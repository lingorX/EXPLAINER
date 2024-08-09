import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """ 
    #rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    #tensors_gather[rank] = tensor
    output = torch.cat(tensors_gather, dim=0)
    
    return output


def _batch_vector_diag(bvec):
    """
    Returns the diagonal matrices of a batch of vectors.
    """
    n = bvec.size(-1)
    bmat = bvec.new_zeros(bvec.shape + (n,))
    bmat.view(bvec.shape[:-1] + (-1,))[..., ::n + 1] = bvec
    return bmat


class MultivariateNormalDiag(Distribution):
    
    arg_constraints = {"loc": constraints.real,
                       "scale_diag": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale_diag, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        event_shape = loc.shape[-1:] 
        if scale_diag.shape[-1:] != event_shape:
            raise ValueError("scale_diag must be a batch of vectors with shape {}".format(event_shape))

        try:
            self.loc, self.scale_diag = torch.broadcast_tensors(loc, scale_diag)
        except RuntimeError:
            raise ValueError("Incompatible batch shapes: loc {}, scale_diag {}"
                             .format(loc.shape, scale_diag.shape))
        batch_shape = self.loc.shape[:-1]
        super(MultivariateNormalDiag, self).__init__(batch_shape, event_shape,
                                                        validate_args=validate_args)

    @property
    def mean(self):
        return self.loc
    
    @lazy_property
    def variance(self):
        return self.scale_diag.pow(2)

    @lazy_property
    def covariance_matrix(self):
        return _batch_vector_diag(self.scale_diag.pow(2))

    @lazy_property
    def precision_matrix(self):
        return _batch_vector_diag(self.scale_diag.pow(-2))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.loc.new_empty(shape).normal_()
        return self.loc + self.scale_diag * eps

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        means_detached = self.loc.clone().detach()
        diff = value - means_detached
        return (
            -0.5 * (math.log(2 * math.pi) * self.scale_diag).log().sum(-1)
            -0.5 * (diff / self.scale_diag).pow(2).sum(-1)
        )

    def entropy(self):
        return (
            0.5 * self._event_shape[0] * (math.log(2 * math.pi) + 1) +
            self.scale_diag.log().sum(-1)
        )
    
    '''experimental'''
    def prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc  
        return (1 / (self.scale_diag * math.sqrt(2 * math.pi)).sum(-1) * torch.exp((diff / self.scale_diag).pow(2).sum(-1) * 0.5))

# K * #Guass as input
@torch.no_grad()
def AGD_torch_no_grad_gpu(M, maxIter=20, eps=0.05):
    # st = datetime.datetime.now()
    M = M.t() # [#Guass, K]
    p = M.shape[0] # #Guass
    n = M.shape[1] # K 
    
    '''new-added'''
    X = torch.zeros((p,n), dtype=torch.float64).cuda()
    # X = Input_X

    r = torch.ones((p,), dtype=torch.float64).to(M.device) / p # .to(L.device) / K
    c = torch.ones((n,), dtype=torch.float64).to(M.device) / n # .to(L.device) / B 先不要 等会加上

    max_el = torch.max(abs(M)) #np.linalg.norm(M, ord=np.inf)
    gamma = eps/(3*math.log(n)) 

    A = torch.zeros((maxIter, 1), dtype=torch.float64).to(M.device) #init array of A_k
    L = torch.zeros((maxIter, 1), dtype=torch.float64).to(M.device) #init array of L_k

    #set initial values for APDAGD
    L[0,0] = 1; #set L_0

    #set starting point for APDAGD
    y = torch.zeros((n+p, maxIter), dtype=torch.float64).cuda() #init array of points y_k for which usually the convergence rate is proved (eta)
    z = torch.zeros((n+p, maxIter), dtype=torch.float64).cuda() #init array of points z_k. this is the Mirror Descent sequence. (zeta)    
    j = 0
    #main cycle of APDAGD
    for k in range(0,(maxIter-1)):
                         
        L_t = (2**(j-1))*L[k,0] #current trial for L            
        a_t = (1  + torch.sqrt(1 + 4*L_t*A[k,0]))/(2*L_t) #trial for calculate a_k as solution of quadratic equation explicitly
        A_t = A[k,0] + a_t; #trial of A_k
        tau = a_t / A_t; #trial of \tau_{k}     
        x_t = tau*z[:,k] + (1 - tau)*y[:,k]; #trial for x_k
        
        #calculate trial oracle at xi      
        # 分别计算x方向和y方向     
        #calculate function \psi(\lambda,\mu) value and gradient at the trial point of x_{k}
        lamb = x_t[:n,]
        mu = x_t[n:n+p,]    
        
        # 1) [K,1] * [1, #Gauss] --> [K, #Gauss].T -->[#Gauss, K]; 2) [K, 1] * [#Guass, 1].T --> [K, #Guass]--.T--> [#Guass, K]
        M_new = -M - torch.matmul(lamb.reshape(-1,1).cuda(), torch.ones((1,p), dtype=torch.float64).cuda()).T - torch.matmul(torch.ones((n,1), dtype=torch.float64).cuda(), mu.reshape(-1,1).T.cuda()).T

        X_lamb = torch.exp(M_new/gamma)
        sum_X = torch.sum(X_lamb)
        X_lamb = X_lamb/sum_X
        grad_psi_x_t = torch.zeros((n+p,), dtype=torch.float64).cuda() 
        grad_psi_x_t[:p,] = r - torch.sum(X_lamb, axis=1)
        grad_psi_x_t[p:p+n,] = c - torch.sum(X_lamb, axis=0).T

        #update model trial
        z_t = z[:,k] - a_t*grad_psi_x_t #trial of z_k 
        y_t = tau*z_t + (1 - tau)*y[:,k] #trial of y_k

        #calculate function \psi(\lambda,\mu) value and gradient at the trial point of y_{k}
        lamb = y_t[:n,]
        mu = y_t[n:n+p,]           
        M_new = -M - torch.matmul(lamb.reshape(-1,1).cuda(), torch.ones((1,p), dtype=torch.float64).cuda()).T - torch.matmul(torch.ones((n,1), dtype=torch.float64).cuda(), mu.reshape(-1,1).T.cuda()).T
        Z = torch.exp(M_new/gamma)
        sum_Z = torch.sum(Z)

        X = tau*X_lamb + (1-tau)*X #set primal variable 
             
        L[k+1,0] = L_t
        j += 1
    
    X = X.t()

    indexs = torch.argmax(X, dim=1) # torch.Size([1001])
    G = F.gumbel_softmax(X, tau=0.5, hard=True) # torch.Size([1004, 10])
    G_probs = torch.sum(G, dim=0)/len(indexs)
    
    return G.to(torch.float32), indexs, G_probs #change into G as well

def torch_wasserstein_loss(tensor_a,tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a,tensor_b,p=1))

def torch_cdf_loss(tensor_a,tensor_b,p=1):
    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)
    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss