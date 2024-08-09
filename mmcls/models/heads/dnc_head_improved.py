# Copyright (c) OpenMMLab. All rights reserved.
# Success_on_Cifar100 1st. Anotated.
import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

from ..builder import HEADS
from .cls_head import ClsHead

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from .dnc_util import concat_all_gather, MultivariateNormalDiag, AGD_torch_no_grad_gpu, torch_wasserstein_loss


@HEADS.register_module()
class DNCHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 act_cfg=dict(type='ReLU'), ##
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(DNCHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.update_prototype = True 
        self.pretrain_prototype = False 
        self.use_prototype = True
        self.proto_contrast_loss = True # should be True for limitation # change this to False for none contrastive exp.
        self.proto_contrast_loss_weights = 0.05
        self.loss_average_loss_weights = 0.0001 # 0.01 # 0.0001 success one!
        self.in_channels = in_channels
        # self.iteration_counter = 0

        self.num_prototype = 3
        self.embedding_dim = self.in_channels 
        
        proto_list = [torch.ones(self.num_prototype, 1) for _ in range(self.num_classes)] # length c, each torch.[10,1]
        
        self.sinkhorn_iterations = 20
        self.gamma_center = 0.9 # 0.999
        self.gamma_covar = 0.999 # 0.999

        self.class_uper_bound = self.num_classes # should change here

        self.expand_dim = True # for reduce dim in resnet
        if self.expand_dim == True:
            self.wanted_dim = 128 # self.wanted_dim = self.embedding_dim if not need for expand dim
        else:
            self.wanted_dim = self.embedding_dim

        self.K = 2000 # 2000   

        self.max_sample_size = -1 # 1000 # (should a little bit smaller than b*#gpu?) 
        self.register_buffer("queue_log_p", torch.randn(self.num_classes, self.num_prototype, self.K)) # * c g memo
        self.register_buffer("queue", torch.randn(self.num_classes, self.wanted_dim, self.K)) # c e memo
        self.queue = nn.functional.normalize(self.queue, dim=-2)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.memory_sampling_mode = 'uniform'
        self.Ks = torch.tensor([self.K for _c in range(self.num_classes)], dtype=torch.long) # c  (each value=self.K)
        self.apply(init_weights)
        self.requireGrad = True
        self.means = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, self.wanted_dim), requires_grad=self.requireGrad) # * c g e
        trunc_normal_(self.means, std=0.02)

        # cov matrix
        self.diagonal = nn.Parameter(torch.ones(self.num_classes,self.num_prototype,self.wanted_dim), requires_grad=False) 
        
        self.means_old = None
        self.diagonal_old = None
        self.firstRun = True
        self.eye_alpha = 0.01
        self.eye_matrix = nn.Parameter(torch.ones(self.wanted_dim), requires_grad=False)
        
        # self.loss_contrast_G = False
        self.final_prob_average_assign = True
        # self.loss_across_embedding = False
        '''testing equivalent'''
        self.debug_distribution = False
        if self.debug_distribution is True:
            self.saving_positions = torch.zeros((self.num_classes, self.num_prototype))
            self._sum_prob_position = None
            self.pred_position = None
            self.proto_position = None
        
        # * to make it be embedded in state dict
        self.iteration_counter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.update_queue_log_p() 
        self.class_prior = nn.Parameter(torch.ones(self.num_classes).float(), requires_grad=False)

        # CK times embedding_dim
        self.feat_norm = nn.LayerNorm(self.wanted_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)
    
    def update_queue_log_p(self):
        covariances = self.diagonal # * c g e
        _c = self.queue.transpose(-1,-2) # * c, k, d == class * self.K(buffer_size) * embedding

        _distance_mem = []
        for k in range(self.num_classes):
            _c_k = _c[k] 
            _k_gauss = MultivariateNormalDiag(self.means[k], scale_diag=covariances[k]) # 做高斯的样本空间始终相同
            _prob_a = []
            
            _n_group = 1
            for _n in range(0,_c_k.shape[0],_n_group):  # self.K
                _prob_a.append(_k_gauss.log_prob(_c_k[_n:_n+_n_group,None,...])) 
            probs = torch.cat(_prob_a, dim=0) 
            _distance_mem.append(probs) 

        # * c, k, p 
        _init_value = torch.stack(_distance_mem, dim=0).transpose(-1,-2)
        # * c, p, k 
        self.queue_log_p = _init_value
    
    
    @torch.no_grad()
    def _dequeue_and_enqueue_k(self, _c, _c_embs, _c_mask, _c_log_prob):

        if _c_mask is None: 
            print('_c_mask_return None value')
            _c_mask = torch.zeros(_c_embs.shape[0]).detach_() 
        _sample_size = _c_mask.int().sum() # 这一类的个数
        if _sample_size == 0: return;

        # * adaptive to number of prototypes
        _k_max_sample_size = self.max_sample_size

        ptr = int(self.queue_ptr[_c])
        _c_embs = _c_embs[_c_mask>0]
        _c_log_prob = _c_log_prob[_c_mask>0] # 提该类别log.prob
        
        # replace the embs at ptr (dequeue and enqueue)
        if ptr + _sample_size >= self.Ks[_c]: 
        
            _fir = self.Ks[_c] - ptr
            _sec = _sample_size - self.Ks[_c] + ptr
            self.queue[_c, :, ptr:self.Ks[_c]] = _c_embs[:_fir].T
            self.queue[_c, :, :_sec] = _c_embs[_fir:].T

            self.queue_log_p[_c, :, ptr:self.Ks[_c]] = _c_log_prob[:_fir].T
            self.queue_log_p[_c, :, :_sec] = _c_log_prob[_fir:].T

        else: 
            self.queue[_c, :, ptr:ptr + _sample_size] = _c_embs.T
            self.queue_log_p[_c, :, ptr:ptr + _sample_size] = _c_log_prob.T
        
        ptr = (ptr + _sample_size) % int(self.Ks[_c]) 
        self.queue_ptr[_c] = ptr

    def compute_log_prob_new(self, _fea, gt_label):
        
        # simple added here to detach computation graph(Nope)
        if self.firstRun is True:
            self.means_old = self.means.data.clone()

        _prob_n = []
        _prob_n_notlog = []
        _n_group, _c_group = 1, 1

        if gt_label is not None:
            
            if self.iteration_counter > 39100: # after step of cifar10, change gamma into a smaller value for acc # cifar10/cifar100 39100
                self.gamma_center = 0.99
            
            # mult-if for simplicity multi-stage setting for cifar-10 # cifar-10/cifar-100 58650
            if self.iteration_counter > 58650:
                self.gamma_center = 0.999
            new_means = torch.add(self.gamma_center * self.means_old, (1 - self.gamma_center) * self.means)

            old_means_data_clone = self.means_old
            '''avoid NAN values in new_means'''
            try: assert torch.isnan(new_means).int().sum() <= 0
            except: 
                print('NAN in new_means: {}'.format(torch.isnan(new_means).int().sum()))               
                new_means[torch.isnan(new_means)] = 0.
            
            try: assert torch.is_complex(new_means) is False
            except: 
                print('Complex number in new_means: {}'.format(torch.is_complex(new_means)))
                '''set it to previously exist number'''               
                new_means[torch.is_complex(new_means)] = old_means_data_clone[torch.is_complex(new_means)]
            
            new_covariances = self.diagonal

        else:
            new_means = self.means_old
            new_covariances = self.diagonal

        '''version2: accelerate batch size'''
        _n_group, _c_group = 1, 1
        
        _c_gauss = MultivariateNormalDiag(new_means.view(-1, self.wanted_dim), scale_diag=new_covariances.view(-1, self.wanted_dim)) # 原版这里把前面维度拉平了 所以是128维的 但是前面都是属于这个类的来做gaussian
            
        _c_probs = _c_gauss.log_prob(_fea.view(_fea.shape[0], -1, _fea.shape[1]))

        _c_probs = _c_probs.contiguous().view(_c_probs.shape[0], self.num_classes, self.num_prototype) # *b c g 
        
        probs = _c_probs
        
        if gt_label is not None:

            '''keep covariance constant'''
            mem_log_p = self.queue_log_p.data.transpose(-1,-2)

            prob_memo = []
            prob_memo_onehot = []
            unique_c_list = gt_label.unique().long()
            n_memo = []

            new_means_sup = self.means.data.clone()
            for _c in unique_c_list:
                if _c == self.class_uper_bound: continue # origin 255
                _c = _c.item()
                prob_log_new = probs[gt_label == _c, _c:_c+1, :]

                wanted_shape = prob_log_new.shape[0]
                _c_init_q_log = mem_log_p[_c:_c+1,:(self.K - wanted_shape),:]
                _c_init_q_log = torch.cat([prob_log_new, _c_init_q_log.transpose(0, 1)], dim=0)
                _c_init_q_log = _c_init_q_log / _c_init_q_log.sum(dim=-1, keepdim=True) # K, 1, g

                indexs = torch.argmax(_c_init_q_log, dim=-1)
                oneHot_Ver = torch.nn.functional.one_hot(indexs, num_classes=self.num_prototype)
                prob_memo_onehot.append(oneHot_Ver)

                _mem_fea_k = self.queue[_c:_c+1,:,:].data.clone().transpose(-1,-2)
                n = torch.sum(_c_init_q_log, dim=0) # 1, g
                n_memo.append(n)

                f = oneHot_Ver.float().permute((1, 2, 0)) @ _mem_fea_k
                f = self.l2_normalize(f)

                new_means_sup[_c:_c+1,:, :] = f
            
            n_saved = torch.cat(n_memo, dim=1)
            n_supervise = torch.ones_like(n_saved) * (self.K / self.num_prototype)

        if gt_label is None:
            return probs.contiguous().view(probs.shape[0],-1), None, None, None
        if gt_label is not None:
            '''originally entropy here'''
            return probs.contiguous().view(probs.shape[0],-1), n_saved, n_supervise, new_means_sup
        else:
            print('Do not support other options, check compute_log_prob_new')


    def simple_test(self, x, softmax=True, post_process=True):
        x = self.pre_logits(x)
        cls_score = self.forward(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
            
            if self.debug_distribution is True:
                self.pred_position = torch.argmax(pred, dim=-1)
                for i in range(len(pred)):
                    cls_num = self.pred_position[i]
                    proto_num = self._sum_prob_position[i, cls_num]
                    self.saving_positions[cls_num, proto_num] += 1
                print(self.saving_positions)
                self.pred_position = None
                self._sum_prob_position = None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred
    
    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def online_contrast(self, gt_seg, simi_logits):

        # compute logits and apply temperature
        proto_logits = simi_logits.flatten(1) 
        proto_target = gt_seg.clone().float()

        unique_c_list = gt_seg.unique().long()

        mem_q = self.queue_log_p.transpose(-1,-2) # c,n,p 
        
        # clustering for each class
        for k in unique_c_list:
            if k == self.class_uper_bound: continue # origin 255
            # get initial assignments for the k-th class
            init_q = simi_logits[:, k, :]
            init_q = init_q[gt_seg == k, ...] # n,p
            
            mem_init_q = mem_q[k][:self.Ks[k],:] # m,p
            init_q = torch.cat([init_q, mem_init_q], dim=0)
            init_q = init_q / torch.abs(init_q).max()

            # clustering q.shape = n x self.num_prototypes
            q, indexs, G_probs = AGD_torch_no_grad_gpu(init_q, maxIter=self.sinkhorn_iterations)

            try:
                assert torch.isnan(q).int().sum() <= 0
            except:
                print('NAN in online contrast: {}'.format(torch.isnan(q).int().sum()))
                # * process nan
                q[torch.isnan(q)] = 0
                indexs[torch.isnan(q).int().sum(dim=1)>0] = self.class_uper_bound - (self.num_prototype * k)

            proto_target[gt_seg == k] = indexs[:-mem_init_q.shape[0]].float() + (self.num_prototype * k)

        return proto_logits, proto_target

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        if self.pretrain_prototype is False and self.use_prototype is True and gt_label is not None:
            seg_logits, contrast_logits, contrast_target, n_saved, n_supervise, new_means_sup = self.forward(x, gt_label=gt_label.argmax(-1) if gt_label.dim()==2 else gt_label) # 

            losses = self.loss(seg_logits, gt_label, **kwargs)

            if self.proto_contrast_loss is True:
                loss_proto_contrast = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.class_uper_bound)

                losses['loss_proto_contrast'] = loss_proto_contrast * self.proto_contrast_loss_weights 

                loss_new_means = nn.MSELoss(reduction='none')
                loss_result_means = torch.mean(loss_new_means(self.means, new_means_sup.float()), dim = -1)
                losses['loss_means'] = loss_result_means * self.loss_average_loss_weights

                loss_result_n = torch_wasserstein_loss(n_saved, n_supervise.float())
                losses['loss_wass100'] = loss_result_n * 100.0 
        else:
            seg_logits = self.forward(x, gt_label.argmax(-1) if gt_label.dim()==2 else gt_label)
            losses = self.loss(seg_logits, gt_label)

        self.iteration_counter += 1
        '''both return losses'''
        return losses
    
    def forward(self, x, img_metas=None, gt_label=None):          

        x = self.feat_norm(x)
        x = self.l2_normalize(x)

        self.means.data.copy_(self.l2_normalize(self.means))

        _log_prob, n_saved, n_supervise, new_means_sup = self.compute_log_prob_new(x, gt_label)  

        self.firstRun = False
        final_probs = _log_prob.contiguous().view(-1, self.num_classes, self.num_prototype)

        # sum of probability on gaussian dimension
        _sum_prob = torch.amax(final_probs, dim=-1)

        '''for testing equivalence'''
        if self.debug_distribution is True:
            self._sum_prob_position = torch.argmax(final_probs, dim=-1)

        out_seg = self.mask_norm(_sum_prob) if self.mask_norm is not None else _sum_prob
        out_seg = out_seg * self.class_prior[None,...] # this should be the output

        if gt_label is not None: 
            gt_seg = gt_label
            
            with torch.no_grad():
                if self.K > 0:
                    # * update memory
                    _c_mem = concat_all_gather(x) # save origin _c
                    _gt_seg_mem = concat_all_gather(gt_seg)
                    _log_prob_mem = concat_all_gather(final_probs) # *(b*gpu) c g

                    unique_c_list = _gt_seg_mem.unique().int()
                    # 每个cls都单独拿一个 拿到标号
                    for k in unique_c_list:
                        if k == self.class_uper_bound: continue
                        k = k.item()
                        self._dequeue_and_enqueue_k(k, _c_mem, (_gt_seg_mem == k), _log_prob_mem[:,k,:])

            if self.proto_contrast_loss == True: # self_changed
                # same architecture as prototype learning (self understanding)
                contrast_logits, contrast_target = self.online_contrast(gt_seg, final_probs)
            else: contrast_logits, contrast_target = None, None

            return out_seg, contrast_logits, contrast_target, n_saved, n_supervise, new_means_sup

        return out_seg

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
