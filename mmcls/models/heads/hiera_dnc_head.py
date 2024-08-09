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
# from .cls_head import ClsHead
from .multi_label_head import MultiLabelClsHead

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from .dnc_util import concat_all_gather, MultivariateNormalDiag, AGD_torch_no_grad_gpu, torch_wasserstein_loss

hiera = {
    "hiera_middle":{
        "aquatic mammals": [0, 5],
        "fish": [5, 10],
        "flowers": [10, 15],
        "food containers": [15, 20],
        "fruit and vegetables": [20, 25],
        "household electrical devices": [25, 30],
        "household furniture": [30, 35],
        "insects": [35, 40],
        "large carnivores": [40, 45],
        "large man-made outdoor things": [45, 50],
        "large natural outdoor scenes": [50, 55],
        "large omnivores and herbivores": [55, 60],
        "medium-sized mammals": [60, 65],
        "non-insect invertebrates": [65, 70],
        "people": [70, 75],
        "reptiles": [75, 80],
        "small mammals": [80, 85],
        "trees": [85, 90],
        "vehicles 1": [90, 95],
        "vehicles 2": [95, 100]
    },
    "hiera_high":{
        "animals": [0,1,8,11,12,15,16],
        "plant": [2,4,17],
        "man-made indoor": [3,5,6],
        "man-made outdoor":[9],
        "scenes": [10],
        "invertebrates": [7,13],
        "people": [14],
        "vehicles": [18,19]
    }
}

@HEADS.register_module()
class HieraDNCHead(MultiLabelClsHead):
    def __init__(self,
                 num_classes,
                 in_channels=128,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=None):
        super(HieraDNCHead, self).__init__(loss=loss, init_cfg=init_cfg)

        self.num_classes = num_classes
        self.wanted_dim = in_channels
        self.update_prototype = True 
        self.pretrain_prototype = False 
        self.use_prototype = True
        self.proto_contrast_loss = True # should be True for limitation # change this to False for none contrastive exp.
        self.proto_contrast_loss_weights = 0.05
#         self.loss_average_loss_weights = 0.0001 # 0.01 # 0.0001 success one!

        self.num_prototype = 3 
        
        proto_list = [torch.ones(self.num_prototype, 1) for _ in range(self.num_classes)] # length c, each torch.[10,1]
        
        self.sinkhorn_iterations = 20
        self.gamma_center = 0.9 # 0.999
        self.gamma_covar = 0.999 # 0.999

        self.class_uper_bound = self.num_classes # should change here

        self.K = 2000 # 2000   

        self.max_sample_size = -1 # 1000 # (should a little bit smaller than b*#gpu?) 
        self.register_buffer("queue_log_p", torch.randn(self.num_classes, self.num_prototype, self.K)) # * c g memo
        self.register_buffer("queue", torch.randn(self.num_classes, self.wanted_dim, self.K)) # c e memo
        self.queue = nn.functional.normalize(self.queue, dim=-2)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.memory_sampling_mode = 'uniform'
        self.Ks = torch.tensor([self.K for _c in range(self.num_classes)], dtype=torch.long) # c  (each value=self.K)
        self.apply(init_weights)
        
        #---------------------hiera-----------------------#
        self.num_class_low = 100
        self.num_class_middle = 20

        self.means = nn.Parameter(torch.zeros(self.num_class_low, self.num_prototype, self.wanted_dim), requires_grad=False)
        trunc_normal_(self.means, std=0.02)
        self.diagonal = nn.Parameter(torch.ones(self.num_class_low, self.num_prototype,self.wanted_dim), requires_grad=False) 
        
        self.means_middle = nn.Parameter(torch.zeros(self.num_class_middle, self.num_prototype, self.wanted_dim), requires_grad=False)
        trunc_normal_(self.means_middle, std=0.02)
        self.diagonal_middle = nn.Parameter(torch.ones(self.num_class_middle, self.num_prototype,self.wanted_dim), requires_grad=False) 
        #---------------------hiera-----------------------#
        
        self.eye_alpha = 0.01
        self.eye_matrix = nn.Parameter(torch.ones(self.wanted_dim), requires_grad=False)
        self.final_prob_average_assign = True
        self.iteration_counter = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.update_queue_log_p() 
        self.class_prior = nn.Parameter(torch.ones(self.num_classes).float(), requires_grad=False)

        # CK times embedding_dim
        self.feat_norm = nn.LayerNorm(self.wanted_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)
    
    def update_queue_log_p(self):
        _c = self.queue.transpose(-1,-2)
        _distance_mem = []
        
        #---------------------hiera-----------------------#
        covariances = self.diagonal 
        covariances_middle = self.diagonal_middle 
        for k in range(self.num_class_low):
            _c_k = _c[k] 
            _k_gauss = MultivariateNormalDiag(self.means[k], scale_diag=covariances[k]) # 做高斯的样本空间始终相同
            _prob_a = []
            
            _n_group = 1
            for _n in range(0,_c_k.shape[0],_n_group):  # self.K
                _prob_a.append(_k_gauss.log_prob(_c_k[_n:_n+_n_group,None,...])) 
            probs = torch.cat(_prob_a, dim=0) 
            _distance_mem.append(probs) 
        
        for k in range(self.num_class_low, self.num_class_middle):
            _c_k = _c[k] 
            _k_gauss = MultivariateNormalDiag(self.means_middle[k], scale_diag=covariances_middle[k]) # 做高斯的样本空间始终相同
            _prob_a = []
            
            _n_group = 1
            for _n in range(0,_c_k.shape[0],_n_group):  # self.K
                _prob_a.append(_k_gauss.log_prob(_c_k[_n:_n+_n_group,None,...])) 
            probs = torch.cat(_prob_a, dim=0) 
            _distance_mem.append(probs) 
        
        #---------------------hiera-----------------------#
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

        _prob_n = []
        _prob_n_notlog = []
        _n_group, _c_group = 1, 1

        if self.iteration_counter > 39100: # after steps, change gamma into a smaller value for acc # cifar10/cifar100 39100
            self.gamma_center = 0.99
        if self.iteration_counter > 58650: # mult-if for simplicity multi-stage setting for cifar-10 # cifar-10/cifar-100 58650
            self.gamma_center = 0.999

        new_means = self.means
        new_means_middle = self.means_middle
        new_covariances = self.diagonal
        new_covariances_middle = self.diagonal_middle
        
        '''version2: accelerate batch size'''
        _n_group, _c_group = 1, 1
        
        _c_gauss = MultivariateNormalDiag(new_means.view(-1, self.wanted_dim), scale_diag=new_covariances.view(-1, self.wanted_dim)) # 原版这里把前面维度拉平了 所以是128维的 但是前面都是属于这个类的来做gaussian
        _c_probs = _c_gauss.log_prob(_fea.view(_fea.shape[0], -1, _fea.shape[1]))
        _c_probs = _c_probs.contiguous().view(_c_probs.shape[0], self.num_class_low, self.num_prototype) # *b c g 
        
        _c_gauss_middle = MultivariateNormalDiag(new_means_middle.view(-1, self.wanted_dim), scale_diag=new_covariances_middle.view(-1, self.wanted_dim)) # 原版这里把前面维度拉平了 所以是128维的 但是前面都是属于这个类的来做gaussian
        _c_probs_middle = _c_gauss_middle.log_prob(_fea.view(_fea.shape[0], -1, _fea.shape[1]))
        _c_probs_middle = _c_probs_middle.contiguous().view(_c_probs_middle.shape[0], self.num_class_middle, self.num_prototype)
        
        probs = torch.cat([_c_probs, _c_probs_middle], dim=1) #b, num_classes, c
        
        if gt_label is not None:

            '''keep covariance constant'''
            mem_log_p = self.queue_log_p.data.transpose(-1,-2)

            prob_memo = []
            prob_memo_onehot = []
            unique_c_list = gt_label.unique().long()
            n_memo = []

            new_means_sup = self.means.data.clone()
            new_means_middle_sup = self.means_middle.data.clone()
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
                
                if _c < self.num_class_low:
                    new_means_sup[_c:_c+1,:, :] = f
                else:
                    new_means_middle_sup[_c-self.num_class_low:_c-self.num_class_low+1,:, :] = f
            
            # update
            new_means = torch.add(self.gamma_center * new_means, (1 - self.gamma_center) * new_means_sup)
            new_means_middle = torch.add(self.gamma_center * new_means_middle, (1 - self.gamma_center) * new_means_middle_sup)

            self.means = nn.Parameter(new_means, requires_grad=False)
            self.means_middle = nn.Parameter(new_means_middle, requires_grad=False)
            
            n_saved = torch.cat(n_memo, dim=1)
            n_supervise = torch.ones_like(n_saved) * (self.K / self.num_prototype)

        if gt_label is None:
            return probs.contiguous().view(probs.shape[0],-1), None, None
        if gt_label is not None:
            '''originally entropy here'''
            return probs.contiguous().view(probs.shape[0],-1), n_saved, n_supervise
        else:
            print('Do not support other options, check compute_log_prob_new')
    
    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def loss(self, cls_score, gt_label, weight):
        gt_label = gt_label.type_as(cls_score)
        num_samples = len(cls_score)
        losses = dict()
        # map difficult examples to positive ones
        _gt_label = torch.abs(gt_label)
        # compute loss
        loss = self.compute_loss(cls_score, _gt_label, weight=weight, avg_factor=num_samples)
        losses['loss'] = loss
        return losses
    
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

#     def prepare_target(self, gt_label):
#         b = gt_label.shape
#         gt_label_middle = torch.zeros((b), dtype=gt_label.dtype, device=gt_label.device)
#         gt_label_high = torch.zeros((b), dtype=gt_label.dtype, device=gt_label.device)
#         for index, middle in enumerate(hiera["hiera_middle"].keys()):
#             indices = hiera["hiera_middle"][middle]
#             for ii in range(indices[0], indices[1]):
#                 gt_label_middle[gt_label==ii] = index
                
#         for index, high in enumerate(hiera["hiera_high"].keys()):
#             indices = hiera["hiera_high"][high]
#             for ii in indices:
#                 gt_label_high[gt_label_middle==ii] = index

#         return gt_label_middle, gt_label_high

    def prepare_target(self, gt_label):
        b = gt_label.shape
        gt_label_middle = torch.zeros((b), dtype=gt_label.dtype, device=gt_label.device)
        for index, middle in enumerate(hiera["hiera_middle"].keys()):
            indices = hiera["hiera_middle"][middle]
            for ii in range(indices[0], indices[1]):
                gt_label_middle[gt_label==ii] = index

        return gt_label_middle
    
    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        gt_label_middle = self.prepare_target(gt_label)
        gt_label = F.one_hot(gt_label, self.num_classes-20)
        gt_label_middle = F.one_hot(gt_label_middle, 20)
        new_gt = torch.cat((gt_label, gt_label_middle), dim=1)
        weight = torch.ones_like(new_gt)
        weight[:,100:] = 0.1
#         new_gt = new_gt.type_as(x)

        if self.pretrain_prototype is False and self.use_prototype is True and gt_label is not None:
            seg_logits, contrast_logits, contrast_target, n_saved, n_supervise = self.forward(x, gt_label=gt_label.argmax(-1) if gt_label.dim()==2 else gt_label) # 

            losses = self.loss(seg_logits, new_gt, weight)

            if self.proto_contrast_loss is True:
                loss_proto_contrast = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.class_uper_bound)

                losses['loss_proto_contrast'] = loss_proto_contrast * self.proto_contrast_loss_weights 

#                 loss_new_means = nn.MSELoss(reduction='none')
#                 loss_result_means = torch.mean(loss_new_means(self.means, new_means_sup.float()), dim = -1)
#                 loss_result_means_middle = torch.mean(loss_new_means(self.means_middle, new_means_sup_middle.float()), dim = -1)
#                 losses['loss_means'] = (loss_result_means*self.num_class_low/self.num_classes + loss_result_means_middle*self.num_class_middle/self.num_classes) * self.loss_average_loss_weights

                loss_result_n = torch_wasserstein_loss(n_saved, n_supervise.float())
                losses['loss_wass100'] = loss_result_n * 100.0 
        else:
            seg_logits = self.forward(x, gt_label.argmax(-1) if gt_label.dim()==2 else gt_label)
            losses = self.loss(seg_logits, new_gt, weight)

        self.iteration_counter += 1
        '''both return losses'''
        return losses
    
    def forward(self, x, img_metas=None, gt_label=None):          

        x = self.feat_norm(x)
        x = self.l2_normalize(x)

        self.means.data.copy_(self.l2_normalize(self.means))
        self.means_middle.data.copy_(self.l2_normalize(self.means_middle))

        _log_prob, n_saved, n_supervise = self.compute_log_prob_new(x, gt_label)  

        final_probs = _log_prob.contiguous().view(-1, self.num_classes, self.num_prototype)

        # sum of probability on gaussian dimension
        _sum_prob = torch.amax(final_probs, dim=-1)

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

            return out_seg, contrast_logits, contrast_target, n_saved, n_supervise

        return out_seg
    
    def simple_test(self, x, softmax=True, post_process=True):
        x = self.pre_logits(x)
        cls_score = self.forward(x)

        if softmax:
            pred = (
                F.softmax(cls_score[:,:100], dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


