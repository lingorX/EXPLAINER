# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .multi_label_head import MultiLabelClsHead



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
class HieraHead(MultiLabelClsHead):
    """Linear classification head for multilabel task.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01)):
        super(HieraHead, self).__init__(
            loss=loss, init_cfg=init_cfg)

        if num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.fc = nn.Linear(self.in_channels, self.num_classes)

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
    
    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        
#         gt_label_middle, gt_label_high = self.prepare_target(gt_label)
#         gt_label = F.one_hot(gt_label, self.num_classes-28)
#         gt_label_middle = F.one_hot(gt_label_middle, 20)
#         gt_label_high = F.one_hot(gt_label_high, 8)
#         new_gt = torch.cat((gt_label, gt_label_middle, gt_label_high), dim=1)
        gt_label_middle = self.prepare_target(gt_label)
        gt_label = F.one_hot(gt_label, self.num_classes-20)
        gt_label_middle = F.one_hot(gt_label_middle, 20)
        new_gt = torch.cat((gt_label, gt_label_middle), dim=1)
        
        new_gt = new_gt.type_as(x)
        cls_score = self.fc(x)
        
        weight = torch.ones_like(cls_score)
        weight[:,100:] = 0.1

        losses = self.loss(cls_score, new_gt, weight, **kwargs)
        return losses

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
    
    def simple_test(self, x, sigmoid=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            sigmoid (bool): Whether to sigmoid the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        
        if sigmoid:
            pred = F.softmax(cls_score[:,:100], dim=1) if cls_score is not None else None
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred
