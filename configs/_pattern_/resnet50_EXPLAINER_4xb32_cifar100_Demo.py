_base_ = [
    '../_base_/models/resnet50_cifar.py',
    '../_base_/datasets/cifar100_bs32.py',
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
#     head=dict(
# #         type='DNCHead',
#         num_classes=100,
#         loss=dict(
# #             type='FocalLoss', gamma=2.0, alpha=0.25),
#             type='AsymmetricLoss', gamma_pos=0.0, gamma_neg=3.0, clip=0.15)
#     ),
#     head=dict(
#         type='HieraHead',
#         num_classes=128,
#         loss=dict(
# #             type='FocalLoss', gamma=2.0, alpha=0.25),
#             type='AsymmetricLoss', gamma_pos=0.0, gamma_neg=3.0, clip=0.18)
#     ),
    head=dict(
        type='HieraHead',
        num_classes=120,
        loss=dict(
#             type='FocalLoss', gamma=2.0, alpha=0.25),
            type='AsymmetricLoss', gamma_pos=0.0, gamma_neg=3.0, clip=0.15)
    ),
#     head=dict(
#         type='HieraDNCHead',
#         num_classes=120,
#         in_channels=128,
#         loss=dict(
# #             type='FocalLoss', gamma=2.0, alpha=0.25),
#             # type='AsymmetricLoss', gamma_pos=0.0, gamma_neg=3.0, clip=0.15)
#             type='StructuredLoss', num_classes=100)
#     ),
#     head=dict(
#         type='DNCSigmoidHead',
#         num_classes=100,
#         in_channels=128,
#         loss=dict(
# #             type='FocalLoss', gamma=2.0, alpha=0.25),
#             type='AsymmetricLoss', gamma_pos=0.0, gamma_neg=3.0, clip=0.15)
#     ),
    # head=dict(
    #     type='DNCSoftmaxHead',
    #     num_classes=100,
    #     in_channels=128,
    # ),
)
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)

runner = dict(type='EpochBasedRunner', max_epochs=1)
