_base_ = [
    '../_base_/models/swin_transformer/small_224.py',
]

# model settings
model = dict(
    head=dict(
        type='DNCHead',
        num_classes=1000,
        # loss=dict(
        #     type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
)
