_base_ = [
    './swin-dnc-small_224.py',
    # '../_base_/models/swin_transformer/small_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(batch_size=128) # 128*8=1024
