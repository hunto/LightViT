_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/default_runtime.py'
]
model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='lightvit_small',
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/hunto/LightViT/releases/download/v0.0.1/lightvit_small_80.9.ckpt'),
        out_indices=(0, 1, 2),
        stem_norm_eval=True,    # fix the BN running stats of the stem layer
        ),
    neck=dict(
        type='LightViTFPN',
        in_channels=[96, 192, 384],
        out_channels=256,
        num_outs=5,
        num_extra_trans_convs=1,
        ))
# data
data = dict(samples_per_gpu=2)  # 2 x 8 = 16
# optimizer
optimizer = dict(type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.04,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'stem.1': dict(decay_mult=0.),
                                                 'stem.4': dict(decay_mult=0.),
                                                 'stem.7': dict(decay_mult=0.),
                                                 'stem.10': dict(decay_mult=0.),
                                                 'global_token': dict(decay_mult=0.)
                                                 }))
# optimizer_config = dict(grad_clip=None)
# do not use mmdet version fp16
# fp16 = None
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

