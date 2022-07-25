_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance_ms.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/hunto/LightViT/releases/download/v0.0.1/lightvit_base_82.1.ckpt'),
        type='lightvit_base',
        drop_path_rate=0.1,
        out_indices=(0, 1, 2),
        stem_norm_eval=True,    # fix the BN running stats of the stem layer
        ),
    neck=dict(
        type='LightViTFPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        num_outs=5,
        num_extra_trans_convs=1,
        ))

data = dict(samples_per_gpu=2)

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
# optimizer
# learning policy
lr_config = dict(step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
total_epochs = 36
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
total_epochs = 36

