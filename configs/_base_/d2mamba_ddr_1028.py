# D²Mamba on DDR (1028x1028), shared settings for all five folds.
# Derived from the resolved training configs of the paper runs.
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'd2mamba.models.mm_backbone',
        'd2mamba.models.cclr_head',
        'd2mamba.datasets.ddr_lesion_dataset',
        'd2mamba.transforms.retina_preprocess',
        'd2mamba.losses.safe_ce_loss',
        'd2mamba.metrics.ddr_metrics',
        'd2mamba.hooks.experiment_hooks',
    ])

default_scope = 'mmseg'
data_root = 'data/DDR_mmseg'
dataset_type = 'DDRLesionDataset'
crop_size = (1028, 1028)
class_weight = [0.5, 0.7, 1.5, 0.7, 1.1]  # background, EX, MA, SE, HE

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='D2MambaBackbone',
        pretrained=None,  # trained from scratch
        embed_dim=64,     # widths 64/128/256/512
        depths=(2, 4, 8, 4),
        num_heads=(2, 4, 8, 16),
        attn_types=['mamba2', 'mamba2', 'mamba2', 'standard'],  # SDM x3 stages, MHSA in stage 4
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.2,
        d_state=64,
        ssd_expand=2,
        ssd_chunk_size=256,
        ssd_positve_dA=True,
        linear_attn_duality=True,  # NC-SSD
        bidirection=False,
        simple_downsample=False,
        simple_patch_embed=False,
        use_defmamba_branch=True,  # DMSM
        defmamba_scale_init=1e-3,  # gamma in Y = Y_nc + gamma * Y_def
        defmamba_d_state=16,
        defmamba_ssm_ratio=1.0,
        defmamba_drop=0.0),
    decode_head=dict(
        type='CCLRHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        decoder_channels=(256, 192, 128),
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='LN2d', eps=1e-6, requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='SafeCrossEntropyLoss', use_sigmoid=False,
                 class_weight=class_weight, loss_weight=1.0, loss_name='loss_ce')
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(type='SafeCrossEntropyLoss', use_sigmoid=False,
                 class_weight=class_weight, loss_weight=0.4, loss_name='loss_aux_ce')
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

retina_preprocess = dict(type='RetinaPreprocess', output_size=crop_size, sigma_x=15, crop_tol=7)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    retina_preprocess,
    dict(type='RandomResize', scale=crop_size, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    retina_preprocess,
    dict(type='PackSegInputs'),
]

_dataset = dict(type=dataset_type, data_root=data_root, ann_file='',
                data_prefix=dict(img_path='', seg_map_path=''))
train_dataloader = dict(
    batch_size=4, num_workers=8, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(**_dataset, manifest_file=None, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(**_dataset, manifest_file=None, pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(**_dataset, manifest_file=None, pipeline=test_pipeline))
val_evaluator = dict(type='DDRLesionMetric', num_classes=5, ignore_index=255, aupr_bins=1000)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0))))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, by_epoch=False, begin=1500, end=100000),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,
                    max_keep_ckpts=3, save_best='mDice', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
# Runs the test split after every validation and writes train_history.xlsx /
# best_metrics.xlsx into work_dir (consumed by tools/aggregate_5fold.py).
custom_hooks = [dict(type='ValTestXlsxHook')]
env_cfg = dict(cudnn_benchmark=True,
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
