# training schedule for 1x
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/et_data.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]

# dataset settings
train_list = [_base_.toy_rec_train]
test_list = [_base_.toy_rec_test]

default_hooks = dict(checkpoint=dict(interval=5, type='CheckpointHook'),logger=dict(type='LoggerHook', interval=100), )

train_dataloader = dict(
    batch_size=32,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))
test_dataloader = val_dataloader

_base_.model.decoder.dictionary.update(
    dict(with_unknown=True, unknown_token=None))
_base_.train_cfg.update(dict(max_epochs=25, val_interval=20))

val_evaluator = [
    dict(type='WordMetric', mode=['exact', 'ignore_case', 'ignore_case_symbol']),
    dict(type='OneMinusNEDMetric')
]

test_evaluator = val_evaluator
