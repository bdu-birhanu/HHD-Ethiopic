_base_ = [
  
    '../_base_/datasets/et_data.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_step_5e.py',
    '_base_satrn_shallow.py',
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)

# dataset settings
train_list = [_base_.toy_rec_train]
test_list = [
    _base_.toy_rec_test
]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=3e-4))

train_dataloader = dict(
    batch_size=128,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

val_evaluator = dict(dataset_prefixes=['et_data'],  metrics=[dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
            dict(type='OneMinusNEDMetric') ])
            
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64 * 8)
