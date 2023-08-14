# training schedule for 1x
_base_ = [
    '_base_aster.py',
    '../_base_/datasets/et_data.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamw_cos_6e.py',
]

# dataset settings
train_list = [
    _base_.toy_rec_train
   
]
test_list = [
    _base_.toy_rec_test
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=100))

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=32,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

auto_scale_lr = dict(base_batch_size=1024)

test_dataloader = dict(
    batch_size=32,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

_base_.train_cfg.update(dict(max_epochs=25, val_interval=10))

val_dataloader = test_dataloader

#val_evaluator = dict(
    #dataset_prefixes=['et_data'])
#val_evaluator = dict(dataset_prefixes=['et_data'],  metrics=[dict(
            #type='WordMetric',
            #mode=['exact', 'ignore_case', 'ignore_case_symbol']),
            #dict(type='OneMinusNEDMetric') ])
#val_evaluator = dict(metrics=[...])
val_evaluator = [
    dict(type='WordMetric', mode=['exact', 'ignore_case', 'ignore_case_symbol']),
    dict(type='OneMinusNEDMetric')
]
    
test_evaluator = val_evaluator
