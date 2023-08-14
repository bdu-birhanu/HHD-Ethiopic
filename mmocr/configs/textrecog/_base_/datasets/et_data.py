#toy_data_root = 'tests/data/rec_toy_dataset/'
#toy_data_root = '../new_hhd/train/train_raw/new_split/'
#toy_data_root_tr = '../new_hhd/test/test_18th/test_18th_raw/'
#toy_data_root_tr = '../new_hhd/test/test_rand/test_rand_raw/'
toy_data_root = 'tests/data/rec_toy_dataset/'

toy_rec_train = dict(
    type='OCRDataset',
    data_root=toy_data_root,
    data_prefix=dict(img_path='imgs/'),
    ann_file='toy_label.json',
    pipeline=None,
    test_mode=False)

toy_rec_test = dict(
    type='OCRDataset',
    data_root=toy_data_root,
    data_prefix=dict(img_path='imgs/'),
    ann_file='toy_label.json',
    pipeline=None,
    test_mode=True)
