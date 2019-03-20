from collections import OrderedDict
import os.path as osp

# baseline softmax
param_base = \
    {
        # system fix params
        'workers': 4,
        'split': 0,
        'height': 256,
        'width': 128,
        'data_dir': osp.join(osp.dirname(osp.abspath(__file__)), 'data'),
        'logs_dir': osp.join(osp.dirname(osp.abspath(__file__)), 'logs'),
        'dataset': 'market1501',  # 'cuhk03'  'market1501'  'MSMT17_V1'
        'arch': 'resnet50',
        'momentum': 0.9,
        'weight_decay': 5e-4,

        # display and evaluate
        'resume': False,
        'evaluate': False,
        'start_save': 0,
        'step_size': 35,
        'epochs': 120,
        'print_freq': 20,
        'W_std': 0.001,
        'loss': 'softmax',  # 'softmax', 'hingeloss', 'std_hingeloss'

        # some basis hyper-params
        'batch_size': 512,
        'combine_trainval': True,
        'features': 0,
        'dropout': 0.5,
        'lr': 0.1,

        # hyper-params
        'gamma': 1.0,
        'F_norm': True,
        'W_norm': True,
        'norm_test': False,
        'CCS' : False,
        'up_term' : False,

        # pretrained
        'pretrained': True,
    }


# metric triplet
param_metric_triplet = \
    {
        # system fix params
        'workers': 4,
        'split': 0,
        'height': 256,
        'width': 128,
        'data_dir': osp.join(osp.dirname(osp.abspath(__file__)), 'data'),
        'logs_dir': osp.join(osp.dirname(osp.abspath(__file__)), 'logs'),
        'dataset': 'market1501',  # 'cuhk03'  'market1501'  'MSMT17_V1'
        'arch': 'resnet50',
        'dist_metric': 'tmm',
        'momentum': 0.9,
        'weight_decay': 5e-4,

        # display and evaluate
        'resume': False,
        'evaluate': False,
        'start_save': 0,
        'step_size': 35,
        'epochs': 45,
        'print_freq': 20,
        'W_std': 0.001,
        'loss': 'softmax',  # 'softmax', 'hingeloss', 'std_hingeloss'

        # some basis hyper-params
        'num_instances': 4,
        'batch_size': 256,
        'combine_trainval': True,
        'features': 128,
        'dropout': 0,
        'lr': 0.0002,

        # hyper-params

        'gamma': 1.0,
        # 'F_norm': False,
        # 'W_norm': False,
        'norm_test': False,
        'norm_train': False,
        'margin': 0.5,

    }



