from __future__ import print_function, absolute_import

from utils import batch_experiments
import os.path as osp
import os
import torch


def bth_train_softmax():
    from softmax_loss import main as softmaxfunc
    from Param_Config import param_base
    dir_name = "logs/Market1501-ResNet50/softmax_baseline"

    header = "with-initial-ckpt-"
    hyper_param_keys   = ('lr',    'features')
    hyper_param_values = [(0.1,       512),
                          # (0.1,       1024),
                          # (0.1,       2048),
                          # (0.1,       4096),
                          # (0.1,       8192),
                          ]

    results = batch_experiments(
        hyper_param_keys, hyper_param_values, param_base,
        header, dir_name, softmaxfunc
    )

def bth_train_softmax_mod():
    from softmax_loss_modified import main as softmaxfunc
    from Param_Config import param_base
    dir_name = "logs/Market1501-ResNet50/softmax_baseline"

    header = "with-initial-ckpt-"
    hyper_param_keys   = ('lr',    'features', 'gamma',  'batch_size')
    hyper_param_values = [
        #(0.01, 1024, 1, 256),
        #(0.1, 1024, 10, 256),
        (0.1, 512, 0.01, 512),
        (0.1, 512, 0.1, 512),
        (0.1, 512, 1.0, 512),
        #(0.1, 512, 5, 512),
        #(0.1, 512, 10, 512),
        #(0.1, 512, 20, 512),
        #(0.1, 256, 15, 128),

        #(0.1,       16284,       15.0,    264 ),
        #(0.1,       16284,       15.0,    264),
        #(0.1,       65136,       15.0,    512),
        #(0.1,       32568,       15.0,    512),
        #(0.1,       16284,       15.0,    512),
        #(0.1,       8192,       15.0,    512),
        #(0.1,       4096,       15.0,    512),
        #(0.1,       2048,       15.0,    512),
        #(0.1,       1024,       15.0,    512),
        #(0.1,       512,       15.0,    512),
                          # (0.1,       1024),
                          # (0.1,       2048),
                          # (0.1,       4096),
                          # (0.1,       8192),
                          ]

    results = batch_experiments(
        hyper_param_keys, hyper_param_values, param_base,
        header, dir_name, softmaxfunc
    )
def bth_train_metric_triplet():
    from metric_triplet_loss import main as softmaxfunc
    from Param_Config import param_metric_triplet
    # dir_name = "logs/Market1501-ResNet50/metric_triplet"
    dir_name = "logs/Market1501-ResNet50/triplet"

    header = "try-"
    # resume = osp.join(osp.dirname(osp.abspath(__file__)), dir_name)
    # resume = resume + "/try-lr_0.001_features_128_/model_best.pth.tar"
    param_metric_triplet['dist_metric'] = 'euclidean'
    param_metric_triplet['epochs'] = 100
    param_metric_triplet['batch_size'] = 512
    param_metric_triplet['norm_train'] = True
    param_metric_triplet['norm_test'] = True
    hyper_param_keys =    ('lr',    'features',  'margin')
    hyper_param_values = [(0.001,       512,       1.1),
                          (0.001, 512, 1.2),
                          (0.001, 512, 1.3),
                          (0.001, 512, 1.4),
                          (0.001, 512, 1.5),
                          (0.001, 1024, 1.1),
                          (0.001, 1024, 1.2),
                          (0.001, 1024, 1.3),
                          (0.001, 1024, 1.4),
                          (0.001, 1024, 1.5),
                          # (0.01,        128),
                          # (0.1,         128),
                          # (0.001,       512),
                          # (0.01,        512),
                          ]

    results = batch_experiments(
        hyper_param_keys, hyper_param_values, param_metric_triplet,
        header, dir_name, softmaxfunc
    )


# def bth_train_triplet():




if __name__ == '__main__':
     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
     #bth_train_softmax()
     bth_train_softmax_mod()
    #bth_train_metric_triplet()