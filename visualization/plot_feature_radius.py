# from Extract_raw_data_feature import extract_raw_data_feature
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import os

from torch import nn
from torch.backends import cudnn

import Model.our_norm_model as models
from evaluate.evaluators import extract_features
from softmax_loss import get_data
from utils import load_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'


def reallocate_features(features, labels, num_classes):
    print("Re-allocate features...")
    class_features = [[] for _ in range(num_classes)]
    n_samples = features.shape[0]
    for i in range(n_samples):
        label = int(labels[i])
        class_features[label].append(features[i, :])
    return class_features


def compute_class_radius(class_features):
    print("compute class radius...")
    num_classes = len(class_features)
    class_radius = np.zeros((num_classes, ))
    for i in range(num_classes):
        one_class_features = class_features[i]
        f_m = np.vstack(one_class_features)
        m, n = f_m.shape
        dist = np.sum(np.power(f_m, 2), axis=1, keepdims=True).repeat(m, axis=1)
        dis_m = dist + dist.T - 2 * np.matmul(f_m, f_m.T)
        class_radius[i] = np.max(dis_m)
        # class_radius[i] = np.min(dis_m + np.eye(m)*10000)
    return class_radius


def compute_class_volume(class_features):
    import math
    print("compute class volume...")
    num_classes = len(class_features)
    class_volume = np.zeros((num_classes,))
    total_volume = []
    for i in range(num_classes):
        one_class_features = class_features[i]
        f_m = np.vstack(one_class_features)
        f_v = np.max(f_m, axis=0) - np.min(f_m, axis=0)
        total_volume.append(np.max(f_m, axis=0))
        total_volume.append(np.min(f_m, axis=0))
        class_volume[i] = np.prod(f_v)
        if math.isinf(class_volume[i]):
            print("is inf")
            print(f_m)
            print(f_v)
            print(f_v.shape)
            print(prod(f_v))
            exit()

    total_volume = np.vstack(total_volume)
    total_volume = np.prod(np.max(total_volume, axis=0) - np.min(total_volume, axis=0))
    return class_volume, total_volume

def prod(x):
    k = 1.0
    for i in range(len(x)):
        k *= x[i]
    return k


def extract_raw_data_feature(height, width, data_dir, arch='resnet50', batch_size=128, pretrained=True, resume=None,
                             F_norm=False, W_norm=False, embedding_size=2048, dropout=0, norm_test=False,
                             dataset='market1501', split=0, workers=4, combine_trainval=False):
    cudnn.benchmark = True
    if height is None or width is None:
        height, width = (144, 56) if arch == 'inception' else \
            (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(dataset, split, data_dir, height, width,
                 batch_size, workers, combine_trainval)

    model = models.Norm_ResNet50(pretrained=pretrained,
                                 num_features=embedding_size, dropout=dropout,
                                 F_norm=F_norm, W_norm=W_norm,
                                 num_classes=num_classes)
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        model.load_state_dict(checkpoint['state_dict'])

    model = nn.DataParallel(model).cuda()
    features, labels = None, None
    # features, labels = extract_features(model, train_loader, print_freq=10, norm_test=norm_test)
    weights = model.module.classifier.weight.data.cpu().numpy()

    return features, labels, weights


def plot_volume_hist(features, labels, title):
    num_classes = int(np.max(labels) + 1)
    class_features = reallocate_features(features, labels, num_classes)

    # class_radius = compute_class_radius(class_features)
    class_volume, total_volume = compute_class_volume(class_features)

    plt.figure()
    plt.hist(class_volume, 100)
    plt.title(title)
    print('total_volume', total_volume)
    plt.show()


def plot_weight_offset(w1, w2, title):
    from sklearn import preprocessing
    w1_norm = preprocessing.normalize(w1)
    w2_norm = preprocessing.normalize(w2)
    w_cos = np.sum(w1_norm * w2_norm, axis=1)
    w_degree = np.arccos(w_cos)*180/np.pi
    plt.figure()
    plt.hist(w_degree, 100)
    plt.title(title)
    plt.show()


if __name__ == '__main__':

    # ================= parameter ====================
    height, width = (256, 128)
    batch_size = 256
    embedding_size = 512
    norm_test = False

    resume = r"/home/xldai/SharedSSD/daixili/PersonReID/NMM-reid/logs/Market1501-ResNet50/" \
             r"softmax_baseline/with-initial-ckpt-lr_0.100_features_512_/initial_checkpoint.pth.tar"  # checkpoint.pth.tar, model_best.pth.tar

    resume1 = r"/home/xldai/SharedSSD/daixili/PersonReID/NMM-reid/logs/Market1501-ResNet50/" \
             r"softmax_baseline/with-initial-ckpt-lr_0.100_features_512_/checkpoint.pth.tar"  # checkpoint.pth.tar, model_best.pth.tar

    F_norm, W_norm = False, False
    data_dir = osp.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'data')
    # =================================================

    _, _, weights_init = extract_raw_data_feature(
        height, width, data_dir, batch_size=batch_size,
        pretrained=False, resume=resume,
        F_norm=F_norm, W_norm=W_norm,
        embedding_size=embedding_size, norm_test=norm_test, combine_trainval=True)

    _, _, weights_end = extract_raw_data_feature(
        height, width, data_dir, batch_size=batch_size,
        pretrained=False, resume=resume1,
        F_norm=F_norm, W_norm=W_norm,
        embedding_size=embedding_size, norm_test=norm_test, combine_trainval=True)

    plot_weight_offset(weights_init, weights_end, title='weight_offset_before_after_train')

    # plot_hist(features, labels, title="market1501_W0.001_class_volume_for_train_set")
