import os.path as osp
import time
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import reid_datasets as datasets
from reid_datasets import transforms as T
from reid_datasets.preprocessor import Preprocessor
from utils import Logger, save_checkpoint, load_checkpoint, adjust_learning_rate, RandomIdentitySampler
from Model import our_norm_model as models
from evaluate.evaluators import Evaluator
from trainers import MetricTripletTrainer as Trainer
from losses import MetricTripletLoss, TripletLoss
from dist_metric import DistanceMetric


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(**kwargs):

    cudnn.benchmark = True

    # Redirect print to both console and log file
    if kwargs['logs_dir'] is not None:
        sys.stdout = Logger(osp.join(kwargs['logs_dir'], 'log.txt'))

    # print configure
    print("+============ parameter ==================+")
    for key in kwargs.keys():
        print(key + ":", kwargs[key])
    print("+======================================+")

    # Create data loaders
    assert kwargs['num_instances'] > 1, "num_instances should be greater than 1"
    assert kwargs['batch_size'] % kwargs['num_instances'] == 0, \
        'num_instances should divide batch_size'
    if kwargs['height'] is None or kwargs['width'] is None:
        kwargs['height'], kwargs['width'] = (144, 56) if kwargs['arch'] == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(kwargs['dataset'], kwargs['split'], kwargs['data_dir'], kwargs['height'],
                 kwargs['width'], kwargs['batch_size'], kwargs['num_instances'], kwargs['workers'],
                 kwargs['combine_trainval'])

    # Create model
    model = models.Norm_ResNet50(num_features=kwargs['features'], dropout=kwargs['dropout'], W_std=kwargs['W_std'],
                                 F_norm=False, W_norm=False,
                                 is_norm_detach=False, is_gaussian_detach=False,
                                 num_classes=num_classes, gamma=kwargs['gamma'])

    # Load from checkpoint
    start_epoch = best_mAP = 0
    if kwargs['resume']:
        checkpoint = load_checkpoint(kwargs['resume'])
        # filter_unneed_key(model, checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))
    model = nn.DataParallel(model).cuda()

    # Criterion
    if kwargs['dist_metric'] == 'tmm':
        criterion = MetricTripletLoss(feature_size=kwargs['features'],
                                      margin=kwargs['margin']).cuda()
        metric = DistanceMetric(algorithm=kwargs['dist_metric'], metric_triplet=criterion)
    else:
        criterion = TripletLoss(margin=kwargs['margin']).cuda()
        metric = DistanceMetric(algorithm=kwargs['dist_metric'])

    # # Distance metric
    # metric = DistanceMetric(algorithm=kwargs['dist_metric'], metric_triplet=criterion)

    # Evaluator
    evaluator = Evaluator(model, norm_test=kwargs['norm_test'])
    if kwargs['evaluate']:
        metric.train(model, train_loader)
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
        return

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'],
                                 weight_decay=kwargs['weight_decay'])

    # Trainer
    trainer = Trainer(model, kwargs['norm_train'], criterion)
    save_checkpoint({
        'state_dict': model.module.state_dict(),
        'epoch': 1,
        'best_mAP': best_mAP,
    }, False, fpath=osp.join(kwargs['logs_dir'], 'initial_checkpoint.pth.tar'))
    best_epoch = 0
    begain_time = time.time()
    for epoch in range(start_epoch, kwargs['epochs']):
        adjust_learning_rate(optimizer, epoch, kwargs['lr'], kwargs['step_size'])
        trainer.train(epoch, train_loader, optimizer, kwargs['print_freq'])

        if epoch < kwargs['start_save']:
            continue
        # _, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, no_cmc=True)
        _, mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, is_cmc=True)

        is_best = mAP >= best_mAP
        if is_best:
            best_epoch = epoch
        best_mAP = max(mAP, best_mAP)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_mAP': best_mAP,
        }, is_best, fpath=osp.join(kwargs['logs_dir'], 'checkpoint.pth.tar'))
        print("time: ", time.time() - begain_time)
        print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{} (epoch: {:3d})\n'.
              format(epoch, mAP, best_mAP, ' *' if is_best else '', best_epoch))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(kwargs['logs_dir'], 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    _, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric=metric, is_cmc=True)
    # print("best mAP: %.4f (epoch: %d)" % (best_mAP, best_epoch))
    temp = sys.stdout
    temp.close()
    sys.stdout = temp.console
    return best_mAP