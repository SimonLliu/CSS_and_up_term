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
from utils import Logger, save_checkpoint, load_checkpoint, adjust_learning_rate
from Model import our_norm_model as models
from evaluate.evaluators import Evaluator
from trainers import BaseTrainer as Trainer
from torchsummary import summary


def get_data(name, split_id, data_dir, height, width, batch_size, workers,
             combine_trainval, identity_least_imgnum=0):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes= (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)
    num_classes_query = dataset.num_query_ids
    num_classes_gallery = dataset.num_gallery_ids
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
        shuffle=True, pin_memory=True, drop_last=True)

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

    query_loader = DataLoader(
        Preprocessor(dataset.query, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, num_classes_query, num_classes_gallery, train_loader, val_loader, test_loader, query_loader, gallery_loader


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
    if kwargs['height'] is None or kwargs['width'] is None:
        kwargs['height'], kwargs['width'] = (144, 56) if kwargs['arch'] == 'inception' else \
                                  (256, 128)
    dataset, num_classes, num_classes_query, num_classes_gallery, train_loader, val_loader, test_loader, query_loader, gallery_loader = \
        get_data(kwargs['dataset'], kwargs['split'], kwargs['data_dir'], kwargs['height'],
                 kwargs['width'], kwargs['batch_size'], kwargs['workers'],
                 kwargs['combine_trainval'])
    print("train_set_classes", num_classes)
    print("query_set_classes", num_classes_query)
    print("gallery_set_classes", num_classes_gallery)
    # Create model
    model = models.Norm_ResNet50(num_features=kwargs['features'], dropout=kwargs['dropout'], W_std=kwargs['W_std'],
                                 F_norm=kwargs['F_norm'], W_norm=kwargs['W_norm'],
                                 is_norm_detach=False, is_gaussian_detach=False,
                                 num_classes=num_classes, gamma=kwargs['gamma'])

    #summary(model, (3, kwargs['height'], kwargs['width']))
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

    # Evaluator
    evaluator = Evaluator(model, norm_test=kwargs['norm_test'])
    if kwargs['evaluate']:
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        temp = sys.stdout
        temp.close()
        sys.stdout = temp.console
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()
    print(hasattr(model.module, 'base'))
    # Optimizer
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=kwargs['lr'],
                                momentum=kwargs['momentum'],
                                weight_decay=kwargs['weight_decay'],
                                nesterov=True)

    # Trainer
    trainer = Trainer(model, criterion)
    save_checkpoint({
        'state_dict': model.module.state_dict(),
        'epoch': 1,
        'best_mAP': best_mAP,
    }, False, fpath=osp.join(kwargs['logs_dir'], 'initial_checkpoint.pth.tar'))
    best_epoch = 0
    begain_time = time.time()
    if not kwargs['pretrained']:
        for epoch in range(start_epoch, kwargs['epochs']):
            adjust_learning_rate(optimizer, epoch, kwargs['lr'], kwargs['step_size'])
            trainer.train(epoch, train_loader, optimizer, kwargs['print_freq'])

            acc=evaluator.count_acc(train_loader)
            print("Accuracy: %.4f" % acc.item())

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
                'acc': acc.item(),
            }, is_best, fpath=osp.join(kwargs['logs_dir'], 'checkpoint.pth.tar'))

            print("time: ", time.time() - begain_time)
            print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{} (epoch: {:3d})\n'.
                format(epoch, mAP, best_mAP, ' *' if is_best else '', best_epoch))

    print('+++++++++++test_start++++++++++')
    #load model
    #start_epoch = best_mAP = kwargs['epochs']
    model_test = models.Norm_ResNet50(num_features=kwargs['features'], dropout=kwargs['dropout'],
                                      W_std=kwargs['W_std'],
                                      F_norm=kwargs['F_norm'], W_norm=kwargs['W_norm'],
                                      is_norm_detach=False, is_gaussian_detach=False,
                                      num_classes=num_classes, gamma=kwargs['gamma'],fixed=True)

    checkpoint = load_checkpoint(osp.join(kwargs['logs_dir'], 'model_best.pth.tar'))
    model_test.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['state_dict'])

    print(checkpoint['epoch'])
    print(checkpoint['acc'])
    model_test = nn.DataParallel(model_test).cuda()

    evaluator_2 = Evaluator(model_test, norm_test=kwargs['norm_test'])
    acc_query = evaluator_2.count_acc(query_loader)
    acc_train = evaluator_2.count_acc(train_loader)
    print("Accuracy_train: %.4f" % acc_train.item())
    print("Accuracy_query: %.4f" % acc_query.item())
    #revise model
    #model_test.classifier = nn.Linear(2048, num_classes)
    trainer_test = Trainer(model_test, criterion)
    #optimizer_t = torch.optim.SGD(filter(lambda p: p.requires_grad, model_test.parameters()), lr=kwargs['lr'],
    #                              momentum=kwargs['momentum'],
    #                              weight_decay=kwargs['weight_decay'],
    #                              nesterov=True)
    optimizer_t = torch.optim.SGD(model_test.parameters(), lr=kwargs['lr'],
                                  momentum=kwargs['momentum'],
                                  weight_decay=kwargs['weight_decay'],
                                  nesterov=True)
    #retrain model
    for epo in range(0, 10):
        adjust_learning_rate(optimizer_t, epo, kwargs['lr'], kwargs['step_size'])
        trainer_test.train(epo, train_loader, optimizer_t, kwargs['print_freq'])
        if epo % 1 == 0:
            acc_query=evaluator_2.count_acc(train_loader)
            print("Accuracy_query: %.4f" % acc_query.item())

    #test model
    acc_gallery = evaluator_2.count_acc(gallery_loader)
    print("Accuracy_gallery: %.4f" % acc_gallery.item())

    print('+++++++++++test_end++++++++++++')

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(kwargs['logs_dir'], 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    _, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, is_cmc=False)
    # print("best mAP: %.4f (epoch: %d)" % (best_mAP, best_epoch))
    temp = sys.stdout
    temp.close()
    sys.stdout = temp.console
    return best_mAP