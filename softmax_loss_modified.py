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
from Model import norm_model as model_norm
from evaluate.evaluators import Evaluator
from trainers import BaseTrainer as Trainer
from torchsummary import summary
from dataset_market import create_market
from cifar10 import create_cifar10



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
    data_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data'),
    print(data_dir)
    data, num_classes = create_market(data_dir[0], kwargs['height'], kwargs['width'], kwargs['batch_size'])
    #data, num_classes = create_cifar10()
    # Create model
    '''
    model = models.Norm_ResNet50(num_features=kwargs['features'], dropout=kwargs['dropout'], W_std=kwargs['W_std'],
                                 F_norm=kwargs['F_norm'], W_norm=kwargs['W_norm'],
                                 is_norm_detach=False, is_gaussian_detach=False,
                                 num_classes=num_classes, gamma=kwargs['gamma'])
    '''
    model = model_norm.Norm_ResNet50(num_features=kwargs['features'], dropout=kwargs['dropout'], CCS=kwargs['CCS'],
                                     up_term=kwargs['up_term'],
                                    is_norm_detach=True, is_gaussian_detach=False,
                                    num_classes=num_classes, gamma=kwargs['gamma'])
    #summary(model, (3, kwargs['height'], kwargs['width']))
    # Load from checkpoint
    start_epoch = best_mAP = 0
    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model, norm_test=kwargs['norm_test'])

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
    begin_time = time.time()
    t = 0
    accuary_train = []
    accuary_test = []
    losses = []
    losses_alpha=[]
    losses_up_term=[]
    losses_crossentropy=[]
    acc_train_max=0
    acc_test_max = 0
    print('+++++++++++train_start++++++++++')
    params = model.state_dict()
    #for k, v in params.items():
    #   print(k)  # 打印网络中的变量名
    #print(model)
    for epoch in range(start_epoch, kwargs['epochs']):
        print(epoch)
        #params = model.state_dict()
        #print(params['module.classifier.weight'])
        adjust_learning_rate(optimizer, epoch, kwargs['lr'], kwargs['step_size'])
        #loss=trainer.train(epoch, data['train'], optimizer, kwargs['print_freq'],method='2')
        loss, loss_a, loss_up = trainer.train(epoch, data['train'], optimizer, kwargs['print_freq'], method='3', CCS=kwargs['CCS'], up_term=kwargs['up_term'])
        #print(loss)
        #print(loss_a)
        loss_alpha = loss_a / kwargs['gamma']
        loss_crossentropy = loss - loss_a - loss_up
        loss_up_term = loss_up

        print("loss: ",loss.item())
        print("loss_alpha: ", loss_alpha.item())
        print("loss_up_term: ", loss_up_term.item())
        print("loss_crossentropy: ", loss_crossentropy.item())
        losses.append(loss.item())

        losses_alpha.append(loss_alpha.item())
        losses_up_term.append(loss_up_term.item())
        losses_crossentropy.append(loss_crossentropy.item())

        #acc_train=evaluator.count_acc(data['train'], method='2')
        acc_train = evaluator.count_acc(data['train'], method='3')
        print("Accuracy_train: %.6f" % acc_train.item())
        accuary_train.append(acc_train.item())
        if acc_train.item() > acc_train_max: acc_train_max = acc_train.item()

        #acc_test = evaluator.count_acc(data['test'], method='2')
        acc_test = evaluator.count_acc(data['test'], method='3')
        print("Accuracy_test: %.6f" % acc_test.item())
        accuary_test.append(acc_test.item())
        if acc_test.item() > acc_test_max: acc_test_max = acc_test.item()

        t = time.time() - begin_time
        print("time: %.4f" % t)
    import matplotlib.pyplot as plt
    def picture(accuary_train, accuary_test, losses, loss_crossentropy, loss_alpha, loss_up_term,
                lr, gamma, feature, batch_size, acc_test_max, CCS=False, up_term=False):
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

        ax1.plot(losses)
        ax1.set_title('losses')
        ax1.set_ylabel("Losses")
        ax1.set_xlabel("Epoch")
        #ax1.legend()

        ax2.plot(loss_crossentropy)
        ax2.set_title('loss_crossentropy')
        ax2.set_ylabel("loss_crossentropy")
        ax2.set_xlabel("Epoch")
        #ax2.legend()

        ax3.plot(loss_alpha)
        ax3.set_title('loss_alpha')
        ax3.set_ylabel("loss_alpha")
        ax3.set_xlabel("Epoch")
        #ax3.legend()

        ax4.plot(loss_up_term)
        ax4.set_title('loss_up_term')
        ax4.set_ylabel("loss_up_term")
        ax4.set_xlabel("Epoch")
        # ax3.legend()
        ax5.plot(accuary_train)
        ax5.set_title('Accuracy'+'_'+str(lr)+'_'+str(gamma)+'_'+str(feature)+'_'+str(batch_size))
        ax5.set_ylabel('Accuracy_train')
        ax5.set_xlabel('Epoch')

        ax6.plot(accuary_test)
        ax6.set_title(str(acc_test_max * 100))
        ax6.set_ylabel('Accuracy_test')
        ax6.set_xlabel('Epoch')
        if CCS and not up_term:
            plt.savefig('./logs/' + str(lr) + '_' + str(gamma) + '_' + str(feature) + '_' + str(
                batch_size) + '_CCS' + '.jpg')
        if up_term and not CCS:
            plt.savefig('./logs/'+str(lr)+'_'+str(gamma)+'_'+str(feature)+'_'+str(batch_size)+'_up_term'+'.jpg')
        if up_term and CCS:
            plt.savefig('./logs/' + str(lr) + '_' + str(gamma) + '_' + str(feature) + '_' + str(
                batch_size) + '_up+CCS' + '.jpg')
        if not CCS and not up_term:
            plt.savefig('./logs/' + str(lr) + '_'  + str(feature) + '_' + str(
                batch_size) + '.jpg')
    picture(accuary_train, accuary_test, losses, losses_crossentropy, losses_alpha, losses_up_term,
                kwargs['lr'], kwargs['gamma'], kwargs['features'], kwargs['batch_size'], acc_test_max,
                CCS=kwargs['CCS'], up_term=kwargs['up_term'])

    print('+++++++++++train_end+++++++++++')
    print('+++++++++++test_start++++++++++')

    '''
    acc_test = evaluator.count_acc(data['test'], method='2')
    acc_train = evaluator.count_acc(data['train'], method='2')
    print("Accuracy_train: %.6f" % acc_train.item())
    print("Accuracy_test: %.6f" % acc_test.item())
    '''
    for key in ['lr', 'gamma', 'batch_size', 'features']:
        key1 = key
        if key == 'gamma':
            key1 = 'lambda'
        print(key1 + ":", kwargs[key])
    print("Accuracy_train_max: %.6f" % acc_train_max)
    print("Accuracy_test_max: %.6f" % acc_test_max)

    print('+++++++++++test_end++++++++++++')

    temp = sys.stdout
    temp.close()
    sys.stdout = temp.console
    return 0