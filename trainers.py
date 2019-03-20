from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import numpy as np

from utils import AverageMeter
from evaluate.ranking import accuracy


def display(epoch, i, data_len, keys, values):
    string = "Epoch: [%d][%d/%d]\t" % (epoch, i, data_len)
    for key, value in zip(keys, values):
        stri = "%s %.3f (%.3f)\t" % (key, value.val, value.avg)
        string = string + stri
    print(string)


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        # self.loss_t = []
        # self.loss_inter = []

    def train(self, epoch, data_loader, optimizer, print_freq=1, method='1', CCS=False, up_term=False):
        self.model.train()

        batch_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        display_keys = ('batch_time', 'Loss', 'Prec')

        end = time.time()
        if method=='3':
            for i, (imgs,pids) in enumerate(data_loader):
                inputs = Variable(imgs.cuda())
                targets = Variable(pids.cuda())
                _, outputs, loss_a, loss_up = self.model(inputs, targets)
                #print(outputs)
                loss = self.criterion(outputs, targets)
                #print('1',loss)
                loss_a=torch.sum(loss_a)
                loss_up=torch.sum(loss_up)
                #print('loss_temp:', loss)
                #print('loss_a', loss_a)
                #print(loss)
                if CCS and not up_term:
                    #print('1')
                    loss = loss+loss_a
                if up_term and not CCS:
                    #print('2')
                    loss = loss+loss_up
                if up_term and  CCS:
                    #print('3')
                    loss = loss+loss_a+loss_up
                prec, = accuracy(outputs.data, targets.data)

                losses.update(loss.data, targets.size(0))
                #print('2',loss)
                # precisions.update(prec[0], targets.size(0))

                optimizer.zero_grad()
                loss.backward(torch.ones_like(loss))
                optimizer.step()
                #print(loss)

                batch_time.update(time.time() - end)
            return loss, loss_a, loss_up
        if method=='2':
            for i, (imgs,pids) in enumerate(data_loader):
                inputs = Variable(imgs.cuda())
                targets = Variable(pids.cuda())
                _, outputs, auxiliary_ouput = self.model(inputs)
                loss = self.criterion(outputs, targets)
                prec, = accuracy(outputs.data, targets.data)

                losses.update(loss.data, targets.size(0))
                # precisions.update(prec[0], targets.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                '''
                if (i + 0) % print_freq == 0:
                    display(epoch, i + 1, len(data_loader), keys=display_keys,
                            values=(batch_time, losses, precisions))
                '''
        if method=='1':
            for i, (imgs, _, pids, _) in enumerate(data_loader):
                inputs = Variable(imgs.cuda())
                targets = Variable(pids.cuda())
                _, outputs, auxiliary_ouput = self.model(inputs)
                loss = self.criterion(outputs, targets)
                prec, = accuracy(outputs.data, targets.data)

                losses.update(loss.data, targets.size(0))
                #precisions.update(prec[0], targets.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)

                if (i + 0) % print_freq == 0:
                    display(epoch, i + 1, len(data_loader), keys=display_keys,
                            values=(batch_time, losses, precisions))

        return loss


class MetricTripletTrainer(object):
    def __init__(self, model, norm_train, criterion):
        super(MetricTripletTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.norm_feature = norm_train
        # self.loss_t = []
        # self.loss_inter = []

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        display_keys = ('batch_time', 'Loss', 'Prec')
        loss=0
        end = time.time()
        for i, (imgs, _, pids, _) in enumerate(data_loader):
            inputs = Variable(imgs.cuda())
            targets = Variable(pids.cuda())
            outputs, _, auxiliary_ouput = self.model(inputs, Norm_test=self.norm_feature)
            loss, prec = self.criterion(outputs, targets)
            # prec, = accuracy(outputs.data, targets.data)

            losses.update(loss.data, targets.size(0))
            precisions.update(prec.data, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)

            if (i + 0) % print_freq == 0:
                display(epoch, i + 1, len(data_loader), keys=display_keys,
                        values=(batch_time, losses, precisions))

