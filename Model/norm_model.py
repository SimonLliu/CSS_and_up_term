from __future__ import absolute_import

# from .inception import *
from .resnet_our import *
import torch
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy


def norm_detach(input, is_detach):
    if is_detach:
        input_ = input / input.norm(p=2, dim=0, keepdim=True).clamp(min=1e-12).expand_as(input).detach()
    else:
        input_ = F.normalize(input)

    return input_
def norm_detach_up(input, is_detach):
    if is_detach:
        input_ = input.norm(p=2, dim=0, keepdim=True).clamp(min=1e-12).expand_as(input).detach()
    else:
        input_ = F.normalize(input)


def W_softplus(input, is_softplus):
    if is_softplus:
        return F.softplus(input)
    else:
        return input


class Norm_ResNet50(nn.Module):

    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=0, dropout=0,  CCS=True, up_term=True,
                 is_norm_detach=True, is_gaussian_detach=True,
                 num_classes=0, gamma=1.0, sigma=1.0, fixed = False):
        super(Norm_ResNet50, self).__init__()

        self.gamma = gamma
        self.sigma = sigma
        self.CCS = CCS
        self.up_term=up_term

        self.is_norm_detach = is_norm_detach
        self.is_gaussian_detach = is_gaussian_detach
        self.fixed=fixed
        #self.loss_up = torch.Tensor(0)




        self.resnet34 = resnet34(pretrained=pretrained, cut_at_pooling=cut_at_pooling,
                                dropout=dropout, num_features=num_features)

        # self.resnet50 = sphere20()
        self.num_classes = num_classes
        self.num_features = num_features if num_features > 0 else 2048
        self.weight = Parameter(torch.Tensor(self.num_features, num_classes))

        self.classifier = nn.Linear(in_features=self.num_features, out_features=num_classes)
            # init.normal_(self.classifier.weight, std=self.W_std)
            # init.constant_(self.classifier.bias, 0)

    def forward(self, inputs, targets, Norm_test=True):
        auxiliary_ouput = {}
        x = self.resnet34(inputs)

        feature = F.normalize(x) if Norm_test else x
        loss_a = torch.Tensor(0).cuda()
        loss_up = torch.Tensor(0).cuda()
        #print(self.classifier.weight)
        #print(self.classifier.weight.size())
        #print(self.weight.size())
        if self.CCS:
            #W=self.weight
            W = self.classifier.weight
            #print(W)
            W_ = norm_detach(W, self.is_norm_detach) if self.CCS else W
            input_ = norm_detach(x, self.is_norm_detach) if self.CCS else x
            #print(W_.size())
            #print(input_.size(0))
            temp=torch.mm(input_, W_.t())
            #temp = torch.mm(input_, W_)
            #print(temp.size())
            #batch_size=len(targets.cpu().numpy().tolist())
            #print(batch_size)
            class_num=1501
            #one_hot_label = torch.zeros(batch_size, class_num).scatter_(1, targets.cpu(), 1)
            index = torch.eye(class_num)
            one_hot_label = torch.index_select(index, dim=0, index=targets.cpu()).cuda()
            #print(one_hot_label.size())
            #loss_a=-1*torch.sum(torch.mul(temp, one_hot_label))/input_.size(0) #normalize
            loss_a = -1 * torch.sum(torch.mul(temp, one_hot_label))  #not normalize
            #print(loss_a)

        base_set=751
        low_shot_set=750
        #alpha_temp=torch.Tensor(0).cuda()

        if self.up_term:
            W = self.classifier.weight
            #print(W)
            W_k_base=W[0:base_set,:].norm(p=2, dim=1, keepdim=True)
            #print(W[:,0:base_set].size())
            #print(W_k_base.size())
            W_k_base=torch.mul(W_k_base, W_k_base)
            alpha_temp=torch.sum(W_k_base)
            alpha=1 / base_set * alpha_temp
            #print(W.size())
            #print(W[:, base_set:base_set+low_shot_set])
            #print(alpha)
            W_k_low = W[base_set:base_set+low_shot_set,: ].norm(p=2, dim=1, keepdim=True)

            W_k_low = torch.mul(W_k_low, W_k_low)-alpha

            alpha_temp = torch.sum(W_k_low)
            #print(alpha_temp)
            loss_up = (1 / low_shot_set) * alpha_temp

            #print(loss_up)
        #print(loss_up)
        output = self.classifier(x)
        #print('1.5',loss_a)
        return feature, output, self.gamma * loss_a, loss_up

    def gaussian_similar(self, input, is_detach):
        norm_input = self.norm_detach(input=input, is_detach=is_detach)
        norm_weight = self.norm_detach(input=self.classifier.weight, is_detach=is_detach)

        m, n = norm_input.size(0), norm_weight.size(0)
        dist = torch.pow(norm_input, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(norm_weight, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, norm_input, norm_weight.t())

        similarity = torch.exp(dist / (-2.0 * self.sigma))
        return similarity


