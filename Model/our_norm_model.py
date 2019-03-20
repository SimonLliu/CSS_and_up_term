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


def norm_detach(input, is_detach):
    if is_detach:
        input_ = input / input.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12).expand_as(input).detach()
    else:
        input_ = F.normalize(input)

    return input_


def W_softplus(input, is_softplus):
    if is_softplus:
        return F.softplus(input)
    else:
        return input


class Norm_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Norm_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, F_norm, W_norm, W_softp, is_detach):

        input_ = norm_detach(input, is_detach) if F_norm else input

        W = self.weight
        # W = W_softplus(W, W_softp)

        W_ = norm_detach(W, is_detach) if W_norm else W

        return F.linear(input_, W_, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Norm_ResNet50(nn.Module):

    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=0, dropout=0, W_std=0.001,
                 F_norm=True, W_norm=True, W_softp=False, is_norm_detach=True, is_gaussian_detach=True,
                 num_classes=0, gamma=1.0, sigma=1.0, fixed = False):
        super(Norm_ResNet50, self).__init__()

        self.gamma = gamma
        self.sigma = sigma
        self.F_norm = F_norm
        self.W_norm = W_norm
        self.W_softp = W_softp
        self.W_std = W_std
        self.is_norm_detach = is_norm_detach
        self.is_gaussian_detach = is_gaussian_detach
        self.fixed=fixed

        self.resnet34 = resnet34(pretrained=pretrained, cut_at_pooling=cut_at_pooling,
                                dropout=dropout, num_features=num_features)

        # self.resnet50 = sphere20()
        self.num_classes = num_classes
        self.num_features = num_features if num_features > 0 else 2048

        if self.F_norm or self.W_norm:
            self.classifier = Norm_Linear(in_features=self.num_features,
                                          out_features=num_classes,
                                          bias=False)
            # init.normal_(self.classifier.weight, std=self.W_std)
        else:
            self.classifier = nn.Linear(in_features=self.num_features, out_features=num_classes)
            # init.normal_(self.classifier.weight, std=self.W_std)
            # init.constant_(self.classifier.bias, 0)

    def forward(self, inputs, Norm_test=True):
        auxiliary_ouput = {}
        x = self.resnet34(inputs)

        feature = F.normalize(x) if Norm_test else x

        if self.F_norm or self.W_norm:
            # x_standard = self.classifier(x, False, False, False, False)
            # norm_weight = F.normalize(self.classifier.weight)
            # cosine_m = torch.mm(norm_weight, norm_weight.t())
            # cosine_m = cosine_m.mul(torch.ones_like(cosine_m) - Variable(torch.eye(cosine_m.size(0)).cuda()))
            # auxiliary_ouput['w_cosine_std'] = torch.std(cosine_m).view(1, 1)
            # auxiliary_ouput['w_cosine_mean'] = torch.sum(cosine_m.clamp(min=0.0).div(cosine_m.size(0))).view(1, 1)

            # auxiliary_ouput['x_standard'] = x_standard
            # auxiliary_ouput['gaussian_similarity'] = self.gaussian_similar(
            #     input=x,
            #     is_detach=self.is_gaussian_detach
            # )
            cosine = self.classifier(x, self.F_norm, self.W_norm, self.W_softp, self.is_norm_detach)
            auxiliary_ouput['cosine'] = cosine
            output = torch.mul(cosine, self.gamma)
        else:
            output = self.classifier(x)

        return feature, output, auxiliary_ouput

    def gaussian_similar(self, input, is_detach):
        norm_input = self.norm_detach(input=input, is_detach=is_detach)
        norm_weight = self.norm_detach(input=self.classifier.weight, is_detach=is_detach)

        m, n = norm_input.size(0), norm_weight.size(0)
        dist = torch.pow(norm_input, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(norm_weight, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, norm_input, norm_weight.t())

        similarity = torch.exp(dist / (-2.0 * self.sigma))
        return similarity


