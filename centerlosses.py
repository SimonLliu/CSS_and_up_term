from __future__ import absolute_import

import torch
import math
from torch.nn.parameter import Parameter
from torch import nn
from torch.autograd import Variable


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(torch.autograd.Variable(classes.expand(batch_size, self.num_classes)))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss

        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


class HingeLoss(nn.Module):
    # loss formulation:
    #  sum_i[max{0, s_{i,y_i}}+ sum_{j not y_i}[max{0, s_{i,j}}]]

    def __init__(self, max_margin=-25.0, min_margin=50.0, lamda=1.0, inv_lamda=1.0):
        super(HingeLoss, self).__init__()
        self.max_margin = max_margin
        self.min_margin = min_margin
        self.lamda = lamda
        self.inv_lamda = inv_lamda

    def forward(self, inputs, targets):
        n = targets.size(0)
        c = inputs.size(1)

        mask_ = torch.ones(n, c).mul(self.inv_lamda).cuda()
        mask_negative = torch.ones(n, c).cuda()
        margin_ = torch.ones(n, c).mul(self.max_margin).cuda()

        for i in range(n):
            mask_[i][targets.data[i]] = -self.lamda
            mask_negative[i][targets.data[i]] = 0
            margin_[i][targets.data[i]] = self.min_margin

        loss_m = torch.mul(Variable(mask_), inputs-Variable(margin_)).clamp(min=0.0)
        loss_inter = torch.mul(loss_m, Variable(mask_negative))
        return loss_m.sum() / n, loss_inter.sum() / n


class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, gamma=30.0, margin=0.40):
        super(MarginCosineProduct, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        self.s = gamma
        self.m = margin
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform(self.weight)

    def forward(self, cosine, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = Variable(torch.zeros(cosine.size()))
        one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class MetricTripletLoss(nn.Module):
    def __init__(self, feature_size, margin=0):
        super(MetricTripletLoss, self).__init__()
        self.margin = margin
        self.M_ = Parameter(torch.Tensor(feature_size, feature_size))
        stdv = 1. / math.sqrt(self.M_.size(1))
        self.M_.data.uniform_(-stdv, stdv)
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def distance(self, inputs):
        dist = torch.mm(inputs, self.M_).mm(inputs.t())
        return dist

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist = self.distance(inputs)  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
