from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['ResNet50']

class ResNet50(nn.Module):
    def __init__(self, num_classes, loss = {'center_loss'}, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50  = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.fc   = nn.Linear(2048, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.feat_dim = 512

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        f = self.fc(x)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'center_loss'}:
            return y, f
        else:
            raise KeyError('Unsupported loss:{}'.format(self.loss))



