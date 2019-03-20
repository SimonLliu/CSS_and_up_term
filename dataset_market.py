import torchvision.datasets as td
from torch.backends import cudnn
from torch.utils.data import DataLoader
import os.path as osp
import time
import numpy as np
import sys
from reid_datasets import transforms as T
from reid_datasets.preprocessor import Preprocessor

def create_market(data_dir, height, width, batch_size):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ]),
        'test': T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
        ]),
    }
    root = data_dir+'/'+'market'

    image_datasets = {x: td.ImageFolder(osp.join(root, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}
    print(len(image_datasets['train'].classes))
    print(len(image_datasets['test'].classes))
    #print(image_datasets['test'].classes)
    #print(image_datasets['train'].class_to_idx)
    #print(image_datasets['test'].class_to_idx)
    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    return dataloaders, len(image_datasets['train'].classes)

if __name__ == '__main__':
    data_dir=osp.join(osp.dirname(osp.abspath(__file__)), 'data'),
    print(data_dir)
    height = 256
    width = 128
    dataloader, num_class=create_market(data_dir[0], height, width, 512)
    print(num_class)
    print(len(dataloader['train']))
    print(len(dataloader['test']))
    for i,(imgs,pids) in enumerate(dataloader['train']):
        print(pids)
        break
            #print(pids.size())
            #print(imgs.size())
    for i,(imgs,pids) in enumerate(dataloader['test']):
        print(pids)
        break
            #print(pids.size())
            #print(imgs.size())
