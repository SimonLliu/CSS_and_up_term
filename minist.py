import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
def create_mnist():
# hyper parameter
    input_size = 28 * 28 # image size of MNIST data
    num_classes = 10
    num_epochs = 10
    batch_size = 100
    learning_rate = 1e-3

# MNIST dataset
    train_dataset = dsets.MNIST(root = './data/mnist', #选择数据的根目录
                           train = True, # 选择训练集
                           transform = transforms.ToTensor(), #转换成tensor变量
                           download = True) # 不从网络上download图片
    test_dataset = dsets.MNIST(root = './data/mnist', #选择数据的根目录
                           train = False, # 选择训练集
                           transform = transforms.ToTensor(), #转换成tensor变量
                           download = True) # 不从网络上download图片
#加载数据
    data={}

    data['train'] = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)  # 将数据打乱
    data['test'] = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = True)
    print()
    return data, 10
if __name__ == '__main__':
    data, _=create_mnist()
    print(len(data['train']))