import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

def create_cifar10():
    show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

# 定义对数据的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ])

# 训练集
    trainset = tv.datasets.CIFAR10(
        root='./data/cifar10',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = t.utils.data.DataLoader(
        trainset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
    )

# 测试集
    testset = tv.datasets.CIFAR10(
        root='./data/cifar10',
        train=False,
        download=True,
        transform=transform,
    )
    testloader = t.utils.data.DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )
    data={}
    data['train']=trainloader
    data['test']=testloader

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return data, 10
if __name__ == '__main__':
    data, _=create_cifar10()
    print(len(data['train']))
