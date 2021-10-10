import os

import numpy as np
import torch

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def getdataloader(datatype,train_db_path, test_db_path, batch_size):
    # get transformations
    transform_train, transform_test = _getdatatransformsdb(datatype=datatype)
    n_classes = 0
    # Data loaders
    if datatype.upper() == "CIFAR10":
        print("Using CIFAR10 dataset.")
        trainset = torchvision.datasets.CIFAR10(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 10
    elif datatype.upper() == "CIFAR100":
        print("Using CIFAR100 dataset.")
        trainset = torchvision.datasets.CIFAR100(root=train_db_path,
                                                 train=True, download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=test_db_path,
                                                train=False, download=True,
                                                transform=transform_test)
        n_classes = 100
    elif datatype.upper() == "MNIST":
        print("Using MNIST dataset.")
        trainset = torchvision.datasets.MNIST(root=train_db_path,
                                                 train=True, download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.MNIST(root=test_db_path,
                                                train=False, download=True,
                                                transform=transform_test)
        n_classes = 10
    else:
        print("Dataset is not supported.")
        return None, None, None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)#在windows系统下num_workers=4用不了
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)#在windows系统下num_workers=4用不了
    return trainloader, testloader, n_classes
def _getdatatransformsdb(datatype):
    transform_train, transform_test = None, None
    if datatype.upper() == "CIFAR10" or datatype.upper() == "CIFAR100" :
        # Data preperation
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.CenterCrop(32),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif datatype.upper() == "MNIST":
        transform_train = transforms.Compose([
            transforms.Resize(28),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]) 
    else:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform_train, transform_test