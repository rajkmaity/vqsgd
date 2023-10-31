from __future__ import print_function
import torch
from torch import utils
from torch.autograd import Variable
import numpy as np
from numpy import linalg as LA
import torchvision
import argparse
from torchvision import datasets, transforms
import sys
import scipy.io as sio

def data_load(data_name=None):
    if data_name == "CIFAR10":
        transform = transforms.Compose(
                    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                                shuffle=True, num_workers=1)

        #testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    #   download=True, transform=transform)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                        #                 shuffle=False, num_workers=1)
    
    if data_name == "FashionMNIST":
        trainset = datasets.FashionMNIST('./data', download=True, train=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=True)

        #testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    if data_name == "MNIST":
        trainloader = torch.utils.data.DataLoader(datasets.MNIST('./data',
                                                download=True,
                                                train=True,
                                                transform=transforms.Compose([
                                                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) #rst, convert image to PyTorch tensor # normalize inputs
                                                ])),
                                            batch_size=60000,
                                            shuffle=True)
    else:
        return(" Not a valid Dataset")
    for data,target in trainloader:
        x=Variable(data)
        y=Variable(target)
    return x, y