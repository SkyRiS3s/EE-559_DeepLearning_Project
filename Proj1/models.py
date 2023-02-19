import torch
from torch import nn
from torch import optim
from torch.nn import functional


import dlc_practical_prologue as prologue

class CNN_base(nn.Module):
    """CNN without weight sharing and without auxiliary loss"""
    def __init__(self):
        super(CNN_base, self).__init__()
        self.convolution1 = nn.Conv2d(2, 16, kernel_size=5, padding = 3)
        self.convolution2 = nn.Conv2d(16, 32, kernel_size=3, padding = 2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm1d(200)
        self.fullyconnected1 = nn.Linear(800, 200)
        self.fullyconnected2 = nn.Linear(200, 2)
        
    def forward(self, x):

        x = self.batchnorm1(functional.relu(functional.max_pool2d(self.convolution1(x), kernel_size=2)))
        x = self.batchnorm2(functional.relu(functional.max_pool2d(self.convolution2(x), kernel_size=2)))
        x = self.batchnorm3(functional.relu(self.fullyconnected1(x.view(x.size()[0], -1))))
        x = self.fullyconnected2(x)
        
        return x, None

class CNN_AL(nn.Module):
    """CNN without weight sharing and with auxiliary loss"""
    def __init__(self):
        super(CNN_AL, self).__init__()
        self.convolution1a = nn.Conv2d(1, 16, kernel_size=5, padding = 3)
        self.convolution1b = nn.Conv2d(1, 16, kernel_size=5, padding = 3)
        self.convolution2a = nn.Conv2d(16, 32, kernel_size=3, padding = 2)
        self.convolution2b = nn.Conv2d(16, 32, kernel_size=3, padding = 2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm1d(100)
        self.fullyconnected1a = nn.Linear(800, 100)
        self.fullyconnected1b = nn.Linear(800, 100)
        self.fullyconnected2 = nn.Linear(200, 2)
        self.fullyconnected2a = nn.Linear(100, 10)
        self.fullyconnected2b = nn.Linear(100, 10)
           
    def forward(self, x):
        xa = x[:,0].view(-1, 1, 14, 14)
        xb = x[:,1].view(-1, 1, 14, 14)
        xa = self.batchnorm1(functional.relu(functional.max_pool2d(self.convolution1a(xa), kernel_size=2)))
        xb = self.batchnorm1(functional.relu(functional.max_pool2d(self.convolution1b(xb), kernel_size=2)))
        xa = self.batchnorm2(functional.relu(functional.max_pool2d(self.convolution2a(xa), kernel_size=2)))
        xb = self.batchnorm2(functional.relu(functional.max_pool2d(self.convolution2b(xb), kernel_size=2)))
        xa = self.batchnorm3(functional.relu(self.fullyconnected1a(xa.view(x.size()[0], -1))))
        xb = self.batchnorm3(functional.relu(self.fullyconnected1b(xb.view(x.size()[0], -1))))
        x = self.fullyconnected2(torch.cat((xa, xb), 1))
        xa = self.fullyconnected2a(xa)
        xb = self.fullyconnected2b(xb)

        return x, [xa, xb]

class CNN_WS(nn.Module):
    """CNN with weight sharing and without auxiliary loss"""
    def __init__(self):
        super(CNN_WS, self).__init__()
        self.convolution1 = nn.Conv2d(1, 16, kernel_size=5, padding = 3)
        self.convolution2 = nn.Conv2d(16, 32, kernel_size=3, padding = 2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm1d(100)
        self.fullyconnected1 = nn.Linear(800, 100)
        self.fullyconnected2 = nn.Linear(200, 2)
        
    def forward(self, data_in):
        data_out = []
        for i in range(2):
            x = data_in[:, i].reshape(-1, 1, 14, 14)
            x = functional.relu(functional.max_pool2d(self.convolution1(x), kernel_size=2))
            x = self.batchnorm2(functional.relu(functional.max_pool2d(self.convolution2(x), kernel_size=2)))
            x = self.batchnorm3(functional.relu(self.fullyconnected1(x.view(x.size()[0], -1))))
            data_out.append(x)
            
        x = self.fullyconnected2(torch.cat((data_out[0], data_out[1]), dim=1))

        return x, None

class CNN_WS_AL(nn.Module):
    """CNN with weight sharing and with auxiliary loss"""
    def __init__(self):
        super(CNN_WS_AL, self).__init__()
        self.convolution1 = nn.Conv2d(1, 16, kernel_size=5, padding = 3)
        self.convolution2 = nn.Conv2d(16, 32, kernel_size=3, padding = 2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm1d(100)
        self.batchnorm4 = nn.BatchNorm1d(100)
        self.fullyconnected1 = nn.Linear(800, 100)
        self.fullyconnected2 = nn.Linear(100, 10)
        self.fullyconnected3 = nn.Linear(20, 100)
        self.fullyconnected4 = nn.Linear(100, 2)
        
    def forward(self, data_in):
        data_out = []
        for i in range(2):
            x = data_in[:, i].reshape(-1, 1, 14, 14)
            x = self.batchnorm1(functional.relu(functional.max_pool2d(self.convolution1(x), kernel_size=2)))
            x = self.batchnorm2(functional.relu(functional.max_pool2d(self.convolution2(x), kernel_size=2)))
            x = self.batchnorm3(functional.relu(self.fullyconnected1(x.view(x.size()[0], -1))))
            data_out.append(x)
            
        data_out[0] = self.fullyconnected2(data_out[0])
        data_out[1] = self.fullyconnected2(data_out[1])
        x = self.batchnorm4(functional.relu(self.fullyconnected3(torch.cat((data_out[0], data_out[1]), dim=1))))
        x = self.fullyconnected4(x)
        
        return x, data_out
