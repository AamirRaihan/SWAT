import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_layers.custom_conv as C
import custom_layers.custom_linear as L
import custom_layers.custom_batchnorm as B
import torch.nn.init as init
import sys
import numpy as np

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        if m.bias is not None:
            init.constant(m.bias, 0)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = B.CustomBatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = C.CustomConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = B.CustomBatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = C.CustomConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and C.CustomConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor,dropout_rate, num_classes):#, widen_factor=1, dropRate=0.0):
        super(Wide_ResNet, self).__init__()
        dropRate=dropout_rate
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = C.CustomConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = B.CustomBatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = L.CustomLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, C.CustomConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, B.CustomBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, L.CustomLinear):
                m.bias.data.zero_()
        self.num_conv_layers=0
        self.num_linear_layers=0
        self.count_conv_linear_layers()
        print("Model WideResNet: Number of Convolution Layers: ",self.num_conv_layers)
        print("Model WideResNet: Number of Linear Layers: ",self.num_linear_layers)


    def count_conv_linear_layers(self):
        self._count_conv_linear_layers(self)
    
    def _count_conv_linear_layers(self,network):
        for layer in network.children():
            if type(layer) != []: 
                self._count_conv_linear_layers(layer)
            if list(layer.children()) == []: # if leaf node, add it to list
                if isinstance(layer, C.CustomConv2d):
                    self.num_conv_layers+=1
                if isinstance(layer, L.CustomLinear):
                    self.num_linear_layers+=1

    
    def remove_sequential(self):
        for layer in self.children():
            if type(layer) == nn.Sequential: # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(layer)
            if list(layer.children()) == []: # if leaf node, add it to list
                all_layers.append(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

numC=0
numL=0
def count_conv_linear_layers(network):
    global numC, numL
    for layer in network.children():
        print("-------------")
        print(layer)
        if type(layer) != []:#nn.Sequential: # if sequential layer, apply recursively to layers in sequential layer
            print("HEHE:",layer)
            count_conv_linear_layers(layer)
        if list(layer.children()) == []: # if leaf node, add it to list
            print(layer)
            if isinstance(layer, C.CustomConv2d):
                numC+=1
            if isinstance(layer, L.CustomLinear):
                numL+=1

def test():
    global numC, numL
    net =Wide_ResNet(16, 8,0.3,10)
    pdb.set_trace()
    count_conv_linear_layers(net)
    print(numC,numL)
    #x = torch.randn(1,3,32,32)
    #y = net(x)
    #print(y)

#test()
