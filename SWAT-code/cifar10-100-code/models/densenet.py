'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_layers.custom_conv as C
import custom_layers.custom_linear as L
import custom_layers.custom_batchnorm as B 
import pdb

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = B.CustomBatchNorm2d(in_planes)
        self.conv1 = C.CustomConv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = B.CustomBatchNorm2d(4*growth_rate)
        self.conv2 = C.CustomConv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = B.CustomBatchNorm2d(in_planes)
        self.conv = C.CustomConv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = C.CustomConv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = B.CustomBatchNorm2d(num_planes)
        self.linear = L.CustomLinear(num_planes, num_classes)
        self.num_conv_layers=0
        self.num_linear_layers=0
        self.count_conv_linear_layers()
        print("Model DenseNet: Number of Convolution Layers: ",self.num_conv_layers)
        print("Model DenseNet: Number of Linear Layers: ",self.num_linear_layers)


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

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(num_classes):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32,reduction=0.5,num_classes=num_classes)

def DenseNet169(num_classes):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32,reduction=0.5,num_classes=num_classes)

def DenseNet201(num_classes):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32,reduction=0.5,num_classes=num_classes)

def DenseNet161(num_classes):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48,reduction=0.5,num_classes=num_classes)

def SBP_densenet_cifar(num_classes):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12,reduction=0.5,num_classes=num_classes)
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
    net = DenseNet121(10) #densenet_cifar()
    count_conv_linear_layers(net)
    print(numC,numL)
    pdb.set_trace()
    #x = torch.randn(1,3,32,32)
    #y = net(x)
    #print(y)

#test()
