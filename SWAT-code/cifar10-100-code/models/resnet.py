'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import custom_layers.custom_conv as C
import custom_layers.custom_linear as L
import custom_layers.custom_batchnorm as B
import pdb


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1,layer_number=[-1]):
        super(BasicBlock, self).__init__()
        self.conv1 = C.CustomConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True,name="conv_layer"+str(layer_number[0]))
        layer_number[0]+=1 #increment the layer number
        self.bn1 = B.CustomBatchNorm2d(planes)
        self.conv2 = C.CustomConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True,name="conv_layer"+str(layer_number[0]))
        layer_number[0]+=1 #increment the layer number
        self.bn2 = B.CustomBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                    C.CustomConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True,name="conv_layer"+str(layer_number[0])),
                B.CustomBatchNorm2d(self.expansion*planes)
            )
            layer_number[0]+=1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,layer_number=-1):
        super(Bottleneck, self).__init__()
        self.conv1 = C.CustomConv2d(in_planes, planes, kernel_size=1, bias=True,name="conv_layer"+str(layer_number[0]))
        layer_number[0]+=1 #increment the layer number
        self.bn1 = B.CustomBatchNorm2d(planes)
        self.conv2 = C.CustomConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True,name="conv_layer"+str(layer_number[0]))
        layer_number[0]+=1 #increment the layer number
        self.bn2 = B.CustomBatchNorm2d(planes)
        self.conv3 = C.CustomConv2d(planes, self.expansion*planes, kernel_size=1, bias=True,name="conv_layer"+str(layer_number[0]))
        layer_number[0]+=1 #increment the layer number
        self.bn3 = B.CustomBatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                C.CustomConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True,name="conv_layer"+str(layer_number[0])),
                B.CustomBatchNorm2d(self.expansion*planes)
            )
            layer_number[0]+=1 #increment the layer number

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, numchannels, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        layer_number=[1]  #layer identifier
        self.conv1 = C.CustomConv2d(numchannels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        layer_number[0]+=1 #increment the layer number
        self.bn1 = B.CustomBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, layer_no=layer_number)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,layer_no=layer_number)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,layer_no=layer_number)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,layer_no=layer_number)
        self.linear = L.CustomLinear(512*block.expansion, num_classes)
        self.all_layers = []
        self.num_conv_layers=0
        self.num_linear_layers=0
        self.count_conv_linear_layers()
        print("Model ResNet: Number of Convolution Layers: ", self.num_conv_layers)
        print("Model ResNet: Number of Linear Layers: ", self.num_linear_layers)
    
   
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

    def _make_layer(self, block, planes, num_blocks, stride,layer_no):
        strides = [stride] + [1]*(num_blocks-1)
        layers  = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,layer_no))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out6 = out5.view(out5.size(0), -1)
        out7 = self.linear(out6)
        return out7


def ResNet18(numchannels,numclasses):
    return ResNet(BasicBlock, [2,2,2,2], numchannels,numclasses)

def ResNet34(numchannels,numclasses):
    return ResNet(BasicBlock, [3,4,6,3], numchannels,numclasses)

def ResNet50(numchannels,numclasses):
    return ResNet(Bottleneck, [3,4,6,3], numchannels,numclasses)

def ResNet101(numchannels,numclasses):
    return ResNet(Bottleneck, [3,4,23,3], numchannels,numclasses)

def ResNet152(numchannels,numclasses):
    return ResNet(Bottleneck, [3,8,36,3], numchannels,numclasses)

def test():
    global numC,numL
    net = ResNet18(3,10)
    count_conv_linear_layers(net)
    print(numC,numL)
    pdb.set_trace()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

#test()
