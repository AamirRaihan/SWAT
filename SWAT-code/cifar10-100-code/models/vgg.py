"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''
import pdb
import torch
import torch.nn as nn
import custom_layers.custom_conv as C
import custom_layers.custom_linear as L
import custom_layers.custom_batchnorm as B

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):

    def __init__(self, vgg_name, num_class=100):
        super().__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            L.CustomLinear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            L.CustomLinear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            L.CustomLinear(4096, num_class)
        )
        self.num_conv_layers=0
        self.num_linear_layers=0
        self.count_conv_linear_layers()
        print("Model VGG: Number of convolution layers: ",self.num_conv_layers)
        print("Model VGG: Number of linear layers: ",self.num_linear_layers)


    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

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

    def _make_layers(self,cfg, batch_norm=True):
        layers = []
    
        input_channel = 3
        for index,l in enumerate(cfg):
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            #if index==0:
            #    layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
            #else:
            #    layers += [C.CustomConv2d(input_channel, l, kernel_size=3, padding=1)]
            layers += [C.CustomConv2d(input_channel, l, kernel_size=3, padding=1)]
    
            if batch_norm:
                layers += [B.CustomBatchNorm2d(l)]
            
            layers += [nn.ReLU(inplace=True)]
            input_channel = l
        
        return nn.Sequential(*layers)

#def vgg11_bn():
#    return VGG(make_layers(cfg['A'], batch_norm=True))
#
#def vgg13_bn():
#    return VGG(make_layers(cfg['B'], batch_norm=True))
#
#def vgg16_bn():
#    return VGG(make_layers(cfg['D'], batch_norm=True))
#
#def vgg19_bn():
#    return VGG(make_layers(cfg['E'], batch_norm=True))

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
    net = VGG('VGG16',10)
    count_conv_linear_layers(net)
    print(numC,numL)
    pdb.set_trace()
    #x = torch.randn(1,3,32,32)
    #y = net(x)
    #print(y)

#test()
