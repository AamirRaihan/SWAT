"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import custom_layers.conv as C
import custom_layers.mylinear as L
import custom_layers.mybatchnorm as B

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
            L.MyLinear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            L.MyLinear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            L.MyLinear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
    
        return output

    def _make_layers(self,cfg, batch_norm=True):
        layers = []
    
        input_channel = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
    
            layers += [C.MyConv2d(input_channel, l, kernel_size=3, padding=1)]
    
            if batch_norm:
                layers += [B.MyBatchNorm2d(l)]
            
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
