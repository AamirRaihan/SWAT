
import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as G
import time
import mypkg
import topk

from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.backends.cudnn as cudnn
class customconv2d(Function):
    def __init__(self, stride,padding,dilation,groups,name):
        super(customconv2d, self).__init__()
        self.stride= stride
        self.padding= padding 
        self.dilation=dilation 
        self.groups=groups
        self.name=name
    
    def load_configuration(self):
        self.gpuid=torch.cuda.current_device()
        configuration_data=torch.load('./configuration_data_'+str(self.gpuid))
        self.epoch    =int(configuration_data['epoch'])
        self.batch_idx=int(configuration_data['batch_idx'])
        self.layer    =int(configuration_data['layer'])
        self.istrain     =int(configuration_data['type'])
        self.period   =int(configuration_data['period'])
        self.sparsity =configuration_data['sparsity']
        self.act_sparsity=configuration_data['global_sparsity']
        self.warmup   =int(configuration_data['warmup'])
        self.pruning_type=configuration_data['pruning_type']
   
    def release_memory(self):
        del self.gpuid    
        del self.epoch    
        del self.batch_idx
        del self.layer    
        del self.istrain    
        del self.period   
        del self.sparsity 
        del self.act_sparsity
        del self.warmup   
        del self.pruning_type
        torch.cuda.empty_cache()
    
    def print_configuration(self):
        if self.gpuid==0:
            print("----------------------")
            print("Gpu:",self.gpuid,end="\n")
            print("Epoch:",self.epoch, end="\n")
            print("BatchIdx:",self.batch_idx, end="\n")
            print("Layer:",self.layer, end="\n")
            print("WarmUp:",self.warmup) 
            print("TopK-Period:",self.period, end="\n")
            print("Sparsity:",self.sparsity[self.layer], end="\n")
    
    def load_threshold(self):
        self.threshold=torch.load('./dump/threshold_'+str(self.layer)+'_'+str(self.gpuid))
        self.wt_threshold=self.threshold[0]
        self.in_threshold=self.threshold[1]
        
    def sparsify_weight(self, weight):
        sparsity           = self.sparsity[self.layer][1]
        select_percentage  = 1 - sparsity
        
        if self.pruning_type=='unstructured':
            if self.epoch >= self.warmup:
                if self.batch_idx%self.period==0:
                    weight,self.wt_threshold = topk.drop_nhwc_send_th(weight,select_percentage)
                else:
                    weight            = topk.drop_threshold(weight,self.wt_threshold)
            else:
                weight                = topk.drop_nhwc(weight,select_percentage)
        else:
            if self.pruning_type=='structured_channel':
                weight                = topk.drop_structured(weight,select_percentage)
            elif self.pruning_type=='structured_filter':
                weight                = topk.drop_structured_filter(weight,select_percentage)
            else:
                assert(0, "Illegal Pruning Type")
        return weight
    
    def sparsify_activation(self,input):
        sparsity           = self.sparsity[self.layer][1]
        select_percentage  = 1 - sparsity
        if self.epoch >= self.warmup:
            if self.batch_idx%self.period==0:
                input,self.in_threshold  = topk.drop_nhwc_send_th(input,select_percentage)
            else:
                input              = topk.drop_threshold(input,self.in_threshold)
        else:
            input                  = topk.drop_nhwc(input,select_percentage)
        
        return input

    def update_threshold(self):
        if self.batch_idx%self.period==0:
            threshold=torch.tensor([self.wt_threshold,self.in_threshold])
            torch.save(threshold,'./dump/threshold_'+str(self.layer)+'_'+str(self.gpuid))
    
    def update_configuration(self):
        next_layer_configuration_data ={'global_sparsity':self.act_sparsity,
                                        'epoch':self.epoch,
                                        'batch_idx':self.batch_idx,
                                        'layer':self.layer+1,
                                        'type':self.istrain,
                                        'period':self.period,
                                        'sparsity':self.sparsity,
                                        'warmup':self.warmup,
                                        'pruning_type':self.pruning_type}
        torch.save(next_layer_configuration_data,'./configuration_data_'+str(self.gpuid))

    
    def forward(self, input,weight,bias=None):
        self.load_configuration()
        self.load_threshold()
        weight=self.sparsify_weight(weight)
        output=F.conv2d(input,weight,bias,self.stride,self.padding,self.dilation,self.groups)
        input=self.sparsify_activation(input)
        self.update_threshold()
        self.update_configuration()
        self.save_for_backward(input,weight,bias)
        self.release_memory()
        return output
    
    def backward(self, grad_output):
        input,weight,bias=self.saved_tensors
        grad_output_shape=grad_output.shape 
        grad_output=grad_output.contiguous()
        grad_weight=mypkg.myConvBackwardWeight(weight.shape, grad_output, input, self.stride, self.padding, self.dilation, self.groups)
        grad_input=mypkg.myConvBackwardInput(input.shape, grad_output, weight, self.stride, self.padding, self.dilation, self.groups)
        if bias is not None:
            grad_bias=mypkg.myConvBackwardBias(grad_output)
        else:
            grad_bias =None
        return  grad_input, grad_weight,grad_bias


class myConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias,name):
        super(myConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.name = name
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class CustomConv2d(myConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,name="CustomConvolutionLayer"):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CustomConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,name)

    def forward(self, input):
        if self.bias is None:
            return customconv2d(self.stride,self.padding,self.dilation,self.groups,self.name)(input, self.weight)
        else:
            return customconv2d(self.stride,self.padding,self.dilation,self.groups,self.name)(input, self.weight,self.bias)

class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()
        self.conv1 = CustomConv2d(input_channel, output_channel, filter_size)

    def forward(self, x):
       out = self.conv1(x)
       return out

