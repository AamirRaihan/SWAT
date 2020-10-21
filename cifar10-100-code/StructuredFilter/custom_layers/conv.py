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

num_gpus= torch.cuda.device_count()

class myconv2d(Function):
    def __init__(self, stride,padding,dilation,groups,name):
        super(myconv2d, self).__init__()
        self.stride= stride
        self.padding= padding 
        self.dilation=dilation 
        self.groups=groups
        self.name=name

    def forward(self, input,weight,bias):
        gpuid=torch.cuda.current_device()
        temp_data=torch.load('./temp_data_'+str(gpuid))
        epoch    =int(temp_data[0])
        batch_idx=int(temp_data[1])
        layer    =int(temp_data[2])
        TYPE     =int(temp_data[3])
        PERIOD   =int(temp_data[4])
        SPARSITY =float(temp_data[5])
        WARMUP   =int(temp_data[6])
        #if gpuid==0:
        #    print(layer)
        
        #print("GPU:",gpuid,"EPOCH:",epoch,"BATCHIDX:",batch_idx,"Layer:",layer,"PERIOD:",PERIOD,"SPARSITY:",SPARSITY,"WARMUP:",WARMUP)
        threshold=torch.load('./dump/threshold_'+str(layer)+'_'+str(gpuid))
        wt_threshold=threshold[0]
        in_threshold=threshold[1]

        sparsity           = SPARSITY 
        select_percentage  = 1 - sparsity
        
       # if epoch >= WARMUP:
       #     if batch_idx%PERIOD==0:
       #         weight,wt_threshold = topk.drop_nhwc_send_th(weight,select_percentage)
       #     else:
       #         weight            = topk.drop_threshold(weight,wt_threshold)
       # else:
       #     weight                = topk.drop_nhwc(weight,select_percentage)
       
        weight                = topk.drop_structured_filter(weight,select_percentage)
        output=F.conv2d(input,weight,bias,self.stride,self.padding,self.dilation,self.groups)
        
        if epoch >= WARMUP:
            if batch_idx%PERIOD==0:
                input,in_threshold  = topk.drop_nhwc_send_th(input,select_percentage)
            else:
                input              = topk.drop_threshold(input,in_threshold)
        else:
            input                  = topk.drop_nhwc(input,select_percentage)

        
        #wt_per=((weight==0).sum().float())/(weight.view(-1).shape[0])
        #in_per=((input==0).sum().float())/(input.view(-1).shape[0])
        #print("CONV:WT ",wt_per,"IN ",in_per)

        if batch_idx%PERIOD==0:
            threshold=torch.tensor([wt_threshold,in_threshold])
            torch.save(threshold,'./dump/threshold_'+str(layer)+'_'+str(gpuid))

        temp_data =torch.tensor([epoch,batch_idx,(layer+1),TYPE,PERIOD,SPARSITY,WARMUP])
        torch.save(temp_data,'./temp_data_'+str(gpuid))

        self.save_for_backward(input,weight,bias)
        return output
    
    def backward(self, grad_output):
        input,weight,bias=self.saved_tensors
        grad_output_shape=grad_output.shape 

        #wt_per=((weight==0).sum().float())/(weight.view(-1).shape[0])
        #in_per=((input==0).sum().float())/(input.view(-1).shape[0])
        #print("CONV WEIGHT", wt_per)
        #print("CONV INPUT" , in_per)

        grad_weight=mypkg.myConvBackwardWeight(weight.shape, grad_output, input, self.stride, self.padding, self.dilation, self.groups)
        grad_input=mypkg.myConvBackwardInput(input.shape, grad_output, weight, self.stride, self.padding, self.dilation, self.groups)
        grad_bias=mypkg.myConvBackwardBias(grad_output)
        
        return  grad_input, grad_weight,grad_bias
class myconv2dNB(Function):
    def __init__(self, stride,padding,dilation,groups,name):
        super(myconv2dNB, self).__init__()
        self.stride= stride
        self.padding= padding 
        self.dilation=dilation 
        self.groups=groups
        self.name=name

    def forward(self, input,weight):
        gpuid=torch.cuda.current_device()
        temp_data=torch.load('./temp_data_'+str(gpuid))
        epoch    =int(temp_data[0])
        batch_idx=int(temp_data[1])
        layer    =int(temp_data[2])
        TYPE     =int(temp_data[3])
        PERIOD   =int(temp_data[4])
        SPARSITY =float(temp_data[5])
        WARMUP   =int(temp_data[6])
        #if gpuid==0:
        #    print(layer)
        
        #print("GPU:",gpuid,"EPOCH:",epoch,"BATCHIDX:",batch_idx,"Layer:",layer,"PERIOD:",PERIOD,"SPARSITY:",SPARSITY,"WARMUP:",WARMUP)
        threshold=torch.load('./dump/threshold_'+str(layer)+'_'+str(gpuid))
        wt_threshold=threshold[0]
        in_threshold=threshold[1]

        sparsity           = SPARSITY 
        select_percentage  = 1 - sparsity
        
        if epoch > WARMUP:
            if batch_idx%PERIOD==0:
                weight,wt_threshold = topk.drop_nhwc_send_th(weight,select_percentage)
            else:
                weight            = topk.drop_threshold(weight,wt_threshold)
        else:
            weight                = topk.drop_nhwc(weight,select_percentage)

        output=F.conv2d(input,weight,None,self.stride,self.padding,self.dilation,self.groups)
        
        if epoch >WARMUP:
            if batch_idx%PERIOD==0:
                input,in_threshold  = topk.drop_nhwc_send_th(input,select_percentage)
            else:
                input              = topk.drop_threshold(input,in_threshold)
        else:
            input                  = topk.drop_nhwc(input,select_percentage)

        wt_per=((weight==0).sum().float())/(weight.view(-1).shape[0])
        in_per=((input==0).sum().float())/(input.view(-1).shape[0])
        #print("CONV:WT ",wt_per,"IN ",in_per)

        if batch_idx%PERIOD==0:
            threshold=torch.tensor([wt_threshold,in_threshold])
            torch.save(threshold,'./dump/threshold_'+str(layer)+'_'+str(gpuid))

        temp_data =torch.tensor([epoch,batch_idx,(layer+1),TYPE,PERIOD,SPARSITY,WARMUP])
        torch.save(temp_data,'./temp_data_'+str(gpuid))

        self.save_for_backward(input,weight)
        return output
    
    def backward(self, grad_output):
        input,weight=self.saved_tensors
        grad_output_shape=grad_output.shape 
        grad_output=grad_output.contiguous()
        #wt_per=((weight==0).sum().float())/(weight.view(-1).shape[0])
        #in_per=((input==0).sum().float())/(input.view(-1).shape[0])
        #print("CONV WEIGHT", wt_per)
        #print("CONV INPUT" , in_per)

        grad_weight=mypkg.myConvBackwardWeight(weight.shape, grad_output, input, self.stride, self.padding, self.dilation, self.groups)
        grad_input=mypkg.myConvBackwardInput(input.shape, grad_output, weight, self.stride, self.padding, self.dilation, self.groups)
        
        return  grad_input, grad_weight

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
        self.in_threshold=Variable(torch.zeros(1))
        self.wt_threshold=Variable(torch.zeros(1))
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

class MyConv2d(myConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,name="defaultName"):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,name)

    def forward(self, input):
        out= myconv2d(self.stride,self.padding,self.dilation,self.groups,self.name)(input, self.weight,self.bias)
        return out

class MyConv2dNB(myConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,name="defaultName"):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2dNB, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,name)

    def forward(self, input):
        out= myconv2dNB(self.stride,self.padding,self.dilation,self.groups,self.name)(input, self.weight)
        return out


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = MyConv2d(input_channel, output_channel, filter_size)

    def forward(self, x):
       out = self.conv1(x)
       return out
