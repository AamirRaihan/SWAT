import math
import copy 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.backends.cudnn as cudnn
import mypkg
from torch.nn import init
import pdb
import topk
import torch.optim as optim
num_gpus= torch.cuda.device_count()

class mylinear(Function):
    def __init__(self):
        super(mylinear, self).__init__()
    
    def forward( self, input, weight, bias):
        gpuid=torch.cuda.current_device()
        temp_data=torch.load('./temp_data_'+str(gpuid))
        epoch=int(temp_data[0])
        batch_idx=int(temp_data[1])
        layer    =int(temp_data[2])
        TYPE     =int(temp_data[3])
        PERIOD   =int(temp_data[4])
        SPARSITY =float(temp_data[5])
        WARMUP   =int(temp_data[6])        
        
        #if gpuid==0:
        #    print(layer)
        
        sparsity           = SPARSITY 
        select_percentage  = 1 - sparsity 
        weight             = topk.matrix_drop(weight,select_percentage)
        #print("LINEAR:SPARSITY:",sparsity)

        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            ret = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
            ret = output
        
        input        = topk.matrix_drop(input,select_percentage)
        self.save_for_backward(input, weight, bias)
        temp_data =torch.tensor([epoch,batch_idx,(layer+1),TYPE,PERIOD,SPARSITY,WARMUP])
        torch.save(temp_data,'./temp_data_'+str(gpuid))
        
        return ret
    
    def backward(self, dy):
        x, w, b = self.saved_tensors
        
        
        #wt_per=((w==0).sum().float())/(w.view(-1).shape[0])
        #in_per=((x==0).sum().float())/(x.view(-1).shape[0])
        #print("LINEAR: WT ", wt_per,"IN " , in_per)
        
        dx = dw = db = None

        dx = torch.mm(dy, w)
        dw = torch.mm(x.t(), dy)
        db = torch.sum(dy, 0)

        return dx, dw.t(), db

class MyLinear(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return mylinear()(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



if __name__ == "__main__":            
    device = torch.device('cuda:0')
    def compare_linear(l1, l2):
        err = False
        if not torch.allclose(l1.weight, l2.weight):
            print('Diff in weight: {} vs {}'.format(
                l1.weight, l2.weight))
            err = True
        
        if not torch.allclose(l1.bias, l2.bias):
            print('Diff in bias: {} vs {}'.format(
                l1.bias, l2.bias))
            err = True
    
        if not err:
            print('All parameters are equal!')

    # Init BatchNorm layers
    my_linear = MyLinear(512, 100)
    linear = nn.Linear(512,100)
    
    my_linear.to(device)
    linear.to(device)
    
    #compare_linear(my_linear, linear)  # weight and bias should be different
    # Load weight and bias
    my_linear.load_state_dict(linear.state_dict())
    compare_linear(my_linear, linear)
    ## Run train
    for index in range(10):
        print("################: Training",index," ############# ")
        x = torch.randn(32,512)
        x=Variable(x)
        x=x.to(device)
        out1 = my_linear(x)
        out2 = linear(x)
        compare_linear(my_linear, linear)
        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())
        
    # Run eval
    my_linear.eval()
    linear.eval()
    for index in range(10):
        print("################: Evaluation",index," ############# ")
        x = torch.randn(32,512)
        x=Variable(x)
        x=x.to(device)
        out1 = my_linear(x)
        out2 = linear(x)
        compare_linear(my_linear, linear)
        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())
    
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear1= nn.Linear(512,100)
            self.linear2= nn.Linear(100,10)
    
        def forward(self, x):
           out = self.linear1(x)
           out = self.linear2(out)
           return out
    
    class myModel(nn.Module):
        def __init__(self):
            super(myModel, self).__init__()
            self.linear1= MyLinear(512,100)
            self.linear2= MyLinear(100,10)
    
        def forward(self, x):
           out = self.linear1(x)
           out = self.linear2(out)
           return out
    
    #Fixing the seed for deterministic execution
    torch.manual_seed(0)
    ref_model=Model()
    ref_model=ref_model.to(device)
    model=myModel()
    model=model.to(device)
    
    print("COMPARING MODULE")
    #compare_bn(model.linear1,ref_model.linear1)
    #compare_bn(model.linear2,ref_model.linear2)
    model.linear1.load_state_dict(ref_model.linear1.state_dict())
    model.linear2.load_state_dict(ref_model.linear2.state_dict())
    optimizer = optim.SGD(model.parameters(), lr=1, momentum=0, weight_decay=0)
    ref_optimizer = optim.SGD(ref_model.parameters(), lr=1, momentum=0, weight_decay=0)
    #
    ##for i,j in zip(model.parameters(),ref_model.parameters()):
    ##    print("HEHE: ",i,j)
    #    
    print("STARTING_LOOP")
    for i in range(10):
        print("-----------------")
        compare_linear(model.linear1,ref_model.linear1)
        compare_linear(model.linear2,ref_model.linear2)
        x=Variable(torch.randn(32,512),requires_grad=False)
        x=x.to(device)
        ref_x=x.clone()
        out=model(x)
        ref_out=ref_model(ref_x)
        torch.allclose(out, ref_out)
        print('Max diff: ', (out - ref_out).abs().max())
        y=torch.randn(out.shape).to(device)
        criterion=nn.MSELoss()
        loss=criterion(out,y)
        ref_loss=criterion(ref_out,y)
        optimizer.zero_grad()
        ref_optimizer.zero_grad()
        loss.backward()
        ref_loss.backward()
        
        print("Linear1",(model.linear1.weight.grad==ref_model.linear1.weight.grad).sum()==model.linear1.weight.numel())
        print("Linear1",(model.linear1.bias.grad==ref_model.linear1.bias.grad).sum()==model.linear1.bias.numel())
        print("Linear2",(model.linear2.weight.grad==ref_model.linear2.weight.grad).sum()==model.linear2.weight.numel())
        print("Linear2",(model.linear2.bias.grad==ref_model.linear2.bias.grad).sum()==model.linear2.bias.numel())
        ref_optimizer.step()
        optimizer.step()
        print("-----------------")
