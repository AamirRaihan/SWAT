import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.backends.cudnn as cudnn
import mypkg
from torch.nn import init
import pdb
import topk

device = torch.device('cuda:0')

class custombatchnorm2d(Function):
    def __init__(self):
        super(custombatchnorm2d, self).__init__()
    
    def forward(self, input, weight,bias, running_mean, running_var, training, momentum, eps):
        training=float(training)==1
        momentum = float(momentum)
        eps =float(eps)
        output,save1,save2 = mypkg.myBatchNormForward(input, weight, bias, running_mean, running_var, training, momentum, eps)
        
        self.save_for_backward(input,weight,running_mean,running_var,save1,save2,torch.tensor(eps))
        return output 
    
    def backward(self, grad_output):
        input,weight, running_mean, running_var,save_mean,save_var,eps= self.saved_tensors
        a,b,c=(mypkg.myBatchNormBackward(input,grad_output,weight,running_mean,running_var,save_mean,save_var,float(eps)))

        return a,b,c,None,None,None,None,None

class CustomBatchNorm(Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training or not self.track_running_stats:
            return custombatchnorm2d()( input, self.weight,self.bias, self.running_mean, self.running_var, torch.tensor(1),torch.tensor(exponential_average_factor), torch.tensor(self.eps))
        else:
            return custombatchnorm2d()( input, self.weight,self.bias, self.running_mean, self.running_var, torch.tensor(0),torch.tensor(exponential_average_factor), torch.tensor(self.eps))


    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(CustomBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class CustomBatchNorm2d(CustomBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

if __name__ == "__main__":            
    def compare_bn(bn1, bn2):
        err = False
        if not torch.allclose(bn1.running_mean, bn2.running_mean):
            print('Diff in running_mean: {} vs {}'.format(
                bn1.running_mean, bn2.running_mean))
            err = True
    
        if not torch.allclose(bn1.running_var, bn2.running_var):
            print('Diff in running_var: {} vs {}'.format(
                bn1.running_var, bn2.running_var))
            err = True
    
        if bn1.affine and bn2.affine:
            if not torch.allclose(bn1.weight, bn2.weight):
                print('Diff in weight: {} vs {}'.format(
                    bn1.weight, bn2.weight))
                err = True
    
            if not torch.allclose(bn1.bias, bn2.bias):
                print('Diff in bias: {} vs {}'.format(
                    bn1.bias, bn2.bias))
                err = True
    
        if not err:
            print('All parameters are equal!')
    
    
    # Init BatchNorm layers
    my_bn = CustomBatchNorm2d(3, affine=True)#.to_device(device)
    bn = nn.BatchNorm2d(3, affine=True)#.to_device(device)
    
    my_bn.to(device)
    bn.to(device)
    compare_bn(my_bn, bn)  # weight and bias should be different
    # Load weight and bias
    my_bn.load_state_dict(bn.state_dict())
    compare_bn(my_bn, bn)
    ## Run train
    for index in range(10):
        print("################: Training",index," ############# ")
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100)* scale + bias
        x=Variable(x)
        x=x.to(device)
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)
        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())
        
    ## Run eval
    my_bn.eval()
    bn.eval()
    for index in range(10):
        print("################: Evaluation",index," ############# ")
        scale = torch.randint(1, 10, (1,)).float()
        bias = torch.randint(-10, 10, (1,)).float()
        x = torch.randn(10, 3, 100, 100) * scale + bias
        x=Variable(x)
        x=x.to(device)
        out1 = my_bn(x)
        out2 = bn(x)
        compare_bn(my_bn, bn)
        torch.allclose(out1, out2)
        print('Max diff: ', (out1 - out2).abs().max())
    
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.batchnorm= nn.BatchNorm2d(input_channel)
    
        def forward(self, x):
           #logging.info("forward-model\n",x)
           out = self.batchnorm(x)
           return out
    
    class myModel(nn.Module):
        def __init__(self):
            super(myModel, self).__init__()
            self.batchnorm=CustomBatchNorm2d(input_channel) 
    
        def forward(self, x):
           #logging.info("forward-mymodel\n",x)
           out = self.batchnorm(x)
           return out
    
    batch_size=10
    input_channel=3
    input_xdim=5
    input_ydim=5
    custom_grad_input=custom_grad_output=custom_grad_weight=custom_grad_bias=torch.empty(1)
    #Fixing the seed for deterministic execution
    torch.manual_seed(0)
    ref_model=Model()
    ref_model=ref_model.to(device)
    model=myModel()
    model=model.to(device)
    
    print("COMPARING MODULE")
    compare_bn(model.batchnorm,ref_model.batchnorm)
    model.batchnorm.load_state_dict(ref_model.batchnorm.state_dict())
    for i in range(20):
        compare_bn(model.batchnorm,ref_model.batchnorm)
        x=Variable(torch.randn(batch_size,input_channel,input_xdim,input_ydim),requires_grad=True)
        x=x.to(device)
        ref_x=x.clone()
        out=model(x)
        ref_out=ref_model(ref_x)
        torch.allclose(out, ref_out)
        print('Max diff: ', (out - ref_out).abs().max())
        y=torch.randn(out.shape).to(device)
        criterion=nn.MSELoss()
        loss=criterion(out,y)
        loss.backward()
        ref_loss=criterion(ref_out,y)
        ref_loss.backward()
        print("GRAD_EQUAL",ref_x.grad==x.grad)
