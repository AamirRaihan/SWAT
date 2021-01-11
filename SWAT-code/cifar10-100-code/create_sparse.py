import torch
import sys
import topk
# This file differ from the similar file in ImageNet directory. 
# Since for CIFAR10 and CIFAR100 even the connection from first layer is dropped.
# Moreover the layer in the network is named differently. 
def get_conv_linear_layers(model):
    layers=torch.load('conv_linear_name')
    conv_layers=[]
    linear_layers=[]
    for i in layers:
        if 'classifier' in i or 'fc' in i or 'linear' in i:
            linear_layers.append(i)
        else:
            conv_layers.append(i)
    
    return conv_layers,linear_layers

def get_sparsity_value(sparsity_budget):
    sparsity={} 
    for i in sparsity_budget:
        name=sparsity_budget[i][0]
        curr_sparsity=sparsity_budget[i][1]
        sparsity[name]=curr_sparsity
    return sparsity 

def sparse_unstructured(model,sparsity_budget):
    conv_layers,linear_layers=get_conv_linear_layers(model)
    #print(conv_layers)
    #print(len(conv_layers))
    sparsity_value=get_sparsity_value(sparsity_budget)

    for layer in conv_layers[:]:
        assert layer in  sparsity_value
        print(layer,sparsity_value[layer])
        model[layer]=topk.drop_nhwc(model[layer],1-sparsity_value[layer])

    #removing linear
    for layer in linear_layers:
        assert layer in  sparsity_value
        print(layer,sparsity_value[layer])
        model[layer]=topk.matrix_drop(model[layer],1-sparsity_value[layer])

    return model

def sparse_channel(model,sparsity_budget):
    conv_layers,linear_layers=get_conv_linear_layers(model)
    #print(conv_layers)
    #print(len(conv_layers))
    sparsity_value=get_sparsity_value(sparsity_budget)
    
    for layer in conv_layers[:]:
        assert layer in  sparsity_value
        print(layer,sparsity_value[layer])
        model[layer]=topk.drop_structured(model[layer],1-sparsity_value[layer])

    #removing linear
    for layer in linear_layers:
        assert layer in  sparsity_value
        print(layer,sparsity_value[layer])
        model[layer]=topk.matrix_drop(model[layer],1-sparsity_value[layer])

    return model

def sparse_filter(model,sparsity_budget):
    conv_layers,linear_layers=get_conv_linear_layers(model)
    #print(conv_layers)
    #print(len(conv_layers))
    sparsity_value=get_sparsity_value(sparsity_budget)
    for layer in conv_layers[:]:
        assert layer in  sparsity_value
        print(layer,sparsity_value[layer])
        model[layer]=topk.drop_structured_filter(model[layer],1-sparsity_value[layer])

    #removing linear
    for layer in linear_layers:
        assert layer in  sparsity_value
        print(layer,sparsity_value[layer])
        model[layer]=topk.matrix_drop(model[layer],1-sparsity_value[layer])

    return model

def print_sparsity_unstructured(model):
    conv_layers,linear_layers=get_conv_linear_layers(model)
    total_param=0
    zero_param=0
    for layer in conv_layers+linear_layers:
        weight=model[layer]
        total_conn=float(weight.numel())
        zero_conn=float((weight==0).sum())
        total_param+=total_conn
        zero_param+=zero_conn
        sparsity=zero_conn/total_conn
        print("Layer:",layer, "Num Connections",total_conn," Zero Connections:",zero_conn, "%Connection Zero",sparsity)
    print("Total Sparsity: ",zero_param*100/total_param)

def print_sparsity_structured(model):
    conv_layers,linear_layers=get_conv_linear_layers(model)
    for layer in conv_layers:
        weight=model[layer]
        weight=(weight.reshape(weight.shape[0],weight.shape[1],-1)).abs()>0
        weight=torch.sum(weight,2)==0

        Total_Channel=weight.shape[0]*weight.shape[1]
        Total_Filter=weight.shape[0]
        Channel_Zero=float(torch.sum(weight))
        weight=torch.sum(weight,1)==weight.shape[1]
        Filter_Zero=float(torch.sum(weight))
        print("Layer:",layer," Total Channel:",Total_Channel," Num Channel Zero:",Channel_Zero, "%Channel Zero:", float(Channel_Zero*100)/(Total_Channel), "Total Filter: ",Total_Filter, "Num Filter Zero:",Filter_Zero)

    for layer in linear_layers:
        weight=model[layer]
        total_conn=float(weight.numel())
        zero_conn=float((weight==0).sum())
        sparsity=zero_conn/total_conn
        print("Layer:",layer, "Num Connections",total_conn," Zero Connections:",zero_conn, "%Connection Zero",sparsity)
import pdb
if __name__ =="__main__":
    a=torch.load(sys.argv[1])

    #pdb.set_trace()
    #sparsity_budget=torch.load(sys.argv[2])
    model=a['net']
    #model=sparse_unstructured(model,sparsity_budget)
    print_sparsity_unstructured(model)
    #a['net']=model
    #torch.save(a,'./test.t7')
