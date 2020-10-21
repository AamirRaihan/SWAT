import torch
import sys
import topk
#THIS FILE DIFFER FROM IMAGENET FILE SINCE ON CIFAR10 & CIFAR100 
#EVEN THE CONNECTION FROM FIRST LAYER IS DROPPED
def get_conv_linear_layers(model):
    layers=[]
    for i in model:
        if 'bias'in i or 'shortcut.1' in i or 'bn' in i or 'running' in i or 'num_batches' in i:
            continue
        layers.append(i)
    conv_layers=[]
    linear_layers=[]
    for i in layers:
        if 'linear' in i:
            linear_layers.append(i)
        else:
            conv_layers.append(i)
    
    return conv_layers,linear_layers

def sparse_unstructured(model,select_percent):
    conv_layers,linear_layers=get_conv_linear_layers(model)

    #print(conv_layers)
    #print(len(conv_layers))
    
    for layer in conv_layers[:]:
        model[layer]=topk.drop_nhwc(model[layer],select_percent)

    #removing linear
    for layer in linear_layers:
        model[layer]=topk.matrix_drop(model[layer],select_percent)

    return model

def sparse_channel(model,select_percent):
    conv_layers,linear_layers=get_conv_linear_layers(model)

    #print(conv_layers)
    #print(len(conv_layers))
    
    for layer in conv_layers[:]:
        model[layer]=topk.drop_structured(model[layer],select_percent)

    #removing linear
    for layer in linear_layers:
        model[layer]=topk.matrix_drop(model[layer],select_percent)

    return model

def sparse_filter(model,select_percent):
    conv_layers,linear_layers=get_conv_linear_layers(model)

    #print(conv_layers)
    #print(len(conv_layers))
    
    for layer in conv_layers[:]:
        model[layer]=topk.drop_structured_filter(model[layer],select_percent)

    #removing linear
    for layer in linear_layers:
        model[layer]=topk.matrix_drop(model[layer],select_percent)

    return model

def print_sparsity_unstructured(model):
    conv_layers,linear_layers=get_conv_linear_layers(model)
    for layer in conv_layers+linear_layers:
        weight=model[layer]
        total_conn=float(weight.numel())
        zero_conn=float((weight==0).sum())
        sparsity=zero_conn/total_conn
        print("Layer:",layer, "Num Connections",total_conn," Zero Connections:",zero_conn, "%Connection Zero",sparsity)


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

if __name__ =="__main__":
    a=torch.load(sys.argv[1])
    select_percent=float(sys.argv[2])
    model=a['net']
    model=sparse_channel(model,select_percent)
    print_sparsity_structured(model)
    a['net']=model
    torch.save(a,'./test.t7')
