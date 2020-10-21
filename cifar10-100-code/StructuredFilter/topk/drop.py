import math
import torch
import numpy as np
import pdb

def drop_structured_filter(tensor,select_percentage):
	tensor_shape    =   tensor.shape
	k               =   int(math.ceil(select_percentage*(tensor_shape[0])))
	tensor          =   tensor.reshape(tensor_shape[0],tensor_shape[1]*tensor_shape[2]*tensor_shape[3])
	
	new_tensor_shape=   tensor.shape
	value           =   torch.sum(tensor.abs(),1)
	topk            =   value.view(-1).abs().topk(k)
	interleaved     =   topk[0][-1]
	index           =   value.abs()>=(interleaved)
	index           =   index.repeat_interleave(tensor_shape[1]*tensor_shape[2]*tensor_shape[3]).type_as(tensor).reshape(new_tensor_shape)
	
	tensor          =   (tensor*index)
	tensor          =   tensor.reshape(tensor_shape)
	return tensor

def drop_structured(tensor,select_percentage):
	tensor_shape    =   tensor.shape
	k               =   int(math.ceil(select_percentage*(tensor_shape[0]*tensor_shape[1])))
	tensor          =   tensor.reshape(tensor_shape[0],tensor_shape[1],tensor_shape[2]*tensor_shape[3])
	
	new_tensor_shape=   tensor.shape
	value           =   torch.sum(tensor.abs(),2)
	topk            =   value.view(-1).abs().topk(k)
	interleaved     =   topk[0][-1]
	index           =   value.abs()>=(interleaved)
	index           =   index.repeat_interleave(tensor_shape[2]*tensor_shape[3]).type_as(tensor).reshape(new_tensor_shape)
	
	tensor          =   (tensor*index)
	tensor          =   tensor.reshape(tensor_shape)
	return tensor

def matrix_drop(tensor,select_percentage):
        tensor_shape    =   tensor.shape
        k               =   int(math.ceil(select_percentage*(tensor_shape[0]*tensor_shape[1])))
        topk            =   tensor.view(-1).abs().topk(k)
        threshold       =   topk[0][-1]
        index           =   tensor.abs()>=(threshold)
        index           =   index.type_as(tensor)
        tensor          =   (tensor*index)
        return tensor


def drop_nhwc(tensor,select_percentage):
        tensor_shape    =   tensor.shape
        k               =   int(math.ceil(select_percentage*(tensor_shape[0]*tensor_shape[1]*tensor_shape[2]*tensor_shape[3])))
        topk            =   tensor.view(-1).abs().topk(k)
        threshold       =   topk[0][-1]
        index           =   tensor.abs()>=(threshold)
        index           =   index.type_as(tensor)
        tensor          =   (tensor*index)
        return tensor

def drop_nhwc_send_th(tensor,select_percentage):
        tensor_shape    =   tensor.shape
        k               =   int(math.ceil(select_percentage*(tensor_shape[0]*tensor_shape[1]*tensor_shape[2]*tensor_shape[3])))
        topk            =   tensor.view(-1).abs().topk(k)
        threshold       =   topk[0][-1]
        index           =   tensor.abs()>=(threshold)
        index           =   index.type_as(tensor)
        tensor          =   (tensor*index)
        return tensor,topk[0][-1]

def drop_threshold(tensor,threshold):
        index           =   tensor.abs()>=(threshold.cuda())
        index           =   index.type_as(tensor)
        tensor          =   (tensor*index)
        return tensor

def drop_hwc(tensor,select_percentage):
        tensor_shape    =   tensor.shape
        k               =   int(math.ceil(select_percentage*(tensor_shape[1]*tensor_shape[2]*tensor_shape[3])))
        tensor          =   tensor.reshape(tensor_shape[0],tensor_shape[1]*tensor_shape[2]*tensor_shape[3])
        new_tensor_shape=   tensor.shape
        topk            =   tensor.abs().topk(k)
        interleaved     =   topk[0][:,-1].repeat_interleave(tensor_shape[1]*tensor_shape[2]*tensor_shape[3])
        interleaved     =   interleaved.reshape(new_tensor_shape)
        index           =   tensor.abs()>=(interleaved)
        index           =   index.type_as(tensor)
        tensor          =   (tensor*index)
        tensor          =   tensor.reshape(tensor_shape)
        return tensor

def drop_hwn(tensor,select_percentage):
        tensor=tensor.permute(1,0,2,3).contiguous()
        tensor=drop_hwc(tensor,select_percentage)
        tensor=tensor.permute(1,0,2,3).contiguous()
        return tensor

def drop_hw(tensor,select_percentage):
        tensor_shape    =   tensor.shape
        k               =   int(math.ceil(select_percentage*(tensor_shape[2]*tensor_shape[3])))
        tensor          =   tensor.reshape(tensor_shape[0],tensor_shape[1],tensor_shape[2]*tensor_shape[3])
        new_tensor_shape=   tensor.shape
        topk            =   tensor.abs().topk(k)
        interleaved     =   topk[0][:,:,-1].repeat_interleave(tensor_shape[2]*tensor_shape[3])
        interleaved     =   interleaved.reshape(new_tensor_shape)
        index           =   tensor.abs()>=(interleaved)
        index           =   index.type_as(tensor)
        tensor          =   (tensor*index)
        tensor          =   tensor.reshape(tensor_shape)
        return tensor

def drop_random(tensor,select_percentage):
        tensor_shape    =   tensor.shape
        drop_percentage =   1-select_percentage
        tensor          =   tensor.reshape(tensor_shape[0]*tensor_shape[1]*tensor_shape[2]*tensor_shape[3])
        index           =   torch.ones(tensor.shape).type_as(tensor)
        index           = index.uniform_() > drop_percentage 
        index           =   index.type_as(tensor)
        tensor          =   (tensor*index)
        tensor          =   tensor.reshape(tensor_shape)
        return tensor


