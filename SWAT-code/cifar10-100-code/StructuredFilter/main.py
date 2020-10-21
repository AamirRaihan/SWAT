'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
import shutil
import numpy as np
import time
import yaml

from models import *
from utils import progress_bar
import create_sparse

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    cudnn.deterministic=True
    cudnn.benchmark=False


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--gpuid', '-g', type=int, default=0, help='gpuid ')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', '-model', type=str, help='model')
parser.add_argument('--dataset', '-dataset', type=str, help='dataset')
parser.add_argument('--message', '-m', type=str, default='', help='appended in the name of checkpoint and log file')
parser.add_argument('--inference', '-inference', default=0, type=int, help='inference')
parser.add_argument('--checkpoint', '-checkpoint', default=None, type=str, help='ckpt')
parser.add_argument('--progress_bar', '-p', default=0, type=int, help='progress bar')
parser.add_argument('--dump_parameter', '-dump_parameter', default=0, type=int, help='dump_parameter')
parser.add_argument('--reproducibility', '-reproducibility', default=0, type=int, help='reproducibility')
parser.add_argument('--schedule-file', default='./schedule.yaml', type=str, help='yaml file containing learning rate schedule and rewire period schedule')

args = parser.parse_args()
device = 'cuda:'+str(args.gpuid)
print("-------------------------------------")
print("Argument Passed")
print("Model Selected:",args.model)
print("Dataset Selected:",args.dataset)
print("Hyperparameter File:",args.schedule_file)
print("Message:",args.message)
print("Num GPUs:",torch.cuda.device_count())
print("-------------------------------------")
with open(args.schedule_file, 'r') as stream:
    try:
        loaded_schedule = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

lr_schedule = loaded_schedule['lr_schedule']
yaml_sparsity=loaded_schedule['sparsity']
yaml_topk_period=loaded_schedule['topk_period']
yaml_num_conv_layer=loaded_schedule['num_conv_layer']
yaml_bn_weight_decay=loaded_schedule['bn_weight_decay']
yaml_nesterov=loaded_schedule['nesterov']
yaml_train_batch_size=loaded_schedule['train_batch_size']
yaml_test_batch_size=loaded_schedule['test_batch_size']
yaml_weight_decay=loaded_schedule['weight_decay']
yaml_momentum=loaded_schedule['momentum']
yaml_warmup=loaded_schedule['warmup']
yaml_total_epoch=loaded_schedule['total_epoch']
print("-------------------------------------")
print("Hyper Parameters")
print("Total Epoch:",yaml_total_epoch)
print("LR Schedule:",lr_schedule)
print("Warm Up:",yaml_warmup)
print("Sparsity:",yaml_sparsity)
print("TopK Period:",yaml_topk_period)
print("Num Conv Layer:",yaml_num_conv_layer)
print("train_batch_size:",yaml_train_batch_size)
print("test_batch_size:",yaml_test_batch_size)
print("weight_decay:",yaml_weight_decay)
print("bn_weight_decay:",yaml_bn_weight_decay)
print("momentum:",yaml_momentum)
print("nesterov:",yaml_nesterov)
print("-------------------------------------")
print("-------------------------------------")
assert args.model, 'Error: no model selected!'
assert args.dataset, 'Error: no dataset selected!'
ngpus_per_node = torch.cuda.device_count()
assert ngpus_per_node == 1


train_batch_size=yaml_train_batch_size
test_batch_size =yaml_test_batch_size
best_acc        =0  
start_epoch     =0 
train_acc_list  =[]
test_acc_list   =[] 
train_loss_list =[] 
test_loss_list  =[]

tag_string      =args.model + '_' + args.message +'_'+ args.dataset + '_batch_size_' + str(train_batch_size) + '_' + str(test_batch_size)
print("TagString: {}".format(tag_string))
print("-------------------------------------")
print("-------------------------------------")

# Data
cur_num_workers=2
if args.reproducibility:
    cur_num_workers=0
    seed_everything()

print('==> Preparing data..')
trainset=testset=trainloader=testloader=numchannels=numclasses=None
if args.dataset != 'Mnist' and args.dataset !='Cifar10' and args.dataset != 'Cifar100':
    assert(0)

#Note I have removed the distributed processing otherwise train_sampler should be provided
if args.dataset=='Cifar10' or args.dataset=='Cifar100':
    print('Dataset Selected {}'.format(args.dataset))
    numchannels=3
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset=='Cifar10':
        numclasses=10
        dataloader=datasets.CIFAR10
        datadirectory='./data/cifar10'
    else:
        numclasses=100
        dataloader=datasets.CIFAR100
        datadirectory='./data/cifar100'

    trainset = dataloader(root=datadirectory, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, train_batch_size, shuffle=True, num_workers=cur_num_workers)
    testset = dataloader(root=datadirectory, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, test_batch_size, shuffle=False, num_workers=cur_num_workers)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.dataset=='Mnist':
    print('Dataset Selected MNIST')
    numchannels=1
    numclasses=10
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),         # mean and std of mnist dataset
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(( 0.1307, ), (0.3081, )),
    ])
    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, train_batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, test_batch_size, shuffle=False, num_workers=2)


def select_model(x,numchannels,numclasses):
    return {
        'ResNet18':ResNet18(numchannels,numclasses),
        'ResNet34':ResNet34(numchannels,numclasses),
        'ResNet50':ResNet50(numchannels,numclasses),
        'ResNet101':ResNet101(numchannels,numclasses),
        'VGG16' : VGG('VGG16',numclasses),
        'DenseNet121' :DenseNet121(numclasses),
    }[x]



net= select_model(args.model,numchannels,numclasses)
net = net.to(device)
#DataParallel is used for multiGPUs configuration
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True

# This resume is not loading the optimizer checkpoint! What does that means?
def resume(checkpt=None):
    # Load checkpoint.
    global start_epoch
    global best_acc
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if checkpt==None:
        checkpt='./checkpoint/ckpt.t7'
    checkpoint = torch.load(checkpt)
    print("Loading checkpoint: ",checkpt)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.resume:
    resume()

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    global train_acc_list
    global train_loss_list
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    temp_data =torch.tensor([epoch,0,0,0,yaml_topk_period,yaml_sparsity,yaml_warmup])
    torch.save(temp_data,'temp_data_0')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        temp_data =torch.tensor([epoch,batch_idx,0,0,yaml_topk_period,yaml_sparsity,yaml_warmup])
        torch.save(temp_data,'temp_data_0')
        outputs = net(inputs)


        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if(args.progress_bar):
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    train_acc_list.append(acc)
    cum_train_loss= train_loss/(batch_idx+1)
    train_loss_list.append(cum_train_loss)
    print("Epoch: ", epoch, "Train Accuracy: ", acc)
def test(epoch):
    global best_acc
    global test_acc_list
    global test_loss_list
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            temp_data =torch.tensor([epoch,batch_idx,0,0,yaml_topk_period,yaml_sparsity,yaml_warmup])
            torch.save(temp_data,'temp_data_0')
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if(args.progress_bar):
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    test_acc_list.append(acc)
    cum_test_loss= (test_loss)/(batch_idx+1)
    test_loss_list.append(cum_test_loss)
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    print("Epoch: ", epoch, "Test Accuracy: ", acc)
       
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    if 1:#acc > best_acc: #comment the line for not using greedy algorithm in that scenario best_acc will become the acc
        print('Saving..')
        torch.save(state, './checkpoint/ckpt.t7')
        if acc > best_acc: #comment the line for not using greedy algorithm in that scenario best_acc will become the acc
            torch.save(state, './checkpoint/best.t7')
            best_acc = acc
    
    if args.dump_parameter:
        torch.save(state,'./checkpoint/ckpt_'+tag_string+'_Epoch'+str(epoch)+'.t7')
    
    return acc


def get_schedule_val(schedule,query):
    val = list(schedule[-1].values())[0]
    for i,entry in enumerate(schedule):
        if query < list(entry)[0]:
            val = list(schedule[i-1].values())[0]
            break
    return val

def adjust_learning_rate(optimizer, epoch,schedule):
    """Sets the learning rate to the initial LR divided by 5 at 30th, 60th and 90th epochs"""
    lr = get_schedule_val(schedule,epoch)

    print('setting learning rate to ' + repr(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

num_gpus= torch.cuda.device_count()
if os.path.exists('./temp_data_0'):
    os.remove('./temp_data_0')
for i in range(num_gpus):
    for layer in range(yaml_num_conv_layer):
        threshold_data=torch.tensor([torch.zeros(1),torch.zeros(1)])
        torch.save(threshold_data,'./dump/threshold_'+str(layer)+'_'+str(i))


#uncomment for printing the function name
#import sys
#sys.settrace(tracefunc)
if args.inference:
    print("######INFERENCE######")
    assert args.checkpoint != None
    optimizer = optim.SGD(net.parameters(),lr=0.1)
    resume(args.checkpoint)
    print("restored_from_epoch",start_epoch)
    print("best_acc",best_acc)
    test(start_epoch+1)
    print("######INFERENCE######")

else:
    # Resume is not allowed for training
    if args.resume:
        assert 0
    
    #Default LR=0.1
    optimizer = optim.SGD(net.parameters(),lr=0.1, momentum=yaml_momentum, weight_decay=yaml_weight_decay)
    print("-------------------------------------")
    print("-------------------------------------")
    print("Training Started")
    
    for epoch in range(0, yaml_total_epoch):
        adjust_learning_rate(optimizer, epoch,lr_schedule)
        train(epoch)
        test(epoch)


    checkpoint = torch.load("./checkpoint/best.t7")
    net.load_state_dict(checkpoint['net'])
    net.load_state_dict(create_sparse.sparse_filter(net.state_dict(),1-yaml_sparsity))
    create_sparse.print_sparsity_structured(net.state_dict())
    top1_acc=test(yaml_total_epoch)
    state = {
        'net': net.state_dict(),
        'acc': top1_acc,
        'epoch': yaml_total_epoch,
    }
    torch.save(state, './checkpoint/best.t7')
    shutil.copy('./checkpoint/best.t7', './checkpoint/ckpt_'+ tag_string +'.t7')
    acc_dict={'TrainAccuracy':train_acc_list,'TestAccuracy':test_acc_list,'TrainLoss':train_loss_list,'TestLoss':test_loss_list}
    np.save( './checkpoint/acc_log_'+ tag_string+  '.npy', acc_dict)
