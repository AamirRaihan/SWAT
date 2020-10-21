#!/bin/bash
python main.py -a resnet50  /datasets/IMAGENET-UNCROPPED/ --label_smoothing 0.1 --warmup 0 --sparsity 0.5 --topk_period 5 --nesterov --lrwarmup 5 -b 256 
#python main.py -a resnet50  /datasets/IMAGENET-UNCROPPED/ --label_smoothing 0.1 --warmup 0 --sparsity 0.5 --topk_period 5 --nesterov --lrwarmup 5 -b 256 --resume "./model_best.pth.tar" --evaluate 
