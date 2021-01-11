#!/bin/sh
python main.py -model "DenseNet121" -dataset="Cifar100" --schedule-file $1 
