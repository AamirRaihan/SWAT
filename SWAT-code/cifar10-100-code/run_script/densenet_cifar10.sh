#!/bin/sh
python main.py -model "DenseNet121" -dataset="Cifar10" --schedule-file $1 
