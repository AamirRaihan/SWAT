#!/bin/sh
python main.py -model "VGG16" -dataset="Cifar100" --schedule-file $1 
