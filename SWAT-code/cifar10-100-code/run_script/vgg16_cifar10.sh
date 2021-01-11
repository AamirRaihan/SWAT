#!/bin/sh
python main.py -model "VGG16" -dataset="Cifar10" --schedule-file $1 
