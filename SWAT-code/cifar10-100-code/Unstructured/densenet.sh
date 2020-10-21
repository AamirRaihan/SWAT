#!/bin/sh
python main.py -model "DenseNet121" -dataset="Cifar100" --schedule-file densenet_schedule.yaml
