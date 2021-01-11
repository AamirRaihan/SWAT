#!/bin/bash
#python main.py -model "WRN-28-10" -dataset="Cifar10" --schedule-file ./run_configurations/wide_resnet_schedule.yaml
python main.py -model "WRN-16-8" -dataset="Cifar10" --schedule-file $1 
