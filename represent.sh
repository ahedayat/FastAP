#!/bin/bash

#### Analysis Number ####
analysis_num=$1

#### Hyperparameters ####
start_epoch=9
num_workers=1

#### Resnet Specifications ####
resnet_type='resnet18'
pretrained=True

#### Devices Specification ####
devices='cuda:0' # cpu, multi, cuda:*
gpu=True

python resnet_representation.py    --analysis $analysis_num \
                            --start-epoch $start_epoch \
                            --worker $num_workers \
                            --resnet $resnet_type \
                            --pretrained \
                            --gpu \
                            --device $devices
