#!/bin/bash

#### Analysis Number ####
analysis_num=$1

#### Training Hyperparameters ####
num_epoch=10
start_epoch=0
batch_size=128
num_workers=3

#### Optimization Hyperparameters ####
optimizer='adam'
learning_rate=1e-5
momentum=0.
weight_decay=0.

#### Resnet Specifications ####
resnet_type='resnet18'
pretrained=True

#### Devices Specification ####
devices='cuda:0' # cpu, multi, cuda:*
gpu=True

python resnet_eval.py   --analysis $analysis_num \
                        --epochs $num_epoch \
                        --start-epoch $start_epoch \
                        --batch-size $batch_size \
                        --worker $num_workers \
                        --learning-rate $learning_rate \
                        --optimizer $optimizer \
                        --resnet $resnet_type \
                        --pretrained \
                        --gpu \
                        --device $devices
