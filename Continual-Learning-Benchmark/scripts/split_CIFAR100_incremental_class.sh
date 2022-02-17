#!/bin/bash

GPUID=$1
OUTDIR=outputs/split_CIFAR100_incremental_class
REPEAT=2
mkdir -p $OUTDIR


export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

source /lus/theta-fs0/software/thetagpu/conda/pt_master/2021-05-12/mconda3/setup.sh



# source activate torch
# source activate torch

# export http_proxy="http://proxy:3128"
# export https_proxy="https://proxy:3128"
# export ftp_proxy="ftp://proxy:3128"


python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 100 --no_class_remap --first_split_size 20 --other_split_size 20 --schedule 3  --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization --agent_name NASH_5600 --lr 0.001  |tee  ${OUTDIR}/NASH.log
