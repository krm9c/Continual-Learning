#!/bin/bash

OUTDIR=outputs/split_MNIST_incremental_class
REPEAT=5
source activate torch
export http_proxy="http://proxy:3128"
export https_proxy="https://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

mkdir -p $OUTDIR

python -u ../iBatchLearn.py --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 2 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name MAS --lr 0.001 --reg_coef 1        |tee  ${OUTDIR}/MAS.log
