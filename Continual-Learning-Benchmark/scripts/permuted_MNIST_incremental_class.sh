#!/bin/bash

GPUID=$0
OUTDIR=../outputs/permuted_MNIST_incremental_class
REPEAT=5
mkdir -p $OUTDIR


source activate torch
export http_proxy="http://proxy:3128"
export https_proxy="https://proxy:3128"
export ftp_proxy="ftp://proxy:3128"



python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 3 --force_out_dim 100  --schedule 2  --batch_size 128 --model_name MLP1000                                                     --lr 0.0001  --offline_training  | tee ${OUTDIR}/Offline.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000                                                     --lr 0.0001                      | tee ${OUTDIR}/Adam.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000                                                     --lr 0.001                       | tee ${OUTDIR}/SGD.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adagrad --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000                                                         --lr 0.001                   | tee ${OUTDIR}/Adagrad.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name EWC_online_mnist --lr 0.0001 --reg_coef 50  | tee ${OUTDIR}/EWC_online.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name EWC_mnist        --lr 0.0001 --reg_coef 10  | tee ${OUTDIR}/EWC.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type regularization --agent_name SI         --lr 0.0001 --reg_coef 0.3       | tee ${OUTDIR}/SI.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type regularization --agent_name L2         --lr 0.0001 --reg_coef 0         | tee ${OUTDIR}/L2.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_4000   --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_16000  --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_16000.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer Adam    --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type regularization --agent_name MAS        --lr 0.0001 --reg_coef 0.003     | tee ${OUTDIR}/MAS.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name GEM_4000   --lr 0.1 --reg_coef 0.5          | tee ${OUTDIR}/GEM_4000.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name DPMCL_16k  --lr 0.1 --reg_coef 0.5          | tee ${OUTDIR}/DPMCL.log
python -u ../iBatchLearn.py  --repeat $REPEAT --incremental_class --optimizer SGD     --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name NASH_16k  --lr 0.1 --reg_coef 0.5           | tee ${OUTDIR}/NASH.log
