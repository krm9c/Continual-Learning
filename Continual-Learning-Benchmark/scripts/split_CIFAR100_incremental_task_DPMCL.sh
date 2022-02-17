#!/bin/bash

GPUID=$1
OUTDIR=outputs/split_CIFAR100_incremental_task
REPEAT=2
mkdir -p $OUTDIR


export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh" ]; then
        . "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh"
    else
        export PATH="/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate torchRL


# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer SGD     --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --momentum 0.9 --weight_decay 1e-4           --lr 0.1   --offline_training         | tee ${OUTDIR}/Offline_SGD.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug  --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet                                              --lr 0.001 --offline_training         | tee ${OUTDIR}/Offline_adam.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet                                              --lr 0.001                            | tee ${OUTDIR}/Adam.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer SGD     --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet                                              --lr 0.01                             | tee ${OUTDIR}/SGD.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adagrad --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet                                              --lr 0.01                             | tee ${OUTDIR}/Adagrad.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 3000     | tee ${OUTDIR}/EWC_online.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 100      | tee ${OUTDIR}/EWC.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type regularization --agent_name SI  --lr 0.001 --reg_coef 2               | tee ${OUTDIR}/SI.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type regularization --agent_name L2  --lr 0.001 --reg_coef 1               | tee ${OUTDIR}/L2.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name Naive_Rehearsal_1400  --lr 0.001          | tee ${OUTDIR}/Naive_Rehearsal_1400.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization  --agent_name Naive_Rehearsal_5600  --lr 0.001          | tee ${OUTDIR}/Naive_Rehearsal_5600.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type regularization --agent_name MAS --lr 0.001 --reg_coef 10              | tee ${OUTDIR}/MAS.log
# python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization --agent_name NASH_16k --lr 0.001         | tee ${OUTDIR}/NASH_16k.log


python -u ../iBatchLearn.py --dataset CIFAR100 --train_aug --repeat $REPEAT --optimizer Adam    --force_out_dim 0 --first_split_size 20 --other_split_size 20 --schedule 80 120 160 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization --agent_name DPMCL_5600 --lr 0.001     | tee ${OUTDIR}/DPMCL_16k.log