#!/bin/bash

OUTDIR=outputs/permuted_MNIST_incremental_class
REPEAT=5
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


python -u   ../iBatchLearn.py --repeat $REPEAT --incremental_class --optimizer SGD --n_permutation 10 --force_out_dim 100 --schedule 2  --batch_size 128 --model_name MLP1000 --agent_type customization  --agent_name NASH_16k  --lr 0.1 --reg_coef 0.5  | tee ${OUTDIR}/NASH.log
