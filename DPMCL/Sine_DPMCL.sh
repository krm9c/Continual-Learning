#!/bin/bash
#COBALT -n 1
#COBALT -q gpu_mules
#COBALT -A Performance
#COBALT -t 5:00:00
# export PATH=/soft/interpreters/python/intelpython/27/bin:$PATH

export https_proxy="https://proxy:3128"
export http_proxy="http://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

for seed in 100
do
    source activate torch
    python py_run.py --save_dir "omni/DPMCL_${seed}_"  --kappa ${seed} --opt "DPMCL" --json_file "omni.json" --total_runs 1
    source deactivate 
done