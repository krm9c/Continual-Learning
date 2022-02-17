#!/bin/bash

# COBALT -n 1
# COBALT -A Performance
# COBALT -t 6:00:00

export https_proxy="https://proxy:3128"
export http_proxy="http://proxy:3128"
export ftp_proxy="ftp://proxy:3128"

source activate torch
# python get_data.py all
python py_run.py --save_dir "Res/wind_NASH" --opt "NASH" --json_file "wind.json" --total_runs 1
source deactivate 
