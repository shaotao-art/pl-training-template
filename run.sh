#!/bin/bash
device_idx=0
config_file=./configs/config.py
run_name=long_fit

CUDA_VISIBLE_DEVICES=$device_idx  python run.py \
                        --config $config_file\
                        --run_name $run_name \
