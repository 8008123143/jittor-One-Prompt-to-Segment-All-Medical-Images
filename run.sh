#!/bin/bash
# One-Prompt Jittor — ISIC 2016 training
python train.py -net oneprompt -mod one_adpt -exp_name basic_exp -b 8 -image_size 1024 -dataset isic -data_path ../data -baseline 'unet' -vis 50 -val_freq 10
