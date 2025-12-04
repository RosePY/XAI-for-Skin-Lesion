#!/bin/bash

for img_idx in {0..2419}; do
    CUDA_VISIBLE_DEVICES=5 python shap_exp_iv4.py $img_idx
done