#!/bin/bash


isic18_root="../isic-data/HAM10000/ISIC2018_Task3_Training_Input/"
train_i18_csv="csv_splits/isic2018n_train.csv"
val_i18_csv="csv_splits/isic2018n_val.csv"
test_i18_csv="csv_splits/isic2018n_test.csv"


RUNS=5
GPU=5
for time in $(seq 0 $RUNS); do
    # CUDA_VISIBLE_DEVICES=$GPU python train_iv4.py with train_root=${isic18_root} train_csv=${train_i18_csv} epochs=100\
      # val_root=${isic18_root} val_csv=${val_i18_csv} model_name="inceptionv4" exp_name="iv4_i18nf_5runs_${time}"
    CUDA_VISIBLE_DEVICES=$GPU python test_model.py results-comet-iv4/iv4_i18nf_5runs_${time}/checkpoints/model_best.pth\
     ${isic18_root} ${test_i18_csv} -n 50 -p -jpg -name iv4_i18nf_5runs_${time}_test > results-comet-iv4/iv4_i18nf_5runs_${time}/test_isic2018t3.txt	
done

# for time in $(seq 0 $RUNS); do
#     CUDA_VISIBLE_DEVICES=$GPU python train_rn50.py with train_root=${isic18_root} train_csv=${train_i18_csv} epochs=100\
#       val_root=${isic18_root} val_csv=${val_i18_csv} model_name="resnet50" exp_name="rn50_i18nf_5runs_${time}"
#     CUDA_VISIBLE_DEVICES=$GPU python test_rn50.py results-comet-iv4/rn50_i18nf_5runs_${time}/checkpoints/model_best.pth\
#      ${isic18_root} ${test_i18_csv} -n 50 -p -jpg -name rn50_i18nf_5runs_${time}_test > results-comet-iv4/rn50_i18nf_5runs_${time}/test_isic2018t3.txt	
# done
