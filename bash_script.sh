#!/bin/bash 

for L2 in 0.0005 0.0001; do
  for RUN in 1 2 3; do
    for BS in 64 128; do 
      python train_dcn.py \
          --model_name WSComplexNetUpStride \
          --framework upstride_type1 \
          --factor 2 \
          --checkpoint_dir ./grid_dcn_${RUN}_${BS}_${L2} \
          --log_dir ./log/grid_dcn_${RUN}_${BS}_${L2} \
          --input_size 3 32 32 \
          --num_classes 10 \
          --dataloader.name cifar10 \
          --dataloader.train_list Normalize RandomHorizontalFlip Translate \
          --dataloader.Translate.width_shift_range 0.125 \
          --dataloader.Translate.height_shift_range 0.125 \
          --dataloader.val_list Normalize \
          --dataloader.data_dir /CIFAR10/ \
          --dataloader.name tfrecords \
          --dataloader.train_split_id train \
          --dataloader.val_split_id validation \
          --dataloader.batch_size ${BS} \
          --dataloader.Normalize.only_subtract_mean true \
          --dataloader.Normalize.scale_in_zero_to_one false \
          --dataloader.Normalize.mean 0.491 0.482 0.446 \
          --early_stopping 200 \
          --num_epochs 200 \
          --optimizer.lr 0.1 \
          --optimizer.lr_decay_strategy.lr_params.strategy explicit_schedule \
          --optimizer.lr_decay_strategy.lr_params.drop_schedule 10 100 120 150 \
          --optimizer.lr_decay_strategy.lr_params.list_lr 0.1 0.01 0.001 0.0001 \
          --optimizer.clipnorm 1.0 \
          --configuration.profiler true \
          --conversion_params.tf2up_strategy learned \
          --conversion_params.up2tf_strategy concat \
          --weight_decay ${L2} > grid_dcn_${RUN}_${BS}_${L2}.log
      done
  done
done