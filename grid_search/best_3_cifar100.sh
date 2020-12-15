#!/bin/bash 

CLASSES=100
LR=0.05
BATCH_SIZE=64
SGD_MOM=0.8
PROJECT="CIFAR100"

for RUN in  1 2 3; do 
    for FACTOR in 2 4; do 
        for FRAMEWORK in "upstride_type2" "tensorflow"; do 
            if [[ "$FRAMEWORK" == "tensorflow" ]]; then 
                FACTOR=1
            fi
            python train.py --model_name MobileNetV3LargeCIFAR \
            --num_epochs 400 \
            --early_stopping 40 \
            --factor $FACTOR \
            --num_classes $CLASSES \
            --project $PROJECT \
            --framework $FRAMEWORK \
            --use_wandb false \
            --run_name "best_3_${RUN}_${PROJECT}_MobileNetV3LargeCIFAR_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
            --log_dir "/results/best_3_cifar100/best_3_${RUN}_${PROJECT}_MobileNetV3LargeCIFAR_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
            --checkpoint_dir "/checkpoints/best_3_${RUN}_${PROJECT}_MobileNetV3LargeCIFAR_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
            --configuration.with_mixed_precision True \
            --configuration.profiler True \
            --input_size 32 32 3 \
            --dataloader.train_list Normalize RandomHorizontalFlip Cutout Translate \
            --dataloader.Normalize.scale_in_zero_to_one True \
            --dataloader.Normalize.only_subtract_mean True \
            --dataloader.Cutout.length 4 \
            --dataloader.Translate.width_shift_range 0.25 \
            --dataloader.Translate.height_shift_range 0.25 \
            --dataloader.val_list Normalize \
            --dataloader.name cifar100 \
            --dataloader.train_split_id train \
            --dataloader.val_split_id test \
            --dataloader.batch_size 128 \
            --optimizer.name sgd_momentum \
            --optimizer.momentum $SGD_MOM \
            --optimizer.lr $LR \
            --optimizer.lr_decay_strategy.activate True \
            --optimizer.lr_decay_strategy.lr_params.patience 20 \
            --optimizer.lr_decay_strategy.lr_params.strategy lr_reduce_on_plateau \
            --optimizer.lr_decay_strategy.lr_params.decay_rate 0.3 \
            > "best_3_${RUN}_${PROJECT}_MobileNetV3LargeCIFAR_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}.log" 
        done
    done
done