#!/bin/bash 

CLASSES=196
PROJECT="cars"

for BATCH_SIZE in 128 64 32; do
    for SGD_MOM in  0.9 0.85 0.80; do 
        for MODEL_NAME in "MobileNetV3Large"; do 
            for FRAMEWORK in "tensorflow" "upstride_type2"; do 
                if [[ "$FRAMEWORK" == "upstride_type2" ]]; then 
                    FACTOR=4
                else
                    FACTOR=1
                fi
                python train.py --model_name $MODEL_NAME \
                    --num_epochs 400 \
                    --early_stopping 40 \
                    --factor $FACTOR \
                    --num_classes $CLASSES \
                    --project $PROJECT \
                    --framework $FRAMEWORK \
                    --use_wandb True \
                    --run_name "${PROJECT}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
                    --log_dir "/results/${PROJECT}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
                    --checkpoint_dir "/checkpoints/${PROJECT}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
                    --configuration.with_mixed_precision True \
                    --configuration.profiler True \
                    --dataloader.train_list ColorJitter Normalize RandomCropThenResize RandomHorizontalFlip Cutout Translate RandomRotate \
                    --dataloader.Normalize.scale_in_zero_to_one True \
                    --dataloader.Normalize.only_subtract_mean True \
                    --dataloader.ColorJitter.clip 0. 255. \
                    --dataloader.RandomRotate.angle 15 \
                    --dataloader.Cutout.length 32 \
                    --dataloader.Translate.width_shift_range 0.3 \
                    --dataloader.Translate.height_shift_range 0.3 \
                    --dataloader.val_list Normalize CentralCrop \
                    --dataloader.CentralCrop.size 224 224 \
                    --dataloader.data_dir /datasets/stanford_cars/tfrecords \
                    --dataloader.name cars_196 \
                    --dataloader.train_split_id train \
                    --dataloader.val_split_id validation \
                    --dataloader.batch_size $BATCH_SIZE \
                    --optimizer.name sgd_momentum \
                    --optimizer.momentum $SGD_MOM \
                    --optimizer.lr 0.1 \
                    --optimizer.lr_decay_strategy.activate True \
                    --optimizer.lr_decay_strategy.lr_params.patience 20 \
                    --optimizer.lr_decay_strategy.lr_params.strategy lr_reduce_on_plateau \
                    --optimizer.lr_decay_strategy.lr_params.decay_rate 0.3 \
                     > "${PROJECT}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}.log" 
            done
        done
    done
done