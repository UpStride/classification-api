#!/bin/bash 

CLASSES=10
LR=0.1
SGD_MOM=0.8
FACTOR=2
FRAMEWORK="upstride_type2"

for RUN in 1 2 3; do 
    for PROJECT in "imagenette" "cifar10" ; do 
        if [[ "$PROJECT" == "imagenette" ]]; then 
            python train.py --model_name MobileNetV3Large \
                --num_epochs 400 \
                --early_stopping 40 \
                --factor $FACTOR \
                --num_classes $CLASSES \
                --project $PROJECT \
                --framework $FRAMEWORK \
                --use_wandb false \
                --run_name "best_3_${RUNS}_${PROJECT}_MobileNetV3Large_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
                --log_dir "/results/best_3/best_3_${RUNS}_${PROJECT}_MobileNetV3Large_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
                --checkpoint_dir "/checkpoints/best_3_${RUNS}_${PROJECT}_MobileNetV3Large_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
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
                --dataloader.data_dir /datasets/tfrecords \
                --dataloader.name imagenette \
                --dataloader.train_split_id train \
                --dataloader.val_split_id validation \
                --dataloader.batch_size 128 \
                --optimizer.name sgd_momentum \
                --optimizer.momentum $SGD_MOM \
                --optimizer.lr $LR \
                --optimizer.lr_decay_strategy.activate True \
                --optimizer.lr_decay_strategy.lr_params.patience 20 \
                --optimizer.lr_decay_strategy.lr_params.strategy lr_reduce_on_plateau \
                --optimizer.lr_decay_strategy.lr_params.decay_rate 0.3 \
                > "best_3_${RUN}_${PROJECT}_MobileNetV3Large_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}.log" 
        else
                python train.py --model_name MobileNetV3LargeCIFAR \
                --num_epochs 400 \
                --early_stopping 40 \
                --factor $FACTOR \
                --num_classes $CLASSES \
                --project $PROJECT \
                --framework $FRAMEWORK \
                --use_wandb True \
                --run_name "best_3_${RUN}_${PROJECT}_MobileNetV3Large_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
                --log_dir "/results/best_3/best_3_${RUN}_${PROJECT}_MobileNetV3Large_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
                --checkpoint_dir "/checkpoints/best_3_${RUN}_${PROJECT}_MobileNetV3Large_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}" \
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
                --dataloader.name cifar10 \
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
                > "best_3_${RUN}_${PROJECT}_MobileNetV3Large_${FRAMEWORK}_MOM_${SGD_MOM}_LR_${LR}.log" 
            fi
    done
done