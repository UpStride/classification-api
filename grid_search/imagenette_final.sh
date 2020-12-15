#!/bin/bash 

CLASSES=10
MODEL_NAME="MobileNetV3Large"
DATA_DIR="/datasets/imagenette/imagenette2/tfrecords"
DATASET_NAME="imagenette"
BATCH_SIZE=128
SGD_MOM=0.85
FRAMEWORK="tensorflow"

for RUN in 1 2 3; do 
    for FACTOR in 8 4 2; do
        python train.py --model_name $MODEL_NAME \
            --num_epochs 400 \
            --early_stopping 40 \
            --factor $FACTOR \
            --num_classes $CLASSES \
            --framework $FRAMEWORK \
            --use_wandb false \
            --run_name "${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
            --log_dir "/results/tf_runs/${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
            --checkpoint_dir "/checkpoints/tf_runs/${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
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
            --dataloader.data_dir $DATA_DIR \
            --dataloader.name $DATASET_NAME \
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
                > "tf_run_${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}.log" 
    done
done

# # run 0.8 64 
# BATCH_SIZE=64
# SGD_MOM=0.8

# for RUN in 1 2 ; do
#     for FRAMEWORK in 'upstride_type2' 'tensorflow'; do
#         if [[ "$FRAMEWORK" == "upstride_type2" ]]; then 
#             FACTOR=4
#         else
#             FACTOR=1
#         fi
#         python train.py --model_name $MODEL_NAME \
#             --num_epochs 400 \
#             --early_stopping 40 \
#             --factor $FACTOR \
#             --num_classes $CLASSES \
#             --framework $FRAMEWORK \
#             --use_wandb false \
#             --run_name "${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
#             --log_dir "/results/${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
#             --checkpoint_dir "/checkpoints/${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}" \
#             --configuration.with_mixed_precision True \
#             --configuration.profiler True \
#             --dataloader.train_list ColorJitter Normalize RandomCropThenResize RandomHorizontalFlip Cutout Translate RandomRotate \
#             --dataloader.Normalize.scale_in_zero_to_one True \
#             --dataloader.Normalize.only_subtract_mean True \
#             --dataloader.ColorJitter.clip 0. 255. \
#             --dataloader.RandomRotate.angle 15 \
#             --dataloader.Cutout.length 32 \
#             --dataloader.Translate.width_shift_range 0.3 \
#             --dataloader.Translate.height_shift_range 0.3 \
#             --dataloader.val_list Normalize CentralCrop \
#             --dataloader.CentralCrop.size 224 224 \
#             --dataloader.data_dir $DATA_DIR \
#             --dataloader.name $DATASET_NAME \
#             --dataloader.train_split_id train \
#             --dataloader.val_split_id validation \
#             --dataloader.batch_size $BATCH_SIZE \
#             --optimizer.name sgd_momentum \
#             --optimizer.momentum $SGD_MOM \
#             --optimizer.lr 0.1 \
#             --optimizer.lr_decay_strategy.activate True \
#             --optimizer.lr_decay_strategy.lr_params.patience 20 \
#             --optimizer.lr_decay_strategy.lr_params.strategy lr_reduce_on_plateau \
#             --optimizer.lr_decay_strategy.lr_params.decay_rate 0.3 \
#                 > "${RUN}_${MODEL_NAME}_${FRAMEWORK}_MOM_${SGD_MOM}_BS_${BATCH_SIZE}.log" 
#     done
# done

