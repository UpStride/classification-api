# train a mobilenet channel first with tensorflow
# need a least 6 GB
python3 train.py \
    --model.name MobileNetV2NCHW \
    --model.factor 1 \
    --model.num_classes 10 \
    --model.input_size 224 224 3 \
    --num_epochs 2 \
    --checkpoint_dir /tmp/checkpoint \
    --log_dir /tmp/results \
    --dataloader.batch_size 128 \
    --dataloader.name imagenette/full-size-v2 \
    --early_stopping 100 \
    --dataloader.train_list RandomCropThenResize Normalize RandomHorizontalFlip Cutout ColorJitter Translate \
    --dataloader.val_list Normalize CentralCrop \
    --dataloader.val_split_id validation \
    --dataloader.train_split_id train \
    --dataloader.Translate.width_shift_range 0.2 \
    --dataloader.Translate.height_shift_range 0.2 \
    --dataloader.RandomCrop.size 224 224 3 \
    --dataloader.CentralCrop.size 224 224 \
    --dataloader.Cutout.length 16 \
    --optimizer.name sgd_nesterov \
    --optimizer.lr 0.1 \
    --optimizer.lr_decay_strategy.lr_params.patience 20 \
    --optimizer.lr_decay_strategy.lr_params.strategy cosine_decay \
    --optimizer.lr_decay_strategy.lr_params.decay_rate 0.3 \
    --config.mixed_precision
