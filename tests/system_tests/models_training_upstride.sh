# train a mobilenet channel first with upstride
# need a least 6 GB of VRAM

python train.py \
    --model_name MobileNetV2Cifar10NCHW \
    --model.upstride_type 2 \
    --model.factor 4 \
    --model.num_classes 10 \
    --model.input_size 32 32 3 \
    --num_epochs 1000 \
    --checkpoint_dir /tmp/checkpointdata2345 \
    --log_dir log/translate \
    --dataloader.name cifar10 \
    --dataloader.train_list RandomHorizontalFlip Translate Cutout Normalize \
    --dataloader.val_list Normalize \
    --dataloader.val_split_id test \
    --dataloader.Resize.size 36 36 \
    --dataloader.RandomCrop.size 32 32 3 \
    --dataloader.Translate.width_shift_range 0.25 \
    --dataloader.Translate.height_shift_range 0.25 \
    --dataloader.Cutout.length 4 \
    --dataloader.batch_size 128 \
    --early_stopping 40 \
    --optimizer.lr 0.1 \
    --optimizer.lr_decay_strategy.lr_params.patience 20 \
    --optimizer.lr_decay_strategy.lr_params.strategy lr_reduce_on_plateau \
    --optimizer.lr_decay_strategy.lr_params.decay_rate 0.3 \
    --config.mixed_precision

rm -r /tmp/results 
rm -r /tmp/checkpoint
