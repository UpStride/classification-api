# train a mobilenet channel first with tensorflow
# need a least 6 GB of VRAM
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

# Total params: 2,270,794
# Trainable params: 2,236,682
# Non-trainable params: 34,112
# Epoch 2 takes 40 second using a GTX 1080 and should reach >30% validation accuracy

rm -r /tmp/results 
rm -r /tmp/checkpoint

# train a mobilenet channel last with tensorflow
python3 train.py \
    --model.name MobileNetV2 \
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

# Total params: 2,270,794
# Trainable params: 2,236,682
# Non-trainable params: 34,112
# Epoch 2 takes 40 second using a GTX 1080 and should reach >30% validation accuracy

rm -r /tmp/results 
rm -r /tmp/checkpoint

# train a resnet channel last with tensorflow
python3 train.py \
    --model.name ResNet18 \
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

# Total params: 11,188,728
# Trainable params: 11,180,920
# Non-trainable params: 7,808

# train a resnet channel first with tensorflow
python3 train.py \
    --model.name ResNet18NCHW \
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

# Total params: 11,188,728
# Trainable params: 11,180,920
# Non-trainable params: 7,808

# train a resnet cifar channel last with tensorflow
python3 train.py \
    --model.name ResNet20CIFAR \
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

# Total params: 294,250
# Trainable params: 292,874
# Non-trainable params: 1,376


# train a mobilenet v3 channel first with tensorflow
python3 train.py \
    --model.name MobileNetV3SmallNCHW \
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

rm -r /tmp/results 
rm -r /tmp/checkpoint

# train a mobilenet v3 channel last with tensorflow
python3 train.py \
    --model.name MobileNetV3Small \
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

rm -r /tmp/results 
rm -r /tmp/checkpoint

# Total params: 1,538,914
# Trainable params: 1,526,802
# Non-trainable params: 12,112

