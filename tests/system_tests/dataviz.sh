

python dataviz.py  \
    --dataloader.batch_size 128 \
    --dataloader.name imagenette/full-size-v2 \
    --dataloader.train_list RandomCropThenResize RandomHorizontalFlip Cutout ColorJitter Translate \
    --dataloader.val_list CentralCrop \
    --dataloader.val_split_id validation \
    --dataloader.train_split_id train \
    --dataloader.Translate.width_shift_range 0.2 \
    --dataloader.Translate.height_shift_range 0.2 \
    --dataloader.RandomCrop.size 224 224 3 \
    --dataloader.CentralCrop.size 224 224 \
    --dataloader.Cutout.length 16 \
    