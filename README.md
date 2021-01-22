# UpStride Classification API (branch: experiment/papers)

This code repository contains the work done migration, validation and run DCN original source code in TF 2.4

In order to validation or reproduce the tests check out upstride_python engine from the commit 47c9e1303556bee70a636fe1d866366a620e48f3 (code_cleaning branch)

One way is to mount upstride_python and do a ``` pip install . ```

legend: 
dcn ours: dcn with python engine 
dcn source: code from deep_complex_networks repository

# DCN WideShallow ComplexNet

Use the below configuration start the experiment. Note: We use the same dataloader as DCN source (Deep complex Networks repository) so some configuration here are not required. For the purpose of comparision to our inital experiments using our own dataloader the parameters are kept for reference in the below configuration file. 

```bash
python train_dcn.py \
            --model_name WSComplexNetUpStride \
            --framework upstride_type1 \
            --factor 2 \
            --checkpoint_dir ./tmp/bench_dcn \
            --log_dir ./tmp/log/bench_dcn \
            --input_size 3 32 32 \
            --num_classes 10 \
            --dataloader.name cifar10 \
            --dataloader.train_list Normalize RandomHorizontalFlip Translate \
            --dataloader.Translate.width_shift_range 0.125 \
            --dataloader.Translate.height_shift_range 0.125 \
            --dataloader.val_list Normalize \
            --dataloader.data_dir /CIFAR10/tfrecords \
            --dataloader.name CIFAR10 \
            --dataloader.train_split_id train \
            --dataloader.val_split_id validation \
            --dataloader.batch_size 64 \
            --dataloader.Normalize.only_subtract_mean true \
            --dataloader.Normalize.scale_in_zero_to_one false \
            --dataloader.Normalize.mean 0.491 0.482 0.446 \
            --early_stopping 200 \
            --num_epochs 200 \
            --optimizer.lr 0.01 \
            --optimizer.lr_decay_strategy.lr_params.strategy explicit_schedule \
            --optimizer.lr_decay_strategy.lr_params.drop_schedule 10 100 120 150 \
            --optimizer.lr_decay_strategy.lr_params.list_lr 0.1 0.01 0.001 0.0001 \
            --optimizer.clipnorm 1.0 \
            --configuration.profiler true \
            --conversion_params.tf2up_strategy learned \
            --conversion_params.up2tf_strategy concat \
            --debug.write_graph true \
            --debug.write_histogram true \
            --debug.log_gradients true > dcn_bench.log
```

Corresponding run config in the Deep Complex Networks is 
```bash
python scripts/run.py train -w WORKDIR --model complex --sf 12 --nb 16
```

To run DCN_source code in TF 2.4 use the below run config
```bash
python run.py train -w WORKDIR --model complex --sf 12 --nb 16
```

Changes: 
- channel first 
- modified complexnet to use complex batch norm
- dcn.py contains the code migrated from deep_complex_networks repo
- generic_model.py file modified to include weight regularization and parameters for TF2Upstride (learned strategy)
- train_dcn.py to run dcn with python engine using dcn source dataloader 
- run.py and training_from_dcn.py are used to run DCN source on TF 2.4

Validations/tests done so far: 
- Unit tests on Complex Batch Norm and Complex Initializers gives identical values
- ported DCN source to TF 2.4 to validate the changes. 
    - the performance is even less compared to using Theano backend. 

Known differences: 
- line 181 in conv_from_dcn.py (upstride_python) requires output filters to be twice as much to match the complex independent initalization.
    - This difference doesn't cause the changes in total parameters when run with Keras 2.2.5 (Theano backend), which is weird. In the latest version of keras, without multiplying the output filters by 2 raises an error in the shape.

Grid search - in progress