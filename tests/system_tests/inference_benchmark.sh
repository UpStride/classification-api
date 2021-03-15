#!/bin/sh 

set -e

# Call the benchmarking script without going through the experiment management
python src/inference_benchmark.py

# With basic TensorFlow
python inference_benchmark.py \
    --batch_size 32 \
    --comments plop \
    --cuda_visible_device 0 \
    --docker_images local \
    --engines tensorflow \
    --factor 1 \
    --models MobileNetV2NCHW \
    --output /tmp/results.md \
    --profiling_dir /tmp/profiling \
    --n_steps 10

# With tensorRT FP32
python inference_benchmark.py \
    --batch_size 32 \
    --comments plop \
    --cuda_visible_device 0 \
    --docker_images local \
    --engines tensorflow \
    --factor 1 \
    --models MobileNetV2NCHW \
    --output /tmp/results.md \
    --profiling_dir /tmp/profiling \
    --n_steps 10 \
    --tensorrt \
    --tensorrt_precision FP32

# With tensorRT FP16
python inference_benchmark.py \
    --batch_size 32 \
    --comments plop \
    --cuda_visible_device 0 \
    --docker_images local \
    --engines tensorflow \
    --factor 1 \
    --models MobileNetV2NCHW \
    --output /tmp/results.md \
    --profiling_dir /tmp/profiling \
    --n_steps 10 \
    --tensorrt \
    --tensorrt_precision FP16
