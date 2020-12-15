build_tensorflow:
	docker build -t bharath/classification_api:tensorflow-1.0 -f dockerfiles/tensorflow.dockerfile .

build:
	docker build -t bharath/classification_api:upstride-1.0 -f dockerfiles/upstride.dockerfile .

run:
	@docker run -it --rm --runtime=nvidia --privileged \
		--name exp_mobilenetv3 \
		-v $$(pwd):/opt \
		-v ~/tensorflow_datasets/:/root/tensorflow_datasets \
		-v /home/ubuntu/results:/results \
		-v /home/ubuntu/checkpoints:/checkpoints \
		-v /home/ubuntu/logs:/logs \
		-v /home/ubuntu/datasets:/datasets \
		-e PYTHONPATH=/opt \
		-e CUDA_VISIBLE_DEVICES="0" \
		bharath/classification_api:upstride-1.0 \
		bash

run_tensorflow:
	@docker run -it --rm --gpus all --privileged \
		--name exp_mobilenetv3 \
		-v $$(pwd):/opt \
		-v /home/bharath/results:/results \
		-v /home/bharath/checkpoints:/checkpoints \
		-v /home/bharath/logs:/logs \
		-v /home/upstride/datasets:/datasets \
		bharath/classification_api:tensorflow-1.0 \
		bash
