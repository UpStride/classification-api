build_tensorflow:
	docker build -t upstride/classification_api:tensorflow-2.0 -f dockerfiles/tensorflow.dockerfile .

build:
	docker build -t upstride/classification_api:upstride-2.0 -f dockerfiles/upstride.dockerfile .

run:
	@docker run -it --rm --gpus all --privileged \
		-v $$(pwd):/opt \
		-v ~/tensorflow_datasets/:/root/tensorflow_datasets \
		-v ~/.keras/datasets:/root/.keras/datasets \
		upstride/classification_api:upstride-2.0 \
		bash

run_tensorflow:
	@docker run -it --rm --gpus all --privileged \
		-v $$(pwd):/opt \
		-v ~/tensorflow_datasets/:/root/tensorflow_datasets \
		-v ~/.keras/datasets:/root/.keras/datasets \
		upstride/classification_api:tensorflow-2.0 \
		bash
		