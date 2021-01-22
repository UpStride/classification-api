from tensorflow.keras import backend as K 
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Convolution2D, Conv2D, Concatenate, Input, Flatten, Dense, Add, Layer, AveragePooling2D
from tensorflow.keras.regularizers import l2
from upstride.src_test.bn_from_dcn import ComplexBatchNormalization as ComplexBN
from upstride.src_test.conv_from_dcn import ComplexConv2D
from upstride.src_test.utils_from_dcn import GetImag, GetReal


def learnConcatRealImagBlock(I, filter_size, featmaps, stage, block, convArgs, bnArgs, d):
	"""Learn initial imaginary component for input."""
	
	conv_name_base = 'res'+str(stage)+block+'_branch'
	bn_name_base   = 'bn' +str(stage)+block+'_branch'
	
	O = BatchNormalization(name=bn_name_base+'2a', **bnArgs)(I)
	O = Activation(d.act)(O)
	O = Convolution2D(featmaps[0], filter_size,
	                  name               = conv_name_base+'2a',
	                  padding            = 'same',
	                  kernel_initializer = 'he_normal',
	                  use_bias           = False,
	                  kernel_regularizer = l2(0.0001))(O)
	
	O = BatchNormalization(name=bn_name_base+'2b', **bnArgs)(O)
	O = Activation(d.act)(O)
	O = Convolution2D(featmaps[1], filter_size,
	                  name               = conv_name_base+'2b',
	                  padding            = 'same',
	                  kernel_initializer = 'he_normal',
	                  use_bias           = False,
	                  kernel_regularizer = l2(0.0001))(O)
	
	return O

def getResidualBlock(I, filter_size, featmaps, stage, block, shortcut, convArgs, bnArgs, d):
	"""Get residual block."""
	
	activation           = d.act
	drop_prob            = d.dropout
	nb_fmaps1, nb_fmaps2 = featmaps
	conv_name_base       = 'res'+str(stage)+block+'_branch'
	bn_name_base         = 'bn' +str(stage)+block+'_branch'
	if K.image_data_format() == 'channels_first' and K.ndim(I) != 3:
		channel_axis = 1
	else:
		channel_axis = -1
	
	
	if   d.model == "real":
		O = BatchNormalization(name=bn_name_base+'_2a', **bnArgs)(I)
	elif d.model == "complex":
		O = ComplexBN(name=bn_name_base+'_2a', **bnArgs)(I)
	O = Activation(activation)(O)
	
	if shortcut == 'regular' or d.spectral_pool_scheme == "nodownsample":
		if   d.model == "real":
			O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', **convArgs)(O)
		elif d.model == "complex":
			O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', **convArgs)(O)
	elif shortcut == 'projection':
		# if d.spectral_pool_scheme == "proj":
        #     continue
		if   d.model == "real":
			O = Conv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
		elif d.model == "complex":
			O = ComplexConv2D(nb_fmaps1, filter_size, name=conv_name_base+'2a', strides=(2, 2), **convArgs)(O)
	
	if   d.model == "real":
		O = BatchNormalization(name=bn_name_base+'_2b', **bnArgs)(O)
		O = Activation(activation)(O)
		O = Conv2D(nb_fmaps2, filter_size, name=conv_name_base+'2b', **convArgs)(O)
	elif d.model == "complex":
		O = ComplexBN(name=bn_name_base+'_2b', **bnArgs)(O)
		O = Activation(activation)(O)
		O = ComplexConv2D(nb_fmaps2, filter_size, name=conv_name_base+'2b', **convArgs)(O)
	
	if   shortcut == 'regular':
		O = Add()([O, I])
	elif shortcut == 'projection':
		# if d.spectral_pool_scheme == "proj":
        #     continue
		if   d.model == "real":
			X = Conv2D(nb_fmaps2, (1, 1),
			           name    = conv_name_base+'1',
			           strides = (2, 2) if d.spectral_pool_scheme != "nodownsample" else
			                     (1, 1),
			           **convArgs)(I)
			O      = Concatenate(channel_axis)([X, O])
		elif d.model == "complex":
			X = ComplexConv2D(nb_fmaps2, (1, 1),
			                  name    = conv_name_base+'1',
			                  strides = (2, 2) if d.spectral_pool_scheme != "nodownsample" else
			                            (1, 1),
			                  **convArgs)(I)
			O_real = Concatenate(channel_axis)([GetReal()(X), GetReal()(O)])
			O_imag = Concatenate(channel_axis)([GetImag()(X), GetImag()(O)])
			O      = Concatenate(      1     )([O_real,     O_imag])
	
	return O

def getResnetModel(d):
    n             = d.num_blocks
    sf            = d.start_filter
    dataset       = d.dataset
    activation    = d.act
    advanced_act  = d.aact
    drop_prob     = d.dropout
    inputShape    = (3, 32, 32) if K.image_data_format() == "channels_first" else (32, 32, 3)
    channelAxis   = 1 if K.image_data_format() == 'channels_first' else -1
    filsize       = (3, 3)
    convArgs      = {
        "padding":                  "same",
        "use_bias":                 False,
        "kernel_regularizer":       l2(0.0001),
    }
    bnArgs        = {
        "axis":                     channelAxis,
        "momentum":                 0.9,
        "epsilon":                  1e-04
    }

    if   d.model == "real":
        # The below if uncommented is causing the overall paramters for WS real to be ~7M which is not 
        # reported in the paper. The value mentioned in the paper is roughly 1.7M. 
        # sf *= 2 
        o.update({"kernel_initializer": Orthogonal(float(np.sqrt(2)))})
    if d.model == "complex":
        convArgs.update({"spectral_parametrization": d.spectral_param,
                            "kernel_initializer": d.comp_init})


    #
    # Input Layer
    #

    I = Input(shape=inputShape)

    #
    # Stage 1
    #

    O = learnConcatRealImagBlock(I, (1, 1), (3, 3), 0, '0', convArgs, bnArgs, d)
    O = Concatenate(channelAxis)([I, O])
    if d.model == "real":
        O = Conv2D(sf, filsize, name='conv1', **convArgs)(O)
        O = BatchNormalization(name="bn_conv1_2a", **bnArgs)(O)
    else:
        O = ComplexConv2D(sf, filsize, name='conv1', **convArgs)(O)
        O = ComplexBN(name="bn_conv1_2a", **bnArgs)(O)
    O = Activation(activation)(O)

    #
    # Stage 2
    #

    for i in range(n):
        O = getResidualBlock(O, filsize, [sf, sf], 2, str(i), 'regular', convArgs, bnArgs, d)
        if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
            O = applySpectralPooling(O, d)

    #
    # Stage 3
    #

    O = getResidualBlock(O, filsize, [sf, sf], 3, '0', 'projection', convArgs, bnArgs, d)
    if d.spectral_pool_scheme == "nodownsample":
        O = applySpectralPooling(O, d)

    for i in range(n-1):
        O = getResidualBlock(O, filsize, [sf*2, sf*2], 3, str(i+1), 'regular', convArgs, bnArgs, d)
        if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
            O = applySpectralPooling(O, d)

    #
    # Stage 4
    #

    O = getResidualBlock(O, filsize, [sf*2, sf*2], 4, '0', 'projection', convArgs, bnArgs, d)
    if d.spectral_pool_scheme == "nodownsample":
        O = applySpectralPooling(O, d)

    for i in range(n-1):
        O = getResidualBlock(O, filsize, [sf*4, sf*4], 4, str(i+1), 'regular', convArgs, bnArgs, d)
        if i == n//2 and d.spectral_pool_scheme == "stagemiddle":
            O = applySpectralPooling(O, d)

    #
    # Pooling
    #

    if d.spectral_pool_scheme == "nodownsample":
        # O = applySpectralPooling(O, d)
        O = AveragePooling2D(pool_size=(32, 32))(O)
    else:
        O = AveragePooling2D(pool_size=(8,  8))(O)

    #
    # Flatten
    #

    O = Flatten()(O)

    #
    # Dense
    #

    if   dataset == 'cifar10':
        O = Dense(10,  activation='softmax', kernel_regularizer=l2(0.0001))(O)
    elif dataset == 'cifar100':
        O = Dense(100, activation='softmax', kernel_regularizer=l2(0.0001))(O)
    # elif dataset == 'svhn':
    # 	O = Dense(10,  activation='softmax', kernel_regularizer=l2(0.0001))(O)
    else:
        raise ValueError("Unknown dataset "+d.dataset)

    # Return the model
    return Model(I, O)
