import tensorflow as tf
import numpy as np
from .generic_model import GenericModelBuilder


weight_init = tf.keras.initializers.VarianceScaling()

is_channel_fist = False


class WideResNet(GenericModelBuilder):
  def __init__(self, *args, **kwargs):
    super(WideResNet, self).__init__(*args, **kwargs)

  def model(self, x):
    if is_channel_fist:
      x = tf.transpose(x, [0, 3, 1, 2])
      tf.keras.backend.set_image_data_format('channels_first')

    layers = self.layers
    weight_regularizer = self.weight_regularizer
    num_blocks_per_resnet = self.blocks_per_group
    filters = [int(16/self.factor), 
               int(16*self.channel_multiplier/self.factor), 
               int(32*self.channel_multiplier/self.factor), 
               int(64*self.channel_multiplier/self.factor)]
    strides = [1, 2, 2]  # stride for each resblock
    final_stride_val = np.prod(strides)

    ch = filters[0]
    x = layers.Conv2D(ch, 3, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, padding="same", name='conv')(x)

    first_x = x  # Res from the beginning

    for block_num in range(1, 4):
      orig_x = x  # Res from previous block
      activate_before_residual = True if block_num == 1 else False
      block_name = f'resblock_{block_num}'
      x = self.resblock(x, filters[block_num - 1], filters[block_num], stride=strides[block_num-1], 
               activate_before_residual=activate_before_residual, block_name=block_name+'_0')
      for i in range(1, num_blocks_per_resnet):
        x = self.resblock(x, filters[block_num], filters[block_num], stride=1, 
               activate_before_residual=False, block_name=block_name+f'_{i}')
      orig_x = self._conform_size(filters[block_num - 1], filters[block_num],
                            strides[block_num - 1], orig_x, block_name=block_name+f'_{i}')
      x = layers.Add()([x, orig_x])

    orig_x = self._conform_size(filters[0], filters[-1],final_stride_val, first_x, 'last_block')
    x = layers.Add()([x, orig_x])

    x = layers.BatchNormalization(name='batch_norm_last')(x)
    x = layers.Activation('relu', name='relu_last')(x)
    x = layers.GlobalAveragePooling2D()(x)
    return x


  def resblock(self, x, in_filter, out_filter, stride=1, use_bias=False, activate_before_residual=False, block_name='resblock'):
    layers = self.layers
    weight_regularizer = self.weight_regularizer
    if activate_before_residual:
      x = layers.BatchNormalization(name=block_name + '/batch_norm_0')(x)
      x = layers.Activation('relu', name=block_name + '/relu_0')(x)
      x_init = x
    else:
      x_init = x
      x = layers.BatchNormalization(name=block_name + '/batch_norm_0')(x)
      x = layers.Activation('relu', name=block_name + '/relu_0')(x)

    x = layers.Conv2D(out_filter, 3, stride, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                             use_bias=use_bias, padding='same', name=block_name + '/conv_0')(x)
    x = layers.BatchNormalization(name=block_name + '/batch_norm_1')(x)
    x = layers.Activation('relu', name=block_name + '/relu_1')(x)
    x = layers.Conv2D(out_filter, 3, 1, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                           use_bias=use_bias, padding='same', name=block_name + '/conv_1')(x)

    x_init = self._conform_size(in_filter, out_filter, stride, x_init, block_name)
    x = layers.Add()([x, x_init])
    return x

  def _conform_size(self, in_filter, out_filter, stride, x_init, block_name):
    layers = self.layers
    if in_filter != out_filter:
      x_init = layers.AveragePooling2D(pool_size=(stride, stride), name=block_name + '/avg_pool_0')(x_init)
      # hack to pad the channels
      if is_channel_fist:
        x_init = tf.transpose(x_init, [0, 2, 3, 1]) # put the channels at index 3
        x_init = layers.ZeroPadding2D(padding=(0,(out_filter-in_filter)//2), name=block_name + '/zero_pad_0')(x_init)
        x_init = tf.transpose(x_init, [0, 3, 1, 2]) # put the channels back at index 1
      else:
        x_init = tf.transpose(x_init, [0, 3, 1, 2]) # put the channels at index 1
        x_init = layers.ZeroPadding2D(padding=((out_filter-in_filter)//2,0), name=block_name + '/zero_pad_0')(x_init)
        x_init = tf.transpose(x_init, [0, 2, 3, 1]) # put the channels back at index 3
    return x_init

class WideResNet28_10(WideResNet):
  def __init__(self, *args, **kwargs):
    self.channel_multiplier = 10
    self.blocks_per_group = 4
    super().__init__(*args, **kwargs)

class WideResNet40_2(WideResNet):
  def __init__(self, *args, **kwargs):
    self.channel_multiplier = 2
    self.blocks_per_group = 6
    super().__init__(*args, **kwargs)



