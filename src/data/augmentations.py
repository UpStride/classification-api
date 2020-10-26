import math
from typing import Dict, List
import tensorflow as tf

TFA_AVAILABLE = True
try:
  import tensorflow_addons as tfa
except Exception as e:
  print(f'can\'t import tensorflow addons: {e}')
  TFA_AVAILABLE = False


INTERPOLATION_METHODS = [
    'area',
    'bicubic',
    'bilinear',
    'gaussian',
    'lanczos3',
    'lanczos5',
    'lanczos5',
    'mitchellcubic',
    'nearest',
]

AUGMENTATION_LIST = [
    'CentralCrop',
    'ColorJitter',
    'Normalize',
    'RandomHorizontalFlip',
    'RandomRotate',
    'RandomRotate90',
    'RandomVerticalFlip',
    'RandomCrop',
    'RandomCropThenResize',
    'Resize',
    'ResizeThenRandomCrop',
    'Translate',
    'Cutout'
]


arguments = [
    ['namespace', 'Normalize', [
        ['list[float]', "mean", [0.485, 0.456, 0.406], 'mean of training data'],
        ['list[float]', "std", [0.229, 0.224, 0.225], 'std of training data'],
        [bool, "scale_in_zero_to_one", True, 'only scale the image in the  range (0~1)'],
        [bool, "only_subtract_mean", False, 'if True, subtract only mean from input without dividing by std']
    ]],
    ['namespace', 'ColorJitter', [
        [float, "brightness", 0.05, 'brightness factor to jitter'],
        ['list[float]', "contrast", [0.7, 1.3], 'contrast range to jitter'],
        ['list[float]', "saturation", [0.6, 1.6], 'saturation range to jitter'],
        [float, "hue", 0.08, 'hue factor to jitter'],
        ['list[float]', "clip", [0., 0.], 'clipping range, if both (min, max) are 0, no clipping will be performed']
    ]],
    ['namespace', 'RandomRotate', [
        [int, "angle", 10, 'angle will be selected randomly within range [-angle, angle]'],
        [str, "interpolation", 'nearest', 'interpolation method']
    ]],
    ['namespace', 'CentralCrop', [
        ['list[int]', "size", [224, 224], 'size of the central crop'],
        [float, "crop_proportion", 0.875, 'proportion of image to retain along the less-cropped side'],
        [str, "interpolation", 'bicubic', 'interpolation method']
    ]],
    ['namespace', 'RandomCrop', [
        ['list[int]', "size", [224, 224, 3], 'Random crop shape']
    ]],
    ['namespace', 'Resize', [
        ['list[int]', "size", [224, 224], 'shape for resizing the image'],
        [str, "interpolation", 'bicubic', 'interpolation method']
    ]],
    ['namespace', 'ResizeThenRandomCrop', [
        ['list[int]', "size", [256, 256], 'shape for resizing the image'],
        ['list[int]', "crop_size", [224, 224, 3], 'Random crop shape'],
        [str, "interpolation", 'bicubic', 'interpolation method']
    ]],
    ['namespace', 'RandomCropThenResize', [
        ['list[int]', "size", [224, 224], 'shape for resizing the image'],
        ['list[float]', "scale", [0.8, 1.0], 'range of size of the origin size cropped'],
        ['list[float]', "ratio", [0.75, 1.33], 'range of aspect ratio of the origin aspect ratio cropped'],
        [str, "interpolation", 'bicubic', 'interpolation method']
    ]],
    ['namespace', 'Translate', [
        [float, "width_shift_range", 0.1, 'randomly shift images horizontally (fraction of total width)'],
        [float, "height_shift_range", 0.1, 'randomly shift images vertically (fraction of total height)']
    ]],
    ['namespace', 'Cutout', [
        [int, "length", 16, 'cutout length'],
    ]]
]


def get_tfa_available():
  """function that return the status of tfa
  """
  global TFA_AVAILABLE
  return TFA_AVAILABLE


def apply_transformation_randomly(transform, p, x):
  """ Randomly apply transformation to x with probability p 
  """
  return tf.cond(
      tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(p, tf.float32)),
      lambda: transform(x),
      lambda: x)


def apply_list_of_transformations(image: tf.Tensor, transformation_list: List[str], config: Dict):
  for t in transformation_list:
    assert t in AUGMENTATION_LIST
  image = tf.cast(image, tf.float32)
  for transformation in transformation_list:
    if transformation in config:
      t_config = config[transformation]
    else:
      t_config = None
    image = eval(transformation)(t_config)(image)
  return image


class Cutout:
  def __init__(self, config):
    self.length = config['length']

  def __call__(self, x):
    height = tf.shape(x)[0]
    width = tf.shape(x)[1]
    channel = tf.shape(x)[2]

    x1 = tf.random.uniform([], 0, height - self.length, tf.int32)
    y1 = tf.random.uniform([], 0, width - self.length, tf.int32)

    erase_area = tf.zeros([self.length, self.length, channel], dtype=x.dtype)
    padding_dim = [[x1, height - (x1 + self.length)], [y1, width - (y1 + self.length)], [0, 0]]

    mask = tf.pad(erase_area, padding_dim, constant_values=1)
    # TODO change with tf.where and bool, maybe can be faster ?
    return mask * x


class Normalize:
  """ Normalize the image with mean and standard deviation, default values are taken from imagenet dataset

      Args:
          x (Tensor): Image to normalize in the range (0~255)
          mean (List): List of means for each channel.
          std (List): List of standard deviations for each channel.
          scale_in_zero_to_one (bool): only scale the image in the  range (0~1)
          only_subtract_mean (bool): if True, subtract only mean from input without dividing by std

      Returns:
          normalized image
  """

  def __init__(self, config):
    self.mean = config['mean']
    self.std = config['std']
    self.only_subtract_mean = config['only_subtract_mean']
    self.scale_in_zero_to_one = config['scale_in_zero_to_one']

  def __call__(self, x):
    x /= 255.
    if not self.scale_in_zero_to_one:
      x -= self.mean
      if not self.only_subtract_mean:
        x /= self.std
    return x


class ColorJitter:
  """Randomly change the hue, saturation, brightness and contrast of an image.

  Args:
      x: Image
      brightness (float): How much to jitter brightness. brightness_factor is chosen uniformly from [-brightness, brightness]
      contrast (list or tuple) : How much to jitter contrast. contrast_factor is randomly chosen in the interval [contrast[0], contrast[1]]
      saturation (list or tuple) : How much to jitter saturation. saturation_factor is randomly chosen in the interval [saturation[0], saturation[1]]
      hue: How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue]
      clip: Clipping range
  Returns:
      Augmented image
  """

  def __init__(self, config):
    self.brightness = config['brightness']
    self.contrast = config['contrast']
    self.saturation = config['saturation']
    self.hue = config['hue']
    self.clip = config['clip']

  def color_jitter_rand(self, x):
    def change_brightness():
      return x if self.brightness == 0 else tf.image.random_brightness(x, self.brightness)

    def change_contrast():
      return x if self.contrast[0] + self.contrast[1] == 0 else tf.image.random_contrast(x, self.contrast[0], self.contrast[1])

    def change_saturation():
      return x if self.saturation[0] + self.saturation[1] == 0 else tf.image.random_saturation(x, self.saturation[0], self.saturation[1])

    def change_hue():
      return x if self.hue == 0 else tf.image.random_hue(x, self.hue)

    perm = tf.random.shuffle(tf.range(4))
    for i in range(4):
      index = tf.gather(perm, i)
      x = tf.switch_case(index, branch_fns={
          0: change_brightness,
          1: change_contrast,
          2: change_saturation,
          3: change_hue
      })
      if self.clip[0] + self.clip[1] != 0:
        x = tf.clip_by_value(x, self.clip[0], self.clip[1])
    return x

  def grayscale(self, x, keep_channels=True):
    x = tf.image.rgb_to_grayscale(x)
    if keep_channels:
      x = tf.tile(x, [1, 1, 3])
    return x

  def __call__(self, x):
    x = apply_transformation_randomly(self.color_jitter_rand, p=0.8, x=x)
    x = apply_transformation_randomly(self.grayscale, p=0.2, x=x)
    return x


class RandomHorizontalFlip:
  """ horizontally flip image

      Args:
          x: Image to flip

      Returns:
          horizontally flipped image
  """

  def __init__(self, config):
    pass

  def __call__(self, x):
    return tf.image.random_flip_left_right(x)


class RandomVerticalFlip:
  """ vertically flip image

      Args:
          x: Image to flip

      Returns:
          vertically flipped image
  """

  def __init__(self, config):
    pass

  def __call__(self, x):
    return tf.image.random_flip_up_down(x)


class RandomRotate90:
  """ Randomly Rotate image counter-clockwise by 90/180/270/360 degrees.

      Args:
          x: Image

      Returns:
          rotated image
  """

  def __init__(self, config):
    pass

  def __call__(self, x):
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(x, k=k)


class RandomRotate:
  """ Randomly Rotate image counter-clockwise by the passed angle in degrees.

      Args:
          x: A tensor of shape (num_images, num_rows, num_columns, num_channels) (NHWC),
                  (num_rows, num_columns, num_channels) (HWC, or (num_rows, num_columns) (HW).
          angle: A scalar angle range value, angle will be selected randomly within range [-angle, angle]
          interpolation: supported interpolation mode ("nearest", "bilinear")

      Returns:
          rotated image
  """

  def __init__(self, config):
    self.angle = config['angle']
    self.interpolation = config['interpolation']
    if self.interpolation not in ['nearest', 'bilinear']:
      raise ValueError(f"supported interpolation methods are ('nearest', 'bilinear'), provided {self.interpolation}")
    self.interpolation = self.interpolation.upper()

  def __call__(self, x):
    global TFA_AVAILABLE
    if TFA_AVAILABLE:
      try:
        k = tf.random.uniform(shape=[], minval=-self.angle, maxval=self.angle, dtype=tf.float32)
        # Apply random rotation 50% of the time
        prob = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        x = tf.cond(tf.math.less(prob, 0.5), lambda: x, lambda: tfa.image.rotate(x, k * tf.constant(math.pi) / 180, interpolation=self.interpolation))
      except Exception as e:
        print(f"error in random rotate : {e}")
        TFA_AVAILABLE = False
    return x


class CentralCrop:
  """ Crop the central region of the image(s)

      Args:
          x: Either a 3-D float Tensor of shape [height, width, depth], or a 4-D Tensor of shape [batch_size, height, width, depth]
          crop_proportion: Proportion of image to retain along the less-cropped side. (0.875 Standard for ImageNet)
          size: size of the central crop
          interpolation: interpolation method
      Returns:
          centrally cropped image
  """

  def __init__(self, config):
    self.crop_proportion = config['crop_proportion']
    self.size = config['size']
    self.interpolation = config['interpolation']
    if self.interpolation not in INTERPOLATION_METHODS:
      raise ValueError(f"supported interpolation methods are ({', '.join(INTERPOLATION_METHODS)}), provided {self.interpolation}")

  def _compute_crop_shape(self, image_height: tf.Tensor, image_width: tf.Tensor, aspect_ratio: float, crop_proportion: float):
    """Compute aspect ratio-preserving shape for central crop.
    The resulting shape retains `crop_proportion` along one side and a proportion
    less than or equal to `crop_proportion` along the other side.
    source: https://github.com/google-research/simclr/blob/01ddaf0bd692ee945dad7ff5fb07b26df1b9edbe/data_util.py#L167
    Args:
      image_height: Height of image to be cropped.
      image_width: Width of image to be cropped.
      aspect_ratio: Desired aspect ratio (width / height) of output.
      crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
      crop_height: Height of image after cropping.
      crop_width: Width of image after cropping.
    """
    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)

    def _requested_aspect_ratio_wider_than_image():
      crop_height = tf.cast(tf.math.rint(crop_proportion / aspect_ratio * image_width_float), tf.int32)
      crop_width = tf.cast(tf.math.rint(crop_proportion * image_width_float), tf.int32)
      return crop_height, crop_width

    def _image_wider_than_requested_aspect_ratio():
      crop_height = tf.cast(tf.math.rint(crop_proportion * image_height_float), tf.int32)
      crop_width = tf.cast(tf.math.rint(crop_proportion * aspect_ratio * image_height_float), tf.int32)
      return crop_height, crop_width

    return tf.cond(
        aspect_ratio > image_width_float / image_height_float,
        _requested_aspect_ratio_wider_than_image,
        _image_wider_than_requested_aspect_ratio)

  def __call__(self, x):
    shape = tf.shape(x)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = self._compute_crop_shape(image_height, image_width, self.size[0]/self.size[1], self.crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    x = tf.image.crop_to_bounding_box(x, offset_height, offset_width, crop_height, crop_width)
    x = tf.image.resize(x, self.size, method=self.interpolation)
    return x


class RandomCrop:
  """ Randomly crops a tensor to a given size

      Args:
          x: input image
          size: shape of the crop
      Returns:
          randomly cropped image
  """

  def __init__(self, config):
    self.size = config['size']

  def __call__(self, x):
    return tf.image.random_crop(x, self.size)


class Resize:
  """ Resize a tensor to a given size

      Args:
          x: input image
          size: 1D Tensor/List of new size [height, width]
          interpolation: interpolation method
      Returns:
          Resized image
  """

  def __init__(self, config):
    self.size = config['size']
    self.interpolation = config['interpolation']
    if self.interpolation not in INTERPOLATION_METHODS:
      raise ValueError(f"supported interpolation methods are ({', '.join(INTERPOLATION_METHODS)}), provided {self.interpolation}")

  def __call__(self, x):
    return tf.image.resize(x, self.size, method=self.interpolation)


class ResizeThenRandomCrop:
  """ Resize a Tensor to a defined shape and then randomly crop it

      Args:
          x: input image
          size: 1D Tensor/List of new size [height, width]
          crop_size: 1D Tensor/List of crop size: e.g for rgb image [crop_height, crop_width, 3]
          interpolation: interpolation method
      Returns:
          randomly cropped image
  """

  def __init__(self, config):
    self.size = config['size']
    self.crop_size = config['crop_size']
    self.interpolation = config['interpolation']
    if self.interpolation not in INTERPOLATION_METHODS:
      raise ValueError(f"supported interpolation methods are ({', '.join(INTERPOLATION_METHODS)}), provided {self.interpolation}")

  def __call__(self, x):
    x = tf.image.resize(x, self.size, method=self.interpolation)
    x = tf.image.random_crop(x, self.crop_size)
    return x


class RandomCropThenResize:
  """Crop the given Image tensor to random size and aspect ratio, then resize

  A crop of random size (default: of 0.08 to 1.0) of the original size and a random
  aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
  is finally resized to given size.
  This is popularly used to train the Inception networks.

  Args:
      size: 1D Tensor/List of new size [height, width]
      scale: range of size of the origin size cropped
      ratio: range of aspect ratio of the origin aspect ratio cropped
      interpolation: interpolation method
  """

  def __init__(self, config):
    self.scale = config['scale']
    self.ratio = config['ratio']
    self.interpolation = config['interpolation']
    self.size = config['size']
    if self.interpolation not in INTERPOLATION_METHODS:
      raise ValueError(f"supported interpolation methods are ({', '.join(INTERPOLATION_METHODS)}), provided {self.interpolation}")
    if not isinstance(self.size, (tuple, list)):
      self.size = (self.size, self.size)
    if (self.scale[0] > self.scale[1]) or (self.ratio[0] > self.ratio[1]):
      raise Exception('range should be of kind (min, max)')

  def _get_params(self, x, bbox, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 1.0), max_attempts=100):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        x: Image to be cropped.
        bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image (used for classification).
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
        aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.

    Returns:
        tuple: params (offset_y, offset_x, height, width) to be passed for a random sized crop.
    """
    shape = tf.shape(x)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    height, width, _ = tf.unstack(bbox_size)
    return offset_y, offset_x, height, width

  def __call__(self, x):
    """
    Args:
        x: Image to be cropped and resized

    Returns:
        Randomly cropped and resized image.
    """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    offset_y, offset_x, height, width = self._get_params(
        x,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=self.ratio,
        area_range=self.scale,
        max_attempts=15)

    x = tf.image.crop_to_bounding_box(x, offset_y, offset_x, height, width)
    x = tf.image.resize(x, self.size, method=self.interpolation)
    return x


class Translate:
  """ Translates image in X or Y dimension.

      Args:
          x: A tensor of shape (num_images, num_rows, num_columns, num_channels) (NHWC),
                (num_rows, num_columns, num_channels) (HWC), or
                (num_rows, num_columns) (HW)
          width_shift_range: randomly shift images horizontally (fraction of total width)
          height_shift_range: randomly shift images vertically (fraction of total height)
      Returns:
          translated image
  """

  def __init__(self, config):
    self.width_shift_range = config['width_shift_range']
    self.height_shift_range = config['height_shift_range']
    if not 0. <= self.width_shift_range <= 1.:
      raise ValueError(f"width_shift_range should be in range 0~1, but given {self.width_shift_range}")
    if not 0. <= self.height_shift_range <= 1.:
      raise ValueError(f"height_shift_range should be in range 0~1, but given {self.height_shift_range}")

  def __call__(self, x):
    tensor_shape = tf.cast(tf.shape(x), tf.float32)
    height_shift_range = tf.cast(self.height_shift_range * tensor_shape[0], tf.int32)
    width_shift_range = tf.cast(self.width_shift_range * tensor_shape[1], tf.int32)

    dx = tf.random.uniform(shape=[], minval=-width_shift_range, maxval=width_shift_range, dtype=tf.int32)
    dy = tf.random.uniform(shape=[], minval=-height_shift_range, maxval=height_shift_range, dtype=tf.int32)

    prob_dx = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    prob_dy = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    dx = tf.cond(tf.math.less(prob_dx, 0.5), lambda: 0, lambda: dx)
    dy = tf.cond(tf.math.less(prob_dy, 0.5), lambda: 0, lambda: dy)

    # translate x
    pad_left = tf.math.maximum(dx, 0)
    pad_right = tf.math.maximum(-dx, 0)
    x = tf.cond(tf.math.equal(pad_right, 0), lambda: x, lambda: x[:, pad_right:, :])
    x = tf.cond(tf.math.equal(pad_left, 0), lambda: x, lambda: x[:, :-pad_left, :])

    # translate y
    pad_up = tf.math.maximum(dy, 0)
    pad_down = tf.math.maximum(-dy, 0)
    x = tf.cond(tf.math.equal(pad_down, 0), lambda: x, lambda: x[pad_down:, :, :])
    x = tf.cond(tf.math.equal(pad_up, 0), lambda: x, lambda: x[:-pad_up, :, :])

    x = tf.pad(x, [[pad_up, pad_down], [pad_left, pad_right], [0, 0]])
    return x
