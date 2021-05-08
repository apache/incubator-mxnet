# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"Image transforms."
import numpy as np

from ....block import Block, HybridBlock
from ..... import image
from .....base import numeric_types
from .....util import is_np_array

__all__ = ['ToTensor', 'Normalize', 'Rotate', 'RandomRotation',
           'RandomResizedCrop', 'CropResize', 'CropResize', 'RandomCrop',
           'CenterCrop', 'Resize', 'RandomFlipLeftRight', 'RandomFlipTopBottom',
           'RandomBrightness', 'RandomContrast', 'RandomSaturation', 'RandomHue',
           'RandomColorJitter', 'RandomLighting', 'RandomGray']

def _append_return(*args):
    """Append multiple args together.
    This allows many transform functions to bypass additional arguments.
    """
    if args:
        if len(args) == 1:
            return args[0]
        return tuple(args)
    return None


class ToTensor(HybridBlock):
    """Converts an image NDArray or batch of image NDArray to a tensor NDArray.

    Converts an image NDArray of shape (H x W x C) in the range
    [0, 255] to a float32 tensor NDArray of shape (C x H x W) in
    the range [0, 1].

    If batch input, converts a batch image NDArray of shape (N x H x W x C) in the
    range [0, 255] to a float32 tensor NDArray of shape (N x C x H x W).

    Inputs:
        - **data**: input tensor with (H x W x C) or (N x H x W x C) shape and uint8 type.

    Outputs:
        - **out**: output tensor with (C x H x W) or (N x C x H x W) shape and float32 type.

    Examples
    --------
    >>> transformer = vision.transforms.ToTensor()
    >>> image = mx.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    [[[ 0.85490197  0.72156864]
      [ 0.09019608  0.74117649]
      [ 0.61960787  0.92941177]
      [ 0.96470588  0.1882353 ]]
     [[ 0.6156863   0.73725492]
      [ 0.46666667  0.98039216]
      [ 0.44705883  0.45490196]
      [ 0.01960784  0.8509804 ]]
     [[ 0.39607844  0.03137255]
      [ 0.72156864  0.52941179]
      [ 0.16470589  0.7647059 ]
      [ 0.05490196  0.70588237]]]
    <NDArray 3x4x2 @cpu(0)>
    """
    def __init__(self):
        super(ToTensor, self).__init__()

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.to_tensor(x), *args)


class Normalize(HybridBlock):
    """Normalize an tensor of shape (C x H x W) or (N x C x H x W) with mean and
    standard deviation.

    Given mean `(m1, ..., mn)` and std `(s1, ..., sn)` for `n` channels,
    this transform normalizes each channel of the input tensor with::

        output[i] = (input[i] - mi) / si

    If mean or std is scalar, the same value will be applied to all channels.

    Parameters
    ----------
    mean : float or tuple of floats
        The mean values.
    std : float or tuple of floats
        The standard deviation values.


    Inputs:
        - **data**: input tensor with (C x H x W) or (N x C x H x W) shape.

    Outputs:
        - **out**: output tensor with the shape as `data`.

    Examples
    --------
    >>> transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    >>> image = mx.nd.random.uniform(0, 1, (3, 4, 2))
    >>> transformer(image)
    [[[ 0.18293785  0.19761486]
      [ 0.23839645  0.28142193]
      [ 0.20092112  0.28598186]
      [ 0.18162774  0.28241724]]
     [[-0.2881726  -0.18821815]
      [-0.17705294 -0.30780914]
      [-0.2812064  -0.3512327 ]
      [-0.05411351 -0.4716435 ]]
     [[-1.0363373  -1.7273437 ]
      [-1.6165586  -1.5223348 ]
      [-1.208275   -1.1878313 ]
      [-1.4711051  -1.5200229 ]]]
    <NDArray 3x4x2 @cpu(0)>
    """
    def __init__(self, mean=0.0, std=1.0):
        super(Normalize, self).__init__()
        self._mean = mean
        self._std = std

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.normalize(x, self._mean, self._std), *args)


class Rotate(Block):
    """Rotate the input image by a given angle. Keeps the original image shape.

    Parameters
    ----------
    rotation_degrees : float32
        Desired rotation angle in degrees.
    zoom_in : bool
        Zoom in image so that no padding is present in final output.
    zoom_out : bool
        Zoom out image so that the entire original image is present in final output.


    Inputs:
        - **data**: input tensor with (C x H x W) or (N x C x H x W) shape.

    Outputs:
        - **out**: output tensor with (C x H x W) or (N x C x H x W) shape.
    """
    def __init__(self, rotation_degrees, zoom_in=False, zoom_out=False):
        super(Rotate, self).__init__()
        self._args = (rotation_degrees, zoom_in, zoom_out)

    def forward(self, x, *args):
        if np.dtype(x.dtype) is not np.dtype(np.float32):
            raise TypeError("This transformation only supports float32. "
                            "Consider calling it after ToTensor, given: {}".format(x.dtype))
        return _append_return(image.imrotate(x, *self._args), *args)


class RandomRotation(Block):
    """Random rotate the input image by a random angle.
       Keeps the original image shape and aspect ratio.

    Parameters
    ----------
    angle_limits: tuple
        Tuple of 2 elements containing the upper and lower limit
        for rotation angles in degree.
    zoom_in : bool
        Zoom in image so that no padding is present in final output.
    zoom_out : bool
        Zoom out image so that the entire original image is present in final output.
    rotate_with_proba : float32


    Inputs:
        - **data**: input tensor with (C x H x W) or (N x C x H x W) shape.

    Outputs:
        - **out**: output tensor with (C x H x W) or (N x C x H x W) shape.
    """
    def __init__(self, angle_limits, zoom_in=False, zoom_out=False, rotate_with_proba=1.0):
        super(RandomRotation, self).__init__()
        lower, upper = angle_limits
        if lower >= upper:
            raise ValueError("`angle_limits` must be an ordered tuple")
        if rotate_with_proba < 0 or rotate_with_proba > 1:
            raise ValueError("Probability of rotating the image should be between 0 and 1")
        self._args = (angle_limits, zoom_in, zoom_out)
        self._rotate_with_proba = rotate_with_proba

    def forward(self, x, *args):
        if np.random.random() > self._rotate_with_proba:
            return x
        if np.dtype(x.dtype) is not np.dtype(np.float32):
            raise TypeError("This transformation only supports float32. "
                            "Consider calling it after ToTensor")
        return _append_return(image.random_rotate(x, *self._args), *args)


class RandomResizedCrop(HybridBlock):
    """Crop the input image with random scale and aspect ratio.

    Makes a crop of the original image with random size (default: 0.08
    to 1.0 of the original image size) and random aspect ratio (default:
    3/4 to 4/3), then resize it to the specified size.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of the final output.
    scale : tuple of two floats
        If scale is `(min_area, max_area)`, the cropped image's area will
        range from min_area to max_area of the original image's area
    ratio : tuple of two floats
        Range of aspect ratio of the cropped image before resizing.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) shape.
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0),
                 interpolation=1):
        super(RandomResizedCrop, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        if isinstance(scale, numeric_types):
            scale = (scale, 1.0)
        self._kwargs = {'width': size[0], 'height': size[1],
                        'area': scale, 'ratio': ratio,
                        'interp': interpolation, 'max_trial': 10}

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.random_resized_crop(x, **self._kwargs), *args)


class CropResize(HybridBlock):
    r"""Crop the input image with and optionally resize it.

    Makes a crop of the original image then optionally resize it to the specified size.

    Parameters
    ----------
    x : int
        Left boundary of the cropping area
    y : int
        Top boundary of the cropping area
    w : int
        Width of the cropping area
    h : int
        Height of the cropping area
    size : int or tuple of (w, h)
        Optional, resize to new size after cropping
    interpolation : int, optional
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=resize#resize
        Note that the Resize on gpu use contrib.bilinearResize2D operator
        which only support bilinear interpolation(1).


    Inputs:
        - **data**: input tensor with (H x W x C) or (N x H x W x C) shape.

    Outputs:
        - **out**: input tensor with (H x W x C) or (N x H x W x C) shape.

    Examples
    --------
    >>> transformer = vision.transforms.CropResize(x=0, y=0, width=100, height=100)
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 100x100x3 @cpu(0)>
    >>> image = mx.nd.random.uniform(0, 255, (3, 224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x100x100x3 @cpu(0)>
    >>> transformer = vision.transforms.CropResize(x=0, y=0, width=100, height=100, size=(50, 50), interpolation=1)
    >>> transformer(image)
    <NDArray 3x50x50 @cpu(0)>
    """
    def __init__(self, x, y, width, height, size=None, interpolation=None):
        super(CropResize, self).__init__()
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._size = size
        self._interpolation = interpolation

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            _image = F.npx.image
        else:
            _image = F.image
        out = _image.crop(x, self._x, self._y, self._width, self._height)
        if self._size:
            out = _image.resize(out, self._size, False, self._interpolation)
        return _append_return(out, *args)

class RandomCrop(HybridBlock):
    """Randomly crop `src` with `size` (width, height).
    Padding is optional.
    Upsample result if `src` is smaller than `size`
    .
    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of the final output.
    pad: int or tuple
        if int, size of the zero-padding
        if tuple, number of values padded to the edges of each axis.
            ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
            ((before, after),) yields same before and after pad for each axis.
            (pad,) or int is a shortcut for before = after = pad width for all axes.
    pad_value : int
        The value to use for padded pixels
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.
    Outputs:
        - **out**: output tensor with ((H+2*pad) x (W+2*pad) x C) shape.
    """

    def __init__(self, size, pad=None, pad_value=0, interpolation=1):
        super(RandomCrop, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = ((0, 1), (0, 1), size[0], size[1], interpolation)
        self._pad_value = pad_value
        if isinstance(pad, int):
            self.nd_pad = (0, 0, 0, 0, pad, pad, pad, pad, 0, 0)  # workaround as 5D
            self.np_pad = ((pad, pad), (pad, pad), (0, 0))
        elif pad is not None:
            assert len(pad) >= 4
            self.nd_pad = tuple([0] * 4 + list(pad) + [0] * (6 - len(pad)))
            self.np_pad = ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0))
        else:
            self.nd_pad = pad
            self.np_pad = pad

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            if self.np_pad:
                x = F.np.pad(x, pad_width=self.np_pad, mode='constant', constant_values=self._pad_value)
            return _append_return(F.npx.image.random_crop(x, *self._args), *args)
        else:
            if self.nd_pad:
                x = F.cast(F.expand_dims(F.expand_dims(x, 0), 0), 'float32')
                x_pad = F.pad(x, pad_width=self.nd_pad, mode='constant', constant_value=self._pad_value)
                x = F.cast(x_pad.squeeze(0).squeeze(0), 'uint8')
            return _append_return(F.image.random_crop(x, *self._args), *args)


class CenterCrop(HybridBlock):
    """Crops the image `src` to the given `size` by trimming on all four
    sides and preserving the center of the image. Upsamples if `src` is
    smaller than `size`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) shape.

    Examples
    --------
    >>> transformer = vision.transforms.CenterCrop(size=(1000, 500))
    >>> image = mx.nd.random.uniform(0, 255, (2321, 3482, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 500x1000x3 @cpu(0)>
    """
    def __init__(self, size, interpolation=1):
        super(CenterCrop, self).__init__()
        if isinstance(size, numeric_types):
            size = (size, size)
        self._args = (size[0], size[1], interpolation)

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.random_crop(x, (0.5, 0.5), (0.5, 0.5), *self._args), *args)


class Resize(HybridBlock):
    """Resize an image or a batch of image NDArray to the given size.
    Should be applied before `mxnet.gluon.data.vision.transforms.ToTensor`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    keep_ratio : bool
        Whether to resize the short edge or both edges to `size`,
        if size is give as an integer.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
        Note that the Resize on gpu use contrib.bilinearResize2D operator
        which only support bilinear interpolation(1).


    Inputs:
        - **data**: input tensor with (H x W x C) or (N x H x W x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) or (N x H x W x C) shape.

    Examples
    --------
    >>> transformer = vision.transforms.Resize(size=(1000, 500))
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 500x1000x3 @cpu(0)>
    >>> image = mx.nd.random.uniform(0, 255, (3, 224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 3x500x1000x3 @cpu(0)>
    """
    def __init__(self, size, keep_ratio=False, interpolation=1):
        super(Resize, self).__init__()
        self._keep = keep_ratio
        self._size = size
        self._interpolation = interpolation

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.resize(x, self._size, self._keep, self._interpolation), *args)

class RandomFlipLeftRight(HybridBlock):
    """Randomly flip the input image left to right with a probability
    of p(0.5 by default).

    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, p=0.5):
        super(RandomFlipLeftRight, self).__init__()
        self.p = p

    def hybrid_forward(self, F, x, *args):
        if self.p <= 0:
            return _append_return(x, *args)

        if is_np_array():
            if self.p >= 1:
                return _append_return(F.npx.image.flip_left_right(x), *args)
            return _append_return(F.npx.image.random_flip_left_right(x, p=self.p), *args)
        else:
            if self.p >= 1:
                return _append_return(F.image.flip_left_right(x), *args)
            return _append_return(F.image.random_flip_left_right(x, p=self.p), *args)


class RandomFlipTopBottom(HybridBlock):
    """Randomly flip the input image top to bottom with a probability
    of p(0.5 by default).

    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, p=0.5):
        super(RandomFlipTopBottom, self).__init__()
        self.p = p

    def hybrid_forward(self, F, x, *args):
        if self.p <= 0:
            return _append_return(x, *args)

        if is_np_array():
            if self.p >= 1:
                return _append_return(F.npx.image.flip_top_bottom(x), *args)
            return _append_return(F.npx.image.random_flip_top_bottom(x, p=self.p), *args)
        else:
            if self.p >= 1:
                return _append_return(F.image.flip_top_bottom(x), *args)
            return _append_return(F.image.random_flip_top_bottom(x, p=self.p), *args)


class RandomBrightness(HybridBlock):
    """Randomly jitters image brightness with a factor
    chosen from `[max(0, 1 - brightness), 1 + brightness]`.

    Parameters
    ----------
    brightness: float
        How much to jitter brightness. brightness factor is randomly
        chosen from `[max(0, 1 - brightness), 1 + brightness]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, brightness):
        super(RandomBrightness, self).__init__()
        self._args = (max(0, 1-brightness), 1+brightness)

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.random_brightness(x, *self._args), *args)


class RandomContrast(HybridBlock):
    """Randomly jitters image contrast with a factor
    chosen from `[max(0, 1 - contrast), 1 + contrast]`.

    Parameters
    ----------
    contrast: float
        How much to jitter contrast. contrast factor is randomly
        chosen from `[max(0, 1 - contrast), 1 + contrast]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, contrast):
        super(RandomContrast, self).__init__()
        self._args = (max(0, 1-contrast), 1+contrast)

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.random_contrast(x, *self._args), *args)


class RandomSaturation(HybridBlock):
    """Randomly jitters image saturation with a factor
    chosen from `[max(0, 1 - saturation), 1 + saturation]`.

    Parameters
    ----------
    saturation: float
        How much to jitter saturation. saturation factor is randomly
        chosen from `[max(0, 1 - saturation), 1 + saturation]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, saturation):
        super(RandomSaturation, self).__init__()
        self._args = (max(0, 1-saturation), 1+saturation)

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.random_saturation(x, *self._args), *args)


class RandomHue(HybridBlock):
    """Randomly jitters image hue with a factor
    chosen from `[max(0, 1 - hue), 1 + hue]`.

    Parameters
    ----------
    hue: float
        How much to jitter hue. hue factor is randomly
        chosen from `[max(0, 1 - hue), 1 + hue]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, hue):
        super(RandomHue, self).__init__()
        self._args = (max(0, 1-hue), 1+hue)

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.random_hue(x, *self._args), *args)


class RandomColorJitter(HybridBlock):
    """Randomly jitters the brightness, contrast, saturation, and hue
    of an image.

    Parameters
    ----------
    brightness : float
        How much to jitter brightness. brightness factor is randomly
        chosen from `[max(0, 1 - brightness), 1 + brightness]`.
    contrast : float
        How much to jitter contrast. contrast factor is randomly
        chosen from `[max(0, 1 - contrast), 1 + contrast]`.
    saturation : float
        How much to jitter saturation. saturation factor is randomly
        chosen from `[max(0, 1 - saturation), 1 + saturation]`.
    hue : float
        How much to jitter hue. hue factor is randomly
        chosen from `[max(0, 1 - hue), 1 + hue]`.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(RandomColorJitter, self).__init__()
        self._args = (brightness, contrast, saturation, hue)

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.random_color_jitter(x, *self._args), *args)


class RandomLighting(HybridBlock):
    """Add AlexNet-style PCA-based noise to an image.

    Parameters
    ----------
    alpha : float
        Intensity of the image.


    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """
    def __init__(self, alpha):
        super(RandomLighting, self).__init__()
        self._alpha = alpha

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            F = F.npx
        return _append_return(F.image.random_lighting(x, self._alpha), *args)


class RandomGray(HybridBlock):
    """Randomly convert to gray image.

    Parameters
    ----------
    p : float
        Probability to convert to grayscale
    """
    def __init__(self, p=0.5):
        super(RandomGray, self).__init__()
        self.p = p

    def hybrid_forward(self, F, x, *args):
        if is_np_array():
            mat = F.np.concatenate((F.np.full((3, 1), 0.2989),
                                    F.np.full((3, 1), 0.5870),
                                    F.np.full((3, 1), 0.114)), axis=1)
            x = x.astype(dtype='float32')
            gray = F.np.where(self.p < F.np.random.uniform(), x, F.np.dot(x, mat))
        else:
            mat = F.concat(F.full((3, 1), 0.2989),
                           F.full((3, 1), 0.5870),
                           F.full((3, 1), 0.114), dim=1)
            cond = self.p < F.random.uniform(shape=1)
            x = F.cast(x, dtype='float32')
            gray = F.contrib.cond(cond, lambda: x, lambda: F.dot(x, mat))
        return _append_return(gray, *args)
