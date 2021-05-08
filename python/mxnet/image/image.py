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

# pylint: disable=no-member, too-many-lines, redefined-builtin, protected-access, unused-import, invalid-name
# pylint: disable=too-many-arguments, too-many-locals, no-name-in-module, too-many-branches, too-many-statements
"""Read individual image files and perform augmentations."""


import sys
import os
import random
import logging
import json
import warnings

from numbers import Number

import numpy as np

from .. import numpy as _mx_np  # pylint: disable=reimported


try:
    import cv2
except ImportError:
    cv2 = None

from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi


def imread(filename, *args, **kwargs):
    """Read and decode an image to an NDArray.

    .. note:: `imread` uses OpenCV (not the CV2 Python library).
       MXNet must have been built with USE_OPENCV=1 for `imdecode` to work.

    Parameters
    ----------
    filename : str
        Name of the image file to be loaded.
    flag : {0, 1}, default 1
        1 for three channel color output. 0 for grayscale output.
    to_rgb : bool, default True
        True for RGB formatted output (MXNet default).
        False for BGR formatted output (OpenCV default).
    out : NDArray, optional
        Output buffer. Use `None` for automatic allocation.

    Returns
    -------
    NDArray
        An `NDArray` containing the image.

    Example
    -------
    >>> mx.img.imread("flower.jpg")
    <NDArray 224x224x3 @cpu(0)>

    Set `flag` parameter to 0 to get grayscale output

    >>> mx.img.imread("flower.jpg", flag=0)
    <NDArray 224x224x1 @cpu(0)>

    Set `to_rgb` parameter to 0 to get output in OpenCV format (BGR)

    >>> mx.img.imread("flower.jpg", to_rgb=0)
    <NDArray 224x224x3 @cpu(0)>
    """
    if is_np_array():
        read_fn = _npi.cvimread
    else:
        read_fn = _internal._cvimread
    return read_fn(filename, *args, **kwargs)


def imresize(src, w, h, *args, **kwargs):
    r"""Resize image with OpenCV.

    .. note:: `imresize` uses OpenCV (not the CV2 Python library). MXNet must have been built
       with USE_OPENCV=1 for `imresize` to work.

    Parameters
    ----------
    src : NDArray
        source image
    w : int, required
        Width of resized image.
    h : int, required
        Height of resized image.
    interp : int, optional, default=1
        Interpolation method (default=cv2.INTER_LINEAR).
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> new_image = mx.img.resize(image, 240, 360)
    >>> new_image
    <NDArray 240x360x3 @cpu(0)>
    """
    resize_fn = _npi.cvimresize if is_np_array() else _internal._cvimresize
    return resize_fn(src, w, h, *args, **kwargs)


def imdecode(buf, *args, **kwargs):
    """Decode an image to an NDArray.

    .. note:: `imdecode` uses OpenCV (not the CV2 Python library).
       MXNet must have been built with USE_OPENCV=1 for `imdecode` to work.

    Parameters
    ----------
    buf : str/bytes/bytearray or numpy.ndarray
        Binary image data as string or numpy ndarray.
    flag : int, optional, default=1
        1 for three channel color output. 0 for grayscale output.
    to_rgb : int, optional, default=1
        1 for RGB formatted output (MXNet default). 0 for BGR formatted output (OpenCV default).
    out : NDArray, optional
        Output buffer. Use `None` for automatic allocation.

    Returns
    -------
    NDArray
        An `NDArray` containing the image.

    Example
    -------
    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 224x224x3 @cpu(0)>

    Set `flag` parameter to 0 to get grayscale output

    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image, flag=0)
    >>> image
    <NDArray 224x224x1 @cpu(0)>

    Set `to_rgb` parameter to 0 to get output in OpenCV format (BGR)

    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image, to_rgb=0)
    >>> image
    <NDArray 224x224x3 @cpu(0)>
    """
    if not isinstance(buf, nd.NDArray):
        if not isinstance(buf, (bytes, bytearray, np.ndarray)):
            raise ValueError('buf must be of type bytes, bytearray or numpy.ndarray,'
                             'if you would like to input type str, please convert to bytes')
        array_fn = _mx_np.array if is_np_array() else nd.array
        buf = array_fn(np.frombuffer(buf, dtype=np.uint8), dtype=np.uint8)

    cvimdecode = _npi.cvimdecode if is_np_array() else _internal._cvimdecode
    return cvimdecode(buf, *args, **kwargs)


def scale_down(src_size, size):
    """Scales down crop size if it's larger than image size.

    If width/height of the crop is larger than the width/height of the image,
    sets the width/height to the width/height of the image.

    Parameters
    ----------
    src_size : tuple of int
        Size of the image in (width, height) format.
    size : tuple of int
        Size of the crop in (width, height) format.

    Returns
    -------
    tuple of int
        A tuple containing the scaled crop size in (width, height) format.

    Example
    --------
    >>> src_size = (640,480)
    >>> size = (720,120)
    >>> new_size = mx.img.scale_down(src_size, size)
    >>> new_size
    (640,106)
    """
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w * sh) / h, sh
    if sw < w:
        w, h = sw, float(h * sw) / w
    return int(w), int(h)


def copyMakeBorder(src, top, bot, left, right, *args, **kwargs):
    """Pad image border with OpenCV.

    Parameters
    ----------
    src : NDArray
        source image
    top : int, required
        Top margin.
    bot : int, required
        Bottom margin.
    left : int, required
        Left margin.
    right : int, required
        Right margin.
    type : int, optional, default='0'
        Filling type (default=cv2.BORDER_CONSTANT).
        0 - cv2.BORDER_CONSTANT - Adds a constant colored border.
        1 - cv2.BORDER_REFLECT - Border will be mirror reflection of the
        border elements, like this : fedcba|abcdefgh|hgfedcb
        2 - cv2.BORDER_REFLECT_101 or cv.BORDER_DEFAULT - Same as above,
        but with a slight change, like this : gfedcb|abcdefgh|gfedcba
        3 - cv2.BORDER_REPLICATE - Last element is replicated throughout,
        like this: aaaaaa|abcdefgh|hhhhhhh
        4 - cv2.BORDER_WRAP - it will look like this : cdefgh|abcdefgh|abcdefg
    value : double, optional, default=0
        (Deprecated! Use ``values`` instead.) Fill with single value.
    values : tuple of <double>, optional, default=[]
        Fill with value(RGB[A] or gray), up to 4 channels.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    --------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> new_image = mx_border = mx.image.copyMakeBorder(mx_img, 1, 2, 3, 4, type=0)
    >>> new_image
    <NDArray 2324x3489x3 @cpu(0)>
    """
    return _internal._cvcopyMakeBorder(src, top, bot, left, right, *args, **kwargs)


def _get_interp_method(interp, sizes=()):
    """Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Parameters
    ----------
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    sizes : tuple of int
        (old_height, old_width, new_height, new_width), if None provided, auto(9)
        will return Area(2) anyway.

    Returns
    -------
    int
        interp method from 0 to 4
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw < ow:
                return 3
            else:
                return 1
        else:
            return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def resize_short(src, size, interp=2):
    """Resizes shorter edge to size.

    .. note:: `resize_short` uses OpenCV (not the CV2 Python library).
       MXNet must have been built with OpenCV for `resize_short` to work.

    Resizes the original image by setting the shorter edge to size
    and setting the longer edge accordingly.
    Resizing function is called from OpenCV.

    Parameters
    ----------
    src : NDArray
        The original image.
    size : int
        The length to be set for the shorter edge.
    interp : int, optional, default=2
        Interpolation method used for resizing the image.
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.

    Returns
    -------
    NDArray
        An 'NDArray' containing the resized image.

    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> size = 640
    >>> new_image = mx.img.resize_short(image, size)
    >>> new_image
    <NDArray 2321x3482x3 @cpu(0)>
    """
    h, w, _ = src.shape
    if h > w:
        new_h, new_w = size * h // w, size
    else:
        new_h, new_w = size, size * w // h
    return imresize(src, new_w, new_h, interp=_get_interp_method(interp, (h, w, new_h, new_w)))


def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """Crop src at fixed location, and (optionally) resize it to size.

    Parameters
    ----------
    src : NDArray
        Input image
    x0 : int
        Left boundary of the cropping area
    y0 : int
        Top boundary of the cropping area
    w : int
        Width of the cropping area
    h : int
        Height of the cropping area
    size : tuple of (w, h)
        Optional, resize to new size after cropping
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.

    Returns
    -------
    NDArray
        An `NDArray` containing the cropped image.
    """
    out = src[y0:y0+h, x0:x0+w]
    if size is not None and (w, h) != size:
        sizes = (h, w, size[1], size[0])
        out = imresize(out, *size, interp=_get_interp_method(interp, sizes))
    return out


def random_crop(src, size, interp=2):
    """Randomly crop `src` with `size` (width, height).
    Upsample result if `src` is smaller than `size`.

    Parameters
    ----------
    src: Source image `NDArray`
    size: Size of the crop formatted as (width, height). If the `size` is larger
           than the image, then the source image is upsampled to `size` and returned.
    interp: int, optional, default=2
        Interpolation method. See resize_short for details.
    Returns
    -------
    NDArray
        An `NDArray` containing the cropped image.
    Tuple
        A tuple (x, y, width, height) where (x, y) is top-left position of the crop in the
        original image and (width, height) are the dimensions of the cropped image.

    Example
    -------
    >>> im = mx.nd.array(cv2.imread("flower.jpg"))
    >>> cropped_im, rect  = mx.image.random_crop(im, (100, 100))
    >>> print cropped_im
    <NDArray 100x100x1 @cpu(0)>
    >>> print rect
    (20, 21, 100, 100)
    """

    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)


def center_crop(src, size, interp=2):
    """Crops the image `src` to the given `size` by trimming on all four
    sides and preserving the center of the image. Upsamples if `src` is smaller
    than `size`.

    .. note:: This requires MXNet to be compiled with USE_OPENCV.

    Parameters
    ----------
    src : NDArray
        Binary source image data.
    size : list or tuple of int
        The desired output image size.
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.

    Returns
    -------
    NDArray
        The cropped image.
    Tuple
        (x, y, width, height) where x, y are the positions of the crop in the
        original image and width, height the dimensions of the crop.

    Example
    -------
    >>> with open("flower.jpg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.image.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> cropped_image, (x, y, width, height) = mx.image.center_crop(image, (1000, 500))
    >>> cropped_image
    <NDArray 500x1000x3 @cpu(0)>
    >>> x, y, width, height
    (1241, 910, 1000, 500)
    """

    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)


def color_normalize(src, mean, std=None):
    """Normalize src with mean and std.

    Parameters
    ----------
    src : NDArray
        Input image
    mean : NDArray
        RGB mean to be subtracted
    std : NDArray
        RGB standard deviation to be divided

    Returns
    -------
    NDArray
        An `NDArray` containing the normalized image.
    """
    if mean is not None:
        src -= mean
    if std is not None:
        src /= std
    return src


def random_size_crop(src, size, area, ratio, interp=2, **kwargs):
    """Randomly crop src with size. Randomize area and aspect ratio.

    Parameters
    ----------
    src : NDArray
        Input image
    size : tuple of (int, int)
        Size of the crop formatted as (width, height).
    area : float in (0, 1] or tuple of (float, float)
        If tuple, minimum area and maximum area to be maintained after cropping
        If float, minimum area to be maintained after cropping, maximum area is set to 1.0
    ratio : tuple of (float, float)
        Aspect ratio range as (min_aspect_ratio, max_aspect_ratio)
    interp: int, optional, default=2
        Interpolation method. See resize_short for details.
    Returns
    -------
    NDArray
        An `NDArray` containing the cropped image.
    Tuple
        A tuple (x, y, width, height) where (x, y) is top-left position of the crop in the
        original image and (width, height) are the dimensions of the cropped image.

    """
    h, w, _ = src.shape
    src_area = h * w

    if 'min_area' in kwargs:
        warnings.warn('`min_area` is deprecated. Please use `area` instead.',
                      DeprecationWarning)
        area = kwargs.pop('min_area')
    assert not kwargs, "unexpected keyword arguments for `random_size_crop`."

    if isinstance(area, numeric_types):
        area = (area, 1.0)
    for _ in range(10):
        target_area = random.uniform(area[0], area[1]) * src_area
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        new_ratio = np.exp(random.uniform(*log_ratio))

        new_w = int(round(np.sqrt(target_area * new_ratio)))
        new_h = int(round(np.sqrt(target_area / new_ratio)))

        if new_w <= w and new_h <= h:
            x0 = random.randint(0, w - new_w)
            y0 = random.randint(0, h - new_h)

            out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
            return out, (x0, y0, new_w, new_h)

    # fall back to center_crop
    return center_crop(src, size, interp)


def imrotate(src, rotation_degrees, zoom_in=False, zoom_out=False):
    """Rotates the input image(s) of a specific rotation degree.

    Parameters
    ----------
    src : NDArray
        Input image (format CHW) or batch of images (format NCHW),
        in both case is required a float32 data type.
    rotation_degrees: scalar or NDArray
        Wanted rotation in degrees. In case of `src` being a single image
        a scalar is needed, otherwise a mono-dimensional vector of angles
        or a scalar.
    zoom_in: bool
        If True input image(s) will be zoomed in a way so that no padding
        will be shown in the output result.
    zoom_out: bool
        If True input image(s) will be zoomed in a way so that the whole
        original image will be contained in the output result.
    Returns
    -------
    NDArray
        An `NDArray` containing the rotated image(s).
    """
    if zoom_in and zoom_out:
        raise ValueError("`zoom_in` and `zoom_out` cannot be both True")
    if np.dtype(src.dtype) is not np.dtype(np.float32):
        raise TypeError("Only `float32` images are supported by this function")
    # handles the case in which a single image is passed to this function
    expanded = False
    if src.ndim == 3:
        expanded = True
        src = _mx_np.expand_dims(src, 0) if is_np_array() else src.expand_dims(axis=0)
        if not isinstance(rotation_degrees, Number):
            raise TypeError("When a single image is passed the rotation angle is "
                            "required to be a scalar.")
    elif src.ndim != 4:
        raise ValueError("Only 3D and 4D are supported by this function")

    # when a scalar is passed we wrap it into an array
    if isinstance(rotation_degrees, Number):
        rotation_degrees = nd.array([rotation_degrees] * len(src),
                                    ctx=src.ctx)

    if len(src) != len(rotation_degrees):
        raise ValueError(
            "The number of images must be equal to the number of rotation angles"
        )

    rotation_degrees = rotation_degrees.as_in_context(src.ctx)
    rotation_rad = np.pi * rotation_degrees / 180
    # reshape the rotations angle in order to be broadcasted
    # over the `src` tensor
    rotation_rad = rotation_rad.expand_dims(axis=1).expand_dims(axis=2)
    _, _, h, w = src.shape

    # Generate a grid centered at the center of the image
    hscale = (float(h - 1) / 2)
    wscale = (float(w - 1) / 2)
    h_matrix = (
        nd.repeat(nd.arange(h, ctx=src.ctx).astype('float32').reshape(h, 1), w, axis=1) - hscale
    ).expand_dims(axis=0)
    w_matrix = (
        nd.repeat(nd.arange(w, ctx=src.ctx).astype('float32').reshape(1, w), h, axis=0) - wscale
    ).expand_dims(axis=0)
    # perform rotation on the grid
    c_alpha = nd.cos(rotation_rad)
    s_alpha = nd.sin(rotation_rad)
    w_matrix_rot = w_matrix * c_alpha - h_matrix * s_alpha
    h_matrix_rot = w_matrix * s_alpha + h_matrix * c_alpha
    # NOTE: grid normalization must be performed after the rotation
    #       to keep the aspec ratio
    w_matrix_rot = w_matrix_rot / wscale
    h_matrix_rot = h_matrix_rot / hscale

    h, w = nd.array([h], ctx=src.ctx), nd.array([w], ctx=src.ctx)
    # compute the scale factor in case `zoom_in` or `zoom_out` are True
    if zoom_in or zoom_out:
        rho_corner = nd.sqrt(h * h + w * w)
        ang_corner = nd.arctan(h / w)
        corner1_x_pos = nd.abs(rho_corner * nd.cos(ang_corner + nd.abs(rotation_rad)))
        corner1_y_pos = nd.abs(rho_corner * nd.sin(ang_corner + nd.abs(rotation_rad)))
        corner2_x_pos = nd.abs(rho_corner * nd.cos(ang_corner - nd.abs(rotation_rad)))
        corner2_y_pos = nd.abs(rho_corner * nd.sin(ang_corner - nd.abs(rotation_rad)))
        max_x = nd.maximum(corner1_x_pos, corner2_x_pos)
        max_y = nd.maximum(corner1_y_pos, corner2_y_pos)
        if zoom_out:
            scale_x = max_x / w
            scale_y = max_y / h
            globalscale = nd.maximum(scale_x, scale_y)
        else:
            scale_x = w / max_x
            scale_y = h / max_y
            globalscale = nd.minimum(scale_x, scale_y)
        globalscale = globalscale.expand_dims(axis=3)
    else:
        globalscale = 1
    grid = nd.concat(w_matrix_rot.expand_dims(axis=1),
                     h_matrix_rot.expand_dims(axis=1), dim=1)
    grid = grid * globalscale
    if is_np_array():
        src = src.as_nd_ndarray()
    rot_img = nd.BilinearSampler(src, grid)
    if is_np_array():
        rot_img = rot_img.as_np_ndarray()
    if expanded:
        return rot_img[0]
    return rot_img


def random_rotate(src, angle_limits, zoom_in=False, zoom_out=False):
    """Random rotates `src` by an angle included in angle limits.

    Parameters
    ----------
    src : NDArray
        Input image (format CHW) or batch of images (format NCHW),
        in both case is required a float32 data type.
    angle_limits: tuple
        Tuple of 2 elements containing the upper and lower limit
        for rotation angles in degree.
    zoom_in: bool
        If True input image(s) will be zoomed in a way so that no padding
        will be shown in the output result.
    zoom_out: bool
        If True input image(s) will be zoomed in a way so that the whole
        original image will be contained in the output result.
    Returns
    -------
    NDArray
        An `NDArray` containing the rotated image(s).
    """
    if src.ndim == 3:
        rotation_degrees = np.random.uniform(*angle_limits)
    else:
        n = src.shape[0]
        rotation_degrees = nd.array(np.random.uniform(
            *angle_limits,
            size=n
        ))
    return imrotate(src, rotation_degrees,
                    zoom_in=zoom_in, zoom_out=zoom_out)


class Augmenter(object):
    """Image Augmenter base class"""
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for k, v in self._kwargs.items():
            if isinstance(v, nd.NDArray):
                v = v.asnumpy()
            if isinstance(v, np.ndarray):
                v = v.tolist()
                self._kwargs[k] = v

    def dumps(self):
        """Saves the Augmenter to string

        Returns
        -------
        str
            JSON formatted string that describes the Augmenter.
        """
        return json.dumps([self.__class__.__name__.lower(), self._kwargs])

    def __call__(self, src):
        """Abstract implementation body"""
        raise NotImplementedError("Must override implementation.")


class SequentialAug(Augmenter):
    """Composing a sequential augmenter list.

    Parameters
    ----------
    ts : list of augmenters
        A series of augmenters to be applied in sequential order.
    """
    def __init__(self, ts):
        super(SequentialAug, self).__init__()
        self.ts = ts

    def dumps(self):
        """Override the default to avoid duplicate dump."""
        return [self.__class__.__name__.lower(), [x.dumps() for x in self.ts]]

    def __call__(self, src):
        """Augmenter body"""
        for aug in self.ts:
            src = aug(src)
        return src


class ResizeAug(Augmenter):
    """Make resize shorter edge to size augmenter.

    Parameters
    ----------
    size : int
        The length to be set for the shorter edge.
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.
    """
    def __init__(self, size, interp=2):
        super(ResizeAug, self).__init__(size=size, interp=interp)
        self.size = size
        self.interp = interp

    def __call__(self, src):
        """Augmenter body"""
        return resize_short(src, self.size, self.interp)


class ForceResizeAug(Augmenter):
    """Force resize to size regardless of aspect ratio

    Parameters
    ----------
    size : tuple of (int, int)
        The desired size as in (width, height)
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.
    """
    def __init__(self, size, interp=2):
        super(ForceResizeAug, self).__init__(size=size, interp=interp)
        self.size = size
        self.interp = interp

    def __call__(self, src):
        """Augmenter body"""
        sizes = (src.shape[0], src.shape[1], self.size[1], self.size[0])
        return imresize(src, *self.size, interp=_get_interp_method(self.interp, sizes))


class RandomCropAug(Augmenter):
    """Make random crop augmenter

    Parameters
    ----------
    size : int
        The length to be set for the shorter edge.
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.
    """
    def __init__(self, size, interp=2):
        super(RandomCropAug, self).__init__(size=size, interp=interp)
        self.size = size
        self.interp = interp

    def __call__(self, src):
        """Augmenter body"""
        return random_crop(src, self.size, self.interp)[0]


class RandomSizedCropAug(Augmenter):
    """Make random crop with random resizing and random aspect ratio jitter augmenter.

    Parameters
    ----------
    size : tuple of (int, int)
        Size of the crop formatted as (width, height).
    area : float in (0, 1] or tuple of (float, float)
        If tuple, minimum area and maximum area to be maintained after cropping
        If float, minimum area to be maintained after cropping, maximum area is set to 1.0
    ratio : tuple of (float, float)
        Aspect ratio range as (min_aspect_ratio, max_aspect_ratio)
    interp: int, optional, default=2
        Interpolation method. See resize_short for details.
    """
    def __init__(self, size, area, ratio, interp=2, **kwargs):
        super(RandomSizedCropAug, self).__init__(size=size, area=area,
                                                 ratio=ratio, interp=interp)
        self.size = size
        if 'min_area' in kwargs:
            warnings.warn('`min_area` is deprecated. Please use `area` instead.',
                          DeprecationWarning)
            self.area = kwargs.pop('min_area')
        else:
            self.area = area
        self.ratio = ratio
        self.interp = interp
        assert not kwargs, "unexpected keyword arguments for `RandomSizedCropAug`."

    def __call__(self, src):
        """Augmenter body"""
        return random_size_crop(src, self.size, self.area, self.ratio, self.interp)[0]


class CenterCropAug(Augmenter):
    """Make center crop augmenter.

    Parameters
    ----------
    size : list or tuple of int
        The desired output image size.
    interp : int, optional, default=2
        Interpolation method. See resize_short for details.
    """
    def __init__(self, size, interp=2):
        super(CenterCropAug, self).__init__(size=size, interp=interp)
        self.size = size
        self.interp = interp

    def __call__(self, src):
        """Augmenter body"""
        return center_crop(src, self.size, self.interp)[0]


class RandomOrderAug(Augmenter):
    """Apply list of augmenters in random order

    Parameters
    ----------
    ts : list of augmenters
        A series of augmenters to be applied in random order
    """
    def __init__(self, ts):
        super(RandomOrderAug, self).__init__()
        self.ts = ts

    def dumps(self):
        """Override the default to avoid duplicate dump."""
        return [self.__class__.__name__.lower(), [x.dumps() for x in self.ts]]

    def __call__(self, src):
        """Augmenter body"""
        random.shuffle(self.ts)
        for t in self.ts:
            src = t(src)
        return src


class BrightnessJitterAug(Augmenter):
    """Random brightness jitter augmentation.

    Parameters
    ----------
    brightness : float
        The brightness jitter ratio range, [0, 1]
    """
    def __init__(self, brightness):
        super(BrightnessJitterAug, self).__init__(brightness=brightness)
        self.brightness = brightness

    def __call__(self, src):
        """Augmenter body"""
        alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
        src *= alpha
        return src


class ContrastJitterAug(Augmenter):
    """Random contrast jitter augmentation.

    Parameters
    ----------
    contrast : float
        The contrast jitter ratio range, [0, 1]
    """
    def __init__(self, contrast):
        super(ContrastJitterAug, self).__init__(contrast=contrast)
        self.contrast = contrast
        self.coef = nd.array([[[0.299, 0.587, 0.114]]])

    def __call__(self, src):
        """Augmenter body"""
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        gray = src * self.coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
        src *= alpha
        src += gray
        return src


class SaturationJitterAug(Augmenter):
    """Random saturation jitter augmentation.

    Parameters
    ----------
    saturation : float
        The saturation jitter ratio range, [0, 1]
    """
    def __init__(self, saturation):
        super(SaturationJitterAug, self).__init__(saturation=saturation)
        self.saturation = saturation
        self.coef = nd.array([[[0.299, 0.587, 0.114]]])

    def __call__(self, src):
        """Augmenter body"""
        alpha = 1.0 + random.uniform(-self.saturation, self.saturation)
        gray = src * self.coef
        gray = nd.sum(gray, axis=2, keepdims=True)
        gray *= (1.0 - alpha)
        src *= alpha
        src += gray
        return src


class HueJitterAug(Augmenter):
    """Random hue jitter augmentation.

    Parameters
    ----------
    hue : float
        The hue jitter ratio range, [0, 1]
    """
    def __init__(self, hue):
        super(HueJitterAug, self).__init__(hue=hue)
        self.hue = hue
        self.tyiq = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.274, -0.321],
                              [0.211, -0.523, 0.311]])
        self.ityiq = np.array([[1.0, 0.956, 0.621],
                               [1.0, -0.272, -0.647],
                               [1.0, -1.107, 1.705]])

    def __call__(self, src):
        """Augmenter body.
        Using approximate linear transfomation described in:
        https://beesbuzz.biz/code/hsv_color_transforms.php
        """
        alpha = random.uniform(-self.hue, self.hue)
        u = np.cos(alpha * np.pi)
        w = np.sin(alpha * np.pi)
        bt = np.array([[1.0, 0.0, 0.0],
                       [0.0, u, -w],
                       [0.0, w, u]])
        t = np.dot(np.dot(self.ityiq, bt), self.tyiq).T
        src = nd.dot(src, nd.array(t))
        return src


class ColorJitterAug(RandomOrderAug):
    """Apply random brightness, contrast and saturation jitter in random order.

    Parameters
    ----------
    brightness : float
        The brightness jitter ratio range, [0, 1]
    contrast : float
        The contrast jitter ratio range, [0, 1]
    saturation : float
        The saturation jitter ratio range, [0, 1]
    """
    def __init__(self, brightness, contrast, saturation):
        ts = []
        if brightness > 0:
            ts.append(BrightnessJitterAug(brightness))
        if contrast > 0:
            ts.append(ContrastJitterAug(contrast))
        if saturation > 0:
            ts.append(SaturationJitterAug(saturation))
        super(ColorJitterAug, self).__init__(ts)


class LightingAug(Augmenter):
    """Add PCA based noise.

    Parameters
    ----------
    alphastd : float
        Noise level
    eigval : 3x1 np.array
        Eigen values
    eigvec : 3x3 np.array
        Eigen vectors
    """
    def __init__(self, alphastd, eigval, eigvec):
        super(LightingAug, self).__init__(alphastd=alphastd, eigval=eigval, eigvec=eigvec)
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, src):
        """Augmenter body"""
        alpha = np.random.normal(0, self.alphastd, size=(3,))
        rgb = np.dot(self.eigvec * alpha, self.eigval)
        src += nd.array(rgb)
        return src


class ColorNormalizeAug(Augmenter):
    """Mean and std normalization.

    Parameters
    ----------
    mean : NDArray
        RGB mean to be subtracted
    std : NDArray
        RGB standard deviation to be divided
    """
    def __init__(self, mean, std):
        super(ColorNormalizeAug, self).__init__(mean=mean, std=std)
        self.mean = mean if mean is None or isinstance(mean, nd.NDArray) else nd.array(mean)
        self.std = std if std is None or isinstance(std, nd.NDArray) else nd.array(std)

    def __call__(self, src):
        """Augmenter body"""
        return color_normalize(src, self.mean, self.std)


class RandomGrayAug(Augmenter):
    """Randomly convert to gray image.

    Parameters
    ----------
    p : float
        Probability to convert to grayscale
    """
    def __init__(self, p):
        super(RandomGrayAug, self).__init__(p=p)
        self.p = p
        self.mat = nd.array([[0.21, 0.21, 0.21],
                             [0.72, 0.72, 0.72],
                             [0.07, 0.07, 0.07]])

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = nd.dot(src, self.mat)
        return src


class HorizontalFlipAug(Augmenter):
    """Random horizontal flip.

    Parameters
    ----------
    p : float
        Probability to flip image horizontally
    """
    def __init__(self, p):
        super(HorizontalFlipAug, self).__init__(p=p)
        self.p = p

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = nd.flip(src, axis=1)
        return src


class CastAug(Augmenter):
    """Cast to float32"""
    def __init__(self, typ='float32'):
        super(CastAug, self).__init__(type=typ)
        self.typ = typ

    def __call__(self, src):
        """Augmenter body"""
        src = src.astype(self.typ)
        return src


def CreateAugmenter(data_shape, resize=0, rand_crop=False, rand_resize=False, rand_mirror=False,
                    mean=None, std=None, brightness=0, contrast=0, saturation=0, hue=0,
                    pca_noise=0, rand_gray=0, inter_method=2):
    """Creates an augmenter list.

    Parameters
    ----------
    data_shape : tuple of int
        Shape for output data
    resize : int
        Resize shorter edge if larger than 0 at the begining
    rand_crop : bool
        Whether to enable random cropping other than center crop
    rand_resize : bool
        Whether to enable random sized cropping, require rand_crop to be enabled
    rand_gray : float
        [0, 1], probability to convert to grayscale for all channels, the number
        of channels will not be reduced to 1
    rand_mirror : bool
        Whether to apply horizontal flip to image with probability 0.5
    mean : np.ndarray or None
        Mean pixel values for [r, g, b]
    std : np.ndarray or None
        Standard deviations for [r, g, b]
    brightness : float
        Brightness jittering range (percent)
    contrast : float
        Contrast jittering range (percent)
    saturation : float
        Saturation jittering range (percent)
    hue : float
        Hue jittering range (percent)
    pca_noise : float
        Pca noise level (percent)
    inter_method : int, default=2(Area-based)
        Interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).

    Examples
    --------
    >>> # An example of creating multiple augmenters
    >>> augs = mx.image.CreateAugmenter(data_shape=(3, 300, 300), rand_mirror=True,
    ...    mean=True, brightness=0.125, contrast=0.125, rand_gray=0.05,
    ...    saturation=0.125, pca_noise=0.05, inter_method=10)
    >>> # dump the details
    >>> for aug in augs:
    ...    aug.dumps()
    """
    auglist = []

    if resize > 0:
        auglist.append(ResizeAug(resize, inter_method))

    crop_size = (data_shape[2], data_shape[1])
    if rand_resize:
        assert rand_crop
        auglist.append(RandomSizedCropAug(crop_size, 0.08, (3.0 / 4.0, 4.0 / 3.0), inter_method))
    elif rand_crop:
        auglist.append(RandomCropAug(crop_size, inter_method))
    else:
        auglist.append(CenterCropAug(crop_size, inter_method))

    if rand_mirror:
        auglist.append(HorizontalFlipAug(0.5))

    auglist.append(CastAug())

    if brightness or contrast or saturation:
        auglist.append(ColorJitterAug(brightness, contrast, saturation))

    if hue:
        auglist.append(HueJitterAug(hue))

    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
        auglist.append(LightingAug(pca_noise, eigval, eigvec))

    if rand_gray > 0:
        auglist.append(RandomGrayAug(rand_gray))

    if mean is True:
        mean = nd.array([123.68, 116.28, 103.53])
    elif mean is not None:
        assert isinstance(mean, (np.ndarray, nd.NDArray)) and mean.shape[0] in [1, 3]

    if std is True:
        std = nd.array([58.395, 57.12, 57.375])
    elif std is not None:
        assert isinstance(std, (np.ndarray, nd.NDArray)) and std.shape[0] in [1, 3]

    if mean is not None or std is not None:
        auglist.append(ColorNormalizeAug(mean, std))

    return auglist


class ImageIter(io.DataIter):
    """Image data iterator with a large number of augmentation choices.
    This iterator supports reading from both .rec files and raw image files.

    To load input images from .rec files, use `path_imgrec` parameter and to load from raw image
    files, use `path_imglist` and `path_root` parameters.

    To use data partition (for distributed training) or shuffling, specify `path_imgidx` parameter.

    Parameters
    ----------
    batch_size : int
        Number of examples per batch.
    data_shape : tuple
        Data shape in (channels, height, width) format.
        For now, only RGB image with 3 channels is supported.
    label_width : int, optional
        Number of labels per example. The default label width is 1.
    path_imgrec : str
        Path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec.
    path_imglist : str
        Path to image list (.lst).
        Created with tools/im2rec.py or with custom script.
        Format: Tab separated record of index, one or more labels and relative_path_from_root.
    imglist: list
        A list of images with the label(s).
        Each item is a list [imagelabel: float or list of float, imgpath].
    path_root : str
        Root folder of image files.
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration or not.
        Can be slow for HDD.
    part_index : int
        Partition index.
    num_parts : int
        Total number of partitions.
    data_name : str
        Data name for provided symbols.
    label_name : str
        Label name for provided symbols.
    dtype : str
        Label data type. Default: float32. Other options: int32, int64, float64
    last_batch_handle : str, optional
        How to handle the last batch.
        This parameter can be 'pad'(default), 'discard' or 'roll_over'.
        If 'pad', the last batch will be padded with data starting from the begining
        If 'discard', the last batch will be discarded
        If 'roll_over', the remaining elements will be rolled over to the next iteration
    kwargs : ...
        More arguments for creating augmenter. See mx.image.CreateAugmenter.
    """

    def __init__(self, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None,
                 data_name='data', label_name='softmax_label', dtype='float32',
                 last_batch_handle='pad', **kwargs):
        # 2020-06-02, mxnet 2.0.0
        warnings.warn(("mxnet.image.ImageIter has been deprecated "
                       "and will be removed from a future release. "
                       "Please use `gluon.contrib.data.vision.ImageDataLoader` instead. "),
                      DeprecationWarning, stacklevel=2)
        super(ImageIter, self).__init__()
        assert path_imgrec or path_imglist or (isinstance(imglist, list))
        assert dtype in ['int32', 'float32', 'int64', 'float64'], dtype + ' label not supported'
        num_threads = os.environ.get('MXNET_CPU_WORKER_NTHREADS', 1)
        logging.info('Using %s threads for decoding...', str(num_threads))
        logging.info('Set enviroment variable MXNET_CPU_WORKER_NTHREADS to a'
                     ' larger number to use more threads.')
        class_name = self.__class__.__name__
        if path_imgrec:
            logging.info('%s: loading recordio %s...',
                         class_name, path_imgrec)
            if path_imgidx:
                self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                self.imgidx = list(self.imgrec.keys)
            else:
                self.imgrec = recordio.MXRecordIO(path_imgrec, 'r')
                self.imgidx = None
        else:
            self.imgrec = None

        array_fn = _mx_np.array if is_np_array() else nd.array
        if path_imglist:
            logging.info('%s: loading image list %s...', class_name, path_imglist)
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = array_fn(line[1:-1], dtype=dtype)
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        elif isinstance(imglist, list):
            logging.info('%s: loading image list...', class_name)
            result = {}
            imgkeys = []
            index = 1
            for img in imglist:
                key = str(index)
                index += 1
                if len(img) > 2:
                    label = array_fn(img[:-1], dtype=dtype)
                elif isinstance(img[0], numeric_types):
                    label = array_fn([img[0]], dtype=dtype)
                else:
                    label = array_fn(img[0], dtype=dtype)
                result[key] = (label, img[-1])
                imgkeys.append(str(key))
            self.imglist = result
        else:
            self.imglist = None
        self.path_root = path_root

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [(label_name, (batch_size, label_width))]
        else:
            self.provide_label = [(label_name, (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width
        self.shuffle = shuffle
        if self.imgrec is None:
            self.seq = imgkeys
        elif shuffle or num_parts > 1 or path_imgidx:
            assert self.imgidx is not None
            self.seq = self.imgidx
        else:
            self.seq = None

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N // num_parts
            self.seq = self.seq[part_index * C:(part_index + 1) * C]
        if aug_list is None:
            self.auglist = CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self._allow_read = True
        self.last_batch_handle = last_batch_handle
        self.num_image = len(self.seq) if self.seq is not None else None
        self._cache_data = None
        self._cache_label = None
        self._cache_idx = None
        self.reset()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.seq is not None and self.shuffle:
            random.shuffle(self.seq)
        if self.last_batch_handle != 'roll_over' or \
            self._cache_data is None:
            if self.imgrec is not None:
                self.imgrec.reset()
            self.cur = 0
            if self._allow_read is False:
                self._allow_read = True

    def hard_reset(self):
        """Resets the iterator and ignore roll over data"""
        if self.seq is not None and self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0
        self._allow_read = True
        self._cache_data = None
        self._cache_label = None
        self._cache_idx = None

    def next_sample(self):
        """Helper function for reading in next sample."""
        if self._allow_read is False:
            raise StopIteration
        if self.seq is not None:
            if self.cur < self.num_image:
                idx = self.seq[self.cur]
            else:
                if self.last_batch_handle != 'discard':
                    self.cur = 0
                raise StopIteration
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return header.label, img
                else:
                    return self.imglist[idx][0], img
            else:
                label, fname = self.imglist[idx]
                return label, self.read_image(fname)
        else:
            s = self.imgrec.read()
            if s is None:
                if self.last_batch_handle != 'discard':
                    self.imgrec.reset()
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img

    def _batchify(self, batch_data, batch_label, start=0):
        """Helper function for batchifying data"""
        i = start
        batch_size = self.batch_size
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = self.imdecode(s)
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                data = self.augmentation_transform(data)
                assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                batch_data[i] = self.postprocess_data(data)
                batch_label[i] = label
                i += 1
        except StopIteration:
            if not i:
                raise StopIteration
        return i

    def next(self):
        """Returns the next batch of data."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        # if last batch data is rolled over
        if self._cache_data is not None:
            # check both the data and label have values
            assert self._cache_label is not None, "_cache_label didn't have values"
            assert self._cache_idx is not None, "_cache_idx didn't have values"
            batch_data = self._cache_data
            batch_label = self._cache_label
            i = self._cache_idx
            # clear the cache data
        else:
            if is_np_array():
                zeros_fn = _mx_np.zeros
                empty_fn = _mx_np.empty
            else:
                zeros_fn = nd.zeros
                empty_fn = nd.empty
            batch_data = zeros_fn((batch_size, c, h, w))
            batch_label = empty_fn(self.provide_label[0][1])
            i = self._batchify(batch_data, batch_label)
        # calculate the padding
        pad = batch_size - i
        # handle padding for the last batch
        if pad != 0:
            if self.last_batch_handle == 'discard':
                raise StopIteration
            # if the option is 'roll_over', throw StopIteration and cache the data
            if self.last_batch_handle == 'roll_over' and \
                self._cache_data is None:
                self._cache_data = batch_data
                self._cache_label = batch_label
                self._cache_idx = i
                raise StopIteration

            _ = self._batchify(batch_data, batch_label, i)
            if self.last_batch_handle == 'pad':
                self._allow_read = False
            else:
                self._cache_data = None
                self._cache_label = None
                self._cache_idx = None

        return io.DataBatch([batch_data], [batch_label], pad=pad)

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        def locate():
            """Locate the image file/index if decode fails."""
            if self.seq is not None:
                idx = self.seq[(self.cur % self.num_image) - 1]
            else:
                idx = (self.cur % self.num_image) - 1
            if self.imglist is not None:
                _, fname = self.imglist[idx]
                msg = "filename: {}".format(fname)
            else:
                msg = "index: {}".format(idx)
            return "Broken image " + msg
        try:
            img = imdecode(s)
        except Exception as e:
            raise RuntimeError("{}, {}".format(locate(), e))
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.
        Examples
        --------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = aug(data)
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        if is_np_array():
            return datum.transpose(2, 0, 1)
        else:
            return nd.transpose(datum, axes=(2, 0, 1))
