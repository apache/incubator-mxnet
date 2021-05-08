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

# pylint: disable=unused-import, too-many-lines
"""Read images and perform augmentations for object detection."""
import json
import logging
import random
import warnings
import numpy as np

# 2020-06-02, mxnet 2.0.0
warnings.warn(("mxnet.image.detection.* functions has been deprecated "
               "and will be removed from a future release. "
               "Please use gluon transform functions instead. "
               "For example, you can replace `ImageDetIter` with "
               "`gluon.contrib.data.vision.ImageBboxDataLoader`."),
              DeprecationWarning, stacklevel=2)

from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray._internal import _cvcopyMakeBorder as copyMakeBorder
from .. import io
from .image import RandomOrderAug, ColorJitterAug, LightingAug, ColorNormalizeAug
from .image import ResizeAug, ForceResizeAug, CastAug, HueJitterAug, RandomGrayAug
from .image import fixed_crop, ImageIter, Augmenter
from ..util import is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported


class DetAugmenter(object):
    """Detection base augmenter"""
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

    def __call__(self, src, label):
        """Abstract implementation body"""
        raise NotImplementedError("Must override implementation.")


class DetBorrowAug(DetAugmenter):
    """Borrow standard augmenter from image classification.
    Which is good once you know label won't be affected after this augmenter.

    Parameters
    ----------
    augmenter : mx.image.Augmenter
        The borrowed standard augmenter which has no effect on label
    """
    def __init__(self, augmenter):
        if not isinstance(augmenter, Augmenter):
            raise TypeError('Borrowing from invalid Augmenter')
        super(DetBorrowAug, self).__init__(augmenter=augmenter.dumps())
        self.augmenter = augmenter

    def dumps(self):
        """Override the default one to avoid duplicate dump."""
        return [self.__class__.__name__.lower(), self.augmenter.dumps()]

    def __call__(self, src, label):
        """Augmenter implementation body"""
        src = self.augmenter(src)
        return (src, label)


class DetRandomSelectAug(DetAugmenter):
    """Randomly select one augmenter to apply, with chance to skip all.

    Parameters
    ----------
    aug_list : list of DetAugmenter
        The random selection will be applied to one of the augmenters
    skip_prob : float
        The probability to skip all augmenters and return input directly
    """
    def __init__(self, aug_list, skip_prob=0):
        super(DetRandomSelectAug, self).__init__(skip_prob=skip_prob)
        if not isinstance(aug_list, (list, tuple)):
            aug_list = [aug_list]
        for aug in aug_list:
            if not isinstance(aug, DetAugmenter):
                raise ValueError('Allow DetAugmenter in list only')
        if not aug_list:
            skip_prob = 1  # disabled

        self.aug_list = aug_list
        self.skip_prob = skip_prob

    def dumps(self):
        """Override default."""
        return [self.__class__.__name__.lower(), [x.dumps() for x in self.aug_list]]

    def __call__(self, src, label):
        """Augmenter implementation body"""
        if random.random() < self.skip_prob:
            return (src, label)
        else:
            random.shuffle(self.aug_list)
            return self.aug_list[0](src, label)


class DetHorizontalFlipAug(DetAugmenter):
    """Random horizontal flipping.

    Parameters
    ----------
    p : float
        chance [0, 1] to flip
    """
    def __init__(self, p):
        super(DetHorizontalFlipAug, self).__init__(p=p)
        self.p = p

    def __call__(self, src, label):
        """Augmenter implementation"""
        if random.random() < self.p:
            src = nd.flip(src, axis=1)
            self._flip_label(label)
        return (src, label)

    def _flip_label(self, label):
        """Helper function to flip label."""
        tmp = 1.0 - label[:, 1]
        label[:, 1] = 1.0 - label[:, 3]
        label[:, 3] = tmp


class DetRandomCropAug(DetAugmenter):
    """Random cropping with constraints

    Parameters
    ----------
    min_object_covered : float, default=0.1
        The cropped area of the image must contain at least this fraction of
        any bounding box supplied. The value of this parameter should be non-negative.
        In the case of 0, the cropped area does not need to overlap any of the
        bounding boxes supplied.
    min_eject_coverage : float, default=0.3
        The minimum coverage of cropped sample w.r.t its original size. With this
        constraint, objects that have marginal area after crop will be discarded.
    aspect_ratio_range : tuple of floats, default=(0.75, 1.33)
        The cropped area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats, default=(0.05, 1.0)
        The cropped area of the image must contain a fraction of the supplied
        image within in this range.
    max_attempts : int, default=50
        Number of attempts at generating a cropped/padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.
    """
    def __init__(self, min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33),
                 area_range=(0.05, 1.0), min_eject_coverage=0.3, max_attempts=50):
        if not isinstance(aspect_ratio_range, (tuple, list)):
            assert isinstance(aspect_ratio_range, numeric_types)
            logging.info('Using fixed aspect ratio: %s in DetRandomCropAug',
                         str(aspect_ratio_range))
            aspect_ratio_range = (aspect_ratio_range, aspect_ratio_range)
        if not isinstance(area_range, (tuple, list)):
            assert isinstance(area_range, numeric_types)
            logging.info('Using fixed area range: %s in DetRandomCropAug', area_range)
            area_range = (area_range, area_range)
        super(DetRandomCropAug, self).__init__(min_object_covered=min_object_covered,
                                               aspect_ratio_range=aspect_ratio_range,
                                               area_range=area_range,
                                               min_eject_coverage=min_eject_coverage,
                                               max_attempts=max_attempts)
        self.min_object_covered = min_object_covered
        self.min_eject_coverage = min_eject_coverage
        self.max_attempts = max_attempts
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.enabled = False
        if (area_range[1] <= 0 or area_range[0] > area_range[1]):
            warnings.warn('Skip DetRandomCropAug due to invalid area_range: %s', area_range)
        elif (aspect_ratio_range[0] > aspect_ratio_range[1] or aspect_ratio_range[0] <= 0):
            warnings.warn('Skip DetRandomCropAug due to invalid aspect_ratio_range: %s',
                          aspect_ratio_range)
        else:
            self.enabled = True

    def __call__(self, src, label):
        """Augmenter implementation body"""
        crop = self._random_crop_proposal(label, src.shape[0], src.shape[1])
        if crop:
            x, y, w, h, label = crop
            src = fixed_crop(src, x, y, w, h, None)
        return (src, label)

    def _calculate_areas(self, label):
        """Calculate areas for multiple labels"""
        heights = np.maximum(0, label[:, 3] - label[:, 1])
        widths = np.maximum(0, label[:, 2] - label[:, 0])
        return heights * widths


    def _intersect(self, label, xmin, ymin, xmax, ymax):
        """Calculate intersect areas, normalized."""
        left = np.maximum(label[:, 0], xmin)
        right = np.minimum(label[:, 2], xmax)
        top = np.maximum(label[:, 1], ymin)
        bot = np.minimum(label[:, 3], ymax)
        invalid = np.where(np.logical_or(left >= right, top >= bot))[0]
        out = label.copy()
        out[:, 0] = left
        out[:, 1] = top
        out[:, 2] = right
        out[:, 3] = bot
        out[invalid, :] = 0
        return out

    def _check_satisfy_constraints(self, label, xmin, ymin, xmax, ymax, width, height):
        """Check if constrains are satisfied"""
        if (xmax - xmin) * (ymax - ymin) < 2:
            return False  # only 1 pixel
        x1 = float(xmin) / width
        y1 = float(ymin) / height
        x2 = float(xmax) / width
        y2 = float(ymax) / height
        object_areas = self._calculate_areas(label[:, 1:])
        valid_objects = np.where(object_areas * width * height > 2)[0]
        if valid_objects.size < 1:
            return False
        intersects = self._intersect(label[valid_objects, 1:], x1, y1, x2, y2)
        coverages = self._calculate_areas(intersects) / object_areas[valid_objects]
        coverages = coverages[np.where(coverages > 0)[0]]
        return coverages.size > 0 and np.amin(coverages) > self.min_object_covered

    def _update_labels(self, label, crop_box, height, width):
        """Convert labels according to crop box"""
        xmin = float(crop_box[0]) / width
        ymin = float(crop_box[1]) / height
        w = float(crop_box[2]) / width
        h = float(crop_box[3]) / height
        out = label.copy()
        out[:, (1, 3)] -= xmin
        out[:, (2, 4)] -= ymin
        out[:, (1, 3)] /= w
        out[:, (2, 4)] /= h
        out[:, 1:5] = np.maximum(0, out[:, 1:5])
        out[:, 1:5] = np.minimum(1, out[:, 1:5])
        coverage = self._calculate_areas(out[:, 1:]) * w * h / self._calculate_areas(label[:, 1:])
        valid = np.logical_and(out[:, 3] > out[:, 1], out[:, 4] > out[:, 2])
        valid = np.logical_and(valid, coverage > self.min_eject_coverage)
        valid = np.where(valid)[0]
        if valid.size < 1:
            return None
        out = out[valid, :]
        return out

    def _random_crop_proposal(self, label, height, width):
        """Propose cropping areas"""
        from math import sqrt

        if not self.enabled or height <= 0 or width <= 0:
            return ()
        min_area = self.area_range[0] * height * width
        max_area = self.area_range[1] * height * width
        for _ in range(self.max_attempts):
            ratio = random.uniform(*self.aspect_ratio_range)
            if ratio <= 0:
                continue
            h = int(round(sqrt(min_area / ratio)))
            max_h = int(round(sqrt(max_area / ratio)))
            if round(max_h * ratio) > width:
                # find smallest max_h satifying round(max_h * ratio) <= width
                max_h = int((width + 0.4999999) / ratio)
            if max_h > height:
                max_h = height
            if h > max_h:
                h = max_h
            if h < max_h:
                # generate random h in range [h, max_h]
                h = random.randint(h, max_h)
            w = int(round(h * ratio))
            assert w <= width

            # trying to fix rounding problems
            area = w * h
            if area < min_area:
                h += 1
                w = int(round(h * ratio))
                area = w * h
            if area > max_area:
                h -= 1
                w = int(round(h * ratio))
                area = w * h
            if not (min_area <= area <= max_area and 0 <= w <= width and 0 <= h <= height):
                continue

            y = random.randint(0, max(0, height - h))
            x = random.randint(0, max(0, width - w))
            if self._check_satisfy_constraints(label, x, y, x + w, y + h, width, height):
                new_label = self._update_labels(label, (x, y, w, h), height, width)
                if new_label is not None:
                    return (x, y, w, h, new_label)
        return ()


class DetRandomPadAug(DetAugmenter):
    """Random padding augmenter.

    Parameters
    ----------
    aspect_ratio_range : tuple of floats, default=(0.75, 1.33)
        The padded area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats, default=(1.0, 3.0)
        The padded area of the image must be larger than the original area
    max_attempts : int, default=50
        Number of attempts at generating a padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.
    pad_val: float or tuple of float, default=(128, 128, 128)
        pixel value to be filled when padding is enabled.
    """
    def __init__(self, aspect_ratio_range=(0.75, 1.33), area_range=(1.0, 3.0),
                 max_attempts=50, pad_val=(128, 128, 128)):
        if not isinstance(pad_val, (list, tuple)):
            assert isinstance(pad_val, numeric_types)
            pad_val = (pad_val)
        if not isinstance(aspect_ratio_range, (list, tuple)):
            assert isinstance(aspect_ratio_range, numeric_types)
            logging.info('Using fixed aspect ratio: %s in DetRandomPadAug',
                         str(aspect_ratio_range))
            aspect_ratio_range = (aspect_ratio_range, aspect_ratio_range)
        if not isinstance(area_range, (tuple, list)):
            assert isinstance(area_range, numeric_types)
            logging.info('Using fixed area range: %s in DetRandomPadAug', area_range)
            area_range = (area_range, area_range)
        super(DetRandomPadAug, self).__init__(aspect_ratio_range=aspect_ratio_range,
                                              area_range=area_range, max_attempts=max_attempts,
                                              pad_val=pad_val)
        self.pad_val = pad_val
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self.enabled = False
        if (area_range[1] <= 1.0 or area_range[0] > area_range[1]):
            warnings.warn('Skip DetRandomPadAug due to invalid parameters: %s', area_range)
        elif (aspect_ratio_range[0] <= 0 or aspect_ratio_range[0] > aspect_ratio_range[1]):
            warnings.warn('Skip DetRandomPadAug due to invalid aspect_ratio_range: %s',
                          aspect_ratio_range)
        else:
            self.enabled = True

    def __call__(self, src, label):
        """Augmenter body"""
        height, width, _ = src.shape
        pad = self._random_pad_proposal(label, height, width)
        if pad:
            x, y, w, h, label = pad
            src = copyMakeBorder(src, y, h-y-height, x, w-x-width, 16, values=self.pad_val)
        return (src, label)

    def _update_labels(self, label, pad_box, height, width):
        """Update label according to padding region"""
        out = label.copy()
        out[:, (1, 3)] = (out[:, (1, 3)] * width + pad_box[0]) / pad_box[2]
        out[:, (2, 4)] = (out[:, (2, 4)] * height + pad_box[1]) / pad_box[3]
        return out

    def _random_pad_proposal(self, label, height, width):
        """Generate random padding region"""
        from math import sqrt
        if not self.enabled or height <= 0 or width <= 0:
            return ()
        min_area = self.area_range[0] * height * width
        max_area = self.area_range[1] * height * width
        for _ in range(self.max_attempts):
            ratio = random.uniform(*self.aspect_ratio_range)
            if ratio <= 0:
                continue
            h = int(round(sqrt(min_area / ratio)))
            max_h = int(round(sqrt(max_area / ratio)))
            if round(h * ratio) < width:
                h = int((width + 0.499999) / ratio)
            if h < height:
                h = height
            if h > max_h:
                h = max_h
            if h < max_h:
                h = random.randint(h, max_h)
            w = int(round(h * ratio))
            if (h - height) < 2 or (w - width) < 2:
                continue  # marginal padding is not helpful

            y = random.randint(0, max(0, h - height))
            x = random.randint(0, max(0, w - width))
            new_label = self._update_labels(label, (x, y, w, h), height, width)
            return (x, y, w, h, new_label)
        return ()


def CreateMultiRandCropAugmenter(min_object_covered=0.1, aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.05, 1.0), min_eject_coverage=0.3,
                                 max_attempts=50, skip_prob=0):
    """Helper function to create multiple random crop augmenters.

    Parameters
    ----------
    min_object_covered : float or list of float, default=0.1
        The cropped area of the image must contain at least this fraction of
        any bounding box supplied. The value of this parameter should be non-negative.
        In the case of 0, the cropped area does not need to overlap any of the
        bounding boxes supplied.
    min_eject_coverage : float or list of float, default=0.3
        The minimum coverage of cropped sample w.r.t its original size. With this
        constraint, objects that have marginal area after crop will be discarded.
    aspect_ratio_range : tuple of floats or list of tuple of floats, default=(0.75, 1.33)
        The cropped area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats or list of tuple of floats, default=(0.05, 1.0)
        The cropped area of the image must contain a fraction of the supplied
        image within in this range.
    max_attempts : int or list of int, default=50
        Number of attempts at generating a cropped/padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.

    Examples
    --------
    >>> # An example of creating multiple random crop augmenters
    >>> min_object_covered = [0.1, 0.3, 0.5, 0.7, 0.9]  # use 5 augmenters
    >>> aspect_ratio_range = (0.75, 1.33)  # use same range for all augmenters
    >>> area_range = [(0.1, 1.0), (0.2, 1.0), (0.2, 1.0), (0.3, 0.9), (0.5, 1.0)]
    >>> min_eject_coverage = 0.3
    >>> max_attempts = 50
    >>> aug = mx.image.det.CreateMultiRandCropAugmenter(min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range, area_range=area_range,
            min_eject_coverage=min_eject_coverage, max_attempts=max_attempts,
            skip_prob=0)
    >>> aug.dumps()  # show some details

    """
    def align_parameters(params):
        """Align parameters as pairs"""
        out_params = []
        num = 1
        for p in params:
            if not isinstance(p, list):
                p = [p]
            out_params.append(p)
            num = max(num, len(p))
        # align for each param
        for k, p in enumerate(out_params):
            if len(p) != num:
                assert len(p) == 1
                out_params[k] = p * num
        return out_params

    aligned_params = align_parameters([min_object_covered, aspect_ratio_range, area_range,
                                       min_eject_coverage, max_attempts])
    augs = []
    for moc, arr, ar, mec, ma in zip(*aligned_params):
        augs.append(DetRandomCropAug(min_object_covered=moc, aspect_ratio_range=arr,
                                     area_range=ar, min_eject_coverage=mec, max_attempts=ma))
    return DetRandomSelectAug(augs, skip_prob=skip_prob)


def CreateDetAugmenter(data_shape, resize=0, rand_crop=0, rand_pad=0, rand_gray=0,
                       rand_mirror=False, mean=None, std=None, brightness=0, contrast=0,
                       saturation=0, pca_noise=0, hue=0, inter_method=2, min_object_covered=0.1,
                       aspect_ratio_range=(0.75, 1.33), area_range=(0.05, 3.0),
                       min_eject_coverage=0.3, max_attempts=50, pad_val=(127, 127, 127)):
    """Create augmenters for detection.

    Parameters
    ----------
    data_shape : tuple of int
        Shape for output data
    resize : int
        Resize shorter edge if larger than 0 at the begining
    rand_crop : float
        [0, 1], probability to apply random cropping
    rand_pad : float
        [0, 1], probability to apply random padding
    rand_gray : float
        [0, 1], probability to convert to grayscale for all channels
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
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
    min_object_covered : float
        The cropped area of the image must contain at least this fraction of
        any bounding box supplied. The value of this parameter should be non-negative.
        In the case of 0, the cropped area does not need to overlap any of the
        bounding boxes supplied.
    min_eject_coverage : float
        The minimum coverage of cropped sample w.r.t its original size. With this
        constraint, objects that have marginal area after crop will be discarded.
    aspect_ratio_range : tuple of floats
        The cropped area of the image must have an aspect ratio = width / height
        within this range.
    area_range : tuple of floats
        The cropped area of the image must contain a fraction of the supplied
        image within in this range.
    max_attempts : int
        Number of attempts at generating a cropped/padded region of the image of the
        specified constraints. After max_attempts failures, return the original image.
    pad_val: float
        Pixel value to be filled when padding is enabled. pad_val will automatically
        be subtracted by mean and divided by std if applicable.

    Examples
    --------
    >>> # An example of creating multiple augmenters
    >>> augs = mx.image.CreateDetAugmenter(data_shape=(3, 300, 300), rand_crop=0.5,
    ...    rand_pad=0.5, rand_mirror=True, mean=True, brightness=0.125, contrast=0.125,
    ...    saturation=0.125, pca_noise=0.05, inter_method=10, min_object_covered=[0.3, 0.5, 0.9],
    ...    area_range=(0.3, 3.0))
    >>> # dump the details
    >>> for aug in augs:
    ...    aug.dumps()
    """
    auglist = []

    if resize > 0:
        auglist.append(DetBorrowAug(ResizeAug(resize, inter_method)))

    if rand_crop > 0:
        crop_augs = CreateMultiRandCropAugmenter(min_object_covered, aspect_ratio_range,
                                                 area_range, min_eject_coverage,
                                                 max_attempts, skip_prob=(1 - rand_crop))
        auglist.append(crop_augs)

    if rand_mirror > 0:
        auglist.append(DetHorizontalFlipAug(0.5))

    # apply random padding as late as possible to save computation
    if rand_pad > 0:
        pad_aug = DetRandomPadAug(aspect_ratio_range,
                                  (1.0, area_range[1]), max_attempts, pad_val)
        auglist.append(DetRandomSelectAug([pad_aug], 1 - rand_pad))

    # force resize
    auglist.append(DetBorrowAug(ForceResizeAug((data_shape[2], data_shape[1]), inter_method)))

    auglist.append(DetBorrowAug(CastAug()))

    if brightness or contrast or saturation:
        auglist.append(DetBorrowAug(ColorJitterAug(brightness, contrast, saturation)))

    if hue:
        auglist.append(DetBorrowAug(HueJitterAug(hue)))

    if pca_noise > 0:
        eigval = np.array([55.46, 4.794, 1.148])
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])
        auglist.append(DetBorrowAug(LightingAug(pca_noise, eigval, eigvec)))

    if rand_gray > 0:
        auglist.append(DetBorrowAug(RandomGrayAug(rand_gray)))

    if mean is True:
        mean = np.array([123.68, 116.28, 103.53])
    elif mean is not None:
        assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]

    if std is True:
        std = np.array([58.395, 57.12, 57.375])
    elif std is not None:
        assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]

    if mean is not None or std is not None:
        auglist.append(DetBorrowAug(ColorNormalizeAug(mean, std)))

    return auglist


class ImageDetIter(ImageIter):
    """Image iterator with a large number of augmentation choices for detection.

    Parameters
    ----------
    aug_list : list or None
        Augmenter list for generating distorted images
    batch_size : int
        Number of examples per batch.
    data_shape : tuple
        Data shape in (channels, height, width) format.
        For now, only RGB image with 3 channels is supported.
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
        Name for detection labels
    last_batch_handle : str, optional
        How to handle the last batch.
        This parameter can be 'pad'(default), 'discard' or 'roll_over'.
        If 'pad', the last batch will be padded with data starting from the begining
        If 'discard', the last batch will be discarded
        If 'roll_over', the remaining elements will be rolled over to the next iteration
    kwargs : ...
        More arguments for creating augmenter. See mx.image.CreateDetAugmenter.
    """
    def __init__(self, batch_size, data_shape,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None,
                 data_name='data', label_name='label', last_batch_handle='pad', **kwargs):
        super(ImageDetIter, self).__init__(batch_size=batch_size, data_shape=data_shape,
                                           path_imgrec=path_imgrec, path_imglist=path_imglist,
                                           path_root=path_root, path_imgidx=path_imgidx,
                                           shuffle=shuffle, part_index=part_index,
                                           num_parts=num_parts, aug_list=[], imglist=imglist,
                                           data_name=data_name, label_name=label_name,
                                           last_batch_handle=last_batch_handle)

        if aug_list is None:
            self.auglist = CreateDetAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list

        # went through all labels to get the proper label shape
        label_shape = self._estimate_label_shape()
        self.provide_label = [(label_name, (self.batch_size, label_shape[0], label_shape[1]))]
        self.label_shape = label_shape

    def _check_valid_label(self, label):
        """Validate label and its shape."""
        if len(label.shape) != 2 or label.shape[1] < 5:
            msg = "Label with shape (1+, 5+) required, %s received." % str(label)
            raise RuntimeError(msg)
        valid_label = np.where(np.logical_and(label[:, 0] >= 0, label[:, 3] > label[:, 1],
                                              label[:, 4] > label[:, 2]))[0]
        if valid_label.size < 1:
            raise RuntimeError('Invalid label occurs.')

    def _estimate_label_shape(self):
        """Helper function to estimate label shape"""
        max_count, label = 0, None
        self.reset()
        try:
            while True:
                label, _ = self.next_sample()
                label = self._parse_label(label)
                max_count = max(max_count, label.shape[0])
        except StopIteration:
            pass
        self.reset()
        return (max_count, label.shape[1] if label is not None else 5)

    def _parse_label(self, label):
        """Helper function to parse object detection label.

        Format for raw label:
        n \t k \t ... \t [id \t xmin\t ymin \t xmax \t ymax \t ...] \t [repeat]
        where n is the width of header, 2 or larger
        k is the width of each object annotation, can be arbitrary, at least 5
        """
        if isinstance(label, nd.NDArray):
            label = label.asnumpy()
        raw = label.ravel()
        if raw.size < 7:
            raise RuntimeError("Label shape is invalid: " + str(raw.shape))
        header_width = int(raw[0])
        obj_width = int(raw[1])
        if (raw.size - header_width) % obj_width != 0:
            msg = "Label shape %s inconsistent with annotation width %d." \
                %(str(raw.shape), obj_width)
            raise RuntimeError(msg)
        out = np.reshape(raw[header_width:], (-1, obj_width))
        # remove bad ground-truths
        valid = np.where(np.logical_and(out[:, 3] > out[:, 1], out[:, 4] > out[:, 2]))[0]
        if valid.size < 1:
            raise RuntimeError('Encounter sample with no valid label.')
        return out[valid, :]

    def reshape(self, data_shape=None, label_shape=None):
        """Reshape iterator for data_shape or label_shape.

        Parameters
        ----------
        data_shape : tuple or None
            Reshape the data_shape to the new shape if not None
        label_shape : tuple or None
            Reshape label shape to new shape if not None
        """
        if data_shape is not None:
            self.check_data_shape(data_shape)
            self.provide_data = [(self.provide_data[0][0], (self.batch_size,) + data_shape)]
            self.data_shape = data_shape
        if label_shape is not None:
            self.check_label_shape(label_shape)
            self.provide_label = [(self.provide_label[0][0], (self.batch_size,) + label_shape)]
            self.label_shape = label_shape

    def _batchify(self, batch_data, batch_label, start=0):
        """Override the helper function for batchifying data"""
        i = start
        batch_size = self.batch_size
        array_fn = _mx_np.array if is_np_array() else nd.array
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = self.imdecode(s)
                try:
                    self.check_valid_image([data])
                    label = self._parse_label(label)
                    data, label = self.augmentation_transform(data, label)
                    self._check_valid_label(label)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                for datum in [data]:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i] = self.postprocess_data(datum)
                    num_object = label.shape[0]
                    batch_label[i][0:num_object] = array_fn(label)
                    if num_object < batch_label[i].shape[0]:
                        batch_label[i][num_object:] = -1
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return i

    def next(self):
        """Override the function for returning next batch."""
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
        else:
            if is_np_array():
                zeros_fn = _mx_np.zeros
                empty_fn = _mx_np.empty
            else:
                zeros_fn = nd.zeros
                empty_fn = nd.empty
            batch_data = zeros_fn((batch_size, c, h, w))
            batch_label = empty_fn(self.provide_label[0][1])
            batch_label[:] = -1
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

    def augmentation_transform(self, data, label):  # pylint: disable=arguments-differ
        """Override Transforms input data with specified augmentations."""
        for aug in self.auglist:
            data, label = aug(data, label)
        return (data, label)

    def check_label_shape(self, label_shape):
        """Checks if the new label shape is valid"""
        if not len(label_shape) == 2:
            raise ValueError('label_shape should have length 2')
        if label_shape[0] < self.label_shape[0]:
            msg = 'Attempts to reduce label count from %d to %d, not allowed.' \
                % (self.label_shape[0], label_shape[0])
            raise ValueError(msg)
        if label_shape[1] != self.provide_label[0][1][2]:
            msg = 'label_shape object width inconsistent: %d vs %d.' \
                % (self.provide_label[0][1][2], label_shape[1])
            raise ValueError(msg)

    def draw_next(self, color=None, thickness=2, mean=None, std=None, clip=True,
                  waitKey=None, window_name='draw_next', id2labels=None):
        """Display next image with bounding boxes drawn.

        Parameters
        ----------
        color : tuple
            Bounding box color in RGB, use None for random color
        thickness : int
            Bounding box border thickness
        mean : True or numpy.ndarray
            Compensate for the mean to have better visual effect
        std : True or numpy.ndarray
            Revert standard deviations
        clip : bool
            If true, clip to [0, 255] for better visual effect
        waitKey : None or int
            Hold the window for waitKey milliseconds if set, skip ploting if None
        window_name : str
            Plot window name if waitKey is set.
        id2labels : dict
            Mapping of labels id to labels name.

        Returns
        -------
            numpy.ndarray

        Examples
        --------
        >>> # use draw_next to get images with bounding boxes drawn
        >>> iterator = mx.image.ImageDetIter(1, (3, 600, 600), path_imgrec='train.rec')
        >>> for image in iterator.draw_next(waitKey=None):
        ...     # display image
        >>> # or let draw_next display using cv2 module
        >>> for image in iterator.draw_next(waitKey=0, window_name='disp'):
        ...     pass
        """
        try:
            import cv2
        except ImportError as e:
            warnings.warn('Unable to import cv2, skip drawing: %s', str(e))
            return
        count = 0
        try:
            while True:
                label, s = self.next_sample()
                data = self.imdecode(s)
                try:
                    self.check_valid_image([data])
                    label = self._parse_label(label)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                count += 1
                data, label = self.augmentation_transform(data, label)
                image = data.asnumpy()

                # revert color_normalize
                if std is True:
                    std = np.array([58.395, 57.12, 57.375])
                elif std is not None:
                    assert isinstance(std, np.ndarray) and std.shape[0] in [1, 3]
                if std is not None:
                    image *= std

                if mean is True:
                    mean = np.array([123.68, 116.28, 103.53])
                elif mean is not None:
                    assert isinstance(mean, np.ndarray) and mean.shape[0] in [1, 3]
                if mean is not None:
                    image += mean

                # swap RGB
                image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]
                if clip:
                    image = np.maximum(0, np.minimum(255, image))
                if color:
                    color = color[::-1]
                image = image.astype(np.uint8)
                height, width, _ = image.shape
                for i in range(label.shape[0]):
                    x1 = int(label[i, 1] * width)
                    if x1 < 0:
                        continue
                    y1 = int(label[i, 2] * height)
                    x2 = int(label[i, 3] * width)
                    y2 = int(label[i, 4] * height)
                    bc = np.random.rand(3) * 255 if not color else color
                    cv2.rectangle(image, (x1, y1), (x2, y2), bc, thickness)
                    if id2labels is not None:
                        cls_id = int(label[i, 0])
                        if cls_id in id2labels:
                            cls_name = id2labels[cls_id]
                            text = "{:s}".format(cls_name)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            text_height = cv2.getTextSize(text, font, font_scale, 2)[0][1]
                            tc = (255, 255, 255)
                            tpos = (x1 + 5, y1 + text_height + 5)
                            cv2.putText(image, text, tpos, font, font_scale, tc, 2)
                if waitKey is not None:
                    cv2.imshow(window_name, image)
                    cv2.waitKey(waitKey)
                yield image
        except StopIteration:
            if not count:
                return

    def sync_label_shape(self, it, verbose=False):
        """Synchronize label shape with the input iterator. This is useful when
        train/validation iterators have different label padding.

        Parameters
        ----------
        it : ImageDetIter
            The other iterator to synchronize
        verbose : bool
            Print verbose log if true

        Returns
        -------
        ImageDetIter
            The synchronized other iterator, the internal label shape is updated as well.

        Examples
        --------
        >>> train_iter = mx.image.ImageDetIter(32, (3, 300, 300), path_imgrec='train.rec')
        >>> val_iter = mx.image.ImageDetIter(32, (3, 300, 300), path.imgrec='val.rec')
        >>> train_iter.label_shape
        (30, 6)
        >>> val_iter.label_shape
        (25, 6)
        >>> val_iter = train_iter.sync_label_shape(val_iter, verbose=False)
        >>> train_iter.label_shape
        (30, 6)
        >>> val_iter.label_shape
        (30, 6)
        """
        assert isinstance(it, ImageDetIter), 'Synchronize with invalid iterator.'
        train_label_shape = self.label_shape
        val_label_shape = it.label_shape
        assert train_label_shape[1] == val_label_shape[1], "object width mismatch."
        max_count = max(train_label_shape[0], val_label_shape[0])
        if max_count > train_label_shape[0]:
            self.reshape(None, (max_count, train_label_shape[1]))
        if max_count > val_label_shape[0]:
            it.reshape(None, (max_count, val_label_shape[1]))
        if verbose and max_count > min(train_label_shape[0], val_label_shape[0]):
            logging.info('Resized label_shape to (%d, %d).', max_count, train_label_shape[1])
        return it
