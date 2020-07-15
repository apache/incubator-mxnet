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

import os
import mxnet as mx
import numpy as np
import scipy.ndimage
from mxnet.test_utils import *
from common import assertRaises, with_seed, xfail_when_nonstandard_decimal_separator
import shutil
import tempfile
import unittest
import pytest

mx.reset_np()

def _get_data(url, dirname):
    import os, tarfile
    download(url, dirname=dirname, overwrite=False)
    fname = os.path.join(dirname, url.split('/')[-1])
    tar = tarfile.open(fname)
    source_images = [os.path.join(dirname, x.name) for x in tar.getmembers() if x.isfile()]
    if len(source_images) < 1 or not os.path.isfile(source_images[0]):
        # skip extracting if exists
        tar.extractall(path=dirname)
    tar.close()
    return source_images

def _generate_objects():
    num = np.random.randint(1, 10)
    xy = np.random.rand(num, 2)
    wh = np.random.rand(num, 2) / 2
    left = (xy[:, 0] - wh[:, 0])[:, np.newaxis]
    right = (xy[:, 0] + wh[:, 0])[:, np.newaxis]
    top = (xy[:, 1] - wh[:, 1])[:, np.newaxis]
    bot = (xy[:, 1] + wh[:, 1])[:, np.newaxis]
    boxes = np.maximum(0., np.minimum(1., np.hstack((left, top, right, bot))))
    cid = np.random.randint(0, 20, size=num)
    label = np.hstack((cid[:, np.newaxis], boxes)).ravel().tolist()
    return [2, 5] + label

def _test_imageiter_last_batch(imageiter_list, assert_data_shape):
    test_iter = imageiter_list[0]
    # test batch data shape
    for _ in range(3):
        for batch in test_iter:
            assert batch.data[0].shape == assert_data_shape
        test_iter.reset()
    # test last batch handle(discard)
    test_iter = imageiter_list[1]
    i = 0
    for batch in test_iter:
        i += 1
    assert i == 5
    # test last_batch_handle(pad)
    test_iter = imageiter_list[2]
    i = 0
    for batch in test_iter:
        if i == 0:
            first_three_data = batch.data[0][:2]
        if i == 5:
            last_three_data = batch.data[0][1:]
        i += 1
    assert i == 6
    assert np.array_equal(first_three_data.asnumpy(), last_three_data.asnumpy())
    # test last_batch_handle(roll_over)
    test_iter = imageiter_list[3]
    i = 0
    for batch in test_iter:
        if i == 0:
            first_image = batch.data[0][0]
        i += 1
    assert i == 5
    test_iter.reset()
    first_batch_roll_over = test_iter.next()
    assert np.array_equal(
        first_batch_roll_over.data[0][1].asnumpy(), first_image.asnumpy())
    assert first_batch_roll_over.pad == 2
    # test iteratopr work properly after calling reset several times when last_batch_handle is roll_over
    for _ in test_iter:
        pass
    test_iter.reset()
    first_batch_roll_over_twice = test_iter.next()
    assert np.array_equal(
        first_batch_roll_over_twice.data[0][2].asnumpy(), first_image.asnumpy())
    assert first_batch_roll_over_twice.pad == 1
    # we've called next once
    i = 1
    for _ in test_iter:
        i += 1
    # test the third epoch with size 6
    assert i == 6
    # test shuffle option for sanity test
    test_iter = imageiter_list[4]
    for _ in test_iter:
        pass


class TestImage(unittest.TestCase):
    IMAGES_URL = "http://data.mxnet.io/data/test_images.tar.gz"

    def setUp(self):
        self.IMAGES_DIR = tempfile.mkdtemp()
        self.IMAGES = _get_data(self.IMAGES_URL, self.IMAGES_DIR)
        print("Loaded {} images".format(len(self.IMAGES)))

    def tearDown(self):
        if self.IMAGES_DIR:
            print("cleanup {}".format(self.IMAGES_DIR))
            shutil.rmtree(self.IMAGES_DIR)

    def test_imread_not_found(self):
        with pytest.raises(mx.base.MXNetError):
            x = mx.img.image.imread("/139810923jadjsajlskd.___adskj/blah.jpg")

    def test_imread_vs_imdecode(self):
        for img in self.IMAGES:
            with open(img, 'rb') as fp:
                str_image = fp.read()
                image = mx.image.imdecode(str_image, to_rgb=0)
                image_read = mx.img.image.imread(img)
                same(image.asnumpy(), image_read.asnumpy())

    def test_imdecode(self):
        try:
            import cv2
        except ImportError:
            raise unittest.SkipTest("Unable to import cv2.")
        for img in self.IMAGES:
            with open(img, 'rb') as fp:
                str_image = fp.read()
                image = mx.image.imdecode(str_image, to_rgb=0)
            cv_image = cv2.imread(img)
            assert_almost_equal(image.asnumpy(), cv_image)

    def test_imdecode_bytearray(self):
        try:
            import cv2
        except ImportError:
            return
        for img in self.IMAGES:
            with open(img, 'rb') as fp:
                str_image = bytearray(fp.read())
                image = mx.image.imdecode(str_image, to_rgb=0)
            cv_image = cv2.imread(img)
            assert_almost_equal(image.asnumpy(), cv_image)

    def test_imdecode_empty_buffer(self):
        with pytest.raises(mx.base.MXNetError):
            mx.image.imdecode(b'', to_rgb=0)

    def test_imdecode_invalid_image(self):
        with pytest.raises(mx.base.MXNetError):
            image = mx.image.imdecode(b'clearly not image content')

    def test_scale_down(self):
        assert mx.image.scale_down((640, 480), (720, 120)) == (640, 106)
        assert mx.image.scale_down((360, 1000), (480, 500)) == (360, 375)
        assert mx.image.scale_down((300, 400), (0, 0)) == (0, 0)

    @with_seed()
    def test_resize_short(self):
        try:
            import cv2
        except ImportError:
            raise unittest.SkipTest("Unable to import cv2")
        for img in self.IMAGES:
            cv_img = cv2.imread(img)
            mx_img = mx.nd.array(cv_img[:, :, (2, 1, 0)])
            h, w, _ = cv_img.shape
            for _ in range(3):
                new_size = np.random.randint(1, 1000)
                if h > w:
                    new_h, new_w = new_size * h // w, new_size
                else:
                    new_h, new_w = new_size, new_size * w // h
                for interp in range(0, 2):
                    # area-based/lanczos don't match with cv2?
                    cv_resized = cv2.resize(cv_img, (new_w, new_h), interpolation=interp)
                    mx_resized = mx.image.resize_short(mx_img, new_size, interp)
                    assert_almost_equal(mx_resized.asnumpy()[:, :, (2, 1, 0)], cv_resized, atol=3)

    @with_seed()
    def test_imresize(self):
        try:
            import cv2
        except ImportError:
            raise unittest.SkipTest("Unable to import cv2")
        for img in self.IMAGES:
            cv_img = cv2.imread(img)
            mx_img = mx.nd.array(cv_img[:, :, (2, 1, 0)])
            new_h = np.random.randint(1, 1000)
            new_w = np.random.randint(1, 1000)
            for interp_val in range(0, 2):
                cv_resized = cv2.resize(cv_img, (new_w, new_h), interpolation=interp_val)
                mx_resized = mx.image.imresize(mx_img, new_w, new_h, interp=interp_val)
                assert_almost_equal(mx_resized.asnumpy()[:, :, (2, 1, 0)], cv_resized, atol=3)
                out_img = mx.nd.zeros((new_h, new_w, 3), dtype=mx_img.dtype)
                mx.image.imresize(mx_img, new_w, new_h, interp=interp_val, out=out_img)
                assert_almost_equal(out_img.asnumpy()[:, :, (2, 1, 0)], cv_resized, atol=3)

    def test_color_normalize(self):
        for _ in range(10):
            mean = np.random.rand(3) * 255
            std = np.random.rand(3) + 1
            width = np.random.randint(100, 500)
            height = np.random.randint(100, 500)
            src = np.random.rand(height, width, 3) * 255.
            mx_result = mx.image.color_normalize(mx.nd.array(src),
                mx.nd.array(mean), mx.nd.array(std))
            assert_almost_equal(mx_result.asnumpy(), (src - mean) / std, atol=1e-3)

    def test_imageiter(self):
        print(self.IMAGES)
        im_list = [[np.random.randint(0, 5), x] for x in self.IMAGES]
        fname = os.path.join(self.IMAGES_DIR, 'test_imageiter.lst')
        file_list = ['\t'.join([str(k), str(np.random.randint(0, 5)), x])
                        for k, x in enumerate(self.IMAGES)]
        with open(fname, 'w') as f:
            for line in file_list:
                f.write(line + '\n')

        test_list = ['imglist', 'path_imglist']
        for dtype in ['int32', 'float32', 'int64', 'float64']:
            for test in test_list:
                imglist = im_list if test == 'imglist' else None
                path_imglist = fname if test == 'path_imglist' else None
                imageiter_list = [
                    mx.image.ImageIter(2, (3, 224, 224), label_width=1, imglist=imglist,
                        path_imglist=path_imglist, path_root=self.IMAGES_DIR, dtype=dtype),
                    mx.image.ImageIter(3, (3, 224, 224), label_width=1, imglist=imglist,
                        path_imglist=path_imglist, path_root=self.IMAGES_DIR, dtype=dtype, last_batch_handle='discard'),
                    mx.image.ImageIter(3, (3, 224, 224), label_width=1, imglist=imglist,
                        path_imglist=path_imglist, path_root=self.IMAGES_DIR, dtype=dtype, last_batch_handle='pad'),
                    mx.image.ImageIter(3, (3, 224, 224), label_width=1, imglist=imglist,
                        path_imglist=path_imglist, path_root=self.IMAGES_DIR, dtype=dtype, last_batch_handle='roll_over'),
                    mx.image.ImageIter(3, (3, 224, 224), label_width=1, imglist=imglist, shuffle=True,
                        path_imglist=path_imglist, path_root=self.IMAGES_DIR, dtype=dtype, last_batch_handle='pad')
                ]
                _test_imageiter_last_batch(imageiter_list, (2, 3, 224, 224))

    @with_seed()
    def test_copyMakeBorder(self):
        try:
            import cv2
        except ImportError:
            raise unittest.SkipTest("Unable to import cv2")
        for img in self.IMAGES:
            cv_img = cv2.imread(img)
            mx_img = mx.nd.array(cv_img)
            top = np.random.randint(1, 10)
            bot = np.random.randint(1, 10)
            left = np.random.randint(1, 10)
            right = np.random.randint(1, 10)
            new_h, new_w, _ = mx_img.shape
            new_h += top + bot
            new_w += left + right
            val = [np.random.randint(1, 255)] * 3
            for type_val in range(0, 5):
                cv_border = cv2.copyMakeBorder(cv_img, top, bot, left, right, borderType=type_val, value=val)
                mx_border = mx.image.copyMakeBorder(mx_img, top, bot, left, right, type=type_val, values=val)
                assert_almost_equal(mx_border.asnumpy(), cv_border)
                out_img = mx.nd.zeros((new_h , new_w, 3), dtype=mx_img.dtype)
                mx.image.copyMakeBorder(mx_img, top, bot, left, right, type=type_val, values=val, out=out_img)
                assert_almost_equal(out_img.asnumpy(), cv_border)

    @with_seed()
    def test_augmenters(self):
        # ColorNormalizeAug
        mean = np.random.rand(3) * 255
        std = np.random.rand(3) + 1
        width = np.random.randint(100, 500)
        height = np.random.randint(100, 500)
        src = np.random.rand(height, width, 3) * 255.
        # We test numpy and mxnet NDArray inputs
        color_norm_aug = mx.image.ColorNormalizeAug(mean=mx.nd.array(mean), std=std)
        out_image = color_norm_aug(mx.nd.array(src))
        assert_almost_equal(out_image.asnumpy(), (src - mean) / std, atol=1e-3)

        # only test if all augmenters will work
        # TODO(Joshua Zhang): verify the augmenter outputs
        im_list = [[0, x] for x in self.IMAGES]
        test_iter = mx.image.ImageIter(2, (3, 224, 224), label_width=1, imglist=im_list,
            resize=640, rand_crop=True, rand_resize=True, rand_mirror=True, mean=True,
            std=np.array([1.1, 1.03, 1.05]), brightness=0.1, contrast=0.1, saturation=0.1,
            hue=0.1, pca_noise=0.1, rand_gray=0.2, inter_method=10, path_root=self.IMAGES_DIR, shuffle=True)
        for batch in test_iter:
            pass

    def test_image_detiter(self):
        im_list = [_generate_objects() + [x] for x in self.IMAGES]
        det_iter = mx.image.ImageDetIter(2, (3, 300, 300), imglist=im_list, path_root=self.IMAGES_DIR)
        for _ in range(3):
            for _ in det_iter:
                pass
        det_iter.reset()
        val_iter = mx.image.ImageDetIter(2, (3, 300, 300), imglist=im_list, path_root=self.IMAGES_DIR)
        det_iter = val_iter.sync_label_shape(det_iter)
        assert det_iter.data_shape == val_iter.data_shape
        assert det_iter.label_shape == val_iter.label_shape

        # test batch_size is not divisible by number of images
        det_iter = mx.image.ImageDetIter(4, (3, 300, 300), imglist=im_list, path_root=self.IMAGES_DIR)
        for _ in det_iter:
            pass

        # test file list with last batch handle
        fname = os.path.join(self.IMAGES_DIR, 'test_imagedetiter.lst')
        im_list = [[k] + _generate_objects() + [x] for k, x in enumerate(self.IMAGES)]
        with open(fname, 'w') as f:
            for line in im_list:
                line = '\t'.join([str(k) for k in line])
                f.write(line + '\n')

        imageiter_list = [
            mx.image.ImageDetIter(2, (3, 400, 400),
                path_imglist=fname, path_root=self.IMAGES_DIR),
            mx.image.ImageDetIter(3, (3, 400, 400),
                path_imglist=fname, path_root=self.IMAGES_DIR, last_batch_handle='discard'),
            mx.image.ImageDetIter(3, (3, 400, 400),
                path_imglist=fname, path_root=self.IMAGES_DIR, last_batch_handle='pad'),
            mx.image.ImageDetIter(3, (3, 400, 400),
                path_imglist=fname, path_root=self.IMAGES_DIR, last_batch_handle='roll_over'),
            mx.image.ImageDetIter(3, (3, 400, 400), shuffle=True,
                path_imglist=fname, path_root=self.IMAGES_DIR, last_batch_handle='pad')
        ]
        _test_imageiter_last_batch(imageiter_list, (2, 3, 400, 400))

    def test_det_augmenters(self):
        # only test if all augmenters will work
        # TODO(Joshua Zhang): verify the augmenter outputs
        im_list = [_generate_objects() + [x] for x in self.IMAGES]
        det_iter = mx.image.ImageDetIter(2, (3, 300, 300), imglist=im_list, path_root=self.IMAGES_DIR,
            resize=640, rand_crop=1, rand_pad=1, rand_gray=0.1, rand_mirror=True, mean=True,
            std=np.array([1.1, 1.03, 1.05]), brightness=0.1, contrast=0.1, saturation=0.1,
            pca_noise=0.1, hue=0.1, inter_method=10, min_object_covered=0.5,
            aspect_ratio_range=(0.2, 5), area_range=(0.1, 4.0), min_eject_coverage=0.5,
            max_attempts=50)
        for batch in det_iter:
            pass

    @with_seed()
    def test_random_size_crop(self):
        # test aspect ratio within bounds
        width = np.random.randint(100, 500)
        height = np.random.randint(100, 500)
        src = np.random.rand(height, width, 3) * 255.
        ratio = (0.75, 1)
        epsilon = 0.05
        out, (x0, y0, new_w, new_h) = mx.image.random_size_crop(mx.nd.array(src), size=(width, height), area=0.08, ratio=ratio)
        _, pts = mx.image.center_crop(mx.nd.array(src), size=(width, height))
        if (x0, y0, new_w, new_h) != pts:
            assert ratio[0] - epsilon <= float(new_w)/new_h <= ratio[1] + epsilon, \
            'ration of new width and height out of the bound{}/{}={}'.format(new_w, new_h, float(new_w)/new_h)

    @xfail_when_nonstandard_decimal_separator
    @with_seed()
    def test_imrotate(self):
        # test correctness
        xlin = np.expand_dims(np.linspace(0, 0.5, 30), axis=1)
        ylin = np.expand_dims(np.linspace(0, 0.5, 60), axis=0)
        np_img = np.expand_dims(xlin + ylin, axis=2)
        # rotate with imrotate
        nd_img = mx.nd.array(np_img.transpose((2, 0, 1)))  # convert to CHW
        rot_angle = 6
        args = {'src': nd_img, 'rotation_degrees': rot_angle, 'zoom_in': False, 'zoom_out': False}
        nd_rot = mx.image.imrotate(**args)
        npnd_rot = nd_rot.asnumpy().transpose((1, 2, 0))
        # rotate with scipy
        scipy_rot = scipy.ndimage.rotate(np_img, rot_angle, axes=(1, 0), reshape=False,
                                         order=1, mode='constant', prefilter=False)
        # cannot compare the edges (where image ends) because of different behavior
        assert_almost_equal(scipy_rot[10:20, 20:40, :], npnd_rot[10:20, 20:40, :])

        # test if execution raises exceptions in any allowed mode
        # batch mode
        img_in = mx.nd.random.uniform(0, 1, (5, 3, 30, 60), dtype=np.float32)
        nd_rots = mx.nd.array([1, 2, 3, 4, 5], dtype=np.float32)
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': False, 'zoom_out': False}
        _ = mx.image.imrotate(**args)
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': False, 'zoom_out': True}
        _ = mx.image.imrotate(**args)
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': True, 'zoom_out': False}
        _ = mx.image.imrotate(**args)
        # single image mode
        nd_rots = 11
        img_in = mx.nd.random.uniform(0, 1, (3, 30, 60), dtype=np.float32)
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': False, 'zoom_out': False}
        _ = mx.image.imrotate(**args)
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': False, 'zoom_out': True}
        _ = mx.image.imrotate(**args)
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': True, 'zoom_out': False}
        _ = mx.image.imrotate(**args)

        # test if exceptions are correctly raised
        # batch exception - zoom_in=zoom_out=True
        img_in = mx.nd.random.uniform(0, 1, (5, 3, 30, 60), dtype=np.float32)
        nd_rots = mx.nd.array([1, 2, 3, 4, 5], dtype=np.float32)
        args={'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': True, 'zoom_out': True}
        self.assertRaises(ValueError, mx.image.imrotate, **args)

        # single image exception - zoom_in=zoom_out=True
        img_in = mx.nd.random.uniform(0, 1, (3, 30, 60), dtype=np.float32)
        nd_rots = 11
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': True, 'zoom_out': True}
        self.assertRaises(ValueError, mx.image.imrotate, **args)

        # batch of images with scalar rotation
        img_in = mx.nd.stack(nd_img, nd_img, nd_img)
        nd_rots = 6
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': False, 'zoom_out': False}
        out = mx.image.imrotate(**args)
        for img in out:
            img = img.asnumpy().transpose((1, 2, 0))
            assert_almost_equal(scipy_rot[10:20, 20:40, :], img[10:20, 20:40, :])

        # single image exception - single image with vector rotation
        img_in = mx.nd.random.uniform(0, 1, (3, 30, 60), dtype=np.float32)
        nd_rots = mx.nd.array([1, 2, 3, 4, 5], dtype=np.float32)
        args = {'src': img_in, 'rotation_degrees': nd_rots, 'zoom_in': False, 'zoom_out': False}
        self.assertRaises(TypeError, mx.image.imrotate, **args)

    @with_seed()
    def test_random_rotate(self):
        angle_limits = [-5., 5.]
        src_single_image = mx.nd.random.uniform(0, 1, (3, 30, 60),
                                                dtype=np.float32)
        out_single_image = mx.image.random_rotate(src_single_image,
                                                  angle_limits)
        self.assertEqual(out_single_image.shape, (3, 30, 60))
        src_batch_image = mx.nd.stack(src_single_image,
                                      src_single_image,
                                      src_single_image)
        out_batch_image = mx.image.random_rotate(src_batch_image,
                                                 angle_limits)
        self.assertEqual(out_batch_image.shape, (3, 3, 30, 60))

