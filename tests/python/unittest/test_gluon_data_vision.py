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
from __future__ import print_function
from collections import namedtuple

import mxnet as mx
import mxnet.ndarray as nd
from mxnet.base import MXNetError
from mxnet import gluon
from mxnet.base import MXNetError
from mxnet.gluon.data.vision import transforms
from mxnet import image
from mxnet.test_utils import *
from common import assertRaises, setup_module, with_seed, teardown_module, \
    xfail_when_nonstandard_decimal_separator

import numpy as np

mx.reset_np()

@with_seed()
def test_to_tensor():
    # 3D Input
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(out_nd.asnumpy(), np.transpose(
                        data_in.astype(dtype=np.float32) / 255.0, (2, 0, 1)))

    # 4D Input
    data_in = np.random.uniform(0, 255, (5, 300, 300, 3)).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(out_nd.asnumpy(), np.transpose(
                        data_in.astype(dtype=np.float32) / 255.0, (0, 3, 1, 2)))

    # Invalid Input
    invalid_data_in = nd.random.uniform(0, 255, (5, 5, 300, 300, 3)).astype(dtype=np.uint8)
    transformer = transforms.ToTensor()
    assertRaises(MXNetError, transformer, invalid_data_in)

    # Bounds (0->0, 255->1)
    data_in = np.zeros((10, 20, 3)).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert same(out_nd.asnumpy(), np.transpose(np.zeros(data_in.shape, dtype=np.float32), (2, 0, 1)))

    data_in = np.full((10, 20, 3), 255).astype(dtype=np.uint8)
    out_nd = transforms.ToTensor()(nd.array(data_in, dtype='uint8'))
    assert same(out_nd.asnumpy(), np.transpose(np.ones(data_in.shape, dtype=np.float32), (2, 0, 1)))


@with_seed()
def test_normalize():
    # 3D Input
    data_in_3d = nd.random.uniform(0, 1, (3, 300, 300))
    out_nd_3d = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))(data_in_3d)
    data_expected_3d = data_in_3d.asnumpy()
    data_expected_3d[:][:][0] = data_expected_3d[:][:][0] / 3.0
    data_expected_3d[:][:][1] = (data_expected_3d[:][:][1] - 1.0) / 2.0
    data_expected_3d[:][:][2] = data_expected_3d[:][:][2] - 2.0
    assert_almost_equal(data_expected_3d, out_nd_3d.asnumpy())

    # 4D Input
    data_in_4d = nd.random.uniform(0, 1, (2, 3, 300, 300))
    out_nd_4d = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))(data_in_4d)
    data_expected_4d = data_in_4d.asnumpy()
    data_expected_4d[0][:][:][0] = data_expected_4d[0][:][:][0] / 3.0
    data_expected_4d[0][:][:][1] = (data_expected_4d[0][:][:][1] - 1.0) / 2.0
    data_expected_4d[0][:][:][2] = data_expected_4d[0][:][:][2] - 2.0
    data_expected_4d[1][:][:][0] = data_expected_4d[1][:][:][0] / 3.0
    data_expected_4d[1][:][:][1] = (data_expected_4d[1][:][:][1] - 1.0) / 2.0
    data_expected_4d[1][:][:][2] = data_expected_4d[1][:][:][2] - 2.0
    assert_almost_equal(data_expected_4d, out_nd_4d.asnumpy())

    # Invalid Input - Neither 3D or 4D input
    invalid_data_in = nd.random.uniform(0, 1, (5, 5, 3, 300, 300))
    normalize_transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    assertRaises(MXNetError, normalize_transformer, invalid_data_in)

    # Invalid Input - Channel neither 1 or 3
    invalid_data_in = nd.random.uniform(0, 1, (5, 4, 300, 300))
    normalize_transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    assertRaises(MXNetError, normalize_transformer, invalid_data_in)


@with_seed()
def test_resize():
    def _test_resize_with_diff_type(dtype):
        # test normal case
        data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
        out_nd = transforms.Resize(200)(data_in)
        data_expected = mx.image.imresize(data_in, 200, 200, 1)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test 4D input
        data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
        out_batch_nd = transforms.Resize(200)(data_bath_in)
        for i in range(len(out_batch_nd)):
            assert_almost_equal(mx.image.imresize(data_bath_in[i], 200, 200, 1).asnumpy(),
                out_batch_nd[i].asnumpy())
        # test interp = 2
        out_nd = transforms.Resize(200, interpolation=2)(data_in)
        data_expected = mx.image.imresize(data_in, 200, 200, 2)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test height not equals to width
        out_nd = transforms.Resize((200, 100))(data_in)
        data_expected = mx.image.imresize(data_in, 200, 100, 1)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test keep_ratio
        out_nd = transforms.Resize(150, keep_ratio=True)(data_in)
        data_expected = mx.image.imresize(data_in, 150, 225, 1)
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test size below zero
        invalid_transform = transforms.Resize(-150, keep_ratio=True)
        assertRaises(MXNetError, invalid_transform, data_in)
        # test size more than 2:
        invalid_transform = transforms.Resize((100, 100, 100), keep_ratio=True)
        assertRaises(MXNetError, invalid_transform, data_in)

    for dtype in ['uint8', 'float32', 'float64']:
        _test_resize_with_diff_type(dtype)


@with_seed()
def test_crop_resize():
    def _test_crop_resize_with_diff_type(dtype):
        # test normal case
        data_in = nd.arange(60).reshape((5, 4, 3)).astype(dtype)
        out_nd = transforms.CropResize(0, 0, 3, 2)(data_in)
        out_np = out_nd.asnumpy()
        assert(out_np.sum() == 180)
        assert((out_np[0:2,1,1].flatten() == [4, 16]).all())
        # test 4D input
        data_bath_in = nd.arange(180).reshape((2, 6, 5, 3)).astype(dtype)
        out_batch_nd = transforms.CropResize(1, 2, 3, 4)(data_bath_in)
        out_batch_np = out_batch_nd.asnumpy()
        assert(out_batch_np.sum() == 7524)
        assert((out_batch_np[0:2,0:4,1,1].flatten() == [37,  52,  67,  82, 127, 142, 157, 172]).all())
        # test normal case with resize
        data_in = nd.random.uniform(0, 255, (300, 200, 3)).astype(dtype)
        out_nd = transforms.CropResize(0, 0, 100, 50, (25, 25), 1)(data_in)
        data_expected = transforms.Resize(size=25, interpolation=1)(nd.slice(data_in, (0, 0, 0), (50, 100, 3)))
        assert_almost_equal(out_nd.asnumpy(), data_expected.asnumpy())
        # test 4D input with resize
        data_bath_in = nd.random.uniform(0, 255, (3, 300, 200, 3)).astype(dtype)
        out_batch_nd = transforms.CropResize(0, 0, 100, 50, (25, 25), 1)(data_bath_in)
        for i in range(len(out_batch_nd)):
            actual = transforms.Resize(size=25, interpolation=1)(nd.slice(data_bath_in[i], (0, 0, 0), (50, 100, 3))).asnumpy()
            expected = out_batch_nd[i].asnumpy()
            assert_almost_equal(expected, actual)
        # test with resize height and width should be greater than 0
        transformer = transforms.CropResize(0, 0, 100, 50, (-25, 25), 1)
        assertRaises(MXNetError, transformer, data_in)
        # test height and width should be greater than 0
        transformer = transforms.CropResize(0, 0, -100, -50)
        assertRaises(MXNetError, transformer, data_in)
        # test cropped area is bigger than input data
        transformer = transforms.CropResize(150, 200, 200, 500)
        assertRaises(MXNetError, transformer, data_in)
        assertRaises(MXNetError, transformer, data_bath_in)

    for dtype in ['uint8', 'float32', 'float64']:
        _test_crop_resize_with_diff_type(dtype)

    # test nd.image.crop backward
    def test_crop_backward(test_nd_arr, TestCase):
        a_np = test_nd_arr.asnumpy()
        b_np = a_np[(slice(TestCase.y, TestCase.y + TestCase.height), slice(TestCase.x, TestCase.x + TestCase.width), slice(0, 3))]

        data = mx.sym.Variable('data')
        crop_sym = mx.sym.image.crop(data, TestCase.x, TestCase.y, TestCase.width, TestCase.height)

        expected_in_grad = np.zeros_like(a_np)
        expected_in_grad[(slice(TestCase.y, TestCase.y + TestCase.height), slice(TestCase.x, TestCase.x + TestCase.width), slice(0, 3))] = b_np
        check_symbolic_backward(crop_sym, [a_np], [b_np], [expected_in_grad])

    TestCase = namedtuple('TestCase', ['x', 'y', 'width', 'height'])
    test_list = [TestCase(0, 0, 3, 3), TestCase(2, 1, 1, 2), TestCase(0, 1, 3, 2)]

    for dtype in ['uint8', 'float32', 'float64']:
        data_in = nd.arange(60).reshape((5, 4, 3)).astype(dtype)
        for test_case in test_list:
            test_crop_backward(data_in, test_case)


@with_seed()
def test_flip_left_right():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    flip_in = data_in[:, ::-1, :]
    data_trans = nd.image.flip_left_right(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(flip_in, data_trans.asnumpy())


@with_seed()
def test_flip_top_bottom():
    data_in = np.random.uniform(0, 255, (300, 300, 3)).astype(dtype=np.uint8)
    flip_in = data_in[::-1, :, :]
    data_trans = nd.image.flip_top_bottom(nd.array(data_in, dtype='uint8'))
    assert_almost_equal(flip_in, data_trans.asnumpy())


@with_seed()
def test_transformer():
    from mxnet.gluon.data.vision import transforms

    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.Resize(300, keep_ratio=True),
        transforms.CenterCrop(256),
        transforms.RandomCrop(256, pad=16),
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomBrightness(0.1),
        transforms.RandomContrast(0.1),
        transforms.RandomSaturation(0.1),
        transforms.RandomHue(0.1),
        transforms.RandomLighting(0.1),
        transforms.ToTensor(),
        transforms.RandomRotation([-10., 10.]),
        transforms.Normalize([0, 0, 0], [1, 1, 1])])

    transform(mx.nd.ones((245, 480, 3), dtype='uint8')).wait_to_read()

@with_seed()
def test_random_crop():
    x = mx.nd.ones((245, 480, 3), dtype='uint8')
    y = mx.nd.image.random_crop(x, width=100, height=100)
    assert y.shape == (100, 100, 3)

@with_seed()
def test_random_resize_crop():
    x = mx.nd.ones((245, 480, 3), dtype='uint8')
    y = mx.nd.image.random_resized_crop(x, width=100, height=100)
    assert y.shape == (100, 100, 3)

@with_seed()
def test_hybrid_transformer():
    from mxnet.gluon.data.vision import transforms

    transform = transforms.HybridCompose([
        transforms.Resize(300),
        transforms.Resize(300, keep_ratio=True),
        transforms.CenterCrop(256),
        transforms.RandomCrop(256, pad=16),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.RandomBrightness(0.1),
        transforms.RandomContrast(0.1),
        transforms.RandomSaturation(0.1),
        transforms.RandomHue(0.1),
        transforms.RandomLighting(0.1),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])])

    transform(mx.nd.ones((245, 480, 3), dtype='uint8')).wait_to_read()

@with_seed()
def test_rotate():
    transformer = transforms.Rotate(10.)
    assertRaises(TypeError, transformer, mx.nd.ones((3, 30, 60), dtype='uint8'))
    single_image = mx.nd.ones((3, 30, 60), dtype='float32')
    single_output = transformer(single_image)
    assert same(single_output.shape, (3, 30, 60))
    batch_image = mx.nd.ones((3, 3, 30, 60), dtype='float32')
    batch_output = transformer(batch_image)
    assert same(batch_output.shape, (3, 3, 30, 60))

    input_image = nd.array([[[0., 0., 0.],
                             [0., 0., 1.],
                             [0., 0., 0.]]])
    rotation_angles_expected_outs = [
        (90., nd.array([[[0., 1., 0.],
                         [0., 0., 0.],
                         [0., 0., 0.]]])),
        (180., nd.array([[[0., 0., 0.],
                          [1., 0., 0.],
                          [0., 0., 0.]]])),
        (270., nd.array([[[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 1., 0.]]])),
        (360., nd.array([[[0., 0., 0.],
                          [0., 0., 1.],
                          [0., 0., 0.]]])),
    ]
    for rot_angle, expected_result in rotation_angles_expected_outs:
        transformer = transforms.Rotate(rot_angle)
        ans = transformer(input_image)
        print(ans, expected_result)
        assert_almost_equal(ans, expected_result, atol=1e-6)


@with_seed()
def test_random_rotation():
    # test exceptions for probability input outside of [0,1]
    assertRaises(ValueError, transforms.RandomRotation, [-10, 10.], rotate_with_proba=1.1)
    assertRaises(ValueError, transforms.RandomRotation, [-10, 10.], rotate_with_proba=-0.3)
    # test `forward`
    transformer = transforms.RandomRotation([-10, 10.])
    assertRaises(TypeError, transformer, mx.nd.ones((3, 30, 60), dtype='uint8'))
    single_image = mx.nd.ones((3, 30, 60), dtype='float32')
    single_output = transformer(single_image)
    assert same(single_output.shape, (3, 30, 60))
    batch_image = mx.nd.ones((3, 3, 30, 60), dtype='float32')
    batch_output = transformer(batch_image)
    assert same(batch_output.shape, (3, 3, 30, 60))
    # test identity (rotate_with_proba = 0)
    transformer = transforms.RandomRotation([-100., 100.], rotate_with_proba=0.0)
    data = mx.nd.random_normal(shape=(3, 30, 60))
    assert_almost_equal(data, transformer(data))


@xfail_when_nonstandard_decimal_separator
@with_seed()
def test_rotate():
    transformer = transforms.Rotate(10.)
    assertRaises(TypeError, transformer, mx.nd.ones((3, 30, 60), dtype='uint8'))
    single_image = mx.nd.ones((3, 30, 60), dtype='float32')
    single_output = transformer(single_image)
    assert same(single_output.shape, (3, 30, 60))
    batch_image = mx.nd.ones((3, 3, 30, 60), dtype='float32')
    batch_output = transformer(batch_image)
    assert same(batch_output.shape, (3, 3, 30, 60))

    input_image = nd.array([[[0., 0., 0.],
                             [0., 0., 1.],
                             [0., 0., 0.]]])
    rotation_angles_expected_outs = [
        (90., nd.array([[[0., 1., 0.],
                         [0., 0., 0.],
                         [0., 0., 0.]]])),
        (180., nd.array([[[0., 0., 0.],
                          [1., 0., 0.],
                          [0., 0., 0.]]])),
        (270., nd.array([[[0., 0., 0.],
                          [0., 0., 0.],
                          [0., 1., 0.]]])),
        (360., nd.array([[[0., 0., 0.],
                          [0., 0., 1.],
                          [0., 0., 0.]]])),
    ]
    for rot_angle, expected_result in rotation_angles_expected_outs:
        transformer = transforms.Rotate(rot_angle)
        ans = transformer(input_image)
        print(ans, expected_result)
        assert_almost_equal(ans, expected_result, atol=1e-6)


@with_seed()
def test_random_rotation():
    # test exceptions for probability input outside of [0,1]
    assertRaises(ValueError, transforms.RandomRotation, [-10, 10.], rotate_with_proba=1.1)
    assertRaises(ValueError, transforms.RandomRotation, [-10, 10.], rotate_with_proba=-0.3)
    # test `forward`
    transformer = transforms.RandomRotation([-10, 10.])
    assertRaises(TypeError, transformer, mx.nd.ones((3, 30, 60), dtype='uint8'))
    single_image = mx.nd.ones((3, 30, 60), dtype='float32')
    single_output = transformer(single_image)
    assert same(single_output.shape, (3, 30, 60))
    batch_image = mx.nd.ones((3, 3, 30, 60), dtype='float32')
    batch_output = transformer(batch_image)
    assert same(batch_output.shape, (3, 3, 30, 60))
    # test identity (rotate_with_proba = 0)
    transformer = transforms.RandomRotation([-100., 100.], rotate_with_proba=0.0)
    data = mx.nd.random_normal(shape=(3, 30, 60))
    assert_almost_equal(data, transformer(data))


@with_seed()
def test_random_transforms():
    from mxnet.gluon.data.vision import transforms

    tmp_t = transforms.Compose([transforms.Resize(300), transforms.RandomResizedCrop(224)])
    transform = transforms.Compose([transforms.RandomApply(tmp_t, 0.5)])

    img = mx.nd.ones((10, 10, 3), dtype='uint8')
    iteration = 1000
    num_apply = 0
    for _ in range(iteration):
        out = transform(img)
        if out.shape[0] == 224:
            num_apply += 1
    assert_almost_equal(num_apply/float(iteration), 0.5, 0.1)

@xfail_when_nonstandard_decimal_separator
@with_seed()
def test_random_gray():
    from mxnet.gluon.data.vision import transforms

    transform = transforms.RandomGray(0.5)
    img = mx.nd.ones((4, 4, 3), dtype='uint8')
    pixel = img[0, 0, 0].asnumpy()
    iteration = 1000
    num_apply = 0
    for _ in range(iteration):
        out = transform(img)
        if out[0][0][0].asnumpy() != pixel:
            num_apply += 1
    assert_almost_equal(num_apply/float(iteration), 0.5, 0.1)

    transform = transforms.RandomGray(0.5)
    transform.hybridize()
    img = mx.nd.ones((4, 4, 3), dtype='uint8')
    pixel = img[0, 0, 0].asnumpy()
    iteration = 1000
    num_apply = 0
    for _ in range(iteration):
        out = transform(img)
        if out[0][0][0].asnumpy() != pixel:
            num_apply += 1
    assert_almost_equal(num_apply/float(iteration), 0.5, 0.1)

@with_seed()
def test_bbox_random_flip():
    from mxnet.gluon.contrib.data.vision.transforms.bbox import ImageBboxRandomFlipLeftRight

    transform = ImageBboxRandomFlipLeftRight(0.5)
    iteration = 200
    num_apply = 0
    for _ in range(iteration):
        img = mx.nd.ones((10, 10, 3), dtype='uint8')
        img[0, 0, 0] = 10
        bbox = mx.nd.array([[1, 2, 3, 4, 0]])
        im_out, im_bbox = transform(img, bbox)
        if im_bbox[0][0].asnumpy() != 1 and im_out[0, 0, 0].asnumpy() != 10:
            num_apply += 1
    assert_almost_equal(np.array([num_apply])/float(iteration), 0.5, 0.5)

@with_seed()
def test_bbox_crop():
    from mxnet.gluon.contrib.data.vision.transforms.bbox import ImageBboxCrop

    transform = ImageBboxCrop((0, 0, 3, 3))
    img = mx.nd.ones((10, 10, 3), dtype='uint8')
    bbox = mx.nd.array([[0, 1, 3, 4, 0]])
    im_out, im_bbox = transform(img, bbox)
    assert im_out.shape == (3, 3, 3)
    assert im_bbox[0][2] == 3

