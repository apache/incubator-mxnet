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

# pylint: skip-file
from __future__ import absolute_import
import numpy as _np
import mxnet as mx
from mxnet import np, npx
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.base import MXNetError
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient, use_np, collapse_sum_like
from common import assertRaises, with_seed
import random
import collections


@with_seed()
@use_np
def test_np_tensordot():
    class TestTensordot(HybridBlock):
        def __init__(self, axes):
            super(TestTensordot, self).__init__()
            self._axes = axes
            
        def hybrid_forward(self, F, a, b):
            return F.np.tensordot(a, b, self._axes)

    def tensordot_backward(a, b, axes=2):
        if (a.ndim < 1) or (b.ndim < 1):
            raise ValueError('An input is zero-dim')

        if _np.isscalar(axes):
            a_axes_summed = [i + a.ndim - axes for i in range(axes)]
            b_axes_summed = [i for i in range(axes)]
        else:
            if len(axes) != 2:
                raise ValueError('Axes must consist of two arrays.')
            a_axes_summed, b_axes_summed = axes
            if _np.isscalar(a_axes_summed):
                a_axes_summed = a_axes_summed,
            if _np.isscalar(b_axes_summed):
                b_axes_summed = b_axes_summed,

            for i in range(len(a_axes_summed)):
                a_axes_summed[i] = (a_axes_summed[i] + a.ndim) % a.ndim

            for i in range(len(b_axes_summed)):
                b_axes_summed[i] = (b_axes_summed[i] + b.ndim) % b.ndim

        if len(a_axes_summed) != len(b_axes_summed):
            raise ValueError('Axes length mismatch') 

        a_axes_remained = []
        for i in range(a.ndim):
            if not (i in a_axes_summed):
                a_axes_remained.append(i)
        a_axes = a_axes_remained[:] + a_axes_summed[:]

        b_axes_remained = []
        for i in range(b.ndim):
            if not (i in b_axes_summed):
                b_axes_remained.append(i)
        b_axes = b_axes_summed[:] + b_axes_remained[:]

        ad1 = _np.prod([a.shape[i] for i in a_axes_remained]) if len(a_axes_remained) > 0 else 1
        ad2 = _np.prod([a.shape[i] for i in a_axes_summed]) if len(a_axes_summed) > 0 else 1
        bd1 = _np.prod([b.shape[i] for i in b_axes_summed]) if len(b_axes_summed) > 0 else 1
        bd2 = _np.prod([b.shape[i] for i in b_axes_remained]) if len(b_axes_remained) > 0 else 1

        out_grad = _np.ones((ad1, bd2))

        new_a = _np.transpose(a, a_axes)
        new_a_shape = new_a.shape[:]
        new_a = new_a.reshape((ad1, ad2))
        new_b = _np.transpose(b, b_axes)
        new_b_shape = new_b.shape[:]
        new_b = new_b.reshape((bd1, bd2))

        reverse_a_axes = [0 for i in a_axes]
        for i in range(len(a_axes)):
            reverse_a_axes[a_axes[i]] = i

        reverse_b_axes = [0 for i in b_axes]
        for i in range(len(b_axes)):
            reverse_b_axes[b_axes[i]] = i

        grad_b = _np.dot(new_a.T, out_grad).reshape(new_b_shape)
        grad_b = _np.transpose(grad_b, reverse_b_axes)
        grad_a = _np.dot(out_grad, new_b.T).reshape(new_a_shape)
        grad_a = _np.transpose(grad_a, reverse_a_axes)

        return [grad_a, grad_b]

    # test non zero size input
    tensor_shapes = [
        ((3, 5), (5, 4), 1),  # (a_shape, b_shape, axes)
        ((3,), (3,), 1),
        ((3, 4, 5, 3, 2), (5, 3, 2, 1, 2), 3),
        ((3, 5, 4, 3, 2), (2, 3, 5, 1, 2), [[1, 3, 4], [2, 1, 0]]),
        ((3, 5, 4), (5, 4, 3), [[1, 0, 2], [0, 2, 1]]),
        ((3, 5, 4), (5, 3, 4), [[2, 0], [-1, -2]]),
        ((2, 2), (2, 2), 2),
        ((3, 5, 4), (5, ), [[-2], [0]]),
        ((3, 5, 4), (5, ), [[1], [0]]),
        ((2,), (2, 3), 1),
        ((3,), (3,), 0),
        ((2,), (2, 3), 0),
        ((3, 5, 4), (5, ), 0),
        ((2, 3, 4), (4, 3, 2), [[], []]),
        ((3, 0), (0, 5), 1),
        ((3, 0), (0, 4), [[1], [0]]),
        ((0, 3), (3, 5), 1),
        ((0, 3), (5, 0), [[0], [1]])
    ]

    for hybridize in [True, False]:
        for a_shape, b_shape, axes in tensor_shapes:
            for dtype in [_np.float32, _np.float64]:
                test_tensordot = TestTensordot(axes)
                if hybridize:
                    test_tensordot.hybridize()
                a = rand_ndarray(shape = a_shape, dtype = dtype).as_np_ndarray()
                b = rand_ndarray(shape = b_shape, dtype = dtype).as_np_ndarray()
                a.attach_grad()
                b.attach_grad()

                np_out = _np.tensordot(a.asnumpy(), b.asnumpy(), axes)
                with mx.autograd.record():
                    mx_out = test_tensordot(a, b)
                assert mx_out.shape == np_out.shape
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol = 1e-3, atol = 1e-5)
                mx_out.backward()
                np_backward = tensordot_backward(a.asnumpy(), b.asnumpy(), axes)
                assert_almost_equal(a.grad.asnumpy(), np_backward[0], rtol = 1e-3, atol=1e-5)
                assert_almost_equal(b.grad.asnumpy(), np_backward[1], rtol = 1e-3, atol=1e-5)

                # Test imperative once again
                mx_out = np.tensordot(a, b, axes)
                np_out = _np.tensordot(a.asnumpy(), b.asnumpy(), axes)
                assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)

                # test numeric gradient
                if (_np.prod(a_shape) > 0 and _np.prod(b_shape) > 0):
                    a_sym = mx.sym.Variable("a").as_np_ndarray()
                    b_sym = mx.sym.Variable("b").as_np_ndarray()
                    mx_sym = mx.sym.np.tensordot(a_sym, b_sym, axes).as_nd_ndarray()
                    check_numeric_gradient(mx_sym, [a.as_nd_ndarray(), b.as_nd_ndarray()],
                      rtol=1e-1, atol=1e-1, dtype = dtype)


@with_seed()
@use_np
def test_np_dot():
    shapes = [
        ((3, 0), (0, 4)),
        ((3,), (3,)),        # Case 1
        ((3, 4), (4, 5)),    # Case 2
        ((), ()),            # Case 3
        ((3, 4, 5), ()),     # Case 3.5.1
        ((), (3, 4, 5)),     # Case 3.5.2
        ((3, 4, 5), (5, )),  # Case 4
        ((3, 4, 5), (5, 2)), # Case 5
        ((5,), (5, 2)),
        ((3, 5, 4), (5, 4, 3)),  
        ((3, 4), (5, 4, 3)),
        ((4,), (5, 4, 3))
    ]

    eps = 1e-3

    for shape_a, shape_b in shapes:
        np_a = _np.random.uniform(-1.0, 1.0, shape_a)
        np_a[abs(np_a) < eps] = 2 * eps
        np_b = _np.random.uniform(-1.0, 1.0, shape_b)
        np_b[abs(np_b) < eps] = 2 * eps
        a = mx.nd.array(np_a)
        b = mx.nd.array(np_b)
        np_res = _np.dot(np_a, np_b)
        mx_res = np.dot(a.as_np_ndarray(), b.as_np_ndarray())
        assert mx_res.shape == np_res.shape
        assert_almost_equal(np_res, mx_res.asnumpy(), rtol=1e-5, atol=1e-5)
        mx_a = mx.sym.Variable("a")
        mx_b = mx.sym.Variable("b")
        mx_sym = mx.sym.np.dot(mx_a.as_np_ndarray(), mx_b.as_np_ndarray()).as_nd_ndarray()
        if (len(shape_a) > 0 and len(shape_b) > 0 and _np.prod(shape_a) > 0 and _np.prod(shape_b) > 0):
            check_numeric_gradient(mx_sym, {"a": a, "b": b}, numeric_eps=eps, rtol=1e-2, atol=1e-3)

    bad_shapes = [((4, 5), (2, 3)), ((3, 4, 5), (6, ))]

    for shape_a, shape_b in bad_shapes:
        a = mx.nd.array(random.random()) if len(shape_a) == 0 else rand_ndarray(shape_a)
        b = mx.nd.array(random.random()) if len(shape_b) == 0 else rand_ndarray(shape_b)
        try:
            mx_res = np.dot(a.as_np_ndarray(), b.as_np_ndarray())
        except mx.base.MXNetError:
            continue
        assert False


@with_seed()
@use_np
def test_np_sum():
    class TestSum(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestSum, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.sum(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    def is_int(dtype):
        return 'int' in dtype

    in_data_dim = random.choice([2, 3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    acc_type = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64',
                'int8': 'int32', 'int32': 'int64', 'int64': 'int64'}
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']:
                    for dtype in ['float16', 'float32', 'float64', 'int8', 'int32', 'int64']:
                        if is_int(dtype) and not is_int(itype):
                            continue
                        # test gluon
                        test_sum = TestSum(axis=axis, dtype=dtype, keepdims=keepdims)
                        if hybridize:
                            test_sum.hybridize()
                        if is_int(itype):
                            x = _np.random.randint(-128, 128, shape, dtype=itype)
                            x = mx.nd.array(x)
                        else:
                            x = mx.nd.random.uniform(-1.0, 1.0, shape=shape, dtype=itype)
                        x = x.as_np_ndarray()
                        x.attach_grad()
                        expected_ret = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        with mx.autograd.record():
                            y = test_sum(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                            atol=1e-5 if dtype == 'float16' else 1e-5, use_broadcast=False)

                        y.backward()
                        assert same(x.grad.asnumpy(), _np.ones(shape=x.shape, dtype=x.dtype))

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x").as_np_ndarray()
                            mx_sym = mx.sym.np.sum(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                            check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                                   numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@with_seed()
@use_np
def test_np_linspace():
    configs = [
        (0.0, 1.0, 10),
        (-2, 4, 30),
        (5.234324, 8.98324, 324),
        (2, 10, 100)
    ]
    exception_configs = [
        (0, 10, -1),
        (0, 1, 2.5)
    ]
    dtypes = ['int32', 'float16', 'float32', 'float64', None]
    for config in configs:
        for dtype in dtypes:
            for endpoint in [False, True]:
                for retstep in [False, True]:
                    if isinstance(config, tuple):
                        mx_ret = np.linspace(*config, endpoint=endpoint, retstep=retstep, dtype=dtype)
                        np_ret = _np.linspace(*config, endpoint=endpoint, retstep=retstep, dtype=dtype)
                    else:
                        mx_ret = np.linspace(config, endpoint=endpoint, retstep=retstep, dtype=dtype)
                        np_ret = _np.linspace(config, endpoint=endpoint, retstep=retstep, dtype=dtype)
                    if retstep:
                        assert_almost_equal(mx_ret[0].asnumpy(), np_ret[0], atol=1e-3, rtol=1e-5)
                        same(mx_ret[1], np_ret[1])
                    else:
                        assert_almost_equal(mx_ret.asnumpy(), np_ret, atol=1e-3, rtol=1e-5)
    # check for exception input
    for config in exception_configs:
        assertRaises(MXNetError, np.linspace, *config)
    # check linspace equivalent to arange
    for test_index in range(1000):
        assert_almost_equal(mx.np.linspace(0, test_index, test_index + 1).asnumpy(), _np.arange(test_index + 1))
    @use_np
    class TestLinspace(HybridBlock):
        def __init__(self, start, stop, num=50, endpoint=None, retstep=False, dtype=None, axis=0):
            super(TestLinspace, self).__init__()
            self._start = start
            self._stop = stop
            self._num = num
            self._endpoint = endpoint
            self._retstep = retstep
            self._dtype = dtype

        def hybrid_forward(self, F, x):
            if self._retstep:
                raise ValueError("linspace didn't support retstep = True inside HybridBlock")
            else:
                return x + F.np.linspace(self._start, self._stop, self._num, \
                self._endpoint, self._retstep, self._dtype)

    for dtype in dtypes:
        x = np.zeros(shape=(), dtype=dtype)
        for config in configs:
            for hybridize in [False, True]:
                for endpoint in [False, True]:
                    if isinstance(config, tuple):
                        net = TestLinspace(*config, endpoint=endpoint, dtype=dtype)
                        np_out = _np.linspace(*config, endpoint=endpoint, dtype=dtype)
                    else:
                        net = TestLinspace(config, endpoint=endpoint, dtype=dtype)
                        np_out = _np.linspace(config, endpoint=endpoint, dtype=dtype)
                    if hybridize:
                        net.hybridize()
                    mx_out = net(x)
                    assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-3, rtol=1e-5)


@with_seed()
@use_np
def test_npx_slice():
    class TestSlice(HybridBlock):
        def __init__(self, begin, end, step):
            super(TestSlice, self).__init__()
            self._begin = begin
            self._end = end
            self._step = step

        def hybrid_forward(self, F, a):
            return F.npx.slice(a, begin=self._begin, end=self._end, step=self._step)

    shape = (8, 16, 9, 9)
    np_array = _np.arange(_np.prod(shape), dtype='int32').reshape(shape)
    configs = [
        ([], [], None),
        ([], [], []),
        ([1], [4], None),
        ([1], [10], [3]),
        ([10], [0], [-2]),
        ([None], [None], [None]),
        ([None], [None], [-1]),
        ([10], [None], [-1]),
        ([1, 0, 3], [-2, 10, -4], [None, 2, 3]),
        ([-2, -3, -5, -6], [1, 3, 4, 5], None),
        ([-2, -3, -5, -6], [1, 3, 4, 5], [-1, -2, -3, -4]),
        ([2, -3, -5, -6], [2, 3, 4, 5], None),
        ([2, -3, -5, 5], [3, 3, 4, 5], None),
    ]

    for hybridize in [True, False]:
        for config in configs:
            start, end, step = config[0], config[1], config[2]
            test_slice = TestSlice(begin=start, end=end, step=step)
            if hybridize:
                test_slice.hybridize()

            a = np.array(np_array, dtype=np_array.dtype)
            a.attach_grad()
            basic_index = tuple([
                slice(start[i], end[i], step[i]) if step is not None else slice(start[i], end[i])
                for i in range(len(start))
            ])
            expected_ret = np_array[basic_index]
            with mx.autograd.record():
                y = test_slice(a)

            assert same(y.asnumpy(), expected_ret)

            # test backward
            mx.autograd.backward(y)
            expected_grad = _np.zeros(shape)
            expected_grad[basic_index] = 1
            assert same(a.grad.asnumpy(), expected_grad)


@with_seed()
@use_np
def test_np_reshape():
    class TestReshape(HybridBlock):
        def __init__(self, newshape):
            super(TestReshape, self).__init__()
            self._newshape = newshape

        def hybrid_forward(self, F, a):
            return F.np.reshape(a, self._newshape)

    shape_pairs = [((2, 6), (6, 2)), ((2, 6), (3, 4)), ((1, 0), (0,)), ((0, 0), (0,)), ((), (1, 1, 1))]
    for hybridize in [True, False]:
        for shape_pair in shape_pairs:
            shape1, shape2 = shape_pair
            print(shape1, shape2)
            test_reshape = TestReshape(shape2)
            if hybridize:
                test_reshape.hybridize()
            x = rand_ndarray(shape1).as_np_ndarray()
            x.attach_grad()
            np_out = _np.reshape(x.asnumpy(), shape2)
            with mx.autograd.record():
                mx_out = test_reshape(x)
            assert mx_out.shape == np_out.shape
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)
            mx_out.backward()
            np_backward = _np.ones(shape1)
            assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5, use_broadcast=False)

            mx_out = np.reshape(x, shape2)
            np_out = _np.reshape(x.asnumpy(), shape2)
            assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@with_seed()
@use_np
def test_np_squeeze():
    config = [((), None),
              ((), -1),
              ((), 0),
              ((4, 1, 2), None),
              ((1, 1, 1), None),
              ((1, 0, 1, 5), 2),
              ((1, 0, 1, 1), (-1, -4))]

    class TestSqueeze(HybridBlock):
        def __init__(self, axis):
            super(TestSqueeze, self).__init__()
            self._axis = axis

        def hybrid_forward(self, F, x):
            return F.np.squeeze(x, axis=self._axis)

    for shape, axis in config:
        data_np = _np.random.uniform(size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = _np.squeeze(data_np, axis=axis)
        ret_mx = np.squeeze(data_mx, axis=axis)
        assert_almost_equal(ret_mx.asnumpy(), ret_np, rtol=1e-5, atol=1e-6, use_broadcast=False)

        net = TestSqueeze(axis)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            data_mx.attach_grad()
            with mx.autograd.record():
                ret_mx = net(data_mx)
            assert_almost_equal(ret_mx.asnumpy(), ret_np, rtol=1e-5, atol=1e-6, use_broadcast=False)
            ret_mx.backward()
            assert_almost_equal(data_mx.grad.asnumpy(), _np.ones_like(data_np),
                                rtol=1e-5, atol=1e-6, use_broadcast=False)


@with_seed()
@use_np
def test_np_prod():
    class TestProd(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):
            super(TestProd, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.np.prod(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

    in_data_dim = random.choice([3, 4])
    shape = rand_shape_nd(in_data_dim, dim=3)
    for hybridize in [False, True]:
        for keepdims in [True, False]:
            for axis in ([i for i in range(in_data_dim)] + [(), None]):
                for itype in ['float32', 'float64']:
                    for dtype in ['float32', 'float64']:
                        # test gluon
                        test_prod = TestProd(axis=axis, dtype=dtype, keepdims=keepdims)
                        if hybridize:
                            test_prod.hybridize()
                        x = np.array(_np.random.uniform(-2.0, 2.0, size=shape), dtype=itype)
                        x.attach_grad()
                        print(x.grad.dtype)
                        expected_ret = _np.prod(x.asnumpy(), axis=axis, keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        with mx.autograd.record():
                            y = test_prod(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3, atol=1e-5, use_broadcast=False)
                        y.backward()
                        # use keepdims=True so that broadcast divide can be used to calculate
                        # grad of input
                        expected_ret = _np.prod(x.asnumpy(), axis=axis, keepdims=True)
                        assert_almost_equal(x.grad.asnumpy(), expected_ret / x.asnumpy(), rtol=1e-3, atol=1e-3,
                                            use_broadcast=False)

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x").as_np_ndarray()
                            mx_sym = mx.sym.np.prod(x_sym, axis=axis, dtype=dtype, keepdims=keepdims).as_nd_ndarray()
                            check_numeric_gradient(mx_sym, [x.as_nd_ndarray()],
                                                   numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.prod(x.asnumpy(), axis=axis, keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@with_seed()
@use_np
def test_np_flatten():
    class TestFlatten(HybridBlock):
        def hybrid_forward(self, F, x):
            return x.flatten()

    shapes = [(), (2, 0, 1), (3, 4, 5), 6, (0,), (0, 0, 0)]
    for shape in shapes:
        for hybridize in [True, False]:
            test_flatten = TestFlatten()
            if hybridize:
                test_flatten.hybridize()
            a_np = _np.random.uniform(size=shape).astype('float32')
            a_mx = np.array(a_np, dtype=a_np.dtype)
            a_mx.attach_grad()
            with mx.autograd.record():
                ret = test_flatten(a_mx)
            expected_ret = a_np.flatten()
            assert_almost_equal(expected_ret, ret.asnumpy(), rtol=1e-5, atol=1e-6, use_broadcast=False)
            # check gradient
            ret.backward()
            assert_almost_equal(a_mx.grad.asnumpy(), _np.ones_like(a_np), rtol=1e-5, atol=1e-6, use_broadcast=False)


@with_seed()
@use_np
def test_np_broadcast_to():
    class TestBroadcastTo(HybridBlock):
        def __init__(self, dst_shape):
            super(TestBroadcastTo, self).__init__()
            self._dst_shape = dst_shape

        def hybrid_forward(self, F, x):
            return F.np.broadcast_to(x, self._dst_shape)

    shapes = [
        ((), (1, 2, 4, 5)),
        ((1,), (4, 5, 6)),
        ((1, 0), (2, 4, 0)),
        ((1, 1), (2, 4, 0)),
        ((4, 1), (1, 2, 3, 4, 5)),
        ((4, 1), (1, 0, 3, 4, 5))
    ]
    for src_shape, dst_shape in shapes:
        for hybridize in [True, False]:
            test_broadcast_to = TestBroadcastTo(dst_shape)
            if hybridize:
                test_broadcast_to.hybridize()

            a = _np.random.uniform(size=src_shape).astype(np.float32)
            expected_ret = _np.broadcast_to(a, dst_shape)
            a_mx = np.array(a, dtype=a.dtype)
            a_mx.attach_grad()
            with mx.autograd.record():
                ret = test_broadcast_to(a_mx)
            assert_almost_equal(ret.asnumpy(), expected_ret, rtol=1e-5, atol=1e-6, use_broadcast=False)
            ret.backward()
            expected_grad = collapse_sum_like(_np.ones_like(expected_ret), src_shape)
            assert_almost_equal(a_mx.grad.asnumpy(), expected_grad, rtol=1e-5, atol=1e-6, use_broadcast=False)


@with_seed()
@use_np
def test_np_transpose():
    def np_transpose_grad(out_shape, dtype, axes=None):
        ograd = _np.ones(out_shape, dtype=dtype)
        if axes is None or axes == ():
            return _np.transpose(ograd, axes)
        np_axes = _np.array(list(axes))
        return _np.transpose(ograd, tuple(list(_np.argsort(np_axes))))

    class TestTranspose(HybridBlock):
        def __init__(self, axes=None):
            super(TestTranspose, self).__init__()
            self.axes = axes

        def hybrid_forward(self, F, a):
            return F.np.transpose(a, self.axes)

    for hybridize in [True, False]:
        for dtype in [_np.int32, _np.float32]:
            for ndim in range(7):
                shape = rand_shape_nd(ndim, dim=5, allow_zero_size=True)
                axeses = [None]
                if ndim == 0:
                    axeses += [()]
                else:
                    axes = [i for i in range(ndim)]
                    axeses.append(tuple(axes))
                    random.shuffle(axes)
                    axeses.append(tuple(axes))
                for axes in axeses:
                    test_trans = TestTranspose(axes)
                    if hybridize:
                        test_trans.hybridize()
                    x = rand_ndarray(shape).as_np_ndarray()
                    x = x.astype(dtype)
                    x.attach_grad()
                    np_out = _np.transpose(x.asnumpy(), axes)
                    with mx.autograd.record():
                        mx_out = test_trans(x)
                    assert mx_out.shape == np_out.shape
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)
                    mx_out.backward()
                    np_backward = np_transpose_grad(np_out.shape, dtype, axes)
                    assert_almost_equal(x.grad.asnumpy(), np_backward, rtol=1e-3, atol=1e-5, use_broadcast=False)

                    mx_out = np.transpose(x, axes)
                    np_out = _np.transpose(x.asnumpy(), axes)
                    assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5, use_broadcast=False)


@with_seed()
@use_np
def test_np_meshgrid():
    nx, ny = (4, 5)
    x = np.array(_np.linspace(0, 1, nx), dtype=np.float32)
    y = np.array(_np.linspace(0, 1, ny), dtype=np.float32)
    z = np.ones(())
    xv, yv, zv = np.meshgrid(x, y, z)
    xv_expected, yv_expected, zv_expected = _np.meshgrid(x.asnumpy(), y.asnumpy(), z.asnumpy())
    assert same(xv.asnumpy(), xv_expected)
    assert same(yv.asnumpy(), yv_expected)
    assert same(zv.asnumpy(), zv_expected)


@with_seed()
@use_np
def test_np_broadcast_arrays():
    shape_config = [
        [(), (2, 1), (1, 3), (4, 1, 1), (5, 4, 2, 3)],
        [(0,), (), (2, 1), (1, 0), (3, 2, 1)]
    ]
    for shapes in shape_config:
        arrays_np = [_np.random.randint(low=0, high=1000, size=shape, dtype=_np.int32) for shape in shapes]
        arrays_mx = [np.array(arr, dtype=arr.dtype) for arr in arrays_np]
        expected_rets = _np.broadcast_arrays(*arrays_np)
        rets = np.broadcast_arrays(*arrays_mx)
        for expected_ret, ret in zip(expected_rets, rets):
            assert same(expected_ret, ret.asnumpy())


@with_seed()
@use_np
def test_np_tile():
    config = [
        ((), ()),
        ((), 0),
        ((), (2, 0)),
        ((), (2, 3)),
        ((4, 2), (2,)),
        ((4, 2), (2, 3)),
        ((4, 2), (2, 1, 4)),
        ((4, 2), (2, 3, 4)),
        ((4, 2), (2, 0)),
        ((4, 2), (2, 0, 3)),
        ((4, 2), (2, 0, 3)),
        ((4, 0), (2, 0, 3)),
    ]

    class TestTile(HybridBlock):
        def __init__(self, reps):
            super(TestTile, self).__init__()
            self._reps = reps

        def hybrid_forward(self, F, x):
            return F.np.tile(x, reps=self._reps)

    for shape, reps in config:
        data_np = _np.random.randint(low=0, high=1000, size=shape)
        data_mx = np.array(data_np, dtype=data_np.dtype)
        ret_np = _np.tile(data_np, reps=reps)
        ret_mx = np.tile(data_mx, reps=reps)
        assert same(ret_mx.asnumpy(), ret_np)

        net = TestTile(reps)
        for hybrid in [False, True]:
            if hybrid:
                net.hybridize()
            ret_mx = net(data_mx)
            assert same(ret_mx.asnumpy(), ret_np)


@with_seed()
@use_np
def test_np_arange():
    configs = [
        (1, 10, 2),
        (1, 10, 4),
        (1, -10, 4),
        (1, -10, -2),
        (1, -10, -4),
        (2, 3),
        (2, -3),
        (-2, -3),
        (-2, 3),
        (4, 0, 5),
        (-4, 0, 5),
        (-4, 0, -5),
        (0, 0),
        (11, 11),
        (0, 0, 2),
        (0, 0, -2),
        (0, 5, None),
        (0, -5, None),
        0,
        6,
    ]
    dtypes = ['int32', 'float16', 'float32', 'float64', None]
    for config in configs:
        for dtype in dtypes:
            if isinstance(config, tuple):
                mx_ret = np.arange(*config, dtype=dtype)
                np_ret = _np.arange(*config, dtype=dtype)
            else:
                mx_ret = np.arange(config, dtype=dtype)
                np_ret = _np.arange(config, dtype=dtype)
            assert same(mx_ret.asnumpy(), np_ret)

    class TestRange(HybridBlock):
        def __init__(self, start, stop=None, step=None, dtype=None):
            super(TestRange, self).__init__()
            self._start = start
            self._stop = stop
            self._step = step
            self._dtype = dtype

        def hybrid_forward(self, F, x):
            return x + F.np.arange(self._start, self._stop, self._step, dtype=self._dtype)

    for dtype in dtypes:
        x = np.zeros(shape=(), dtype=dtype)
        for config in configs:
            for hybridize in [False, True]:
                if isinstance(config, tuple):
                    net = TestRange(*config, dtype=dtype)
                    np_out = _np.arange(*config, dtype=dtype)
                else:
                    net = TestRange(config, dtype=dtype)
                    np_out = _np.arange(config, dtype=dtype)
                if hybridize:
                    net.hybridize()
                mx_out = net(x)
                assert same(mx_out.asnumpy(), np_out)


if __name__ == '__main__':
    import nose
    nose.runmodule()
