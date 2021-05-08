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
from __future__ import division
import itertools
import os
import pytest
import numpy as _np
import mxnet as mx
from mxnet import np, npx, autograd
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray, use_np
from common import retry, TemporaryDirectory, xfail_when_nonstandard_decimal_separator
from mxnet.test_utils import verify_generator, gen_buckets_probs_with_ppf, assert_exception, is_op_runnable, collapse_sum_like
from mxnet.ndarray.ndarray import py_slice
from mxnet.base import integer_types


@use_np
def test_np_empty():
    # (input dtype, expected output dtype)
    dtype_pairs = [
        (np.int8, np.int8),
        (np.int16, np.int16),
        (np.int32, np.int32),
        (np.float16, np.float16),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.bool_, np.bool_),
        (np.bool, np.bool_),
        ('int8', np.int8),
        ('int16', np.int16),
        ('int32', np.int32),
        ('float16', np.float16),
        ('float32', np.float32),
        ('float64', np.float64),
        ('bool', np.bool_),
        (None, np.float32),
    ]
    orders = ['C', 'F', 'A']
    shapes = [
        (),
        0,
        (0,),
        (0, 0),
        2,
        (2,),
        (3, 0),
        (4, 5),
        (1, 1, 1, 1),
    ]
    ctxes = [npx.current_context(), None]
    for dtype, expected_dtype in dtype_pairs:
        for shape in shapes:
            for order in orders:
                for ctx in ctxes:
                    if order == 'C':
                        ret = np.empty(shape, dtype, order, ctx)
                        assert ret.dtype == expected_dtype
                        assert ret.shape == shape if isinstance(shape, tuple) else (shape,)
                        assert ret.ctx == npx.current_context()
                    else:
                        assert_exception(np.empty, NotImplementedError, shape, dtype, order, ctx)


@use_np
def test_np_array_creation():
    dtypes = [_np.int8, _np.int16, _np.int32, _np.float16, _np.float32, _np.float64, _np.bool, _np.bool_,
              'int8', 'int16', 'int32', 'float16', 'float32', 'float64', 'bool', None]
    objects = [
        [],
        (),
        [[1, 2], [3, 4]],
        _np.random.randint(-10, 10, size=rand_shape_nd(3)),
        _np.random.uniform(size=rand_shape_nd(3)),
        _np.random.uniform(size=(3, 0, 4))
    ]
    for dtype in dtypes:
        for src in objects:
            mx_arr = np.array(src, dtype=dtype)
            assert mx_arr.ctx == mx.current_context()
            if dtype is None:
                dtype = src.dtype if isinstance(src, _np.ndarray) else _np.float32
            if isinstance(src, mx.nd.NDArray):
                np_arr = _np.array(src.asnumpy(), dtype=dtype)
            else:
                np_arr = _np.array(src, dtype=dtype)
            assert mx_arr.dtype == np_arr.dtype
            assert same(mx_arr.asnumpy(), np_arr)


@use_np
@pytest.mark.serial
def test_np_zeros():
    # test np.zeros in Gluon
    class TestZeros(HybridBlock):
        def __init__(self, shape, dtype=None):
            super(TestZeros, self).__init__()
            self._shape = shape
            self._dtype = dtype

        def hybrid_forward(self, F, x, *args, **kwargs):
            return x + F.np.zeros(shape, dtype)

    class TestZerosOutputType(HybridBlock):
        def hybrid_forward(self, F, x, *args, **kwargs):
            return x, F.np.zeros(shape=())

    # test np.zeros in imperative
    def check_zero_array_creation(shape, dtype):
        np_out = _np.zeros(shape=shape, dtype=dtype)
        mx_out = np.zeros(shape=shape, dtype=dtype)
        assert same(mx_out.asnumpy(), np_out)
        if dtype is None:
            assert mx_out.dtype == _np.float32
            assert np_out.dtype == _np.float64
        else:
            assert mx_out.dtype == np_out.dtype

    shapes = [(0,), (2, 0, 2), (0, 0, 0, 0), ()]
    shapes += [rand_shape_nd(ndim, allow_zero_size=True) for ndim in range(5)]
    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, None]
    for shape in shapes:
        for dtype in dtypes:
            check_zero_array_creation(shape, dtype)
            x = np.array(_np.random.uniform(size=shape), dtype=dtype)
            if dtype is None:
                x = x.astype('float32')
            for hybridize in [True, False]:
                test_zeros = TestZeros(shape, dtype)
                test_zeros_output_type = TestZerosOutputType()
                if hybridize:
                    test_zeros.hybridize()
                    test_zeros_output_type.hybridize()
                y = test_zeros(x)
                assert type(y) == np.ndarray
                assert same(x.asnumpy(), y.asnumpy())
                y = test_zeros_output_type(x)
                assert type(y[1]) == np.ndarray

    for shape in shapes:
        for dtype in [_np.bool, bool, _np.bool, 'bool']:
            check_zero_array_creation(shape, dtype)


@use_np
def test_np_ones():
    # test np.ones in Gluon
    class TestOnes(HybridBlock):
        def __init__(self, shape, dtype=None):
            super(TestOnes, self).__init__()
            self._shape = shape
            self._dtype = dtype

        def hybrid_forward(self, F, x, *args, **kwargs):
            return x * F.np.ones(shape, dtype)

    class TestOnesOutputType(HybridBlock):
        def hybrid_forward(self, F, x, *args, **kwargs):
            return x, F.np.ones(shape=())

    # test np.ones in imperative
    def check_ones_array_creation(shape, dtype):
        np_out = _np.ones(shape=shape, dtype=dtype)
        mx_out = np.ones(shape=shape, dtype=dtype)
        assert same(mx_out.asnumpy(), np_out)
        if dtype is None:
            assert mx_out.dtype == _np.float32
            assert np_out.dtype == _np.float64
        else:
            assert mx_out.dtype == np_out.dtype

    shapes = [(0,), (2, 0, 2), (0, 0, 0, 0), ()]
    shapes += [rand_shape_nd(ndim, allow_zero_size=True) for ndim in range(5)]
    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, None]
    for shape in shapes:
        for dtype in dtypes:
            check_ones_array_creation(shape, dtype)
            x = mx.nd.array(_np.random.uniform(size=shape), dtype=dtype).as_np_ndarray()
            if dtype is None:
                x = x.astype('float32')
            for hybridize in [True, False]:
                test_ones = TestOnes(shape, dtype)
                test_ones_output_type = TestOnesOutputType()
                if hybridize:
                    test_ones.hybridize()
                    test_ones_output_type.hybridize()
                y = test_ones(x)
                assert type(y) == np.ndarray
                assert same(x.asnumpy(), y.asnumpy())
                y = test_ones_output_type(x)
                assert type(y[1]) == np.ndarray

    for shape in shapes:
        for dtype in [_np.bool, bool, _np.bool, 'bool']:
            check_ones_array_creation(shape, dtype)


@use_np
@pytest.mark.serial
def test_identity():
    class TestIdentity(HybridBlock):
        def __init__(self, shape, dtype=None):
            super(TestIdentity, self).__init__()
            self._n = n
            self._dtype = dtype

        def hybrid_forward(self, F, x):
            return x * F.np.identity(self._n, self._dtype)

    class TestIdentityOutputType(HybridBlock):
        def hybrid_forward(self, F, x):
            return x, F.np.identity(0)

    def check_identity_array_creation(shape, dtype):
        np_out = _np.identity(n=n, dtype=dtype)
        mx_out = np.identity(n=n, dtype=dtype)
        assert same(mx_out.asnumpy(), np_out)
        if dtype is None:
            assert mx_out.dtype == _np.float32
            assert np_out.dtype == _np.float64

    ns = [0, 1, 2, 3, 5, 15, 30, 200]
    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, None]
    for n in ns:
        for dtype in dtypes:
            check_identity_array_creation(n, dtype)
            x = mx.nd.array(_np.random.uniform(size=(n, n)), dtype=dtype).as_np_ndarray()
            if dtype is None:
                x = x.astype('float32')
            for hybridize in [True, False]:
                test_identity = TestIdentity(n, dtype)
                test_identity_output_type = TestIdentityOutputType()
                if hybridize:
                    test_identity.hybridize()
                    test_identity_output_type.hybridize()
                y = test_identity(x)
                assert type(y) == np.ndarray
                assert same(x.asnumpy() * _np.identity(n, dtype), y.asnumpy())
                y = test_identity_output_type(x)
                assert type(y[1]) == np.ndarray


@xfail_when_nonstandard_decimal_separator
@pytest.mark.serial
def test_np_ndarray_binary_element_wise_ops():
    np_op_map = {
        '+': _np.add,
        '*': _np.multiply,
        '-': _np.subtract,
        '/': _np.divide,
        'mod': _np.mod,
        'pow': _np.power,

    }

    if is_op_runnable():
        np_op_map.update({
            '==': _np.equal,
            '!=': _np.not_equal,
            '>': _np.greater,
            '>=': _np.greater_equal,
            '<': _np.less,
            '<=': _np.less_equal
        })

    def _get_grad_func(op, scalar=None, reverse=False):
        if op == '+':
            if scalar is None:
                return lambda ograd, x1, x2, out: (collapse_sum_like(ograd, x1.shape),
                                                   collapse_sum_like(ograd, x2.shape))
            elif not reverse:
                return lambda ograd, x1, x2, out: ograd
            else:
                return lambda ograd, x1, x2, out: ograd
        elif op == '-':
            if scalar is None:
                return lambda ograd, x1, x2, out: (collapse_sum_like(ograd, x1.shape),
                                                   -collapse_sum_like(ograd, x2.shape))
            elif not reverse:
                return lambda ograd, x1, x2, out: ograd
            else:
                return lambda ograd, x1, x2, out: -ograd
        elif op == '*':
            if scalar is None:
                return lambda ograd, x1, x2, out: (collapse_sum_like(ograd * x2, x1.shape),
                                                   collapse_sum_like(ograd * x1, x2.shape))
            elif not reverse:
                return lambda ograd, x1, x2, out: ograd * x2
            else:
                return lambda ograd, x1, x2, out: ograd * x1
        elif op == '/':
            if scalar is None:
                return lambda ograd, x1, x2, out: (collapse_sum_like(ograd / x2, x1.shape),
                                                   collapse_sum_like(-x1 * ograd / (x2 * x2), x2.shape))
            elif not reverse:
                return lambda ograd, x1, x2, out: ograd / x2
            else:
                return lambda ograd, x1, x2, out: -x1 * ograd / (x2 * x2)
        elif op == 'mod':
            if scalar is None:
                return lambda ograd, x1, x2, out: (collapse_sum_like(ograd, x1.shape),
                                                   collapse_sum_like(-ograd * _np.floor(x1 / x2), x2.shape))
            elif not reverse:
                return lambda ograd, x1, x2, out: ograd
            else:
                return lambda ograd, x1, x2, out: -ograd * _np.floor(x1 / x2)
        elif op == 'pow':
            if scalar is None:
                return lambda ograd, x1, x2, out: (collapse_sum_like(ograd * x2 * _np.power(x1, x2 - 1), x1.shape),
                                                   collapse_sum_like(ograd * out * _np.log(x1), x2.shape))
            elif not reverse:
                return lambda ograd, x1, x2, out: ograd * x2 * _np.power(x1, x2 - 1)
            else:
                return lambda ograd, x1, x2, out: ograd * out * _np.log(x1)
        elif op in ('==', '!=', '<', '<=', '>', '>='):
            if scalar is None:
                return lambda ograd, x1, x2, out: (_np.zeros_like(x1), _np.zeros_like(x2))
            else:
                return lambda ograd, x1, x2, out: _np.zeros_like(ograd)
        return None

    def get_np_ret(x1, x2, op):
        return np_op_map[op](x1, x2)

    @use_np
    class TestBinaryElementWiseOp(HybridBlock):
        def __init__(self, op, scalar=None, reverse=False):
            super(TestBinaryElementWiseOp, self).__init__()
            self._op = op
            self._scalar = scalar
            self._reverse = reverse  # if false, scalar is the right operand.

        def hybrid_forward(self, F, x, *args):
            if self._op == '+':
                if self._scalar is not None:
                    return x + self._scalar if not self._reverse else self._scalar + x
                else:
                    return x + args[0] if not self._reverse else args[0] + x
            elif self._op == '*':
                if self._scalar is not None:
                    return x * self._scalar if not self._reverse else self._scalar * x
                else:
                    return x * args[0] if not self._reverse else args[0] * x
            elif self._op == '-':
                if self._scalar is not None:
                    return x - self._scalar if not self._reverse else self._scalar - x
                else:
                    return x - args[0] if not self._reverse else args[0] - x
            elif self._op == '/':
                if self._scalar is not None:
                    return x / self._scalar if not self._reverse else self._scalar / x
                else:
                    return x / args[0] if not self._reverse else args[0] / x
            elif self._op == 'mod':
                if self._scalar is not None:
                    return x % self._scalar if not self._reverse else self._scalar % x
                else:
                    return x % args[0] if not self._reverse else args[0] % x
            elif self._op == 'pow':
                if self._scalar is not None:
                    return x ** self._scalar if not self._reverse else self._scalar ** x
                else:
                    return x ** args[0] if not self._reverse else args[0] ** x
            elif self._op == '>':
                if self._scalar is not None:
                    return x > self._scalar if not self._reverse else self._scalar > x
                else:
                    return x > args[0]
            elif self._op == '>=':
                if self._scalar is not None:
                    return x >= self._scalar if not self._reverse else self._scalar >= x
                else:
                    return x >= args[0]
            elif self._op == '<':
                if self._scalar is not None:
                    return x < self._scalar if not self._reverse else self._scalar < x
                else:
                    return x < args[0]
            elif self._op == '<=':
                if self._scalar is not None:
                    return x <= self._scalar if not self._reverse else self._scalar <= x
                else:
                    return x <= args[0]
            elif self._op == '==':
                if self._scalar is not None:
                    return x == self._scalar if not self._reverse else self._scalar == x
                else:
                    return x == args[0]
            elif self._op == '!=':
                if self._scalar is not None:
                    return x != self._scalar if not self._reverse else self._scalar != x
                else:
                    return x != args[0]
            else:
                print(self._op)
                assert False

    logic_ops = ['==', '!=', '>', '<', '>=', '<=']
    @use_np
    def check_binary_op_result(shape1, shape2, op, dtype=None):
        if shape1 is None:
            mx_input1 = abs(_np.random.uniform()) + 1
            np_input1 = mx_input1
        else:
            mx_input1 = (rand_ndarray(shape1, dtype=dtype).abs() + 1).as_np_ndarray()
            mx_input1.attach_grad()
            np_input1 = mx_input1.asnumpy()
        if shape2 is None:
            mx_input2 = abs(_np.random.uniform()) + 1
            np_input2 = mx_input2
        else:
            mx_input2 = (rand_ndarray(shape2, dtype=dtype).abs() + 1).as_np_ndarray()
            mx_input2.attach_grad()
            np_input2 = mx_input2.asnumpy()

        scalar = None
        reverse = False
        if isinstance(mx_input1, mx.nd.NDArray) and not isinstance(mx_input2, mx.nd.NDArray):
            scalar = mx_input2
            reverse = False
        elif isinstance(mx_input2, mx.nd.NDArray) and not isinstance(mx_input1, mx.nd.NDArray):
            scalar = mx_input1
            reverse = True

        grad_func = _get_grad_func(op, scalar, reverse)
        np_out = get_np_ret(np_input1, np_input2, op)
        ograd = _np.ones_like(np_out)
        for hybridize in [True, False]:
            if scalar is None:
                get_mx_ret_np = TestBinaryElementWiseOp(op)
                get_mx_ret_classic = TestBinaryElementWiseOp(op)
                if hybridize:
                    get_mx_ret_np.hybridize()
                    get_mx_ret_classic.hybridize()
                if grad_func is None:
                    mx_out = get_mx_ret_np(mx_input1, mx_input2)
                else:
                    with mx.autograd.record():
                        mx_out = get_mx_ret_np(mx_input1, mx_input2)
                    mx_out.backward()
                assert type(mx_out) == np.ndarray
                if op in logic_ops:
                    assert np_out.dtype == mx_out.dtype
                assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-6, rtol=1e-5, use_broadcast=False)

                if grad_func is not None:
                    x1_grad_expected, x2_grad_expected = grad_func(ograd, np_input1, np_input2, np_out)
                    assert_almost_equal(mx_input1.grad.asnumpy(), x1_grad_expected, atol=1e-5, rtol=1e-3,
                                        use_broadcast=False)
                    assert_almost_equal(mx_input2.grad.asnumpy(), x2_grad_expected, atol=1e-5, rtol=1e-3,
                                        use_broadcast=False)
            else:
                get_mx_ret = TestBinaryElementWiseOp(op, scalar=scalar, reverse=reverse)
                if hybridize:
                    get_mx_ret.hybridize()
                if reverse:
                    mx_input = mx_input2
                else:
                    mx_input = mx_input1

                if grad_func is None:
                    mx_out = get_mx_ret(mx_input)
                else:
                    with mx.autograd.record():
                        mx_out = get_mx_ret(mx_input)
                    mx_out.backward()
                assert type(mx_out) == np.ndarray

                if op in logic_ops:
                    assert np_out.dtype == mx_out.dtype
                assert_almost_equal(mx_out.asnumpy(), np_out, atol=1e-6, rtol=1e-5, use_broadcast=False)

                # check grad
                if grad_func is not None:
                    x_grad_expected = grad_func(ograd, np_input1, np_input2, np_out)
                    assert_almost_equal(mx_input.grad.asnumpy(), x_grad_expected, atol=1e-5, rtol=1e-3,
                                        use_broadcast=False)

    dtypes = [_np.float32, _np.float64, None]
    ops = np_op_map.keys()
    for dtype in dtypes:
        for op in ops:
            check_binary_op_result((3, 4), (3, 4), op, dtype)
            check_binary_op_result(None, (3, 4), op, dtype)
            check_binary_op_result((3, 4), None, op, dtype)
            check_binary_op_result((1, 4), (3, 1), op, dtype)
            check_binary_op_result(None, (3, 1), op, dtype)
            check_binary_op_result((1, 4), None, op, dtype)
            check_binary_op_result((1, 4), (3, 5, 4), op, dtype)
            check_binary_op_result((), (3, 5, 4), op, dtype)
            check_binary_op_result((), None, op, dtype)
            check_binary_op_result(None, (), op, dtype)
            check_binary_op_result((0, 2), (1, 1), op, dtype)
            check_binary_op_result((0, 2), None, op, dtype)
            check_binary_op_result(None, (0, 2), op, dtype)


@pytest.mark.serial
def test_np_hybrid_block_multiple_outputs():
    @use_np
    class TestAllNumpyOutputs(HybridBlock):
        def hybrid_forward(self, F, x, *args, **kwargs):
            return F.np.add(x, x), F.np.multiply(x, x)

    class TestAllClassicOutputs(HybridBlock):
        def hybrid_forward(self, F, x, *args, **kwargs):
            return x.as_nd_ndarray() + x.as_nd_ndarray(), x.as_nd_ndarray() * x.as_nd_ndarray()

    data_np = np.ones((2, 3))
    for block, expected_out_type in [(TestAllClassicOutputs, mx.nd.NDArray),
                                     (TestAllNumpyOutputs, np.ndarray)]:
        net = block()
        for hybridize in [True, False]:
            if hybridize:
                net.hybridize()
            out1, out2 = net(data_np)
            assert type(out1) is expected_out_type
            assert type(out2) is expected_out_type

    @use_np
    class TestMixedTypeOutputsFailure(HybridBlock):
        def hybrid_forward(self, F, x, *args, **kwargs):
            return x.as_nd_ndarray() + x.as_nd_ndarray(), F.np.multiply(x, x)

    net = TestMixedTypeOutputsFailure()
    assert_exception(net, TypeError, data_np)
    net.hybridize()
    assert_exception(net, TypeError, data_np)


@use_np
def test_np_grad_ndarray_type():
    data = np.array(2, dtype=_np.float32)
    data.attach_grad()
    assert type(data.grad) == np.ndarray
    assert type(data.detach()) == np.ndarray


@use_np
@pytest.mark.serial
def test_np_ndarray_astype():
    class TestAstype(HybridBlock):
        def __init__(self, dtype, copy):
            super(TestAstype, self).__init__()
            self._dtype = dtype
            self._copy = copy

        def hybrid_forward(self, F, x):
            return x.astype(dtype=self._dtype, copy=self._copy)

    def check_astype_equal(itype, otype, copy, expect_zero_copy=False, hybridize=False):
        expect_zero_copy = copy is False and itype == otype
        mx_data = np.array([2, 3, 4, 5], dtype=itype)
        np_data = mx_data.asnumpy()
        test_astype = TestAstype(otype, copy)
        if hybridize:
            test_astype.hybridize()
        mx_ret = test_astype(mx_data)
        assert type(mx_ret) is np.ndarray
        np_ret = np_data.astype(dtype=otype, copy=copy)
        assert mx_ret.dtype == np_ret.dtype
        assert same(mx_ret.asnumpy(), np_ret)
        if expect_zero_copy and not hybridize:
            assert id(mx_ret) == id(mx_data)
            assert id(np_ret) == id(np_data)

    dtypes = [np.int8, np.uint8, np.int32, np.float16, np.float32, np.float64, np.bool, np.bool_,
              'int8', 'uint8', 'int32', 'float16', 'float32', 'float64', 'bool']

    for itype, otype in itertools.product(dtypes, dtypes):
        for copy in [True, False]:
            for hybridize in [True, False]:
                check_astype_equal(itype, otype, copy, hybridize)


def test_np_ndarray_copy():
    mx_data = np.array([2, 3, 4, 5], dtype=_np.int32)
    assert_exception(mx_data.copy, NotImplementedError, order='F')
    mx_ret = mx_data.copy()
    np_ret = mx_data.asnumpy().copy()
    assert same(mx_ret.asnumpy(), np_ret)


def test_formatting():
    def test_0d():
        a = np.array(np.pi)
        _a = a.asnumpy()
        assert '{:0.3g}'.format(a) == '{:0.3g}'.format(_a)
        assert '{:0.3g}'.format(a[()]) == '{:0.3g}'.format(_a[()])

    def test_nd_format():
        a = np.array([np.pi])
        assert_exception('{:30}'.format, TypeError, a)

    def test_nd_no_format():
        a = np.array([np.pi])
        _a = a.asnumpy()
        assert '{}'.format(a) == '{}'.format(_a)
        b = np.arange(8).reshape(2,2,2)
        assert '{}'.format(a) == '{}'.format(_a)

    context = mx.context.current_context()
    if str(context)[:3] != 'gpu':
        test_0d()
        test_nd_format()
        test_nd_no_format()
    # if the program is running in GPU, the formatted string would be appended with context notation
    # for exmpale, if a = np.array([np.pi]), the return value of '{}'.format(a) is '[3.1415927] @gpu(0)'


@use_np
@pytest.mark.serial
def test_np_ndarray_indexing():
    def np_int(index, int_type=np.int32):
        """
        Helper function for testing indexing that converts slices to slices of ints or None, and tuples to
        tuples of ints or None.
        """
        def convert(num):
            if num is None:
                return num
            else:
                return int_type(num)

        if isinstance(index, slice):
            return slice(convert(index.start), convert(index.stop), convert(index.step))
        elif isinstance(index, tuple):  # tuple of slices and integers
            ret = []
            for elem in index:
                if isinstance(elem, slice):
                    ret.append(slice(convert(elem.start), convert(elem.stop), convert(elem.step)))
                else:
                    ret.append(convert(elem))
            return tuple(ret)
        else:
            assert False

    # Copied from test_ndarray.py. Under construction.
    def test_getitem(np_array, index):
        np_index = index
        if type(index) == mx.nd.NDArray:  # use of NDArray is prohibited
            assert False
        if isinstance(index, np.ndarray):
            np_index = index.asnumpy()
        if isinstance(index, tuple):
            np_index = tuple([
                idx.asnumpy() if isinstance(idx, mx.nd.NDArray) else idx
                for idx in index]
            )
        np_indexed_array = np_array[np_index]
        mx_np_array = np.array(np_array, dtype=np_array.dtype)
        for autograd in [True, False]:
            try:
                if autograd:
                    with mx.autograd.record():
                        mx_indexed_array = mx_np_array[index]
                else:
                    mx_indexed_array = mx_np_array[index]
            except Exception as e:
                print('Failed with index = {}'.format(index))
                raise e
            mx_indexed_array = mx_indexed_array.asnumpy()
            assert same(np_indexed_array, mx_indexed_array), 'Failed with index = {}'.format(index)

    def test_getitem_slice_bound():
        mx_array = np.arange(10)
        np_array = mx_array.asnumpy()
        assert_almost_equal(mx_array[100:], np_array[100:])
        assert_almost_equal(mx_array[:100], np_array[:100])
        assert_almost_equal(mx_array[-100:], np_array[-100:])
        assert_almost_equal(mx_array[:-100], np_array[:-100])

        mx_array = np.arange(81).reshape(3, 3, 3, 3)
        np_array = mx_array.asnumpy()
        assert_almost_equal(mx_array[100:], np_array[100:])
        assert_almost_equal(mx_array[:100], np_array[:100])
        assert_almost_equal(mx_array[-100:], np_array[-100:])
        assert_almost_equal(mx_array[:-100], np_array[:-100])

    def test_setitem(np_array, index):
        def assert_same(np_array, np_index, mx_array, mx_index, mx_value, np_value=None):
            if np_value is not None:
                np_array[np_index] = np_value
            elif isinstance(mx_value, np.ndarray):
                np_array[np_index] = mx_value.asnumpy()
            else:
                np_array[np_index] = mx_value
            try:
                mx_array[mx_index] = mx_value
            except Exception as e:
                print('Failed with index = {}, value.shape = {}'.format(mx_index, mx_value.shape))
                raise e

            assert same(np_array, mx_array.asnumpy())

        def _is_basic_index(index):
            if isinstance(index, (integer_types, py_slice)):
                return True
            if isinstance(index, tuple) and all(isinstance(i, (integer_types, py_slice)) for i in index):
                return True
            return False

        np_index = index  # keep this native numpy type
        if isinstance(index, np.ndarray):
            np_index = index.asnumpy()
        if isinstance(index, tuple):
            np_index = []
            for idx in index:
                if isinstance(idx, np.ndarray):
                    np_index.append(idx.asnumpy())
                else:
                    np_index.append(idx)
            np_index = tuple(np_index)

        mx_array = np.array(np_array, dtype=np_array.dtype)  # mxnet.np.ndarray
        np_array = mx_array.asnumpy()  # native numpy array
        indexed_array_shape = np_array[np_index].shape
        np_indexed_array = _np.random.randint(low=-10000, high=0, size=indexed_array_shape)
        # test value is a native numpy array without broadcast
        assert_same(np_array, np_index, mx_array, index, np_indexed_array)
        # test value is a list without broadcast
        assert_same(np_array, np_index, mx_array, index, np_indexed_array.tolist())
        # test value is a mxnet numpy array without broadcast
        assert_same(np_array, np_index, mx_array, index, np.array(np_indexed_array))
        # test value is an numeric_type
        assert_same(np_array, np_index, mx_array, index, _np.random.randint(low=-10000, high=0))

        np_value = _np.random.randint(low=-10000, high=0,
                                      size=(indexed_array_shape[-1],) if len(indexed_array_shape) > 0 else ())
        # test mxnet ndarray with broadcast
        assert_same(np_array, np_index, mx_array, index, np.array(np_value))
        # test native numpy array with broadcast
        assert_same(np_array, np_index, mx_array, index, np_value)
        # test python list with broadcast
        assert_same(np_array, np_index, mx_array, index, np_value.tolist())

        # test value shape are expanded to be longer than index array's shape
        # this is currently only supported in basic indexing
        if _is_basic_index(index):
            expanded_value_shape = (1, 1) + np_value.shape
            assert_same(np_array, np_index, mx_array, index, np.array(np_value.reshape(expanded_value_shape)))
            assert_same(np_array, np_index, mx_array, index, np_value.reshape(expanded_value_shape))
            if len(expanded_value_shape) <= np_array[index].ndim:
                # NumPy does not allow value.ndim > np_array[index].ndim when value is a python list.
                # It may be a bug of NumPy.
                assert_same(np_array, np_index, mx_array, index, np_value.reshape(expanded_value_shape).tolist())

        # test list with broadcast
        assert_same(np_array, np_index, mx_array, index,
                    [_np.random.randint(low=-10000, high=0)] * indexed_array_shape[-1] if len(indexed_array_shape) > 0
                    else _np.random.randint(low=-10000, high=0))

    def test_getitem_autograd(np_array, index):
        """
        np_array: native numpy array.
        """
        x = np.array(np_array, dtype=np_array.dtype)
        x.attach_grad()
        with mx.autograd.record():
            y = x[index]
        y.backward()
        value = np.ones_like(y)
        x_grad = np.zeros_like(x)
        x_grad[index] = value
        assert same(x_grad.asnumpy(), x.grad.asnumpy())

    def test_setitem_autograd(np_array, index):
        """
        np_array: native numpy array.
        """
        x = np.array(np_array, dtype=np_array.dtype)
        out_shape = x[index].shape
        y = np.array(_np.random.uniform(size=out_shape))
        y.attach_grad()
        try:
            with mx.autograd.record():
                x[index] = y
                x.backward()
                y_grad = np.ones_like(y)
                assert same(y_grad.asnumpy(), y.grad.asnumpy())
        except mx.base.MXNetError as err:
            assert str(err).find('Inplace operations (+=, -=, x[:]=, etc) are not supported when recording with') != -1

    shape = (8, 16, 9, 9)
    np_array = _np.arange(_np.prod(_np.array(shape)), dtype='int32').reshape(shape)  # native np array

    # Test sliced output being ndarray:
    index_list = [
        (),
        # Basic indexing
        # Single int as index
        0,
        np.int32(0),
        np.int64(0),
        np.array(0, dtype='int32'),
        np.array(0, dtype='int64'),
        5,
        np.int32(5),
        np.int64(5),
        np.array(5, dtype='int32'),
        np.array(5, dtype='int64'),
        -1,
        np.int32(-1),
        np.int64(-1),
        np.array(-1, dtype='int32'),
        np.array(-1, dtype='int64'),
        # Slicing as index
        slice(5),
        np_int(slice(5), np.int32),
        np_int(slice(5), np.int64),
        slice(1, 5),
        np_int(slice(1, 5), np.int32),
        np_int(slice(1, 5), np.int64),
        slice(1, 5, 2),
        slice(1, 2, 2),
        np_int(slice(1, 5, 2), np.int32),
        np_int(slice(1, 5, 2), np.int64),
        slice(7, 0, -1),
        np_int(slice(7, 0, -1)),
        np_int(slice(7, 0, -1), np.int64),
        slice(None, 6),
        np_int(slice(None, 6)),
        np_int(slice(None, 6), np.int64),
        slice(None, 6, 3),
        np_int(slice(None, 6, 3)),
        np_int(slice(None, 6, 3), np.int64),
        slice(1, None),
        np_int(slice(1, None)),
        np_int(slice(1, None), np.int64),
        slice(1, None, 3),
        np_int(slice(1, None, 3)),
        np_int(slice(1, None, 3), np.int64),
        slice(None, None, 2),
        np_int(slice(None, None, 2)),
        np_int(slice(None, None, 2), np.int64),
        slice(None, None, -1),
        np_int(slice(None, None, -1)),
        np_int(slice(None, None, -1), np.int64),
        slice(None, None, -2),
        np_int(slice(None, None, -2), np.int32),
        np_int(slice(None, None, -2), np.int64),
        # Multiple ints as indices
        (1, 2, 3),
        np_int((1, 2, 3)),
        np_int((1, 2, 3), np.int64),
        (-1, -2, -3),
        np_int((-1, -2, -3)),
        np_int((-1, -2, -3), np.int64),
        (1, 2, 3, 4),
        np_int((1, 2, 3, 4)),
        np_int((1, 2, 3, 4), np.int64),
        (-4, -3, -2, -1),
        (-4, mx.np.array(-3, dtype='int32'), -2, -1),
        (-4, mx.np.array(-3, dtype='int64'), -2, -1),
        np_int((-4, -3, -2, -1)),
        np_int((-4, -3, -2, -1), np.int64),
        # slice(None) as indices
        (slice(None), slice(None), 1, 8),
        (slice(None), slice(None), np.array(1, dtype='int32'), 8),
        (slice(None), slice(None), np.array(1, dtype='int64'), 8),
        (slice(None), slice(None), -1, 8),
        (slice(None), slice(None), 1, -8),
        (slice(None), slice(None), -1, -8),
        np_int((slice(None), slice(None), 1, 8)),
        np_int((slice(None), slice(None), 1, 8), np.int64),
        (slice(None), slice(None), 1, 8),
        np_int((slice(None), slice(None), -1, -8)),
        np_int((slice(None), slice(None), -1, -8), np.int64),
        (slice(None), 2, slice(1, 5), 1),
        np_int((slice(None), 2, slice(1, 5), 1)),
        np_int((slice(None), 2, slice(1, 5), 1), np.int64),
        # Mixture of ints and slices as indices
        (slice(None, None, -1), 2, slice(1, 5), 1),
        np_int((slice(None, None, -1), 2, slice(1, 5), 1)),
        np_int((slice(None, None, -1), 2, slice(1, 5), 1), np.int64),
        (slice(None, None, -1), 2, slice(1, 7, 2), 1),
        np_int((slice(None, None, -1), 2, slice(1, 7, 2), 1)),
        np_int((slice(None, None, -1), 2, slice(1, 7, 2), 1), np.int64),
        (slice(1, 8, 2), slice(14, 2, -2), slice(3, 8), slice(0, 7, 3)),
        np_int((slice(1, 8, 2), slice(14, 2, -2), slice(3, 8), slice(0, 7, 3))),
        np_int((slice(1, 8, 2), slice(14, 2, -2), slice(3, 8), slice(0, 7, 3)), np.int64),
        (slice(1, 8, 2), 1, slice(3, 8), 2),
        np_int((slice(1, 8, 2), 1, slice(3, 8), 2)),
        np_int((slice(1, 8, 2), 1, slice(3, 8), 2), np.int64),
        # Test Ellipsis ('...')
        (1, Ellipsis, -1),
        (slice(2), Ellipsis, None, 0),
        # Test newaxis
        None,
        (1, None, -2, 3, -4),
        (1, slice(2, 5), None),
        (slice(None), slice(1, 4), None, slice(2, 3)),
        (slice(1, 3), slice(1, 3), slice(1, 3), slice(1, 3), None),
        (slice(1, 3), slice(1, 3), None, slice(1, 3), slice(1, 3)),
        (None, slice(1, 2), 3, None),
        (1, None, 2, 3, None, None, 4),
        # Advanced indexing
        ([1, 2], slice(3, 5), None, None, [3, 4]),
        (slice(None), slice(3, 5), None, None, [2, 3], [3, 4]),
        (slice(None), slice(3, 5), None, [2, 3], None, [3, 4]),
        (None, slice(None), slice(3, 5), [2, 3], None, [3, 4]),
        [1],
        [1, 2],
        [2, 1, 3],
        [7, 5, 0, 3, 6, 2, 1],
        np.array([6, 3], dtype=np.int32),
        np.array([[3, 4], [0, 6]], dtype=np.int32),
        np.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int32),
        np.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int64),
        np.array([[2], [0], [1]], dtype=np.int32),
        np.array([[2], [0], [1]], dtype=np.int64),
        np.array([4, 7], dtype=np.int32),
        np.array([4, 7], dtype=np.int64),
        np.array([[3, 6], [2, 1]], dtype=np.int32),
        np.array([[3, 6], [2, 1]], dtype=np.int64),
        np.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int32),
        np.array([[7, 3], [2, 6], [0, 5], [4, 1]], dtype=np.int64),
        (1, [2, 3]),
        (1, [2, 3], np.array([[3], [0]], dtype=np.int32)),
        (1, [2, 3]),
        (1, [2, 3], np.array([[3], [0]], dtype=np.int64)),
        (1, [2], np.array([[5], [3]], dtype=np.int32), slice(None)),
        (1, [2], np.array([[5], [3]], dtype=np.int64), slice(None)),
        (1, [2, 3], np.array([[6], [0]], dtype=np.int32), slice(2, 5)),
        (1, [2, 3], np.array([[6], [0]], dtype=np.int64), slice(2, 5)),
        (1, [2, 3], np.array([[4], [7]], dtype=np.int32), slice(2, 5, 2)),
        (1, [2, 3], np.array([[4], [7]], dtype=np.int64), slice(2, 5, 2)),
        (1, [2], np.array([[3]], dtype=np.int32), slice(None, None, -1)),
        (1, [2], np.array([[3]], dtype=np.int64), slice(None, None, -1)),
        (1, [2], np.array([[3]], dtype=np.int32), np.array([[5, 7], [2, 4]], dtype=np.int64)),
        (1, [2], np.array([[4]], dtype=np.int32), np.array([[1, 3], [5, 7]], dtype='int64')),
        [0],
        [0, 1],
        [1, 2, 3],
        [2, 0, 5, 6],
        ([1, 1], [2, 3]),
        ([1], [4], [5]),
        ([1], [4], [5], [6]),
        ([[1]], [[2]]),
        ([[1]], [[2]], [[3]], [[4]]),
        (slice(0, 2), [[1], [6]], slice(0, 2), slice(0, 5, 2)),
        ([[[[1]]]], [[1]], slice(0, 3), [1, 5]),
        ([[[[1]]]], 3, slice(0, 3), [1, 3]),
        ([[[[1]]]], 3, slice(0, 3), 0),
        ([[[[1]]]], [[2], [12]], slice(0, 3), slice(None)),
        ([1, 2], slice(3, 5), [2, 3], [3, 4]),
        ([1, 2], slice(3, 5), (2, 3), [3, 4]),
        range(4),
        range(3, 0, -1),
        (range(4,), [1]),
        (1, 1, slice(None), 1),
        (1, 1, slice(None, 3), 1),
        (1, 1, slice(None, 8, 3), 1),
    ]
    for index in index_list:
        test_getitem(np_array, index)
        test_setitem(np_array, index)
        test_getitem_autograd(np_array, index)
        test_setitem_autograd(np_array, index)

    # Test indexing to zero-size tensors
    index_list = [
        (slice(0, 0), slice(0, 0), 1, 2),
        (slice(0, 0), slice(0, 0), slice(0, 0), slice(0, 0)),
    ]
    for index in index_list:
        test_getitem(np_array, index)
        test_setitem(np_array, index)
        test_getitem_autograd(np_array, index)
        test_setitem_autograd(np_array, index)

    # test zero-size tensors get and setitem
    shapes_indices = [
        ((0), [slice(None, None, None)]),
        ((3, 0), [2, (slice(None, None, None)), (slice(None, None, None), None)]),
    ]
    for shape, indices in shapes_indices:
        np_array = _np.zeros(shape)
        for index in indices:
            test_getitem(np_array, index)
            test_setitem(np_array, index)
            test_getitem_autograd(np_array, index)
            test_setitem_autograd(np_array, index)
    test_getitem_slice_bound()


@use_np
@pytest.mark.parametrize('load_fn', [_np.load, npx.load])
def test_np_save_load_large_ndarrays(load_fn, tmp_path):
    weight = mx.np.arange(32768 * 512).reshape((32768, 512))
    mx.npx.savez(str(tmp_path / 'params.npz'), weight=weight)
    arr_loaded = load_fn(str(tmp_path / 'params.npz'))['weight']
    assert _np.array_equal(arr_loaded.asnumpy() if load_fn is npx.load
                           else arr_loaded, weight)


@use_np
@pytest.mark.serial
@pytest.mark.parametrize('load_fn', [_np.load, npx.load])
def test_np_save_load_ndarrays(load_fn):
    shapes = [(2, 0, 1), (0,), (), (), (0, 4), (), (3, 0, 0, 0), (2, 1), (0, 5, 0), (4, 5, 6), (0, 0, 0)]
    array_list = [_np.random.randint(0, 10, size=shape) for shape in shapes]
    array_list = [np.array(arr, dtype=arr.dtype) for arr in array_list]
    # test save/load single ndarray to npy format
    for i, arr in enumerate(array_list):
        with TemporaryDirectory() as work_dir:
            fname = os.path.join(work_dir, 'dataset.npy')
            npx.save(fname, arr)
            arr_loaded = load_fn(fname)
            assert _np.array_equal(arr_loaded.asnumpy() if load_fn is npx.load
                                   else arr_loaded, array_list[i].asnumpy())

    # test save/load a list of ndarrays
    with TemporaryDirectory() as work_dir:
        fname = os.path.join(work_dir, 'dataset.npz')
        npx.savez(fname, *array_list)
        if load_fn is _np.load:
            with load_fn(fname) as array_dict_loaded:  # Ensure NPZFile is closed
                array_list_loaded = [
                    array_dict_loaded['arr_{}'.format(str(i))]
                    for i in range(len(array_dict_loaded))
                ]
        else:
            array_dict_loaded = load_fn(fname)
            array_list_loaded = [
                array_dict_loaded['arr_{}'.format(str(i))]
                for i in range(len(array_dict_loaded))
            ]
        assert len(array_list) == len(array_list_loaded)
        assert all(isinstance(arr, np.ndarray) for arr in arr_loaded)
        for a1, a2 in zip(array_list, array_list_loaded):
            assert _np.array_equal(a1.asnumpy(), a2.asnumpy() if load_fn is npx.load else a2)

    # test save/load a dict of str->ndarray
    arr_dict = {}
    keys = [str(i) for i in range(len(array_list))]
    for k, v in zip(keys, array_list):
        arr_dict[k] = v
    with TemporaryDirectory() as work_dir:
        fname = os.path.join(work_dir, 'dataset.npz')
        npx.savez(fname, **arr_dict)
        if load_fn is _np.load:
            with load_fn(fname) as arr_dict_loaded:  # Ensure NPZFile is closed
                assert isinstance(arr_dict_loaded, _np.lib.npyio.NpzFile)
                assert len(arr_dict_loaded) == len(arr_dict)
                for k, v in arr_dict_loaded.items():
                    assert k in arr_dict
                    assert _np.array_equal(v.asnumpy() if load_fn is npx.load else v, arr_dict[k].asnumpy())
        else:
            arr_dict_loaded = load_fn(fname)
            assert isinstance(arr_dict_loaded, dict)
            assert len(arr_dict_loaded) == len(arr_dict)
            for k, v in arr_dict_loaded.items():
                assert k in arr_dict
                assert _np.array_equal(v.asnumpy() if load_fn is npx.load else v, arr_dict[k].asnumpy())


@retry(5)
@use_np
@pytest.mark.serial
def test_np_multinomial():
    pvals_list = [[0.0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0.0]]
    sizes = [None, (), (3,), (2, 5, 7), (4, 9)]
    experiements = 10000
    for pvals_mx_np_array in [False, True]:
        for have_size in [False, True]:
            for pvals in pvals_list:
                if pvals_mx_np_array:
                    pvals = mx.np.array(pvals)
                if have_size:
                    for size in sizes:
                        freq = mx.np.random.multinomial(experiements, pvals, size=size).asnumpy() / _np.float32(experiements)
                        # for those cases that didn't need reshape
                        if size in [None, ()]:
                            if type(pvals) == np.ndarray:
                                mx.test_utils.assert_almost_equal(freq, pvals.asnumpy(), rtol=0.20, atol=1e-1)
                            else:
                                mx.test_utils.assert_almost_equal(freq, pvals, rtol=0.20, atol=1e-1)
                        else:
                            # check the shape
                            assert freq.shape == size + (len(pvals),), 'freq.shape={}, size + (len(pvals))={}'.format(freq.shape, size + (len(pvals)))
                            freq = freq.reshape((-1, len(pvals)))
                            # check the value for each row
                            for i in range(freq.shape[0]):
                                if type(pvals) == np.ndarray:
                                    mx.test_utils.assert_almost_equal(freq[i, :], pvals.asnumpy(), rtol=0.20, atol=1e-1)
                                else:
                                    mx.test_utils.assert_almost_equal(freq[i, :], pvals, rtol=0.20, atol=1e-1)
                else:
                    freq = mx.np.random.multinomial(experiements, pvals).asnumpy() / _np.float32(experiements)
                    if type(pvals) == np.ndarray:
                        mx.test_utils.assert_almost_equal(freq, pvals.asnumpy(), rtol=0.20, atol=1e-1)
                    else:
                        mx.test_utils.assert_almost_equal(freq, pvals, rtol=0.20, atol=1e-1)
    # check the zero dimension
    sizes = [(0), (0, 2), (4, 0, 2), (3, 0, 1, 2, 0)]
    for pvals_mx_np_array in [False, True]:
        for pvals in pvals_list:
            for size in sizes:
                if pvals_mx_np_array:
                    pvals = mx.np.array(pvals)
                freq = mx.np.random.multinomial(experiements, pvals, size=size).asnumpy()
                assert freq.size == 0
    # check [] as pvals
    for pvals_mx_np_array in [False, True]:
        for pvals in [[], ()]:
            if pvals_mx_np_array:
                pvals = mx.np.array(pvals)
            freq = mx.np.random.multinomial(experiements, pvals).asnumpy()
            assert freq.size == 0
            for size in sizes:
                freq = mx.np.random.multinomial(experiements, pvals, size=size).asnumpy()
                assert freq.size == 0
    # test small experiment for github issue
    # https://github.com/apache/incubator-mxnet/issues/15383
    small_exp, total_exp = 20, 10000
    for pvals_mx_np_array in [False, True]:
        for pvals in pvals_list:
            if pvals_mx_np_array:
                pvals = mx.np.array(pvals)
            x = np.random.multinomial(small_exp, pvals)
            for i in range(total_exp // small_exp):
                x = x + np.random.multinomial(20, pvals)
        freq = (x.asnumpy() / _np.float32(total_exp)).reshape((-1, len(pvals)))
        for i in range(freq.shape[0]):
            if type(pvals) == np.ndarray:
                mx.test_utils.assert_almost_equal(freq[i, :], pvals.asnumpy(), rtol=0.20, atol=1e-1)
            else:
                mx.test_utils.assert_almost_equal(freq[i, :], pvals, rtol=0.20, atol=1e-1)


@pytest.mark.skipif(not is_op_runnable(), reason="Comparison ops can only run on either CPU instances, or GPU instances with"
                                                 " compute capability >= 53 if MXNet is built with USE_TVM_OP=ON")
@use_np
def test_boolean_index_single():
    # adapted from numpy's test_indexing.py
    # Single boolean index
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=np.int32)
    assert same(a[np.array(True, dtype=np.bool_)].asnumpy(), a[None].asnumpy())
    assert same(a[np.array(False, dtype=np.bool_)].asnumpy(), a[None][0:0].asnumpy())

@pytest.mark.skipif(not is_op_runnable(), reason="Comparison ops can only run on either CPU instances, or GPU instances with"
                                                 " compute capability >= 53 if MXNet is built with USE_TVM_OP=ON")
@use_np
def test_boolean_index_catch_exception():
    # adapted from numpy's test_indexing.py
    arr = np.ones((5, 4, 3))

    index = np.array([True], dtype=np.bool_)
    assert_exception(arr.__getitem__, IndexError, index)

    index = np.array([False] * 6, dtype=np.bool_)
    assert_exception(arr.__getitem__, IndexError, index)

    index = np.zeros((4, 4), dtype=bool)
    assert_exception(arr.__getitem__, IndexError, index)

@pytest.mark.skipif(not is_op_runnable(), reason="Comparison ops can only run on either CPU instances, or GPU instances with"
                                                 " compute capability >= 53 if MXNet is built with USE_TVM_OP=ON")
@use_np
def test_boolean_index_onedim():
    # adapted from numpy's test_indexing.py
    # Indexing a 2-dimensional array with
    # boolean array of length one
    a = np.array([[0.,  0.,  0.]])
    b = np.array([True], dtype=bool)
    assert same(a[b].asnumpy(), a.asnumpy())

@pytest.mark.skipif(not is_op_runnable(), reason="Comparison ops can only run on either CPU instances, or GPU instances with"
                                                 " compute capability >= 53 if MXNet is built with USE_TVM_OP=ON")
@use_np
def test_boolean_index_twodim():
    # adapted from numpy's test_indexing.py
    # Indexing a 2-dimensional array with
    # 2-dimensional boolean array
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=np.int32)
    b = np.array([[ True, False,  True],
                  [False,  True, False],
                  [ True, False,  True]], dtype=np.bool_)
    assert same(a[b].asnumpy(), _np.array([1, 3, 5, 7, 9], dtype=a.dtype))
    assert same(a[b[1]].asnumpy(), _np.array([[4, 5, 6]], dtype=a.dtype))
    assert same(a[b[0]].asnumpy(), a[b[2]].asnumpy())

@pytest.mark.skipif(not is_op_runnable(), reason="Comparison ops can only run on either CPU instances, or GPU instances with"
                                                 " compute capability >= 53 if MXNet is built with USE_TVM_OP=ON")
@use_np
def test_boolean_index_list():
    # adapted from numpy's test_indexing.py
    a = np.array([1, 2, 3], dtype=np.int32)
    b = [True, False, True]
    # Two variants of the test because the first takes a fast path
    assert same(a[b].asnumpy(), _np.array([1, 3], dtype=a.dtype))
    (a[None, b], [[1, 3]])

@pytest.mark.skipif(not is_op_runnable(), reason="Comparison ops can only run on either CPU instances, or GPU instances with"
                                                 " compute capability >= 53 if MXNet is built with USE_TVM_OP=ON")
@use_np
def test_boolean_index_tuple():
    # case arr[:, mask, :] and arr[1, mask, 0]
    # when a boolean array is in a tuple
    a = np.array([[[0, 1],
                   [2, 3]],
                  [[4, 5],
                   [6, 7]]], dtype=np.int32)
    b = np.array([[False,True],
                  [True,False]],dtype=np.bool)
    _np_a = a.asnumpy()
    _np_b = b.asnumpy()
    assert same(a[:, b].asnumpy(), _np_a[:, _np_b])
    assert same(a[b, :].asnumpy(), _np_a[_np_b, :])
    assert same(a[0, b].asnumpy(), _np_a[0, _np_b])
    assert same(a[b, 1].asnumpy(), _np_a[_np_b, 1])

    a = np.arange(12).reshape(4,3)
    b = np.array([1.,2.,3.])
    _np_a = a.asnumpy()
    _np_b = b.asnumpy()
    assert same(a[:, b > 2].shape, _np_a[:, _np_b > 2].shape)
    assert same(a[:, b > 2].asnumpy(), _np_a[:, _np_b > 2])

    a = np.array([[1,2,3],[3,4,5]])
    _np_a = a.asnumpy()
    assert same(a[:,a[1,:] > 0].shape, _np_a[:,_np_a[1,: ] > 0].shape)
    assert same(a[:,a[1,:] > 0].asnumpy(), _np_a[:,_np_a[1,: ] > 0])

    a = np.ones((3,2), dtype='bool')
    b = np.array([1,2,3])
    _np_a = a.asnumpy()
    _np_b = b.asnumpy()
    assert same(a[b > 1].asnumpy(), _np_a[_np_b > 1])

@pytest.mark.skipif(not is_op_runnable(), reason="Comparison ops can only run on either CPU instances, or GPU instances with"
                                                 " compute capability >= 53 if MXNet is built with USE_TVM_OP=ON")
@use_np
@pytest.mark.xfail(reason='Flaky boolean index assign. See #18334')
def test_boolean_index_assign():
    # test boolean indexing assign
    shape = (3, 2, 3)
    mx_data = np.random.uniform(size=shape)
    mx_mask = np.array([[False,True], [True,False], [True,False]],dtype=np.bool)
    np_data = mx_data.asnumpy()
    np_mask = mx_mask.asnumpy()

    np_data[np_data>0.5] = 0
    mx_data[mx_data>0.5] = 0
    assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)
    np_data[np_mask] = 1
    mx_data[mx_mask] = 1
    assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)

    np_data[np_mask, 1] = 2
    mx_data[mx_mask, 1] = 2
    assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)

    np_data[np_mask, :] = 3
    mx_data[mx_mask, :] = 3
    assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)

    mx_mask = np.array([[False,True, True],[False, True,False]],dtype=np.bool)
    np_mask = mx_mask.asnumpy()

    np_data[0, np_mask] = 5
    mx_data[0, mx_mask] = 5
    assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)
    np_data[:, np_mask] = 6
    mx_data[:, mx_mask] = 6
    assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)

    np_data[0, True, True, np_mask] = 7
    mx_data[0, True, True, mx_mask] = 7
    assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)

    np_data[False, 1] = 8
    mx_data[False, 1] = 8
    assert_almost_equal(mx_data.asnumpy(), np_data, rtol=1e-3, atol=1e-5, use_broadcast=False)

@pytest.mark.skipif(not is_op_runnable(), reason="Comparison ops can only run on either CPU instances, or GPU instances with"
                                                 " compute capability >= 53 if MXNet is built with USE_TVM_OP=ON")
@use_np
def test_boolean_index_autograd():
    a = np.random.uniform(size=(3, 4, 5))
    a.attach_grad()
    with mx.autograd.record():
        out_mx = a[a < 0.5]
    out_mx.backward()

    a_np = a.asnumpy()
    out_np = a_np[a_np < 0.5]
    assert_almost_equal(out_mx.asnumpy(), out_np, rtol=1e-4, atol=1e-5, use_broadcast=False)

    a_grad_np = _np.zeros(a.shape, dtype=a.dtype)
    a_grad_np[a_np < 0.5] = 1
    assert_almost_equal(a.grad.asnumpy(), a_grad_np, rtol=1e-4, atol=1e-5, use_broadcast=False)


@use_np
def test_np_get_dtype():
    dtypes = [_np.int8, _np.int32, _np.float16, _np.float32, _np.float64, _np.bool, _np.bool_,
              'int8', 'int32', 'float16', 'float32', 'float64', 'bool', None]
    objects = [
        [],
        (),
        [[1, 2], [3, 4]],
        _np.random.uniform(size=rand_shape_nd(3)),
        _np.random.uniform(size=(3, 0, 4))
    ]
    for dtype in dtypes:
        for src in objects:
            mx_arr = np.array(src, dtype=dtype)
            assert mx_arr.ctx == mx.current_context()
            if isinstance(src, mx.nd.NDArray):
                np_arr = _np.array(src.asnumpy(), dtype=dtype if dtype is not None else _np.float32)
            else:
                np_arr = _np.array(src, dtype=dtype if dtype is not None else _np.float32)
            assert type(mx_arr.dtype) == type(np_arr.dtype)


@use_np
def test_np_ndarray_pickle():
    a = np.random.uniform(size=(4, 5))
    a_copy = a.copy()
    import pickle

    with TemporaryDirectory() as work_dir:
        fname = os.path.join(work_dir, 'np_ndarray_pickle_test_file')
        with open(fname, 'wb') as f:
            pickle.dump(a_copy, f)
        with open(fname, 'rb') as f:
            a_load = pickle.load(f)
        same(a.asnumpy(), a_load.asnumpy())

@pytest.mark.parametrize('dtype', [np.float32, np.int32])
@pytest.mark.parametrize('size', [
    (3, 4, 5, 6),
    (2, 10),
    (15,),
    ()
])
@use_np
def test_dlpack(dtype, size):
    a = mx.np.random.uniform(size=size)
    a_np = a.copy()
    a += 1

    pack = mx.npx.to_dlpack_for_read(a)
    b = mx.npx.from_dlpack(pack)

    a_copy = a.copy()
    pack2 = mx.npx.to_dlpack_for_write(a_copy)
    c = mx.npx.from_dlpack(pack2)
    c += 1

    del a, pack, pack2

    same(a_np+1, b)
    same(a_np+2, c)
    same(a_np+2, a_copy)

@use_np
@pytest.mark.parametrize('np_array', [
    # ordinary numpy array
    _np.array([[1, 2], [3, 4], [5, 6]], dtype="float32"),
    # 0-dim
    _np.array((1, )).reshape(()),
    # 0-size
    _np.array(()).reshape((1, 0, 2)),
])
@pytest.mark.parametrize('zero_copy', [False, True])
def test_from_numpy(np_array, zero_copy):
    # Test zero_copy
    mx_array = mx.npx.from_numpy(np_array, zero_copy=zero_copy)
    mx.test_utils.assert_almost_equal(np_array, mx_array.asnumpy())

def test_from_numpy_exception():
    np_array = _np.array([[1, 2], [3, 4], [5, 6]], dtype="float32")
    mx_array = mx.npx.from_numpy(np_array)
    with pytest.raises(ValueError):
        np_array[2, 1] = 0

    mx_array[2, 1] = 100
    mx.test_utils.assert_almost_equal(np_array, mx_array.asnumpy())
    np_array = _np.array([[1, 2], [3, 4], [5, 6]]).transpose()
    assert not np_array.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        mx_array = mx.nd.from_numpy(np_array)

    np_array = _np.array([[1, 2], [3, 4], [5, 6]], dtype="float32")
    mx_array = mx.npx.from_numpy(np_array, zero_copy=False)
    np_array[2, 1] = 0 # no error

def test_mixed_array_types():
    np_array = _np.array([[1, 2], [3, 4], [5, 6]], dtype="float32")
    mx_array = mx.np.ones((3, 1))
    assert_almost_equal(mx_array + np_array, 1+np_array)

def test_mixed_array_types_share_memory():
    np_array = _np.array([[1, 2], [3, 4], [5, 6]], dtype="float32")
    mx_array = mx.npx.from_numpy(np_array)
    assert _np.may_share_memory(np_array, mx_array)
    assert _np.shares_memory(np_array, mx_array)

    np_array_slice = np_array[:2]
    mx_array_slice = mx_array[1:]
    assert _np.may_share_memory(np_array_slice, mx_array)
    assert _np.shares_memory(np_array_slice, mx_array)

    mx_pinned_array = mx_array.as_in_ctx(mx.cpu_pinned(0))
    assert not _np.may_share_memory(np_array, mx_pinned_array)
    assert not _np.shares_memory(np_array, mx_pinned_array)
