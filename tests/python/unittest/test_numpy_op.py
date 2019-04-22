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
from mxnet import numpy as np
from mxnet.gluon import HybridBlock
from mxnet.test_utils import same, assert_almost_equal, rand_shape_nd, rand_ndarray
from mxnet.test_utils import check_numeric_gradient
from common import with_seed
import random


@mx.use_np_compat
@with_seed()
def test_np_sum():
    class TestSum(HybridBlock):
        def __init__(self, axis=None, dtype=None, keepdims=False):# , initial=None):
            super(TestSum, self).__init__()
            self._axis = axis
            self._dtype = dtype
            self._keepdims = keepdims

        def hybrid_forward(self, F, a, *args, **kwargs):
            return F.numpy.sum(a, axis=self._axis, dtype=self._dtype, keepdims=self._keepdims)

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
                        x.attach_grad()
                        expected_ret = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims)
                        expected_ret = expected_ret.astype(dtype)
                        with mx.autograd.record():
                            y = test_sum(x)
                        assert y.shape == expected_ret.shape
                        assert_almost_equal(y.asnumpy(), expected_ret, rtol=1e-3 if dtype == 'float16' else 1e-3,
                                            atol=1e-5 if dtype == 'float16' else 1e-5)

                        y.backward()
                        assert same(x.grad.asnumpy(), _np.ones(shape=x.shape, dtype=x.dtype))

                        # test numeric
                        if itype == 'float32' and dtype == 'float32':
                            x_sym = mx.sym.Variable("x")
                            mx_sym = mx.sym.numpy.sum(x_sym, axis=axis, dtype=dtype, keepdims=keepdims)
                            check_numeric_gradient(mx_sym, [x], numeric_eps=1e-3, rtol=1e-3, atol=1e-4, dtype=_np.float32)

                        # test imperative
                        mx_out = np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
                        np_out = _np.sum(x.asnumpy(), axis=axis, dtype=acc_type[itype], keepdims=keepdims).astype(dtype)
                        assert_almost_equal(mx_out.asnumpy(), np_out, rtol=1e-3, atol=1e-5)


@mx.use_np_compat
@with_seed()
def test_np_dot():
    shapes = [
        ((3,), (3,)),        # Case 1
        ((3, 4), (4, 5)),    # Case 2
        ((), ()),            # Case 3
        ((3, 4, 5), ()),     # Case 3.5.1
        ((), (3, 4, 5)),     # Case 3.5.2
        ((3, 4, 5), (5, )),  # Case 4
    ]

    eps = 1e-3

    for shape_a, shape_b in shapes:
        print(shape_a, shape_b)
        np_a = _np.random.uniform(-1.0, 1.0, shape_a)
        np_a[abs(np_a) < eps] = 2 * eps;
        np_b = _np.random.uniform(-1.0, 1.0, shape_b)
        np_b[abs(np_b) < eps] = 2 * eps;
        a = mx.nd.array(np_a)
        b = mx.nd.array(np_b)
        np_res = _np.dot(np_a, np_b)
        mx_res = np.dot(a, b)
        assert mx_res.shape == np_res.shape
        assert_almost_equal(np_res, mx_res.asnumpy(), rtol=1e-5, atol=1e-5)
        mx_a = mx.sym.Variable("a")
        mx_b = mx.sym.Variable("b")
        mx_sym = mx.sym.numpy.dot(mx_a, mx_b)
        check_numeric_gradient(mx_sym, {"a": a, "b": b}, numeric_eps=eps, rtol=1e-2, atol=1e-3)

    bad_shapes = [((4, 5), (2, 3)), ((3, 4, 5), (6, ))]

    for shape_a, shape_b in bad_shapes:
        a = mx.nd.array(random.random()) if len(shape_a) == 0 else rand_ndarray(shape_a)
        b = mx.nd.array(random.random()) if len(shape_b) == 0 else rand_ndarray(shape_b)
        try:
            mx_res = np.dot(a, b)
        except mx.base.MXNetError:
            continue
        assert False

@mx.use_np_compat
@with_seed()
def test_np_unary_funcs():
    func_names = ['abs', 'cbrt', 'ceil', 'exp', 'expm1', 'fix', 'floor', 'log', 'log10',
                  'log1p', 'log2', 'logical_not', 'negative', 'reciprocal', 'rint', 'sign',
                  'sqrt', 'square', 'trunc', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
                  'degrees', 'radians', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh']
    ndim = random.choice([2, 3, 4])
    shape = rand_shape_nd(ndim, dim=3)
    mx_test_data = rand_ndarray(shape)
    np_test_data = mx_test_data.asnumpy()
    for func in func_names:
        np_func = getattr(_np, func)
        mx_func = getattr(np, func)
        np_out = np_func(np_test_data)
        mx_out = mx_func(mx_test_data)
        assert_almost_equal(np_out, mx_out.asnumpy(), rtol=1e-5, atol=1e-5, equal_nan=True)


@with_seed()
def test_np_clip():
    ndim = random.choice([1, 2, 3])
    shape = rand_shape_nd(ndim, dim=5)
    mx_test_data = rand_ndarray(shape)
    np_test_data = mx_test_data.asnumpy()
    mx_out = np.clip(mx_test_data, -0.1, 0.1)
    np_out = _np.clip(np_test_data, -0.1, 0.1)
    assert_almost_equal(np_out, mx_out.asnumpy(), rtol=1e-5, atol=1e-5, equal_nan=True)


if __name__ == '__main__':
    import nose
    nose.runmodule()
