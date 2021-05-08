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

import mxnet as mx
from mxnet import np, npx
from mxnet.test_utils import same, use_np, assert_almost_equal
import random
import pytest

@use_np
@pytest.mark.parametrize('shape',
    [(3, 2), (9,17), (2, 7, 1, 8)] + [(i,) for i in range(1,65)])
def test_contrib_intgemm_maxabsolute(shape):
    if "intgemm_maxabsolute" not in dir(mx.nd.contrib):
        return
    # mx.nd API
    m = mx.nd.random_uniform(low=-100.0, high=100.0, shape=shape)
    fast = mx.nd.contrib.intgemm_maxabsolute(m)
    slow = mx.nd.max(mx.nd.abs(m))
    assert same(fast, slow)
    # np API
    m = np.random.uniform(low=-100.0, high=100.0, size=shape)
    fast = npx.intgemm_maxabsolute(m).reshape(())
    slow = np.max(np.abs(m))
    assert same(fast, slow)

@use_np
@pytest.mark.parametrize('shape', [(i,) for i in range(1, 67)] + [(2,3), (130, 12)])
@pytest.mark.parametrize('max_quant', [2.0])
def test_contrib_intgemm_prepare_data(shape, max_quant):
    if "intgemm_prepare_data" not in dir(mx.nd.contrib):
        return
    m = mx.nd.random_uniform(low=-3.0, high=3.0, shape=shape)
    scaled = m * 127.0 / max_quant
    # Rounding 0.5 can go up or down.  Move values away from 0.5.
    too_close = mx.nd.abs(mx.nd.round(scaled) - scaled) > 0.45
    # Add 0.2 in scaled space so (0.45, 0.55) maps to (0.65, 0.75) which will round consistently.
    m += max_quant / 127.0 * 0.2 * too_close

    # Reference: scale and round
    ref = mx.nd.round(m * 127.0 / max_quant)
    # Clip to [-127, 127].  Because otherwise e.g. -129 casts to +127.
    ref = mx.nd.broadcast_maximum(ref, mx.nd.array([-127.0]))
    ref = mx.nd.broadcast_minimum(ref, mx.nd.array([127.0]))
    # Reference: cast to int8
    ref = mx.nd.cast(ref, dtype='int8')
    # Reference: ban -128
    ref = mx.nd.broadcast_maximum(ref, mx.nd.array([-127], dtype = 'int8'))

    test = mx.nd.contrib.intgemm_prepare_data(m, mx.nd.array([max_quant]))
    assert same(test, ref)
    test = npx.intgemm_prepare_data(m.as_np_ndarray(), np.array([max_quant]))
    assert same(test, ref.as_np_ndarray())

@use_np  
@pytest.mark.parametrize('shape', [(8, 64), (16, 64), (8, 128), (16, 128), (2, 4, 64)])
@pytest.mark.parametrize('max_quant', [0.2, 3.0])
@pytest.mark.parametrize('api', [(mx.nd.contrib, mx.nd), (npx, np)])
def test_contrib_intgemm_weight_consistent(shape, max_quant, api):
    # The weight format is actually CPU-dependent so we don't directly test the
    # output, but indirectly test that it works.
    if "intgemm_prepare_weight" not in dir(mx.nd.contrib):
        return
    contrib, top = api
    max_array = top.array([max_quant])
    if top == mx.nd:
        m = top.random_uniform(low=-3.0, high=3.0, shape=shape)
    else:
        m = np.random.uniform(size=shape)
    direct = contrib.intgemm_prepare_weight(m, max_array)
    quant = contrib.intgemm_prepare_data(m, max_array) 
    indirect = contrib.intgemm_prepare_weight(quant, already_quantized=True)
    # Should get the same data from direct call and already_quantized version.
    assert same(direct, indirect)
    
@use_np
@pytest.mark.parametrize('indices', [
        [0,1,2,3,4,5,6,7],
        [1,2,1,2,1,2,1,2],
        [7,6,5,4,3,2,1,0],
        [3,1,4,1,5,9,2,6],
        # Since random_uniform doesn't support int8, use python
        [random.randint(0,15) for i in range(8)],
        [random.randint(0,15) for i in range(16)],
        [random.randint(0,15) for i in range(24)]
    ])
@pytest.mark.parametrize('api', [(mx.nd.contrib, mx.nd), (npx, np)])
def test_contrib_intgemm_take_weight(indices, api):
    if "intgemm_take_weight" not in dir(mx.nd.contrib):
        return
    contrib, top = api
    m = top.array([random.randint(-127,127) for i in range(16 * 64)], dtype='int8')
    m = m.reshape((16, 64))
    indices = top.array(indices, dtype='int32')
    # Prepare weight then take.
    test = contrib.intgemm_prepare_weight(m, already_quantized=True)
    test = contrib.intgemm_take_weight(test, indices)
    # Take then prepare.
    ref = m.take(indices, axis=0)
    ref = contrib.intgemm_prepare_weight(ref, already_quantized=True)
    assert same(test, ref)

@use_np
@pytest.mark.parametrize('data_rows', range(1, 5))
@pytest.mark.parametrize('inner', range(64, 256, 64))
@pytest.mark.parametrize('weight_cols', range(8, 24, 8))
@pytest.mark.parametrize('api', [
    (mx.nd.contrib, mx.nd, mx.nd.FullyConnected, mx.nd.cast),
    (npx, np, npx.fully_connected, mx.nd.np._api_internal.cast)])
def test_contrib_intgemm_multiply(data_rows, inner, weight_cols, api):
    if "intgemm_fully_connected" not in dir(mx.nd.contrib):
        return
    contrib, top, fully_connected, cast_api = api
    def cast(a, dtype=None):
        return cast_api(a, dtype)
    #The multiplication routine has approximations so everything is tested
    #deterministically to ensure bounds are met.
    random.seed(1)

    # Don't use full range (-127, 127) to avoid saturation.
    data = [random.randint(-64, 64) for i in range(data_rows * inner)]
    data = top.array(data, dtype='int8').reshape((data_rows, inner))
    weight = [random.randint(-64, 64) for i in range(inner * weight_cols)]
    weight = top.array(weight, dtype='int8').reshape((weight_cols, inner))
    weight_prepared = contrib.intgemm_prepare_weight(weight, already_quantized=True)

    # int32 output, no bias
    test = contrib.intgemm_fully_connected(data,
                                           weight_prepared,
                                           no_bias=True,
                                           flatten=False,
                                           out_type='int32',
                                           num_hidden=weight_cols)
    ref = fully_connected(cast(data, dtype='float32'),
                          cast(weight, dtype='float32'),
                          no_bias=True,
                          flatten=False,
                          num_hidden=weight_cols)
    assert_almost_equal(cast(test, dtype='float32'), ref, rtol=0.01, atol=0.01)

    # float32 output, no bias
    scale = 3.0
    test = contrib.intgemm_fully_connected(data,
                                           weight_prepared,
                                           top.array([scale]),
                                           no_bias=True,
                                           flatten=False,
                                           out_type='float32',
                                           num_hidden=weight_cols)
    assert_almost_equal(test, ref * scale, rtol=0.01, atol=0.01)

    # int32 output, bias
    bias = top.array([random.randint(-60000, 60000) for i in range(weight_cols)], dtype = 'int32')
    test = contrib.intgemm_fully_connected(data,
                                           weight_prepared,
                                           bias,
                                           no_bias=False,
                                           flatten=False,
                                           out_type='int32',
                                           num_hidden=weight_cols)
    ref = fully_connected(cast(data, dtype='float32'),
                               cast(weight, dtype='float32'),
                               cast(bias, dtype='float32'),
                               no_bias=False,
                               flatten=False,
                               num_hidden=weight_cols)
    assert_almost_equal(cast(test, dtype='float32'), ref, rtol=0.01, atol=0.01)

    # float32 output, bias
    # Scaling is applied before bias (and bias is not scaled). So to make the
    # reference comparison easy, just scale the bias beforehand.
    test = contrib.intgemm_fully_connected(data,
                                           weight_prepared,
                                           top.array([scale]),
                                           cast(bias, dtype='float32') * scale,
                                           no_bias=False,
                                           flatten=False,
                                           out_type='float32',
                                           num_hidden=weight_cols)
    assert_almost_equal(test, ref * scale, rtol=0.01, atol=0.01)

    # float32 input should work the same as manually prepared int8 input.
    data_float = top.array([random.uniform(-3.14, 3.14) for i in range(data_rows * inner)])
    data_float = data_float.reshape(data_rows, inner)
    direct = contrib.intgemm_fully_connected(data_float,
                                             weight_prepared,
                                             top.array([scale]),
                                             cast(bias, dtype='float32'),
                                             no_bias=False,
                                             flatten=False,
                                             out_type='float32',
                                             num_hidden=weight_cols)
    maxabs = contrib.intgemm_maxabsolute(data_float)
    data_prepared = contrib.intgemm_prepare_data(data_float, maxabs)
    cooked = contrib.intgemm_fully_connected(data_prepared,
                                             weight_prepared,
                                             top.array(scale * maxabs / 127.0),
                                             cast(bias, dtype='float32'),
                                             no_bias=False,
                                             flatten=False,
                                             out_type='float32',
                                             num_hidden=weight_cols)
    assert_almost_equal(direct, cooked, rtol=0.01, atol=0.01)
