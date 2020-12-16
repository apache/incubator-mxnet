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
from mxnet.gluon import HybridBlock, nn
import numpy as np
import onnxruntime as rt
from mxnet.test_utils import assert_almost_equal
import pytest
import tempfile

def def_model(op_name, **params):
    class Model(HybridBlock):
        def __init__(self, **kwargs):
            super(Model, self).__init__(**kwargs)

        def hybrid_forward(self, F, *inputs):
            names = op_name.split('.')
            func = F
            for name in names:
                func = getattr(func, name)
            out = func(*inputs, **params)
            return out
    return Model

def op_export_test(model_name, Model, inputs, tmp_path):
    def export_to_onnx(model, model_name, inputs):
        model_path = '{}/{}'.format(tmp_path, model_name)
        model.export(model_path, epoch=0)
        sym_file = '{}-symbol.json'.format(model_path)
        params_file = '{}-0000.params'.format(model_path)
        dtype = inputs[0].dtype
        onnx_file = '{}/{}.onnx'.format(tmp_path, model_name)
        mx.contrib.onnx.export_model(sym_file, params_file, [inp.shape for inp in inputs],
                                     dtype, onnx_file)
        return onnx_file

    def onnx_rt(onnx_file, inputs):
        sess = rt.InferenceSession(onnx_file)
        input_dict = dict((sess.get_inputs()[i].name, inputs[i].asnumpy()) for i in range(len(inputs)))
        pred = sess.run(None, input_dict)[0]
        return pred

    # create a new model 
    model = Model()
    model.initialize(ctx=mx.cpu(0))
    model.hybridize()
    pred_nat = model(*inputs)
    onnx_file = export_to_onnx(model, model_name, inputs)
    pred_onx = onnx_rt(onnx_file, inputs)
    assert_almost_equal(pred_nat, pred_onx)


def test_onnx_export_abs(tmp_path):
    M = def_model('abs')
    x = mx.nd.array([[-2, -1], [0, 99]], dtype='float32')
    op_export_test('abs', M, [x], tmp_path)


def test_onnx_export_slice(tmp_path):
    M = def_model('slice', begin=(0,1), end=(2,4))
    x = mx.nd.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype='float32')
    op_export_test('slice', M, [x], tmp_path)


def test_onnx_export_stack(tmp_path):
    M = def_model('stack')
    x = mx.nd.array([1, 2], dtype='float32')
    y = mx.nd.array([3, 4], dtype='float32')
    op_export_test('stack', M, [x, y], tmp_path)


def test_onnx_export_zeros_like(tmp_path):
    M = def_model('zeros_like')
    x = mx.nd.array([[-2,-1,0],[0,50,99],[4,5,6],[7,8,9]], dtype='float32')
    op_export_test('zeros_like', M, [x], tmp_path)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_onnx_export_arange_like(tmp_path, dtype):
    M = def_model('contrib.arange_like')
    x = mx.nd.array([[-2,-1,0],[0,50,99],[4,5,6],[7,8,9]], dtype=dtype)
    op_export_test('arange_like', M, [x], tmp_path)


def test_onnx_export_layernorm(tmp_path):
    M = def_model('LayerNorm', axis=1)
    x = mx.nd.array([[1,3],[2,4]], dtype='float32')
    gamma = mx.random.uniform(0, 1, x[0].shape, dtype='float32')
    beta = mx.random.uniform(0, 1, x[0].shape, dtype='float32')
    op_export_test('LayerNorm', M, [x, gamma, beta], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32'])
def test_onnx_export_broadcast_axis(tmp_path, dtype):
    M1 = def_model('broadcast_axis', axis=(0, 2), size=(3, 4))
    M2 = def_model('broadcast_axis', axis=(0, 2), size=(1, 5))
    x1 = mx.nd.array([[[1], [2]]], dtype=dtype)
    op_export_test('broadcast_axis_1', M1, [x1], tmp_path)
    op_export_test('broadcast_axis_2', M2, [x1], tmp_path)
    M3 = def_model('broadcast_axis', axis=(1, 4), size=(3, 5))
    x2 = mx.nd.ones((1, 1, 3, 1, 1, 1), dtype=dtype)
    op_export_test('broadcast_axis_3', M3, [x2], tmp_path)


@pytest.mark.parametrize('dtype', ['float32'])
def test_onnx_export_SequenceMask(tmp_path, dtype):
    M1 = def_model('SequenceMask', use_sequence_length=True, axis=1, value=-5)
    M2 = def_model('SequenceMask', use_sequence_length=True, axis=0, value=-99)
    x = mx.nd.array([[[[  1.,   2.,   3.,  3.5]],
                      [[  4.,   5.,   6.,  6.5]]],
                     [[[  7.,   8.,   9.,  9.5]],
                      [[ 10.,  11.,  12., 12.5]]],
                     [[[ 13.,  14.,  15., 15.5]],
                      [[ 16.,  17.,  18., 18.5]]]], dtype=dtype)
    seq_len1 = mx.nd.array([1, 2, 1], dtype=dtype)
    seq_len2 = mx.nd.array([1, 2], dtype=dtype)
    op_export_test('SequenceMask_1', M1, [x, seq_len1], tmp_path)
    op_export_test('SequenceMask_2', M2, [x, seq_len2], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32'])
def test_onnx_export_contrib_interleaved_matmul_selfatt_qk(tmp_path, dtype):
    M1 = def_model('contrib.interleaved_matmul_selfatt_qk', heads=3)
    x1 = mx.nd.random.uniform(0, 1, (3, 3, 3*3*3))
    op_export_test('contrib_interleaved_matmul_selfatt_qk_1', M1, [x1], tmp_path)
    M2 = def_model('contrib.interleaved_matmul_selfatt_qk', heads=5)
    x2 = mx.nd.random.uniform(0, 1, (7, 5, 4*5*6))
    op_export_test('contrib_interleaved_matmul_selfatt_qk_2', M2, [x2], tmp_path)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
@pytest.mark.parametrize('num_hidden', [1, 5, 10, 20])
@pytest.mark.parametrize('no_bias', [False, True])
@pytest.mark.parametrize('flatten', [True, False])
def test_onnx_export_fully_connected(tmp_path, dtype, num_hidden, no_bias, flatten):
    M = def_model('FullyConnected', num_hidden=num_hidden, no_bias=no_bias, flatten=flatten)
    x = mx.nd.random.uniform(-0.5, 0.5, (5, 325))
    weight = mx.nd.random.uniform(0, 1, (num_hidden, 325))
    args = [x, weight]
    if not no_bias:
        args.append(mx.nd.random.uniform(0,1,(num_hidden,)))
    op_export_test('FullyConnected', M, args, tmp_path)


