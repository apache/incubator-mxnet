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
from __future__ import print_function
from __future__ import division
import numpy as np
import mxnet as mx
import copy
import math
import random
import itertools
from distutils.version import LooseVersion
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
from mxnet.operator import *
from mxnet.base import py_str, MXNetError, _as_list
from common import setup_module, with_seed, teardown, assert_raises_cudnn_not_satisfied, assert_raises_cuda_not_satisfied, assertRaises
from common import run_in_spawned_process, with_environment
from nose.tools import assert_raises, ok_
import unittest
import os
import locale

def check_rnn_consistency(cell1, cell2, T, N, I, H, grad_req, rtol=1e-2, atol=1e-4):
    dshape = (N, T, I)
    data = mx.sym.Variable('data')

    Y1, _ = cell1.unroll(T, data, layout='NTC', merge_outputs=True)
    mod1 = mx.mod.Module(Y1, label_names=None, context=default_context())
    mod1.bind(data_shapes=[('data', dshape)], label_shapes=None, inputs_need_grad=True, grad_req=grad_req)

    Y2, _ = cell2.unroll(T, data, layout='NTC', merge_outputs=True)
    mod2 = mx.mod.Module(Y2, label_names=None, context=default_context())
    mod2.bind(data_shapes=[('data', dshape)], label_shapes=None, inputs_need_grad=True, grad_req=grad_req)

    mod1.init_params()
    args, auxs = mod1.get_params()
    args = cell1.unpack_weights(args)
    args = cell2.pack_weights(args)
    mod2.set_params(args, auxs)

    x = mx.random.uniform(shape=dshape)
    batch=mx.io.DataBatch(data=[x])
    # check inference
    mod1.forward(batch, is_train=False)
    mod2.forward(batch, is_train=False)
    assert_allclose(mod1.get_outputs()[0].asnumpy(), mod2.get_outputs()[0].asnumpy(), rtol=rtol, atol=atol)

    # check training
    mod1.forward(batch, is_train=True)
    mod2.forward(batch, is_train=True)
    assert_allclose(mod1.get_outputs()[0].asnumpy(), mod2.get_outputs()[0].asnumpy(), rtol=rtol, atol=atol)

    dy = mx.random.uniform(shape=mod1.get_outputs()[0].shape)
    mod1.backward(out_grads=[dy])
    mod2.backward(out_grads=[dy])
    if type(grad_req) is dict and grad_req['data'] == 'null' or grad_req == 'null':
        assert(mod1.get_input_grads()[0] == None)
        assert(mod2.get_input_grads()[0] == None)
    else:
        assert_allclose(mod1.get_input_grads()[0].asnumpy(), mod2.get_input_grads()[0].asnumpy(), rtol=rtol, atol=atol)


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnn_with_new_param():
    rnn_modes = ['rnn_relu', 'rnn_tanh', 'gru', 'lstm']
    ngates_ = [1, 1, 3, 4]
    num_layers, input_size, seq_len, batch_size, state_size = 3, 128, 5, 64, 8
    for bidirectional in [False, True]:
        directions = 2 if bidirectional else 1
        for mode, ngates in zip(rnn_modes, ngates_):
            first_layer_size = (input_size * state_size + state_size * state_size + state_size * 2) * ngates
            rest_layer_size = (state_size * directions * state_size + state_size * state_size + state_size * 2) \
                * ngates * (num_layers - 1)
            param_size = (first_layer_size + rest_layer_size) * directions
            sym = mx.sym.RNN(mode=mode, num_layers=num_layers, bidirectional=bidirectional,
                state_outputs=False, state_size=state_size, name='rnn')

            bind_dict = {
                'rnn_data': mx.ndarray.random.uniform(low=-1, high=1, shape=(seq_len, batch_size, input_size)),
                'rnn_parameters': mx.ndarray.random.uniform(low=-1, high=1, shape=(param_size)),
                'rnn_state': mx.ndarray.zeros(shape=(num_layers * directions, batch_size, state_size))
            }
            if mode == 'lstm':
                bind_dict['rnn_state_cell'] = mx.ndarray.zeros(
                    shape=(num_layers * directions, batch_size, state_size))

            ex = sym.bind(default_context(), bind_dict)
            ex.forward(is_train=True)
            ex01 = ex.output_dict['rnn_output'].asnumpy()
            ex.forward(is_train=False)
            ex02 = ex.output_dict['rnn_output'].asnumpy()
            assert_allclose(ex01, ex02, rtol=1e-2, atol=1e-4)
            bind_dict['rnn_parameters'] = mx.ndarray.random.uniform(low=-1, high=1, shape=(param_size))
            ex.copy_params_from(bind_dict)
            ex.forward(is_train=True)
            ex03 = ex.output_dict['rnn_output'].asnumpy()
            ex.forward(is_train=False)
            ex04 = ex.output_dict['rnn_output'].asnumpy()
            assert_allclose(ex03, ex04, rtol=1e-2, atol=1e-4)


@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_lstm_sym():
    Ts = [1, 5]
    Ns = [1, 32]
    Is = [32, 128, 512]
    Hs = [32, 128, 512]
    for T, N, I, H in itertools.product(Ts, Ns, Is, Hs):
        fused = mx.rnn.FusedRNNCell(H, num_layers=3, mode='lstm', get_next_state=True, prefix='')
        stack = mx.rnn.SequentialRNNCell()
        stack.add(mx.rnn.LSTMCell(H, prefix='l0_'))
        stack.add(mx.rnn.LSTMCell(H, prefix='l1_'))
        stack.add(mx.rnn.LSTMCell(H, prefix='l2_'))

        check_rnn_consistency(fused, stack, T, N, I, H, 'write')
        check_rnn_consistency(fused, stack, T, N, I, H, 'add')
        check_rnn_consistency(fused, stack, T, N, I, H, 'null')

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_lstm_bidirectional():
    Ts = [1, 5]
    Ns = [1, 32]
    Is = [32, 128, 512]
    Hs = [32, 128, 512]
    for T, N, I, H in itertools.product(Ts, Ns, Is, Hs):
        fused = mx.rnn.FusedRNNCell(H, num_layers=2, mode='lstm',
                                    bidirectional=True, get_next_state=True, prefix='')

        stack = mx.rnn.SequentialRNNCell()
        stack.add(mx.rnn.BidirectionalCell(
                    mx.rnn.LSTMCell(H, prefix='l0_'),
                    mx.rnn.LSTMCell(H, prefix='r0_'),
                    output_prefix='bi_lstm_0_'))
        stack.add(mx.rnn.BidirectionalCell(
                    mx.rnn.LSTMCell(H, prefix='l1_'),
                    mx.rnn.LSTMCell(H, prefix='r1_'),
                    output_prefix='bi_lstm_1_'))

        check_rnn_consistency(fused, stack, T, N, I, H, 'write')
        check_rnn_consistency(fused, stack, T, N, I, H, 'add')
        check_rnn_consistency(fused, stack, T, N, I, H, 'null')
        check_rnn_consistency(fused, stack, T, N, I, H, {'data': 'add', 'parameters': 'null'})

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_gru_sym():
    Ts = [1, 5]
    Ns = [1, 32]
    Is = [32, 128, 512]
    Hs = [32, 128, 512]
    for T, N, I, H in itertools.product(Ts, Ns, Is, Hs):
        fused = mx.rnn.FusedRNNCell(H, num_layers=3, mode='gru', get_next_state=True, prefix='')
        stack = mx.rnn.SequentialRNNCell()
        stack.add(mx.rnn.GRUCell(H, prefix='l0_'))
        stack.add(mx.rnn.GRUCell(H, prefix='l1_'))
        stack.add(mx.rnn.GRUCell(H, prefix='l2_'))

        check_rnn_consistency(fused, stack, T, N, I, H, 'write')
        check_rnn_consistency(fused, stack, T, N, I, H, 'add')
        check_rnn_consistency(fused, stack, T, N, I, H, 'null')

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_gru_bidirectional():
    Ts = [1, 5]
    Ns = [1, 32]
    Is = [32, 128, 512]
    Hs = [32, 128, 512]
    for T, N, I, H in itertools.product(Ts, Ns, Is, Hs):
        fused = mx.rnn.FusedRNNCell(H, num_layers=2, mode='gru',
                                    bidirectional=True, get_next_state=True, prefix='')

        stack = mx.rnn.SequentialRNNCell()
        stack.add(mx.rnn.BidirectionalCell(
                    mx.rnn.GRUCell(H, prefix='l0_'),
                    mx.rnn.GRUCell(H, prefix='r0_'),
                    output_prefix='bi_gru_0_'))

        stack.add(mx.rnn.BidirectionalCell(
                    mx.rnn.GRUCell(H, prefix='l1_'),
                    mx.rnn.GRUCell(H, prefix='r1_'),
                    output_prefix='bi_gru_1_'))

        check_rnn_consistency(fused, stack, T, N, I, H, 'write')
        check_rnn_consistency(fused, stack, T, N, I, H, 'add')
        check_rnn_consistency(fused, stack, T, N, I, H, 'null')

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnntanh_sym():
    Ts = [1, 5]
    Ns = [1, 32]
    Is = [32, 128, 512]
    Hs = [32, 128, 512]
    for T, N, I, H in itertools.product(Ts, Ns, Is, Hs):
        fused = mx.rnn.FusedRNNCell(H, num_layers=3, mode='rnn_tanh', get_next_state=True, prefix='')
        stack = mx.rnn.SequentialRNNCell()
        stack.add(mx.rnn.RNNCell(H, activation='tanh', prefix='l0_'))
        stack.add(mx.rnn.RNNCell(H, activation='tanh', prefix='l1_'))
        stack.add(mx.rnn.RNNCell(H, activation='tanh', prefix='l2_'))

        check_rnn_consistency(fused, stack, T, N, I, H, 'write')
        check_rnn_consistency(fused, stack, T, N, I, H, 'add')
        check_rnn_consistency(fused, stack, T, N, I, H, 'null')

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnntanh_bidirectional():
    Ts = [1, 5]
    Ns = [1, 32]
    Is = [32, 128, 512]
    Hs = [32, 128, 512]
    for T, N, I, H in itertools.product(Ts, Ns, Is, Hs):
        fused = mx.rnn.FusedRNNCell(H, num_layers=2, mode='rnn_tanh',
                                    bidirectional=True, get_next_state=True, prefix='')

        stack = mx.rnn.SequentialRNNCell()
        stack.add(mx.rnn.BidirectionalCell(
                    mx.rnn.RNNCell(H, activation='tanh', prefix='l0_'),
                    mx.rnn.RNNCell(H, activation='tanh', prefix='r0_'),
                    output_prefix='bi_rnntanh_0_'))
        stack.add(mx.rnn.BidirectionalCell(
                    mx.rnn.RNNCell(H, activation='tanh', prefix='l1_'),
                    mx.rnn.RNNCell(H, activation='tanh', prefix='r1_'),
                    output_prefix='bi_rnntanh_1_'))

        check_rnn_consistency(fused, stack, T, N, I, H, 'write')
        check_rnn_consistency(fused, stack, T, N, I, H, 'add')
        check_rnn_consistency(fused, stack, T, N, I, H, 'null')

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnnrelu_sym():
    T, N, I, H = 5, 32, 200, 200

    fused = mx.rnn.FusedRNNCell(H, num_layers=3, mode='rnn_relu', get_next_state=True, prefix='')
    stack = mx.rnn.SequentialRNNCell()
    stack.add(mx.rnn.RNNCell(H, activation='relu', prefix='l0_'))
    stack.add(mx.rnn.RNNCell(H, activation='relu', prefix='l1_'))
    stack.add(mx.rnn.RNNCell(H, activation='relu', prefix='l2_'))

    check_rnn_consistency(fused, stack, T, N, I, H, 'write')
    check_rnn_consistency(fused, stack, T, N, I, H, 'add')
    check_rnn_consistency(fused, stack, T, N, I, H, 'null')

@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_rnnrelu_bidirectional():
    Ts = [1, 5]
    Ns = [1, 32]
    Is = [32, 128, 512]
    Hs = [32, 128, 512]
    for T, N, I, H in itertools.product(Ts, Ns, Is, Hs):
        fused = mx.rnn.FusedRNNCell(H, num_layers=2, mode='rnn_relu',
                                    bidirectional=True, get_next_state=True, prefix='')

        stack = mx.rnn.SequentialRNNCell()
        stack.add(mx.rnn.BidirectionalCell(
                    mx.rnn.RNNCell(H, activation='relu', prefix='l0_'),
                    mx.rnn.RNNCell(H, activation='relu', prefix='r0_'),
                    output_prefix='bi_rnnrelu_0_'))
        stack.add(mx.rnn.BidirectionalCell(
                    mx.rnn.RNNCell(H, activation='relu', prefix='l1_'),
                    mx.rnn.RNNCell(H, activation='relu', prefix='r1_'),
                    output_prefix='bi_rnnrelu_1_'))

        check_rnn_consistency(fused, stack, T, N, I, H, 'write', rtol=1e-2, atol=1e-2)
        check_rnn_consistency(fused, stack, T, N, I, H, 'add', rtol=1e-2, atol=1e-2)
        check_rnn_consistency(fused, stack, T, N, I, H, 'null', rtol=1e-2, atol=1e-2)

@with_seed()
def test_lstm_dropout():
    X = mx.sym.Variable('x')
    Params = mx.sym.Variable('params')
    HX = mx.sym.Variable('state')
    CX = mx.sym.Variable('state_cell')
    T, N, I, H = 300, 20, 800, 800
    rnn = mx.sym.RNN(data=X, parameters=Params, state=HX, state_cell=CX,
                     state_size=H, num_layers=5, mode='lstm', p=0.5, state_outputs=True, name='LSTM')
    exe = rnn.simple_bind(ctx=mx.cpu(), x=(T, N, I))
    out = exe.forward(is_train=True)
    out[0].wait_to_read()

@with_seed()
def test_gru_dropout():
    X = mx.sym.Variable('x')
    Params = mx.sym.Variable('params')
    HX = mx.sym.Variable('state')
    T, N, I, H = 300, 20, 800, 800
    rnn = mx.sym.RNN(data=X, parameters=Params, state=HX,
                     state_size=H, num_layers=5, mode='gru', p=0.5, state_outputs=True, name='GRU')
    exe = rnn.simple_bind(ctx=mx.cpu(), x=(T, N, I))
    out = exe.forward(is_train=True)
    out[0].wait_to_read()

@with_seed()
def test_rnntanh_dropout():
    X = mx.sym.Variable('x')
    Params = mx.sym.Variable('params')
    HX = mx.sym.Variable('state')
    T, N, I, H = 300, 20, 800, 800
    rnn = mx.sym.RNN(data=X, parameters=Params, state=HX,
                     state_size=H, num_layers=5, mode='rnn_tanh', p=0.5, state_outputs=True, name='RNN_TANH')
    exe = rnn.simple_bind(ctx=mx.cpu(), x=(T, N, I))
    out = exe.forward(is_train=True)
    out[0].wait_to_read()

@with_seed()
def test_rnnrelu_dropout():
    X = mx.sym.Variable('x')
    Params = mx.sym.Variable('params')
    HX = mx.sym.Variable('state')
    T, N, I, H = 300, 20, 800, 800
    rnn = mx.sym.RNN(data=X, parameters=Params, state=HX,
                     state_size=H, num_layers=5, mode='rnn_relu', p=0.5, state_outputs=True, name='RNN_RELU')
    exe = rnn.simple_bind(ctx=mx.cpu(), x=(T, N, I))
    out = exe.forward(is_train=True)
    out[0].wait_to_read()

def test_RNN_float64():
    if default_context().device_type == 'gpu':
        return
    sym = mx.sym.RNN(
        mx.sym.Variable('in'),
        mx.sym.Variable('par'),
        mx.sym.Variable('s'),
        state_size = (2),
        num_layers = 1,
        mode = 'rnn_tanh'
    )

    dtype = 'float64'
    explicit_grad = {
        'in': mx.nd.ones([2, 1, 2], dtype=dtype),
        'par': mx.nd.ones([12], dtype=dtype),
        's': mx.nd.ones([1, 1, 2], dtype=dtype)
    }

    args_grad = explicit_grad
    grad_req = 'write'

    ex = sym.bind(default_context(),
        {
            'in': mx.nd.ones([2, 1, 2], dtype=dtype),
            'par': mx.nd.ones([12], dtype=dtype),
            's': mx.nd.ones([1, 1, 2], dtype=dtype)
        },
        args_grad = args_grad,
        grad_req = grad_req
    )
    ex.forward()
    ex.outputs[0].wait_to_read()

def np_softmax(x, axis=-1, temperature=1.0):
    x = x - np.max(x, axis=axis, keepdims=True)
    x = np.exp(x/temperature)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


def check_elementwise_sum_with_shape(shape, n):
    # forward
    inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
    out = mx.symbol.ElementWiseSum(*inputs, name='esum')
    arr = [mx.nd.empty(shape) for i in range(n)]
    arr_grad = [mx.nd.empty(shape) for i in range(n)]
    for i in range(n):
        arr[i][:] = np.random.uniform(-10, 10, shape)
    exec1 = out.bind(default_context(),
                     args=arr,
                     args_grad=arr_grad)

    exec1.forward(is_train=True)
    out1 = exec1.outputs[0]
    out = sum(a.asnumpy() for a  in arr)
    assert_almost_equal(out, out1, rtol=1e-5, atol=1e-5)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = np.random.uniform(-10, 10, shape)
    # backward
    exec1.backward([out_grad])
    for a in arr_grad:
        assert_almost_equal(a, out_grad, rtol=1e-5, atol=1e-5)


@with_seed()
def test_elementwise_sum():
    nrepeat = 2
    maxdim = 4
    for repeat in range(nrepeat):
        for dim in range(1, maxdim):
            shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
            check_elementwise_sum_with_shape(shape, np.random.randint(1, 8))


def check_concat_with_shape(shapes, dimension, skip_second):
    # if skip_second is True, second argument will not have gradient.
    # it is to test #1130
    n = len(shapes)
    # forward
    target_dim = 0
    for shape in shapes:
        target_dim += shape[dimension]

    inputs = [mx.symbol.Variable('arg%d' % i) for i in range(n)]
    out = mx.symbol.Concat(*inputs, name='conc',dim=dimension)
    arr = [mx.nd.empty(shape) for shape in shapes]
    for i in range(n):
        arr[i][:] = shapes[i][dimension]
    arr_np = [np.copy(narray.asnumpy()) for narray in arr]
    arr_grad = [mx.nd.empty(shape) for shape in shapes]
    dict_grad = {}
    arg_names = out.list_arguments()

    for name, g in zip(arg_names, arr_grad):
        if not skip_second or name != 'arg1':
            dict_grad[name] = g

    args = out.list_arguments()
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(**dict(zip(args, shapes)))
    out_grad = mx.nd.empty(out_shapes[0])
    exec1 = out.bind(default_context(),
                     args=arr,
                     args_grad=dict_grad)
    exec1.forward(is_train=True)
    out1 = exec1.outputs[0]
    ret = np.concatenate([narray.asnumpy() for narray in arr], axis=dimension)
    assert_almost_equal(out1, ret)
    # backward
    out1.copyto(out_grad)
    out_grad[:] += 1
    exec1.backward([out_grad])

    for i, name in enumerate(arg_names):
        if not skip_second or name != 'arg1':
            grad = dict_grad[name]
            np_grad = arr_np[i]
            assert_almost_equal(grad, np_grad + 1)


@with_seed()
def test_concat():
    for dimension in range(4):
        n = 2
        merge = [2, 3, 4, 5, 6]
        a = 2
        b = 3
        c = 4
        # test  2D
        if dimension<2:
            for dim in range(2, 6):
                shapes = []
                for i in range(dim):
                    if dimension == 0:
                        shapes.append((merge[i], a))
                    elif dimension == 1:
                        shapes.append((a, merge[i]))
                    check_concat_with_shape(shapes,dimension,True)
                    check_concat_with_shape(shapes,dimension,False)
                    # Test negative dim
                    check_concat_with_shape(shapes, dimension - 2, True)
                    check_concat_with_shape(shapes, dimension - 2, False)

        #test 3D
        if dimension<3:
            for dim in range(2, 6):
                shapes = []
                for i in range(dim):
                    if dimension == 0:
                        shapes.append((merge[i], a,b))
                    elif dimension ==1:
                        shapes.append((a,merge[i],b))
                    elif dimension ==2:
                        shapes.append((a,b,merge[i]))
                check_concat_with_shape(shapes,dimension,True)
                check_concat_with_shape(shapes,dimension,False)
                # Test negative dim
                check_concat_with_shape(shapes, dimension - 3, True)
                check_concat_with_shape(shapes, dimension - 3, False)
        # test 4D
        for dim in range(2, 6):
            shapes = []
            for i in range(dim):
                if dimension == 0:
                    shapes.append((merge[i],a,b,c))
                elif dimension == 1:
                    shapes.append((a,merge[i],b,c))
                elif dimension ==2:
                    shapes.append((a,b,merge[i],c))
                elif dimension ==3:
                    shapes.append((a,b,c,merge[i]))
            check_concat_with_shape(shapes,dimension,True)
            check_concat_with_shape(shapes,dimension,False)
            # Test negative dim
            check_concat_with_shape(shapes, dimension - 4, True)
            check_concat_with_shape(shapes, dimension - 4, False)

@with_seed()
def test_slice_channel():
    def check_slice_channel(data_ndim, axis, num_outputs, squeeze_axis):
        ins = []
        if squeeze_axis:
            shape = np.random.randint(2, 5, data_ndim).tolist()
            shape[axis] = num_outputs
            out_ele_shape = [ele for ele in shape]
            del out_ele_shape[axis]
        else:
            shape = np.random.randint(1, 5, data_ndim).tolist()
            shape[axis] *= num_outputs
            out_ele_shape = [ele for ele in shape]
            out_ele_shape[axis] //= num_outputs
        data_npy = np.random.normal(size=shape)
        out_grads_npy = [np.random.normal(size=out_ele_shape) for i in range(num_outputs)]
        data = mx.sym.Variable('data')
        sym = mx.sym.SliceChannel(data=data, num_outputs=num_outputs, axis=axis, squeeze_axis=squeeze_axis)
        exe = sym.simple_bind(ctx=default_context(), data=data_npy.shape)
        assert len(exe.outputs) == num_outputs
        outputs = exe.forward(is_train=True, data=data_npy)
        for i in range(num_outputs):
            gt = data_npy.take(np.arange(i * shape[axis]/num_outputs,
                                         (i+1) * shape[axis]/num_outputs).astype(np.int), axis=axis)
            if squeeze_axis:
                assert_almost_equal(outputs[i], gt.reshape(outputs[i].shape))
            else:
                assert_almost_equal(outputs[i], gt)
        # test backward
        exe.backward(out_grads=[mx.nd.array(ele, ctx=default_context()) for ele in out_grads_npy])
        if squeeze_axis:
            assert_almost_equal(exe.grad_arrays[0],
                                np.concatenate([np.expand_dims(ele, axis=axis) for ele in out_grads_npy],
                                               axis=axis))
        else:
            assert_almost_equal(exe.grad_arrays[0],
                                np.concatenate(out_grads_npy, axis=axis))
    check_slice_channel(data_ndim=2, axis=1, num_outputs=3, squeeze_axis=True)
    check_slice_channel(data_ndim=4, axis=2, num_outputs=3, squeeze_axis=False)
    check_slice_channel(data_ndim=3, axis=-1, num_outputs=2, squeeze_axis=False)
    check_slice_channel(data_ndim=5, axis=-2, num_outputs=3, squeeze_axis=True)

@with_seed()
def test_regression():
    ''' test regression operator '''
    def check_regression(symbol, forward, backward, shape, stype='default', densities=[0, 0.5, 1]):
        # init executor
        data = mx.symbol.Variable('data')
        label = mx.symbol.Variable('label', stype=stype)
        out = symbol(data, label)
        grad_req = {'data': 'write', 'label': 'null'}
        out_exec = out.simple_bind(default_context(), grad_req=grad_req,
            data=shape, label=shape)
        arg_map = dict(zip(out.list_arguments(), out_exec.arg_arrays))
        grad_map = dict(zip(out.list_arguments(), out_exec.grad_arrays))
        # init data
        arr_data = mx.random.uniform(-1, 1, shape)
        arg_map["data"][:] = arr_data
        # init label based on density
        arr_label = arg_map["label"]
        atol = 1e-5
        for density in densities:
            arr_label[:] = rand_ndarray(shape, stype, density=density)
            out_exec.forward(is_train=True)
            out_exec.backward()
            np_out = forward(arr_data.asnumpy())
            out_grad = backward(np_out, arr_label.asnumpy().reshape(np_out.shape)) / shape[1]
            assert_almost_equal(out_exec.outputs[0], np_out, atol=atol)
            assert_almost_equal(grad_map["data"], out_grad, atol=atol)

    shape = (50, 30)

    check_regression(mx.symbol.LogisticRegressionOutput,
                     lambda x: 1.0 / (1.0 + np.exp(-x)),
                     lambda x, y : x - y,
                     shape)
    check_regression(mx.symbol.LinearRegressionOutput,
                     lambda x: x,
                     lambda x, y : x - y,
                     shape)
    check_regression(mx.symbol.MAERegressionOutput,
                     lambda x: x,
                     lambda x, y : np.where(x > y, np.ones(x.shape), -np.ones(x.shape)),
                     shape)
    check_regression(mx.symbol.LogisticRegressionOutput,
                     lambda x: 1.0 / (1.0 + np.exp(-x)),
                     lambda x, y : x - y,
                     shape, stype='csr')
    check_regression(mx.symbol.LinearRegressionOutput,
                     lambda x: x,
                     lambda x, y : x - y,
                     shape, stype='csr')


def check_softmax_grad(xpu):
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((1,4), ctx=xpu)
    label_nd = mx.nd.array([1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, use_ignore=False)
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.zeros((1,4))
    expected_grad_out[0, k] = -1
    assert np.isclose(grad_out - softmax_out, expected_grad_out).all()


def check_smoothed_softmax_grad(xpu):
    alpha = 0.2
    x = mx.sym.Variable('x')
    label = mx.sym.Variable('label')
    x_nd = mx.nd.array([[1, 6, 4, 2]], ctx=xpu)
    grad_x = mx.nd.zeros((1,4), ctx=xpu)
    label_nd = mx.nd.array([1], ctx=xpu)

    sym = mx.sym.SoftmaxOutput(data=x, label=label, ignore_label=0, use_ignore=False, smooth_alpha=alpha)
    ex = sym.bind(ctx=xpu, args={'x': x_nd, 'label': label_nd}, args_grad={'x': grad_x})

    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()
    expected_softmax_out = [[0.005806628, 0.861780069, 0.116629249, 0.015784052]]
    assert np.isclose(softmax_out, expected_softmax_out).all()

    ex.backward(is_train=True)
    grad_out = ex.grad_arrays[0].asnumpy()
    k = int(label_nd[0].asscalar())
    expected_grad_out = np.full((1,4), fill_value=-alpha/float(4-1))
    expected_grad_out[0, k] = - (1 - alpha)
    assert np.isclose(grad_out - softmax_out, expected_grad_out).all()


def check_softmax_with_ignore_label(xpu):
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SoftmaxOutput(data=X, label=L, ignore_label=0, use_ignore=True)

    shape = (20, 10)
    x = mx.nd.empty(shape, ctx = xpu)
    l = mx.nd.empty((shape[0],), ctx = xpu)
    x_np = np.random.rand(*shape)
    l_np = np.random.randint(0, shape[1]-1, (shape[0],))
    x[:] = x_np
    l[:] = l_np

    grad = mx.nd.empty(shape, ctx = xpu)

    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward(is_train=True)
    exec1.backward()

    grad0 = grad.asnumpy()

    for i in range(int(shape[0]/2)):
        l_np[i] = 0
    l[:] = l_np

    exec1.forward(is_train=True)
    exec1.backward()
    grad1 = grad.asnumpy()

    assert abs(np.sum(grad1[:int(shape[0]/2)])) < 1e-5
    assert_almost_equal(grad0[int(shape[0]/2):], grad1[int(shape[0]/2):])


def check_softmax_with_shape(shape, xpu, preserve_shape=False):
    # bind with label
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SoftmaxOutput(data=X, label=L, preserve_shape=preserve_shape)
    x = mx.random.uniform(-1, 1, shape, ctx=xpu)
    l = mx.random.uniform(-1, 1, shape, ctx=xpu)
    l[:] = np_softmax(l.asnumpy())
    grad = mx.nd.empty(shape, ctx = xpu)
    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward(is_train=True)
    out = exec1.outputs[0].asnumpy()
    # Non-zero atol required by test_softmax with seed 781663739
    rtol = 1e-4
    atol = 1e-6
    assert_almost_equal(out, np_softmax(x.asnumpy()), rtol=rtol, atol=atol)
    exec1.backward()
    assert_almost_equal(grad, np_softmax(x.asnumpy()) - l.asnumpy(), rtol=rtol, atol=atol)


def test_python_op():
    X = mx.symbol.Variable('X')
    op = mx.operator.NumpyOp()
    s = op.get_symbol(X, name='numpy_op')

    x = mx.ndarray.ones((10))*10
    dx = mx.ndarray.zeros((10))
    dy = mx.ndarray.ones((10))
    exec1 = s.bind(default_context(), args=[x], args_grad = {'X': dx})
    exec1.forward(is_train=True)
    assert_almost_equal(x, exec1.outputs[0])
    exec1.backward(dy)
    assert_almost_equal(dy, dx)


def test_swapaxes():
    data = mx.symbol.Variable('data')
    shape = (2, 3, 4)
    data_tmp = np.ones(shape)
    data_tmp[0] = 1
    data_tmp[1] = 2
    arr_data = mx.nd.array(data_tmp)
    swap0 = mx.symbol.SwapAxis(data=data, dim1=0, dim2=2)
    swap = mx.symbol.SwapAxis(data=swap0, dim1=1, dim2=2)
    exe_c = swap.bind(default_context(), args=[arr_data])
    exe_c.forward(is_train=True)
    out = exe_c.outputs[0]

    swap0_ = np.swapaxes(data_tmp, 0, 2)
    swap_ = np.swapaxes(swap0_, 1, 2)

    assert_almost_equal(out, swap_)

    config = [((1, 1, 2), 0, 1),
              ((1, 1, 2), -1, -2),
              ((4, 5, 6, 7), 1, 1),
              ((4, 5, 6, 7), 2, 3),
              ((4, 5, 6, 7), -2, 2),
              ((4, 5, 6, 7), -2, -3)]

    for shape, axis1, axis2 in config:
        data_np = np.random.uniform(size=shape)
        data_mx = mx.nd.array(data_np, dtype=data_np.dtype)
        ret_np = np.swapaxes(data_np, axis1=axis1, axis2=axis2)
        ret_mx = mx.symbol.SwapAxis(data, dim1=axis1, dim2=axis2)
        exe_c = ret_mx.bind(default_context(), args=[data_mx])
        exe_c.forward(is_train=True)
        out = exe_c.outputs[0]
        assert_almost_equal(out, ret_np)


@with_seed()
def test_scalarop():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)*5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = 2 / (4-((1+data+1)*2/5)-0.8-(data!=0))

    npout_1 = (4-((1+data_tmp+1)*2/5)-0.8-(data_tmp!=0))
    npout = 2/npout_1

    check_symbolic_forward(test, [data_tmp], [npout])

    npout_grad = 2.*2/5
    npout_grad = 2*npout_grad /(npout_1 *npout_1 )

    check_symbolic_backward(test, [data_tmp], [np.ones(shape)*2], [npout_grad])


@with_seed()
def test_scalar_pow():
    data = mx.symbol.Variable('data')
    shape = (1, 1)
    data_tmp = np.ones(shape)
    test = data ** 2
    check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, [data_tmp], [data_tmp ** 2])
    check_symbolic_backward(test, [data_tmp], [np.ones(shape)], [2 * data_tmp])


@with_seed()
def test_symbol_pow():
    shape = (1, 1)

    data = mx.symbol.Variable('data')
    data_tmp = np.ones(shape)*2

    exp = mx.symbol.Variable('exp')
    exp_tmp = np.ones(shape)*3

    test = data**exp

    check_numeric_gradient(test, [data_tmp, exp_tmp])
    check_symbolic_forward(test, [data_tmp, exp_tmp], [data_tmp**exp_tmp])

    data_dir = data_tmp**(exp_tmp - 1) * exp_tmp
    exp_dir = data_tmp**(exp_tmp) * np.log(data_tmp)
    check_symbolic_backward(test, [data_tmp, exp_tmp], [np.ones(shape)], [data_dir, exp_dir])


@with_seed()
def test_fully_connected():
    # Create data of given shape as a uniform distribution centered on 0.0
    def random_data(shape, dtype=np.float32):
        return mx.nd.random.uniform(low=-0.5,
                                    high=0.5, shape=shape, dtype=dtype)
    data = mx.sym.var("data")
    fc_weight = mx.sym.var("weight")
    fc_bias = mx.sym.var("bias")
    fc = mx.sym.FullyConnected(data=data, weight=fc_weight, bias=fc_bias, num_hidden=10, no_bias=False, name='fc')

    data = random_data(shape=(5, 5, 5, 13))
    fc_weight = random_data(shape=(10, 325))
    fc_bias = random_data(shape=(10))
    fc_bias2 = random_data(shape=(10, 1))

    data_np = data.asnumpy().reshape(5, 325)
    fc_weight_np = np.transpose(fc_weight.asnumpy())
    fc_bias_np = fc_bias.asnumpy()
    res = np.dot(data_np, fc_weight_np) + fc_bias.asnumpy()
    check_symbolic_forward(fc, {'data': data_np, 'weight': fc_weight.asnumpy(), 'bias': fc_bias_np}, {'fc_output': res})
    check_numeric_gradient(fc, {'data': data_np, 'weight': fc_weight.asnumpy(), 'bias': fc_bias_np})
    # TODO: Fix Bug #15032 when bias has ndim > 1
    #check_symbolic_forward(fc, {'data': data_np, 'weight': fc_weight.asnumpy(), 'bias': fc_bias2.asnumpy()}, {'fc_output': res})


@with_seed()
def test_pow_fn():
    shape = (3, 4)
    exp = mx.symbol.Variable("exp")
    x = np.ones(shape)*3
    for y in [mx.sym.pow(2, exp), mx.sym.power(2, exp)]:
        check_numeric_gradient(y, [x], numeric_eps=1E-3)
        check_symbolic_forward(y, [x], [2**x])
        check_symbolic_backward(y, [x], [np.ones(shape)], [np.log(2) * 2**x])


@with_seed()
def test_relu():
    def frelu(x):
        return np.maximum(x, 0.0)
    def frelu_grad(x):
        return 1.0 * (x > 0.0)
    shape = (3, 4)
    x = mx.symbol.Variable("x")
    y = mx.sym.relu(x)
    xa = np.random.uniform(low=-1.0,high=1.0,size=shape)
    eps = 1e-4
    # Avoid finite difference method inaccuracies due to discontinuous gradient at the origin.
    # Here we replace small problematic inputs with 1.0.  Repro issue with seed 97264195.
    xa[abs(xa) < eps] = 1.0
    ya = frelu(xa)
    ga = frelu_grad(xa)
    check_numeric_gradient(y, [xa], numeric_eps=eps)
    check_symbolic_forward(y, [xa], [ya])
    check_symbolic_backward(y, [xa], [np.ones(shape)], [ga])


# NOTE(haojin2): Skipping the numeric check tests for float16 data type due to precision issues,
# the analytical checks are still performed on each and every data type to verify the correctness.
@with_seed()
def test_leaky_relu():
    def fleaky_relu(x, act_type, slope=0.25):
        neg_indices = x < 0
        out = x.copy()
        if act_type == 'elu':
            out[neg_indices] = slope * np.expm1(out[neg_indices])
        elif act_type == 'leaky':
            out[neg_indices] = slope * out[neg_indices]
        return out
    def fleaky_relu_grad(grad, x, y, act_type, slope=0.25):
        neg_indices = x < 0
        out = np.ones(x.shape)
        if act_type == 'elu':
            out[neg_indices] = y[neg_indices] + slope
        elif act_type == 'leaky':
            out[neg_indices] = slope
        return out * grad
    for ndim in range(1, 4):
        shape = rand_shape_nd(ndim)
        x = mx.symbol.Variable("x")
        slp = 0.25
        for dtype in [np.float16, np.float32, np.float64]:
            xa = np.random.uniform(low=-1.0,high=1.0,size=shape).astype(dtype)
            eps = 1e-4
            rtol = 1e-2
            atol = 1e-3
            xa[abs(xa) < eps] = 1.0
            for act_type in ['elu', 'leaky']:
                y = mx.symbol.LeakyReLU(data=x, slope=slp, act_type=act_type)
                ya = fleaky_relu(xa, slope=slp, act_type=act_type)
                ga = fleaky_relu_grad(np.ones(shape), xa, ya, slope=slp, act_type=act_type)
                # Skip numeric check for float16 type to get rid of flaky behavior
                if dtype is not np.float16:
                    check_numeric_gradient(y, [xa], numeric_eps=eps, rtol=rtol, atol=atol, dtype=dtype)
                check_symbolic_forward(y, [xa], [ya], rtol=rtol, atol=atol, dtype=dtype)
                check_symbolic_backward(y, [xa], [np.ones(shape)], [ga], rtol=rtol, atol=atol, dtype=dtype)


# NOTE(haojin2): Skipping the numeric check tests for float16 data type due to precision issues,
# the analytical checks are still performed on each and every data type to verify the correctness.
@with_seed()
def test_prelu():
    def fprelu(x, gamma):
        pos_indices = x > 0
        out = x.copy()
        if len(x.shape) == 4:
            out = out.transpose(2,3,0,1)
            out = np.multiply(out, gamma)
            out = out.transpose(2,3,0,1)
        else:
            out = np.multiply(out, gamma)
        out[pos_indices] = x[pos_indices]
        return out
    def fprelu_grad(x, y, gamma):
        pos_indices = x > 0
        if len(x.shape) == 4:
            grad_x = np.multiply(np.ones(x.shape).transpose(2,3,0,1), gamma)
            grad_x = grad_x.transpose(2,3,0,1)
        else:
            grad_x = np.multiply(np.ones(x.shape), gamma)
        grad_gam = np.zeros(gamma.shape)
        copy_x = x.copy()
        copy_x[pos_indices] = 0.0
        grad_x[pos_indices] = 1.0
        if len(gamma.shape) > 1 and len(x.shape) != 4:
            grad_gam = copy_x
        elif len(gamma.shape) > 1 and len(x.shape) == 4:
            grad_gam = np.sum(copy_x, axis=(2,3))
        elif gamma.shape[0] == 1:
            grad_gam = np.sum(np.sum(copy_x))
        elif gamma.shape[0] > 1 and len(x.shape) != 4:
            grad_gam = np.sum(copy_x, axis=0)
        elif gamma.shape[0] > 1 and len(x.shape) == 4:
            grad_gam = np.sum(copy_x, axis=(0,2,3))
        return (grad_x, grad_gam)
    x = mx.symbol.Variable("x")
    gamma = mx.symbol.Variable("gamma")
    for shape in [(3,4), (3,4,4,5)]:
        for dtype in [np.float16, np.float32, np.float64]:
            for gam in [np.array([0.1, 0.2, 0.3, 0.4], dtype=dtype)]:
                gam_full = np.array([gam, gam, gam])
                xa = np.random.uniform(low=-1.0,high=1.0,size=shape).astype(dtype)
                rtol = 1e-2
                atol = 1e-3
                eps = 1e-4
                xa[abs(xa) < eps] = 1.0
                y = mx.symbol.LeakyReLU(data=x, gamma=gamma, act_type='prelu')
                ya = fprelu(xa, gam)
                ya_full = fprelu(xa, gam_full)
                g_xa, g_gam = fprelu_grad(xa, ya, gamma=gam)
                g_xa_full, g_gam_full = fprelu_grad(xa, ya_full, gamma=gam_full)
                # Skip numeric check for float16 type to get rid of flaky behavior
                if dtype is not np.float16:
                    check_numeric_gradient(y, [xa, gam], numeric_eps=eps, rtol=rtol, atol=atol, dtype=dtype)
                    check_numeric_gradient(y, [xa, gam_full], numeric_eps=eps, rtol=rtol, atol=atol, dtype=dtype)
                check_symbolic_forward(y, [xa, gam], [ya], rtol=rtol, atol=atol, dtype=dtype)
                check_symbolic_backward(y, [xa, gam], [np.ones(shape), np.ones(gam.shape)], [g_xa, g_gam], rtol=rtol, atol=atol, dtype=dtype)
                check_symbolic_forward(y, [xa, gam_full], [ya_full], rtol=rtol, atol=atol, dtype=dtype)
                check_symbolic_backward(y, [xa, gam_full], [np.ones(shape), np.ones(gam_full.shape)],
                                        [g_xa_full, g_gam_full], rtol=rtol, atol=atol, dtype=dtype)

@with_seed()
def test_selu():
    alpha = 1.6732632423543772848170429916717
    lamb = 1.0507009873554804934193349852946
    def fselu(x):
        neg_indices = x < 0
        out = x.copy()
        out[neg_indices] = alpha * np.expm1(out[neg_indices])
        return out * lamb
    def fselu_grad(grad, x, y):
        neg_indices = x < 0
        out = np.ones(x.shape).astype(x.dtype)
        out[neg_indices] = y[neg_indices] + alpha
        return out * lamb

    shape = (3, 4)
    x = mx.sym.Variable("x")
    y = mx.sym.LeakyReLU(data=x, act_type="selu")
    for dtype in [np.float16, np.float32, np.float64]:
        xa = np.random.uniform(low=-0.1,high=0.1,size=shape).astype(dtype)
        eps, rtol, atol = (7.5e-4, 1e-1, 1e-2) if dtype is np.float16 else (1e-4, 1e-2, 1e-4)
        if dtype is np.float16:
            xa /= 10.0
        xa[abs(xa) < eps] = 0.01
        ya = fselu(xa)
        ga = fselu_grad(np.ones(shape).astype(dtype), xa, ya)
        check_numeric_gradient(y, [xa], numeric_eps=eps, rtol=rtol, atol=atol, dtype=dtype)
        check_symbolic_forward(y, [xa], [ya], rtol=rtol, atol=atol, dtype=dtype)
        check_symbolic_backward(y, [xa], [np.ones(shape)], [ga], rtol=rtol, atol=atol, dtype=dtype)


@with_seed()
def test_gelu():
    CUBE_CONSTANT = 0.044715
    ROOT_TWO_OVER_PI = 0.7978845608028654
    def g(x):
        return ROOT_TWO_OVER_PI * (x + CUBE_CONSTANT * np.power(x, 3))
    def g_grad(x):
        return ROOT_TWO_OVER_PI * (1.0 + 3.0 * CUBE_CONSTANT * np.power(x, 2))
    def f(x):
        return 1.0 + np.tanh(g(x))
    def f_grad(x):
        return (1.0 - np.tanh(g(x)) * np.tanh(g(x))) * g_grad(x)
    def fgelu(x):
        return 0.5 * x * f(x)
    def fgelu_grad(grad, x, y):
        return grad * (y / x + y * (1 - np.tanh(g(x))) * g_grad(x))

    shape = (3, 4)
    x = mx.sym.Variable("x")
    y = mx.sym.LeakyReLU(data=x, act_type="gelu")
    for dtype in [np.float16, np.float32, np.float64]:
        xa = np.random.uniform(low=-0.1,high=0.1,size=shape).astype(dtype)
        eps, rtol, atol = (7.5e-4, 2e-2, 1e-3) if dtype is np.float16 else (1e-4, 1e-3, 1e-5)
        if dtype is np.float16:
            xa /= 10.0
        xa[abs(xa) < eps] = 0.01
        ya = fgelu(xa)
        ga = fgelu_grad(np.ones(shape).astype(dtype), xa, ya)
        check_numeric_gradient(y, [xa], numeric_eps=eps, rtol=rtol, atol=atol, dtype=dtype)
        check_symbolic_forward(y, [xa], [ya], rtol=rtol, atol=atol, dtype=dtype)
        check_symbolic_backward(y, [xa], [np.ones(shape)], [ga], rtol=rtol, atol=atol, dtype=dtype)


@with_seed()
def test_sigmoid():
    def fsigmoid(a):
        return np.divide(1.0, (1.0 + np.exp(-a)))
    shape = (3, 4)
    x = mx.symbol.Variable("x")
    y = mx.sym.sigmoid(x)
    xa = np.random.uniform(low=-1.0,high=1.0,size=shape)
    ya = fsigmoid(xa)
    check_numeric_gradient(y, [xa], numeric_eps=1E-3)
    check_symbolic_forward(y, [xa], [ya])
    check_symbolic_backward(y, [xa], [np.ones(shape)], [ya * (1 - ya)])

@with_seed()
def test_shape_array():
    for i in range(1,6):
        shape = rand_shape_nd(i)
        x = mx.sym.var('x')
        y = mx.sym.shape_array(x)
        xa = mx.nd.array(np.random.ranf(shape))
        xg = mx.nd.empty(xa.shape)
        ya = np.shape(xa)
        yg = mx.nd.ones(ya)
        exe = y.bind(ctx=default_context(), args={'x': xa},
                     args_grad={'x': xg})
        exe.forward(is_train=True)
        exe.backward([yg])
        yo = exe.outputs[0].asnumpy()
        same(yo, ya)
        assert_almost_equal(xg, np.zeros_like(xg.asnumpy()))

@with_seed()
def test_size_array():
    for i in range(1,6):
        shape = rand_shape_nd(i)
        x = mx.sym.var('x')
        y = mx.sym.size_array(x)
        xa = mx.nd.array(np.random.ranf(shape))
        xg = mx.nd.empty(xa.shape)
        ya = np.size(xa)
        yg = mx.nd.ones(ya)
        exe = y.bind(ctx=default_context(), args={'x': xa},
                     args_grad={'x': xg})
        exe.forward(is_train=True)
        exe.backward([yg])
        yo = exe.outputs[0].asnumpy()
        same(yo, ya)
        assert_almost_equal(xg, np.zeros_like(xg.asnumpy()))

@with_seed()
def test_hard_sigmoid():
    def fhardsigmoid(a, alpha=0.2, beta=0.5):
        return np.maximum(np.zeros(a.shape, dtype=a.dtype),
                          np.minimum(np.ones(a.shape, dtype=a.dtype), alpha*a+beta))
    def fhardsigmoid_grad(a, out_grad, alpha=0.2, beta=0.5):
        orig_out = fhardsigmoid(a, alpha, beta)
        res = out_grad * alpha
        res[orig_out <= 0.0] = 0.0
        res[orig_out >= 1.0] = 0.0
        return res
    shape = (3, 4)
    x = mx.symbol.Variable("x")
    y = mx.sym.hard_sigmoid(x)
    for dtype in [np.float16, np.float32, np.float64]:
        if dtype is np.float16:
            rtol = 1e-2
        else:
            rtol = 1e-3
        atol = 1e-3
        eps = 1e-3
        xa = np.random.uniform(low=-3.0,high=3.0,size=shape).astype(dtype)
        # function not differentiable at x=2.5 and -2.5
        xa[abs(xa-2.5) < eps] -= 2 * eps
        xa[abs(xa+2.5) < eps] += 2 * eps
        ya = fhardsigmoid(xa)
        grad_xa = fhardsigmoid_grad(xa, np.ones(shape))
        if dtype is not np.float16:
            check_numeric_gradient(y, [xa], numeric_eps=eps, rtol=rtol, atol=atol, dtype=dtype)
        check_symbolic_forward(y, [xa], [ya], rtol=rtol, atol=atol, dtype=dtype)
        check_symbolic_backward(y, [xa], [np.ones(shape)], [grad_xa], rtol=rtol, atol=atol, dtype=dtype)

@with_seed()
def test_softsign():
    def fsoftsign(a):
        return np.divide(a, (1.0 + np.abs(a)))
    def fsoftsign_grad(a):
        return np.divide(1.0, np.square((1.0 + np.abs(a))))
    shape = (3, 4)
    x = mx.symbol.Variable("x")
    y = mx.sym.softsign(x)
    xa = np.random.uniform(low=-1.0,high=1.0,size=shape)
    ya = fsoftsign(xa)
    ya_grad = fsoftsign_grad(xa)
    check_numeric_gradient(y, [xa], numeric_eps=1E-3)
    check_symbolic_forward(y, [xa], [ya])
    check_symbolic_backward(y, [xa], [np.ones(shape)], [ya_grad])

@with_seed()
def test_binary_logic():
    def _inner_test(forward_gt, logic_sym, x_shape, y_shape, test_scalar=True):
        x = mx.symbol.Variable("x")
        y = mx.symbol.Variable("y")
        z = logic_sym(x, y)
        x_npy = np.random.randint(0, 4, size=x_shape).astype(np.float32)
        y_npy = np.random.randint(0, 4, size=y_shape).astype(np.float32)
        exe = z.simple_bind(ctx=default_context(), x=x_shape, y=y_shape)
        mx_out = exe.forward(is_train=True, x=x_npy, y=y_npy)[0]
        assert_almost_equal(mx_out, forward_gt(x_npy, y_npy))
        exe.backward()
        if test_scalar:
            z_lscalar = logic_sym(1, y)
            z_rscalar = logic_sym(x, 1)
            exe_lscalar = z_lscalar.simple_bind(ctx=default_context(), y=y_shape)
            exe_rscalar = z_rscalar.simple_bind(ctx=default_context(), x=x_shape)
            mx_lscalar_out = exe_lscalar.forward(is_train=True, y=y_npy)[0]
            mx_rscalar_out = exe_rscalar.forward(is_train=True, x=x_npy)[0]
            assert_almost_equal(mx_lscalar_out, forward_gt(1, y_npy))
            assert_almost_equal(mx_rscalar_out, forward_gt(x_npy, 1))
            exe_lscalar.backward()
            exe_rscalar.backward()
    # Test the no-broadcasting binary logic ops + scalar logic ops
    _inner_test(forward_gt=lambda x, y: x == y,
                logic_sym=lambda x, y: x == y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x > y,
                logic_sym=lambda x, y: x > y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x >= y,
                logic_sym=lambda x, y: x >= y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x < y,
                logic_sym=lambda x, y: x < y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x <= y,
                logic_sym=lambda x, y: x <= y, x_shape=(10, 10), y_shape=(10, 10))
    _inner_test(forward_gt=lambda x, y: x != y,
                logic_sym=lambda x, y: x != y, x_shape=(10, 10), y_shape=(10, 10))
    # Test the broadcasting binary logic ops
    _inner_test(forward_gt=lambda x, y: x == y,
                logic_sym=lambda x, y: mx.sym.broadcast_equal(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x > y,
                logic_sym=lambda x, y: mx.sym.broadcast_greater(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x >= y,
                logic_sym=lambda x, y: mx.sym.broadcast_greater_equal(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x < y,
                logic_sym=lambda x, y: mx.sym.broadcast_lesser(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x <= y,
                logic_sym=lambda x, y: mx.sym.broadcast_lesser_equal(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)
    _inner_test(forward_gt=lambda x, y: x != y,
                logic_sym=lambda x, y: mx.sym.broadcast_not_equal(x, y),
                x_shape=(1, 10), y_shape=(10, 1), test_scalar=False)


@with_seed()
def test_unary_logic():
    def reference(a, dtype):
        return np.logical_not(a).astype(dtype)
    shape = (3, 4)
    xa = np.random.randint(-2, 2, size=shape).astype(np.float32)
    mx_xa = mx.nd.array(xa)
    mx_out = mx.nd.logical_not(mx_xa)
    assert_almost_equal(mx_out, reference(xa, dtype=xa.dtype))
    x = mx.sym.Variable('x')
    y = mx.sym.logical_not(data=x)
    exe = y.simple_bind(ctx=default_context(), x=shape)
    sym_out = exe.forward(is_train=True, x=mx_xa)[0]
    assert_almost_equal(sym_out, reference(xa, dtype=xa.dtype))


@with_seed()
def test_embedding():
    in_dim = 10
    out_dim = 4
    batch = 24

    data = mx.sym.Variable("data")
    embed = mx.sym.Embedding(data=data, input_dim=in_dim, output_dim=out_dim, name="embed")
    exe_test = embed.simple_bind(default_context(), grad_req={'data': 'null', 'embed_weight': 'write'}, data=(batch,))
    arg_map = dict(zip(embed.list_arguments(), exe_test.arg_arrays))
    grad_map = dict(zip(embed.list_arguments(), exe_test.grad_arrays))
    np_data = np.random.randint(low=0, high=in_dim, size=batch)
    np_weight = np.random.uniform(-0.01, 0.01, arg_map["embed_weight"].shape)
    np_onehot = np.zeros((batch, in_dim))
    np_onehot[np.arange(batch), np_data] = 1.0
    # forward
    arg_map["data"][:] = np_data
    arg_map["embed_weight"][:] = np_weight
    exe_test.forward(is_train=True)
    # Non-zero atol required, as exposed by seed 781663739
    rtol = 1e-5
    atol = 1e-5
    assert_almost_equal(exe_test.outputs[0], np.dot(np_onehot, np_weight), rtol=rtol, atol=atol)
    # backward
    np_grad = np.random.uniform(-1, 1, exe_test.outputs[0].shape)
    grad = mx.nd.zeros(np_grad.shape)
    grad[:] = np_grad
    exe_test.backward([grad])
    assert_almost_equal(grad_map["embed_weight"], np.dot(np_onehot.T, np_grad), rtol=rtol, atol=atol)


# check ops handle duplicate input correctly.
@with_seed()
def test_binary_op_duplicate_input():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:] = 5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:] = 3
    out_grad = mx.nd.empty(shape)
    out_grad[:] = 1
    square = data * data
    exe_square = square.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_square.forward(is_train=True)
    assert_almost_equal(exe_square.outputs[0], data_tmp * data_tmp)
    exe_square.backward(out_grad)
    assert_almost_equal(arr_grad, 2.0 * data_tmp)


@with_seed()
def test_sign():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = mx.sym.sign(data)
    exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout = np.sign(data_tmp)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = 0;
    exe_test.backward(out_grad)
    assert_almost_equal(arr_grad, npout_grad)


@with_seed()
def test_round_ceil_floor():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5.543
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]= 2

    test = mx.sym.round(data) + mx.sym.ceil(data) +  mx.sym.floor(data)
    exe_test = test.bind(default_context(), args=[arr_data])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout = np.round(data_tmp) + np.ceil(data_tmp) + np.floor(data_tmp)
    assert_almost_equal(out, npout)


@with_seed()
def test_trunc():
    data_tmp = np.random.rand(3, 4) * 10 - 5
    arr_data = mx.nd.array(data_tmp)
    data = mx.symbol.Variable('data')
    test = mx.sym.trunc(data)

    exe_test = test.bind(default_context(), args=[arr_data])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    # 'trunc' is sensitive to the precision of the calculation.  Force numpy to match mxnet's float32.
    # Repro issue with seed 1660190454
    npout = np.trunc(np.float32(data_tmp))

    assert_almost_equal(out, npout)


@with_seed()
def test_rsqrt_cos_sin():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test =  mx.sym.rsqrt(data) + mx.sym.cos(data) + mx.sym.sin(data)
    exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout =  1/ np.sqrt(data_tmp) + np.cos(data_tmp) + np.sin(data_tmp)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2
    npout_grad = out_grad.asnumpy()
    npout_grad = npout_grad * -(1.0 / (2.0 * data_tmp * np.sqrt(data_tmp))) + npout_grad * -1 * np.sin(data_tmp) + npout_grad * np.cos(data_tmp)
    exe_test.backward(out_grad)
    assert_almost_equal(arr_grad, npout_grad)


@with_seed()
def test_maximum_minimum():
    data1 = mx.symbol.Variable('data1')
    data2 = mx.symbol.Variable('data2')
    shape = (3, 4)
    data_tmp1 = np.random.rand(3,4)
    data_tmp2 = np.random.rand(3,4)
    data_tmp1[:] = 2
    data_tmp2[:] = 3

    arr_data1 = mx.nd.array(data_tmp1)
    arr_data2 = mx.nd.array(data_tmp2)

    arr_grad1 = mx.nd.empty(shape)
    arr_grad2 = mx.nd.empty(shape)

    test =  mx.sym.maximum(data1,data2) + mx.sym.minimum(data1,data2)
    exe_test = test.bind(default_context(), args=[arr_data1,arr_data2], args_grad=[arr_grad1,arr_grad2])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout =  np.maximum(data_tmp1,data_tmp2) + np.minimum(data_tmp1,data_tmp2)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2
    exe_test.backward(out_grad)

    npout_grad = np.ones(shape)
    npout_grad[:] = 2
    mask1 = (data_tmp1 > data_tmp2).astype('float')
    mask2 = (data_tmp1 < data_tmp2).astype('float')
    npout_grad1 = npout_grad * mask1 + npout_grad * mask2
    npout_grad2 = (npout_grad - npout_grad * mask1) + (npout_grad - npout_grad * mask2)

    assert_almost_equal(arr_grad1, npout_grad1)
    assert_almost_equal(arr_grad2, npout_grad2)


@with_seed()
def test_maximum_minimum_scalar():
    data1 = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp1 = np.random.rand(3,4)
    data_tmp1[:] = 2

    arr_data1 = mx.nd.array(data_tmp1)
    arr_grad1 = mx.nd.empty(shape)

    test =  mx.sym.maximum(data1,3) + mx.sym.maximum(9,data1) + mx.sym.minimum(5,data1) + mx.sym.minimum(data1,4)
    exe_test = test.bind(default_context(), args=[arr_data1], args_grad=[arr_grad1])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout =  np.maximum(data_tmp1,3) + np.maximum(9,data_tmp1) + np.minimum(5,data_tmp1) + np.minimum(data_tmp1,4)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2
    exe_test.backward(out_grad)

    npout_grad = np.ones(shape)
    npout_grad[:] = 2
    mask1 = (data_tmp1 > 3).astype('float')
    mask2 = (9 > data_tmp1).astype('float')
    mask3 = (5 < data_tmp1).astype('float')
    mask4 = (data_tmp1 < 4).astype('float')
    npout_grad1 = npout_grad * mask1 + (npout_grad - npout_grad * mask2) + (npout_grad - npout_grad * mask3) + npout_grad * mask4

    assert_almost_equal(arr_grad1, npout_grad1)


@with_seed()
def test_abs():
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:]=5
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:]=3

    test = mx.sym.abs(data)
    exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout = abs(data_tmp)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = 2;
    npout_grad = out_grad.asnumpy()
    npout_grad = npout_grad * np.sign(data_tmp)
    exe_test.backward(out_grad)
    assert_almost_equal(arr_grad, npout_grad)


def check_deconvolution_forward_backward(input_shape, num_filter, kernel, stride, pad):
    """configure A: input --> conv --> deconv --> output.
       the convolution and deconvoluiton has similar parameter which ensure
       the input shape is the same as output, and the same weights between conv
       and deconv;
       If the input value of forward() and backwrad() is the same, then
       the output value of them should also the same;
    """
    assert input_shape[1] == num_filter
    data = mx.sym.Variable(name="data")
    conv = mx.sym.Convolution(
        data=data, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "conv")
    deconv = mx.sym.Deconvolution(
        data=conv, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "deconv")

    arg_names = deconv.list_arguments()
    arg_shapes, out_shapes, _ = deconv.infer_shape(data=input_shape)
    input_data = mx.random.uniform(-5, 5, input_shape, ctx=mx.cpu()).copyto(default_context())
    out_grad = input_data
    args = {}
    args["data"] = input_data
    args['conv_weight'] = args['deconv_weight'] = mx.random.normal(0, 1,
        (num_filter, input_shape[1]) + kernel, ctx=mx.cpu()).copyto(default_context())
    args_grad = [mx.nd.empty(s) for s in arg_shapes]

    exe = deconv.bind(default_context(), args=args, args_grad=args_grad)
    exe.forward(is_train=True)
    out = exe.outputs[0]
    exe.backward(out_grad)
    assert_almost_equal(out, args_grad[0], rtol=1E-3, atol=1e-3)

    args_grad_addto_npy = [np.random.normal(size=s) for s in arg_shapes]
    args_grad_addto = [mx.nd.array(ele) for ele in args_grad_addto_npy]
    exe = deconv.bind(default_context(), args=args, args_grad=args_grad_addto, grad_req="add")
    exe.forward(is_train=True)
    out = exe.outputs[0].asnumpy()
    exe.backward(out_grad)
    assert_almost_equal(out + args_grad_addto_npy[0], args_grad_addto[0].asnumpy(), rtol=1e-3, atol=1e-3)


def check_deconvolution_gradient(input_shape, num_filter, pad):
    """configure A: input --> conv --> output.
       configure B: input --> deconv --> output
       the convolution and deconvoluiton has similar parameter which ensure
       the input shape is the same as output;
       During backward(), if the input of A equals output of B, and the output
       of A equals input of B, then the grad of weight should be the same;
    """
    ndim = len(pad)
    stride = (1,) * ndim
    kernel = tuple(2 * np.array(pad) + 1)
    data_conv = mx.sym.Variable(name="data_conv")
    conv = mx.sym.Convolution(
        data=data_conv, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "conv")
    data_deconv = mx.sym.Variable(name="data_deconv")
    deconv = mx.sym.Deconvolution(
        data=data_deconv, kernel=kernel, stride=stride, pad=pad,
        num_filter=num_filter, no_bias = "true", name = "deconv")

    conv_data = mx.random.uniform(-5, 5, input_shape, ctx=mx.cpu()).copyto(default_context())
    conv_args = {}
    conv_args["data_conv"] = conv_data
    conv_args['conv_weight'] = \
        mx.random.normal(0, 1,(num_filter, input_shape[1]) + kernel, ctx=mx.cpu()).copyto(default_context())
    conv_args_grad = [mx.nd.zeros(conv_data.shape),
        mx.nd.zeros((num_filter, input_shape[1]) + kernel)]
    exe_conv = conv.bind(default_context(), args=conv_args, args_grad=conv_args_grad)
    exe_conv.forward(is_train=True)
    conv_out_grad = mx.random.normal(0, 2, exe_conv.outputs[0].shape, ctx=mx.cpu()).copyto(default_context())
    exe_conv.backward(conv_out_grad)

    deconv_data = conv_out_grad
    deconv_args = {}
    deconv_args['data_deconv'] = deconv_data
    deconv_args['deconv_weight'] = conv_args['conv_weight']
    deconv_args_grad = [mx.nd.zeros(deconv_data.shape),
        mx.nd.zeros((num_filter, input_shape[1]) + kernel)]
    deconv_addto_args_grad_npy = [np.random.normal(size=deconv_data.shape),
                                  np.random.normal(size=(num_filter, input_shape[1]) + kernel)]
    deconv_addto_args_grad = [mx.nd.array(deconv_addto_args_grad_npy[0]),
                              mx.nd.array(deconv_addto_args_grad_npy[1])]
    exe_deconv = deconv.bind(default_context(), args=deconv_args, args_grad=deconv_args_grad)
    exe_deconv.forward(is_train=True)
    deconv_out_grad = conv_data[:]
    exe_deconv.backward(deconv_out_grad)
    assert_almost_equal(conv_args_grad[1], deconv_args_grad[1], rtol=1e-3, atol=1e-2)
    # Test AddTo
    exe_deconv_addto = deconv.bind(default_context(), args=deconv_args,
                                   args_grad=deconv_addto_args_grad,
                                   grad_req="add")
    exe_deconv_addto.forward(is_train=True)
    deconv_out_grad = conv_data[:]
    exe_deconv_addto.backward(deconv_out_grad)
    assert_almost_equal(conv_args_grad[1].asnumpy() + deconv_addto_args_grad_npy[1],
                        deconv_addto_args_grad[1].asnumpy(), rtol=1e-3, atol=1e-2)


def check_deconvolution_target_shape(input_shape, kernel, stride, pad, adj, target_shape=None):
    data = mx.sym.Variable(name="data")
    if target_shape:
        deconv = mx.sym.Deconvolution(
            data=data, kernel=kernel, stride=stride, pad=pad, adj=adj, num_filter=5,
            target_shape = target_shape)
    else:
        deconv = mx.sym.Deconvolution(
            data=data, kernel=kernel, stride=stride, pad=pad, adj=adj, num_filter=5)
    arg_names = deconv.list_arguments()
    arg_shapes, out_shapes, _ = deconv.infer_shape(data=input_shape)
    default_target_size = 8
    if target_shape is None:
        target_shape = (default_target_size,) * len(kernel)
    assert out_shapes[0] == (input_shape[0], 5) + target_shape


@with_seed()
def test_deconvolution():
    # 2D
    check_deconvolution_target_shape(
        input_shape         = (2,3,4,4),
        kernel              = (3,3),
        stride              = (2,2),
        target_shape        = (8,8),
        pad                 = (99,99),  # will be ignored
        adj                 = (101,101),  # will be ignored
    )
    check_deconvolution_target_shape(
        input_shape         = (2,3,4,4),
        kernel              = (3,3),
        stride              = (2,2),
        pad                 = (1,1),
        adj                 = (1,1),
    )
    check_deconvolution_forward_backward(
        input_shape         = (1,1,5,5),
        num_filter          = 1,
        kernel              = (3,3),
        stride              = (1,1),
        pad                 = (1,1)
    )
    check_deconvolution_forward_backward(
        input_shape         = (32,3,28,28),
        num_filter          = 3,
        kernel              = (3,3),
        stride              = (1,1),
        pad                 = (1,1)
    )
    check_deconvolution_forward_backward(
        input_shape         = (10, 3, 403, 403),
        num_filter          = 3,
        kernel              = (7,7),
        stride              = (5,5),
        pad                 = (2,2)
    )
    check_deconvolution_gradient(
        input_shape = (1,3,5,5),
        num_filter = 3,
        pad = (1,1)
    )
    check_deconvolution_gradient(
        input_shape = (5,3,100,100),
        num_filter = 3,
        pad = (3,3)
    )
    # 1D
    check_deconvolution_target_shape(
        input_shape         = (2,3,4),
        kernel              = (3,),
        stride              = (2,),
        target_shape        = (8,),
        pad                 = (99,),  # will be ignored
        adj                 = (101,),  # will be ignored
    )
    check_deconvolution_target_shape(
        input_shape         = (2,3,4),
        kernel              = (3,),
        stride              = (2,),
        pad                 = (1,),
        adj                 = (1,),
    )
    check_deconvolution_forward_backward(
        input_shape         = (1,1,5),
        num_filter          = 1,
        kernel              = (3,),
        stride              = (1,),
        pad                 = (1,)
    )
    check_deconvolution_forward_backward(
        input_shape         = (32,3,28),
        num_filter          = 3,
        kernel              = (3,),
        stride              = (1,),
        pad                 = (1,)
    )
    check_deconvolution_forward_backward(
        input_shape         = (10, 3, 403),
        num_filter          = 3,
        kernel              = (7,),
        stride              = (5,),
        pad                 = (2,)
    )
    check_deconvolution_gradient(
        input_shape = (1,3,5),
        num_filter = 3,
        pad = (1,)
    )
    check_deconvolution_gradient(
        input_shape = (5,3,100),
        num_filter = 3,
        pad = (3,)
    )

@with_seed()
def test_deconvolution_forward_with_bias():
    """Check if deconvolution forward can work well with bias=True
    """
    def check_deconvolution_forward_with_bias(shape=(1, 16, 5, 5), num_filter=32, num_group=1, kernel=(3, 3), pad=(1, 1)):
        x = mx.sym.Variable('x')
        w = mx.sym.Variable('w')
        b = mx.sym.Variable('b')
        y_nb = mx.sym.Deconvolution(data=x, weight=w, num_filter=num_filter, num_group=num_group, kernel=kernel, no_bias=True, pad=pad)
        y_b = mx.sym.Deconvolution(data=x, weight=w, bias=b, num_filter=num_filter, num_group=num_group, kernel=kernel, no_bias=False, pad=pad)
        
        
        exe_nb = y_nb.simple_bind(ctx=mx.cpu(), x=shape, grad_req='null')
        exe_b = y_b.simple_bind(ctx=mx.cpu(), x=shape, grad_req='null')

        
        data = np.random.uniform(-5, 5, size=exe_b.arg_arrays[0].shape)
        weights = np.random.normal(size=exe_b.arg_arrays[1].shape)
        bias = np.random.normal(size=exe_b.arg_arrays[2].shape)
        
        def exe_forward(exe):
            exe.arg_arrays[0][:] = data
            exe.arg_arrays[1][:] = weights
            if len(exe.arg_arrays) == 3:
                exe.arg_arrays[2][:] = bias
            return exe.forward(is_train=False)[0].asnumpy()
        
        out_nb = exe_forward(exe_nb)
        out_b = exe_forward(exe_b)
        bias = np.broadcast_to(bias, [np.prod(out_nb.shape[2:])] + [num_filter]).T
        bias = np.broadcast_to(bias.reshape((num_filter, *out_nb.shape[2:])), out_b.shape)
        assert_almost_equal(out_nb + bias, out_b)


    check_deconvolution_forward_with_bias((1, 16, 5), 32, 1, (3,), (1,))
    check_deconvolution_forward_with_bias((32, 16, 5), 32, 1, (3,), (1,))
    check_deconvolution_forward_with_bias((1, 16, 5, 5), 32, 1, (3, 3), (1, 1))
    check_deconvolution_forward_with_bias((32, 16, 5, 5), 32, 1, (3, 3), (1, 1))

def check_nearest_upsampling_with_shape(shapes, scale, root_scale):
    arr = {'arg_%d'%i: mx.random.uniform(-10.0, 10.0, shape, ctx=mx.cpu()).copyto(default_context()) for i, shape in zip(range(len(shapes)), shapes)}
    arr_grad = {'arg_%d'%i: mx.nd.zeros(shape) for i, shape in zip(range(len(shapes)), shapes)}

    up = mx.sym.UpSampling(*[mx.sym.Variable('arg_%d'%i) for i in range(len(shapes))], sample_type='nearest', scale=root_scale)
    exe = up.bind(default_context(), args=arr, args_grad=arr_grad)
    exe.forward(is_train=True)
    exe.backward(exe.outputs)
    for k in range(len(shapes)):
        name = 'arg_%d'%k
        assert_allclose(arr[name].asnumpy()*root_scale**2*scale**(2*k), arr_grad[name].asnumpy(), rtol=1e-4)


def check_bilinear_upsampling_with_shape(data_shape, weight_shape, scale, root_scale, num_filter):
    def _init_bilinear(arr, f):
        weight = np.zeros(np.prod(arr.shape), dtype='float32')
        shape = arr.shape
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % shape[3]
            y = (i // shape[3]) % shape[2]
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        arr[:] = weight.reshape(shape)
        return arr

    up = mx.sym.UpSampling(mx.sym.Variable("data"),
        mx.sym.Variable('weight'), sample_type='bilinear', scale=root_scale,
        num_filter=num_filter, num_args=2)
    arg_shapes, out_shapes, _ = up.infer_shape(data=data_shape)
    arr = {'data': mx.random.uniform(-5, 5, data_shape, ctx=mx.cpu()).copyto(default_context()),
        'weight':  mx.nd.array(_init_bilinear(mx.ndarray.empty(arg_shapes[1]).asnumpy(), root_scale))}

    arr_grad = [mx.nd.empty(s) for s in arg_shapes]
    exe = up.bind(default_context(), args=arr, args_grad=arr_grad)
    exe.forward(is_train=True)
    out = exe.outputs[0].asnumpy()
    exe.backward(exe.outputs)
    target_shape = (data_shape[2] * root_scale, data_shape[3] * root_scale)
    assert out.shape == data_shape[:2] + target_shape


@with_seed()
def test_nearest_upsampling():
    for root_scale in [1,2,3]:
        for scale in [1,2,3]:
            for num_shape in [1,2,3]:
                for base in [1,2,3]:
                    shapes = [(1,3,base*root_scale*scale**(num_shape-1-i),base*root_scale*scale**(num_shape-1-i)) for i in range(num_shape)]
                    check_nearest_upsampling_with_shape(shapes, scale, root_scale)


@with_seed()
def test_bilinear_upsampling():
    rootscale = [2,3]
    scales = [1,2,3]
    filters = [1,2,3]
    bases = [1,2,3]
    for params in itertools.product(rootscale, scales, filters, bases):
        root_scale, scale, num_filter, base = params
        # bilinear upsampling takes only 1 data and 1 weight
        # multi input mode is not applicable
        dimension = base*root_scale*scale
        kernel = 2 * root_scale - root_scale % 2
        data_shape = (1, num_filter, dimension, dimension)
        weight_shape = (1, num_filter, kernel, kernel)
        check_bilinear_upsampling_with_shape(data_shape, weight_shape, scale, root_scale, num_filter)

@with_seed()
def test_batchnorm_training():
    def check_batchnorm_training(stype):
        for shape in [(2, 3), (2, 3, 2, 2), (2, 8, 2, 2)]:
            data_tmp = np.random.normal(-0.1, 0.1, size=shape)
            s = shape[1],
            gamma = np.ones(s)
            beta = np.ones(s)
            gamma[1] = 3
            beta[0] = 3

            rolling_mean = np.random.uniform(size=s)
            rolling_std = np.random.uniform(size=s)

            data = mx.symbol.Variable('data', stype=stype)
            in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(gamma).tostype(stype),
                           mx.nd.array(beta).tostype(stype)]
            mean_std = [mx.nd.array(rolling_mean).tostype(stype), mx.nd.array(rolling_std).tostype(stype)]

            test = mx.symbol.BatchNorm_v1(data, fix_gamma=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

            test = mx.symbol.BatchNorm(data, fix_gamma=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

            test = mx.symbol.BatchNorm_v1(data, fix_gamma=True, use_global_stats=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

            test = mx.symbol.BatchNorm(data, fix_gamma=True, use_global_stats=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

            test = mx.symbol.BatchNorm_v1(data, fix_gamma=False)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

            test = mx.symbol.BatchNorm(data, fix_gamma=False)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

            test = mx.symbol.BatchNorm_v1(data, fix_gamma=False, use_global_stats=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

            test = mx.symbol.BatchNorm(data, fix_gamma=False, use_global_stats=True)
            check_numeric_gradient(test, in_location, mean_std, numeric_eps=1e-2, rtol=0.16, atol=1e-2)

            # Test varying channel axis
            dim = len(shape)
            for chaxis in range(-dim, dim):
                chaxis_true = chaxis
                if chaxis < 0:
                    chaxis_true = dim + chaxis

                shapex = shape

                channel_count = shapex[chaxis_true]
                data_tmp = np.random.normal(-0.1, 0.1, size=shapex)

                gamma = np.ones(channel_count)
                beta = np.ones(channel_count)
                if channel_count > 1:
                    gamma[1] = 3
                beta[0] = 3

                in_location = [mx.nd.array(data_tmp).tostype(stype), mx.nd.array(gamma).tostype(stype),
                               mx.nd.array(beta).tostype(stype)]

                xrolling_mean = np.random.uniform(size=channel_count)
                xrolling_std = np.random.uniform(size=channel_count)
                xmean_std = [mx.nd.array(xrolling_mean).tostype(stype),
                             mx.nd.array(xrolling_std).tostype(stype)]

                test = mx.symbol.BatchNorm(data, fix_gamma=True, axis=chaxis)
                check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-2, rtol=0.2, atol=0.01)

                test = mx.symbol.BatchNorm(data, fix_gamma=True, use_global_stats=True, axis=chaxis)
                check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-2, rtol=0.2, atol=0.01)

                test = mx.symbol.BatchNorm(data, fix_gamma=False, axis=chaxis)
                check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-2, rtol=0.2, atol=0.01)

                test = mx.symbol.BatchNorm(data, fix_gamma=False, use_global_stats=True, axis=chaxis)
                check_numeric_gradient(test, in_location, xmean_std, numeric_eps=1e-2, rtol=0.2, atol=0.01)

    check_batchnorm_training('default')


@with_seed()
def test_batchnorm():
    momentum = 0.9
    epsilon = 1e-5

    def _test_batchnorm_impl(op_name, shape, fix_gamma, cudnn_off, output_mean_var,
                             axis,
                             data_grad_req, gamma_grad_req, beta_grad_req):

        if op_name == 'BatchNorm':
            op = mx.nd.BatchNorm
        elif op_name == 'SyncBatchNorm':
            op = mx.nd.contrib.SyncBatchNorm
        else:
            raise ValueError('Not supported {}'.format(op_name))
        kwargs = dict(output_mean_var=output_mean_var)
        if op_name == 'SyncBatchNorm':
            if axis != 1:
                return
            key = str(op) + str(shape) + str(axis)
            kwargs.update(dict(key=key))
            if cudnn_off:
                return
        else:
            kwargs.update(dict(axis=axis, cudnn_off=cudnn_off))
        nch = shape[axis]

        if not fix_gamma:
            bn_gamma = mx.nd.random.uniform(shape=(nch,))
            bn_gamma.attach_grad(grad_req=gamma_grad_req)
        else:
            bn_gamma = mx.nd.ones(shape=(nch,))

        bn_beta = mx.nd.random.uniform(shape=(nch,))
        bn_beta.attach_grad(grad_req=beta_grad_req)

        bn_running_mean = mx.nd.zeros(nch)
        bn_running_var = mx.nd.ones(nch)

        running_mean = mx.nd.zeros(nch)
        running_var = mx.nd.ones(nch)
        num_iters = 10
        expand_shape = [1] * len(shape)
        expand_shape[axis] = shape[axis]
        data = mx.nd.random.uniform(shape=shape)
        data.attach_grad(grad_req=data_grad_req)
        adX, adW, adb = 0, 0, 0
        is_train = data_grad_req != 'null' or \
            (not fix_gamma and gamma_grad_req != 'null') or \
            beta_grad_req != 'null'
        for _ in range(num_iters):
            if data_grad_req != 'add':
                data = mx.nd.random.uniform(shape=shape)
                data.attach_grad(grad_req=data_grad_req)
            ograd = mx.nd.random.uniform(shape=shape)
            with mx.autograd.record():
                output = op(data, bn_gamma, bn_beta,
                            bn_running_mean, bn_running_var,
                            momentum=momentum, eps=epsilon,
                            fix_gamma=fix_gamma, **kwargs)
                if output_mean_var:
                    output, output_mean, output_std = output
                if is_train:
                    output.backward(ograd)
            mx.nd.waitall()

            data_mean = data.mean(
                axis=axis, exclude=True, keepdims=True)
            data_var = (data - data_mean).square().mean(axis=axis,
                                                        exclude=True,
                                                        keepdims=True)

            target_output = (data - data_mean) / \
                (data_var + epsilon).sqrt() * \
                bn_gamma.reshape(expand_shape) + \
                bn_beta.reshape(expand_shape)

            # squeeze data_mean and data_var
            data_mean_flat = data_mean.squeeze()
            data_var_flat = data_var.squeeze()

            running_mean = running_mean * momentum + \
                data_mean_flat * (1 - momentum)

            m = np.prod(shape) / shape[axis]
            # cudnn uses m-1 in the denominator of its sample variance calculation, not m
            sample_var_adjust = 1.0 if cudnn_off or fix_gamma else m / (m-1)
            running_var = running_var * momentum + \
                data_var_flat * sample_var_adjust * (1 - momentum)

            W = bn_gamma.reshape(expand_shape)
            dnx = ograd * W
            xsm = data - data_mean
            nd = 1.0 / mx.nd.sqrt(data_var + epsilon)
            nx = xsm * nd
            dvar = (dnx * xsm).sum(axis=axis, keepdims=True,
                                   exclude=True) * (-0.5) * mx.nd.power(nd, 3)
            dmean = -nd * dnx.sum(axis=axis, keepdims=True, exclude=True) - \
                dvar * xsm.mean(axis=axis, keepdims=True,
                                exclude=True) * 2.0
            dX = dnx * nd + dvar * xsm * (2.0 / m) + dmean * (1.0 / m)
            dW = (ograd * nx).sum(axis=axis, exclude=True)
            db = ograd.sum(axis=axis, exclude=True)
            adX = dX if data_grad_req != 'add' else adX + dX
            adW = dW if gamma_grad_req != 'add' else adW + dW
            adb = db if beta_grad_req != 'add' else adb + db

            atol, rtol = 5e-2, 5e-2

            if output_mean_var:
                assert_almost_equal(output_mean.asnumpy(),
                                    data_mean_flat.asnumpy(),
                                    atol=atol, rtol=rtol)
                if op != mx.nd.contrib.SyncBatchNorm:
                    assert_almost_equal(output_std.asnumpy(),
                                        (1.0 / (data_var_flat +
                                                epsilon).sqrt()).asnumpy(),
                                        atol=atol, rtol=rtol)
                else:
                    assert_almost_equal(output_std.asnumpy(),
                                        data_var_flat.asnumpy(),
                                        atol=atol, rtol=rtol)
            assert_almost_equal(output.asnumpy(), target_output.asnumpy(),
                                atol=atol, rtol=rtol)
            if is_train:
                assert_almost_equal(bn_running_mean.asnumpy(
                ), running_mean.asnumpy(), atol=atol, rtol=rtol)
                assert_almost_equal(bn_running_var.asnumpy(
                ), running_var.asnumpy(), atol=atol, rtol=rtol)

            if data_grad_req != 'null':
                assert_almost_equal(data.grad.asnumpy(),
                                    adX.asnumpy(), atol=atol, rtol=rtol)
            if not fix_gamma:
                if gamma_grad_req != 'null':
                    assert_almost_equal(
                        bn_gamma.grad.asnumpy(), adW.asnumpy(),
                        atol=atol, rtol=rtol)
            else:
                assert((bn_gamma.asnumpy() == 1).all())
            if beta_grad_req != 'null':
                assert_almost_equal(
                    bn_beta.grad.asnumpy(), adb.asnumpy(), atol=atol, rtol=rtol)

    op_names = ['BatchNorm', 'SyncBatchNorm']
    shapes = [(4, 2), (4, 3, 4), (4, 6, 4, 5), (4, 5, 6, 4, 5)]
    bools = [False, True]
    for op_name, shape, fix_gamma, cudnn_off, output_mean_var in itertools.product(
            op_names, shapes, bools, bools, bools):
        grad_reqs = ['write'] if len(shape) != 4 else ['null', 'write', 'add']
        for data_grad_req in grad_reqs:
            for gamma_grad_req in grad_reqs:
                if fix_gamma and gamma_grad_req != 'null':
                    continue
                for beta_grad_req in grad_reqs:
                    for axis in range(len(shape)):
                        _test_batchnorm_impl(
                            op_name, shape, fix_gamma, cudnn_off, output_mean_var,
                            axis,
                            data_grad_req,
                            gamma_grad_req, beta_grad_req)

@with_seed()
def test_groupnorm():
    acc_types = {'float16': 'float32', 'float32': 'float64', 'float64': 'float64'}
    def x_hat_helper(x, num_groups, eps):
        dtype = x.dtype
        dshape = x.shape
        assert len(dshape) == 4
        acc_type = acc_types[str(dtype)]
        new_shape = (dshape[0], num_groups, int(dshape[1] / num_groups), dshape[2], dshape[3])
        new_moments_shape = (dshape[0], num_groups, 1, 1, 1)
        data = x.reshape(new_shape)
        mean = np.mean(data, axis=(2, 3, 4), keepdims=False, dtype=acc_type).astype(dtype)
        std = np.sqrt(np.var(data, axis=(2, 3, 4), dtype=acc_type, keepdims=False).astype(dtype) + eps)
        x_hat = (data - mean.reshape(new_moments_shape)) / std.reshape(new_moments_shape)
        return x_hat, mean, std

    def np_groupnorm(data, gamma, beta, num_groups, eps):
        new_param_shape = (1, num_groups, 1, 1, 1)
        x_hat, mean, std = x_hat_helper(data, num_groups, eps)
        out = x_hat * gamma.reshape(new_param_shape) + beta.reshape(new_param_shape)
        return out.reshape(dshape), mean, std

    def np_groupnorm_grad(ograd, data, gamma, beta, mean, std, num_groups, eps):
        x_hat, mean, std = x_hat_helper(data, num_groups, eps)
        new_shape = x_hat.shape
        dshape = data.shape
        dtype = data.dtype
        new_moments_shape = (new_shape[0], num_groups, 1, 1, 1)
        new_param_shape = (1, num_groups, 1, 1, 1)
        acc_type = acc_types[str(dtype)]
        ograd = ograd.reshape(new_shape)
        data = data.reshape(new_shape)
        gamma = gamma.reshape(new_param_shape)
        beta = beta.reshape(new_param_shape)
        mean = mean.reshape(new_moments_shape)
        std = std.reshape(new_moments_shape)
        beta_grad = np.sum(ograd, axis=(0, 2, 3, 4), dtype=acc_type, keepdims=False).astype(dtype)
        gamma_grad = np.sum(x_hat * ograd, axis=(0, 2, 3, 4), dtype=acc_type, keepdims=False).astype(dtype)
        x_hat_grad = ograd * gamma
        ograd_mult = x_hat_grad / std
        red_out = np.mean(ograd_mult, axis=(2, 3, 4), dtype=acc_type, keepdims=True).astype(dtype)
        data_grad = ograd_mult - red_out
        red_out = np.mean(ograd_mult * x_hat, axis=(2, 3, 4), dtype=acc_type, keepdims=True).astype(dtype)
        data_grad = data_grad - x_hat * red_out
        return data_grad.reshape(dshape), gamma_grad, beta_grad


    batch_size = random.randint(1, 8)
    num_groups = random.randint(2, 3)
    num_channels = random.randint(2, 3) * num_groups
    height = random.randint(1, 5)
    width = random.randint(1, 5)
    dshape = (batch_size, num_channels, height, width)
    param_shape = (num_groups,)
    temp_shape = (batch_size, num_groups, int(num_channels / num_groups), height, width)
    np_data = np.random.uniform(0.2, 1.0, dshape)
    np_gamma = np.random.uniform(-1.0, 1.0, param_shape)
    np_beta = np.random.uniform(-1.0, 1.0, param_shape)
    data_sym = mx.sym.Variable("data")
    gamma_sym = mx.sym.Variable("gamma")
    beta_sym = mx.sym.Variable("beta")
    for dtype in [np.float16, np.float32, np.float64]:
        eps = 1e-2 if dtype == np.float16 else 1e-5
        mx_data = mx.nd.array(np_data, dtype=dtype)
        mx_gamma = mx.nd.array(np_gamma, dtype=dtype)
        mx_beta = mx.nd.array(np_beta, dtype=dtype)
        np_out, np_mean, np_std = np_groupnorm(np_data.astype(dtype),
                                               np_gamma.astype(dtype),
                                               np_beta.astype(dtype),
                                               num_groups=num_groups,
                                               eps=eps)
        mx_sym = mx.sym.GroupNorm(data=data_sym, gamma=gamma_sym, beta=beta_sym,
                                  num_groups=num_groups, eps=eps, output_mean_var=True)
        check_symbolic_forward(mx_sym, [mx_data, mx_gamma, mx_beta], [np_out, np_mean, np_std],
                               rtol=1e-2 if dtype == np.float16 else 1e-3,
                               atol=5e-3 if dtype == np.float16 else 1e-4, dtype=dtype)
        mx_sym = mx.sym.GroupNorm(data=data_sym, gamma=gamma_sym, beta=beta_sym,
                                  num_groups=num_groups, eps=eps, output_mean_var=False)
        np_ograd = np.random.uniform(-1.0, 1.0, dshape).astype(dtype)
        np_data_grad, np_gamma_grad, np_beta_grad = np_groupnorm_grad(np_ograd,
                                                                      np_data.astype(dtype),
                                                                      np_gamma.astype(dtype),
                                                                      np_beta.astype(dtype),
                                                                      np_mean, np_std,
                                                                      num_groups, eps)
        check_symbolic_backward(mx_sym, [mx_data, mx_gamma, mx_beta], [mx.nd.array(np_ograd)],
                                [np_data_grad, np_gamma_grad, np_beta_grad],
                                rtol=1e-2 if dtype == np.float16 else 1e-3,
                                atol=5e-2 if dtype == np.float16 else 1e-4, dtype=dtype)


@with_seed()
def test_convolution_grouping():
    for dim in [1, 2, 3]:
        num_filter = 4
        for num_group in [1, 2]:
            kernel = (3,) * dim
            shape = (1, 4) + (9,) * dim

            x = mx.sym.Variable('x')
            w = mx.sym.Variable('w')
            b = mx.sym.Variable('b')
            y1 = mx.sym.Convolution(data=x, weight=w, bias=b, num_filter=num_filter, num_group=num_group, kernel=kernel)
            xslice = mx.sym.SliceChannel(data=x, num_outputs=num_group, axis=1)
            wslice = mx.sym.SliceChannel(data=w, num_outputs=num_group, axis=0)
            bslice = mx.sym.SliceChannel(data=b, num_outputs=num_group, axis=0)
            y2 = mx.sym.Concat(*[mx.sym.Convolution(data=xslice[i], weight=wslice[i], bias=bslice[i],
                                                    num_filter=num_filter//num_group, kernel=kernel)
                            for i in range(num_group)])

            exe1 = y1.simple_bind(default_context(), x=shape)
            exe2 = y2.simple_bind(default_context(), x=shape, w=(num_filter, shape[1]//num_group) + kernel, b=(num_filter,))
            for arr1, arr2 in zip(exe1.arg_arrays, exe2.arg_arrays):
                arr1[:] = np.float32(np.random.normal(size=arr1.shape))
                arr2[:] = arr1
            exe1.forward(is_train=True)
            exe1.backward(exe1.outputs[0])
            exe2.forward(is_train=True)
            exe2.backward(exe2.outputs[0])

            for arr1, arr2 in zip(exe1.outputs + exe1.grad_arrays, exe2.outputs + exe2.grad_arrays):
                np.testing.assert_allclose(arr1.asnumpy(), arr2.asnumpy(), rtol=1e-3, atol=1e-3)


@unittest.skip("Flaky test https://github.com/apache/incubator-mxnet/issues/14052")
@with_seed()
def test_depthwise_convolution():
    for dim in [1,2]:
        for num_base in [1, 4, 16, 32, 64]:
            for kernel_x in [3, 5]:
                for stride_x in [1, 2]:
                    for pad_x in [0, 1]:
                        for in_size in [7, 32]:
                            kernel = (kernel_x,) * dim
                            stride = (stride_x,) * dim
                            pad = (pad_x,) * dim
                            num_filter = num_base
                            num_group = num_base
                            shape = (2, num_base) + (in_size,) * dim

                            x = mx.sym.Variable('x')
                            w = mx.sym.Variable('w')
                            b = mx.sym.Variable('b')
                            y1 = mx.sym.Convolution(data=x, weight=w, bias=b, num_filter=num_filter, num_group=num_group,
                                    kernel=kernel, stride=stride, pad=pad)
                            xslice = mx.sym.SliceChannel(data=x, num_outputs=num_group, axis=1)
                            wslice = mx.sym.SliceChannel(data=w, num_outputs=num_group, axis=0)
                            bslice = mx.sym.SliceChannel(data=b, num_outputs=num_group, axis=0)
                            y2 = mx.sym.Concat(*[mx.sym.Convolution(data=xslice[i], weight=wslice[i], bias=bslice[i],
                                                                    num_filter=num_filter//num_group, kernel=kernel,
                                                                    stride=stride, pad=pad)
                                                for i in range(num_group)])

                            dev = default_context()
                            exe1 = y1.simple_bind(dev, x=shape)
                            exe2 = y2.simple_bind(dev, x=shape, w=(num_filter, shape[1]//num_group)+kernel,
                                    b=(num_filter,))
                            for arr1, arr2 in zip(exe1.arg_arrays, exe2.arg_arrays):
                                arr1[:] = np.random.normal(size=arr1.shape)
                                arr2[:] = arr1
                            exe1.forward(is_train=True)
                            exe1.backward(exe1.outputs[0])
                            exe2.forward(is_train=True)
                            exe2.backward(exe2.outputs[0])

                            for arr1, arr2 in zip(exe1.outputs + exe1.grad_arrays, exe2.outputs + exe2.grad_arrays):
                                assert_allclose(arr1, arr2, rtol=1e-3, atol=1e-3)


@with_seed()
def test_convolution_independent_gradients():
    # NOTE(zixuanweeei): Flaky test tracked by https://github.com/apache/incubator-mxnet/issues/15603.
    # GPU context will be enabled after figuring out the possible issue tracked at
    # https://github.com/apache/incubator-mxnet/issues/15638.
    ctx = mx.cpu()
    atol = 1.0e-3
    rtol = 1.0e-3
    reqs = ["null", "write", "add"]
    var_names = ["x", "w", "b"]
    dims = [1, 2]
    num_bases = [1, 8]
    kernel_xs = [3, 5]
    stride_xs = [1, 2]
    pad_xs = [0, 1]
    in_sizes = [7, 32]
    no_biases = [True, False]
    for dim, num_base, kernel_x, stride_x, pad_x , in_size, no_bias in \
            itertools.product(dims, num_bases, kernel_xs, stride_xs, pad_xs, in_sizes, no_biases):
        # Prepare params shape
        kernel = (kernel_x,) * dim
        stride = (stride_x,) * dim
        pad = (pad_x,) * dim
        num_filter = num_base
        x_shape = (2, num_base) + (in_size,) * dim
        w_shape = (num_filter, num_base) + kernel

        # Symbols definition
        x = mx.sym.Variable('x')
        w = mx.sym.Variable('w')
        b = mx.sym.Variable('b') if not no_bias else None
        conv = mx.sym.Convolution(x, w, b, num_filter=num_filter,
            kernel=kernel, stride=stride, pad=pad, no_bias=no_bias)

        for req_kind in reqs:
            # Binding args for conv with possible dependent gradients
            base_args = {
                'x': mx.nd.random.normal(shape=x_shape, ctx=ctx),
                'w': mx.nd.random.normal(shape=w_shape, ctx=ctx),
                'b': mx.nd.random.normal(shape=(num_filter, ), ctx=ctx) if not no_bias else None}
            args1 = copy.deepcopy(base_args)
            grad1 = {
                'x': mx.nd.zeros(shape=x_shape, ctx=ctx),
                'w': mx.nd.zeros(shape=w_shape, ctx=ctx),
                'b': mx.nd.zeros(shape=(num_filter, ), ctx=ctx) if not no_bias else None}

            grad_req1 = [req_kind] * 3
            grad_req1 = dict(zip(var_names, grad_req1))

            exe1 = conv.bind(ctx, args1, args_grad=grad1, grad_req=grad_req1)
            exe1.forward(is_train=True)
            exe1.backward(exe1.outputs[0])

            for x_req, w_req, b_req in itertools.product(reqs, repeat=3):
                # Binding args for conv with independent gradients
                args2 = copy.deepcopy(base_args)    # Deepcopy the same params of `exe1`
                grad2 = {
                    'x': mx.nd.zeros(shape=x_shape, ctx=ctx),
                    'w': mx.nd.zeros(shape=w_shape, ctx=ctx),
                    'b': mx.nd.zeros(shape=(num_filter, ), ctx=ctx) if not no_bias else None}
                grad_req2 = {"x": x_req, "w": w_req, "b": b_req}
                exe2 = conv.bind(ctx, args2, args_grad=grad2, grad_req=grad_req2)

                exe2.forward(is_train=True)
                np.testing.assert_allclose(exe1.outputs[0].asnumpy(),
                    exe2.outputs[0].asnumpy(), rtol=rtol, atol=atol)

                exe2.backward(exe2.outputs[0])
                for var_name in var_names:
                    if var_name == "b" and no_bias:
                        continue
                    if grad_req2[var_name] == "null":
                        exe2_var_grad = grad2[var_name].asnumpy()
                        np.testing.assert_allclose(exe2_var_grad,
                            np.zeros_like(exe2_var_grad), rtol=rtol, atol=atol)
                    if grad_req2[var_name] != grad_req1[var_name]:
                        continue
                    np.testing.assert_allclose(args1[var_name].asnumpy(),
                        args2[var_name].asnumpy(), rtol=rtol, atol=atol)
                    np.testing.assert_allclose(grad1[var_name].asnumpy(),
                        grad2[var_name].asnumpy(), rtol=rtol, atol=atol)


def gen_broadcast_data(idx):
    # Manually set test cases
    binary_op_data_shape = np.array(
        [[[2, 5, 1, 30, 7], [1, 5, 448, 30, 1]],
        [[10, 49, 1, 77, 17], [10, 1, 2, 1, 17]],
        [[13, 2, 65, 2,  1], [13, 1, 65, 1, 225]],
        [[9, 434, 4, 2, 37], [9, 1, 4, 1, 37]],
        [[2, 52, 1, 4, 1], [1, 52, 60, 1, 37]],
        [[1, 23, 7, 122, 50], [2, 1, 7, 1, 50]],
        [[1, 17, 1, 5, 1], [22, 1, 2, 1, 28]],
        [[29, 1, 2, 1, 8], [29, 22, 1, 130, 1]],
        [[2, 36, 1, 427, 3], [1, 36, 11, 427, 1]],
        [[1, 2, 1, 100, 7], [1, 2, 448, 100, 1]],
        [[1, 2, 495, 77, 7], [1, 2, 1, 1, 7]],
        [[1, 43, 65, 2, 1], [1, 43, 65, 1, 225]],
        [[1, 92, 434, 2, 2], [1, 92, 1, 2, 2]],
        [[1, 92, 1, 4, 1], [1, 92, 134, 1, 17]],
        [[1, 53, 2, 122, 143], [1, 1, 2, 1, 143]],
        [[1, 179, 1, 87, 17], [1, 179, 1, 1, 17]],
        [[1, 1, 17, 5, 1], [1, 22, 1, 1, 28]],
        [[1, 2, 1, 1, 8], [1, 2, 52, 430, 1]],
        [[1, 163, 1, 22, 3], [1, 163, 116, 22, 1]],
        [[1, 1, 44, 30, 7], [1, 1, 44, 30, 1]],
        [[1, 1, 1, 1, 28], [1, 127, 1, 5, 28]],
        [[1, 2, 394, 38, 1], [1, 2, 394, 38, 16]],
        [[1, 10, 49, 77, 17], [1, 1, 1, 1, 17]],
        [[1, 431, 6, 2, 225], [1, 1, 6, 2, 225]],
        [[1, 15, 1, 28, 1], [1, 15, 1, 28, 463]],
        [[1, 129, 2, 48, 96], [1, 129, 2, 1, 1]],
        [[1, 1, 403, 17, 2], [1, 44, 403, 17, 2]],
        [[1, 1, 65, 2, 22], [1, 1, 65, 1, 1]],
        [[1, 24, 103, 17, 18], [1, 24, 1, 1, 1]],
        [[1, 1, 1, 1, 2], [1, 24, 194, 50, 1]],
        [[1, 1, 107, 84, 9], [1, 1, 1, 1, 1]]])
    if idx < binary_op_data_shape.shape[0]:
        l_shape = binary_op_data_shape[idx][0]
        r_shape = binary_op_data_shape[idx][1]
    else:
        # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
        ndim = np.random.randint(1, 6)
        shape = np.random.randint(1, 6, size=(ndim,))
        l_same_dim = np.random.randint(0, 5)
        r_same_dim = np.random.randint(0, 5)
        l_axis_flags = np.random.randint(0, 2, size=ndim)
        r_axis_flags = np.random.randint(0, 2, size=ndim)
        if l_same_dim == 4:
            l_axis_flags = np.ones(ndim)
        if r_same_dim == 4:
            r_axis_flags = np.ones(ndim)
        l_shape = shape.copy()
        r_shape = shape.copy()
        l_shape[np.where(l_axis_flags == 0)] = 1
        r_shape[np.where(r_axis_flags == 0)] = 1
    return [np.random.random(l_shape), np.random.random(r_shape)]


def gen_broadcast_data_int(idx):
    d = gen_broadcast_data(idx);
    return [np.round(d[0]*100).astype(int), np.round(d[1]*100).astype(int)]


def gen_binary_data(dummy):
    ndim = np.random.randint(1, 6)
    shape = np.random.randint(1, 6, size=(ndim,))
    #print("gen shape {}".format(shape))
    return [np.random.random(shape), np.random.random(shape)]


def gen_binary_data_int(dummy):
    d = gen_binary_data(dummy);
    return [np.round(d[0]*100).astype(int), np.round(d[1]*100).astype(int)]


def check_binary_op_forward(symbol, baseline, gen_data, rtol=1e-3, atol=1e-5, mx_nd_func=None):
    sample_num = 200
    for i in range(sample_num):
        d = gen_data(i)
        y = symbol.bind(default_context(), args={'a': mx.nd.array(d[0]), 'b': mx.nd.array(d[1])})
        y.forward(is_train=True)
        y = y.outputs[0].asnumpy()
        x = baseline(d[0], d[1]).astype(y.dtype)

        #np.set_printoptions(precision=20)

        a = d[0]
        b = d[1]
        #print("a: {} {}".format(a.dtype, a))
        #print("a: {} {}".format(b.dtype, b))

        #print("x: {} {}".format(x.dtype, x))
        #print("y: {} {}".format(y.dtype, y))
        if mx_nd_func is not None:
            d0 = mx.nd.array(d[0], dtype=d[0].dtype)
            d1 = mx.nd.array(d[1], dtype=d[1].dtype)
            assert_almost_equal(y, mx_nd_func(d0, d1).asnumpy(), rtol=rtol, atol=atol)
        idx = np.abs(x-y) > atol+rtol*np.abs(x)
        if idx.any():
            import binascii
            np.set_printoptions(precision=20)
            logging.error('found precision problem:')
            d[0] = np.broadcast_to(d[0], x.shape)
            d[1] = np.broadcast_to(d[1], x.shape)
            logging.error('input a: {}'.format(d[0][idx]))
            logging.error('input b: {}'.format(d[1][idx]))
            logging.error("output x: {} {}".format(x.dtype, x))
            logging.error("output y: {} {}".format(y.dtype, y))
            def ftohex(xs):
                import struct
                return list(map(lambda x: binascii.hexlify(struct.pack('d', x)), xs.flatten()))
            logging.error('output x in baseline(a, b): {}'.format(x[idx]))
            logging.error('output y in symbol(a, b): {}'.format(y[idx]))
            logging.error('output x in baseline(a,b) hex: {}'.format(ftohex(x[idx])))
            logging.error('output y in symbol(a,b) hex: {}'.format(ftohex(y[idx])))
            logging.error('input a hex: {}'.format(ftohex(d[0][idx])))
            logging.error('input a hex: {}'.format(ftohex(d[1][idx])))

            logging.error('diff: {}'.format(np.abs(x-y)[idx] - atol-rtol*np.abs(x)[idx]))
        assert_allclose(y, x, rtol=rtol, atol=atol)


def check_binary_op_backward(symbol, baseline, gen_data, rtol=1e-3, atol=1e-5):
    sample_num = 200
    for i in range(sample_num):
        d = gen_data(i)
        out = np.random.random((d[0] + d[1]).shape)

        def reduce_op(shape, x):
            if shape == x.shape:
                return x
            keepdims_shape = list(x.shape)
            for i in range(len(shape)):
                if x.shape[i] != shape[i]:
                    keepdims_shape[i] = 1
                    x = np.sum(x, axis=i).reshape(keepdims_shape)
            return x

        baseline_grad1, baseline_grad2 = baseline(out, d[0], d[1])
        x_1 = reduce_op(d[0].shape, baseline_grad1)
        x_2 = reduce_op(d[1].shape, baseline_grad2)
        y_1 = mx.nd.empty(d[0].shape)
        y_2 = mx.nd.empty(d[1].shape)
        y = symbol.bind(default_context(), args={'a': mx.nd.array(d[0]), 'b': mx.nd.array(d[1])},
                        args_grad=[y_1, y_2])
        y.forward(is_train=True)
        y.backward([mx.nd.array(out)])
        assert_allclose(y_1.asnumpy(), x_1, rtol=rtol, atol=atol)
        assert_allclose(y_2.asnumpy(), x_2, rtol=rtol, atol=atol)


@with_seed()
def test_binary_op():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')

    def test_bplus(a, b):
        c = a + b
        check_binary_op_forward(c, lambda a, b: a + b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, g_out), gen_binary_data)

    def test_bminus(a, b):
        c = a - b
        check_binary_op_forward(c, lambda a, b: a - b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, - g_out), gen_binary_data)

    def test_bmul(a, b):
        c = a * b
        check_binary_op_forward(c, lambda a, b: a * b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out * b, g_out * a), gen_binary_data)

    def test_bdiv(a, b):
        c = a / b
        check_binary_op_forward(c, lambda a, b: a / b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out / b, - g_out * a / (b * b)), gen_binary_data)

    def test_bmod(a, b):
        # Python and numpy operate only in double so to avoid numerical errors we have to use
        # doubles as well. This was a flaky test before when using float32. seed 1688524483, 1768433044
        #c = a % b
        c = mx.sym.cast(a, dtype='float64') % mx.sym.cast(b, dtype='float64')
        # '%' is sensitive to the precision of the calculation.  Force numpy to match mxnet's float32.
        check_binary_op_forward(c, lambda a, b: np.float32(a) % np.float32(b), gen_binary_data, rtol=0, atol=0)
        check_binary_op_backward(c,
            lambda g_out, a, b: (g_out, - g_out * (np.float32(a) // np.float32(b))), gen_binary_data)

    def test_bmod_int(a, b):
        c = mx.sym.cast(a, dtype='int32') % mx.sym.cast(b, dtype='int32')
        check_binary_op_forward(c, lambda a, b: a % b, gen_binary_data_int)
        check_binary_op_backward(c, lambda g_out, a, b: (np.zeros_like(a), np.zeros_like(b)), gen_binary_data_int)

    def test_bpow(a, b):
        c = a ** b
        check_binary_op_forward(c, lambda a, b: a ** b, gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out * a **(b - 1) * b,
                                        g_out * a ** b * np.log(a)), gen_binary_data)

    def test_bneq(a, b):
        c = a != b
        # '!=' is sensitive to the precision of the comparison.  Force numpy to match mxnet's float32.
        # Issue exposed with seed 1644387363
        check_binary_op_forward(c, lambda a, b: (np.float32(a) != np.float32(b)).astype(a.dtype), gen_binary_data)
        check_binary_op_backward(c, lambda g_out, a, b: (np.zeros_like(a), np.zeros_like(b)), gen_binary_data)

    test_bplus(a, b)
    test_bminus(a, b)
    test_bmul(a, b)
    test_bdiv(a, b)
    test_bmod(a, b)
    test_bmod_int(a, b)
    test_bpow(a, b)
    test_bneq(a, b)

@with_seed()
def test_broadcast_binary_op():
    def check_bmaxmin_gradient(test_sym, x, y, delta, rtol, atol):
        """This function ensures that checking the numerical gradient of
        broadcast_max/min is not crossing the boundary y=x where there
        is no gradient definition at those sigularities."""
        x_max = np.max(x)
        y = x_max + 2 * delta + np.random.random(y.shape)
        check_numeric_gradient(test_sym, [x, y], numeric_eps=delta, rtol=rtol, atol=atol)

        x_min = np.min(x)
        y = x_min - 2 * delta - np.random.random(y.shape)
        check_numeric_gradient(test_sym, [x, y], numeric_eps=delta, rtol=rtol, atol=atol)

    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')

    def test_bplus(a, b):
        c = mx.sym.broadcast_plus(a, b)
        check_binary_op_forward(c, lambda a, b: a + b, gen_broadcast_data, mx_nd_func=mx.nd.add)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, g_out), gen_broadcast_data)

    def test_bminus(a, b):
        c = mx.sym.broadcast_minus(a, b)
        check_binary_op_forward(c, lambda a, b: a - b, gen_broadcast_data, mx_nd_func=mx.nd.subtract)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out, - g_out), gen_broadcast_data)

    def test_bmul(a, b):
        c = mx.sym.broadcast_mul(a, b)
        check_binary_op_forward(c, lambda a, b: a * b, gen_broadcast_data, mx_nd_func=mx.nd.multiply)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out * b, g_out * a), gen_broadcast_data)

    def test_bdiv(a, b):
        c = mx.sym.broadcast_div(a, b)
        check_binary_op_forward(c, lambda a, b: a / b, gen_broadcast_data, mx_nd_func=mx.nd.divide)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out / b, - g_out * a / (b * b)), gen_broadcast_data)

    def test_bmod(a_, b_):
        # Python and numpy operate only in double so to avoid numerical errors we have to use
        # doubles as well. This was a flaky test before when using float32. seed 1688524483, 1768433044
        a = mx.sym.cast(a_, dtype='float64')
        b = mx.sym.cast(b_, dtype='float64')
        # '%' is sensitive to the precision of the calculation.  Force numpy to match mxnet's float32.
        c = mx.sym.broadcast_mod(a, b)
        check_binary_op_forward(c, lambda a, b: a % b, gen_broadcast_data, atol=1, mx_nd_func=mx.nd.modulo)
        check_binary_op_backward(c,
                                 lambda g_out, a, b: (g_out, - g_out * (np.float32(a) // np.float32(b))), gen_binary_data)

    def test_bmod_int(a, b):
        c = mx.sym.broadcast_mod(mx.sym.cast(a, dtype='int32'), mx.sym.cast(b, dtype='int32'))
        check_binary_op_forward(c, lambda a, b: a % b, gen_broadcast_data_int, mx_nd_func=mx.nd.modulo)
        check_binary_op_backward(c, lambda g_out, a, b: (np.zeros_like(a), np.zeros_like(b)), gen_broadcast_data_int)

    def test_bpow(a, b):
        c = mx.sym.broadcast_power(a, b)
        check_binary_op_forward(c, lambda a, b: a ** b, gen_broadcast_data, mx_nd_func=mx.nd.power)
        check_binary_op_backward(c, lambda g_out, a, b: (g_out * a **(b - 1) * b,
                                                         g_out * a ** b * np.log(a)), gen_broadcast_data)

    def test_bequal(a, b):
        c = mx.sym.broadcast_equal(a, b)
        check_binary_op_forward(c, lambda a, b: (a == b).astype(a.dtype), gen_broadcast_data_int,
                                mx_nd_func=mx.nd.equal)
        check_binary_op_backward(c, lambda g_out, a, b: (np.zeros_like(a), np.zeros_like(b)), gen_broadcast_data_int)

    def test_bmax(a, b):
        c = mx.sym.broadcast_maximum(a, b)
        check_binary_op_forward(c, lambda x, y: np.maximum(x, y), gen_broadcast_data, mx_nd_func=mx.nd.maximum)
        # pass idx=200 to gen_broadcast_data so that generated ndarrays' sizes are not too big
        data = gen_broadcast_data(idx=200)
        check_bmaxmin_gradient(c, data[0], data[1], 0.001, 1e-2, 1e-3)

    def test_bmin(a, b):
        c = mx.sym.broadcast_minimum(a, b)
        check_binary_op_forward(c, lambda x, y: np.minimum(x, y), gen_broadcast_data, mx_nd_func=mx.nd.minimum)
        # pass idx=200 to gen_broadcast_data so that generated ndarrays' sizes are not too big
        data = gen_broadcast_data(idx=200)
        check_bmaxmin_gradient(c, data[0], data[1], 0.001, 1e-2, 1e-3)

    def test_band(a, b):
        c = mx.sym.broadcast_logical_and(a, b)
        check_binary_op_forward(c, lambda x, y: np.logical_and(x, y), gen_broadcast_data, mx_nd_func=mx.nd.logical_and)
        # pass idx=200 to gen_broadcast_data so that generated ndarrays' sizes are not too big
        data = gen_broadcast_data(idx=200)
        check_bmaxmin_gradient(c, data[0], data[1], 0.001, 1e-2, 1e-3)

    def test_bor(a, b):
        c = mx.sym.broadcast_logical_or(a, b)
        check_binary_op_forward(c, lambda x, y: np.logical_or(x, y), gen_broadcast_data, mx_nd_func=mx.nd.logical_or)
        # pass idx=200 to gen_broadcast_data so that generated ndarrays' sizes are not too big
        data = gen_broadcast_data(idx=200)
        check_bmaxmin_gradient(c, data[0], data[1], 0.001, 1e-2, 1e-3)

    def test_bxor(a, b):
        c = mx.sym.broadcast_logical_xor(a, b)
        check_binary_op_forward(c, lambda x, y: np.logical_xor(x, y), gen_broadcast_data, mx_nd_func=mx.nd.logical_xor)
        # pass idx=200 to gen_broadcast_data so that generated ndarrays' sizes are not too big
        data = gen_broadcast_data(idx=200)
        check_bmaxmin_gradient(c, data[0], data[1], 0.001, 1e-2, 1e-3)

    test_bplus(a, b)
    test_bminus(a, b)
    test_bmul(a, b)
    test_bdiv(a, b)
    test_bmod(a, b)
    test_bmod_int(a, b)
    test_bpow(a, b)
    test_bequal(a, b)
    test_bmax(a, b)
    test_bmin(a, b)
    test_band(a, b)
    test_bor(a, b)
    test_bxor(a, b)

@with_seed()
def test_run_convolution_dilated_impulse_response(dil=(1,1), kernel_shape=(3,3), verbose=False):
    dim = len(dil)
    assert(len(kernel_shape) == dim)
    # Input for spike response
    data_size = 33
    data_shape = (1, 1) + (data_size,) * dim
    center = (0,0) + (data_size // 2,) * dim
    spike_imgs = np.zeros(shape=data_shape, dtype=np.float32)
    spike_imgs[center] = 1.0
    spike_img = mx.nd.array(spike_imgs)
    spike_img2 = mx.nd.array(spike_imgs)

    kernel_weights = mx.nd.ones(shape=tuple([1,1]+list(kernel_shape)), dtype=np.float32)
    kernel_weights2 = mx.nd.ones(shape=tuple([1,1]+list(kernel_shape)), dtype=np.float32)

    kernel = mx.symbol.Variable('kernel')
    in_img = mx.symbol.Variable('input')
    net = mx.symbol.Convolution(in_img, num_filter=1,kernel=kernel_shape, dilate=dil, no_bias="true", name='test_convolution')
    net.list_arguments()
    be = net.bind(default_context(), args={ 'input' : spike_img, 'test_convolution_weight' : kernel_weights},
                args_grad={'input' : spike_img2, 'test_convolution_weight' : kernel_weights2 } )
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    ndo = be.outputs[0]

    out_grads = np.zeros(shape=be.outputs[0].shape, dtype=np.float32)
    out_grads[center] = 1.0
    out_grad = mx.nd.array(out_grads)
    be.backward([out_grad])
    vgrad = be.grad_arrays[0].asnumpy()
    out = out_o.reshape(out_o.shape[2:])
    nz_loc = np.nonzero(out)
    assert_allclose(np.sum(out),np.prod(kernel_shape),atol=1e-5)
    assert_allclose(np.sum(vgrad),np.prod(kernel_shape),atol=1e-5)

    # Now check whether the input gradient was computed correctly
    input_grad = mx.nd.array(vgrad)

    be = net.bind(default_context(), args={ 'input' : input_grad, 'test_convolution_weight' : kernel_weights})
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    assert_allclose(out_o[center],np.prod(kernel_shape),atol=1e-5)

    rnd_kernel_s = np.random.uniform(low=0.0, high=1.0, size=tuple([1,1]+list(kernel_shape))).astype(np.float32)
    impulse_error = mx.nd.array(out_o/np.sum(out_o)) # This should be 1.0 at [0,0,16,16]
    rnd_kernel = mx.nd.array(rnd_kernel_s)

    rnd_kernel2 = mx.nd.array(rnd_kernel_s)
    white_in = mx.nd.ones(shape=data_shape)
    white_in2 = mx.nd.ones(shape=data_shape)

    be = net.bind(default_context(), args={ 'input' : white_in, 'test_convolution_weight' : rnd_kernel},
                args_grad={'input' : white_in2, 'test_convolution_weight' : rnd_kernel2 } )

    be.forward(True)
    be.backward([impulse_error])
    out_orig = be.outputs[0].asnumpy()
    kernel_gradient = be.grad_arrays[1].asnumpy()

    dkernel = mx.nd.array(rnd_kernel_s + kernel_gradient)

    be = net.bind(default_context(), args={ 'input' : white_in, 'test_convolution_weight' : dkernel})

    be.forward(True)
    out = be.outputs[0].asnumpy()
    # Now do a simple check of the kernel gradient
    assert(out[center] - np.sum(kernel_gradient) - out_orig[center] < 0.001)


@with_seed()
def test_convolution_dilated_impulse_response():
    # 1D
    for dil in [ (1,), (2,), (3,) ]:
        for ks in [ (1,), (2,), (3,), (4,)]:
            test_run_convolution_dilated_impulse_response(dil=dil, kernel_shape=ks)
    # 2D
    for dil in [ (1,1), (2,2), (3,3) ]:
        for ks in [ (3,3), (4,4), (2,3), (3,2), (1,1) ]:
            test_run_convolution_dilated_impulse_response(dil=dil, kernel_shape=ks)
    # 3D
    for dil in [ (1,1,1), (2,2,2), (3,3,3) ]:
        for ks in [ (3,3,3), (4,4,4), (2,3,4), (3,2,4), (1,1,1) ]:
            test_run_convolution_dilated_impulse_response(dil=dil, kernel_shape=ks)


@with_seed()
def test_reshape():

    def test_reshape_new(src_shape, shape_args, reverse, dst_shape):
        net = mx.sym.Variable("data")
        net = mx.sym.Reshape(net, shape=shape_args, reverse=reverse)
        js = net.tojson()
        net = mx.sym.load_json(js)
        _, output_shape, __ = net.infer_shape(data=src_shape)
        assert output_shape[0] == dst_shape, \
            'Src Shape = %s, Shape Arguments = %s, Reverse = %s, Dst Shape = %s, ' \
            'Output Shape = %s' %(str(src_shape), str(shape_args), str(reverse),
                                  str(dst_shape), str(output_shape[0]))
        dat_npy = np.random.rand(*src_shape)
        grad_npy = np.random.rand(*dst_shape)
        exe = net.simple_bind(default_context(), data=src_shape)
        exe.arg_dict['data'][:] = dat_npy
        exe.forward(is_train=True)
        assert np.square(exe.outputs[0].asnumpy() - dat_npy.reshape(dst_shape)).mean() < 1E-7, \
            'Src Shape = %s, Shape Arguments = %s, Reverse = %s, Dst Shape = %s'\
            %(str(src_shape), str(shape_args), str(reverse), str(dst_shape))
        exe.backward(out_grads=mx.nd.array(grad_npy))
        assert np.square(exe.grad_dict['data'].asnumpy() - grad_npy.reshape(src_shape)).mean() < 1E-7, \
            'Src Shape = %s, Shape Arguments = %s, Reverse = %s, Dst Shape = %s'\
            %(str(src_shape), str(shape_args), str(reverse), str(dst_shape))

        for i in range(len(src_shape)):
            holdout_src_shape = list(src_shape)
            holdout_src_shape[i] = 0
            holdout_src_shape = tuple(holdout_src_shape)
            net = mx.sym.Variable('data')
            net = mx.sym.elemwise_add(net.reshape(shape_args, reverse=reverse), mx.sym.ones(shape=dst_shape))
            input_shape, output_shape, __ = net.infer_shape(data=holdout_src_shape)
            assert output_shape[0] == dst_shape, \
                'Holdout Src Shape = %s, Shape Arguments = %s, Reverse = %s, Dst Shape = %s, ' \
                'Output Shape = %s' %(str(holdout_src_shape), str(shape_args), str(reverse),
                                      str(dst_shape), str(output_shape[0]))
            assert input_shape[0] == src_shape, \
                'Holdout Src Shape = %s, Shape Arguments = %s, Reverse = %s, Dst Shape = %s, ' \
                'Output Shape = %s' %(str(holdout_src_shape), str(shape_args), str(reverse),
                                      str(dst_shape), str(output_shape[0]))

    # Test new api (Using shape)
    test_cases = [
        [(2, 3, 5, 5),  (0, -1),          False, (2, 75)],
        [(2, 3, 5, 5),  (0, 0, -1),       False, (2, 3, 25)],
        [(5, 3, 4, 5),  (0, -1, 0),       False, (5, 15, 4)],
        [(2, 3, 5, 4),  (-1, 0, 0),       False, (8, 3, 5)],
        [(2, 3, 5, 5),  (0, 0, 0, 0),     False, (2, 3, 5, 5)],
        [(2, 4, 5, 3),  (-1, 2, 2, 1),    False, (30, 2, 2, 1)],
        [(2, 3, 5, 6),  (-2,),            False, (2, 3, 5, 6)],
        [(2, 3, 5, 6),  (6, 1, -2),       False, (6, 1, 5, 6)],
        [(2, 3, 5, 6),  (-3, -3),         False, (6, 30)],
        [(2, 3, 5, 6),  (-3, -1),         False, (6, 30)],
        [(64,),         (-4, 16, 4),      False, (16, 4)],
        [(64,),         (-4, 16, -1),     False, (16, 4)],
        [(64, 1, 2, 3), (-4, 16, -1, -2), False, (16, 4, 1, 2, 3)],
        [(2, 3, 5, 5),  (0, -1),          True,  (5, 30)],
        [(2, 3, 5, 5),  (0, 0, -1),       True,  (3, 5, 10)],
        [(5, 3, 4, 5),  (0, -1, 0),       True,  (3, 20, 5)],
        [(2, 3, 5, 4),  (-1, 0, 0),       True,  (6, 5, 4)],
        [(2, 3, 4, 5),  (3, -1, 0),       True,  (3, 8, 5)],
        [(2, 3, 5, 5),  (5, 3, 0, -1),    True,  (5, 3, 5, 2)],
        [(2, 3, 5, 5),  (0, 0, 0, 0),     True,  (2, 3, 5, 5)],
        [(2, 3, 5, 6),  (-2,),            True,  (2, 3, 5, 6)],
        [(2, 3, 5, 6),  (-2, 1, 30),      True,  (2, 3, 1, 30)],
        [(2, 3, 5, 6),  (-3, -3),         True,  (6, 30)],
        [(64,),         (16, 4, -4),      True,  (16, 4)],
        [(64,),         (16, -1, -4),     True,  (16, 4)],
        [(1, 2, 3, 64), (-2, -1, 16, -4), True,  (1, 2, 3, 4, 16)]]
    for test_case in test_cases:
        test_reshape_new(*test_case)
    # Test old api
    net = mx.sym.Variable("data")
    net = mx.sym.Reshape(net, target_shape=(2, 0))
    js = net.tojson()
    net = mx.sym.load_json(js)
    _, output_shape, __ = net.infer_shape(data=(2, 3, 5, 5))
    assert(output_shape[0] == (2, 75))
    # Test for Flatten
    data = mx.sym.Variable("data")
    net = mx.sym.Flatten(data)
    exe = net.simple_bind(ctx=default_context(), data=(5, 4, 3, 7))
    data_npy = np.random.normal(size=(5, 4, 3, 7))
    out_grad_npy = np.random.normal(size=(5, 4 * 3 * 7))
    outputs = exe.forward(is_train=True, data=data_npy)[0].asnumpy()
    assert_allclose(outputs, data_npy.reshape((5, 4 * 3 * 7)))
    exe.backward(out_grads=[mx.nd.array(out_grad_npy, ctx=default_context())])
    assert_allclose(exe.grad_arrays[0].asnumpy(), out_grad_npy.reshape((5, 4, 3, 7)))


@with_seed()
def test_reshape_like():
    def test_reshape_like_new(lhs_shape, rhs_shape, lbeg, lend, rbeg, rend, dst_shape):
        lhs = mx.sym.Variable("lhs")
        rhs = mx.sym.Variable("rhs")
        net = mx.sym.reshape_like(lhs, rhs, lhs_begin=lbeg, lhs_end=lend, rhs_begin=rbeg, rhs_end=rend)
        js = net.tojson()
        net = mx.sym.load_json(js)
        _, output_shape, __ = net.infer_shape(lhs=lhs_shape, rhs=rhs_shape)

        assert output_shape[0] == dst_shape, \
            'LHS Shape = %s, RHS Shape = %s, lhs_begin = %s, lhs_end = %s, rhs_begin= %s, rhs_end= %s'\
            %(str(lhs_shape), str(rhs_shape), str(lbeg), str(lend), str(rbeg), str(rend))

        lhs_npy = np.random.rand(*lhs_shape)
        rhs_npy = np.random.rand(*rhs_shape)
        grad_npy = np.random.rand(*dst_shape)

        exe = net.simple_bind(default_context(), lhs=lhs_shape, rhs=rhs_shape)
        exe.arg_dict['lhs'][:] = lhs_npy
        exe.arg_dict['rhs'][:] = rhs_npy
        exe.forward(is_train=True)
        assert np.square(exe.outputs[0].asnumpy() - lhs_npy.reshape(dst_shape)).mean() < 1E-7, \
            'LHS Shape = %s, RHS Shape = %s, lhs_begin = %s, lhs_end = %s, rhs_begin= %s, rhs_end= %s'\
            %(str(lhs_shape), str(rhs_shape), str(lbeg), str(lend), str(rbeg), str(rend))
        exe.backward(out_grads=mx.nd.array(grad_npy))
        assert np.square(exe.grad_dict['lhs'].asnumpy() - grad_npy.reshape(lhs_shape)).mean() < 1E-7, \
            'LHS Shape = %s, RHS Shape = %s, lhs_begin = %s, lhs_end = %s, rhs_begin= %s, rhs_end= %s'\
            %(str(lhs_shape), str(rhs_shape), str(lbeg), str(lend), str(rbeg), str(rend))
    # Test new api (Using shape)
    test_cases = [
        [(30,), (15,2,4), 0, None, 0, 2, (15,2)],
        [(30,), (15,2,4), None, 1, None, 2, (15,2)],
        [(30,7), (15,2,4), 0, 1, 0, 2, (15,2,7)],
        [(3,5), (1,15,4), 0, 2, 1, 2, (15,)],
        [(3,5), (1,15,4), 0, None, 1, -1, (15,)],
        [(30,12), (4,2,2,3), -1, None, 1, None, (30,2,2,3)],
        [(1,1,7,3,1,1), (81,1,1,21), 1, -1, 1, None, (1,1,1,21,1)]
    ]
    # for test_case in test_cases:
    for test_case in test_cases:
        test_reshape_like_new(*test_case)

    # Test old api
    lhs = mx.sym.Variable("lhs")
    rhs = mx.sym.Variable("rhs")
    net = mx.sym.reshape_like(lhs, rhs)
    js = net.tojson()
    net = mx.sym.load_json(js)
    _, output_shape, __ = net.infer_shape(lhs=(40, 30), rhs=(30,20,2))
    assert(output_shape[0] == (30,20,2))


@with_seed()
def test_reduce():
    sample_num = 500
    def test_reduce_inner(numpy_reduce_func, numpy_reduce_grad_func, mx_reduce_sym, nan_prob=0,
                          test_exclude=True, test_none_axis=False):
        for i in range(sample_num):
            # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
            # Insert a NaN with probability equal to nan_prob
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 6, size=(ndim,))
            axis_num = np.random.randint(0, ndim, size=1)
            axis_flags = np.random.randint(0, 2, size=ndim)
            if test_exclude:
                exclude = np.random.randint(0, 2)
            else:
                exclude = False
            axes = []
            for (axis, flag) in enumerate(axis_flags):
                if flag:
                    axes.append(axis)
            if 0 == len(axes):
                axes = None
            elif 1 == len(axes):
                axes = axes[0]
            else:
                axes = tuple(axes)
            keepdims = np.random.randint(0, 2)
            a = mx.symbol.Variable('a')
            if axes is None:
                if test_none_axis:
                    b = mx_reduce_sym(a, keepdims=keepdims, axis=axes)
                else:
                    b = mx_reduce_sym(a, keepdims=keepdims)
            elif exclude and isinstance(axes, tuple) and len(axes) < ndim:
                naxes = [i for i in range(ndim) if i not in axes]
                b = mx_reduce_sym(a, axis=naxes, keepdims=keepdims, exclude=True)
            else:
                b = mx_reduce_sym(a, axis=axes, keepdims=keepdims)
            dat_npy = np.random.rand(*shape)
            # Test with both negative and positive values (randomly).  Avoid having both in the same
            # test, which can be problematic for error checking due to near-zero values.
            if np.random.rand() > 0.5:
                dat_npy = -dat_npy
            if nan_prob > 0:
                dat_npy[np.random.rand(*shape) < nan_prob] = np.nan
            sum_groundtruth = np.array(numpy_reduce_func(dat_npy, axis=axes, keepdims=keepdims))
            if sum_groundtruth.shape == ():
                sum_groundtruth = np.array([sum_groundtruth])
            grad_nd = mx.nd.empty(shape)
            outgrad_npy = np.array(np.random.rand(*sum_groundtruth.shape))

            keepdim_shape = np_reduce(dat_npy, axes, 1, np.sum).shape
            grad_groundtruth = numpy_reduce_grad_func(outgrad=outgrad_npy, data=dat_npy,
                                                      outdata=sum_groundtruth,
                                                      axis=axes, keepdims=keepdims,
                                                      keepdim_shape=keepdim_shape)
            net = b.bind(default_context(), args={'a': mx.nd.array(dat_npy)},
                         args_grad={'a': grad_nd})
            net.forward(is_train=True)

            # check forward
            assert_almost_equal_ignore_nan(net.outputs[0].asnumpy(), sum_groundtruth, rtol=1e-4, atol=1e-4)

            net.backward(out_grads=mx.nd.array(outgrad_npy))
            bc_grad_groundtruth = np.broadcast_to(grad_groundtruth, grad_nd.shape)
            # check backward
            assert_almost_equal_ignore_nan(grad_nd.asnumpy(), bc_grad_groundtruth, rtol=1e-4, atol=1e-4)

    test_none_axis = [True, False]
    for test_none in test_none_axis:
        test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.sum),
                          lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                            outgrad.reshape(keepdim_shape),
                          mx.symbol.sum, test_none_axis=test_none)
        test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.mean),
                          lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                            outgrad.reshape(keepdim_shape)/(data.size/outdata.size),
                          mx.symbol.mean, test_none_axis=test_none)
        test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.prod),
                          lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                            outgrad.reshape(keepdim_shape) * (outdata.reshape(keepdim_shape) / data),
                          mx.symbol.prod, test_none_axis=test_none)
        test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.nansum),
                          lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                            np.where(np.isnan(data), 0, outgrad.reshape(keepdim_shape)),
                          mx.symbol.nansum, 0.3, test_none_axis=test_none)
        test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.nanprod),
                          lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                            np.where(np.isnan(data), 0, outgrad.reshape(keepdim_shape) *
                                   (outdata.reshape(keepdim_shape) / data)),
                          mx.symbol.nanprod, 0.3, test_none_axis=test_none)
        # grad of max and min are sensitive to the precision of the calculation.
        # Force numpy to match mxnet's float32.
        test_reduce_inner(lambda data, axis, keepdims:np_reduce(np.float32(data), axis, keepdims, np.max),
                          lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                            outgrad.reshape(keepdim_shape) *
                            (np.equal(np.float32(data), outdata.reshape(keepdim_shape))),
                          mx.symbol.max)
        test_reduce_inner(lambda data, axis, keepdims:np_reduce(np.float32(data), axis, keepdims, np.min),
                          lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                            outgrad.reshape(keepdim_shape) *
                            (np.equal(np.float32(data), outdata.reshape(keepdim_shape))),
                          mx.symbol.min)
        test_reduce_inner(lambda data, axis, keepdims:np_reduce(data, axis, keepdims, np.linalg.norm),
                          lambda outgrad, data, outdata, axis, keepdims, keepdim_shape:
                            outgrad.reshape(keepdim_shape) * (data / outdata.reshape(keepdim_shape)),
                          mx.symbol.norm, test_exclude=False, test_none_axis=test_none)


@with_seed()
def test_broadcast():
    sample_num = 200
    for i in range(sample_num):
        # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
        ndim = np.random.randint(1, 6)
        target_shape = np.random.randint(1, 6, size=(ndim,))
        axis = tuple(set(np.random.randint(0, ndim, np.random.randint(1, ndim + 1))))
        shape = target_shape.copy()
        size = tuple([shape[ele] for ele in axis])
        for ele in axis:
            shape[ele] = 1
        target_shape_with_zero = list(target_shape)
        for idx in range(len(target_shape_with_zero)):
            if idx not in axis:
                target_shape_with_zero[idx] = 0
                break

        a = mx.symbol.Variable('a')
        sym_bcast_axis = mx.symbol.broadcast_axis(a, axis=axis, size=size)
        sym_bcast_to = mx.symbol.broadcast_to(a, shape=tuple(target_shape))
        sym_bcast_to_with_zero = mx.symbol.broadcast_to(a, shape=tuple(target_shape_with_zero))
        sym_bcast_like = mx.symbol.broadcast_like(a, sym_bcast_to)

        def test_broadcasting_ele(sym_bcast):
            dat_npy = np.random.rand(*shape)
            groundtruth = dat_npy
            grad_nd = mx.nd.empty(shape)
            outgrad_npy = np.random.rand(*target_shape)
            grad_groundtruth = np_reduce(outgrad_npy, axis=axis, keepdims=True,
                                         numpy_reduce_func=np.sum)
            net = sym_bcast.bind(default_context(), args={'a': mx.nd.array(dat_npy)},
                                 args_grad={'a': grad_nd})
            net.forward(is_train=True)
            assert (net.outputs[0].shape == target_shape).all()
            assert_almost_equal(net.outputs[0], groundtruth, rtol=1e-4)
            net.backward(out_grads=mx.nd.array(outgrad_npy))
            assert_almost_equal(grad_nd, grad_groundtruth, rtol=1e-4)
        test_broadcasting_ele(sym_bcast_axis)
        test_broadcasting_ele(sym_bcast_to)
        test_broadcasting_ele(sym_bcast_to_with_zero)
        test_broadcasting_ele(sym_bcast_like)


@with_seed()
def test_transpose():
    for ndim in range(1, 7):
        for t in range(5):
            dims = list(np.random.randint(1, 10, size=ndim))
            axes = list(range(ndim))
            random.shuffle(axes)
            axes = tuple(axes)
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.transpose(x, axes=axes)
            assert_allclose(np.transpose(x.asnumpy(), axes=axes), y.asnumpy())

            y = mx.nd.transpose(x)
            assert_allclose(np.transpose(x.asnumpy()), y.asnumpy())


@with_seed()
def test_pseudo2dtranspose():
    def getTwoInts(mn, mx):
        n1 = np.random.randint(mn, mx)
        n2 = np.random.randint(mn, mx-1)
        n2 = n2 if n2 < n1 else n2+1
        return tuple(np.sort([n1, n2]))

    def getTranspAxes(ndim):
        axes = list(range(ndim))
        n1, n2 = getTwoInts(0,ndim)
        return tuple(axes[:n1]+axes[n2:]+axes[n1:n2])

    for ndim in range(2, 7):
        for dt in ['int8', 'half', 'int32', 'int64']:
            for _ in range(5):
                dims = list(np.random.randint(5, 20, size=ndim))
                axes = getTranspAxes(ndim)
                x = mx.nd.array(np.random.normal(size=dims), dtype=dt)
                y = mx.nd.transpose(x, axes=axes)
                assert_allclose(np.transpose(x.asnumpy(), axes=axes), y.asnumpy())


@with_seed()
def test_big_transpose():
    n = [1]
    d = list(np.random.randint(132, 160, size=1))
    hw = list(np.random.randint(256, 320, size=2))
    c = [10]
    dims = n + d + hw + c
    axes = (0,4,1,2,3)
    x_np = np.random.normal(size=dims).astype('uint8')
    x = mx.nd.array(x_np, dtype='uint8')
    y = mx.nd.transpose(x, axes=axes)
    assert_allclose(np.transpose(x_np, axes=axes), y.asnumpy().astype('uint8'))
    axes = (0,2,3,4,1)
    z = mx.nd.transpose(y, axes=axes)
    assert_allclose(x_np, z.asnumpy().astype('uint8'))


@with_seed()
def test_larger_transpose():
    x = mx.nd.random.normal(shape=(50,51))
    y = mx.nd.transpose(x)
    assert_allclose(np.transpose(x.asnumpy()), y.asnumpy())


@with_seed()
def test_expand_dims():
    for ndim in range(1, 6):
        for axis in range(-ndim + 1, ndim):
            x = np.random.normal(size=list(np.random.randint(1, 10, size=ndim)))
            y = mx.nd.array(x)
            x1 = np.expand_dims(x, axis=axis)
            y1 = mx.nd.expand_dims(y, axis=axis)
            assert_allclose(x1, y1.asnumpy())
            assert_allclose(x1.shape, y1.shape)


@with_seed()
def test_crop():
    for ndim in range(1, 6):
        for t in range(5):
            dims = []
            begin = []
            end = []
            idx = []
            for i in range(ndim):
                d = random.randint(1, 5)
                b = random.randint(0, d-1)
                e = random.randint(b+1, d)
                if b == 0 and random.randint(0, 1):
                    b = None
                elif b != 0 and random.randint(0, 1):
                    b -= d
                if e == d and random.randint(0, 1):
                    e = None
                elif e != d and random.randint(0, 1):
                    e -= d
                dims.append(d)
                begin.append(b)
                end.append(e)
                idx.append(slice(b, e))
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.crop(x, begin=tuple(begin), end=tuple(end))
            assert_allclose(x.asnumpy()[idx], y.asnumpy())

            vx = mx.sym.Variable('x')
            vy = mx.sym.crop(vx, begin=tuple(begin), end=tuple(end))
            check_numeric_gradient(vy, [x.asnumpy()])


@with_seed()
def test_slice_axis():
    for ndim in range(1, 6):
        shape = np.random.randint(1, 11, size=(ndim,))
        for t in range(ndim):
            d = shape[t]
            b = random.randint(0, d-1)
            e = random.randint(b+1, d)
            if np.random.rand() > 0.6:
                e = None
            else:
                if e < d and np.random.rand() > 0.5:
                    e = e - d
            if np.random.rand() > 0.5:
                b = b - d
            idx = []
            for i in range(ndim):
                idx.append(slice(0, shape[i]))
            idx[t] = slice(b, e)

            X = mx.symbol.Variable('X')
            x = mx.nd.array(np.random.normal(size=shape))
            Y = mx.symbol.slice_axis(data=X, axis=t, begin=b, end=e)

            xgrad = mx.nd.empty(x.shape)
            exec1 = Y.bind(default_context(), args = [x], args_grad = {'X': xgrad})
            exec1.forward(is_train=True)
            y = exec1.outputs[0]
            assert_allclose(x.asnumpy()[idx], y.asnumpy())
            exec1.backward([y])
            xx = x.asnumpy()
            xx[:] = 0.0
            xx[idx] = x.asnumpy()[idx]
            assert_allclose(xx, xgrad.asnumpy())
            x_grad_npy = np.random.normal(size=x.shape)
            xgrad = mx.nd.array(x_grad_npy)
            exec2 = Y.bind(default_context(), args=[x], args_grad={'X': xgrad}, grad_req="add")
            exec2.forward(is_train=True)
            exec2.backward([exec2.outputs[0]])
            xx = np.zeros(shape=x.shape, dtype=np.float32)
            xx[idx] = x.asnumpy()[idx]
            assert_allclose(xx + x_grad_npy, xgrad.asnumpy(), atol=1E-5)

@with_seed()
def test_slice_like():
    for ndim in range(1, 6):
        from_shape = np.random.randint(1, 11, size=(ndim,))
        shape = [s + np.random.randint(0, 3) for s in from_shape]
        for t in range(ndim):
            if t > 0:
                axes = np.random.randint(0, ndim, size=t).tolist()
            else:
                axes = []
            idx = []
            for i in range(ndim):
                idx.append(slice(0, shape[i]))
                if i in axes or not axes:
                    idx[i] = slice(0, from_shape[i])

            if axes:
                pos = np.random.randint(0, t)
                if axes[pos] > 0:
                    axes[pos] -= ndim  # negative index

            X = mx.symbol.Variable('X')
            X_1 = mx.symbol.Variable('X1')
            x = mx.nd.array(np.random.normal(size=shape))
            x1 = mx.nd.array(np.random.normal(size=from_shape))
            Y = mx.symbol.slice_like(data=X, shape_like=X_1, axes=axes)

            xgrad = mx.nd.empty(x.shape)
            xgrad1 = mx.nd.empty(x1.shape)
            exec1 = Y.bind(default_context(), args = [x, x1],
                           args_grad = {'X': xgrad, 'X1': xgrad1})
            exec1.forward(is_train=True)
            y = exec1.outputs[0]
            assert_allclose(x.asnumpy()[idx], y.asnumpy())
            exec1.backward([y])
            xx = x.asnumpy()
            xx[:] = 0.0
            xx[idx] = x.asnumpy()[idx]
            assert_allclose(xx, xgrad.asnumpy())
            assert_allclose(xgrad1.asnumpy(), mx.nd.zeros_like(xgrad1).asnumpy())

@with_seed()
def test_slice_like_different_types():
    x = [[  1.,   2.,   3.,   4.],
         [  5.,   6.,   7.,   8.],
         [  9.,  10.,  11.,  12.]]

    y = [[  0.,   0.,   0.],
         [  0.,   0.,   0.]]

    x = mx.nd.array(x)
    y = mx.nd.array(y).astype('int32')
    z = mx.nd.slice_like(x, y)
    assert_allclose(z.asnumpy(), [[1,2,3],[5,6,7]])

@with_seed()
def test_reshape_like_different_types():
    x = mx.nd.zeros((2, 3))

    y = mx.nd.array([[1, 2], [3, 4], [5, 6]])

    y = mx.nd.array(y).astype('int32')
    z = mx.nd.reshape_like(x, y)
    assert_allclose(z.asnumpy(), [[0,0],[0,0],[0,0]])

@with_seed()
def test_broadcast_like_different_types():
    x = mx.nd.zeros((2, 1))
    y = mx.nd.ones((2, 2))

    y = mx.nd.array(y).astype('int32')
    z = mx.nd.broadcast_like(x, y)
    assert_allclose(z.asnumpy(), [[0,0],[0,0]])
    assert x.dtype == z.dtype

@with_seed()
def test_flip():
    for ndim in range(1, 6):
        for t in range(5):
            dims = [random.randint(1,10) for i in range(ndim)]
            axis = random.randint(0, ndim-1)
            idx = [slice(None, None, -1) if i == axis else slice(None, None) for i in range(ndim)]
            x = mx.nd.array(np.random.normal(size=dims))
            y = mx.nd.flip(x, axis=axis)
            assert_allclose(x.asnumpy()[idx], y.asnumpy())


@with_seed()
def test_stn():
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    num_filter = 2  # conv of loc net
    kernel = (3, 3)  # conv of loc net
    num_hidden = 6  # fc of loc net
    for n in [1, 2, 3, 4]:
        for c in [1, 2, 3, 4]:
            for h in [5, 9, 13, 17]:  # for convenience test, this third and forth input dim should be 4x + 1
                for w in [5, 9, 13, 17]:
                    data_shape = (n, c, h, w)
                    target_shape = (int((data_shape[2]+1)/2), int((data_shape[3]+1)/2))
                    data = mx.sym.Variable(name="data")
                    loc = mx.sym.Convolution(data=data, kernel=kernel, pad=(1, 1), num_filter=num_filter, name="loc_conv")
                    loc = mx.sym.Flatten(data=loc)
                    loc = mx.sym.FullyConnected(data=loc, num_hidden=num_hidden, name="loc_fc")
                    stn = mx.sym.SpatialTransformer(data=data, loc=loc, target_shape=target_shape,
                                                    transform_type="affine", sampler_type="bilinear")
                    arg_names = stn.list_arguments()
                    arg_shapes, out_shapes, _ = stn.infer_shape(data=data_shape)
                    # check shape
                    assert out_shapes[0] == (data_shape[0], data_shape[1], target_shape[0], target_shape[1])
                    dev = default_context()
                    #dev = mx.gpu(0)
                    args = {}
                    args['data'] = mx.random.normal(0, 1, data_shape, ctx=mx.cpu()).copyto(dev)
                    args['loc_conv_weight'] = mx.nd.zeros((num_filter, data_shape[1], kernel[0], kernel[1]), ctx=dev)
                    args['loc_conv_bias'] = mx.nd.zeros((num_filter,), ctx=dev)
                    args['loc_fc_weight'] = mx.nd.zeros((6, num_filter*data_shape[2]*data_shape[3]), ctx=dev)
                    args['loc_fc_bias'] = mx.nd.array([0.5, 0, 0, 0, 0.5, 0], ctx=dev)
                    grad_grad = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
                    exe = stn.bind(dev, args=args, args_grad=grad_grad)
                    exe.forward(is_train=True)
                    out = exe.outputs[0]
                    # check forward
                    assert_almost_equal(out, args['data'].asnumpy()[:, :, h//4:h-h//4, w//4:w-w//4], rtol=1e-2, atol=1e-4)
                    out_grad = mx.nd.ones(out.shape, ctx=dev)
                    exe.backward([out_grad])
                    # check backward
                    assert_almost_equal(out_grad, grad_grad[0].asnumpy()[:, :, h//4:h-h//4, w//4:w-w//4], rtol=1e-2, atol=1e-4)


def test_stn_valid_sampling():
    target_shape = (
        28,
        28,
    )
    src_shape = (
        42,
        42,
    )

    data = mx.sym.Variable(name="data")
    loc = mx.sym.Variable(name="loc")

    data_array = np.zeros((
        1,
        1,
    ) + src_shape)
    # Have an ever so slight rotation.
    loc_array = np.array(
        [[9.03887e-05, 1.00015, 0.00174931, 1.0003, 0.000311901,
          -0.000919065]])

    stn = mx.sym.SpatialTransformer(
        data=data,
        loc=loc,
        target_shape=target_shape,
        transform_type="affine",
        sampler_type="bilinear")

    grad_req = {k: 'write' for k in stn.list_arguments()}
    grads = {
        'data': mx.nd.array(np.zeros_like(data_array)),
        'loc': mx.nd.array(np.zeros_like(loc_array))
    }
    executor = stn.bind(
        ctx=default_context(),
        args={'data': mx.nd.array(data_array),
              'loc': mx.nd.array(loc_array)},
        grad_req=grad_req,
        args_grad=grads)
    executor.forward(is_train=True)
    executor.backward(mx.nd.ones((
        1,
        1,
    ) + target_shape))


@with_seed()
def test_dot():
    ctx = default_context()
    dtypes = ['float32', 'float64']
    ndims = [2]
    if ctx.device_type == 'gpu':
        dtypes += ['float16']
        ndims += [1]

    # Test normal dot.
    for ndim in ndims:
        for data_type in dtypes:
            tol = 1e-2 if data_type == 'float16' else 1e-3
            for m in range(1, 5):
                for k in range(1, 5):
                    if ndim == 1 and k != 1:
                        pass
                    for n in range(1, 5):
                        a_shape = (m, k) if ndim == 2 else (m,)
                        b_shape = (k, n) if ndim == 2 else (n,)
                        a_npy = np.random.normal(0, 1, (m, k))
                        a_npy = a_npy.astype(data_type)
                        b_npy = np.random.normal(0, 1, (k, n))
                        b_npy = b_npy.astype(data_type)
                        c_npy = np.empty((m, n), dtype=data_type)
                        ograd_npy = np.random.normal(0, 1, (m, n))
                        ograd_npy = ograd_npy.astype(data_type)
                        agrad_npy = np.empty((m, k), dtype=data_type)
                        bgrad_npy = np.empty((k, n), dtype=data_type)
                        c_npy[:, :] = np.dot(a_npy[:, :], b_npy[:, :])
                        bgrad_npy[:, :] = np.dot(a_npy[:, :].T, ograd_npy[:, :])
                        agrad_npy[:, :] = np.dot(ograd_npy[:, :], b_npy[:, :].T)
                        a = mx.sym.Variable('a', dtype=data_type)
                        b = mx.sym.Variable('b', dtype=data_type)
                        c = mx.sym.dot(a, b)
                        exe = c.simple_bind(ctx=ctx, a=a_npy.shape, b=b_npy.shape)
                        outputs = exe.forward(is_train=True, a=a_npy, b=b_npy)
                        assert_almost_equal(outputs[0], c_npy, rtol=tol, atol=tol)
                        exe.backward(out_grads=[mx.nd.array(ograd_npy, mx.cpu()).astype(data_type)])
                        assert_almost_equal(exe.grad_dict['a'], agrad_npy, rtol=tol, atol=tol)
                        assert_almost_equal(exe.grad_dict['b'], bgrad_npy, rtol=tol, atol=tol)

    # Test dot with transpose flag using gradient checker.
    def dot_sym(data_type):
        x = mx.sym.Variable('x', dtype=data_type)
        y = mx.sym.Variable('y', dtype=data_type)
        return mx.sym.dot(x, y)

    def dot_sym_xT(data_type):
        x = mx.sym.Variable('x', dtype=data_type)
        y = mx.sym.Variable('y', dtype=data_type)
        return mx.sym.dot(x, y, transpose_a=True)

    def dot_sym_yT(data_type):
        x = mx.sym.Variable('x', dtype=data_type)
        y = mx.sym.Variable('y', dtype=data_type)
        return mx.sym.dot(x, y, transpose_b=True)

    def dot_sym_xT_yT(data_type):
        x = mx.sym.Variable('x', dtype=data_type)
        y = mx.sym.Variable('y', dtype=data_type)
        return mx.sym.dot(x, y, transpose_a=True, transpose_b=True)

    for data_type in dtypes:
        for ashape, bshape in [((3, 4), (4, 5)), ((2, 3, 4), (4, 5, 6))]:
            m1_npy = np.random.uniform(-1, 1, ashape)
            m1_npy = m1_npy.astype(data_type)
            m2_npy = np.random.uniform(-1, 1, bshape)
            m2_npy = m2_npy.astype(data_type)
            check_numeric_gradient(dot_sym(data_type), [m1_npy, m2_npy], numeric_eps=1e-1, rtol=2e-2, atol=1e-3)
            check_numeric_gradient(dot_sym_xT(data_type), [m1_npy.T, m2_npy], numeric_eps=1e-1, rtol=2e-2, atol=1e-3)
            check_numeric_gradient(dot_sym_yT(data_type), [m1_npy, m2_npy.T], numeric_eps=1e-1, rtol=2e-2, atol=1e-3)
            check_numeric_gradient(dot_sym_xT_yT(data_type), [m1_npy.T, m2_npy.T], numeric_eps=1e-1, rtol=2e-2, atol=1e-3)


@with_seed()
def test_batch_dot():
    ctx = default_context()
    dtypes = ['float32', 'float64']
    if ctx.device_type == 'gpu':
        dtypes += ['float16']

    for data_type in dtypes:
        for batch_size in range(1, 5):
            for m in range(1, 5):
                for k in range(1, 5):
                    for n in range(1, 5):
                        transpose_a = (np.random.rand() > 0.5)
                        transpose_b = (np.random.rand() > 0.5)
                        a_npy = np.random.normal(0, 1, (batch_size, m, k))
                        a_npy = a_npy.astype(data_type)
                        b_npy = np.random.normal(0, 1, (batch_size, k, n))
                        b_npy = b_npy.astype(data_type)
                        c_npy = np.empty((batch_size, m, n), dtype=data_type)
                        ograd_npy = np.random.normal(0, 1, (batch_size, m, n))
                        ograd_npy = ograd_npy.astype(data_type)
                        agrad_npy = np.empty((batch_size, m, k), dtype=data_type)
                        bgrad_npy = np.empty((batch_size, k, n), dtype=data_type)
                        a_init_grad_npy = np.random.normal(size=(batch_size, m, k))
                        a_init_grad_npy = a_init_grad_npy.astype(data_type)
                        b_init_grad_npy = np.random.normal(size=(batch_size, k, n))
                        b_init_grad_npy = b_init_grad_npy.astype(data_type)
                        for i in range(batch_size):
                            c_npy[i, :, :] = np.dot(a_npy[i, :, :], b_npy[i, :, :])
                            bgrad_npy[i, :, :] = np.dot(a_npy[i, :, :].T, ograd_npy[i, :, :])
                            agrad_npy[i, :, :] = np.dot(ograd_npy[i, :, :], b_npy[i, :, :].T)
                        a = mx.sym.Variable('a', dtype=data_type)
                        b = mx.sym.Variable('b', dtype=data_type)
                        c = mx.sym.batch_dot(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
                        if transpose_a:
                            a_npy = np.transpose(a_npy, axes=(0, 2, 1))
                            agrad_npy = np.transpose(agrad_npy, axes=(0, 2, 1))
                            a_init_grad_npy = np.transpose(a_init_grad_npy, axes=(0, 2, 1))
                        if transpose_b:
                            b_npy = np.transpose(b_npy, axes=(0, 2, 1))
                            bgrad_npy = np.transpose(bgrad_npy, axes=(0, 2, 1))
                            b_init_grad_npy = np.transpose(b_init_grad_npy, axes=(0, 2, 1))
                        exe = c.simple_bind(ctx=ctx,
                            a=a_npy.shape, b=b_npy.shape, grad_req='write')
                        exe_add = c.simple_bind(ctx=ctx,
                            a=a_npy.shape, b=b_npy.shape, grad_req='add')
                        exe_add.grad_dict['a'][:] = a_init_grad_npy
                        exe_add.grad_dict['b'][:] = b_init_grad_npy
                        outputs = exe.forward(is_train=True, a=a_npy, b=b_npy)
                        assert_almost_equal(outputs[0], c_npy,
                                            rtol=1e-2 if data_type == 'float16' else 1e-3,
                                            atol=1e-2 if data_type == 'float16' else 1e-4)
                        exe.backward(out_grads=[mx.nd.array(ograd_npy, ctx=exe._ctx)])
                        assert_almost_equal(exe.grad_dict['a'], agrad_npy,
                                            rtol=1e-2 if data_type == 'float16' else 1e-3,
                                            atol=1e-2 if data_type == 'float16' else 1e-4)
                        assert_almost_equal(exe.grad_dict['b'], bgrad_npy,
                                            rtol=1e-2 if data_type == 'float16' else 1e-3,
                                            atol=1e-2 if data_type == 'float16' else 1e-4)
                        exe_add.forward(is_train=True, a=a_npy, b=b_npy)
                        exe_add.backward(out_grads=[mx.nd.array(ograd_npy, ctx=exe._ctx)])
                        assert_almost_equal(exe_add.grad_dict['a'],
                                            agrad_npy + a_init_grad_npy,
                                            rtol=1e-2 if data_type == 'float16' else 1e-3,
                                            atol=1e-2 if data_type == 'float16' else 1e-4)
                        assert_almost_equal(exe_add.grad_dict['b'],
                                            bgrad_npy + b_init_grad_npy,
                                            rtol=1e-2 if data_type == 'float16' else 1e-3,
                                            atol=1e-2 if data_type == 'float16' else 1e-4)


def get_correlation(data1,data2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply):

    img1 = mx.sym.Variable('img1')
    img2 = mx.sym.Variable('img2')
    return mx.sym.Correlation(data1=img1,data2=img2,kernel_size =kernel_size,max_displacement = max_displacement,
                              stride1 = stride1,stride2 = stride2,pad_size= pad_size,is_multiply = is_multiply)


def correlation_forward(data1,data2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply):

    # compute output's dimension
    paddedbottomheight = data1.shape[2] + 2 * pad_size
    paddedbottomwidth = data1.shape[3] + 2 * pad_size
    kernel_radius = (kernel_size - 1) // 2
    border_size = max_displacement + kernel_radius
    top_width = (paddedbottomwidth - border_size * 2) // stride1
    top_height = (paddedbottomheight - border_size  * 2) // stride1
    neighborhood_grid_radius = max_displacement // stride2
    neighborhood_grid_width = neighborhood_grid_radius * 2 + 1
    top_channels = neighborhood_grid_width * neighborhood_grid_width

    out = np.zeros((data1.shape[0], top_channels, top_height, top_width))
    tmp1 = np.zeros((data1.shape[0],data1.shape[1],paddedbottomheight, paddedbottomwidth))
    tmp2 = np.zeros((data1.shape[0],data1.shape[1],paddedbottomheight, paddedbottomwidth))

    tmp1[:, :, pad_size:pad_size + data1.shape[2], pad_size:pad_size + data1.shape[3]] = data1[:,:,:,:]
    tmp2[:, :, pad_size:pad_size + data2.shape[2], pad_size:pad_size + data2.shape[3]] = data2[:,:,:,:]

    for i in range(top_height):
        for j in range(top_width):
            for nbatch in range(data1.shape[0]):

                # x1,y1 is the location in data1 , i,j is the location in output
                x1 = j * stride1 + max_displacement
                y1 = i * stride1 + max_displacement

                for top_channel in range(top_channels):

                    s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2
                    s2p = (top_channel // neighborhood_grid_width - neighborhood_grid_radius) * stride2

                    # location in data2
                    x2 = x1 + s2o
                    y2 = y1 + s2p

                    for h in range(kernel_size):
                        for w in range(kernel_size):
                            for channel in range(data1.shape[1]):
                                if is_multiply:
                                    out[nbatch, top_channel, i, j] += tmp1[nbatch, channel,y1 + h, x1 + w] * tmp2[nbatch, channel, y2 + h,x2 + w]
                                else:
                                    out[nbatch, top_channel, i, j] += abs(tmp1[nbatch, channel, y1 + h, x1 + w] - tmp2[nbatch, channel, y2 + h, x2 + w])
    out /= float(kernel_size**2*data1.shape[1])
    return out,tmp1,tmp2


def correlation_backward(out_grad,tmp1,tmp2,data1,data2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply):

    # compute output's dimension
    paddedbottomheight = data1.shape[2] + 2 * pad_size
    paddedbottomwidth = data1.shape[3] + 2 * pad_size
    kernel_radius = (kernel_size - 1) // 2
    border_size = max_displacement + kernel_radius
    top_width = (paddedbottomwidth - border_size * 2) // stride1
    top_height = (paddedbottomheight - border_size  * 2) // stride1
    neighborhood_grid_radius = max_displacement // stride2
    neighborhood_grid_width = neighborhood_grid_radius * 2 + 1
    top_channels = neighborhood_grid_width * neighborhood_grid_width

    out = np.zeros((data1.shape[0], top_channels, top_height, top_width))
    tmp1_grad = np.zeros(tmp1.shape)
    tmp2_grad = np.zeros(tmp2.shape)

    for i in range(top_height):
        for j in range(top_width):
            for nbatch in range(data1.shape[0]):

                # x1,y1 is the location in data1 , i,j is the location in output
                x1 = j * stride1 + max_displacement
                y1 = i * stride1 + max_displacement

                for top_channel in range(top_channels):

                    s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2
                    s2p = (top_channel // neighborhood_grid_width - neighborhood_grid_radius) * stride2

                    # location in data2
                    x2 = x1 + s2o
                    y2 = y1 + s2p

                    for h in range(kernel_size):
                        for w in range(kernel_size):
                            for channel in range(data1.shape[1]):
                                if is_multiply:
                                    tmp1_grad[nbatch,channel,y1+h,x1+w]+= out_grad[nbatch,top_channel,i,j]*tmp2[nbatch, channel, y2 + h,x2 + w]
                                    tmp2_grad[nbatch,channel,y2+h,x2+w]+= out_grad[nbatch,top_channel,i,j]*tmp1[nbatch, channel, y1 + h,x1 + w]
                                else:
                                    sgn = 1 if (tmp1[nbatch, channel, y1 + h,x1 + w]>=tmp2[nbatch, channel, y2 + h,x2 + w]) else -1
                                    tmp1_grad[nbatch,channel,y1+h,x1+w]+= out_grad[nbatch,top_channel,i,j]*sgn
                                    tmp2_grad[nbatch,channel,y2+h,x2+w]+= out_grad[nbatch,top_channel,i,j]*(-sgn)

    tmp1_grad = tmp1_grad / float(kernel_size**2*data1.shape[1])
    tmp2_grad = tmp2_grad / float(kernel_size**2*data1.shape[1])
    return tmp1_grad[:,:,pad_size:pad_size+data1.shape[2],pad_size:pad_size+data1.shape[3]],tmp2_grad[:,:,pad_size:pad_size+data1.shape[2],pad_size:pad_size+data1.shape[3]],


def unittest_correlation(data_shape,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply,dtype):

    img1 = np.random.random(data_shape)
    img1 = img1.astype(dtype)
    img2 = np.random.random(data_shape)
    img2 = img2.astype(dtype)

    net1 = get_correlation(img1,img2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply)
    net2 = get_correlation(img1,img2,kernel_size,max_displacement,stride1,stride2,pad_size,is_multiply )

    exe1 = net1.simple_bind(default_context(),img1=img1.shape,img2=img1.shape)
    exe1.arg_dict['img1'][:] = img1
    exe1.arg_dict['img2'][:] = img2

    #cpu forward
    exe1.forward(is_train=True)
    # python forward
    forward_result,tmp1,tmp2 = correlation_forward(img1,img2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply)

    # forward error
    assert_almost_equal(exe1.outputs[0], forward_result, rtol=1e-4, atol=1e-4)

    # out_grad
    a = np.ones(forward_result.shape)
    out_grad1 = mx.nd.array(a,default_context())
    # cpu backward
    exe1.backward(out_grads=out_grad1)
    # python backward
    grad1,grad2 = correlation_backward(a,tmp1,tmp2,img1,img2,pad_size,kernel_size,stride1,stride2,max_displacement,is_multiply)

    # backward error
    assert_almost_equal(exe1.grad_dict['img1'], grad1, rtol=1e-3, atol=1e-4)
    assert_almost_equal(exe1.grad_dict['img2'], grad2, rtol=1e-3, atol=1e-4)


@with_seed()
def test_correlation():
    def test_infer_type(dtype):
        a = mx.sym.Variable('a')
        b = mx.sym.Variable('b')
        corr = mx.sym.Correlation(data1=a, data2=b)
        arg_type1, out_type1, _ = corr.infer_type(a=dtype)
        if arg_type1[0] != np.dtype(dtype) and arg_type1[1] != np.dtype(dtype) and out_type1[0] != np.dtype(dtype):
            msg = npt.npt.build_err_msg([a, b],
                                        err_msg="Inferred type from a is not as expected, "
                                                "Expected :%s %s %s, Got: %s %s %s"
                                                % (dtype, dtype, dtype, arg_type1[0], arg_type1[1], out_type1[0]),
                                                names=['a', 'b'])
            raise AssertionError(msg)
        arg_type2, out_type2, _ = corr.infer_type(b=dtype)
        if arg_type2[0] != np.dtype(dtype) and arg_type2[1] != np.dtype(dtype) and out_type2[0] != np.dtype(dtype):
            msg = npt.npt.build_err_msg([a, b],
                                        err_msg="Inferred type from b is not as expected, "
                                                "Expected :%s %s %s, Got: %s %s %s"
                                                % (dtype, dtype, dtype, arg_type1[0], arg_type1[1], out_type1[0]),
                                                names=['a', 'b'])
            raise AssertionError(msg)

    for dtype in ['float16', 'float32']:
        test_infer_type(dtype)
        unittest_correlation((1,3,10,10), kernel_size = 1,max_displacement = 4,stride1 = 1,stride2 = 1,pad_size = 4,is_multiply = False, dtype = dtype)
        unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 5,stride1 = 1,stride2 = 1,pad_size = 5,is_multiply = False, dtype = dtype)
        unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 5,stride1 = 1,stride2 = 1,pad_size = 5,is_multiply = True, dtype = dtype)
        unittest_correlation((5,1,15,15), kernel_size = 1,max_displacement = 10,stride1 = 1,stride2 = 2,pad_size = 10,is_multiply = True, dtype = dtype)
        unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 1,stride2 = 1,pad_size = 2,is_multiply = True, dtype = dtype)
        unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = True, dtype = dtype)
        unittest_correlation((5,1,4,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = False, dtype = dtype)
        unittest_correlation((5,1,6,4), kernel_size = 3,max_displacement = 1,stride1 = 2,stride2 = 1,pad_size = 2,is_multiply = False, dtype = dtype)
        unittest_correlation((5,1,11,11), kernel_size = 5,max_displacement = 1,stride1 = 1,stride2 = 1,pad_size = 2,is_multiply = False, dtype = dtype)


@with_seed()
def test_support_vector_machine_l1_svm():
    xpu = default_context()
    shape = (20, 10)

    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SVMOutput(data=X, label=L, use_linear=True)
    x = mx.nd.empty(shape, ctx = xpu)
    l = mx.nd.empty((shape[0],), ctx = xpu)
    x_np = np.random.rand(*shape)
    l_np = np.random.randint(0, shape[1], (shape[0],))
    x[:] = x_np
    l[:] = l_np

    grad = mx.nd.empty(shape, ctx = xpu)
    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward(is_train=True)

    assert_almost_equal(x_np, exec1.outputs[0])

    exec1.backward()

    l_mask = np.equal(l_np.reshape(shape[0],1),range(shape[1]))
    l_mask = np.array(l_mask, dtype=np.float32)*2 -1
    grad_np = (-1) * l_mask * np.greater(1 - l_mask * x_np, 0)

    assert_almost_equal(grad_np, grad)


@with_seed()
def test_support_vector_machine_l2_svm():
    xpu = default_context()
    shape = (20, 10)

    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L')
    Y = mx.symbol.SVMOutput(data=X, label=L)
    x = mx.nd.empty(shape, ctx = xpu)
    l = mx.nd.empty((shape[0],), ctx = xpu)
    x_np = np.random.rand(*shape)
    x_np = x_np.astype(np.float32)
    l_np = np.random.randint(0, shape[1], (shape[0],))
    x[:] = x_np
    l[:] = l_np

    grad = mx.nd.empty(shape, ctx = xpu)
    exec1 = Y.bind(xpu, args = [x, l], args_grad = {'X': grad})
    exec1.forward(is_train=True)

    assert_almost_equal(x_np, exec1.outputs[0])

    exec1.backward()

    l_mask = np.equal(l_np.reshape(shape[0],1),range(shape[1]))
    l_mask = np.array(l_mask, dtype=np.float32)*2 -1
    grad_np = (-2)*l_mask*np.maximum(1-l_mask*x_np,0)
    grad_np = grad_np.astype(np.float32)
    assert_almost_equal(grad_np, grad)


# Seed set because the test is not robust enough to operate on random data
@with_seed(1234)
def test_roipooling():

    data = mx.symbol.Variable(name='data')
    rois = mx.symbol.Variable(name='rois')
    test = mx.symbol.ROIPooling(data=data, rois=rois, pooled_size=(4, 4), spatial_scale=1)

    x1 = np.random.rand(4, 3, 12, 8).astype('float32')
    x2 = np.array([[0, 1.1, 1.1, 6.2, 6.2], [2, 6.1, 2.1, 8.2, 11.2], [1, 3.1, 1.1, 5.2, 10.2], [0, 3, 3, 3, 3]], dtype='float32')

    check_numeric_gradient(sym=test, location=[x1, x2],
                           grad_nodes={'data':'write', 'rois':'null'},
                           numeric_eps=1e-4, rtol=1e-1, atol=1e-4)
    check_numeric_gradient(sym=test, location=[x1, x2],
                           grad_nodes={'data':'add', 'rois':'null'},
                           numeric_eps=1e-4, rtol=1e-1, atol=1E-4)


def check_pad_with_shape(shape, xpu, pad_width, mode, dtype="float64"):
    # bind with label
    X = mx.symbol.Variable('X', dtype=dtype)
    Y = mx.symbol.Pad(data=X, mode=mode, pad_width=pad_width)
    x = mx.random.uniform(-1, 1, shape, ctx=mx.cpu(), dtype=dtype).copyto(xpu)
    # numpy result
    pad_grouped = list(zip(*[iter(list(pad_width))] * 2))
    np_out = np.pad(x.asnumpy(), pad_grouped, mode)
    # mxnet result
    grad = mx.nd.empty(shape, ctx = xpu, dtype=dtype)
    exec1 = Y.bind(xpu, args = [x], args_grad = {'X': grad})
    exec1.forward(is_train=True)
    out = exec1.outputs[0]
    # compare numpy + mxnet
    assert_almost_equal(out, np_out)
    # grad check
    check_numeric_gradient(Y, [x.asnumpy()], numeric_eps=1e-2, rtol=1e-2)


@with_seed()
def test_pad():
    ctx = default_context()
    shape1 = (2, 3, 3, 5)
    pad1 = (0, 0, 0, 0, 1, 2, 3, 4)
    shape2 = (2, 3, 3, 5, 4)
    pad2 = (0, 0, 0, 0, 1, 2, 3, 4, 3, 1)
    # note: this op doesn't support ints yet. Add tests when supported
    dtypes = ["float16", "float32", "float64"]
    for dtype in dtypes:
        check_pad_with_shape(shape1, ctx, pad1, 'constant', dtype)
        check_pad_with_shape(shape1, ctx, pad1, 'edge', dtype)
        check_pad_with_shape(shape2, ctx, pad2, 'constant', dtype)
        check_pad_with_shape(shape2, ctx, pad2, 'edge', dtype)
        check_pad_with_shape(shape1, ctx, pad1, 'reflect', dtype)
        check_pad_with_shape(shape2, ctx, pad2, 'reflect', dtype)


def np_instance_norm(data, weight, bias, eps):
    spatial_dims = data.shape[2::]
    num_spatial_vals = np.prod(np.array(spatial_dims))
    scale = 1/float(num_spatial_vals)
    sum_axis = tuple(range(2, data.ndim))
    mean = scale * np.sum(data, axis = sum_axis)
    mean = np.reshape(np.repeat(mean, num_spatial_vals), data.shape)
    var = scale * np.sum((data - mean)**2, axis = sum_axis)
    var = np.reshape(np.repeat(var, num_spatial_vals), data.shape)

    weightBatch = np.tile(weight, (data.shape[0], 1))
    weightBatch = np.reshape(np.repeat(weightBatch, num_spatial_vals), data.shape)
    biasBatch = np.tile(bias, (data.shape[0], 1))
    biasBatch = np.reshape(np.repeat(biasBatch, num_spatial_vals), data.shape)
    return weightBatch * (data - mean)/np.sqrt(var + eps) + biasBatch


def check_instance_norm_with_shape(shape, xpu):
    # bind with label
    eps = 0.001
    X = mx.symbol.Variable('X')
    G = mx.symbol.Variable('G')
    B = mx.symbol.Variable('B')

    Y = mx.symbol.InstanceNorm(data=X, beta=B, gamma=G, eps=eps)
    x = mx.random.normal(0, 1, shape, ctx=mx.cpu()).copyto(xpu)
    gamma = mx.random.normal(0, 1, shape[1], ctx=mx.cpu()).copyto(xpu)
    beta = mx.random.normal(0, 1, shape[1], ctx=mx.cpu()).copyto(xpu)

    np_out = np_instance_norm(x.asnumpy(), gamma.asnumpy(), beta.asnumpy(), eps)
    exec1 = Y.bind(xpu, args = {'X':x, 'G':gamma, 'B':beta})
    exec1.forward(is_train=False)
    out = exec1.outputs[0]
    assert_almost_equal(out, np_out, rtol=1e-4, atol=1e-4)
    check_numeric_gradient(Y, {'X':x.asnumpy(), 'G':gamma.asnumpy(), 'B':beta.asnumpy()},
                           numeric_eps=1e-2, rtol=1e-2, atol=1e-2)


@with_seed()
def test_instance_normalization():
    check_instance_norm_with_shape((1, 1, 1), default_context())
    check_instance_norm_with_shape((2, 1, 2), default_context())
    check_instance_norm_with_shape((2,4,5,6), default_context())
    check_instance_norm_with_shape((3,3,2,3,2,1,1), default_context())


def check_l2_normalization(in_shape, mode, dtype, norm_eps=1e-10):
    ctx = default_context()
    data = mx.symbol.Variable('data')
    out = mx.symbol.L2Normalization(data=data, mode=mode, eps=norm_eps)
    in_data = np.random.uniform(-1, 1, in_shape).astype(dtype)
    # calculate numpy results
    if mode == 'channel':
        assert in_data.ndim > 2
        np_norm = np.linalg.norm(in_data, axis=1) + norm_eps
        np_norm = np.repeat(1. / np.expand_dims(np_norm, axis=1), in_data.shape[1], axis=1)
        np_out = np.multiply(in_data, np_norm)
    elif mode == 'spatial':
        assert in_data.ndim > 2
        s = in_data.shape
        np_norm = np.linalg.norm(in_data.reshape((s[0], s[1], -1)), axis=2) + norm_eps
        np_norm = np.repeat(1. / np_norm[:, np.newaxis], in_data.size / s[0] / s[1], axis=2)
        np_out = np.multiply(in_data, np_norm.reshape(s))
    elif mode == 'instance':
        assert in_data.ndim > 1
        s = in_data.shape
        np_norm = np.linalg.norm(in_data.reshape((s[0], -1)), axis=1) + norm_eps
        np_norm = np.repeat(1. / np_norm[:, np.newaxis], in_data.size / s[0], axis=1)
        np_out = np.multiply(in_data, np_norm.reshape(s))
    else:
        raise RuntimeError('Unknown l2 normalization mode')
    exe = out.simple_bind(ctx=ctx, data=in_data.shape)
    output = exe.forward(is_train=True, data=in_data)
    # compare numpy + mxnet
    assert_almost_equal(exe.outputs[0], np_out, rtol=1e-2 if dtype is 'float16' else 1e-5, atol=1e-5)
    # check gradient
    check_numeric_gradient(out, [in_data], numeric_eps=1e-3, rtol=1e-2, atol=5e-3)


@with_seed()
def test_l2_normalization():
    for dtype in ['float16', 'float32', 'float64']:
        for mode in ['channel', 'spatial', 'instance']:
            nbatch = random.randint(1, 4)
            nchannel = random.randint(3, 5)
            height = random.randint(4, 6)
            check_l2_normalization((nbatch, nchannel, height), mode, dtype)
            width = random.randint(5, 7)
            check_l2_normalization((nbatch, nchannel, height, width), mode, dtype)


def check_layer_normalization(in_shape, axis, eps, dtype=np.float32,
                              forward_check_eps=1E-3, backward_check_eps=1E-3,
                              npy_grad_check=True, finite_grad_check=True):
    def npy_layer_norm(data, gamma, beta, axis=1, eps=1E-5):
        if axis < 0:
            axis += data.ndim
        broadcast_shape = [1 for _ in range(data.ndim)]
        broadcast_shape[axis] = data.shape[axis]
        mean = data.mean(axis=axis, keepdims=True).astype(dtype)
        var = data.var(axis=axis, keepdims=True).astype(dtype)
        std = np.sqrt(var + dtype(eps)).astype(dtype)
        out = np.reshape(gamma, broadcast_shape) * (data - mean) / std + \
              np.reshape(beta, broadcast_shape)
        return out

    def npy_layer_norm_grad(data, gamma, out_grad, axis, eps):
        if axis < 0:
            axis += data.ndim
        exclude_axis = tuple([ele for ele in range(data.ndim) if ele != axis])
        data_mean = data.mean(axis=axis, keepdims=True)
        data_var = data.var(axis=axis, keepdims=True)
        data_std = np.sqrt(data_var + eps)
        centered_data = (data - data_mean) / data_std
        gamma_grad = (centered_data * out_grad).sum(axis=exclude_axis, keepdims=True)
        beta_grad = out_grad.sum(axis=exclude_axis, keepdims=True)
        w = out_grad * gamma.reshape([1 if i != axis else data.shape[axis] for i in range(data.ndim)])\
            / data_std
        data_grad = w - w.mean(axis=axis, keepdims=True)\
                    - centered_data * (w * centered_data).mean(axis=axis, keepdims=True)
        gamma_grad = gamma_grad.reshape((-1,))
        beta_grad = beta_grad.reshape((-1,))
        return data_grad, gamma_grad, beta_grad

    ctx = default_context()
    data = np.random.normal(0, 1, in_shape).astype(dtype)
    gamma = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
    beta = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
    data_s = mx.symbol.Variable('data')
    gamma_s = mx.symbol.Variable('gamma')
    beta_s = mx.symbol.Variable('beta')
    out_s = mx.symbol.LayerNorm(data=data_s, gamma=gamma_s, beta=beta_s, axis=axis, eps=eps)
    exe = out_s.simple_bind(ctx, data=in_shape)
    exe.arg_dict['data'][:] = data
    exe.arg_dict['gamma'][:] = gamma
    exe.arg_dict['beta'][:] = beta
    out_nd = exe.forward()[0]
    out = npy_layer_norm(data, gamma, beta, axis, eps)
    assert_almost_equal(out, out_nd, forward_check_eps, forward_check_eps)

    if finite_grad_check:
        for req in ['write', 'add']:
            check_numeric_gradient(out_s, {'data': data, 'gamma': gamma, 'beta': beta},
                                   grad_nodes={'data': req, 'gamma': req, 'beta': req},
                                   numeric_eps=1e-2, rtol=1e-2, atol=1e-2)

    if npy_grad_check:
        # Test for grad_req = write
        out_grad = np.random.normal(0, 1, in_shape).astype(dtype)
        exe = out_s.simple_bind(ctx, data=in_shape, grad_req='write')
        exe.arg_dict['data'][:] = data
        exe.arg_dict['gamma'][:] = gamma
        exe.arg_dict['beta'][:] = beta
        exe.forward()
        exe.backward([mx.nd.array(out_grad, ctx=ctx)])
        gt_data_grad, gt_gamma_grad, gt_beta_grad =\
            npy_layer_norm_grad(data, gamma, out_grad, axis, eps)
        assert_almost_equal(exe.grad_dict['data'].asnumpy(), gt_data_grad, backward_check_eps, backward_check_eps)
        assert_almost_equal(exe.grad_dict['gamma'].asnumpy(), gt_gamma_grad, backward_check_eps, backward_check_eps)
        assert_almost_equal(exe.grad_dict['beta'].asnumpy(), gt_beta_grad, backward_check_eps, backward_check_eps)

        # Test for grad_req = add
        out_grad = np.random.normal(0, 1, in_shape).astype(dtype)
        init_data_grad = np.random.normal(0, 1, in_shape).astype(dtype)
        init_gamma_grad = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
        init_beta_grad = np.random.normal(0, 1, (in_shape[axis],)).astype(dtype)
        exe = out_s.simple_bind(ctx, data=in_shape, grad_req='add')
        exe.arg_dict['data'][:] = data
        exe.arg_dict['gamma'][:] = gamma
        exe.arg_dict['beta'][:] = beta
        exe.grad_dict['data'][:] = init_data_grad
        exe.grad_dict['gamma'][:] = init_gamma_grad
        exe.grad_dict['beta'][:] = init_beta_grad
        exe.forward()
        exe.backward([mx.nd.array(out_grad, ctx=ctx)])
        gt_data_grad, gt_gamma_grad, gt_beta_grad = \
            npy_layer_norm_grad(data, gamma, out_grad, axis, eps)
        assert_almost_equal(exe.grad_dict['data'].asnumpy(),
                            gt_data_grad + init_data_grad, backward_check_eps, backward_check_eps)
        assert_almost_equal(exe.grad_dict['gamma'].asnumpy(),
                            gt_gamma_grad + init_gamma_grad, backward_check_eps, backward_check_eps)
        assert_almost_equal(exe.grad_dict['beta'].asnumpy(),
                            gt_beta_grad + init_beta_grad, backward_check_eps, backward_check_eps)


@with_seed()
def test_norm():
    try:
        import scipy
        assert LooseVersion(scipy.__version__) >= LooseVersion('0.1')
        from scipy.linalg import norm as sp_norm
    except (AssertionError, ImportError):
        print("Could not import scipy.linalg.norm or scipy is too old. "
              "Falling back to numpy.linalg.norm which is not numerically stable.")
        from numpy.linalg import norm as sp_norm

    def l1norm(input_data, axis=0, keepdims=True):
        return np.sum(abs(input_data), axis=axis, keepdims=keepdims)

    def l2norm(input_data, axis=0, keepdims=True):
        return sp_norm(input_data, axis=axis, keepdims=keepdims)

    ctx = default_context()
    data = mx.symbol.Variable('data')
    in_data_dim = random_sample([2,3,4], 1)[0]
    in_shape = rand_shape_nd(in_data_dim, dim=5)
    epsilon = 1e-3
    acc_type = {np.float16: np.float32, np.float32: np.float32, np.float64: np.float64,
                np.int32: np.int32, np.int64: np.int64}
    dtype_to_str = {np.float16: 'float16', np.float32: 'float32', np.float64: 'float64',
                    np.int32: 'int32', np.int64: 'int64'}
    for enforce_safe_acc in ['1', '0']:
        with environment('MXNET_SAFE_ACCUMULATION', enforce_safe_acc):
            for order in [1, 2]:
                for dtype in [np.float16, np.float32, np.float64]:
                    for i in range(in_data_dim):
                        for out_dtype in ['float32', 'float64']:
                            backward_dtype = np.float32 if out_dtype == 'float32' else np.float64
                            accumulation_type = acc_type[dtype]
                            if enforce_safe_acc == "0":
                                backward_dtype = dtype
                                out_dtype = dtype_to_str[dtype]
                                accumulation_type = dtype
                            skip_backward = 'int' in out_dtype
                            in_data = np.random.uniform(-1, 1, in_shape).astype(accumulation_type)
                            in_data[abs(in_data) < epsilon] = 2 * epsilon
                            norm_sym = mx.symbol.norm(data=data, ord=order, axis=i, out_dtype=out_dtype, keepdims=True)
                            npy_out = l1norm(in_data, i) if order is 1 else l2norm(in_data, i)
                            npy_out_backward = np.sign(in_data) if order is 1 else in_data/npy_out
                            check_symbolic_forward(norm_sym, [in_data.astype(dtype)], [npy_out.astype(out_dtype)],
                                                   rtol=1e-2 if dtype == np.float16 else 1e-3,
                                                   atol=1e-4 if dtype == np.float16 else 1e-5, ctx=ctx, dtype=dtype)
                            if dtype is not np.float16 and not skip_backward:
                                check_symbolic_backward(norm_sym, [in_data.astype(dtype)],
                                                        [np.ones(npy_out.shape).astype(out_dtype)],
                                                        [npy_out_backward], rtol=1e-3, atol=1e-5, ctx=ctx,
                                                        dtype=backward_dtype)
                                # Disable numeric gradient https://github.com/apache/incubator-mxnet/issues/11509
                                # check gradient
                                if dtype is not np.float16 and not skip_backward:
                                    check_numeric_gradient(norm_sym, [in_data], numeric_eps=epsilon,
                                                   rtol=1e-1, atol=1e-3, dtype=backward_dtype)
                            if i < in_data_dim-1:
                                norm_sym = mx.symbol.norm(data=data, ord=order, axis=(i, i+1), keepdims=True)
                                npy_out = l1norm(in_data, (i, i+1)) if order is 1 else l2norm(in_data, (i, i+1))
                                npy_out_backward = np.sign(in_data) if order is 1 else in_data/npy_out
                                check_symbolic_forward(norm_sym, [in_data], [npy_out.astype(dtype)],
                                                       rtol=1e-2 if dtype is np.float16 else 1e-3,
                                                       atol=1e-4 if dtype is np.float16 else 1e-5, ctx=ctx)
                                if dtype is not np.float16 and not skip_backward:
                                    check_symbolic_backward(norm_sym, [in_data],
                                                            [np.ones(npy_out.shape).astype(out_dtype)],
                                                            [npy_out_backward.astype(out_dtype)],
                                                            rtol=1e-3, atol=1e-5, ctx=ctx, dtype=backward_dtype)
                                # check gradient
                                if dtype is not np.float16 and not skip_backward:
                                    check_numeric_gradient(norm_sym, [in_data], numeric_eps=epsilon,
                                                           rtol=1e-1, atol=1e-3, dtype=backward_dtype)


def test_layer_norm():
    for enforce_safe_acc in ["1", "0"]:
        with environment('MXNET_SAFE_ACCUMULATION', enforce_safe_acc):
            for dtype, forward_check_eps, backward_check_eps in zip([np.float16, np.float32, np.float64],
                                                                    [1E-2, 1E-3, 1E-4],
                                                                    [1E-2, 1E-3, 1E-4]):
                if dtype != np.float16:
                    in_shape_l, finite_grad_check_l = [(10, 6, 5), (10, 10), (128 * 32, 512)], [True, True, False]
                else:
                    in_shape_l, finite_grad_check_l = [(10, 6, 5), (10, 10)], [True, True]  # large input + fp16 does not pass the forward check
                for in_shape, finite_grad_check in zip(in_shape_l, finite_grad_check_l):
                    for axis in range(-len(in_shape), len(in_shape)):
                        for eps in [1E-2, 1E-3]:
                            if dtype == np.float16:
                                npy_grad_check = False
                            else:
                                npy_grad_check = True
                            check_layer_normalization(in_shape, axis, eps, dtype=dtype,
                                                      forward_check_eps=forward_check_eps,
                                                      backward_check_eps=backward_check_eps,
                                                      npy_grad_check=npy_grad_check,
                                                      finite_grad_check=finite_grad_check)


# Numpy Implementation of Sequence Ops
def sequence_last_numpy(array, lengths, axis):
    # create new array of dims [batch, seqlen, ...]
    array2 = np.moveaxis(array, axis, 1)
    dims = array2.shape
    if lengths is None:
        return array2[:, -1]
    lengths = list(lengths)
    return np.array([array2[i, int(lengths[i]) - 1] for i in range(dims[0])])


def sequence_mask_numpy(array, lengths, axis, value):
    if lengths is None:
        return array
    arrayMask = array.copy()
    # conform to [batch, seqlen, ...]
    arrayMask = np.moveaxis(arrayMask, axis, 1)
    shape = arrayMask.shape
    lengths = list(lengths)
    for i in range(shape[0]):
        arrayMask[i, int(lengths[i]):] = value
    return np.moveaxis(arrayMask, 1, axis)


def sequence_reverse_numpy(array, lengths, axis):
    rarray = array.copy()
    # conform to [batch, seqlen, ...]
    rarray = np.moveaxis(rarray, axis, 1)
    shape = rarray.shape
    if lengths is None:
        lengths = [shape[1]] * shape[0]
    lengths = list(lengths)
    for i in range(shape[0]):
        j = int(lengths[i])
        rarray[i,:j] = rarray[i,:j][::-1]
    return np.moveaxis(rarray, 1, axis)


def check_sequence_func(ftype, mask_value=0, axis=0):
    # bind with label
    xpu = default_context()
    X = mx.symbol.Variable('X')
    L = mx.symbol.Variable('L') # lengths
    shapes = [(3, 4), (1, 1), (3, 4, 3, 1, 1)]
    for seqlenQ in [True, False]:
        for ary_dtype in [np.float32]:
            for idx_dtype in [np.int32, np.float32]:
                for s in shapes:
                    x = mx.random.uniform(-1, 1, s, ctx=mx.cpu()).astype(ary_dtype).copyto(xpu)
                    batch = s[1] if (axis == 0) else s[0]
                    seqlen = s[axis]
                    l_np = np.random.randint(1, seqlen + 1, batch)
                    l = mx.nd.array(l_np, ctx=mx.cpu(), dtype=idx_dtype).copyto(xpu)
                    if not seqlenQ:
                        l_np = None
                    args = {'data':X, 'use_sequence_length':seqlenQ, "axis":axis}
                    if seqlenQ:
                        args['sequence_length'] = L
                    if ftype == "last":
                        Y = mx.symbol.SequenceLast(**args)
                        np_out = sequence_last_numpy(x.asnumpy(), l_np, axis)
                    elif ftype == "mask":
                        args['value'] = mask_value
                        Y = mx.symbol.SequenceMask(**args)
                        np_out = sequence_mask_numpy(x.asnumpy(), l_np, axis, mask_value)
                    elif ftype == "reverse":
                        Y = mx.symbol.SequenceReverse(**args)
                        np_out = sequence_reverse_numpy(x.asnumpy(), l_np, axis)
                    fargs = [x, l] if seqlenQ else [x]
                    gargs = [x.asnumpy(), l_np] if seqlenQ else [x.asnumpy()]
                    check_symbolic_forward(Y, fargs, [np_out], dtype="asnumpy")
                    check_numeric_gradient(Y, gargs, grad_nodes={'X':'write'},
                        numeric_eps=1e-2, rtol=1e-2)
                    check_numeric_gradient(Y, gargs, grad_nodes={'X':'add'},
                        numeric_eps=1e-3, rtol=1e-2, atol=1E-4)
                    check_numeric_gradient(Y, gargs, grad_nodes={'X':'null'},
                        numeric_eps=1e-3, rtol=1e-2, atol=1E-4)


@with_seed()
@unittest.skip("Flaky test: https://github.com/apache/incubator-mxnet/issues/11395")
def test_sequence_last():
    check_sequence_func("last", axis=0)
    check_sequence_func("last", axis=1)


@with_seed()
def test_sequence_mask():
    check_sequence_func("mask", axis = 0, mask_value=-2.3)
    check_sequence_func("mask", axis = 1, mask_value=0.3)


def check_sequence_reverse(xpu):
    # sample data
    arr = np.array(
        [[[  1.,   2.,   3.],
          [  4.,   5.,   6.]],
         [[  7.,   8.,   9.],
          [ 10.,  11.,  12.]],
         [[ 13.,  14.,   15.],
          [ 16.,  17.,   18.]]])

    arr1 = np.array(
        [[[  13.,   14.,   15.],
          [  16.,   17.,   18.]],
         [[  7.,   8.,   9.],
          [ 10.,  11.,  12.]],
         [[ 1.,  2.,   3.],
          [ 4.,  5.,   6.]]])

    arr2 = np.array(
        [[[  7.,   8.,   9.],
          [  10.,   11.,   12.]],
         [[  1.,   2.,   3.],
          [ 4.,  5.,   6.]],
         [[ 13.,  14.,   15.],
          [ 16.,  17.,   18.]]])

    arr3 = np.array(
        [[[  7.,   8.,   9.],
          [  16.,   17.,   18.]],
         [[  1.,   2.,   3.],
          [ 10.,  11.,  12.]],
         [[ 13.,  14.,   15.],
          [ 4.,  5.,   6.]]])

    # test for matrix case
    seq_len_1 = [1, 2, 2]
    arr_4 = np.array([[7., 8., 9.], [16., 17., 5.4]], dtype=np.float32)
    arr_5 = np.array([[7., 17., 5.4], [16., 8., 9.]], dtype=np.float32)

    def test_wrapper(arr, xpu, sequence_length=None, use_sequence_length=False):
        # MxNet symbol creation
        seq = mx.sym.Variable('seq')
        if sequence_length and use_sequence_length:
            seq_len = mx.sym.Variable('seq_len')
        else:
           # ensure that both are disabled, not just one
           seq_len=None
           use_sequence_length=False
        rev = mx.sym.SequenceReverse(data=seq, sequence_length=seq_len, use_sequence_length=use_sequence_length)
        # MxNet symbol execution
        if sequence_length:
            bound = rev.bind(xpu, {'seq': mx.nd.array(arr), 'seq_len': mx.nd.array(sequence_length)})
        else:
            bound = rev.bind(xpu, {'seq': mx.nd.array(arr)})
        fwd = bound.forward()
        return fwd[0].asnumpy()

    # test cases
    assert_array_equal(test_wrapper(arr, xpu, use_sequence_length=False), arr1)
    assert_array_equal(test_wrapper(arr, xpu, sequence_length=[3, 3], use_sequence_length=True), arr1)
    assert_array_equal(test_wrapper(arr, xpu, sequence_length=[2, 2], use_sequence_length=True), arr2)
    assert_array_equal(test_wrapper(arr, xpu, sequence_length=[2, 3], use_sequence_length=True), arr3)
    assert_array_equal(test_wrapper(arr_4, xpu, sequence_length=seq_len_1, use_sequence_length=True), arr_5)


@with_seed()
def test_sequence_reverse():
    check_sequence_func("reverse", axis=0)
    check_sequence_reverse(mx.cpu())


def mathematical_core_binary(name,
                             forward_mxnet_call,
                             forward_numpy_call,
                             backward_numpy_call1,
                             backward_numpy_call2,
                             data1_init=2.,
                             data2_init=3.,
                             grad_init=2.):
    data1 = mx.symbol.Variable('data1')
    data2 = mx.symbol.Variable('data2')
    shape = (3, 4)
    data_tmp1 = np.random.rand(3, 4)
    data_tmp2 = np.random.rand(3, 4)
    data_tmp1[:] = data1_init
    data_tmp2[:] = data2_init

    arr_data1 = mx.nd.array(data_tmp1)
    arr_data2 = mx.nd.array(data_tmp2)

    arr_grad1 = mx.nd.empty(shape)
    arr_grad2 = mx.nd.empty(shape)

    test = forward_mxnet_call(data1, data2)
    exe_test = test.bind(default_context(), args=[arr_data1, arr_data2], args_grad=[arr_grad1, arr_grad2])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout = forward_numpy_call(data_tmp1, data_tmp2)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = grad_init
    exe_test.backward(out_grad)

    npout_grad = np.ones(shape)
    npout_grad[:] = grad_init

    npout_grad1 = npout_grad * backward_numpy_call1(data_tmp1, data_tmp2)
    npout_grad2 = npout_grad * backward_numpy_call2(data_tmp1, data_tmp2)

    assert_almost_equal(arr_grad1, npout_grad1)
    assert_almost_equal(arr_grad2, npout_grad2)


def mathematical_core(name, forward_mxnet_call, forward_numpy_call, backward_numpy_call, data_init=5., grad_init=2.):
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:] = data_init
    arr_data = mx.nd.array(data_tmp)
    arr_grad = mx.nd.empty(shape)
    arr_grad[:] = 3

    test = forward_mxnet_call(data)
    exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout = forward_numpy_call(data_tmp)
    assert_almost_equal(out, npout)

    out_grad = mx.nd.empty(shape)
    out_grad[:] = grad_init
    npout_grad = out_grad.asnumpy()
    temp = backward_numpy_call(data_tmp)
    npout_grad = npout_grad * temp
    exe_test.backward(out_grad)
    assert_almost_equal(arr_grad, npout_grad)


@with_seed()
def test_special_functions_using_scipy():
    try:
        from scipy import special as scipy_special
    except:
        print("Could not import scipy. Skipping unit tests for special functions")
        return

    # gamma
    mathematical_core("gamma", lambda x: mx.sym.gamma(x), lambda x: scipy_special.gamma(x),
                     lambda x: scipy_special.gamma(x) * scipy_special.psi(x), 0.5, 0.5)

    # gammaln
    mathematical_core("gammaln", lambda x: mx.sym.gammaln(x), lambda x: scipy_special.gammaln(x),
                     lambda x: scipy_special.psi(x), 0.5, 0.5)

    # erf
    mathematical_core("erf", lambda x: mx.sym.erf(x), lambda x: scipy_special.erf(x),
                     lambda x: 2.0 / math.sqrt(math.pi) * np.exp(-(x ** 2)), 0.5, 0.5)

    # erfinv
    mathematical_core("erfinv", lambda x: mx.sym.erfinv(x), lambda x: scipy_special.erfinv(x),
                     lambda x: 0.5 * math.sqrt(math.pi) * np.exp(scipy_special.erfinv(x) ** 2), 0.5, 0.5)


def rounding(name, forward_mxnet_call, forward_numpy_call, data_init=5., grad_init=2.):
    data = mx.symbol.Variable('data')
    shape = (3, 4)
    data_tmp = np.ones(shape)
    data_tmp[:] = data_init
    arr_data = mx.nd.array(data_tmp)

    test = forward_mxnet_call(data)
    exe_test = test.bind(default_context(), args=[arr_data])
    exe_test.forward(is_train=True)
    out = exe_test.outputs[0]
    npout = forward_numpy_call(data_tmp)
    assert_almost_equal(out, npout)


@with_seed()
def test_mathematical():
    # rsqrt
    mathematical_core("rsqrt",
                      lambda x: mx.sym.rsqrt(x),
                      lambda x: 1 / np.sqrt(x),
                      lambda x: -(1.0 / (2.0 * x * np.sqrt(x))))
    # tan
    mathematical_core("tan", lambda x: mx.sym.tan(x), lambda x: np.tan(x), lambda x: np.tan(x) ** 2 + 1)
    # arcsin
    mathematical_core("arcsin", lambda x: mx.sym.arcsin(x), lambda x: np.arcsin(x),
                      lambda x: 1. / (1. - x ** 2) ** (1. / 2.), 0.5, 0.5)
    # arccos
    mathematical_core("arccos", lambda x: mx.sym.arccos(x), lambda x: np.arccos(x),
                      lambda x: -1. / (1. - x ** 2.) ** (1. / 2.), 0.5, 0.5)
    # arctan
    mathematical_core("arctan", lambda x: mx.sym.arctan(x), lambda x: np.arctan(x),
                      lambda x: 1. / (x ** 2. + 1.), 0.5, 0.5)
    # hypot
    mathematical_core_binary("hypot",
                             lambda x, y: mx.sym.hypot(x, y),
                             lambda x, y: np.hypot(x, y),
                             lambda x, y: x / np.hypot(x, y),
                             lambda x, y: y / np.hypot(x, y),
                             0.5, 0.5, 0.5)

    # hypot scalar
    mathematical_core("hypot scalar",
                      lambda x: mx.sym.hypot(x, 3),
                      lambda x: np.hypot(x, 3),
                      lambda x: x / np.hypot(x, 3),
                      0.5, 0.5)

    # degrees
    mathematical_core("degrees",
                      lambda x: mx.sym.degrees(x),
                      lambda x: np.degrees(x),
                      lambda x: 180./np.pi,
                      0.5, 0.5)
    # radians
    mathematical_core("radians",
                      lambda x: mx.sym.radians(x),
                      lambda x: np.radians(x),
                      lambda x: np.pi / 180.,
                      0.6, 1)
    # sinh
    mathematical_core("sinh", lambda x: mx.sym.sinh(x), lambda x: np.sinh(x), lambda x: np.cosh(x))

    # cosh
    mathematical_core("cosh", lambda x: mx.sym.cosh(x), lambda x: np.cosh(x), lambda x: np.sinh(x), 5, 5)

    # tanh
    mathematical_core("tanh", lambda x: mx.sym.tanh(x), lambda x: np.tanh(x), lambda x: 1. - np.tanh(x) ** 2, 0.5, 1)

    # arcsinh
    mathematical_core("arcsinh", lambda x: mx.sym.arcsinh(x), lambda x: np.arcsinh(x),
                      lambda x: 1./(x**2 + 1.)**(1./2.))

    # arccosh
    mathematical_core("arccosh", lambda x: mx.sym.arccosh(x), lambda x: np.arccosh(x),
                      lambda x: 1./(x**2 - 1.)**(1./2.))

    # arctanh
    mathematical_core("arctanh", lambda x: mx.sym.arctanh(x), lambda x: np.arctanh(x),
                      lambda x: -1./(x**2 - 1.), 0.5)

    # log1p
    mathematical_core("log1p", lambda x: mx.sym.log1p(x), lambda x: np.log1p(x),
                      lambda x: 1. / (1.0 + x), 0.5, 0.5)
    # expm1
    mathematical_core("expm1", lambda x: mx.sym.expm1(x), lambda x: np.expm1(x),
                      lambda x: np.exp(x), 0.5, 0.5)

    # log10
    mathematical_core("log10", lambda x: mx.sym.log10(x), lambda x: np.log10(x),
                      lambda x: 1. / (x * np.log(10.)))

    # log2
    mathematical_core("log2", lambda x: mx.sym.log2(x), lambda x: np.log2(x),
                      lambda x: 1. / (x * np.log(2.)))

    # rint
    rounding("rint", lambda x: mx.sym.rint(x), lambda x: np.rint(x))

    # fix
    rounding("fix", lambda x: mx.sym.fix(x), lambda x: np.fix(x))


@with_seed()
def test_special_functions_using_scipy():
    try:
        from scipy import special as scipy_special
    except:
        print("Could not import scipy. Skipping unit tests for special functions")
        return

    # gamma
    mathematical_core("gamma", lambda x: mx.sym.gamma(x), lambda x: scipy_special.gamma(x),
                     lambda x: scipy_special.gamma(x) * scipy_special.psi(x), 0.5, 0.5)

    # gammaln
    mathematical_core("gammaln", lambda x: mx.sym.gammaln(x), lambda x: scipy_special.gammaln(x),
                     lambda x: scipy_special.psi(x), 0.5, 0.5)


@with_seed()
def test_clip():
    data = mx.symbol.Variable('data')
    shape = (30, 30)
    data_tmp = np.random.uniform(-1, 1, shape).astype('float32')
    test = mx.sym.clip(data, a_max=0.6, a_min=-0.6)
    check_symbolic_forward(test, [data_tmp], [np.clip(data_tmp, -0.6, 0.6)])
    check_symbolic_backward(test, [data_tmp], [np.ones(shape)],
                            [np.where(data_tmp <= 0.6, [1], [0]) * np.where(data_tmp >= -0.6, [1], [0])])

    # Test monitor on symbol using clip

    def simple_callback(name, arr):
        pass

    exe = test.simple_bind(ctx=mx.current_context(), data=shape)
    exe.set_monitor_callback(simple_callback, monitor_all=True)
    exe.forward(is_train=True)
    exe.backward(out_grads=mx.nd.ones(shape))
    mx.nd.waitall()


@with_seed()
def test_init():
    def test_basic_val_init(sym_func, np_func, shape, dtype):
        x = sym_func(shape=shape, dtype=dtype)
        exe = x.bind(default_context(), args=[], args_grad=[])
        exe.forward(is_train=True)
        assert_almost_equal(exe.outputs[0], np_func(shape=shape, dtype=dtype))
        assert exe.outputs[0].asnumpy().dtype == dtype

    def test_arange():
        # General Random Tests
        dtype_list = [np.float32, np.float64, np.int32, np.uint8]
        config_list = [(10,),
                       (0, 10),
                       (5, 100, 4),
                       (50, -50, -2),
                       (-100, 100, 1),
                       (1.3, 456.6, 1.3)]
        for dtype in dtype_list:
            for config in config_list:
                repeats = random.choice([1, 3])
                np_out = np.repeat(np.arange(*config, dtype=dtype), repeats)
                nd_out = mx.nd.arange(*config, repeat=repeats, dtype=dtype)
                assert_almost_equal(np_out, nd_out)

    def test_arange_inferstop():
        s = mx.sym.arange(start=0, stop=None, infer_range=True)
        s = mx.sym.elemwise_add(s, mx.sym.zeros(shape=[5]))
        exe = s.bind(ctx=mx.cpu(), args={})
        exe.forward()
        assert_almost_equal(exe.outputs[0], np.array([0,1,2,3,4]))

    def test_arange_like():
        shape_list = [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)]
        axis_list = [0, -1]
        for sh in shape_list:
            for axis in axis_list:
                val = np.random.rand(*sh)
                data = mx.nd.array(val)
                nd_out = mx.nd.contrib.arange_like(data, start=0, axis=axis)
                np_out = np.arange(start=0, stop=sh[axis])
                assert_almost_equal(nd_out.asnumpy(), np_out)

    def test_arange_like_without_axis():
        shape_list = [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)]
        for sh in shape_list:
            val = np.random.rand(*sh)
            data = mx.nd.array(val)
            nd_out = mx.nd.contrib.arange_like(data, start=0)
            np_out = np.arange(start=0, stop=val.size)
            assert_almost_equal(nd_out.asnumpy(), np_out.reshape(sh))

    test_basic_val_init(mx.sym.zeros, np.zeros, (3, 4), np.float32)
    test_basic_val_init(mx.sym.ones, np.ones, 3, np.int32)
    test_basic_val_init(mx.sym.ones, np.ones, (2, 2, 3), np.float16)
    test_arange()
    test_arange_inferstop()
    test_arange_like()
    test_arange_like_without_axis()


@with_seed()
def test_order():
    ctx = default_context()

    def gt_topk(dat, axis, ret_typ, k, is_ascend):
        if ret_typ == "indices":
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            ret = np.take(dat.argsort(axis=axis), axis=axis, indices=indices, mode='wrap')
        elif ret_typ == "value":
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            ret = np.take(np.sort(dat, axis=axis), axis=axis, indices=indices, mode='wrap')
        else:
            assert dat.shape == (5, 5, 5, 5)
            assert axis is None or axis == 1
            ret = np.zeros(dat.shape)
            if is_ascend:
                indices = np.arange(k)
            else:
                indices = np.arange(-1, -k-1, -1)
            gt_argsort = np.take(dat.argsort(axis=axis), axis=axis, indices=indices, mode='wrap')
            if axis is None:
                ret.ravel()[gt_argsort] = 1
            else:
                for i in range(5):
                    for j in range(5):
                        for k in range(5):
                            ret[i, gt_argsort[i, :, j, k], j, k] = 1
        return ret

    dshape = (5, 5, 5, 5)
    a_npy = np.arange(np.prod(dshape)).astype(np.float32)
    np.random.shuffle(a_npy)
    a_npy = a_npy.reshape(dshape)
    a = mx.sym.Variable('a')

    def get_large_matrix():
      data = np.array([np.arange(300096).astype(np.float32)])
      data = np.repeat(data, 100, axis=0)
      np.apply_along_axis(np.random.shuffle, 1, data)
      return data

    large_matrix_npy = get_large_matrix()

    for axis in [1, 3, None]:
        for is_ascend in [True, False]:
            b = mx.sym.sort(a, axis=axis, is_ascend=is_ascend)
            if axis is None:
                out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=a_npy.size, is_ascend=is_ascend)
            else:
                out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=5, is_ascend=is_ascend)
            check_numeric_gradient(b, location={'a': a_npy}, numeric_eps=1e-2, rtol=1e-2, ctx=ctx)
            check_symbolic_forward(b, location={'a': a_npy}, expected=[out_npy])

    b = mx.sym.topk(a, axis=1, is_ascend=is_ascend, ret_typ="indices", k=5)
    check_symbolic_backward(sym=b, location={'a': large_matrix_npy},
                            out_grads=[np.random.normal(size=(100, 5))],
                            expected=[np.zeros((100, 300096))])
    check_symbolic_forward(b, location={'a': large_matrix_npy},
                           expected=[gt_topk(dat=large_matrix_npy, axis=1,
                                             ret_typ="indices", k=5,
                                             is_ascend=is_ascend)])

    b = mx.sym.argsort(a, axis=1, is_ascend=False)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=5,
                                             is_ascend=False)])

    b = mx.sym.argmax(a, axis=1, keepdims=True)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=1,
                                             is_ascend=False)])

    b = mx.sym.argmin(a, axis=1, keepdims=True)
    check_symbolic_backward(sym=b, location={'a': a_npy},
                            out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                            expected=[np.zeros((5, 5, 5, 5))])
    check_symbolic_forward(b, location={'a': a_npy},
                           expected=[gt_topk(dat=a_npy, axis=1, ret_typ="indices", k=1,
                                             is_ascend=True)])

    for dtype in [np.float16, np.float32, np.float64]:
        dshape = (5, 5, 5, 5)
        a_npy = np.arange(np.prod(dshape)).astype(dtype)
        np.random.shuffle(a_npy)
        a_npy = a_npy.reshape(dshape)
        a = mx.sym.Variable('a')
        for axis in [1, 3, None]:
            K = [1, 3, 5, 7] if axis is None else [1, 3, 5]
            for k in K:
                for is_ascend in [True, False]:
                    b = mx.sym.topk(a, axis=axis, is_ascend=is_ascend, ret_typ="value", k=k)
                    out_npy = gt_topk(dat=a_npy, axis=axis, ret_typ="value", k=k, is_ascend=is_ascend)
                    check_numeric_gradient(b, location={'a': a_npy}, numeric_eps=1e-2, rtol=1e-2, ctx=ctx)
                    check_symbolic_forward(b, location={'a': a_npy}, expected=[out_npy])

        b = mx.sym.topk(a, axis=1, is_ascend=is_ascend, ret_typ="indices", k=5)
        check_symbolic_backward(sym=b, location={'a': large_matrix_npy},
                out_grads=[np.random.normal(size=(100, 5))],
                expected=[np.zeros((100, 300096))])
        check_symbolic_forward(b, location={'a': large_matrix_npy},
                expected=[gt_topk(dat=large_matrix_npy, axis=1,
                    ret_typ="indices", k=5, is_ascend=is_ascend)])

        b = mx.sym.topk(a, axis=3, is_ascend=is_ascend, ret_typ="indices", k=3)
        check_symbolic_backward(sym=b, location={'a': a_npy},
                out_grads=[np.random.normal(size=(5, 5, 5, 3))],
                expected=[np.zeros((5, 5, 5, 5))])
        check_symbolic_forward(b, location={'a': a_npy},
                expected=[gt_topk(dat=a_npy, axis=3, ret_typ="indices", k=3,
                    is_ascend=False)])

        b = mx.sym.topk(a, axis=1, is_ascend=True, ret_typ="mask", k=3)
        check_symbolic_backward(sym=b, location={'a': a_npy},
                out_grads=[np.random.normal(size=(5, 5, 5, 5))],
                expected=[np.zeros((5, 5, 5, 5))])
        check_symbolic_forward(b, location={'a': a_npy},
                expected=[gt_topk(dat=a_npy, axis=1, ret_typ="mask", k=3,
                    is_ascend=True)])


@with_seed()
def test_blockgrad():
    a = mx.sym.Variable('a')
    b = mx.sym.BlockGrad(a)
    exe = b.simple_bind(ctx=default_context(), a=(10, 10))
    a_npy = np.random.rand(10, 10)
    exe.forward(is_train=True, a=a_npy)
    assert_almost_equal(exe.outputs[0], a_npy)
    exe.backward()  # No error if BlockGrad works


@with_seed()
def test_take():
    def grad_helper(grad_in, axis, idx):
        if axis == 0:
            if axis == len(grad_in.shape) - 1:
                grad_in[idx] += 1.0
            else:
                grad_in[idx, :] += 1.0
        elif axis == 1:
            if axis == len(grad_in.shape) - 1:
                grad_in[:, idx] += 1.0
            else:
                grad_in[:, idx, :] += 1.0
        elif axis == 2:
            if axis == len(grad_in.shape) - 1:
                grad_in[:, :, idx] += 1.0
            else:
                grad_in[:, :, idx, :] += 1.0
        elif axis == 3:
            if axis == len(grad_in.shape) - 1:
                grad_in[:, :, :, idx] += 1.0
            else:
                grad_in[:, :, :, idx, :] += 1.0
        elif axis == 4:
            grad_in[:, :, :, :, idx] += 1.0
        else:
            raise ValueError("axis %d is not supported..." % axis)

    def check_output_n_grad(data_shape, idx_shape, axis, mode, out_of_range=True):
        data = mx.sym.Variable('a')
        idx = mx.sym.Variable('indices')
        idx = mx.sym.BlockGrad(idx)
        result = mx.sym.take(a=data, indices=idx, axis=axis, mode=mode)
        exe = result.simple_bind(default_context(), a=data_shape,
                                 indices=idx_shape, axis=axis, mode=mode)
        data_real = np.random.normal(size=data_shape).astype('float32')
        if out_of_range:
            idx_real = np.random.randint(low=-data_shape[axis], high=data_shape[axis], size=idx_shape)
            if mode == 'raise':
                idx_real[idx_real == 0] = 1
                idx_real *= data_shape[axis]
        else:
            idx_real = np.random.randint(low=0, high=data_shape[axis], size=idx_shape)
        if axis < 0:
            axis += len(data_shape)

        grad_out = np.ones((data_shape[0:axis] if axis > 0 else ()) + idx_shape + (data_shape[axis+1:] if axis < len(data_shape) - 1 else ()), dtype='float32')
        grad_in = np.zeros(data_shape, dtype='float32')

        exe.arg_dict['a'][:] = mx.nd.array(data_real)
        exe.arg_dict['indices'][:] = mx.nd.array(idx_real)
        exe.forward(is_train=True)
        if out_of_range and mode == 'raise':
            try:
                mx_out = exe.outputs[0].asnumpy()
            except MXNetError as e:
                return
            else:
                # Did not raise exception
                assert False, "did not raise %s" % MXNetError.__name__

        assert_almost_equal(exe.outputs[0], np.take(data_real, idx_real, axis=axis, mode=mode))

        for i in np.nditer(idx_real):
            if mode == 'clip':
                i = np.clip(i, 0, data_shape[axis])
            grad_helper(grad_in, axis, i)

        exe.backward([mx.nd.array(grad_out)])
        assert_almost_equal(exe.grad_dict['a'], grad_in)

    def check_autograd_req():
        row_len = 2
        col_len = 8
        shape = (row_len, col_len)
        sc = mx.nd.random.uniform(-1.0, 1.0, shape=shape, dtype="float32")
        sc.attach_grad()
        i = mx.nd.array([0], dtype="int64")
        j = mx.nd.array([0], dtype="int64")
        with mx.autograd.record(train_mode=True):
            xs = []
            for _ in range(row_len):
                x_i = []
                for _ in range(col_len):
                    x_ij = sc.take(i).squeeze(axis=0).take(j).squeeze(axis=0)
                    x_i.append(x_ij)
                    j = j + 1
                i = i + 1
                j = j - col_len  # reset j
                xs.append(mx.nd.stack(*x_i))
            x = mx.nd.stack(*xs)
            x = x.sum()

        x.backward()
        assert_almost_equal(np.ones(sc.grad.shape), sc.grad)

    for mode in ['clip', 'wrap', 'raise']:
        for data_ndim in range(1, 5):
            for idx_ndim in range(1, 4):
                for axis in range(-data_ndim, data_ndim):
                    data_shape = ()
                    for _ in range(data_ndim):
                        data_shape += (np.random.randint(low=1, high=5), )
                    idx_shape = ()
                    for _ in range(idx_ndim):
                        idx_shape += (np.random.randint(low=1, high=5), )
                    if mode == 'raise':
                        check_output_n_grad(data_shape, idx_shape, axis, 'raise', False)
                    check_output_n_grad(data_shape, idx_shape, axis, mode)

    check_autograd_req()


@with_seed()
def test_grid_generator():
    # transform_type =  affine
    test_case = [(20,21),(4,3),(6,12),(15,17)]
    for target_shape in test_case:
        affine_matrix =  mx.sym.Variable('affine')
        grid = mx.sym.GridGenerator(data=affine_matrix,transform_type='affine', target_shape=target_shape)
        exe = grid.simple_bind(ctx=default_context(), affine=(1,6), grad_req='write')

        # check forward
        exe.arg_dict['affine'][:] = np.array([[1.0,0,0,0,1.0,0]])
        exe.forward(is_train=True)
        output = exe.outputs[0]
        output[0,0,:,:] = (output[0,0,:,:] + 1) * (target_shape[1] - 1) / 2.0
        output[0,1,:,:] = (output[0,1,:,:] + 1) * (target_shape[0] - 1) / 2.0
        xv, yv = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]))
        assert_almost_equal(output[0,0], yv.T)
        assert_almost_equal(output[0,1], xv.T)

        # check backward
        out_grad = np.random.normal(size=(1,2)+target_shape)
        exe.backward(mx.nd.array(out_grad))
        tmp = np.zeros((3,target_shape[0]*target_shape[1]))
        tmp[0] = -1.0 + (np.arange(target_shape[0]*target_shape[1]) % target_shape[1]) * (2.0 / (target_shape[1]-1))
        tmp[1] = -1.0 + (np.arange(target_shape[0]*target_shape[1]) // target_shape[1]) * (2.0 / (target_shape[0]-1))
        tmp[2] = 1
        grad_est = np.dot(out_grad[0].reshape(2,target_shape[0]*target_shape[1]),tmp.T).reshape(1,6)
        assert_almost_equal(exe.grad_dict['affine'], grad_est)
        # check addto
        exe = grid.simple_bind(ctx=default_context(), affine=(1,6), grad_req='add')
        grid_grad_npy = np.random.normal(size=exe.grad_dict['affine'].shape)
        exe.grad_dict['affine'][:] = grid_grad_npy
        exe.arg_dict['affine'][:] = np.array([[1.0, 0, 0, 0, 1.0, 0]])
        exe.forward(is_train=True)
        exe.backward(mx.nd.array(out_grad))
        assert_almost_equal(exe.grad_dict['affine'], grad_est + grid_grad_npy)

    # transform_type = warp
    test_case = [(12,21),(4,3),(6,12)]
    for target_shape in test_case:
        flow = mx.sym.Variable('flow')
        grid = mx.sym.GridGenerator(data=flow,transform_type='warp', target_shape=target_shape)
        exe = grid.simple_bind(ctx=default_context(), flow=(1,2)+target_shape, grad_req='write')
        # check forward
        exe.arg_dict['flow'][:] = np.ones((1,2)+target_shape)
        exe.forward(is_train=True)
        output = exe.outputs[0].asnumpy()
        output[0,0,:,:] = (output[0,0,:,:] + 1) * (target_shape[1] - 1) / 2.0
        output[0,1,:,:] = (output[0,1,:,:] + 1) * (target_shape[0] - 1) / 2.0
        xv, yv = np.meshgrid(np.arange(target_shape[0])+1, np.arange(target_shape[1])+1)
        assert_almost_equal(output[0,0], yv.T)
        assert_almost_equal(output[0,1], xv.T)
        # check backward
        out_grad = np.random.normal(size=(1,2)+target_shape)
        exe.backward(mx.nd.array(out_grad))
        grad_est = np.zeros((1,2)+target_shape)
        grad_est[0,0] = out_grad[0,0] / ((target_shape[1]-1.0) / 2.0)
        grad_est[0,1] = out_grad[0,1] / ((target_shape[0]-1.0) / 2.0)
        assert_almost_equal(exe.grad_dict['flow'], grad_est, rtol=1e-3)
        # check addto
        exe_add = grid.simple_bind(ctx=default_context(), flow=(1, 2) + target_shape, grad_req='add')
        flow_grad_npy = np.random.normal(size=exe_add.grad_dict['flow'].shape)
        exe_add.arg_dict['flow'][:] = np.ones((1, 2) + target_shape)
        exe_add.grad_dict['flow'][:] = flow_grad_npy
        exe_add.forward(is_train=True)
        exe_add.backward(mx.nd.array(out_grad))
        assert_almost_equal(exe_add.grad_dict['flow'], grad_est + flow_grad_npy, rtol=1e-3, atol=1e-5)


@with_seed()
def test_index2d():
    for _ in range(30):
        n = np.random.randint(1, 100)
        m = np.random.randint(1, 500)
        data = mx.random.uniform(-1, 1, shape=(n, m), ctx=default_context())
        x = mx.nd.array(np.random.randint(0, m, size=n), ctx=default_context(), dtype='int32')
        r = mx.nd.batch_take(data, x)
        assert_almost_equal(r, data.asnumpy()[np.arange(n), x.asnumpy()])


@with_seed()
def test_cast():
    for srctype in [np.int32, np.float32, np.float16]:
        for dsttype in [np.float32, np.int32, np.float16]:
            x = mx.sym.Variable('x', dtype=srctype)
            y = mx.sym.Cast(x, dtype=dsttype)
            exe = y.simple_bind(ctx=default_context(), x=(10, 10))
            assert exe.arg_arrays[0].dtype == srctype
            assert exe.outputs[0].dtype == dsttype
            X = np.random.uniform(-10, 10, size=(10, 10))
            exe.arg_arrays[0][:] = X
            exe.forward(is_train=True)
            exe.backward(mx.nd.array(X, dtype=dsttype, ctx=default_context()))
            assert_almost_equal(exe.outputs[0], X.astype(srctype).astype(dsttype), rtol=1e-3, atol=1e-5)
            assert_almost_equal(exe.grad_arrays[0], X.astype(dsttype).astype(srctype), rtol=1e-3, atol=1e-5)

def get_cast_op_data():
    FP16_FRACTION_BITS = 10
    FP32_FRACTION_BITS = 23
    FP32_EXP_MIN = -126
    FP32_EXP_MAX = 127
    # generate test cases in the vicinity of representable float16 mantissas
    # and mid-way between them, but over the full range of float32 exponents.

    for sign_bit in [0, 1]:
        for exponent in range(FP32_EXP_MIN - FP32_FRACTION_BITS - 1, FP32_EXP_MAX + 2):
            denominator = 2**(FP16_FRACTION_BITS + 1)
            for numerator in range(0, denominator):
                fraction = numerator / float(denominator)
                for y in [-1.0, 0.0, 1.0]:
                    small_delta = y / 2**FP32_FRACTION_BITS
                    val = (-1.0)**sign_bit * 2.0**exponent * (1.0 + fraction + small_delta)
                    yield val
    # Add np.nan as a final data value to process
    yield np.nan

# Test requires all platforms to round float32->float16 with same round-to-nearest-even policy.
@with_seed()
def test_cast_float32_to_float16():
    input_np = np.array(list(get_cast_op_data())).astype(np.float32)
    # The intermediate cast to np.float64 below gets around a numpy rounding bug that is fixed
    # as of numpy 1.17 by PR https://github.com/numpy/numpy/pull/12722
    expected_output = input_np.astype(np.float64).astype(np.float16)

    def check_cast(op, input_np, expected_output):
        x = mx.sym.Variable('x', dtype=np.float32)
        sym = op(x, dtype=np.float16)
        ctx = default_context()
        exe = sym.bind(ctx, {'x': mx.nd.array(input_np, dtype=np.float32, ctx=ctx)})
        assert exe.arg_arrays[0].dtype == np.float32
        assert exe.outputs[0].dtype == np.float16
        exe.forward(is_train=True)
        sym_output = exe.outputs[0].asnumpy()
        for fp32_val, model_fp16_val, np_fp16_val in zip(input_np, sym_output, expected_output):
            assert (model_fp16_val == np_fp16_val) or \
                   (np.isnan(model_fp16_val) and np.isnan(np_fp16_val)), \
                   'fp32->fp16 cast mismatch: with fp32 value {}, model_fp16 = {}, numpy_fp16 = {}'.format(
                    fp32_val, model_fp16_val, np_fp16_val)

    check_cast(mx.sym.Cast, input_np, expected_output)
    check_cast(mx.sym.amp_cast, input_np, expected_output)


@with_seed()
def test_amp_multicast():
    if default_context().device_type == 'cpu':
        return
    x = mx.sym.Variable('x', dtype=np.float16)
    y = mx.sym.Variable('y', dtype=np.float32)
    z = mx.sym.Variable('z', dtype=np.float16)
    ctx = default_context()
    res = mx.sym.amp_multicast(x, y, z, num_outputs=3)
    exe = res.bind(ctx, {'x': mx.nd.random.uniform(shape=(3, 3), dtype=np.float16, ctx=ctx),
                         'y': mx.nd.random.uniform(shape=(3, 3), dtype=np.float32, ctx=ctx),
                         'z': mx.nd.random.uniform(shape=(3, 3), dtype=np.float16, ctx=ctx)})
    exe.forward(is_train=True)
    out1, out2, out3 = exe.outputs
    assert out1.asnumpy().dtype == np.float32
    assert out2.asnumpy().dtype == np.float32
    assert out3.asnumpy().dtype == np.float32

    def check_amp_multicast(input_np, expected_output):
        x = mx.sym.Variable('x', dtype=np.float16)
        y = mx.sym.Variable('y', dtype=np.float32)
        z = mx.sym.Variable('z', dtype=np.float16)
        ctx = default_context()
        res = mx.sym.amp_multicast(x, y, z, num_outputs=3)
        exe = res.bind(ctx, {'x': mx.nd.array(input_np, dtype=np.float16, ctx=ctx),
                             'y': mx.nd.array(input_np, dtype=np.float32, ctx=ctx),
                             'z': mx.nd.array(input_np, dtype=np.float16, ctx=ctx)})
        exe.forward(is_train=True)
        sym_output = exe.outputs[0].asnumpy()
        for fp32_val, model_fp16_val, np_fp16_val in zip(input_np, sym_output, expected_output):
            assert (model_fp16_val == np_fp16_val) or \
                   (np.isnan(model_fp16_val) and np.isnan(np_fp16_val)), \
                   'fp32->fp16 cast mismatch: with fp32 value {}, model_fp16 = {}, numpy_fp16 = {}'.format(
                    fp32_val, model_fp16_val, np_fp16_val)

    input_np = np.array(list(get_cast_op_data()), dtype=np.float16)
    expected_output = input_np.astype(np.float32)
    check_amp_multicast(input_np, expected_output)


@with_seed()
def test_all_finite():
    data = mx.sym.Variable("data", dtype=np.float32)
    data2 = mx.sym.Variable("data2", dtype=np.float32)
    finite_arr = mx.nd.array([[0, 0]])
    inf_arr = mx.nd.array([[np.inf, np.inf]])
    z = mx.sym.all_finite(data)
    ctx = default_context()
    exe = z.bind(ctx, {'data': inf_arr})
    exe.forward(is_train=False)
    sym_output = exe.outputs[0].asnumpy()
    assert sym_output[0] == 0
    exe = z.bind(ctx, {'data': finite_arr})
    exe.forward(is_train=False)
    sym_output = exe.outputs[0].asnumpy()
    assert sym_output[0] == 1
    z = mx.sym.multi_all_finite(data, data2, num_arrays=2)
    exe = z.bind(ctx, {'data': finite_arr, 'data2': inf_arr})
    exe.forward(is_train=False)
    sym_output = exe.outputs[0].asnumpy()
    assert sym_output[0] == 0
    z = mx.sym.multi_all_finite(data, data2, num_arrays=2)
    exe = z.bind(ctx, {'data': finite_arr, 'data2': finite_arr})
    exe.forward(is_train=False)
    sym_output = exe.outputs[0].asnumpy()
    assert sym_output[0] == 1


@with_seed()
def test_repeat():
    def test_repeat_forward():
        ndim_max = 6 # max number of dims of the ndarray
        size_max = 10 # max number of elements in each dim
        repeats = 3
        for ndim in range(1, ndim_max+1):
            shape = ()
            for i in range(0, ndim):
                shape += (np.random.randint(1, size_max+1), )
            a = np.random.random_sample(size=shape)
            aa = np.repeat(a, repeats)
            b = mx.nd.array(a, ctx=default_context())
            bb = mx.nd.repeat(b, repeats)
            assert_almost_equal(aa, bb)

            for axis in range(0, ndim):
                aa = np.repeat(a, repeats, axis)
                bb = mx.nd.repeat(b, repeats, axis)
                assert_almost_equal(aa, bb)

    def test_repeat_backward(axis):
        data = mx.sym.Variable('data')
        n1 = 3
        n2 = 4
        shape = (n1, n2)
        data_tmp = np.random.randint(0, 10, n1 * n2).reshape(shape)
        arr_data = mx.nd.array(data_tmp)
        arr_grad = mx.nd.empty(shape)
        repeats = 2
        test = mx.sym.repeat(data, repeats=repeats, axis=axis)
        exe = test.bind(ctx=default_context(), args=[arr_data], args_grad=[arr_grad])
        npout_grad = np.random.randint(0, 10, n1 * n2 * repeats)
        if axis == 0:
            npout_grad = npout_grad.reshape(n1 * repeats, n2)
        elif axis == 1:
            npout_grad = npout_grad.reshape(n1, n2 * repeats)
        else:
            raise RuntimeError("Invalid axis value")
        out_grad = mx.nd.array(npout_grad)
        exe.backward(out_grad)

        expected_grad = np.zeros(shape)
        if axis == 0:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    k = i * repeats
                    expected_grad[i][j] = sum(npout_grad[k:k + repeats, j])
        elif axis == 1:
            for j in range(shape[1]):
                for i in range(shape[0]):
                    k = j * repeats
                    expected_grad[i][j] = sum(npout_grad[i, k:k + repeats])
        else:
            raise RuntimeError("Invalid axis value")

        assert_almost_equal(expected_grad, arr_grad, rtol=1e-3)

    def test_repeat_numeric_gradient():
        data = mx.sym.Variable('data')
        n1 = 3
        n2 = 4
        shape = (n1, n2)
        data_tmp = np.random.randint(0, 10, n1 * n2).reshape(shape)
        repeats = 2

        test = mx.sym.repeat(data, repeats=repeats, axis=0)
        check_numeric_gradient(test, [data_tmp], numeric_eps=1e-3, rtol=1e-2)

    test_repeat_forward()
    test_repeat_backward(axis=0)
    test_repeat_backward(axis=1)
    test_repeat_numeric_gradient()


@with_seed()
def test_reverse():
    data = mx.symbol.Variable('data')
    shape = (5, 5, 5)
    data_tmp = np.random.uniform(-1, 1, shape)
    test = mx.sym.reverse(data, axis=[1, 2])
    grad = np.random.uniform(-1, 1, shape)
    check_numeric_gradient(test, [data_tmp], numeric_eps=2E-2)
    check_symbolic_forward(test, [data_tmp], [data_tmp[:, ::-1, ::-1]])
    check_symbolic_backward(test, [data_tmp], [grad], [grad[:, ::-1, ::-1]])


@with_seed()
def test_tile():
    def test_normal_case():
        ndim_min = 1
        ndim_max = 5  # max number of dims of the ndarray
        size_max = 10  # max number of elements in each dim
        length_max = 3  # max length of reps
        rep_max = 10  # max number of tiling in each dim
        for ndim in range(ndim_min, ndim_max+1):
            shape = []
            for i in range(1, ndim+1):
                shape.append(np.random.randint(1, size_max+1))
            shape = tuple(shape)
            a = np.random.randint(0, 100, shape)
            b = mx.nd.array(a, dtype=a.dtype)

            reps_len = np.random.randint(1, length_max+1)
            reps_tuple = ()
            for i in range(1, reps_len):
                reps_tuple += (np.random.randint(1, rep_max), )
            reps_array = np.asarray(reps_tuple)

            a_tiled = np.tile(a, reps_array)
            b_tiled = mx.nd.tile(b, reps_tuple).asnumpy()
            assert same(a_tiled, b_tiled)

    def test_empty_tensor():
        shape = (2, 3, 0, 4)
        with mx.np_shape():
            a = np.array([], dtype=np.int32).reshape(shape)
            b = mx.nd.array(a, ctx=default_context(), dtype=a.dtype)

            reps = (2, 4, 6)
            a_tiled = np.tile(a, reps)
            b_tiled = mx.nd.tile(b, reps).asnumpy()
            assert same(a_tiled, b_tiled)

    def test_empty_reps():
        a = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32)
        b = mx.nd.array(a, ctx=default_context(), dtype=a.dtype)
        a_tiled = np.tile(a, ())
        b_tiled = mx.nd.tile(b, ()).asnumpy()
        assert same(a_tiled, b_tiled)

    def test_tile_backward():
        data = mx.sym.Variable('data')
        n1 = 2
        n2 = 2
        shape = (n1, n2)
        data_tmp = np.random.randint(0, 10, n1 * n2).reshape(shape)
        arr_data = mx.nd.array(data_tmp)
        arr_grad = mx.nd.empty(shape)
        reps1 = 2
        reps2 = 2
        reps = (reps1, reps2)
        test = mx.sym.tile(data, reps=reps)
        exe = test.bind(ctx=default_context(), args=[arr_data], args_grad=[arr_grad])
        npout_grad = np.random.randint(0, 10, n1 * n2 * reps1 * reps2).reshape(n1 * reps1, n2 * reps2)
        out_grad = mx.nd.array(npout_grad)
        exe.backward(out_grad)

        expected_grad = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                expected_grad[i][j] += sum(sum(npout_grad[i:(n1 * reps1):reps1, j:(n2 * reps2):reps2]))

        assert_almost_equal(expected_grad, arr_grad, rtol=1e-3)

    def test_tile_numeric_gradient():
        data = mx.sym.Variable('data')
        n1 = 2
        n2 = 2
        shape = (n1, n2)
        data_tmp = np.random.randint(0, 10, n1 * n2).reshape(shape)
        reps1 = 2
        reps2 = 2
        reps = (reps1, reps2)
        test = mx.sym.tile(data, reps=reps)
        check_numeric_gradient(test, [data_tmp], numeric_eps=1e-2, rtol=1e-2)

    def test_invalid_reps():
        data = mx.nd.arange(16).reshape((4, 4))
        assert_exception(mx.nd.tile, MXNetError, data, (1, 2, -3))
        assert_exception(mx.nd.tile, MXNetError, data, (1, 0, 3))

    test_normal_case()
    with mx.np_shape():
        test_empty_tensor()
    test_empty_reps()
    test_tile_backward()
    test_tile_numeric_gradient()
    test_invalid_reps()


@with_seed()
def test_one_hot():
    def test_normal_case(index_type=np.int32):
        ndim_max = 6
        dim_size_max = 20
        depth = int(dim_size_max / 2)
        on_value = 1
        off_value = 0
        for ndim in range(1, ndim_max+1):
            shape = ()
            for i in range(1, ndim+1):
                shape += (np.random.randint(1, dim_size_max+1), )
            indices = np.random.randint(-dim_size_max, dim_size_max+1,
                                        size=np.prod(shape)).reshape(shape)
            mx_one_hot_array = mx.nd.one_hot(
                mx.nd.array(indices, ctx=default_context(), dtype=index_type),
                depth=depth, dtype=np.int32)
            expected_array = np.zeros((np.prod(shape), depth), dtype=np.int32)
            expected_array[:] = off_value
            indices_1d = indices.flatten()
            row = 0
            for idx in indices_1d:
                if 0 <= idx < depth:
                    expected_array[row, idx] = on_value
                row += 1
            expected_array = expected_array.reshape(shape + (depth, ))
            one_hot_array = mx_one_hot_array.asnumpy()
            assert same(expected_array, one_hot_array)

    def test_empty_indices():
        shape = (2, 0, 9, 3)
        with mx.np_shape():
            indices = np.array([]).reshape(shape)
            depth = 10
            mx_one_hot_array = mx.nd.one_hot(
                mx.nd.array(indices, ctx=default_context(), dtype=np.int32),
                depth=depth, dtype=np.int32
            ).asnumpy()
            expected_array = np.array([], dtype=np.int32).reshape(shape + (depth,))
            assert same(expected_array, mx_one_hot_array)

    def test_zero_depth():
        shape = (2, 4, 9, 3)
        indices = np.ones(shape)
        depth = 0
        mx_one_hot_array = mx.nd.one_hot(
            mx.nd.array(indices, ctx=default_context(), dtype=np.int32),
            depth=depth, dtype=np.int32).asnumpy()
        expected_array = np.array([], dtype=np.int32).reshape(shape + (depth, ))
        assert same(expected_array, mx_one_hot_array)

    test_normal_case(index_type=np.int32)
    test_normal_case(index_type=np.float64)
    test_normal_case(index_type=np.float32)
    test_normal_case(index_type=np.float16)
    with mx.np_shape():
        test_empty_indices()
    test_zero_depth()


@with_seed()
def test_where():
    def get_forward_expected_output(condition, x, y):
        original_shape = x.shape
        out = np.zeros(original_shape)
        if condition.shape == x.shape:
            for index, c in np.ndenumerate(condition):
                if c != 0:
                    out[index] = x[index]
                else:
                    out[index] = y[index]
        elif condition.shape == (x.shape[0], ):
            s = x.shape
            m = s[0]
            n = int(np.prod(s)/s[0])
            x2d = x.reshape((m, n))
            y2d = y.reshape((m, n))
            out = out.reshape((m, n))
            for i in range(0, m):
                if condition[i] != 0:
                    for j in range(0, n):
                        out[i, j] = x2d[i, j]
                else:
                    for j in range(0, n):
                        out[i, j] = y2d[i, j]
        else:
            raise RuntimeError("Invalid condition shape for where op")

        out = out.reshape(original_shape)
        return out

    def get_forward_inputs_same_shape(shape):
        condition_np = np.random.randint(0, 2, np.prod(shape)).reshape(shape)
        x_np = np.random.randint(1, 6, np.prod(shape)).reshape(shape)
        y_np = np.random.randint(7, 11, np.prod(shape)).reshape(shape)
        return condition_np, x_np, y_np

    def get_forward_inputs_condition_vector(shape):
        condition_np = np.random.randint(0, 2, shape[0])
        x_np = np.random.randint(1, 6, np.prod(shape)).reshape(shape)
        y_np = np.random.randint(7, 11, np.prod(shape)).reshape(shape)
        return condition_np, x_np, y_np

    def get_backward_input(shape):
        return np.random.randint(20, 30, np.prod(shape)).reshape(shape)

    def get_backward_expected_outputs(grad_in, condition):
        shape = grad_in.shape
        grad_cond = np.zeros(condition.shape)
        grad_x = np.empty(shape)
        grad_y = np.empty(shape)

        for index, c in np.ndenumerate(condition):
            if 0 != c:
                grad_x[index] = grad_in[index]
                grad_y[index] = 0
            else:
                grad_x[index] = 0
                grad_y[index] = grad_in[index]

        return grad_cond, grad_x, grad_y

    def test_where_helper(shape, same_shape):
        if same_shape:
            condition_np, x_np, y_np = get_forward_inputs_same_shape(shape)
        else:
            condition_np, x_np, y_np = get_forward_inputs_condition_vector(shape)

        out_expected = get_forward_expected_output(condition_np, x_np, y_np)

        grad_in_np = get_backward_input(shape)
        grad_expected_cond, grad_expected_x, grad_expected_y\
            = get_backward_expected_outputs(grad_in_np, condition_np)

        condition = mx.sym.Variable('condition')
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        grad_in_mx = mx.nd.array(grad_in_np, dtype=np.int32)
        where_sym = mx.sym.where(condition, x, y)

        # test req='write'
        where_exe_write = where_sym.simple_bind(ctx=default_context(),
                                                condition=condition_np.shape,
                                                x=x_np.shape, y=y_np.shape,
                                                grad_req='write')
        # test forward req='write'
        outputs = where_exe_write.forward(is_train=True, condition=condition_np,
                                          x=x_np, y=y_np)
        assert same(outputs[0].asnumpy(), out_expected)
        # test backward req='write'
        where_exe_write.backward(grad_in_mx)
        assert same(where_exe_write.grad_dict['x'].asnumpy(), grad_expected_x)
        assert same(where_exe_write.grad_dict['y'].asnumpy(), grad_expected_y)
        assert same(where_exe_write.grad_dict['condition'].asnumpy(), grad_expected_cond)

        # test req='add'
        x_grad_init = np.random.randint(30, 40, np.prod(shape)).reshape(shape)
        y_grad_init = np.random.randint(40, 50, np.prod(shape)).reshape(shape)
        where_exe_add = where_sym.simple_bind(ctx=default_context(),
                                              condition=condition_np.shape,
                                              x=x_np.shape, y=y_np.shape,
                                              grad_req='add')
        where_exe_add.grad_dict['x'][:] = x_grad_init
        where_exe_add.grad_dict['y'][:] = y_grad_init
        # test forward req='add'
        outputs = where_exe_add.forward(is_train=True, condition=condition_np, x=x_np, y=y_np)
        assert same(outputs[0].asnumpy(), out_expected)
        # test backward req='add'
        where_exe_add.backward(grad_in_mx)
        x_ograd = where_exe_add.grad_dict['x'].asnumpy()
        y_ograd = where_exe_add.grad_dict['y'].asnumpy()
        assert same(x_ograd, grad_expected_x+x_grad_init)
        assert same(y_ograd, grad_expected_y+y_grad_init)

    def test_where_numeric_gradient(shape, same_shape):
        condition = mx.sym.Variable('condition')
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        where_sym = mx.sym.where(condition, x, y)
        if same_shape:
            condition_np, x_np, y_np = get_forward_inputs_same_shape(shape)
        else:
            condition_np, x_np, y_np = get_forward_inputs_condition_vector(shape)
        check_numeric_gradient(where_sym, [condition_np, x_np, y_np], grad_nodes=['x', 'y'])

    def test_invalid_shape():
        condition = mx.sym.Variable('condition')
        x = mx.sym.Variable('x')
        y = mx.sym.Variable('y')
        where_sym = mx.sym.where(condition, x, y)

        assert_exception(lambda: where_sym.eval(x=mx.nd.array([[2,3],[4,5],[6,7]]),
                                                y=mx.nd.array([[8,9],[10,11],[12,13]]),
                                                condition=mx.nd.array([1,0])), MXNetError)

        assert_exception(lambda: mx.nd.where(x=mx.nd.array([[2,3],[4,5],[6,7]]),
                                             y=mx.nd.array([[8,9],[10,11],[12,13]]),
                                             condition=mx.nd.array([1,0])), MXNetError)

    def test_1d_cond():
        cond = mx.nd.array([1, 0, 1])
        x = mx.nd.array([[2, 3], [4, 5], [6, 7]])
        y = mx.nd.array([[7, 8], [9, 10], [10, 11]])
        expect_out = np.array([[2, 3], [9, 10], [6, 7]])
        out = mx.nd.where(cond, x, y).asnumpy()
        assert(expect_out.all() == out.all())

    test_where_helper((5, 9), True)
    test_where_helper((5, 9), False)
    test_where_helper((5, 7, 9), True)
    test_where_helper((5, 7, 9), False)
    test_where_helper((10, 8, 15, 3), True)
    test_where_helper((10, 8, 15, 3), False)
    test_where_numeric_gradient((5, 9), True)
    test_where_numeric_gradient((5, 9), False)
    test_where_numeric_gradient((5, 7, 9), True)
    test_where_numeric_gradient((5, 7, 9), False)
    test_invalid_shape()
    test_1d_cond()


@with_seed()
def test_softmin():
    for ndim in range(1, 5):
        for dtype in [np.float16, np.float32, np.float64]:
            rtol, atol = (1e-2, 5e-3) if dtype is np.float16 else (1e-3, 1e-3)
            shape = np.random.randint(1, 5, size=ndim)
            axis = np.random.randint(-ndim, ndim)
            data = np.random.uniform(-2, 2, size=shape).astype(dtype)
            data = data / 10 if dtype is np.float16 else data
            sym = mx.sym.softmin(axis=axis)
            expected_fwd = np_softmax(-data, axis=axis)
            expected_bwd = np.zeros(shape)
            check_symbolic_forward(sym, [data], [expected_fwd], atol=atol, dtype=dtype)
            for req in ['null', 'add', 'write']:
                check_symbolic_backward(sym, [data], [np.ones(expected_fwd.shape)], [expected_bwd],
                                        rtol=rtol, atol=atol, grad_req=req, dtype=dtype)
            if dtype is not np.float16:
                check_numeric_gradient(sym, [data], rtol=rtol, atol=atol, dtype=dtype)


@with_seed()
def test_new_softmax():
    for ndim in range(1, 5):
        shape = np.random.randint(1, 5, size=ndim)
        axis = np.random.randint(-ndim, ndim)
        data = np.random.uniform(-2, 2, size=shape)
        sym = mx.sym.softmax(axis=axis)
        expected_fwd = np_softmax(data, axis=axis)
        expected_bwd = np.zeros(shape)
        check_symbolic_forward(sym, [data], [expected_fwd])
        for req in ['null', 'add', 'write']:
            check_symbolic_backward(sym, [data], [np.ones(expected_fwd.shape)], [expected_bwd],
                                    rtol=1e-2, atol=1e-3, grad_req=req)
        check_numeric_gradient(sym, [data], rtol=1e-2, atol=1e-3)


@with_seed()
def test_softmax_with_temperature():
    for ndim in range(1, 5):
        shape = np.random.randint(1, 5, size=ndim)
        data = np.random.uniform(-2, 2, size=shape)
        for temp in range(1, 11):
            sym = mx.sym.softmax(axis=0, temperature=temp)
            expected_fwd = np_softmax(data, axis=0, temperature=temp)
            expected_bwd = np.zeros(shape)
            check_symbolic_forward(sym, [data], [expected_fwd], rtol=0.05, atol=1e-3)
            check_symbolic_backward(sym, [data], [np.ones(shape)], [expected_bwd], rtol=0.05, atol=1e-3)
            check_numeric_gradient(sym, [data], rtol=0.05, atol=1e-3)

@with_seed()
def test_log_softmax():
    for ndim in range(1, 5):
        for _ in range(5):
            shape = np.random.randint(1, 5, size=ndim)
            axis = np.random.randint(0, ndim)
            data = np.random.uniform(-2, 2, size=shape)
            sym = mx.sym.log_softmax(axis=axis-ndim)
            check_symbolic_forward(sym, [data], [np.log(np_softmax(data, axis=axis)+1e-20)], rtol=1e-3, atol=1e-4)
            check_numeric_gradient(sym, [data], rtol=1e-1, atol=1e-2)

def test_softmax_with_large_inputs():
    def softmax_forward(input_data, true_output):
        data = mx.sym.Variable('data')
        out1 = data.softmax(axis=1)
        exec1 = out1.bind(default_context(), args={'data': input_data})
        exec1.forward()[0].wait_to_read()
        ndarr = exec1.outputs[0][0][0][0]
        assert_almost_equal(ndarr, true_output, rtol=1e-5, atol=1e-5)

    softmax_forward(mx.nd.array([[[[-1e30,-1e30]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[1e30,1e30]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[-3.4e38,-3.4e38]]]]), np.array([1.0,1.0]))
    softmax_forward(mx.nd.array([[[[3.4e38,3.4e38]]]]), np.array([1.0,1.0]))

@with_seed()
@with_environment('MXNET_SAFE_ACCUMULATION', '1')
def test_softmax_dtype():
    def check_dtypes_almost_equal(op_name,
                                  atol, rtol,
                                  grad_atol, grad_rtol,
                                  idtype, ref_dtype, odtype=None):
        op = getattr(mx.nd, op_name)
        input_data = mx.random.uniform(shape=(100, 500))
        dtype_input = input_data.astype(idtype)
        ref_input = input_data.astype(ref_dtype)
        dtype_input.attach_grad()
        ref_input.attach_grad()
        with mx.autograd.record():
            dtype_softmax = op(dtype_input, axis=-1, dtype=odtype)
            ref_softmax = op(ref_input, axis=-1, dtype=odtype)
        assert_almost_equal(dtype_softmax, ref_softmax, rtol=rtol, atol=atol)
        dtype_softmax.backward()
        ref_softmax.backward()
        assert_almost_equal(dtype_input.grad, ref_input.grad, rtol=grad_rtol, atol=grad_atol)

    check_dtypes_almost_equal('softmax', 1e-5, 1e-5, 1e-5, 1e-5, 'float16', 'float32')
    check_dtypes_almost_equal('softmax', 1e-5, 1e-5, 1e-5, 1e-5, 'float16', 'float32', 'float32')
    check_dtypes_almost_equal('softmax', 1e-5, 1e-5, 1e-5, 1e-5, 'float32', 'float64')
    check_dtypes_almost_equal('softmax', 1e-5, 1e-5, 1e-5, 1e-5, 'float32', 'float64', 'float64')
    check_dtypes_almost_equal('softmin', 1e-5, 1e-5, 1e-5, 1e-5, 'float16', 'float32')
    check_dtypes_almost_equal('softmin', 1e-5, 1e-5, 1e-5, 1e-5, 'float16', 'float32', 'float32')
    check_dtypes_almost_equal('softmin', 1e-5, 1e-5, 1e-5, 1e-5, 'float32', 'float64')
    check_dtypes_almost_equal('softmin', 1e-5, 1e-5, 1e-5, 1e-5, 'float32', 'float64', 'float64')
    check_dtypes_almost_equal('log_softmax', 1e-2, 1e-2, 1e-2, 1e-2,
                              'float16', 'float32')
    check_dtypes_almost_equal('log_softmax', 1e-2, 1e-2, 1e-2, 1e-2,
                              'float16', 'float32', 'float32')
    check_dtypes_almost_equal('log_softmax', 1e-3, 1e-3, 1e-3, 1e-3,
                              'float32', 'float64')
    check_dtypes_almost_equal('log_softmax', 1e-3, 1e-3, 1e-3, 1e-3,
                              'float32', 'float64', 'float64')


@with_seed()
def test_softmax_with_length():
    def np_softmax_with_length(data, length):
        res = np.zeros(data.shape)
        for i in range(length.shape[0]):
            for j in range(length.shape[1]):
                leng = int(length[i, j])
                res[i, 0:leng, j] = np_softmax(data[i, 0:leng, j])
        return res

    ndim = 3
    shape = rand_shape_nd(ndim, dim=10)
    len_shape = list(shape)
    del len_shape[1]
    len_shape = tuple(len_shape)
    for dtype in [np.float16, np.float32, np.float64]:
        mx_data = rand_ndarray(shape, dtype=dtype)
        np_data = mx_data.asnumpy()
        np_length = np.random.randint(1, shape[1] + 1, len_shape)
        mx_length = mx.nd.array(np_length, dtype=np.int32)
        np_out = np_softmax_with_length(np_data, np_length)
        data = mx.sym.Variable("data")
        length = mx.sym.Variable("length")
        mx_sym = mx.sym.softmax(data=data, length=length, use_length=True, axis=1)
        location = {"data": mx_data, "length": mx_length}
        rtol = 1e-2 if dtype == np.float16 else 1e-3
        atol = 1e-4 if dtype == np.float16 else 1e-5
        check_symbolic_forward(mx_sym, location, [np_out], rtol=rtol, atol=atol, dtype="asnumpy")
        check_symbolic_backward(mx_sym, location, [np.ones(shape, dtype=dtype)],
                                [np.zeros(shape), np.zeros(len_shape, dtype=np.int32)],
                                rtol=1e-2, atol=2e-3 if dtype == np.float16 else 1e-3, dtype="asnumpy")


@with_seed()
def test_pick():
    def test_pick_helper(index_type=np.int32):
        for mode in ['clip', 'wrap']:
            ndim = np.random.randint(1, 5)
            bshape = np.random.randint(1, 10, size=ndim)
            axis = np.random.randint(0, ndim)
            sshape = bshape.copy()
            sshape[axis] = 1
            data = np.random.uniform(-1, 1, size=bshape)

            if mode == 'wrap':
                index = np.random.randint(-2*bshape[axis], 2*bshape[axis], size=sshape)
            else:
                index = np.random.randint(0, bshape[axis], size=sshape)
            exp = []
            for i in range(ndim):
                if i == axis:
                    if mode == 'wrap':
                        exp.append(index % bshape[axis])
                    else:
                        exp.append(index)
                else:
                    ishape = [1 for _ in range(ndim)]
                    ishape[i] = bshape[i]
                    exp.append(np.arange(bshape[i]).reshape(ishape))
            expected = data[exp]
            data = mx.nd.array(data, dtype='float32')
            index = mx.nd.array(index, dtype=index_type)
            out = mx.nd.pick(data, index, axis=axis, keepdims=True, mode=mode)
            assert_almost_equal(out.asnumpy(), expected)

            data_holder = data
            index_holder = index
            data = mx.sym.Variable('data')
            index = mx.sym.Variable('index')
            sym = mx.sym.pick(data, index, axis=axis, keepdims=True, mode=mode)
            check_numeric_gradient(sym, [data_holder, index_holder], grad_nodes=['data'])

    test_pick_helper(np.int32)
    test_pick_helper(np.float32)


def check_ctc_loss(acts, labels, loss_truth, contrib=False):
    in_var = mx.sym.Variable('input')
    labels_var = mx.sym.Variable('labels')
    if contrib:
        ctc = mx.sym.contrib.ctc_loss(in_var, labels_var)
    else:
        ctc = mx.sym.ctc_loss(in_var, labels_var)
    acts_nd = mx.nd.array(acts, ctx=default_context())
    labels_nd = mx.nd.array(labels, ctx=default_context())
    exe = ctc.bind(ctx=default_context(), args=[acts_nd, labels_nd])
    # test forward with grad calc
    exe.forward(is_train=True)
    outTest = exe.outputs[0].copy()
    # test forward without grad calc
    exe.forward(is_train=False)
    outTrain = exe.outputs[0]
    # make sure losses calculated with both modes are the same
    assert_almost_equal(outTest, outTrain)

    # test against ground truth, if available
    if loss_truth is not None:
        assert_almost_equal(outTest, loss_truth)
    # test grad
    check_numeric_gradient(ctc, [acts, labels], grad_nodes=['input'], rtol=0.05, atol=1e-3)

@with_seed()
def test_ctc_loss():
    # Test 1: check that batches are same + check against Torch WarpCTC
    acts = np.array([
        [[1.2, 3.4, 1.2, -0.1, -2.34], [1.2, 3.4, 1.2, -0.1, -2.34]],
        [[0.1, 0.2, 0.3, 0.22, 0.123], [0.1, 0.2, 0.3, 0.22, 0.123]],
        [[-15, -14, -13, -12, -11], [-15, -14, -13, -12, -11]]],
                    dtype=np.float32)
    labels = np.array([[2, 3, 0], [2, 3, 0]])
    true_loss = np.array([4.04789, 4.04789], dtype=np.float32) # from Torch
    for contrib in [False, True]:
        check_ctc_loss(acts, labels, true_loss, contrib=contrib)


    # Test 2:
    acts2 = np.array([
        [[-5, -4, -3, -2, -1], [1.2, 3.4, 1.2, -0.1, -2.34]],
        [[-10, -9, -8, -7, -6], [0.1, 0.2, 0.3, 0.22, 0.123]],
        [[-15, -14, -13, -12, -11], [-15, -14.2, -13.5, -12.2, -11.22]]], dtype=np.float32)
    labels2 = np.array([[2, 3, 1], [2, 0, 0]], dtype=np.float32)
    true_loss = np.array([7.3557, 5.4091], dtype=np.float32) # from Torch
    for contrib in [False, True]:
        check_ctc_loss(acts2, labels2, true_loss, contrib=contrib)

    # Test 3: check use integer type as label
    labels3 = np.array([[2, 3, 1], [2, 0, 0]], dtype=np.int32)
    true_loss = np.array([7.3557, 5.4091], dtype=np.float32) # from Torch
    for contrib in [False, True]:
        check_ctc_loss(acts2, labels3, true_loss, contrib=contrib)

@with_seed()
def test_ctc_loss_with_large_classes():
    ctx = default_context()
    num_classes = 6000
    seq_len = 8
    batch_size = 2
    data = np.empty((num_classes, 0))
    for i in range(seq_len * batch_size) :
        row = np.roll(np.arange(num_classes, dtype=np.float32), i).reshape(num_classes, 1)
        data = np.append(data, row/13, axis=1)
    data = data.reshape(seq_len, batch_size, num_classes)
    label = np.array([
        [100, 200, 300, 400, 500, 0, 0, 0],
        [1000, 2000, 3000, 4000, 0, 5000, 0, 0]], dtype=np.int32)
    nd_data = mx.nd.array(data)
    nd_label = mx.nd.array(label)
    loss = mx.nd.ctc_loss(data=nd_data, label=nd_label)
    expected_loss = np.array([688.02826, 145.34462])
    assert_almost_equal(loss, expected_loss)

@with_seed()
def test_ctc_loss_grad():
    def check_ctc_loss_grad(blank_label, contrib=False): # from tf
        vocab_size = 5
        max_label_len = 5
        padding_mask = -1+ (blank_label=='first')

        targets_0 = [0, 1, 2, 1, 0]
        loss_log_prob_0 = -3.34211
        input_prob_matrix_0 = np.asarray(
            [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
             [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
             [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
             [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
             [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
            dtype=np.float32)
        gradient_log_prob_0 = np.asarray(
            [[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
             [0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
             [0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
             [0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
             [-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
            dtype=np.float32)

        targets_1 = [0, 1, 1, 0]
        loss_log_prob_1 = -5.42262
        input_prob_matrix_1 = np.asarray(
            [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
             [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
            dtype=np.float32)
        gradient_log_prob_1 = np.asarray(
            [[-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
             [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549],
             [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544],
             [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345],
             [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
            dtype=np.float32)

        inputs = [
            np.vstack(
                [input_prob_matrix_0[t, :], input_prob_matrix_1[t, :]])
            for t in range(5)
        ] + 2 * [np.nan * np.ones((2, vocab_size+1), np.float32)]
        inputs = np.log(np.asarray(inputs, dtype=np.float32))

        grad_truth = np.array([
            np.vstack(
                [gradient_log_prob_0[t, :], gradient_log_prob_1[t, :]])
            for t in range(5)
        ] + 2 * [np.zeros((2, vocab_size+1), np.float32)])

        if blank_label == 'first':
            inputs = np.roll(inputs, 1, axis=2)
            grad_truth = np.roll(grad_truth, 1, axis=2)

        labels = (np.asarray([x + [padding_mask]*(max_label_len-len(x))
                             for x in [targets_0, targets_1]])+(blank_label == 'first'))

        seq_lens = np.array([5, 5], dtype=np.int32)
        label_lens = np.array([5, 4], dtype=np.int32)
        loss_truth = np.array([-loss_log_prob_0, -loss_log_prob_1], np.float32)

        with default_context():
            data = mx.nd.array(inputs)
            label = mx.nd.array(labels)
            data.attach_grad()
            with mx.autograd.record():
                if contrib:
                    l = mx.contrib.ndarray.CTCLoss(data, label,
                                           use_data_lengths=True,
                                           use_label_lengths=True,
                                           data_lengths=mx.nd.array(seq_lens),
                                           label_lengths=mx.nd.array(label_lens),
                                           blank_label=blank_label)
                else:
                    l = mx.ndarray.CTCLoss(data, label,
                                           use_data_lengths=True,
                                           use_label_lengths=True,
                                           data_lengths=mx.nd.array(seq_lens),
                                           label_lengths=mx.nd.array(label_lens),
                                           blank_label=blank_label)
                l.backward()

            assert_almost_equal(l, loss_truth, atol=1e-5, rtol=1e-5)
            assert_almost_equal(data.grad, grad_truth, atol=1e-5, rtol=1e-5)

    for contrib in [False, True]:
        for label in ['first', 'last']:
            check_ctc_loss_grad(label, contrib=contrib)

@with_seed()
def test_quantization_op():
    min0 = mx.nd.array([0.0])
    max0 = mx.nd.array([1.0])
    a  = mx.nd.array([[0.1392, 0.5928], [0.6027, 0.8579]])
    qa, min1, max1 = mx.nd.contrib.quantize(a, min0, max0, out_type='int8')
    a_ = mx.nd.contrib.dequantize(qa, min1, max1, out_type='float32')

    qa_real = mx.nd.array([[18, 75], [77, 109]])
    a_real  = mx.nd.array([[0.14173228, 0.5905512], [0.6062992, 0.8582677]])
    print(a_.asnumpy())
    print(a_real.asnumpy())
    assert same(qa.asnumpy(), qa_real.asnumpy())
    assert_almost_equal(a_.asnumpy(),  a_real.asnumpy(), rtol=1e-2)

@with_seed()
def test_index_copy():
    x = mx.nd.zeros((5,3))
    t = mx.nd.array([[1,2,3],[4,5,6],[7,8,9]])
    index = mx.nd.array([0,4,2], dtype=np.int64)
    tensor = mx.nd.array([[1,2,3],[0,0,0],[7,8,9],[0,0,0],[4,5,6]])
    x_grad = mx.nd.array([[0,0,0],[1,1,1],[0,0,0],[1,1,1],[0,0,0]])
    t_grad = mx.nd.array([[1,1,1],[1,1,1],[1,1,1]])

    t.attach_grad()
    with mx.autograd.record():
        out = mx.nd.contrib.index_copy(x, index, t)
    out.backward()
    assert same(out.asnumpy(), tensor.asnumpy())
    assert same(t.grad.asnumpy(), t_grad.asnumpy())

    x.attach_grad()
    t.attach_grad()
    with mx.autograd.record():
        out = mx.nd.contrib.index_copy(x, index, t)
    out.backward()
    assert same(out.asnumpy(), tensor.asnumpy())
    assert same(x.grad.asnumpy(), x_grad.asnumpy())
    assert same(t.grad.asnumpy(), t_grad.asnumpy())


@with_seed()
def test_boolean_mask():
    data = mx.nd.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    index = mx.nd.array([0, 1, 0])
    data.attach_grad()
    with mx.autograd.record():
        out = mx.nd.contrib.boolean_mask(data, index)
    out.backward()
    data.grad.wait_to_read()
    expected = np.array([[4, 5, 6]])
    expected_grad = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    assert same(out.asnumpy(), expected)
    assert same(data.grad.asnumpy(), expected_grad)

    # test 0-size output
    mx.set_np_shape(True)
    data = mx.nd.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    index = mx.nd.array([0, 0, 0])
    data.attach_grad()
    with mx.autograd.record():
        out = mx.nd.contrib.boolean_mask(data, index)
    out.backward()
    data.grad.wait_to_read()
    expected = np.zeros((0, 3))
    expected_grad = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert same(out.asnumpy(), expected)
    assert same(data.grad.asnumpy(), expected_grad)
    mx.set_np_shape(False)

    # test gradient
    shape = (100, 30)
    a = mx.nd.random.randint(0, 100, shape=shape)
    a.attach_grad()
    bi = mx.nd.random.randint(0, 100, shape=shape[0:1]) > 50
    ci = mx.nd.random.randint(0, 100, shape=shape[0:1]) < 50
    mx_grad = mx.nd.zeros_like(a)
    mx.autograd.mark_variables([a], [mx_grad], grad_reqs='add')
    T = 3
    for _ in range(T):
        with mx.autograd.record():
            b = mx.nd.contrib.boolean_mask(a, bi)
            c = mx.nd.contrib.boolean_mask(a, ci)
            su = b.sum() + c.sum()
            su.backward()
    grad = (bi + ci).asnumpy().reshape((-1,) + (1,) * (len(shape)-1))
    grad = np.tile(grad, (1,) + shape[1:])
    # T times
    grad *= T
    assert_allclose(a.grad.asnumpy(), grad)
    a_np = a.asnumpy()
    assert same(b.asnumpy(), a_np[bi.asnumpy().astype('bool')])
    assert same(c.asnumpy(), a_np[ci.asnumpy().astype('bool')])


@with_seed()
def test_div_sqrt_dim():
    data_tmp = np.random.normal(0, 1, (5, 10, 8))
    data = mx.symbol.Variable('data')
    test = mx.sym.contrib.div_sqrt_dim(data)

    check_numeric_gradient(test, [data_tmp], numeric_eps=1E-2)
    check_symbolic_forward(test, [data_tmp], [data_tmp / np.sqrt(data_tmp.shape[-1])])


# helper function to identify inputs likely to fail check_numeric_gradient tol test
# due to finite difference method inaccuracies or function discontuities at the origin
def bad_input_finder(f, f_grad, dtype):
    eps = default_numeric_eps()[np.dtype(dtype)]
    rtol = default_rtols()[np.dtype(dtype)]
    def expected_relative_error(x):
        fd_gradient = (f(x+eps/2) - f(x-eps/2)) / eps
        return abs(fd_gradient/f_grad(x) - 1)
    def is_fd_problem_input(x):
        return abs(x) < eps/2 or expected_relative_error(x) > rtol
    return np.vectorize(is_fd_problem_input)

@with_seed()
def test_reciprocal_op():
    data_tmp = np.random.rand(3, 4).astype(np.float32) * 10 - 5

    # Avoid possible division by 0 errors and finite difference method
    # inaccuracies by replacing problem inputs with 1.0.
    is_bad_input = bad_input_finder(np.reciprocal,
                                    lambda x: -np.reciprocal(x)**2, np.float32)
    data_tmp[is_bad_input(data_tmp)] = 1.0
    data = mx.symbol.Variable('data')
    test = mx.sym.reciprocal(data)

    check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, [data_tmp], [np.reciprocal(data_tmp)])


@with_seed()
def test_cbrt_op():
    data_tmp = np.random.rand(3, 4).astype(np.float32) * 10 - 5

    # Avoid possible division by 0 errors and finite difference method
    # inaccuracies by replacing problem inputs with 1.0.
    is_bad_input = bad_input_finder(np.cbrt,
                                    lambda x: 1./(3 * np.cbrt(x)**2), np.float32)
    data_tmp[is_bad_input(data_tmp)] = 1.0
    data = mx.symbol.Variable('data')
    test = mx.sym.cbrt(data)
    check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, [data_tmp], [np.cbrt(data_tmp)])


@with_seed()
def test_rcbrt_op():
    data_tmp = np.random.rand(3, 4).astype(np.float32) * 10 - 5

    # Avoid possible division by 0 errors and finite difference method
    # inaccuracies by replacing problem inputs with 1.0.
    is_bad_input = bad_input_finder(lambda x: 1./np.cbrt(x),
                                    lambda x: -1./(3 * np.cbrt(x)**4), np.float32)
    data_tmp[is_bad_input(data_tmp)] = 1.0
    data = mx.symbol.Variable('data')
    test = mx.sym.rcbrt(data)

    check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, [data_tmp], [1/np.cbrt(data_tmp)])


@with_seed()
def test_custom_op():
    class Sqr(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            if in_data[0].stype == 'default':
                aux[0][:] = 1
                self.assign(out_data[0], req[0], in_data[0]*in_data[0])
            else:
                inp = in_data[0]
                csr_m = inp.data * inp.data
                out = mx.nd.sparse.csr_matrix((csr_m, inp.indices, inp.indptr), shape=inp.shape)
                self.assign(out_data[0], req[0], out)
                if (in_data[0].stype == 'csr'):
                    assert(isinstance(out_data[0], mx.nd.sparse.CSRNDArray))


        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            self.assign(in_grad[0], req[0], 2 * mx.nd.sparse.elemwise_mul(in_data[0], out_grad[0]))
            if in_data[0].stype == 'default':
                assert (aux[0].asnumpy() == 1).all()

    @mx.operator.register("sqr")
    class SqrProp(mx.operator.CustomOpProp):
        def __init__(self):
            super(SqrProp, self).__init__(need_top_grad=True)

        def list_arguments(self):
            return ['data']

        def list_outputs(self):
            return ['output']

        def list_auxiliary_states(self):
            return ['aux']

        def infer_shape(self, in_shape):
            return in_shape, [in_shape[0]], [in_shape[0]]

        def infer_type(self, in_type):
            return in_type, [in_type[0]], [in_type[0]]

        def infer_storage_type(self, in_stype):
            if in_stype[0] == 'default':
                return ['default'], ['default'], ['default']
            return ['csr'], ['csr'], ['csr']

        def infer_storage_type_backward(self, ograd_stype, in_stype,
                                        out_stype, igrad_stype, aux_stype):
            if in_stype[0] == 'default':
                return ['default'], ['default'], ['default'], ['default'], ['default']
            return ['default'], ['csr'], ['csr'], ['csr'], ['csr']

        def create_operator(self, ctx, shapes, dtypes):
            return Sqr()

    data = mx.symbol.Variable('data')
    aux = mx.symbol.Variable('aux')
    op = mx.symbol.Custom(data=data, aux=aux, name='sqr', op_type='sqr')
    x = mx.nd.array(np.random.uniform(-1, 1, size=(4, 10)))
    aux = mx.nd.zeros_like(x)
    check_numeric_gradient(op, [x], [aux])

    data = mx.symbol.cast(data, dtype='float64')
    op = mx.symbol.cast(op, dtype='float32')
    check_numeric_gradient(op, [x], [aux])

    data = mx.symbol.Variable('data', stype='csr')
    aux = mx.symbol.Variable('aux')
    op2 = mx.symbol.Custom(data=data, aux=aux, name='sqr', op_type='sqr')
    x = x.tostype('csr')
    aux = mx.nd.zeros_like(x)
    check_numeric_gradient(op2, [x], [aux], grad_stype_dict={"data": "csr"})

    x2 = mx.nd.array(np.random.uniform(-1, 1, size=(4, 10)))
    x2 = x2.tostype('csr')
    aux2 = mx.nd.zeros_like(x2)
    x2.attach_grad()
    with mx.autograd.record():
        output = mx.nd.Custom(x2, aux2, name='sqr', op_type='sqr')
        output.backward()
    expected_output = mx.nd.sparse.square(x2)
    expected_grad = 2 * x2
    rtol = 1e-4
    atol = 1e-6
    assert_almost_equal(output, expected_output, rtol=rtol, atol=atol)
    assert_almost_equal(x2.grad, expected_grad, rtol=rtol, atol=atol)


    # test for backward compatibility, i.e. the correctness of default implementation of
    # infer storage in custom operator
    class Mult(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            self.assign(out_data[0], req[0], in_data[0]*in_data[1])

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            self.assign(in_grad[0], req[0], in_data[1])
            self.assign(in_grad[1], req[1], in_data[0])

    @mx.operator.register("mult")
    class MultProp(mx.operator.CustomOpProp):
        def __init__(self):
            super(MultProp, self).__init__(need_top_grad=True)

        def list_arguments(self):
            return ['lhs', 'rhs']

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shape):
            return in_shape, [in_shape[0]], []

        def create_operator(self, ctx, shapes, dtypes):
            return Mult()

    lhs = mx.nd.array(np.random.uniform(-1, 1, size=(4, 10)))
    rhs = mx.nd.array(np.random.uniform(-1, 1, size=(4, 10)))
    lhs.attach_grad()
    rhs.attach_grad()
    with mx.autograd.record():
        y = mx.nd.Custom(lhs, rhs, name='mult', op_type='mult')
        y.backward()
    assert_almost_equal(rhs, lhs.grad, rtol=rtol, atol=atol)
    assert_almost_equal(lhs, rhs.grad, rtol=rtol, atol=atol)

    class MultNoGrad(mx.operator.CustomOp):
        def forward(self, is_train, req, in_data, out_data, aux):
            self.assign(out_data[0], req[0], in_data[0]*in_data[1])

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            self.assign(in_grad[0], req[0], in_data[1])
            self.assign(in_grad[1], req[1], in_data[0])

    @mx.operator.register("mult_no_grad")
    class MultNoGradProp(mx.operator.CustomOpProp):
        def __init__(self):
            super(MultNoGradProp, self).__init__(need_top_grad=False)

        def list_arguments(self):
            return ['lhs', 'rhs']

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shape):
            return in_shape, [in_shape[0]], []

        def create_operator(self, ctx, shapes, dtypes):
            return MultNoGrad()

        def infer_storage_type_backward(self, ograd_stype, in_stype, out_stype, igrad_stype, aux_stype):
            return ograd_stype, in_stype, out_stype, igrad_stype, aux_stype

    with mx.autograd.record():
        y2 = mx.nd.Custom(lhs, rhs, name="mult_no_grad", op_type="mult_no_grad")
        y2.backward()
    assert_almost_equal(rhs, lhs.grad, rtol=rtol, atol=atol)
    assert_almost_equal(lhs, rhs.grad, rtol=rtol, atol=atol)

    class NoInputOp(mx.operator.CustomOp):
        def __init__(self, length, depth):
            super(NoInputOp, self).__init__()
            self.output = np.ones(shape=(length, depth), dtype=np.float32)

        def forward(self, is_train, req, in_data, out_data, aux):
            self.assign(out_data[0], req[0], self.output)

        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            pass

    @mx.operator.register("no_input_op")
    class NoInputOpProp(mx.operator.CustomOpProp):
        def __init__(self, length, depth):
            super(NoInputOpProp, self).__init__()
            self.length = int(length)
            self.depth = int(depth)

        def list_arguments(self):
            return []

        def list_outputs(self):
            return ['output']

        def infer_shape(self, in_shape):
            return [], [(self.length, self.depth)], []

        def infer_type(self, in_type):
            return [], [np.float32], []

        def create_operator(self, ctx, shapes, dtypes):
            return NoInputOp(length=self.length, depth=self.depth)

    with mx.autograd.record():
        x = mx.nd.Custom(length=10, depth=10, op_type="no_input_op")
    assert_almost_equal(x, np.ones(shape=(10, 10), dtype=np.float32))

@unittest.skip("Flaky test, tracked at https://github.com/apache/incubator-mxnet/issues/17467")
@with_seed()
def test_custom_op_fork():
    # test custom operator fork
    # see https://github.com/apache/incubator-mxnet/issues/14396
    class AdditionOP(mx.operator.CustomOp):
        def __init__(self):
            super(AdditionOP, self).__init__()
        def forward(self, is_train, req, in_data, out_data, aux):
            out_data[0][:] = in_data[0] + in_data[1]
        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            in_grad[0][:] = out_grad[0]
            in_grad[1][:] = out_grad[0]

    @mx.operator.register("AdditionOP")
    class AdditionOPProp(mx.operator.CustomOpProp):
        def __init__(self):
            super(AdditionOPProp, self).__init__()
        def list_arguments(self):
            return ['a', 'b']
        def list_outputs(self):
            return ['output']
        def infer_shape(self, in_shape):
            return in_shape, [in_shape[0]]
        def create_operator(self, ctx, shapes, dtypes):
            return AdditionOP()

    if not sys.platform.startswith('win'):  # no fork in windows
        def custom_add():
            a = mx.nd.array([1, 2, 3])
            b = mx.nd.array([4, 5, 6])
            c = mx.nd.Custom(a, b, op_type='AdditionOP')
            assert_almost_equal((a + b).asnumpy(), c.asnumpy())

        custom_add()
        from multiprocessing import Process
        p = Process(target=custom_add)
        p.daemon = True
        p.start()
        p.join(5)
        assert not p.is_alive() and p.exitcode == 0


def _build_dot_custom(fun_forward, name):
    class Dot(mx.operator.CustomOp):
        def __init__(self):
            super(Dot, self).__init__()
        def forward(self, is_train, req, in_data, out_data, aux):
            fun_forward(in_data, out_data)
        def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
            pass

    @mx.operator.register(name)
    class DotProp(mx.operator.CustomOpProp):
        def __init__(self):
            super(DotProp, self).__init__()
        def list_arguments(self):
            return ['a', 'b']
        def list_outputs(self):
            return ['output']
        def infer_shape(self, in_shape):
            return in_shape, [(in_shape[0][0], in_shape[1][1])]
        def create_operator(self, ctx, shapes, dtypes):
            return Dot()

@with_seed()
def test_custom_op_exc():
    # test except handling
    # see https://github.com/apache/incubator-mxnet/pull/14693
    # 1. error in python code
    def custom_exc1():
        def f(in_data, out_data):
            assert False
            out_data[0][:] = mx.nd.dot(in_data[0], in_data[1])
        _build_dot_custom(f, 'Dot1')
        a = mx.nd.zeros((4, 1))
        b = mx.nd.zeros((1, 4))
        c = mx.nd.Custom(a, b, op_type='Dot1')
        c.wait_to_read()
    assert_raises(MXNetError, custom_exc1)

    # 2. error in pushing operator to engine
    def custom_exc2():
        def f(in_data, out_data):
            out_data[0][:] = mx.nd.dot(in_data[0], in_data[1])
        _build_dot_custom(f, 'Dot2')
        a = mx.nd.zeros((4, 2))
        b = mx.nd.zeros((1, 4))
        # trigger error by invalid input shapes of operands
        c = mx.nd.Custom(a, b, op_type='Dot2')
        c.wait_to_read()
    assert_raises(MXNetError, custom_exc2)

    # 3. error in real execution
    if default_context().device_type == 'cpu':
        def custom_exc3():
            def f(in_data, out_data):
                dot = mx.nd.dot(in_data[0], in_data[1])
                # input to Cholesky factorization should be
                # symmetric positive-definite, error will be
                # triggered in op execution on cpu
                out_data[0][:] = mx.nd.linalg.potrf(dot)
                out_data[0].wait_to_read()
            _build_dot_custom(f, 'Dot3')
            a = mx.nd.zeros((2, 1))
            b = mx.nd.zeros((1, 2))
            c = mx.nd.Custom(a, b, op_type='Dot3')
            c.wait_to_read()
        assert_raises(MXNetError, custom_exc3)

        def custom_exc4():
            def f(in_data, out_data):
                dot = mx.nd.dot(in_data[0], in_data[1])
                # input to Cholesky factorization should be
                # symmetric positive-definite, error will be
                # triggered in op execution on cpu
                out_data[0][:] = mx.nd.linalg.potrf(dot)
            _build_dot_custom(f, 'Dot4')
            a = mx.nd.zeros((2, 1))
            b = mx.nd.zeros((1, 2))
            c = mx.nd.Custom(a, b, op_type='Dot4')
            c.wait_to_read()
        assert_raises(MXNetError, custom_exc4)


@with_seed()
def test_psroipooling():
    for num_rois in [1, 2]:
        for num_classes, num_group in itertools.product([2, 3], [2, 3]):
            for image_height, image_width in itertools.product([168, 224], [168, 224]):
                for grad_nodes in [['im_data']]:
                    spatial_scale = 0.0625
                    feat_height = np.int(image_height * spatial_scale)
                    feat_width = np.int(image_width * spatial_scale)
                    im_data = np.random.rand(1, num_classes*num_group*num_group, feat_height, feat_width)
                    rois_data = np.zeros([num_rois, 5])
                    rois_data[:, [1,3]] = np.sort(np.random.rand(num_rois, 2)*(image_width-1))
                    rois_data[:, [2,4]] = np.sort(np.random.rand(num_rois, 2)*(image_height-1))

                    im_data_var = mx.symbol.Variable(name="im_data")
                    rois_data_var = mx.symbol.Variable(name="rois_data")
                    op = mx.sym.contrib.PSROIPooling(data=im_data_var, rois=rois_data_var, spatial_scale=spatial_scale,
                                                     group_size=num_group, pooled_size=num_group,
                                                     output_dim=num_classes, name='test_op')
                    rtol, atol = 1e-2, 1e-3
                    check_numeric_gradient(op, [im_data, rois_data], rtol=rtol, atol=atol,
                                           grad_nodes=grad_nodes)


@with_seed()
def test_psroipooling_with_type():
    arg_params = {
        'psroipool_rois': np.array([[0, 10, 22, 161, 173], [0, 20, 15, 154, 160]])}

    # plain psroipooling
    sym = mx.sym.contrib.PSROIPooling(spatial_scale=0.0625, output_dim=2, pooled_size=3, name='psroipool')
    ctx_list = [{'ctx': mx.cpu(0),
                 'psroipool_data': (1, 18, 14, 14),
                 'psroipool_rois': (2, 5),
                 'type_dict': {'psroipool_data': np.float64, 'psroipool_rois': np.float64}},
                {'ctx': mx.cpu(0),
                 'psroipool_data': (1, 18, 14, 14),
                 'psroipool_rois': (2, 5),
                 'type_dict': {'psroipool_data': np.float32, 'psroipool_rois': np.float32}},
                {'ctx': mx.cpu(0),
                 'psroipool_data': (1, 18, 14, 14),
                 'psroipool_rois': (2, 5),
                 'type_dict': {'psroipool_data': np.float16, 'psroipool_rois': np.float16}},
                ]

    check_consistency(sym, ctx_list, grad_req={'psroipool_data': 'write',
                                               'psroipool_rois': 'null'}, arg_params=arg_params)


@with_seed()
def test_deformable_convolution():
    for num_batch in [1, 2]:
        for num_channel_data, num_deformable_group in itertools.product([4, 8], [1, 2]):
            for input_height, input_width in itertools.product([5, 6], [5, 6]):
                for dilate in [(1, 1), (2, 2)]:
                    for grad_nodes in [['im_data'], ['offset_data'], ['weight']]:
                        output_height = input_height
                        output_width = input_width
                        im_data = np.random.rand(num_batch, num_channel_data, input_height, input_width)
                        offset_data = \
                            np.random.rand(num_batch, num_deformable_group * 3 * 3 * 2, output_height, output_width)\
                            * 0.8 + 0.1

                        weight = np.random.normal(0, 0.001, (num_channel_data, num_channel_data, 3, 3))
                        bias = np.zeros(num_channel_data)

                        im_data_var = mx.symbol.Variable(name="im_data")
                        offset_data_var = mx.symbol.Variable(name="offset_data")
                        weight_var = mx.symbol.Variable(name="weight")
                        bias_var = mx.symbol.Variable(name="bias")
                        op = mx.sym.contrib.DeformableConvolution(name='test_op', data=im_data_var,
                                                                  offset=offset_data_var,
                                                                  weight=weight_var, bias=bias_var,
                                                                  num_filter=num_channel_data, pad=dilate,
                                                                  kernel=(3, 3), stride=(1, 1), dilate=dilate,
                                                                  num_deformable_group=num_deformable_group)
                        if grad_nodes[0] == 'offset_data':
                            # wider tolerance needed for coordinate differential
                            rtol, atol = 1.0, 1e-2
                        else:
                            rtol, atol = 0.05, 1e-3
                        # By now we only have gpu implementation
                        if default_context().device_type == 'gpu':
                            check_numeric_gradient(op, [im_data, offset_data, weight, bias], rtol=rtol, atol=atol,
                                                   grad_nodes=grad_nodes, ctx=mx.gpu(0), numeric_eps=1.0/64)


def _validate_sample_location(input_rois, input_offset, spatial_scale, pooled_w, pooled_h, sample_per_part, part_size, output_dim, num_classes, trans_std, feat_h, feat_w):
    num_rois = input_rois.shape[0]
    output_offset = input_offset.copy()
    # simulate deformable psroipooling forward function
    for roi_idx in range(num_rois):
        sub_rois = input_rois[roi_idx, :].astype(np.float32)
        img_idx, x0, y0, x1, y1 = int(sub_rois[0]), sub_rois[1], sub_rois[2], sub_rois[3], sub_rois[4]
        roi_start_w = round(x0) * spatial_scale - 0.5
        roi_start_h = round(y0) * spatial_scale - 0.5
        roi_end_w = round(x1 + 1) * spatial_scale - 0.5
        roi_end_h = round(y1 + 1) * spatial_scale - 0.5
        roi_w, roi_h = roi_end_w - roi_start_w, roi_end_h - roi_start_h
        bin_size_w, bin_size_h = roi_w / pooled_w, roi_h / pooled_h
        sub_bin_size_w, sub_bin_size_h = bin_size_w / sample_per_part, bin_size_h / sample_per_part
        for c_top in range(output_dim):
            channel_each_cls = output_dim / num_classes
            class_id = int(c_top / channel_each_cls)
            for ph in range(pooled_h):
                for pw in range(pooled_w):
                    part_h = int(math.floor(float(ph) / pooled_h * part_size))
                    part_w = int(math.floor(float(pw) / pooled_w * part_size))
                    trans_x = input_offset[roi_idx, class_id * 2, part_h, part_w] * trans_std
                    trans_y = input_offset[roi_idx, class_id * 2 + 1, part_h, part_w] * trans_std
                    bin_h_start, bin_w_start = ph * bin_size_h + roi_start_h, pw * bin_size_w + roi_start_w

                    need_check = True
                    while need_check:
                        pass_check = True
                        for ih in range(sample_per_part):
                            for iw in range(sample_per_part):
                                h = bin_h_start + trans_y * roi_h + ih * sub_bin_size_h
                                w = bin_w_start + trans_x * roi_w + iw * sub_bin_size_w

                                if w < -0.5 or w > feat_w - 0.5 or h < -0.5 or h > feat_h - 0.5:
                                    continue

                                w = min(max(w, 0.1), feat_w - 1.1)
                                h = min(max(h, 0.1), feat_h - 1.1)
                                # if the following condiiton holds, the sampling location is not differentiable
                                # therefore we need to re-do the sampling process
                                if h - math.floor(h) < 1e-3 or math.ceil(h) - h < 1e-3 or w - math.floor(w) < 1e-3 or math.ceil(w) - w < 1e-3:
                                    trans_x, trans_y = random.random() * trans_std, random.random() * trans_std
                                    pass_check = False
                                    break
                            if not pass_check:
                                break
                        if pass_check:
                            output_offset[roi_idx, class_id * 2 + 1, part_h, part_w] = trans_y / trans_std
                            output_offset[roi_idx, class_id * 2, part_h, part_w] = trans_x / trans_std
                            need_check = False

    return output_offset

@unittest.skip("Flaky test, tracked at https://github.com/apache/incubator-mxnet/issues/11713")
@with_seed()
def test_deformable_psroipooling():
    sample_per_part = 4
    trans_std = 0.1
    for num_rois in [1, 2]:
        for num_classes, num_group in itertools.product([2, 3], [2, 3]):
            for image_height, image_width in itertools.product([160, 224], [160, 224]):
                for grad_nodes in [['im_data'], ['offset_data']]:
                    spatial_scale = 0.0625
                    stride = int(1 / spatial_scale)
                    feat_height = np.int(image_height * spatial_scale)
                    feat_width = np.int(image_width * spatial_scale)
                    im_data = np.random.rand(1, num_classes*num_group*num_group, feat_height, feat_width)
                    rois_data = np.zeros([num_rois, 5])
                    rois_data[:, [1,3]] = np.sort(np.random.rand(num_rois, 2)*(image_width-1 - 2 * stride)) + stride
                    rois_data[:, [2,4]] = np.sort(np.random.rand(num_rois, 2)*(image_height-1 - 2 * stride)) + stride
                    offset_data = np.random.rand(num_rois, 2*num_classes, num_group, num_group)
                    # at certain points, the bilinear interpolation function may be non-differentiable
                    # to avoid this, we check whether the input locates on the valid points
                    offset_data = _validate_sample_location(rois_data, offset_data, spatial_scale, num_group, num_group,
                                                            sample_per_part, num_group, num_classes, num_classes, trans_std, feat_height, feat_width)
                    im_data_var = mx.symbol.Variable(name="im_data")
                    rois_data_var = mx.symbol.Variable(name="rois_data")
                    offset_data_var = mx.symbol.Variable(name="offset_data")
                    op = mx.sym.contrib.DeformablePSROIPooling(data=im_data_var, rois=rois_data_var,
                                                               trans=offset_data_var, spatial_scale=spatial_scale,
                                                               sample_per_part=4, group_size=num_group,
                                                               pooled_size=num_group, output_dim=num_classes,
                                                               trans_std=0.1, no_trans=False, name='test_op')
                    rtol, atol = 1e-2, 1e-3
                    # By now we only have gpu implementation
                    if default_context().device_type == 'gpu':
                        check_numeric_gradient(op, [im_data, rois_data, offset_data], rtol=rtol, atol=atol,
                                               grad_nodes=grad_nodes, ctx=mx.gpu(0))


def _gemm_test_helper(dtype, grad_check, rtol_fw = None, atol_fw = None,
                                         rtol_bw = None, atol_bw = None, num_eps = None):
    def np_random_data(shape, dtype=np.float32):
        return np.random.uniform(low=-0.5,
                                 high=0.5, size=shape).astype(dtype)

    data1 = mx.symbol.Variable('data1')
    data2 = mx.symbol.Variable('data2')
    data3 = mx.symbol.Variable('data3')

    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol_bw,
                               atol=atol_bw, dtype=dtype)
    rep_3x = lambda a, m, n :\
        np.reshape(np.tile(np.array(a).flatten(), 3), (3, 1, m, n))

    shape1 = (2, 3)
    shape2 = (3, 2)
    shape3 = (3, 3)
    shape4 = (2, 2)
    data_in1 = np_random_data(shape1, dtype)
    data_in2 = np_random_data(shape2, dtype)
    data_in3 = np_random_data(shape3, dtype)
    data_in4 = np_random_data(shape4, dtype)
    # Check all transpositions of gemm operator.
    data_in1_t = np.transpose(data_in1)
    data_in2_t = np.transpose(data_in2)
    res_gemm = 4. * np.dot(data_in1, data_in2) + 7. * data_in4
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.)
    check_fw(test_gemm, [data_in1, data_in2, data_in4], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in2, data_in4])
    res_gemm = 4. * np.dot(data_in1_t, data_in2_t) + 7. * data_in3
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.,
                                   transpose_a=True, transpose_b=True)
    check_fw(test_gemm, [data_in1, data_in2, data_in3], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in2, data_in3])
    res_gemm = 4. * np.dot(data_in1_t, data_in1) + 7. * data_in3
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.,
                                   transpose_a=True)
    check_fw(test_gemm, [data_in1, data_in1, data_in3], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in1, data_in3])
    res_gemm = 4. * np.dot(data_in1, data_in1_t) + 7. * data_in4
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.,
                                   transpose_b=True)
    check_fw(test_gemm, [data_in1, data_in1, data_in4], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in1, data_in4])

    # Check batch of gemm.
    a = rep_3x(data_in1, 2, 3)
    b = rep_3x(data_in2, 3, 2)
    c = rep_3x(data_in4, 2, 2)
    r = 4. * np.dot(data_in1, data_in2) + 7. * data_in4
    r = rep_3x(r, 2, 2)
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7.)
    check_fw(test_gemm, [a, b, c], [r])
    if grad_check == 1:
        check_grad(test_gemm, [a, b, c])
    # Check for different axis that describes matrix rows.
    a2 = np.copy(np.swapaxes(a, 0, 2))
    b2 = np.copy(np.swapaxes(b, 0, 2))
    c2 = np.copy(np.swapaxes(c, 0, 2))
    r2 = np.copy(np.swapaxes(r, 0, 2))
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7., axis = 0)
    check_fw(test_gemm, [a2, b2, c2], [r2])
    if grad_check == 1:
        check_grad(test_gemm, [a2, b2, c2])
    a2 = np.copy(np.swapaxes(a, 1, 2))
    b2 = np.copy(np.swapaxes(b, 1, 2))
    c2 = np.copy(np.swapaxes(c, 1, 2))
    r2 = np.copy(np.swapaxes(r, 1, 2))
    test_gemm = mx.sym.linalg.gemm(data1, data2, data3, alpha=4., beta=7., axis = -3)
    check_fw(test_gemm, [a2, b2, c2], [r2])
    if grad_check == 1:
        check_grad(test_gemm, [a2, b2, c2])

    # Check gemm2 operator same way as gemm.
    res_gemm = 4. * np.dot(data_in1, data_in2)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4.)
    check_fw(test_gemm, [data_in1, data_in2], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in2])
    res_gemm = 4. * np.dot(data_in1_t, data_in2_t)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4., transpose_a=True,
                                    transpose_b=True)
    check_fw(test_gemm, [data_in1, data_in2], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in2])
    res_gemm = 4. * np.dot(data_in1_t, data_in1)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4., transpose_a=True)
    check_fw(test_gemm, [data_in1, data_in1], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in1])
    res_gemm = 4. * np.dot(data_in1, data_in1_t)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4., transpose_b=True)
    check_fw(test_gemm, [data_in1, data_in1], [res_gemm])
    if grad_check == 1:
        check_grad(test_gemm, [data_in1, data_in1])

    # Check batch of gemm2.
    a = rep_3x(data_in1, 2, 3)
    b = rep_3x(data_in2, 3, 2)
    r = rep_3x(4. * np.dot(data_in1, data_in2), 2, 2)
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4.)
    check_fw(test_gemm, [a, b], [r])
    if grad_check == 1:
        check_grad(test_gemm, [a, b])
    a2 = np.copy(np.swapaxes(a, 0, 2))
    b2 = np.copy(np.swapaxes(b, 0, 2))
    r2 = np.copy(np.swapaxes(r, 0, 2))
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4., axis = 0)
    check_fw(test_gemm, [a2, b2], [r2])
    if grad_check == 1:
        check_grad(test_gemm, [a2, b2])
    a2 = np.copy(np.swapaxes(a, 1, 2))
    b2 = np.copy(np.swapaxes(b, 1, 2))
    r2 = np.copy(np.swapaxes(r, 1, 2))
    test_gemm = mx.sym.linalg.gemm2(data1, data2, alpha=4., axis = -3)
    check_fw(test_gemm, [a2, b2], [r2])
    if grad_check == 1:
        check_grad(test_gemm, [a2, b2])

# Test gemm separately from other la-operators.
@with_seed()
def test_gemm():
    _gemm_test_helper(np.float64, True)
    with environment('MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION', '0'):
        _gemm_test_helper(np.float32, True)
    if default_context().device_type == 'gpu':
        with environment('MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION', '1'):
            _gemm_test_helper(np.float32, True)


# Helper functions for test_laop

def _make_symm_symbol(a, ndims):
    assert ndims >= 2
    tr_shape = list(range(ndims))
    tr_shape[-1] = ndims-2
    tr_shape[-2] = ndims-1
    tr_shape = tuple(tr_shape)
    return 0.5 * (a + mx.sym.transpose(a, axes=tr_shape))

def _make_triangle_symm(a, ndims, m, lower, dtype=np.float32):
    assert ndims >= 2
    # The last two dimensions must both be m
    # Create mask for lower triangle and diagonal
    index = mx.sym.arange(start=0, stop=m, step=1, dtype=np.int32)
    lt_mask = mx.sym.one_hot(index, depth=m, dtype=dtype)
    for j in range(1, m):
        part1 = mx.sym.zeros(shape=(j, m), dtype=dtype)
        index = mx.sym.arange(start=0, stop=m-j, step=1, dtype=np.int32)
        part2 = mx.sym.one_hot(index, depth=m, dtype=dtype)
        lt_mask = lt_mask + mx.sym.concat(*[part1, part2], dim=0)
    if not lower:
        lt_mask = mx.sym.reshape(lt_mask, shape=(m, m))
        lt_mask = mx.sym.transpose(lt_mask, axes=(1, 0))
    shp = tuple([1]*(ndims-2) + [m, m])
    lt_mask = mx.sym.reshape(lt_mask, shape=shp)
    return mx.sym.broadcast_mul(a, lt_mask)

# @ankkhedia: Getting rid of fixed seed as flakiness could not be reproduced
# tracked at https://github.com/apache/incubator-mxnet/issues/11718
@with_seed()
def test_laop():
    dtype = np.float64
    rtol_fw = 1e-7
    atol_fw = 1e-9
    num_eps = 2e-6
    rtol_bw = 1e-5
    atol_bw = 1e-5
    # enable numerical checking of gradients
    grad_check = 1

    data1 = mx.symbol.Variable('data1')
    data2 = mx.symbol.Variable('data2')

    rep_3x = lambda a, m, n :\
        np.reshape(np.tile(np.array(a).flatten(), 3), (3, 1, m, n))

    def check_fw_grad(sym, location, expected):
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)
        if grad_check == 1:
            check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol_bw,
                                   atol=atol_bw, dtype=dtype)

    matrix = np.array([[9., 3., -6., 12.],
                       [3., 26., -7., -11.],
                       [-6., -7., 9., 7.],
                       [12., -11., 7., 65.]])
    trian  = np.array([[3., 0., 0., 0.],
                       [1., 5., 0., 0.],
                       [-2., -1., 2., 0.],
                       [4., -3., 6., 2.]])
    pow    = np.array([[2., 1., 1., 1.],
                       [1., 4., 1., 1.],
                       [1., 1., 8., 1.],
                       [1., 1., 1., 16.]])
    inv    = np.array([[8.95/3., 0.05/3., 2.65, -2.5/3.],
                       [0.05/3., 0.05, 0.05, 0.],
                       [2.65, 0.05, 2.5, -0.75],
                       [-2.5/3., 0., -0.75, 0.25]])
    ident  = np.eye(4)
    shape = (4, 4, 1, 1)
    ones = mx.nd.ones(shape).asnumpy()

    for lower in [True, False]:
        upper = not lower

        # Tests with trivial 1x1 matrices.
        data_in = np.random.uniform(1, 10, shape)
        # test potrf
        # Note: Have to symmetrize input, for gradient test to work
        res_potrf = np.sqrt(data_in)
        test_potrf = mx.sym.linalg.potrf(data1, lower=lower)
        check_fw_grad(test_potrf, [data_in], [res_potrf])
        # test potri
        res_potri = np.divide(ones, data_in * data_in)
        test_potri = mx.sym.linalg.potri(data1, lower=lower)
        check_fw_grad(test_potri, [data_in], [res_potri])
        # test trsm
        trian_in = data_in * 7.
        test_trsm = mx.sym.linalg.trsm(data1, data2, alpha=7., lower=lower)
        check_fw_grad(test_trsm, [trian_in, data_in], [ones])
        # test trmm
        trian_in = np.divide(ones, trian_in)
        test_trmm = mx.sym.linalg.trmm(data1, data2, alpha=7., transpose=True,
                                       rightside=True, lower=lower)
        check_fw_grad(test_trmm, [trian_in, data_in], [ones])
        # test sumlogdiag
        res_sumlogdiag = np.reshape(np.log(data_in), (4, 4))
        test_sumlogdiag = mx.sym.linalg.sumlogdiag(data1)
        check_fw_grad(test_sumlogdiag, [data_in], [res_sumlogdiag])

        # more elaborate example of Cholesky factorization
        low_trian = trian
        if upper:
            trian = np.transpose(trian)

        # test potrf
        test_potrf = mx.sym.linalg.potrf(_make_symm_symbol(data1, ndims=4), lower=lower)
        a = rep_3x(matrix, 4, 4)
        r = rep_3x(trian, 4, 4)
        check_fw_grad(test_potrf, [a], [r])

        #test potri
        data1_ltri = _make_triangle_symm(
            data1, ndims=4, m=4, lower=lower, dtype=dtype)
        test_potri = mx.sym.linalg.potri(data1_ltri, lower=lower)
        a = rep_3x(trian, 4, 4)
        r = rep_3x(inv, 4, 4)
        check_fw_grad(test_potri, [a], [r])

        # test trsm
        test_trsm = mx.sym.linalg.trsm(data1_ltri, data2, alpha=7., transpose=upper, lower=lower)
        b = rep_3x(matrix, 4, 4)
        r = rep_3x(7. * np.transpose(low_trian), 4, 4)
        check_fw_grad(test_trsm, [a, b], [r])

        test_trsm2 = mx.sym.linalg.trsm(
            data1_ltri, data2, alpha=-2., rightside=True, transpose=lower, lower=lower)
        r = rep_3x(-2. * low_trian, 4, 4)
        check_fw_grad(test_trsm2, [a, b], [r])

        test_trsm3 = mx.sym.linalg.trsm(
            data1_ltri, data2, alpha=0.5, transpose=lower, lower=lower)
        b = rep_3x(np.transpose(low_trian), 4, 4)
        r = rep_3x(0.5 * ident, 4, 4)
        check_fw_grad(test_trsm3, [a, b], [r])

        test_trsm4 = mx.sym.linalg.trsm(
            data1_ltri, data2, alpha=-0.5, rightside=True, transpose=upper, lower=lower)
        b = rep_3x(low_trian, 4, 4)
        r = rep_3x(-0.5 * ident, 4, 4)
        check_fw_grad(test_trsm4, [a, b], [r])

        # test trmm
        test_trmm = mx.sym.linalg.trmm(
            data1_ltri, data2, alpha=7., transpose=True, rightside=True, lower=lower)
        a = [a, rep_3x(matrix, 4, 4)]
        r = rep_3x(7. * np.dot(matrix, trian.T), 4, 4)
        check_fw_grad(test_trmm, a, [r])

        test_trmm2 = mx.sym.linalg.trmm(data1_ltri, data2, alpha=-2., lower=lower)
        r = rep_3x(-2. * np.dot(trian, matrix), 4, 4)
        check_fw_grad(test_trmm2, a, [r])

        test_trmm3 = mx.sym.linalg.trmm(data1_ltri, data2, rightside=True, lower=lower)
        r = rep_3x(np.dot(matrix, trian), 4, 4)
        check_fw_grad(test_trmm3, a, [r])

        test_trmm4 = mx.sym.linalg.trmm(
            data1_ltri, data2, alpha=1.2, transpose=True, lower=lower)
        r = rep_3x(1.2 * np.dot(trian.T, matrix), 4, 4)
        check_fw_grad(test_trmm4, a, [r])

        # test sumlogdiag
        r = np.reshape(np.tile(10. * np.log(np.array([2.])), 3), (3,))
        check_fw_grad(test_sumlogdiag, [rep_3x(pow, 4, 4)], [r])


# Tests for operators linalg.syrk, linalg.gelqf

def _gelqf_combined_symbol(a):
    q, l = mx.sym.linalg.gelqf(a)
    q_qt = mx.sym.linalg.syrk(q, transpose=False, alpha=1., name='Q_times_Qt')
    l_q = mx.sym.linalg.trmm(l, q, alpha=1., name='L_times_Q')
    return mx.sym.Group([q_qt, l_q])

# NOTE: If we leave the unused output dangling, things break if dtype=np.float64. Namely, the
# backward gradient for the unused output is of dtype np.float32 then.
# ==> Very annoying!
def _gelqf_first_output(a):
    q, l = mx.sym.linalg.gelqf(a)
    bogus_scal = mx.sym.sum(mx.sym.BlockGrad(l), axis=(), keepdims=True) * 0.0
    return mx.sym.broadcast_add(q, bogus_scal)

def _gelqf_second_output(a):
    q, l = mx.sym.linalg.gelqf(a)
    bogus_scal = mx.sym.sum(mx.sym.BlockGrad(q), axis=(), keepdims=True) * 0.0
    return mx.sym.broadcast_add(l, bogus_scal)

def _syevd_combined_symbol(a):
    u, lam = mx.sym.linalg.syevd(a)
    u_ut = mx.sym.linalg.syrk(u, transpose=False, alpha=1., name='U_times_Ut')
    lam_u = mx.sym.broadcast_mul(mx.sym.reshape(lam, shape=(-2, 1)), u)
    ut_lam_u = mx.sym.linalg.gemm2(u, lam_u, alpha=1., transpose_a=True,
                                   transpose_b=False, name='Ut_L_U')
    return mx.sym.Group([u_ut, ut_lam_u])

@with_seed()
def test_laop_2():
    dtype = np.float64
    rtol_fw = 1e-7
    atol_fw = 1e-9
    num_eps = 1e-6
    rtol_bw = 1e-5
    atol_bw = 1e-6
    # enable numerical checking of gradients
    grad_check = 1

    data1 = mx.symbol.Variable('data1')

    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol_bw,
                               atol=atol_bw, dtype=dtype)
    rep_3x = lambda a, m, n :\
        np.reshape(np.tile(np.array(a).flatten(), 3), (3, 1, m, n))

    # Tests for linalg.syrk
    mnalpha_lst = [(2, 3, 1.), (5, 3, -2.), (1, 6, 5.), (3, 3, 0.5), (4, 1, 10.), (1, 1, 1.)]
    for m, n, alpha in mnalpha_lst:
        #print('syrk: m={}, n={}, alpha={}'.format(m, n, alpha))
        data_in1 = np.random.uniform(1, 10, (m, n))
        res_syrk1 = alpha * np.dot(data_in1, data_in1.T)
        test_syrk1 = mx.sym.linalg.syrk(data1, transpose=False, alpha=alpha)
        check_fw(test_syrk1, [data_in1], [res_syrk1])
        if grad_check == 1:
            check_grad(test_syrk1, [data_in1])
        res_syrk2 = alpha * np.dot(data_in1.T, data_in1)
        test_syrk2 = mx.sym.linalg.syrk(data1, transpose=True, alpha=alpha)
        check_fw(test_syrk2, [data_in1], [res_syrk2])
        if grad_check == 1:
            check_grad(test_syrk2, [data_in1])
        # Batch mode (3x the same thing)
        a_batch = rep_3x(data_in1, m, n)
        r1_batch = rep_3x(res_syrk1, m, m)
        check_fw(test_syrk1, [a_batch], [r1_batch])
        if grad_check == 1:
            check_grad(test_syrk1, [a_batch])
        r2_batch = rep_3x(res_syrk2, n, n)
        check_fw(test_syrk2, [a_batch], [r2_batch])
        if grad_check == 1:
            check_grad(test_syrk2, [a_batch])

    # Tests for linalg.gelqf
    # Currently disabled on GPU as they need cuda8
    # and MxNet builds use cuda 7.5
    if not (default_context() == mx.cpu()):
        return

    test_gelqf2 = _gelqf_combined_symbol(data1)  # Outputs (dot(Q, Q.T), dot(L, Q))
    test_gelqf_q = _gelqf_first_output(data1)  # Output Q (L is not dangling)
    test_gelqf_l = _gelqf_second_output(data1)  # Output L (Q is not dangling)
    mn_lst = [(4, 4), (1, 1), (5, 20), (1, 10), (15, 50)]
    for m, n in mn_lst:
        #print('gelqf: m={}, n={}'.format(m, n))
        data_in1 = np.random.normal(0., 10., (m, n))
        res_eye = np.eye(m)
        res_a = data_in1
        check_fw(test_gelqf2, [data_in1], [res_eye, res_a])
        if grad_check == 1:
            # A => Q
            check_grad(test_gelqf_q, [data_in1])
            # A => L
            check_grad(test_gelqf_l, [data_in1])
        # Batch mode (3x the same thing)
        a_batch = rep_3x(data_in1, m, n)
        reye_batch = rep_3x(res_eye, m, m)
        ra_batch = a_batch
        check_fw(test_gelqf2, [a_batch], [reye_batch, ra_batch])
        if grad_check == 1:
            # A => Q
            check_grad(test_gelqf_q, [a_batch])
            # A => L
            check_grad(test_gelqf_l, [a_batch])


# Tests for operator linalg.syevd

def _syevd_first_output(a):
    u, lam = mx.sym.linalg.syevd(a)
    bogus_scal = mx.sym.sum(mx.sym.BlockGrad(lam), axis=(), keepdims=True) * 0.0
    return mx.sym.broadcast_add(u, bogus_scal)

def _syevd_second_output(a):
    u, lam = mx.sym.linalg.syevd(a)
    bogus_scal = mx.sym.sum(mx.sym.BlockGrad(u), axis=(), keepdims=True) * 0.0
    return mx.sym.broadcast_add(lam, bogus_scal)

def _syevd_forward(a):
    lam, ut = np.linalg.eig(a)
    ind = np.argsort(lam)
    lam = lam[ind]
    u = ut[:, ind].T
    for i in range(0, a.shape[0]):
        _syevd_forw_eigvec_sign(u[i])
    return u, lam

def _syevd_forw_eigvec_sign(v):
    ind = np.argmax(np.abs(v))
    if v[ind] < 0.:
        v[:] = -v

def _syevd_backward(grad_u, grad_l, u, l):
    n = l.size
    assert grad_l.size == n
    assert grad_u.shape == (n, n)
    assert u.shape == (n, n)
    temp = np.dot(grad_u, u.T)
    temp2 = np.diag(grad_l)
    for i in range(1, n):
        for j in range(0, i):
            denom = 2. * (l[i] - l[j])
            elem = (temp[i, j] - temp[j, i])/denom
            temp2[i, j] = elem
            temp2[j, i] = elem
    temp3 = np.dot(u.T, temp2)
    return np.dot(temp3, u)

# Seed set because the test is not robust enough to operate on random data
@with_seed(1896893923)
def test_laop_3():
    # Currently disabled on GPU as syevd needs cuda8
    # and MxNet builds use cuda 7.5
    if not (default_context() == mx.cpu()):
        return

    dtype = np.float64
    rtol_fw = 1e-6
    atol_fw = 1e-6
    num_eps = 1e-4
    rtol_bw = 1e-2
    atol_bw = 1e-2
    # enable numerical checking of gradients
    grad_check = 1

    data1 = mx.symbol.Variable('data1')
    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol_bw,
                               atol=atol_bw, dtype=dtype)
    rep_3x = lambda a, m, n :\
        np.reshape(np.tile(np.array(a).flatten(), 3), (3, 1, m, n))
    check_bw = lambda sym, location, out_grads, expected :\
        check_symbolic_backward(sym, location, out_grads, expected,
                                rtol=rtol_fw, atol=atol_fw, dtype=dtype)

    # Tests for linalg.syevd
    test_syevd2 = _syevd_combined_symbol(data1)  # Outputs (U U^T, U^T (diag L) U)
    data1_s2 = _make_symm_symbol(data1, ndims=2)
    test_syevd_u_2 = _syevd_first_output(data1_s2)
    test_syevd_l_2 = _syevd_second_output(data1_s2)
    data1_s4 = _make_symm_symbol(data1, ndims=4)
    test_syevd_u_4 = _syevd_first_output(data1_s4)
    test_syevd_l_4 = _syevd_second_output(data1_s4)
    n_lst = [4, 1, 2, 10, 14]
    for n in n_lst:
        #print('\n** syevd: n={}'.format(n))
        data_in1 = np.random.normal(0., 10., (n, n))
        data_in1 = 0.5 * (data_in1 + data_in1.T)
        res_eye = np.eye(n)
        res_a = data_in1
        check_fw(test_syevd2, [data_in1], [res_eye, res_a])
        # Check backward
        grad_u = np.random.normal(0., 2., (n, n))
        grad_l = np.random.normal(0., 2., (n,))
        bw_u, bw_l = _syevd_forward(data_in1)
        grad_a = _syevd_backward(grad_u, grad_l, bw_u, bw_l)
        check_bw(mx.sym.linalg.syevd(data1), [data_in1], [grad_u, grad_l], [grad_a])
        if grad_check == 1:
            # A => U
            check_grad(test_syevd_u_2, [data_in1])
            # A => L
            check_grad(test_syevd_l_2, [data_in1])
        # Batch mode (3x the same thing)
        a_batch = rep_3x(data_in1, n, n)
        reye_batch = rep_3x(res_eye, n, n)
        ra_batch = a_batch
        check_fw(test_syevd2, [a_batch], [reye_batch, ra_batch])
        if grad_check == 1:
            # A => U
            check_grad(test_syevd_u_4, [a_batch])
            # A => L
            check_grad(test_syevd_l_4, [a_batch])


# @piyushghai - Removing the fixed seed for this test.
# Issue for flakiness is tracked at - https://github.com/apache/incubator-mxnet/issues/11721
@with_seed()
def test_laop_4():
    # Currently disabled on GPU as syevd needs cuda8
    # and MxNet builds use cuda 7.5
    if not (default_context() == mx.cpu()):
        return

    rtol_fw = 1e-6
    atol_fw = 1e-6

    data1 = mx.symbol.Variable('data1')

    check_fw = lambda sym, location, expected, dtype :\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)

    a_np = np.array([[1., 2.], [2., 4.]])
    u_np = np.array([[0.89442718, -0.44721359], [0.44721359, 0.89442718]])
    l_np = np.array([0., 5.])
    test_syevd = mx.sym.linalg.syevd(data1)
    # float64
    #print('float64')
    check_fw(test_syevd, [a_np], [u_np, l_np], np.float64)
    # float32
    #print('float32')
    check_fw(test_syevd, [a_np], [u_np, l_np], np.float32)

def test_laop_5():
    # tests for diagonal and triangular matrix extraction and generation
    data = mx.symbol.Variable('data')
    # test complete range of small matrices to cover corner cases
    for n in range(1, 5):
        # test batched and non-batched processing
        for b in range(3):
            shape = (n, n) if b == 0 else (b, n, n)
            data_in = np.random.uniform(1, 10, shape)
            # test all legal offsets of the diagonal
            for offs in range(1-n, n):
                # test extraction of diagonal
                test_diag = mx.sym.linalg.extractdiag(data, offset=offs)
                res_diag = np.diagonal(data_in, offset=offs) if b==0 else np.diagonal(data_in, axis1=1, axis2=2, offset=offs)
                check_symbolic_forward(test_diag, [data_in], [res_diag])
                check_numeric_gradient(test_diag, [data_in])
                # test generation of diagonal matrix
                test_diag2 = mx.sym.linalg.makediag(data, offset=offs)
                res_diag2 = None
                if b == 0:
                    res_diag2 = np.diagflat(res_diag, k=offs)
                else:
                    for i in range(b):
                        res = np.reshape(np.diagflat(res_diag[i], k=offs), (1, n, n))
                        res_diag2 = res if res_diag2 is None else np.concatenate((res_diag2, res), axis=0)
                check_symbolic_forward(test_diag2, [res_diag], [res_diag2])
                check_numeric_gradient(test_diag2, [res_diag])
                # check both settings for parameter "lower" in case of zero offset
                lower_vals = [True] if offs != 0 else [True, False]
                for lower in lower_vals:
                    # test extraction of triangle by doing a full roundtrip as the intermediate extracted
                    # triangle has different orderings than numpy.
                    test_trian = mx.sym.linalg.extracttrian(data, offset=offs, lower=lower)
                    test_trian = mx.sym.linalg.maketrian(test_trian, offset=offs, lower=lower)
                    extracts_lower = (offs < 0) or ((offs == 0) and lower)
                    res_trian = None
                    if b == 0:
                        res_trian = np.tril(data_in, offs) if extracts_lower else np.triu(data_in, offs)
                    else:
                        for i in range(b):
                            res = np.tril(data_in[i], offs) if extracts_lower else np.triu(data_in[i], offs)
                            res = np.reshape(res, (1, n, n))
                            res_trian = res if res_trian is None else np.concatenate((res_trian, res), axis=0)
                    check_symbolic_forward(test_trian, [data_in], [res_trian])
                    check_numeric_gradient(test_trian, [data_in])

# Tests for linalg.inverse
@with_seed()
@unittest.skip("Test crashes https://github.com/apache/incubator-mxnet/issues/15975")
def test_laop_6():
    dtype = np.float64
    rtol_fw = 1e-7
    atol_fw = 1e-9
    num_eps = 1e-6
    rtol_bw = 1e-4
    atol_bw = 1e-6

    data = mx.symbol.Variable('data')

    check_fw = lambda sym, location, expected:\
        check_symbolic_forward(sym, location, expected, rtol=rtol_fw,
                               atol=atol_fw, dtype=dtype)
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol_bw,
                               atol=atol_bw, dtype=dtype)

    ## det(I + dot(v, v.T)) = 1 + dot(v.T, v) >= 1, so it's always invertible;
    ## det is away from zero, so the value of logdet is stable
    v = np.random.random(4)
    a = np.eye(4) + np.outer(v, v)
    a = np.tile(a, (3, 1, 1))
    permute_mat = np.eye(4)[[1, 0, 2, 3]]

    # test matrix inverse
    r = np.eye(4)
    r = np.tile(r, (3, 1, 1))
    test_inverse = mx.sym.linalg.inverse(data)
    test_eye = mx.sym.linalg.gemm2(data, test_inverse)
    check_fw(test_eye, [a], [r])
    check_grad(test_inverse, [a])

    # test matrix determinant
    # det
    r = np.linalg.det(a)
    test_det = mx.sym.linalg.det(data)
    check_fw(test_det, [a], [r])
    check_grad(test_det, [a])
    # test slogdet
    r1 = np.array([1., 1., 1.])
    r2 = np.log(np.abs(np.linalg.det(a)))
    test_sign, test_logabsdet = mx.sym.linalg.slogdet(data)
    check_fw(test_sign, [a], [r1])
    check_fw(test_sign, [np.dot(a, permute_mat)], [-r1])
    check_fw(test_logabsdet, [a], [r2])
    check_grad(test_logabsdet, [a])

@with_seed()
def test_stack():
    for _ in range(100):
        ndim = random.randint(1, 5)
        axis = random.randint(0, ndim)
        if random.randint(0, 1):
            axis = axis - ndim - 1
        nin = random.randint(1, 3)
        dshape = [random.randint(1, 5) for _ in range(ndim)]
        inputs = [np.random.uniform(size=dshape) for _ in range(nin)]
        output = np.stack(inputs, axis=axis)
        sym_ins = [mx.sym.var('x%d'%i) for i in range(nin)]
        out = mx.sym.stack(*sym_ins, axis=axis)
        check_symbolic_forward(out, inputs, [output])
        check_numeric_gradient(out, inputs)


## TODO: test fails intermittently when cudnn on. temporarily disabled cudnn until gets fixed.
## tracked at https://github.com/apache/incubator-mxnet/issues/14288
@with_seed()
def test_dropout():
    def zero_count(array, ratio):
        zeros = 0
        for i in array:
            if i == 0:
                zeros += 1
            elif math.isnan(i):
                assert ratio == 1  # Only valid for ratio = 1
                zeros += 1
        return zeros

    def check_correctness(executor, input, ratio):
        input = input.ravel()
        output = executor.outputs[0].asnumpy().ravel()
        input_sum = np.sum(input)
        output_sum = np.sum(output)

        # Make sure input zeroes are none (test data setup check)
        assert zero_count(input, ratio) == 0

        # count number of zeroes in output
        output_zeroes = zero_count(output, ratio)

        # Hopefully should be within ratio/2 %
        error = abs(output_sum - input_sum) / input_sum
        if ratio == 1.0:
            assert output_zeroes == len(input)
        elif ratio > 0.2:
            assert output_zeroes > 0
            assert error < (ratio/2)
        elif ratio == 0:
            assert output_zeroes == 0

    def check_dropout_ratio(ratio, shape, cudnn_off=True):
        # test dropout
        x = mx.sym.var('data')
        y = mx.sym.Dropout(x, p=ratio, cudnn_off=cudnn_off)
        exe = y.simple_bind(ctx=default_context(), data=shape)

        if ratio == 1:
            max_value = float('nan')
        else:
            max_value = 1 if ratio == 0 else 1/ratio

        if ratio == 1:
            min_value = float('nan')
        else:
            min_value = 1 if ratio == 0 else 0

        exe.arg_arrays[0][:] = 1
        exe.forward(is_train=True)

        if not math.isnan(max_value):
            assert exe.outputs[0].asnumpy().max() > 0
        else:
            assert math.isnan(exe.outputs[0].asnumpy().max())
        if not math.isnan(min_value):
            assert exe.outputs[0].asnumpy().min() == min_value
        else:
            assert math.isnan(exe.outputs[0].asnumpy().min())

        check_correctness(exe, exe.arg_arrays[0].asnumpy(), ratio)

        if ratio == 0.5:
            exe.backward([mx.nd.ones(shape)])
            assert (exe.grad_arrays[0].asnumpy() == exe.outputs[0].asnumpy()).all()

            exe.forward(is_train=False)
            assert (exe.outputs[0].asnumpy() == exe.arg_arrays[0].asnumpy()).all()
            exe.backward([mx.nd.ones(shape)], is_train=False)
            assert (exe.grad_arrays[0].asnumpy() == exe.arg_arrays[0].asnumpy()).all()

            # test permanent dropout
            x = mx.sym.var('data')
            y = mx.sym.Dropout(x, p=ratio, mode='always', cudnn_off=cudnn_off)
            exe = y.simple_bind(ctx=default_context(), data=shape)

            exe.arg_arrays[0][:] = 1
            exe.forward(is_train=True)
            assert exe.outputs[0].asnumpy().max() == max_value
            assert exe.outputs[0].asnumpy().min() == min_value
            exe.backward([mx.nd.ones(shape)])
            assert (exe.grad_arrays[0].asnumpy() == exe.outputs[0].asnumpy()).all()

            exe.forward(is_train=False)
            assert exe.outputs[0].asnumpy().max() == max_value
            assert exe.outputs[0].asnumpy().min() == min_value
            exe.backward([mx.nd.ones(shape)], is_train=False)
            assert (exe.grad_arrays[0].asnumpy() == exe.outputs[0].asnumpy()).all()

    def get_slice(x, axis, idx):
        ix = ()
        for i in range(x.ndim):
            if i == axis:
                ix += (idx,)
            else:
                ix += (slice(None, None, None),)
        return x[ix]

    def check_dropout_axes(ratio, shape, axes, cudnn_off=True):
        compactshape = list(shape)
        for axis in axes:
            compactshape[axis] = 1
        compactx = mx.random.uniform(shape=tuple(compactshape))
        broadcastx = compactx.broadcast_to(shape)
        dropouty = mx.nd.Dropout(broadcastx, p=ratio, axes=axes, cudnn_off=cudnn_off)
        for axis in axes:
            target = get_slice(dropouty, axis, 0).asnumpy()
            for i in range(1, shape[axis]):
                assert(get_slice(dropouty, axis, i).asnumpy() == target).all()

    def check_passthrough(ratio, shape, cudnn_off=True):
        # test inference_mode forward and then backward
        a = mx.random.uniform(shape=shape)
        a.attach_grad()
        with mx.autograd.record(train_mode=False):
            b = mx.nd.Dropout(a, ratio, cudnn_off=cudnn_off) # dropout acts as identity
        b.backward()
        assert_almost_equal(a.grad.asnumpy(), mx.nd.ones_like(b).asnumpy())

    shape = (100, 100)
    check_dropout_ratio(0.5, shape)
    check_dropout_ratio(0.0, shape)
    check_dropout_ratio(1.0, shape)
    check_dropout_ratio(0.75, shape)
    check_dropout_ratio(0.25, shape)
    # check_dropout_ratio(0.5, shape, cudnn_off=False)
    # check_dropout_ratio(0.0, shape, cudnn_off=False)
    # check_dropout_ratio(1.0, shape, cudnn_off=False)
    # check_dropout_ratio(0.75, shape, cudnn_off=False)
    # check_dropout_ratio(0.25, shape, cudnn_off=False)

    check_passthrough(0.5, shape)
    check_passthrough(0.0, shape)
    check_passthrough(1.0, shape)
    # check_passthrough(0.5, shape, cudnn_off=False)
    # check_passthrough(0.0, shape, cudnn_off=False)
    # check_passthrough(1.0, shape, cudnn_off=False)

    nshape = (10, 10, 10, 10)
    with mx.autograd.train_mode():
        check_dropout_axes(0.25, nshape, axes = (0,))
        check_dropout_axes(0.25, nshape, axes = (1,))
        check_dropout_axes(0.25, nshape, axes = (2,))
        check_dropout_axes(0.25, nshape, axes = (3,))
        check_dropout_axes(0.25, nshape, axes = (0, 1))
        check_dropout_axes(0.25, nshape, axes = (0, 2))
        check_dropout_axes(0.25, nshape, axes = (0, 3))
        check_dropout_axes(0.25, nshape, axes = (1, 2))
        check_dropout_axes(0.25, nshape, axes = (1, 3))
        check_dropout_axes(0.25, nshape, axes = (2, 3))
        check_dropout_axes(0.25, nshape, axes = (0, 1, 2))
        check_dropout_axes(0.25, nshape, axes = (0, 2, 3))
        check_dropout_axes(0.25, nshape, axes = (1, 2, 3))
        # check_dropout_axes(0.25, nshape, axes = (0,), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (1,), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (2,), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (3,), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (0, 1), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (0, 2), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (0, 3), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (1, 2), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (1, 3), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (2, 3), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (0, 1, 2), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (0, 2, 3), cudnn_off=False)
        # check_dropout_axes(0.25, nshape, axes = (1, 2, 3), cudnn_off=False)


@unittest.skip("test fails intermittently. temporarily disabled till it gets fixed. tracked at https://github.com/apache/incubator-mxnet/issues/11290")
@with_seed()
def test_scatter_gather_nd():
    def check(data, idx):
        data.attach_grad()
        with mx.autograd.record():
            y = mx.nd.gather_nd(data, idx)
            y.backward(y)
        npidx = tuple(i.asnumpy() for i in idx)
        assert (data.asnumpy()[npidx] == y.asnumpy()).all()
        npdata = np.zeros_like(data.asnumpy())
        npdata[npidx] = y.asnumpy()
        assert (npdata == data.grad.asnumpy()).all()
        assert (mx.nd._internal._backward_gather_nd(y, idx, shape=data.shape).asnumpy() == data.grad.asnumpy()).all()
    for dtype in ['int32', 'int64', 'float16', 'float32', 'float64']:
        data = mx.nd.arange(360, dtype=dtype).reshape((3,4,5,6))
        idx = mx.nd.array([[1,1,2], [3, 3, 0], [3,2,1]], dtype='int32')
        check(data, idx)

        idx = mx.nd.array([[1,1,2], [3,3,0], [3,2,1], [5,2,4]], dtype='int32')

        check(data, idx)

        data = mx.nd.array([2, 3, 0], dtype=dtype)
        idx = mx.nd.array([[1, 1, 0], [0, 1, 0]], dtype='int32')
        assert (mx.nd.scatter_nd(data, idx, shape=(2, 2)).asnumpy() == [[0, 0], [2, 3]]).all()

        data = mx.nd.array([2, 3, 0], dtype=dtype)
        idx = mx.nd.array([[1, 1, 0], [1, 1, 0]], dtype='int32')
        assert (mx.nd._internal._backward_gather_nd(data, idx, shape=(2, 2)).asnumpy() == [[0, 0], [0, 5]]).all()
        data_npy = np.random.randint(0, 10, (100,))
        data = mx.nd.array(data_npy, dtype=dtype)
        idx = mx.nd.zeros(shape=(1, 100), dtype='int32')
        assert (mx.nd._internal._backward_gather_nd(data, idx, shape=(1,)).asscalar() == data_npy.sum())
        if dtype == 'int64':
            data = mx.nd.array([2123162361283621, -31231236374787,
                                -112372937128970, -1378278798172378], dtype=dtype)
            idx = mx.nd.array([[0, 0, 0, 0]], dtype='int32')
            assert (mx.nd._internal._backward_gather_nd(data, idx, shape=(1,)).asscalar() == data.asnumpy().sum())

@with_seed()
def test_gather_nd_check_bound():
    def _test_gather_nd_exception(data, indices):
        output = mx.nd.gather_nd(data, indices).asnumpy()
    # check if indices is out of bound
    data = mx.nd.array([[0, 1, 2], [3, 4, 5]])
    indices1 = mx.nd.array([[0, 1, 0], [0, 1, 3]])
    indices2 = mx.nd.array([[0, 1, 0], [0, 1, -5]])
    assertRaises(IndexError, _test_gather_nd_exception, data, indices1)
    # IndexError: index 3 is out of bounds for axis 1 with size 3
    assertRaises(IndexError, _test_gather_nd_exception, data, indices2)
    # IndexError: index -5 is out of bounds for axis 1 with size 3

    # check if the negative indices are wrapped correctly
    indices1 = mx.nd.array([[0, 1, -1], [0, 1, -2]])
    indices2 = mx.nd.array([[0, 1, 1], [0, 1, 1]])
    data1 = mx.nd.gather_nd(data, indices1)
    data2 = mx.nd.gather_nd(data, indices2)
    assert_almost_equal(data1, data2, rtol=1e-5, atol=1e-5)


def compare_forw_backw_unary_op(
        name, forward_mxnet_call, forward_numpy_call,
        backward_numpy_call, shape, input_low, input_high, rtol, atol,
        dtype=np.float32):
    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol,
                               atol=atol, dtype=dtype)
    check_bw = lambda sym, location, out_grads, expected :\
        check_symbolic_backward(sym, location, out_grads, expected,
                                rtol=rtol, atol=atol, dtype=dtype)
    op_name = 'unary_op={}, dtype={}'.format(name, dtype)
    data = mx.symbol.Variable(op_name + '_data', dtype=dtype)
    # Comparison: Forward expression
    data_np = np.random.uniform(input_low, input_high, shape).astype(dtype)
    res_np = forward_numpy_call(data_np)
    op_ex = mx.sym.broadcast_add(
        forward_mxnet_call(data), mx.sym.zeros_like(data),
        name=op_name)
    check_fw(op_ex, [data_np], [res_np])
    # Comparison: Backward expression
    res_grad = np.random.uniform(-2.0, 2.0, shape).astype(dtype)
    data_grad = backward_numpy_call(data_np) * res_grad
    check_bw(op_ex, [data_np], [res_grad], [data_grad])

def finite_diff_unary_op(
        name, forward_mxnet_call, shape, input_low, input_high, rtol, atol,
        num_eps):
    # Finite difference tests are done in float64
    dtype = np.float64
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol,
                               atol=atol, dtype=dtype)
    data_np = np.random.uniform(input_low, input_high, shape).astype(dtype)
    data = mx.symbol.Variable('data', dtype=dtype)
    op_name = 'unary_op={}, dtype={}'.format(name, dtype)
    op_ex = mx.sym.broadcast_add(
        forward_mxnet_call(data), mx.sym.zeros_like(data),
        name=op_name)
    check_grad(op_ex, [data_np])

def np_smooth_l1(x, sigma):
    issq = 1. / sigma / sigma
    absx = np.abs(x)
    temp = x * sigma
    return np.where(absx < issq, 0.5 * (temp ** 2), absx - 0.5 * issq)

def np_smooth_l1_grad(x, sigma):
    ssq = sigma * sigma
    return np.where(np.abs(x) < 1. / ssq, x * ssq, np.sign(x))

# Tests for unary operators (basic mathematical functions):
# - Forward: Comparison to NumPy (several dtype)
# - Backward: Comparison to NumPy (several dtype)
# - Finite difference tests (only dtype = float64)
# Seed set because the test is not robust enough to operate on random data
@with_seed(192837465)
def test_unary_math_operators():
    have_scipy = True
    try:
        from scipy import special as scipy_special
    except:
        print("Could not import scipy. Skipping unit tests for special functions")
        have_scipy = False
    shape=(9, 10)
    dtype_l = [np.float64, np.float32, np.float16]
    rtol_l = [1e-7, 1e-6, 1e-2]
    rtol_less_l = [1e-6, 1e-5, 1e-2]
    atol_l = [1e-7, 1e-6, 1e-2]
    atol_less_l = [1e-6, 1e-5, 1e-2]
    rtol_fd = 1e-5
    atol_fd = 1e-6
    num_eps = 1e-6
    unary_ops = {
        'arccos' : [lambda x: mx.sym.arccos(x),
                    lambda x: np.arccos(x),
                    lambda x: -1. / np.sqrt(1. - x ** 2.),
                    -0.95, 0.95],
        'arccosh': [lambda x: mx.sym.arccosh(x),
                    lambda x: np.arccosh(x),
                    lambda x: 1. / np.sqrt(x ** 2 - 1.),
                    1.05, 10.0],
        'arcsin': [lambda x: mx.sym.arcsin(x),
                   lambda x: np.arcsin(x),
                   lambda x: 1. / np.sqrt(1. - x ** 2),
                   -0.95, 0.95],
        'arcsinh': [lambda x: mx.sym.arcsinh(x),
                    lambda x: np.arcsinh(x),
                    lambda x: 1. / np.sqrt(x**2 + 1.),
                    -5.0, 5.0],
        'arctan': [lambda x: mx.sym.arctan(x),
                   lambda x: np.arctan(x),
                   lambda x: 1. / (x ** 2. + 1.),
                   -5.0, 5.0],
        'arctanh': [lambda x: mx.sym.arctanh(x),
                    lambda x: np.arctanh(x),
                    lambda x: 1. / (1. - x ** 2),
                    -0.95, 0.95],
        'cbrt': [lambda x: mx.sym.cbrt(x),
                 lambda x: np.cbrt(x),
                 lambda x: 1. / (3. * np.cbrt(x) ** 2),
                 -10.0, 10.0],
        'cos': [lambda x: mx.sym.cos(x),
                lambda x: np.cos(x),
                lambda x: -np.sin(x),
                -5.0, 5.0],
        'cosh': [lambda x: mx.sym.cosh(x),
                 lambda x: np.cosh(x),
                 lambda x: np.sinh(x),
                 -2.0, 2.0],
        'exp': [lambda x: mx.sym.exp(x),
                lambda x: np.exp(x),
                lambda x: np.exp(x),
                -4.0, 4.0],
        'expm1': [lambda x: mx.sym.expm1(x),
                  lambda x: np.expm1(x),
                  lambda x: np.exp(x),
                  -0.1, 0.1],
        'log': [lambda x: mx.sym.log(x),
                lambda x: np.log(x),
                lambda x: 1. / x,
                0.01, 100.0],
        'log10': [lambda x: mx.sym.log10(x),
                lambda x: np.log10(x),
                lambda x: 1. / (x * np.log(10.)),
                0.01, 100.0],
        'log2': [lambda x: mx.sym.log2(x),
                lambda x: np.log2(x),
                lambda x: 1. / (x * np.log(2.)),
                0.01, 100.0],
        'log1p': [lambda x: mx.sym.log1p(x),
                  lambda x: np.log1p(x),
                  lambda x: 1. / (1. + x),
                  -0.1, 0.1],
        'rcbrt': [lambda x: mx.sym.rcbrt(x),
                  lambda x: 1. / np.cbrt(x),
                  lambda x: -1. / (3. * x * np.cbrt(x)),
                  0.01, 100.0],
        'reciprocal': [lambda x: mx.sym.reciprocal(x),
                       lambda x: 1. / x,
                       lambda x: -1. / (x ** 2),
                       0.01, 100.0],
        'relu': [lambda x: mx.sym.relu(x),
                 lambda x: np.maximum(x, 0.),
                 lambda x: 1. * (x > 0.),
                 -5.0, 5.0],
        'rsqrt': [lambda x: mx.sym.rsqrt(x),
                  lambda x: 1. / np.sqrt(x),
                  lambda x: -0.5 / (x * np.sqrt(x)),
                  0.01, 100.0],
        'sigmoid': [lambda x: mx.sym.sigmoid(x),
                    lambda x: 1. / (np.exp(-x) + 1.),
                    lambda x: 1. / (np.exp(-x) + 1.) / (np.exp(x) + 1.),
                    -3.0, 3.0],
        'softsign': [lambda x: mx.sym.softsign(x),
                    lambda x: x / (1. + np.abs(x)),
                    lambda x: 1. / np.square(1. + np.abs(x)),
                    -3.0, 3.0],
        'sin': [lambda x: mx.sym.sin(x),
                lambda x: np.sin(x),
                lambda x: np.cos(x),
                -5.0, 5.0],
        'sinh': [lambda x: mx.sym.sinh(x),
                 lambda x: np.sinh(x),
                 lambda x: np.cosh(x),
                 -2.0, 2.0],
        'sqrt': [lambda x: mx.sym.sqrt(x),
                 lambda x: np.sqrt(x),
                 lambda x: 0.5 / np.sqrt(x),
                 0.01, 100.0],
        'tan': [lambda x: mx.sym.tan(x),
                lambda x: np.tan(x),
                lambda x: np.tan(x) ** 2 + 1.,
                -1.5, 1.5],
        'tanh': [lambda x: mx.sym.tanh(x),
                 lambda x: np.tanh(x),
                 lambda x: 1. - np.tanh(x) ** 2,
                 -4.0, 4.0],
        'smooth_l1_sig1': [lambda x: mx.sym.smooth_l1(x, scalar=1.),
                           lambda x: np_smooth_l1(x, 1.),
                           lambda x: np_smooth_l1_grad(x, 1.),
                           -2.0, 2.0],
        'smooth_l1_sig_default': [lambda x: mx.sym.smooth_l1(x),
                                  lambda x: np_smooth_l1(x, 1.),
                                  lambda x: np_smooth_l1_grad(x, 1.),
                                  -2.0, 2.0],
        'smooth_l1_sig2': [lambda x: mx.sym.smooth_l1(x, scalar=2.),
                           lambda x: np_smooth_l1(x, 2.),
                           lambda x: np_smooth_l1_grad(x, 2.),
                           -1.0, 1.0]
    }
    if have_scipy:
        unary_ops['gamma'] = [lambda x: mx.sym.gamma(x),
                              lambda x: scipy_special.gamma(x),
                              lambda x: scipy_special.gamma(x) * scipy_special.psi(x),
                              0.01, 5.0]
        unary_ops['gammaln'] = [lambda x: mx.sym.gammaln(x),
                                lambda x: scipy_special.gammaln(x),
                                lambda x: scipy_special.psi(x),
                                0.01, 20.0]
    # Loop over operators
    for name, op in unary_ops.items():
        # Loop over dtype's
        for ind in range(len(dtype_l)):
            dtype = dtype_l[ind]
            if name == 'gammaln' or name == 'gamma':
                rtol = rtol_less_l[ind]
                atol = atol_less_l[ind]
            else:
                rtol = rtol_l[ind]
                atol = atol_l[ind]
            compare_forw_backw_unary_op(
                name, op[0], op[1], op[2], shape, op[3], op[4], rtol, atol,
                dtype)
        # Finite difference testing
        finite_diff_unary_op(
            name, op[0], shape, op[3], op[4], rtol_fd, atol_fd, num_eps)

def compare_forw_backw_binary_op(
        name, forward_mxnet_call, forward_numpy_call,
        backward1_numpy_call, backward2_numpy_call, shape, input1_low,
        input1_high, input2_low, input2_high, rtol, atol, dtype=np.float32):
    check_fw = lambda sym, location, expected :\
        check_symbolic_forward(sym, location, expected, rtol=rtol,
                               atol=atol, dtype=dtype)
    check_bw = lambda sym, location, out_grads, expected :\
        check_symbolic_backward(sym, location, out_grads, expected,
                                rtol=rtol, atol=atol, dtype=dtype)
    op_name = 'binary_op={}, dtype={}'.format(name, dtype)
    data1 = mx.symbol.Variable(op_name + '_data1', dtype=dtype)
    data2 = mx.symbol.Variable(op_name + '_data2', dtype=dtype)
    # Comparison: Forward expression
    data1_np = np.random.uniform(input1_low, input1_high, shape).astype(dtype)
    data2_np = np.random.uniform(input2_low, input2_high, shape).astype(dtype)
    res_np = forward_numpy_call(data1_np, data2_np)
    op_ex = mx.sym.broadcast_add(
        forward_mxnet_call(data1, data2), mx.sym.zeros_like(data1),
        name=op_name)
    check_fw(op_ex, [data1_np, data2_np], [res_np])
    # Comparison: Backward expression
    res_grad = np.random.uniform(-2.0, 2.0, shape).astype(dtype)
    data1_grad = backward1_numpy_call(data1_np, data2_np) * res_grad
    data2_grad = backward2_numpy_call(data1_np, data2_np) * res_grad
    check_bw(op_ex, [data1_np, data2_np], [res_grad], [data1_grad, data2_grad])

def finite_diff_binary_op(
        name, forward_mxnet_call, shape, input1_low, input1_high, input2_low,
        input2_high, rtol, atol, num_eps):
    # Finite difference tests are done in float64
    dtype = np.float64
    check_grad = lambda sym, location:\
        check_numeric_gradient(sym, location, numeric_eps=num_eps, rtol=rtol,
                               atol=atol, dtype=dtype)
    data1_np = np.random.uniform(input1_low, input1_high, shape).astype(dtype)
    data2_np = np.random.uniform(input2_low, input2_high, shape).astype(dtype)
    data1 = mx.symbol.Variable('data1', dtype=dtype)
    data2 = mx.symbol.Variable('data2', dtype=dtype)
    op_name = 'binary_op={}, dtype={}'.format(name, dtype)
    op_ex = mx.sym.broadcast_add(
        forward_mxnet_call(data1, data2), mx.sym.zeros_like(data1),
        name=op_name)
    check_grad(op_ex, [data1_np, data2_np])

# Tests for unary operators (basic mathematical functions):
# - Forward: Comparison to NumPy (several dtype)
# - Backward: Comparison to NumPy (several dtype)
# - Finite difference tests (only dtype = float64)
@with_seed()
def test_binary_math_operators():
    shape=(9, 10)
    dtype_l = [np.float64, np.float32, np.float16]
    rtol_l = [1e-7, 1e-6, 1e-2]
    atol_l = [1e-7, 1e-6, 1e-2]
    rtol_fd = 1e-5
    atol_fd = 1e-6
    num_eps = 1e-6
    binary_ops = {
        'hypot' : [lambda x, y: mx.sym.hypot(x, y),
                   lambda x, y: np.hypot(x, y),
                   lambda x, y: x / np.hypot(x, y),
                   lambda x, y: y / np.hypot(x, y),
                    -5.0, 5.0, -5.0, 5.0],
        'pow': [lambda x, y: mx.sym.pow(x, y),
                lambda x, y: np.power(x, y),
                lambda x, y: np.power(x, y - 1.) * y,
                lambda x, y: np.power(x, y) * np.log(x),
                0.2, 5.0, -4.0, 4.0],
        'power': [lambda x, y: mx.sym.power(x, y),
                  lambda x, y: np.power(x, y),
                  lambda x, y: np.power(x, y - 1.) * y,
                  lambda x, y: np.power(x, y) * np.log(x),
                  0.2, 5.0, -4.0, 4.0]
    }
    # Loop over operators
    for name, op in binary_ops.items():
        # Loop over dtype's
        for ind in range(len(dtype_l)):
            dtype = dtype_l[ind]
            compare_forw_backw_binary_op(
                name, op[0], op[1], op[2], op[3], shape, op[4], op[5], op[6],
                op[7], rtol_l[ind], atol_l[ind], dtype)
        # Finite difference testing
        finite_diff_binary_op(
            name, op[0], shape, op[4], op[5], op[6], op[7], rtol_fd, atol_fd,
            num_eps)


@with_seed()
def test_softmax():
    check_softmax_with_shape((3, 4), default_context(), preserve_shape=False)
    check_softmax_with_shape((3, 4), default_context(), preserve_shape=True)
    check_softmax_with_shape((3, 4, 2), default_context(), preserve_shape=True)
    check_softmax_grad(default_context())
    check_smoothed_softmax_grad(default_context())


@with_seed()
def test_softmax_output_normalization():
    def _softmaxoutput_normalization(multi_output, use_ignore, normalization):
        grad_scale = np.random.random()
        batch_size = 8
        num_labels = 6
        H, W = 3, 3
        ignore_label = np.random.randint(0, num_labels) if use_ignore else -1

        if multi_output:
            data_shape = (batch_size, num_labels, H, W)
            label_shape = (batch_size, H, W)
        else:
            data_shape = (batch_size, num_labels)
            label_shape = (batch_size, )

        data = mx.nd.random.uniform(-1, 1, shape=data_shape)
        label = mx.nd.random.randint(
            0, num_labels, shape=label_shape).astype('float32')
        data.attach_grad()

        kwargs = dict(grad_scale=grad_scale,
                      normalization=normalization, multi_output=multi_output)
        if use_ignore:
            kwargs.update(use_ignore=True, ignore_label=ignore_label)

        with mx.autograd.record():
            out = mx.nd.SoftmaxOutput(data=data, label=label, **kwargs)
        out.backward(mx.nd.ones_like(data))

        exp_data = mx.nd.exp(data)
        softmax_data = exp_data / exp_data.sum(1, keepdims=True)
        argmax_data = mx.nd.argmax(data, axis=1)

        assert_almost_equal(out.asnumpy(), softmax_data.asnumpy())
        one_hot_label = mx.nd.one_hot(label, num_labels)
        if multi_output:
            one_hot_label = one_hot_label.transpose((0, 3, 1, 2))
        data_grad = softmax_data - one_hot_label

        if use_ignore:
            if multi_output:
                data_grad *= (label !=
                              ignore_label).reshape((batch_size, 1, H, W))
            else:
                data_grad *= (label != ignore_label).reshape((batch_size, 1))

        valid_cnt = 1
        if normalization == 'batch':
            valid_cnt = batch_size
        elif normalization == 'valid':
            valid_cnt = mx.nd.maximum(1, (label != ignore_label).sum())
        scale = grad_scale / valid_cnt

        if multi_output:
            if normalization != 'valid':
                scale /= H * W

        data_grad *= scale

        assert_almost_equal(data.grad.asnumpy(), data_grad.asnumpy())

    for multi_output in [False, True]:
        for use_ignore in [False, True]:
            for normalization in ['null', 'batch', 'valid']:
                _softmaxoutput_normalization(
                    multi_output, use_ignore, normalization)


@with_seed()
def test_slice():
    def test_slice_forward_backward(a, index):
        a_np = a.asnumpy()
        begin = []
        end = []
        step = []
        for slice_i in index:
            begin.append(slice_i.start)
            end.append(slice_i.stop)
            step.append(slice_i.step)
        b = mx.nd.slice(a, begin=begin, end=end, step=step)
        b_np = a_np[index]
        assert same(b.asnumpy(), b_np)

        data = mx.sym.Variable('data')
        slice_sym = mx.sym.slice(data, begin=begin, end=end, step=step)
        expected_in_grad = np.zeros_like(a_np)
        expected_in_grad[index] = b_np
        check_symbolic_backward(slice_sym, [a_np], [b_np], [expected_in_grad])

    shape = (16, 14, 17, 20)
    arr = mx.nd.arange(np.prod(shape)).reshape(shape=shape)
    index_list = [(slice(None),), (slice(None), slice(None)), (slice(1, 10),), (slice(1, 10), slice(3, 9)),
                  (slice(1, 10), slice(2, 5), slice(3, 6), slice(7, 10)),
                  (slice(1, 10, 2), slice(2, 9, 3), slice(3, 6, 5), slice(7, 10, 2)),
                  (slice(None, None, -1), slice(None, None, -1), slice(None, None, -1)),
                  (slice(10, 0, -2), slice(5, 2, -1), slice(7, None, 3), slice(None, 12, 4))]
    for index in index_list:
        test_slice_forward_backward(arr, index)

    # check numeric gradient
    in_data = np.arange(36).reshape(2, 2, 3, 3)
    data = mx.sym.Variable('data')
    slice_sym = mx.sym.slice(data, begin=[0, None], end=[1, None], step=[2, -1])
    check_numeric_gradient(slice_sym, [in_data])


def test_slice_partial_infer():
    def check_slice_partial_infer(data, begin, end, step, expected_out_shape):
        out = mx.sym.slice(data, begin=begin, end=end, step=step)
        assert (out.infer_shape_partial()[1][0] == expected_out_shape), out.infer_shape_partial()[1]

    def check_slice_axis_partial_infer(data, axis, begin, end, expected_out_shape):
        out = mx.sym.slice_axis(data, axis=axis, begin=begin, end=end)
        assert (out.infer_shape_partial()[1][0] == expected_out_shape), out.infer_shape_partial()[1]

    var1 = mx.sym.var(name="data", shape=(0, 20))
    check_slice_partial_infer(var1, (None, None), (None, 10), [], (0, 10))
    check_slice_partial_infer(var1, (None, None), (None, 10), (None, 2), (0, 5))
    check_slice_partial_infer(var1, (None, 3), (None, 10), [], (0, 7))
    check_slice_partial_infer(var1, (None, 3), (5, 10), [], (0, 7))
    check_slice_partial_infer(var1, (2, 3), (None, 10), [], (0, 7))
    check_slice_partial_infer(var1, (2, 3), (None, 10), (None, 1), (0, 7))
    check_slice_partial_infer(var1, (2, 3), (None, 10), (3, 3), (0, 3))

    var1 = mx.sym.var(name="data", shape=(10, 0))
    check_slice_axis_partial_infer(var1, 0, 0, 5, (5, 0))
    check_slice_axis_partial_infer(var1, 1, 0, 5, (10, 0))

    with mx.np_shape():
        var1 = mx.sym.var(name="data", shape=(-1, 20))
        check_slice_partial_infer(var1, (None, None), (None, 10), [], (-1, 10))
        check_slice_partial_infer(var1, (None, None), (None, 10), (None, 2), (-1, 5))
        check_slice_partial_infer(var1, (None, 3), (None, 10), [], (-1, 7))
        check_slice_partial_infer(var1, (None, 3), (5, 10), [], (-1, 7))
        check_slice_partial_infer(var1, (2, 3), (None, 10), [], (-1, 7))
        check_slice_partial_infer(var1, (2, 3), (None, 10), (None, 1), (-1, 7))
        check_slice_partial_infer(var1, (2, 3), (None, 10), (3, 3), (-1, 3))

        var1 = mx.sym.var(name='data', shape=(10, -1))
        check_slice_axis_partial_infer(var1, 0, 0, 5, (5, -1))
        check_slice_axis_partial_infer(var1, 1, 0, 5, (10, -1))


@with_seed()
def test_float16_min_max():
    """Test for issue: https://github.com/apache/incubator-mxnet/issues/9007"""
    a = mx.nd.array([np.finfo('float16').min, np.finfo('float16').max], dtype='float16')
    assert a.dtype == np.float16
    assert np.finfo('float16').min == mx.nd.min(a).asscalar()
    assert np.finfo('float16').max == mx.nd.max(a).asscalar()


@with_seed()
@mx.use_np_shape
def test_zero_size_min_max():
    def min():
        a = mx.nd.zeros(shape=(5, 0))
        a.min()

    def max():
        a = mx.nd.zeros(shape=(5, 0))
        a.max()

    assert_raises(MXNetError, min)
    assert_raises(MXNetError, max)


@with_seed()
def test_squeeze_op():
    def check_squeeze_op(shape, axis=None):
        data = mx.nd.random.uniform(low=-10.0, high=10.0, shape=shape)
        if axis is None:
            out = mx.nd.squeeze(data).asnumpy()
            out_expected = np.squeeze(data.asnumpy())
        else:
            out = mx.nd.squeeze(data, axis=axis).asnumpy()
            out_expected = np.squeeze(data.asnumpy(), axis=axis)
        if out.shape == (1,):  # as an exception (1, 1, 1) will be squeezed to (1,)
            out_expected = np.squeeze(data.asnumpy(), axis=tuple([i for i in range(1, len(shape))]))
        assert same(out, out_expected)

    # check forward
    check_squeeze_op((1, 5, 1, 3, 1), 0)
    check_squeeze_op((1, 5, 1, 3, 1), 2)
    check_squeeze_op((1, 5, 1, 3, 1), 4)
    check_squeeze_op((1, 5, 1, 3, 1), (0, 4))
    check_squeeze_op((1, 5, 1, 3, 1), (0, 2, 4))
    check_squeeze_op((1, 5, 1, 3, 1))
    check_squeeze_op((1, 1, 1, 1))

    # check gradient
    data = mx.symbol.Variable('data')
    shape = (1, 2, 1, 3, 1)
    data_tmp = np.ones(shape)
    test = mx.sym.squeeze(data)
    check_numeric_gradient(test, [data_tmp])
    test = mx.sym.squeeze(data, axis=2)
    check_numeric_gradient(test, [data_tmp])
    test = mx.sym.squeeze(data, axis=(2, 4))
    check_numeric_gradient(test, [data_tmp])

@with_seed()
def test_adaptive_avg_pool_op():
    def py_adaptive_avg_pool(x, height, width):
        # 2D per frame adaptive avg pool
        def adaptive_avg_pool_frame(x, y):
            isizeH, isizeW = x.shape
            osizeH, osizeW = y.shape
            for oh in range(osizeH):
                istartH = int(np.floor(1.0 * (oh * isizeH) / osizeH))
                iendH = int(np.ceil(1.0 * (oh + 1) * isizeH / osizeH))
                kH = iendH - istartH
                for ow in range(osizeW):
                    istartW = int(np.floor(1.0 * (ow * isizeW) / osizeW))
                    iendW = int(np.ceil(1.0 * (ow + 1) * isizeW / osizeW))
                    kW = iendW - istartW
                    xsum = 0
                    for ih in range(kH):
                        for iw in range(kW):
                            xsum += x[istartH+ih][istartW+iw]
                    y[oh][ow] = xsum / kH / kW

        B,C,_,_ = x.shape
        y = np.empty([B,C,height, width], dtype=x.dtype)
        for b in range(B):
            for c in range(C):
                adaptive_avg_pool_frame(x[b][c], y[b][c])
        return y
    def check_adaptive_avg_pool_op(shape, output_height, output_width=None):
        x = mx.nd.random.uniform(shape=shape)
        if output_width is None:
            y = mx.nd.contrib.AdaptiveAvgPooling2D(x, output_size=output_height)
            npy = py_adaptive_avg_pool(x.asnumpy(), output_height, output_height)
        else:
            y = mx.nd.contrib.AdaptiveAvgPooling2D(x, output_size=(output_height, output_width))
            npy = py_adaptive_avg_pool(x.asnumpy(), output_height, output_width)
        assert_almost_equal(y.asnumpy(), npy)
    shape = (2, 2, 10, 10)
    for i in range(1, 11):
        check_adaptive_avg_pool_op(shape, i)
        for j in range(1, 11):
            check_adaptive_avg_pool_op(shape, i, j)

@with_seed()
def test_bilinear_resize_op():
    def py_bilinear_resize(x, outputHeight, outputWidth):
        batch, channel, inputHeight, inputWidth = x.shape
        if outputHeight == inputHeight and outputWidth == inputWidth:
            return x
        y = np.empty([batch, channel, outputHeight, outputWidth])
        rheight = 1.0 * (inputHeight - 1) / (outputHeight - 1) if outputHeight > 1 else 0.0
        rwidth = 1.0 * (inputWidth - 1) / (outputWidth - 1) if outputWidth > 1 else 0.0
        for h2 in range(outputHeight):
            h1r = 1.0 * h2 * rheight
            h1 = int(np.floor(h1r))
            h1lambda = h1r - h1
            h1p = 1 if h1 < (inputHeight - 1) else 0
            for w2 in range(outputWidth):
                w1r = 1.0 * w2 * rwidth
                w1 = int(np.floor(w1r))
                w1lambda = w1r - w1
                w1p = 1 if w1 < (inputWidth - 1) else 0
                for b in range(batch):
                    for c in range(channel):
                        y[b][c][h2][w2] = (1-h1lambda)*((1-w1lambda)*x[b][c][h1][w1] + \
                            w1lambda*x[b][c][h1][w1+w1p]) + \
                            h1lambda*((1-w1lambda)*x[b][c][h1+h1p][w1] + \
                            w1lambda*x[b][c][h1+h1p][w1+w1p])
        return y
    def py_bilinear_resize_backward(x, incoming_grads, mode='size'):
        data1 = np.zeros_like(x)
        data2 = incoming_grads
        batchsize = data1.shape[0]
        channels = data1.shape[1]
        height1 = data1.shape[2]
        width1 = data1.shape[3]
        height2 = data2.shape[2]
        width2 = data2.shape[3]
        rheight = float(height1 - 1) / (height2 - 1) if (height2 > 1) else 0
        rwidth = float(width1 - 1) / (width2 - 1) if (width2 > 1) else 0
        # special case: just copy
        if height1 == height2 and width1 == width2:
            data1 += data2
            return [data1]
        for h2 in range(0, height2):
            for w2 in range(0, width2):
                h1r = rheight * h2
                h1 = int(h1r)
                h1p = 1 if (h1 < height1 - 1) else 0
                h1lambda = h1r - h1
                h0lambda = 1 - h1lambda
                #
                w1r = rwidth * w2
                w1 = int(w1r)
                w1p = 1 if (w1 < width1 - 1) else 0
                w1lambda = w1r - w1
                w0lambda = 1 - w1lambda
                #
                for n in range(0, batchsize):
                    for c in range(0, channels):
                        d2val = data2[n][c][h2][w2]
                        data1[n][c][h1][w1] += h0lambda * w0lambda * d2val
                        data1[n][c][h1][w1 + w1p] += h0lambda * w1lambda * d2val
                        data1[n][c][h1 + h1p][w1] += h1lambda * w0lambda * d2val
                        data1[n][c][h1 + h1p][w1 + w1p] += h1lambda * w1lambda * d2val
        if mode == 'like':
            return data1, np.zeros_like(incoming_grads)
        return [data1]
    def check_bilinear_resize_op(shape, height, width):
        x = mx.nd.random.uniform(shape=shape)
        y = mx.nd.contrib.BilinearResize2D(x, height=height, width=width)
        assert_almost_equal(y, py_bilinear_resize(x.asnumpy(), height, width))

        x_scale = width / shape[-1]
        y_scale = height / shape[-2]
        y = mx.nd.contrib.BilinearResize2D(x, scale_height=y_scale, scale_width=x_scale)
        assert_almost_equal(y.asnumpy(), py_bilinear_resize(x.asnumpy(), height, width))

    def check_bilinear_resize_align_corners_op():
        img_shape = [1, 1, 3, 2]
        data = [64, 32, 32, 64, 50, 100]
        target_height = 6
        target_width = 4
        expected_data = {}

        # align_corners = False
        expected_data[0] = [
            64.000, 56.000, 40.000, 32.000, 56.000, 52.000, 44.000, 40.000, 40.000, 44.000, 52.000, 56.000,
            36.500, 45.625, 63.875, 73.000, 45.500, 56.875, 79.625, 91.000, 50.000, 62.500, 87.500, 100.000
        ]

        # align_corners = True
        expected_data[1] = [
            64.000, 53.333, 42.667, 32.000, 51.200, 49.067, 46.933, 44.800, 38.400, 44.800, 51.200, 57.600,
            35.600, 47.467, 59.333, 71.200, 42.800, 57.067, 71.333, 85.600, 50.000, 66.667, 83.333, 100.000
        ]

        x = np.array(data, dtype=np.float32).reshape(img_shape)
        x_nd = mx.nd.array(x)

        y0 = np.array(expected_data[0]).reshape((1, 1, target_height, target_width))
        y0_nd = mx.nd.contrib.BilinearResize2D(x_nd, height=target_height, width=target_width, mode='size', align_corners=False)
        assert_almost_equal(y0, y0_nd.asnumpy(), atol=1e-3)

        y1 = np.array(expected_data[1]).reshape((1, 1, target_height, target_width))
        y1_nd = mx.nd.contrib.BilinearResize2D(x_nd, height=target_height, width=target_width, mode='size', align_corners=True)
        assert_almost_equal(y1, y1_nd.asnumpy(), atol=1e-3)

    def check_bilinear_resize_modes_op(shape, scale_height=None, scale_width=None, shape_1=None, mode=None):
        x = mx.nd.random.uniform(shape=shape)
        original_h = shape[2]
        original_w = shape[3]
        if mode == 'odd_scale':
            assert scale_height is not None and scale_width is not None
            new_h = int(original_h * scale_height) if (original_h % 2) == 0 else \
                int((original_h - 1) * scale_height) + 1
            new_w = int(original_w * scale_width) if (original_w % 2) == 0 \
                else int((original_w - 1) * scale_width) + 1
            y = mx.nd.contrib.BilinearResize2D(x, scale_height=scale_height,
                                               scale_width=scale_width,
                                               mode='odd_scale')
        elif mode == 'to_even_down':
            new_h = original_h if (original_h % 2) == 0 else original_h - 1
            new_w = original_w if (original_w % 2) == 0 else original_w - 1
            y = mx.nd.contrib.BilinearResize2D(x, mode='to_even_down')
        elif mode == 'to_even_up':
            new_h = original_h if (original_h % 2) == 0 else original_h + 1
            new_w = original_w if (original_w % 2) == 0 else original_w + 1
            y = mx.nd.contrib.BilinearResize2D(x, mode='to_even_up')
        elif mode == 'to_odd_down':
            new_h = original_h if (original_h % 2) == 1 else original_h - 1
            new_w = original_w if (original_w % 2) == 1 else original_w - 1
            y = mx.nd.contrib.BilinearResize2D(x, mode='to_odd_down')
        elif mode == 'to_odd_up':
            new_h = original_h if (original_h % 2) == 1 else original_h + 1
            new_w = original_w if (original_w % 2) == 1 else original_w + 1
            y = mx.nd.contrib.BilinearResize2D(x, mode='to_odd_up')
        elif mode == 'like':
            x_1 = mx.nd.random.uniform(shape=shape_1)
            new_h = x_1.shape[2]
            new_w = x_1.shape[3]
            y = mx.nd.contrib.BilinearResize2D(x, x_1, mode='like')
        new_shape_desired = np.array([shape[0], shape[1], new_h, new_w], dtype='int')
        new_shape_got = np.array(y.shape, dtype='int')
        data_sym = mx.sym.var('data')
        data_np = x.asnumpy()
        expected = py_bilinear_resize(data_np, new_h, new_w)
        out_grads = np.ones([shape[0], shape[1], new_h, new_w])
        expected_backward = py_bilinear_resize_backward(data_np, out_grads, mode)
        assert_array_equal(new_shape_desired, new_shape_got, "Desired and got shapes are not equal. {} vs {}".format(
            str(new_shape_desired.tolist()), str(new_shape_got.tolist())))
        assert_almost_equal(y.asnumpy(), expected, 1e-3, 0)
        if mode != 'like':
            resize_sym = mx.sym.contrib.BilinearResize2D(data_sym, None, scale_height=scale_height, scale_width=scale_width, mode=mode)
            check_symbolic_forward(resize_sym, [data_np], [expected], rtol=1e-3, atol=1e-5)
            check_symbolic_backward(resize_sym, [data_np], [out_grads], expected_backward, rtol=1e-3, atol=1e-5)
            check_numeric_gradient(resize_sym, [data_np], rtol=1e-2, atol=1e-4)
        else:
            data_sym_like = mx.sym.var('data_like')
            resize_sym = mx.sym.contrib.BilinearResize2D(data_sym, data_sym_like, mode=mode)
            date_np_like = x_1.asnumpy()
            check_symbolic_forward(resize_sym, [data_np, date_np_like], [expected], rtol=1e-3, atol=1e-5)
            check_symbolic_backward(resize_sym, [data_np, date_np_like], [out_grads], expected_backward, rtol=1e-3, atol=1e-5)
            check_numeric_gradient(resize_sym, [data_np, date_np_like], rtol=1e-2, atol=1e-4)

    shape = (2, 2, 10, 10)
    check_bilinear_resize_op(shape, 5, 5)
    check_bilinear_resize_op(shape, 10, 10)
    check_bilinear_resize_op(shape, 15, 15)
    check_bilinear_resize_op(shape, 3, 7)
    check_bilinear_resize_op(shape, 13, 17)
    shape = (2, 2, 20, 20)
    check_bilinear_resize_modes_op(shape, scale_height=0.5, scale_width=0.5, mode='odd_scale')
    check_bilinear_resize_modes_op(shape, scale_height=5, scale_width=10, mode='odd_scale')
    check_bilinear_resize_modes_op(shape, scale_height=0.1, scale_width=0.2, mode='odd_scale')
    check_bilinear_resize_modes_op(shape, mode='to_even_down')
    check_bilinear_resize_modes_op(shape, mode='to_even_up')
    check_bilinear_resize_modes_op(shape, mode='to_odd_down')
    check_bilinear_resize_modes_op(shape, mode='to_odd_up')
    shape = (2, 2, 21, 21)
    check_bilinear_resize_modes_op(shape, scale_height=0.5, scale_width=0.5, mode='odd_scale')
    check_bilinear_resize_modes_op(shape, scale_height=5, scale_width=10, mode='odd_scale')
    check_bilinear_resize_modes_op(shape, scale_height=0.1, scale_width=0.2, mode='odd_scale')
    check_bilinear_resize_modes_op(shape, mode='to_even_down')
    check_bilinear_resize_modes_op(shape, mode='to_even_up')
    check_bilinear_resize_modes_op(shape, mode='to_odd_down')
    check_bilinear_resize_modes_op(shape, mode='to_odd_up')
    shape_0 = (2, 2, 21, 21)
    shape_1 = (2, 2, 10, 10)
    check_bilinear_resize_modes_op(shape_0, shape_1=shape_1, mode='like')
    check_bilinear_resize_modes_op(shape_1, shape_1=shape_0, mode='like')
    check_bilinear_resize_align_corners_op()

def test_multi_proposal_op():
    # paramters
    feature_stride = 16
    scales = (8, 16, 32)
    ratios = (0.5, 1, 2)
    rpn_pre_nms_top_n = 12000
    rpn_post_nms_top_n = 2000
    threshold = 0.7
    rpn_min_size = 16

    batch_size = 20
    feat_len = (1000 + 15) // 16
    H, W = feat_len, feat_len
    num_anchors = len(scales) * len(ratios)
    count_anchors = H * W * num_anchors

    '''
    cls_prob: (batch_size, 2 * num_anchors, H, W)
    bbox_pred: (batch_size, 4 * num_anchors, H, W)
    im_info: (batch_size, 3)
    '''

    cls_prob = mx.nd.empty((batch_size, 2 * num_anchors, H, W), dtype = np.float32)
    bbox_pred = mx.nd.empty((batch_size, 4 * num_anchors, H, W), dtype = np.float32)
    im_info = mx.nd.empty((batch_size, 3), dtype = np.float32)

    cls_prob = mx.nd.array(np.random.random(cls_prob.shape))
    bbox_pred = mx.nd.array(np.random.random(bbox_pred.shape))

    for i in range(batch_size):
        im_size = np.random.randint(100, feat_len * feature_stride, size = (2,))
        im_scale = np.random.randint(70, 100) / 100.0
        im_info[i, :] = [im_size[0], im_size[1], im_scale]

    def get_sub(arr, i):
        new_shape = list(arr.shape)
        new_shape[0] = 1
        res = arr[i].reshape(new_shape)
        return res

    def check_forward(rpn_pre_nms_top_n, rpn_post_nms_top_n):
        single_proposal = []
        single_score = []
        for i in range(batch_size):
            rois, score = mx.nd.contrib.Proposal(
                    cls_prob = get_sub(cls_prob, i),
                    bbox_pred = get_sub(bbox_pred, i),
                    im_info = get_sub(im_info, i),
                    feature_stride = feature_stride,
                    scales = scales,
                    ratios = ratios,
                    rpn_pre_nms_top_n = rpn_pre_nms_top_n,
                    rpn_post_nms_top_n = rpn_post_nms_top_n,
                    threshold = threshold,
                    rpn_min_size = rpn_min_size, output_score = True)
            single_proposal.append(rois)
            single_score.append(score)

        multi_proposal, multi_score = mx.nd.contrib.MultiProposal(
                cls_prob = cls_prob,
                bbox_pred = bbox_pred,
                im_info = im_info,
                feature_stride = feature_stride,
                scales = scales,
                ratios = ratios,
                rpn_pre_nms_top_n = rpn_pre_nms_top_n,
                rpn_post_nms_top_n = rpn_post_nms_top_n,
                threshold = threshold,
                rpn_min_size = rpn_min_size, output_score = True)

        single_proposal = mx.nd.stack(*single_proposal).reshape(multi_proposal.shape)
        single_score = mx.nd.stack(*single_score).reshape(multi_score.shape)

        single_proposal_np = single_proposal.asnumpy()
        multi_proposal_np = multi_proposal.asnumpy()

        single_score_np = single_score.asnumpy()
        multi_score_np = multi_score.asnumpy()

        # check rois x1,y1,x2,y2
        assert np.allclose(single_proposal_np[:, 1:], multi_proposal_np[:, 1:])
        # check rois batch_idx
        for i in range(batch_size):
            start = i * rpn_post_nms_top_n
            end = start + rpn_post_nms_top_n
            assert (multi_proposal_np[start:end, 0] == i).all()
        # check score
        assert np.allclose(single_score_np, multi_score_np)

    def check_backward(rpn_pre_nms_top_n, rpn_post_nms_top_n):

        im_info_sym = mx.sym.Variable('im_info')
        cls_prob_sym = mx.sym.Variable('cls_prob')
        bbox_pred_sym = mx.sym.Variable('bbox_pred')

        sym = mx.sym.contrib.MultiProposal(
                cls_prob = cls_prob_sym,
                bbox_pred = bbox_pred_sym,
                im_info = im_info_sym,
                feature_stride = feature_stride,
                scales = scales,
                ratios = ratios,
                rpn_pre_nms_top_n = rpn_pre_nms_top_n,
                rpn_post_nms_top_n = rpn_post_nms_top_n,
                threshold = threshold,
                rpn_min_size = rpn_min_size, output_score = False)

        location = [cls_prob.asnumpy(), bbox_pred.asnumpy(), im_info.asnumpy()]

        expected = [np.zeros_like(e) for e in location]

        out_grads = [np.ones((rpn_post_nms_top_n, 5))]

        check_symbolic_backward(sym, location, out_grads, expected)

    check_forward(rpn_pre_nms_top_n, rpn_post_nms_top_n)
    check_forward(rpn_pre_nms_top_n, 1500)
    check_forward(1000, 500)
    check_backward(rpn_pre_nms_top_n, rpn_post_nms_top_n)

@with_seed()
def test_quadratic_function():
    def f(x, a, b, c):
        return a * x**2 + b * x + c

    a = np.random.random_sample()
    b = np.random.random_sample()
    c = np.random.random_sample()
    data = mx.symbol.Variable('data')
    quad_sym = mx.sym.contrib.quadratic(data=data, a=a, b=b, c=c)
    for dtype in [np.float16, np.float32, np.float64]:
        tol = 1e-2 if dtype is np.float16 else 1e-5
        for ndim in range(1, 6):
            shape = rand_shape_nd(ndim, 5)
            data_np = np.random.randn(*shape).astype(dtype)
            expected = f(data_np, a, b, c)
            backward_expected = 2 * a * data_np + b

            # check imperative forward
            output = mx.nd.contrib.quadratic(mx.nd.array(data_np), a=a, b=b, c=c)
            assert_almost_equal(output, expected, rtol=tol, atol=tol)
            # check forward
            check_symbolic_forward(quad_sym, [data_np], [expected], rtol=tol, atol=tol)
            # check backward
            check_symbolic_backward(quad_sym, [data_np], [np.ones(expected.shape)],
                                    [backward_expected], rtol=tol, atol=tol)
            # check backward using finite difference
            check_numeric_gradient(quad_sym, [data_np], atol=0.001)

def allclose_function(contexts):
    def getRandom(base, percent = 1.):
        return base * (1 + percent * (2 * np.random.random_sample() - 1.) / 100)

    title = 'exp'
    for ctx in contexts:
        title += ' cpu' if ctx == mx.cpu() else ' gpu'

    title += '        nElem     shape'
    num_ctx = len(contexts)
    result = [False, False]
    for dtype in [np.float16, np.float32, np.float64]:
        rtol = getRandom(1e-2 if dtype is np.float16 else 1e-5)
        atol = getRandom(1e-4 if dtype is np.float16 else 1e-7)
        print('\nnumpy.{}:  atol = {}  rtol = {}'.format(dtype.__name__, atol, rtol))
        print(title)
        for ndim in range(1, 10):
            shape = rand_shape_nd(ndim, 8)
            a_np = np.random.randn(*shape).astype(dtype)
            b_np = (a_np + np.random.randn(*shape).astype(dtype) / 10000000).astype(dtype)
            expected = np.allclose(a_np, b_np, rtol, atol)

            for n, ctx in enumerate(contexts):
                a_ctx = mx.nd.array(a_np, dtype = dtype, ctx=ctx)
                b_ctx = mx.nd.array(b_np, dtype = dtype, ctx=ctx)
                output = mx.nd.contrib.allclose(a_ctx, b_ctx, rtol=rtol, atol=atol)
                result[n] = output.asnumpy() == 1
                if expected != result[n]:
                    # Preparing the output of elements of the array, which are considered as "not close" AND
                    # corresponding elements of comparison CPU/GPU/Python vectors, which are considered as "close"
                    v_ctx = 'CPU' if ctx == mx.cpu() else 'GPU'
                    if expected:
                        v_cmp = 'Python'
                        a_b = a_ctx.asnumpy()
                        b_b = b_ctx.asnumpy()
                        a_g = np.asarray(a_np)
                        b_g = np.asarray(b_np)

                    else:
                        v_cmp = v_ctx
                        v_ctx = 'Python'
                        a_b = np.asarray(a_np)
                        b_b = np.asarray(b_np)
                        a_g = a_ctx.asnumpy()
                        b_g = b_ctx.asnumpy()

                    print('\n *** Violations found on %s, but not on %s side  ***' % (v_ctx, v_cmp))
                    frmt = "                 a[{0:d}]:                 b[{0:d}]:"  \
                           "          abs(a[{0:d}]-b[{0:d}]) - atol + rtol*abs(b[{0:d}]):"

                    # Define the indices of all violations and corresponding values of coordinates
                    bad_indexes = np.abs(a_b - b_b) >= atol + rtol * abs(b_b)
                    a_values = [a_b[bad_indexes], a_g[bad_indexes]]
                    b_values = [b_b[bad_indexes], b_g[bad_indexes]]
                    idx = np.asarray(np.where(bad_indexes == True))
                    idx = idx.reshape(1, idx.size)
                    idx_flat = np.asarray(np.where(bad_indexes.flatten() == True)).flatten()
                    for i in range(len(a_values[0])):
                        flat_idx = idx_flat[i]
                        print('{}:  index = {}   flat_index = {}'.format('%4d'%i, idx[i], flat_idx))
                        print(frmt.format(flat_idx))
                        for j in range(2):
                            diff = np.abs(a_values[j][i]-b_values[j][i]) - atol + rtol*abs(b_values[j][i])
                            print('{}:  {}  {}              {}'.format('%6s'%v_ctx, a_values[j][i], b_values[j][i], diff))


            if num_ctx == 1:
                print(' {0:d}   {1:d}    {2:10d}   {3:}'.format(expected, result[0], np.prod(shape), shape))
            else:
                print(' {0:d}   {1:d}   {2:d}    {3:10d}   {4:}'.format(expected, result[0], result[1], np.prod(shape), shape))

            if expected != result[0] or num_ctx > 1 and expected != result[1]:
                assert False

@with_seed()
def test_allclose_function():
    allclose_function([default_context()])

@with_seed()
def test_histogram():
    def f(x, bins=10, range=None):
        return np.histogram(x, bins, range=range)

    for ndim in range(1, 6):
        shape = rand_shape_nd(ndim)
        x = rand_ndarray(shape, stype='default', dtype=np.float64)
        mx_bins = mx.nd.array([-1.0, 0.5, 2.0, 4.5, 50.0], dtype=np.float64)
        np_bins = mx_bins.asnumpy()
        bin_cnt = random.randint(2, 10)
        bin_range = (-2.5, 2.5)
        mx_histo1, mx_bins1 = mx.nd.histogram(x, bins=bin_cnt, range=bin_range)
        np_histo1, np_bins1 = f(x.asnumpy(), bins=bin_cnt, range=bin_range)
        assert_almost_equal(mx_bins1, np_bins1)
        assert_almost_equal(mx_histo1, np_histo1, rtol=1e-3, atol=1e-5)
        mx_histo2, mx_bins2 = mx.nd.histogram(x, bins=mx_bins)
        np_histo2, np_bins2 = f(x.asnumpy(), bins=np_bins)
        assert_almost_equal(mx_histo2, np_histo2, rtol=1e-3, atol=1e-5)
        assert_almost_equal(mx_bins2, np_bins2, rtol=1e-3, atol=1e-5)

        data = mx.sym.Variable("data")

        bins = mx.sym.Variable("bins")
        histo1 = mx.sym.histogram(a=data, bins=bin_cnt, range=bin_range)
        histo2 = mx.sym.histogram(a=data, bins=bins)
        executor1 = histo1.bind(ctx=default_context(), args={"data" : x})
        executor1.forward(is_train=False)
        assert_almost_equal(np_histo1, executor1.outputs[0].asnumpy(), 0, 0, ("EXPECTED_histo1", "FORWARD_histo1"), equal_nan=False)
        executor2 = histo2.bind(ctx=default_context(), args={"data" : x, "bins" : mx_bins})
        executor2.forward(is_train=False)
        assert_almost_equal(np_histo2, executor2.outputs[0].asnumpy(), 0, 0, ("EXPECTED_histo2", "FORWARD_histo2"), equal_nan=False)


def test_op_output_names_monitor():
    def check_name(op_sym, expected_names):
        output_names = []

        def get_output_names_callback(name, arr):
            output_names.append(py_str(name))

        op_exe = op_sym.simple_bind(ctx=mx.current_context(), grad_req='null')
        op_exe.set_monitor_callback(get_output_names_callback, monitor_all=False)
        try:
            op_exe.forward()
            mx.nd.waitall()
        except mx.base.MXNetError:
            # skip errors since test is to check output names
            pass
        for output_name, expected_name in zip(output_names, expected_names):
            assert output_name == expected_name
    is_windows = sys.platform.startswith('win')
    if (is_windows):
        # Windows doesn't support set environment variable on the fly, so disable it for now
        pass
    else:
        # Disable subgraph in case subgraph will replace symbol
        os.environ['MXNET_SUBGRAPH_BACKEND'] = "NONE"
        data = mx.sym.Variable('data', shape=(10, 3, 10, 10))
        conv_sym = mx.sym.Convolution(data, kernel=(2, 2), num_filter=1, name='conv')
        check_name(conv_sym, ['conv_output'])

        deconv_sym = mx.sym.Deconvolution(data, kernel=(2, 2), num_filter=1, name='deconv')
        check_name(deconv_sym, ['deconv_output'])

        fc_sym = mx.sym.FullyConnected(data, num_hidden=10, name='fc')
        check_name(fc_sym, ['fc_output'])

        lrn_sym = mx.sym.LRN(data, nsize=1, name='lrn')
        check_name(lrn_sym, ['lrn_output', 'lrn_tmp_norm'])

        act_sym = mx.sym.Activation(data, act_type='relu', name='act')
        check_name(act_sym, ['act_output'])

        cc_sym = mx.sym.concat(data, data, dim=0, name='concat')
        check_name(cc_sym, ['concat_output'])

        sm_sym = mx.sym.softmax(data, name='softmax')
        check_name(sm_sym, ['softmax_output'])

        sa_sym = mx.sym.SoftmaxActivation(data, name='softmax')
        check_name(sa_sym, ['softmax_output'])

        us_sym = mx.sym.UpSampling(data, scale=2, sample_type='nearest',
                                name='upsampling')
        check_name(us_sym, ['upsampling_output'])

        us_sym = mx.sym.Pooling(data, kernel=(2, 2), pool_type='avg',
                                name='pooling')
        check_name(us_sym, ['pooling_output'])
        del os.environ['MXNET_SUBGRAPH_BACKEND']

def test_op_all_names_monitor():
    def check_name(op_sym, expected_names):
        output_names = []

        def get_output_names_callback(name, arr):
            output_names.append(py_str(name))

        op_exe = op_sym.simple_bind(ctx=mx.current_context(), grad_req='null')
        op_exe.set_monitor_callback(get_output_names_callback, monitor_all=True)
        try:
            op_exe.forward()
            mx.nd.waitall()
        except mx.base.MXNetError:
            # skip errors since test is to check all names
            pass
        for output_name, expected_name in zip(output_names, expected_names):
            assert output_name == expected_name
    is_windows = sys.platform.startswith('win')
    if (is_windows):
        # Windows doesn't support set environment variable on the fly, so disable it for now
        pass
    else:
        # Disable subgraph in case subgraph will replace symbol
        os.environ['MXNET_SUBGRAPH_BACKEND'] = "NONE"

        data = mx.sym.Variable('data', shape=(10, 3, 10, 10))
        conv_sym = mx.sym.Convolution(data, kernel=(2, 2), num_filter=1, name='conv')
        check_name(conv_sym, ['data', 'conv_data', 'conv_weight', 'conv_weight', 'conv_bias', 'conv_bias', 'conv_output'])

        deconv_sym = mx.sym.Deconvolution(data, kernel=(2, 2), num_filter=1, name='deconv')
        check_name(deconv_sym, ['data', 'deconv_data', 'deconv_weight', 'deconv_weight', 'deconv_output'])

        fc_sym = mx.sym.FullyConnected(data, num_hidden=10, name='fc')
        check_name(fc_sym, ['data', 'fc_data', 'fc_weight', 'fc_weight', 'fc_bias', 'fc_bias', 'fc_output'])

        lrn_sym = mx.sym.LRN(data, nsize=1, name='lrn')
        check_name(lrn_sym, ['data', 'lrn_data', 'lrn_output', 'lrn_tmp_norm'])

        act_sym = mx.sym.Activation(data, act_type='relu', name='act')
        check_name(act_sym, ['data', 'act_input0', 'act_output'])

        cc_sym = mx.sym.concat(data, data, dim=0, name='concat')
        check_name(cc_sym, ['data', 'concat_arg0', 'data', 'concat_arg1', 'concat_output'])

        sm_sym = mx.sym.softmax(data, name='softmax')
        check_name(sm_sym, ['data', 'softmax_data', 'softmax_output'])

        length = mx.sym.Variable("length", shape=(10, 10, 10))
        sm_sym = mx.sym.softmax(data, length, axis=1, use_length=True, name='softmax')
        check_name(sm_sym, ['data', 'softmax_data', 'length', 'softmax_length', 'softmax_output'])

        sa_sym = mx.sym.SoftmaxActivation(data, name='softmax')
        check_name(sa_sym, ['data', 'softmax_input0', 'softmax_output'])

        us_sym = mx.sym.UpSampling(data, scale=2, sample_type='nearest',
                                name='upsampling')
        check_name(us_sym, ['data', 'upsampling_arg0', 'upsampling_output'])

        us_sym = mx.sym.Pooling(data, kernel=(2, 2), pool_type='avg',
                                name='pooling')
        check_name(us_sym, ['data', 'pooling_data', 'pooling_output'])
        del os.environ['MXNET_SUBGRAPH_BACKEND']

@with_seed()
def test_monitor_with_variable_input_shape():
    output = {}

    def get_output_min_callback(name, arr):
        name = py_str(name)
        handle = ctypes.cast(arr, NDArrayHandle)
        arr = NDArray(handle, writable=False)
        min_val = mx.ndarray.min(arr).asscalar()
        if name in output:
            output[name] = min(output[name], min_val)
        else:
            output[name] = min_val

    def check_result(output, names):
        assert len(output) > 0
        for k, v in output.items():
            assert k in names
            assert v is not None

    is_windows = sys.platform.startswith('win')
    if (is_windows):
        # Windows doesn't support set environment variable on the fly, so disable it for now
        pass
    else:
        # Disable subgraph in case subgraph will replace symbol
        os.environ['MXNET_SUBGRAPH_BACKEND'] = "NONE"

        batch_size = 1
        op_name = 'conv'
        dshape = (batch_size, 3, 10, 10)
        data = mx.sym.Variable('data', shape=dshape)
        sym = mx.sym.Convolution(data, kernel=(1, 1), num_filter=1, name=op_name)

        mod = mx.module.Module(symbol=sym, label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', dshape)])
        mod.init_params()
        mod._exec_group.execs[0].set_monitor_callback(get_output_min_callback, monitor_all=True)

        new_dshape = dshape[:-1] + (dshape[-1] + 4,)
        new_data = mx.nd.random.uniform(shape=new_dshape)
        new_data = mx.io.NDArrayIter(data=new_data, batch_size=batch_size)
        new_data = DummyIter(new_data)

        for batch in new_data:
            mod.forward(data_batch=batch, is_train=False)
            mx.nd.waitall()
            break

        name_list = ['data', 'conv_data', 'conv_weight', 'conv_bias', 'conv_output']
        check_result(output, name_list)
        del os.environ['MXNET_SUBGRAPH_BACKEND']

@with_seed()
@unittest.skip("test fails intermittently. temporarily disabled till it gets fixed. tracked at https://github.com/apache/incubator-mxnet/issues/13915")
def test_activation():
    shapes = [(9,), (9, 10), (9, 10, 10), (1, 9, 10, 10)]
    dtype_l = [np.float64, np.float32, np.float16]
    rtol_l = [1e-7, 1e-6, 1e-2]
    atol_l = [1e-7, 1e-6, 1e-2]
    rtol_fd = 1e-5
    atol_fd = 1e-6
    num_eps = 1e-6
    unary_ops = {
        'relu': [lambda x: mx.sym.Activation(x, act_type='relu'),
                 lambda x: np.maximum(x, 0.),
                 lambda x: 1. * (x > 0.),
                 -5.0, 5.0],
        'sigmoid': [lambda x: mx.sym.Activation(x, act_type='sigmoid'),
                    lambda x: 1. / (np.exp(-x) + 1.),
                    lambda x: 1. / (np.exp(-x) + 1.) / (np.exp(x) + 1.),
                    -3.0, 3.0],
        'tanh': [lambda x: mx.sym.Activation(x, act_type='tanh'),
                 lambda x: np.tanh(x),
                 lambda x: 1. - np.tanh(x) ** 2,
                 -4.0, 4.0],
        'softrelu': [lambda x: mx.sym.Activation(x, act_type='softrelu'),
                    lambda x: np.log(1. + np.exp(x)),
                    lambda x: 1. - 1 / (1 + np.exp(x)),
                    -3.0, 3.0],
        'softsign': [lambda x: mx.sym.Activation(x, act_type='softsign'),
                     lambda x: x / (1. + np.abs(x)),
                     lambda x: 1. / np.square(1. + np.abs(x)),
                     -3.0, 3.0],
    }
    # Loop over operators
    for name, op in unary_ops.items():
        # Loop over shapes
        for shape in shapes:
            # Loop over dtype's
            for ind in range(len(dtype_l)):
                dtype = dtype_l[ind]
                rtol = rtol_l[ind]
                atol = atol_l[ind]
                compare_forw_backw_unary_op(
                    name, op[0], op[1], op[2], shape, op[3], op[4], rtol, atol,
                    dtype)
            # Finite difference testing
            finite_diff_unary_op(
                name, op[0], shape, op[3], op[4], rtol_fd, atol_fd, num_eps)

@with_seed()
def test_ravel():
    # be aware that check_symbolic_forward will use float type internally
    # for the arrays and that limits the representable flat index range.
    # Taking dim==4 and a range of [0,..,100] for the data can already
    # cause precision issues and break this test.
    for dim in [1, 2, 3, 4]:
      data = np.random.randint(50, size=(dim, 500))
      shape = tuple(np.add(np.amax(data, axis=1), [1]))
      a = mx.sym.Variable('a')
      ravel_npy = np.ravel_multi_index(data, shape)
      b = mx.sym.ravel_multi_index(a, shape=shape)
      check_symbolic_forward(b, location={'a': data}, expected=[ravel_npy])
      c = mx.sym.unravel_index(a, shape=shape)
      check_symbolic_forward(c, location={'a': ravel_npy}, expected=[data])
      # Test with leading dimension set to -1.
      shape2 = shape
      shape2 = (-1,)+shape[1:]
      b = mx.sym.ravel_multi_index(a, shape=shape2)
      check_symbolic_forward(b, location={'a': data}, expected=[ravel_npy])
      c = mx.sym.unravel_index(a, shape=shape2)
      check_symbolic_forward(c, location={'a': ravel_npy}, expected=[data])

def test_context_num_gpus():
    try:
        # Note: the test is run both on GPU and CPU hosts, so that we can not assert
        # on a specific number here.
        assert mx.context.num_gpus() >= 0
    except mx.MXNetError as e:
        # Note: On a CPU only host CUDA sometimes is not able to determine the number
        # of GPUs
        if str(e).find("CUDA") == -1:
            raise e


@with_seed()
def test_op_roi_align():
    T = np.float32

    def assert_same_dtype(dtype_a, dtype_b):
        '''
        Assert whether the two data type are the same
        Parameters
        ----------
        dtype_a, dtype_b: type
            Input data types to compare
        '''
        assert dtype_a == dtype_b,\
            TypeError('Unmatched data types: %s vs %s' % (dtype_a, dtype_b))

    def bilinear_interpolate(bottom, height, width, y, x):
        if y < -1.0 or y > height or x < -1.0 or x > width:
            return T(0.0), []
        x = T(max(0.0, x))
        y = T(max(0.0, y))
        x_low = int(x)
        y_low = int(y)
        if x_low >= width - 1:
            x_low = x_high = width - 1
            x = T(x_low)
        else:
            x_high = x_low + 1
        if y_low >= height - 1:
            y_low = y_high = height - 1
            y = T(y_low)
        else:
            y_high = y_low + 1
        ly = y - T(y_low)
        lx = x - T(x_low)
        hy = T(1.0) - ly
        hx = T(1.0) - lx
        v1 = bottom[y_low, x_low]
        v2 = bottom[y_low, x_high]
        v3 = bottom[y_high, x_low]
        v4 = bottom[y_high, x_high]
        w1 = hy * hx
        w2 = hy * lx
        w3 = ly * hx
        w4 = ly * lx
        assert_same_dtype(w1.dtype, T)
        assert_same_dtype(w2.dtype, T)
        assert_same_dtype(w3.dtype, T)
        assert_same_dtype(w4.dtype, T)
        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
        assert_same_dtype(val.dtype, T)
        grad = [(y_low, x_low, w1), (y_low, x_high, w2),
                (y_high, x_low, w3), (y_high, x_high, w4)
                ]
        return val, grad

    def roialign_forward_backward(data, rois, pooled_size, spatial_scale, sampling_ratio,
                                  position_sensitive, dy):
        N, C, H, W = data.shape
        R = rois.shape[0]
        PH, PW = pooled_size
        assert rois.ndim == 2,\
            ValueError(
                'The ndim of rois should be 2 rather than %d' % rois.ndim)
        assert rois.shape[1] == 5,\
            ValueError(
                'The length of the axis 1 of rois should be 5 rather than %d' % rois.shape[1])
        assert_same_dtype(data.dtype, T)
        assert_same_dtype(rois.dtype, T)

        C_out = C // PH // PW if position_sensitive else C
        out = np.zeros((R, C_out, PH, PW), dtype=T)
        dx = np.zeros_like(data)
        drois = np.zeros_like(rois)

        for r in range(R):
            batch_ind = int(rois[r, 0])
            sw, sh, ew, eh = rois[r, 1:5] * T(spatial_scale)
            roi_w = T(max(ew - sw, 1.0))
            roi_h = T(max(eh - sh, 1.0))
            bin_h = roi_h / T(PH)
            bin_w = roi_w / T(PW)
            bdata = data[batch_ind]
            if sampling_ratio > 0:
                roi_bin_grid_h = roi_bin_grid_w = sampling_ratio
            else:
                roi_bin_grid_h = int(np.ceil(roi_h / T(PH)))
                roi_bin_grid_w = int(np.ceil(roi_w / T(PW)))
            count = T(roi_bin_grid_h * roi_bin_grid_w)
            for c in range(C_out):
                for ph in range(PH):
                    for pw in range(PW):
                        val = T(0.0)
                        c_in = c * PH * PW + ph * PW + pw if position_sensitive else c
                        for iy in range(roi_bin_grid_h):
                            y = sh + T(ph) * bin_h + (T(iy) + T(0.5)) * \
                                bin_h / T(roi_bin_grid_h)
                            for ix in range(roi_bin_grid_w):
                                x = sw + T(pw) * bin_w + (T(ix) + T(0.5)) * \
                                    bin_w / T(roi_bin_grid_w)
                                v, g = bilinear_interpolate(
                                    bdata[c_in], H, W, y, x)
                                assert_same_dtype(v.dtype, T)
                                val += v
                                # compute grad
                                for qy, qx, qw in g:
                                    assert_same_dtype(qw.dtype, T)
                                    dx[batch_ind, c_in, qy, qx] += dy[r,
                                                                      c, ph, pw] * qw / count
                        out[r, c, ph, pw] = val / count
        assert_same_dtype(out.dtype, T)
        return out, [dx, drois]

    def test_roi_align_value(sampling_ratio=0, position_sensitive=False):
        ctx = default_context()
        dtype = np.float32
        dlen = 224
        N, C, H, W = 5, 3, 16, 16
        R = 7
        pooled_size = (3, 4)
        C = C * pooled_size[0] * pooled_size[1] if position_sensitive else C
        spatial_scale = H * 1.0 / dlen
        data = mx.nd.array(
            np.arange(N * C * W * H).reshape((N, C, H, W)), ctx=ctx, dtype=dtype)
        center_xy = mx.nd.random.uniform(0, dlen, (R, 2), ctx=ctx, dtype=dtype)
        wh = mx.nd.random.uniform(0, dlen, (R, 2), ctx=ctx, dtype=dtype)
        batch_ind = mx.nd.array(np.random.randint(0, N, size=(R, 1)), ctx=ctx)
        pos = mx.nd.concat(center_xy - wh / 2, center_xy + wh / 2, dim=1)
        rois = mx.nd.concat(batch_ind, pos, dim=1)

        data.attach_grad()
        rois.attach_grad()
        with mx.autograd.record():
            output = mx.nd.contrib.ROIAlign(data, rois, pooled_size=pooled_size,
                                            spatial_scale=spatial_scale, sample_ratio=sampling_ratio,
                                            position_sensitive=position_sensitive)
        C_out = C // pooled_size[0] // pooled_size[1] if position_sensitive else C
        dy = mx.nd.random.uniform(-1, 1, (R, C_out) +
                                  pooled_size, ctx=ctx, dtype=dtype)
        output.backward(dy)
        real_output, [dx, drois] = roialign_forward_backward(data.asnumpy(), rois.asnumpy(), pooled_size,
                                                             spatial_scale, sampling_ratio,
                                                             position_sensitive, dy.asnumpy())

        assert_almost_equal(output, real_output, atol=1e-3)
        assert_almost_equal(data.grad, dx, atol=1e-3)
        assert_almost_equal(rois.grad, drois, atol=1e-3)

    # modified from test_roipooling()
    def test_roi_align_autograd(sampling_ratio=0):
        ctx = default_context()
        data = mx.symbol.Variable(name='data')
        rois = mx.symbol.Variable(name='rois')
        test = mx.symbol.contrib.ROIAlign(data=data, rois=rois, pooled_size=(4, 4), spatial_scale=1,
                                          sample_ratio=sampling_ratio)

        x1 = np.random.rand(4, 1, 12, 12).astype('float64')
        x2 = np.array([[0, 1.1, 1.1, 6.2, 6.2], [2, 6.1, 2.1, 8.2, 11.2],
                       [1, 3.1, 1.1, 5.2, 10.2]], dtype='float64')

        check_numeric_gradient(sym=test, location=[x1, x2],
                               grad_nodes={'data': 'write', 'rois': 'null'},
                               numeric_eps=1e-4, rtol=1e-1, atol=1e-4, ctx=ctx)
        check_numeric_gradient(sym=test, location=[x1, x2],
                               grad_nodes={'data': 'add', 'rois': 'null'},
                               numeric_eps=1e-4, rtol=1e-1, atol=1e-4, ctx=ctx)

    test_roi_align_value()
    test_roi_align_value(sampling_ratio=2)
    test_roi_align_value(position_sensitive=True)
    test_roi_align_autograd()

@with_seed()
def test_op_rroi_align():
    T = np.float32

    def assert_same_dtype(dtype_a, dtype_b):
        '''
        Assert whether the two data type are the same
        Parameters
        ----------
        dtype_a, dtype_b: type
            Input data types to compare
        '''
        assert dtype_a == dtype_b,\
            TypeError('Unmatched data types: %s vs %s' % (dtype_a, dtype_b))

    def bilinear_interpolate(bottom, height, width, y, x):
        if y < -1.0 or y > height or x < -1.0 or x > width:
            return T(0.0)
        x = T(max(0.0, x))
        y = T(max(0.0, y))
        x_low = int(x)
        y_low = int(y)
        if x_low >= width - 1:
            x_low = x_high = width - 1
            x = T(x_low)
        else:
            x_high = x_low + 1
        if y_low >= height - 1:
            y_low = y_high = height - 1
            y = T(y_low)
        else:
            y_high = y_low + 1
        ly = y - T(y_low)
        lx = x - T(x_low)
        hy = T(1.0) - ly
        hx = T(1.0) - lx
        v1 = bottom[y_low, x_low]
        v2 = bottom[y_low, x_high]
        v3 = bottom[y_high, x_low]
        v4 = bottom[y_high, x_high]
        w1 = hy * hx
        w2 = hy * lx
        w3 = ly * hx
        w4 = ly * lx
        assert_same_dtype(w1.dtype, T)
        assert_same_dtype(w2.dtype, T)
        assert_same_dtype(w3.dtype, T)
        assert_same_dtype(w4.dtype, T)
        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
        assert_same_dtype(val.dtype, T)

        return val

    def rroialign_forward(data, rois, pooled_size, spatial_scale, sampling_ratio):
        N, C, H, W = data.shape
        R = rois.shape[0]
        PH, PW = pooled_size
        assert rois.ndim == 2,\
            ValueError(
                'The ndim of rois should be 2 rather than %d' % rois.ndim)
        assert rois.shape[1] == 6,\
            ValueError(
                'The length of the axis 1 of rois should be 6 rather than %d' % rois.shape[1])
        assert_same_dtype(data.dtype, T)
        assert_same_dtype(rois.dtype, T)

        out = np.zeros((R, C, PH, PW), dtype=T)

        for r in range(R):
            batch_ind = int(rois[r, 0])
            roi_center_w, roi_center_h, roi_w, roi_h = rois[r, 1:5] * T(spatial_scale)
            roi_theta = T(rois[r,5]  * np.pi / 180.0)
            roi_w = T(max(roi_w, 1.0))
            roi_h = T(max(roi_h, 1.0))
            bin_h = roi_h / T(PH)
            bin_w = roi_w / T(PW)
            bdata = data[batch_ind]
            if sampling_ratio > 0:
                roi_bin_grid_h = roi_bin_grid_w = sampling_ratio
            else:
                roi_bin_grid_h = int(np.ceil(roi_h / T(PH)))
                roi_bin_grid_w = int(np.ceil(roi_w / T(PW)))
            count = T(roi_bin_grid_h * roi_bin_grid_w)
            roi_start_h = T(-roi_h / 2.0)
            roi_start_w = T(-roi_w / 2.0)
            for c in range(C):
                for ph in range(PH):
                    for pw in range(PW):
                        val = T(0.0)
                        for iy in range(roi_bin_grid_h):
                            yy = roi_start_h + T(ph) * bin_h + (T(iy) + T(0.5)) * \
                                bin_h / T(roi_bin_grid_h)
                            for ix in range(roi_bin_grid_w):
                                xx = roi_start_w + T(pw) * bin_w + (T(ix) + T(0.5)) * \
                                    bin_w / T(roi_bin_grid_w)
                                x = xx * np.cos(roi_theta, dtype=T) + yy * np.sin(roi_theta, dtype=T) + roi_center_w
                                y = yy * np.cos(roi_theta, dtype=T) - xx * np.sin(roi_theta, dtype=T) + roi_center_h
                                v = bilinear_interpolate(
                                    bdata[c], H, W, y, x)
                                assert_same_dtype(v.dtype, T)
                                val += v

                        out[r, c, ph, pw] = val / count
        assert_same_dtype(out.dtype, T)

        return out

    def test_rroi_align_value(sampling_ratio=-1):
        ctx = default_context()
        if ctx.device_type == 'gpu':
            print('skipped testing rroi align for gpu since it is not supported yet')
            return

        dtype = np.float32
        dlen = 224
        N, C, H, W = 5, 3, 16, 16
        R = 7
        pooled_size = (3, 4)
        spatial_scale = H * 1.0 / dlen
        data = mx.nd.array(
            np.arange(N * C * W * H).reshape((N, C, H, W)), ctx=ctx, dtype=dtype)
        center_xy = mx.nd.random.uniform(0, dlen, (R, 2), ctx=ctx, dtype=dtype)
        wh = mx.nd.random.uniform(0, dlen, (R, 2), ctx=ctx, dtype=dtype)
        theta = mx.nd.random.uniform(0, 180, (R,1), ctx=ctx, dtype=dtype)
        batch_ind = mx.nd.array(np.random.randint(0, N, size=(R, 1)), ctx=ctx)
        pos = mx.nd.concat(center_xy, wh, theta, dim=1)
        rois = mx.nd.concat(batch_ind, pos, dim=1)

        output = mx.nd.contrib.RROIAlign(data, rois, pooled_size=pooled_size,
                                        spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
        real_output = rroialign_forward(data.asnumpy(), rois.asnumpy(), pooled_size,
                                        spatial_scale, sampling_ratio)

        assert_almost_equal(output.asnumpy(), real_output, atol=1e-3)

    test_rroi_align_value()
    test_rroi_align_value(sampling_ratio=2)

@with_seed()
def test_diag():

    # Test 2d input
    h = np.random.randint(2,9)
    w = np.random.randint(2,9)
    a_np = np.random.random((h, w)).astype(np.float32)
    a = mx.nd.array(a_np).astype('float32')

    for k in [0, 1, -1, np.random.randint(-min(h,w) + 1, min(h,w))]:
        assert_almost_equal(mx.nd.diag(a, k=k), np.diag(a_np, k=k))

    # invalid k
    k = max(h,w) + 1
    assertRaises(MXNetError, mx.nd.diag, a, k=k)

    # Test 2d backward, k=0
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data)
    check_numeric_gradient(diag_sym, [a_np])

    # Test 2d backward, k=1
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data, k=1)
    check_numeric_gradient(diag_sym, [a_np])

    # Test 2d backward, k=-1
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data, k=-1)
    check_numeric_gradient(diag_sym, [a_np])

    # test 1d input
    d = np.random.randint(2,9)
    a_np = np.random.random((d))
    a = mx.nd.array(a_np)

    # k is random
    k = np.random.randint(-d,d)
    assert_almost_equal(mx.nd.diag(a, k=k), np.diag(a_np, k=k))

    # Test 2d backward, k=0
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data)
    check_numeric_gradient(diag_sym, [a_np])

    # Test 2d backward, k=1
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data, k=1)
    check_numeric_gradient(diag_sym, [a_np])

    # Test 2d backward, k=-1
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data, k=-1)
    check_numeric_gradient(diag_sym, [a_np])

    # Test 4d input
    x1 = np.random.randint(3,9)
    x2 = np.random.randint(3,9)
    x3 = np.random.randint(3,9)
    x4 = np.random.randint(3,9)
    a_np = np.random.random((x1, x2, x3, x4)).astype(np.float32)
    a = mx.nd.array(a_np).astype('float32')

    # k = 0, axis1=0, axis2=1
    r = mx.nd.diag(data=a, k=0, axis1=0, axis2=1)
    assert_almost_equal(r, np.diagonal(a_np, offset=0, axis1=0, axis2=1))

    # k = 1, axis1=1, axis2=0
    r = mx.nd.diag(data=a, k=1, axis1=1, axis2=0)
    assert_almost_equal(r, np.diagonal(a_np, offset=1, axis1=1, axis2=0))

    # k = -1 axis1=1, axis3=3
    r = mx.nd.diag(data=a, k=-1, axis1=1, axis2=3)
    assert_almost_equal(r, np.diagonal(a_np, offset=-1, axis1=1, axis2=3))

    # k = 2, axis1=-2, axis2=0
    r = mx.nd.diag(data=a, k=2, axis1=-2, axis2=0)
    assert_almost_equal(r, np.diagonal(a_np, offset=2, axis1=-2, axis2=0))

    # Test 4d backward, k=0, axis1=3, axis2=0
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data, k=0, axis1=3, axis2=0)
    check_numeric_gradient(diag_sym, [a_np])

    # Test 4d backward, k=1, axis1=1, axis2=2
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data, k=1, axis1=1, axis2=2)
    check_numeric_gradient(diag_sym, [a_np])

    # Test 4d backward, k=-1, axis1=2, axis2=0
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data, k=-1, axis1=2, axis2=0)
    check_numeric_gradient(diag_sym, [a_np])

    # Test 4d backward, k=-2, axis1=1, axis2=-1
    data = mx.sym.Variable('data')
    diag_sym = mx.sym.diag(data=data, k=-2, axis1=1, axis2=-1)
    check_numeric_gradient(diag_sym, [a_np])

@with_seed()
def test_depthtospace():
    def f(x, blocksize):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
        tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
        y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
        return y

    block = random.randint(2, 4)
    rand_mul1 = random.randint(1, 4)
    n = random.randint(1, 5)
    c = block * block * rand_mul1
    h = random.randint(1, 5)
    w = random.randint(1, 5)
    shape_inp = (n, c, h, w)
    data = rand_ndarray(shape_inp, 'default')
    data_np = data.asnumpy()
    expected = f(data_np, block)
    output = mx.nd.depth_to_space(data, block)
    assert_almost_equal(output, expected, atol=1e-3, rtol=1e-3)

    shape_out = (n, c // (block ** 2), h * block, w * block)
    data = mx.sym.Variable('data')
    dts_sym = mx.sym.depth_to_space(data, block)
    check_numeric_gradient(dts_sym, [np.ones(shape_inp)])

    check_symbolic_forward(dts_sym, [data_np], [expected])
    check_symbolic_backward(dts_sym, [data_np], [np.ones(shape_out)], [np.ones(shape_inp)])

    def test_invalid_depth_dim():
        invalid_shape_inp = (n, block - 1, h, w)
        data = rand_ndarray(invalid_shape_inp, 'default')
        assertRaises(MXNetError, mx.nd.depth_to_space, data, block)

    def test_invalid_space_dim():
        invalid_shape_inp = (n, block ** 2, 0, block + 1)
        data = rand_ndarray(invalid_shape_inp, 'default')
        assertRaises(MXNetError, mx.nd.depth_to_space, data, block)

    def test_invalid_block_size():
        block = 0
        invalid_shape_inp = (n , c, h, w)
        data = rand_ndarray(invalid_shape_inp, 'default')
        assertRaises(MXNetError, mx.nd.depth_to_space, data, block)

    test_invalid_depth_dim()
    test_invalid_space_dim()
    test_invalid_block_size()

@with_seed()
def test_spacetodepth():
    def f(x, blocksize):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        tmp = np.reshape(x, [b, c, h // blocksize, blocksize, w // blocksize, blocksize])
        tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
        y = np.reshape(tmp, [b, c * (blocksize**2), h // blocksize, w // blocksize])
        return y

    block = random.randint(2, 4)
    rand_mul1 = random.randint(1, 4)
    rand_mul2 = random.randint(1, 4)
    n = random.randint(1, 5)
    c = random.randint(1, 5)
    h = block * rand_mul1
    w = block * rand_mul2
    shape_inp = (n, c, h, w)
    data = rand_ndarray(shape_inp, 'default')
    data_np = data.asnumpy()
    expected = f(data_np, block)
    output = mx.nd.space_to_depth(data, block)
    assert_almost_equal(output, expected, atol=1e-3, rtol=1e-3)

    shape_out = (n, c * (block ** 2), h // block, w // block)
    data = mx.sym.Variable('data')
    dts_sym = mx.sym.space_to_depth(data, block)
    check_numeric_gradient(dts_sym, [np.ones(shape_inp)])

    check_symbolic_forward(dts_sym, [data_np], [expected])
    check_symbolic_backward(dts_sym, [data_np], [np.ones(shape_out)], [np.ones(shape_inp)])

    def test_invalid_space_dim():
        invalid_shape_inp = (n , c, block - 1, w)
        data = rand_ndarray(invalid_shape_inp, 'default')
        assertRaises(MXNetError, mx.nd.space_to_depth, data, block)

    def test_invalid_block_size():
        block = 0
        invalid_shape_inp = (n, c, h, w)
        data = rand_ndarray(invalid_shape_inp, 'default')
        assertRaises(MXNetError, mx.nd.space_to_depth, data, block)

    def test_invalid_depth_dim():
        invalid_shape_inp = (n, 0, h, w)
        data = rand_ndarray(invalid_shape_inp, 'default')
        assertRaises(MXNetError, mx.nd.space_to_depth, data, block)

    test_invalid_space_dim()
    test_invalid_block_size()
    test_invalid_depth_dim()


@with_seed()
def test_softmax_cross_entropy():
    def f_sm_ce(data, label):
        return np.sum(-np.log(data) * label)

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    sym = mx.sym.softmax_cross_entropy(data=data, label=label)
    num_labels = random.randint(100, 200)
    batch_size = random.randint(100, 200)
    np_data = rand_ndarray((batch_size, num_labels), stype='default').asnumpy()
    np_sm = np_softmax(np_data)
    np_label = np.random.randint(0, num_labels, (batch_size, ))
    np_one_hot_label = np.zeros((batch_size, num_labels))
    np_one_hot_label[np.arange(batch_size), np_label] = 1.
    check_symbolic_forward(sym, {'data' : np_data, 'label' : np_label}, [np.array([f_sm_ce(np_sm, np_one_hot_label)])], rtol=1e-3, atol=1e-5)


@with_seed()
def test_split_v2():
    dim = random.randint(2, 6)
    shape = rand_shape_nd(dim)
    axis = random.randint(-dim, dim-1)
    axis_size = shape[axis]
    samples = random.randint(0, axis_size - 1)
    indices = sorted(random.sample([i for i in range(1, axis_size)], samples))
    indices = tuple(indices)
    mx_data = rand_ndarray(shape)
    np_data = mx_data.asnumpy()
    np_out = np.split(np_data, indices_or_sections=indices, axis=axis)
    data = mx.sym.Variable("data")
    sym = mx.sym.split_v2(data, indices_or_sections=indices, axis=axis)
    check_symbolic_forward(sym, {"data": mx_data}, np_out, rtol=1e-3, atol=1e-5)
    out_grad = [np.ones(arr.shape) for arr in np_out]
    check_symbolic_backward(sym, {"data": mx_data}, out_grad, [np.concatenate(out_grad, axis=axis)])


@with_seed()
def test_moments():
    dim = random.randint(2, 5)
    shape = rand_shape_nd(dim, dim=5)
    axes = [i for i in range(dim)]
    test_dims = random.sample(axes, random.randint(1, dim))
    test_axes = tuple(sorted(test_dims))
    np_a = np.random.uniform(-1.0, 1.0, shape)
    a = mx.nd.array(np_a)
    for keepdims in [True, False]:
        eps = 1e-3
        np_a[abs(np_a) < eps] = 2 * eps
        np_mean = np.mean(np_a, axis=test_axes, keepdims=keepdims)
        np_var = np.var(np_a, axis=test_axes, keepdims=keepdims)
        mx_mean, mx_var = mx.nd.moments(a, keepdims=keepdims, axes=test_axes)
        N = np_a.size / np_mean.size
        mx_sym = mx.sym.Variable("data")
        mx_moments = mx.sym.moments(mx_sym, axes=test_axes, keepdims=keepdims)
        mx_test_sym = mx.sym.elemwise_add(mx_moments[0], mx_moments[1])
        if len(np_mean.shape) == 0:
            np_mean = np_mean.reshape(mx_mean.shape)
            np_var = np_var.reshape(mx_var.shape)
        assert np_mean.shape == mx_mean.shape
        assert np_var.shape == mx_var.shape
        check_symbolic_forward(mx_test_sym, [np_a], [np_mean + np_var], rtol=1e-3, atol=1e-5)
        check_numeric_gradient(mx_test_sym, [np_a], numeric_eps=eps, rtol=1e-2, atol=2e-4)


@with_seed()
def test_invalid_kernel_size():
    invalid_kernel_size = 28
    assert_exception(
        mx.nd.Correlation,
        MXNetError,
        mx.nd.array(np.random.rand(1, 1, 28, 28)),
        mx.nd.array(np.random.rand(1, 1, 28, 28)),
        kernel_size=invalid_kernel_size)

@with_seed()
def test_valid_kernel_size():
    valid_kernel_size = 9
    mx.nd.Correlation(
        mx.nd.array(np.random.rand(1, 1, 28, 28)),
        mx.nd.array(np.random.rand(1, 1, 28, 28)),
        kernel_size=valid_kernel_size)

@with_seed()
def test_valid_max_pooling_pad_type_same():
    import math
    input_data = mx.nd.array(np.random.rand(1,1,10))
    stride = 2
    kernel = 2
    output_data=mx.nd.Pooling(
        input_data,
        kernel=kernel,
        stride=stride,
        pad=(0,0,0),
        pool_type='max',
        name='pooling',
        pooling_convention="same")
    assert(math.ceil(input_data.shape[2]/stride) == output_data.shape[2])

@with_seed()
def test_invalid_max_pooling_pad_type_same():
    import math
    input_data = mx.nd.array(np.random.rand(1,1,10))
    stride = 2
    kernel = 2
    pad = 2
    assert_exception(
        mx.nd.Pooling,
        MXNetError,
        input_data,
        stride=stride,
        kernel=kernel,
        pad=pad,
        pool_type='max',
        name='pooling',
        pooling_convention="same")


@with_seed()
def test_image_normalize():
    # Part 1 - Test 3D input with 3D mean/std
    shape_3d = (3, 28, 28)
    mean = (0, 1, 2)
    std = (3, 2, 1)

    data_in_3d = mx.nd.random.uniform(0, 1, shape_3d)
    data_expected_3d = data_in_3d.asnumpy()
    data_expected_3d[:][:][0] = data_expected_3d[:][:][0] / 3.0
    data_expected_3d[:][:][1] = (data_expected_3d[:][:][1] - 1.0) / 2.0
    data_expected_3d[:][:][2] = data_expected_3d[:][:][2] - 2.0

    data = mx.symbol.Variable('data')
    img_norm_sym = mx.sym.image.normalize(data=data, mean=mean, std=std)

    # check forward
    check_symbolic_forward(img_norm_sym, [data_in_3d], [data_expected_3d],
                           rtol=1e-5, atol=1e-5)

    # Gradient is 1/std_dev
    grad_expected_3d = np.ones(shape_3d)
    grad_expected_3d[:][:][0] = 1 / 3.0
    grad_expected_3d[:][:][1] = 1 / 2.0
    grad_expected_3d[:][:][2] = 1 / 1.0

    # check backward
    check_symbolic_backward(img_norm_sym, location=[data_in_3d], out_grads=[mx.nd.ones(shape_3d)],
                            expected=[grad_expected_3d], rtol=1e-5, atol=1e-5)

    # check backward using finite difference
    check_numeric_gradient(img_norm_sym, [data_in_3d], atol=0.001)

    # Part 2 - Test 4D input with 3D mean/std
    shape_4d = (2, 3, 28, 28)

    data_in_4d = mx.nd.random.uniform(0, 1, shape_4d)
    data_expected_4d = data_in_4d.asnumpy()
    data_expected_4d[0][:][:][0] = data_expected_4d[0][:][:][0] / 3.0
    data_expected_4d[0][:][:][1] = (data_expected_4d[0][:][:][1] - 1.0) / 2.0
    data_expected_4d[0][:][:][2] = data_expected_4d[0][:][:][2] - 2.0
    data_expected_4d[1][:][:][0] = data_expected_4d[1][:][:][0] / 3.0
    data_expected_4d[1][:][:][1] = (data_expected_4d[1][:][:][1] - 1.0) / 2.0
    data_expected_4d[1][:][:][2] = data_expected_4d[1][:][:][2] - 2.0

    # check forward
    check_symbolic_forward(img_norm_sym, [data_in_4d], [data_expected_4d],
                           rtol=1e-5, atol=1e-5)

    # Gradient is 1/std_dev
    grad_expected_4d = np.ones(shape_4d)
    grad_expected_4d[0][:][:][0] = 1 / 3.0
    grad_expected_4d[0][:][:][1] = 1 / 2.0
    grad_expected_4d[0][:][:][2] = 1 / 1.0
    grad_expected_4d[1][:][:][0] = 1 / 3.0
    grad_expected_4d[1][:][:][1] = 1 / 2.0
    grad_expected_4d[1][:][:][2] = 1 / 1.0

    # check backward
    check_symbolic_backward(img_norm_sym, location=[data_in_4d], out_grads=[mx.nd.ones(shape_4d)],
                            expected=[grad_expected_4d], rtol=1e-5, atol=1e-5)

    # check backward using finite difference
    check_numeric_gradient(img_norm_sym, [data_in_4d], atol=0.001)

    # Part 3 - Test 3D input with scalar mean/std
    shape_3d = (3, 28, 28)
    mean = 1.0
    std = 2.0

    data_in_3d = mx.nd.random.uniform(0, 1, shape_3d)
    data_expected_3d = data_in_3d.asnumpy()
    data_expected_3d[:][:][:] = (data_expected_3d[:][:][:] - 1.0) / 2.0

    data = mx.symbol.Variable('data')
    img_norm_sym = mx.sym.image.normalize(data=data, mean=mean, std=std)

    # check forward
    check_symbolic_forward(img_norm_sym, [data_in_3d], [data_expected_3d],
                           rtol=1e-5, atol=1e-5)

    # Gradient is 1/std_dev
    grad_expected_3d = np.ones(shape_3d)
    grad_expected_3d[:][:][:] = 1 / 2.0

    # check backward
    check_symbolic_backward(img_norm_sym, location=[data_in_3d], out_grads=[mx.nd.ones(shape_3d)],
                            expected=[grad_expected_3d], rtol=1e-5, atol=1e-5)

    # check backward using finite difference
    check_numeric_gradient(img_norm_sym, [data_in_3d], atol=0.001)

    # Part 4 - Test 4D input with scalar mean/std
    shape_4d = (2, 3, 28, 28)

    data_in_4d = mx.nd.random.uniform(0, 1, shape_4d)
    data_expected_4d = data_in_4d.asnumpy()
    data_expected_4d[:][:][:][:] = (data_expected_4d[:][:][:][:] - 1.0) / 2.0

    # check forward
    check_symbolic_forward(img_norm_sym, [data_in_4d], [data_expected_4d],
                           rtol=1e-5, atol=1e-5)

    # Gradient is 1/std_dev
    grad_expected_4d = np.ones(shape_4d)
    grad_expected_4d[:][:][:][:] = 1 / 2.0

    # check backward
    check_symbolic_backward(img_norm_sym, location=[data_in_4d], out_grads=[mx.nd.ones(shape_4d)],
                            expected=[grad_expected_4d], rtol=1e-5, atol=1e-5)

    # check backward using finite difference
    check_numeric_gradient(img_norm_sym, [data_in_4d], atol=0.001)

@with_seed()
def test_index_array():
    def test_index_array_default():
        for shape in [(10,), (7, 5, 29), (5, 7, 11, 13, 17, 19)]:
            data  = mx.symbol.Variable("data")
            index_array = mx.sym.contrib.index_array(data)

            input_array = np.ones(shape)
            mgrid = np.mgrid[tuple(slice(0, x) for x in shape)]
            expected = np.stack(mgrid, axis=-1)

            check_symbolic_forward(index_array, [input_array], [expected])
            check_symbolic_backward(index_array, [input_array], [np.ones(expected.shape)], [np.zeros_like(input_array)])

    @mx.use_np_shape
    def test_index_array_default_zero_dim():
        data = mx.symbol.Variable("data")
        index_array = mx.sym.contrib.index_array(data)

        input_array = np.ones(())
        expected = np.zeros((0,))

        check_symbolic_forward(index_array, [input_array], [expected])
        check_symbolic_backward(index_array, [input_array], [np.ones(expected.shape)], [np.zeros_like(input_array)])

    @mx.use_np_shape
    def test_index_array_default_zero_size():
        data  = mx.symbol.Variable("data")
        index_array = mx.sym.contrib.index_array(data)

        input_array = np.ones((0, 0, 0))
        expected = np.zeros((0, 0, 0, 3))

        check_symbolic_forward(index_array, [input_array], [expected])
        check_symbolic_backward(index_array, [input_array], [np.ones(expected.shape)], [np.zeros_like(input_array)])

    def test_index_array_select_axes():
        shape = (5, 7, 11, 13, 17, 19)
        for axes in [(3,), (4, 1), (5, 1, 3), (-1,), (-5, -1, -3)]:
            data  = mx.symbol.Variable("data")
            index_array = mx.sym.contrib.index_array(data, axes=axes)

            input_array = np.ones(shape)
            mgrid = np.mgrid[tuple(slice(0, x) for x in shape)]
            expected = np.stack(mgrid, axis=-1)[..., axes]

            check_symbolic_forward(index_array, [input_array], [expected])
            check_symbolic_backward(index_array, [input_array], [np.ones(expected.shape)], [np.zeros_like(input_array)])

    @mx.use_np_shape
    def test_index_array_select_axes_zero_size():
        data  = mx.symbol.Variable("data")
        index_array = mx.sym.contrib.index_array(data, axes=(2, 1))

        input_array = np.ones((0, 0, 0, 0))
        expected = np.zeros((0, 0, 2))

        check_symbolic_forward(index_array, [input_array], [expected])
        check_symbolic_backward(index_array, [input_array], [np.ones(expected.shape)], [np.zeros_like(input_array)])

    test_index_array_default()
    test_index_array_default_zero_dim()
    test_index_array_default_zero_size()
    test_index_array_select_axes()
    test_index_array_select_axes_zero_size()


@with_seed()
def test_scalar_tensor_creation():
    assertRaises(MXNetError, mx.nd.zeros, shape=())
    assertRaises(MXNetError, mx.nd.ones, shape=())
    with mx.np_shape():
        data_mx = mx.nd.ones(shape=())
        data_np = np.ones((), dtype=data_mx.dtype)
        assert same(data_mx.asnumpy(), data_np)


@with_seed()
def test_zero_size_tensor_creation():
    assertRaises(MXNetError, mx.nd.zeros, shape=(0, 1, 3, 0))
    assertRaises(MXNetError, mx.nd.ones, shape=(0, 1, 3, 0))
    with mx.np_shape():
        data_mx = mx.nd.ones(shape=(0, 1, 0, 4))
        data_np = np.ones(shape=data_mx.shape, dtype=data_mx.dtype)
        assert same(data_mx.asnumpy(), data_np)


@with_seed()
def test_concat_with_zero_size_tensor():
    with mx.np_shape():
        data1 = mx.nd.ones((0, 8, 12))
        data2 = mx.nd.ones((3, 8, 12))
        data3 = mx.nd.ones((0, 8, 12))
        ret = mx.nd.Concat(data1, data2, data3, dim=0)
        assert ret.shape == (3, 8, 12)

        data1 = mx.nd.ones((0, 3, 10))
        data2 = mx.nd.ones((0, 4, 10))
        data3 = mx.nd.ones((0, 5, 10))
        ret = mx.nd.Concat(data1, data2, data3, dim=1)
        assert ret.shape == (0, 12, 10)


@with_seed()
def test_np_shape_decorator():
    @mx.use_np_shape
    def check_scalar_one():
        """Generate scalar one tensor"""
        return mx.nd.ones(shape=())
    assert check_scalar_one.__name__ == "check_scalar_one"
    assert check_scalar_one.__doc__ == "Generate scalar one tensor"
    assert check_scalar_one().shape == ()
    for active in [True, False]:
        with mx.np_shape(active=active):
            assert check_scalar_one.__name__ == "check_scalar_one"
            assert check_scalar_one.__doc__ == "Generate scalar one tensor"
            assert check_scalar_one().shape == ()

    @mx.use_np_shape
    def check_concat(shape1, shape2, axis):
        data1 = mx.nd.ones(shape1)
        data2 = mx.nd.ones(shape2)
        ret = mx.nd.Concat(data1, data2, dim=axis)
        expected_ret = np.concatenate((data1.asnumpy(), data2.asnumpy()), axis=axis)
        assert ret.shape == expected_ret.shape

    check_concat((0, 3, 4), (5, 3, 4), 0)
    check_concat((8, 0, 5), (8, 7, 5), 1)
    check_concat((8, 0, 0), (8, 0, 0), 2)
    for active in [True, False]:
        check_concat((0, 3, 4), (5, 3, 4), 0)
        check_concat((8, 0, 5), (8, 7, 5), 1)
        check_concat((8, 0, 0), (8, 0, 0), 2)


@with_seed()
def test_add_n():
    data_shape = (2, 2)
    input_num = 5
    data = [mx.nd.random.uniform(shape=data_shape) for i in range(input_num)]
    rslt = mx.nd.zeros(shape=data_shape)
    for i in range(input_num):
        rslt += data[i]
    add_n_rslt = mx.nd.add_n(*data, out=data[0])
    assert_almost_equal(rslt.asnumpy(), add_n_rslt.asnumpy(), atol=1e-5)


def test_get_all_registered_operators():
    ops = get_all_registered_operators()
    ok_(isinstance(ops, list))
    ok_(len(ops) > 0)
    ok_('Activation' in ops)


def test_get_operator_arguments():
    operator_arguments = get_operator_arguments('Activation')
    ok_(isinstance(operator_arguments, OperatorArguments))
    ok_(operator_arguments.names == ['data', 'act_type'])
    ok_(operator_arguments.types
        == ['NDArray-or-Symbol', "{'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}, required"])
    ok_(operator_arguments.narg == 2)


def test_transpose_infer_shape_back():
    o1 = mx.sym.ones(shape=[2,3])
    o2 = mx.sym.ones(shape=[-1,-1])
    t = mx.sym.transpose(o2)
    b = o1 + t
    x = b.bind(mx.cpu(), args={})
    y = x.forward()
    assert(y[0].shape == (2,3))


def test_transpose_infer_shape_mixed():
    o1 = mx.sym.ones(shape=[2,-1])
    o2 = mx.sym.ones(shape=[3,-1])
    t = mx.sym.transpose(o2)
    b = o1 + t
    x = b.bind(mx.cpu(), args={})
    y = x.forward()
    assert(y[0].shape == (2,3))


@with_seed()
def test_sample_normal_default_shape():
    # Test case from https://github.com/apache/incubator-mxnet/issues/16135
    s = mx.nd.sample_normal(mu=mx.nd.array([10.0]), sigma=mx.nd.array([0.5]))
    assert s.shape == (1,)
    s = mx.nd.sample_normal(mu=mx.nd.array([10.0]), sigma=mx.nd.array([0.5]), shape=())
    assert s.shape == (1,)
    s = mx.nd.sample_normal(mu=mx.nd.array([10.0]), sigma=mx.nd.array([0.5]), shape=1)
    assert s.shape == (1, 1)
    s = mx.nd.sample_normal(mu=mx.nd.array([10.0]), sigma=mx.nd.array([0.5]), shape=(1,))
    assert s.shape == (1, 1)


def test_large_tensor_disabled_err_msg():
    LARGE_X = 4300000000
    MEDIUM_X = 1000000000
    SMALL_Y = 1
    shape = (2, LARGE_X)

    def check_nd_array():
        x = np.arange(0, LARGE_X)
        assertRaises(MXNetError, mx.nd.array, x)

    def check_nd_ones():
        assertRaises(MXNetError, mx.nd.ones, shape)

    def check_nd_zeros():
        assertRaises(MXNetError, mx.nd.zeros, shape)

    def check_nd_full():
        val = 1
        assertRaises(Exception, mx.nd.full, shape, val)

    def check_nd_arange():
        start = 0
        stop = LARGE_X
        assertRaises(Exception, mx.nd.arange, start, stop)

    def check_nd_random():
        shape = (2, LARGE_X)
        def check_random_exp():
            lam = 4
            assertRaises(MXNetError, mx.nd.random_exponential, lam, shape)

        def check_random_gamma():
            alpha = 9
            beta = 0.5
            assertRaises(MXNetError, mx.nd.random_gamma, alpha, beta, shape)

        def check_random_normal():
            loc = 0
            scale = 1
            assertRaises(MXNetError, mx.nd.random_normal, loc, scale, shape)

        def check_random_poisson():
            lam = 4
            assertRaises(MXNetError, mx.nd.random_poisson, alpha, lam, shape)

        def check_random_randint():
            low = 0
            high = 1000000
            assertRaises(MXNetError, mx.nd.random_randint, low, high, shape)

        def check_random_uniform():
            low = 0
            hight = 1
            assertRaises(MXNetError, mx.nd.random_uniform, alpha, beta, shape)

def check_multihead_attention_selfatt(dtype):
    def convert_weight(F, q_weight, k_weight, v_weight, num_heads):
        q_weight = F.reshape(q_weight, shape=(num_heads, -1, 0), reverse=True)
        k_weight = F.reshape(k_weight, shape=(num_heads, -1, 0), reverse=True)
        v_weight = F.reshape(v_weight, shape=(num_heads, -1, 0), reverse=True)
        all_weights = F.concat(q_weight, k_weight, v_weight, dim=-2)
        all_weights = F.reshape(all_weights, shape=(-1, 0), reverse=True)
        return all_weights

    def convert_bias(F, q_bias, k_bias, v_bias, num_heads):
        q_bias = F.reshape(q_bias, shape=(num_heads, -1))
        k_bias = F.reshape(k_bias, shape=(num_heads, -1))
        v_bias = F.reshape(v_bias, shape=(num_heads, -1))
        all_bias = F.stack(q_bias, k_bias, v_bias, axis=1)
        all_bias = F.reshape(all_bias, shape=(-1,))
        return all_bias

    batch_size = 2
    qkv_length = 7  # length of a sequence
    qkv_dim = 9     # dimension of encoding
    num_heads = 3   # number of attention head
    head_dim = 5    # head size
    out_dim = 13 * num_heads
    qkv_units = num_heads * head_dim

    arg_params = {
        'qkv': mx.nd.array(np.random.rand(*(batch_size, qkv_length, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'q_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'k_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'v_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'q_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype(dtype) * 0.1, dtype=dtype),
        'k_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype(dtype) * 0.1, dtype=dtype),
        'v_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype(dtype) * 0.1, dtype=dtype),
        'out_weight': mx.nd.array(np.random.rand(*(out_dim, qkv_units)).astype(dtype) * 0.1, dtype=dtype),
        'out_bias': mx.nd.array(np.random.rand(*(out_dim,)).astype(dtype) * 0.1, dtype=dtype),
        }

    qkv = mx.sym.Variable('qkv')
    sonde = mx.sym.Variable('sonde')
    q_weight = mx.sym.Variable('q_weight')
    k_weight = mx.sym.Variable('k_weight')
    v_weight = mx.sym.Variable('v_weight')
    q_bias = mx.sym.Variable('q_bias')
    k_bias = mx.sym.Variable('k_bias')
    v_bias = mx.sym.Variable('v_bias')
    out_weight = mx.sym.Variable('out_weight')
    out_bias = mx.sym.Variable('out_bias')
    qkv_weight = convert_weight(mx.sym, q_weight, k_weight, v_weight, num_heads)
    qkv_bias = convert_bias(mx.sym, q_bias, k_bias, v_bias, num_heads)
    qkv = mx.sym.transpose(qkv, axes=(1, 0, 2))
    qkv_proj = mx.sym.FullyConnected(qkv, weight=qkv_weight, bias=qkv_bias, flatten=False,
                                     num_hidden=qkv_units * 3, no_bias=False)
    att_score = mx.sym.contrib.interleaved_matmul_selfatt_qk(
            qkv_proj, heads=num_heads)
    att_score = att_score + sonde
    weighted_value = mx.sym.contrib.interleaved_matmul_selfatt_valatt(
            qkv_proj, att_score, heads=num_heads)
    output = mx.sym.FullyConnected(weighted_value, weight=out_weight, bias=out_bias, flatten=False,
                                   num_hidden=out_dim, no_bias=False)
    output = mx.sym.transpose(output, axes=(1, 0, 2))
    output = mx.sym.Group([output, att_score])
    executor = output.simple_bind(ctx=default_context(),
                                  qkv=(batch_size, qkv_length, qkv_dim),
                                  q_weight=(qkv_units, qkv_dim),
                                  q_bias=(qkv_units,),
                                  k_weight=(qkv_units, qkv_dim),
                                  k_bias=(qkv_units,),
                                  v_weight=(qkv_units, qkv_dim),
                                  v_bias=(qkv_units,),
                                  type_dict={'qkv': dtype,
                                             'q_weight': dtype,
                                             'k_weight': dtype,
                                             'v_weight': dtype,
                                             'q_bias': dtype,
                                             'k_bias': dtype,
                                             'v_bias': dtype,
                                             'sonde': dtype},
                                  grad_req='write', force_rebind=True)
    output_shape = executor.outputs[0].shape
    output_grads = np.random.rand(*output_shape).astype(dtype) * 0.1
    executor.copy_params_from(arg_params, {})
    executor.arg_dict['sonde'][:] = 0.
    executor.arg_dict['sonde'].wait_to_read()
    executor.forward(is_train=True)
    output_opti = executor.outputs[0].asnumpy()
    att_score_opti = executor.outputs[1].asnumpy()
    executor.backward([mx.nd.array(output_grads, dtype=dtype),
                       mx.nd.zeros(att_score_opti.shape, dtype=dtype)])
    grads_opti = {k: v.asnumpy() for k, v in executor.grad_dict.items()}
    qkv = mx.sym.Variable('qkv')
    sonde = mx.sym.Variable('sonde')
    q_weight = mx.sym.Variable('q_weight')
    k_weight = mx.sym.Variable('k_weight')
    v_weight = mx.sym.Variable('v_weight')
    q_bias = mx.sym.Variable('q_bias')
    k_bias = mx.sym.Variable('k_bias')
    v_bias = mx.sym.Variable('v_bias')
    out_weight = mx.sym.Variable('out_weight')
    out_bias = mx.sym.Variable('out_bias')

    q = mx.sym.FullyConnected(qkv, weight=q_weight, bias=q_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    k = mx.sym.FullyConnected(qkv, weight=k_weight, bias=k_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    v = mx.sym.FullyConnected(qkv, weight=v_weight, bias=v_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    q = mx.sym.reshape(q, shape=(0, 0, num_heads, -1))
    q = mx.sym.transpose(q, axes=(0, 2, 1, 3))
    q = mx.sym.reshape(q, shape=(-1, 0, 0), reverse=True)
    k = mx.sym.reshape(k, shape=(0, 0, num_heads, -1))
    k = mx.sym.transpose(k, axes=(0, 2, 1, 3))
    k = mx.sym.reshape(k, shape=(-1, 0, 0), reverse=True)
    q = mx.sym.contrib.div_sqrt_dim(q)
    att_score = mx.sym.batch_dot(q, k, transpose_b=True)
    att_score = att_score + sonde
    v = mx.sym.reshape(v, shape=(0, 0, num_heads, -1))
    v = mx.sym.transpose(v, axes=(0, 2, 1, 3))
    v = mx.sym.reshape(v, shape=(-1, 0, 0), reverse=True)
    weighted_value = mx.sym.batch_dot(att_score, v)
    weighted_value = mx.sym.reshape(weighted_value, shape=(-1, num_heads, 0, 0),
                                    reverse=True)
    weighted_value = mx.sym.transpose(weighted_value, axes=(0, 2, 1, 3))
    weighted_value = mx.sym.reshape(weighted_value, shape=(0, 0, -1))
    output = mx.sym.FullyConnected(weighted_value, weight=out_weight, bias=out_bias, flatten=False,
                                   num_hidden=out_dim, no_bias=False)
    output = mx.sym.Group([output, att_score])
    executor = output.simple_bind(ctx=default_context(),
                                  qkv=(batch_size, qkv_length, qkv_dim),
                                  type_dict={'qkv': dtype},
                                  grad_req='write', force_rebind=True)
    executor.copy_params_from(arg_params, {})
    executor.arg_dict['sonde'][:] = 0.
    executor.arg_dict['sonde'].wait_to_read()
    executor.forward(is_train=True)
    output_orig = executor.outputs[0].asnumpy()
    att_score_orig = executor.outputs[1].asnumpy()
    executor.backward([mx.nd.array(output_grads, dtype=dtype),
                       mx.nd.zeros(att_score_orig.shape, dtype=dtype)])
    grads_orig = {k : v.asnumpy() for k, v in executor.grad_dict.items()}
    assert_allclose(att_score_orig, att_score_opti, rtol=1e-2, atol=1e-3)
    assert_allclose(output_orig, output_opti, rtol=1e-2, atol=1e-3)

    for k in grads_opti.keys():
        assert(grads_orig[k].dtype == grads_opti[k].dtype)
        assert(grads_orig[k].shape == grads_opti[k].shape)
        assert_allclose(grads_orig[k], grads_opti[k], rtol=1e-2, atol=1e-3)


@with_seed()
@assert_raises_cuda_not_satisfied(min_version='9.1')
def test_multihead_attention_selfatt():
    dtypes = ['float32']
    if default_context().device_type == 'gpu':
        dtypes += ['float16']

    for dtype in dtypes:
        check_multihead_attention_selfatt(dtype=dtype)

def check_multihead_attention_encdec(dtype):
    def convert_weight(F, k_weight, v_weight, num_heads):
        k_weight = F.reshape(k_weight, shape=(num_heads, -1, 0), reverse=True)
        v_weight = F.reshape(v_weight, shape=(num_heads, -1, 0), reverse=True)
        all_weights = F.concat(k_weight, v_weight, dim=-2)
        all_weights = F.reshape(all_weights, shape=(-1, 0), reverse=True)
        return all_weights

    def convert_bias(F, k_bias, v_bias, num_heads):
        k_bias = F.reshape(k_bias, shape=(num_heads, -1))
        v_bias = F.reshape(v_bias, shape=(num_heads, -1))
        all_bias = F.stack(k_bias, v_bias, axis=1)
        all_bias = F.reshape(all_bias, shape=(-1,))
        return all_bias

    batch_size = 2
    qkv_length = 7  # length of a sequence
    qkv_dim = 9     # dimension of encoding
    num_heads = 3   # number of attention head
    head_dim = 5    # head size
    out_dim = 13 * num_heads
    qkv_units = num_heads * head_dim

    arg_params = {
        'q': mx.nd.array(np.random.rand(*(batch_size, qkv_length, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'kv': mx.nd.array(np.random.rand(*(batch_size, qkv_length, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'q_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'k_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'v_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype(dtype) * 0.1, dtype=dtype),
        'q_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype(dtype) * 0.1, dtype=dtype),
        'k_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype(dtype) * 0.1, dtype=dtype),
        'v_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype(dtype) * 0.1, dtype=dtype),
        'out_weight': mx.nd.array(np.random.rand(*(out_dim, qkv_units)).astype(dtype) * 0.1, dtype=dtype),
        'out_bias': mx.nd.array(np.random.rand(*(out_dim,)).astype(dtype) * 0.1, dtype=dtype),
        }

    q = mx.sym.Variable('q')
    kv = mx.sym.Variable('kv')
    sonde = mx.sym.Variable('sonde')
    q_weight = mx.sym.Variable('q_weight')
    k_weight = mx.sym.Variable('k_weight')
    v_weight = mx.sym.Variable('v_weight')
    q_bias = mx.sym.Variable('q_bias')
    k_bias = mx.sym.Variable('k_bias')
    v_bias = mx.sym.Variable('v_bias')
    out_weight = mx.sym.Variable('out_weight')
    out_bias = mx.sym.Variable('out_bias')
    kv_weight = convert_weight(mx.sym, k_weight, v_weight, num_heads)
    kv_bias = convert_bias(mx.sym, k_bias, v_bias, num_heads)
    kv = mx.sym.transpose(kv, axes=(1, 0, 2))
    kv_proj = mx.sym.FullyConnected(kv, weight=kv_weight, bias=kv_bias, flatten=False,
                                    num_hidden=qkv_units * 2, no_bias=False)
    q = mx.sym.transpose(q, axes=(1, 0, 2))
    q_proj = mx.sym.FullyConnected(q, weight=q_weight, bias=q_bias, flatten=False,
                                   num_hidden=qkv_units, no_bias=False)
    att_score = mx.sym.contrib.interleaved_matmul_encdec_qk(
            q_proj, kv_proj, heads=num_heads)
    att_score = att_score + sonde
    weighted_value = mx.sym.contrib.interleaved_matmul_encdec_valatt(
            kv_proj, att_score, heads=num_heads)
    output = mx.sym.FullyConnected(weighted_value, weight=out_weight, bias=out_bias, flatten=False,
                                   num_hidden=out_dim, no_bias=False)
    output = mx.sym.transpose(output, axes=(1, 0, 2))
    output = mx.sym.Group([output, att_score])
    executor = output.simple_bind(ctx=default_context(),
                                  q=(batch_size, qkv_length, qkv_dim),
                                  kv=(batch_size, qkv_length, qkv_dim),
                                  q_weight=(qkv_units, qkv_dim),
                                  q_bias=(qkv_units,),
                                  k_weight=(qkv_units, qkv_dim),
                                  k_bias=(qkv_units,),
                                  v_weight=(qkv_units, qkv_dim),
                                  v_bias=(qkv_units,),
                                  out_weight=(out_dim, qkv_units),
                                  out_bias=(out_dim,),
                                  type_dict={'q': dtype,
                                             'kv': dtype,
                                             'q_weight': dtype,
                                             'q_bias': dtype,
                                             'k_weight': dtype,
                                             'k_bias': dtype,
                                             'v_weight': dtype,
                                             'v_bias': dtype,
                                             'out_weight': dtype,
                                             'out_bias': dtype,
                                              },
                                  grad_req='write', force_rebind=True)
    output_shape = executor.outputs[0].shape
    output_grads = np.random.rand(*output_shape).astype(dtype) * 0.1
    executor.copy_params_from(arg_params, {})
    executor.arg_dict['sonde'][:] = 0.
    executor.arg_dict['sonde'].wait_to_read()
    executor.forward(is_train=True)
    output_opti = executor.outputs[0].asnumpy()
    att_score_opti = executor.outputs[1].asnumpy()
    executor.backward([mx.nd.array(output_grads, dtype=dtype), mx.nd.zeros(att_score_opti.shape, dtype=dtype)])

    grads_opti = {k: v.asnumpy() for k, v in executor.grad_dict.items()}

    q = mx.sym.Variable('q')
    kv = mx.sym.Variable('kv')
    sonde = mx.sym.Variable('sonde')
    q_weight = mx.sym.Variable('q_weight')
    k_weight = mx.sym.Variable('k_weight')
    v_weight = mx.sym.Variable('v_weight')
    q_bias = mx.sym.Variable('q_bias')
    k_bias = mx.sym.Variable('k_bias')
    v_bias = mx.sym.Variable('v_bias')
    out_weight = mx.sym.Variable('out_weight')
    out_bias = mx.sym.Variable('out_bias')

    q = mx.sym.FullyConnected(q, weight=q_weight, bias=q_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    k = mx.sym.FullyConnected(kv, weight=k_weight, bias=k_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    v = mx.sym.FullyConnected(kv, weight=v_weight, bias=v_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    q = mx.sym.reshape(q, shape=(0, 0, num_heads, -1))
    q = mx.sym.transpose(q, axes=(0, 2, 1, 3))
    q = mx.sym.reshape(q, shape=(-1, 0, 0), reverse=True)
    k = mx.sym.reshape(k, shape=(0, 0, num_heads, -1))
    k = mx.sym.transpose(k, axes=(0, 2, 1, 3))
    k = mx.sym.reshape(k, shape=(-1, 0, 0), reverse=True)
    q = mx.sym.contrib.div_sqrt_dim(q)
    att_score = mx.sym.batch_dot(q, k, transpose_b=True)
    att_score = att_score + sonde
    v = mx.sym.reshape(v, shape=(0, 0, num_heads, -1))
    v = mx.sym.transpose(v, axes=(0, 2, 1, 3))
    v = mx.sym.reshape(v, shape=(-1, 0, 0), reverse=True)
    weighted_value = mx.sym.batch_dot(att_score, v)
    weighted_value = mx.sym.reshape(weighted_value, shape=(-1, num_heads, 0, 0),
                                    reverse=True)
    weighted_value = mx.sym.transpose(weighted_value, axes=(0, 2, 1, 3))
    weighted_value = mx.sym.reshape(weighted_value, shape=(0, 0, -1))
    output = mx.sym.FullyConnected(weighted_value, weight=out_weight, bias=out_bias, flatten=False,
                                   num_hidden=out_dim, no_bias=False)
    output = mx.sym.Group([output, att_score])
    executor = output.simple_bind(ctx=default_context(),
                                  q=(batch_size, qkv_length, qkv_dim),
                                  kv=(batch_size, qkv_length, qkv_dim),
                                  type_dict={'q': dtype,
                                             'kv': dtype},
                                  grad_req='write', force_rebind=True)
    executor.copy_params_from(arg_params, {})
    executor.arg_dict['sonde'][:] = 0.
    executor.arg_dict['sonde'].wait_to_read()
    executor.forward(is_train=True)
    output_orig = executor.outputs[0].asnumpy()
    att_score_orig = executor.outputs[1].asnumpy()
    executor.backward([mx.nd.array(output_grads, dtype=dtype), mx.nd.zeros(att_score_orig.shape, dtype=dtype)])
    grads_orig = {k : v.asnumpy() for k, v in executor.grad_dict.items()}
    assert_allclose(att_score_orig, att_score_opti, rtol=1e-2, atol=1e-3)
    assert_allclose(output_orig, output_opti, rtol=1e-2, atol=1e-3)

    for k in grads_opti.keys():
        assert(grads_orig[k].dtype == grads_opti[k].dtype)
        assert(grads_orig[k].shape == grads_opti[k].shape)
        assert_allclose(grads_orig[k], grads_opti[k], rtol=1e-2, atol=1e-3)

@with_seed()
@assert_raises_cuda_not_satisfied(min_version='9.1')
def test_multihead_attention_encdec():
    dtypes = ['float32']
    if default_context().device_type == 'gpu':
        dtypes += ['float16']

    for dtype in dtypes:
        check_multihead_attention_encdec(dtype=dtype)

@with_seed()
def test_im2col_col2im():
    def compute_output_size(spatial, kernel, stride=1, dilate=1, pad=0):
        pad_size = spatial + 2 * pad
        dilated_kernel = dilate * (kernel - 1) + 1
        return (pad_size - dilated_kernel) // stride + 1

    def build_kwargs(kernel, stride=1, dilate=1, pad=0):
        return {'kernel': (kernel, kernel),
                'stride': (stride, stride),
                'dilate': (dilate, dilate),
                'pad': (pad, pad)}

    # use im2col to compute convolution
    def test_conv_compute(input_shape, num_filter, kernel, stride=1, dilate=1, pad=0):
        batch_size = input_shape[0]
        channel = input_shape[1]
        kwargs = build_kwargs(kernel, stride, dilate, pad)
        data = mx.nd.uniform(shape=input_shape)
        col = mx.nd.im2col(data, **kwargs)
        w = mx.nd.uniform(shape=(num_filter, channel, kernel, kernel))
        c1 = mx.nd.dot(col.transpose((0, 2, 1)), w.reshape(num_filter, -1).T).transpose((0, 2, 1))
        hos = compute_output_size(input_shape[2], kernel, stride, dilate, pad)
        wos = compute_output_size(input_shape[3], kernel, stride, dilate, pad)
        c1 = c1.reshape((batch_size, num_filter, hos, wos))

        c2 = mx.nd.Convolution(data, num_filter=num_filter, weight=w, no_bias=True, **kwargs)
        assert_almost_equal(c1.asnumpy(), c2.asnumpy(), rtol=1e-5, atol=1e-5)

    test_conv_compute(
        input_shape = (5, 3, 30, 20),
        num_filter  = 10,
        kernel      = 3
    )

    test_conv_compute(
        input_shape = (5, 3, 30, 20),
        num_filter  = 10,
        kernel      = 3,
        stride      = 2
    )

    test_conv_compute(
        input_shape = (5, 3, 30, 20),
        num_filter  = 10,
        kernel      = 3,
        stride      = 2,
        dilate      = 2
    )

    test_conv_compute(
        input_shape = (5, 3, 30, 20),
        num_filter  = 10,
        kernel      = 3,
        stride      = 2,
        dilate      = 2,
        pad         = 1
    )

    # use composite of im2col and col2im to reconstruct image
    def test_reconstruct(input_shape, kernel, stride=1, dilate=1, pad=0):
        batch_size = input_shape[0]
        channel = input_shape[1]
        kwargs = build_kwargs(kernel, stride, dilate, pad)
        data = mx.nd.uniform(shape=input_shape)
        col = mx.nd.im2col(data, **kwargs)
        im1 = mx.nd.col2im(col, input_shape[2:], **kwargs)

        im2 = mx.nd.col2im(mx.nd.ones_like(col), input_shape[2:], **kwargs) * data
        assert_almost_equal(im1.asnumpy(), im2.asnumpy(), rtol=1e-5, atol=1e-5)

    test_reconstruct(
        input_shape = (5, 3, 30, 20),
        kernel      = 3
    )

    test_reconstruct(
        input_shape = (5, 3, 30, 20),
        kernel      = 3,
        stride      = 2
    )

    test_reconstruct(
        input_shape = (5, 3, 30, 20),
        kernel      = 3,
        stride      = 2,
        dilate      = 2
    )

    test_reconstruct(
        input_shape = (5, 3, 30, 20),
        kernel      = 3,
        stride      = 2,
        dilate      = 2,
        pad         = 1
    )

    # test gradient
    # the grad of im2col is col2im, and vice versa
    def test_grad(input_shape, kernel, stride=1, dilate=1, pad=0):
        # im2col
        data = mx.sym.Variable('data')
        kwargs = build_kwargs(kernel, stride, dilate, pad)
        sym = mx.sym.im2col(data, **kwargs)

        im = mx.nd.uniform(shape=input_shape)
        col = mx.nd.im2col(im, **kwargs)
        col_shape = col.shape
        expected = mx.nd.col2im(col, input_shape[2:], **kwargs)
        check_symbolic_backward(sym, [im.asnumpy()], [col.asnumpy()], [expected.asnumpy()])

        # col2im
        data = mx.sym.Variable('data')
        sym = mx.sym.col2im(data, input_shape[2:], **kwargs)

        col = mx.nd.uniform(shape=col_shape)
        im = mx.nd.col2im(col, input_shape[2:], **kwargs)
        expected = mx.nd.im2col(im, **kwargs)
        check_symbolic_backward(sym, [col.asnumpy()], [im.asnumpy()], [expected.asnumpy()])

    test_grad(
        input_shape = (5, 3, 30, 20),
        kernel      = 3
    )

    test_grad(
        input_shape = (5, 3, 30, 20),
        kernel      = 3,
        stride      = 2
    )

    test_grad(
        input_shape = (5, 3, 30, 20),
        kernel      = 3,
        stride      = 2,
        dilate      = 2
    )

    test_grad(
        input_shape = (5, 3, 30, 20),
        kernel      = 3,
        stride      = 2,
        dilate      = 2,
        pad         = 1
    )

def test_elemwise_sum_for_gradient_accumulation():
    for nrepeat in range(1, 10):
        stored_grad = dict()
        for grad_req in ['write', 'add']:
            a = mx.nd.array([1])
            b = mx.nd.array([2])
            if grad_req == 'write':
                a.attach_grad(grad_req='write')
            elif grad_req == 'add':
                a.attach_grad(grad_req='add')
            a.grad[:] = 0
            with mx.autograd.record():
                for _ in range(nrepeat):
                    b = b * a
                b.backward()
            stored_grad[grad_req] = a.grad.asscalar()
        assert stored_grad['write'] == stored_grad['add']
        assert stored_grad['write'] == 2 * nrepeat

def test_scalarop_locale_invariance():
    arr = mx.nd.zeros((1,))
    prev = locale.getlocale(locale.LC_NUMERIC)
    try:
        locales_to_try = [
            "en_DK.UTF-8", # Non-standard locale that uses , as the decimal separator, installed
                           # on English Ubuntu by default.
            "ru_RU.UTF-8", # Uses , as the decimal separator. May not be installed on the system.
            "German"       # A Windows locale. Uses , as the decimal separator.
        ]

        locale_set = False
        for loc in locales_to_try:
            try:
                locale.setlocale(locale.LC_NUMERIC, loc)
                locale_set = True
                break
            except locale.Error as e:
                print("Couldn't enable locale", loc, ": ", str(e))

        if locale_set:
            scalar = 0.3
            assert "," in locale.str(scalar)
            assert_almost_equal(
                arr.asnumpy() + scalar,
                (arr + scalar).asnumpy(),
                rtol=1e-5,
                atol=1e-5
            )
        else:
            # Shouldn't happen on Windows: "German" should always be available.
            raise unittest.SkipTest("Couldn't find a locale suitable for "
                                    "test_scalarop_locale_invariance. Please install en_DK.UTF-8 "
                                    "or ru_RU.UTF-8 to run this test.")
    finally:
        locale.setlocale(locale.LC_NUMERIC, prev)

@with_seed()
def test_take_grads():
    # Test for https://github.com/apache/incubator-mxnet/issues/19817
    from mxnet.gluon.nn import HybridBlock, Conv1D, HybridSequential, HybridLambda, Dense
    from mxnet import autograd, nd
    from mxnet.gluon.loss import L2Loss

    def get_grads(model, grads, ctx=mx.cpu()):
        pd = model.collect_params()
        total_grad_l2 = 0
        total_grad_l1 = 0
        total_grad_linf = 0
        for p in pd:
            try:
                g = pd[p].grad(ctx) / N
                g2 = (g**2).sum().as_in_context(mx.cpu()).asscalar()
                g1 = g.abs().sum().as_in_context(mx.cpu()).asscalar()
                ginf = g.max().as_in_context(mx.cpu()).asscalar()
                total_grad_linf = max(total_grad_linf, ginf)
                total_grad_l2 += g2
                total_grad_l1 += g1
                print(f"||g_param||_2: {g2**0.5:.2E} | Param: {p}")
            except Exception:
                pass

        grads.append(total_grad_l1)
        grads.append(total_grad_l2)
        grads.append(total_grad_linf)

    def run_model(model, loss, X, Y, num_iters=5):
        grads = []
        for i in range(num_iters):
            with autograd.record():
                Y_hat = model(X)
                ll = loss(Y_hat, Y)
                ll = ll.sum()
            ll.backward()
            get_grads(model, grads)
        return grads

    def conv_layer(atrous_rates, num_channels):
        convs = HybridSequential()
        for rate in atrous_rates:
            convs.add(Conv1D(num_channels, 3, padding=rate, dilation=rate, activation='tanh'))
        return convs

    class Model(HybridBlock):
        def __init__(self, conv_units, atrous_rates, use_take=False, **kwargs):
            super().__init__(prefix=kwargs.get('prefix', None), params=kwargs.get('params', None))
            self.use_take = use_take
            with self.name_scope():
                self.convs = conv_layer(atrous_rates, conv_units)
                self.dense_out = Dense(1, flatten=False, activation='tanh')

        def hybrid_forward(self, F, X, axis=-1):
            X1 = X
            X2 = self.convs(X1)
            if self.use_take:
                X3 = F.take(X2, nd.array([0]), axis=axis)
            else:
                X3 = F.slice_axis(X2, begin=0, end=1, axis=axis)
            return X3

    N = 30
    T = 20
    C = 8
    conv_units = 5
    atrous_rates = [1]

    X = np.random.normal(size=(N, T, C))
    Y = np.random.normal(size=(N, T))
    Y = np.random.normal(size=(N, conv_units))
    X, Y = nd.array(X), nd.array(Y)
    seed = np.random.randint(1000)

    # Using F.take
    mx.random.seed(seed)
    model = Model(conv_units, atrous_rates, use_take=True)
    model.initialize()
    loss = L2Loss()
    grads1 = run_model(model, loss, X, Y)

    # Using F.slice_axis
    mx.random.seed(seed)
    model2 = Model(conv_units, atrous_rates, use_take=False)
    model2.initialize()
    grads2 = run_model(model2, loss, X, Y)

    for i in range(len(grads1)):
        assert_almost_equal(grads1[i], grads2[i])

if __name__ == '__main__':
    import nose
    nose.runmodule()