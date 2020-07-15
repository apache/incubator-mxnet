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

import os
from uuid import uuid4
import numpy as _np
import mxnet as mx
from mxnet import gluon, autograd, np
from mxnet.test_utils import use_np, assert_almost_equal, check_gluon_hybridize_consistency
from mxnet.gluon import nn
from common import with_seed
import random


@with_seed()
def test_create_np_param():
    M, K, N = 10, 9, 20

    def check_block_params(x, TestBlock, hybridize, expected_type, initializer):
        net = TestBlock()
        net.initialize(initializer())
        if hybridize:
            net.hybridize()
        net(x)
        params = net.collect_params()
        for k, v in params.items():
            assert type(v.data()) is expected_type

    mx.reset_np()

    class TestBlock1(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock1, self).__init__()
            self.w = gluon.Parameter('w', shape=(K, N), allow_deferred_init=True)

        def hybrid_forward(self, F, x, w):
            return F.dot(x, w)

    @use_np
    class TestBlock2(gluon.HybridBlock):
        def __init__(self):
            super(TestBlock2, self).__init__()
            self.w = gluon.Parameter('w', shape=(K, N), allow_deferred_init=True)

        def hybrid_forward(self, F, x, w):
            return F.np.dot(x, w)

    x = mx.nd.random.uniform(shape=(M, K))
    for initializer in [mx.initializer.Uniform, mx.initializer.Normal]:
        check_block_params(x, TestBlock1, False, mx.nd.NDArray, initializer)
        check_block_params(x, TestBlock1, True, mx.nd.NDArray, initializer)
        check_block_params(x.as_np_ndarray(), TestBlock2, False, np.ndarray, initializer)
        check_block_params(x.as_np_ndarray(), TestBlock2, True, np.ndarray, initializer)


@with_seed()
@use_np
def test_optimizer_with_np_ndarrays():
    class LinearRegression(gluon.HybridBlock):
        def __init__(self, num_input_dim=0, num_hidden_dim=100, num_output_dim=10):
            super(LinearRegression, self).__init__()
            self.w1 = gluon.Parameter('w1', shape=(num_input_dim, num_hidden_dim),
                                      allow_deferred_init=True)
            self.w2 = gluon.Parameter('w2', shape=(num_hidden_dim, num_output_dim),
                                      allow_deferred_init=True)

        def hybrid_forward(self, F, x, w1, w2):
            h = x.dot(w1)  # equivalent to F.np.dot(x, w1)
            h_relu = F.npx.relu(h)  # equivalent to F.relu(h) but generating np.ndarray
            y_pred = h_relu.dot(w2)  # equivalent to F.np.dot(h_relu, w2)
            return y_pred

    class TotalLoss(gluon.HybridBlock):
        def hybrid_forward(self, F, pred, label):
            return ((pred - label) ** 2).sum()  # equivalent to F.np.sum(F.np.square(pred - label))

    regressor = LinearRegression()
    regressor.initialize(mx.init.Uniform())
    regressor.hybridize()

    # Create random input and output data
    x = np.random.uniform(size=(64, 1000))  # x is of type mxnet.numpy.ndarray
    regressor(x)
    y = np.random.uniform(size=(64, 10))  # y is of type mxnet.numpy.ndarray

    total_loss = TotalLoss()
    total_loss.hybridize()

    trainer = gluon.Trainer(regressor.collect_params(),
                            'sgd',
                            {'learning_rate': 1e-3, 'momentum': 0.9})

    for t in range(2):
        with autograd.record():
            output = regressor(x)  # output is a type of np.ndarray because np.dot is the last op in the network
            loss = total_loss(output, y)  # loss is a scalar np.ndarray
        loss.backward()
        trainer.step(1)


@with_seed()
@use_np
def test_optimizer_backward_compat():
    optimizer = mx.optimizer.SGD()
    delattr(optimizer, "allow_np_array")
    updater = mx.optimizer.Updater(optimizer)
    updater(0, np.ones((0, 0)), np.zeros((0, 0)))


@with_seed()
@use_np
def test_np_loss_ndarray():
    # Ported from test_loss.test_loss_ndarray
    output = np.array([1, 2, 3, 4])
    label = np.array([1, 3, 5, 7])
    weighting = np.array([0.5, 1, 0.5, 1])

    loss = gluon.loss.L1Loss()
    assert float(np.sum(loss(output, label))) == 6.
    loss = gluon.loss.L1Loss(weight=0.5)
    assert float(np.sum(loss(output, label))) == 3.
    loss = gluon.loss.L1Loss()
    assert float(np.sum(loss(output, label, weighting))) == 5.

    loss = gluon.loss.L2Loss()
    assert float(np.sum(loss(output, label))) == 7.
    loss = gluon.loss.L2Loss(weight=0.25)
    assert float(np.sum(loss(output, label))) == 1.75
    loss = gluon.loss.L2Loss()
    assert float(np.sum(loss(output, label, weighting))) == 6

    output = np.array([[0, 2], [1, 4]])
    label = np.array([0, 1])
    weighting = np.array([[0.5], [1.0]])

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    L = loss(output, label).asnumpy()
    assert_almost_equal(L, _np.array([2.12692809,  0.04858733]), use_broadcast=False, rtol=1e-3)

    L = loss(output, label, weighting).asnumpy()
    assert_almost_equal(L, _np.array([1.06346405,  0.04858733]), use_broadcast=False, rtol=1e-3)


@with_seed()
@use_np
def test_np_get_constant():
    const_arr = _np.random.uniform(0, 100, size=(10, 10)).astype(_np.float32)

    class Foo(gluon.HybridBlock):
        def __init__(self):
            super(Foo, self).__init__()
            self.weight = gluon.Constant(const_arr)

        def hybrid_forward(self, F, x, weight):
            return x + weight.astype(np.float32)

    x = np.random.uniform(size=const_arr.shape, dtype=const_arr.dtype)
    for hybridize in [False, True]:
        foo = Foo()
        if hybridize:
            foo.hybridize()
        foo.initialize()
        out = foo(x)
        assert_almost_equal(out.asnumpy(), (x.asnumpy() + const_arr), atol=1e-5, rtol=1e-4, use_broadcast=False)


@use_np
def test_parameters_zero_grad():
    for hybridize in [False, True]:
        net = gluon.nn.HybridSequential()
        for _ in range(5):
            net.add(gluon.nn.Dense(10))
        if hybridize:
            net.hybridize()
        net.initialize()
        out = net(mx.np.ones((32, 8)))
        for v in net.collect_params().values():
            v.grad()[()] = 1
        net.zero_grad()
        for v in net.collect_params().values():
            assert_almost_equal(v.grad().asnumpy(), mx.np.zeros_like(v.grad()).asnumpy())


def check_gluon_save_load(net_builder, data_l):
    """Verify the consistency between the loaded network and the original network.

    Known limitations: Currently it only supports loading

    Parameters
    ----------
    net_builder : function
        The builder of the HybridBlock.
    data_l : list of numpy.ndarray
        List of the input data that we will use to verify the correctness of the loaded network.

    """
    net = net_builder()
    net.hybridize()
    net.initialize()
    out = net(*data_l)
    out_np = out.asnumpy()
    prefix = str(uuid4())
    net.export(prefix)
    input_names = 'data' if len(data_l) == 1 else ['data{}'.format(i) for i in range(len(data_l))]
    net_imported = gluon.SymbolBlock.imports('{}-symbol.json'.format(prefix),
                                             input_names, param_file='{}-0000.params'.format(prefix))
    # Clean up the directory
    os.remove('{}-symbol.json'.format(prefix))
    os.remove('{}-0000.params'.format(prefix))
    loaded_out = net_imported(*data_l).asnumpy()
    assert_almost_equal(out_np, loaded_out)


def hashable_index(tuple_idx):
    """Return an hashable representation of a tuple of slice object

    We add this because the slice object in python is not hashable.

    Parameters
    ----------
    tuple_idx : tuple
        A tuple of slice/int objects

    Returns
    -------
    ret : tuple
        A hashable representation of the slice data
    """
    l = []
    for ele in tuple_idx:
        if isinstance(ele, slice):
            l.append(ele.__reduce__())
        else:
            l.append(ele)
    return tuple(l)


@with_seed()
@use_np
def test_symbolic_basic_slicing():
    def random_slice_index(shape):
        index = []
        step_switch = random.randint(-1, 1)
        for i in range(len(shape)):
            if shape[i] == 0:
                index.append(slice(None))
                continue
            r = random.randint(0, 5)
            if r < 2:
                index.append(random.randint(1 - shape[i], shape[i] - 1))
                continue
            elif r < 3:
                index.append(slice(None))
                continue
            s = random.randint(0, shape[i] - 1)
            e = random.randint(s + 1, shape[i])
            if step_switch == 1:
                index.append(slice(s, e, 1))
            elif step_switch == -1:
                e -= 1
                s -= 1
                index.append(slice(e, s, -1))
            else:
                index.append(slice(s, e, 2))
        return tuple(index)

    shapes = [
        (4, 6, 8, 5),
        (1, 1, 1, 6),
        (5, 6, 4),
        (5, 6),
        (10,),
        (100, 0, 10, 0, 0),
        (100, 0, 0),
        (0, 0, 0),
        (),
    ]
    for shape in shapes:
        cache = set()
        x = mx.np.random.normal(0, 1, shape)
        y = mx.np.random.normal(0, 1, shape)
        for _ in range(200):
            index1 = random_slice_index(shape)
            index2 = random_slice_index(x.asnumpy()[index1].shape)
            if (hashable_index(index1), hashable_index(index2)) in cache:
                continue
            cache.add((hashable_index(index1), hashable_index(index2)))
            # Test basic slicing on a single symbol
            class TestSlicingSingleSymbol1(gluon.HybridBlock):
                def hybrid_forward(self, F, x, y):
                    return x[()][index1] + y[()][index1]

            # Test basic slicing on a single symbol
            class TestSlicingSingleSymbol2(gluon.HybridBlock):
                def hybrid_forward(self, F, x, y):
                    return (x[()][index1] + y[()][index1])[index2]

            check_gluon_hybridize_consistency(TestSlicingSingleSymbol1, [x, y],
                                              numpy_func=lambda a, b: a[()][index1] + b[()][index1])
            check_gluon_hybridize_consistency(TestSlicingSingleSymbol2, [x, y],
                                              numpy_func=lambda a, b:
                                              (a[()][index1] + b[()][index1])[index2])
        # Test for split/hsplit/vsplit
        class TestSlicingWithSplit(gluon.HybridBlock):
            def hybrid_forward(self, F, x):
                x = F.np.split(x, shape[2], axis=2)
                x = x[1:-1]
                x = F.np.concatenate(x, axis=2)
                return x

        class TestSlicingWithSplit2(gluon.HybridBlock):
            def __init__(self):
                super(TestSlicingWithSplit2, self).__init__()
                self.layer = gluon.nn.Dense(16, flatten=False)

            def hybrid_forward(self, F, x, y):
                x = F.np.split(x, 1)
                x = x[0]
                return self.layer(x[:, -1, :] + y[:, -1, :])

        class TestSlicingWithHSplit(gluon.HybridBlock):
            def hybrid_forward(self, F, x):
                x = F.np.hsplit(x, shape[1])
                x = x[1:-1]
                x = F.np.concatenate(x, axis=1)
                return x

        class TestSlicingWithVSplit(gluon.HybridBlock):
            def hybrid_forward(self, F, x):
                x = F.np.vsplit(x, shape[0])
                x = x[1:-1]
                x = F.np.concatenate(x, axis=0)
                return x

        if len(shape) > 2 and shape[2] > 2:
            check_gluon_hybridize_consistency(
                TestSlicingWithSplit, [x],
                numpy_func=lambda a: _np.concatenate(_np.split(a, shape[2], axis=2)[1:-1],
                                                     axis=2))
        if len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
            check_gluon_hybridize_consistency(TestSlicingWithSplit2, [x, y])

        if len(shape) > 1 and shape[1] > 2:
            check_gluon_hybridize_consistency(
                TestSlicingWithHSplit, [x],
                numpy_func=lambda a: _np.concatenate(_np.hsplit(a, shape[1])[1:-1], axis=1))
        if len(shape) > 1 and shape[0] > 2:
            check_gluon_hybridize_consistency(
                TestSlicingWithVSplit, [x],
                numpy_func=lambda a: _np.concatenate(_np.vsplit(a, shape[0])[1:-1], axis=0))

    for data_shape, idx in [((4, 3), 2),
                            ((3,), -1),
                            ((3,), 0)]:
        class IntegerIndexing(gluon.HybridBlock):
            def hybrid_forward(self, F, x):
                return x[idx]
        check_gluon_hybridize_consistency(IntegerIndexing,
                                          [mx.np.ones(data_shape)],
                                          numpy_func=lambda a: a[idx])


@with_seed()
@use_np
def test_net_symbol_save_load():
    class Case1(gluon.HybridBlock):
        def __init__(self):
            super(Case1, self).__init__()
            self.layer = gluon.nn.Dense(64, flatten=False)

        def hybrid_forward(self, F, x, y):
            x = F.np.split(x, 1)
            x = x[0]
            return self.layer(x[:, -1, :] + y[:, -1, :])
    check_gluon_save_load(Case1, [mx.np.random.normal(0, 1, (10, 5, 8, 6)),
                                  mx.np.random.normal(0, 1, (10, 5, 8, 6))])

    class Case2(gluon.HybridBlock):
        def __init__(self):
            super(Case2, self).__init__()
            self.layer1 = gluon.nn.Dense(64, flatten=False)
            self.layer2 = gluon.nn.Dense(64, flatten=False)

        def hybrid_forward(self, F, x, y):
            x = F.np.split(x, 1)
            x = x[0]
            return self.layer1(x[:, -1, :]) + self.layer2(y[:, -1, :])
    check_gluon_save_load(Case2, [mx.np.random.normal(0, 1, (10, 5, 8)),
                                  mx.np.random.normal(0, 1, (10, 5, 8))])


@with_seed()
@use_np
def test_hybridize_boolean_dtype():
    class Foo(gluon.HybridBlock):
        def __init__(self):
            super(Foo, self).__init__()

        def hybrid_forward(self, F, valid_length):
            mask = ((F.np.ones((10,)) / 2) < valid_length)
            return mask

    valid_length = mx.np.random.uniform(size=(10,))
    foo = Foo()
    out1 = foo(valid_length)

    foo = Foo()
    foo.hybridize()
    out2 = foo(valid_length)

    assert mx.test_utils.same(out1.asnumpy(), out2.asnumpy())


@with_seed()
@use_np
def test_activations_leakyrelu():
    # Currently, all the activation tests, we will just test for runnable.
    act_layer = nn.LeakyReLU(0.1)
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@with_seed()
@use_np
def test_activations_prelu():
    act_layer = nn.PReLU()
    act_layer.initialize()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@with_seed()
@use_np
def test_activations_elu():
    act_layer = nn.ELU(1.0)
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@with_seed()
@use_np
def test_activations_selu():
    act_layer = nn.SELU()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@with_seed()
@use_np
def test_activations_gelu():
    act_layer = nn.GELU()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()


@with_seed()
@use_np
def test_activations_swish():
    act_layer = nn.Swish()
    out = act_layer(mx.np.random.uniform(size=(10,)))
    out.asnumpy()
