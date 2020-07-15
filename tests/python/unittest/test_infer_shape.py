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
import mxnet as mx
from common import models
import pytest

mx.reset_np()

def test_mlp2_infer_shape():
    # Build MLP
    out = models.mlp2()
    # infer shape
    data_shape = (100, 100)
    arg_shapes, out_shapes, aux_shapes = out.infer_shape(data=data_shape)
    arg_shape_dict = dict(zip(out.list_arguments(), arg_shapes))
    assert len(out_shapes) == 1
    assert out_shapes[0] == (100, 10)
    true_shapes = {'fc2_bias': (10,),
                   'fc2_weight' : (10, 1000),
                   'fc1_bias' : (1000,),
                   'fc1_weight' : (1000,100) }
    for k, v in true_shapes.items():
        assert arg_shape_dict[k] == v

def test_mlp2_infer_error():
    # Test shape inconsistent case
    out = models.mlp2()
    weight_shape= (1, 100)
    data_shape = (100, 100)
    with pytest.raises(mx.MXNetError):
        arg_shapes, out_shapes, aux_shapes = out.infer_shape(data=data_shape, fc1_weight=weight_shape)


def test_backward_infer():
    w = mx.sym.Variable("weight")
    wshift = mx.sym.Variable("wshift", shape=(1,))
    data = mx.sym.Variable("data")
    # broadcast add here, not being able to deduce shape correctly
    wt = mx.sym.broadcast_add(w, wshift)
    # shape constraint, this is what enables backward shape inference
    wt = mx.symbol._internal._identity_with_attr_like_rhs(wt, w)
    net = mx.sym.FullyConnected(data=data, weight=wt, num_hidden=11, no_bias=True)
    data_shape = (7, 100)
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=data_shape)
    arg_shape_dict = dict(zip(net.list_arguments(), arg_shapes))
    true_shapes = {'weight': (11, 100)}
    for k, v in true_shapes.items():
        assert arg_shape_dict[k] == v


def test_incomplete_infer_elewise():
    a = mx.sym.Variable('a', shape=(0, 10))
    b = mx.sym.Variable('b', shape=(12, 0))
    c = a + b
    arg_shapes, _, _ = c.infer_shape()
    arg_names = c.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (12, 10)
    assert arg_shapes['b'] == (12, 10)


def test_incomplete_infer_mlp():
    a = mx.sym.Variable('a', shape=(0, 10))
    b = mx.sym.FullyConnected(data=a, num_hidden=21)
    c = mx.sym.Variable('c', shape=(5, 0))
    d = b + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (5, 10)
    assert arg_shapes['c'] == (5, 21)


def test_incomplete_infer_slicechannel():
    a = mx.sym.Variable('a', shape=(0, 10))
    b = mx.sym.SliceChannel(data=a, num_outputs=10, axis=1, squeeze_axis=True)
    c = mx.sym.Variable('c', shape=(5,))
    d = b[1] + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (5, 10)

    a = mx.sym.Variable('a', shape=(0, 15, 0))
    b = mx.sym.SliceChannel(data=a, num_outputs=3, squeeze_axis=False)
    c = mx.sym.Variable('c', shape=(3, 5, 2))
    d = b[1] + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (3, 15, 2)


def test_incomplete_infer_convolution():
    a = mx.sym.Variable('a', shape=(0, 10, 0, 0))
    b = mx.sym.Convolution(data=a, num_filter=21, kernel=(3, 3), dilate=(1, 1), pad=(1, 1))
    c = mx.sym.Variable('c', shape=(5, 21, 32, 32))
    d = b + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (5, 10, 32, 32)


def test_incomplete_infer_concat():
    a = mx.sym.Variable('a', shape=(0, 10))
    b = mx.sym.Variable('b', shape=(0, 5))
    c = mx.sym.Concat(a, b, num_args=2, dim=1)
    d = mx.sym.Variable('d', shape=(2, 0))
    d = d + c
    arg_shapes, _, _ = d.infer_shape()
    arg_names = d.list_arguments()
    arg_shapes = {k: v for k, v in zip(arg_names, arg_shapes)}
    assert arg_shapes['a'] == (2, 10)
    assert arg_shapes['b'] == (2, 5)
    assert arg_shapes['d'] == (2, 15)

def test_fc_infer_type():
    mx_real_t = mx.base.mx_real_t
    data = mx.symbol.Variable('data')
    out = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=1000)

    # infer type
    data_type = mx_real_t
    arg_types, out_types, aux_types = out.infer_type(data=data_type)
    arg_type_dict = dict(zip(out.list_arguments(), arg_types))
    assert len(out_types) == 1
    assert out_types[0] == mx_real_t
    true_types = {
                   'fc1_bias' : mx_real_t,
                   'fc1_weight' : mx_real_t }
    for k, v in true_types.items():
        assert arg_type_dict[k] == v


def test_shape_completely_unknown():
    data = mx.sym.var("data")
    ret = mx.sym.sin(data)
    arg_shapes, out_shapes, _ = ret.infer_shape_partial()
    assert arg_shapes[0] == ()
    assert out_shapes[0] == ()

    with mx.np_shape():
        data = mx.sym.var("data")
        ret = mx.sym.sin(data)
        arg_shapes, out_shapes, _ = ret.infer_shape_partial()
        assert arg_shapes[0] is None
        assert out_shapes[0] is None


def test_dot_partial_shape():
    x = mx.sym.Variable("x")
    y = mx.sym.Variable("y")
    z = mx.sym.dot(x, y)
    # batch size(first dim) of lhs unknown
    _, result_shape, _ = z.infer_shape_partial(x=(0, 3, 4), y=(4, 5))
    assert result_shape == [(0, 3, 5)]
    with mx.np_shape(True):
        _, result_shape, _ =  z.infer_shape_partial(x=(-1, 3, 4), y=(4, 5))
        assert result_shape == [(-1, 3, 5)]


def test_batch_dot_partial_shape():
    x = mx.sym.Variable("x")
    y = mx.sym.Variable("y")
    z = mx.sym.batch_dot(x, y)
    # lhs and rhs batch size unknown
    _, result_shape, _ = z.infer_shape_partial(x=(0, 3, 4), y=(0, 4, 5))
    assert result_shape == [(0, 3, 5)]
    # rhs second dim unknown
    _, result_shape, _ = z.infer_shape_partial(x=(0, 3, 4), y=(0, 0, 5))
    assert result_shape == [()]
    with mx.np_shape(True):
        _, result_shape, _ =  z.infer_shape_partial(x=(-1, 3, 4), y=(-1, 4, 5))
        assert result_shape == [(-1, 3, 5)]
        _, result_shape, _ =  z.infer_shape_partial(x=(-1, 3, 4), y=(-1, -1, 5))
        assert result_shape == [None]


def test_embedding_partial_shape():
    # testing embedding with batch size unknown
    x = mx.sym.Variable("x")
    w = mx.sym.Variable("w")
    y = mx.sym.Embedding(data=x, weight=w, input_dim=100, output_dim=10)
    _, result_shape, _ = y.infer_shape_partial(x=(0, 5), w=(100, 10))
    assert result_shape  == [(0, 5, 10)]
    with mx.np_shape(True):
        _, result_shape, _ = y.infer_shape_partial(x=(-1, 5), w=(100, 10))
        assert result_shape == [(-1, 5, 10)]


def test_transpose_partial_shape():
    # test converting tensor shape
    # from channels first to channels last
    # with batch size unknown
    axes = [0, 3, 2, 1]
    x = mx.sym.Variable("x")
    y = mx.sym.transpose(x, axes=axes)
    _, result, _ = y.infer_shape_partial(x=(0, 3, 224, 224))
    assert result == [(0, 224, 224, 3)]

    with mx.np_shape(True):
        _, result, _ = y.infer_shape_partial(x=(-1, 3, 224, 224))
        assert result == [(-1, 224, 224, 3)]


def test_pick_partial_shape():
    x = mx.sym.Variable("x")
    index = mx.sym.Variable("index")
    y = mx.sym.pick(x, index, axis=1)
    # batch size unknown
    _, result, _ =  y.infer_shape_partial(x=(0, 3, 3), index=(0, 3,))
    assert result == [(0, 3)]
    with mx.np_shape(True):
        _, result, _ =  y.infer_shape_partial(x=(-1, 3, 3), index=(-1, 3,))
        assert result == [(-1, 3)]


def test_where_partial_shape():
    x = mx.sym.Variable("x")
    y = mx.sym.Variable("y")
    cond = mx.sym.Variable("cond")
    where_op = mx.sym.where(cond, x, y)
    # condition must be fully known to infer shape
    _, result, _ = where_op.infer_shape_partial(cond=(0, 2), x=(0, 2), y =(0, 2))
    assert result == [()]
    _, result, _ = where_op.infer_shape_partial(cond=(0,), x=(2, 2), y =(2, 2))
    assert result == [()]
    with mx.np_shape(True):
        _, result, _ =  where_op.infer_shape_partial(cond=(-1, 2), x=(-1, 2), y =(-1, 2))
        assert result == [None]
        _, result, _ = where_op.infer_shape_partial(cond=(-1,), x=(2, 2), y=(2, 2))
        assert result == [None]

