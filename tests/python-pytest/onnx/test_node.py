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

"""
Tests for individual operators
This module contains operator tests which currently do not exist on
ONNX backend test framework. Once we have PRs on the ONNX repo and get
those PRs merged, this file will get EOL'ed.
"""
# pylint: disable=too-many-locals,wrong-import-position,import-error
from __future__ import absolute_import
import sys
import os
import unittest
import logging
import tarfile
from collections import namedtuple
import numpy as np
import numpy.testing as npt
from onnx import checker, numpy_helper, helper, load_model
from onnx import TensorProto
from mxnet.test_utils import download
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx
import backend

CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(CURR_PATH, '../../python/unittest'))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_rnd(shape, low=-1.0, high=1.0, dtype=np.float32):
    if dtype == np.float32:
        return (np.random.uniform(low, high,
                                  np.prod(shape)).reshape(shape).astype(np.float32))
    elif dtype == np.int32:
        return (np.random.randint(low, high,
                                  np.prod(shape)).reshape(shape).astype(np.float32))
    elif dtype == np.bool_:
        return np.random.choice(a=[False, True], size=shape).astype(np.float32)


def _fix_attributes(attrs, attribute_mapping):
    new_attrs = attrs
    attr_modify = attribute_mapping.get('modify', {})
    for k, v in attr_modify.items():
        new_attrs[v] = new_attrs.pop(k, None)

    attr_add = attribute_mapping.get('add', {})
    for k, v in attr_add.items():
        new_attrs[k] = v

    attr_remove = attribute_mapping.get('remove', [])
    for k in attr_remove:
        if k in new_attrs:
            del new_attrs[k]

    return new_attrs


def forward_pass(sym, arg, aux, data_names, input_data):
    """ Perform forward pass on given data
    :param sym: Symbol
    :param arg: Arg params
    :param aux: Aux params
    :param data_names: Input names (list)
    :param input_data: Input data (list). If there is only one input,
                        pass it as a list. For example, if input is [1, 2],
                        pass input_data=[[1, 2]]
    :return: result of forward pass
    """
    data_shapes = []
    data_forward = []
    for idx in range(len(data_names)):
        val = input_data[idx]
        data_shapes.append((data_names[idx], np.shape(val)))
        data_forward.append(mx.nd.array(val))
    # create module
    mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)
    if not arg and not aux:
        mod.init_params()
    else:
        mod.set_params(arg_params=arg, aux_params=aux,
                       allow_missing=True, allow_extra=True)
    # run inference
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch(data_forward), is_train=False)

    return mod.get_outputs()[0].asnumpy()


def get_input_tensors(input_data):
    input_tensor = []
    input_names = []
    input_sym = []
    for idx, ip in enumerate(input_data):
        name = "input" + str(idx + 1)
        input_sym.append(mx.sym.Variable(name))
        input_names.append(name)
        input_tensor.append(helper.make_tensor_value_info(name,
                                                          TensorProto.FLOAT, shape=np.shape(ip)))
    return input_names, input_tensor, input_sym


def get_onnx_graph(testname, input_names, inputs, output_name, output_shape, attr):
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=output_shape)]

    nodes = [helper.make_node(output_name, input_names, ["output"], **attr)]

    graph = helper.make_graph(nodes, testname, inputs, outputs)

    model = helper.make_model(graph)
    return model

class TestNode(unittest.TestCase):
    """ Tests for models.
    Tests are dynamically added.
    Therefore edit test_models to add more tests.
    """
    def test_import_export(self):
        for test in test_cases:
            test_name, mxnet_op, onnx_name, inputs, attrs, mxnet_specific, fix_attrs, check_value, check_shape = test
            with self.subTest(test_name):
                names, input_tensors, inputsym = get_input_tensors(inputs)
                if inputs:
                    test_op = mxnet_op(*inputsym, **attrs)
                    mxnet_output = forward_pass(test_op, None, None, names, inputs)
                    outputshape = np.shape(mxnet_output)
                else:
                    test_op = mxnet_op(**attrs)
                    shape = attrs.get('shape', (1,))
                    x = mx.nd.zeros(shape, dtype='float32')
                    xgrad = mx.nd.zeros(shape, dtype='float32')
                    exe = test_op.bind(ctx=mx.cpu(), args={'x': x}, args_grad={'x': xgrad})
                    mxnet_output = exe.forward(is_train=False)[0].asnumpy()
                    outputshape = np.shape(mxnet_output)

                if mxnet_specific:
                    onnxmodelfile = onnx_mxnet.export_model(test_op, {}, [np.shape(ip) for ip in inputs],
                                                            np.float32,
                                                            onnx_name + ".onnx")
                    onnxmodel = load_model(onnxmodelfile)
                else:
                    onnx_attrs = _fix_attributes(attrs, fix_attrs)
                    onnxmodel = get_onnx_graph(test_name, names, input_tensors, onnx_name, outputshape, onnx_attrs)

                bkd_rep = backend.prepare(onnxmodel, operation='export')
                output = bkd_rep.run(inputs)

                if check_value:
                    npt.assert_almost_equal(output[0], mxnet_output)

                if check_shape:
                    npt.assert_equal(output[0].shape, outputshape)

        input1 = get_rnd((1, 10, 2, 3))
        ipsym = mx.sym.Variable("input1")
        for test in test_scalar_ops:
            if test == 'Add':
                outsym = 2 + ipsym
            if test == "Sub":
                outsym = ipsym - 2
            if test == "rSub":
                outsym = ipsym.__rsub__(2)
            if test == "Mul":
                outsym = 2 * ipsym
            if test == "Div":
                outsym = ipsym / 2
            if test == "Pow":
                outsym = ipsym ** 2
            forward_op = forward_pass(outsym, None, None, ['input1'], input1)
            converted_model = onnx_mxnet.export_model(outsym, {}, [np.shape(input1)], np.float32,
                                                      onnx_file_path=outsym.name + ".onnx")

            sym, arg_params, aux_params = onnx_mxnet.import_model(converted_model)
        result = forward_pass(sym, arg_params, aux_params, ['input1'], input1)

        npt.assert_almost_equal(result, forward_op)

    def test_imports(self):
        for test in import_test_cases:
            test_name, onnx_name, inputs, np_op, attrs = test
            with self.subTest(test_name):
                names, input_tensors, inputsym = get_input_tensors(inputs)
                np_out = [np_op(*inputs, **attrs)]
                output_shape = np.shape(np_out)
                onnx_model = get_onnx_graph(test_name, names, input_tensors, onnx_name, output_shape, attrs)
                bkd_rep = backend.prepare(onnx_model, operation='import')
                mxnet_out = bkd_rep.run(inputs)
                npt.assert_almost_equal(np_out, mxnet_out, decimal=4)

    def test_exports(self):
        for test in export_test_cases:
            test_name, onnx_name, mx_op, input_shapes, attrs = test
            input_syms = [mx.sym.var('data' + str(i)) for i in range(len(input_shapes))]
            outsym = mx_op(*input_syms, **attrs)
            converted_model = onnx_mxnet.export_model(outsym, {}, input_shapes, np.float32,
                                                      onnx_file_path=outsym.name + ".onnx")
            model = load_model(converted_model)
            checker.check_model(model)


# test_case = ("test_case_name", mxnet op, "ONNX_op_name", [input_list], attribute map, MXNet_specific=True/False,
# fix_attributes = {'modify': {mxnet_attr_name: onnx_attr_name},
#                   'remove': [attr_name],
#                   'add': {attr_name: value},
# check_value=True/False, check_shape=True/False)
test_cases = [
    ("test_equal", mx.sym.broadcast_equal, "Equal", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False, {}, True,
     False),
    ("test_greater", mx.sym.broadcast_greater, "Greater", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False, {}, True,
     False),
    ("test_less", mx.sym.broadcast_lesser, "Less", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False, {}, True,
     False),
    ("test_and", mx.sym.broadcast_logical_and, "And",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False, {}, True, False),
    ("test_xor", mx.sym.broadcast_logical_xor, "Xor",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False, {}, True, False),
    ("test_or", mx.sym.broadcast_logical_or, "Or",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False, {}, True, False),
    ("test_not", mx.sym.logical_not, "Not", [get_rnd((3, 4, 5), dtype=np.bool_)], {}, False, {}, True, False),
    ("test_square", mx.sym.square, "Pow", [get_rnd((2, 3), dtype=np.int32)], {}, True, {}, True, False),
    ("test_spacetodepth", mx.sym.space_to_depth, "SpaceToDepth", [get_rnd((1, 1, 4, 6))],
     {'block_size': 2}, False, {}, True, False),
    ("test_softmax", mx.sym.SoftmaxOutput, "Softmax", [get_rnd((1000, 1000)), get_rnd(1000)],
     {'ignore_label': 0, 'use_ignore': False}, True, {}, True, False),
    ("test_logistic_regression", mx.sym.LogisticRegressionOutput, "Sigmoid",
     [get_rnd((1000, 1000)), get_rnd((1000, 1000))], {}, True, {}, True, False),
    ("test_fullyconnected", mx.sym.FullyConnected, "Gemm", [get_rnd((4, 3)), get_rnd((4, 3)), get_rnd(4)],
     {'num_hidden': 4, 'name': 'FC'}, True, {}, True, False),
    ("test_lppool1", mx.sym.Pooling, "LpPool", [get_rnd((2, 3, 20, 20))],
     {'kernel': (4, 5), 'pad': (0, 0), 'stride': (1, 1), 'p_value': 1, 'pool_type': 'lp'}, False,
     {'modify': {'kernel': 'kernel_shape', 'pad': 'pads', 'stride': 'strides', 'p_value': 'p'},
      'remove': ['pool_type']}, True, False),
    ("test_lppool2", mx.sym.Pooling, "LpPool", [get_rnd((2, 3, 20, 20))],
     {'kernel': (4, 5), 'pad': (0, 0), 'stride': (1, 1), 'p_value': 2, 'pool_type': 'lp'}, False,
     {'modify': {'kernel': 'kernel_shape', 'pad': 'pads', 'stride': 'strides', 'p_value': 'p'},
      'remove': ['pool_type']}, True, False),
    ("test_globallppool1", mx.sym.Pooling, "GlobalLpPool", [get_rnd((2, 3, 20, 20))],
     {'kernel': (4, 5), 'pad': (0, 0), 'stride': (1, 1), 'p_value': 1, 'pool_type': 'lp', 'global_pool': True}, False,
     {'modify': {'p_value': 'p'},
      'remove': ['pool_type', 'kernel', 'pad', 'stride', 'global_pool']}, True, False),
    ("test_globallppool2", mx.sym.Pooling, "GlobalLpPool", [get_rnd((2, 3, 20, 20))],
     {'kernel': (4, 5), 'pad': (0, 0), 'stride': (1, 1), 'p_value': 2, 'pool_type': 'lp', 'global_pool': True}, False,
     {'modify': {'p_value': 'p'},
      'remove': ['pool_type', 'kernel', 'pad', 'stride', 'global_pool']}, True, False),
    ("test_roipool", mx.sym.ROIPooling, "MaxRoiPool",
     [[[get_rnd(shape=(8, 6), low=1, high=100, dtype=np.int32)]], [[0, 0, 0, 4, 4]]],
     {'pooled_size': (2, 2), 'spatial_scale': 0.7}, False,
     {'modify': {'pooled_size': 'pooled_shape'}}, True, False),

    # since results would be random, checking for shape alone
    ("test_multinomial", mx.sym.sample_multinomial, "Multinomial",
     [np.array([0, 0.1, 0.2, 0.3, 0.4]).astype("float32")],
     {'shape': (10,)}, False, {'modify': {'shape': 'sample_size'}}, False, True),
    ("test_random_normal", mx.sym.random_normal, "RandomNormal", [],
     {'shape': (2, 2), 'loc': 0, 'scale': 1}, False, {'modify': {'loc': 'mean'}}, False, True),
    ("test_random_uniform", mx.sym.random_uniform, "RandomUniform", [],
     {'shape': (2, 2), 'low': 0.5, 'high': 1.0}, False, {}, False, True)
]

test_scalar_ops = ['Add', 'Sub', 'rSub' 'Mul', 'Div', 'Pow']

# test_case = ("test_case_name", "ONNX_op_name", [input_list], np_op, attribute map)
import_test_cases = [
    ("test_lpnormalization_default", "LpNormalization", [get_rnd([5, 3, 3, 2])], np.linalg.norm, {'ord':2, 'axis':-1}),
    ("test_lpnormalization_ord1", "LpNormalization", [get_rnd([5, 3, 3, 2])], np.linalg.norm, {'ord':1, 'axis':-1}),
    ("test_lpnormalization_ord2", "LpNormalization", [get_rnd([5, 3, 3, 2])], np.linalg.norm, {'ord':2, 'axis':1})
]

# test_case = ("test_case_name", "ONNX_op_name", mxnet_op, input_shapes, attribute map)
export_test_cases = [
    ("test_expand", "Expand", mx.sym.broadcast_to, [(2,1,3,1)], {'shape': (2,1,3,1)}),
    ("test_tile", "Tile", mx.sym.tile, [(2,1,3,1)], {'reps': (2,3)}),
    ("test_topk", "TopK", mx.sym.topk, [(2, 10, 2)], {'k': 3, 'axis': 1, 'ret_typ': 'both', 'dtype': np.int64}),
    ("test_slice_axis", "Slice", mx.sym.slice_axis, [(2, 10, 2)], {'begin': 3, 'end': 7, 'axis': 1}),
    ("test_arange", "ValueInfo", mx.sym.arange, [], {'start':2, 'stop': 10, 'step': 2}),
    ("test_zeros_like", "Expand", mx.sym.zeros_like, [(2, 1, 3, 1)], {}),
    ("test_ones_like", "Expand", mx.sym.ones_like, [(2, 1, 3, 1)], {}),
    ("test_lesser_scalar", "Less", mx.sym._internal._lesser_scalar, [(2, 1, 3, 1)], {'scalar': 0}),
    ("test_greater_scalar", "Greater", mx.sym._internal._greater_scalar, [(2, 1, 3, 1)], {'scalar': 0}),
    ("test_equal_scalar", "Equal", mx.sym._internal._equal_scalar, [(2, 1, 3, 1)], {'scalar': 0}),
    ("test_slice_like", "Slice", mx.sym.slice_like, [(2, 10, 20), (7, 5, 3)], {'axes': (1, 2)}),
    ("test_where", "Where", mx.sym.where, [(2, 2), (2, 2), (2, 2)], {}),
]

if __name__ == '__main__':
    unittest.main()
