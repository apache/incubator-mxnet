
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

# coding: utf-8
"""GluonNLP BERT specific translation logics"""
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
try:
    import onnx
except ImportError:
    onnx = None

def get_cheat_sheet(kwargs):
    cheat_sheet = kwargs.get('cheat_sheet', None)
    if cheat_sheet is None:
        logging.warning('cheat_sheet not found, using default vallues')
        cheat_sheet = {
            'qkv_hidden': 768,
            'num_heads': 12,
            'head_dim': 64}
    return cheat_sheet


def get_inputs(node, kwargs):
    """Helper function to get inputs"""
    name = node["name"]
    outputs_lookup = kwargs["outputs_lookup"]
    inputs = node["inputs"]
    attrs = node.get("attrs", {})

    input_nodes = []
    for ip in inputs:
        input_node_name = outputs_lookup[ip[0]][ip[1]].name
        input_nodes.append(input_node_name)

    return name, input_nodes, attrs


def get_input_dtypes(node, kwargs):
    outputs_lookup = kwargs['outputs_lookup']
    inputs = node['inputs']
    input_dtypes = []
    for ip in inputs:
        input_node_dtype = outputs_lookup[ip[0]][ip[1]].dtype
        input_dtypes.append(input_node_dtype)
    return input_dtypes


def get_boolean_attribute_value(attrs, attr_name):
    """ Helper function to convert a string version
    of Boolean attributes to integer for ONNX.
    Takes attribute dictionary and attr_name as
    parameters.
    """
    return 1 if attrs.get(attr_name, 0) in ["True", "1"] else 0


def create_const_scalar_node(input_name, value, kwargs):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    from onnx.helper import make_tensor
    initializer = kwargs["initializer"]
    dtype = value.dtype
    if dtype == 'float16':
        # when using float16, we must convert it to np.uint16 view first
        value = np.float16(value).view(np.uint16) #pylint: disable=too-many-function-args
    input_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    tensor_node = make_tensor(input_name, input_type, (), ([value]))
    initializer.append(tensor_node)

def create_const_node(input_name, value, kwargs):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    from onnx.helper import make_tensor
    initializer = kwargs["initializer"]
    dtype = value.dtype
    if dtype == 'float16':
        # when using float16, we must convert it to np.uint16 view first
        value = np.float16(value).view(np.uint16) #pylint: disable=too-many-function-args
    input_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[dtype]
    input_shape = value.shape
    tensor_node = make_tensor(input_name, input_type, input_shape, value)
    initializer.append(tensor_node)


def create_tensor(tensor_list, tensor_name, initializer, dtype='int64'):
    """Helper function to create a tensor value node and a
    initializer tensor node with constant value."""
    tensor_np = np.array(tensor_list, dtype=dtype)
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[tensor_np.dtype]
    dims = np.shape(tensor_np)
    if dtype == np.float16:
        tensor_np = tensor_np.view(dtype=np.uint16)
    initializer.append(
        onnx.helper.make_tensor(
            name=tensor_name,
            data_type=data_type,
            dims=dims,
            vals=tensor_np.flatten().tolist(),
            raw=False
        )
    )


@mx_op.register("FullyConnected", opset_version='gluonnlp_bert_uninterleaved')
def convert_fully_connected(node, **kwargs):
    """Map MXNet's FullyConnected operator attributes to onnx's Gemm operator
    and return the created node.
    """
    from onnx.helper import make_node
    name, input_nodes, attrs = get_inputs(node, kwargs)
    input_dtypes = get_input_dtypes(node, kwargs)
    dtype = input_dtypes[0]
    flatten = get_boolean_attribute_value(attrs, 'flatten')
    no_bias = get_boolean_attribute_value(attrs, 'no_bias')
    num_hidden = int(attrs.get('num_hidden'))

    nodes = []

    if 'dotproductselfattentioncell' in name:
        cheat_sheet = get_cheat_sheet(kwargs)
        qkv_hidden = cheat_sheet['qkv_hidden']
        num_heads = cheat_sheet['num_heads']
        head_dim = cheat_sheet['head_dim']
        create_tensor([num_heads, 3 * head_dim, qkv_hidden], name+'_interleaved_w_shape', kwargs['initializer'])
        create_tensor([num_heads, 3 * head_dim], name+'_interleaved_b_shape', kwargs['initializer'])
        create_tensor([qkv_hidden, qkv_hidden], name+'_w_shape', kwargs['initializer'])
        create_tensor([qkv_hidden], name+'_b_shape', kwargs['initializer'])
        nodes += [
            make_node('Reshape', [input_nodes[1], name+'_interleaved_w_shape'], [name+'_interleaved_w']),
            make_node('Split', [name+'_interleaved_w'], [name+'_q_w_', name+'_k_w_', name+'_v_w_'], axis=1),
            make_node('Reshape', [name+'_q_w_', name+'_w_shape'], [name+'_q_w_reshaped']),
            make_node('Reshape', [name+'_k_w_', name+'_w_shape'], [name+'_k_w_reshaped']),
            make_node('Reshape', [name+'_v_w_', name+'_w_shape'], [name+'_v_w_reshaped']),
            make_node('Transpose', [name+'_q_w_reshaped'], [name+'_q_w']),
            make_node('Transpose', [name+'_k_w_reshaped'], [name+'_k_w']),
            make_node('Transpose', [name+'_v_w_reshaped'], [name+'_v_w']),
            make_node('Reshape', [input_nodes[2], name+'_interleaved_b_shape'], [name+'_interleaved_b']),
            make_node('Split', [name+'_interleaved_b'], [name+'_q_b_', name+'_k_b_', name+'_v_b_'], axis=1),
            make_node('Reshape', [name+'_q_b_', name+'_b_shape'], [name+'_q_b']),
            make_node('Reshape', [name+'_k_b_', name+'_b_shape'], [name+'_k_b']),
            make_node('Reshape', [name+'_v_b_', name+'_b_shape'], [name+'_v_b']),
            make_node('MatMul', [input_nodes[0], name+'_q_w'], [name+'_q_']),
            make_node('MatMul', [input_nodes[0], name+'_k_w'], [name+'_k_']),
            make_node('MatMul', [input_nodes[0], name+'_v_w'], [name+'_v_']),
            make_node('Add', [name+'_q_', name+'_q_b'], [name+'0']),
            make_node('Add', [name+'_k_', name+'_k_b'], [name+'1']),
            make_node('Add', [name+'_v_', name+'_v_b'], [name+'2']),
        ]
        return nodes

    if flatten:
        nodes += [
            make_node('Flatten', [input_nodes[0]], [name+'_data_flattened'])
        ]
    else:
        nodes += [
            make_node('Shape', [input_nodes[0]], [name+'_orig_shape']),
            make_node('Shape', [name+'_orig_shape'], [name+'_dim']),
            make_node('Flatten', [input_nodes[0]], [name+'_data_flattened'], axis=-1),
        ]
    in_nodes = [name+'_data_flattened', input_nodes[1]]
    if no_bias:
        create_const_scalar_node(name+'_bias', np.int32(0).astype(dtype), kwargs)
        in_nodes.append(name+'_bias')
    else:
        in_nodes.append(input_nodes[2])
    if flatten:
        nodes += [
            make_node('Gemm', in_nodes, [name], alpha=1.0, beta=1.0, transA=0, transB=1, name=name)
        ]
    else:
        create_tensor([0], name+'_0', kwargs['initializer'])
        create_tensor([1], name+'_1', kwargs['initializer'])
        create_tensor([num_hidden], name+'_num_hidden', kwargs['initializer'])
        nodes += [
            make_node('Gemm', in_nodes, [name+'_gemm'], alpha=1.0, beta=1.0, transA=0, transB=1),
            make_node('Sub', [name+'_dim', name+'_1'], [name+'dim_minus_1']),
            make_node('Slice', [name+'_orig_shape', name+'_0', name+'dim_minus_1'],
                      [name+'_shape_sliced']),
            make_node('Concat', [name+'_shape_sliced', name+'_num_hidden'],
                      [name+'_shape_new'], axis=0),
            make_node('Reshape', [name+'_gemm', name+'_shape_new'], [name], name=name)
        ]
    return nodes


@mx_op.register("_contrib_interleaved_matmul_selfatt_qk", opset_version='gluonnlp_bert_uninterleaved')
def convert_matmul_selfatt_qk(node, **kwargs):
    """Map MXNet's _contrib_interleaved_matmul_selfatt_qk operator
    """
    from onnx.helper import make_node
    import copy

    inp0 = node['inputs'][0]
    inp1 = copy.deepcopy(inp0)
    inp1[1] = 1
    node['inputs'] = [inp0, inp1]
    name, input_nodes, _ = get_inputs(node, kwargs)

    cheat_sheet = get_cheat_sheet(kwargs)
    num_heads = cheat_sheet['num_heads']
    head_dim = cheat_sheet['head_dim']

    create_tensor([-1], name+'_m1', kwargs['initializer'])
    create_tensor([int(head_dim ** 0.5)], name+'_sqrt_head_dim', kwargs['initializer'], dtype='float32')
    create_tensor([0, 0, num_heads, head_dim], name+"_qkv_shape", kwargs['initializer'])
    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Split', [name+'_shape'], [name+'_sq', name+'_bs', name+'___'], axis=0),
        make_node('Reshape', [input_nodes[0], name+'_qkv_shape'], [name+'_q_']),
        make_node('Reshape', [input_nodes[1], name+'_qkv_shape'], [name+'_k_']),
        make_node('Transpose', [name+'_q_'], [name+'_q'], perm=[1, 2, 0, 3]),
        make_node('Transpose', [name+'_k_'], [name+'_k'], perm=[1, 2, 3, 0]),
        make_node('MatMul', [name+'_q', name+'_k'], [name+'_qk']),
        make_node('Concat', [name+'_m1', name+'_sq', name+'_sq'], [name+'_out_shape'], axis=0),
        make_node('Reshape', [name+'_qk', name+'_out_shape'], [name+'_qk_reshaped']),
        make_node('Div', [name+'_qk_reshaped', name+'_sqrt_head_dim'], [name])
    ]

    return nodes


@mx_op.register("_contrib_interleaved_matmul_selfatt_valatt", opset_version='gluonnlp_bert_uninterleaved')
def convert_contrib_interleaved_matmul_selfatt_valatt(node, **kwargs):
    """Map MXNet's _contrib_interleaved_matmul_selfatt_valatt operator attributes to onnx's operator.
    """
    from onnx.helper import make_node
    inp0 = node['inputs'][0]
    inp0[1] = 2
    inp1 = node['inputs'][1]
    node['inputs'] = [inp0, inp1]
    name, input_nodes, _ = get_inputs(node, kwargs)

    cheat_sheet = get_cheat_sheet(kwargs)
    num_heads = cheat_sheet['num_heads']
    head_dim = cheat_sheet['head_dim']

    create_tensor([head_dim], name+'_head_dim', kwargs["initializer"])
    create_tensor([0], name+'_0', kwargs["initializer"])
    create_tensor([-1], name+'_m1', kwargs["initializer"])
    create_tensor([0, 0, num_heads, head_dim], name+"_qkv_shape", kwargs["initializer"])
    create_tensor([0, 0, -1], name+"_out_shape", kwargs["initializer"])
    nodes = [
        make_node('Shape', [input_nodes[0]], [name+'_shape']),
        make_node('Split', [name+'_shape'], [name+'_sq', name+'_bs', name+'___'], axis=0),
        make_node('Reshape', [input_nodes[0], name+"_qkv_shape"], [name+'_v__']),
        make_node('Transpose', [name+'_v__'], [name+'_v_'], perm=[1, 2, 0, 3]),
        make_node('Concat', [name+'_m1', name+'_sq', name+'_head_dim'], [name+'_v_shape'], axis=0),
        make_node('Reshape', [name+'_v_', name+'_v_shape'], [name+'_v']),
        make_node('MatMul', [input_nodes[1], name+'_v'], [name+'_matmul']),
        make_node('Concat', [name+'_bs', name+'_m1', name+'_sq', name+'_head_dim'],
                  [name+'_before_transpose'], axis=0),
        make_node('Reshape', [name+'_matmul', name+'_before_transpose'], [name+'_bt']),
        make_node('Transpose', [name+'_bt'], [name+'_transpose'], perm=[2, 0, 1, 3]),
        make_node('Reshape', [name+'_transpose', name+'_out_shape'], [name])
    ]

    return nodes
