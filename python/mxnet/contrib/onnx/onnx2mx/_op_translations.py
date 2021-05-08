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
""" Module for translating ONNX operators into Mxnet operatoes"""
# pylint: disable=unused-argument,protected-access
import numpy as np
from . import _translation_utils as translation_utils
from .... import symbol
# Method definitions for the callable objects mapped in the import_helper module

def identity(attrs, inputs, proto_obj):
    """Returns the identity function of the input."""
    return 'identity', attrs, inputs

def random_uniform(attrs, inputs, proto_obj):
    """Draw random samples from a uniform distribtuion."""
    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          "Instructions to install - https://github.com/onnx/onnx")
    new_attrs = translation_utils._remove_attributes(attrs, ['seed'])
    new_attrs['dtype'] = TENSOR_TYPE_TO_NP_TYPE[int(new_attrs.get('dtype', 1))]
    return 'random_uniform', new_attrs, inputs

def random_normal(attrs, inputs, proto_obj):
    """Draw random samples from a Gaussian distribution."""
    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          "Instructions to install - https://github.com/onnx/onnx")
    new_attr = translation_utils._remove_attributes(attrs, ['seed'])
    new_attr = translation_utils._fix_attribute_names(new_attr, {'mean': 'loc'})
    new_attr['dtype'] = TENSOR_TYPE_TO_NP_TYPE[int(new_attr.get('dtype', 1))]
    return 'random_normal', new_attr, inputs

def sample_multinomial(attrs, inputs, proto_obj):
    """Draw random samples from a multinomial distribution."""
    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")
    new_attrs = translation_utils._remove_attributes(attrs, ['seed'])
    new_attrs = translation_utils._fix_attribute_names(new_attrs, {'sample_size': 'shape'})
    new_attrs['dtype'] = TENSOR_TYPE_TO_NP_TYPE[int(attrs.get('dtype', 6))]
    return 'sample_multinomial', new_attrs, inputs

def mean(attrs, inputs, proto_obj):
    """Mean of all the input tensors."""
    concat_input = [symbol.expand_dims(op_input, axis=0) for op_input in inputs]
    concat_sym = symbol.concat(*concat_input, dim=0)
    mean_sym = symbol.mean(concat_sym, axis=0)
    return mean_sym, attrs, inputs

def logical_and(attrs, inputs, proto_obj):
    """Logical and of two input arrays."""
    return 'broadcast_logical_and', attrs, inputs

def logical_or(attrs, inputs, proto_obj):
    """Logical or of two input arrays."""
    return 'broadcast_logical_or', attrs, inputs

def logical_xor(attrs, inputs, proto_obj):
    """Logical xor of two input arrays."""
    return 'broadcast_logical_xor', attrs, inputs

def logical_not(attrs, inputs, proto_obj):
    """Logical not of two input arrays."""
    return 'logical_not', attrs, inputs

def absolute(attrs, inputs, proto_obj):
    """Returns element-wise absolute value of the input."""
    return 'abs', attrs, inputs

def negative(attrs, inputs, proto_obj):
    """Negation of every element in a tensor"""
    return 'negative', attrs, inputs

def add_n(attrs, inputs, proto_obj):
    """Elementwise sum of arrays"""
    return 'add_n', attrs, inputs

# Sorting and Searching
def argmax(attrs, inputs, proto_obj):
    """Returns indices of the maximum values along an axis"""
    axis = attrs.get('axis', 0)
    keepdims = attrs.get('keepdims', 1)
    argmax_op = symbol.argmax(inputs[0], axis=axis, keepdims=keepdims)
    # onnx argmax operator always expects int64 as output type
    cast_attrs = {'dtype': 'int64'}
    return 'cast', cast_attrs, argmax_op

def argmin(attrs, inputs, proto_obj):
    """Returns indices of the minimum values along an axis."""
    axis = attrs.get('axis', 0)
    keepdims = attrs.get('keepdims', 1)
    argmin_op = symbol.argmin(inputs[0], axis=axis, keepdims=keepdims)
    # onnx argmax operator always expects int64 as output type
    cast_attrs = {'dtype': 'int64'}
    return 'cast', cast_attrs, argmin_op

def maximum(attrs, inputs, proto_obj):
    """
    Elementwise maximum of arrays.
    MXNet maximum compares only two symbols at a time.
    ONNX can send more than two to compare.
    Breaking into multiple mxnet ops to compare two symbols at a time
    """
    if len(inputs) > 1:
        mxnet_op = symbol.maximum(inputs[0], inputs[1])
        for op_input in inputs[2:]:
            mxnet_op = symbol.maximum(mxnet_op, op_input)
    else:
        mxnet_op = symbol.maximum(inputs[0], inputs[0])
    return mxnet_op, attrs, inputs

def minimum(attrs, inputs, proto_obj):
    """Elementwise minimum of arrays."""
    # MXNet minimum compares only two symbols at a time.
    # ONNX can send more than two to compare.
    # Breaking into multiple mxnet ops to compare two symbols at a time
    if len(inputs) > 1:
        mxnet_op = symbol.minimum(inputs[0], inputs[1])
        for op_input in inputs[2:]:
            mxnet_op = symbol.minimum(mxnet_op, op_input)
    else:
        mxnet_op = symbol.minimum(inputs[0], inputs[0])
    return mxnet_op, attrs, inputs

def lesser(attrs, inputs, proto_obj):
    """Logical Lesser operator with broadcasting."""
    return 'broadcast_lesser', attrs, inputs

def greater(attrs, inputs, proto_obj):
    """Logical Greater operator with broadcasting."""
    return 'broadcast_greater', attrs, inputs

def equal(attrs, inputs, proto_obj):
    """Logical Equal operator with broadcasting."""
    return 'broadcast_equal', attrs, inputs

#Hyperbolic functions
def tanh(attrs, inputs, proto_obj):
    """Returns the hyperbolic tangent of the input array."""
    return 'tanh', attrs, inputs

# Rounding
def ceil(attrs, inputs, proto_obj):
    """ Calculate ceil value for input """
    return 'ceil', attrs, inputs

def floor(attrs, inputs, proto_obj):
    """ Calculate floor value for input """
    return 'floor', attrs, inputs

# Joining and spliting
def concat(attrs, inputs, proto_obj):
    """ Joins input arrays along a given axis. """
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axis': 'dim'})
    return 'concat', new_attrs, inputs

# Basic neural network functions
def softsign(attrs, inputs, proto_obj):
    """Computes softsign of x element-wise."""
    return 'softsign', attrs, inputs

def sigmoid(attrs, inputs, proto_obj):
    """Computes elementwise sigmoid of the input array"""
    return 'sigmoid', attrs, inputs

def hardsigmoid(attrs, inputs, proto_obj):
    """Computes elementwise hard sigmoid of the input array"""
    return 'hard_sigmoid', attrs, inputs

def relu(attrs, inputs, proto_obj):
    """Computes rectified linear function."""
    return 'relu', attrs, inputs

def pad(attrs, inputs, proto_obj):
    """ Add padding to input tensor"""
    opset_version = proto_obj.opset_version
    if 'mode' not in attrs.keys():
        attrs['mode'] = 'constant'
    if opset_version >= 11:
        pads = list(proto_obj._params[inputs[1].name].asnumpy())
        pads = tuple([int(i) for i in pads])
        new_attrs = translation_utils._add_extra_attributes(attrs, {'pad_width': pads})
        if len(inputs) == 3:
            const = proto_obj._params[inputs[2].name].asnumpy()[0]
            new_attrs = translation_utils._add_extra_attributes(new_attrs, {'constant_value': const})
        new_attrs['pad_width'] = translation_utils._pad_sequence_fix(new_attrs.get('pad_width'))
        return 'pad', new_attrs, inputs[0]
    else:
        new_attrs = translation_utils._fix_attribute_names(attrs, {'pads'  : 'pad_width',
                                                                   'value' : 'constant_value'
                                                                  })
        new_attrs['pad_width'] = translation_utils._pad_sequence_fix(new_attrs.get('pad_width'))
        return 'pad', new_attrs, inputs

def matrix_multiplication(attrs, inputs, proto_obj):
    """Performs general matrix multiplication"""
    return 'linalg_gemm2', attrs, inputs

def batch_norm(attrs, inputs, proto_obj):
    """Batch normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'epsilon': 'eps',
                                                               'is_test': 'fix_gamma'})
    new_attrs = translation_utils._remove_attributes(new_attrs,
                                                     ['spatial', 'consumed_inputs'])
    # Disable cuDNN BN only if epsilon from model is < than minimum cuDNN eps (1e-5)
    cudnn_min_eps = 1e-5
    cudnn_off = 0 if attrs.get('epsilon', cudnn_min_eps) >= cudnn_min_eps else 1
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'cudnn_off': cudnn_off})

    # in test mode "fix_gamma" should be unset.
    new_attrs['fix_gamma'] = not attrs.get('is_test', 1)
    return 'BatchNorm', new_attrs, inputs

def instance_norm(attrs, inputs, proto_obj):
    """Instance Normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'epsilon' : 'eps'})
    new_attrs['eps'] = attrs.get('epsilon', 1e-5)
    return 'InstanceNorm', new_attrs, inputs

def leaky_relu(attrs, inputs, proto_obj):
    """Leaky Relu function"""
    if 'alpha' in attrs:
        new_attrs = translation_utils._fix_attribute_names(attrs, {'alpha' : 'slope'})
    else:
        new_attrs = translation_utils._add_extra_attributes(attrs, {'slope': 0.01})
    return 'LeakyReLU', new_attrs, inputs

def _elu(attrs, inputs, proto_obj):
    """Elu function"""
    if 'alpha' in attrs:
        new_attrs = translation_utils._fix_attribute_names(attrs, {'alpha' : 'slope'})
    else:
        new_attrs = translation_utils._add_extra_attributes(attrs, {'slope': 1.0})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'act_type': 'elu'})
    return 'LeakyReLU', new_attrs, inputs

def _prelu(attrs, inputs, proto_obj):
    """PRelu function"""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type': 'prelu'})
    return 'LeakyReLU', new_attrs, inputs

def _selu(attrs, inputs, proto_obj):
    """Selu function"""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type': 'selu'})
    return 'LeakyReLU', new_attrs, inputs

def softmax(attrs, inputs, proto_obj):
    """Softmax function."""
    if 'axis' not in attrs:
        attrs = translation_utils._add_extra_attributes(attrs, {'axis': 1})
    return 'softmax', attrs, inputs

def log_softmax(attrs, inputs, proto_obj):
    """Computes the log softmax of the input. This is equivalent to
    computing softmax followed by log."""
    return 'log_softmax', attrs, inputs

def softplus(attrs, inputs, proto_obj):
    """Applies the sofplus activation function element-wise to the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'act_type' : 'softrelu'})
    return 'Activation', new_attrs, inputs

def conv(attrs, inputs, proto_obj):
    """Compute N-D convolution on (N+2)-D input."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'kernel_shape' : 'kernel',
                                                               'strides' : 'stride',
                                                               'pads': 'pad',
                                                               'dilations': 'dilate',
                                                               'group': 'num_group'})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'num_group' : 1})
    new_attrs = translation_utils._fix_bias('Convolution', new_attrs, len(inputs))

    new_attrs = translation_utils._fix_channels('Convolution', new_attrs, inputs, proto_obj)
    kernel = new_attrs['kernel']
    stride = new_attrs['stride'] if 'stride' in new_attrs else []
    padding = new_attrs['pad'] if 'pad' in new_attrs else []
    dilations = new_attrs['dilate'] if 'dilate' in new_attrs else []
    num_filter = new_attrs['num_filter']
    num_group = new_attrs['num_group']
    no_bias = new_attrs['no_bias'] if 'no_bias' in new_attrs else 0
    bias = None if no_bias is True else inputs[2]

    mxnet_pad = translation_utils._pad_sequence_fix(padding, kernel_dim=len(kernel))

    left_pads = mxnet_pad[0::2]
    right_pads = mxnet_pad[1::2]
    is_pad_sym = left_pads == right_pads

    if not is_pad_sym:
        # Unlike ONNX, MXNet's convolution operator does not support asymmetric padding, so we first
        # use 'Pad' operator, which supports asymmetric padding. Then use the convolution operator.
        pad_width = (0, 0, 0, 0) + mxnet_pad
        pad_op = symbol.pad(inputs[0], mode='constant', pad_width=pad_width)
        conv_op = symbol.Convolution(pad_op, inputs[1], bias,
                                     kernel=kernel, stride=stride, dilate=dilations,
                                     num_filter=num_filter, num_group=num_group, no_bias=no_bias)
    else:
        pad_width = left_pads
        conv_op = symbol.Convolution(inputs[0], inputs[1], bias,
                                     kernel=kernel, stride=stride, dilate=dilations, pad=pad_width,
                                     num_filter=num_filter, num_group=num_group, no_bias=no_bias)

    return conv_op, new_attrs, inputs

def deconv(attrs, inputs, proto_obj):
    """Computes transposed convolution of the input tensor."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'kernel_shape' : 'kernel',
                                                               'strides' : 'stride',
                                                               'pads': 'pad',
                                                               'dilations': 'dilate',
                                                               'group': 'num_group',
                                                               'output_padding': 'adj'})
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'num_group' : 1})
    new_attrs = translation_utils._fix_bias('Deconvolution', new_attrs, len(inputs))

    new_attrs = translation_utils._fix_channels('Deconvolution', new_attrs, inputs, proto_obj)
    kernel = new_attrs['kernel'] if 'kernel' in new_attrs else []
    stride = new_attrs['stride'] if 'stride' in new_attrs else []
    padding = new_attrs['pad'] if 'pad' in new_attrs else []
    dilations = new_attrs['dilate'] if 'dilate' in new_attrs else []
    num_filter = new_attrs['num_filter']
    num_group = new_attrs['num_group']
    no_bias = new_attrs['no_bias'] if 'no_bias' in new_attrs else False
    bias = None if no_bias is True else inputs[2]
    adj = new_attrs['adj'][:len(kernel)]

    # Unlike ONNX, MXNet's deconvolution operator does not support asymmetric padding, so we first
    # use 'Pad' operator, which supports asymmetric padding. Then use the deconvolution operator.
    pad_width = (0, 0, 0, 0) + translation_utils._pad_sequence_fix(padding, kernel_dim=len(kernel))
    pad_op = symbol.pad(inputs[0], mode='constant', pad_width=pad_width)

    deconv_op = symbol.Deconvolution(pad_op, inputs[1], bias,
                                     kernel=kernel, stride=stride, dilate=dilations,
                                     num_filter=num_filter, num_group=num_group, no_bias=no_bias,
                                     adj=adj)

    return deconv_op, new_attrs, inputs

def fully_connected(attrs, inputs, proto_obj):
    """Applies a linear transformation: Y=XWT+b."""
    new_attrs = translation_utils._remove_attributes(attrs, ['axis'])

    new_attrs = translation_utils._fix_bias('FullyConnected', new_attrs, len(inputs))

    new_attrs = translation_utils._fix_channels('FullyConnected', new_attrs, inputs, proto_obj)

    return 'FullyConnected', new_attrs, inputs


def global_maxpooling(attrs, inputs, proto_obj):
    """Performs max pooling on the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'global_pool': True,
                                                                'kernel': (1, 1),
                                                                'pool_type': 'max'})
    return 'Pooling', new_attrs, inputs


def global_avgpooling(attrs, inputs, proto_obj):
    """Performs avg pooling on the input."""
    new_attrs = translation_utils._add_extra_attributes(attrs, {'global_pool': True,
                                                                'kernel': (1, 1),
                                                                'pool_type': 'avg'})
    return 'Pooling', new_attrs, inputs

def global_lppooling(attrs, inputs, proto_obj):
    """Performs global lp pooling on the input."""
    p_value = attrs.get('p', 2)
    new_attrs = translation_utils._add_extra_attributes(attrs, {'global_pool': True,
                                                                'kernel': (1, 1),
                                                                'pool_type': 'lp',
                                                                'p_value': p_value})
    new_attrs = translation_utils._remove_attributes(new_attrs, ['p'])
    return 'Pooling', new_attrs, inputs

def linalg_gemm(attrs, inputs, proto_obj):
    """Performs general matrix multiplication and accumulation"""
    trans_a = 0
    trans_b = 0
    alpha = 1
    beta = 1
    if 'transA' in attrs:
        trans_a = attrs['transA']
    if 'transB' in attrs:
        trans_b = attrs['transB']
    if 'alpha' in attrs:
        alpha = attrs['alpha']
    if 'beta' in attrs:
        beta = attrs['beta']
    flatten_a = symbol.flatten(inputs[0])
    matmul_op = symbol.linalg_gemm2(A=flatten_a, B=inputs[1],
                                    transpose_a=trans_a, transpose_b=trans_b,
                                    alpha=alpha)
    gemm_op = symbol.broadcast_add(matmul_op, beta*inputs[2])
    new_attrs = translation_utils._fix_attribute_names(attrs, {'transA': 'transpose_a',
                                                               'transB': 'transpose_b'})
    new_attrs = translation_utils._remove_attributes(new_attrs, ['broadcast'])
    return gemm_op, new_attrs, inputs

def local_response_norm(attrs, inputs, proto_obj):
    """Local Response Normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'bias': 'knorm',
                                                        'size' : 'nsize'})
    return 'LRN', new_attrs, inputs

def dropout(attrs, inputs, proto_obj):
    """Dropout Regularization."""
    mode = 'training'
    opset_version = proto_obj.opset_version
    if 'is_test' in attrs and attrs['is_test'] == 0:
        mode = 'always'
    new_attrs = translation_utils._remove_attributes(attrs, ['is_test'])
    new_attrs = translation_utils._add_extra_attributes(new_attrs, {'mode': mode})
    if opset_version >= 12:
        new_attrs = translation_utils._remove_attributes(new_attrs, ['seed'])
        if len(inputs) == 2:
            ratio_float = proto_obj._params[inputs[1].name].asnumpy()[0]
            new_attrs = translation_utils._remove_attributes(new_attrs, ['p'])
            new_attrs = translation_utils._add_extra_attributes(new_attrs, {'p': ratio_float})
        elif len(inputs) == 1:
            new_attrs = translation_utils._fix_attribute_names(new_attrs, {'ratio': 'p'})
        return 'Dropout', new_attrs, inputs[0]
    else:
        new_attrs = translation_utils._fix_attribute_names(new_attrs, {'ratio': 'p'})
    return 'Dropout', new_attrs, inputs

# Changing shape and type.
def reshape(attrs, inputs, proto_obj):
    """Reshape the given array by the shape attribute."""
    if len(inputs) == 1:
        return 'reshape', attrs, inputs[0]
    reshape_shape = list(proto_obj._params[inputs[1].name].asnumpy())
    reshape_shape = [int(i) for i in reshape_shape]
    new_attrs = {'shape': reshape_shape}
    return 'reshape', new_attrs, inputs[:1]

def cast(attrs, inputs, proto_obj):
    """ Cast input to a given dtype"""
    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError:
        raise ImportError("Onnx and protobuf need to be installed. "
                          + "Instructions to install - https://github.com/onnx/onnx")
    new_attrs = translation_utils._fix_attribute_names(attrs, {'to' : 'dtype'})
    new_attrs['dtype'] = TENSOR_TYPE_TO_NP_TYPE[int(new_attrs['dtype'])]
    return 'cast', new_attrs, inputs

def split(attrs, inputs, proto_obj):
    """Splits an array along a particular axis into multiple sub-arrays."""
    split_list = attrs.get('split') if 'split' in attrs else []
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'split' : 'num_outputs'})
    if 'axis' not in attrs:
        new_attrs = translation_utils._add_extra_attributes(new_attrs, {'axis': 0})

    if not split_list:
        num_outputs = len(proto_obj.model_metadata.get('output_tensor_data'))
    else:
        if len(set(split_list)) == 1:
            num_outputs = len(split_list)
        else:
            raise NotImplementedError("Operator {} in MXNet does not support variable splits."
                                      "Tracking the issue to support variable split here: "
                                      "https://github.com/apache/incubator-mxnet/issues/11594"
                                      .format('split'))

    new_attrs['num_outputs'] = num_outputs
    return 'split', new_attrs, inputs

def _slice(attrs, inputs, proto_obj):
    """Returns a slice of the input tensor along multiple axes."""
    input_tensor_data = proto_obj.model_metadata.get('input_tensor_data')[0]
    input_shape = input_tensor_data[1]

    if proto_obj.opset_version >= 10:
        begin = proto_obj._params[inputs[1].name].asnumpy()
        end = proto_obj._params[inputs[2].name].asnumpy()
        if len(inputs) >= 4:
            axes = list(proto_obj._params[inputs[3].name].asnumpy())
            axes = tuple([int(i) for i in axes])
        else:
            axes = tuple(range(len(begin)))
        new_attrs = translation_utils._add_extra_attributes(attrs, {'axes' : axes,
                                                                    'begin' : begin,
                                                                    'end' : end
                                                                   })
    else:
        new_attrs = translation_utils._fix_attribute_names(attrs,
                                                           {'axes' : 'axis',
                                                            'ends' : 'end',
                                                            'starts' : 'begin'})
        # onnx slice provides slicing on multiple axis. Adding multiple slice_axis operator
        # for multiple axes from mxnet
        begin = new_attrs.get('begin')
        end = list(new_attrs.get('end'))
        axes = new_attrs.get('axis', tuple(range(len(begin))))

    for i, axis in enumerate(axes):
        end[i] = None if end[i] >= input_shape[axis] else end[i]
    slice_op = symbol.slice_axis(inputs[0], axis=axes[0], begin=begin[0], end=end[0])
    if len(axes) > 1:
        for i, axis in enumerate(axes):
            slice_op = symbol.slice_axis(slice_op, axis=axis, begin=begin[i], end=end[i])
    return slice_op, new_attrs, inputs

def transpose(attrs, inputs, proto_obj):
    """Transpose the input array."""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'perm' : 'axes'})
    return 'transpose', new_attrs, inputs

def squeeze(attrs, inputs, proto_obj):
    """Remove single-dimensional entries from the shape of a tensor."""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'axes' : 'axis'})
    return 'squeeze', new_attrs, inputs

def unsqueeze(attrs, inputs, cls):
    """Inserts a new axis of size 1 into the array shape"""
    # MXNet can only add one axis at a time.
    mxnet_op = inputs[0]
    for axis in attrs["axes"]:
        mxnet_op = symbol.expand_dims(mxnet_op, axis=axis)

    return mxnet_op, attrs, inputs

def flatten(attrs, inputs, proto_obj):
    """Flattens the input array into a 2-D array by collapsing the higher dimensions."""
    #Mxnet does not have axis support. By default uses axis=1
    if 'axis' in attrs and attrs['axis'] != 1:
        raise RuntimeError("Flatten operator only supports axis=1")
    new_attrs = translation_utils._remove_attributes(attrs, ['axis'])
    return 'Flatten', new_attrs, inputs

def clip(attrs, inputs, proto_obj):
    """Clips (limits) the values in an array."""
    opset_version = proto_obj.opset_version
    if opset_version >= 11:
        if len(inputs) == 1:
            new_attrs = translation_utils._add_extra_attributes(new_attrs, {'a_max' : np.inf,
                                                                            'a_min' : -np.inf})
        elif len(inputs) == 2:
            min_float = proto_obj._params[inputs[1].name].asnumpy()
            new_attrs = translation_utils._add_extra_attributes(attrs, {'a_min': min_float[0],
                                                                        'a_max': np.inf})
        elif len(inputs) == 3:
            min_float = proto_obj._params[inputs[1].name].asnumpy()
            max_float = proto_obj._params[inputs[2].name].asnumpy()
            new_attrs = translation_utils._add_extra_attributes(attrs, {'a_min': min_float[0],
                                                                        'a_max': max_float[0]})
    else:
        new_attrs = translation_utils._fix_attribute_names(attrs, {'min' : 'a_min',
                                                                   'max' : 'a_max'})
        if 'a_max' not in new_attrs:
            new_attrs = translation_utils._add_extra_attributes(new_attrs, {'a_max' : np.inf})
        if 'a_min' not in new_attrs:
            new_attrs = translation_utils._add_extra_attributes(new_attrs, {'a_min' : -np.inf})
    return 'clip', new_attrs, inputs[0]

def gather(attrs, inputs, proto_obj):
    """Gather elements from an input array along the given axis."""
    return 'take', attrs, inputs

#Powers
def reciprocal(attrs, inputs, proto_obj):
    """Returns the reciprocal of the argument, element-wise."""
    return 'reciprocal', attrs, inputs

def squareroot(attrs, inputs, proto_obj):
    """Returns element-wise square-root value of the input."""
    return 'sqrt', attrs, inputs

def power(attrs, inputs, proto_obj):
    """Returns element-wise result of base element raised to powers from exp element."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'exponent':'exp'})
    if 'broadcast' in attrs:
        new_attrs = translation_utils._remove_attributes(new_attrs, ['broadcast'])
        if attrs['broadcast'] == 1:
            return 'broadcast_power', new_attrs, inputs
        else:
            mxnet_op = symbol.pow(inputs[0], inputs[1])
            return mxnet_op, new_attrs, inputs
    mxnet_op = symbol.broadcast_power(inputs[0], inputs[1])
    return mxnet_op, new_attrs, inputs

def exponent(attrs, inputs, proto_obj):
    """Elementwise exponent of input array."""
    return 'exp', attrs, inputs

def _cos(attrs, inputs, proto_obj):
    """Elementwise cosine of input array."""
    return 'cos', attrs, inputs

def _sin(attrs, inputs, proto_obj):
    """Elementwise sine of input array."""
    return 'sin', attrs, inputs

def _tan(attrs, inputs, proto_obj):
    """Elementwise tan of input array."""
    return 'tan', attrs, inputs

def arccos(attrs, inputs, proto_obj):
    """Elementwise inverse cos of input array."""
    return 'arccos', attrs, inputs

def arcsin(attrs, inputs, proto_obj):
    """Elementwise inverse sin of input array."""
    return 'arcsin', attrs, inputs

def arctan(attrs, inputs, proto_obj):
    """Elementwise inverse tan of input array."""
    return 'arctan', attrs, inputs

def _log(attrs, inputs, proto_obj):
    """Elementwise log of input array."""
    return 'log', attrs, inputs

# Reduce Functions
def reduce_max(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by maximum value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'max', new_attrs, inputs

def reduce_mean(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by mean value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'mean', new_attrs, inputs

def reduce_min(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by minimum value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'min', new_attrs, inputs

def reduce_sum(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by sum value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'sum', new_attrs, inputs

def reduce_prod(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by product value"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'prod', new_attrs, inputs

def reduce_log_sum(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by log sum value"""
    keep_dims = True if 'keepdims' not in attrs else attrs.get('keepdims')
    sum_op = symbol.sum(inputs[0], axis=attrs.get('axes'),
                        keepdims=keep_dims)
    log_sym = symbol.log(sum_op)
    return log_sym, attrs, inputs

def reduce_log_sum_exp(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by log sum exp value"""
    keep_dims = True if 'keepdims' not in attrs else attrs.get('keepdims')
    exp_op = symbol.exp(inputs[0])
    sum_op = symbol.sum(exp_op, axis=attrs.get('axes'),
                        keepdims=keep_dims)
    log_sym = symbol.log(sum_op)
    return log_sym, attrs, inputs

def reduce_sum_square(attrs, inputs, proto_obj):
    """Reduce the array along a given axis by sum square value"""
    square_op = symbol.square(inputs[0])
    sum_op = symbol.sum(square_op, axis=attrs.get('axes'),
                        keepdims=attrs.get('keepdims'))
    return sum_op, attrs, inputs

def reduce_l1(attrs, inputs, proto_obj):
    """Reduce input tensor by l1 normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    new_attrs = translation_utils._add_extra_attributes(new_attrs,
                                                        {'ord' : 1})
    return 'norm', new_attrs, inputs

def shape(attrs, inputs, proto_obj):
    """Returns shape of input array."""
    return 'shape_array', attrs, inputs

def size(attrs, inputs, proto_obj):
    """Returns array containing size of data."""
    return "size_array", attrs, inputs

def reduce_l2(attrs, inputs, proto_obj):
    """Reduce input tensor by l2 normalization."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'axes':'axis'})
    return 'norm', new_attrs, inputs

def avg_pooling(attrs, inputs, proto_obj):
    """ Average pooling"""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'kernel_shape': 'kernel',
                                                        'strides': 'stride',
                                                        'pads': 'pad',
                                                       })
    new_attrs = translation_utils._add_extra_attributes(new_attrs,
                                                        {'pooling_convention': 'valid'
                                                        })
    new_op = translation_utils._fix_pooling('avg', inputs, new_attrs)

    return new_op, new_attrs, inputs

def lp_pooling(attrs, inputs, proto_obj):
    """LP Pooling"""
    p_value = attrs.get('p', 2)
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'kernel_shape': 'kernel',
                                                        'strides': 'stride',
                                                        'pads': 'pad'
                                                       })
    new_attrs = translation_utils._remove_attributes(new_attrs, ['p'])
    new_attrs = translation_utils._add_extra_attributes(new_attrs,
                                                        {'pooling_convention': 'valid',
                                                         'p_value': p_value
                                                        })
    new_op = translation_utils._fix_pooling('lp', inputs, new_attrs)
    return new_op, new_attrs, inputs

def max_pooling(attrs, inputs, proto_obj):
    """ Average pooling"""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'kernel_shape': 'kernel',
                                                        'strides': 'stride',
                                                        'pads': 'pad',
                                                       })

    new_attrs = translation_utils._add_extra_attributes(new_attrs,
                                                        {'pooling_convention': 'valid'
                                                        })
    new_op = translation_utils._fix_pooling('max', inputs, new_attrs)

    return new_op, new_attrs, inputs

def max_roi_pooling(attrs, inputs, proto_obj):
    """Max ROI Pooling."""
    new_attrs = translation_utils._fix_attribute_names(attrs,
                                                       {'pooled_shape': 'pooled_size',
                                                        'spatial_scale': 'spatial_scale'
                                                       })
    return 'ROIPooling', new_attrs, inputs

def depthtospace(attrs, inputs, proto_obj):
    """Rearranges data from depth into blocks of spatial data."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'blocksize':'block_size'})

    return "depth_to_space", new_attrs, inputs

def spacetodepth(attrs, inputs, proto_obj):
    """Rearranges blocks of spatial data into depth."""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'blocksize':'block_size'})

    return "space_to_depth", new_attrs, inputs

def hardmax(attrs, inputs, proto_obj):
    """Returns batched one-hot vectors."""
    input_tensor_data = proto_obj.model_metadata.get('input_tensor_data')[0]
    input_shape = input_tensor_data[1]

    axis = int(attrs.get('axis', 1))
    axis = axis if axis >= 0 else len(input_shape) + axis

    if axis == len(input_shape) - 1:
        amax = symbol.argmax(inputs[0], axis=-1)
        one_hot = symbol.one_hot(amax, depth=input_shape[-1])
        return one_hot, attrs, inputs

    # since reshape doesn't take a tensor for shape,
    # computing with np.prod. This needs to be changed to
    # to use mx.sym.prod() when mx.sym.reshape() is fixed.
    # (https://github.com/apache/incubator-mxnet/issues/10789)
    new_shape = (int(np.prod(input_shape[:axis])),
                 int(np.prod(input_shape[axis:])))
    reshape_op = symbol.reshape(inputs[0], new_shape)
    amax = symbol.argmax(reshape_op, axis=-1)
    one_hot = symbol.one_hot(amax, depth=new_shape[-1])
    hardmax_op = symbol.reshape(one_hot, input_shape)
    return hardmax_op, attrs, inputs

def lpnormalization(attrs, inputs, proto_obj):
    """ONNX does not have eps attribute, so cannot map it to L2normalization in MXNet
     without that, it works as norm operator discussion in PR:
     https://github.com/onnx/onnx/pull/1330"""
    new_attrs = translation_utils._fix_attribute_names(attrs, {'p': 'ord'})
    axis = int(attrs.get("axis", -1))
    new_attrs.update(axis=axis)
    return 'norm', new_attrs, inputs


def topk(attrs, inputs, proto_obj):
    """Returns the top k elements in an input array along the given axis."""
    new_attrs = translation_utils._add_extra_attributes(attrs,
                                                        {'ret_typ': 'both',
                                                         'dtype': 'int64'})
    opset_version = proto_obj.opset_version
    if opset_version >= 10:
        k_vals = proto_obj._params[inputs[1].name].asnumpy()
        new_attrs = translation_utils._add_extra_attributes(new_attrs, {'k': k_vals})
        return 'topk', new_attrs, inputs[0]
    else:
        return 'topk', new_attrs, inputs
