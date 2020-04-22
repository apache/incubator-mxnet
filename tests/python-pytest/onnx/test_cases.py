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

IMPLEMENTED_OPERATORS_TEST = {
    'both': ['test_add',
             'test_sub',
             'test_mul',
             'test_div',
             'test_neg',
             'test_abs',
             'test_sum',
             'test_tanh',
             'test_ceil',
             'test_floor',
             'test_concat',
             'test_identity',
             'test_sigmoid',
             'test_relu',
             'test_constant_pad',
             'test_edge_pad',
             'test_reflect_pad',
             'test_softmax_example',
             'test_softmax_large_number',
             'test_softmax_axis_2',
             'test_transpose',
             'test_globalmaxpool',
             'test_globalaveragepool',
             'test_reciprocal',
             'test_sqrt',
             'test_pow',
             'test_exp_',
             'test_argmax',
             'test_argmin',
             'test_min',
             # pytorch operator tests
             'test_exp_',
             'test_operator_maxpool',
             'test_operator_params',
             'test_operator_permute2',
             'test_cos[^h]',
             'test_sin[^h]',
             'test_tan',
             'test_acos[^h]',
             'test_asin[^h]',
             'test_atan[^h]',
             'test_squeeze',
             'test_matmul_',
             'test_depthtospace',
             'test_hardsigmoid',
             'test_instancenorm',
             'test_shape',
             'test_cast((?!STRING).)*$',
             'test_clip',
             'test_size',
             'test_dropout',
             'test_unsqueeze',
             'test_log_',
             'test_flatten_default_axis',
             'test_leakyrelu',
             'test_selu_default',
             'test_elu',
             'test_max_',
             'test_softplus',
             'test_reduce_',
             'test_split_equal',
             'test_gather'
             ],
    'import': ['test_softsign',
               'test_mean',
               'test_averagepool_1d',
               'test_averagepool_2d_pads_count_include_pad',
               'test_averagepool_2d_precomputed_pads_count_include_pad',
               'test_averagepool_2d_precomputed_strides',
               'test_averagepool_2d_strides',
               'test_averagepool_3d',
               'test_hardmax'
               ],
    'export': ['test_random_uniform',
               'test_random_normal',
               'test_reduce_min',
               'test_reduce_max',
               'test_reduce_mean',
               'test_reduce_prod',
               'test_reduce_sum_d',
               'test_reduce_sum_keepdims_random',
               'test_lrn'
               ]
}

BASIC_MODEL_TESTS = {
    'both': ['test_AvgPool2D',
             'test_BatchNorm',
             'test_ConstantPad2d'
             'test_Conv2d',
             'test_MaxPool',
             'test_PReLU',
             'test_Softmax',
             'test_softmax_functional',
             'test_softmax_lastdim',
             ],
    'export': ['test_ConvTranspose2d']
}

STANDARD_MODEL = {
    'both': ['test_bvlc_alexnet',
             'test_densenet121',
             # 'test_inception_v1',
             # 'test_inception_v2',
             'test_resnet50',
             # 'test_shufflenet',
             'test_squeezenet',
             'test_vgg19'
             ],
    'import': ['test_zfnet512'],
    'export': ['test_vgg16']
}
