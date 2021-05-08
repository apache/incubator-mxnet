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
import pytest
from subgraph_common import check_fusion, check_neg_fusion
from subgraph_common import CustomNormalInit, DATA_SHAPE, TailNegBlock
from mxnet.contrib import quantization
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal_with_err

fc_post_ops_list=['relu', 'sigmoid', 'tanh', 'softrelu', 'gelu', 'elu', 'leaky',
                  'square', 'square_root', 'abs', 'exp', 'bounded_relu']

def test_float64_fallback():
  dtype = 'float64'
  net = nn.Dense(units=3, dtype=dtype)
  in_data = mx.nd.random.normal(shape=[3,3,3,3], dtype=dtype)
  net.initialize()
  out = net(in_data)
  out.wait_to_read()
  assert in_data.dtype == out.dtype

@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_single_fc(data_shape, use_bias, flatten):

  class SingleFC(nn.HybridBlock):
    def __init__(self, use_bias, flatten, **kwargs):
      super(SingleFC, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten)

    def hybrid_forward(self, F, x):
      return self.fc(x)

  attrs = {'fc': {}}
  net = SingleFC(use_bias, flatten)
  check_fusion(net, data_shape, attrs, check_quantization=flatten)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
@pytest.mark.parametrize('alg', fc_post_ops_list)
def test_fc_eltwise(data_shape, use_bias, flatten, alg):
  # fc + eltwise fusion case
  class FCEltwise(nn.HybridBlock):
    def __init__(self, use_bias, flatten, alg, **kwargs):
      super(FCEltwise, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten,
                          weight_initializer=CustomNormalInit(mean=0.5, sigma=0.1) if alg == 'square_root' else None)
                                            #avoid calculating square root of negative values
      self.alg = alg

    def hybrid_forward(self, F, x):
      fc_out = self.fc(x)
      if self.alg in ['relu', 'sigmoid', 'tanh', 'softrelu']:
        out = F.Activation(fc_out, act_type=self.alg)
      elif self.alg in ['gelu', 'elu', 'leaky']:
        out = F.LeakyReLU(fc_out, act_type=self.alg)
      elif self.alg == 'square':
        out = F.square(fc_out)
      elif self.alg == 'square_root':
        out = F.sqrt(fc_out)
      elif self.alg == 'abs':
        out = F.abs(fc_out)
      elif self.alg == 'exp':
        out = F.exp(fc_out)
      else:
        out = F.clip(fc_out, 0, 1.0)
      return out

  attrs = {'fc': {'with_eltwise': 'true'}}
  net = FCEltwise(use_bias, flatten, alg)
  check_fusion(net, data_shape, attrs, check_quantization=flatten)


@pytest.mark.parametrize('data_shape', DATA_SHAPE)
@pytest.mark.parametrize('use_bias', [True, False])
@pytest.mark.parametrize('flatten', [True, False])
def test_neg_fc_relu(data_shape, use_bias, flatten):
  # fc + relu can't be fusion case
  # eg.1
  # fc -----------> relu
  #  |
  #  |
  #  ---------------> [custom op]
  class NegFCReLU(nn.HybridBlock):
    def __init__(self, use_bias, flatten, **kwargs):
      super(NegFCReLU, self).__init__(**kwargs)
      self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten)
      self.act1 = nn.Activation('relu')
      self.act2 = nn.Activation('sigmoid')
      self.tail_neg = TailNegBlock()

    def hybrid_forward(self, F, x):
      fc_out = self.fc(x)
      return self.tail_neg(self.act1(fc_out), self.act2(fc_out))


  attrs, excluded_attrs = [], []
  net = NegFCReLU(use_bias, flatten)
  check_neg_fusion(net, attrs, excluded_attrs, data_shape, name='fc')


@pytest.mark.parametrize('data_min,data_max,weight_min,weight_max', [
    (-1, 1, 0, 0),
    (-1, 1, -1e-6, +1e-6),
    (0, 0, 1, 1),
    (-1e-6, +1e-6, -1, 1),
    (-1e-6, +1e-6, -1e-6, +1e-6),
    (0, 0, 0, 0)
])
def test_quantized_fc_bias_overflow(data_min, data_max, weight_min, weight_max):
  data_shape = (1, 32)
  data_nd = mx.random.uniform(data_min, data_max, shape=data_shape, ctx=mx.cpu())
  weight_nd = mx.random.uniform(weight_min, weight_max, shape=[64, 32], ctx=mx.cpu())
  bias_nd = mx.random.uniform(-1, +1, shape=[64], ctx=mx.cpu())

  class FCBiasOverflow(nn.HybridBlock):
    def __init__(self, dtype='float32', **kwargs):
        super(FCBiasOverflow, self).__init__(**kwargs)
        self.weight = mx.gluon.Parameter('weight', dtype=dtype, allow_deferred_init=True)
        self.bias = mx.gluon.Parameter('bias', dtype=dtype, allow_deferred_init=True)

    def hybrid_forward(self, F, x, weight, bias):
        conv1 = F.FullyConnected(x, num_hidden=64, weight=weight, no_bias=False, bias=bias)
        return conv1

  net = FCBiasOverflow()
  net.initialize()
  net(data_nd) # dummy run

  net.weight.data()[:] = weight_nd
  net.bias.data()[:] = bias_nd
  out = net(data_nd)

  calib_data = mx.gluon.data.DataLoader(data_nd, batch_size=1)
  qnet = quantization.quantize_net(net,
                                   ctx=mx.cpu(),
                                   exclude_layers=None,
                                   exclude_operators=None,
                                   quantized_dtype='int8',
                                   calib_mode='naive',
                                   calib_data=calib_data,
                                   num_calib_batches=1,
                                   quantize_mode='full')
  out_quantized = qnet(data_nd)
  assert_almost_equal_with_err(out.asnumpy(), out_quantized.asnumpy(),
                               rtol=1e-2, atol=1e-2, etol=0.01)
