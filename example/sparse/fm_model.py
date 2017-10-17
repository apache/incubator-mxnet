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

def fm_model(factor_size, num_features, init):
    # data with csr storage type to enable feeding data with CSRNDArray
    x = mx.symbol.Variable("data", stype='csr')
    # weight with row_sparse storage type to enable sparse gradient updates
    v = mx.symbol.Variable("v", shape=(num_features, factor_size),
                           init=init, stype='row_sparse')
    w1_weight = mx.symbol.var('w1_weight', shape=(num_features, 1),
                              init=init, stype='row_sparse')
    w1_bias = mx.symbol.var('w1_bias', shape=(1))
    w1 = mx.symbol.broadcast_add(mx.symbol.dot(x, w1_weight), w1_bias)

    v_s = mx.symbol._internal._square_sum(data=v, axis=1, keepdims=True)
    x_s = mx.symbol.square(data=x)
    bd_sum = mx.sym.dot(x_s, v_s)

    w2 = mx.symbol.dot(x, v)
    w2_squared = 0.5 * mx.symbol.square(data=w2)

    w_all = mx.symbol.Concat(w1, w2_squared, dim=1)
    sum1 = mx.symbol.sum(data=w_all, axis=1, keepdims=True)
    sum2 = 0.5 * mx.symbol.negative(bd_sum)
    model = mx.sym.elemwise_add(sum1, sum2)

    y = mx.symbol.Variable("softmax_label")
    model = mx.symbol.LogisticRegressionOutput(data=model, label=y)
    return model
