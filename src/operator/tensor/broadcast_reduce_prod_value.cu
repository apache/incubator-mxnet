/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_prod_value.cu
 * \brief GPU Implementation of broadcast and reduce prod functions based on value.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(prod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesRTCCompute<ReduceAxesParam, 0>
                                     {"identity", "red::product{}", false});

NNVM_REGISTER_OP(_backward_prod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::rdiv>);

NNVM_REGISTER_OP(nanprod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesRTCCompute<ReduceAxesParam, 0>
                                     {"identity", "red::nanprod{}", false});

NNVM_REGISTER_OP(_backward_nanprod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::nanprod_grad>);

}  // namespace op
}  // namespace mxnet
