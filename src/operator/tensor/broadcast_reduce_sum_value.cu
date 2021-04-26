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
 * \file broadcast_reduce_sum_value.cu
 * \brief GPU Implementation of broadcast and reduce sum (and related) functions based on value.
 */
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesRTCCompute<ReduceAxesParam, 0>{"identity",
                                                                           "red::sum{}", false});

NNVM_REGISTER_OP(_backward_sum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseNone<gpu>);

NNVM_REGISTER_OP(mean)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesRTCCompute<ReduceAxesParam, 0>{"identity",
                                                                           "red::sum{}", true});

NNVM_REGISTER_OP(_backward_mean)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseNone<gpu, true>);

NNVM_REGISTER_OP(nansum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesRTCCompute<ReduceAxesParam, 0>{"identity",
                                                                           "red::nansum{}", false});

NNVM_REGISTER_OP(_backward_nansum)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesBackwardUseInOut<gpu, mshadow_op::nansum_grad>);

}  // namespace op
}  // namespace mxnet
