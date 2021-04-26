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
 *  Copyright (c) 2019 by Contributors
 * \file np_reduce_op_value.cu
 * \brief GPU Implementation of reduce functions based on value.
 */
#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_npi_sum)
.set_attr<FCompute>("FCompute<gpu>",
                    ReduceAxesRTCCompute<NumpyReduceAxesParam, 0>{"identity", "red::sum{}", false});

NNVM_REGISTER_OP(_backward_npi_sum)
.set_attr<FCompute>("FCompute<gpu>", NumpyReduceAxesBackwardUseNone<gpu>);

NNVM_REGISTER_OP(_npi_max)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesRTCCompute<NumpyReduceAxesNoDTypeParam, 0>
                                     {"identity", "red::maximum{}", false});

NNVM_REGISTER_OP(_backward_npi_max)
.set_attr<FCompute>("FCompute<gpu>", NumpyReduceAxesNoDTypeBackward<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(_npi_min)
.set_attr<FCompute>("FCompute<gpu>",
                    ReduceAxesRTCCompute<NumpyReduceAxesNoDTypeParam, 0>{"identity",
                      "red::minimum{}", false});

NNVM_REGISTER_OP(_backward_npi_min)
.set_attr<FCompute>("FCompute<gpu>", NumpyReduceAxesNoDTypeBackward<gpu, mshadow_op::eq>);

NNVM_REGISTER_OP(_npi_prod)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesRTCCompute<NumpyReduceAxesParam, 1>{"identity",
                                       "red::product{}", false});

NNVM_REGISTER_OP(_backward_npi_prod)
.set_attr<FCompute>("FCompute<gpu>", NumpyReduceAxesBackwardUseInOut<gpu, mshadow_op::rdiv>);

NNVM_REGISTER_OP(_npi_average)
.set_attr<FCompute>("FCompute<gpu>", NumpyWeightedAverageForward<gpu>);

NNVM_REGISTER_OP(_backward_np_average)
.set_attr<FCompute>("FCompute<gpu>", NumpyWeightedAverageBackward<gpu>);

NNVM_REGISTER_OP(_npi_mean)
.set_attr<FCompute>("FCompute<gpu>", ReduceAxesRTCCompute<NumpyReduceAxesParam, 0>{"identity",
                                       "red::sum{}", true});

NNVM_REGISTER_OP(_backward_np_mean)
.set_attr<FCompute>("FCompute<gpu>", NumpyReduceAxesBackwardUseNone<gpu, true>);

NNVM_REGISTER_OP(_npi_std)
.set_attr<FCompute>("FCompute<gpu>", NumpyMomentsForward<gpu, true>);

NNVM_REGISTER_OP(_npi_var)
.set_attr<FCompute>("FCompute<gpu>", NumpyMomentsForward<gpu, false>);

NNVM_REGISTER_OP(_npi_broadcast_to)
.set_attr<FCompute>("FCompute<gpu>", NumpyBroadcastToForward<gpu>);

NNVM_REGISTER_OP(_backward_np_broadcast_to)
.set_attr<FCompute>("FCompute<gpu>", NumpyBroadcastToBackward<gpu>);

}  // namespace op
}  // namespace mxnet
