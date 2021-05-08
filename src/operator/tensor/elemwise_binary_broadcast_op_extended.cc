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
 * \file elemwise_binary_broadcast_op_extended.cc
 * \brief CPU Implementation of extended functions for elementwise binary broadcast operator.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"
#include "./elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_power)
.describe(R"code(Returns result of first array elements raised to powers from second array, element-wise with broadcasting.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_power(x, y) = [[ 2.,  2.,  2.],
                            [ 4.,  4.,  4.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::power>)
.set_attr<nnvm::FGradient>("FGradient", NonlossGradFGradient{
  [](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
  auto head_grad_z = ograds[0];
  auto x = nnvm::NodeEntry{mxnet::op::MakeNode("broadcast_like",
      n->attrs.name + "_broadcast_like", {n->inputs[0], head_grad_z}, nullptr, &n)};
  auto y =  nnvm::NodeEntry{mxnet::op::MakeNode("broadcast_like",
      n->attrs.name + "_broadcast_like", {n->inputs[1], head_grad_z}, nullptr, &n)};

  auto one_like  = nnvm::NodeEntry{mxnet::op::MakeNode("ones_like",
      n->attrs.name + "_ones_like", {y}, nullptr, &n)};
  auto y_sub_1 = nnvm::NodeEntry{MakeNode("elemwise_sub",
      n->attrs.name + "_rhs_sub_1",  {y, one_like}, nullptr, &n)};
  auto x_power_y_sub_1 = nnvm::NodeEntry{MakeNode("broadcast_power",
      n->attrs.name + "_lhs_power_rhs_sub_1",  {x, y_sub_1}, nullptr, &n)};
  auto dzdx = nnvm::NodeEntry{MakeNode("elemwise_mul",
     n->attrs.name + "dpower/dlhs",  {y, x_power_y_sub_1}, nullptr, &n)};

  auto lnx = nnvm::NodeEntry{MakeNode("log",
      n->attrs.name + "_ln_lhs",  {x}, nullptr, &n)};
  auto x_power_y = nnvm::NodeEntry{n};
  auto dzdy = nnvm::NodeEntry{MakeNode("elemwise_mul",
     n->attrs.name + "dpower/drhs", {x_power_y, lnx}, nullptr, &n)};

  auto broadcasted_lhs_backward = nnvm::NodeEntry{MakeNode("elemwise_mul",
             n->attrs.name + "_broadcasted_lhs_backward", {head_grad_z, dzdx}, nullptr, &n)};
  auto broadcasted_rhs_backward = nnvm::NodeEntry{MakeNode("elemwise_mul",
             n->attrs.name + "_broadcasted_rhs_backward", {head_grad_z, dzdy}, nullptr, &n)};

  std::vector<nnvm::NodeEntry> ret;
  ret.emplace_back(MakeNode("_reduce_sum_brodcasted", n->attrs.name + "_lhs_backward",
                            {broadcasted_lhs_backward, n->inputs[0]}, nullptr, &n));
  ret.emplace_back(MakeNode("_reduce_sum_brodcasted", n->attrs.name + "rhs_backward",
                            {broadcasted_rhs_backward, n->inputs[1]}, nullptr, &n));
  return ret;
}});

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_maximum)
.add_alias("_npi_maximum")
.describe(R"code(Returns element-wise maximum of the input arrays with broadcasting.

This function compares two input arrays and returns a new array having the element-wise maxima.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
                              [ 1.,  1.,  1.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::maximum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_maximum"});

NNVM_REGISTER_OP(_backward_broadcast_maximum)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::ge,
                                                              mshadow_op::lt>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_minimum)
.add_alias("_npi_minimum")
.describe(R"code(Returns element-wise minimum of the input arrays with broadcasting.

This function compares two input arrays and returns a new array having the element-wise minima.

Example::

   x = [[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]]

   y = [[ 0.],
        [ 1.]]

   broadcast_maximum(x, y) = [[ 0.,  0.,  0.],
                              [ 1.,  1.,  1.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::minimum>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_broadcast_minimum"});

NNVM_REGISTER_OP(_backward_broadcast_minimum)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::le,
                                                              mshadow_op::gt>);

MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(broadcast_hypot)
.describe(R"code( Returns the hypotenuse of a right angled triangle, given its "legs"
with broadcasting.

It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.

Example::

   x = [[ 3.,  3.,  3.]]

   y = [[ 4.],
        [ 4.]]

   broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
                            [ 5.,  5.,  5.]]

   z = [[ 0.],
        [ 4.]]

   broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
                            [ 5.,  5.,  5.]]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, mshadow_op::hypot>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_broadcast_hypot" });

NNVM_REGISTER_OP(_backward_broadcast_hypot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> > {{0, 1}};
})
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastBackwardUseIn<cpu, mshadow_op::hypot_grad_left,
                    mshadow_op::hypot_grad_right>);

}  // namespace op
}  // namespace mxnet
