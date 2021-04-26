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
 *  Copyright (c) 2020 by Contributors
 * \file np_broadcast_reduce_op.cc
 * \brief Function definitions of NumPy-compatible
 *        broadcast and reduce operators
 */

#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDA

void NumpyArgMinMaxRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[0] == kNullOp) return;
  // parse param
  const auto& param = nnvm::get<ReduceAxisParam>(attrs.parsed);
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  TBlob out = outputs[0];
  TBlob in = inputs[0];
  // do some shape checks
  if (in.shape_.ndim() != 0) {
    if (param.axis.has_value()) {
      // cannot do argmax in an empty dimension
      int axis = param.axis.value();
      axis = CheckAxis(axis, in.shape_.ndim());
      CHECK_NE(in.shape_[axis], 0)
          << "searching input tensor of shape " << inputs[0].shape_
          << " along axis = " << axis << " of zero dim-size is not allowed";
    } else {
      // cannot do argmax on an empty array
      CHECK_NE(in.shape_.Size(), 0U) << "attempt to search an empty sequence";
    }
  }
  if (in.shape_.Size() == 0U) return;  // zero-size tensor
  // prepare shape
  dmlc::optional<mxnet::Tuple<int>> axes;
  if (param.axis.has_value()) {
    mxnet::Tuple<int> t({param.axis.value()});
    axes = dmlc::optional<mxnet::Tuple<int>>(t);
  }
  TShape small;
  small = NumpyReduceAxesShapeImpl(in.shape_, axes, true);
  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(in.shape_, small, &src_shape, &dst_shape);
  const TBlob in_data = in.reshape(src_shape);
  // request a work space
  size_t workspace_size = broadcast::ReduceWorkspaceSize(s, dst_shape, req[0], src_shape);
  Tensor<gpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
  BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, outputs[0].reshape(dst_shape), req[0], workspace, in_data,
                           reducer, NDim, "identity", true);
  });
}

#endif  // MXNET_USE_CUDA

}  // namespace op
}  // namespace mxnet
