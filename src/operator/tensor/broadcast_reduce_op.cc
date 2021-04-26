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

#include "broadcast_reduce_op.h"
#include <limits>
#include "../numpy/np_broadcast_reduce_op.h"
#include "elemwise_binary_scalar_op.h"
#include "mxnet/tuple.h"

namespace mxnet {
namespace op {

#if MXNET_USE_CUDA

void ReduceAxesRTCComputeImpl(const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs,
                              const mxnet::TShape& small,
                              const std::string& reducer,
                              const mshadow::Tensor<gpu, 1, char>* workspace,
                              const bool normalize,
                              const std::string& OP,
                              const int ddof) {
  using namespace mshadow;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<gpu>* s = ctx.get_stream<gpu>();
  Tensor<gpu, 1, char> w;
  if (workspace == nullptr) {
    size_t workspace_size = broadcast::ReduceWorkspaceSize(
        s, dst_shape, req[0], src_shape);
    w = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
    workspace = &w;
  }
  const TBlob in_data = inputs[0].reshape(src_shape);
  const TBlob out_data = outputs[0].reshape(dst_shape);
  BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
    broadcast::RTCReduce(ctx, out_data, req[0], *workspace, in_data, reducer, NDim, OP);
  });
  if (normalize) {
    NumpyBinaryScalarParam p{};
    p.scalar = static_cast<double>(src_shape.Size()/dst_shape.Size() - ddof);
    NodeAttrs a;
    a.parsed = p;
    BinaryScalarRTCCompute {"div"}(a, ctx, {out_data}, {kWriteInplace}, {out_data});
  }
}

namespace {
template <typename Param>
void PrepareReduce(const Param& param,
                   const std::vector<TBlob>& inputs,
                   const std::vector<TBlob>& outputs,
                   mxnet::TShape* shape, int* ddof);

template <>
void PrepareReduce<ReduceAxesParam>(const ReduceAxesParam& param,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<TBlob>& outputs,
                                    mxnet::TShape* small, int* ddof) {
  if (param.keepdims) {
    *small = outputs[0].shape_;
  } else {
    *small = ReduceAxesShapeImpl(inputs[0].shape_, param.axis, true, param.exclude);
  }

  *ddof = 0;
}

template <>
void PrepareReduce<NumpyReduceAxesNoDTypeParam>(const NumpyReduceAxesNoDTypeParam& param,
                                                const std::vector<TBlob>& inputs,
                                                const std::vector<TBlob>& outputs,
                                                mxnet::TShape* small, int* ddof) {
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  if (param.keepdims) {
    *small = outputs[0].shape_;
  } else {
    *small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  *ddof = 0;
}

template <>
void PrepareReduce<NumpyReduceAxesParam>(const NumpyReduceAxesParam& param,
                                         const std::vector<TBlob>& inputs,
                                         const std::vector<TBlob>& outputs,
                                         mxnet::TShape* small, int* ddof) {
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  if (param.keepdims) {
    *small = outputs[0].shape_;
  } else {
    *small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  *ddof = 0;
}

template <>
void PrepareReduce<NumpyReduceAxesBoolParam>(const NumpyReduceAxesBoolParam& param,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<TBlob>& outputs,
                                             mxnet::TShape* small, int* ddof) {
  if (param.keepdims) {
    *small = outputs[0].shape_;
  } else {
    *small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  *ddof = 0;
}

}  // namespace

template <typename Param, int init>
void ReduceAxesRTCCompute<Param, init>::operator()(const nnvm::NodeAttrs& attrs,
                                                   const OpContext& ctx,
                                                   const std::vector<TBlob>& inputs,
                                                   const std::vector<OpReqType>& req,
                                                   const std::vector<TBlob>& outputs) {
  if (req[0] == kNullOp) return;
  mxnet::TShape small;
  int ddof;
  const auto& param = nnvm::get<Param>(attrs.parsed);
  CHECK_NE(req[0], kWriteInplace) << "Reduce does not support write in-place";
  PrepareReduce(param, inputs, outputs, &small, &ddof);
  if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  if (inputs[0].shape_.Size() == 0) {
    if (normalize && mxnet::common::is_float(outputs[0].type_flag_)) {
      LOG(WARNING) << "WARNING: Mean of empty slice.";
        NumpyBinaryScalarParam p{};
        p.scalar = std::numeric_limits<float>::quiet_NaN();
        NodeAttrs a;
        a.parsed = p;
        BinaryScalarRTCCompute {"right"} (a, ctx, outputs, {kWriteTo}, outputs);
    } else {
      if (normalize) {
        LOG(WARNING) << "WARNING: nan is outside the range of"<<
                        "representable values of type 'int'";
      }
      if (init == 0 && req[0] == kAddTo) return;
      NumpyBinaryScalarParam p{};
      p.scalar = init;
      NodeAttrs a;
      a.parsed = p;
      BinaryScalarRTCCompute {"right"} (a, ctx, outputs, {req[0]}, outputs);
    }
    return;
  }

  ReduceAxesRTCComputeImpl(ctx, inputs, req, outputs, small, reducer, nullptr, normalize, OP, ddof);
}

template struct ReduceAxesRTCCompute<ReduceAxesParam, 0>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesParam, 0>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesParam, 1>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesNoDTypeParam, 0>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesBoolParam, 0>;
template struct ReduceAxesRTCCompute<NumpyReduceAxesBoolParam, 1>;

#endif

}  // namespace op
}  // namespace mxnet
