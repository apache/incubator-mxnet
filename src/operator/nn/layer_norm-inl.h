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
 * Copyright (c) 2015 by Contributors
 * \file layer_norm-inl.h
 * \brief Implements Ba et. al, Layer Normalization (https://arxiv.org/abs/1607.06450).
*/
#ifndef MXNET_OPERATOR_NN_LAYER_NORM_INL_H_
#define MXNET_OPERATOR_NN_LAYER_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/base.h>
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

namespace layernorm {
enum LayerNormOpInputs {kData, kGamma, kBeta};  // kGamma: scaling parameters, kBeta: shift biases
enum LayerNormOpOutputs {kOut, kMean, kStd};  // indices for req, out_data
enum LayerNormOpInputsBwd {kBOgrad, kBData, kBGamma, kBMean, kBStd, kBBeta};
                           // kBGamma: scaling parameters, kBBeta: shift biases
enum LayerNormOpOutputsBwd {kBDgrad, kBGgrad, kBBgrad};  // indices for req, out_data;
                            // kBDgrad - data grad, kBGgrad - gamma grad, kBBgrad - beta grad
}  // namespace layernorm

struct LayerNormParam : public dmlc::Parameter<LayerNormParam> {
  int axis;
  float eps;
  bool output_mean_var;
  DMLC_DECLARE_PARAMETER(LayerNormParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1)
      .describe("The axis to perform layer normalization. "
                "Usually, this should be be axis of the channel dimension. "
                "Negative values means indexing from right to left.");
    DMLC_DECLARE_FIELD(eps).set_default(1e-5f)
      .describe("An `epsilon` parameter to prevent division by 0.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
      .describe("Output the mean and std calculated along the given axis.");
  }

  bool operator==(const LayerNormParam& other) const {
    return (this->axis == other.axis &&
            this->eps == other.eps &&
            this->output_mean_var == other.output_mean_var);
  }
};

static int GetRealAxis(int axis, int ndim) {
  return axis < 0 ? (axis + ndim) : axis;
}

template<typename xpu>
void LayerNormCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs);

template<typename xpu>
void LayerNormComputeGeneral(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx, const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kAddTo);
  int axis = GetRealAxis(param.axis, inputs[0].ndim());
  CHECK(axis >= 0 && axis < inputs[0].ndim()) << "Channel axis out of range: " << param.axis;
  CHECK_EQ(inputs.size(), 3U);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Reshape gamma and beta to be broadcastable
  mxnet::TShape new_param_shape(inputs[0].shape_.begin(), inputs[0].shape_.end());
  for (int i = 0; i < inputs[0].ndim(); i++) {
    if (i != axis) {
      new_param_shape[i] = 1;
    }
  }
  const TBlob gamma = inputs[1].reshape(new_param_shape);
  const TBlob beta = inputs[2].reshape(new_param_shape);
  // Compute necessary data for the reduce operation.
  mxnet::TShape red_src_shape, red_dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, outputs[layernorm::kMean].shape_,
                              &red_src_shape, &red_dst_shape);
  const TBlob in_data = inputs[0].reshape(red_src_shape);
  const TBlob mean_data = outputs[layernorm::kMean].reshape(red_dst_shape);
  const TBlob std_data = outputs[layernorm::kStd].reshape(red_dst_shape);
  int channel_size = red_src_shape.Size() / red_dst_shape.Size();
  // Initialize the workspace
  Tensor<xpu, 1, char> workspace;
  size_t workspace_size = 0;
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    workspace_size =
      broadcast::ReduceWorkspaceSize(s, mean_data.shape_, req[0],
                                     in_data.shape_, sizeof(DType));
  });
  workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);

  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
  if (!safe_acc && inputs[0].type_flag_ == mshadow::kFloat16) {
    common::LogOnce("MXNET_SAFE_ACCUMULATION=1 is recommended for float16 inputs for LayerNorm. "
                    "See https://mxnet.apache.org/api/faq/env_var "
                    "for more details.");
  }

  // Calculate mean
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      if (!safe_acc) {
        broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, false>(
          s, mean_data, req[0], workspace, in_data);
      } else {
        broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
          s, mean_data, req[0], workspace, in_data);
      }
      Tensor<xpu, 1, DType> mean_data_tensor = mean_data.FlatTo1D<xpu, DType>(s);
      mean_data_tensor /= scalar<DType>(channel_size);
    });
  });
  // Calculate data = data - mean
#if !defined(__CUDACC__)
  BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx,
                                                     {inputs[0], outputs[layernorm::kMean]},
                                                     {kWriteTo}, {outputs[0]});
#else
  BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                    {inputs[0], outputs[layernorm::kMean]},
                                    {kWriteTo}, {outputs[0]});
#endif  // !defined(__CUDACC__)
  // Calculate std
  const TBlob centered_out = outputs[0].reshape(red_src_shape);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      if (!safe_acc) {
        broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::square, false>(
          s, std_data, req[0], workspace, centered_out);
      } else {
        broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::square, true>(
          s, std_data, req[0], workspace, centered_out);
      }
      Tensor<xpu, 1, DType> std_data_tensor = std_data.FlatTo1D<xpu, DType>(s);
      std_data_tensor = F<mshadow_op::square_root>(std_data_tensor / scalar<DType>(channel_size)
                        + scalar<DType>(param.eps));
    });
  });
#if !defined(__CUDACC__)
  // Calculate data = data / std
  BinaryBroadcastCompute<xpu, mshadow_op::div>(attrs, ctx,
                                               {outputs[0], outputs[layernorm::kStd]},
                                               {kWriteTo}, {outputs[0]});
  // Calculate data = data * gamma
  BinaryBroadcastCompute<xpu, mshadow_op::mul>(attrs, ctx,
                                               {outputs[0], gamma},
                                               {kWriteTo}, {outputs[0]});
  // Calculate data = data + beta
  BinaryBroadcastCompute<xpu, mshadow_op::plus>(attrs, ctx,
                                                {outputs[0], beta},
                                                {kWriteTo}, {outputs[0]});
#else
  // Calculate data = data / std
  BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                    {outputs[0], outputs[layernorm::kStd]},
                                    {kWriteTo}, {outputs[0]});
  // Calculate data = data * gamma
  BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                    {outputs[0], gamma},
                                    {kWriteTo}, {outputs[0]});
  // Calculate data = data + beta
  BinaryBroadcastRTCCompute {"add"}(attrs, ctx,
                                    {outputs[0], beta},
                                    {kWriteTo}, {outputs[0]});
#endif  // !defined(__CUDACC__)
}

template<typename xpu>
void LayerNormGradCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs);

/*
Calculate the gradient of layer normalization.
We have the following gradient for gamma, beta and x:

\bar{x} = (x - mean) / std
w = og * r / std
grad_gamma = sum(\bar{x} og, exclude_axis)
grad_beta = sum(og, exclude_axis)
grad_x = w - mean(w, axis) - \bar{x} * mean(w * \bar{x}, axis)
*/
template<typename xpu>
void LayerNormGradComputeGeneral(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx, const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
#if MXNET_USE_MKLDNN == 1
  CHECK_EQ(inputs.size(), 6U);  // additional beta tensor
#else
  CHECK_EQ(inputs.size(), 5U);
#endif
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  int axis = param.axis;
  if (axis < 0) {
    axis += inputs[0].ndim();
  }
  CHECK(axis >= 0 && axis < inputs[0].ndim()) << "Channel axis out of range: " << param.axis;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Reshape gamma to be broadcastable
  mxnet::TShape new_param_shape(inputs[0].shape_.begin(), inputs[0].shape_.end());
  for (int i = 0; i < inputs[0].ndim(); i++) {
    if (i != axis) {
      new_param_shape[i] = 1;
    }
  }
  const TBlob ograd = inputs[0];
  const TBlob data = inputs[1];
  const TBlob gamma = inputs[2].reshape(new_param_shape);
  const TBlob mean = inputs[3];
  const TBlob std = inputs[4];
  // Prepare the necessary shapes for reduction
  mxnet::TShape red_src_shape, red_dst_shape, red_exclude_src_shape, red_exclude_dst_shape;
  BroadcastReduceShapeCompact(ograd.shape_, mean.shape_, &red_src_shape, &red_dst_shape);
  BroadcastReduceShapeCompact(ograd.shape_, gamma.shape_,
                              &red_exclude_src_shape, &red_exclude_dst_shape);
  int channel_size = red_src_shape.Size() / red_dst_shape.Size();
  // Initialize the workspace + Construct the temporary TBlobs
  Tensor<xpu, 1, char> workspace;
  size_t reduce_workspace_size = 0;
  size_t data_size = 0;
  size_t red_out_size = 0;
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    data_size = sizeof(DType) * data.Size();
    red_out_size = sizeof(DType) * mean.Size();
    // There are two types of reduction workloads: reduce over axis and reduce exclude axis
    // We take the maximum of the workspace sizes required by these workloads.
    // Also, we explicitly set the req_type=kAddto in case we want to use it.
    reduce_workspace_size =
      std::max(reduce_workspace_size,
               broadcast::ReduceWorkspaceSize(s, red_dst_shape,
                                              kAddTo, red_src_shape,
                                              sizeof(DType)));
    reduce_workspace_size =
      std::max(reduce_workspace_size,
               broadcast::ReduceWorkspaceSize(s, red_exclude_dst_shape, kAddTo,
                                              red_exclude_src_shape,
                                              sizeof(DType)));
  });
  workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(
    Shape1(reduce_workspace_size + data_size * 2 + red_out_size), s);
  const TBlob normalized_data = TBlob(workspace.dptr_ + reduce_workspace_size,
                                      data.shape_, data.dev_mask(), data.type_flag_, data.dev_id());
  const TBlob ograd_mult = TBlob(workspace.dptr_ + reduce_workspace_size + data_size,
                                 ograd.shape_, ograd.dev_mask(), ograd.type_flag_, ograd.dev_id());
  const TBlob red_out = TBlob(workspace.dptr_ + reduce_workspace_size + data_size * 2,
                              mean.shape_, mean.dev_mask(), mean.type_flag_, mean.dev_id());
  // Compute normalized_data = (data - mean) / std
#if !defined(__CUDACC__)
  BinaryBroadcastCompute<xpu, mshadow_op::minus>(attrs, ctx,
                                                 {data, mean},
                                                 {kWriteTo}, {normalized_data});
  BinaryBroadcastCompute<xpu, mshadow_op::div>(attrs, ctx,
                                               {normalized_data, std},
                                               {kWriteTo}, {normalized_data});
#else
  BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                    {data, mean},
                                    {kWriteTo}, {normalized_data});
  BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                    {normalized_data, std},
                                    {kWriteTo}, {normalized_data});
#endif  // !defined(__CUDACC__)
  // Calculate grad_beta
  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
  if (req[2] != kNullOp) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[2].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
        if (!safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, false>(
            s, outputs[2].reshape(red_exclude_dst_shape), req[2], workspace,
            ograd.reshape(red_exclude_src_shape));
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
            s, outputs[2].reshape(red_exclude_dst_shape), req[2], workspace,
            ograd.reshape(red_exclude_src_shape));
        }
      });
    });
  }
  // Calculate grad_gamma, it will be sum(ograd * normalized_data, exclude_axis)
#if !defined(__CUDACC__)
  ElemwiseBinaryOp::Compute<xpu, op::mshadow_op::mul>(attrs, ctx, {normalized_data, ograd},
                                                      {kWriteTo}, {ograd_mult});
#else
  ElemwiseBinaryRTCCompute {"mul"}(attrs, ctx, {normalized_data, ograd},
                                   {kWriteTo}, {ograd_mult});
#endif  // !defined(__CUDACC__)
  if (req[1] != kNullOp) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[1].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
        if (!safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, false>(
            s, outputs[1].reshape(red_exclude_dst_shape), req[1], workspace,
            ograd_mult.reshape(red_exclude_src_shape));
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
            s, outputs[1].reshape(red_exclude_dst_shape), req[1], workspace,
            ograd_mult.reshape(red_exclude_src_shape));
        }
      });
    });
  }
  // Calculate grad_data:
  //   ograd_mult = ograd * gamma / std
  //   grad_data = ograd_mult - mean(ograd_mult, axis)
  //               + normalized_data * (-mean(normalized_data * ograd_mult, axis))
  if (req[0] != kNullOp) {
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx,
                                                    {ograd, gamma},
                                                    {kWriteTo}, {ograd_mult});
    BinaryBroadcastCompute<xpu, op::mshadow_op::div>(attrs, ctx,
                                                    {ograd_mult, std},
                                                    {kWriteTo}, {ograd_mult});
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {ograd, gamma},
                                      {kWriteTo}, {ograd_mult});
    BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                      {ograd_mult, std},
                                      {kWriteTo}, {ograd_mult});
#endif  // !defined(__CUDACC__)
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        if (!safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, false>(
            s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
            ograd_mult.reshape(red_src_shape));
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
            s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
            ograd_mult.reshape(red_src_shape));
        }
      });
      Tensor<xpu, 1, DType> red_out_tensor = red_out.FlatTo1D<xpu, DType>(s);
      red_out_tensor /= scalar<DType>(channel_size);
    });
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx,
                                                      {ograd_mult, red_out},
                                                      {req[0]}, {outputs[0]});
    ElemwiseBinaryOp::Compute<xpu, op::mshadow_op::mul>(attrs, ctx, {ograd_mult, normalized_data},
                                                        {kWriteTo}, {ograd_mult});
#else
    BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                      {ograd_mult, red_out},
                                      {req[0]}, {outputs[0]});
    ElemwiseBinaryRTCCompute {"mul"}(attrs, ctx, {ograd_mult, normalized_data},
                                     {kWriteTo}, {ograd_mult});
#endif  // !defined(__CUDACC__)
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        if (!safe_acc) {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, false>(
            s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
            ograd_mult.reshape(red_src_shape));
        } else {
          broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
            s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
            ograd_mult.reshape(red_src_shape));
        }
      });
      Tensor<xpu, 1, DType> red_out_tensor = red_out.FlatTo1D<xpu, DType>(s);
      red_out_tensor /=  scalar<DType>(- channel_size);
    });
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, mshadow_op::mul>(attrs, ctx,
                                                 {normalized_data, red_out},
                                                 {kAddTo}, {outputs[0]});
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {normalized_data, red_out},
                                      {kAddTo}, {outputs[0]});
#endif  // !defined(__CUDACC__)
  }
}

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::LayerNormParam> {
  size_t operator()(const mxnet::op::LayerNormParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.axis);
    ret = dmlc::HashCombine(ret, val.eps);
    ret = dmlc::HashCombine(ret, val.output_mean_var);
    return ret;
  }
};
}  // namespace std
#endif  // MXNET_OPERATOR_NN_LAYER_NORM_INL_H_
