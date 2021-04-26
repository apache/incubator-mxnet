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
 * Copyright (c) 2019 by Contributors
 * \file group_norm-inl.h
 * \brief Implements Group Normalization (https://arxiv.org/abs/1803.08494).
 * \author Hao Jin
*/

#ifndef MXNET_OPERATOR_NN_GROUP_NORM_INL_H_
#define MXNET_OPERATOR_NN_GROUP_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/base.h>
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "./moments-inl.h"
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

namespace groupnorm {
enum GroupNormOpInputs {kData, kGamma, kBeta};  // kGamma: scaling parameters, kBeta: shift biases
enum GroupNormOpOutputs {kOut, kMean, kStd};  // req, out_data
}  // namespace groupnorm

struct GroupNormParam : public dmlc::Parameter<GroupNormParam> {
  int num_groups;
  float eps;
  bool output_mean_var;
  DMLC_DECLARE_PARAMETER(GroupNormParam) {
    DMLC_DECLARE_FIELD(num_groups).set_default(1)
      .describe("Total number of groups.");
    DMLC_DECLARE_FIELD(eps).set_default(1e-5f)
      .describe("An `epsilon` parameter to prevent division by 0.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
      .describe("Output the mean and std calculated along the given axis.");
  }
};


template<typename xpu>
void GroupNormCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const GroupNormParam& param = nnvm::get<GroupNormParam>(attrs.parsed);
  const int num_groups = param.num_groups;
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kAddTo);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[groupnorm::kData];
  const TBlob& mean = outputs[groupnorm::kMean];
  const TBlob& std = outputs[groupnorm::kStd];
  const mxnet::TShape& data_shape = data.shape_;
  CHECK_GE(data_shape.ndim(), 3U)
    << "input should have at least 3 dims and "
    << "the first 2 dims should be batch and channel respectively";
  CHECK_EQ(data_shape[1] % num_groups, 0)
    << "number of channel should be divisible by num_groups.";

  mxnet::TShape temp_data_shape(data_shape.ndim() + 1, 1);
  temp_data_shape[0] = data_shape[0];
  temp_data_shape[1] = num_groups;
  temp_data_shape[2] = data_shape[1] / num_groups;
  for (int i = 2; i < data_shape.ndim(); ++i) {
    temp_data_shape[i+1] = data_shape[i];
  }

  mxnet::TShape moments_shape(temp_data_shape.ndim(), 1);
  for (int i = 0; i < data.shape_.ndim(); ++i) {
    moments_shape[i] = (i < mean.shape_.ndim()) ? mean.shape_[i] : 1;
  }

  mxnet::TShape red_src_shape, red_dst_shape;
  BroadcastReduceShapeCompact(temp_data_shape, moments_shape, &red_src_shape, &red_dst_shape);
  int channel_size = red_src_shape.Size() / red_dst_shape.Size();

  TBlob data_ = data.reshape(red_src_shape);
  const TBlob& mean_ = mean.reshape(red_dst_shape);
  const TBlob& std_ = std.reshape(red_dst_shape);

  Tensor<xpu, 1, char> workspace;

  size_t workspace_size = broadcast::ReduceWorkspaceSize(s, red_dst_shape, req[0],
                                                         red_src_shape);

  workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);

  // Calculate mean
#if !defined(__CUDACC__)
  MSHADOW_REAL_TYPE_SWITCH(data.type_flag_, DType, {
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::identity, true>(
        s, mean_, req[0], workspace, data_);
    });
  });
#else
  BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
    broadcast::RTCReduce(ctx, mean_, req[0], workspace,
                         data_, "red::sum{}", NDim, "identity");
  });
#endif  // !defined(__CUDACC__)
  MSHADOW_REAL_TYPE_SWITCH(data.type_flag_, DType, {
    Tensor<xpu, 1, DType> mean_data_tensor = mean_.FlatTo1D<xpu, DType>(s);
    mean_data_tensor /= scalar<DType>(channel_size);
  });

  TBlob data_grp = data.reshape(temp_data_shape);
  const TBlob& mean_grp = mean.reshape(moments_shape);
  const TBlob& std_grp = std.reshape(moments_shape);
  const TBlob& output_grp = outputs[groupnorm::kOut].reshape(temp_data_shape);

  // Calculate data = data - mean
#if !defined(__CUDACC__)
  BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx,
                                                     {data_grp, mean_grp},
                                                     {kWriteTo}, {output_grp});
#else
  BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                    {data_grp, mean_grp},
                                    {kWriteTo}, {output_grp});
#endif  // !defined(__CUDACC__)

  // Calculate std
  const TBlob centered_out = outputs[groupnorm::kOut].reshape(red_src_shape);
#if !defined(__CUDACC__)
  MSHADOW_REAL_TYPE_SWITCH(output_grp.type_flag_, DType, {
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      broadcast::Reduce<mshadow_op::sum, NDim, DType, mshadow_op::square, true>(
        s, std_, req[0], workspace, centered_out);
    });
  });
#else
  BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
    broadcast::RTCReduce(ctx, std_, req[0],
                         workspace, centered_out,
                         "red::sum{}", NDim, "square");
  });
#endif
  MSHADOW_REAL_TYPE_SWITCH(output_grp.type_flag_, DType, {
    Tensor<xpu, 1, DType> std_data_tensor = std_.FlatTo1D<xpu, DType>(s);
    std_data_tensor = F<mshadow_op::square_root>(std_data_tensor / scalar<DType>(channel_size)
                      + scalar<DType>(param.eps));
  });

  // Calculate data = data / std
#if !defined(__CUDACC__)
  BinaryBroadcastCompute<xpu, mshadow_op::div>(attrs, ctx,
                                               {output_grp, std_grp},
                                               {kWriteTo}, {output_grp});
#else
  BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                    {output_grp, std_grp},
                                    {kWriteTo}, {output_grp});
#endif  // !defined(__CUDACC__)

  const TBlob& output = outputs[groupnorm::kOut];
  mxnet::TShape new_param_shape(data_shape.ndim(), 1);
  new_param_shape[1] = data_shape[1];

  const TBlob& gamma = inputs[groupnorm::kGamma].reshape(new_param_shape);
  const TBlob& beta = inputs[groupnorm::kBeta].reshape(new_param_shape);

#if !defined(__CUDACC__)
  // Calculate data = data * gamma
  BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx,
                                                   {output, gamma},
                                                   {kWriteTo}, {output});
  // Calculate data = data + beta
  BinaryBroadcastCompute<xpu, op::mshadow_op::plus>(attrs, ctx,
                                                   {output, beta},
                                                   {kWriteTo}, {output});
#else
  // Calculate data = data * gamma
  BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                    {output, gamma},
                                    {kWriteTo}, {output});
  // Calculate data = data + beta
  BinaryBroadcastRTCCompute {"add"}(attrs, ctx,
                                    {output, beta},
                                    {kWriteTo}, {output});
#endif  // !defined(__CUDACC__)
}

/*
Calculate the gradient of group normalization.
We have the following gradient for gamma, beta and x:

\bar{x} = (x - mean) / std
w = og * r / std
grad_gamma = sum(\bar{x} og, exclude_axis)
grad_beta = sum(og, exclude_axis)
grad_x = w - mean(w, axis) - \bar{x} * mean(w * \bar{x}, axis)
*/
template<typename xpu>
void GroupNormGradCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 5U);
  CHECK_EQ(outputs.size(), 3U);
  const GroupNormParam& param = nnvm::get<GroupNormParam>(attrs.parsed);
  const int num_groups = param.num_groups;

  const TBlob& data = inputs[1];
  const mxnet::TShape& dshape = data.shape_;

  mxnet::TShape temp_dshape(dshape.ndim() + 1, 1);
  temp_dshape[0] = dshape[0];
  temp_dshape[1] = num_groups;
  temp_dshape[2] = dshape[1] / num_groups;
  for (int i = 2; i < dshape.ndim(); ++i) {
    temp_dshape[i+1] = dshape[i];
  }
  const TBlob& data_ = data.reshape(temp_dshape);
  const TBlob& ograd = inputs[0].reshape(temp_dshape);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Reshape gamma to be broadcastable
  mxnet::TShape new_param_shape(dshape.ndim(), 1);
  new_param_shape[1] = dshape[1];

  const TBlob& gamma = inputs[2].reshape(new_param_shape);

  const TBlob& mean = inputs[3];
  const TBlob& std = inputs[4];

  mxnet::TShape moments_shape(temp_dshape.ndim(), 1);
  for (int i = 0; i < dshape.ndim(); ++i) {
    moments_shape[i] = (i < mean.shape_.ndim()) ? mean.shape_[i] : 1;
  }
  const TBlob& mean_ = mean.reshape(moments_shape);
  const TBlob& std_ = std.reshape(moments_shape);

  // Prepare the necessary shapes for reduction
  mxnet::TShape red_src_shape, red_dst_shape, red_exclude_src_shape, red_exclude_dst_shape;
  BroadcastReduceShapeCompact(temp_dshape, mean_.shape_, &red_src_shape, &red_dst_shape);
  BroadcastReduceShapeCompact(dshape, gamma.shape_,
                              &red_exclude_src_shape, &red_exclude_dst_shape);

  int N = red_src_shape.Size() / red_dst_shape.Size();

  // Initialize the workspace + Construct the temporary TBlobs
  Tensor<xpu, 1, char> workspace;
  size_t dtype_size = common::mshadow_type_info(outputs[0].type_flag_).size;
  size_t data_size = data.Size() * dtype_size;
  size_t red_out_size = mean.Size() * dtype_size;
  // There are two types of reduction workloads: reduce over axis and reduce exclude axis
  // We take the maximum of the workspace sizes required by these workloads.
  // Also, we explicitly set the req_type=kAddto in case we want to use it.
  size_t reduce_workspace_size =
    std::max(broadcast::ReduceWorkspaceSize(s, red_dst_shape,
                                            kAddTo, red_src_shape),
             broadcast::ReduceWorkspaceSize(s, red_exclude_dst_shape, kAddTo,
                                            red_exclude_src_shape));
  workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(
    Shape1(reduce_workspace_size + data_size * 2 + red_out_size), s);
  const TBlob normalized_data =
    TBlob(workspace.dptr_ + reduce_workspace_size,
          data_.shape_, data.dev_mask(), data.type_flag_, data.dev_id());
  const TBlob ograd_mult = TBlob(workspace.dptr_ + reduce_workspace_size + data_size,
                                 data_.shape_, ograd.dev_mask(), ograd.type_flag_, ograd.dev_id());
  const TBlob red_out = TBlob(workspace.dptr_ + reduce_workspace_size + data_size * 2,
                              mean_.shape_, mean.dev_mask(), mean.type_flag_, mean.dev_id());
  // Compute normalized_data = (data - mean) / std
#if !defined(__CUDACC__)
  BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx,
                                                    {data_, mean_},
                                                    {kWriteTo}, {normalized_data});
  BinaryBroadcastCompute<xpu, op::mshadow_op::div>(attrs, ctx,
                                                   {normalized_data, std_},
                                                   {kWriteTo}, {normalized_data});
  // Calculate grad_beta
  if (req[2] != kNullOp) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[2].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
        broadcast::Reduce<red::sum, NDim, DType, op::mshadow_op::identity, true>(
          s, outputs[2].reshape(red_exclude_dst_shape), req[2], workspace,
          ograd.reshape(red_exclude_src_shape));
      });
    });
  }
  // Calculate grad_gamma, it will be sum(ograd * normalized_data, exclude_axis)
  ElemwiseBinaryOp::Compute<xpu, op::mshadow_op::mul>(attrs, ctx, {normalized_data, ograd},
                                                      {kWriteTo}, {ograd_mult});
  if (req[1] != kNullOp) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[1].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
        broadcast::Reduce<mshadow_op::sum, NDim, DType, op::mshadow_op::identity, true>(
          s, outputs[1].reshape(red_exclude_dst_shape), req[1], workspace,
          ograd_mult.reshape(red_exclude_src_shape));
      });
    });
  }
#else
  BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                    {data_, mean_},
                                    {kWriteTo}, {normalized_data});
  BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                    {normalized_data, std_},
                                    {kWriteTo}, {normalized_data});
  // Calculate grad_beta
  if (req[2] != kNullOp) {
    BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, outputs[2].reshape(red_exclude_dst_shape),
                           req[2], workspace, ograd.reshape(red_exclude_src_shape),
                           "red::sum{}", NDim, "identity");
    });
  }
  // Calculate grad_gamma, it will be sum(ograd * normalized_data, exclude_axis)
  ElemwiseBinaryRTCCompute {"mul"}(attrs, ctx, {normalized_data, ograd},
                                   {kWriteTo}, {ograd_mult});
  if (req[1] != kNullOp) {
    BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, outputs[1].reshape(red_exclude_dst_shape),
                           req[1], workspace, ograd_mult.reshape(red_exclude_src_shape),
                           "red::sum{}", NDim, "identity");
    });
  }
#endif  // !defined(__CUDACC__)

  // Calculate grad_data:
  //   ograd_mult = ograd * gamma / std
  //   grad_data = ograd_mult - mean(ograd_mult, axis)
  //               + normalized_data * (-mean(normalized_data * ograd_mult, axis))
  if (req[0] != kNullOp) {
    const TBlob output_ = outputs[0].reshape(data_.shape_);
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx,
                                                    {inputs[0], gamma},
                                                    {kWriteTo},
                                                    {ograd_mult.reshape(data.shape_)});
    BinaryBroadcastCompute<xpu, op::mshadow_op::div>(attrs, ctx,
                                                    {ograd_mult, std_},
                                                    {kWriteTo}, {ograd_mult});
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        broadcast::Reduce<mshadow_op::sum, NDim, DType, op::mshadow_op::identity, true>(
          s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
          ograd_mult.reshape(red_src_shape));
      });
      Tensor<xpu, 1, DType> red_out_tensor = red_out.FlatTo1D<xpu, DType>(s);
      red_out_tensor /= scalar<DType>(N);
    });
    BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx,
                                                      {ograd_mult, red_out},
                                                      {req[0]}, {output_});
    ElemwiseBinaryOp::Compute<xpu, op::mshadow_op::mul>(attrs, ctx, {ograd_mult, normalized_data},
                                                        {kWriteTo}, {ograd_mult});
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        broadcast::Reduce<mshadow_op::sum, NDim, DType, op::mshadow_op::identity, true>(
          s, red_out.reshape(red_dst_shape), kWriteTo, workspace,
          ograd_mult.reshape(red_src_shape));
      });
      Tensor<xpu, 1, DType> red_out_tensor = red_out.FlatTo1D<xpu, DType>(s);
      red_out_tensor /= scalar<DType>(-N);
    });
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx,
                                                     {normalized_data, red_out},
                                                     {kAddTo}, {output_});
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {inputs[0], gamma},
                                      {kWriteTo},
                                      {ograd_mult.reshape(data.shape_)});
    BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                      {ograd_mult, std_},
                                      {kWriteTo}, {ograd_mult});
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, red_out.reshape(red_dst_shape), kWriteTo, workspace,
                           ograd_mult.reshape(red_src_shape), "red::sum{}", NDim, "identity");
    });
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> red_out_tensor = red_out.FlatTo1D<xpu, DType>(s);
      red_out_tensor /= scalar<DType>(N);
    });
    BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                      {ograd_mult, red_out},
                                      {req[0]}, {output_});
    ElemwiseBinaryRTCCompute {"mul"}(attrs, ctx, {ograd_mult, normalized_data},
                                     {kWriteTo}, {ograd_mult});
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, red_out.reshape(red_dst_shape), kWriteTo, workspace,
                           ograd_mult.reshape(red_src_shape), "red::sum{}", NDim, "identity");
    });
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> red_out_tensor = red_out.FlatTo1D<xpu, DType>(s);
      red_out_tensor /= scalar<DType>(-N);
    });
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {normalized_data, red_out},
                                      {kAddTo}, {output_});
#endif  // !defined(__CUDACC__)
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_GROUP_NORM_INL_H_
