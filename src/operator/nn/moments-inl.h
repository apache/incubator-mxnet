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
 * \file moments-inl.h
 * \brief Moments operator
 * \author Hao Jin
*/

#ifndef MXNET_OPERATOR_NN_MOMENTS_INL_H_
#define MXNET_OPERATOR_NN_MOMENTS_INL_H_

#include <vector>
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct MomentsParam : public dmlc::Parameter<MomentsParam> {
  dmlc::optional<mxnet::TShape> axes;
  bool keepdims;
  DMLC_DECLARE_PARAMETER(MomentsParam) {
    DMLC_DECLARE_FIELD(axes).set_default(dmlc::optional<mxnet::TShape>())
      .describe("Array of ints. Axes along which to compute mean and variance.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("produce moments with the same dimensionality as the input.");
  }
};

inline bool MomentsShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  const MomentsParam& param = nnvm::get<MomentsParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);

  mxnet::TShape out_shape =
    ReduceAxesShapeImpl((*in_attrs)[0], param.axes, param.keepdims, false);
  if (!param.axes.has_value() || param.axes.value().ndim() == 0) {
    LOG(FATAL) << "Empty axes is not supported, if you would like to do global moments, "
               << "please pass all axes to axes argument";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, out_shape);
  return true;
}

inline bool MomentsType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 2U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(1));
  return out_attrs->at(0) != -1 && out_attrs->at(1) != -1;
}

struct VarBroadcastKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType *out,
                                  const DType *data,
                                  const DType *mean,
                                  mshadow::Shape<6> data_shape,
                                  mshadow::Shape<6> mean_shape) {
    index_t data_idx = i;
    index_t mean_idx = i;
    index_t data_stride = 1;
    index_t mean_stride = 1;
    for (int axis = 5; axis >= 0; --axis) {
      index_t axis_idx = data_idx % data_shape[axis];
      mean_idx -= axis_idx * data_stride;
      if (mean_shape[axis] != 1) {
        mean_idx += axis_idx * mean_stride;
      }
      data_idx /= data_shape[axis];
      data_stride *= data_shape[axis];
      mean_stride *= mean_shape[axis];
    }
    DType res = (data[i] - mean[mean_idx]);
    out[i] = res * res;
  }
};

template<typename xpu>
inline void MomentsForwardImpl(const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs,
                               const dmlc::optional<mxnet::TShape>& axes,
                               const bool keepdims) {
  using namespace mshadow;
  using namespace mshadow_op;
  using namespace mxnet_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& data = inputs[0];
  const TBlob& mean = outputs[0];
  const TBlob& var = outputs[1];

  mxnet::TShape small;
  if (keepdims) {
    small = outputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(inputs[0].shape_, axes, true, false);
  }

#if !defined(__CUDACC__)
  ReduceAxesComputeImpl<xpu, mshadow_op::sum, true, true>(ctx, {data}, {req[0]}, {mean}, small);
#else
  ReduceAxesRTCComputeImpl(ctx, {data}, {req[0]}, {mean}, small, "red::sum{}", nullptr, true);
#endif
  TBlob temp;
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    Shape<6> data_shape, mean_shape;
    for (int i = 0; i < 6; ++i) {
      data_shape[i] = (i < data.shape_.ndim()) ? data.shape_[i] : 1;
      mean_shape[i] = (i < small.ndim()) ? small[i] : 1;
    }
    Tensor<xpu, 1, DType> temp_data =
      ctx.requested[0].get_space_typed<xpu, 1, DType>(Shape1(data.shape_.Size()), s);;
    Kernel<VarBroadcastKernel, xpu>::Launch(s, data.shape_.Size(), temp_data.dptr_,
      data.dptr<DType>(), mean.dptr<DType>(), data_shape, mean_shape);
    temp = TBlob(temp_data);
  });
#if !defined(__CUDACC__)
  ReduceAxesComputeImpl<xpu, mshadow_op::sum, true, true>(
    ctx, {temp.reshape(data.shape_)}, {kWriteTo}, {var}, small);
#else
  ReduceAxesRTCComputeImpl(ctx, {temp.reshape(data.shape_)},
                           {kWriteTo}, {var}, small, "red::sum{}", nullptr, true);
#endif
}

template<typename xpu>
inline void MomentsForward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;
  using namespace mxnet_op;

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);

  const MomentsParam& param = nnvm::get<MomentsParam>(attrs.parsed);

  MomentsForwardImpl<xpu>(ctx, inputs, req, outputs, param.axes, param.keepdims);
}

template<int req>
struct VarBackwardKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType *igrad,
                                  const DType *ograd,
                                  const DType *data,
                                  const DType *mean,
                                  mshadow::Shape<6> data_shape,
                                  mshadow::Shape<6> mean_shape,
                                  const float N,
                                  const float ddof = 0.0f) {
    size_t data_idx = i;
    size_t mean_idx = i;
    size_t data_stride = 1;
    size_t mean_stride = 1;
    for (int axis = 5; axis >= 0; --axis) {
      size_t axis_idx = data_idx % data_shape[axis];
      mean_idx -= axis_idx * data_stride;
      if (mean_shape[axis] != 1) {
        mean_idx += axis_idx * mean_stride;
      }
      data_idx /= data_shape[axis];
      data_stride *= data_shape[axis];
      mean_stride *= mean_shape[axis];
    }
    KERNEL_ASSIGN(igrad[i], req, ograd[mean_idx] * (data[i] - mean[mean_idx]) * 2 / (N - ddof));
  }
};

template<typename xpu>
inline void MomentsBackwardImpl(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs,
                                const dmlc::optional<mxnet::TShape>& axes) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  using namespace mxnet_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& mean_grad = inputs[0];
  const TBlob& var_grad = inputs[1];
  const TBlob& data = inputs[2];
  const TBlob& mean = inputs[3];
  const TBlob& var = inputs[4];
  const TBlob& data_grad = outputs[0];

  mxnet::TShape small = ReduceAxesShapeImpl(data.shape_, axes, true, false);
  BroadcastComputeImpl<xpu>(attrs, ctx, {mean_grad}, req, outputs, small);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> igrad = outputs[0].FlatTo1D<xpu, DType>(s);
    igrad /= scalar<DType>(outputs[0].Size()/inputs[0].Size());
  });

  Shape<6> data_shape, var_shape;
  float N = data_grad.Size() / var.Size();
  for (int i = 0; i < 6; ++i) {
    data_shape[i] = (i < data.shape_.ndim()) ? data.shape_[i] : 1;
    var_shape[i] = (i < small.ndim()) ? small[i] : 1;
  }
  MSHADOW_TYPE_SWITCH(data_grad.type_flag_, DType, {
    Kernel<VarBackwardKernel<kAddTo>, xpu>::Launch(
      s, data_grad.shape_.Size(), data_grad.dptr<DType>(), var_grad.dptr<DType>(),
      data.dptr<DType>(), mean.dptr<DType>(), data_shape, var_shape, N);
  });
}

template<typename xpu>
inline void MomentsBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;
  using namespace mxnet_op;

  CHECK_EQ(inputs.size(), 5U);
  CHECK_EQ(outputs.size(), 1U);

  const MomentsParam& param = nnvm::get<MomentsParam>(attrs.parsed);

  MomentsBackwardImpl<xpu>(attrs, ctx, inputs, req, outputs, param.axes);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_MOMENTS_INL_H_
