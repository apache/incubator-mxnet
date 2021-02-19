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
 * Copyright (c) 2018 by Contributors
 * \file adaptive_avg_pooling-inl.h
 * \brief adaptive average pooling operator
 * \author Hang Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_ADAPTIVE_AVG_POOLING_INL_H_
#define MXNET_OPERATOR_CONTRIB_ADAPTIVE_AVG_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
/* contrib
#include "../ndarray/ndarray_function.h"
#include "./operator_common.h"
#include "./mxnet_op.h"
#include "./mshadow_op.h"
*/
#include "../../ndarray/ndarray_function.h"
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#if MXNET_USE_MKLDNN == 1
#include "../nn/mkldnn/mkldnn_adaptive_pooling-inl.h"
#endif

namespace mxnet {
namespace op {

struct AdaptiveAvgPoolParam : public dmlc::Parameter<AdaptiveAvgPoolParam> {
  mxnet::Tuple<int> output_size;
  DMLC_DECLARE_PARAMETER(AdaptiveAvgPoolParam) {
    DMLC_DECLARE_FIELD(output_size).set_default(mxnet::Tuple<int>())
    .describe("int (output size) or a tuple of int for output (height, width).");
  }
  bool operator==(const AdaptiveAvgPoolParam &other) const {
    return this->output_size == other.output_size;
  }
};

static inline bool IsWriting(const OpReqType ort) {
  return ort == kWriteTo || ort == kWriteInplace;
}

template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateOutput(mshadow::Stream<cpu> *s,
                                 const std::vector<TBlob> &input,
                                 const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateGradInput(mshadow::Stream<cpu> *s,
                                    const std::vector<TBlob> &input,
                                    const std::vector<TBlob> &output);

#if MXNET_USE_CUDA
template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateOutput(mshadow::Stream<gpu> *s,
                                 const std::vector<TBlob> &input,
                                 const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void AdaptiveAvgPoolUpdateGradInput(mshadow::Stream<gpu> *s,
                                    const std::vector<TBlob> &input,
                                    const std::vector<TBlob> &output);
#endif  // MXNET_USE_CUDA

template <typename xpu>
inline void AdaptiveAvgPoolOpForward(const nnvm::NodeAttrs& attrs,
                                     const OpContext &ctx,
                                     const std::vector<TBlob> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    AdaptiveAvgPoolUpdateOutput<xpu, DType, AccReal>(s, inputs, outputs);
  });
}


template <typename xpu>
inline void AdaptiveAvgPoolOpBackward(const nnvm::NodeAttrs& attrs,
                                      const OpContext &ctx,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (IsWriting(req[0])) {
    // zero grad before backwarding
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      Fill<false>(s, outputs[0], kWriteTo, 0);
    })
  }
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    AdaptiveAvgPoolUpdateGradInput<xpu, DType, AccReal>(s, inputs, outputs);
  });
}

static bool AdaptiveAvgPoolOpInferShape(const nnvm::NodeAttrs& attrs,
                                       mxnet::ShapeVector *in_shape,
                                       mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";
  CHECK_EQ(out_shape->size(), 1U) << "Output:[data]";
  const AdaptiveAvgPoolParam& param = nnvm::get<AdaptiveAvgPoolParam>(attrs.parsed);
  mxnet::TShape dshape(in_shape->at(0));
  if (mxnet::op::shape_is_none(dshape)) return false;
  if (param.output_size.ndim() == 0) {
    dshape[2] = 1;
    dshape[3] = 1;
  } else if (param.output_size.ndim() == 1) {
    dshape[2] = param.output_size[0];
    dshape[3] = param.output_size[0];
  } else if (param.output_size.ndim() == 2) {
    dshape[2] = param.output_size[0];
    dshape[3] = param.output_size[1];
  } else {
    dshape[2] = 1;
    dshape[3] = 1;
  }
  out_shape->clear();
  out_shape->push_back(dshape);
  return true;
}

using namespace mshadow;
template<typename xpu, int Dim, typename DType>
MSHADOW_XINLINE int get_stride(Tensor<xpu, Dim, DType> tensor, int idx) {
  int stride = 1;
  for (int i = Dim-2; i >= idx; --i) {
    stride *= tensor.size(i+1);
  }
  return stride;
}

}  // namespace op
}  // namespace mxnet
namespace std {
template <>
struct hash<mxnet::op::AdaptiveAvgPoolParam> {
  size_t operator()(const mxnet::op::AdaptiveAvgPoolParam &val) {
    size_t ret = 0;
    return ret;
  }
};
}  // namespace std
#endif  // MXNET_OPERATOR_CONTRIB_ADAPTIVE_AVG_POOLING_INL_H_
