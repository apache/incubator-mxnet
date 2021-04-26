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
 * \file np_exponential_op.h
 * \brief Operator for numpy sampling from exponential distribution.
 */

#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_EXPONENTIAL_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_EXPONENTIAL_OP_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_map>
#include "../../elemwise_op_common.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "./dist_common.h"

namespace mxnet {
namespace op {

struct NumpyExponentialParam : public dmlc::Parameter<NumpyExponentialParam> {
  dmlc::optional<float> scale;
  dmlc::optional<mxnet::Tuple<index_t>> size;
  std::string ctx;
  DMLC_DECLARE_PARAMETER(NumpyExponentialParam) {
      DMLC_DECLARE_FIELD(scale)
      .set_default(dmlc::optional<float>(1.0));
      DMLC_DECLARE_FIELD(size)
      .set_default(dmlc::optional<mxnet::Tuple<index_t>>())
      .describe("Output shape. If the given shape is, "
          "e.g., (m, n, k), then m * n * k samples are drawn. "
          "Default is None, in which case a single value is returned.");
      DMLC_DECLARE_FIELD(ctx).set_default("cpu").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
  }
    void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream scale_s, size_s;
    scale_s << scale;
    size_s << size;
    (*dict)["scale"] = scale_s.str();
    (*dict)["size"] = size_s.str();
  }
};

template <typename DType>
struct scalar_exponential_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float scale, float *threshold,
                                  DType *out) {
    out[i] = -scale * log(threshold[i]);
  }
};

namespace mxnet_op {

template <typename IType>
struct check_legal_scale_kernel {
  MSHADOW_XINLINE static void Map(index_t i, IType *scalar, float* flag) {
    if (scalar[i] < 0.0) {
      flag[0] = -1.0;
    }
  }
};


template <int ndim, typename IType, typename OType>
struct exponential_kernel {
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape<ndim> &stride,
                                  const Shape<ndim> &oshape,
                                  IType *scales, float* threshold, OType *out) {
    Shape<ndim> coord = unravel(i, oshape);
    auto idx = static_cast<index_t>(dot(coord, stride));
    threshold[i] = -log(threshold[i]);
    out[i] =  scales[idx] * threshold[i];
  }
};

}  // namespace mxnet_op

template <typename xpu>
void NumpyExponentialForward(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyExponentialParam &param = nnvm::get<NumpyExponentialParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  Tensor<xpu, 1, float> workspace =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(1), s);
  Tensor<xpu, 1, float> uniform_tensor = outputs[1].FlatTo1D<xpu, float>(s);
  Tensor<xpu, 1, float> indicator_device = workspace;
  float indicator_host = 1.0;
  float *indicator_device_ptr = indicator_device.dptr_;
  Kernel<set_zero, xpu>::Launch(s, 1, indicator_device_ptr);
  prnd->SampleUniform(&uniform_tensor, 0.0, 1.0);
  if (param.scale.has_value()) {
    CHECK_GE(param.scale.value(), 0.0) << "ValueError: expect scale >= 0";
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Kernel<scalar_exponential_kernel<DType>, xpu>::Launch(
        s, outputs[0].Size(), param.scale.value(),
        uniform_tensor.dptr_, outputs[0].dptr<DType>());
    });
  } else {
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      Kernel<check_legal_scale_kernel<IType>, xpu>::Launch(
      s, inputs[0].Size(), inputs[0].dptr<IType>(), indicator_device_ptr);
    });
    _copy<xpu>(s, &indicator_host, indicator_device_ptr);
    CHECK_GE(indicator_host, 0.0) << "ValueError: expect scale >= 0";
    mxnet::TShape new_lshape, new_oshape;
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> oshape = new_oshape.get<NDim>();
          Shape<NDim> stride = calc_stride(new_lshape.get<NDim>());
          Kernel<exponential_kernel<NDim, IType, OType>, xpu>::Launch(
              s, outputs[0].Size(), stride, oshape, inputs[0].dptr<IType>(),
              uniform_tensor.dptr_, outputs[0].dptr<OType>());
        });
      });
    });
  }
}

template<typename xpu, int ndim, typename DType>
inline void ExponentialReparamBackwardImpl(const OpContext& ctx,
                                           const std::vector<TBlob>& inputs,
                                           const std::vector<OpReqType>& req,
                                           const std::vector<TBlob>& outputs,
                                           const mxnet::TShape& new_ishape,
                                           const mxnet::TShape& new_oshape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob igrad = outputs[0].reshape(new_ishape);
  // inputs: [grad_from_samples, grad_from_noise(invisible), input_tensor,
  //          samples, noise]
  const TBlob ograd = inputs[0].reshape(new_oshape);
  const TBlob itensor = inputs[2].reshape(new_ishape);
  const TBlob samples = inputs[3].reshape(new_oshape);
  const TBlob noise = inputs[4].reshape(new_oshape);
  size_t workspace_size =
      ReduceWorkspaceSize(s, igrad.shape_, req[0], ograd.shape_);
  Tensor<xpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
#if !defined(__CUDACC__)
  Reduce<red::sum, ndim, DType, op::mshadow_op::mul, op::mshadow_op::left>(
      s, igrad, req[0], workspace, ograd, noise, noise);
#else
  RTCReduce(ctx, igrad, req[0], workspace, ograd, noise, noise,
            "red::sum{}", ndim, "mul", "left");
#endif
}

template<typename xpu>
void ExponentialReparamBackward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  // skip kernel launch for zero-size tensors
  if (inputs[0].shape_.Size() == 0U) {
    return;
  }
  // [scalar] case
  if (outputs.size() == 0U) {
    return;
  }
  // [tensor] case
  if (inputs.size() == 5U) {
    mxnet::TShape new_ishape, new_oshape;
    int ndim = FillShape(outputs[0].shape_, outputs[0].shape_, inputs[0].shape_,
                         &new_ishape, &new_ishape, &new_oshape);
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        ExponentialReparamBackwardImpl<xpu, NDim, DType>(
          ctx, inputs, req, outputs, new_ishape, new_oshape);
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_EXPONENTIAL_OP_H_
