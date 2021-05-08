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
 * \file np_geometric_op.h
 * \brief Operator for numpy sampling from geometric distribution.
 */
#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_GEOMETRIC_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_GEOMETRIC_OP_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <string>
#include <vector>
#include "../../elemwise_op_common.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../../api/operator/op_utils.h"
#include "./dist_common.h"

namespace mxnet {
namespace op {

struct NumpyGeometricParam : public dmlc::Parameter<NumpyGeometricParam> {
  dmlc::optional<float> prob;
  std::string ctx;
  int dtype;
  dmlc::optional<mxnet::Tuple<int>> size;
  DMLC_DECLARE_PARAMETER(NumpyGeometricParam) {
    DMLC_DECLARE_FIELD(prob);
    DMLC_DECLARE_FIELD(size)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
        .describe(
            "Output shape. If the given shape is, "
            "e.g., (m, n, k), then m * n * k samples are drawn. "
            "Default is None, in which case a single value is returned.");
    DMLC_DECLARE_FIELD(ctx).set_default("cpu").describe(
        "Context of output, in format [cpu|gpu|cpu_pinned](n)."
        " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
        .add_enum("uint8", mshadow::kUint8)
        .add_enum("int32", mshadow::kInt32)
        .add_enum("float32", mshadow::kFloat32)
        .add_enum("float64", mshadow::kFloat64)
        .add_enum("float16", mshadow::kFloat16)
        .add_enum("bool", mshadow::kBool)
        .set_default(mshadow::kInt32)
        .describe(
            "DType of the output in case this can't be inferred. "
            "Defaults to int32 if not defined (dtype=None).");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream prob_s, dtype_s, size_s;
    prob_s << prob;
    dtype_s << dtype;
    size_s << size;
    (*dict)["prob"] = prob_s.str();
    (*dict)["dtype"] = dtype_s.str();
    (*dict)["size"] = size_s.str();
  }
};

inline bool NumpyGeometricOpType(const nnvm::NodeAttrs &attrs,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  const NumpyGeometricParam &param = nnvm::get<NumpyGeometricParam>(attrs.parsed);
  int otype = param.dtype;
  (*out_attrs)[0] = otype;
  return true;
}

template <int ndim, typename IType, typename OType>
struct geometric_kernel {
  MSHADOW_XINLINE static void Map(index_t i,
                                  const mshadow::Shape<ndim> &stride,
                                  const mshadow::Shape<ndim> &oshape,
                                  IType *probs, float* uniforms, OType *out) {
    mshadow::Shape<ndim> coord = mxnet_op::unravel(i, oshape);
    auto idx = static_cast<index_t>(mxnet_op::dot(coord, stride));
    IType prob = probs[idx];
    out[i] = math::floor(math::log(uniforms[i]) / math::log(1 - prob)) + 1;
  }
};

template <typename OType>
struct scalar_geometric_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float prob, float *uniforms,
                                  OType *out) {
    out[i] = floor(log(uniforms[i]) / log(1 - prob)) + 1;
  }
};

template <typename IType>
struct check_legal_prob_kernel {
  MSHADOW_XINLINE static void Map(index_t i, IType *scalar, float* flag) {
    if (scalar[i] <= 0.0 || scalar[i] > 1.0) {
      flag[0] = -1.0;
    }
  }
};

template <typename xpu>
void NumpyGeometricForward(const nnvm::NodeAttrs &attrs,
                           const OpContext &ctx,
                           const std::vector<TBlob> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyGeometricParam &param = nnvm::get<NumpyGeometricParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  index_t output_len = outputs[0].Size();
  Random<xpu, float> *prnd = ctx.requested[0].get_random<xpu, float>(s);
  Tensor<xpu, 1, float> workspace =
      ctx.requested[1].get_space_typed<xpu, 1, float>(Shape1(output_len + 1), s);
  Tensor<xpu, 1, float> uniform_tensor = workspace.Slice(0, output_len);
  Tensor<xpu, 1, float> indicator_device = workspace.Slice(output_len, output_len + 1);
  float indicator_host = 1.0;
  float *indicator_device_ptr = indicator_device.dptr_;
  Kernel<set_zero, xpu>::Launch(s, 1, indicator_device_ptr);
  prnd->SampleUniform(&uniform_tensor, 0.0, 1.0);
  if (inputs.size() == 0U) {
    // scalar prob input
    CHECK_LE(param.prob.value(), 1.0) << "ValueError: expect probs > 0 && probs <= 1";
    CHECK_GE(param.prob.value(), 0.0) << "ValueError: expect probs > 0 && probs <= 1";
    MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, OType, {
      Kernel<scalar_geometric_kernel<OType>, xpu>::Launch(
        s, outputs[0].Size(), param.prob.value(),
        uniform_tensor.dptr_, outputs[0].dptr<OType>());
    });
  } else if (inputs.size() == 1U) {
    mxnet::TShape new_lshape, new_oshape;
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> oshape = new_oshape.get<NDim>();
          Shape<NDim> stride = calc_stride(new_lshape.get<NDim>());
          // tensor prob input
          MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
            Kernel<check_legal_prob_kernel<IType>, xpu>::Launch(
              s, inputs[0].Size(), inputs[0].dptr<IType>(), indicator_device_ptr);
          });
          _copy<xpu>(s, &indicator_host, indicator_device_ptr);
          CHECK_GE(indicator_host, 0.0)
              << "ValueError: expect probs > 0 && probs <= 1";
          Kernel<geometric_kernel<NDim, IType, OType>, xpu>::Launch(
            s, outputs[0].Size(), stride, oshape, inputs[0].dptr<IType>(),
            uniform_tensor.dptr_, outputs[0].dptr<OType>());
        });
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_GEOMETRIC_OP_H_
