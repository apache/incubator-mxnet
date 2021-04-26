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
 * \file np_weibull_op.h
 * \brief Operator for numpy sampling from weibull distribution.
 */

#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_WEIBULL_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_WEIBULL_OP_H_

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

struct NumpyWeibullParam : public dmlc::Parameter<NumpyWeibullParam> {
  dmlc::optional<float> a;
  dmlc::optional<mxnet::Tuple<index_t>> size;
  std::string ctx;
  DMLC_DECLARE_PARAMETER(NumpyWeibullParam) {
      DMLC_DECLARE_FIELD(a)
      .set_default(dmlc::optional<float>());
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
    std::ostringstream a_s, size_s;
    a_s << a;
    size_s << size;
    (*dict)["a"] = a_s.str();
    (*dict)["size"] = size_s.str();
  }
};

template <typename DType>
struct scalar_weibull_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float a, float *noise,
                                  DType *out) {
    out[i] = powf(-log(noise[i]), DType(1.0/a));
  }
};

namespace mxnet_op {

template <typename IType>
struct check_legal_a_kernel {
  MSHADOW_XINLINE static void Map(index_t i, IType *a, float *flag) {
    if (a[i] <= 0.0) {
      flag[0] = -1.0;
    }
  }
};


template <int ndim, typename IType, typename OType>
struct weibull_kernel {
  MSHADOW_XINLINE static void Map(index_t i,
                                  const Shape<ndim> &stride,
                                  const Shape<ndim> &oshape,
                                  IType *aparams, float *noise, OType *out) {
    Shape<ndim> coord = unravel(i, oshape);
    auto idx = static_cast<index_t>(dot(coord, stride));
    noise[i] = -log(noise[i]);
    out[i] =  powf(noise[i], IType(1.0/aparams[idx]));
    // get grad
    noise[i] = -log(noise[i]) * out[i] * (1.0/(aparams[idx] * aparams[idx]));
  }
};

}  // namespace mxnet_op

template <typename xpu>
void NumpyWeibullForward(const nnvm::NodeAttrs &attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyWeibullParam &param = nnvm::get<NumpyWeibullParam>(attrs.parsed);
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
  if (param.a.has_value()) {
    CHECK_GT(param.a.value(), 0.0) << "ValueError: expect a > 0";
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Kernel<scalar_weibull_kernel<DType>, xpu>::Launch(
        s, outputs[0].Size(), param.a.value(),
        uniform_tensor.dptr_, outputs[0].dptr<DType>());
    });
  } else {
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      Kernel<check_legal_a_kernel<IType>, xpu>::Launch(
      s, inputs[0].Size(), inputs[0].dptr<IType>(), indicator_device_ptr);
    });
    _copy<xpu>(s, &indicator_host, indicator_device_ptr);
    CHECK_GE(indicator_host, 0.0) << "ValueError: expect a > 0";
    mxnet::TShape new_lshape, new_oshape;
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          Shape<NDim> oshape = new_oshape.get<NDim>();
          Shape<NDim> stride = calc_stride(new_lshape.get<NDim>());
          Kernel<weibull_kernel<NDim, IType, OType>, xpu>::Launch(
              s, outputs[0].Size(), stride, oshape, inputs[0].dptr<IType>(),
              uniform_tensor.dptr_, outputs[0].dptr<OType>());
        });
      });
    });
  }
}

template<typename xpu>
void WeibullReparamBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& reqs,
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
      CommonScalarReparamBackwardImpl<xpu, NDim, DType>(
        ctx, inputs, reqs, outputs, new_ishape, new_oshape);
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_WEIBULL_OP_H_
