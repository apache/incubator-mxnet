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
 * \file np_gamma_op.h
 * \brief Operator for random sampling from gamma distribution
 */

#ifndef MXNET_OPERATOR_NUMPY_RANDOM_NP_GAMMA_OP_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_NP_GAMMA_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <algorithm>
#include "./dist_common.h"
#include "../../../api/operator/op_utils.h"
#include "../../elemwise_op_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"

#define M 2

namespace mxnet {
namespace op {

struct NumpyGammaParam : public dmlc::Parameter<NumpyGammaParam> {
  dmlc::optional<float> shape;
  dmlc::optional<float> scale;
  std::string ctx;
  int dtype;
  dmlc::optional<mxnet::Tuple<index_t>> size;
  DMLC_DECLARE_PARAMETER(NumpyGammaParam) {
    DMLC_DECLARE_FIELD(shape);
    DMLC_DECLARE_FIELD(scale);
    DMLC_DECLARE_FIELD(size)
      .set_default(dmlc::optional<mxnet::Tuple<index_t>>())
      .describe("Output shape. If the given shape is, "
                "e.g., (m, n, k), then m * n * k samples are drawn. "
                "Default is None, in which case a single value is returned.");
    DMLC_DECLARE_FIELD(ctx)
      .set_default("cpu")
      .describe("Context of output, in format [xpu|xpu|xpu_pinned](n)."
                " Only used for imperative calls.");
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("None", -1)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("float16", mshadow::kFloat16)
      .set_default(-1)
      .describe("DType of the output in case this can't be inferred."
                "Defaults to float64 or float32 if not defined (dtype=None),"
                "which depends on your current default dtype.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream shape_s, scale_s, dtype_s, size_s;
    shape_s << shape;
    scale_s << scale;
    dtype_s << dtype;
    size_s << size;
    (*dict)["shape"] = shape_s.str();
    (*dict)["scale"] = scale_s.str();
    (*dict)["dtype"] = MXNetTypeWithBool2String(dtype);
    (*dict)["size"] = size_s.str();
  }
};


namespace mxnet_op {

template <typename IType, typename FType>
MSHADOW_XINLINE void GammaTransform(IType a, IType b,
                                    FType* uniforms, FType* normals) {
  FType d = a < 1 ? a + 2.0 / 3.0 : a - 1.0 / 3.0;
  FType k = sqrt(9.0 * d);
  FType c = 1.0 / k;
  for (size_t i = 0; i < M - 1; i++) {
    FType u = uniforms[i];
    FType n = normals[i];
    uniforms[i] = FType(-1);
    FType ocn = 1+c*n;
    FType v = ocn*ocn*ocn;
    if (v > 0) {
      if (u <= (1 - 0.0331 * (n * n) * (n * n))) {
        // rejection sample. The second operation should be
        // performed with low probability. This is the "squeeze"
        uniforms[i] = FType(d * v * b);
      }
      if (logf(u) < 0.5 * (n * n) + d * (1 - v + logf(v))) {
        // rejection sample. The second operation should be
        // performed with low probability. This is the "squeeze"
        uniforms[i] = FType(d * v * b);
      }
    }
  }
}


template <typename IType, typename FType>
MSHADOW_XINLINE FType GammaReduce(IType a, FType* uniforms) {
  FType u2 = uniforms[M - 1];
  for (size_t i = 0; i < M - 1; i++) {
    FType sample = uniforms[i];
    if (sample > 0) {
      return a < 1 ? sample * powf(u2, FType(1.0 / a)) : sample;
    }
  }
  return -1;
}

template<typename OType, typename FType>
struct CheckSuccessKernel {
  MSHADOW_XINLINE static void Map(int i, OType* out, FType* flag) {
    if (out[i] < 0) {
      flag[0] = -1.0;
    }
  }
};

template<typename DType>
struct StandarizeKernel {
  MSHADOW_XINLINE static void Map(int i, DType* samples, float scale) {
    samples[i] /= scale;
  }
};

template <int ndim, typename IType, typename OType, typename FType>
struct gamma_kernel {
  MSHADOW_XINLINE static void Map(index_t i, const Shape<ndim> &lstride,
                                  const Shape<ndim> &hstride,
                                  const Shape<ndim> &oshape, IType *shape,
                                  IType *scale, FType *uniforms, FType *normals,
                                  OType *out, FType *flag, bool resample) {
    // We know the sampling procedure is in its first stage, if `flag` is
    // nullptr, i.e. there is no need for reseting the indicator
    // variable(flag[0] = 1) nor checking whether a specific element is sampled
    // successfully (out[i] < 0).
    bool in_first_stage = !resample;
    Shape<ndim> coord = unravel(i, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto hidx = static_cast<index_t>(dot(coord, hstride));
    IType shape_value = shape[lidx];
    IType scale_value = scale[hidx];
    if ((shape_value <= 0) || (scale_value <= 0)) {
      flag[0] = -1;
    }
    if (in_first_stage || out[i] < 0) {
      // map phase
      GammaTransform<IType, FType>(shape_value, scale_value, uniforms + i * M,
                                   normals + i * M);
      // reduce phase
      OType sample =
          (OType)GammaReduce<IType, FType>(shape_value, uniforms + i * M);
      out[i] = sample;
    }
  }
};

template <int ndim, typename IType, typename OType, typename FType>
struct gamma_one_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, int scalar_pos,
                                  const Shape<ndim> &stride,
                                  const Shape<ndim> &oshape, IType *array,
                                  float scalar, FType *uniforms, FType *normals,
                                  OType *out, FType *flag, bool resample) {
    // We know the sampling procedure is in its first stage, if `resample` is
    // false, i.e. there is no need for reseting the indicator
    // variable(flag[0] = 1) nor checking whether a specific element is sampled
    // successfully (out[i] < 0). Instead, we would check if `array` is strictly
    // larger than zero.
    bool in_first_stage = !resample;
    Shape<ndim> coord = unravel(i, oshape);
    auto idx = static_cast<index_t>(dot(coord, stride));
    IType shape_value;
    IType scale_value;
    if (scalar_pos == 0) {
      shape_value = scalar;
      scale_value = array[idx];
      // This branch shall only be executed in the first trial.
      if (scale_value <= 0) {
        flag[0] = -1;
      }
    } else {
      shape_value = array[idx];
      scale_value = scalar;
      // This branch shall only be executed in the first trial.
      if (shape_value <= 0) {
        flag[0] = -1;
      }
    }
    if (in_first_stage || out[i] < 0) {
      // map phase
      GammaTransform<IType, FType>(shape_value, scale_value, uniforms + i * M,
                                   normals + i * M);
      // reduce phase
      OType sample =
          (OType)GammaReduce<IType, FType>(shape_value, uniforms + i * M);
      out[i] = sample;
    }
  }
};

template <typename OType, typename FType>
struct gamma_two_scalar_kernel {
  MSHADOW_XINLINE static void Map(index_t i, float shape_value,
                                  float scale_value, FType *uniforms_origin,
                                  FType *normals_origin, OType *out,
                                  FType *flag = nullptr) {
    // We know the sampling procedure is in its first stage, if `flag` is
    // nullptr, i.e. there is no need for reseting the indicator
    // variable(flag[0] = 1) nor checking whether a specific element is sampled
    // successfully (out[i] < 0).
    bool in_first_stage = (flag == nullptr);
    if (!in_first_stage) {
      flag[0] = 1;
    }
    FType *uniforms = uniforms_origin + i * M;
    FType *normals = normals_origin + i * M;
    if (in_first_stage || out[i] < 0) {
      // map phase
      GammaTransform<float, FType>(shape_value, scale_value, uniforms, normals);
      // reduce phase
      OType sample = (OType)GammaReduce<float, FType>(shape_value, uniforms);
      out[i] = sample;
    }
  }
};
}  // namespace mxnet_op

template <typename xpu, typename FType>
void NumpyGammaForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const NumpyGammaParam &param = nnvm::get<NumpyGammaParam>(attrs.parsed);
  CHECK_EQ(outputs.size(), 1);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // Generate base random number.
  Random<xpu, FType> *prnd = ctx.requested[0].get_random<xpu, FType>(s);
  index_t output_len = outputs[0].Size();
  Tensor<xpu, 1, FType> random_tensor =
      ctx.requested[1].get_space_typed<xpu, 1, FType>(Shape1(output_len * 2 * M + 1), s);
  Tensor<xpu, 1, FType> uniform_tensor = random_tensor.Slice(0, output_len * M);
  Tensor<xpu, 1, FType> normal_tensor = random_tensor.Slice(output_len * M, output_len * 2 * M);
  prnd->SampleUniform(&uniform_tensor, 0, 1);
  prnd->SampleGaussian(&normal_tensor, 0, 1);
  mxnet::TShape new_lshape, new_hshape, new_oshape;
  FType failure_indicator = 1.0;
  Tensor<xpu, 1, FType> failure_indic_workspace =
      random_tensor.Slice(output_len * 2 * M, output_len * 2 * M + 1);
  FType *failure_indicator_device = failure_indic_workspace.dptr_;
  // [scalar scalar] case
  if (inputs.size() == 0U) {
    if (param.shape.value() <= 0) {
      CHECK(false) << "ValueError: expect shape > 0";
    }
    if (param.scale.value() <= 0) {
      CHECK(false) << "ValueError: expect scale > 0";
    }
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      bool in_resample_stage = false;
      do {
        if (in_resample_stage) {
          prnd->SampleUniform(&uniform_tensor, 0, 1);
          prnd->SampleGaussian(&normal_tensor, 0, 1);
        }
        Kernel<gamma_two_scalar_kernel<OType, FType>, xpu>::Launch(
          s, outputs[0].Size(), param.shape.value(), param.scale.value(),
          uniform_tensor.dptr_, normal_tensor.dptr_, outputs[0].dptr<OType>(),
          in_resample_stage ? failure_indicator_device : nullptr);
        failure_indicator = 1.0;
        Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
          s, outputs[0].Size(), outputs[0].dptr<OType>(),
          failure_indicator_device);
        _copy<xpu>(s, &failure_indicator, failure_indicator_device);
        in_resample_stage = true;
      } while (failure_indicator < 0);
    });
  } else if (inputs.size() == 1U) {
    // [scalar tensor], [tensor scalar] case
    int ndim = FillShape(inputs[0].shape_, inputs[0].shape_, outputs[0].shape_,
                         &new_lshape, &new_lshape, &new_oshape);
    int scalar_pos;
    float scalar_value;
    if (param.shape.has_value()) {
      scalar_pos = 0;
      scalar_value = param.shape.value();
      if (scalar_value <= 0) {
        LOG(FATAL) << "ValueError: expect shape > 0";
      }
    } else {
      scalar_pos = 1;
      scalar_value = param.scale.value();
      if (scalar_value <= 0) {
        LOG(FATAL) << "ValueError: expect scale > 0";
      }
    }
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
          mshadow::Shape<NDim> stride = calc_stride(new_lshape.get<NDim>());
          bool in_resample_stage = false;
          do {
            if (in_resample_stage) {
              prnd->SampleUniform(&uniform_tensor, 0, 1);
              prnd->SampleGaussian(&normal_tensor, 0, 1);
            }
            Kernel<set_one, xpu>::Launch(s, 1, failure_indicator_device);
            Kernel<gamma_one_scalar_kernel<NDim, IType, OType, FType>, xpu>::Launch(
              s, outputs[0].Size(), scalar_pos, stride, oshape,
              inputs[0].dptr<IType>(), scalar_value,
              uniform_tensor.dptr_, normal_tensor.dptr_,
              outputs[0].dptr<OType>(),
              failure_indicator_device, in_resample_stage);
            // We reuse `failure_indicator` for parameter check.
            failure_indicator = 1.0;
            if (!in_resample_stage) {
              // Only check parameter validity in the first trial.
              _copy<xpu>(s, &failure_indicator, failure_indicator_device);
              if (failure_indicator < 0) {
                if (param.shape.has_value()) {
                  // Problematic tensor contains `scale`.
                  LOG(FATAL) << "ValueError: expect scale > 0";
                } else {
                  // Problematic tensor contains `shape`.
                  LOG(FATAL) << "ValueError: expect shape > 0";
                }
              }
            }
            failure_indicator = 1.0;
            Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
              s, outputs[0].Size(), outputs[0].dptr<OType>(),
              failure_indicator_device);
            _copy<xpu>(s, &failure_indicator, failure_indicator_device);
            in_resample_stage = true;
          } while (failure_indicator < 0);
        });
      });
    });
  } else if (inputs.size() == 2U) {
    // [tensor tensor] case
    int ndim = FillShape(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                         &new_lshape, &new_hshape, &new_oshape);
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, IType, {
      MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> lstride = calc_stride(new_lshape.get<NDim>());
        mshadow::Shape<NDim> hstride = calc_stride(new_hshape.get<NDim>());
        bool in_resample_stage = false;
        do {
          if (in_resample_stage) {
            prnd->SampleUniform(&uniform_tensor, 0, 1);
            prnd->SampleGaussian(&normal_tensor, 0, 1);
          }
          Kernel<set_one, xpu>::Launch(s, 1, failure_indicator_device);
          failure_indicator = 1.0;
          Kernel<gamma_kernel<NDim, IType, OType, FType>, xpu>::Launch(
            s, outputs[0].Size(), lstride, hstride, oshape,
            inputs[0].dptr<IType>(), inputs[1].dptr<IType>(),
            uniform_tensor.dptr_, normal_tensor.dptr_,
            outputs[0].dptr<OType>(), failure_indicator_device, in_resample_stage);
          if (!in_resample_stage) {
            _copy<xpu>(s, &failure_indicator, failure_indicator_device);
            if (failure_indicator < 0) {
              LOG(FATAL) << "ValueError: expect shape and value > 0";
            }
          }
          failure_indicator = 1.0;
          Kernel<CheckSuccessKernel<OType, FType>, xpu>::Launch(
            s, outputs[0].Size(), outputs[0].dptr<OType>(),
            failure_indicator_device);
          _copy<xpu>(s, &failure_indicator, failure_indicator_device);
          in_resample_stage = true;
          } while (failure_indicator < 0);
        });
      });
    });
  }
}

template<typename xpu, int ndim, typename DType>
inline void GammaReparamBackwardImpl(const OpContext& ctx,
                                            const std::vector<TBlob>& inputs,
                                            const std::vector<OpReqType>& req,
                                            const std::vector<TBlob>& outputs,
                                            const mxnet::TShape& new_ishape,
                                            const mxnet::TShape& new_oshape,
                                            const float scale) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob igrad = outputs[0].reshape(new_ishape);
  // inputs: [grad_from_samples, alpha_tensor, samples]
  const TBlob ograd = inputs[0].reshape(new_oshape);
  const TBlob alpha = inputs[1].reshape(new_ishape);
  TBlob samples = inputs[2].reshape(new_oshape);
  size_t workspace_size =
      ReduceWorkspaceSize(s, igrad.shape_, req[0], ograd.shape_);
  // Convert samples to standard gamma
  Kernel<op_with_req<mshadow_op::div, kWriteTo>, xpu>::Launch(
    s, samples.Size(), samples.dptr<DType>(), samples.dptr<DType>(), DType(scale));
  Tensor<xpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
#if !defined(__CUDACC__)
  Reduce<red::sum, ndim, DType, op::mshadow_op::mul, op::mshadow_op::gamma_implicit_grad>(
      s, igrad, req[0], workspace, ograd, alpha, samples);
#else
  RTCReduce(ctx, igrad, req[0], workspace, ograd, alpha, samples, "red::sum{}", ndim,
            "mul", "gamma_implicit_grad");
#endif
  Kernel<op_with_req<mshadow_op::mul, kWriteTo>, xpu>::Launch(
    s, igrad.Size(), igrad.dptr<DType>(), igrad.dptr<DType>(), DType(scale));
  // Convert samples back, otherwise the output would be corrupted.
  Kernel<op_with_req<mshadow_op::mul, kWriteTo>, xpu>::Launch(
    s, samples.Size(), samples.dptr<DType>(), samples.dptr<DType>(), DType(scale));
}

// Allow gamma sampling to be differentiable,
// using implicit reparameterization gradient:
// -(d/d\alpha cdf(x;alpha)) / pdf(x;alpha)
template<typename xpu>
void NumpyGammaGrad(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  // skip kernel launch for zero-size tensors
  if (inputs[0].shape_.Size() == 0U) {
    return;
  }
  // [scalar, scalar] case
  if (outputs.size() == 0U) {
    return;
  }
  const NumpyGammaParam &param = nnvm::get<NumpyGammaParam>(attrs.parsed);
  // [tensor tensor] case, not supported.
  if (inputs.size() == 5U) {
    LOG(FATAL) << "ValueError: two tensor case not supported";
  }

  // [tensor, scalar] case, only scalar scale is supported.
  if (inputs.size() == 3U) {
    if (param.shape.has_value()) {
      LOG(FATAL) << "ValueError: tensor scale case not supported";
    }
    mxnet::TShape new_ishape, new_oshape;
    int ndim = FillShape(outputs[0].shape_, outputs[0].shape_, inputs[0].shape_,
                         &new_ishape, &new_ishape, &new_oshape);
    auto scale = param.scale.value();
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        GammaReparamBackwardImpl<xpu, NDim, DType>(
          ctx, inputs, req, outputs, new_ishape, new_oshape, scale);
      });
    });
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_NP_GAMMA_OP_H_
