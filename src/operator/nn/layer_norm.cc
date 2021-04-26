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
 *
 * Function LayerNormCPUKernel is adapated from Marian
 * https://github.com/marian-nmt/marian-dev/blob/master/src/tensors/cpu/tensor_operators.cpp
 * under the MIT license
 * MIT License
 *
 * Copyright (c) 2016 Marcin Junczys-Dowmunt, the University of Edinburgh, Adam
 * Mickiewicz University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 * 
 * All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 *
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file layer_norm.cc
 * \brief Implements Ba et. al, Layer Normalization (https://arxiv.org/abs/1607.06450).
*/

#include "layer_norm-inl.h"
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"

#if MSHADOW_USE_MKL == 1
#include "../mkl_functions-inl.h"
#endif

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LayerNormParam);

static bool LayerNormShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_shape,
                           mxnet::ShapeVector *out_shape) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
  const mxnet::TShape &dshape = in_shape->at(layernorm::kData);
  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }

  int axis = GetRealAxis(param.axis, dshape.ndim());
  CHECK(axis >= 0 && axis < dshape.ndim())
    << "Channel axis out of range: axis=" << param.axis;

  const index_t channelCount = dshape[axis];

  SHAPE_ASSIGN_CHECK(*in_shape,
                     layernorm::kGamma,
                     mxnet::TShape(Shape1(channelCount)));
  SHAPE_ASSIGN_CHECK(*in_shape,
                     layernorm::kBeta,
                     mxnet::TShape(Shape1(channelCount)));
  out_shape->clear();
  out_shape->push_back(dshape);                // kOut
  mxnet::TShape moments_shape(dshape.begin(), dshape.end());
  moments_shape[axis] = 1;
  out_shape->push_back(moments_shape);  // kMean
  out_shape->push_back(moments_shape);  // kInvstd
  return true;
}

/* CPU optimized kernel for LayerNorm assuming axis = -1.
 * Data is the underlying storage data type.
 * Accum is the type to use for accumulation.
 *   Apparently there isn't a reduction operator for half_t and anyway it isn't
 *   efficient to use on the CPU, so use float for reduction of half_t.
 *
 * width is the number of values being summed to compute a mean.
 * instances is how many independent layer normalization problems are packed into the tensors. 
 *
 * Inputs:
 * data is instances x width
 * gamma is width
 * beta is width
 *
 * Outputs:
 * out is instances x width, can be same as data
 * mean is instances: means of each problem
 * std is instances: standard deviation of each problem
 *
 */
template <typename Data, typename Accum = typename
            /* By default accumulate in float32 for float16.  Otherwise use same type. */
            std::conditional<std::is_same<mshadow::half::half_t, Data>::value,
                             float,
                             Data>::type>
void LayerNormCPUKernel(size_t width,
                        size_t instances,
                        Data eps,
                        const Data *data,
                        const Data *gamma,
                        const Data *beta,
                        Data *out,
                        Data *mean,
                        Data *std) {
  // Parallelize over independent instances to normalize.
  // MSVC says index variable in OpenMP 'for' statement must have signed integral type.
  const mshadow::index_t signed_instances = static_cast<mshadow::index_t>(instances);
#pragma omp parallel for
  for (nnvm::dim_t j = 0; j < signed_instances; ++j) {
    const Data *from = data + j * width;

    // Sum the values to compute mean.
    Accum sum = 0.f;
#pragma omp simd reduction(+ : sum)
    for (size_t i = 0; i < width; ++i) {
      sum += from[i];
    }
    Accum mean_value = sum / width;
    mean[j] = static_cast<Data>(mean_value);

    // Sum squares from mean to compute stddev.
    Accum squares = 0.f;
#pragma omp simd reduction(+ : squares)
    for (size_t i = 0; i < width; ++i) {
      Accum off = from[i] - mean_value;
      squares += off * off;
    }
    Accum sigma = std::sqrt(squares / width + eps);
    std[j] = static_cast<Data>(sigma);

    // Write normalized values.
    Data *to = out + j * width;
    Accum inv_sigma = 1.f / sigma;
#pragma omp simd
    for (size_t i = 0; i < width; ++i) {
      to[i] = (from[i] - mean_value) * gamma[i] * inv_sigma + beta[i];
    }
  }
}

/* Wrap the above LayerNormCPUKernel in MXNet's API.  Returns true if it
 * is able to run.
 */
bool LayerNormCPU(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx, const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);

  switch (req[layernorm::kOut]) {
    case kNullOp:
      return true;
    case kWriteTo:
      break;
    case kWriteInplace:
      break;
    default:
      // Should only be kAddTo, which isn't supported by the others implementation either.
      return false;
  }
  // Axis must be the last one.
  int axis = GetRealAxis(param.axis, inputs[layernorm::kData].ndim());
  if (axis != inputs[layernorm::kData].ndim() - 1) {
    return false;
  }
  MSHADOW_REAL_TYPE_SWITCH(inputs[layernorm::kData].type_flag_, DType, {
    LayerNormCPUKernel<DType>(
        inputs[layernorm::kData].shape_[axis],
        outputs[layernorm::kMean].Size(),
        param.eps,
        inputs[layernorm::kData].dptr<DType>(),
        inputs[layernorm::kGamma].dptr<DType>(),
        inputs[layernorm::kBeta].dptr<DType>(),
        outputs[layernorm::kOut].dptr<DType>(),
        outputs[layernorm::kMean].dptr<DType>(),
        outputs[layernorm::kStd].dptr<DType>());
  });
  return true;
}

#if MSHADOW_USE_MKL == 1 && MXNET_USE_MKL_LAYERNORM == 1
bool LayerNormComputeMKL(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  if (req[0] == kNullOp) return true;
  CHECK_NE(req[0], kAddTo);
  CHECK_EQ(inputs.size(), 3U);
  int axis = GetRealAxis(param.axis, inputs[0].ndim());

  // This optimization only applys for LayerNorm on the last dimension with dtype FP32 or FP64.
  if (axis == (inputs[layernorm::kData].ndim() - 1) &&
      (inputs[0].type_flag_ == kFloat32 || inputs[0].type_flag_ == kFloat64)) {
    // Compute necessary data for the reduce operation.
    mxnet::TShape red_src_shape, red_dst_shape;
    BroadcastReduceShapeCompact(inputs[layernorm::kData].shape_, outputs[layernorm::kMean].shape_,
                                &red_src_shape, &red_dst_shape);
    const TBlob in_data = inputs[layernorm::kData].reshape(red_src_shape);
    const TBlob mean_data = outputs[layernorm::kMean].reshape(red_dst_shape);
    const TBlob std_data = outputs[layernorm::kStd].reshape(red_dst_shape);
    const int outter_size = red_dst_shape.Size();
    const int channel_size = red_src_shape.Size() / red_dst_shape.Size();

    // call
    MSHADOW_SGL_DBL_TYPE_SWITCH(in_data.type_flag_, DType, {
      mkl_func::LayerNormLastDim(outter_size, channel_size,
                                 in_data.dptr<DType>(),
                                 outputs[layernorm::kOut].dptr<DType>(),
                                 inputs[layernorm::kGamma].dptr<DType>(),
                                 inputs[layernorm::kBeta].dptr<DType>(),
                                 outputs[layernorm::kMean].dptr<DType>(),
                                 outputs[layernorm::kStd].dptr<DType>(),
                                 static_cast<DType>(param.eps));
    });
    return true;
  } else {
    // fallback
    return false;
  }
}
#endif

template<>
void LayerNormCompute<cpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
#if MSHADOW_USE_MKL == 1 && MXNET_USE_MKL_LAYERNORM == 1
  if (LayerNormComputeMKL(attrs, ctx, inputs, req, outputs)) return;
#endif
  if (LayerNormCPU(attrs, ctx, inputs, req, outputs)) return;
  LayerNormComputeGeneral<cpu>(attrs, ctx, inputs, req, outputs);
}

template <>
void LayerNormGradComputeGeneralImpl<cpu>(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const TBlob& ograd,
                                          const TBlob& data,
                                          const TBlob& gamma,
                                          const TBlob& mean,
                                          const TBlob& std,
                                          const TBlob& normalized_data,
                                          const TBlob& ograd_mult,
                                          const TBlob& red_out,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<TBlob>& outputs,
                                          const mshadow::Tensor<cpu, 1, char>& workspace,
                                          const mxnet::TShape& red_dst_shape,
                                          const mxnet::TShape& red_src_shape,
                                          const mxnet::TShape& red_exclude_dst_shape,
                                          const mxnet::TShape& red_exclude_src_shape,
                                          const int channel_size) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<cpu> *s = ctx.get_stream<cpu>();
  // Compute normalized_data = (data - mean) / std
  BinaryBroadcastCompute<cpu, mshadow_op::minus>(attrs, ctx,
                                                 {data, mean},
                                                 {kWriteTo}, {normalized_data});
  BinaryBroadcastCompute<cpu, mshadow_op::div>(attrs, ctx,
                                               {normalized_data, std},
                                               {kWriteTo}, {normalized_data});
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
  ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::mul>(attrs, ctx, {normalized_data, ograd},
                                                      {kWriteTo}, {ograd_mult});
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
    BinaryBroadcastCompute<cpu, op::mshadow_op::mul>(attrs, ctx,
                                                    {ograd, gamma},
                                                    {kWriteTo}, {ograd_mult});
    BinaryBroadcastCompute<cpu, op::mshadow_op::div>(attrs, ctx,
                                                    {ograd_mult, std},
                                                    {kWriteTo}, {ograd_mult});
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
      Tensor<cpu, 1, DType> red_out_tensor = red_out.FlatTo1D<cpu, DType>(s);
      red_out_tensor /= scalar<DType>(channel_size);
    });
    BinaryBroadcastCompute<cpu, op::mshadow_op::minus>(attrs, ctx,
                                                      {ograd_mult, red_out},
                                                      {req[0]}, {outputs[0]});
    ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::mul>(attrs, ctx, {ograd_mult, normalized_data},
                                                        {kWriteTo}, {ograd_mult});
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
      Tensor<cpu, 1, DType> red_out_tensor = red_out.FlatTo1D<cpu, DType>(s);
      red_out_tensor /=  scalar<DType>(- channel_size);
    });
    BinaryBroadcastCompute<cpu, mshadow_op::mul>(attrs, ctx,
                                                 {normalized_data, red_out},
                                                 {kAddTo}, {outputs[0]});
  }
}

template<>
void LayerNormGradCompute<cpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  return LayerNormGradComputeGeneral<cpu>(attrs, ctx, inputs, req, outputs);
}

NNVM_REGISTER_OP(LayerNorm)
.add_alias("_npx_layer_norm")
.describe(R"code(Layer normalization.

Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as
well as offset ``beta``.

Assume the input has more than one dimension and we normalize along axis 1.
We first compute the mean and variance along this axis and then 
compute the normalized output, which has the same shape as input, as following:

.. math::

  out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta

Both ``gamma`` and ``beta`` are learnable parameters.

Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
``data_std``. Note that no gradient will be passed through these two outputs.

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is -1, which sets the channel
axis to be the last item in the input shape.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<LayerNormParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mean", "std"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  return param.output_mean_var ? 3 : 1;
})
.set_attr<mxnet::FInferShape>("FInferShape", LayerNormShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 3>)
.set_attr<FCompute>("FCompute<cpu>", LayerNormCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", [](const nnvm::ObjectPtr& n,
                                           const std::vector<nnvm::NodeEntry>& ograds) {
  std::vector<nnvm::NodeEntry> heads;
  heads.push_back(ograds[0]);  // ograd
  heads.push_back(n->inputs[0]);  // data
  heads.push_back(n->inputs[1]);  // gamma
  heads.emplace_back(n, 1, 0);  // mean
  heads.emplace_back(n, 2, 0);  // std
  return MakeGradNode("_backward_LayerNorm", n, heads, n->attrs.dict);
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.add_argument("data", "NDArray-or-Symbol", "Input data to layer normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_arguments(LayerNormParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_LayerNorm)
.set_num_inputs(5)
.set_num_outputs(3)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(ParamParser<LayerNormParam>)
.set_attr<FCompute>("FCompute<cpu>", LayerNormGradCompute<cpu>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
});

}  // namespace op
}  // namespace mxnet
