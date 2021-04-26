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
 *  Copyright (c) 2017 by Contributors
 * \file requantize-inl.h
 * \brief implementation of quantize operation
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_REQUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_REQUANTIZE_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "./quantization_utils.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct RequantizeParam : public dmlc::Parameter<RequantizeParam> {
  int out_type;
  dmlc::optional<float> min_calib_range;  // min float value calculated from calibration dataset
  dmlc::optional<float> max_calib_range;  // max float value calculated from calibration dataset
  DMLC_DECLARE_PARAMETER(RequantizeParam) {
    DMLC_DECLARE_FIELD(out_type)
      .add_enum("auto", QuantizeOutType::kAuto)
      .add_enum("int8", QuantizeOutType::kInt8)
      .add_enum("uint8", QuantizeOutType::kUint8)
      .set_default(QuantizeOutType::kInt8)
      .describe("Output data type. `auto` can be specified to automatically determine output type "
                "according to min_calib_range.");
    DMLC_DECLARE_FIELD(min_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The minimum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to requantize the "
              "int32 data into int8.");
    DMLC_DECLARE_FIELD(max_calib_range)
    .set_default(dmlc::optional<float>())
    .describe("The maximum scalar value in the form of float32 obtained "
              "through calibration. If present, it will be used to requantize the "
              "int32 data into int8.");
  }
};

inline bool RequantizeType(const nnvm::NodeAttrs& attrs,
                           std::vector<int> *in_attrs,
                           std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 3U);
  const RequantizeParam &param = nnvm::get<RequantizeParam>(attrs.parsed);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kInt32);
  TYPE_ASSIGN_CHECK(*in_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*in_attrs, 2, mshadow::kFloat32);
  auto out_type = GetQuantizeOutputType(param);
  if (out_type == mshadow::kUint8) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kUint8);
  } else if (out_type == mshadow::kInt8) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  } else {
    LOG(FATAL) << "requantize op only supports int8 and uint8 as output type";
  }
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

struct RequantizeKernel {
  template<typename T1, typename T2>
  MSHADOW_XINLINE static void Map(int i, T2 *output, float *omin_range, float *omax_range,
      const T1 *input, const float *imin_range, const float *imax_range, const float real_range) {
    const float input_float = QuantizedToFloat<T1>(input[i], *imin_range, *imax_range);
    *omin_range = -real_range;
    *omax_range =  real_range;
    output[i] = FloatToQuantized<T2>(input_float, -real_range, real_range);
  }

  template<typename T1, typename T2>
  MSHADOW_XINLINE static void Map(int i, T2 *output, float *omin_range, float *omax_range,
      const T1 *input, const float *imin_range, const float *imax_range,
      const float *actual_min, const float *actual_max) {
    Map(i, output, omin_range, omax_range, input, imin_range, imax_range,
        MaxAbs(*actual_min, *actual_max));
  }
};

template<typename xpu>
void RequantizeForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  typedef int32_t SrcDType;
  typedef int8_t  DstDType;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const RequantizeParam& param =
    nnvm::get<RequantizeParam>(attrs.parsed);
  auto out_type = GetQuantizeOutputType(param);
  if (out_type == mshadow::kUint8 && std::is_same<xpu, gpu>::value) {
    LOG(FATAL) << "currently, uint8 quantization is only supported by CPU, "
                  "please switch to the context of CPU or int8 data type for GPU.";
  }

  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    Kernel<RequantizeKernel, xpu>::Launch(s, inputs[0].Size(),
        outputs[0].dptr<DstDType>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
        inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
        MaxAbs(param.min_calib_range.value(), param.max_calib_range.value()));
  } else {  // model is not calibrated
    mxnet::TShape src_shape, dst_shape;
    const size_t actual_float_size = sizeof(float);
    const size_t actual_quantized_size = sizeof(SrcDType);
    const size_t temp_reduce_size = ConfigReduce<xpu, SrcDType>(
        s, inputs[0].shape_, mxnet::TShape(1, 1), &src_shape, &dst_shape);
    Tensor<xpu, 1, char> temp_space =
      ctx.requested[0].get_space_typed<xpu, 1, char>(
          Shape1(2*actual_float_size+2*actual_quantized_size+temp_reduce_size), s);
    Tensor<xpu, 1, float> actual_min_float(
        reinterpret_cast<float*>(temp_space.dptr_), Shape1(1), s);
    Tensor<xpu, 1, float> actual_max_float(
        reinterpret_cast<float*>(temp_space.dptr_) + 1, Shape1(1), s);

    const int dev_id = ctx.run_ctx.ctx.dev_id;
    TBlob actual_min_quantized(reinterpret_cast<SrcDType*>(
          temp_space.dptr_ + 8), Shape1(1), xpu::kDevMask, dev_id);
    TBlob actual_max_quantized(reinterpret_cast<SrcDType*>(
          temp_space.dptr_ + 8) + 1, Shape1(1), xpu::kDevMask, dev_id);
    Tensor<xpu, 1, char> workspace(
        temp_space.dptr_+2*actual_float_size+2*actual_quantized_size, Shape1(temp_reduce_size), s);
#if !defined(__CUDACC__)
    broadcast::Reduce<red::minimum, 2, SrcDType, mshadow::op::identity>(
      s, actual_min_quantized.reshape(dst_shape),
      kWriteTo, workspace, inputs[0].reshape(src_shape));
    Kernel<QuantizedToFloatStruct, xpu>::Launch(s, 1,
        actual_min_float.dptr_, actual_min_quantized.dptr<SrcDType>(),
        inputs[1].dptr<float>(), inputs[2].dptr<float>());

    broadcast::Reduce<red::maximum, 2, SrcDType, mshadow::op::identity>(
      s, actual_max_quantized.reshape(dst_shape),
      kWriteTo, workspace, inputs[0].reshape(src_shape));
#else
    broadcast::RTCReduce(ctx, actual_min_quantized.reshape(dst_shape),
                         kWriteTo, workspace, inputs[0].reshape(src_shape),
                         "red::minimum{}", 2, "identity");
    Kernel<QuantizedToFloatStruct, xpu>::Launch(s, 1,
        actual_min_float.dptr_, actual_min_quantized.dptr<SrcDType>(),
        inputs[1].dptr<float>(), inputs[2].dptr<float>());

    broadcast::RTCReduce(ctx, actual_max_quantized.reshape(dst_shape),
                         kWriteTo, workspace, inputs[0].reshape(src_shape),
                         "red::maximum{}", 2, "identity");
#endif
    Kernel<QuantizedToFloatStruct, xpu>::Launch(s, 1,
        actual_max_float.dptr_, actual_max_quantized.dptr<SrcDType>(),
        inputs[1].dptr<float>(), inputs[2].dptr<float>());

    Kernel<RequantizeKernel, xpu>::Launch(s, inputs[0].Size(),
        outputs[0].dptr<DstDType>(), outputs[1].dptr<float>(), outputs[2].dptr<float>(),
        inputs[0].dptr<SrcDType>(), inputs[1].dptr<float>(), inputs[2].dptr<float>(),
        actual_min_float.dptr_, actual_max_float.dptr_);
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_REQUANTIZE_INL_H_
