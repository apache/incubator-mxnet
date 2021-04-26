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
 * \file quantization_utils-inl.h
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZATION_UTILS_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZATION_UTILS_H_

#include <mxnet/base.h>
#include <algorithm>
#include "../mxnet_op.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

using mshadow::red::limits::MaxValue;
using mshadow::red::limits::MinValue;

static const float kUint8Range = 255.5;
static const float kInt8Range = 127.5;
static const size_t kInt32Range = 0x7fffffff;

template<typename T>
MSHADOW_XINLINE int Sign(T val) {
  return (val > T(0)) - (val < T(0));
}

template<typename T>
MSHADOW_XINLINE T Abs(T a) {
#ifdef __CUDACC__
  return ::abs(a);
#else
  return std::abs(a);
#endif
}

template<typename T>
MSHADOW_XINLINE T Max(T a, T b) {
#ifdef __CUDACC__
  return ::max(a, b);
#else
  return std::max(a, b);
#endif
}

template<typename T>
MSHADOW_XINLINE T Min(T a, T b) {
#ifdef __CUDACC__
  return ::min(a, b);
#else
  return std::min(a, b);
#endif
}

template<typename T>
MSHADOW_XINLINE float MaxAbs(T a, T b) {
  return Max(Abs(static_cast<float>(a)), Abs(static_cast<float>(b)));
}

template<typename T>
MSHADOW_XINLINE float MinAbs(T a, T b) {
  return Min(Abs(static_cast<float>(a)), Abs(static_cast<float>(b)));
}

template<typename T>
MSHADOW_XINLINE T FloatToQuantized(float input, float min_range, float max_range) {
  float real_range = MaxAbs(min_range, max_range);
  float quantized_range = MinAbs(MaxValue<T>(), MinValue<T>());
  float scale = quantized_range / real_range;
  return Sign(input) * Min(Abs(input) * scale + 0.5f, quantized_range);
}

template <typename T>
MSHADOW_XINLINE float QuantizedToFloat(T input, float min_range, float max_range) {
  float quantized_range = MinAbs(MinValue<T>(), MaxValue<T>());
  float real_range = MaxAbs(min_range, max_range);
  float scale = real_range / quantized_range;
  return input * scale;
}

struct QuantizedToFloatStruct {
  template<typename T>
  MSHADOW_XINLINE static void Map(int i, float *output, const T *input,
                                  const float *range_min, const float *range_max) {
    output[i] = QuantizedToFloat(input[i], *range_min, *range_max);
  }
};

template <class T1, class T2>
MSHADOW_XINLINE T2 RequantizeInNewRange(T1 input, float min_input, float max_input,
                                        float min_new, float max_new) {
  const float input_float = QuantizedToFloat<T1>(input, min_input, max_input);
  return FloatToQuantized<T2>(input_float, min_new, max_new);
}

template <class T1, class T2>
MSHADOW_XINLINE void RequantizeManyInNewRange(size_t count, T2* output, const T1 *input,
                                              float input_min, float input_max,
                                              float actual_min, float actual_max) {
  for (size_t i = 0; i < count; ++i) {
    const float input_float =
        QuantizedToFloat<T1>(input[i], input_min, input_max);
    output[i] = FloatToQuantized<T2>(input_float, actual_min, actual_max);
  }
}

/*!
 * \brief Get the scaling factor for converting type T to float.
 */
template<typename T>
MSHADOW_XINLINE float FloatForOneQuantizedLevel(float range_min, float range_max, bool all_sign) {
  float range_data = MaxAbs(range_min, range_max);
  float range_T = all_sign ? MinAbs(MinValue<T>(), MaxValue<T>()) : MaxValue<T>();
  return range_data / range_T;
}

template <typename TA, typename TB, typename TC>
MSHADOW_XINLINE void QuantizationRangeForMultiplication(float min_a, float max_a, float min_b,
                                                        float max_b, float *min_c, float *max_c,
                                                        bool all_sign) {
  const float a_float_for_one_quant_level = FloatForOneQuantizedLevel<TA>(min_a, max_a, all_sign);
  const float b_float_for_one_quant_level = FloatForOneQuantizedLevel<TB>(min_b, max_b, all_sign);
  const float range_c =
      MinAbs(static_cast<int64_t>(MinValue<TC>()), static_cast<int64_t>(MaxValue<TC>()));
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;
  *max_c = c_float_for_one_quant_level * range_c;
  *min_c = -*max_c;
}

struct QuantizationRangeForS8S8MultiplicationStruct {
  MSHADOW_XINLINE static void Map(int i,
                                  float *min_c,
                                  float *max_c,
                                  const float *min_a,
                                  const float *max_a,
                                  const float *min_b,
                                  const float *max_b) {
  QuantizationRangeForMultiplication<int8_t, int8_t, int32_t>(
    min_a[i], max_a[i], min_b[i], max_b[i], min_c, max_c, true);
  }
};

struct QuantizationRangeForS8U8MultiplicationStruct {
  MSHADOW_XINLINE static void Map(int i,
                                  float *min_c,
                                  float *max_c,
                                  const float *min_a,
                                  const float *max_a,
                                  const float *min_b,
                                  const float *max_b) {
  QuantizationRangeForMultiplication<int8_t, uint8_t, int32_t>(
    min_a[i], max_a[i], min_b[i], max_b[i], min_c, max_c, false);
  }
};

template<typename xpu, typename DType>
inline size_t ConfigReduce(mshadow::Stream<xpu>* s,
                           const mxnet::TShape& data_shape,
                           const mxnet::TShape& out_shape,
                           mxnet::TShape* src_shape,
                           mxnet::TShape* dst_shape) {
  BroadcastReduceShapeCompact(data_shape, out_shape, src_shape, dst_shape);
  constexpr int NDim = 2;
  CHECK_EQ(src_shape->ndim(), NDim);
  CHECK_EQ(dst_shape->ndim(), NDim);

  return broadcast::ReduceWorkspaceSize(s, *dst_shape, kWriteTo, *src_shape);
}

enum QuantizeOutType { kAuto = 0, kInt8, kUint8 };

template<typename Param>
static mshadow::TypeFlag GetQuantizeOutputType(const Param &param) {
  auto out_type = mshadow::kInt8;
  if (param.out_type == QuantizeOutType::kAuto) {
    if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
      if (param.min_calib_range.value() >= 0.0) {
        out_type = mshadow::kUint8;
      } else {
        out_type = mshadow::kInt8;
      }
    }
  } else if (param.out_type == QuantizeOutType::kInt8) {
    out_type = mshadow::kInt8;
  } else if (param.out_type == QuantizeOutType::kUint8) {
    out_type = mshadow::kUint8;
  } else {
    LOG(FATAL) << "Unsupported out_type in params: " <<param.out_type;
  }
  return out_type;
}

static inline float GetQuantizeScale(const int dtype, const float data_min, const float data_max) {
  const float real_data_range = MaxAbs(data_min, data_max);
  const auto quantized_data_range = (dtype == mshadow::kInt8) ? kInt8Range : kUint8Range;
  // If real_data_range == 0, to avoid `inf` in scale, use a large number here, which is MAX_INT.
  return real_data_range ? quantized_data_range / real_data_range : MaxValue<int32_t>();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZATION_UTILS_H_
