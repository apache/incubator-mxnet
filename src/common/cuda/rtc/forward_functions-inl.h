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

#ifndef MXNET_COMMON_CUDA_RTC_FORWARD_FUNCTIONS_INL_H_
#define MXNET_COMMON_CUDA_RTC_FORWARD_FUNCTIONS_INL_H_

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

const char function_definitions_util[] = R"code(

#define INT_MAX (2147483647)

namespace op {

template <typename DType>
struct LoadType {
  using Type = DType;
};

template <>
struct LoadType<half> {
  using Type = float;
};

template <typename DType>
__device__ inline typename LoadType<DType>::Type load(const DType input) {
  return input;
}

template <>
__device__ inline float load(const half input) {
  return __half2float(input);
}

template <typename DType1, typename DType2>
__device__ inline DType1 store(const DType2 input, DType1* ref) {
  return input;
}

template <typename DType>
__device__ inline half store(const DType input, half* ref) {
  return __float2half(input);
}

template <int ndim>
struct Shape {
   int x[ndim];
   size_t size;
   __device__ inline const int& operator [](const int i) const {
       return x[i];
   }
   __device__ inline int& operator [](const int i) {
       return x[i];
   }
   __device__ inline void set(const int def) {
       #pragma unroll
       for (int i = 0; i < ndim; i++) {
           x[i] = def;
       }
   }
};

template <>
struct Shape<0> {
   size_t size;
};

template <int nvec, typename DType, int ndim>
__device__ inline vector::VectorizedStorage<DType, nvec> load_index(const DType * input, int i,
                                                                    const Shape<ndim> &shape) {
  using V = vector::VectorizedStorage<DType, nvec>;
  if (i < shape.size) {
    const auto* vector_input = reinterpret_cast<const typename V::LType *>(input + i);
    return V(*vector_input);
  } else {
    return V({0});
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline vector::VectorizedStorage<DType, nvec> global_load_index(const DType * input,
                    int i, const Shape<ndim> &shape) {
  using V = vector::VectorizedStorage<DType, nvec>;
  if (i < shape.size) {
    const auto* vector_input = reinterpret_cast<const typename V::LType *>(input + i);
    return V(__ldg(vector_input));
  } else {
    return V({0});
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline vector::VectorizedStorage<DType, nvec> load_slice(const DType * input,
                                                                    const Shape<ndim>& shape,
                                                                    Shape<ndim> begin,
                                                                    Shape<ndim> end,
                                                                    int offset) {
  int idx[nvec];

  Shape<ndim> ref_strides;
  Shape<ndim> strides;
  ref_strides[ndim-1] = 1;
  strides[ndim-1] = 1;
  #pragma unroll
  for (int dim = ndim-1; dim >=0; dim--) {
    if (begin[dim] < 0) begin[dim] = shape[dim] + begin[dim];
    if (end[dim] < 0) end[dim] = shape[dim] + end[dim];
    if (end[dim] == INT_MAX) end[dim] = shape[dim];
    if (dim > 0) {
      ref_strides[dim-1] = ref_strides[dim] * (end[dim] - begin[dim]);
      strides[dim-1] = strides[dim] * shape[dim];
    }
  }
  #pragma unroll
  for (int j = 0; j < nvec; j++) {
    idx[j] = 0;
    int ref_idx = offset + j;
    #pragma unroll
    for (int dim = 0; dim < ndim; dim++) {
       int stride = ref_strides[dim];
       if (shape[dim] > 1) {
         idx[j] += (ref_idx / stride + begin[dim]) * strides[dim];
       }
       ref_idx = ref_idx % stride;
    }
  }
  vector::VectorizedStorage<DType, nvec> ret;
  #pragma unroll
  for (int j = 0; j < nvec; j++) {
      ret.scratch_.separate[j] = idx[j] < shape.size ? *(input + idx[j]) : DType {};
  }
  return ret;
}

template <int nvec, typename DType, int ndim>
__device__ inline vector::VectorizedStorage<DType, nvec> fast_load_slice(const DType * input,
                                                                         const Shape<ndim>& shape,
                                                                         Shape<ndim> begin,
                                                                         Shape<ndim> end,
                                                                         int offset) {
  int idx = 0;

  Shape<ndim> ref_strides;
  Shape<ndim> strides;
  ref_strides[ndim-1] = 1;
  strides[ndim-1] = 1;
  #pragma unroll
  for (int dim = ndim-1; dim >=0; dim--) {
    if (begin[dim] < 0) begin[dim] = shape[dim] + begin[dim];
    if (end[dim] < 0) end[dim] = shape[dim] + end[dim];
    if (end[dim] == INT_MAX) end[dim] = shape[dim];
    if (dim > 0) {
      ref_strides[dim-1] = ref_strides[dim] * (end[dim] - begin[dim]);
      strides[dim-1] = strides[dim] * shape[dim];
    }
  }
  int ref_idx = offset;
  #pragma unroll
  for (int dim = 0; dim < ndim; dim++) {
     int stride = ref_strides[dim];
     if (shape[dim] > 1) {
       idx += (ref_idx / stride + begin[dim]) * strides[dim];
     }
     ref_idx = ref_idx % stride;
  }
  return global_load_index<nvec>(input, idx, shape);
}

template <int nvec, typename DType, int ndim>
__device__ inline void store_index(const vector::VectorizedStorage<DType, nvec> value, int i,
                        DType * output, const Shape<ndim>& shape) {
  if (i < (shape.size + nvec - 1) / nvec) {
    auto vector_output = reinterpret_cast<
                          typename vector::VectorizedStorage<DType, nvec>::LType *>(output);
    vector_output[i] = value.scratch_.aligned;
  }
}

template <int nvec, typename DType, int ndim>
__device__ inline void store_add_index(const vector::VectorizedStorage<DType, nvec> value, int i,
                            DType * output, const Shape<ndim>& shape) {
  if (i < (shape.size + nvec - 1) / nvec) {
    auto vector_output = reinterpret_cast<
                          typename vector::VectorizedStorage<DType, nvec>::LType *>(output);
    vector::VectorizedStorage<DType, nvec> ret(vector_output[i]);
    ret += value;
    vector_output[i] = ret.scratch_.aligned;
  }
}

}  // namespace op
)code";

const char function_definitions_binary[] = R"code(
namespace op {

template <typename DType>
__device__ inline bool isnan(const DType val) {
  return util::isnan(val);
}

template <typename DType>
__device__ inline bool_t isinf(const DType val) {
  return util::isinf(val);
}

template <typename DType>
__device__ inline bool_t isposinf(const DType val) {
  return util::isinf(val) && (val > 0);
}

template <typename DType>
__device__ inline bool_t isneginf(const DType val) {
  return util::isinf(val) && (val < 0);
}

template <typename DType>
__device__ inline bool_t isfinite(const DType val) {
  return !op::isnan(val) && !op::isinf(val);
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
add(const DType a, const DType2 b) {
  return a + b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
sub(const DType a, const DType2 b) {
  return a - b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
rsub(const DType a, const DType2 b) {
  return b - a;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
mul(const DType a, const DType2 b) {
  return a * b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
div(const DType a, const DType2 b) {
  return a / b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
rdiv(const DType a, const DType2 b) {
  return b / a;
}

#define DEFINE_BINARY_MATH_FUNC(name, double_version, float_version) \
template <typename DType, typename DType2> \
__device__ inline typename type_util::mixed_type<DType, DType2>::type \
name (const DType a, const DType2 b) { \
  if (type_util::has_double_or_integral<DType, DType2>::value) { \
    return double_version ((double)a, (double)b); \
  } else { \
    return float_version ((float)a, (float)b); \
  } \
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
power (const DType a, const DType2 b) {
  if (type_util::has_double<DType, DType2>::value) {
    return ::pow ((double)a, (double)b); \
  } else {
    return ::powf ((float)a, (float)b);
  }
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
rpow(const DType a, const DType2 b) {
  return power(b, a);
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
max(const DType a, const DType2 b) {
  if (isnan(a)) return a;
  return a > b ? a : b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
fmax(const DType a, const DType2 b) {
  if (isnan(b)) return a;
  return a > b ? a : b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
min(const DType a, const DType2 b) {
  if (isnan(a)) return a;
  return a < b ? a : b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
fmin(const DType a, const DType2 b) {
  if (isnan(b)) return a;
  return a < b ? a : b;
}

DEFINE_BINARY_MATH_FUNC(hypot, ::hypot, ::hypotf)

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
mod(const DType a, const DType2 b) {
  if (b == 0) {
    return 0;
  }
  const double ad = static_cast<double>(a);
  const double bd = static_cast<double>(b);
  if (bd < 0) {
    if (ad < 0) {
      return -::fmod(-ad, -bd);
    } else {
      return ::fmod(ad, -bd) +
             (::fmod(ad, -bd) != 0 ? bd : 0);
    }
  } else {
    if (ad < 0) {
      return -::fmod(-ad, bd) +
              (::fmod(-ad, bd) != 0 ? bd : 0);
    } else {
      return ::fmod(ad, bd);
    }
  }
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
fmod(const DType a, const DType2 b) {
  if (b == 0) {
    return 0;
  }
  return ::fmod(static_cast<double>(a), static_cast<double>(b));
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
rmod(const DType a, const DType2 b) {
  return op::mod(b, a);
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
rfmod(const DType a, const DType2 b) {
  return op::fmod(b, a);
}

template <typename DType, typename DType2>
__device__ inline DType equal(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a == real_b ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline DType not_equal(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a != real_b ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline DType greater(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a > real_b ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline DType greater_equal(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a >= real_b ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline DType less(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a < real_b ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline DType less_equal(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a <= real_b ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline bool_t np_equal(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a == real_b ? true : false;
}

template <typename DType, typename DType2>
__device__ inline bool_t np_not_equal(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a != real_b ? true : false;
}

template <typename DType, typename DType2>
__device__ inline bool_t np_greater(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a > real_b ? true : false;
}

template <typename DType, typename DType2>
__device__ inline bool_t np_greater_equal(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a >= real_b ? true : false;
}

template <typename DType, typename DType2>
__device__ inline bool_t np_less(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a < real_b ? true : false;
}

template <typename DType, typename DType2>
__device__ inline bool_t np_less_equal(const DType a, const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a <= real_b ? true : false;
}

template <typename DType, typename DType2>
__device__ inline DType logical_and(const DType a, const DType2 b) {
  return a && b ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline DType logical_or(const DType a, const DType2 b) {
  return a || b ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline DType logical_xor(const DType a, const DType2 b) {
  return ((a || b) && !(a && b)) ? 1 : 0;
}

template <typename DType, typename DType2>
__device__ inline DType copysign(const DType a, const DType2 b) {
  return (a >= 0 && b >= 0) || (a < 0 && b < 0) ? a : -a;
}

template <typename DType, typename DType2>
__device__ inline DType2 rcopysign(const DType a, const DType2 b) {
  return copysign(b, a);
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
lcm(const DType a, const DType2 b) {
  if (type_util::is_integral<DType>::value &&
      type_util::is_integral<DType2>::value) {
    DType A = a;
    DType2 B = b;
    // minus cases.
    if (a < 0) {
      A = -a;
    }
    if (b < 0) {
      B = -b;
    }
    // handle zero-valued cases.
    DType c;
    if (a == 0 || b == 0) {
      c = 0;
    } else {
      DType tmp;
      DType tmp_a = A;
      DType tmp_b = B;
      if (A < B) {
        tmp = A;
        A = B;
        B = tmp;
      }
      while (A % B != 0) {
        A = A % B;
        tmp = A;
        A = B;
        B = tmp;
      }
      c = tmp_a / B * tmp_b;
    }
    return c;
  } else {
    return 0;
  }
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
gcd(const DType a, const DType2 b) {
  if (type_util::is_integral<DType>::value &&
      type_util::is_integral<DType2>::value) {
    DType A = a;
    DType2 B = b;
    // minus cases.
    if (a < 0) {
      A = -a;
    }
    if (b < 0) {
      B = -b;
    }
    // handle zero-valued cases.
    DType c;
    if (a == 0 && b != 0) {
      c = B;
    } else if (b == 0 && a != 0) {
      c = A;
    } else if (a == 0 && b == 0) {
      c = 0;
    } else {
      DType tmp;
      if (A < B) {
        tmp = A;
        A = B;
        B = tmp;
      }
      while (A % B != 0) {
        A = A % B;
        tmp = A;
        A = B;
        B = tmp;
      }
      c = B;
    }
    return c;
  } else {
    return 0;
  }
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type bitwise_xor(const DType a,
                                                                       const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a ^ real_b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type bitwise_or(const DType a,
                                                                       const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a | real_b;
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type bitwise_and(const DType a,
                                                                       const DType2 b) {
  using mixed_type = typename type_util::mixed_type<DType, DType2>::type;
  const mixed_type real_a = a;
  const mixed_type real_b = b;
  return real_a & real_b;
}

DEFINE_BINARY_MATH_FUNC(arctan2, ::atan2, ::atan2f)

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
rarctan2(const DType a, const DType2 b) {
  return arctan2(b, a);
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
ldexp(const DType a, const DType2 b) {
  if (type_util::has_double_or_integral<DType, DType2>::value) {
    return a * ::pow(2.0, static_cast<double>(b));
  } else {
    return a * ::powf(2.0f, static_cast<float>(b));
  }
}

template <typename DType, typename DType2>
__device__ inline typename type_util::mixed_type<DType, DType2>::type
rldexp(const DType a, const DType2 b) {
  return ldexp(b, a);
}

#undef DEFINE_BINARY_MATH_FUNC

template <typename DType, typename DType2>
__device__ inline bool np_logical_and(const DType val, const DType2 val2) {
  return (val && val2) ? true : false;
}

template <typename DType, typename DType2>
__device__ inline bool np_logical_or(const DType val, const DType2 val2) {
  return (val || val2) ? true : false;
}

template <typename DType, typename DType2>
__device__ inline bool np_logical_xor(const DType val, const DType2 val2) {
  return ((val || val2) && !(val && val2)) ? true : false;
}

template <typename DType, typename DType2>
__device__ inline DType left(const DType left_val, const DType2 right_val) {
  return left_val;
}

template <typename DType, typename DType2>
__device__ inline DType2 right(const DType left_val, const DType2 right_val) {
  return right_val;
}

}  // namespace op
)code";

const char function_definitions_unary[] = R"code(
namespace op {

template <typename DType>
__device__ inline DType identity(const DType val) {
  return val;
}

template <typename DType>
__device__ inline DType negation(const DType val) {
  return -val;
}

template <typename OutType, typename DType>
__device__ inline typename LoadType<OutType>::Type cast(const DType val) {
  return static_cast<typename LoadType<OutType>::Type>(val);
}

// activations

template <typename DType>
__device__ inline DType relu(const DType val) {
  return (isnan(val) || val > 0) ? val : 0;
}

template <typename DType>
__device__ inline DType sigmoid(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return 1./(1 + ::exp(-val));
  } else {
    return 1.f/(1 + expf(-val));
  }
}

template <typename DType>
__device__ inline DType softrelu(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return ::log(1 + ::exp(val));
  } else {
    return logf(1 + expf(val));
  }
}

template <typename DType>
__device__ inline DType softsign(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return val / (1 + fabs(val));
  } else {
    return val / (1 + fabsf(val));
  }
}

// exp and log

#define DEFINE_UNARY_MATH_FUNC(name, double_version, float_version) \
template <typename DType> \
__device__ inline DType name (const DType a) { \
  if (type_util::has_double_or_integral<DType>::value) { \
    return double_version ((double)a); \
  } else { \
    return float_version (a); \
  } \
}

DEFINE_UNARY_MATH_FUNC(exp, ::exp, ::expf)
DEFINE_UNARY_MATH_FUNC(expm1, ::expm1, ::expm1f)
DEFINE_UNARY_MATH_FUNC(log, ::log, ::logf)
DEFINE_UNARY_MATH_FUNC(log10, ::log10, ::log10f)
DEFINE_UNARY_MATH_FUNC(log2, ::log2, ::log2f)
DEFINE_UNARY_MATH_FUNC(log1p, ::log1p, ::log1pf)

// trigonometric

constexpr double pi = 3.14159265358979323846;

template <typename DType>
__device__ inline DType degrees(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return (val / pi) * 180;
  } else {
    return (val / static_cast<float>(pi)) * 180.f;
  }
}

template <typename DType>
__device__ inline DType radians(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return (val / 180.0) * pi;
  } else {
    return (val / 180.0f) * static_cast<float>(pi);
  }
}

DEFINE_UNARY_MATH_FUNC(sin, ::sin, ::sinf)
DEFINE_UNARY_MATH_FUNC(cos, ::cos, ::cosf)
DEFINE_UNARY_MATH_FUNC(tan, ::tan, ::tanf)
DEFINE_UNARY_MATH_FUNC(arcsin, ::asin, ::asinf)
DEFINE_UNARY_MATH_FUNC(arccos, ::acos, ::acosf)
DEFINE_UNARY_MATH_FUNC(arctan, ::atan, ::atanf)

DEFINE_UNARY_MATH_FUNC(sinh, ::sinh, ::sinhf)
DEFINE_UNARY_MATH_FUNC(cosh, ::cosh, ::coshf)
DEFINE_UNARY_MATH_FUNC(tanh, ::tanh, ::tanhf)
DEFINE_UNARY_MATH_FUNC(arcsinh, ::asinh, ::asinhf)
DEFINE_UNARY_MATH_FUNC(arccosh, ::acosh, ::acoshf)
DEFINE_UNARY_MATH_FUNC(arctanh, ::atanh, ::atanhf)

// sqrt

DEFINE_UNARY_MATH_FUNC(sqrt, ::sqrt, ::sqrtf)
DEFINE_UNARY_MATH_FUNC(rsqrt, ::rsqrt, ::rsqrtf)
DEFINE_UNARY_MATH_FUNC(cbrt, ::cbrt, ::cbrtf)
DEFINE_UNARY_MATH_FUNC(rcbrt, ::rcbrt, ::rcbrtf)

template <typename DType>
__device__ inline DType square(const DType val) {
  return val * val;
}

template <typename DType, typename... DTypes>
__device__ inline typename LoadType<DType>::Type zero(const DType val, const DTypes... args) {
  return 0;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type zero() {
  return 0;
}

template <typename DType, typename... DTypes>
__device__ inline typename LoadType<DType>::Type one(const DType val, const DTypes... args) {
  return 1;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type one() {
  return 1;
}

template <typename DType, typename... DTypes>
__device__ inline typename LoadType<DType>::Type negone(const DType val, const DTypes... args) {
  return -1;
}

template <typename DType>
__device__ inline typename LoadType<DType>::Type negone() {
  return -1;
}

template <typename DType>
__device__ inline DType round(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::round((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::roundf(val);
  }
}

template <typename DType>
__device__ inline DType floor(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::floor((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::floorf(val);
  }
}

template <typename DType>
__device__ inline DType ceil(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::ceil((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::ceilf(val);
  }
}

template <typename DType>
__device__ inline DType rint(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::rint((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::rintf(val);
  }
}

template <typename DType>
__device__ inline DType fix(const DType val) {
  const auto f = floor(val);
  const auto c = ceil(val);
  return (f > 0 ? f : -f) < (c > 0 ? c : -c) ? f : c;
}

template <typename DType>
__device__ inline DType trunc(const DType val) {
  if (type_util::has_double<DType>::value) {
    return ::trunc((double)val);
  } else if (type_util::is_integral<DType>::value) {
    return val;
  } else {
    return ::truncf(val);
  }
}

template <typename DType>
__device__ inline DType clip(const DType val, const float a_min, const float a_max) {
  if (val > a_max) {
    return a_max;
  } else if (val < a_min) {
    return a_min;
  } else {
    return val;
  }
}

template <typename DType>
__device__ inline DType sign(const DType val) {
  if (val < 0) return -1;
  return val > 0 ? 1 : 0;
}

template <typename DType>
__device__ inline DType reciprocal(const DType val) {
  return 1.0f / val;
}

DEFINE_UNARY_MATH_FUNC(abs, ::fabs, ::fabsf)
DEFINE_UNARY_MATH_FUNC(gamma, ::tgamma, ::tgammaf)
DEFINE_UNARY_MATH_FUNC(gammaln, ::lgamma, ::lgammaf)
DEFINE_UNARY_MATH_FUNC(erf, ::erf, ::erff)
DEFINE_UNARY_MATH_FUNC(erfinv, ::erfinv, ::erfinvf)

template <typename DType>
__device__ inline DType gelu(const DType val) {
  return 0.5f * val * (1.0f + op::erf(val / op::sqrt(2.0f)));
}

template <typename DType1, typename DType2>
__device__ inline DType1 smooth_l1(const DType1 val, const DType2 scalar) {
  const auto bsq = scalar * scalar;
  const auto ibsq = 1.0f / bsq;
  if (val > ibsq) {
    return val - 0.5f * ibsq;
  } else if (val < -ibsq) {
    return -val - 0.5f * ibsq;
  } else {
    return 0.5f * val * val * bsq;
  }
}

template <typename DType>
__device__ inline DType digamma(const DType val) {
  if (type_util::has_double_or_integral<DType>::value) {
    return special_functions::cephes::psi<double>(val);
  } else {
    return special_functions::cephes::psi<float>(val);
  }
}

template <typename DType>
__device__ inline DType logical_not(const DType val) {
  return val != DType(0) ? DType(0) : DType(1);
}

template <typename DType>
__device__ inline bool_t np_logical_not(const DType val) {
  return !static_cast<bool>(val);
}

template <typename DType>
__device__ inline bool_t NonZero(const DType val) {
  return val != 0;
}

#undef DEFINE_UNARY_MATH_FUNC

template <typename DType>
__device__ inline DType bitwise_not(const DType a) {
  if (type_util::is_same<DType, bool_t>::value) {
    return !a;
  } else {
    return ~static_cast<int64>(a);
  }
}

}  // namespace op

)code";

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_FORWARD_FUNCTIONS_INL_H_
