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
 * Copyright (c) 2015 by Contributors
 * \file mshadow_op.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_MSHADOW_OP_H_
#define MXNET_OPERATOR_MSHADOW_OP_H_

#include <mxnet/base.h>
#include <mshadow/base.h>
#include "math.h"
#include "math_functions-inl.h"
#include "special_functions-inl.h"
#include "./operator_tune.h"
#include "./contrib/erfinv-inl.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

namespace mxnet {
namespace op {
namespace mshadow_op {

using mshadow::isnan_typed::IsNan;
using mshadow::isinf_typed::IsInf;

#ifdef __CUDA_ARCH__
__constant__ const float PI = 3.14159265358979323846;
__constant__ const float SELU_ALPHA = 1.6732632423543772848170429916717;
__constant__ const float SELU_LAMBDA = 1.0507009873554804934193349852946;
__constant__ const float SQRT_2 = 1.4142135623730950488016887242096;
#else
const float PI = 3.14159265358979323846;
const float SELU_ALPHA = 1.6732632423543772848170429916717;
const float SELU_LAMBDA = 1.0507009873554804934193349852946;
const float SQRT_2 = 1.4142135623730950488016887242096;
#endif
using std::enable_if;
using std::is_unsigned;
using std::is_integral;

#define MXNET_UNARY_MATH_OP(name, expr) \
  struct name : public mxnet_op::tunable { \
    template<typename DType> \
    MSHADOW_XINLINE static DType Map(DType a) { \
      return DType(expr); \
    } \
  }

#define MXNET_UNARY_MATH_OP_NC(name, expr) \
  struct name : public mxnet_op::tunable { \
    template<typename DType> \
    MSHADOW_XINLINE static DType Map(DType a) { \
      return (expr); \
    } \
  }

#define MXNET_UNARY_LOGIC_OP_NC(name, expr) \
  struct name : public mxnet_op::tunable { \
    template<typename DType> \
    MSHADOW_XINLINE static bool Map(DType a) { \
      return (expr); \
    } \
  }

#define MXNET_BINARY_MATH_OP(name, expr) \
  struct name : public mxnet_op::tunable { \
    template<typename DType> \
    MSHADOW_XINLINE static DType Map(DType a, DType b) { \
      return DType(expr); \
    } \
  }

#define MXNET_BINARY_MATH_OP_NC(name, expr) \
  struct name : public mxnet_op::tunable  { \
    template<typename DType> \
    MSHADOW_XINLINE static DType Map(DType a, DType b) { \
      return (expr); \
    } \
  }

#define MXNET_BINARY_MATH_OP_NC_WITH_BOOL(name, expr) \
  struct name : public mxnet_op::tunable  { \
    template<typename DType, \
             typename std::enable_if<!std::is_same<DType, bool>::value, int>::type = 0> \
    MSHADOW_XINLINE static DType Map(DType a, DType b) { \
      return (expr); \
    } \
    MSHADOW_XINLINE static bool Map(bool a, bool b) { \
      return (expr); \
    } \
  }

#define MXNET_BINARY_LOGIC_OP_NC(name, expr) \
  struct name : public mxnet_op::tunable  { \
    template<typename DType, typename EType> \
    MSHADOW_XINLINE static bool Map(DType lhs, EType rhs) { \
      double a = static_cast<double>(lhs); \
      double b = static_cast<double>(rhs); \
      return (expr); \
    } \
  }

#define MXNET_SIMPLE_UNARY_MATH_OP(name) MXNET_UNARY_MATH_OP(name, math::name(a))

#define MXNET_SIMPLE_BINARY_MATH_OP(name) MXNET_BINARY_MATH_OP(name, math::name(a, b))

MXNET_UNARY_MATH_OP_NC(identity, a);

template <typename IType, typename DType>
struct IndexedNum {
  IType idx;
  DType num;

  MSHADOW_XINLINE IndexedNum() : idx(0), num(0) {}

  MSHADOW_XINLINE IndexedNum(DType n) : idx(0), num(n) {}

  MSHADOW_XINLINE IndexedNum& operator+=(const IndexedNum& rhs){
    return *this;
  }
};

template<typename AType, typename IType>
struct set_index_no_op : public mxnet_op::tunable {
  static const bool do_op = false;

  MSHADOW_XINLINE static void Op(AType* const a, IType i) {
  }
};

template<typename AType, typename IType>
struct arg_min_max_set_index : public mxnet_op::tunable {
  static const bool do_op = true;

  MSHADOW_XINLINE static void Op(AType* const a, IType i) {
    a->idx = i;
  }
};


MXNET_UNARY_MATH_OP(identity_grad, 1);

struct identity_with_cast {
  template<typename DTypeIn, typename DTypeOut>
  MSHADOW_XINLINE static void Map(index_t i, DTypeOut *out, DTypeIn *in) {
    out[i] = DTypeOut(in[i]);
  }
};

struct true_divide : public mxnet_op::tunable  {
  template<typename DType,
           typename std::enable_if<!std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a / b;
  }

  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, DType b) {
    return static_cast<float>(a) / static_cast<float>(b);
  }

  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return static_cast<mshadow::half::half_t>(a) / b;
  }

  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return static_cast<float>(a) / b;
  }

  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return static_cast<double>(a) / b;
  }
};

struct rtrue_divide : public mxnet_op::tunable  {
  template<typename DType,
           typename std::enable_if<!std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return b / a;
  }

  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, DType b) {
    return static_cast<float>(b) / static_cast<float>(a);
  }

  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return b / static_cast<mshadow::half::half_t>(a);
  }

  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return b / static_cast<float>(a);
  }

  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return b / static_cast<double>(a);
  }
};

MXNET_BINARY_MATH_OP_NC(left, a);

MXNET_BINARY_MATH_OP_NC(right, b);

struct mixed_plus {
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return static_cast<mshadow::half::half_t>(a) + b;
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return static_cast<float>(a) + b;
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return static_cast<double>(a) + b;
  }
};

struct mixed_minus {
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return static_cast<mshadow::half::half_t>(a) - b;
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return static_cast<float>(a) - b;
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return static_cast<double>(a) - b;
  }
};

struct mixed_rminus {
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return b - static_cast<mshadow::half::half_t>(a);
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return b - static_cast<float>(a);
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return b - static_cast<double>(a);
  }
};

struct mixed_mul {
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return static_cast<mshadow::half::half_t>(a) * b;
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return static_cast<float>(a) * b;
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return static_cast<double>(a) * b;
  }
};

struct mixed_power {
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return static_cast<mshadow::half::half_t>(math::pow(a, b));
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return static_cast<float>(math::pow(a, b));
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return static_cast<double>(math::pow(a, b));
  }
};

struct mixed_rpower {
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return static_cast<mshadow::half::half_t>(math::pow(b, a));
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return static_cast<float>(math::pow(b, a));
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return static_cast<double>(math::pow(b, a));
  }
};

#pragma GCC diagnostic push
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wbool-compare"
#endif
MXNET_BINARY_MATH_OP_NC_WITH_BOOL(mul, a * b);

MXNET_BINARY_MATH_OP_NC_WITH_BOOL(div, a / b);

MXNET_BINARY_MATH_OP_NC_WITH_BOOL(plus, a + b);

MXNET_BINARY_MATH_OP_NC_WITH_BOOL(minus, a - b);
#pragma GCC diagnostic pop

MXNET_UNARY_MATH_OP(negation, -a);

MXNET_UNARY_MATH_OP(reciprocal, 1.0f / math::id(a));

struct bitwise_not : public mxnet_op::tunable {
  template<typename DType,
           typename std::enable_if<!std::is_same<DType, bool>::value, int>::type = 0>
  MSHADOW_XINLINE static DType Map(DType a) {
    return ~static_cast<int64_t>(a);
  }

  MSHADOW_XINLINE static bool Map(bool a) {
    return !a;
  }
};

MXNET_UNARY_MATH_OP(reciprocal_grad, -1.0f / math::sqr(a));

MXNET_UNARY_MATH_OP(sigmoid, 1.0f / (1.0f + math::exp(-a)));

MXNET_UNARY_MATH_OP(sigmoid_grad, math::id(a) * (1.0f - math::id(a)));

MXNET_UNARY_MATH_OP(softsign, a / (1.0f + math::fabs(a)));

MXNET_UNARY_MATH_OP(softsign_grad, 1.0f /  math::sqr(1.0f + math::fabs(a)));

MXNET_UNARY_MATH_OP_NC(selu, DType(SELU_LAMBDA) *
                         (a > DType(0) ? a : DType(math::id(SELU_ALPHA) * math::expm1(a))));

MXNET_UNARY_MATH_OP_NC(selu_grad,
                       DType(SELU_LAMBDA) * (a > DType(0) ? DType(1) : DType(SELU_ALPHA + a)));

MXNET_BINARY_MATH_OP_NC(prelu_grad, a > DType(0) ? DType(0) : a);

MXNET_BINARY_MATH_OP_NC(xelu, a > DType(0) ? a :
                        DType(static_cast<float>(a) * static_cast<float>(b)));

MXNET_BINARY_MATH_OP_NC(xelu_grad, a > DType(0) ? DType(1) : b);

MXNET_BINARY_MATH_OP_NC(elu, a > DType(0) ? a :
                        DType(math::id(b) * math::expm1(a)));

MXNET_BINARY_MATH_OP_NC(elu_grad, a > DType(0) ? DType(1) : DType(b + a));

MXNET_SIMPLE_UNARY_MATH_OP(tanh);

MXNET_UNARY_MATH_OP(tanh_grad, 1.0f - math::sqr(a));

/*! \brief SoftReLU, also known as softplus activation */
struct softrelu : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // Avoid overflow of exp for large inputs.
    // Thresholds 20.0 is chosen such that softrelu(a) = a
    // for a > 20 using floating precision
    if (a > DType(20.0f)) {
      return a;
    } else {
      return DType(math::log1p(math::exp(a)));
    }
  }
};

MXNET_UNARY_MATH_OP(softrelu_grad, -math::expm1(-a));

MXNET_UNARY_MATH_OP(erfinv_grad, 0.5 * math::sqrt(PI) * math::exp(math::sqr(a)));

MXNET_UNARY_MATH_OP(erf_grad, 2.0 / math::sqrt(PI) * math::exp(-(a * a)));

MXNET_SIMPLE_UNARY_MATH_OP(erf);

MXNET_UNARY_MATH_OP(gelu,
  DType(0.5f * static_cast<float>(a) * (1.0f + math::erf(static_cast<float>(a) / SQRT_2))));

MXNET_BINARY_MATH_OP_NC(gelu_grad,
  DType(0.5f * (1.0f + math::erf(static_cast<float>(a) / SQRT_2) +
                static_cast<float>(a) * erf_grad::Map(static_cast<float>(a) / SQRT_2) / SQRT_2)));

MXNET_SIMPLE_UNARY_MATH_OP(exp);

MXNET_SIMPLE_UNARY_MATH_OP(expm1);

MXNET_SIMPLE_UNARY_MATH_OP(log);

MXNET_UNARY_MATH_OP(log_grad, 1.0f / math::id(a));

MXNET_SIMPLE_UNARY_MATH_OP(log10);

// Constant is 1 / log(10)
struct log10_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(0.4342944819f / static_cast<float>(a));
  }
};

template<>
MSHADOW_XINLINE double log10_grad::Map<double>(double a) {
  return 0.43429448190325182765 / a;
}

MXNET_SIMPLE_UNARY_MATH_OP(log2);

// Constant is 1 / log(2)
struct log2_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return DType(1.442695041f / static_cast<float>(a));
  }
};

template<>
MSHADOW_XINLINE double log2_grad::Map<double>(double a) {
  return 1.44269504088896340737 / a;
}

MXNET_SIMPLE_UNARY_MATH_OP(sin);

MXNET_UNARY_MATH_OP(sin_grad, math::cos(a));

MXNET_SIMPLE_UNARY_MATH_OP(log1p);

MXNET_UNARY_MATH_OP(log1p_grad, 1.0f / (1.0f + math::id(a)));

MXNET_SIMPLE_UNARY_MATH_OP(cos);

MXNET_UNARY_MATH_OP(cos_grad, -math::sin(a));

MXNET_SIMPLE_UNARY_MATH_OP(tan);

MXNET_UNARY_MATH_OP(tan_grad, math::sqr(a) + 1.0f);

MXNET_UNARY_MATH_OP(arcsin, math::asin(a));

MXNET_UNARY_MATH_OP(arcsin_grad, 1.0f / math::sqrt(1.0f - math::sqr(a)));

MXNET_UNARY_MATH_OP(arccos, math::acos(a));

MXNET_UNARY_MATH_OP(arccos_grad, -1.0f / math::sqrt(1.0f - math::sqr(a)));

MXNET_UNARY_MATH_OP(arctan, math::atan(a));

MXNET_UNARY_MATH_OP(arctan_grad, 1.0f / (math::sqr(a) + 1.0f));

MXNET_SIMPLE_BINARY_MATH_OP(hypot);

MXNET_BINARY_MATH_OP(hypot_grad_left, math::id(a) / math::hypot(a, b));

MXNET_BINARY_MATH_OP(hypot_grad_right, math::id(b) / math::hypot(a, b));

MXNET_UNARY_MATH_OP(degrees, 180.0f / PI * math::id(a));

MXNET_UNARY_MATH_OP(degrees_grad, 180.0f / PI);

MXNET_UNARY_MATH_OP(radians, PI / 180.0f * math::id(a));

MXNET_UNARY_MATH_OP(radians_grad, PI / 180.0f);

MXNET_SIMPLE_UNARY_MATH_OP(sinh);

MXNET_UNARY_MATH_OP(sinh_grad, math::cosh(a));

MXNET_SIMPLE_UNARY_MATH_OP(cosh);

MXNET_UNARY_MATH_OP(cosh_grad, math::sinh(a));

MXNET_UNARY_MATH_OP(arcsinh, math::asinh(a));

MXNET_UNARY_MATH_OP(arcsinh_grad, 1.0f / math::hypot(a, DType(1)));

MXNET_UNARY_MATH_OP(arccosh, math::acosh(a));

MXNET_UNARY_MATH_OP(arccosh_grad, 1.0f / math::sqrt(math::sqr(a) - 1.0f));

MXNET_UNARY_MATH_OP(arctanh, math::atanh(a));

MXNET_UNARY_MATH_OP(arctanh_grad, 1.0f / (1.0f - math::sqr(a)));

MXNET_UNARY_MATH_OP(square, math::sqr(a));

MXNET_UNARY_MATH_OP(square_grad, 2.0f * math::id(a));

/*! \brief used for generate Bernoulli mask */
MXNET_BINARY_MATH_OP_NC(threshold, a < b ? DType(1) : DType(0));
MXNET_BINARY_MATH_OP_NC(threshold_eq, a <= b ? DType(1) : DType(0));

/*! \brief used for generate element of abs */
MXNET_UNARY_MATH_OP(abs, math::fabs(a)); // NOLINT(*)

/*! \brief used for generate element of sign */
struct sign : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_unsigned<DType>::value, DType>::type
  Map(DType a) {
    if (a < DType(0)) return DType(-DType(1));
    if (a > DType(0)) return DType(1);
    return DType(0);
  }
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_unsigned<DType>::value, DType>::type
  Map(DType a) {
    if (a > DType(0)) return DType(1);
    return DType(0);
  }
};

MXNET_UNARY_MATH_OP_NC(sign_grad, DType(0));

/*! \brief used for generate element of power */
MXNET_BINARY_MATH_OP(power, math::pow(a, b));

MXNET_BINARY_MATH_OP(power_grad, math::pow(a, b - DType(1)) * math::id(b));

MXNET_BINARY_MATH_OP(power_rgrad, math::pow(a, b) * math::log(a));

MXNET_BINARY_MATH_OP(rpower, math::pow(b, a));

MXNET_BINARY_MATH_OP(rpower_grad, math::id(a) * math::log(b));

MXNET_BINARY_MATH_OP(arctan2, math::atan2(a, b));
MXNET_BINARY_MATH_OP(arctan2_grad, math::id(b) / (math::id(a * a + b * b)));

MXNET_BINARY_MATH_OP(arctan2_rgrad, -math::id(a) / (math::id(a * a + b * b)));

MXNET_BINARY_MATH_OP(rarctan2, math::atan2(b, a));

MXNET_BINARY_MATH_OP(rarctan2_grad, math::id(a) / (math::id(a * a + b * b)));

MXNET_UNARY_MATH_OP_NC(nt, a != DType(0) ? DType(0) : DType(1));

MXNET_UNARY_LOGIC_OP_NC(np_logical_not, !static_cast<bool>(a));

MXNET_BINARY_MATH_OP_NC(ge, a >= b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(gt, a > b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(lt, a < b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(le, a <= b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(eq, a == b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP_NC(ne, a != b ? DType(1) : DType(0));

MXNET_BINARY_LOGIC_OP_NC(np_greater_equal, a >= b ? true : false);

MXNET_BINARY_LOGIC_OP_NC(np_greater, a > b ? true : false);

MXNET_BINARY_LOGIC_OP_NC(np_less, a < b ? true : false);

MXNET_BINARY_LOGIC_OP_NC(np_less_equal, a <= b ? true : false);

MXNET_BINARY_LOGIC_OP_NC(np_equal, a == b ? true : false);

MXNET_BINARY_LOGIC_OP_NC(np_not_equal, a != b ? true : false);

MXNET_BINARY_LOGIC_OP_NC(np_logical_and, a && b ? true : false);

MXNET_BINARY_LOGIC_OP_NC(np_logical_or, a || b ? true : false);

MXNET_BINARY_LOGIC_OP_NC(np_logical_xor, (a || b) && !(a && b) ? true : false);

MXNET_BINARY_MATH_OP(logical_and, a && b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP(logical_or, a || b ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP(logical_xor, (a || b) && !(a && b) ? DType(1) : DType(0));

MXNET_BINARY_MATH_OP(bitwise_and, static_cast<int64_t>(a) & static_cast<int64_t>(b));

MXNET_BINARY_MATH_OP(bitwise_xor, static_cast<int64_t>(a) ^ static_cast<int64_t>(b));

MXNET_BINARY_MATH_OP(bitwise_or, static_cast<int64_t>(a) | static_cast<int64_t>(b));

MXNET_UNARY_MATH_OP(square_root, math::sqrt(a));

MXNET_UNARY_MATH_OP(square_root_grad, 0.5f / math::id(a));
MXNET_UNARY_MATH_OP(reciprocal_square_root, 1.0f / math::sqrt(a));

MXNET_UNARY_MATH_OP(reciprocal_square_root_grad, -0.5f / (math::sqrt(a) * math::id(a)));

MXNET_UNARY_MATH_OP(cube_root, math::cbrt(a));

MXNET_UNARY_MATH_OP(cube_root_grad, 1.0f / (3.0f * math::sqr(a)));

MXNET_UNARY_MATH_OP(reciprocal_cube_root, 1.0f / math::cbrt(a));

MXNET_UNARY_MATH_OP(reciprocal_cube_root_grad, -1.0f / (3.0f * math::cbrt(a) * math::id(a)));

/*! \brief used for generate element of ldexp */
MXNET_BINARY_MATH_OP(ldexp, math::id(a) * math::pow(2.0f, b));

MXNET_BINARY_MATH_OP(ldexp_grad, math::pow(2.0f, b));

MXNET_BINARY_MATH_OP(ldexp_rgrad, math::id(a) * math::pow(2.0f, b) * math::log(2.0f));

MXNET_BINARY_MATH_OP(rldexp, math::id(b) * math::pow(2.0f, a));  // swap a and b if a is scalar.

MXNET_BINARY_MATH_OP(rldexp_grad, math::id(b) * math::pow(2.0f, a) * math::log(2.0f));

/*! \brief used for generate element of round */
MXNET_SIMPLE_UNARY_MATH_OP(round);

/*! \brief used for generate element of ceil */
MXNET_SIMPLE_UNARY_MATH_OP(ceil);

/*! \brief used for generate element of floor */
MXNET_SIMPLE_UNARY_MATH_OP(floor);

/*! \brief used to round towards zero */
MXNET_SIMPLE_UNARY_MATH_OP(trunc);

/*! \brief used to round number to nearest integer */
struct rint : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    auto floor = math::floor(a);
    auto ceil = math::ceil(a);
    auto af = math::id(a);
    return DType((af - floor) <= (ceil - af) ? floor : ceil);
  }
};

/*! \brief used to round number to integer nearest to 0 */
struct fix : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    auto floor = math::floor(a);
    auto ceil = math::ceil(a);
    return DType((floor > 0 ? floor : -floor) < (ceil > 0 ? ceil : -ceil) ? floor : ceil);
  }
};

#pragma GCC diagnostic push
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wbool-compare"
#endif
/*! \brief used to determine whether a number is Not A Number*/
struct isnan : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static bool Map(DType a) {
    return IsNan(a);
  }
};

/*! \brief used to determine whether a number is infinite*/
struct isinf : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static bool Map(DType a) {
    return IsInf(a);
  }
};

/*! \brief used to determine whether a number is finite*/
struct isfinite : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static bool Map(DType a) {
    return !IsNan(a) && !IsInf(a);
  }
};

/*! \brief used to determine whether a number is positive infinity*/
struct isposinf : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static bool Map(DType a) {
    return IsInf(a) && a > 0;
  }
};

/*! \brief used to determine whether a number is negative infinity*/
struct isneginf : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static bool Map(DType a) {
    return IsInf(a) && a < 0;
  }
};
#pragma GCC diagnostic pop

/*! \brief used for generate gradient of MAE loss*/
MXNET_BINARY_MATH_OP_NC(minus_sign, a - b > DType(0) ? DType(1) : -DType(1));

MXNET_BINARY_MATH_OP(rminus, b - a);

MXNET_BINARY_MATH_OP_NC(posone, 1);

MXNET_BINARY_MATH_OP_NC(negone, -1);

MXNET_BINARY_MATH_OP(div_grad, 1.0f / math::id(b));

MXNET_BINARY_MATH_OP(div_rgrad, -math::id(a) / math::sqr(b));

MXNET_BINARY_MATH_OP(rdiv, math::id(b) / math::id(a));

MXNET_BINARY_MATH_OP(rdiv_grad, -math::id(b) / math::sqr(a));

MXNET_BINARY_MATH_OP(copysign, (a >= 0 && b >= 0) || (a < 0 && b < 0) ? a : -a);

MXNET_BINARY_MATH_OP(copysign_grad, (a >= 0 && b >= 0) || (a < 0 && b < 0) ? 1: -1);

MXNET_BINARY_MATH_OP(copysign_rgrad, 0);

MXNET_BINARY_MATH_OP(rcopysign, (b >= 0 && a >= 0) || (b < 0 && a < 0) ? b : -b);

struct mod : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_unsigned<DType>::value, DType>::type
  Map(DType a, DType b) {
    if (b == DType(0)) {
      return DType(0);
    } else if (b < DType(0)) {
      if (a < DType(0)) {
        return DType(-::fmod(-static_cast<double>(a), -static_cast<double>(b)));
      } else {
        return DType(::fmod(static_cast<double>(a), -static_cast<double>(b)) +
                     (::fmod(static_cast<double>(a), -static_cast<double>(b)) != DType(0)
                      ? b : DType(0)));
      }
    } else {
      if (a < DType(0)) {
        return DType(-::fmod(-static_cast<double>(a), static_cast<double>(b)) +
                     (::fmod(-static_cast<double>(a), static_cast<double>(b)) != DType(0)
                      ? b : DType(0)));
      } else {
        return DType(::fmod(static_cast<double>(a), static_cast<double>(b)));
      }
    }
  }
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_unsigned<DType>::value, DType>::type
  Map(DType a, DType b) {
    if (b == DType(0)) {
      return DType(0);
    } else {
      return DType(::fmod(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};

struct mixed_mod {
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return mod::Map(static_cast<mshadow::half::half_t>(a), b);
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return mod::Map(static_cast<float>(a), b);
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return mod::Map(static_cast<double>(a), b);
  }
};

struct mixed_rmod {
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static mshadow::half::half_t Map(DType a, mshadow::half::half_t b) {
    return mod::Map(b, static_cast<mshadow::half::half_t>(a));
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static float Map(DType a, float b) {
    return mod::Map(b, static_cast<float>(a));
  }

  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static double Map(DType a, double b) {
    return mod::Map(b, static_cast<double>(a));
  }
};

struct fmod : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    if (b == DType(0)) {
      return DType(0);
    } else {
        return DType(::fmod(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};

struct rfmod : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    if (a == DType(0)) {
      return DType(0);
    } else  {
      return DType(::fmod(static_cast<double>(b), static_cast<double>(a)));
    }
  }
};

struct mod_grad : public mxnet_op::tunable  {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(0);
  }
};
template<>
MSHADOW_XINLINE double mod_grad::Map<double>(double a, double b) {
  return 1.0;
}
template<>
MSHADOW_XINLINE float mod_grad::Map<float>(float a, float b) {
  return 1.0f;
}

template<>
MSHADOW_XINLINE mshadow::half::half_t mod_grad::Map<mshadow::half::half_t>
                                                   (mshadow::half::half_t a,
                                                    mshadow::half::half_t b) {
  return mshadow::half::half_t(1.0f);
}

struct mod_rgrad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(0);
  }
};
template<>
MSHADOW_XINLINE double mod_rgrad::Map<double>(double a, double b) {
  return -::floor(a/b);
}
template<>
MSHADOW_XINLINE float mod_rgrad::Map<float>(float a, float b) {
  return -::floorf(a/b);
}

template<>
MSHADOW_XINLINE mshadow::half::half_t mod_rgrad::Map<mshadow::half::half_t>
                                                    (mshadow::half::half_t a,
                                                     mshadow::half::half_t b) {
  return mshadow::half::half_t(-::floorf(static_cast<float>(a)/static_cast<float>(b)));
}

struct rmod : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_unsigned<DType>::value, DType>::type
  Map(DType a, DType b) {
    if (a == DType(0)) {
      return DType(0);
    } else if (a < DType(0)) {
      if (b < DType(0)) {
        return DType(-::fmod(-static_cast<double>(b), -static_cast<double>(a)));
      } else {
        return DType(::fmod(static_cast<double>(b), -static_cast<double>(a)) +
                     (::fmod(static_cast<double>(b), -static_cast<double>(a)) != DType(0)
                      ? a : DType(0)));
      }
    } else {
      if (b < DType(0)) {
        return DType(-::fmod(-static_cast<double>(b), static_cast<double>(a)) +
                     (::fmod(-static_cast<double>(b), static_cast<double>(a)) != DType(0)
                      ? a : DType(0)));
      } else {
        return DType(::fmod(static_cast<double>(b), static_cast<double>(a)));
      }
    }
  }
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_unsigned<DType>::value, DType>::type
  Map(DType a, DType b) {
    if (a == DType(0)) {
      return DType(0);
    } else {
      return DType(::fmod(static_cast<double>(b), static_cast<double>(a)));
    }
  }
};

struct rmod_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(0);
  }
};
template<>
MSHADOW_XINLINE double rmod_grad::Map<double>(double a, double b) {
  return -::floor(b/a);
}
template<>
MSHADOW_XINLINE float rmod_grad::Map<float>(float a, float b) {
  return -::floorf(b/a);
}

template<>
MSHADOW_XINLINE mshadow::half::half_t rmod_grad::Map<mshadow::half::half_t>
                                                   (mshadow::half::half_t a,
                                                    mshadow::half::half_t b) {
  return mshadow::half::half_t(-::floorf(static_cast<float>(b/a)));
}

struct clip : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType bound) {
    if (x > bound) {
      return bound;
    } else if (x < -bound) {
      return -bound;
    } else {
      return x;
    }
  }
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType lower_bound, DType upper_bound) {
    if (x > upper_bound) {
      return upper_bound;
    } else if (x < lower_bound) {
      return lower_bound;
    }
    return x;
  }
};

/***** gamma ******/

MXNET_UNARY_MATH_OP(gamma, math::tgamma(a));

struct gamma_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // default implementation using floating precision
    float af(static_cast<float>(a));
    return DType(math::tgamma(af) * special_functions::cephes::psi<float>(af));
  }
};

template<>
MSHADOW_XINLINE double gamma_grad::Map<double>(double a) {
  return math::tgamma(a) * special_functions::cephes::psi<double>(a);
}

/***** gammaln ******/

MXNET_UNARY_MATH_OP(gammaln, math::lgamma(a));

struct gammaln_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // default implementation using floating precision
    return DType(special_functions::cephes::psi<float>(a));
  }
};

template<>
MSHADOW_XINLINE double gammaln_grad::Map<double>(double a) {
  return special_functions::cephes::psi<double>(a);
}

/***** digamma ******/

struct digamma : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // default implementation using floating precision
    return DType(special_functions::cephes::psi<float>(a));
  }
};

template<>
MSHADOW_XINLINE double digamma::Map<double>(double a) {
  return special_functions::cephes::psi<double>(a);
}

/***** trigamma ******/

struct trigamma : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    // default implementation using floating precision
    return DType(special_functions::trigamma<float>(a));
  }
};

template<>
MSHADOW_XINLINE double trigamma::Map<double>(double a) {
  return special_functions::trigamma<double>(a);
}

/* Smooth L1 Loss is a loss specific for R-CNN franchise training
 * Smooth L1 Loss function:
 * f(x) = 0.5 * (sigma * x) ^ 2,     |x| < 1 / sigma^2
 *      = |x| - 0.5 / sigma / sigma, otherwise
 * When sigma = 1, it is equivalent to the Huber loss, evaluated at
 * delta = 1.
 * smooth_l1_loss = w_out * f(w_in * x)
 * with w_in, w_out provided by input_data.
 */
struct smooth_l1_loss : public mxnet_op::tunable {
  // a is x, b is sigma
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    auto bsq = math::sqr(b);
    auto ibsq = 1.0f / bsq;
    auto af = math::id(a);
    if (af > ibsq) {
      return DType(af - 0.5f * ibsq);
    } else if (af < -ibsq) {
      return DType(-af - 0.5f * ibsq);
    } else {
      return DType(0.5f * af * af * bsq);
    }
  }
};  // struct smooth_l1_loss

/* The derivative of smooth l1 loss is
 * f'(x) = sigma^2 * x, |x| < 1 / sigma^2
 *       = sign(x),     otherwise
 */
struct smooth_l1_gradient : public mxnet_op::tunable {
  // a is x, b is sigma2
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    auto bsq = math::sqr(b);
    auto ibsq = 1.0f / bsq;
    auto af = math::id(a);
    if (af > ibsq) {
      return DType(1);
    } else if (af < -ibsq) {
      return DType(-1);
    } else {
      return DType(bsq * af);
    }
  }
};  // struct smooth_l1_derivative

/* Implicti reparameterization gradient for standard x ~ Gamma(\alpha, 1)
 * according to dx/da = -cdf(x;alpha) / pdf(x;alpha)
 */
struct gamma_implicit_grad {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType x) {
    if (x < 0.8f) {
      DType numer = 1;
      DType denom = a;
      DType series1 = numer / denom;
      DType series2 = numer / (denom * denom);
      for (int i = 1; i <= 5; i++) {
        numer *= -x / static_cast<DType>(i);
        denom += 1;
        series1 += numer / denom;
        series2 += numer / (denom * denom);
      }
      DType pow_x_alpha = math::pow(x, a);
      DType gamma_pdf = math::pow(x, a - 1) * math::exp(-x);
      DType gamma_cdf = pow_x_alpha * series1;
      DType gamma_cdf_alpha =
          (math::log(x) - DType(special_functions::cephes::psi<float>(a))) *
              gamma_cdf -
          pow_x_alpha * series2;
      DType result = -gamma_cdf_alpha / gamma_pdf;
      return IsNan(result) ? static_cast<DType>( 0.f ) : static_cast<DType>(result);
    }
    if (a > 8.0f) {
      if (0.9f * a <= x && x <= 1.1f * a) {
        DType numer_1 = 1 + 24 * a * (1 + 12 * a);
        DType numer_2 = 1440 * (a * a) + 6 * x * (53 - 120 * x) -
                        65 * x * x / a + a * (107 + 3600 * x);
        DType denom = 1244160 * (a * a) * (a * a);
        return static_cast<DType>(numer_1 * numer_2 / denom);
      }
      DType denom = math::sqrt(8 * a);
      DType term2 = denom / (a - x);
      DType term3 =
          math::pow(x - a - a * math::log(x / a), static_cast<DType>(-1.5));
      DType term23 = (x < a) ? term2 - term3 : term2 + term3;
      DType term1 = math::log(x / a) * term23 -
                    math::sqrt(2 / a) * (a + x) / ((a - x) * (a - x));
      DType stirling = 1 + 1 / (12 * a) * (1 + 1 / (24 * a));
      DType numer = x * term1;
      return static_cast<DType>(-stirling * numer / denom);
    }
    DType u = math::log(x / a);
    DType v = math::log(a);
    DType coef_uv[3][8] = {
        {0.16009398, -0.094634809, 0.025146376, -0.0030648343, 1, 0.32668115,
         0.10406089, 0.0014179084},
        {0.53487893, 0.1298071, 0.065735949, -0.0015649758, 0.16639465,
         0.020070113, -0.0035938915, -0.00058392623},
        {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777, 0.017050642,
         -0.0021309326, 0.00085092367, -1.5247877e-07},
    };
    DType coef_v[8];
    for (int i = 0; i < 8; i++) {
      coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
    }
    DType p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
    DType q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
    return static_cast<DType>(math::exp(p / q));
  }
};  // gamma_implicit_grad

/*! \brief product reducer */
struct product {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src) { // NOLINT(*)
    dst *= src;
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src, volatile DType& none) { // NOLINT(*)
    Reduce(dst, src);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& dst_residual, volatile DType& src_val, volatile DType& src_residual) { // NOLINT(*)
    Reduce(dst_val, src_val);
  }
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst) {} // NOLINT(*)
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& none) {} // NOLINT(*)
  /*!
  *\brief calculate gradient of redres with respect to redsrc,
  * redres: reduced result, redsrc: one of reduction element
  */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redres / redsrc;
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = 1;
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &none) { // NOLINT(*)
    SetInitValue(initv);
  }
};

MXNET_UNARY_MATH_OP_NC(relu, IsNan(a) || (a > DType(0)) ? a : DType(0));

/*! \brief used for computing gradient of relu operator */
struct relu_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    if (IsNan(a)) {
      return a;
    } else {
      return a > DType(0) ? DType(1) : DType(0);
    }
  }
};

/*! \brief used for computing binary operator maximum */
struct maximum : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    if (IsNan(a)) {
      return a;
    } else {
      return (a > b ? a : b);
    }
  }
};

/*! \brief used for computing binary operator fmax */
struct fmax : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    if (IsNan(b)) {
      return a;
    } else if (IsNan(a)) {
      return b;
    } else {
      return (a > b ? a : b);
    }
  }
};

/*! \brief used for computing binary operator minimum */
struct minimum : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    if (IsNan(a)) {
      return a;
    } else {
      return DType(a < b ? a : b);
    }
  }
};

/*! \brief used for computing binary operator fmin */
struct fmin : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    if (IsNan(b)) {
      return a;
    } else if (IsNan(a)) {
      return b;
    } else {
      return (a < b ? a : b);
    }
  }
};

/*! \brief boolean any/all kernel that determines whether elem is NonZero */
struct NonZero {
  template<typename DType>
  MSHADOW_XINLINE static bool Map(DType a) {
    return (a != DType(0));
  }
};

/*! \brief sum reducer that ignores NaN values in the input */
struct nansum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src) { // NOLINT(*)
    if (IsNan(src)) return;
    dst += src;
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src, volatile DType& residual) { // NOLINT(*)
    if (IsNan(src)) return;
    DType y = src - residual;
    DType t = dst + y;
    residual = (t - dst) - y;
    dst = t;
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& dst_residual, volatile DType& src_val, volatile DType& src_residual) { // NOLINT(*)
    DType t1 = dst_val + src_val;
    DType e = t1 - src_val;
    DType t2 = ((src_val - e) + (dst_val - (t1 - e))) + dst_residual + src_residual;
    dst_val = t1 + t2;
    dst_residual = t2 - (dst_val - t1);
  }
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst) {} // NOLINT(*)
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& residual) {} // NOLINT(*)
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType & initv) { // NOLINT(*)
      initv = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &residual) { // NOLINT(*)
    SetInitValue(initv);
    residual = 0;
  }
};

struct nansum_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return IsNan(a) ? DType(0) : DType(1);
  }
};

/*! \brief product reducer that ignores NaN values in the input */
struct nanprod {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src) { // NOLINT(*)
    if (IsNan(src)) return;
    dst *= src;
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst, volatile DType src, volatile DType& none) { // NOLINT(*)
    Reduce(dst, src);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& dst_residual, volatile DType& src_val, volatile DType& src_residual) { // NOLINT(*)
    Reduce(dst_val, src_val);
  }
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst) {} // NOLINT(*)
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& none) {} // NOLINT(*)
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType & initv) { // NOLINT(*)
    initv = 1;
  }

  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &none) { // NOLINT(*)
    SetInitValue(initv);
  }
};

/*! \brief compute l2 norm */
struct nrm2 {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& sum_of_squares, volatile DType src) { // NOLINT(*)
    sum_of_squares += src * src;
  }
  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& sum_of_squares,  volatile DType src, volatile DType& scale) { // NOLINT(*)
    if (src != 0) {
      DType abs = mshadow_op::abs::Map(src);
      if (scale < abs) {
        sum_of_squares = 1 + sum_of_squares * (scale / abs) * (scale / abs);
        scale = abs;
      } else {
        sum_of_squares = sum_of_squares + (abs / scale) * (abs / scale);
      }
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    dst_val += src_val;
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_ssq, volatile DType& dst_scale, volatile DType& src_ssq, volatile DType& src_scale) { // NOLINT(*)
    if (dst_scale != 0 && dst_scale >= src_scale) {
      dst_ssq = dst_ssq + src_ssq * (src_scale / dst_scale) * (src_scale / dst_scale);
    } else if (src_scale != 0 && dst_scale < src_scale) {
      dst_ssq = src_ssq + dst_ssq * (dst_scale / src_scale) * (dst_scale / src_scale);
      dst_scale = src_scale;
    }
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& sum_of_squares) { // NOLINT(*)
    sum_of_squares = math::sqrt(sum_of_squares);
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& sum_of_squares, volatile DType& scale) { // NOLINT(*)
#pragma GCC diagnostic push
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif
    sum_of_squares = scale * math::sqrt(sum_of_squares);
#pragma GCC diagnostic pop
  }
  /*!
   *\brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redsrc / redres;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &sum_of_squares) { // NOLINT(*)
    sum_of_squares = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &sum_of_squares, DType &scale) { // NOLINT(*)
    SetInitValue(sum_of_squares);
    scale = 0;
  }
};

/*! \brief sum reducer */
struct sum {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& dst,  volatile DType src) { // NOLINT(*)
    dst += src;
  }
  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& dst,  volatile DType src, volatile DType& residual) { // NOLINT(*)
    DType y = src - residual;
    DType t = dst + y;
    residual = (t - dst) - y;
    dst = t;
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& dst_residual, volatile DType& src_val, volatile DType& src_residual) { // NOLINT(*)
    DType t1 = dst_val + src_val;
    DType e = t1 - dst_val;
    DType t2 = ((src_val - e) + (dst_val - (t1 - e))) + dst_residual + src_residual;
    dst_val = t1 + t2;
    dst_residual = t2 - (dst_val - t1);
  }
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst) {} // NOLINT(*)
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& residual) {} // NOLINT(*)
  /*!
   *\brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return 1;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &residual) { // NOLINT(*)
    SetInitValue(initv);
    residual = 0;
  }
};

/*! \brief arg max reducer */
struct argmax {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& dst,  volatile DType src) { // NOLINT(*)
    if (dst.num < src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& dst,  volatile DType src, volatile DType& residual) { // NOLINT(*)
    if (dst.num < src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    if (dst_val.num < src_val.num || (dst_val.num == src_val.num && dst_val.idx > src_val.idx)) {
      dst_val.num = src_val.num;
      dst_val.idx = src_val.idx;
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& dst_residual, volatile DType& src_val, volatile DType& src_residual) { // NOLINT(*)
    if (dst_val.num < src_val.num || (dst_val.num == src_val.num && dst_val.idx > src_val.idx)) {
      dst_val.num = src_val.num;
      dst_val.idx = src_val.idx;
    }
  }
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst) {} // NOLINT(*)
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& residual) {} // NOLINT(*)
  /*!
   *\brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return 1;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv.num = mshadow::red::limits::NegInfValue<decltype(initv.num)>();
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &residual) { // NOLINT(*)
    initv.num = mshadow::red::limits::NegInfValue<decltype(initv.num)>();
  }
};

/*! \brief arg max reducer */
struct argmin {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& dst,  volatile DType src) { // NOLINT(*)
    if (dst.num > src.num) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE static void Reduce(volatile AType& dst,  volatile DType src, volatile DType& residual) { // NOLINT(*)
    if (dst.num > src.num) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    if (dst_val.num > src_val.num) {
      dst_val.num = src_val.num;
      dst_val.idx = src_val.idx;
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& dst_residual, volatile DType& src_val, volatile DType& src_residual) { // NOLINT(*)
    if (dst_val.num > src_val.num) {
      dst_val.num = src_val.num;
      dst_val.idx = src_val.idx;
    }
  }
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst) {} // NOLINT(*)
  /*! \brief finalize reduction */
  template<typename DType>
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& residual) {} // NOLINT(*)
  /*!
   *\brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return 1;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv.num = mshadow::red::limits::PosInfValue<decltype(initv.num)>();
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &residual) { // NOLINT(*)
    initv.num = mshadow::red::limits::PosInfValue<decltype(initv.num)>();
  }
};

struct nanprod_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return IsNan(a) ? DType(0) : b / a;
  }
};

#pragma GCC diagnostic push
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#pragma GCC diagnostic ignored "-Wbool-compare"
#endif

/*! \brief used for computing binary greatest common divisor */
struct gcd : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_integral<DType>::value, DType>::type
  Map(DType a, DType b) {
    // minus cases.
    if (a < 0) {
      a = -a;
    }
    if (b < 0) {
      b = -b;
    }
    // handle zero-valued cases.
    DType c;
    if (a == 0 && b != 0) {
      c = b;
    } else if (b == 0 && a != 0) {
      c = a;
    } else if (a == 0 && b == 0) {
      c = 0;
    } else {
      DType tmp;
      if (a < b) {
        tmp = a;
        a = b;
        b = tmp;
      }
      while (a % b != 0) {
        a = a % b;
        tmp = a;
        a = b;
        b = tmp;
      }
      c = b;
    }
    return c;
  }

  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_integral<DType>::value, DType>::type
  Map(DType a, DType b) {
    return DType(0.0f);
  }
};

/*! \brief used for computing binary lowest common multiple */
struct lcm : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<is_integral<DType>::value, DType>::type
  Map(DType a, DType b) {
    // minus cases.
    if (a < 0) {
      a = -a;
    }
    if (b < 0) {
      b = -b;
    }
    // handle zero-valued cases.
    DType c;
    if (a == 0 || b == 0) {
      c = 0;
    } else {
      DType tmp;
      DType tmp_a = a;
      DType tmp_b = b;
      if (a < b) {
        tmp = a;
        a = b;
        b = tmp;
      }
      while (a % b != 0) {
        a = a % b;
        tmp = a;
        a = b;
        b = tmp;
      }
      c = tmp_a / b * tmp_b;
    }
    return c;
  }
  template<typename DType>
  MSHADOW_XINLINE static typename enable_if<!is_integral<DType>::value, DType>::type
  Map(DType a, DType b) {
    return DType(0.0f);
  }
};
#pragma GCC diagnostic pop

}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MSHADOW_OP_H_
