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
#include <cfloat>
#include <atomic>
#include "./mxnet_op.h"
#include "./mshadow_op.h"
#include "./tensor/init_op.h"
#include "./operator_tune-inl.h"
#include "./tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

/*!
 * \brief Shared static variables for all OperatorTune data types
 */
std::atomic<bool> OperatorTuneBase::calculated_(false);
bool OperatorTuneBase::verbose_tuning_info_ = false;
double OperatorTuneBase::tuning_weight_scale_ = 0.0;

/*!
 * \brief Instantiate static variables for OperatorTune<DType>, where 'DType' is specified
 */
#define IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(__typ$) \
  template<> bool OperatorTune<__typ$>::initialized_ = false; \
  template<> std::unique_ptr<__typ$[]> OperatorTune<__typ$>::data_set_ = nullptr; \
  template<> volatile tune::TuningMode OperatorTuneByType<__typ$>::tuning_mode_ = tune::kAuto; \
  template<> volatile int OperatorTune<__typ$>::volatile_int_ = 9;  /* arbitrary number */ \
  template<> std::unordered_set<std::string> OperatorTune<__typ$>::operator_names_({}); \
  template<> bool OperatorTune<__typ$>::output_tuning_data_ = false; \
  template<> std::list<void (*)()> *OperatorTune<__typ$>::GetTuningList() { \
    static std::list<void (*)()> ll; \
    return &ll; \
  }

/*!
 * \brief Static variables for different types (ie OperatorTune<float>, OperatorTune<double>, etc.
 */
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(float);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(double);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(mshadow::half::half_t);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(mshadow::bfloat::bf16_t);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(int8_t);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(uint8_t);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(int16_t);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(uint16_t);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(int32_t);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(int64_t);
IMPLEMENT_OPERATOR_TUNE_STATICS_FOR_TYPE(bool);

/*!
 * \brief Init variable used to facilitate registering a tunable operator during
 *        static initialization
 * \tparam OP Operator type
 * \tparam DType Data type
 */
template<typename OP, typename DType>
struct static_init_var {
  static bool init_;
};

/*!
 * \brief Repeat the given macro and associated arguments for each data type,
 *        appending the data type to the end of the arguments
 */
#define MSHADOW_MACRO_FOREACH_TYPE(__macro$, ...) \
  __macro$(__VA_ARGS__, float);                   \
  __macro$(__VA_ARGS__, double);                  \
  __macro$(__VA_ARGS__, mshadow::half::half_t);   \
  __macro$(__VA_ARGS__, mshadow::bfloat::bf16_t); \
  __macro$(__VA_ARGS__, uint8_t);                 \
  __macro$(__VA_ARGS__, int8_t);                  \
  __macro$(__VA_ARGS__, uint16_t);                 \
  __macro$(__VA_ARGS__, int16_t);                  \
  __macro$(__VA_ARGS__, int32_t);                 \
  __macro$(__VA_ARGS__, int64_t);

#define MSHADOW_MACRO_FOREACH_TYPE_WITH_BOOL(__macro$, ...) \
  __macro$(__VA_ARGS__, float);                             \
  __macro$(__VA_ARGS__, double);                            \
  __macro$(__VA_ARGS__, mshadow::half::half_t);             \
  __macro$(__VA_ARGS__, mshadow::bfloat::bf16_t);           \
  __macro$(__VA_ARGS__, uint8_t);                           \
  __macro$(__VA_ARGS__, int8_t);                            \
  __macro$(__VA_ARGS__, uint16_t);                           \
  __macro$(__VA_ARGS__, int16_t);                            \
  __macro$(__VA_ARGS__, int32_t);                           \
  __macro$(__VA_ARGS__, int64_t);                           \
  __macro$(__VA_ARGS__, bool)

#define IMPLEMENT_WORKLOAD_VALUE_FOR_TYPE(__op$, __typ$) \
  namespace mxnet_op { \
  template<> std::vector<float> mxnet::op::mxnet_op::tuned_op<__op$, __typ$>::workload_ = \
    { static_cast<float>(INT_MAX >> 3) }; \
  }  /* namespace mxnet_op */
/*!
 * \brief Implement tuning objects for a forward blank (no arguments) kernel operator
 */
#define _IMPLEMENT_BLANK_WORKLOAD_FWD(__op$, __typ$) \
  IMPLEMENT_WORKLOAD_VALUE_FOR_TYPE(__op$, __typ$); \
  namespace mxnet_op { \
  template<> bool ::mxnet::op::mxnet_op::tuned_op<__op$, __typ$>::UseOMP( \
    size_t N, size_t omp_threads) { \
    return ::mxnet::op::UnaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op<__op$, __typ$>>( \
      N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<__op$, __typ$>::init_ = \
    ::mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>( \
      ::mxnet::op::UnaryOpTune<__typ$>::TuneBlankOperatorEx<__op$>)

/*!
 * \brief Implement tuning objects for a forward unary kernel operator
 */
#define _IMPLEMENT_UNARY_WORKLOAD_FWD(__op$, __typ$) \
  IMPLEMENT_WORKLOAD_VALUE_FOR_TYPE(__op$, __typ$); \
  namespace mxnet_op { \
  template<> bool ::mxnet::op::mxnet_op::tuned_op<__op$, __typ$>::UseOMP( \
    size_t N, size_t omp_threads) { \
    return ::mxnet::op::UnaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op<__op$, __typ$>>( \
      N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<__op$, __typ$>::init_ = \
    ::mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>( \
      ::mxnet::op::UnaryOpTune<__typ$>::TuneUnaryOperator<__op$>)

/*!
 * \brief Implement tuning objects for a backward unary kernel operator
 */
#define _IMPLEMENT_UNARY_WORKLOAD_BWD(__op$, __typ$) \
  IMPLEMENT_WORKLOAD_VALUE_FOR_TYPE(::mxnet::op::mxnet_op::backward_grad_tuned<__op$>, __typ$); \
  namespace mxnet_op { \
  template<> \
  bool ::mxnet::op::mxnet_op::tuned_op<::mxnet::op::mxnet_op::backward_grad_tuned<__op$>, __typ$>::\
    UseOMP(size_t N, size_t omp_threads) { \
    return ::mxnet::op::UnaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op< \
      ::mxnet::op::mxnet_op::backward_grad_tuned<__op$>, __typ$>>(N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<::mxnet::op::mxnet_op::backward_grad_tuned<__op$>, __typ$>:: \
    init_ = ::mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>( \
      ::mxnet::op::UnaryOpTune<__typ$>::TuneUnaryBackwardOperator<__op$>)

/*!
 * \brief Implement tuning objects for a forward binary kernel operator
 */
#define _IMPLEMENT_BINARY_WORKLOAD_FWD(__op$, __typ$) \
  IMPLEMENT_WORKLOAD_VALUE_FOR_TYPE(__op$, __typ$); \
  namespace mxnet_op { \
  template<> bool ::mxnet::op::mxnet_op::tuned_op<__op$, __typ$>::UseOMP( \
    size_t N, size_t omp_threads) { \
    return ::mxnet::op::BinaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op<__op$, __typ$>>( \
      N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<__op$, __typ$>::init_ = \
    ::mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>( \
      ::mxnet::op::BinaryOpTune<__typ$>::TuneBinaryOperator<__op$>)

/*!
 * \brief Implement tuning objects for a backward binary kernel operator
 */
#define _IMPLEMENT_BINARY_WORKLOAD_BWD(__op$, __typ$) \
  IMPLEMENT_WORKLOAD_VALUE_FOR_TYPE(::mxnet::op::mxnet_op::backward_grad_tuned<__op$>, __typ$); \
  namespace mxnet_op { \
  template<> \
    bool ::mxnet::op::mxnet_op::tuned_op< \
      ::mxnet::op::mxnet_op::backward_grad_tuned<__op$>, __typ$>:: \
      UseOMP(size_t N, size_t omp_threads) { \
    return ::mxnet::op::BinaryOpTune<__typ$>::UseOMP<mxnet_op::tuned_op< \
      ::mxnet::op::mxnet_op::backward_grad_tuned<__op$>, __typ$>>(N, omp_threads); \
  }}  /* namespace mxnet_op */ \
  template<> bool static_init_var<::mxnet::op::mxnet_op::backward_grad_tuned<__op$>, \
    __typ$>::init_ = \
    ::mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$>(  \
      ::mxnet::op::BinaryOpTune<__typ$>::TuneBinaryBackwardOperator<__op$>)

/*!
 * \brief Implement tuning objects for a custom forward kernel operator
 */
#define _IMPLEMENT_CUSTOM_WORKLOAD_FWD(__op$, __typ$) \
  IMPLEMENT_WORKLOAD_VALUE_FOR_TYPE(__op$<__typ$>, __typ$); \
  template<> bool static_init_var<__op$<__typ$>, __typ$>::init_ = \
    ::mxnet::op::OperatorTune<__typ$>::ScheduleTune<__op$<__typ$>>(\
      __op$<__typ$>::Tune)

/*!
 * \brief Macros for manually adding new blank, unary and binary operators to the tuning set
 */
#define IMPLEMENT_UNARY_WORKLOAD_FWD(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_UNARY_WORKLOAD_FWD, __op$)

#define IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE_WITH_BOOL(_IMPLEMENT_UNARY_WORKLOAD_FWD, __op$)

#define IMPLEMENT_BLANK_WORKLOAD_FWD(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_BLANK_WORKLOAD_FWD, __op$)

#define IMPLEMENT_BLANK_WORKLOAD_FWD_WITH_BOOL(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE_WITH_BOOL(_IMPLEMENT_BLANK_WORKLOAD_FWD, __op$)

#define IMPLEMENT_UNARY_WORKLOAD_BWD(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_UNARY_WORKLOAD_BWD, __op$)

#define IMPLEMENT_BINARY_WORKLOAD_FWD(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_BINARY_WORKLOAD_FWD, __op$)

#define IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE_WITH_BOOL(_IMPLEMENT_BINARY_WORKLOAD_FWD, __op$)

#define IMPLEMENT_BINARY_WORKLOAD_BWD(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_BINARY_WORKLOAD_BWD, __op$)

#define IMPLEMENT_CUSTOM_WORKLOAD_FWD(__op$) \
  MSHADOW_MACRO_FOREACH_TYPE(_IMPLEMENT_CUSTOM_WORKLOAD_FWD, __op$)

/*!
 * \brief Tuning data and default weights in the case that MXNET_ENABLE_OPERATOR_AUTOTUNE is set
 *        to zero (thus turning off auto-tuning)
 * \note This code can be automatically generated
 *       by setting the environment variable MXNET_OUTPUT_TUNING_DATA to a positive
 *       integer value
 */
OperatorTuneBase::duration_t OperatorTuneBase::omp_overhead_ns_ = 5000;
IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::identity);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::identity_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::negation);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::reciprocal);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::reciprocal_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::sigmoid);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::sigmoid_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::softsign);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::softsign_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::relu);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::relu_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::selu);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::selu_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::gelu);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::tanh);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::tanh_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::softrelu);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::softrelu_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::exp);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::exp);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::expm1);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::log);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::log_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::log1p);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::log1p_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::log2);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::log2_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::log10);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::log10_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::erf);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::erf_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::erfinv);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::erfinv_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::sin);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::sin_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::sinh);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::sinh_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::arcsin);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::arcsin_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::arcsinh);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::arcsinh_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::cos);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::cos_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::cosh);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::cosh_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::arccos);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::arccos_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::arccosh);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::arccosh_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::tan);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::tan_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::arctan);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::arctan_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::arctanh);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::arctanh_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::square);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::square_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::square_root);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::square_root_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::reciprocal_square_root);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::reciprocal_square_root_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::cube_root);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::cube_root_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::reciprocal_cube_root);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::reciprocal_cube_root_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::abs);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::sign);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::sign);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::round);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::floor);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::trunc);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rint);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::fix);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::gamma);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::gamma_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::gammaln);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::gammaln_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::digamma);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::trigamma);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::ceil);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::degrees);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::degrees_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::radians);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::radians_grad);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD(mxnet::op::mshadow_op::nt);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_logical_not);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::bitwise_not);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::isnan);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::isinf);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::isposinf);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::isneginf);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::isfinite);  // NOLINT()
IMPLEMENT_UNARY_WORKLOAD_BWD(mxnet::op::mshadow_op::nt);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::clip);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::clip);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::plus);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::minus);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::mul);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::div);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::true_divide);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::minus_sign);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rminus);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::rdiv);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::plus);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::minus);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::mul);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::div);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::minus_sign);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::rminus);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rdiv);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rtrue_divide);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::div_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::div_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::div_rgrad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::div_rgrad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::rdiv_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::mod);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::fmod);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rfmod);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::mod_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::mod_rgrad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rmod);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::rmod_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::left);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::left);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::right);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::right);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::power);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rpower);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::xelu); // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::elu); // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::power_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::rpower_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::power_rgrad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::copysign);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rcopysign);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::copysign_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::copysign_rgrad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::arctan2);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rarctan2);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::arctan2_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::rarctan2_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::arctan2_rgrad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::xelu_grad); // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::gelu_grad); // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::prelu_grad); // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::elu_grad); // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::maximum);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::fmax);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::minimum);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::fmin);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::hypot);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::hypot_grad_left);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::hypot_grad_left);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::hypot_grad_right);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::hypot_grad_right);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::lt);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::lt);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::le);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::le);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::gt);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::gt);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::ge);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::ge);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::ne);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::ne);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::eq);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::eq);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_equal);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_not_equal);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_greater);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_greater_equal);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_less);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_less_equal);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_logical_and);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_logical_or);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::np_logical_xor);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::logical_and);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::logical_and);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::logical_or);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::logical_or);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::logical_xor);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::logical_xor);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::bitwise_and);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::bitwise_xor);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::bitwise_or);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::smooth_l1_loss);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::smooth_l1_gradient);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::gcd);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mshadow_op::lcm);  // NOLINT()
IMPLEMENT_BLANK_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mxnet_op::set_to_int<0>);  // NOLINT()
IMPLEMENT_BLANK_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mxnet_op::set_to_int<1>);  // NOLINT()
IMPLEMENT_BLANK_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mxnet_op::set_to_bool<false>);  // NOLINT()
IMPLEMENT_BLANK_WORKLOAD_FWD_WITH_BOOL(mxnet::op::mxnet_op::set_to_bool<true>);  // NOLINT()
IMPLEMENT_BLANK_WORKLOAD_FWD(mxnet::op::PopulateFullIdxRspKernel);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::ldexp);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_FWD(mxnet::op::mshadow_op::rldexp);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::ldexp_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::ldexp_rgrad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::rldexp_grad);  // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::posone); // NOLINT()
IMPLEMENT_BINARY_WORKLOAD_BWD(mxnet::op::mshadow_op::negone); // NOLINT()
/*!
 * \brief Tuner objects, *not* automatically generated
 */
#ifdef MXNET_USE_OPERATOR_TUNING
static BinaryOpTune<float>                   binaryOpTuneFloat;
static BinaryOpTune<double>                  binaryOpTuneDouble;
static BinaryOpTune<mshadow::half::half_t>   binaryOpTuneHalf;
static BinaryOpTune<mshadow::bfloat::bf16_t> binaryOpTuneBf16;
static BinaryOpTune<int8_t>                  binaryOpTuneInt8;
static BinaryOpTune<uint8_t>                 binaryOpTuneUInt8;
static BinaryOpTune<int16_t>                 binaryOpTuneInt16;
static BinaryOpTune<uint16_t>                binaryOpTuneUInt16;
static BinaryOpTune<int32_t>                 binaryOpTuneInt32;
static BinaryOpTune<int64_t>                 binaryOpTuneInt64;
#endif  // MXNET_USE_OPERATOR_TUNING
}  // namespace op
}  // namespace mxnet
