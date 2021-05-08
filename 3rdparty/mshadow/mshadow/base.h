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
 *  Copyright (c) 2014 by Contributors
 * \file base.h
 * \brief definitions of base types, operators, macros functions
 *
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_BASE_H_
#define MSHADOW_BASE_H_
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <sstream>
#include <string>

#ifdef _MSC_VER
//! \cond Doxygen_Suppress
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
//! \endcond
#else
#include <inttypes.h>
#endif
// macro defintiions
/*!
 * \brief if this macro is define to be 1,
 * mshadow should compile without any of other libs
 */
#ifndef MSHADOW_STAND_ALONE
#define MSHADOW_STAND_ALONE 0
#endif
/*! \brief whether do padding during allocation */
#ifndef MSHADOW_ALLOC_PAD
#define MSHADOW_ALLOC_PAD true
#endif
/*!
 * \brief
 *  x dimension of data must be bigger pad_size * ratio to be alloced padded memory,
 *  otherwise use tide allocation
 *  for example, if pad_ratio=2, GPU memory alignement size is 32,
 *  then we will only allocate padded memory if x dimension > 64
 *  set it to 0 then we will always allocate padded memory
 */
#ifndef MSHADOW_MIN_PAD_RATIO
  #define MSHADOW_MIN_PAD_RATIO 2
#endif

#if MSHADOW_STAND_ALONE
  #define MSHADOW_USE_CBLAS 0
  #define MSHADOW_USE_MKL   0
  #define MSHADOW_USE_CUDA  0
#endif

/*!
 * \brief force user to use GPU stream during computation
 *  error will be shot when default stream NULL is used
 */
#ifndef MSHADOW_FORCE_STREAM
#define MSHADOW_FORCE_STREAM 1
#endif

/*! \brief use CBLAS for CBLAS */
#ifndef MSHADOW_USE_CBLAS
  #define MSHADOW_USE_CBLAS 0
#endif
/*! \brief use MKL for BLAS */
#ifndef MSHADOW_USE_MKL
  #define MSHADOW_USE_MKL   1
#endif

/*!
 * \brief use CUDA support, must ensure that the cuda include path is correct,
 * or directly compile using nvcc
 */
#ifndef MSHADOW_USE_CUDA
  #define MSHADOW_USE_CUDA   1
#endif

/*!
 * \brief use CUDNN support, must ensure that the cudnn include path is correct
 */
#ifndef MSHADOW_USE_CUDNN
  #define MSHADOW_USE_CUDNN 0
#endif

/*!
 * \brief use CUTENSOR support, must ensure that the cutensor include path is correct
 */
#ifndef MSHADOW_USE_CUTENSOR
  #define MSHADOW_USE_CUTENSOR 0
#endif

/*!
 * \brief use CUSOLVER support
 */
#ifndef MSHADOW_USE_CUSOLVER
  #define MSHADOW_USE_CUSOLVER MSHADOW_USE_CUDA
#endif

/*!
 * \brief seems CUDAARCH is deprecated in future NVCC
 * set this to 1 if you want to use CUDA version smaller than 2.0
 */
#ifndef MSHADOW_OLD_CUDA
#define MSHADOW_OLD_CUDA 0
#endif

/*! \brief whether use SSE */
#ifndef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 1
#endif

/*! \brief whether use F16C instruction set architecture extension */
#ifndef MSHADOW_USE_F16C
  #if defined(_MSC_VER) || defined(__CUDACC__)
    #define MSHADOW_USE_F16C 0
  #elif defined(__clang__) && \
        ((__clang_major__ < 8) || ((__clang_major__ == 8) && (__clang_minor__ < 1)))
    #define MSHADOW_USE_F16C 0
  #else
    #define MSHADOW_USE_F16C 1
  #endif
#endif

/*! \brief whether use NVML to get dynamic info */
#ifndef MSHADOW_USE_NVML
  #define MSHADOW_USE_NVML 0
#endif
// SSE is conflict with cudacc
#ifdef __CUDACC__
  #undef MSHADOW_USE_SSE
  #define MSHADOW_USE_SSE 0
#endif

#if MSHADOW_USE_CBLAS
extern "C" {
    #include <cblas.h>
}
#elif MSHADOW_USE_MKL
  #if MSHADOW_INT64_TENSOR_SIZE == 1
    // Define MKL_INT here to use exactly the same 64bits integer type definitions.
    // If MKL_INT will not be defined here, the mkl header defines it as long long int.
    #define MKL_INT int64_t
    #define MKL_UINT uint64_t
  #endif
  #include <mkl_blas.h>
  #include <mkl_cblas.h>
  #include <mkl_vsl.h>
  #include <mkl_vsl_functions.h>
  #include <mkl_version.h>
#endif

#if MSHADOW_USE_CUDA
  #include <cuda.h>
  #include <cublas_v2.h>
  #include <curand.h>
#endif

#if MSHADOW_USE_CUDNN == 1
  #include <cudnn.h>
#endif

#if MSHADOW_USE_CUTENSOR == 1
  #include <cutensor.h>
#endif

#if MSHADOW_USE_CUSOLVER == 1
  #include <cusolverDn.h>
#endif

#if MSHADOW_USE_NVML
  #include <nvml.h>
#endif

// --------------------------------
// MSHADOW_XINLINE is used for inlining template code for both CUDA and CPU code
#ifdef MSHADOW_XINLINE
  #error "MSHADOW_XINLINE must not be defined"
#endif
#ifdef _MSC_VER
#define MSHADOW_FORCE_INLINE __forceinline
#pragma warning(disable : 4068)
#else
#define MSHADOW_FORCE_INLINE inline __attribute__((always_inline))
#endif
#ifdef __CUDACC__
  #define MSHADOW_XINLINE MSHADOW_FORCE_INLINE __device__ __host__
#else
  #define MSHADOW_XINLINE MSHADOW_FORCE_INLINE
#endif
/*! \brief cpu force inline */
#define MSHADOW_CINLINE MSHADOW_FORCE_INLINE

/*!
 * \brief default data type for tensor string
 *  in code release, change it to default_real_t
 *  during development, change it to empty string so that missing
 *  template arguments can be detected
 */
#ifndef MSHADOW_DEFAULT_DTYPE
#define MSHADOW_DEFAULT_DTYPE = ::mshadow::default_real_t
#endif

/*!
 * \brief DMLC marco for logging
 */
#ifndef MSHADOW_USE_GLOG
#define MSHADOW_USE_GLOG DMLC_USE_GLOG
#endif  // MSHADOW_USE_GLOG

#define MSHADOW_THROW_EXCEPTION noexcept(false)
#define MSHADOW_NO_EXCEPTION  noexcept(true)

#if defined(_MSC_VER)
#define MSHADOW_ALIGNED(x) __declspec(align(x))
#else
#define MSHADOW_ALIGNED(x) __attribute__ ((aligned(x)))
#endif

/*!
 * \brief Protected cuda call in mshadow
 * \param func Expression to call.
 * It checks for CUDA errors after invocation of the expression.
 */
#define MSHADOW_CUDA_CALL(func)                                    \
  {                                                                \
    cudaError_t e = (func);                                        \
    if (e == cudaErrorCudartUnloading) {                           \
      throw dmlc::Error(cudaGetErrorString(e));                    \
    }                                                              \
    CHECK_EQ(e, cudaSuccess)                                       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

/*!
 * \brief Run function and catch error, log unknown error.
 * \param func Expression to call.
 */
#define MSHADOW_CATCH_ERROR(func)                                     \
  {                                                                   \
    try {                                                             \
      (func);                                                         \
    } catch (const dmlc::Error &e) {                                    \
      std::string what = e.what();                                      \
      if (what.find("driver shutting down") == std::string::npos) {     \
        LOG(ERROR) << "Ignore CUDA Error " << what;                     \
      }                                                                 \
    }                                                                   \
  }

#include "./half.h"
#include "./bfloat.h"
#define MSHADOW_HALF_BF_OPERATOR(RTYPE, OP)                                               \
  MSHADOW_XINLINE RTYPE operator OP(mshadow::half::half_t a, mshadow::bfloat::bf16_t b) { \
    return float(a) OP float(b); /* NOLINT(*) */                                          \
  }                                                                                       \
  MSHADOW_XINLINE RTYPE operator OP(mshadow::bfloat::bf16_t a, mshadow::half::half_t b) { \
    return float(a) OP float(b); /* NOLINT(*) */                                          \
  }

/*! \brief overloaded + operator between half_t and bf16_t */
MSHADOW_HALF_BF_OPERATOR(float, +)
/*! \brief overloaded - operator between half_t and bf16_t */
MSHADOW_HALF_BF_OPERATOR(float, -)
/*! \brief overloaded * operator between half_t and bf16_t */
MSHADOW_HALF_BF_OPERATOR(float, *)
/*! \brief overloaded / operator between half_t and bf16_t */
MSHADOW_HALF_BF_OPERATOR(float, /)
/*! \brief overloaded > operator between half_t and bf16_t */
MSHADOW_HALF_BF_OPERATOR(bool, >)
/*! \brief overloaded < operator between half_t and bf16_t */
MSHADOW_HALF_BF_OPERATOR(bool, <)
/*! \brief overloaded >= operator between half_t and bf16_t */
MSHADOW_HALF_BF_OPERATOR(bool, >=)
/*! \brief overloaded <= operator between half_t and bf16_t */
MSHADOW_HALF_BF_OPERATOR(bool, <=)

#include "./logging.h"
/*! \brief namespace for mshadow */
namespace mshadow {
/*! \brief buffer size for each random number generator */
const unsigned kRandBufferSize = 1000000;
/*! \brief pi  */
const float kPi = 3.1415926f;
/*! \brief type that will be used for index */
#if MSHADOW_INT64_TENSOR_SIZE == 1
  typedef int64_t index_t;
#else
  typedef int32_t index_t;
#endif

#ifdef _WIN32
  /*! \brief openmp index for windows */
  typedef int64_t openmp_index_t;
#else
  /*! \brief openmp index for linux */
  typedef index_t openmp_index_t;
#endif


#if (MSHADOW_USE_MKL && MXNET_USE_LAPACK) || MXNET_USE_ILP64_LAPACKE
  // lapack_index_t could be replaced by index_t and removed when all blas library support large tensor
  typedef index_t lapack_index_t;
#else
  typedef int lapack_index_t;
#endif

/*! \brief float point type that will be used in default by mshadow */
typedef float default_real_t;

/*! \brief data type flag */
enum TypeFlag {
  kFloat32 = 0,
  kFloat64 = 1,
  kFloat16 = 2,
  kUint8 = 3,
  kInt32 = 4,
  kInt8  = 5,
  kInt64 = 6,
  kBool = 7,
  kInt16 = 8,
  kUint16 = 9,
  kUint32 = 10,
  kUint64 = 11,
  kBfloat16 = 12
};

template<typename DType>
struct DataType;

template<>
struct DataType<float> {
  static const int kFlag = kFloat32;
  static const int kLanes = 1;
#if MSHADOW_USE_CUDA
#if (CUDA_VERSION >= 8000)
  static const cudaDataType_t kCudaFlag = CUDA_R_32F;
#endif
#if MSHADOW_USE_CUDNN
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_FLOAT;
  typedef float ScaleType;
#endif
#endif
};
template<>
struct DataType<double> {
  static const int kFlag = kFloat64;
  static const int kLanes = 1;
#if MSHADOW_USE_CUDA
#if (CUDA_VERSION >= 8000)
  static const cudaDataType_t kCudaFlag = CUDA_R_64F;
#endif
#if MSHADOW_USE_CUDNN
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_DOUBLE;
  typedef double ScaleType;
#endif
#endif
};
template<>
struct DataType<half::half_t> {
  static const int kFlag = kFloat16;
  static const int kLanes = 1;
#if MSHADOW_USE_CUDA
#if (CUDA_VERSION >= 8000)
  static const cudaDataType_t kCudaFlag = CUDA_R_16F;
#endif
#if MSHADOW_USE_CUDNN
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_HALF;
  typedef float ScaleType;
#endif
#endif
};
template<>
struct DataType<bfloat::bf16_t> {
  static const int kFlag = kBfloat16;
  static const int kLanes = 1;
};
template<>
struct DataType<uint8_t> {
  static const int kFlag = kUint8;
  static const int kLanes = 1;
#if MSHADOW_USE_CUDA
#if (CUDA_VERSION >= 8000)
  static const cudaDataType_t kCudaFlag = CUDA_R_8U;
#endif
#if (MSHADOW_USE_CUDNN == 1 && CUDNN_MAJOR >= 6)
  // no uint8 in cudnn for now
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_INT8;
  typedef uint8_t ScaleType;
#endif
#endif
};
template<>
struct DataType<int8_t> {
  static const int kFlag = kInt8;
  static const int kLanes = 1;
#if MSHADOW_USE_CUDA
#if (CUDA_VERSION >= 8000)
  static const cudaDataType_t kCudaFlag = CUDA_R_8I;
#endif
#if (MSHADOW_USE_CUDNN == 1 && CUDNN_MAJOR >= 6)
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_INT8;
  typedef int8_t ScaleType;
#endif
#endif
};
template<>
struct DataType<uint16_t> {
  static const int kFlag = kUint16;
  static const int kLanes = 1;
};
template<>
struct DataType<int16_t> {
  static const int kFlag = kInt16;
  static const int kLanes = 1;
};
template<>
struct DataType<int32_t> {
  static const int kFlag = kInt32;
  static const int kLanes = 1;
#if MSHADOW_USE_CUDA
#if (CUDA_VERSION >= 8000)
  static const cudaDataType_t kCudaFlag = CUDA_R_32I;
#endif
#if (MSHADOW_USE_CUDNN == 1 && CUDNN_MAJOR >= 6)
  static const cudnnDataType_t kCudnnFlag = CUDNN_DATA_INT32;
  typedef int32_t ScaleType;
#endif
#endif
};
template<>
struct DataType<int64_t> {
  static const int kFlag = kInt64;
  static const int kLanes = 1;
};
template<>
struct DataType<bool> {
  static const int kFlag = kBool;
  static const int kLanes = 1;
};

/*! \brief type enum value for default real type */
const int default_type_flag = DataType<default_real_t>::kFlag;

/*! \brief TypeFlag value for type of indexes */
const int index_type_flag = DataType<lapack_index_t>::kFlag;

/*! layout flag */
enum LayoutFlag {
  kNCHW = 0,
  kNHWC,
  kCHWN,

  kNCW = 1 << 3,
  kNWC,
  kCWN,

  kNCDHW = 1 << 5,
  kNDHWC,
  kCDHWN
};

template<int layout>
struct LayoutType;

template<>
struct LayoutType<kNCHW> {
  static const index_t kNdim = 4;
#if (MSHADOW_USE_CUDA && MSHADOW_USE_CUDNN == 1 && CUDNN_MAJOR >= 4)
  static const cudnnTensorFormat_t kCudnnFlag = CUDNN_TENSOR_NCHW;
#else
  static const int kCudnnFlag = -1;
#endif
};

template<>
struct LayoutType<kNHWC> {
  static const index_t kNdim = 4;
#if (MSHADOW_USE_CUDA && MSHADOW_USE_CUDNN == 1 && CUDNN_MAJOR >= 4)
  static const cudnnTensorFormat_t kCudnnFlag = CUDNN_TENSOR_NHWC;
#else
  static const int kCudnnFlag = -1;
#endif
};

/*! \brief default layout for 4d tensor */
const int default_layout = kNCHW;

template<>
struct LayoutType<kNCDHW> {
  static const index_t kNdim = 5;
#if (MSHADOW_USE_CUDA && MSHADOW_USE_CUDNN == 1 && CUDNN_MAJOR >= 4)
  static const cudnnTensorFormat_t kCudnnFlag = CUDNN_TENSOR_NCHW;
#else
  static const int kCudnnFlag = -1;
#endif
};

template<>
struct LayoutType<kNDHWC> {
  static const index_t kNdim = 5;
#if (MSHADOW_USE_CUDA && MSHADOW_USE_CUDNN == 1 && CUDNN_MAJOR >= 4)
  static const cudnnTensorFormat_t kCudnnFlag = CUDNN_TENSOR_NHWC;
#else
  static const int kCudnnFlag = -1;
#endif
};

/*! \brief default layout for 5d tensor */
const int default_layout_5d = kNCDHW;

/*! \brief namespace for operators */
namespace op {
// binary operator
/*! \brief mul operator */
struct mul{
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a * b;
  }
};
/*! \brief plus operator */
struct plus {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a + b;
  }
};
/*! \brief minus operator */
struct minus {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a - b;
  }
};
/*! \brief divide operator */
struct div {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a / b;
  }
};
/*! \brief get rhs */
struct right {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return b;
  }
};
// unary operator/ function: example
// these operators can be defined by user,
// in the same style as binary and unary operator
// to use, simply write F<op::identity>( src )
/*! \brief identity function that maps a real number to it self */
struct identity{
  /*! \brief map a to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    return a;
  }
};
}  // namespace op
/*! \brief namespace for savers */
namespace sv {
/*! \brief save to saver: = */
struct saveto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a = b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return 1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 0.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::right OPType;
};
/*! \brief save to saver: += */
struct plusto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a += b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return 1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 1.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::plus OPType;
};
/*! \brief minus to saver: -= */
struct minusto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a -= b;
  }
  /*! \brief helper constant to use BLAS, alpha */
  inline static default_real_t AlphaBLAS(void) { return -1.0f; }
  /*! \brief helper constant to use BLAS, beta */
  inline static default_real_t BetaBLAS(void) { return 1.0f; }
  /*! \brief corresponding binary operator type */
  typedef op::minus OPType;
};
/*! \brief multiply to saver: *= */
struct multo {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType &a, DType b) { // NOLINT(*)
    a *= b;
  }
  /*! \brief corresponding binary operator type */
  typedef op::mul OPType;
};
/*! \brief divide to saver: /= */
struct divto {
  /*! \brief save b to a using save method */
  template<typename DType>
  MSHADOW_XINLINE static void Save(DType& a, DType b) { // NOLINT(*)
    a /= b;
  }
  /*! \brief corresponding binary operator type */
  typedef op::div OPType;
};
}  // namespace sv

#ifndef __CUDA_ARCH__
using std::isnan;
using std::isinf;
#endif

/*! \brief
 *  determines if the given floating point
 *  number is not a number */
namespace isnan_typed {
  template<typename DType>
  MSHADOW_XINLINE bool IsNan(volatile DType val) {
    return false;
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile float val) {
    return isnan(val);
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile double val) {
    return isnan(val);
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile long double val) {
    return isnan(val);
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile mshadow::half::half_t val) {
    return (val.half_ & (~MSHADOW_HALF_SIGN_BIT)) > MSHADOW_HALF_EXPONENT_BITS;
  }
}  // namespace isnan_typed

/*! \brief
 *  determines if the given floating point
 *  number is a positive or negative infinity */
namespace isinf_typed {
  template<typename DType>
  MSHADOW_XINLINE bool IsInf(volatile DType val) {
    return false;
  }
  template<>
  MSHADOW_XINLINE bool IsInf(volatile float val) {
    return isinf(val);
  }
  template<>
  MSHADOW_XINLINE bool IsInf(volatile double val) {
    return isinf(val);
  }
  template<>
  MSHADOW_XINLINE bool IsInf(volatile long double val) {
    return isinf(val);
  }
  template<>
  MSHADOW_XINLINE bool IsInf(volatile mshadow::half::half_t val) {
    return (val.half_ & (~MSHADOW_HALF_SIGN_BIT)) == MSHADOW_HALF_EXPONENT_BITS;
  }
}  // namespace isinf_typed

/*! \brief namespace for potential reducer operations */
namespace red {
namespace limits {
/*!
 * \brief minimum value of certain types
 * \tparam DType data type
 */
template<typename DType>
MSHADOW_XINLINE DType MinValue(void);
/*! \brief minimum value of float */
template<>
MSHADOW_XINLINE float MinValue<float>(void) {
  return -FLT_MAX;
}
/*! \brief minimum value of double */
template<>
MSHADOW_XINLINE double MinValue<double>(void) {
  return -DBL_MAX;
}
/*! \brief minimum value of half */
template<>
MSHADOW_XINLINE half::half_t MinValue<half::half_t>(void) {
  return MSHADOW_HALF_MIN;
}
/*! \brief minimum value of bf16 */
template<>
MSHADOW_XINLINE bfloat::bf16_t MinValue<bfloat::bf16_t>(void) {
  return MSHADOW_BF16_MIN;
}
/*! \brief minimum value of uint8_t */
template<>
MSHADOW_XINLINE uint8_t MinValue<uint8_t>(void) {
  return 0;
}
/*! \brief minimum value of int8_t */
template<>
MSHADOW_XINLINE int8_t MinValue<int8_t>(void) {
  return SCHAR_MIN;
}
/*! \brief minimum value of uint16_t */
template<>
MSHADOW_XINLINE uint16_t MinValue<uint16_t>(void) {
  return 0;
}
/*! \brief minimum value of int16_t */
template<>
MSHADOW_XINLINE int16_t MinValue<int16_t>(void) {
  return SHRT_MIN;
}
/*! \brief minimum value of int32_t */
template<>
MSHADOW_XINLINE int MinValue<int32_t>(void) {
  return INT_MIN;
}
/*! \brief minimum value of int64_t */
template<>
MSHADOW_XINLINE int64_t MinValue<int64_t>(void) {
  return LLONG_MIN;
}
/*! \brief minimum value of bool */
template<>
MSHADOW_XINLINE bool MinValue<bool>(void) {
  return false;
}
/*! \brief minimum value of unsigned int */
template<>
MSHADOW_XINLINE unsigned int MinValue<unsigned int>(void) {
  return 0;
}

/*!
 * \brief negative infinity of certain types
 * \tparam DType data type
 */
template<typename DType>
MSHADOW_XINLINE DType NegInfValue(void) {
  return MinValue<DType>();
}
/*! \brief negative infinity value of float */
template<>
MSHADOW_XINLINE float NegInfValue<float>(void) {
  return -HUGE_VALF;
}
/*! \brief negative infinity value of double */
template<>
MSHADOW_XINLINE double NegInfValue<double>(void) {
  return -HUGE_VAL;
}
/*! \brief negative infinity value of float16 */
template<>
MSHADOW_XINLINE half::half_t NegInfValue<half::half_t>(void) {
  return half::half_t::Binary(
      MSHADOW_HALF_SIGN_BIT | MSHADOW_HALF_EXPONENT_BITS);
}

/*!
 * \brief maximum value of certain types
 * \tparam DType data type
 */
template<typename DType>
MSHADOW_XINLINE DType MaxValue(void);
/*! \brief maximum value of float */
template<>
MSHADOW_XINLINE float MaxValue<float>(void) {
  return FLT_MAX;
}
/*! \brief maximum value of double */
template<>
MSHADOW_XINLINE double MaxValue<double>(void) {
  return DBL_MAX;
}
/*! \brief maximum value of half */
template<>
MSHADOW_XINLINE half::half_t MaxValue<half::half_t>(void) {
  return MSHADOW_HALF_MAX;
}
/*! \brief maximum value of bf16 */
template<>
MSHADOW_XINLINE bfloat::bf16_t MaxValue<bfloat::bf16_t>(void) {
  return MSHADOW_BF16_MAX;
}
/*! \brief maximum value of uint8_t */
template<>
MSHADOW_XINLINE uint8_t MaxValue<uint8_t>(void) {
  return UCHAR_MAX;
}
/*! \brief maximum value of int8_t */
template<>
MSHADOW_XINLINE int8_t MaxValue<int8_t>(void) {
  return SCHAR_MAX;
}
/*! \brief maximum value of uint16_t */
template<>
MSHADOW_XINLINE uint16_t MaxValue<uint16_t>(void) {
  return USHRT_MAX;
}
/*! \brief maximum value of int16_t */
template<>
MSHADOW_XINLINE int16_t MaxValue<int16_t>(void) {
  return SHRT_MAX;
}
/*! \brief maximum value of int32_t */
template<>
MSHADOW_XINLINE int MaxValue<int32_t>(void) {
  return INT_MAX;
}
/*! \brief maximum value of int64_t */
template<>
MSHADOW_XINLINE int64_t MaxValue<int64_t>(void) {
  return LLONG_MAX;
}
/*! \brief maximum value of bool */
template<>
MSHADOW_XINLINE bool MaxValue<bool>(void) {
  return true;
}
/*! \brief maximum value of uint32_t */
template<>
MSHADOW_XINLINE uint32_t MaxValue<uint32_t>(void) {
  return std::numeric_limits<uint32_t>::max();
}

/*!
 * \brief positive infinity of certain types
 * \tparam DType data type
 */
template<typename DType>
MSHADOW_XINLINE DType PosInfValue(void) {
  return MaxValue<DType>();
}
/*! \brief positive infinity value of float */
template<>
MSHADOW_XINLINE float PosInfValue<float>(void) {
  return HUGE_VALF;
}
/*! \brief positive infinity value of double */
template<>
MSHADOW_XINLINE double PosInfValue<double>(void) {
  return HUGE_VAL;
}
/*! \brief positive infinity value of float16 */
template<>
MSHADOW_XINLINE half::half_t PosInfValue<half::half_t>(void) {
  return half::half_t::Binary(MSHADOW_HALF_EXPONENT_BITS);
}

}  // namespace limits

/*! \brief sum reducer */
struct sum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    dst += src;
  }
  /*! \brief do stable reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src, volatile DType& residual) { // NOLINT(*)
    DType y = src - residual;
    DType t = dst + y;
    if (isinf_typed::IsInf(t)) {
      residual = 0;
    } else {
      residual = (t - dst) - y;
    }
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
    if (isinf_typed::IsInf(t1)) {
      dst_val = t1;
      dst_residual = 0;
    } else {
      DType e = t1 - dst_val;
      DType t2 = ((src_val - e) + (dst_val - (t1 - e))) + dst_residual + src_residual;
      dst_val = t1 + t2;
      dst_residual = t2 - (dst_val - t1);
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
/*! \brief maximum reducer */
struct maximum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    if (!isnan_typed::IsNan(dst)) {
      if (!(dst >= src)) dst = src;
    }
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src, volatile DType &none) { // NOLINT(*)
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
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& residual) {} // NOLINT(*)
  /*!
   * \brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redres == redsrc ? 1: 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = limits::NegInfValue<DType>();
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &none) { // NOLINT(*)
    SetInitValue(initv);
  }
};
/*! \brief minimum reducer */
struct minimum {
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src) { // NOLINT(*)
    if (!isnan_typed::IsNan(dst)) {
      if (!(dst <= src)) dst = src;
    }
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  MSHADOW_XINLINE static void Reduce(volatile DType& dst,  volatile DType src, volatile DType &none) { // NOLINT(*)
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
  MSHADOW_XINLINE static void Finalize(volatile DType& dst, volatile DType& residual) {} // NOLINT(*)
  /*!
   * \brief calculate gradient of redres with respect to redsrc,
   * redres: reduced result, redsrc: one of reduction element
   */
  template<typename DType>
  MSHADOW_XINLINE static DType PartialGrad(DType redres, DType redsrc) {
    return redres == redsrc ? 1: 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv) { // NOLINT(*)
    initv = limits::PosInfValue<DType>();
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &initv, DType &none) { // NOLINT(*)
    SetInitValue(initv);
  }
};
}  // namespace red

#ifndef __NVCC__
#define MSHADOW_TYPE_SWITCH(type, DType, ...)       \
  switch (type) {                                   \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat16:                           \
    {                                               \
      typedef mshadow::half::half_t DType;          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kBfloat16:                          \
    {                                               \
      typedef mshadow::bfloat::bf16_t DType;        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint8:                             \
    {                                               \
      typedef uint8_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt8:                              \
    {                                               \
      typedef int8_t DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint16:                            \
    {                                               \
      typedef uint16_t DType;                       \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt16:                             \
    {                                               \
      typedef int16_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt32:                             \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt64:                             \
    {                                               \
      typedef int64_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type;     \
  }
#else
#define MSHADOW_TYPE_SWITCH(type, DType, ...)       \
  switch (type) {                                   \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat16:                           \
    {                                               \
      typedef mshadow::half::half_t DType;          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint8:                             \
    {                                               \
      typedef uint8_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt8:                              \
    {                                               \
      typedef int8_t DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint16:                            \
    {                                               \
      typedef uint16_t DType;                       \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt16:                             \
    {                                               \
      typedef int16_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt32:                             \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt64:                             \
    {                                               \
      typedef int64_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type;     \
  }
#endif

#define MSHADOW_SGL_DBL_TYPE_SWITCH(type, DType, ...)  \
  switch (type) {                                      \
  case mshadow::kFloat32:                              \
    {                                                  \
      typedef float DType;                             \
      {__VA_ARGS__}                                    \
    }                                                  \
    break;                                             \
  case mshadow::kFloat64:                              \
    {                                                  \
      typedef double DType;                            \
      {__VA_ARGS__}                                    \
    }                                                  \
    break;                                             \
  default:                                             \
    LOG(FATAL) << "This operation only supports "      \
                  "32-bit and 64-bit floating point";  \
  }

#define MSHADOW_REAL_TYPE_SWITCH(type, DType, ...)  \
  switch (type) {                                   \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat16:                           \
    {                                               \
      typedef mshadow::half::half_t DType;          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint8:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not uint8"; \
    break;                                          \
  case mshadow::kInt8:                              \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not int8";  \
    break;                                          \
  case mshadow::kUint16:                            \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not uint16";\
    break;                                          \
  case mshadow::kInt16:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not int16"; \
    break;                                          \
  case mshadow::kInt32:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types, not int32";\
    break;                                          \
  case mshadow::kInt64:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types, not int64";\
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type;     \
  }

#define MSHADOW_NO_INT16_TYPE_SWITCH(type, DType, ...)  \
  switch (type) {                                   \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType;                          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat16:                           \
    {                                               \
      typedef mshadow::half::half_t DType;          \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint8:                             \
    {                                               \
      typedef uint8_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt8:                              \
    {                                               \
      typedef int8_t DType;                         \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint16:                            \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not uint16";\
    break;                                          \
  case mshadow::kInt16:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not int16"; \
    break;                                          \
  case mshadow::kInt32:                             \
    {                                               \
      typedef int32_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kInt64:                             \
    {                                               \
      typedef int64_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type;     \
  }

#ifndef __NVCC__
#define MSHADOW_REAL_TYPE_SWITCH_EX(type$, DType$, DLargeType$, ...)  \
  switch (type$) {                                  \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType$;                         \
      typedef float DLargeType$;                    \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType$;                        \
      typedef double DLargeType$;                   \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat16:                           \
    {                                               \
      typedef mshadow::half::half_t DType$;         \
      typedef float DLargeType$;                    \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kBfloat16:                          \
    {                                               \
      typedef mshadow::bfloat::bf16_t DType$;       \
      typedef float DLargeType$;                    \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint8:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not uint8"; \
    break;                                          \
  case mshadow::kInt8:                              \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not int8";  \
    break;                                          \
  case mshadow::kUint16:                            \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not uint16";\
    break;                                          \
  case mshadow::kInt16:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not int16"; \
    break;                                          \
  case mshadow::kInt32:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types, not int32";\
    break;                                          \
  case mshadow::kInt64:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types, not int64";\
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type$;    \
  }
#else
#define MSHADOW_REAL_TYPE_SWITCH_EX(type$, DType$, DLargeType$, ...)  \
  switch (type$) {                                  \
  case mshadow::kFloat32:                           \
    {                                               \
      typedef float DType$;                         \
      typedef float DLargeType$;                    \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat64:                           \
    {                                               \
      typedef double DType$;                        \
      typedef double DLargeType$;                   \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kFloat16:                           \
    {                                               \
      typedef mshadow::half::half_t DType$;         \
      typedef float DLargeType$;                    \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kUint8:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not uint8"; \
    break;                                          \
  case mshadow::kInt8:                              \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not int8";  \
    break;                                          \
  case mshadow::kUint16:                            \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not uint16";\
    break;                                          \
  case mshadow::kInt16:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types not int16"; \
    break;                                          \
  case mshadow::kInt32:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types, not int32";\
    break;                                          \
  case mshadow::kInt64:                             \
    LOG(FATAL) << "This operation only support "    \
                  "floating point types, not int64";\
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type$;    \
  }
#endif
#define MSHADOW_LAYOUT_SWITCH(layout, Layout, ...)  \
  switch (layout) {                                 \
  case mshadow::kNCHW:                              \
    {                                               \
      const int Layout = kNCHW;                     \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kNHWC:                              \
    {                                               \
      const int Layout = kNHWC;                     \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kNCDHW:                             \
    {                                               \
      const int Layout = kNCDHW;                    \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case mshadow::kNDHWC:                             \
    {                                               \
      const int Layout = kNDHWC;                    \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown layout enum " << layout; \
  }

/*!
 * \brief Only supports int64 index type for aux_data
 * in NDArray class fow now.
 */
#define MSHADOW_IDX_TYPE_SWITCH(type, DType, ...)   \
  switch (type) {                                   \
  case mshadow::kInt64:                             \
    {                                               \
      typedef int64_t DType;                        \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    LOG(FATAL) << "Unknown type enum " << type;     \
  }

#define MSHADOW_TYPE_SWITCH_WITH_BOOL(type, DType, ...)       \
  switch (type) {                                             \
  case mshadow::kFloat32:                                     \
    {                                                         \
      typedef float DType;                                    \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kFloat64:                                     \
    {                                                         \
      typedef double DType;                                   \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kFloat16:                                     \
    {                                                         \
      typedef mshadow::half::half_t DType;                    \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kBfloat16:                                    \
    {                                                         \
      typedef mshadow::bfloat::bf16_t DType;                  \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kUint8:                                       \
    {                                                         \
      typedef uint8_t DType;                                  \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kInt8:                                        \
    {                                                         \
      typedef int8_t DType;                                   \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kUint16:                                      \
    {                                                         \
      typedef uint16_t DType;                                 \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kInt16:                                       \
    {                                                         \
      typedef int16_t DType;                                  \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kInt32:                                       \
    {                                                         \
      typedef int32_t DType;                                  \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kInt64:                                       \
    {                                                         \
      typedef int64_t DType;                                  \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  case mshadow::kBool:                                        \
    {                                                         \
      typedef bool DType;                                     \
      {__VA_ARGS__}                                           \
    }                                                         \
    break;                                                    \
  default:                                                    \
    LOG(FATAL) << "Unknown type enum " << type;               \
  }

/*! \brief get data type size from type enum */
inline size_t mshadow_sizeof(int type) {
  int size = 0;
  MSHADOW_TYPE_SWITCH_WITH_BOOL(type, DType, size = sizeof(DType););
  return size;
}

/*/ \brief get string with the type name from type enum */
inline std::string dtype_string(const int dtype) {
  switch (dtype) {
    case mshadow::kFloat32:
      return "float";
    case mshadow::kFloat64:
      return "double";
    case mshadow::kFloat16:
      return "half";
    case mshadow::kUint8:
      return "unsigned char";
    case mshadow::kInt8:
      return "char";
    case mshadow::kUint16:
      return "unsigned short";
    case mshadow::kInt16:
      return "short";
    case mshadow::kInt32:
      return "int";
    case mshadow::kInt64:
      return "long long";
    case mshadow::kBool:
      return "bool";
    default:
      LOG(FATAL) << "Unknown type enum " << dtype;
  }
  return "unknown";
}

}  // namespace mshadow
#endif  // MSHADOW_BASE_H_
