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
 * Copyright (c) 2017 by Contributors
 * \file mxnet_op.h
 * \brief
 * \author Junyuan Xie
*/
#ifndef MXNET_OPERATOR_MXNET_OP_H_
#define MXNET_OPERATOR_MXNET_OP_H_

#include <dmlc/omp.h>
#include <mxnet/base.h>
#include <mxnet/engine.h>
#include <mxnet/op_attr_types.h>
#include <algorithm>
#include <limits>
#include "./operator_tune.h"
#include "../engine/openmp.h"

#ifdef __CUDACC__
#include "../common/cuda/utils.h"
#endif  // __CUDACC__

namespace mxnet {
namespace op {
namespace mxnet_op {
using namespace mshadow;

#ifdef __CUDA_ARCH__
__constant__ const float PI = 3.14159265358979323846;
#else
const float PI = 3.14159265358979323846;
using std::isnan;
#endif

template<typename xpu>
int get_num_threads(const int N);

#ifdef __CUDACC__
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

inline cudaDeviceProp cuda_get_device_prop() {
  int device;
  CUDA_CALL(cudaGetDevice(&device));
  cudaDeviceProp deviceProp;
  CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
  return deviceProp;
}

/*!
 * \brief Get the number of blocks for cuda kernel given N
 */
inline int cuda_get_num_blocks(const int N) {
  using namespace mshadow::cuda;
  return std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
}

template<>
inline int get_num_threads<gpu>(const int N) {
  using namespace mshadow::cuda;
  return kBaseThreadNum * cuda_get_num_blocks(N);
}

#endif  // __CUDACC__

template<>
inline int get_num_threads<cpu>(const int N) {
  return engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
}

/*! \brief operator request type switch */
#define MXNET_ASSIGN_REQ_SWITCH(req, ReqType, ...)  \
  switch (req) {                                    \
  case kNullOp:                                     \
    break;                                          \
  case kWriteInplace:                               \
  case kWriteTo:                                    \
    {                                               \
      const OpReqType ReqType = kWriteTo;           \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kAddTo:                                      \
    {                                               \
      const OpReqType ReqType = kAddTo;             \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    break;                                          \
  }


/*! \brief operator request type switch */
#define MXNET_REQ_TYPE_SWITCH(req, ReqType, ...)  \
  switch (req) {                                    \
  case kNullOp:                                     \
    {                                               \
      const OpReqType ReqType = kNullOp;            \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kWriteInplace:                               \
  case kWriteTo:                                    \
    {                                               \
      const OpReqType ReqType = kWriteTo;           \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  case kAddTo:                                      \
    {                                               \
      const OpReqType ReqType = kAddTo;             \
      {__VA_ARGS__}                                 \
    }                                               \
    break;                                          \
  default:                                          \
    break;                                          \
  }


#define MXNET_NDIM_SWITCH(NDim, ndim, ...)         \
  if (NDim == 0) {                                 \
  } else if (NDim == 1) {                          \
    const int ndim = 1;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 2) {                          \
    const int ndim = 2;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 3) {                          \
    const int ndim = 3;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 4) {                          \
    const int ndim = 4;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 5) {                          \
    const int ndim = 5;                            \
    {__VA_ARGS__}                                  \
  } else {                                         \
    LOG(FATAL) << "ndim=" << NDim << "too large "; \
  }

#define MXNET_NDIM_SWITCH_EX(NDim, ndim, ...)      \
  if (NDim == 0) {                                 \
  } else if (NDim == 1) {                          \
    const int ndim = 1;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 2) {                          \
    const int ndim = 2;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 3) {                          \
    const int ndim = 3;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 4) {                          \
    const int ndim = 4;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 5) {                          \
    const int ndim = 5;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 6) {                          \
    const int ndim = 6;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 7) {                          \
    const int ndim = 7;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 8) {                          \
    const int ndim = 8;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 9) {                          \
    const int ndim = 9;                            \
    {__VA_ARGS__}                                  \
  } else if (NDim == 10) {                         \
    const int ndim = 10;                           \
    {__VA_ARGS__}                                  \
  }  else {                                        \
    LOG(FATAL) << "ndim=" << NDim << "too large "; \
  }

#define MXNET_NO_INT8_TYPE_SWITCH(type, DType, ...)        \
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      typedef float DType;                                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      typedef double DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
  case mshadow::kBfloat16:                                 \
    {                                                      \
      typedef mshadow::half::half_t DType;                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kUint8:                                    \
    LOG(FATAL) << "This operation does not "               \
                  "support int8 or uint8";                 \
    break;                                                 \
  case mshadow::kInt8:                                     \
    LOG(FATAL) << "This operation does not "               \
                  "support int8 or uint8";                 \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      typedef int32_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      typedef int64_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
  }

#define MXNET_NO_INT16_TYPE_SWITCH(type, DType, ...)        \
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      typedef float DType;                                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      typedef double DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
  case mshadow::kBfloat16:                                 \
    {                                                      \
      typedef mshadow::half::half_t DType;                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kUint8:                                    \
    {                                                      \
      typedef uint8_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt8:                                     \
    {                                                      \
      typedef int8_t DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt16:                                    \
    LOG(FATAL) << "This operation does not "               \
                  "support int16 or uint16";               \
    break;                                                 \
  case mshadow::kUint16:                                   \
    LOG(FATAL) << "This operation does not "               \
                  "support int16 or uint16";               \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      typedef int32_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      typedef int64_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
  }

#define MXNET_NO_BFLOAT16_TYPE_SWITCH(type, DType, ...)     \
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      typedef float DType;                                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      typedef double DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
    {                                                      \
      typedef mshadow::half::half_t DType;                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kBfloat16:                                 \
    LOG(FATAL) << "This operation does not "               \
                  "support bfloat16";                      \
    break;                                                 \
  case mshadow::kInt8:                                     \
    {                                                      \
      typedef int32_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      typedef int32_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      typedef int64_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
  }

#define MXNET_NO_FLOAT16_TYPE_SWITCH(type, DType, ...)     \
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      typedef float DType;                                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      typedef double DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
    LOG(FATAL) << "This operation does not "               \
                  "support float16";                       \
    break;                                                 \
  case mshadow::kUint8:                                    \
    {                                                      \
      typedef uint8_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt8:                                     \
    {                                                      \
      typedef int8_t DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      typedef int32_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      typedef int64_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
  }

template <typename T>
struct AccType {
  using type = T;
};

template <>
struct AccType<mshadow::half::half_t> {
  using type = float;
};

#define MXNET_REAL_ACC_TYPE_SWITCH(type, DType, AType, ...)\
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      typedef float DType;                                 \
      typedef double AType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      typedef double DType;                                \
      typedef double AType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
    {                                                      \
      typedef mshadow::half::half_t DType;                 \
      typedef float AType;                                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kUint8:                                    \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "floating point types not uint8";      \
    }                                                      \
    break;                                                 \
  case mshadow::kInt8:                                     \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "floating point types not int8";       \
    }                                                      \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "floating point types, not int32";     \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "floating point types, not int64";     \
    }                                                      \
    break;                                                 \
  case mshadow::kBool:                                     \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "floating point types, not bool";      \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
  }

#define MXNET_ACC_TYPE_SWITCH(type, DType, AType, ...)\
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      typedef float DType;                                 \
      typedef double AType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      typedef double DType;                                \
      typedef double AType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
    {                                                      \
      typedef mshadow::half::half_t DType;                 \
      typedef float AType;                                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kUint8:                                    \
    {                                                      \
      typedef uint8_t DType;                               \
      typedef uint32_t AType;                              \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt8:                                     \
    {                                                      \
      typedef int8_t DType;                                \
      typedef int32_t AType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      typedef int32_t DType;                               \
      typedef int64_t AType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      typedef int64_t DType;                               \
      typedef int64_t AType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kBool:                                     \
    {                                                      \
      typedef bool DType;                                  \
      typedef int64_t AType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
  }

#define MXNET_INT_TYPE_SWITCH(type, DType, ...)\
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not float32";          \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not float64";          \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not float16";          \
    }                                                      \
    break;                                                 \
  case mshadow::kUint8:                                    \
    {                                                      \
      typedef uint8_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt8:                                     \
    {                                                      \
      typedef int8_t DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      typedef int32_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      typedef int64_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kBool:                                     \
    {                                                      \
      typedef bool DType;                                  \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
  }

#define MXNET_INT32_INT64_TYPE_SWITCH(type, DType, ...)\
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not float32";          \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not float64";          \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not float16";          \
    }                                                      \
    break;                                                 \
  case mshadow::kUint8:                                    \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not uint8";            \
    }                                                      \
    break;                                                 \
  case mshadow::kInt8:                                     \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not int8";             \
    }                                                      \
    break;                                                 \
  case mshadow::kInt32:                                    \
    {                                                      \
      typedef int32_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kInt64:                                    \
    {                                                      \
      typedef int64_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kBool:                                     \
    {                                                      \
      LOG(FATAL) << "This operation only support "         \
                    "integer types, not bool";             \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Unknown type enum " << type;            \
  }

#define MXNET_LOAD_TYPE_SWITCH(type, DType, ...)           \
  switch (type) {                                          \
  case mshadow::kFloat32:                                  \
    {                                                      \
      typedef float DType;                                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat64:                                  \
    {                                                      \
      typedef double DType;                                \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kFloat16:                                  \
    {                                                      \
      typedef mshadow::half::half_t DType;                 \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  case mshadow::kUint8:                                    \
    {                                                      \
      typedef uint8_t DType;                               \
      {__VA_ARGS__}                                        \
    }                                                      \
    break;                                                 \
  default:                                                 \
    LOG(FATAL) << "Invalid loading enum type " << type;    \
  }

/*!
 * \brief assign the val to out according
 * to request in Kernel::Launch
 * \param out the data to be assigned
 * \param req the assignment request
 * \param val the value to be assigned to out
 * \tparam OType output type
 * \tparam VType value type
 */
#define KERNEL_ASSIGN(out, req, val)  \
  {                                   \
    switch (req) {                    \
      case kNullOp:                   \
        break;                        \
      case kWriteTo:                  \
      case kWriteInplace:             \
        (out) = (val);                \
        break;                        \
      case kAddTo:                    \
        (out) += (val);               \
        break;                        \
      default:                        \
        break;                        \
    }                                 \
  }


#define MXNET_ADD_ALL_TYPES \
  .add_enum("float32", mshadow::kFloat32) \
  .add_enum("float64", mshadow::kFloat64) \
  .add_enum("float16", mshadow::kFloat16) \
  .add_enum("bfloat16", mshadow::kBfloat16) \
  .add_enum("uint8", mshadow::kUint8) \
  .add_enum("int8", mshadow::kInt8) \
  .add_enum("int32", mshadow::kInt32) \
  .add_enum("int64", mshadow::kInt64)


#define MXNET_ADD_ALL_TYPES_WITH_BOOL \
  .add_enum("float32", mshadow::kFloat32) \
  .add_enum("float64", mshadow::kFloat64) \
  .add_enum("float16", mshadow::kFloat16) \
  .add_enum("bfloat16", mshadow::kBfloat16) \
  .add_enum("uint8", mshadow::kUint8) \
  .add_enum("int8", mshadow::kInt8) \
  .add_enum("int32", mshadow::kInt32) \
  .add_enum("int64", mshadow::kInt64) \
  .add_enum("bool", mshadow::kBool)


/* \brief Compute flattened index given coordinates and shape. */
template<int ndim>
MSHADOW_XINLINE index_t ravel(const Shape<ndim>& coord, const Shape<ndim>& shape) {
  index_t ret = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i) {
    ret = ret * shape[i] + (shape[i] > coord[i]) * coord[i];
  }
  return ret;
}


/* Compute coordinates from flattened index given shape */
template<int ndim>
MSHADOW_XINLINE Shape<ndim> unravel(const index_t idx, const Shape<ndim>& shape) {
  Shape<ndim> ret;
  #pragma unroll
  for (index_t i = ndim-1, j = idx; i >=0; --i) {
    auto tmp = j / shape[i];
    ret[i] = j - tmp*shape[i];
    j = tmp;
  }
  return ret;
}


/* Compute dot product of two vector */
template<int ndim>
MSHADOW_XINLINE index_t dot(const Shape<ndim>& coord, const Shape<ndim>& stride) {
  index_t ret = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i) {
    ret += coord[i] * stride[i];
  }
  return ret;
}


/* Combining unravel and dot */
template<int ndim>
MSHADOW_XINLINE index_t unravel_dot(const index_t idx, const Shape<ndim>& shape,
  const Shape<ndim>& stride) {
  index_t ret = 0;
  #pragma unroll
  for (index_t i = ndim-1, j = idx; i >=0; --i) {
    auto tmp = j / shape[i];
    ret += (j - tmp*shape[i])*stride[i];
    j = tmp;
  }
  return ret;
}


/* Calculate stride of each dim from shape */
template<int ndim>
MSHADOW_XINLINE Shape<ndim> calc_stride(const Shape<ndim>& shape) {
  Shape<ndim> stride;
  index_t cumprod = 1;
  #pragma unroll
  for (int i = ndim - 1; i >= 0; --i) {
    stride[i] = (shape[i] > 1) ? cumprod : 0;
    cumprod *= shape[i];
  }
  return stride;
}

/* Increment coordinates */
template<int ndim>
MSHADOW_XINLINE bool inc(Shape<ndim>* coord, const Shape<ndim>& shape) {
  ++(*coord)[ndim-1];
  #pragma unroll
  for (int i = ndim - 1; i > 0 && (*coord)[i] >= shape[i]; --i) {
    (*coord)[i] -= shape[i];
    ++(*coord)[i-1];
  }
  return (*coord)[0] < shape[0];
}

/* Increment coordinates and modify index */
template<int ndim>
MSHADOW_XINLINE void inc(Shape<ndim>* coord, const Shape<ndim>& shape,
                         index_t* idx, const Shape<ndim>& stride) {
  ++(*coord)[ndim-1];
  *idx += stride[ndim-1];
  #pragma unroll
  for (int i = ndim - 1; i > 0 && (*coord)[i] >= shape[i]; --i) {
    (*coord)[i] -= shape[i];
    ++(*coord)[i-1];
    *idx = *idx + stride[i-1] - shape[i] * stride[i];
  }
}

/* Increment coordinates and modify index */
template<int ndim>
MSHADOW_XINLINE void inc(Shape<ndim>* coord, const Shape<ndim>& shape,
                         index_t* idx1, const Shape<ndim>& stride1,
                         index_t* idx2, const Shape<ndim>& stride2) {
  ++(*coord)[ndim-1];
  *idx1 += stride1[ndim-1];
  *idx2 += stride2[ndim-1];
  #pragma unroll
  for (int i = ndim - 1; i > 0 && (*coord)[i] >= shape[i]; --i) {
    (*coord)[i] -= shape[i];
    ++(*coord)[i-1];
    *idx1 = *idx1 + stride1[i-1] - shape[i] * stride1[i];
    *idx2 = *idx2 + stride2[i-1] - shape[i] * stride2[i];
  }
}

/*!
 * \brief Simple copy data from one blob to another
 * \param to Destination blob
 * \param from Source blob
 */
template <typename xpu>
MSHADOW_CINLINE void copy(mshadow::Stream<xpu> *s, const TBlob& to, const TBlob& from) {
  CHECK_EQ(from.Size(), to.Size());
  CHECK_EQ(from.dev_mask(), to.dev_mask());
  MSHADOW_TYPE_SWITCH_WITH_BOOL(to.type_flag_, DType, {
    if (to.type_flag_ == from.type_flag_) {
      mshadow::Copy(to.FlatTo1D<xpu, DType>(s), from.FlatTo1D<xpu, DType>(s), s);
    } else {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(from.type_flag_, SrcDType, {
        to.FlatTo1D<xpu, DType>(s) = mshadow::expr::tcast<DType>(from.FlatTo1D<xpu, SrcDType>(s));
      })
    }
  })
}

/*! \brief Binary op backward gradient OP wrapper */
template<typename GRAD_OP>
struct backward_grad {
  /* \brief Backward calc with grad
   * \param a - output grad
   * \param args... - data to grad calculation op (what this is -- input, output, etc. -- varies)
   * \return input grad
   */
  template<typename DType, typename ...Args>
  MSHADOW_XINLINE static DType Map(DType a, Args... args) {
    return DType(a * GRAD_OP::Map(args...));
  }
};

template<typename OP, int req>
struct mixed_type_unary_op {
  typedef OP Operation;

  /*! \brief input is one tensor */
  template<typename OType, typename IType>
  MSHADOW_XINLINE static void Map(index_t i, OType *out, const IType *in) {
    KERNEL_ASSIGN(out[i], req, OP::Map(OType(in[i])));
  }
};

/*! \brief Binary op backward gradient OP wrapper (tuned) */
template<typename GRAD_OP>
struct backward_grad_tuned : public backward_grad<GRAD_OP>, public tunable {
  using backward_grad<GRAD_OP>::Map;
};

/*! \brief Select assignment operation based upon the req value
 * Also useful for mapping mshadow Compute (F<OP>) to Kernel<OP>::Launch
 */
template<typename OP, int req>
struct op_with_req {
  typedef OP Operation;

  /*! \brief input is one tensor */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *in) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i]));
  }

  /*! \brief inputs are two tensors */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *lhs, const DType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief input is tensor and a scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value));
  }

  /*! \brief input is tensor and two scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType *in,
                                  const DType value_1, const DType value_2) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value_1, value_2));
  }

  /*! \brief No inputs (ie fill to constant value) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out) {
    KERNEL_ASSIGN(out[i], req, OP::Map());
  }

  /*! \brief input is single scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(value));
  }

  /*! \brief inputs are two tensors and a scalar value */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out,
                                  const DType *input_1, const DType *input_2, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(input_1[i], input_2[i], value));
  }

  /*! \brief inputs are three tensors (ie backward grad with binary grad function) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out,
                                  const DType *input_1,
                                  const DType *input_2,
                                  const DType *input_3) {
    KERNEL_ASSIGN(out[i], req, OP::Map(input_1[i], input_2[i], input_3[i]));
  }

  /*! \brief input is a tensor and the output is a boolean tensor */
  template<typename DType,
           typename std::enable_if<!std::is_same<DType, bool>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, bool *out, const DType *in) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i]));
  }

  /*! \brief inputs are two tensors with a boolean output tensor */
  template<typename DType,
           typename std::enable_if<!std::is_same<DType, bool>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, bool *out, const DType *lhs, const DType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief input is tensor and two scalar value with a boolean output tensor */
  template<typename DType,
           typename std::enable_if<!std::is_same<DType, bool>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, bool *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value));
  }

  /*! \brief input is two tensors with different type and with a boolean output tensor */
  template<typename LType, typename RType,
           typename std::enable_if<!std::is_same<LType, RType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, bool *out, const LType *lhs, const RType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief inputs are two tensors with a half_t output tensor */
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i,
                                  mshadow::half::half_t *out,
                                  const DType *lhs,
                                  const mshadow::half::half_t *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief inputs are two tensors with a float output tensor */
  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, float *out, const DType *lhs, const float *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief inputs are two tensors with a double output tensor */
  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, double *out, const DType *lhs, const double *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief inputs are two tensors with a half_t output tensor */
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i,
                                  mshadow::half::half_t *out,
                                  const DType *lhs,
                                  const mshadow::half::half_t value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], value));
  }

  /*! \brief inputs are two tensors with a float output tensor */
  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, float *out, const DType *lhs, const float value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], value));
  }

  /*! \brief inputs are two tensors with a double output tensor */
  template<typename DType,
           typename std::enable_if<std::is_same<DType, mshadow::half::half_t>::value ||
                                   std::is_same<DType, float>::value ||
                                   std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, double *out, const DType *lhs, const double value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], value));
  }

  /*! \brief inputs are two tensors with a float output tensor */
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, float *out, const DType *lhs, const DType *rhs) {
    KERNEL_ASSIGN(out[i], req, OP::Map(lhs[i], rhs[i]));
  }

  /*! \brief input is a tensor and a scalar value with a float output tensor */
  template<typename DType,
           typename std::enable_if<std::is_integral<DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t i, float *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], req, OP::Map(in[i], value));
  }
};

template<typename OP, typename xpu>
struct Kernel;

/*!
 * \brief CPU Kernel launcher
 * \tparam OP Operator to launch
 */
template<typename OP>
struct Kernel<OP, cpu> {
  /*!
   * \brief Launch a generic CPU kernel.
   * When using this for a new kernel op, add declaration and tuning objects to
   * operator_tune.cc
   * \tparam Args Varargs type to eventually pass to the OP::Map() function
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the OP::Map() function
   */
  template<typename ...Args>
  inline static bool Launch(mshadow::Stream<cpu> *, const size_t N, Args... args) {
#ifdef _OPENMP
    const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    if (omp_threads < 2) {
      for (size_t i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (index_t i = 0; i < static_cast<index_t>(N); ++i) {
        OP::Map(i, args...);
      }
    }
#else
    for (size_t i = 0; i < N; ++i) {
      OP::Map(i, args...);
    }
#endif
    return true;
  }

  /*!
   * \brief Launch a generic CPU kernel with dynamic schedule. This is recommended
   * for irregular workloads such as spmv.
   * When using this for a new kernel op, add declaration and tuning objects to
   * operator_tune.cc
   * \tparam Args Varargs type to eventually pass to the OP::Map() function
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the OP::Map() function
   */
  template<typename ...Args>
  inline static bool LaunchDynamic(mshadow::Stream<cpu> *, const int64_t N, Args... args) {
#ifdef _OPENMP
    const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount(false);
    if (omp_threads < 2) {
      for (int64_t i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads) schedule(dynamic)
      for (int64_t i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    }
#else
    for (int64_t i = 0; i < N; ++i) {
      OP::Map(i, args...);
    }
#endif
    return true;
  }

  /*!
   * \brief Launch CPU kernel which has OMP tuning data available.
   * When using this for a new kernel op, add declaration and tuning objects to
   * operator_tune.cc
   * \tparam PRIMITIVE_OP The primitive operation to use for tuning
   * \tparam DType Data type
   * \tparam Args Varargs type to eventually pass to the OP::Map() function
   * \param N Number of iterations
   * \param dest Destination pointer (used to infer DType)
   * \param args Varargs to eventually pass to the OP::Map() function
   */
  template<typename PRIMITIVE_OP, typename DType, typename ...Args>
  static void LaunchTuned(mshadow::Stream<cpu> *, const size_t N, Args... args) {
#ifdef _OPENMP
    const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    if (omp_threads < 2 || !tuned_op<PRIMITIVE_OP, DType>::UseOMP(
      N, static_cast<size_t>(omp_threads))) {
      for (size_t i = 0; i < N; ++i) {
        OP::Map(i, args...);
      }
    } else {
      #pragma omp parallel for num_threads(omp_threads)
      for (index_t i = 0; i < static_cast<index_t>(N); ++i) {
        OP::Map(i, args...);
      }
    }
#else
    for (size_t i = 0; i < N; ++i) {
      OP::Map(i, args...);
    }
#endif
  }

  /*!
   * \brief Launch custom-tuned kernel where each thread is set to
   *        operate on a contiguous partition
   * \tparam Args Varargs type to eventually pass to the OP::Map() function
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the UseOMP() and OP::Map() functions
   */
  template<typename ...Args>
  inline static void LaunchEx(mshadow::Stream<cpu> *s, const size_t N, Args... args) {
#ifdef _OPENMP
    const int omp_threads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    if (omp_threads < 2) {
      OP::Map(0, N, args...);
    } else {
      const auto length = (N + omp_threads - 1) / omp_threads;
      #pragma omp parallel for num_threads(omp_threads)
      for (index_t i = 0; i < static_cast<index_t>(N); i += length) {
        OP::Map(i, i + length > N ? N - i : length, args...);
      }
    }
#else
    OP::Map(0, N, args...);
#endif
  }

  /*!
   * \brief Launch a tunable OP with implicitly-supplied data type
   * \tparam DType Data type
   * \tparam T OP type
   * \tparam Args Varargs type to eventually pass to the OP::Map() function
   * \param s Stream (usually null for CPU)
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the OP::Map() function
   * \return Always true
   */
  template<typename DType, typename T = OP, typename ...Args>
  static MSHADOW_CINLINE
  typename std::enable_if<std::is_base_of<tunable, T>::value, bool>::type
  Launch(mshadow::Stream<cpu> *s, const size_t N, DType *dest, Args... args) {
    LaunchTuned<T, DType>(s, N, dest, args...);
    return true;
  }

  /*!
   * \brief Launch a tunable OP wrapper with explicitly-supplied data type (ie op_with_req)
   * \tparam DType Data type
   * \tparam T Wrapper type
   * \tparam Args Varargs type to eventually pass to the OP::Map() function
   * \param s Stream (usually null for CPU)
   * \param N Number of iterations
   * \param args Varargs to eventually pass to the OP::Map() function
   * \return Always true
   */
  template<typename DType, typename T = OP, typename ...Args>
  static MSHADOW_CINLINE
  typename std::enable_if<std::is_base_of<tunable, typename T::Operation>::value, bool>::type
  Launch(mshadow::Stream<cpu> *s, const size_t N, DType *dest, Args... args) {
    LaunchTuned<typename T::Operation, DType>(s, N, dest, args...);
    return true;
  }
};



#ifdef __CUDACC__
template<typename OP, typename ...Args>
__global__ void mxnet_generic_kernel(int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    OP::Map(i, args...);
  }
}

template<typename OP, typename ...Args>
__global__ void mxnet_generic_kernel_ex(int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    OP::Map(i, 1, args...);
  }
}

template<typename OP>
struct Kernel<OP, gpu> {
  /*! \brief Launch GPU kernel */
  template<typename ...Args>
  inline static void Launch(mshadow::Stream<gpu> *s, int N, Args... args) {
    if (0 == N) return;
    using namespace mshadow::cuda;
    int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    mxnet_generic_kernel<OP, Args...>
      <<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
        N, args...);
    MSHADOW_CUDA_POST_KERNEL_CHECK(mxnet_generic_kernel);
  }

  template<typename ...Args>
  inline static void LaunchEx(mshadow::Stream<gpu> *s, const int N, Args... args) {
    if (0 == N) return;
    using namespace mshadow::cuda;
    int ngrid = std::min(kMaxGridNum, (N + kBaseThreadNum - 1) / kBaseThreadNum);
    mxnet_generic_kernel_ex<OP, Args...>
      <<<ngrid, kBaseThreadNum, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
        N, args...);
    MSHADOW_CUDA_POST_KERNEL_CHECK(mxnet_generic_kernel_ex);
  }
};
#endif  // __CUDACC__

/*!
 * \brief Set to immediate scalar value kernel
 * \tparam val Scalar immediate
 */
template<int val>
struct set_to_int : public tunable {
  // mxnet_op version (when used directly with Kernel<>::Launch()) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out) {
    out[i] = DType(val);
  }
  // mshadow_op version (when used with op_with_req<>)
  MSHADOW_XINLINE static int Map() {
    return val;
  }
};

/*!
 * \brief Special-case kernel shortcut for setting to zero and one
 */
using set_zero = set_to_int<0>;
using set_one  = set_to_int<1>;

/*!
 * \brief Set to immediate scalar value kernel
 * \tparam val Scalar immediate
 */
template<bool val>
struct set_to_bool : public tunable {
  // mxnet_op version (when used directly with Kernel<>::Launch()) */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out) {
    out[i] = DType(val);
  }
  // mshadow_op version (when used with op_with_req<>)
  MSHADOW_XINLINE static int Map() {
    return val;
  }
};

/*!
 * \brief Special-case kernel shortcut for setting to true and false
 */
using set_true = set_to_bool<true>;
using set_false = set_to_bool<false>;
}  // namespace mxnet_op

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MXNET_OP_H_
