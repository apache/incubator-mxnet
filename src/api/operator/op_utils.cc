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
 * \file op_utils.cc
 * \brief Utility functions for modification in src/operator
 */

#include "op_utils.h"
#include <mxnet/base.h>
#include "../../operator/numpy/np_percentile_op-inl.h"

namespace mxnet {

std::string MXNetTypeWithBool2String(int dtype) {
  switch (dtype) {
    case mshadow::kFloat32:
      return "float32";
    case mshadow::kFloat64:
      return "float64";
    case mshadow::kFloat16:
      return "float16";
    case mshadow::kUint8:
      return "uint8";
    case mshadow::kInt8:
      return "int8";
    case mshadow::kUint16:
      return "uint16";
    case mshadow::kInt16:
      return "int16";
    case mshadow::kInt32:
      return "int32";
    case mshadow::kInt64:
      return "int64";
    case mshadow::kBool:
      return "bool";
    default:
      LOG(FATAL) << "Unknown type enum " << dtype;
  }
  LOG(FATAL) << "should not reach here ";
  return "";
}

std::string MXNetPercentileType2String(int interpolation) {
  using namespace op;
  switch (interpolation) {
    case percentile_enum::kLinear:
      return "linear";
    case percentile_enum::kLower:
      return "lower";
    case percentile_enum::kHigher:
      return "higher";
    case percentile_enum::kMidpoint:
      return "midpoint";
    case percentile_enum::kNearest:
      return "nearest";
    default:
      LOG(FATAL) << "Unknown type enum " << interpolation;
  }
  LOG(FATAL) << "should not reach here ";
  return "";
}

}  // namespace mxnet
