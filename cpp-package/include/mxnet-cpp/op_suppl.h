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
*  Copyright (c) 2016 by Contributors
* \file op_suppl.h
* \brief A supplement and amendment of the operators from op.h
* \author Zhang Chen, zhubuntu, Xin Li
*/

#ifndef MXNET_CPP_OP_SUPPL_H_
#define MXNET_CPP_OP_SUPPL_H_

#include <cassert>
#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/shape.h"
#include "mxnet-cpp/operator.h"
#include "mxnet-cpp/MxNetCpp.h"

namespace mxnet {
namespace cpp {

inline Symbol _Plus(Symbol lhs, Symbol rhs) {
  return Operator("_Plus")(lhs, rhs)
           .CreateSymbol();
}
inline Symbol _Mul(Symbol lhs, Symbol rhs) {
  return Operator("_Mul")(lhs, rhs)
           .CreateSymbol();
}
inline Symbol _Minus(Symbol lhs, Symbol rhs) {
  return Operator("_Minus")(lhs, rhs)
           .CreateSymbol();
}
inline Symbol _Div(Symbol lhs, Symbol rhs) {
  return Operator("_Div")(lhs, rhs)
           .CreateSymbol();
}
inline Symbol _Mod(Symbol lhs, Symbol rhs) {
  return Operator("_Mod")(lhs, rhs)
           .CreateSymbol();
}
inline Symbol _Power(Symbol lhs, Symbol rhs) {
  return Operator("_Power")(lhs, rhs)
           .CreateSymbol();
}
inline Symbol _Maximum(Symbol lhs, Symbol rhs) {
  return Operator("_Maximum")(lhs, rhs)
           .CreateSymbol();
}
inline Symbol _Minimum(Symbol lhs, Symbol rhs) {
  return Operator("_Minimum")(lhs, rhs)
           .CreateSymbol();
}
inline Symbol _PlusScalar(Symbol lhs, mx_float scalar) {
  return Operator("_PlusScalar")(lhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _MinusScalar(Symbol lhs, mx_float scalar) {
  return Operator("_MinusScalar")(lhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _RMinusScalar(mx_float scalar, Symbol rhs) {
  return Operator("_RMinusScalar")(rhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _MulScalar(Symbol lhs, mx_float scalar) {
  return Operator("_MulScalar")(lhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _DivScalar(Symbol lhs, mx_float scalar) {
  return Operator("_DivScalar")(lhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _RDivScalar(mx_float scalar, Symbol rhs) {
  return Operator("_RDivScalar")(rhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _ModScalar(Symbol lhs, mx_float scalar) {
  return Operator("_ModScalar")(lhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _RModScalar(mx_float scalar, Symbol rhs) {
  return Operator("_RModScalar")(rhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _PowerScalar(Symbol lhs, mx_float scalar) {
  return Operator("_PowerScalar")(lhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _RPowerScalar(mx_float scalar, Symbol rhs) {
  return Operator("_RPowerScalar")(rhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _MaximumScalar(Symbol lhs, mx_float scalar) {
  return Operator("_MaximumScalar")(lhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
inline Symbol _MinimumScalar(Symbol lhs, mx_float scalar) {
  return Operator("_MinimumScalar")(lhs)
           .SetParam("scalar", scalar)
           .CreateSymbol();
}
// TODO(zhangcheng-qinyinghua)
//  make crop function run in op.h
//  This function is due to [zhubuntu](https://github.com/zhubuntu)
inline Symbol Crop(const std::string& symbol_name,
    int num_args,
    Symbol data,
    Symbol crop_like,
    Shape offset = Shape(0, 0),
    Shape h_w = Shape(0, 0),
    bool center_crop = false) {
  return Operator("Crop")
    .SetParam("num_args", num_args)
    .SetParam("offset", offset)
    .SetParam("h_w", h_w)
    .SetParam("center_crop", center_crop)
    .SetInput("arg0", data)
    .SetInput("arg1", crop_like)
    .CreateSymbol(symbol_name);
}


/*!
 * \brief Apply activation function to input.
 *        Softmax Activation is only available with CUDNN on GPUand will be
 *        computed at each location across channel if input is 4D.
 * \param symbol_name name of the resulting symbol.
 * \param data Input data to activation function.
 * \param act_type Activation function to be applied.
 * \return new symbol
 */
inline Symbol Activation(const std::string& symbol_name,
                         Symbol data,
                         const std::string& act_type) {
  assert(act_type == "relu" ||
         act_type == "sigmoid" ||
         act_type == "softrelu" ||
         act_type == "tanh");
  return Operator("Activation")
           .SetParam("act_type", act_type.c_str())
           .SetInput("data", data)
           .CreateSymbol(symbol_name);
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_OP_SUPPL_H_

