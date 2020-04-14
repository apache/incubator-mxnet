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
 * \file np_broadcast_reduce_op_value.cc
 * \brief Implementation of the API of functions in
 * src/operator/tensor/np_broadcast_reduce_op_value.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/broadcast_reduce_op.h"
#include "../../../operator/numpy/np_broadcast_reduce_op.h"

namespace mxnet {

inline int String2MXNetProdType(const std::string& s) {
  using namespace op;
  if (s == "float16") {
    return mshadow::kFloat16;
  } else if (s == "float32") {
    return mshadow::kFloat32;
  } else if (s == "float64") {
    return mshadow::kFloat64;
  } else if (s == "int8") {
    return mshadow::kInt8;
  } else if (s == "int32") {
    return mshadow::kInt32;
  } else if (s == "int64") {
    return mshadow::kInt64;
  } else if (s == "bool") {
    return mshadow::kBool;
  } else {
    LOG(FATAL) << "unknown type" << s;
  }
  LOG(FATAL) << "should not reach here";
  return 0;
}

MXNET_REGISTER_API("_npi.broadcast_to")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_broadcast_to");
  nnvm::NodeAttrs attrs;
  op::BroadcastToParam param;
  if (args[1].type_code() == kDLInt) {
    param.shape = TShape(1, args[1].operator int64_t());
  } else {
    param.shape = TShape(args[1].operator ObjectRef());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::BroadcastToParam>(&attrs);

  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.sum")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_sum");
  op::NumpyReduceAxesParam param;
  nnvm::NodeAttrs attrs;
  attrs.op = op;

  // parse axis
  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else {
    if (args[1].type_code() == kDLInt) {
      param.axis = Tuple<int>(1, args[1].operator int64_t());
    } else {
      param.axis = Tuple<int>(args[1].operator ObjectRef());
    }
  }

  // parse dtype
  if (args[2].type_code() == kNull) {
    param.dtype = dmlc::nullopt;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[2].operator std::string());
  }

  // parse keepdims
  if (args[3].type_code() == kNull) {
    param.keepdims = false;
  } else {
    param.keepdims = args[3].operator bool();
  }

  // parse initial
  if (args[4].type_code() == kNull) {
    param.initial = dmlc::nullopt;
  } else {
    param.initial = args[4].operator double();
  }

  attrs.parsed = std::move(param);

  SetAttrDict<op::NumpyReduceAxesParam>(&attrs);

  NDArray* inputs[] = {args[0].operator NDArray*()};
  int num_inputs = 1;

  NDArray* outputs[] = {args[5].operator NDArray*()};
  NDArray** out = (outputs[0] == nullptr) ? nullptr : outputs;
  int num_outputs = (outputs[0] != nullptr);
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, out);

  if (out) {
    *ret = PythonArg(5);
  } else {
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

MXNET_REGISTER_API("_npi.mean")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_mean");
  nnvm::NodeAttrs attrs;
  op::NumpyReduceAxesParam param;
  if (args[1].type_code() == kNull) {
    param.axis = dmlc::optional<mxnet::Tuple<int>>();
  } else {
    param.axis = mxnet::Tuple<int>(args[1].operator ObjectRef());
  }
  if (args[2].type_code() == kNull) {
    param.dtype = dmlc::optional<int>();
  } else {
    param.dtype = String2MXNetTypeWithBool(args[2].operator std::string());
  }

  if (args[3].type_code() == kNull) {
    param.keepdims = false;
  } else {
    param.keepdims = args[3].operator bool();
  }
  param.initial = dmlc::optional<double>();
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::NumpyReduceAxesParam>(&attrs);
  int num_inputs = 1;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  NDArray* out = args[4].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(4);
  } else {
    *ret = ndoutputs[0];
  }
});

MXNET_REGISTER_API("_npi.prod")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_prod");
  nnvm::NodeAttrs attrs;
  op::NumpyReduceAxesParam param;
  if (args[1].type_code() == kNull) {
    param.axis = dmlc::nullopt;
  } else if (args[1].type_code() == kDLInt) {
    param.axis = Tuple<int>(1, args[1].operator int64_t());
  } else {
    param.axis = Tuple<int>(args[1].operator ObjectRef());
  }
  if (args[2].type_code() == kNull) {
    param.dtype = dmlc::nullopt;
  } else {
    param.dtype = String2MXNetProdType(args[2].operator std::string());
  }
  param.keepdims = args[3].operator bool();
  if (args[4].type_code() == kNull) {
    param.initial = dmlc::nullopt;
  } else {
    param.initial = args[4].operator double();
  }
  attrs.op = op;
  attrs.parsed = std::move(param);
  SetAttrDict<op::NumpyReduceAxesParam>(&attrs);
  // inputs
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  // outputs
  NDArray* out = args[5].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(3);
  } else {
    *ret = reinterpret_cast<mxnet::NDArray*>(ndoutputs[0]);
  }
});

}  // namespace mxnet
