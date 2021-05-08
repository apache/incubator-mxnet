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
 * \file np_cast.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_cast.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/amp_cast.h"
#include "../../../operator/tensor/elemwise_unary_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.amp_cast")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npi_amp_cast");
  op::AMPCastParam param;
  // dtype
  if (args[1].type_code() == kNull) {
    param.dtype = mxnet::common::GetDefaultDtype();
  } else {
    param.dtype = String2MXNetTypeWithBool(args[1].operator std::string());
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::AMPCastParam>(&attrs);
  // inputs
  NDArray* inputs[] = {args[0].operator NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.cast")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  nnvm::NodeAttrs attrs;
  const nnvm::Op* op = Op::Get("_npi_cast");
  op::CastParam param;
  // dtype
  if (args[1].type_code() == kNull) {
    param.dtype = mxnet::common::GetDefaultDtype();
  } else {
    param.dtype = String2MXNetTypeWithBool(args[1].operator std::string());
  }
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::CastParam>(&attrs);
  // inputs
  NDArray* inputs[] = {args[0].operator NDArray*()};
  int num_inputs = 1;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet
