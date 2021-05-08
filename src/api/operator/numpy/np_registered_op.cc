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
 * \file np_registered_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_registered_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <vector>
#include "../utils.h"

namespace mxnet {

MXNET_REGISTER_GLOBAL("ndarray.invoke")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  int args_size = args.size();
  const nnvm::Op* op = static_cast<nnvm::Op*>(args[0].value().v_handle);
  int num_inputs = args_size - 2;
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 1; i < num_inputs + 1; ++i) {
    ndinputs.push_back(static_cast<NDArray*>(args[i]));
  }
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  int out_type_code = args[args_size - 1].type_code();
  NDArray* out = args[args_size - 1].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;

  auto ndoutputs = Invoke(op, &attrs, num_inputs, ndinputs.data(), &num_outputs, outputs);

  if (out_type_code == kNull) {
    if (num_outputs == 1) {
      *ret = reinterpret_cast<NDArray*>(ndoutputs[0]);
    } else {
      std::vector<ObjectRef> outputs_obj;
      outputs_obj.reserve(num_outputs);
      for (int i = 0; i < num_outputs; ++i) {
        ObjectRef out = NDArrayHandle(ndoutputs[i]);
        outputs_obj.push_back(out);
        delete ndoutputs[i];
      }
      *ret = runtime::ADT(0, outputs_obj.begin(), outputs_obj.end());
    }
  } else {
    *ret = PythonArg(args_size - 1);
  }
});

}  // namespace mxnet
