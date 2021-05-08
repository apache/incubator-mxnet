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
 * \file cached_op_api.cc
 * \brief The API of function to invoke CachedOp in src/imperative/cached_op.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include <mxnet/runtime/container_ext.h>
#include "../imperative/cached_op.h"
#include "../imperative/cached_op_threadsafe.h"

namespace mxnet {

MXNET_REGISTER_GLOBAL("ndarray.cached_op_invoke")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  CachedOpPtr op_shared = *static_cast<CachedOpPtr*>(args[0].value().v_handle);
  // CachedOp* points to CachedOpThreadSafe object if CreateCachedOpEX
  // was called with thread_safe=true
  CachedOp* op = dynamic_cast<CachedOp*>(op_shared.get());

  int num_inputs = args[1];
  int args_size = args.size();
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 2; i < num_inputs + 2; ++i) {
    ndinputs.push_back(static_cast<mxnet::NDArray*>(args[i]));
  }

  std::vector<NDArray*> ndoutputs;
  ndoutputs.reserve(op->num_outputs());
  if (args[num_inputs + 4].type_code() == kNull) {
    for (int i = 0; i < op->num_outputs(); ++i) ndoutputs.push_back(new NDArray());
  } else {
    int array_size = args_size - num_inputs - 4;
    CHECK_EQ(array_size, op->num_outputs())
        << "CachedOp expects " << op->num_outputs() << " outputs, but "
        << array_size << " was given.";
    for (int i = num_inputs + 4; i < array_size; ++i) {
      ndoutputs.push_back(args[i].operator mxnet::NDArray*());
    }
  }

  int default_dev_type;
  int default_dev_id;
  if (args[num_inputs + 2].type_code() != kNull) {
    default_dev_type = args[num_inputs + 2];
    default_dev_id = args[num_inputs + 3];
  } else {
    const Context &ctx = ndinputs[0]->ctx();
    default_dev_type = ctx.dev_type;
    default_dev_id = ctx.dev_id;
  }

  // construct default context
  Context ctx = Context::Create(static_cast<Context::DeviceType>(default_dev_type),
                                default_dev_id);
  op->Forward(op_shared, ndinputs, ndoutputs, ctx);

  if (op->num_outputs() == 1) {
    *ret = ndoutputs[0];
  } else {
    std::vector<ObjectRef> outputs;
    outputs.reserve(op->num_outputs());
    for (int i = 0; i < op->num_outputs(); ++i) {
      ObjectRef out = NDArrayHandle(ndoutputs[i]);
      outputs.push_back(out);
      delete ndoutputs[i];
    }
    *ret = runtime::ADT(0, outputs.begin(), outputs.end());
  }
});

MXNET_REGISTER_GLOBAL("ndarray.cached_op_create")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(args[0].value().v_handle);
  Object* flags_ptr = static_cast<Object*>(args[1].value().v_handle);
  auto* n = static_cast<const runtime::MapObj*>(flags_ptr);
  int num_flags = static_cast<int>(n->size());
  bool thread_safe = args[2];
  std::vector<std::pair<std::string, std::string> > flags;
  flags.reserve(num_flags);
  for (const auto& kv : *n) {
    flags.emplace_back(std::string(runtime::Downcast<runtime::String>(kv.first)),
                       std::string(runtime::Downcast<runtime::String>(kv.second)));
  }
  mxnet::CachedOpPtr *out;
  if (!thread_safe) {
    out = new CachedOpPtr(new CachedOp(*sym, flags));
  } else {
    out = new CachedOpPtr(new CachedOpThreadSafe(*sym, flags));
  }
  *ret = static_cast<void*>(out);
});

MXNET_REGISTER_GLOBAL("ndarray.cached_op_free")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  CachedOpPtr* g = static_cast<CachedOpPtr*>(args[0].value().v_handle);
  delete g;
});

MXNET_REGISTER_GLOBAL("ndarray.cached_op_get_optimized_symbol")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  auto s = new nnvm::Symbol();
  CachedOpPtr op = *static_cast<CachedOpPtr*>(args[0].value().v_handle);
  *s = op->GetOptimizedSymbol();
  *ret = static_cast<void*>(static_cast<SymbolHandle>(s));
});

MXNET_REGISTER_GLOBAL("ndarray.cached_op_register_op_hook")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  CachedOpHandle handle = static_cast<CachedOpHandle>(args[0].value().v_handle);
  CachedOpMonitorCallback callback = reinterpret_cast<CachedOpMonitorCallback>(
    reinterpret_cast<void (*)(const char *, const char *, void *)>(args[1].value().v_handle));
  bool monitor_all = args[2];
  CachedOpMonitorCallback callback_temp = nullptr;
  std::function<void(const char *, const char *, void*)> clbk;
  if (callback) {
    callback_temp = callback;
    clbk = [callback_temp](const char *name, const char *opr_name,
                           void *handle) {
      callback_temp(name, opr_name, handle);
    };
  } else {
      clbk = nullptr;
  }
  CachedOpPtr op = *static_cast<CachedOpPtr *>(handle);
  op->RegisterOpHook(clbk, monitor_all);
});

}  // namespace mxnet
