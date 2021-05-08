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
 * \file c_api_ndarray.cc
 * \brief C API of mxnet
 */

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/imperative.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <string>
#include "./c_api_common.h"
#include "../common/utils.h"
#include "../common/exec_utils.h"
#include "../imperative/imperative_utils.h"
#include "../imperative/cached_op.h"
#include "../imperative/cached_op_threadsafe.h"
#include "../profiler/profiler.h"

using namespace mxnet;

void SetNDInputsOutputs(const nnvm::Op* op,
                        std::vector<NDArray*>* ndinputs,
                        std::vector<NDArray*>* ndoutputs,
                        int num_inputs,
                        const NDArrayHandle *inputs,
                        int *num_outputs,
                        int infered_num_outputs,
                        int num_visible_outputs,
                        NDArrayHandle **outputs) {
  NDArray** out_array = *reinterpret_cast<NDArray***>(outputs);

  ndinputs->clear();
  ndinputs->reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    NDArray* inp = reinterpret_cast<NDArray*>(inputs[i]);
    if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
      if (shape_is_known(inp->shape())) {  // Shape may be unknown after dynamic shape operators
        CHECK_LT(inp->shape().Size(), (int64_t{1} << 31) - 1) <<
          "[SetNDInputsOutputs] Size of tensor you are trying to allocate is larger than "
          "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
      }
    }
    ndinputs->emplace_back(inp);
  }

  ndoutputs->clear();
  ndoutputs->reserve(infered_num_outputs);
  if (out_array == nullptr) {
    for (int i = 0; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
    *num_outputs = num_visible_outputs;
  } else {
    CHECK(*num_outputs == infered_num_outputs || *num_outputs == num_visible_outputs)
      << "Operator expects " << infered_num_outputs << " (all) or "
      << num_visible_outputs << " (visible only) outputs, but got "
      << *num_outputs << " instead.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs->emplace_back(out_array[i]);
    }
    for (int i = *num_outputs; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
  }
}

void MXImperativeInvokeImpl(AtomicSymbolCreator creator,
                            int num_inputs,
                            NDArrayHandle *inputs,
                            int *num_outputs,
                            NDArrayHandle **outputs,
                            int num_params,
                            const char **param_keys,
                            const char **param_vals) {
  const nnvm::Op* op = static_cast<nnvm::Op*>(creator);
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();

  nnvm::NodeAttrs attrs = imperative::ParseAttrs(op, num_inputs, num_params,
                                                 param_keys, param_vals);
  attrs.dict["__profiler_scope__"] = profiler::ProfilerScope::Get()->GetCurrentProfilerScope();
  if (attrs.op) {
    attrs.name = attrs.op->name;
  }

  int infered_num_outputs;
  int num_visible_outputs;
  imperative::SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<NDArray*> ndinputs, ndoutputs;
  SetNDInputsOutputs(op, &ndinputs, &ndoutputs, num_inputs, inputs,
      num_outputs, infered_num_outputs, num_visible_outputs, outputs);

  if (Imperative::Get()->is_deferred_compute()) {
    Imperative::Get()->RecordDeferredCompute(std::move(attrs), ndinputs, ndoutputs);
  } else {
    for (NDArray* input : ndinputs) {
      Imperative::DCInfo::Compute(*input);
    }
    auto state = Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);
    if (Imperative::Get()->is_recording()) {
      Imperative::Get()->RecordOp(std::move(attrs), ndinputs, ndoutputs, state);
    }
  }

  for (int i = *num_outputs; i < infered_num_outputs; ++i) delete ndoutputs[i];

  if (*outputs == nullptr) {
    ret->ret_handles.clear();
    ret->ret_handles.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) ret->ret_handles.push_back(ndoutputs[i]);
    *outputs = reinterpret_cast<NDArrayHandle*>(dmlc::BeginPtr(ret->ret_handles));
  }
}

int MXImperativeInvoke(AtomicSymbolCreator creator,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       int num_params,
                       const char **param_keys,
                       const char **param_vals,
                       const int **out_stypes) {  // outputs storage types
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
                         num_params, param_keys, param_vals);
  if (out_stypes != nullptr) {
    NDArray** out_array = *reinterpret_cast<NDArray***>(outputs);
    ret->out_types.clear();
    ret->out_types.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) {
      ret->out_types.emplace_back(out_array[i]->storage_type());
    }
    *out_stypes = dmlc::BeginPtr(ret->out_types);
  }
  API_END();
}

int MXCreateCachedOp(SymbolHandle handle,
                     int num_flags,
                     const char** keys,
                     const char** vals,
                     CachedOpHandle *out,
                     bool thread_safe) {
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(handle);
  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > flags;
  flags.reserve(num_flags);
  for (int i = 0; i < num_flags; ++i) {
    flags.emplace_back(keys[i], vals[i]);
  }
  if (!thread_safe) {
    *out = new CachedOpPtr(new CachedOp(*sym, flags));
  } else {
    *out = new CachedOpPtr(new CachedOpThreadSafe(*sym, flags));
  }
  API_END();
}

int MXFreeCachedOp(CachedOpHandle handle) {
  CachedOpPtr* g = static_cast<CachedOpPtr*>(handle);
  API_BEGIN();
  delete g;
  API_END();
}

/*!
 * \brief get optimized graph from the cached op
 */
int MXCachedOpGetOptimizedSymbol(CachedOpHandle handle,
                                 SymbolHandle *out) {
  auto s = new nnvm::Symbol();
  API_BEGIN();
  CachedOpPtr op = *static_cast<CachedOpPtr*>(handle);
  *s = op->GetOptimizedSymbol();
  *out = s;
  API_END_HANDLE_ERROR(delete s);
}

int MXInvokeCachedOp(CachedOpHandle handle,
                     int num_inputs,
                     NDArrayHandle *inputs,
                     int default_dev_type,
                     int default_dev_id,
                     int *num_outputs,
                     NDArrayHandle **outputs,
                     const int **out_stypes) {  // outputs storage types
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();

  API_BEGIN();
  CachedOpPtr op_shared = *static_cast<CachedOpPtr*>(handle);
  // CachedOp* points to CachedOpThreadSafe object if CreateCachedOpEX
  // was called with thread_safe=true
  CachedOp* op = dynamic_cast<CachedOp*>(op_shared.get());
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.push_back(reinterpret_cast<NDArray*>(inputs[i]));
  }

  std::vector<NDArray*> ndoutputs;
  ndoutputs.reserve(op->num_outputs());
  if (*outputs == nullptr) {
    *num_outputs = op->num_outputs();
    for (int i = 0; i < *num_outputs; ++i) ndoutputs.push_back(new NDArray());
  } else {
    CHECK_EQ(*num_outputs, op->num_outputs())
        << "CachedOp expects " << op->num_outputs() << " outputs, but "
        << *num_outputs << " was given.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs.push_back(reinterpret_cast<NDArray*>((*outputs)[i]));
    }
  }
  // construct default context
  Context ctx = Context::Create(static_cast<Context::DeviceType>(default_dev_type),
                                default_dev_id);
  op->Forward(op_shared, ndinputs, ndoutputs, ctx);

  if (*outputs == nullptr) {
    ret->ret_handles.clear();
    ret->ret_handles.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) {
      ret->ret_handles.push_back(ndoutputs[i]);
    }
    *outputs = dmlc::BeginPtr(ret->ret_handles);
  }
  if (out_stypes != nullptr) {
    NDArray** out_array = reinterpret_cast<NDArray**>(*outputs);
    ret->out_types.clear();
    ret->out_types.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) {
      ret->out_types.emplace_back(out_array[i]->storage_type());
    }
    *out_stypes = dmlc::BeginPtr(ret->out_types);
  }

  API_END();
}

int MXAutogradIsTraining(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_training();
  API_END();
}

int MXAutogradSetIsTraining(int is_training, int* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_training(static_cast<bool>(is_training));
  API_END();
}

int MXAutogradIsRecording(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_recording();
  API_END();
}

int MXAutogradSetIsRecording(int is_recording, int* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_recording(static_cast<bool>(is_recording));
  API_END();
}

int MXIsNumpyShape(int* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_np_shape();
  API_END();
}

int MXSetIsNumpyShape(int is_np_shape, int* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_np_shape(is_np_shape);
  API_END();
}

int MXIsNumpyDefaultDtype(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_np_default_dtype();
  API_END();
}

int MXSetIsNumpyDefaultDtype(bool default_dtype, bool* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_np_default_dtype(default_dtype);
  API_END();
}

int MXAutogradMarkVariables(uint32_t num_var,
                            NDArrayHandle *var_handles,
                            uint32_t *reqs_array,
                            NDArrayHandle *grad_handles) {
  API_BEGIN();
  std::vector<NDArray*> variables, gradients;
  std::vector<uint32_t> grad_reqs;
  variables.reserve(num_var);
  gradients.reserve(num_var);
  grad_reqs.reserve(num_var);
  for (uint32_t i = 0; i < num_var; ++i) {
    variables.emplace_back(static_cast<NDArray*>(var_handles[i]));
    gradients.emplace_back(static_cast<NDArray*>(grad_handles[i]));
    grad_reqs.emplace_back(reqs_array[i]);
  }
  Imperative::Get()->MarkVariables(variables, grad_reqs, gradients);
  API_END();
}

int MXAutogradComputeGradient(uint32_t num_output,
                              NDArrayHandle *output_handles) {
  return MXAutogradBackward(num_output, output_handles, nullptr, 0);
}

int MXAutogradBackward(uint32_t num_output,
                       NDArrayHandle *output_handles,
                       NDArrayHandle *ograd_handles,
                       int retain_graph) {
  return MXAutogradBackwardEx(num_output, output_handles, ograd_handles,
                              0, nullptr, retain_graph, false, true,
                              nullptr, nullptr);
}

int MXAutogradBackwardEx(uint32_t num_output,
                         NDArrayHandle *output_handles,
                         NDArrayHandle *ograd_handles,
                         uint32_t num_variables,
                         NDArrayHandle *var_handles,
                         int retain_graph,
                         int create_graph,
                         int is_train,
                         NDArrayHandle **grad_handles,
                         int **grad_stypes) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();

  std::vector<NDArray*> outputs, ograds, variables;
  outputs.reserve(num_output);
  for (uint32_t i = 0; i < num_output; ++i) {
    outputs.emplace_back(reinterpret_cast<NDArray*>(output_handles[i]));
  }

  ograds.reserve(num_output);
  for (uint32_t i = 0; i < num_output; ++i) {
    if (ograd_handles != nullptr) {
      ograds.emplace_back(reinterpret_cast<NDArray*>(ograd_handles[i]));
    } else {
      ograds.emplace_back(nullptr);
    }
  }

  variables.reserve(num_variables);
  for (uint32_t i = 0; i < num_variables; ++i) {
    variables.emplace_back(reinterpret_cast<NDArray*>(var_handles[i]));
  }

  auto grads = Imperative::Get()->Backward(outputs, ograds, variables, is_train,
                                                  retain_graph, create_graph);
  if (num_variables != 0) {
    ret->ret_handles.clear();
    ret->out_types.clear();
    ret->ret_handles.reserve(grads.size());
    ret->out_types.reserve(grads.size());
    for (const auto& i : grads) {
      ret->ret_handles.push_back(i);
      ret->out_types.push_back(i->storage_type());
    }
    *grad_handles = dmlc::BeginPtr(ret->ret_handles);
    *grad_stypes = dmlc::BeginPtr(ret->out_types);
  }
  API_END();
}

int MXAutogradGetSymbol(NDArrayHandle handle, SymbolHandle *out) {
  API_BEGIN();
  NDArray *head = reinterpret_cast<NDArray*>(handle);
  auto sym = new nnvm::Symbol(head->get_autograd_symbol());
  *out = reinterpret_cast<SymbolHandle>(sym);
  API_END();
}

int MXCachedOpRegisterOpHook(CachedOpHandle handle,
                             CachedOpMonitorCallback callback,
                             bool monitor_all) {
  API_BEGIN();
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
  API_END();
}

int MXNDArrayIsDeferredCompute(int *curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_deferred_compute();
  API_END();
}

int MXNDArraySetIsDeferredCompute(int deferred_compute, int *prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_deferred_compute(static_cast<bool>(deferred_compute));
  API_END();
}

int MXNDArraySetDeferredComputeVariable(NDArrayHandle *arrays, SymbolHandle *variables, int num) {
  API_BEGIN();
  Imperative::Get()->SetDeferredComputeVariable(arrays, variables, num);
  API_END();
}

int MXNDArrayGetDeferredComputeSymbol(NDArrayHandle *output_handles, int num_outputs,
                                      SymbolHandle *out) {
  nnvm::Symbol *s = new nnvm::Symbol();
  API_BEGIN();
  std::vector<NDArray *> outputs;
  outputs.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    NDArray *array = reinterpret_cast<NDArray *>(output_handles[i]);
    outputs.emplace_back(array);
  }
  // Obtain Symbol
  *s = Imperative::Get()->GetDeferredComputeSymbol(outputs);
  *out = s;
  API_END_HANDLE_ERROR(delete s;);
}
