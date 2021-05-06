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
 *  Copyright (c) 2015 by Contributors
 * \file c_api.cc
 * \brief C API of mxnet
 */
#include <vector>
#include <sstream>
#include <string>
#include <mutex>
#include <memory>
#include <functional>
#include <unordered_map>
#include <utility>
#include "dmlc/base.h"
#include "dmlc/logging.h"
#include "dmlc/io.h"
#include "dmlc/memory_io.h"
#include "dmlc/recordio.h"
#include "dmlc/omp.h"
#include "mxnet/base.h"
#include "mxnet/ndarray.h"
#include "mxnet/operator.h"
#include "mxnet/io.h"
#include "mxnet/c_api.h"
#include "mxnet/kvstore.h"
#include "mxnet/rtc.h"
#include "mxnet/storage.h"
#include "mxnet/libinfo.h"
#include "mxnet/imperative.h"
#include "mxnet/lib_api.h"
#include "../initialize.h"
#include "./c_api_common.h"
#include "../operator/custom/custom-inl.h"
#include "../operator/operator_common.h"
#include "../operator/subgraph/common.h"
#include "../operator/tensor/matrix_op-inl.h"
#include "../operator/tvmop/op_module.h"
#include "../operator/subgraph/partitioner/custom_subgraph_property.h"
#include "../operator/subgraph/subgraph_property.h"
#include "../common/utils.h"
#include "nnvm/pass_functions.h"

using namespace mxnet;

// Internal function to get the information
// from function registry
// Used to implement MXSymbolGetAtomicSymbolInfo and MXFuncGetInfo
template<typename FunRegType>
inline int MXAPIGetFunctionRegInfo(const FunRegType *e,
                                   const char **name,
                                   const char **description,
                                   uint32_t *num_args,
                                   const char ***arg_names,
                                   const char ***arg_type_infos,
                                   const char ***arg_descriptions,
                                   const char **return_type) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();

  API_BEGIN();
  *name = e->name.c_str();
  *description = e->description.c_str();
  *num_args = static_cast<uint32_t>(e->arguments.size());
  if (return_type) *return_type = e->return_type.c_str();
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < e->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(e->arguments[i].name.c_str());
  }
  for (size_t i = 0; i < e->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(e->arguments[i].type_info_str.c_str());
  }
  for (size_t i = 0; i < e->arguments.size(); ++i) {
    ret->ret_vec_charp.push_back(e->arguments[i].description.c_str());
  }
  *arg_names = dmlc::BeginPtr(ret->ret_vec_charp);
  *arg_type_infos = dmlc::BeginPtr(ret->ret_vec_charp) + e->arguments.size();
  *arg_descriptions = dmlc::BeginPtr(ret->ret_vec_charp) + (e->arguments.size() * 2);
  API_END();
}

size_t MXGetBytesInUse(int dev_type, int dev_id) {
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  return Storage::Get()->GetMemoryInUseInBytes(ctx);
}

// NOTE: return value is added in API_END

std::string getExtensionMsgs(mxnet::ext::msgSize_t msgSize,
                             mxnet::ext::msgGet_t msgGet) {
  std::string str;
  if (msgSize() > 0) {
    str = "\nExtension Traceback:\n";
    for (int i = 0; i < msgSize(); i++) {
      const char* tmp;
      msgGet(i, &tmp);
      // format: [i] message
      str += std::string("\t[") + std::to_string(i) + std::string("] ")
        + std::string(tmp) + std::string("\n");
    }
  }
  return str;
}

/*!
 * \brief Common compute function dispatcher for forward/backward and stateful forward/backward
 * state_ptr will be nullptr for regular ops; fcomp_fp is nullptr for stateful ops
 */
void CustomFComputeDispatcher(const std::string op_name,
                              const mxnet::ext::opCallFComp_t callFComp,
                              const mxnet::ext::fcomp_t fcomp_fp,
                              const nnvm::NodeAttrs* attrs,
                              const mxnet::ext::opCallFStatefulComp_t callFStatefulComp,
                              int stateful_forward_flag,
                              const OpStatePtr* state_ptr,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs,
                              mxnet::ext::msgSize_t msgSize,
                              mxnet::ext::msgGet_t msgGet) {
  using namespace mxnet::ext;

  std::vector<void*> in_data, out_data;
  std::vector<const int64_t*> in_shapes, out_shapes;
  std::vector<int> in_dims, out_dims;
  std::vector<int> in_types, out_types;
  std::vector<size_t> in_verIDs, out_verIDs;
  std::vector<const char*> in_dev_type, out_dev_type;
  std::vector<int> in_dev_id, out_dev_id;
  std::vector<NDArray> conv_mkl;  // converted NDArrays from MKLDNN format

  // Extra data for sparse inputs and outputs.
  std::vector<int> in_stypes(inputs.size(), 0), out_stypes(outputs.size(), 0);
  std::vector<void*> in_indices(inputs.size(), nullptr), out_indices(outputs.size(), nullptr);
  std::vector<void*> in_indptr(inputs.size(), nullptr), out_indptr(outputs.size(), nullptr);
  std::vector<int64_t> in_indices_shapes(inputs.size(), 0), out_indices_shapes(outputs.size(), 0);
  std::vector<int64_t> in_indptr_shapes(inputs.size(), 0), out_indptr_shapes(outputs.size(), 0);

  // convert inputs/outpus NDArray to C types to be passed to lib_api.h
  for (size_t i = 0; i < inputs.size(); i++) {
    NDArray const* in_nd = &(inputs[i]);
#if MXNET_USE_MKLDNN == 1
    // reorder data if in MKLDNN format
    if (in_nd->IsMKLDNNData()) {
      // convert from MKLDNN
      conv_mkl.push_back(in_nd->Reorder2Default());
      in_nd = &(conv_mkl.back());
    }
#endif
    // pull out parts to pass over to library
    in_data.push_back(in_nd->data().dptr_);
    in_shapes.push_back(in_nd->shape().data());
    in_dims.push_back(in_nd->shape().ndim());
    in_types.push_back(in_nd->dtype());
    in_verIDs.push_back(in_nd->version());
    // string repr of supported context for custom library, currently only "cpu" and "gpu"
    const char* ctx_str = in_nd->ctx().dev_mask() == Context::kCPU ? "cpu" : "gpu";
    in_dev_type.push_back(ctx_str);

    in_dev_id.push_back(in_nd->ctx().real_dev_id());
    if (inputs[i].storage_type() == mxnet::kRowSparseStorage) {
      in_stypes[i] = 1;
      in_indices[i] = inputs[i].aux_data(rowsparse::kIdx).dptr_;
      in_indices_shapes[i] = inputs[i].aux_shape(rowsparse::kIdx).Size();
    } else if (inputs[i].storage_type() == mxnet::kCSRStorage) {
      in_stypes[i] = 2;
      in_indices[i] = inputs[i].aux_data(csr::kIdx).dptr_;
      in_indptr[i] = inputs[i].aux_data(csr::kIndPtr).dptr_;
      in_indices_shapes[i] = inputs[i].aux_shape(csr::kIdx).Size();
      in_indptr_shapes[i] = inputs[i].aux_shape(csr::kIndPtr).Size();
    }
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    out_data.push_back(outputs[i].data().dptr_);
    out_shapes.push_back(outputs[i].shape().data());
    out_dims.push_back(outputs[i].shape().ndim());
    out_types.push_back(outputs[i].dtype());
    out_verIDs.push_back(outputs[i].version());
    const char* ctx_str = outputs[i].ctx().dev_mask() == Context::kCPU ? "cpu" : "gpu";
    out_dev_type.push_back(ctx_str);
    out_dev_id.push_back(outputs[i].ctx().real_dev_id());

    if (outputs[i].storage_type() == mxnet::kRowSparseStorage) {
      out_stypes[i] = 1;
      out_indices[i] = outputs[i].aux_data(rowsparse::kIdx).dptr_;
      out_indices_shapes[i] = outputs[i].aux_shape(rowsparse::kIdx).Size();
    } else if (outputs[i].storage_type() == mxnet::kCSRStorage) {
      out_stypes[i] = 2;
      out_indices[i] = outputs[i].aux_data(csr::kIdx).dptr_;
      out_indptr[i] = outputs[i].aux_data(csr::kIndPtr).dptr_;
      out_indices_shapes[i] = outputs[i].aux_shape(csr::kIdx).Size();
      out_indptr_shapes[i] = outputs[i].aux_shape(csr::kIndPtr).Size();
    }
  }

  // get memory resource and mxnet backend streams
  CHECK(ctx.requested.size() >= 2)
    << "Custom operator should register at least memory resource and parallel random resource";
  const Resource &resource = ctx.requested.at(0);
  mshadow::Stream<mxnet::cpu> *cpu_stream = ctx.get_stream<mxnet::cpu>();
  mshadow::Stream<mxnet::gpu> *gpu_stream = ctx.get_stream<mxnet::gpu>();

  // create lambda that captures stream & resource objects
  // this temp workspace holds memory allocated by custom library via OpResource
  auto cpu_alloc = [&](int size) {
    mshadow::Tensor<mxnet::cpu, 1, char> workspace =
      resource.get_space_typed<mxnet::cpu, 1, char>(mshadow::Shape1(size), cpu_stream);
    return workspace.dptr_;
  };
  auto gpu_alloc = [&](int size) {
    mshadow::Tensor<mxnet::gpu, 1, char> workspace =
      resource.get_space_typed<mxnet::gpu, 1, char>(mshadow::Shape1(size), gpu_stream);
    return workspace.dptr_;
  };

  // create lambda that allocates memory for sparse and
  // returns allocated arrays for data, indices and indptr.
  auto sparse_alloc = [&](int index, int indices_len, int idxptr_len,
                          void** data, int64_t** indices, int64_t** indptr) {
    if (idxptr_len == 0) {
      // Row Sparse
      outputs[index].CheckAndAlloc({mshadow::Shape1(indices_len)});
      *data = outputs[index].data().dptr_;
      *indices = reinterpret_cast<int64_t*>(outputs[index].aux_data(rowsparse::kIdx).dptr_);
    } else {
      // CSR
      outputs[index].CheckAndAlloc({mshadow::Shape1(idxptr_len), mshadow::Shape1(indices_len)});
      *data = outputs[index].data().dptr_;
      *indices = reinterpret_cast<int64_t*>(outputs[index].aux_data(csr::kIdx).dptr_);
      *indptr = reinterpret_cast<int64_t*>(outputs[index].aux_data(csr::kIndPtr).dptr_);
    }
  };

  // create no-capture lambda so that we can cast it to function pointer
  // lambda with captures cannot be cast to function pointer and pass to lib_api.h
  // this needs to be a lambda function so that we can do the decltype cast
  typedef decltype(cpu_alloc) alloc_type_cpu;
  auto cpu_malloc = [](void* _cpu_alloc, int size) {
    // cast the void* argument to the type for the cpu_alloc lambda function
    alloc_type_cpu* cpualloc = static_cast<alloc_type_cpu*>(_cpu_alloc);
    // call cpu_alloc to actually allocate memory and return the pointer
    return static_cast<void*>((*cpualloc)(size));
  };

  typedef decltype(gpu_alloc) alloc_type_gpu;
  auto gpu_malloc = [](void* _gpu_alloc, int size) {
    alloc_type_gpu* gpualloc = static_cast<alloc_type_gpu*>(_gpu_alloc);
    return static_cast<void*>((*gpualloc)(size));
  };

  typedef decltype(sparse_alloc) alloc_type_sparse;
  auto sparse_malloc = [](void* _sparse_alloc, int index, int indices_len, int idxptr_len,
                           void** data, int64_t** indices, int64_t** indptr) {
    alloc_type_sparse* sparsealloc = static_cast<alloc_type_sparse*>(_sparse_alloc);
    (*sparsealloc)(index, indices_len, idxptr_len, data, indices, indptr);
  };

  // get actual cudaStream_t out of mxnet gpu stream and pass to lib_api.h
  void *cuda_stream = nullptr;
#if MXNET_USE_CUDA
  if ((inputs.size() > 0 && inputs[0].ctx().dev_mask() == Context::kGPU) ||
      (outputs.size() > 0 && outputs[0].ctx().dev_mask() == Context::kGPU)) {
    cuda_stream = static_cast<void*>(gpu_stream->stream_);
  }
#endif

  // get mxnet initialized and seeded RNG states and pass to lib_api.h
  void *rng_cpu_states = nullptr, *rng_gpu_states = nullptr;
  using mxnet::common::random::RandGenerator;
  RandGenerator<cpu, float> *pgen_cpu = ctx.requested.at(1).get_parallel_random<cpu, float>();
  rng_cpu_states = pgen_cpu->GetStates();
#if MXNET_USE_CUDA
  RandGenerator<gpu, float> *pgen_gpu = ctx.requested.at(1).get_parallel_random<gpu, float>();
  rng_gpu_states = pgen_gpu->GetStates();
#endif

  CHECK((fcomp_fp != nullptr && state_ptr == nullptr)
        || (fcomp_fp == nullptr && state_ptr != nullptr))
    << "Can only register either regular op or stateful op for '" << op_name << "'";

  if (fcomp_fp != nullptr) {
    // convert attributes to vector of char*
    std::vector<const char*> attr_keys, attr_vals;
    for (auto &kv : attrs->dict) {
      attr_keys.push_back(kv.first.c_str());
      attr_vals.push_back(kv.second.c_str());
    }

    // call fcompute function
    int retval = callFComp(fcomp_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                           in_shapes.data(), in_dims.data(), in_data.data(), in_types.data(),
                           in_verIDs.data(), in_dev_type.data(), in_dev_id.data(), in_data.size(),
                           out_shapes.data(), out_dims.data(), out_data.data(), out_types.data(),
                           out_verIDs.data(), out_dev_type.data(), out_dev_id.data(),
                           out_data.size(),
                           cpu_malloc, &cpu_alloc, gpu_malloc, &gpu_alloc, cuda_stream,
                           sparse_malloc, &sparse_alloc, in_stypes.data(), out_stypes.data(),
                           in_indices.data(), out_indices.data(), in_indptr.data(),
                           out_indptr.data(),
                           in_indices_shapes.data(), out_indices_shapes.data(),
                           in_indptr_shapes.data(), out_indptr_shapes.data(),
                           rng_cpu_states, rng_gpu_states);
    std::string msgs = getExtensionMsgs(msgSize, msgGet);
    CHECK(retval) << "Error calling FCompute for custom operator '" << op_name << "'" << msgs;
  }

  if (state_ptr != nullptr) {
    // retrieve op state object created from CreateOpState
    CustomStatefulOpWrapper& op = state_ptr->get_state<CustomStatefulOpWrapper>();
    CustomStatefulOp* state_op_inst = op.get_instance();
    std::string msgs = getExtensionMsgs(msgSize, msgGet);
    CHECK(state_op_inst != nullptr)
      << "Error custom stateful operator is null for operator '" << op_name << "'" << msgs;

    // call fcompute function
    int retval = callFStatefulComp(stateful_forward_flag, state_op_inst,
                                   in_shapes.data(), in_dims.data(), in_data.data(),
                                   in_types.data(),
                                   in_verIDs.data(), in_dev_type.data(), in_dev_id.data(),
                                   in_data.size(),
                                   out_shapes.data(), out_dims.data(), out_data.data(),
                                   out_types.data(),
                                   out_verIDs.data(), out_dev_type.data(), out_dev_id.data(),
                                   out_data.size(),
                                   cpu_malloc, &cpu_alloc, gpu_malloc, &gpu_alloc, cuda_stream,
                                   sparse_malloc, &sparse_alloc, in_stypes.data(),
                                   out_stypes.data(), in_indices.data(), out_indices.data(),
                                   in_indptr.data(), out_indptr.data(),
                                   in_indices_shapes.data(), out_indices_shapes.data(),
                                   in_indptr_shapes.data(), out_indptr_shapes.data(),
                                   rng_cpu_states, rng_gpu_states);
    msgs = getExtensionMsgs(msgSize, msgGet);
    CHECK(retval) << "Error calling FStatefulCompute for custom operator '" << op_name << "'"
                  << msgs;
  }
}

template <typename RescReq, typename AttrParser, typename NumInputs, typename NumOutputs,
          typename NumInOuts,
          typename InferType, typename InferShape, typename InferSType, typename MutateInputs,
          typename SubgraphNumInputs, typename SubgraphInferType, typename SubgraphInferShape,
          typename SubgraphInferSType, typename CreateOpState, typename GradReg>
void registerOp(const char* name, const std::string& name_str, bool isSubgraphOp,
                RescReq resc_req, AttrParser attr_parser, NumInputs num_inputs,
                NumOutputs num_outputs, NumInOuts num_inouts, InferType infer_type,
                InferShape infer_shape, InferSType infer_storage_type,
                MutateInputs mutate_inputs, SubgraphNumInputs num_subgraph_inputs,
                SubgraphInferType infer_subgraph_type, SubgraphInferShape infer_subgraph_shape,
                SubgraphInferSType infer_subgraph_storage_type, CreateOpState create_opstate,
                GradReg grad_reg, mxnet::ext::mutateInputs_t mutate_fp,
                const std::unordered_map<std::string, mxnet::ext::createOpState_t> &createop_map,
                const std::unordered_map<std::string, mxnet::ext::fcomp_t> &forward_ctx_map,
                const std::unordered_map<std::string, mxnet::ext::fcomp_t> &backward_ctx_map,
                mxnet::ext::opCallFComp_t callFComp,
                mxnet::ext::opCallFStatefulComp_t callFStatefulComp,
                mxnet::ext::msgSize_t msgSize,
                mxnet::ext::msgGet_t msgGet) {
  using namespace mxnet::ext;

  // check if operator is already registered
  const nnvm::Op *regOpPtr = dmlc::Registry<nnvm::Op>::Get()->Find(name);
  nnvm::Op &regOp = dmlc::Registry<nnvm::Op>::Get()->__REGISTER_OR_GET__(name);
  int plevel = 10;
  if (regOpPtr != nullptr) {
    // overwrite registration of existing op with custom op
    regOp.arguments.clear();
    // set attribute with higher plevel (11) to allow re-registering once
    // TODO(samskalicky): enable constant overwriting of registertion multiple times
    plevel++;
  }
  // define supported resources for both subgraph ops and regular ops
  regOp.set_attr<FResourceRequest>("FResourceRequest", resc_req, plevel);
  if (!isSubgraphOp) {
    regOp.set_attr_parser(attr_parser);
    regOp.set_num_inputs(num_inputs);
    regOp.set_num_outputs(num_outputs);
    regOp.set_attr<nnvm::FInferType>("FInferType", infer_type, plevel);
    regOp.set_attr<FInferStorageType>("FInferStorageType", infer_storage_type, plevel);
    regOp.set_attr<mxnet::FInferShape>("FInferShape", infer_shape, plevel);
    // optionally add fmutate inputs if user specified a function
    if (mutate_fp != nullptr)
      regOp.set_attr<nnvm::FMutateInputs>("FMutateInputs", mutate_inputs, plevel);
  } else {
    using namespace mxnet::op;
    regOp.set_num_inputs(num_subgraph_inputs);
    regOp.set_num_outputs(DefaultSubgraphOpNumOutputs);
    regOp.set_attr<nnvm::FInferType>("FInferType", infer_subgraph_type, plevel);
    regOp.set_attr<mxnet::FInferShape>("FInferShape", infer_subgraph_shape, plevel);
    regOp.set_attr<FInferStorageType>("FInferStorageType",
                                      infer_subgraph_storage_type, plevel);
    regOp.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                        DefaultSubgraphOpMutableInputs, plevel);
  }
  // optionally add stateful forward
  if (createop_map.size() != 0) {
    regOp.set_attr<FCreateOpState>("FCreateOpState", create_opstate, plevel);
    auto fstate_forward = [=](const OpStatePtr& state_ptr,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
      CustomFComputeDispatcher(name_str, nullptr, nullptr, nullptr,
                               callFStatefulComp, 1, &state_ptr, ctx, inputs, req, outputs,
                               msgSize, msgGet);
    };
    if (createop_map.count("cpu") > 0)
      regOp.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", fstate_forward, plevel);
    if (createop_map.count("gpu") > 0)
      regOp.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", fstate_forward, plevel);
  } else {
    auto forward_lambda = [=](const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
      if (ctx.run_ctx.ctx.dev_mask() == Context::kCPU) {
        CHECK_GT(forward_ctx_map.count("cpu"), 0);
        fcomp_t fcomp = forward_ctx_map.at("cpu");
        CustomFComputeDispatcher(name_str, callFComp, fcomp, &attrs,
                                 nullptr, 0, nullptr, ctx, inputs, req, outputs, msgSize, msgGet);
      } else if (ctx.run_ctx.ctx.dev_mask() == Context::kGPU) {
        CHECK_GT(forward_ctx_map.count("gpu"), 0);
        fcomp_t fcomp = forward_ctx_map.at("gpu");
        CustomFComputeDispatcher(name_str, callFComp, fcomp, &attrs,
                                 nullptr, 0, nullptr, ctx, inputs, req, outputs, msgSize, msgGet);
      }
    };
    if (forward_ctx_map.count("cpu") > 0)
      regOp.set_attr<FComputeEx>("FComputeEx<cpu>", forward_lambda, plevel);
    if (forward_ctx_map.count("gpu") > 0)
      regOp.set_attr<FComputeEx>("FComputeEx<gpu>", forward_lambda, plevel);
  }
  // optionally add fgradient if user specified a function, or for stateful ops
  if (backward_ctx_map.size() != 0 || createop_map.size() != 0) {
    std::string grad_name = "_backward_" + name_str;
    nnvm::Op &gradOp = dmlc::Registry<nnvm::Op>::Get()->__REGISTER_OR_GET__(grad_name);
    regOp.set_attr<nnvm::FGradient>("FGradient", grad_reg, plevel);
    gradOp.set_attr<nnvm::TIsBackward>("TIsBackward", true, plevel);
    gradOp.set_attr<FInferStorageType>("FInferStorageType", infer_storage_type, plevel);
    gradOp.set_attr<FResourceRequest>("FResourceRequest", resc_req, plevel);

    if (!isSubgraphOp) {
      // register attr parser and standard functions for non-subgraph ops
      gradOp.set_attr_parser(attr_parser);
      gradOp.set_num_inputs(num_inouts);
      gradOp.set_num_outputs(num_inputs);
    } else {
      // for subgraph ops use special functions that do not invoke attr_parser
      using namespace mxnet::op;
      auto grad_inouts = [=](const nnvm::NodeAttrs& attrs) {
        // for backward passes, inputs + outputs + input gradients (one for each output)
        uint32_t cnt = num_subgraph_inputs(attrs);
        cnt += 2 * DefaultSubgraphOpNumOutputs(attrs);
        return cnt;
      };
      gradOp.set_num_inputs(grad_inouts);
      gradOp.set_num_outputs(num_subgraph_inputs);
    }

    if (createop_map.size() != 0) {
      // for stateful operators
      gradOp.set_attr<bool>("TIsLayerOpBackward", true, plevel);
      auto fstate_backward = [=](const OpStatePtr& state_ptr,
                                 const OpContext& ctx,
                                 const std::vector<NDArray>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<NDArray>& outputs) {
        CustomFComputeDispatcher(name_str, nullptr, nullptr, nullptr,
                                 callFStatefulComp, 0, &state_ptr, ctx, inputs, req, outputs,
                                 msgSize, msgGet);
      };
      gradOp.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", fstate_backward, plevel);
      gradOp.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", fstate_backward, plevel);
    } else {
      // for stateless operators
      if (backward_ctx_map.count("cpu") > 0) {
        fcomp_t fcomp_back_cpu = backward_ctx_map.at("cpu");
        auto backward_cpu_lambda = [=](const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs) {
          CustomFComputeDispatcher(name_str, callFComp, fcomp_back_cpu, &attrs,
                                   nullptr, 0, nullptr, ctx, inputs, req, outputs, msgSize, msgGet);
        };
        gradOp.set_attr<FComputeEx>("FComputeEx<cpu>", backward_cpu_lambda, plevel);
      }
      if (backward_ctx_map.count("gpu") > 0) {
        fcomp_t fcomp_back_gpu = backward_ctx_map.at("gpu");
        auto backward_gpu_lambda = [=](const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<NDArray>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<NDArray>& outputs) {
          CustomFComputeDispatcher(name_str, callFComp, fcomp_back_gpu, &attrs,
                                   nullptr, 0, nullptr, ctx, inputs, req, outputs, msgSize, msgGet);
        };
        gradOp.set_attr<FComputeEx>("FComputeEx<gpu>", backward_gpu_lambda, plevel);
      }
    }
    }
  regOp.add_argument("data", "NDArray[]", "Source inputs");
}

void registerOperators(void *lib, int verbose, mxnet::ext::msgSize_t msgSize,
                       mxnet::ext::msgGet_t msgGet) {
  using namespace mxnet::ext;

  // get C type interface functions
  opCallFree_t callFree = get_func<opCallFree_t>(lib, const_cast<char*>(MXLIB_OPCALLFREE_STR));

  opCallParseAttrs_t callParseAttrs =
    get_func<opCallParseAttrs_t>(lib, const_cast<char*>(MXLIB_OPCALLPARSEATTRS_STR));

  opCallInferShape_t callInferShape =
    get_func<opCallInferShape_t>(lib, const_cast<char*>(MXLIB_OPCALLINFERSHAPE_STR));

  opCallInferType_t callInferType =
    get_func<opCallInferType_t>(lib, const_cast<char*>(MXLIB_OPCALLINFERTYPE_STR));

  opCallInferSType_t callInferSType =
    get_func<opCallInferSType_t>(lib, const_cast<char*>(MXLIB_OPCALLINFERSTYPE_STR));

  opCallFComp_t callFComp =
    get_func<opCallFComp_t>(lib, const_cast<char*>(MXLIB_OPCALLFCOMP_STR));

  opCallMutateInputs_t callMutateInputs =
    get_func<opCallMutateInputs_t>(lib, const_cast<char*>(MXLIB_OPCALLMUTATEINPUTS_STR));

  opCallCreateOpState_t callCreateOpState =
    get_func<opCallCreateOpState_t>(lib, const_cast<char*>(MXLIB_OPCALLCREATEOPSTATE_STR));

  opCallDestroyOpState_t callDestroyOpState =
    get_func<opCallDestroyOpState_t>(lib, const_cast<char*>(MXLIB_OPCALLDESTROYOPSTATE_STR));

  opCallFStatefulComp_t callFStatefulComp =
    get_func<opCallFStatefulComp_t>(lib, const_cast<char*>(MXLIB_OPCALLFSTATEFULCOMP_STR));

  // get number of operators registered in the library
  opRegSize_t opRegSize = get_func<opRegSize_t>(lib, const_cast<char*>(MXLIB_OPREGSIZE_STR));
  int numOps = opRegSize();
  if (verbose) LOG(INFO) << "Found " << numOps << " operators in library";

  /*
   * Get all custom operators implementation from custom library
   * loop and register each operator in the library to NNVM
   */
  opRegGet_t opRegGet = get_func<opRegGet_t>(lib, const_cast<char*>(MXLIB_OPREGGET_STR));
  for (int i = 0; i < numOps; i++) {
    const char* name;
    // function pointers holding implementation from custom library
    parseAttrs_t parse_fp = nullptr;
    inferType_t type_fp = nullptr;
    inferSType_t stype_fp = nullptr;
    inferShape_t shape_fp = nullptr;
    // optional attributes
    mutateInputs_t mutate_fp = nullptr;
    bool isSubgraphOp = false;
    int _isSubgraphOp = 0;
    // lists of forward and backward function associated with each context
    const char **forward_ctx, **backward_ctx, **createop_ctx;
    fcomp_t *forward_fcomp, *backward_fcomp;
    createOpState_t *createop_fp;
    int forward_count, backward_count, createop_count;

    // main function to get custom operator implemenation from the custom library
    opRegGet(i, &name, &_isSubgraphOp,
             &forward_ctx, &forward_fcomp, &forward_count,
             &backward_ctx, &backward_fcomp, &backward_count,
             &createop_ctx, &createop_fp, &createop_count,
             &parse_fp, &type_fp, &stype_fp, &shape_fp, &mutate_fp);

    // construct maps of context to forward/backward custom library function
    std::unordered_map<std::string, fcomp_t> forward_ctx_map;
    std::unordered_map<std::string, fcomp_t> backward_ctx_map;
    std::unordered_map<std::string, createOpState_t> createop_map;
    for (int i=0; i < forward_count; i++) {
      std::string ctx_str(forward_ctx[i]);
      forward_ctx_map[ctx_str] = forward_fcomp[i];
    }
    for (int i=0; i < backward_count; i++) {
      std::string ctx_str(backward_ctx[i]);
      backward_ctx_map[ctx_str] = backward_fcomp[i];
    }
    for (int i=0; i < createop_count; i++) {
      std::string ctx_str(createop_ctx[i]);
      createop_map[ctx_str] = createop_fp[i];
    }
    // set bool, dont pass bool across ABI boundary
    isSubgraphOp = _isSubgraphOp;

    // validate custom operator functions from the dynamic library
    if (!isSubgraphOp) {
      CHECK(parse_fp != nullptr) << "Error loading '" << name
                                 << "' custom op, ParseAttrs function was not set.";
      CHECK(forward_ctx_map.size() != 0 || createop_map.size() != 0)
                            << "Error loading '" << name
                            << "' custom op, Forward or CreateOpState function was not set.";
      CHECK(type_fp != nullptr) << "Error loading '" << name
                            << "' custom op, InferType function was not set.";
      CHECK(shape_fp != nullptr) << "Error loading '" << name
                            << "' custom op, InferShape function was not set.";
    } else {
      CHECK(createop_map.size() != 0) << "Error loading '" << name
                            << "' custom subgraph op, CreateOpState function was not set.";
    }
    if (verbose) LOG(INFO) << "\tOp[" << i << "] " << name;
    if (verbose && isSubgraphOp) LOG(INFO) << "\t\tisSubgraphOp";
    std::string name_str(name);

    /*
     * Below are a series of lambda functions that will be registered in the NNVM op registration
     * Each one has the standard MXNet signature and converts to types supported by externally
     * registered operators.
     */

    // lambda function to call parse attributes
    auto attr_parser = [=](const NodeAttrs* attrs) {
      // convert attributes to vector of char
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs->dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }
      // convert subgraph symbol from node attributes to char*
      std::string subgraph_json;
      if (!attrs->subgraphs.empty()) {
        nnvm::Graph g;
        g.outputs = attrs->subgraphs[0].get()->outputs;
        subgraph_json = nnvm::pass::SaveJSON(g);
        attr_keys.push_back(MX_STR_SUBGRAPH_SYM_JSON);
        attr_vals.push_back(subgraph_json.c_str());
      }

      int num_in = -1;
      int num_out = -1;
      int retval = callParseAttrs(parse_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                                  &num_in, &num_out);
      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(retval) << "Error calling ParseAttrs for custom operator '" << name_str << "'" << msgs;

      // return type void
    };

    // lambda function to call parse attributes and return the number of inputs
    auto num_inputs = [=](const NodeAttrs& attrs) {
      // convert attributes to vector of char
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      int num_in = -1;
      int num_out = -1;
      int retval = callParseAttrs(parse_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                                  &num_in, &num_out);
      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(retval) << "Error calling ParseAttrs::num_inputs for custom operator '" << name_str
      << "'" << msgs;

      // get extra inputs, if exists
      size_t extra_inputs = 0;
      if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
        extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));

      return num_in + extra_inputs;
    };

    // lambda function to call parse attributes and return the number of inputs for subgraph ops
    auto num_subgraph_inputs = [=](const NodeAttrs& attrs) {
      // get number of inputs for subgraph
      int num_in = mxnet::op::DefaultSubgraphOpNumInputs(attrs);

      // get extra inputs, if exists
      size_t extra_inputs = 0;
      if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
        extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));

      return num_in + extra_inputs;
    };

    // lambda function to call parse attributes and return the number of outputs
    auto num_outputs = [=](const NodeAttrs& attrs) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      int num_in = -1;
      int num_out = -1;
      int retval = callParseAttrs(parse_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                                  &num_in, &num_out);
      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(retval) << "Error calling ParseAttrs::num_outputs for custom operator '" << name_str
      << "'" << msgs;

      return num_out;
    };

    // lambda function to call parse attributes and return the number of inputs and outputs
    // for backward computation
    auto num_inouts = [=](const NodeAttrs& attrs) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      int num_in = -1;
      int num_out = -1;
      int retval = callParseAttrs(parse_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                                  &num_in, &num_out);
      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(retval) << "Error calling ParseAttrs::num_outputs for custom operator '" << name_str
      << "'" << msgs;
      // for backward passes, inputs + outputs + input gradients (one for each output)

      // get extra inputs, if exists
      size_t extra_inputs = 0;
      if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
        extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));

      return num_in + extra_inputs + 2 * num_out;
    };

    // lambda function to call infer shape
    auto infer_shape = [=] (const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_shape,
                            mxnet::ShapeVector *out_shape) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // get extra inputs, if exists
      size_t extra_inputs = 0;
      if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
        extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));
      size_t num_inputs = in_shape->size() - extra_inputs;

      std::vector<uint32_t*> inshapes(num_inputs);
      std::vector<int> indims(num_inputs);

      // determine amount of memory needed to store all the input shapes
      size_t buff_size = 0;
      for (size_t i = 0; i < num_inputs; ++i)
        buff_size += (*in_shape)[i].ndim();

      // copy input shapes from ShapeVector to raw memory layout
      std::vector<uint32_t> inbuff(buff_size);
      uint32_t *ptr = inbuff.data();
      for (size_t i = 0; i < num_inputs; ++i) {
        inshapes[i] = ptr;
        indims[i] = (*in_shape)[i].ndim();
        for (int j = 0; j < (*in_shape)[i].ndim(); ++j, ++ptr) {
          *ptr = static_cast<uint32_t>((*in_shape)[i][j]);
        }
      }

      // modified input shapes will be allocated by infer shape function
      uint32_t** mod_inshapes = nullptr;
      int* mod_indims = nullptr;
      // output shapes will be allocated by infer shape function
      uint32_t** outshapes = nullptr;
      int* outdims = nullptr;

      int retval = callInferShape(shape_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                                  inshapes.data(), indims.data(), num_inputs,
                                  &mod_inshapes, &mod_indims,
                                  &outshapes, &outdims, out_shape->size());
      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(retval) << "Error calling InferShape for custom operator '" << name_str << "'" << msgs;

      std::vector<uint32_t*> in_shapes(num_inputs);
      // determine amount of memory needed to store all the modified input shapes
      buff_size = 0;
      for (size_t i = 0; i < num_inputs; i++) {
        buff_size += mod_indims[i];
      }

      // copy modified input shapes from custom op memory to MXNet memory
      std::vector<uint32_t> mod_inbuff(buff_size);
      ptr = mod_inbuff.data();
      for (size_t i = 0; i < num_inputs; ++i) {
        in_shapes[i] = ptr;
        for (int j = 0; j < mod_indims[i]; ++j, ++ptr) {
          *ptr = static_cast<uint32_t>(mod_inshapes[i][j]);
        }
      }

      // assign modified input shapes to ShapeVector
      for (size_t i = 0; i < num_inputs; ++i) {
        SHAPE_ASSIGN_CHECK(*in_shape, i,
                           mxnet::TShape(in_shapes[i], in_shapes[i]+mod_indims[i]));
      }

      std::vector<uint32_t*> out_shapes(out_shape->size());
      // determine amount of memory needed to store all the output shapes
      buff_size = 0;
      for (size_t i = 0; i < out_shape->size(); i++) {
        buff_size += outdims[i];
      }

      // copy output shapes from custom op memory to MXNet memory
      std::vector<uint32_t> outbuff(buff_size);
      ptr = outbuff.data();
      for (size_t i = 0; i < out_shape->size(); ++i) {
        out_shapes[i] = ptr;
        for (int j = 0; j < outdims[i]; ++j, ++ptr) {
          *ptr = static_cast<uint32_t>(outshapes[i][j]);
        }
      }

      // assign output shapes to ShapeVector
      for (size_t i = 0; i < out_shape->size(); ++i) {
        SHAPE_ASSIGN_CHECK(*out_shape, i,
                           mxnet::TShape(out_shapes[i], out_shapes[i]+outdims[i]));
      }

      // free memory used by custom op to allocate shapes/dims
      callFree(mod_indims);
      for (size_t i = 0; i < num_inputs; i++) {
        callFree(mod_inshapes[i]);
      }
      callFree(mod_inshapes);

      callFree(outdims);
      for (size_t i = 0; i < out_shape->size(); i++) {
        callFree(outshapes[i]);
      }
      callFree(outshapes);

      return true;
    };

    // lambda function to call infer shape for subgraph ops
    auto infer_subgraph_shape = [=] (const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_shape,
                            mxnet::ShapeVector *out_shape) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // get extra inputs, if exists
      size_t extra_inputs = 0;
      if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
        extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));

      auto in_first = in_shape->begin();
      auto in_last  = in_first + in_shape->size() - extra_inputs;
      mxnet::ShapeVector *sg_in_shapes = new mxnet::ShapeVector(in_first, in_last);
      bool res = mxnet::op::DefaultSubgraphOpShape(attrs, sg_in_shapes, out_shape);

      // assign modified input shapes to ShapeVector
      for (unsigned i = 0; i < sg_in_shapes->size(); ++i) {
        SHAPE_ASSIGN_CHECK(*in_shape, i, sg_in_shapes->at(i));
      }
      return res;
    };

    // lambda function to call infer type
    auto infer_type = [=] (const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type,
                            std::vector<int> *out_type) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // get extra inputs, if exists
      size_t extra_inputs = 0;
      if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
        extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));
      size_t num_inputs = in_type->size() - extra_inputs;

      // copy input types from in_type
      std::vector<int> intypes(*in_type);

      // output types will be populated by inferType function
      std::vector<int> outtypes(out_type->size());

      int retval = callInferType(type_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                                 intypes.data(), num_inputs,
                                 outtypes.data(), out_type->size());
      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(retval) << "Error calling InferType for custom operator '" << name_str << "'" << msgs;

      // copy and assign modified input types from custom op to MXNet memory
      for (size_t i = 0; i < num_inputs; i++) {
        TYPE_ASSIGN_CHECK(*in_type, i, intypes[i]);
      }
      // copy and assign output types from custom op to MXNet memory
      for (size_t i = 0; i < out_type->size(); i++) {
        TYPE_ASSIGN_CHECK(*out_type, i, outtypes[i]);
      }

      return true;
    };

    // lambda function to call infer type for subgraph ops
    auto infer_subgraph_type = [=] (const nnvm::NodeAttrs& attrs,
                                    std::vector<int> *in_type,
                                    std::vector<int> *out_type) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // get extra inputs, if exists
      size_t extra_inputs = 0;
      if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
        extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));

      auto in_first = in_type->begin();
      auto in_last  = in_first + in_type->size() - extra_inputs;
      std::vector<int> *sg_in_types = new std::vector<int>(in_first, in_last);

      bool res = mxnet::op::DefaultSubgraphOpType(attrs, sg_in_types, out_type);
      // copy and assign modified input types
      for (size_t i = 0; i < sg_in_types->size(); i++) {
        TYPE_ASSIGN_CHECK(*in_type, i, sg_in_types->at(i));
      }
      return res;
    };

    // lambda function to convert from external mutate_inputs to internal MXNet types
    auto mutate_inputs = [=](const nnvm::NodeAttrs& attrs) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // C type placeholder for mutate input indices vector
      int* mutate_indices = nullptr;
      int indices_size = 0;

      // call mutate inputs function
      int retval = callMutateInputs(mutate_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                                    &mutate_indices, &indices_size);
      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(retval) << "Error calling MutateInputs for custom operator '" << name_str << "'"
      << msgs;

      std::vector<uint32_t> mutate_indices_list(indices_size);
      for (int i=0; i < indices_size; i++) {
        mutate_indices_list[i] = static_cast<uint32_t>(mutate_indices[i]);
      }

      return mutate_indices_list;
    };

    // lambda function to set storage types
    auto infer_storage_type = [=](const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int>* in_stypes,
                                std::vector<int>* out_stypes) {
      if (stype_fp == nullptr) {
        // InferSType is not defined in customized lib.
        CHECK(mxnet::common::ContainsOnlyStorage(*in_stypes, mxnet::kDefaultStorage))
        << "Error input tensors are not dense for custom operator '" << name_str << "'";
        // set outputs as dense
        return op::storage_type_assign(out_stypes, mxnet::kDefaultStorage,
                                       dispatch_mode, DispatchMode::kFComputeEx);
      } else {
        // InferSType is defined in customized lib.
        // convert attributes to vector of char*
        std::vector<const char*> attr_keys, attr_vals;
        for (auto kv : attrs.dict) {
          attr_keys.push_back(kv.first.c_str());
          attr_vals.push_back(kv.second.c_str());
        }

        // get extra inputs, if exists
        size_t extra_inputs = 0;
        if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
          extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));
        size_t num_inputs = in_stypes->size() - extra_inputs;

        // copy input types from in_stype
        std::vector<int> instypes(*in_stypes);

        // output types will be populated by inferType function
        std::vector<int> outstypes(out_stypes->size());
        int retval = callInferSType(stype_fp, attr_keys.data(), attr_vals.data(), attr_keys.size(),
                                    instypes.data(), num_inputs,
                                    outstypes.data(), out_stypes->size());
        std::string msgs = getExtensionMsgs(msgSize, msgGet);
        CHECK(retval) << "Error calling InferSType for custom operator '" << name_str << "'"
        << msgs;

        // copy and assign modified input storage types from custom op to MXNet memory.
        for (size_t i = 0; i < num_inputs; i++) {
          STORAGE_TYPE_ASSIGN_CHECK(*in_stypes, i, instypes[i]);
        }
        // copy and assign output storage types from custom op to MXNet memory.
        for (size_t i = 0; i < out_stypes->size(); i++) {
          STORAGE_TYPE_ASSIGN_CHECK(*out_stypes, i, outstypes[i]);
        }
        // assign dispatch mode
        DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
        return true;
      }
    };

    // lambda function to set storage types for subgraph ops
    auto infer_subgraph_storage_type = [=](const nnvm::NodeAttrs& attrs,
                                           const int dev_mask,
                                           DispatchMode* dispatch_mode,
                                           std::vector<int>* in_stypes,
                                           std::vector<int>* out_stypes) {
        // get extra inputs, if exists
        size_t extra_inputs = 0;
        if (attrs.dict.count(MX_STR_EXTRA_INPUTS) > 0)
          extra_inputs = std::stoi(attrs.dict.at(MX_STR_EXTRA_INPUTS));

        auto in_first = in_stypes->begin();
        auto in_last  = in_first + in_stypes->size() - extra_inputs;
        std::vector<int> *sg_in_stypes = new std::vector<int>(in_first, in_last);

        bool res = mxnet::op::DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                                           sg_in_stypes, out_stypes);
        // copy and assign modified input storage types
        for (size_t i = 0; i < sg_in_stypes->size(); i++) {
          STORAGE_TYPE_ASSIGN_CHECK(*in_stypes, i, sg_in_stypes->at(i));
        }
        return res;
    };

    // FGradient register lambda
    auto grad_reg = [=](const nnvm::ObjectPtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // create node for gradient
      auto p = nnvm::Node::Create();
      std::string grad_name = "_backward_" + name_str;
      p->attrs.op = nnvm::Op::Get(grad_name.c_str());
      p->attrs.name = n->attrs.name + "_backward";
      // copy attributes and subgraphs
      p->attrs.dict = n->attrs.dict;
      for (auto s : n->attrs.subgraphs)
        p->attrs.subgraphs.push_back(s);
      // set control dependency and attr parser
      p->control_deps.emplace_back(n);
      if (p->op()->attr_parser != nullptr) {
        p->op()->attr_parser(&(p->attrs));
      }
      // gradient inputs: copy gradients first
      std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
      // copy inputs second
      for (auto& h : n->inputs) {
        heads.push_back(h);
      }
      // gradient inputs: copy outputs last
      uint32_t n_out = n->num_outputs();
      for (uint32_t i = 0; i < n_out; ++i) {
        heads.emplace_back(n, i, 0);
      }
      // set inputs to gradient node
      p->inputs = heads;
      CHECK_EQ(p->num_inputs(), p->inputs.size())
      << "Number of inputs to operator " << grad_name << " (" << p->num_inputs()
      << ") does not match the actual number of inputs provided to operator "
      << p->attrs.name << " (" << p->inputs.size() << ").";
      // create output node entries
      return mxnet::op::CreateNodeEntries(p);
    };

    auto resc_req = [=](const NodeAttrs& attrs) {
      return std::vector<ResourceRequest>{ResourceRequest::kTempSpace,
                                          ResourceRequest::kParallelRandom};
    };

    // library author should implement and return a 'state' which points to an instance
    // in lambda we create OpStatePtr using the returned 'state'
    auto create_opstate = [=] (const NodeAttrs& attrs,
                               Context ctx,
                               const std::vector<TShape>& in_shapes,
                               const std::vector<int>& in_types) {
      // convert attributes to vector of char*
      std::vector<const char*> attr_keys, attr_vals;
      for (auto &kv : attrs.dict) {
        attr_keys.push_back(kv.first.c_str());
        attr_vals.push_back(kv.second.c_str());
      }

      // string repr of supported context for custom library, currently only "cpu" and "gpu"
      const char* ctx_str = ctx.dev_mask() == Context::kCPU ? "cpu" : "gpu";

      std::vector<uint32_t*> inshapes(in_shapes.size());
      std::vector<int> indims(in_shapes.size());

      // determine amount of memory needed to store all the input shapes
      size_t buff_size = 0;
      for (size_t i = 0; i < in_shapes.size(); ++i)
        buff_size += in_shapes[i].ndim();

      // copy input shapes to raw memory layout
      std::vector<uint32_t> inbuff(buff_size);
      uint32_t *ptr = inbuff.data();
      for (size_t i = 0; i < in_shapes.size(); ++i) {
        inshapes[i] = ptr;
        indims[i] = in_shapes[i].ndim();
        for (int j = 0; j < in_shapes[i].ndim(); ++j, ++ptr) {
          *ptr = static_cast<uint32_t>(in_shapes[i][j]);
        }
      }

      // convert subgraph symbol from node attributes to char*
      std::string subgraph_json;
      if (!attrs.subgraphs.empty()) {
        nnvm::Graph g;
        g.outputs = attrs.subgraphs[0].get()->outputs;
        subgraph_json = nnvm::pass::SaveJSON(g);
        attr_keys.push_back(MX_STR_SUBGRAPH_SYM_JSON);
        attr_vals.push_back(subgraph_json.c_str());
      }

      // create a pointer to hold custom op state object
      // only create one stateful op depending on passing context
      // user can add new supported context and call to custom library
      void* state_op_inst = nullptr;
      if (ctx.dev_mask() == Context::kCPU) {
        CHECK(createop_map.count("cpu") > 0)
          << "CPU CreateOpState not implemented for '" << name_str << "'";
        int retval = callCreateOpState(createop_map.at("cpu"), attr_keys.data(), attr_vals.data(),
                                       attr_keys.size(), ctx_str, ctx.real_dev_id(),
                                       inshapes.data(), indims.data(),
                                       in_shapes.size(), in_types.data(), &state_op_inst);
        std::string msgs = getExtensionMsgs(msgSize, msgGet);
        CHECK(retval) << "Error calling CreateOpState CPU for custom operator '" << name_str << "'"
                      << msgs;
      } else if (ctx.dev_mask() == Context::kGPU) {
        CHECK(createop_map.count("gpu") > 0)
          << "GPU CreateOpState not implemented for '" << name_str << "'";
        int retval = callCreateOpState(createop_map.at("gpu"), attr_keys.data(), attr_vals.data(),
                                       attr_keys.size(), ctx_str, ctx.real_dev_id(),
                                       inshapes.data(), indims.data(),
                                       in_shapes.size(), in_types.data(), &state_op_inst);
        std::string msgs = getExtensionMsgs(msgSize, msgGet);
        CHECK(retval) << "Error calling CreateOpState GPU for custom operator '" << name_str << "'"
        << msgs;
      }

      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(state_op_inst != nullptr)
      << "Error custom library failed to create stateful operator '" << name_str << "'" << msgs;

      CustomStatefulOp* state_op = reinterpret_cast<CustomStatefulOp*>(state_op_inst);
      if (!state_op->wasCreated() && !state_op->ignore_warn)
        LOG(INFO) << "WARNING! Custom stateful op " << state_op_inst << " was created without "
                  << "calling CustomStatefulOp::create(). Please ensure this object was "
                  << "allocated with 'new' since it will be destructed with 'delete'. "
                  << "To suppress this message without calling CustomStatefulOp::create() "
                  << "set ignore_warn to 'true' on custom stateful op instance.";
      return OpStatePtr::Create<CustomStatefulOpWrapper>(state_op, callDestroyOpState);
    };

    /* -------------- BELOW IS THE REGISTRATION FOR CUSTOM OPERATORS --------------- */

    registerOp(name, name_str, isSubgraphOp, resc_req, attr_parser, num_inputs, num_outputs,
               num_inouts, infer_type, infer_shape, infer_storage_type, mutate_inputs,
               num_subgraph_inputs, infer_subgraph_type, infer_subgraph_shape,
               infer_subgraph_storage_type, create_opstate, grad_reg, mutate_fp,
               createop_map, forward_ctx_map, backward_ctx_map, callFComp, callFStatefulComp,
               msgSize, msgGet);
  }
}

void registerPartitioners(void *lib, int verbose, mxnet::ext::msgSize_t msgSize,
                       mxnet::ext::msgGet_t msgGet) {
  using namespace mxnet::ext;

  // get C type interface functions
  opCallFree_t callFree = get_func<opCallFree_t>(lib, const_cast<char*>(MXLIB_OPCALLFREE_STR));

  partCallSupportedOps_t callSupportedOps =
    get_func<partCallSupportedOps_t>(lib, const_cast<char*>(MXLIB_PARTCALLSUPPORTEDOPS_STR));

  partCallCreateSelector_t callCreateSelector =
    get_func<partCallCreateSelector_t>(lib, const_cast<char*>(MXLIB_PARTCALLCREATESELECTOR_STR));

  partCallSelect_t callSelect =
    get_func<partCallSelect_t>(lib, const_cast<char*>(MXLIB_PARTCALLSELECT_STR));

  partCallSelectInput_t callSelectInput =
    get_func<partCallSelectInput_t>(lib, const_cast<char*>(MXLIB_PARTCALLSELECTINPUT_STR));

  partCallSelectOutput_t callSelectOutput =
    get_func<partCallSelectOutput_t>(lib, const_cast<char*>(MXLIB_PARTCALLSELECTOUTPUT_STR));

  partCallFilter_t callFilter =
    get_func<partCallFilter_t>(lib, const_cast<char*>(MXLIB_PARTCALLFILTER_STR));

  partCallReset_t callReset =
    get_func<partCallReset_t>(lib, const_cast<char*>(MXLIB_PARTCALLRESET_STR));

  partCallReviewSubgraph_t callReviewSubgraph =
    get_func<partCallReviewSubgraph_t>(lib, const_cast<char*>(MXLIB_PARTCALLREVIEWSUBGRAPH_STR));

  // get number of partitioners registered in the library
  partRegSize_t partRegSize = get_func<partRegSize_t>(lib,
                                                      const_cast<char*>(MXLIB_PARTREGSIZE_STR));
  int numParts = partRegSize();
  if (verbose) LOG(INFO) << "Found " << numParts << " partitioners in library";

  /*
   * Get all custom partitioners implementation from custom library
   * loop and register each partitioner in the library to NNVM
   */
  partRegGetCount_t partRegGetCount = get_func<partRegGetCount_t>(lib,
                                                  const_cast<char*>(MXLIB_PARTREGGETCOUNT_STR));
  partRegGet_t partRegGet = get_func<partRegGet_t>(lib, const_cast<char*>(MXLIB_PARTREGGET_STR));
  for (int i = 0; i < numParts; i++) {
    const char* name;
    // get custom partitioner strategy count from the dynamic library
    int count = partRegGetCount(i, &name);
    CHECK(count > 0) << "Error loading '" << name
                     << "' custom partitioner, no strategies defined";
    std::string name_str(name);
    if (verbose) LOG(INFO) << "\tPartitioner[" << i << "] " << name;

    mxnet::op::SubgraphBackendRegistry::Get()->__REGISTER_BACKEND__(name);

    for (int j = 0; j < count; j++) {
      const char* strategy;
      // function pointers holding implementation from custom library
      supportedOps_t supportedOps_fp = nullptr;
      createSelector_t createSelector_fp = nullptr;
      reviewSubgraph_t reviewSubgraph_fp = nullptr;
      // name of subgraph op
      const char* op_name = nullptr;

      // get custom partitioner strategy from the dynamic library
      partRegGet(i, j, &strategy, &supportedOps_fp, &createSelector_fp,
                 &reviewSubgraph_fp, &op_name);
      // validate custom partitioner functions from the dynamic library
      if (supportedOps_fp == nullptr && createSelector_fp == nullptr)
        LOG(ERROR) << "Error loading '" << name << "' custom partitioner strategy '"
                   << strategy << "', must implement supportedOps or createSelector";
      std::string strategy_str(strategy);
      std::string op_name_str(op_name);
      if (verbose) LOG(INFO) << "\t\tStrategy[" << j << "] " << strategy_str
                             << " subgraphOp: '" << op_name_str << "'";
      mxnet::op::SubgraphBackendRegistry::Get()->__REGISTER_CUSTOM_PROPERTY__
        (name_str, std::make_shared<mxnet::op::CustomSubgraphProperty>
         (strategy_str, callSupportedOps, supportedOps_fp, callCreateSelector,
          createSelector_fp, callSelect, callSelectInput, callSelectOutput,
          callFilter, callReset, callReviewSubgraph, reviewSubgraph_fp, callFree,
          op_name_str));
    }
  }
}

void registerPasses(void *lib, int verbose, mxnet::ext::msgSize_t msgSize,
                       mxnet::ext::msgGet_t msgGet) {
  using namespace mxnet::ext;

  // get C type interface functions
  opCallFree_t callFree = get_func<opCallFree_t>(lib, const_cast<char*>(MXLIB_OPCALLFREE_STR));

  passCallGraphPass_t callGraphPass =
    get_func<passCallGraphPass_t>(lib, const_cast<char*>(MXLIB_PASSCALLGRAPHPASS_STR));

  // get number of passes registered in the library
  partRegSize_t passRegSize = get_func<passRegSize_t>(lib,
                                                      const_cast<char*>(MXLIB_PASSREGSIZE_STR));
  int numPasses = passRegSize();
  if (verbose) LOG(INFO) << "Found " << numPasses << " graph passes in library";

  /*
   * Get all custom pass implementation from custom library
   * loop and register each pass in the library to NNVM
   */
  passRegGet_t passRegGet = get_func<passRegGet_t>(lib, const_cast<char*>(MXLIB_PASSREGGET_STR));
  for (int i = 0; i < numPasses; i++) {
    const char* name;
    // function pointers holding implementation from custom library
    graphPass_t pass_fp = nullptr;

    // main function to get custom pass implemenation from the custom library
    passRegGet(i, &pass_fp, &name);

    if (verbose) LOG(INFO) << "\tGraph Pass [" << i << "] " << name;

    auto pass_lambda = [=] (nnvm::Graph&& g) {
      // get pass name
      const char* pass_name = g.GetAttr<const char*>("pass_name");
      // get options
      const std::unordered_map<std::string, std::string>& options_map =
            g.GetAttr<const std::unordered_map<std::string, std::string>>("options_map");
      // convert options_map_ to char* to pass to backend library
      std::vector<const char*> opt_keys, opt_vals;
      for (auto& kv : options_map) {
        opt_keys.push_back(kv.first.c_str());
        opt_vals.push_back(kv.second.c_str());
      }

      // get input args and arg names
      std::vector<std::string> in_arg_names = g.GetAttr<std::vector<std::string>>("in_arg_names");
      std::vector<std::string> in_aux_names = g.GetAttr<std::vector<std::string>>("in_aux_names");
      NDArray **in_args_ptr = g.GetAttr<NDArray**>("in_args");
      NDArray **in_aux_ptr = g.GetAttr<NDArray**>("in_aux");

      // get shapes/types
      mxnet::ShapeVector shapes;
      if (g.HasAttr("shape"))
        shapes = g.GetAttr<mxnet::ShapeVector>("shape");
      std::vector<int> dtypes;
      if (g.HasAttr("dtype"))
        dtypes = g.GetAttr<std::vector<int> >("dtype");
      g.attrs.clear();
      const nnvm::IndexedGraph& indexed_graph = g.indexed_graph();

      // set shape attrs for each node in the graph
      if (shapes.size() > 0) {
        for (unsigned nid = 0; nid < indexed_graph.num_nodes(); nid++) {
          nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[nid].source);
          std::stringstream ss;
          ss << "[";
          // set the output shapes for this node
          for (unsigned oid = 0; oid < node->num_outputs(); oid++) {
            const uint32_t out_entry_id = indexed_graph.entry_id(nid, oid);
            mxnet::TShape& shape = shapes[out_entry_id];
            ss << shape;
            if (oid < node->num_outputs()-1) ss << ",";
          }
          ss << "]";
          node->attrs.dict[MX_STR_SHAPE] = ss.str();
        }
      }
      // set dtype attrs for each node in the graph
      if (dtypes.size() > 0) {
        for (unsigned nid = 0; nid < indexed_graph.num_nodes(); nid++) {
          nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[nid].source);
          std::stringstream ss;
          ss << "[";
          // set the output dtypes for this node
          for (unsigned oid = 0; oid < node->num_outputs(); oid++) {
            const uint32_t out_entry_id = indexed_graph.entry_id(nid, oid);
            int dtype = dtypes[out_entry_id];
            ss << dtype;
            if (oid < node->num_outputs()-1) ss << ",";
          }
          ss << "]";
          node->attrs.dict[MX_STR_DTYPE] = ss.str();
        }
      }

      std::vector<const char*> arg_names, aux_names;
      std::vector<void*> arg_data, aux_data;
      std::vector<const int64_t*> arg_shapes, aux_shapes;
      std::vector<int> arg_dims, aux_dims;
      std::vector<int> arg_types, aux_types;
      std::vector<size_t> arg_verIDs, aux_verIDs;
      std::vector<const char*> arg_dev_type, aux_dev_type;
      std::vector<int> arg_dev_id, aux_dev_id;

      // convert input args
      for (size_t i=0; i < in_arg_names.size(); i++) {
        if (in_args_ptr[i] != nullptr) {
          arg_names.push_back(in_arg_names[i].c_str());
          const NDArray &in_arg = *(in_args_ptr[i]);

#if MXNET_USE_MKLDNN == 1
          // reorder data if in MKLDNN format
          if (in_arg.IsMKLDNNData()) {
            in_arg.Reorder2DefaultAsync();
            in_arg.WaitToRead();
          }
#endif

          // pull out parts of NDArray to send to backend
          arg_data.push_back(in_arg.data().dptr_);
          arg_shapes.push_back(in_arg.shape().data());
          arg_dims.push_back(in_arg.shape().ndim());
          arg_types.push_back(in_arg.dtype());
          arg_verIDs.push_back(in_arg.version());
          const char* arg_ctx_str = in_arg.ctx().dev_mask() == Context::kCPU ? "cpu" : "gpu";
          arg_dev_type.push_back(arg_ctx_str);
          arg_dev_id.push_back(in_arg.ctx().real_dev_id());
        }
      }

      // convert input aux
      for (size_t i=0; i < in_aux_names.size(); i++) {
        if (in_aux_ptr[i] != nullptr) {
          aux_names.push_back(in_aux_names[i].c_str());
          const auto &in_aux = *(in_aux_ptr[i]);

#if MXNET_USE_MKLDNN == 1
          // reorder data if in MKLDNN format
          if (in_aux.IsMKLDNNData()) {
            in_aux.Reorder2DefaultAsync();
            in_aux.WaitToRead();
          }
#endif

          // pull out parts of NDArray to send to backend
          aux_data.push_back(in_aux.data().dptr_);
          aux_shapes.push_back(in_aux.shape().data());
          aux_dims.push_back(in_aux.shape().ndim());
          aux_types.push_back(in_aux.dtype());
          aux_verIDs.push_back(in_aux.version());
          const char* aux_ctx_str = in_aux.ctx().dev_mask() == Context::kCPU ? "cpu" : "gpu";
          aux_dev_type.push_back(aux_ctx_str);
          aux_dev_id.push_back(in_aux.ctx().real_dev_id());
        }
      }

      // convert graph to string
      std::string in_json = nnvm::pass::SaveJSON(g);

      std::vector<std::string> new_arg_names, new_aux_names;
      std::vector<NDArray*> new_args, new_aux;

      // create lambda that captures stream & resource objects
      // this temp workspace holds memory allocated by custom library via OpResource
      auto ndarray_alloc = [&](const mxnet::TShape &shape, Context ctx, int dtype,
                               std::string name, bool isArg) {
        NDArray* arr = new NDArray(shape, ctx, false, dtype);
        if (isArg) {
          new_args.push_back(arr);
          new_arg_names.push_back(name);
        } else {
          new_aux.push_back(arr);
          new_aux_names.push_back(name);
        }
        return arr;
      };

      // create no-capture lambda so that we can cast it to function pointer
      // lambda with captures cannot be cast to function pointer and pass to lib_api.h
      // this needs to be a lambda function so that we can do the decltype cast
      typedef decltype(ndarray_alloc) alloc_type_ndarray;
      auto ndarray_malloc = [](const void* _ndarray_alloc, const int64_t* shapes, int num_shapes,
                               const char* dev_str, int dev_id, int dtype, const char* name,
                               int isArg, void** data) {
        mxnet::TShape shape(num_shapes, 0);
        for (int i = 0; i < num_shapes; i++)
          shape[i] = shapes[i];
        int dev_type = -1;
        if (strcmp(dev_str, "cpu") == 0)
          dev_type = kCPU;
        else
          dev_type = kGPU;
        Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);

        // cast the void* argument to the type for the cpu_alloc lambda function
        const alloc_type_ndarray* ndalloc = static_cast<const alloc_type_ndarray*>(_ndarray_alloc);
        // call cpu_alloc to actually allocate memory and return the pointer
        NDArray* arr = (*ndalloc)(shape, ctx, dtype, name, isArg);
        *data = arr->data().dptr_;
      };

      char* out_json;
      int retval = callGraphPass(pass_fp, in_json.c_str(), &out_json, opt_keys.data(),
                                 opt_vals.data(), opt_keys.size(), pass_name,
                                 arg_names.data(), arg_names.size(), arg_data.data(),
                                 arg_shapes.data(), arg_dims.data(), arg_types.data(),
                                 arg_verIDs.data(), arg_dev_type.data(),
                                 arg_dev_id.data(), aux_names.data(), aux_names.size(),
                                 aux_data.data(), aux_shapes.data(), aux_dims.data(),
                                 aux_types.data(), aux_verIDs.data(),
                                 aux_dev_type.data(), aux_dev_id.data(),
                                 ndarray_malloc, &ndarray_alloc);
      std::string msgs = getExtensionMsgs(msgSize, msgGet);
      CHECK(retval) << "Error calling graph pass for '" << pass_name << "'" << msgs;

      std::string out_string(out_json);
      nnvm::Graph out_graph = nnvm::pass::LoadJSON(out_string);

      out_graph.attrs["new_args"] = std::make_shared<nnvm::any>(new_args);
      out_graph.attrs["new_arg_names"] = std::make_shared<nnvm::any>(new_arg_names);
      out_graph.attrs["new_aux"] = std::make_shared<nnvm::any>(new_aux);
      out_graph.attrs["new_aux_names"] = std::make_shared<nnvm::any>(new_aux_names);

      callFree(out_json);
      return out_graph;
    };

    nnvm::PassFunctionReg& pass = dmlc::Registry<nnvm::PassFunctionReg>::Get()->__REGISTER__(name);
    pass.set_body(pass_lambda);
    pass.set_change_graph(true);
  }
}

/*!
 * \brief Loads dynamic custom library and initializes it
 * \param path library path
 */
int MXLoadLib(const char *path, unsigned verbose) {
  API_BEGIN();
  void *lib = LibraryInitializer::Get()->lib_load(path);
  if (!lib)
    LOG(FATAL) << "Unable to load library";

  // check that library and MXNet use same version of library API
  mxnet::ext::opVersion_t opVersion =
    get_func<mxnet::ext::opVersion_t>(lib, const_cast<char*>(MXLIB_OPVERSION_STR));
  int libVersion =  opVersion();
  if (MX_LIBRARY_VERSION != libVersion)
    LOG(FATAL) << "Library version (" << libVersion << ") does not match MXNet version ("
               << MX_LIBRARY_VERSION << ")";

  // get error messaging APIs
  mxnet::ext::msgSize_t msgSize =
    get_func<mxnet::ext::msgSize_t>(lib, const_cast<char*>(MXLIB_MSGSIZE_STR));
  mxnet::ext::msgGet_t msgGet =
    get_func<mxnet::ext::msgGet_t>(lib, const_cast<char*>(MXLIB_MSGGET_STR));

  // initialize library by passing MXNet version
  mxnet::ext::initialize_t initialize =
    get_func<mxnet::ext::initialize_t>(lib, const_cast<char*>(MXLIB_INITIALIZE_STR));
  if (!initialize(static_cast<int>(MXNET_VERSION))) {
    std::string msgs = getExtensionMsgs(msgSize, msgGet);
    LOG(FATAL) << "Library failed to initialize" << msgs;
  }

  // find ops, partitioners, and passes in library
  registerOperators(lib, verbose, msgSize, msgGet);
  registerPartitioners(lib, verbose, msgSize, msgGet);
  registerPasses(lib, verbose, msgSize, msgGet);
  API_END();
}

int MXLibInfoFeatures(const struct LibFeature **lib_features, size_t *size) {
  using namespace features;
  API_BEGIN();
  LibInfo* lib_info = LibInfo::getInstance();
  *lib_features = lib_info->getFeatures().data();
  *size = lib_info->getFeatures().size();
  API_END();
}

int MXRandomSeed(int seed) {
  API_BEGIN();
  mxnet::RandomSeed(seed);
  API_END();
}

int MXRandomSeedContext(int seed, int dev_type, int dev_id) {
  API_BEGIN();
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  mxnet::RandomSeed(ctx, seed);
  API_END();
}

int MXNotifyShutdown() {
  API_BEGIN();
  mxnet::op::custom::CustomOperator::Get()->Stop();
  Engine::Get()->NotifyShutdown();
  Engine::Get()->WaitForAll();
  API_END();
}

int MXSetNumOMPThreads(int thread_num) {
  API_BEGIN();
  omp_set_num_threads(thread_num);
  API_END();
}

int MXEngineSetBulkSize(int bulk_size, int* prev_bulk_size) {
  API_BEGIN();
  *prev_bulk_size = Engine::Get()->set_bulk_size(bulk_size);
  API_END();
}

int MXGetGPUCount(int* out) {
  API_BEGIN();
  *out = Context::GetGPUCount();
  API_END();
}

// Deprecated: use MXGetGPUMemoryInformation64() instead.
int MXGetGPUMemoryInformation(int dev, int *free_mem, int *total_mem) {
  API_BEGIN();
  uint64_t free_mem64 = 0UL;
  uint64_t total_mem64 = 0UL;
  Context::GetGPUMemoryInformation(dev, &free_mem64, &total_mem64);
  *free_mem = static_cast<int>(free_mem64);
  *total_mem = static_cast<int>(total_mem64);
  API_END();
}

int MXGetGPUMemoryInformation64(int dev, uint64_t *free_mem, uint64_t *total_mem) {
  API_BEGIN();
  Context::GetGPUMemoryInformation(dev, free_mem, total_mem);
  API_END();
}

int MXGetVersion(int *out) {
  API_BEGIN();
  *out = static_cast<int>(MXNET_VERSION);
  API_END();
}

#if MXNET_USE_TVM_OP
int MXLoadTVMOp(const char *libpath) {
  API_BEGIN();
  tvm::runtime::TVMOpModule::Get()->Load(libpath);
  API_END();
}

int MXLoadTVMConfig(ConfigSpaces config) {
  API_BEGIN();
  for (int k = 0; k < config.spaces_size; ++k) {
    tvm::runtime::TVMOpConfig& entry = ::dmlc::Registry<tvm::runtime::TVMOpConfig>::Get()
      ->__REGISTER_OR_GET__(std::string(config.spaces_key[k]));
    const ConfigSpace& c = config.spaces_val[k];
    for (int i = 0; i < c.entity_map_size; ++i) {
      entry.add_entity(std::string(c.entity_map_key[i]), c.entity_map_val[i].val);
    }
    for (int i = 0; i < c.space_map_size; ++i) {
      std::string name = std::string(c.space_map_key[i]);
      std::vector<int> entities;
      for (int j = 0; j < c.space_map_val[i].entities_size; ++j) {
        int val = c.space_map_val[i].entities[j].val;
        entities.push_back(val);
      }
      entry.add_space(name, entities);
    }
  }
  API_END();
}

#endif  // MXNET_USE_TVM_OP

int MXNDArrayCreateNone(NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray();
  API_END();
}

template<typename DataType>
void CreateNDArray(const DataType* shape,
                   int ndim,
                   int dev_type,
                   int dev_id,
                   int delay_alloc,
                   int dtype,
                   NDArrayHandle* out) {
  mxnet::TShape requested_shape = mxnet::TShape(shape, shape + ndim);
  if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
    CHECK_LT(requested_shape.Size(), (int64_t{1} << 31) - 1) <<
              "[CreateNDArray] Size of tensor you are trying to allocate is larger than "
              "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
  }
  *out = new NDArray(requested_shape,
                     Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id),
                     delay_alloc != 0, dtype);
}

int MXNDArrayCreate(const uint32_t *shape,
                    uint32_t ndim,
                    int dev_type,
                    int dev_id,
                    int delay_alloc,
                    NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(mxnet::TShape(shape, shape + ndim),
                     Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id),
                     delay_alloc != 0);
  API_END();
}

int MXNDArrayCreateEx64(const int64_t *shape,
                        int ndim,
                        int dev_type,
                        int dev_id,
                        int delay_alloc,
                        int dtype,
                        NDArrayHandle *out) {
  API_BEGIN();
  CreateNDArray<int64_t>(shape, ndim, dev_type, dev_id, delay_alloc, dtype, out);
  API_END();
}

int MXNDArrayCreateEx(const uint32_t *shape,
                      uint32_t ndim,
                      int dev_type,
                      int dev_id,
                      int delay_alloc,
                      int dtype,
                      NDArrayHandle *out) {
  API_BEGIN();
  CreateNDArray<uint32_t>(shape, static_cast<int>(ndim), dev_type, dev_id, delay_alloc, dtype, out);
  API_END();
}

template<typename DType>
void CreateSparseNDArray(int storage_type,
                         const DType *shape,
                         int ndim,
                         int dev_type,
                         int dev_id,
                         int delay_alloc,
                         int dtype,
                         uint32_t num_aux,
                         int *aux_type,
                         int *aux_ndims,
                         const DType *aux_shape,
                         NDArrayHandle *out) {
  std::vector<int> aux_types;
  mxnet::ShapeVector aux_shapes;
  auto shape_start = aux_shape;
  for (size_t i = 0; i < num_aux; i++) {
    // types
    aux_types.push_back(aux_type[i]);
    // shapes
    aux_shapes.emplace_back(shape_start, shape_start + aux_ndims[i]);
    shape_start += aux_ndims[i];
  }
  *out = new NDArray(
      NDArrayStorageType(storage_type),
      mxnet::TShape(shape, shape + ndim),
      Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id),
      delay_alloc != 0,
      dtype, aux_types, aux_shapes);
}

int MXNDArrayCreateSparseEx(int storage_type,
                            const uint32_t *shape,
                            uint32_t ndim,
                            int dev_type,
                            int dev_id,
                            int delay_alloc,
                            int dtype,
                            uint32_t num_aux,
                            int *aux_type,
                            uint32_t *aux_ndims,
                            const uint32_t *aux_shape,
                            NDArrayHandle *out) {
  API_BEGIN();
  CreateSparseNDArray<uint32_t>(storage_type, shape, static_cast<int>(ndim), dev_type, dev_id,
                                delay_alloc, dtype, num_aux, aux_type,
                                reinterpret_cast<int *>(aux_ndims), aux_shape, out);
  API_END();
}


int MXNDArrayCreateSparseEx64(int storage_type,
                            const int64_t *shape,
                            int ndim,
                            int dev_type,
                            int dev_id,
                            int delay_alloc,
                            int dtype,
                            uint32_t num_aux,
                            int *aux_type,
                            int *aux_ndims,
                            const int64_t *aux_shape,
                            NDArrayHandle *out) {
  API_BEGIN();
  CreateSparseNDArray<int64_t>(storage_type, shape, static_cast<int>(ndim), dev_type, dev_id,
                               delay_alloc, dtype, num_aux, aux_type,
                               reinterpret_cast<int *>(aux_ndims), aux_shape, out);
  API_END();
}


int MXNDArrayLoadFromRawBytes(const void *buf,
                              size_t size,
                              NDArrayHandle *out) {
  NDArray *ptr = nullptr;
  API_BEGIN();
  dmlc::MemoryFixedSizeStream strm((void*)buf, size); // NOLINT(*)
  ptr = new NDArray();
  if (!ptr->Load(&strm)) {
    throw dmlc::Error("Invalid NDArray serialization format");
  }
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArraySaveRawBytes(NDArrayHandle handle,
                          size_t *out_size,
                          const char **out_buf) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  ret->ret_str.resize(0);
  dmlc::MemoryStringStream strm(&ret->ret_str);
  static_cast<NDArray*>(handle)->Save(&strm);
  *out_size = ret->ret_str.length();
  *out_buf = ret->ret_str.c_str();
  API_END();
}

int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,
                             const void *data,
                             size_t size) {
  API_BEGIN();
  static_cast<NDArray*>(handle)->SyncCopyFromCPU(data, size);
  API_END();
}

int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
                           void *data,
                           size_t size) {
  API_BEGIN();
  static_cast<NDArray*>(handle)->SyncCopyToCPU(data, size);
  API_END();
}

/*!
 * \brief Copy src.data() to dst.data() if i = -1, else dst.aux_data(i) if i >= 0
 * This function blocks. Do not use it in performance critical code.
 * \param handle_dst handle of a dst ndarray whose data/aux_data has been allocated
 * \param handle_src handle of a src ndarray which has default storage type
 * \param i dst data blob indicator
 */
int MXNDArraySyncCopyFromNDArray(NDArrayHandle handle_dst,
                                 const NDArrayHandle handle_src,
                                 const int i) {
  API_BEGIN();
  NDArray* dst = static_cast<NDArray*>(handle_dst);
  NDArray* src = static_cast<NDArray*>(handle_src);
  dst->SyncCopyFromNDArray(*src, -1, i);
  API_END();
}

int MXNDArraySyncCheckFormat(NDArrayHandle handle, const bool full_check) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  arr->SyncCheckFormat(full_check);
  API_END();
}

int MXNDArrayWaitToRead(NDArrayHandle handle) {
  API_BEGIN();
  static_cast<NDArray*>(handle)->WaitToRead();
  API_END();
}

int MXNDArrayWaitToWrite(NDArrayHandle handle) {
  API_BEGIN();
  static_cast<NDArray*>(handle)->WaitToWrite();
  API_END();
}

int MXNDArrayWaitAll() {
  API_BEGIN();
  Engine::Get()->WaitForAll();
  API_END();
}

int MXNDArraySave(const char* fname,
                  uint32_t num_args,
                  NDArrayHandle* args,
                  const char** keys) {
  API_BEGIN();
  std::vector<NDArray> data(num_args);
  std::vector<std::string> names;
  for (uint32_t i = 0; i < num_args; ++i) {
    data[i] = *static_cast<NDArray*>(args[i]);
  }
  if (keys != nullptr) {
    names.resize(num_args);
    for (uint32_t i = 0; i < num_args; ++i) {
      names[i] = keys[i];
    }
  }
  {
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname, "w"));
    mxnet::NDArray::Save(fo.get(), data, names);
  }
  API_END();
}

int MXNDArrayLoad(const char* fname,
                  uint32_t *out_size,
                  NDArrayHandle** out_arr,
                  uint32_t *out_name_size,
                  const char*** out_names) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  ret->ret_vec_str.clear();
  API_BEGIN();
  std::vector<NDArray> data;
  std::vector<std::string> &names = ret->ret_vec_str;
  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname, "r"));
    mxnet::NDArray::Load(fi.get(), &data, &names);
  }
  ret->ret_handles.resize(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    NDArray *ptr = new NDArray();
    *ptr = data[i];
    ret->ret_handles[i] = ptr;
  }
  ret->ret_vec_charp.resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    ret->ret_vec_charp[i] = names[i].c_str();
  }
  *out_size = static_cast<uint32_t>(data.size());
  *out_arr = dmlc::BeginPtr(ret->ret_handles);
  *out_name_size = static_cast<uint32_t>(names.size());
  *out_names = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXNDArrayLoadFromBuffer(const void *ndarray_buffer,
                            size_t size,
                            uint32_t *out_size,
                            NDArrayHandle** out_arr,
                            uint32_t *out_name_size,
                            const char*** out_names) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  ret->ret_vec_str.clear();
  API_BEGIN();
  CHECK_NOTNULL(ndarray_buffer);
  std::vector<NDArray> data;
  std::vector<std::string> &names = ret->ret_vec_str;
  {
    std::unique_ptr<dmlc::MemoryFixedSizeStream> fi(new dmlc::MemoryFixedSizeStream(
        const_cast<void*>(ndarray_buffer), size));
    mxnet::NDArray::Load(fi.get(), &data, &names);
  }
  ret->ret_handles.resize(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    NDArray *ptr = new NDArray();
    *ptr = data[i];
    ret->ret_handles[i] = ptr;
  }
  ret->ret_vec_charp.resize(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    ret->ret_vec_charp[i] = names[i].c_str();
  }
  *out_size = static_cast<uint32_t>(data.size());
  *out_arr = dmlc::BeginPtr(ret->ret_handles);
  *out_name_size = static_cast<uint32_t>(names.size());
  *out_names = dmlc::BeginPtr(ret->ret_vec_charp);
  API_END();
}

int MXNDArrayFree(NDArrayHandle handle) {
  API_BEGIN();
  delete static_cast<NDArray*>(handle);
  API_END();
}

template<typename dtype>
void SliceArray(NDArrayHandle handle, dtype slice_begin, dtype slice_end, NDArray* ptr,
                NDArrayHandle* out) {
  *ptr = static_cast<NDArray*>(handle)->SliceWithRecord(slice_begin, slice_end);
  *out = ptr;
}

int MXNDArraySlice(NDArrayHandle handle,
                   uint32_t slice_begin,
                   uint32_t slice_end,
                   NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  SliceArray<uint32_t>(handle, slice_begin, slice_end, ptr, out);
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArraySlice64(NDArrayHandle handle,
                     int64_t slice_begin,
                     int64_t slice_end,
                     NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  SliceArray<int64_t>(handle, slice_begin, slice_end, ptr, out);
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayAt(NDArrayHandle handle,
                uint32_t idx,
                NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  *ptr = static_cast<NDArray*>(handle)->AtWithRecord(idx);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayAt64(NDArrayHandle handle,
                  int64_t idx,
                  NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  *ptr = static_cast<NDArray*>(handle)->AtWithRecord(idx);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

MXNET_DLL int MXNDArrayReshape(NDArrayHandle handle,
                               int ndim,
                               int *dims,
                               NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  mxnet::TShape new_shape(dims, dims+ndim);
  int size = 1;
  int pos = -1;
  for (int i = 0; i < ndim; ++i) {
    int dim = dims[i];
    if (dim == -1) {
      CHECK_EQ(pos, -1)
        << "Invalid new shape " << new_shape
        << ": more than one dimensions are -1";
      pos = i;
    } else {
      if (dim == 0) {
        CHECK_LT(i, arr->shape().ndim())
          << "Invalid new shape " << new_shape
          << ": 0 dimension exceeds original shape " << arr->shape();
        dim = arr->shape()[i];
      }
      size *= dim;
      new_shape[i] = dim;
    }
  }
  if (pos >= 0) {
    new_shape[pos] = arr->shape().Size() / size;
  }
  *ptr = arr->ReshapeWithRecord(new_shape);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

MXNET_DLL int MXNDArrayReshape64(NDArrayHandle handle,
                                 int ndim,
                                 dim_t *dims,
                                 bool reverse,
                                 NDArrayHandle *out) {
  NDArray *ptr = new NDArray();
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  mxnet::Tuple<dim_t> shape(dims, dims+ndim);
  mxnet::TShape new_shape = mxnet::op::InferReshapeShape(shape, arr->shape(), reverse);
  *ptr = arr->ReshapeWithRecord(new_shape);
  *out = ptr;
  API_END_HANDLE_ERROR(delete ptr);
}

int MXNDArrayGetStorageType(NDArrayHandle handle,
                     int *out_storage_type) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    *out_storage_type = arr->storage_type();
  } else {
    *out_storage_type = kUndefinedStorage;
  }
  API_END();
}

int MXNDArrayGetShape(NDArrayHandle handle,
                      uint32_t *out_dim,
                      const uint32_t **out_pdata) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    const mxnet::TShape &s = arr->shape();
    *out_dim = s.ndim();
    std::vector<uint32_t>& buffer = ret->arg_shape_buffer;
    buffer.resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), buffer.data());
    *out_pdata = buffer.data();
  } else {
    *out_dim = 0;
  }
  API_END();
}

template<typename dtype>
inline void GetShape(NDArrayHandle handle, const dtype** out_pdata, int* out_dim,
                     MXAPIThreadLocalEntry<dtype>* ret) {
  NDArray* arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    if (!features::is_enabled(features::INT64_TENSOR_SIZE)) {
      CHECK_LT(arr->shape().Size(), (int64_t{1} << 31) - 1) <<
                      "[Get Shape] Size of tensor you are trying to allocate is larger than "
                      "2^31 elements. Please build with flag USE_INT64_TENSOR_SIZE=1";
    }
    mxnet::TShape s = arr->shape();
    if (!Imperative::Get()->is_np_shape()) {
      common::ConvertToLegacyShape(&s);
    }
    *out_dim = s.ndim();
    if (s.ndim() >= 0) {
      std::vector<dtype> &buffer = ret->arg_shape_buffer_ex;
      buffer.resize(s.ndim());
      mxnet::ShapeTypeCast(s.begin(), s.end(), buffer.data());
      *out_pdata = buffer.data();
    }
  } else {
    if (Imperative::Get()->is_np_shape()) {
      *out_dim = -1;
    } else {
      *out_dim = 0;
    }
  }
}

int MXNDArrayGetShapeEx(NDArrayHandle handle,
                        int *out_dim,
                        const int **out_pdata) {
  MXAPIThreadLocalEntry<> *ret = MXAPIThreadLocalStore<>::Get();
  API_BEGIN();
  GetShape<int>(handle, out_pdata, out_dim, ret);
  API_END();
}

int MXNDArrayGetShapeEx64(NDArrayHandle handle,
                          int *out_dim,
                          const int64_t **out_pdata) {
  MXAPIThreadLocalEntry<int64_t> *ret = MXAPIThreadLocalStore<int64_t>::Get();
  API_BEGIN();
  GetShape<int64_t>(handle, out_pdata, out_dim, ret);
  API_END();
}

int MXNDArrayGetData(NDArrayHandle handle,
                     void **out_pdata) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
#if MXNET_USE_MKLDNN == 1
  if (arr->IsMKLDNNData()) {
    arr->Reorder2DefaultAsync();
    arr->WaitToRead();
  }
#endif
  if (!arr->is_none()) {
    *out_pdata = arr->data().dptr_;
  } else {
    *out_pdata = nullptr;
  }
  API_END();
}

int MXNDArrayToDLPack(NDArrayHandle handle,
                      DLManagedTensorHandle *out_dlpack) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out_dlpack = arr->ToDLPack();
  API_END();
}

int MXNDArrayFromDLPack(DLManagedTensorHandle dlpack,
                        NDArrayHandle *out_handle) {
  return MXNDArrayFromDLPackEx(dlpack, false, out_handle);
}

int MXNDArrayFromDLPackEx(DLManagedTensorHandle dlpack,
                          const bool transient_handle,
                          NDArrayHandle *out_handle) {
  API_BEGIN();
  *out_handle = new NDArray(NDArray::FromDLPack(
              static_cast<DLManagedTensor*>(dlpack),
              transient_handle));
  API_END();
}

int MXNDArrayCallDLPackDeleter(DLManagedTensorHandle dlpack) {
  API_BEGIN();
  if (dlpack != nullptr) {
    DLManagedTensor *p_dlpack = static_cast<DLManagedTensor*>(dlpack);
    p_dlpack->deleter(p_dlpack);
  }
  API_END();
}

int MXNDArrayGetDType(NDArrayHandle handle,
                     int *out_dtype) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    *out_dtype = arr->dtype();
  } else {
    *out_dtype = -1;
  }
  API_END();
}

int MXNDArrayGetAuxType(NDArrayHandle handle,
                        uint32_t i,
                        int *out_type) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out_type = arr->aux_type(i);
  API_END();
}

/*!
 * \brief Get a deep copy of the ith aux data blob
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
int MXNDArrayGetAuxNDArray(NDArrayHandle handle,
                           uint32_t i,
                           NDArrayHandle *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out = new NDArray(arr->aux_ndarray(i));
  API_END();
}

/*!
 * \brief Get a deep copy of the data blob
 * in the form of an NDArray of default storage type.
 * This function blocks. Do not use it in performance critical code.
 */
int MXNDArrayGetDataNDArray(NDArrayHandle handle,
                            NDArrayHandle *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out = new NDArray(arr->data_ndarray());
  API_END();
}

int MXNDArrayGetContext(NDArrayHandle handle,
                        int *out_dev_type,
                        int *out_dev_id) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  if (!arr->is_none()) {
    const Context &ctx = arr->ctx();
    *out_dev_type = ctx.dev_type;
    *out_dev_id = ctx.dev_id;
  } else {
    *out_dev_type = 0;
    *out_dev_id = 0;
  }
  API_END();
}


int MXNDArrayGetGrad(NDArrayHandle handle, NDArrayHandle *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  NDArray ret = arr->grad();
  if (ret.is_none()) {
    *out = nullptr;
  } else {
    *out = new NDArray(ret);
  }
  API_END();
}

int MXNDArrayDetach(NDArrayHandle handle, NDArrayHandle *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out = new NDArray(arr->Detach());
  API_END();
}

int MXNDArraySetGradState(NDArrayHandle handle, int state) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  arr->set_fresh_out_grad(static_cast<bool>(state));
  API_END();
}

int MXNDArrayGetGradState(NDArrayHandle handle, int *out) {
  API_BEGIN();
  NDArray *arr = static_cast<NDArray*>(handle);
  *out = arr->fresh_out_grad();
  API_END();
}

int MXListFunctions(uint32_t *out_size,
                    FunctionHandle **out_array) {
  API_BEGIN();
  auto &vec = dmlc::Registry<NDArrayFunctionReg>::List();
  *out_size = static_cast<uint32_t>(vec.size());
  *out_array = (FunctionHandle*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXGetFunction(const char *name,
                  FunctionHandle *out) {
  API_BEGIN();
  *out = dmlc::Registry<NDArrayFunctionReg>::Find(name);
  API_END();
}

int MXFuncGetInfo(FunctionHandle fun,
                  const char **name,
                  const char **description,
                  uint32_t *num_args,
                  const char ***arg_names,
                  const char ***arg_type_infos,
                  const char ***arg_descriptions,
                  const char **return_type) {
  return MXAPIGetFunctionRegInfo(static_cast<const NDArrayFunctionReg *>(fun),
                                 name, description, num_args,
                                 arg_names, arg_type_infos, arg_descriptions,
                                 return_type);
}

int MXFuncDescribe(FunctionHandle fun,
                   uint32_t *num_use_vars,
                   uint32_t *num_scalars,
                   uint32_t *num_mutate_vars,
                   int *type_mask) {
  API_BEGIN();
  auto *f = static_cast<const NDArrayFunctionReg*>(fun);
  *num_use_vars = f->num_use_vars;
  *num_scalars = f->num_scalars;
  *num_mutate_vars = f->num_mutate_vars;
  *type_mask = f->type_mask;
  API_END();
}

int MXFuncInvoke(FunctionHandle fun,
                 NDArrayHandle *use_vars,
                 float *scalar_args,
                 NDArrayHandle *mutate_vars) {
  API_BEGIN();
  auto *f = static_cast<const NDArrayFunctionReg*>(fun);
  f->body((NDArray**)(use_vars),  //  NOLINT(*)
          scalar_args,
          (NDArray**)(mutate_vars),  //  NOLINT(*)
          0,
          nullptr,
          nullptr);
  API_END();
}

int MXFuncInvokeEx(FunctionHandle fun,
                 NDArrayHandle *use_vars,
                 float *scalar_args,
                 NDArrayHandle *mutate_vars,
                 int num_params,
                 char **param_keys,
                 char **param_vals) {
  API_BEGIN();
  auto *f = static_cast<const NDArrayFunctionReg*>(fun);
  f->body((NDArray**)(use_vars),  //  NOLINT(*)
          scalar_args,
          (NDArray**)(mutate_vars),  //  NOLINT(*)
          num_params,
          param_keys,
          param_vals);
  API_END();
}

//--------------------------------------------
// Part 5: IO Interface
//--------------------------------------------
int MXListDataIters(uint32_t *out_size,
                    DataIterCreator **out_array) {
  API_BEGIN();
  auto &vec = dmlc::Registry<DataIteratorReg>::List();
  *out_size = static_cast<uint32_t>(vec.size());
  *out_array = (DataIterCreator*)(dmlc::BeginPtr(vec));  //  NOLINT(*)
  API_END();
}

int MXDataIterGetIterInfo(DataIterCreator creator,
                          const char **name,
                          const char **description,
                          uint32_t *num_args,
                          const char ***arg_names,
                          const char ***arg_type_infos,
                          const char ***arg_descriptions) {
  DataIteratorReg *e = static_cast<DataIteratorReg *>(creator);
  return MXAPIGetFunctionRegInfo(e, name, description, num_args,
                                 arg_names, arg_type_infos, arg_descriptions,
                                 nullptr);
}

int MXDataIterCreateIter(DataIterCreator creator,
                         uint32_t num_param,
                         const char **keys,
                         const char **vals,
                         DataIterHandle *out) {
  IIterator<DataBatch> *iter = nullptr;
  API_BEGIN();
  DataIteratorReg *e = static_cast<DataIteratorReg *>(creator);
  iter = e->body();
  std::vector<std::pair<std::string, std::string> > kwargs;
  for (uint32_t i = 0; i < num_param; ++i) {
    kwargs.push_back({std::string(keys[i]), std::string(vals[i])});
  }
  iter->Init(kwargs);
  *out = iter;
  API_END_HANDLE_ERROR(delete iter);
}

int MXDataIterFree(DataIterHandle handle) {
  API_BEGIN();
  delete static_cast<IIterator<DataBatch> *>(handle);
  API_END();
}

int MXDataIterBeforeFirst(DataIterHandle handle) {
  API_BEGIN();
  static_cast<IIterator<DataBatch>* >(handle)->BeforeFirst();
  API_END();
}

int MXDataIterNext(DataIterHandle handle, int *out) {
  API_BEGIN();
  *out = static_cast<IIterator<DataBatch>* >(handle)->Next();
  API_END();
}

int MXDataIterGetLabel(DataIterHandle handle, NDArrayHandle *out) {
  API_BEGIN();
  const DataBatch& db = static_cast<IIterator<DataBatch>* >(handle)->Value();
  NDArray* pndarray = new NDArray();
  // temp hack to make label 1D
  // TODO(tianjun) make label 1D when label_width=0
  mxnet::TShape shape = db.data[1].shape();
  if (shape.ndim() > 1 && shape[1] == 1) {
    *pndarray = db.data[1].Reshape(mshadow::Shape1(shape[0]));
  } else {
    *pndarray = db.data[1];
  }
  *out = pndarray;
  API_END();
}

int MXDataIterGetIndex(DataIterHandle handle, uint64_t **out_index, uint64_t *out_size) {
  API_BEGIN();
  const DataBatch& db = static_cast<IIterator<DataBatch>* >(handle)->Value();
  *out_size = db.index.size();
  *out_index = const_cast<uint64_t*>(db.index.data());
  API_END();
}

int MXDataIterGetData(DataIterHandle handle, NDArrayHandle *out) {
  API_BEGIN();
  const DataBatch& db = static_cast<IIterator<DataBatch>* >(handle)->Value();
  NDArray* pndarray = new NDArray();
  *pndarray = db.data[0];
  *out = pndarray;
  API_END();
}

int MXDataIterGetPadNum(DataIterHandle handle, int *pad) {
  API_BEGIN();
  const DataBatch& db = static_cast<IIterator<DataBatch>* >(handle)->Value();
  *pad = db.num_batch_padd;
  API_END();
}

int MXKVStoreCreate(const char *type,
                    KVStoreHandle *out) {
  API_BEGIN();
  *out = KVStore::Create(type);
  API_END();
}

int MXKVStoreSetGradientCompression(KVStoreHandle handle, uint32_t num_params,
                                    const char** keys, const char** vals) {
  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > params;
  for (uint32_t i = 0; i < num_params; ++i) {
    std::pair<std::string, std::string> p;
    p.first = keys[i];
    p.second = vals[i];
    params.push_back(p);
  }
  static_cast<KVStore*>(handle)->SetGradientCompression(params);
  API_END();
}

int MXKVStoreFree(KVStoreHandle handle) {
  API_BEGIN();
  delete static_cast<KVStore*>(handle);
  API_END();
}

int MXKVStoreInit(KVStoreHandle handle,
                  uint32_t num,
                  const int* keys,
                  NDArrayHandle* vals) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Init(v_keys, v_vals);
  API_END();
}

int MXKVStoreInitEx(KVStoreHandle handle,
                  uint32_t num,
                  const char** keys,
                  NDArrayHandle* vals) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Init(v_keys, v_vals);
  API_END();
}

int MXKVStorePush(KVStoreHandle handle,
                  uint32_t num,
                  const int* keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Push(v_keys, v_vals, priority);
  API_END();
}

int MXKVStorePushEx(KVStoreHandle handle,
                  uint32_t num,
                  const char** keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Push(v_keys, v_vals, priority);
  API_END();
}

int MXKVStorePull(KVStoreHandle handle,
                  uint32_t num,
                  const int* keys,
                  NDArrayHandle* vals,
                  int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority, true);
  API_END();
}

int MXKVStorePullEx(KVStoreHandle handle,
                    uint32_t num,
                    const char** keys,
                    NDArrayHandle* vals,
                    int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority, true);
  API_END();
}

int MXKVStoreBroadcast(KVStoreHandle handle,
                       mx_uint vnum,
                       const int* vkeys,
                       mx_uint onum,
                       const int* okeys,
                       NDArrayHandle* vals,
                       NDArrayHandle* outs,
                       int priority) {
  API_BEGIN();
  std::vector<int> v_vkeys(vnum);
  std::vector<int> v_okeys(onum);
  std::vector<NDArray> v_vals(vnum);
  std::vector<NDArray*> v_outs(onum);
  for (mx_uint i = 0; i < vnum; ++i) {
    v_vkeys[i] = vkeys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  for (mx_uint i = 0; i < onum; ++i) {
    v_okeys[i] = okeys[i];
    v_outs[i] = static_cast<NDArray*>(outs[i]);
  }
  static_cast<KVStore*>(handle)->Broadcast(v_vkeys, v_okeys, v_vals, v_outs,
    priority);
  API_END();
}

int MXKVStoreBroadcastEx(KVStoreHandle handle,
                         mx_uint vnum,
                         const char** vkeys,
                         mx_uint onum,
                         const char** okeys,
                         NDArrayHandle* vals,
                         NDArrayHandle* outs,
                         int priority) {
  API_BEGIN();
  std::vector<std::string> v_vkeys(vnum);
  std::vector<std::string> v_okeys(onum);
  std::vector<NDArray> v_vals(vnum);
  std::vector<NDArray*> v_outs(onum);
  for (mx_uint i = 0; i < vnum; ++i) {
    v_vkeys[i] = vkeys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  for (mx_uint i = 0; i < onum; ++i) {
    v_okeys[i] = okeys[i];
    v_outs[i] = static_cast<NDArray*>(outs[i]);
  }
  static_cast<KVStore*>(handle)->Broadcast(v_vkeys, v_okeys, v_vals, v_outs,
    priority);
  API_END();
}

int MXKVStorePushPull(KVStoreHandle handle,
                      mx_uint vnum,
                      const int* vkeys,
                      mx_uint onum,
                      const int* okeys,
                      NDArrayHandle* vals,
                      NDArrayHandle* outs,
                      int priority) {
  API_BEGIN();
  std::vector<int> v_vkeys(vnum);
  std::vector<int> v_okeys(onum);
  std::vector<NDArray> v_vals(vnum);
  std::vector<NDArray*> v_outs(onum);
  for (mx_uint i = 0; i < vnum; ++i) {
    v_vkeys[i] = vkeys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  for (mx_uint i = 0; i < onum; ++i) {
    v_okeys[i] = okeys[i];
    v_outs[i] = static_cast<NDArray*>(outs[i]);
  }
  static_cast<KVStore*>(handle)->PushPull(v_vkeys, v_okeys, v_vals, v_outs,
    priority);
  API_END();
}

int MXKVStorePushPullEx(KVStoreHandle handle,
                        mx_uint vnum,
                        const char** vkeys,
                        mx_uint onum,
                        const char** okeys,
                        NDArrayHandle* vals,
                        NDArrayHandle* outs,
                        int priority) {
  API_BEGIN();
  std::vector<std::string> v_vkeys(vnum);
  std::vector<std::string> v_okeys(onum);
  std::vector<NDArray> v_vals(vnum);
  std::vector<NDArray*> v_outs(onum);
  for (mx_uint i = 0; i < vnum; ++i) {
    v_vkeys[i] = vkeys[i];
    v_vals[i] = *static_cast<NDArray*>(vals[i]);
  }
  for (mx_uint i = 0; i < onum; ++i) {
    v_okeys[i] = okeys[i];
    v_outs[i] = static_cast<NDArray*>(outs[i]);
  }
  static_cast<KVStore*>(handle)->PushPull(v_vkeys, v_okeys, v_vals, v_outs,
    priority);
  API_END();
}

int MXKVStorePullWithSparse(KVStoreHandle handle,
                            uint32_t num,
                            const int* keys,
                            NDArrayHandle* vals,
                            int priority,
                            bool ignore_sparse) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority, ignore_sparse);
  API_END();
}

int MXKVStorePullWithSparseEx(KVStoreHandle handle,
                              uint32_t num,
                              const char** keys,
                              NDArrayHandle* vals,
                              int priority,
                              bool ignore_sparse) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<NDArray*> v_vals(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_vals[i] = static_cast<NDArray*>(vals[i]);
  }
  static_cast<KVStore*>(handle)->Pull(v_keys, v_vals, priority, ignore_sparse);
  API_END();
}

int MXKVStorePullRowSparse(KVStoreHandle handle,
                           uint32_t num,
                           const int* keys,
                           NDArrayHandle* vals,
                           const NDArrayHandle* row_ids,
                           int priority) {
  API_BEGIN();
  std::vector<int> v_keys(num);
  std::vector<std::pair<NDArray*, NDArray>> v_val_rowids(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_val_rowids[i] = std::make_pair(static_cast<NDArray*>(vals[i]),
                                     *static_cast<NDArray*>(row_ids[i]));
  }
  static_cast<KVStore*>(handle)->PullRowSparse(v_keys, v_val_rowids, priority);
  API_END();
}

int MXKVStorePullRowSparseEx(KVStoreHandle handle,
                             uint32_t num,
                             const char** keys,
                             NDArrayHandle* vals,
                             const NDArrayHandle* row_ids,
                             int priority) {
  API_BEGIN();
  std::vector<std::string> v_keys(num);
  std::vector<std::pair<NDArray*, NDArray>> v_val_rowids(num);
  for (uint32_t i = 0; i < num; ++i) {
    v_keys[i] = keys[i];
    v_val_rowids[i] = std::make_pair(static_cast<NDArray*>(vals[i]),
                                     *static_cast<NDArray*>(row_ids[i]));
  }
  static_cast<KVStore*>(handle)->PullRowSparse(v_keys, v_val_rowids, priority);
  API_END();
}

void MXKVStoreSetUpdaterImpl(KVStoreHandle handle,
                             MXKVStoreUpdater updater,
                             void* updater_handle) {
  MXKVStoreUpdater * updater_temp = updater;
  void* updater_handle_temp = updater_handle;
  std::function<void(int, const NDArray&, NDArray*)> updt
  = [updater_temp, updater_handle_temp](int key, const NDArray& recv, NDArray* local) {
    NDArray* recv_copy = new NDArray();
    *recv_copy = recv;
    NDArray* local_copy = new NDArray();
    *local_copy = *local;
    updater_temp(key, recv_copy, local_copy, updater_handle_temp);
  };
  static_cast<KVStore*>(handle)->set_updater(updt);
}

int MXKVStoreSetUpdater(KVStoreHandle handle,
                        MXKVStoreUpdater updater,
                        void* updater_handle) {
  API_BEGIN();
  MXKVStoreSetUpdaterImpl(handle, updater, updater_handle);
  API_END();
}

int MXKVStoreSetUpdaterEx(KVStoreHandle handle,
                          MXKVStoreUpdater updater,
                          MXKVStoreStrUpdater str_updater,
                          void* updater_handle) {
  API_BEGIN();
  // set updater with int keys
  MXKVStoreSetUpdaterImpl(handle, updater, updater_handle);
  // set updater with string keys
  MXKVStoreStrUpdater * updater_temp = str_updater;
  void* updater_handle_temp = updater_handle;
  std::function<void(const std::string&, const NDArray&, NDArray*)> updt
  = [updater_temp, updater_handle_temp]
    (const std::string& key, const NDArray& recv, NDArray* local) {
    NDArray* recv_copy = new NDArray();
    *recv_copy = recv;
    NDArray* local_copy = new NDArray();
    *local_copy = *local;
    updater_temp(key.c_str(), recv_copy, local_copy, updater_handle_temp);
  };
  static_cast<KVStore*>(handle)->set_updater(updt);
  API_END();
}

int MXKVStoreGetRank(KVStoreHandle handle, int *rank) {
  API_BEGIN();
  *rank = static_cast<KVStore*>(handle)->get_rank();
  API_END();
}

int MXKVStoreGetGroupSize(KVStoreHandle handle, int *size) {
  API_BEGIN();
  *size = static_cast<KVStore*>(handle)->get_group_size();
  API_END();
}

int MXKVStoreBarrier(KVStoreHandle handle) {
  API_BEGIN();
  static_cast<KVStore*>(handle)->Barrier();
  API_END();
}

int MXKVStoreSetBarrierBeforeExit(KVStoreHandle handle,
                                  const int barrier_before_exit) {
  API_BEGIN();
  static_cast<KVStore*>(handle)->set_barrier_before_exit(barrier_before_exit);
  API_END();
}

int MXInitPSEnv(uint32_t num_vars,
                const char **keys,
                const char **vals) {
  API_BEGIN();
  std::unordered_map<std::string, std::string> kwargs;
  for (uint32_t i = 0; i < num_vars; ++i) {
    kwargs[std::string(keys[i])] = std::string(vals[i]);
  }
  KVStore::InitPSEnv(kwargs);
  API_END();
}

int MXKVStoreIsWorkerNode(int *ret) {
  API_BEGIN();
  *ret = KVStore::IsWorkerNode();
  API_END();
}

int MXKVStoreIsServerNode(int *ret) {
  API_BEGIN();
  *ret = KVStore::IsServerNode();
  API_END();
}

int MXKVStoreIsSchedulerNode(int *ret) {
  API_BEGIN();
  *ret = KVStore::IsSchedulerNode();
  API_END();
}

int MXKVStoreRunServer(KVStoreHandle handle,
                       MXKVStoreServerController controller,
                       void *controller_handle) {
  API_BEGIN();
  MXKVStoreServerController *controller_temp = controller;
  void *controller_handle_temp = controller_handle;
  auto ctrl = [controller_temp, controller_handle_temp](int head, const std::string& body) {
      controller_temp(head, body.c_str(), controller_handle_temp);
  };
  static_cast<KVStore*>(handle)->RunServer(ctrl);
  API_END();
}

int MXKVStoreSendCommmandToServers(KVStoreHandle handle,
                                   int cmd_id,
                                   const char* cmd_body) {
  API_BEGIN();
  static_cast<KVStore*>(handle)->SendCommandToServers(
      cmd_id, std::string(cmd_body));
  API_END();
}

int MXKVStoreGetType(KVStoreHandle handle,
                     const char** type) {
  API_BEGIN();
  *CHECK_NOTNULL(type) = static_cast<KVStore*>(handle)->type().c_str();
  API_END();
}

int MXKVStoreGetNumDeadNode(KVStoreHandle handle,
                            const int node_id,
                            int *number,
                            const int timeout_sec) {
  API_BEGIN();
  *number = static_cast<KVStore*>(handle)->get_num_dead_node(node_id, timeout_sec);
  API_END();
}

struct MXRecordIOContext {
  dmlc::RecordIOWriter *writer;
  dmlc::RecordIOReader *reader;
  dmlc::Stream *stream;
  std::string *read_buff;
};

int MXRecordIOWriterCreate(const char *uri,
                           RecordIOHandle *out) {
  API_BEGIN();
  dmlc::Stream *stream = dmlc::Stream::Create(uri, "w");
  MXRecordIOContext *context = new MXRecordIOContext;
  context->writer = new dmlc::RecordIOWriter(stream);
  context->reader = nullptr;
  context->stream = stream;
  context->read_buff = nullptr;
  *out = reinterpret_cast<RecordIOHandle>(context);
  API_END();
}

int MXRecordIOWriterFree(RecordIOHandle handle) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  delete context->writer;
  delete context->stream;
  delete context;
  API_END();
}

int MXRecordIOWriterWriteRecord(RecordIOHandle handle,
                                const char *buf, size_t size) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  context->writer->WriteRecord(reinterpret_cast<const void*>(buf), size);
  API_END();
}

int MXRecordIOWriterTell(RecordIOHandle handle, size_t *pos) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  *pos = context->writer->Tell();
  API_END();
}

int MXRecordIOReaderCreate(const char *uri,
                           RecordIOHandle *out) {
  API_BEGIN();
  dmlc::Stream *stream = dmlc::Stream::Create(uri, "r");
  MXRecordIOContext *context = new MXRecordIOContext;
  context->reader = new dmlc::RecordIOReader(stream);
  context->writer = nullptr;
  context->stream = stream;
  context->read_buff = new std::string();
  *out = reinterpret_cast<RecordIOHandle>(context);
  API_END();
}

int MXRecordIOReaderFree(RecordIOHandle handle) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  delete context->reader;
  delete context->stream;
  delete context->read_buff;
  delete context;
  API_END();
}

int MXRecordIOReaderReadRecord(RecordIOHandle handle,
                              char const **buf, size_t *size) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  if (context->reader->NextRecord(context->read_buff)) {
    *buf = context->read_buff->c_str();
    *size = context->read_buff->size();
  } else {
    *buf = nullptr;
    *size = 0;
  }
  API_END();
}

int MXRecordIOReaderSeek(RecordIOHandle handle, size_t pos) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  context->reader->Seek(pos);
  API_END();
}

int MXRecordIOReaderTell(RecordIOHandle handle, size_t *pos) {
  API_BEGIN();
  MXRecordIOContext *context =
    reinterpret_cast<MXRecordIOContext*>(handle);
  *pos = context->reader->Tell();
  API_END();
}

int MXRtcCreate(char* name, uint32_t num_input, uint32_t num_output,
                char** input_names, char** output_names,
                NDArrayHandle* inputs, NDArrayHandle* outputs,
                char* kernel, RtcHandle *out) {
  API_BEGIN();
  LOG(FATAL) << "Old rtc API is deprecated. Please use CudaModule";
  API_END();
}

int MXRtcPush(RtcHandle handle, uint32_t num_input, uint32_t num_output,
              NDArrayHandle* inputs, NDArrayHandle* outputs,
              uint32_t gridDimX,
              uint32_t gridDimY,
              uint32_t gridDimZ,
              uint32_t blockDimX,
              uint32_t blockDimY,
              uint32_t blockDimZ) {
  API_BEGIN();
  LOG(FATAL) << "Old rtc API is deprecated. Please use CudaModule";
  API_END();
}

int MXRtcFree(RtcHandle handle) {
  API_BEGIN();
  LOG(FATAL) << "Old rtc API is deprecated. Please use CudaModule";
  API_END();
}

int MXCustomOpRegister(const char* op_type, CustomOpPropCreator creator) {
  API_BEGIN();
  mxnet::op::custom::CustomOperator::Get()->Register(op_type, creator);
  API_END();
}


int MXRtcCudaModuleCreate(const char* source, int num_options,
                          const char** options, int num_exports,
                          const char** exports, CudaModuleHandle *out) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  std::vector<std::string> str_opts;
  for (int i = 0; i < num_options; ++i) str_opts.emplace_back(options[i]);
  std::vector<std::string> str_exports;
  for (int i = 0; i < num_exports; ++i) str_exports.emplace_back(exports[i]);
  *out = new rtc::CudaModule(source, str_opts, str_exports);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXRtcCudaModuleFree(CudaModuleHandle handle) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  delete reinterpret_cast<rtc::CudaModule*>(handle);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXRtcCudaKernelCreate(CudaModuleHandle handle, const char* name, int num_args,
                          int* is_ndarray, int* is_const, int* arg_types,
                          CudaKernelHandle *out) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  auto module = reinterpret_cast<rtc::CudaModule*>(handle);
  std::vector<rtc::CudaModule::ArgType> signature;
  for (int i = 0; i < num_args; ++i) {
    signature.push_back(rtc::CudaModule::ArgType{
        static_cast<bool>(is_ndarray[i]), static_cast<bool>(is_const[i]),
        static_cast<mshadow::TypeFlag>(arg_types[i])});
  }
  auto kernel = module->GetKernel(name, signature);
  *out = new std::shared_ptr<rtc::CudaModule::Kernel>(kernel);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXRtcCudaKernelFree(CudaKernelHandle handle) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  delete reinterpret_cast<std::shared_ptr<rtc::CudaModule::Kernel>*>(handle);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXRtcCudaKernelCall(CudaKernelHandle handle, int dev_id, void** args,
                        uint32_t grid_dim_x, uint32_t grid_dim_y,
                        uint32_t grid_dim_z, uint32_t block_dim_x,
                        uint32_t block_dim_y, uint32_t block_dim_z,
                        uint32_t shared_mem) {
  API_BEGIN();
#if MXNET_USE_CUDA && MXNET_ENABLE_CUDA_RTC
  auto kernel = reinterpret_cast<std::shared_ptr<rtc::CudaModule::Kernel>*>(handle);
  const auto& signature = (*kernel)->signature();
  std::vector<dmlc::any> any_args;
  for (size_t i = 0; i < signature.size(); ++i) {
    if (signature[i].is_ndarray) {
      any_args.emplace_back(*static_cast<NDArray*>(args[i]));
    } else {
      MSHADOW_TYPE_SWITCH(signature[i].dtype, DType, {
        any_args.emplace_back(*static_cast<DType*>(args[i]));
      });
    }
  }
  (*kernel)->Launch(Context::GPU(dev_id), any_args, grid_dim_x, grid_dim_y,
                    grid_dim_z, block_dim_x, block_dim_y, block_dim_z, shared_mem);
#else
  LOG(FATAL) << "Compile with USE_CUDA=1 and ENABLE_CUDA_RTC=1 to have CUDA runtime compilation.";
#endif
  API_END();
}

int MXNDArrayGetSharedMemHandle(NDArrayHandle handle, int* shared_pid, int* shared_id) {
  API_BEGIN();
  NDArray* arr = reinterpret_cast<NDArray*>(handle);
  Storage::Handle shandle;
  if (arr->ctx().dev_type == Context::kCPUShared) {
    arr->WaitToRead();
    shandle = arr->storage_handle();
    Storage::Get()->SharedIncrementRefCount(shandle);
  } else {
    NDArray new_arr(arr->shape(), Context::CPUShared(0), false, arr->dtype());
    CopyFromTo(*arr, new_arr);
    new_arr.WaitToRead();
    shandle = new_arr.storage_handle();
    Storage::Get()->SharedIncrementRefCount(shandle);
  }
  *shared_pid = shandle.shared_pid;
  *shared_id = shandle.shared_id;
  API_END();
}

int MXNDArrayCreateFromSharedMem(int shared_pid, int shared_id, const uint32_t *shape,
                                 uint32_t ndim, int dtype, NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(shared_pid, shared_id, mxnet::TShape(shape, shape + ndim), dtype);
  API_END();
}

int MXNDArrayCreateFromSharedMemEx(int shared_pid, int shared_id, const int *shape,
                                   int ndim, int dtype, NDArrayHandle *out) {
  API_BEGIN();
  *out = new NDArray(shared_pid, shared_id, mxnet::TShape(shape, shape + ndim), dtype);
  API_END();
}

typedef Engine::VarHandle VarHandle;
typedef Engine::CallbackOnComplete CallbackOnComplete;

void AssertValidNumberVars(int num_const_vars, int num_mutable_vars) {
  CHECK_GE(num_const_vars, 0) << "Non-negative number of const vars expected.";
  CHECK_GE(num_mutable_vars, 0) << "Non-negative number of mutable vars expected.";
}

int MXEnginePushAsync(EngineAsyncFunc async_func, void* func_param,
                      EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                      EngineVarHandle const_vars_handle, int num_const_vars,
                      EngineVarHandle mutable_vars_handle, int num_mutable_vars,
                      EngineFnPropertyHandle prop_handle, int priority,
                      const char* opr_name, bool wait) {
  API_BEGIN();

  auto exec_ctx = *static_cast<const Context*>(ctx_handle);
  auto const_vars = static_cast<VarHandle*>(const_vars_handle);
  auto mutable_vars = static_cast<VarHandle*>(mutable_vars_handle);
  auto prop = FnProperty::kNormal;
  if (prop_handle) {
    prop = *static_cast<const FnProperty*>(prop_handle);
  }

  Engine::AsyncFn exec_fn;
  if (deleter == nullptr) {
    exec_fn = [async_func, func_param](RunContext rctx,
                                       CallbackOnComplete on_complete) {
      async_func(&rctx, &on_complete, func_param);
    };
  } else {
    // Wrap func_param in a shared_ptr with deleter such that deleter
    // will be called when the lambda goes out of scope.
    std::shared_ptr<void> shared_func_param(func_param, deleter);
    exec_fn = [async_func, shared_func_param](RunContext rctx,
                                              CallbackOnComplete on_complete) {
      async_func(&rctx, &on_complete, shared_func_param.get());
    };
  }

  AssertValidNumberVars(num_const_vars, num_mutable_vars);
  std::vector<VarHandle> const_var_vec(const_vars, const_vars + num_const_vars);
  std::vector<VarHandle> mutable_var_vec(mutable_vars, mutable_vars + num_mutable_vars);
  Engine::Get()->PushAsync(exec_fn, exec_ctx, const_var_vec, mutable_var_vec,
                           prop, priority, opr_name, wait);

  API_END();
}

int MXEnginePushSync(EngineSyncFunc sync_func, void* func_param,
                     EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                     EngineVarHandle const_vars_handle, int num_const_vars,
                     EngineVarHandle mutable_vars_handle, int num_mutable_vars,
                     EngineFnPropertyHandle prop_handle, int priority,
                     const char* opr_name) {
  API_BEGIN();

  auto exec_ctx = *static_cast<const Context*>(ctx_handle);
  auto const_vars = static_cast<VarHandle*>(const_vars_handle);
  auto mutable_vars = static_cast<VarHandle*>(mutable_vars_handle);
  auto prop = FnProperty::kNormal;
  if (prop_handle) {
    prop = *static_cast<const FnProperty*>(prop_handle);
  }

  Engine::SyncFn exec_fn;
  if (deleter == nullptr) {
    exec_fn = [sync_func, func_param](RunContext rctx) {
      sync_func(&rctx, func_param);
    };
  } else {
    // Wrap func_param in a shared_ptr with deleter such that deleter
    // will be called when the lambda goes out of scope.
    std::shared_ptr<void> shared_func_param(func_param, deleter);
    exec_fn = [sync_func, shared_func_param](RunContext rctx) {
      sync_func(&rctx, shared_func_param.get());
    };
  }

  AssertValidNumberVars(num_const_vars, num_mutable_vars);
  std::vector<VarHandle> const_var_vec(const_vars, const_vars + num_const_vars);
  std::vector<VarHandle> mutable_var_vec(mutable_vars, mutable_vars + num_mutable_vars);
  Engine::Get()->PushSync(exec_fn, exec_ctx, const_var_vec, mutable_var_vec,
                          prop, priority, opr_name);

  API_END();
}

int MXEnginePushAsyncND(EngineAsyncFunc async_func, void* func_param,
                        EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                        NDArrayHandle* const_nds_handle, int num_const_nds,
                        NDArrayHandle* mutable_nds_handle, int num_mutable_nds,
                        EngineFnPropertyHandle prop_handle, int priority,
                        const char* opr_name, bool wait) {
  API_BEGIN();
  NDArray** const_nds = reinterpret_cast<NDArray**>(const_nds_handle);
  NDArray** mutable_nds = reinterpret_cast<NDArray**>(mutable_nds_handle);
  std::vector<VarHandle> const_var_vec(num_const_nds);
  for (int i = 0; i < num_const_nds; ++i) const_var_vec[i] = const_nds[i]->var();
  std::vector<VarHandle> mutable_var_vec(num_mutable_nds);
  for (int i = 0; i < num_mutable_nds; ++i) mutable_var_vec[i] = mutable_nds[i]->var();
  return MXEnginePushAsync(async_func, func_param, deleter, ctx_handle,
                           const_var_vec.data(), num_const_nds,
                           mutable_var_vec.data(), num_mutable_nds,
                           prop_handle, priority, opr_name, wait);
  API_END();
}

int MXEnginePushSyncND(EngineSyncFunc sync_func, void* func_param,
                       EngineFuncParamDeleter deleter, ContextHandle ctx_handle,
                       NDArrayHandle* const_nds_handle, int num_const_nds,
                       NDArrayHandle* mutable_nds_handle, int num_mutable_nds,
                       EngineFnPropertyHandle prop_handle, int priority,
                       const char* opr_name) {
  API_BEGIN();
  NDArray** const_nds = reinterpret_cast<NDArray**>(const_nds_handle);
  NDArray** mutable_nds = reinterpret_cast<NDArray**>(mutable_nds_handle);
  std::vector<VarHandle> const_var_vec(num_const_nds);
  for (int i = 0; i < num_const_nds; ++i) const_var_vec[i] = const_nds[i]->var();
  std::vector<VarHandle> mutable_var_vec(num_mutable_nds);
  for (int i = 0; i < num_mutable_nds; ++i) mutable_var_vec[i] = mutable_nds[i]->var();
  return MXEnginePushSync(sync_func, func_param, deleter, ctx_handle,
                          const_var_vec.data(), num_const_nds,
                          mutable_var_vec.data(), num_mutable_nds,
                          prop_handle, priority, opr_name);
  API_END();
}

int MXStorageEmptyCache(int dev_type, int dev_id) {
  API_BEGIN();
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  Storage::Get()->ReleaseAll(ctx);
  API_END();
}

int MXShallowCopyNDArray(NDArrayHandle src_handle, NDArrayHandle* out) {
  NDArray* ret = nullptr;
  API_BEGIN();
  NDArray* src_array = static_cast<NDArray*>(src_handle);
  ret = new NDArray(*src_array);
  *out = ret;
  API_END_HANDLE_ERROR(delete ret);
}
