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
 * \file c_predict_api.cc
 * \brief C predict API of mxnet
 */
#include <dmlc/base.h>
#include <dmlc/memory_io.h>
#include <mxnet/c_predict_api.h>
#include <mxnet/executor.h>
#include <mxnet/ndarray.h>
#include <nnvm/pass_functions.h>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include "./c_api_common.h"
#include "../operator/operator_common.h"
#include "../executor/exec_pass.h"

using namespace mxnet;

// predictor interface
struct MXAPIPredictor {
  // output arrays
  std::vector<NDArray> out_arrays;
  // argument arrays
  std::vector<NDArray> arg_arrays;
  // auxiliary arrays
  std::vector<NDArray> aux_arrays;
  // output names
  std::vector<std::string> out_names;
  std::vector<const char*> out_names_buffer;
  // output shapes
  mxnet::ShapeVector out_shapes;
  // output types
  nnvm::DTypeVector out_dtypes;

  // uint32_t buffer for input shapes
  std::vector<uint32_t> in_shapes_buffer;
  // uint32_t buffer for output shapes
  std::vector<uint32_t> out_shapes_buffer;
  // key to arguments
  std::unordered_map<std::string, size_t> key2arg;
  // executor
  std::unique_ptr<Executor> exec;
  // symbol
  nnvm::Symbol sym;
  // Context
  Context ctx;
};

struct MXAPINDList {
  std::vector<std::string> keys;
  mxnet::ShapeVector shapes;
  std::vector<uint32_t> shapes_buffer;
  std::vector<size_t> indptr;
  std::vector<float> data;
};

struct MXAPINDListEx {
  std::vector<std::string> keys;
  std::vector<uint32_t> shapes_buffer;
  std::vector<NDArray> arrays;
};

inline void _CreateExecutor(PredictorHandle pred_hnd) {
  MXAPIPredictor *pred = static_cast<MXAPIPredictor*>(pred_hnd);
  if (pred->exec == nullptr) {
    auto sym = pred->sym;
    auto ctx = pred->ctx;
    auto key2arg = pred->key2arg;
    auto arg_arrays = pred->arg_arrays;
    auto aux_arrays = pred->aux_arrays;
    std::map<std::string, Context> ctx_map;
    std::vector<NDArray> grad_store(arg_arrays.size());
    std::vector<OpReqType> grad_req(arg_arrays.size(), kNullOp);
    pred->exec.reset(Executor::Bind(sym, ctx, ctx_map, arg_arrays,
                                    grad_store, grad_req, aux_arrays));
    pred->out_arrays = pred->exec->outputs();
  }
}

int _CreatePartialOut(const char* symbol_json_str,
                      const void* param_bytes,
                      int param_size,
                      int dev_type, int dev_id,
                      const uint32_t num_input_nodes,
                      const char** input_keys,
                      const uint32_t* input_shape_indptr,
                      const uint32_t* input_shape_data,
                      uint32_t num_output_nodes,
                      const char** output_keys,
                      // This is used for parallel inference.
                      int num_threads,
                      bool lazy,
                      const uint32_t num_provided_arg_dtypes,
                      const char** provided_arg_dtype_names,
                      const int* provided_arg_dtypes,
                      PredictorHandle* out) {
  using nnvm::Symbol;

  API_BEGIN();
  Symbol sym;
  // make sure symbols are registered
  {
  uint32_t outSize;
  const char **outArray;
  MXListAllOpNames(&outSize, &outArray);
  }
  // load in the symbol.
  {
    nnvm::Graph g;
    g.attrs["json"] = std::make_shared<nnvm::any>(std::string(symbol_json_str));
    sym.outputs = nnvm::ApplyPass(g, "LoadLegacyJSON").outputs;
  }
  // looks likely to output the internal results
  if (num_output_nodes != 0) {
    Symbol internal = sym.GetInternals();
    std::vector<std::string> all_out = internal.ListOutputNames();
    std::vector<Symbol> out_syms(num_output_nodes);
    for (uint32_t i = 0; i < num_output_nodes; ++i) {
      std::string out_key(output_keys[i]);
      out_key += "_output";
      for (size_t j = 0; j < all_out.size(); ++j) {
        if (all_out[j] == out_key) {
          out_syms[i] = internal[j];
          break;
        }
        CHECK_NE(j, all_out.size() - 1) << "didn't find node name: " << out_key;
      }
    }
    sym = nnvm::Symbol::CreateGroup(out_syms);
  }

  // load the parameters
  std::unordered_map<std::string, NDArray> arg_params, aux_params;
  std::unordered_map<std::string, int> arg_types, aux_types;
  {
    std::unordered_set<std::string> arg_names, aux_names;
    std::vector<std::string> arg_names_vec = sym.ListInputNames(Symbol::kReadOnlyArgs);
    std::vector<std::string> aux_names_vec = sym.ListInputNames(Symbol::kAuxiliaryStates);
    for (const auto &arg_name : arg_names_vec) {
      arg_names.insert(arg_name);
    }
    for (const auto &aux_name : aux_names_vec) {
      aux_names.insert(aux_name);
    }
    std::vector<NDArray> data;
    std::vector<std::string> names;
    dmlc::MemoryFixedSizeStream fi((void*)param_bytes, param_size);  // NOLINT(*)
    NDArray::Load(&fi, &data, &names);
    CHECK_EQ(names.size(), data.size())
        << "Invalid param file format";
    for (size_t i = 0; i < names.size(); ++i) {
      if (!strncmp(names[i].c_str(), "aux:", 4)) {
        std::string name(names[i].c_str() + 4);
        if (aux_names.count(name) != 0) {
          aux_params[name] = data[i];
          aux_types[name] = data[i].dtype();
        }
      }
      if (!strncmp(names[i].c_str(), "arg:", 4)) {
        std::string name(names[i].c_str() + 4);
        if (arg_names.count(name) != 0) {
          arg_params[name] = data[i];
          arg_types[name] = data[i].dtype();
        }
      }
    }

    if (num_provided_arg_dtypes > 0) {
      for (uint32_t i = 0; i < num_provided_arg_dtypes; ++i) {
        if (aux_types.count(provided_arg_dtype_names[i]) == 0 &&
            arg_types.count(provided_arg_dtype_names[i]) == 0) {
          arg_types[provided_arg_dtype_names[i]] = provided_arg_dtypes[i];
        }
      }
    }
  }

  // shape inference and bind
  std::unordered_map<std::string, mxnet::TShape> known_shape;
  for (uint32_t i = 0; i < num_input_nodes; ++i) {
    known_shape[std::string(input_keys[i])] =
        mxnet::TShape(input_shape_data + input_shape_indptr[i],
               input_shape_data + input_shape_indptr[i + 1]);
  }
  std::vector<std::string> arg_names = sym.ListInputNames(Symbol::kReadOnlyArgs);
  std::vector<std::string> aux_names = sym.ListInputNames(Symbol::kAuxiliaryStates);
  std::vector<std::string> out_names = sym.ListOutputNames();
  mxnet::ShapeVector out_shapes(sym.ListOutputNames().size());
  mxnet::ShapeVector aux_shapes(aux_names.size());
  mxnet::ShapeVector arg_shapes;
  nnvm::DTypeVector result_arg_types, result_out_types, result_aux_types;
  std::unordered_map<std::string, size_t> key2arg;
  for (size_t i = 0; i < arg_names.size(); ++i) {
    std::string key = arg_names[i];
    key2arg[key] = i;
  }

  try {
    mxnet::ShapeVector in_shapes;
    nnvm::DTypeVector in_types;
    for (std::string key : sym.ListInputNames(Symbol::kAll)) {
      if (known_shape.count(key) != 0) {
        in_shapes.push_back(known_shape[key]);
      } else {
        in_shapes.emplace_back();
      }
    }

    for (std::string key : sym.ListInputNames(Symbol::kAll)) {
      if (arg_types.count(key) != 0) {
        in_types.push_back(arg_types[key]);
      } else if (aux_types.count(key) != 0) {
        in_types.push_back(aux_types[key]);
      } else {
        // if key not in arg_types or aux_types set to FP32
        in_types.push_back(0);
      }
    }
    nnvm::Graph g; g.outputs = sym.outputs;
    g = mxnet::exec::InferShape(std::move(g), std::move(in_shapes), "__shape__");
    g = mxnet::exec::InferType(std::move(g), std::move(in_types), "__dtype__");
    bool infer_complete = (g.GetAttr<size_t>("shape_num_unknown_nodes") == 0);
    // This is tricky for AMP Use case, for example, with only weights input types
    // cannot be inferred in AMP. Thus for AMP converted model type_dict will be
    // required
    bool infer_type_complete = (g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0);
    CHECK(infer_complete)
      << "The shape information of is not enough to get the shapes";
    CHECK(infer_type_complete)
        << "The type information is not enough, please provide input arg_types "
           "with provided_arg_dtype_names and provided_arg_dtypes."
           "If using amalgamation python frontend you can use type_dict in Predictor API"
           "to provide this information";
    CopyAttr(g.indexed_graph(),
             g.GetAttr<mxnet::ShapeVector>("shape"),
             &arg_shapes, &out_shapes, &aux_shapes);
    CopyAttr(g.indexed_graph(),
             g.GetAttr<nnvm::DTypeVector>("dtype"),
             &result_arg_types, &result_out_types, &result_aux_types);
  } catch (const mxnet::op::InferShapeError &err) {
    throw dmlc::Error(err.msg);
  }

  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);

  std::vector<NDArray> arg_arrays, aux_arrays;
  for (size_t i = 0; i < arg_shapes.size(); ++i) {
    NDArray nd;
    if (result_arg_types[i] != -1) {
      nd = NDArray(arg_shapes[i], ctx, false, result_arg_types[i]);
    } else {
      nd = NDArray(arg_shapes[i], ctx);
    }
    if (arg_params.count(arg_names[i]) != 0) {
      CopyFromTo(arg_params[arg_names[i]], &nd);
    }
    arg_arrays.push_back(nd);
  }

  for (size_t i = 0; i < aux_shapes.size(); ++i) {
    NDArray nd;
    if (result_aux_types[i] != -1) {
      nd = NDArray(aux_shapes[i], ctx, false, result_aux_types[i]);
    } else {
      nd = NDArray(aux_shapes[i], ctx);
    }
    if (aux_params.count(aux_names[i]) != 0) {
      CopyFromTo(aux_params[aux_names[i]], &nd);
    }
    aux_arrays.push_back(nd);
  }

  // bind
  for (int i = 0; i < num_threads; i++) {
    std::unique_ptr<MXAPIPredictor> ret(new MXAPIPredictor());
    ret->sym = sym;
    ret->ctx = ctx;
    ret->key2arg = key2arg;
    ret->arg_arrays = arg_arrays;
    ret->aux_arrays = aux_arrays;
    ret->out_shapes = out_shapes;
    ret->out_dtypes = result_out_types;
    ret->out_names = out_names;
    ret->out_names_buffer.resize(ret->out_names.size());
    std::transform(ret->out_names.begin(), ret->out_names.end(), ret->out_names_buffer.begin(),
        [](const std::string& name) { return name.c_str(); });

    if (!lazy) {
      std::map<std::string, Context> ctx_map;
      std::vector<NDArray> grad_store(arg_arrays.size());
      std::vector<OpReqType> grad_req(arg_arrays.size(), kNullOp);
      ret->exec.reset(Executor::Bind(sym, ctx, ctx_map,
                                     arg_arrays,
                                     grad_store, grad_req,
                                     aux_arrays));
      ret->out_arrays = ret->exec->outputs();
      CHECK(ret->out_shapes.size() == ret->out_arrays.size());
      CHECK(ret->out_names.size() == ret->out_arrays.size());
    }
    out[i] = ret.release();
  }
  API_END_HANDLE_ERROR();
}

int MXPredCreatePartialOut(const char* symbol_json_str,
                           const void* param_bytes,
                           int param_size,
                           int dev_type, int dev_id,
                           uint32_t num_input_nodes,
                           const char** input_keys,
                           const uint32_t* input_shape_indptr,
                           const uint32_t* input_shape_data,
                           uint32_t num_output_nodes,
                           const char** output_keys,
                           PredictorHandle* out) {
  return _CreatePartialOut(
      symbol_json_str,
      param_bytes,
      param_size,
      dev_type, dev_id,
      num_input_nodes,
      input_keys,
      input_shape_indptr,
      input_shape_data,
      num_output_nodes,
      output_keys,
      1,
      false,
      0,
      nullptr,
      nullptr,
      out);
}

int MXPredCreate(const char* symbol_json_str,
                 const void* param_bytes,
                 int param_size,
                 int dev_type, int dev_id,
                 uint32_t num_input_nodes,
                 const char** input_keys,
                 const uint32_t* input_shape_indptr,
                 const uint32_t* input_shape_data,
                 PredictorHandle* out) {
  return _CreatePartialOut(
      symbol_json_str,
      param_bytes,
      param_size,
      dev_type,
      dev_id,
      num_input_nodes,
      input_keys,
      input_shape_indptr,
      input_shape_data,
      0,
      nullptr,
      1,
      false,
      0,
      nullptr,
      nullptr,
      out);
}

int MXPredCreateEx(const char* symbol_json_str,
                   const void* param_bytes,
                   int param_size,
                   int dev_type, int dev_id,
                   uint32_t num_input_nodes,
                   const char** input_keys,
                   const uint32_t* input_shape_indptr,
                   const uint32_t* input_shape_data,
                   const uint32_t num_provided_arg_dtypes,
                   const char** provided_arg_dtype_names,
                   const int* provided_arg_dtypes,
                   PredictorHandle* out) {
  return _CreatePartialOut(
      symbol_json_str,
      param_bytes,
      param_size,
      dev_type,
      dev_id,
      num_input_nodes,
      input_keys,
      input_shape_indptr,
      input_shape_data,
      0,
      nullptr,
      1,
      false,
      num_provided_arg_dtypes,
      provided_arg_dtype_names,
      provided_arg_dtypes,
      out);
}

int MXPredCreateMultiThread(const char* symbol_json_str,
                            const void* param_bytes,
                            int param_size,
                            int dev_type, int dev_id,
                            uint32_t num_input_nodes,
                            const char** input_keys,
                            const uint32_t* input_shape_indptr,
                            const uint32_t* input_shape_data,
                            // This is used for paralle inference.
                            int num_threads,
                            PredictorHandle* out) {
  const char *type = getenv("MXNET_ENGINE_TYPE");
  std::string stype;
  if (type)
    stype = type;
  CHECK(stype == "NaiveEngine") << "Multithread inference only works with NaiveEngine.\n"
      << "Please set MXNET_ENGINE_TYPE to NaiveEngine"
      << std::endl;
  return _CreatePartialOut(
      symbol_json_str,
      param_bytes,
      param_size,
      dev_type,
      dev_id,
      num_input_nodes,
      input_keys,
      input_shape_indptr,
      input_shape_data,
      0,
      nullptr,
      num_threads,
      true,
      0,
      nullptr,
      nullptr,
      out);
}

int MXPredCreateEx2(const char* symbol_json_str,
                    int dev_type, int dev_id,
                    const char** input_keys,
                    const void** input_data,
                    const int* input_dtypes,
                    const mx_uint* input_shape_indptr,
                    const mx_uint* input_shape_data,
                    const mx_uint num_input_nodes,
                    PredictorHandle* out) {
  return MXPredCreatePartialOutEx2(symbol_json_str,
                                   dev_type,
                                   dev_id,
                                   input_keys,
                                   input_data,
                                   input_dtypes,
                                   input_shape_indptr,
                                   input_shape_data,
                                   num_input_nodes,
                                   0,
                                   nullptr,
                                   out);
}

int MXPredCreatePartialOutEx(const char* symbol_json_str,
                             int dev_type, int dev_id,
                             const char** input_keys,
                             const mx_float** input_data,
                             const mx_uint* input_shape_indptr,
                             const mx_uint* input_shape_data,
                             const mx_uint num_input_nodes,
                             const mx_uint num_output_nodes,
                             const char** output_keys,
                             PredictorHandle* out) {
  using nnvm::Symbol;

  MXAPIPredictor* ret = new MXAPIPredictor();
  API_BEGIN();
  Symbol sym;
  // make sure symbols are registered
  {
    mx_uint outSize;
    const char **outArray;
    MXListAllOpNames(&outSize, &outArray);
  }
  // load in the symbol.
  {
    nnvm::Graph g;
    g.attrs["json"] = std::make_shared<nnvm::any>(std::string(symbol_json_str));
    sym.outputs = nnvm::ApplyPass(g, "LoadLegacyJSON").outputs;
  }
  // output the internal results
  if (num_output_nodes != 0) {
    Symbol internal = sym.GetInternals();
    std::vector<std::string> all_out = internal.ListOutputNames();
    std::vector<Symbol> out_syms(num_output_nodes);
    for (mx_uint i = 0; i < num_output_nodes; ++i) {
      std::string out_key(output_keys[i]);
      out_key += "_output";
      for (size_t j = 0; j < all_out.size(); ++j) {
        if (all_out[j] == out_key) {
          out_syms[i] = internal[j];
          break;
        }
        CHECK_NE(j, all_out.size() - 1) << "didn't find node name: " << out_key;
      }
    }
    sym = nnvm::Symbol::CreateGroup(out_syms);
  }

  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  std::unordered_map<std::string, mxnet::TShape> input_shape_map;
  std::unordered_map<std::string, NDArray> input_nd_map;
  for (mx_uint i = 0; i < num_input_nodes; ++i) {
    const std::string input_name = input_keys[i];
    CHECK_EQ(input_shape_map.count(input_name), 0U);
    input_shape_map[input_name] = mxnet::TShape(input_shape_data + input_shape_indptr[i],
                                         input_shape_data + input_shape_indptr[i + 1]);
    NDArray nd(input_shape_map[input_name], ctx);
    nd.SyncCopyFromCPU(input_data[i], nd.shape().Size());
    CHECK_EQ(input_nd_map.count(input_name), 0U);
    input_nd_map.emplace(input_name, nd);
  }
  CHECK_EQ(input_shape_map.size(), input_nd_map.size());

  const std::vector<std::string> arg_names = sym.ListInputNames(Symbol::kReadOnlyArgs);
  const std::vector<std::string> aux_names = sym.ListInputNames(Symbol::kAuxiliaryStates);
  const std::vector<std::string> out_names = sym.ListOutputNames();
  CHECK_EQ(input_shape_map.size(), arg_names.size() + aux_names.size());
  for (size_t i = 0; i < arg_names.size(); ++i) {
    std::string key = arg_names[i];
    ret->key2arg[key] = i;
  }
  // load args
  ret->arg_arrays.clear();
  ret->arg_arrays.reserve(arg_names.size());
  for (const auto& name : arg_names) {
    auto it = input_nd_map.find(name);
    CHECK(it != input_nd_map.end());
    ret->arg_arrays.push_back(it->second);
  }
  // load aux
  ret->aux_arrays.clear();
  ret->aux_arrays.reserve(aux_names.size());
  for (const auto& name : aux_names) {
    auto it = input_nd_map.find(name);
    CHECK(it != input_nd_map.end());
    ret->aux_arrays.push_back(it->second);
  }

  // bind
  std::map<std::string, Context> ctx_map;
  std::vector<NDArray> grad_store(ret->arg_arrays.size());
  std::vector<OpReqType> grad_req(ret->arg_arrays.size(), kNullOp);


  ret->exec.reset(Executor::Bind(sym, ctx, ctx_map,
                                 ret->arg_arrays,
                                 grad_store, grad_req,
                                 ret->aux_arrays));
  ret->out_arrays = ret->exec->outputs();

  CHECK(out_names.size() == ret->out_arrays.size());
  ret->out_names = out_names;
  ret->out_names_buffer.resize(ret->out_names.size());
  std::transform(ret->out_names.begin(), ret->out_names.end(), ret->out_names_buffer.begin(),
      [](const std::string& name) { return name.c_str(); });

  ret->out_shapes.resize(ret->out_arrays.size());
  std::transform(ret->out_arrays.begin(), ret->out_arrays.end(), ret->out_shapes.begin(),
      [](const NDArray& nd) { return nd.shape(); });

  *out = ret;
  API_END_HANDLE_ERROR(delete ret);
}

int MXPredCreatePartialOutEx2(const char* symbol_json_str,
                              int dev_type, int dev_id,
                              const char** input_keys,
                              const void** input_data,
                              const int* input_dtypes,
                              const mx_uint* input_shape_indptr,
                              const mx_uint* input_shape_data,
                              const mx_uint num_input_nodes,
                              const mx_uint num_output_nodes,
                              const char** output_keys,
                              PredictorHandle* out) {
  using nnvm::Symbol;

  MXAPIPredictor* ret = new MXAPIPredictor();
  API_BEGIN();
  Symbol sym;
  // make sure symbols are registered
  {
    mx_uint outSize;
    const char **outArray;
    MXListAllOpNames(&outSize, &outArray);
  }
  // load in the symbol.
  {
    nnvm::Graph g;
    g.attrs["json"] = std::make_shared<nnvm::any>(std::string(symbol_json_str));
    sym.outputs = nnvm::ApplyPass(g, "LoadLegacyJSON").outputs;
  }
  // output the internal results
  if (num_output_nodes != 0) {
    Symbol internal = sym.GetInternals();
    std::vector<std::string> all_out = internal.ListOutputNames();
    std::vector<Symbol> out_syms(num_output_nodes);
    for (mx_uint i = 0; i < num_output_nodes; ++i) {
      std::string out_key(output_keys[i]);
      out_key += "_output";
      for (size_t j = 0; j < all_out.size(); ++j) {
        if (all_out[j] == out_key) {
          out_syms[i] = internal[j];
          break;
        }
        CHECK_NE(j, all_out.size() - 1) << "didn't find node name: " << out_key;
      }
    }
    sym = nnvm::Symbol::CreateGroup(out_syms);
  }

  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  std::unordered_map<std::string, mxnet::TShape> input_shape_map;
  std::unordered_map<std::string, NDArray> input_nd_map;
  for (mx_uint i = 0; i < num_input_nodes; ++i) {
    const std::string input_name = input_keys[i];
    CHECK_EQ(input_shape_map.count(input_name), 0U);
    input_shape_map[input_name] = mxnet::TShape(input_shape_data + input_shape_indptr[i],
                                         input_shape_data + input_shape_indptr[i + 1]);
    NDArray nd(input_shape_map[input_name], ctx, false, input_dtypes[i]);
    nd.SyncCopyFromCPU(input_data[i], nd.shape().Size());
    CHECK_EQ(input_nd_map.count(input_name), 0U);
    input_nd_map.emplace(input_name, nd);
  }
  CHECK_EQ(input_shape_map.size(), input_nd_map.size());

  const std::vector<std::string> arg_names = sym.ListInputNames(Symbol::kReadOnlyArgs);
  const std::vector<std::string> aux_names = sym.ListInputNames(Symbol::kAuxiliaryStates);
  const std::vector<std::string> out_names = sym.ListOutputNames();
  CHECK_EQ(input_shape_map.size(), arg_names.size() + aux_names.size());
  for (size_t i = 0; i < arg_names.size(); ++i) {
    std::string key = arg_names[i];
    ret->key2arg[key] = i;
  }
  // load args
  ret->arg_arrays.clear();
  ret->arg_arrays.reserve(arg_names.size());
  for (const auto& name : arg_names) {
    auto it = input_nd_map.find(name);
    CHECK(it != input_nd_map.end());
    ret->arg_arrays.push_back(it->second);
  }
  // load aux
  ret->aux_arrays.clear();
  ret->aux_arrays.reserve(aux_names.size());
  for (const auto& name : aux_names) {
    auto it = input_nd_map.find(name);
    CHECK(it != input_nd_map.end());
    ret->aux_arrays.push_back(it->second);
  }

  // bind
  std::map<std::string, Context> ctx_map;
  std::vector<NDArray> grad_store(ret->arg_arrays.size());
  std::vector<OpReqType> grad_req(ret->arg_arrays.size(), kNullOp);


  ret->exec.reset(Executor::Bind(sym, ctx, ctx_map,
                                 ret->arg_arrays,
                                 grad_store, grad_req,
                                 ret->aux_arrays));
  ret->out_arrays = ret->exec->outputs();

  CHECK(out_names.size() == ret->out_arrays.size());
  ret->out_names = out_names;
  ret->out_names_buffer.resize(ret->out_names.size());
  std::transform(ret->out_names.begin(), ret->out_names.end(), ret->out_names_buffer.begin(),
      [](const std::string& name) { return name.c_str(); });

  ret->out_dtypes.resize(ret->out_arrays.size());
  std::transform(ret->out_arrays.begin(), ret->out_arrays.end(), ret->out_dtypes.begin(),
      [](const NDArray& nd) { return nd.dtype(); });

  ret->out_shapes.resize(ret->out_arrays.size());
  std::transform(ret->out_arrays.begin(), ret->out_arrays.end(), ret->out_shapes.begin(),
      [](const NDArray& nd) { return nd.shape(); });

  *out = ret;
  API_END_HANDLE_ERROR(delete ret);
}

int MXPredReshape(uint32_t num_input_nodes,
                  const char** input_keys,
                  const uint32_t* input_shape_indptr,
                  const uint32_t* input_shape_data,
                  PredictorHandle handle,
                  PredictorHandle* out) {
  _CreateExecutor(handle);
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  std::unique_ptr<MXAPIPredictor> ret(new MXAPIPredictor());

  API_BEGIN();
  // shape inference
  std::unordered_map<std::string, mxnet::TShape> new_shape;
  for (uint32_t i = 0; i < num_input_nodes; ++i) {
    new_shape[std::string(input_keys[i])] =
        mxnet::TShape(input_shape_data + input_shape_indptr[i],
            input_shape_data + input_shape_indptr[i + 1]);
  }
  ret->sym = p->sym;
  std::vector<std::string> arg_names = ret->sym.ListInputNames(Symbol::kReadOnlyArgs);
  std::vector<std::string> aux_names = ret->sym.ListInputNames(Symbol::kAuxiliaryStates);
  std::vector<std::string> out_names = ret->sym.ListOutputNames();
  mxnet::ShapeVector out_shapes(ret->sym.ListOutputNames().size());
  mxnet::ShapeVector aux_shapes(aux_names.size());
  mxnet::ShapeVector arg_shapes;
  ret->key2arg = p->key2arg;

  try {
    mxnet::ShapeVector in_shapes;
    in_shapes.reserve(arg_names.size());
    for (std::string key : ret->sym.ListInputNames(Symbol::kAll)) {
      if (new_shape.count(key) != 0) {
        in_shapes.push_back(new_shape[key]);
      } else {
        in_shapes.emplace_back();
      }
    }
    nnvm::Graph g; g.outputs = ret->sym.outputs;
    g = mxnet::exec::InferShape(std::move(g), std::move(in_shapes), "__shape__");
    bool infer_complete = (g.GetAttr<size_t>("shape_num_unknown_nodes") == 0);
    CHECK(infer_complete)
      << "The shape information of is not enough to get the shapes";
    CopyAttr(g.indexed_graph(),
             g.GetAttr<mxnet::ShapeVector>("shape"),
             &arg_shapes, &out_shapes, &aux_shapes);
  } catch (const mxnet::op::InferShapeError &err) {
    throw dmlc::Error(err.msg);
  }

  ret->arg_arrays = p->arg_arrays;
  ret->ctx = p->ctx;
  for (size_t i=0; i < arg_names.size(); ++i) {
    mxnet::TShape newShape = arg_shapes[i];
    NDArray &arr = p->arg_arrays[i];
    if (new_shape.count(arg_names[i]) != 0) {
      ret->arg_arrays[i].ReshapeAndAlloc(newShape);
    } else {
       CHECK_EQ(newShape.Size(), arr.shape().Size())
        << "arg " << arg_names[i]
        << " shape has been changed, only allow to change the shape of input data.";
    }
  }

  for (size_t i=0; i < aux_names.size(); ++i) {
    mxnet::TShape newShape = aux_shapes[i];
    NDArray &arr = p->aux_arrays[i];
    CHECK_EQ(newShape.Size(), arr.shape().Size())
      << "aux " << aux_names[i]
      << " shape has been changed, only allow to change the shape of input data.";
  }
  ret->aux_arrays = p->aux_arrays;

  // bind
  {
    std::map<std::string, Context> ctx_map;
    std::vector<NDArray> grad_store;
    grad_store.reserve(ret->arg_arrays.size());
    std::vector<OpReqType> grad_req(ret->arg_arrays.size(), kNullOp);

    ret->exec.reset(Executor::Bind(ret->sym, ret->ctx, ctx_map,
                                   ret->arg_arrays,
                                   grad_store, grad_req,
                                   ret->aux_arrays,
                                   p->exec.get()));
    ret->out_arrays = ret->exec->outputs();
    ret->out_dtypes = p->out_dtypes;

    CHECK(out_shapes.size() == ret->out_arrays.size());
    CHECK(out_names.size() == ret->out_arrays.size());

    ret->out_shapes = out_shapes;
    ret->out_names = out_names;

    ret->out_names_buffer.resize(ret->out_names.size());
    std::transform(ret->out_names.begin(), ret->out_names.end(), ret->out_names_buffer.begin(),
        [](const std::string& name) { return name.c_str(); });
  }
  *out = ret.release();
  API_END();
}

int MXPredGetOutputNames(PredictorHandle handle,
                         mx_uint *out_names_length,
                         const char ***out_names) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  *out_names = p->out_names_buffer.data();
  *out_names_length = p->out_names_buffer.size();
  API_END();
}

int MXPredGetOutputShape(PredictorHandle handle,
                         uint32_t out_index,
                         uint32_t** shape_data,
                         uint32_t* shape_ndim) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(out_index, p->out_arrays.size())
      << "Index exceed number of outputs";

  const mxnet::TShape& s = p->out_shapes[out_index];
  CHECK_GE(s.ndim(), 0);
  p->out_shapes_buffer.resize(s.ndim());
  nnvm::ShapeTypeCast(s.begin(), s.end(), p->out_shapes_buffer.data());
  *shape_data = p->out_shapes_buffer.data();
  *shape_ndim = p->out_shapes[out_index].ndim();
  API_END();
}

int MXPredGetOutputType(PredictorHandle handle,
                        uint32_t out_index,
                        int* out_dtype) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(out_index, p->out_arrays.size())
      << "Index exceed number of outputs, provided out_index should be less than "
      << p->out_arrays.size();

  const int s = p->out_dtypes[out_index];
  CHECK_GE(s, 0);
  out_dtype[out_index] = s;
  API_END();
}

int MXPredGetInputShape(PredictorHandle handle,
                        const char* key,
                        mx_uint** shape_data,
                        mx_uint* shape_ndim,
                        int* key_found) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  auto it = p->key2arg.find(key);
  if (it == p->key2arg.end()) {
    *key_found = 0;
  } else {
    *key_found = 1;
    const NDArray& nd = p->arg_arrays[it->second];
    const mxnet::TShape& s = nd.shape();
    p->in_shapes_buffer.resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), p->in_shapes_buffer.data());
    *shape_data = p->in_shapes_buffer.data();
    *shape_ndim = s.ndim();
  }
  API_END();
}

int MXPredGetInputType(PredictorHandle handle,
                       const char* key,
                       int* dtype,
                       int* key_found) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  auto it = p->key2arg.find(key);
  if (it == p->key2arg.end()) {
    *key_found = 0;
  } else {
    *key_found = 1;
    const NDArray& nd = p->arg_arrays[it->second];
    *dtype =  nd.dtype();
  }
  API_END();
}

int MXPredSetInput(PredictorHandle handle,
                   const char* key,
                   const float* data,
                   uint32_t size) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  auto it = p->key2arg.find(key);
  if (it == p->key2arg.end()) {
    LOG(FATAL) << "cannot find input key " << key;
  }
  NDArray& nd = p->arg_arrays[it->second];
  nd.SyncCopyFromCPU(data, size);
  API_END();
}

int MXPredForward(PredictorHandle handle) {
  _CreateExecutor(handle);
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  p->exec->Forward(false);
  API_END();
}

int MXPredPartialForward(PredictorHandle handle, int step, int* step_left) {
  _CreateExecutor(handle);
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  p->exec->PartialForward(false, step, step_left);
  API_END();
}

int MXPredGetOutput(PredictorHandle handle,
                    uint32_t index,
                    float* data,
                    uint32_t size) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->out_arrays.size())
      << "Output index out of range";
  const NDArray& nd = p->out_arrays[index];
  nd.SyncCopyToCPU(data, size);
  API_END();
}

int MXPredFree(PredictorHandle handle) {
  API_BEGIN();
  delete static_cast<MXAPIPredictor*>(handle);
  API_END();
}

int MXNDListCreate(const char* nd_file_bytes,
                   int nd_file_size,
                   NDListHandle *out,
                   uint32_t* out_length) {
  MXAPINDList* ret = new MXAPINDList();
  API_BEGIN();
  std::vector<NDArray> arrays;
  dmlc::MemoryFixedSizeStream fi((void*)nd_file_bytes, nd_file_size);  // NOLINT(*)
  NDArray::Load(&fi,
                &(arrays),
                &(ret->keys));
  if (ret->keys.size() == 0) {
    ret->keys.resize(arrays.size());
  }
  ret->indptr.push_back(0);
  for (auto &array : arrays) {
    mxnet::TShape shape = array.shape();
    size_t begin = ret->data.size();
    size_t size = shape.Size();
    ret->shapes.push_back(shape);
    ret->data.resize(begin + size);
    array.SyncCopyToCPU(dmlc::BeginPtr(ret->data) + begin, size);
    ret->indptr.push_back(begin + size);
  }
  *out = ret;
  *out_length = static_cast<uint32_t>(arrays.size());
  API_END();
}

int MXNDListGet(NDListHandle handle,
                uint32_t index,
                const char** out_key,
                const float** out_data,
                const uint32_t** out_shape,
                uint32_t* out_ndim) {
  MXAPINDList* p = static_cast<MXAPINDList*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->shapes.size())
      << "Index out of range";
  *out_key = p->keys[index].c_str();
  *out_data = dmlc::BeginPtr(p->data) + p->indptr[index];
  const mxnet::TShape& s = p->shapes[index];
  p->shapes_buffer.resize(s.ndim());
  nnvm::ShapeTypeCast(s.begin(), s.end(), p->shapes_buffer.data());
  *out_shape = p->shapes_buffer.data();
  *out_ndim = p->shapes[index].ndim();
  API_END();
}

int MXNDListGetEx(NDListHandle handle,
                  mx_uint index,
                  const char** out_key,
                  const void** out_data,
                  int* out_dtype,
                  const mx_uint** out_shape,
                  mx_uint* out_ndim) {
  MXAPINDListEx* p = static_cast<MXAPINDListEx*>(handle);
  API_BEGIN();
  CHECK_LT(index, p->arrays.size())
      << "Index out of range";
  *out_key = p->keys[index].c_str();
  const NDArray& nd = p->arrays[index];
  const TBlob& t = nd.data();
  *out_data = t.dptr_;
  *out_dtype = nd.dtype();
  const mxnet::TShape& s = nd.shape();
  p->shapes_buffer.resize(s.ndim());
  nnvm::ShapeTypeCast(s.begin(), s.end(), p->shapes_buffer.data());
  *out_shape = p->shapes_buffer.data();
  *out_ndim = s.ndim();
  API_END();
}


int MXPredSetMonitorCallback(PredictorHandle handle,
                             PredMonitorCallback callback,
                             void* callback_handle,
                             bool monitor_all) {
  MXAPIPredictor* p = static_cast<MXAPIPredictor*>(handle);
  API_BEGIN();
  PredMonitorCallback callback_temp = callback;
  void* callback_handle_temp = callback_handle;
  std::function<void(const char*, void*)> clbk
  = [callback_temp, callback_handle_temp](const char* name, void* handle) {
    callback_temp(name, handle, callback_handle_temp);
  };
  p->exec->SetMonitorCallback(clbk, monitor_all);
  API_END();
}

int MXNDListFree(NDListHandle handle) {
  API_BEGIN();
  delete static_cast<MXAPINDList*>(handle);
  API_END();
}
