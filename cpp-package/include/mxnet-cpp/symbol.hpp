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
 * \file symbol.hpp
 * \brief implementation of the symbol
 * \author Zhang Chen, Chuntao Hong
 */

#ifndef MXNET_CPP_SYMBOL_HPP_
#define MXNET_CPP_SYMBOL_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dmlc/logging.h"
#include "mxnet-cpp/symbol.h"

#include "mxnet-cpp/op_suppl.h"

namespace mxnet {
namespace cpp {
inline OpMap*& Symbol::op_map() {
  static OpMap* op_map_ = new OpMap();
  return op_map_;
}
inline Symbol::Symbol(SymbolHandle handle) {
  blob_ptr_ = std::make_shared<SymBlob>(handle);
}
inline Symbol::Symbol(const char *name) {
  SymbolHandle handle;
  CHECK_EQ(MXSymbolCreateVariable(name, &(handle)), 0);
  blob_ptr_ = std::make_shared<SymBlob>(handle);
}
inline Symbol::Symbol(const std::string &name) : Symbol(name.c_str()) {}
inline Symbol Symbol::Variable(const std::string &name) { return Symbol(name); }
inline Symbol Symbol::operator+(const Symbol &rhs) const { return _Plus(*this, rhs); }
inline Symbol Symbol::operator-(const Symbol &rhs) const { return _Minus(*this, rhs); }
inline Symbol Symbol::operator*(const Symbol &rhs) const { return _Mul(*this, rhs); }
inline Symbol Symbol::operator/(const Symbol &rhs) const { return _Div(*this, rhs); }
inline Symbol Symbol::operator%(const Symbol &rhs) const { return _Mod(*this, rhs); }
inline Symbol Symbol::operator+(mx_float scalar) const {
  return _PlusScalar(*this, scalar);
}
inline Symbol Symbol::operator-(mx_float scalar) const {
  return _MinusScalar(*this, scalar);
}
inline Symbol Symbol::operator*(mx_float scalar) const {
  return _MulScalar(*this, scalar);
}
inline Symbol Symbol::operator/(mx_float scalar) const {
  return _DivScalar(*this, scalar);
}
inline Symbol Symbol::operator%(mx_float scalar) const {
  return _ModScalar(*this, scalar);
}
inline Symbol Symbol::operator[](int index) {
  SymbolHandle out;
  MXSymbolGetOutput(GetHandle(), index, &out);
  return Symbol(out);
}
inline Symbol Symbol::operator[](const std::string &index) {
  auto outputs = ListOutputs();
  for (mx_uint i = 0; i < outputs.size(); ++i) {
    if (outputs[i] == index) {
      return (*this)[i];
    }
  }
  LOG(FATAL) << "Cannot find output that matches name " << index;
  return (*this)[0];
}
inline Symbol Symbol::Group(const std::vector<Symbol> &symbols) {
  SymbolHandle out;
  std::vector<SymbolHandle> handle_list;
  for (const auto &t : symbols) {
    handle_list.push_back(t.GetHandle());
  }
  MXSymbolCreateGroup(handle_list.size(), handle_list.data(), &out);
  return Symbol(out);
}
inline Symbol Symbol::Load(const std::string &file_name) {
  op_map();
  SymbolHandle handle;
  CHECK_EQ(MXSymbolCreateFromFile(file_name.c_str(), &(handle)), 0);
  return Symbol(handle);
}
inline Symbol Symbol::LoadJSON(const std::string &json_str) {
  op_map();
  SymbolHandle handle;
  CHECK_EQ(MXSymbolCreateFromJSON(json_str.c_str(), &(handle)), 0);
  return Symbol(handle);
}
inline void Symbol::Save(const std::string &file_name) const {
  CHECK_EQ(MXSymbolSaveToFile(GetHandle(), file_name.c_str()), 0);
}
inline std::string Symbol::ToJSON() const {
  const char *out_json;
  CHECK_EQ(MXSymbolSaveToJSON(GetHandle(), &out_json), 0);
  return std::string(out_json);
}
inline Symbol Symbol::GetInternals() const {
  SymbolHandle handle;
  CHECK_EQ(MXSymbolGetInternals(GetHandle(), &handle), 0);
  return Symbol(handle);
}
inline Symbol::Symbol(const std::string &operator_name, const std::string &name,
               std::vector<const char *> input_keys,
               std::vector<SymbolHandle> input_values,
               std::vector<const char *> config_keys,
               std::vector<const char *> config_values) {
  SymbolHandle handle;
  AtomicSymbolCreator creator = op_map()->GetSymbolCreator(operator_name);
  MXSymbolCreateAtomicSymbol(creator, config_keys.size(), config_keys.data(),
                             config_values.data(), &handle);
  MXSymbolCompose(handle, operator_name.c_str(), input_keys.size(),
                  input_keys.data(), input_values.data());
  blob_ptr_ = std::make_shared<SymBlob>(handle);
}

inline Symbol Symbol::Copy() const {
  SymbolHandle handle;
  CHECK_EQ(MXSymbolCopy(GetHandle(), &handle), 0);
  return Symbol(handle);
}

inline std::vector<std::string> Symbol::ListArguments() const {
  std::vector<std::string> ret;
  mx_uint size;
  const char **sarr;
  MXSymbolListArguments(GetHandle(), &size, &sarr);
  for (mx_uint i = 0; i < size; ++i) {
    ret.push_back(std::string(sarr[i]));
  }
  return ret;
}

inline std::vector<std::string> Symbol::ListInputs() const {
  std::vector<std::string> ret;
  mx_uint size;
  const char **sarr;
  NNSymbolListInputNames(GetHandle(), 0, &size, &sarr);
  for (mx_uint i = 0; i < size; ++i) {
    ret.push_back(std::string(sarr[i]));
  }
  return ret;
}

inline std::vector<std::string> Symbol::ListOutputs() const {
  std::vector<std::string> ret;
  mx_uint size;
  const char **sarr;
  MXSymbolListOutputs(GetHandle(), &size, &sarr);
  for (mx_uint i = 0; i < size; ++i) {
    ret.push_back(std::string(sarr[i]));
  }
  return ret;
}
inline std::vector<std::string> Symbol::ListAuxiliaryStates() const {
  std::vector<std::string> ret;
  mx_uint size;
  const char **sarr;
  MXSymbolListAuxiliaryStates(GetHandle(), &size, &sarr);
  for (mx_uint i = 0; i < size; ++i) {
    ret.push_back(std::string(sarr[i]));
  }
  return ret;
}

inline std::map<std::string, std::string> Symbol::ListAttributes() const {
    mx_uint size;
    const char** pairs;
    CHECK_EQ(MXSymbolListAttrShallow(GetHandle(), &size, &pairs), 0);
    std::map<std::string, std::string> attributes;
    for (mx_uint i = 0; i < size; ++i) {
        // pairs is 2 * size with key, value pairs according to
        //   https://github.com/apache/incubator-mxnet/blob/master/include/mxnet/c_api.h#L1428
        attributes[pairs[2 * i]] = pairs[2 * i + 1];
    }
    return attributes;
}

inline void Symbol::SetAttribute(const std::string &key, const std::string &value) {
    CHECK_EQ(MXSymbolSetAttr(GetHandle(), key.c_str(), value.c_str()), 0);
}

inline void Symbol::SetAttributes(const std::map<std::string, std::string> &attrs) {
    for (const auto& kv : attrs) {
        SetAttribute(kv.first, kv.second);
    }
}

inline mx_uint Symbol::GetNumOutputs() const {
    mx_uint numOutputs;
    CHECK_EQ(MXSymbolGetNumOutputs(GetHandle(), &numOutputs), 0);
    return numOutputs;
}

inline mxnet::cpp::Symbol Symbol::GetBackendSymbol(const std::string &backendName) const {
    SymbolHandle symbolHandle;
    CHECK_EQ(MXGenBackendSubgraph(GetHandle(), backendName.c_str(), &symbolHandle), 0);
    return mxnet::cpp::Symbol(symbolHandle);
}

inline std::string Symbol::GetName() const {
  int success;
  const char* out_name;
  CHECK_EQ(MXSymbolGetName(GetHandle(), &out_name, &success), 0);
  CHECK_EQ(success, 1);
  return std::string(out_name);
}

inline void Symbol::InferShape(
    const std::map<std::string, std::vector<mx_uint> > &arg_shapes,
    std::vector<std::vector<mx_uint> > *in_shape,
    std::vector<std::vector<mx_uint> > *aux_shape,
    std::vector<std::vector<mx_uint> > *out_shape) const {

  std::vector<const char *> keys;
  std::vector<mx_uint> arg_ind_ptr;
  std::vector<int> arg_shape_data;

  for (const auto &arg : arg_shapes) {
    keys.push_back(arg.first.c_str());
    arg_ind_ptr.push_back(arg_shape_data.size());
    for (auto i : arg.second) {
      arg_shape_data.push_back(i);
    }
  }
  arg_ind_ptr.push_back(arg_shape_data.size());

  mx_uint in_shape_size;
  const int *in_shape_ndim;
  const int **in_shape_data;
  mx_uint out_shape_size;
  const int *out_shape_ndim;
  const int **out_shape_data;
  mx_uint aux_shape_size;
  const int *aux_shape_ndim;
  const int **aux_shape_data;
  int complete;

  CHECK_EQ(MXSymbolInferShape(GetHandle(), keys.size(), keys.data(),
                              arg_ind_ptr.data(), arg_shape_data.data(),
                              &in_shape_size, &in_shape_ndim, &in_shape_data,
                              &out_shape_size, &out_shape_ndim, &out_shape_data,
                              &aux_shape_size, &aux_shape_ndim, &aux_shape_data,
                              &complete),
           0);

  if (complete) {
    for (mx_uint i = 0; i < in_shape_size; ++i) {
      in_shape->push_back(std::vector<mx_uint>());
      for (int j = 0; j < in_shape_ndim[i]; ++j) {
        (*in_shape)[i].push_back(in_shape_data[i][j]);
      }
    }
    for (mx_uint i = 0; i < aux_shape_size; ++i) {
      aux_shape->push_back(std::vector<mx_uint>());
      for (int j = 0; j < aux_shape_ndim[i]; ++j) {
        (*aux_shape)[i].push_back(aux_shape_data[i][j]);
      }
    }
    for (mx_uint i = 0; i < out_shape_size; ++i) {
      out_shape->push_back(std::vector<mx_uint>());
      for (int j = 0; j < out_shape_ndim[i]; ++j) {
        (*out_shape)[i].push_back(out_shape_data[i][j]);
      }
    }
  }
}

inline void Symbol::InferExecutorArrays(
    const Context &context, std::vector<NDArray> *arg_arrays,
    std::vector<NDArray> *grad_arrays, std::vector<OpReqType> *grad_reqs,
    std::vector<NDArray> *aux_arrays,
    const std::map<std::string, NDArray> &args_map,
    const std::map<std::string, NDArray> &arg_grad_store,
    const std::map<std::string, OpReqType> &grad_req_type,
    const std::map<std::string, NDArray> &aux_map) const {

  const auto arg_name_list = ListArguments();
  std::vector<std::vector<mx_uint> > in_shapes, aux_shapes, out_shapes;
  std::map<std::string, std::vector<mx_uint> > arg_shapes;

  for (const auto &arg_name : arg_name_list) {
    auto iter = args_map.find(arg_name);
    if (iter != args_map.end()) {
      arg_shapes[arg_name] = iter->second.GetShape();
    }
  }

  InferShape(arg_shapes, &in_shapes, &aux_shapes, &out_shapes);

  for (size_t i = 0; i < in_shapes.size(); ++i) {
    const auto &shape = in_shapes[i];
    const auto &arg_name = arg_name_list[i];
    auto iter_arg = args_map.find(arg_name);
    if (iter_arg != args_map.end()) {
      arg_arrays->push_back(iter_arg->second);
    } else {
      arg_arrays->push_back(NDArray(shape, context, false));
      NDArray::SampleGaussian(0, 1, &arg_arrays->back());
    }
    auto iter_grad = arg_grad_store.find(arg_name);
    if (iter_grad != arg_grad_store.end()) {
      grad_arrays->push_back(iter_grad->second);
    } else {
      grad_arrays->push_back(NDArray(shape, context, false));
    }
    auto iter_req = grad_req_type.find(arg_name);
    if (iter_req != grad_req_type.end()) {
      grad_reqs->push_back(iter_req->second);
    } else if (arg_name.rfind("data") != std::string::npos
            || arg_name.rfind("label") != std::string::npos) {
      grad_reqs->push_back(OpReqType::kNullOp);
    } else {
      grad_reqs->push_back(OpReqType::kWriteTo);
    }
  }

  const auto aux_name_list = ListAuxiliaryStates();
  for (size_t i = 0; i < aux_shapes.size(); ++i) {
    const auto &shape = aux_shapes[i];
    const auto &aux_name = aux_name_list[i];
    auto iter_aux = aux_map.find(aux_name);
    if (iter_aux != aux_map.end()) {
      aux_arrays->push_back(iter_aux->second);
    } else {
      aux_arrays->push_back(NDArray(shape, context, false));
      NDArray::SampleGaussian(0, 1, &aux_arrays->back());
    }
  }
}
inline void Symbol::InferArgsMap(
    const Context &context, std::map<std::string, NDArray> *args_map,
    const std::map<std::string, NDArray> &known_args) const {

  const auto arg_name_list = ListArguments();
  std::vector<std::vector<mx_uint> > in_shapes, aux_shapes, out_shapes;
  std::map<std::string, std::vector<mx_uint> > arg_shapes;

  for (const auto &arg_name : arg_name_list) {
    auto iter = known_args.find(arg_name);
    if (iter != known_args.end()) {
      arg_shapes[arg_name] = iter->second.GetShape();
    }
  }

  InferShape(arg_shapes, &in_shapes, &aux_shapes, &out_shapes);

  for (size_t i = 0; i < in_shapes.size(); ++i) {
    const auto &shape = in_shapes[i];
    const auto &arg_name = arg_name_list[i];
    auto iter_arg = known_args.find(arg_name);
    if (iter_arg != known_args.end()) {
      (*args_map)[arg_name] = iter_arg->second;
    } else {
      (*args_map)[arg_name] = NDArray(shape, context, false);
      NDArray::SampleGaussian(0, 1, &(*args_map)[arg_name]);
    }
  }
}

inline Executor *Symbol::SimpleBind(
    const Context &context, const std::map<std::string, NDArray> &args_map,
    const std::map<std::string, NDArray> &arg_grad_store,
    const std::map<std::string, OpReqType> &grad_req_type,
    const std::map<std::string, NDArray> &aux_map) {
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<OpReqType> grad_reqs;
  std::vector<NDArray> aux_arrays;

  InferExecutorArrays(context, &arg_arrays, &grad_arrays, &grad_reqs,
                      &aux_arrays, args_map, arg_grad_store, grad_req_type,
                      aux_map);

  return new Executor(*this, context, arg_arrays, grad_arrays, grad_reqs,
                      aux_arrays);
}

inline Executor *Symbol::Bind(const Context &context,
                       const std::vector<NDArray> &arg_arrays,
                       const std::vector<NDArray> &grad_arrays,
                       const std::vector<OpReqType> &grad_reqs,
                       const std::vector<NDArray> &aux_arrays,
                       const std::map<std::string, Context> &group_to_ctx,
                       Executor *shared_exec) {
  return new Executor(*this, context, arg_arrays, grad_arrays, grad_reqs,
                      aux_arrays, group_to_ctx, shared_exec);
}
inline Symbol operator+(mx_float lhs, const Symbol &rhs) { return rhs + lhs; }
inline Symbol operator-(mx_float lhs, const Symbol &rhs) {
  return mxnet::cpp::_RMinusScalar(lhs, rhs);
}
inline Symbol operator*(mx_float lhs, const Symbol &rhs) { return rhs * lhs; }
inline Symbol operator/(mx_float lhs, const Symbol &rhs) {
  return mxnet::cpp::_RDivScalar(lhs, rhs);
}
inline Symbol operator%(mx_float lhs, const Symbol &rhs) {
  return mxnet::cpp::_RModScalar(lhs, rhs);
}
}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_SYMBOL_HPP_
