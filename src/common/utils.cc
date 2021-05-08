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
 * \file utils.cc
 * \brief cpu implementation of util functions
 */

#include "./utils.h"
#include "../operator/tensor/cast_storage-inl.h"
#include "../operator/tensor/sparse_retain-inl.h"

namespace mxnet {
namespace common {

template<>
void CheckFormatWrapper<cpu>(const RunContext &rctx, const NDArray &input,
                             const TBlob &err_cpu, const bool full_check) {
  CheckFormatImpl<cpu>(rctx, input, err_cpu, full_check);
}

template<>
void SparseRetainOpForwardRspWrapper<cpu>(mshadow::Stream<cpu> *s,
                                          const NDArray& input_nd,
                                          const TBlob& idx_data,
                                          const OpReqType req,
                                          NDArray* output_nd) {
  mxnet::op::SparseRetainOpForwardRspImpl<cpu>(s, input_nd, idx_data, req, output_nd);
}

template<>
void CastStorageDispatch<cpu>(const OpContext& ctx,
                              const NDArray& input,
                              const NDArray& output) {
  mxnet::op::CastStorageComputeImpl<cpu>(ctx, input, output);
}

void ExecuteMonInputCallback(
    const nnvm::IndexedGraph &idx, const std::vector<NDArray *> &state_arrays,
    size_t nid, const std::function<void(const char *, const char *, void *)>
                    &monitor_callback) {
  static const auto &flist_inputs =
      nnvm::Op::GetAttr<nnvm::FListInputNames>("FListInputNames");
  std::vector<std::string> input_names;
  const nnvm::IndexedGraph::Node &inode = idx[nid];
  const nnvm::Node *node = inode.source;
  if (flist_inputs.count(node->op())) {
    input_names = flist_inputs[node->op()](node->attrs);
  } else {
    for (size_t i = 0; i < node->num_inputs(); ++i) {
      input_names.emplace_back("input" + std::to_string(i));
    }
  }

  for (size_t i = 0; i < node->num_inputs(); ++i) {
    const nnvm::NodeEntry &input = node->inputs[i];
    if (state_arrays[idx.entry_id(input)]->is_none()) {
      continue;
    }
    NDArray *cpy = new NDArray(*state_arrays[idx.entry_id(input)]);
    std::string name = inode.source->attrs.name + "_" + input_names[i];
    monitor_callback(name.c_str(), inode.source->op()->name.c_str(),
                     reinterpret_cast<void *>(cpy));
  }
}

void ExecuteMonOutputCallback(
    const nnvm::IndexedGraph &idx, const std::vector<NDArray *> &state_arrays,
    size_t nid, const std::function<void(const char *, const char *, void *)>
                    &monitor_callback) {
  static const auto &flist_outputs =
      nnvm::Op::GetAttr<nnvm::FListOutputNames>("FListOutputNames");
  std::vector<std::string> output_names;
  const nnvm::IndexedGraph::Node &inode = idx[nid];
  const nnvm::Node *node = inode.source;
  if (flist_outputs.count(node->op())) {
    output_names = flist_outputs[node->op()](node->attrs);
  } else {
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      output_names.emplace_back(std::to_string(i));
    }
  }

  for (size_t i = 0; i < node->num_outputs(); ++i) {
    if (state_arrays[idx.entry_id(nid, i)]->is_none()) {
      continue;
    }
    NDArray *cpy = new NDArray(*state_arrays[idx.entry_id(nid, i)]);
    std::string name = inode.source->attrs.name + "_" + output_names[i];
    monitor_callback(name.c_str(), inode.source->op()->name.c_str(),
                     reinterpret_cast<void *>(cpy));
  }
}

MShadowTypeInfo mshadow_type_info(const int type_flag) {
  using namespace mshadow;
  switch (type_flag) {
    case kFloat32:
      return MShadowTypeInfo("float32", sizeof(float));
    case kFloat64:
      return MShadowTypeInfo("float64", sizeof(double));
    case kFloat16:
      return MShadowTypeInfo("float16", 2, sizeof(float));
    case kUint8:
      return MShadowTypeInfo("uint8", sizeof(uint8_t), sizeof(index_t));
    case kUint16:
      return MShadowTypeInfo("uint16", sizeof(uint16_t), sizeof(index_t));
    case kInt32:
      return MShadowTypeInfo("int32", sizeof(int32_t));
    case kInt8:
      return MShadowTypeInfo("int8", sizeof(int8_t), sizeof(index_t));
    case kInt16:
      return MShadowTypeInfo("int16", sizeof(int16_t), sizeof(index_t));
    case kInt64:
      return MShadowTypeInfo("int64", sizeof(int64_t));
    case kBool:
      return MShadowTypeInfo("bool", sizeof(bool), sizeof(index_t));
    default:
      LOG(FATAL) << "Unknown type flag " << type_flag;
      return MShadowTypeInfo("INVALID", 1);
  }
}

}  // namespace common
}  // namespace mxnet
