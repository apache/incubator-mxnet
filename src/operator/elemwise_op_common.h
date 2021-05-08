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
* Copyright (c) 2016 by Contributors
* \file elemwise_op_common.h
* \brief common function used for broadcasting and reducing
* \author Xingjian Shi
*/
#ifndef MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
#define MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mxnet_op.h"

namespace mxnet {
namespace op {

/*! \brief storge type inference function for elemwise operators.
 *         It infers output stypes the same as input stypes when input stypes are the same
 *  \tparam cpu_only whether fcompute_ex can only be dispatched on cpu context
 *  \tparam rsp whether row sparse stype is supported
 *  \tparam rsp whether csr stype is supported
 */
template<bool cpu_only, bool rsp, bool csr>
inline bool ElemwiseStorageAttr(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  using namespace common;
  bool dispatched = false;
  const bool invalid_ctx = cpu_only && dev_mask != mshadow::cpu::kDevMask;
  const auto dispatch_ex = invalid_ctx ? DispatchMode::kFComputeFallback :
                                         DispatchMode::kFComputeEx;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, dns ... -> dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && rsp && ContainsOnlyStorage(*in_attrs, kRowSparseStorage)) {
    // rsp, rsp, ... -> rsp
    dispatched = storage_type_assign(out_attrs, kRowSparseStorage,
                                     dispatch_mode, dispatch_ex);
  }
  if (!dispatched && csr && common::ContainsOnlyStorage(*in_attrs, kCSRStorage)) {
    // csr, csr, ... -> csr
    dispatched = storage_type_assign(out_attrs, kCSRStorage,
                                     dispatch_mode, dispatch_ex);
  }
  if (!dispatched && in_attrs->size() == 3U && in_attrs->at(0) == kDefaultStorage &&
      in_attrs->at(1) == kCSRStorage && in_attrs->at(2) == kDefaultStorage) {
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, dispatch_ex);
  }
  if (!dispatched && in_attrs->size() > 4U && ContainsStorageType(*in_attrs, kDefaultStorage)) {
    // *, dense, * -> dense
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, dispatch_ex);
  }
  if (!dispatched) {
    dispatch_fallback(out_attrs, dispatch_mode);
  }
  if (static_cast<DispatchMode>(*dispatch_mode) == DispatchMode::kFComputeFallback) {
    LogStorageFallback(attrs, dev_mask, in_attrs, out_attrs);
  }
  return true;
}

/*! \brief storge type inference function for elemwise operators.
 *         It infers output stypes the same as input stypes when input stypes are the same
 *  \tparam n_in the number of inputs
 *  \tparam n_in the number of outputs
 *  \tparam cpu_only whether fcompute_ex can only be dispatched on cpu context
 *  \tparam rsp whether row sparse stype is supported
 *  \tparam rsp whether csr stype is supported
 */
template<index_t n_in, index_t n_out, bool cpu_only, bool rsp, bool csr>
inline bool ElemwiseStorageType(const nnvm::NodeAttrs& attrs,
                                const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), n_in);
  CHECK_EQ(out_attrs->size(), n_out);
  return ElemwiseStorageAttr<cpu_only, rsp, csr>(attrs, dev_mask, dispatch_mode,
                                                 in_attrs, out_attrs);
}

template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         std::string (*attr_string)(const AttrType&),
         index_t n_in = -1, index_t n_out = -1>
inline bool ElemwiseAttrHelper(const std::string& node_name,
                               std::vector<AttrType> *in_attrs,
                               std::vector<AttrType> *out_attrs,
                               const AttrType& none) {
  AttrType dattr = none;
  size_t in_size = in_attrs->size();
  size_t out_size = out_attrs->size();
  if (n_in != -1)
    in_size = static_cast<size_t>(n_in);
  if (n_out != -1)
    out_size = static_cast<size_t>(n_out);

  CHECK_LE(in_size, in_attrs->size());
  CHECK_LE(out_size, out_attrs->size());
  auto deduce = [&](const std::vector<AttrType>& vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        CHECK(assign(&dattr, vec.at(i)))
          << "Incompatible attr in node " << node_name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string(vec.at(i));
      }
    };
  deduce(*in_attrs, in_size, "input");
  if (reverse_infer)
      deduce(*out_attrs, out_size, "output");

  auto write = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        CHECK(assign(&(vec->at(i)), dattr))
          << "Incompatible attr in node " << node_name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string(vec->at(i));
      }
    };
  write(in_attrs, in_size, "input");
  write(out_attrs, out_size, "output");

  if (is_none(dattr))
      return false;
  return true;
}


template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         std::string (*attr_string)(const AttrType&),
         index_t n_in = -1, index_t n_out = -1>
inline bool ElemwiseAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs,
                         const AttrType& none) {
  return ElemwiseAttrHelper<AttrType, is_none,
                            assign, reverse_infer,
                            attr_string, n_in,
                            n_out>(attrs.name, in_attrs, out_attrs, none);
}

template<index_t n_in, index_t n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<mxnet::TShape, shape_is_none, shape_assign, true, shape_string>(
    attrs, in_attrs, out_attrs, mxnet::TShape());
}

template<index_t n_in, index_t n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

// Special case of ElemwiseType. Constrains dtype to integer types
template<index_t n_in, index_t n_out>
inline bool ElemwiseIntType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK(in_attrs->at(0) == mshadow::kInt64 ||
        in_attrs->at(0) == mshadow::kInt32 ||
        in_attrs->at(0) == mshadow::kInt16 ||
        in_attrs->at(0) == mshadow::kUint16 ||
        in_attrs->at(0) == mshadow::kInt8 ||
        in_attrs->at(0) == mshadow::kUint8 ||
        in_attrs->at(0) == mshadow::kBool) << "Only supports integer types.";
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

// Transfer gradient and input to FGradient function
struct ElemwiseGradUseIn {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    return MakeNonlossGradNode(op_name, n, ograds, n->inputs, n->attrs.dict);
  }
};

// Transfer gradient and output to FGradient function
struct ElemwiseGradUseOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads;
    uint32_t n_out = n->num_outputs();
    for (uint32_t i = 0; i < n_out; ++i) {
      heads.emplace_back(n, i, 0);
    }
    return MakeNonlossGradNode(op_name, n, ograds, heads, n->attrs.dict);
  }
};

// Transfer gradient and input and output to FGradient function
struct ElemwiseGradUseInOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    for (auto& h : n->inputs) {
      heads.push_back(h);
    }
    uint32_t n_out = n->num_outputs();
    for (uint32_t i = 0; i < n_out; ++i) {
      heads.emplace_back(n, i, 0);
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

// Transfer only gradient to FGradient function
struct ElemwiseGradUseNone {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    return MakeNonlossGradNode(op_name, n, ograds, {}, n->attrs.dict);
  }
};

struct CloneGradient {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> ret;
    const size_t input_count = n->inputs.size();
    ret.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
      ret.emplace_back(ograds[0]);
    }
    return ret;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
