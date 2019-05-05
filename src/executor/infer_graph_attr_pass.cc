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
 * \file infer_graph_attr_pass.cc
 * \brief infer graph shape, dtype, and storage type
 */

#include <mxnet/op_attr_types.h>
#include <mxnet/graph_attr_types.h>
#include <mxnet/imperative.h>
#include "./exec_pass.h"
#include "../operator/operator_common.h"
#include "../common/exec_utils.h"

namespace mxnet {
namespace exec {

template<typename AttrType, typename FInfer>
bool ApplyOpInferAttr(const nnvm::Graph& g,
                      const FInfer& finfer,
                      const NodeAttrs& attrs,
                      const uint32_t nid,
                      std::vector<AttrType>* in_attrs,
                      std::vector<AttrType>* out_attrs,
                      DispatchMode* dispatch_mode) {
  return finfer(attrs, in_attrs, out_attrs);
}

template<>
bool ApplyOpInferAttr<int, FInferStorageType>(const nnvm::Graph& g,
                                              const FInferStorageType& finfer,
                                              const NodeAttrs& attrs,
                                              const uint32_t nid,
                                              std::vector<int>* in_attrs,
                                              std::vector<int>* out_attrs,
                                              DispatchMode* dispatch_mode) {
  const DevMaskVector& dev_masks = g.GetAttr<DevMaskVector>("dev_mask");
  const bool success = finfer(attrs, dev_masks[nid], dispatch_mode, in_attrs, out_attrs);
  if (!success) {
    LOG(FATAL) << "Operator not implemented: "
               << common::operator_stype_string(attrs, dev_masks[nid], *in_attrs, *out_attrs);
  }
  if (*dispatch_mode == DispatchMode::kFComputeFallback) {
    common::LogStorageFallback(attrs, dev_masks[nid], in_attrs, out_attrs);
  }
  return true;
}

/*!\brief
 * This is a duplicate of the InferAttr function in nnvm with minor modification
 * to support inferring storage type whose function signature is different from
 * shape/type inference functions'. The nnvm InferAttr will be deprecated
 * in the future. Please use interfaces InferShape, InferType, and InferStorageType
 * to call this function.
 *
 * \param ret graph used for attribute inference
 * \param emmpty_val empty value of the attribute
 * \param infer_name name of the function used for attribute inference
 * \param input_name name of the attribute in the graph used to store the
 *                   input data for attribute inference
 * \param attr_key_name name of the attribute used for inference for variable nodes
 * \param attr_name name of the inferred attribute
 * \param unknown_name name of the attribute storing number of entries
 *                     impossible to infer
 * \param fis_none function returning true for not fully inferred values
 * \param fdefault default function used for inference if the node does not
 *                 provide its own implementation.
 * \param bwd_identity_assign whether the attributes of forward NDArray and backward
 *                            NDArray have to be the same. False only for storage
 *                            type inference
 * \param dispatch_mode_name name of the dispatch mode attribute on the node. Used for
 *                           storage type inference
 * \param default_mode_val default value of the dispatch mode attribute on the node. Used
 *                         for storage type inference
 */
template<typename AttrType, typename FInferType, typename IsNone, typename FDefault>
nnvm::Graph InferAttr(nnvm::Graph &&ret,
                      const AttrType empty_val,
                      const char* infer_name,
                      const char* input_name,
                      const char* attr_key_name,
                      const char* attr_name,
                      const char* unknown_name,
                      IsNone fis_none,
                      FDefault fdefault,
                      bool bwd_identity_assign,
                      const char* dispatch_mode_name,
                      const DispatchMode default_mode_val = DispatchMode::kUndefined) {
  using nnvm::IndexedGraph;
  using nnvm::Op;
  using AttrVector = std::vector<AttrType>;
  using NodeAttrVector = std::vector<DispatchMode>;
  using dmlc::any;

  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_shape =
      Op::GetAttr<FInferType>(infer_name);
  static auto& is_backward =
      Op::GetAttr<nnvm::TIsBackward>("TIsBackward");
  // gradient function, used to get node correspondence.
  static auto& fgrad =
      Op::GetAttr<nnvm::FGradient>("FGradient");
  // reshape shape vector
  AttrVector rshape;
  // dispatch mode vector
  DispatchModeVector dispatch_modes;
  if (ret.attrs.count(attr_name) != 0) {
    rshape = ret.MoveCopyAttr<AttrVector>(attr_name);
  } else {
    rshape.resize(idx.num_node_entries(), empty_val);
  }

  if (ret.attrs.count(input_name) != 0) {
    const AttrVector& shape_args = ret.GetAttr<AttrVector>(input_name);
    CHECK_LE(shape_args.size(), idx.input_nodes().size())
        << "More provided " << attr_name << "s than number of arguments.";
    for (size_t i = 0; i < shape_args.size(); ++i) {
      rshape[idx.entry_id(idx.input_nodes()[i], 0)] = shape_args[i];
    }
  }

  // get the shape hints
  std::string shape_hints_key = std::string(attr_name) + "_hints";
  if (ret.attrs.count(shape_hints_key)) {
    nnvm::NodeEntryMap<AttrType> shape_hints =
      ret.GetAttr<nnvm::NodeEntryMap<AttrType>>(shape_hints_key);
    for (const auto& kv : shape_hints) {
      nnvm::NodeEntry e = kv.first;
      if (idx.exist(e.node.get())) {
        rshape[idx.entry_id(kv.first)] = kv.second;
      }
    }
  }

  std::string shape_attr_key;
  if (ret.attrs.count(attr_key_name) != 0) {
    shape_attr_key = ret.GetAttr<std::string>(attr_key_name);
    // erase the provided arguments
    ret.attrs.erase(attr_key_name);
  }

  // limit inference to part of the graph
  uint32_t node_start = 0, node_end = idx.num_nodes();
  if (ret.attrs.count("node_range")) {
    const auto& range = ret.GetAttr<std::pair<uint32_t, uint32_t> >("node_range");
    node_start = range.first;
    node_end = range.second;
    CHECK_GE(node_start, 0);
    CHECK_LE(node_end, idx.num_nodes());
    ret.attrs.erase("node_range");
  }
  uint32_t entry_start = 0, entry_end = idx.num_node_entries();
  if (ret.attrs.count("entry_range")) {
    const auto& range = ret.GetAttr<std::pair<uint32_t, uint32_t> >("entry_range");
    entry_start = range.first;
    entry_end = range.second;
    CHECK_GE(entry_start, 0);
    CHECK_LE(entry_end, idx.num_node_entries());
    ret.attrs.erase("entry_range");
  }
  // populate the node attribute vector
  if (dispatch_mode_name != nullptr) {
    if (ret.attrs.count(dispatch_mode_name) != 0) {
      dispatch_modes = ret.MoveCopyAttr<NodeAttrVector>(dispatch_mode_name);
    } else {
      LOG(FATAL) << "Node attribute " << dispatch_mode_name << " does not exist in the graph";
    }
  }

  // Temp space for shape inference.
  std::vector<AttrType> ishape, oshape;

  // inference step function for nid
  auto infer_step = [&](uint32_t nid, bool last_iter) {
    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      CHECK(inode.source->op() == nullptr);
      CHECK_EQ(num_outputs, 1U);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
      if (shape_attr_key.length() != 0 && fis_none(rshape[out_ent_id])) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          std::istringstream is(it->second);
          CHECK(is >> rshape[out_ent_id]) << "Invalid attribute";
        }
      }
      // assign a default value to node attribute
      if (dispatch_mode_name != nullptr) {
        op::dispatch_mode_assign(&dispatch_modes[nid], default_mode_val);
      }
    } else if (is_backward.get(inode.source->op(), false) &&
               inode.control_deps.size() && bwd_identity_assign) {
      CHECK(dispatch_mode_name == nullptr)
        << "Backward inference for node attributes is not available";
      CHECK_GE(inode.control_deps.size(), 1U)
        << "BackwardOp need to have control_deps to its forward op";
      const IndexedGraph::Node& fnode = idx[inode.control_deps[0]];
      nnvm::NodePtr fwd_ptr = inode.source->control_deps[0];
      CHECK(fwd_ptr->op() != nullptr) << "Forward op cannot be a variable";
      // use gradient function to find out the correspondence.
      std::vector<nnvm::NodeEntry> ograd(fwd_ptr->num_outputs());
      for (size_t i = 0; i < ograd.size(); ++i) {
        ograd[i].index = static_cast<uint32_t>(i);
      }
      // input gradient list
      auto igrad = fgrad[fwd_ptr->op()](fwd_ptr, ograd);
      const nnvm::Node* igrad_node = nullptr;
      // Input gradient assignement
      for (size_t i = 0; i < igrad.size(); ++i) {
        if (igrad[i].node->op() == inode.source->op()) {
          uint32_t eid = idx.entry_id(nid, igrad[i].index);
          if (fis_none(rshape[eid])) {
            rshape[eid] = rshape[idx.entry_id(fnode.inputs[i])];
          } else if (!fis_none(rshape[idx.entry_id(fnode.inputs[i])])) {
            // Need to skip empty forward shape, because it may not be
            // available now and it is possible to infer the forward
            // shape in one of the next a few passes
            CHECK_EQ(rshape[eid], rshape[idx.entry_id(fnode.inputs[i])])
                << "Backward shape inconsistent with the forward shape";
          }
          if (igrad_node == nullptr) {
            igrad_node = igrad[i].node.get();
          } else {
            CHECK(igrad_node == igrad[i].node.get());
          }
        }
      }
      // out grad entries
      CHECK(igrad_node != nullptr)
        << "Cannot find matching backward op for " << inode.source->attrs.name;
      for (size_t i = 0; i < igrad_node->inputs.size(); ++i) {
        const nnvm::NodeEntry& e = igrad_node->inputs[i];
        if (e.node == nullptr) {
          uint32_t eid = idx.entry_id(inode.inputs[i]);
          if (fis_none(rshape[eid])) {
            rshape[eid] = rshape[idx.entry_id(inode.control_deps[0], e.index)];
          }
        }
      }
    } else {
      DispatchMode* dispatch_mode = nullptr;
      bool forward_known = true;
      // Forward operator inference.
      ishape.resize(num_inputs, empty_val);
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = rshape[idx.entry_id(inode.inputs[i])];
        if (fis_none(ishape[i])) forward_known = false;
      }
      oshape.resize(num_outputs, empty_val);
      for (uint32_t i = 0; i < oshape.size(); ++i) {
        oshape[i] = rshape[idx.entry_id(nid, i)];
        if (fis_none(oshape[i])) forward_known = false;
      }
      if (dispatch_mode_name != nullptr) {
        dispatch_mode = &dispatch_modes[nid];
        if (dispatch_modes[nid] == DispatchMode::kUndefined) forward_known = false;
      }
      auto finfer = finfer_shape.get(inode.source->op(), fdefault);
      if (!forward_known) {
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            forward_known = ApplyOpInferAttr(ret, finfer, inode.source->attrs,
                                             nid, &ishape, &oshape, dispatch_mode);
          } catch (const std::exception& e) {
            throw dmlc::Error("Error in operator " + inode.source->attrs.name + ": " + e.what());
          }
        } else {
          CHECK(!last_iter)
              << "Attribute " << infer_name
              << " is not registed by op " << inode.source->op()->name
              << " we are not able to complete the inference because of this";
        }
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        rshape[idx.entry_id(inode.inputs[i])] = ishape[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        rshape[idx.entry_id(nid, i)] = oshape[i];
      }
    }
  };

  size_t last_num_unknown;
  size_t num_unknown_dispatch_mode = dispatch_mode_name ? node_end - node_start : 0;
  size_t num_unknown_entry_attr = entry_end - entry_start;
  size_t num_unknown = num_unknown_entry_attr + num_unknown_dispatch_mode;
  int i = 0;
  do {
    if (i % 2 == 0) {
      for (uint32_t nid = node_start; nid < node_end; ++nid) {
        infer_step(nid, false);
      }
    } else {
      // backward inference
      for (uint32_t i = node_end; i != node_start; --i) {
        infer_step(i - 1, false);
      }
    }
    last_num_unknown = num_unknown;
    num_unknown = 0;
    for (size_t j = entry_start; j < entry_end; ++j) {
      if (fis_none(rshape[j])) {
        ++num_unknown;
      }
    }
    if (dispatch_mode_name) {
      for (size_t i = node_start; i < node_end; i++) {
        if (dispatch_modes[i] == DispatchMode::kUndefined) ++num_unknown;
      }
    }
    ++i;
  } while (num_unknown > 0 && last_num_unknown > num_unknown);
  // set the shapes
  ret.attrs[attr_name] = std::make_shared<any>(std::move(rshape));
  // set the shapes
  if (dispatch_mode_name) {
    ret.attrs[dispatch_mode_name] = std::make_shared<any>(std::move(dispatch_modes));
  }
  // number of nodes who knows the shape.
  ret.attrs[unknown_name] = std::make_shared<any>(num_unknown);
  return ret;
}

/*!\brief
 * This is a version of the InferAttr function specifically for shape inference.
 *
 * \param ret graph used for attribute inference
 * \param emmpty_val empty value of the attribute
 * \param infer_name name of the function used for attribute inference
 * \param input_name name of the attribute in the graph used to store the
 *                   input data for attribute inference
 * \param attr_key_name name of the attribute used for inference for variable nodes
 * \param attr_name name of the inferred attribute
 * \param unknown_name name of the attribute storing number of entries
 *                     impossible to infer
 * \param fis_none function returning true for not fully inferred values
 * \param fnum_unknown function returning how many elements are unknown in
 *                     partially inferred value of the attribute
 * \param fdefault default function used for inference if the node does not
 *                 provide its own implementation.
 * \param bwd_identity_assign whether the attributes of forward NDArray and backward
 *                            NDArray have to be the same. False only for storage
 *                            type inference
 * \param dispatch_mode_name name of the dispatch mode attribute on the node. Used for
 *                           storage type inference
 * \param default_mode_val default value of the dispatch mode attribute on the node. Used
 *                         for storage type inference
 */
template<typename IsNone, typename FDefault, typename FNumUnknown>
nnvm::Graph InferShapeAttr(nnvm::Graph &&ret,
                           const mxnet::TShape empty_val,
                           const char* infer_name,
                           const char* input_name,
                           const char* attr_key_name,
                           const char* attr_name,
                           const char* unknown_name,
                           IsNone fis_none,
                           FNumUnknown fnum_unknown,
                           FDefault fdefault,
                           bool bwd_identity_assign,
                           const char* dispatch_mode_name,
                           const DispatchMode default_mode_val = DispatchMode::kUndefined) {
  using nnvm::IndexedGraph;
  using nnvm::Op;
  using AttrType = mxnet::TShape;
  using FInferType = mxnet::FInferShape;
  using AttrVector = std::vector<AttrType>;
  using NodeAttrVector = std::vector<DispatchMode>;
  using dmlc::any;
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_shape =
      Op::GetAttr<FInferType>(infer_name);
  static auto& is_backward =
      Op::GetAttr<nnvm::TIsBackward>("TIsBackward");
  // gradient function, used to get node correspondence.
  static auto& fgrad =
      Op::GetAttr<nnvm::FGradient>("FGradient");
  // reshape shape vector
  AttrVector rshape;
  // dispatch mode vector
  DispatchModeVector dispatch_modes;
  if (ret.attrs.count(attr_name) != 0) {
    rshape = ret.MoveCopyAttr<AttrVector>(attr_name);
  } else {
    rshape.resize(idx.num_node_entries(), empty_val);
  }

  if (ret.attrs.count(input_name) != 0) {
    const AttrVector& shape_args = ret.GetAttr<AttrVector>(input_name);
    CHECK_LE(shape_args.size(), idx.input_nodes().size())
        << "More provided " << attr_name << "s than number of arguments.";
    for (size_t i = 0; i < shape_args.size(); ++i) {
      rshape[idx.entry_id(idx.input_nodes()[i], 0)] = shape_args[i];
    }
  }

  // get the shape hints
  std::string shape_hints_key = std::string(attr_name) + "_hints";
  if (ret.attrs.count(shape_hints_key)) {
    nnvm::NodeEntryMap<AttrType> shape_hints =
      ret.GetAttr<nnvm::NodeEntryMap<AttrType>>(shape_hints_key);
    for (const auto& kv : shape_hints) {
      nnvm::NodeEntry e = kv.first;
      if (idx.exist(e.node.get())) {
        rshape[idx.entry_id(kv.first)] = kv.second;
      }
    }
  }

  std::string shape_attr_key;
  if (ret.attrs.count(attr_key_name) != 0) {
    shape_attr_key = ret.GetAttr<std::string>(attr_key_name);
    // erase the provided arguments
    ret.attrs.erase(attr_key_name);
  }

  // limit inference to part of the graph
  uint32_t node_start = 0, node_end = idx.num_nodes();
  if (ret.attrs.count("node_range")) {
    const auto& range = ret.GetAttr<std::pair<uint32_t, uint32_t> >("node_range");
    node_start = range.first;
    node_end = range.second;
    CHECK_GE(node_start, 0);
    CHECK_LE(node_end, idx.num_nodes());
    ret.attrs.erase("node_range");
  }
  uint32_t entry_start = 0, entry_end = idx.num_node_entries();
  if (ret.attrs.count("entry_range")) {
    const auto& range = ret.GetAttr<std::pair<uint32_t, uint32_t> >("entry_range");
    entry_start = range.first;
    entry_end = range.second;
    CHECK_GE(entry_start, 0);
    CHECK_LE(entry_end, idx.num_node_entries());
    ret.attrs.erase("entry_range");
  }
  // populate the node attribute vector
  if (dispatch_mode_name != nullptr) {
    if (ret.attrs.count(dispatch_mode_name) != 0) {
      dispatch_modes = ret.MoveCopyAttr<NodeAttrVector>(dispatch_mode_name);
    } else {
      LOG(FATAL) << "Node attribute " << dispatch_mode_name << " does not exist in the graph";
    }
  }

  // Temp space for shape inference.
  std::vector<AttrType> ishape, oshape;
  // whether a shape is dynamic
  std::vector<int> is_dynamic(rshape.size(), 0);

  // convert to numpy compatible shape to use operator's infer shape function
  if (!Imperative::Get()->is_np_shape()) {
    common::ConvertToNumpyShape(&rshape);
  }

  // inference step function for nid
  auto infer_step = [&](uint32_t nid, bool last_iter) {
    const auto& inode = idx[nid];
    const std::string name = inode.source->attrs.name;
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      CHECK(inode.source->op() == nullptr);
      CHECK_EQ(num_outputs, 1U);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
      if (shape_attr_key.length() != 0 && fis_none(rshape[out_ent_id])) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          std::istringstream is(it->second);
          CHECK(is >> rshape[out_ent_id]) << "Invalid attribute";
          if (!Imperative::Get()->is_np_shape()) {
            common::ConvertToNumpyShape(&rshape[out_ent_id]);
          }
        }
      }
      // assign a default value to node attribute
      if (dispatch_mode_name != nullptr) {
        op::dispatch_mode_assign(&dispatch_modes[nid], default_mode_val);
      }
    } else if (is_backward.get(inode.source->op(), false) &&
               inode.control_deps.size() && bwd_identity_assign) {
      CHECK(dispatch_mode_name == nullptr)
        << "Backward inference for node attributes is not available";
      CHECK_GE(inode.control_deps.size(), 1U)
        << "BackwardOp need to have control_deps to its forward op";
      const IndexedGraph::Node& fnode = idx[inode.control_deps[0]];
      nnvm::NodePtr fwd_ptr = inode.source->control_deps[0];
      CHECK(fwd_ptr->op() != nullptr) << "Forward op cannot be a variable";
      // use gradient function to find out the correspondence.
      std::vector<nnvm::NodeEntry> ograd(fwd_ptr->num_outputs());
      for (size_t i = 0; i < ograd.size(); ++i) {
        ograd[i].index = static_cast<uint32_t>(i);
      }
      // input gradient list
      auto igrad = fgrad[fwd_ptr->op()](fwd_ptr, ograd);
      const nnvm::Node* igrad_node = nullptr;
      // Input gradient assignement
      for (size_t i = 0; i < igrad.size(); ++i) {
        if (igrad[i].node->op() == inode.source->op()) {
          uint32_t eid = idx.entry_id(nid, igrad[i].index);
          if (fis_none(rshape[eid])) {
            rshape[eid] = rshape[idx.entry_id(fnode.inputs[i])];
          } else if (!fis_none(rshape[idx.entry_id(fnode.inputs[i])])) {
            // Need to skip empty forward shape, because it may not be
            // available now and it is possible to infer the forward
            // shape in one of the next a few passes
            CHECK_EQ(rshape[eid], rshape[idx.entry_id(fnode.inputs[i])])
                << "Backward shape inconsistent with the forward shape";
          }
          if (igrad_node == nullptr) {
            igrad_node = igrad[i].node.get();
          } else {
            CHECK(igrad_node == igrad[i].node.get());
          }
        }
      }
      // out grad entries
      CHECK(igrad_node != nullptr)
        << "Cannot find matching backward op for " << inode.source->attrs.name;
      for (size_t i = 0; i < igrad_node->inputs.size(); ++i) {
        const nnvm::NodeEntry& e = igrad_node->inputs[i];
        if (e.node == nullptr) {
          uint32_t eid = idx.entry_id(inode.inputs[i]);
          if (fis_none(rshape[eid])) {
            rshape[eid] = rshape[idx.entry_id(inode.control_deps[0], e.index)];
          }
        }
      }
    } else {
      DispatchMode* dispatch_mode = nullptr;
      bool forward_known = true;
      // Forward operator inference.
      ishape.resize(num_inputs, empty_val);
      bool is_input_dynamic_shape = false;
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = rshape[idx.entry_id(inode.inputs[i])];
        if (!mxnet::ndim_is_known(ishape[i]) && is_dynamic[idx.entry_id(inode.inputs[i])]) {
          is_input_dynamic_shape = true;
        }
        if (fis_none(ishape[i])) forward_known = false;
      }
      oshape.resize(num_outputs, empty_val);
      for (uint32_t i = 0; i < oshape.size(); ++i) {
        oshape[i] = rshape[idx.entry_id(nid, i)];
        if (fis_none(oshape[i])) forward_known = false;
      }
      if (dispatch_mode_name != nullptr) {
        dispatch_mode = &dispatch_modes[nid];
        if (dispatch_modes[nid] == DispatchMode::kUndefined) forward_known = false;
      }
      auto finfer = finfer_shape.get(inode.source->op(), fdefault);
      if (finfer == nullptr || is_input_dynamic_shape) {
        for (uint32_t i = 0; i < oshape.size(); ++i) {
          if (!mxnet::ndim_is_known(oshape[i].ndim())) {
            is_dynamic[idx.entry_id(nid, i)] = 1;
          }
        }
      } else if (!forward_known) {
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            forward_known = ApplyOpInferAttr(ret, finfer, inode.source->attrs,
                                             nid, &ishape, &oshape, dispatch_mode);
          } catch (const std::exception& e) {
            throw dmlc::Error("Error in operator " + inode.source->attrs.name + ": " + e.what());
          }
        } else {
          CHECK(!last_iter)
              << "Attribute " << infer_name
              << " is not registed by op " << inode.source->op()->name
              << " we are not able to complete the inference because of this";
        }
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        rshape[idx.entry_id(inode.inputs[i])] = ishape[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        rshape[idx.entry_id(nid, i)] = oshape[i];
      }
    }
  };

  size_t last_num_unknown;
  size_t num_unknown = static_cast<size_t>(-1);  // Infinity

  int i = 0;
  do {
    if (i % 2 == 0) {
      // forward inference
      for (uint32_t nid = node_start; nid < node_end; ++nid) {
        infer_step(nid, false);
      }
    } else {
      // backward inference
      for (uint32_t i = node_end; i != node_start; --i) {
        infer_step(i - 1, false);
      }
    }
    last_num_unknown = num_unknown;
    num_unknown = 0;
    for (size_t j = entry_start; j < entry_end; ++j) {
      if (fis_none(rshape[j])) {
        num_unknown += fnum_unknown(rshape[j]);
      }
    }
    if (dispatch_mode_name) {
      for (size_t i = node_start; i < node_end; i++) {
        if (dispatch_modes[i] == DispatchMode::kUndefined) {
          ++num_unknown;
        }
      }
    }
    ++i;
  } while (num_unknown > 0 && last_num_unknown > num_unknown);
  // set the shapes
  ret.attrs[attr_name] = std::make_shared<any>(std::move(rshape));
  // set the shapes
  if (dispatch_mode_name) {
    ret.attrs[dispatch_mode_name] = std::make_shared<any>(std::move(dispatch_modes));
  }
  // number of nodes who knows the shape.
  ret.attrs[unknown_name] = std::make_shared<any>(num_unknown);
  return ret;
}

nnvm::Graph InferShape(nnvm::Graph&& graph,
                       mxnet::ShapeVector&& shape_inputs,
                       const std::string& shape_attr_key) {
  using dmlc::any;
  if (shape_inputs.size() != 0) {
    graph.attrs["shape_inputs"] = std::make_shared<any>(std::move(shape_inputs));
  }
  if (shape_attr_key.length() != 0) {
    graph.attrs["shape_attr_key"] = std::make_shared<any>(shape_attr_key);
  }
  std::cout << "Graph attributes before infershape" << std::endl;
  for (const auto& kv : graph.attrs) {
    std::cout << kv.first << std::endl;
  }
  return InferShapeAttr(
      std::move(graph), mxnet::TShape(),
      "FInferShape", "shape_inputs", "shape_attr_key",
      "shape", "shape_num_unknown_nodes",
      [](const mxnet::TShape& s) { return !mxnet::shape_is_known(s); },
      [](const mxnet::TShape& s) {
        if (!mxnet::ndim_is_known(s)) {
          return static_cast<size_t>(1);
        }
        size_t ret = 0;
        for (const auto& val : s) {
          if (!mxnet::dim_size_is_known(val)) {
            ++ret;
          }
        }
        return ret;
      },
      nullptr, true, nullptr);
}

nnvm::Graph InferType(nnvm::Graph&& graph,
                      nnvm::DTypeVector&& dtype_inputs,
                      const std::string& dtype_attr_key) {
  using dmlc::any;
  if (dtype_inputs.size() != 0) {
    graph.attrs["dtype_inputs"] = std::make_shared<any>(std::move(dtype_inputs));
  }
  if (dtype_attr_key.length() != 0) {
    graph.attrs["dtype_attr_key"] = std::make_shared<any>(dtype_attr_key);
  }
  std::cout << "Graph attributes before infertype" << std::endl;
  for (const auto& kv : graph.attrs) {
    std::cout << kv.first << std::endl;
  }
  return InferAttr<int, nnvm::FInferType>(
      std::move(graph), -1,
      "FInferType", "dtype_inputs", "dtype_attr_key",
      "dtype", "dtype_num_unknown_nodes",
      [](const int t) { return t == -1; },
      common::SameType, true, nullptr);
}

nnvm::Graph InferStorageType(nnvm::Graph&& graph,
                             StorageTypeVector&& storage_type_inputs,
                             const std::string& storage_type_attr_key) {
  using dmlc::any;
  if (storage_type_inputs.size() != 0) {
    graph.attrs["storage_type_inputs"] = std::make_shared<any>(std::move(storage_type_inputs));
  }
  if (storage_type_attr_key.length() != 0) {
    graph.attrs["storage_type_attr_key"] = std::make_shared<any>(storage_type_attr_key);
  }
  // initialize unknown values for dispatch modes
  if (graph.attrs.count("dispatch_mode") == 0) {
    DispatchModeVector dispatch_modes(graph.indexed_graph().num_nodes(), DispatchMode::kUndefined);
    graph.attrs["dispatch_mode"] = std::make_shared<any>(std::move(dispatch_modes));
  }
  // initialize the dev_mask vector from the context vector
  if (graph.attrs.count("dev_mask") == 0) {
    CHECK_GT(graph.attrs.count("context"), 0);
    DevMaskVector dev_masks(graph.indexed_graph().num_nodes());
    const ContextVector& vctx = graph.GetAttr<ContextVector>("context");
    for (size_t i = 0; i < vctx.size(); i++) dev_masks[i] = vctx[i].dev_mask();
    graph.attrs["dev_mask"] = std::make_shared<any>(std::move(dev_masks));
  }

  // for storage type, the backward attr is not necessarily the same as it's correspondence
  nnvm::Graph ret = InferAttr<int, FInferStorageType>(
      std::move(graph), -1,
      "FInferStorageType", "storage_type_inputs", "storage_type_attr_key",
      "storage_type", "storage_type_num_unknown_nodes",
      [](const int t) { return t == -1; },
      common::DefaultStorageType, false, "dispatch_mode", DispatchMode::kVariable);

  // log the storage types and dispatch modes of the graph
  static bool log_verbose = dmlc::GetEnv("MXNET_INFER_STORAGE_TYPE_VERBOSE_LOGGING", false);
  if (log_verbose) {
    common::LogInferStorage(ret);
  }
  return ret;
}

}  // namespace exec
}  // namespace mxnet
