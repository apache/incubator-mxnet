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

/*
 * This file contains an implementation of a subgraph property
 * that interfaces between MXNet and custom subgraph properties
 * created by users in external libraries. It does not implement
 * any custom subgraphing logic itself, rather it calls APIs
 * in the user's custom library to enable control of partitioning
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_

#include <nnvm/pass_functions.h>
#include <nnvm/symbolic.h>
#include <string>
#include <utility>
#include <vector>
#include "../common.h"
#include "../subgraph_property.h"
#include "../../include/mxnet/lib_api.h"
namespace mxnet {
namespace op {

/*
 * This selects nodes for a subgraph based on node name as supplied
 * by the supportedOps from an external library. It visits nodes via
 * both input and output links.
 */
class CustomContainOpSelector: public SubgraphSelector {
 public:
  explicit CustomContainOpSelector(std::unordered_set<std::string> supported_nodes) :
    supported_nodes_(supported_nodes) {}
  virtual bool Select(const nnvm::Node &n) {
    return supported_nodes_.count(n.attrs.name) > 0;
  }
  virtual bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) {
    return supported_nodes_.count(new_node.attrs.name) > 0;
  }
  virtual bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) {
    return supported_nodes_.count(new_node.attrs.name) > 0;
  }
  std::unordered_set<std::string> supported_nodes_;
};

/*
 * This subgraph property finds a subgraph that only contains
 * nodes as specified by the supportedOps from an external library.
 * The operators in the subgraph will be executed by the operator
 * specified by the external library too.
 */
class  CustomSubgraphProperty: public SubgraphProperty {
 public:
  CustomSubgraphProperty() :
    subgraph_prop("error"),
    call_supported_ops_(nullptr),
    supported_ops_(nullptr),
    call_review_subgraph_(nullptr),
    review_subgraph_(nullptr),
    subgraph_op_name("error") {}
  CustomSubgraphProperty(std::string subgraph_prop_name,
                         partCallSupportedOps_t call_supported_ops,
                         supportedOps_t supported_ops,
                         partCallReviewSubgraph_t call_review_subgraph,
                         reviewSubgraph_t review_subgraph,
                         opCallFree_t call_free,
                         std::string op_name) :
      subgraph_prop(subgraph_prop_name),
      call_supported_ops_(call_supported_ops),
      supported_ops_(supported_ops),
      call_review_subgraph_(call_review_subgraph),
      review_subgraph_(review_subgraph),
      call_free_(call_free),
      subgraph_op_name(op_name) {}

  // create custom subgraph property
  static SubgraphPropertyPtr Create() {
    return std::make_shared<CustomSubgraphProperty>();
  }

  void PrePartition(const nnvm::Graph& g,
    const std::vector<std::pair<std::string, std::string>>& options_map) {
    // clear supported_nodes to remove state from previous calls
    supported_nodes.clear();
    // get input args and arg names
    in_arg_names = g.GetAttr<std::vector<std::string>>("in_arg_names");
    in_args_ptr = g.GetAttr<NDArray**>("in_args");
    in_aux_names = g.GetAttr<std::vector<std::string>>("in_aux_names");
    in_aux_ptr = g.GetAttr<NDArray**>("in_aux");

    // convert input args
    arg_names.clear();
    arg_data.clear();
    arg_shapes.clear();
    arg_dims.clear();
    arg_types.clear();
    arg_verIDs.clear();
    arg_dev_type.clear();
    arg_dev_id.clear();
    for (size_t i=0; i < in_arg_names.size(); i++) {
      arg_names.push_back(in_arg_names[i].c_str());
      const auto &in_arg = *(in_args_ptr[i]);
      arg_data.push_back(in_arg.data().dptr_);
      arg_shapes.push_back(in_arg.shape().data());
      arg_dims.push_back(in_arg.shape().ndim());
      arg_types.push_back(in_arg.dtype());
      arg_verIDs.push_back(in_arg.version());
      const char* arg_ctx_str = in_arg.ctx().dev_mask() == Context::kCPU ? "cpu" : "gpu";
      arg_dev_type.push_back(arg_ctx_str);
      arg_dev_id.push_back(in_arg.ctx().real_dev_id());
    }

    // convert input aux
    aux_names.clear();
    aux_data.clear();
    aux_shapes.clear();
    aux_dims.clear();
    aux_types.clear();
    aux_verIDs.clear();
    aux_dev_type.clear();
    aux_dev_id.clear();
    for (size_t i=0; i < in_aux_names.size(); i++) {
      aux_names.push_back(in_aux_names[i].c_str());
      const auto &in_aux = *(in_aux_ptr[i]);
      aux_data.push_back(in_aux.data().dptr_);
      aux_shapes.push_back(in_aux.shape().data());
      aux_dims.push_back(in_aux.shape().ndim());
      aux_types.push_back(in_aux.dtype());
      aux_verIDs.push_back(in_aux.version());
      const char* aux_ctx_str = in_aux.ctx().dev_mask() == Context::kCPU ? "cpu" : "gpu";
      aux_dev_type.push_back(aux_ctx_str);
      aux_dev_id.push_back(in_aux.ctx().real_dev_id());
    }

    // remove all graph attrs, some cannot be saved to json
    nnvm::Graph graph = std::move(g);
    graph.attrs.clear();
    const nnvm::IndexedGraph& indexed_graph = graph.indexed_graph();

    // set shape attrs for each node in the graph
    if (g.HasAttr("shape")) {
      mxnet::ShapeVector shapes = g.GetAttr<mxnet::ShapeVector>("shape");
      for (unsigned i = 0; i < indexed_graph.num_nodes(); i++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[i].source);
        mxnet::TShape shape = shapes[i];
        std::stringstream ss;
        ss << shape;
        node->attrs.dict[MX_SHAPE] = ss.str();
      }
    }
    // set dtype attrs for each node in the graph
    if (g.HasAttr("dtype")) {
      std::vector<int> dtypes = g.GetAttr<std::vector<int> >("dtype");
      for (unsigned i = 0; i < indexed_graph.num_nodes(); i++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(indexed_graph[i].source);
        int dtype = dtypes[i];
        std::stringstream ss;
        ss << dtype;
        node->attrs.dict[MX_DTYPE] = ss.str();
      }
    }

    CHECK(supported_ops_ != nullptr)
      << "supported_ops_ is null for " << subgraph_prop << std::endl;
    CHECK(call_supported_ops_ != nullptr)
      << "call_supported_ops_ is null for " << subgraph_prop << std::endl;

    std::string subgraph_json = nnvm::pass::SaveJSON(graph);
    std::vector<int> supported_node_IDs(indexed_graph.num_nodes(), 0);
    const char* json = subgraph_json.c_str();
    int *ids = supported_node_IDs.data();

    // clear options from previous call
    opt_keys_.clear();
    opt_vals_.clear();
    options_map_.clear();
    for (auto kv : options_map) {
      options_map_.push_back(kv);
      opt_keys_.push_back(options_map_.back().first.c_str());
      opt_vals_.push_back(options_map_.back().second.c_str());
    }

    CHECK(call_supported_ops_(supported_ops_, json, supported_node_IDs.size(), ids,
                            opt_keys_.data(), opt_vals_.data(), opt_keys_.size()))
      << "Error calling supported_ops for '" << subgraph_prop << "'";

    const auto& idx = g.indexed_graph();
    // loop and add node names for each supported node ID
    for (unsigned i = 0; i < supported_node_IDs.size(); i++) {
      if (supported_node_IDs[i]) {
        supported_nodes.insert(idx[i].source->attrs.name);
      }
    }
  }
  // override CreateSubgraphNode
  virtual nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                             const int subgraph_id = 0) const {
    int accept = 1;
    int num_attr = 0;
    std::map<std::string, std::string> user_attrs;
    char** attr_keys = nullptr;
    char** attr_vals = nullptr;
    if (review_subgraph_) {
      nnvm::Graph g;
      g.outputs = sym.outputs;
      const auto& idx = g.indexed_graph();

      // set isArg/isAux for each null op/param in the graph
      const std::vector<std::string> aux_state_names =
        sym.ListInputNames(nnvm::Symbol::kAuxiliaryStates);
      std::unordered_set<std::string> aux_set(aux_state_names.begin(), aux_state_names.end());
      for (unsigned i = 0; i < idx.num_nodes(); i++) {
        nnvm::Node* node = const_cast<nnvm::Node*>(idx[i].source);
        // check if this node is input to subgraph
        if (node->is_variable()) {
          // check if this node is an aux param
          if (aux_set.count(node->attrs.name))
            node->attrs.dict["isAux"] = "True";
          else
            node->attrs.dict["isAux"] = "False";
        }
      }

      std::string subgraph_json = nnvm::pass::SaveJSON(g);
      CHECK(call_review_subgraph_(review_subgraph_, subgraph_json.c_str(),  subgraph_id,
                                  &accept, opt_keys_.data(), opt_vals_.data(),
                                  opt_keys_.size(),  &attr_keys, &attr_vals, &num_attr,
                                  arg_names.data(), arg_names.size(), arg_data.data(),
                                  arg_shapes.data(), arg_dims.data(), arg_types.data(),
                                  arg_verIDs.data(), arg_dev_type.data(),
                                  arg_dev_id.data(), aux_names.data(), aux_names.size(),
                                  aux_data.data(), aux_shapes.data(), aux_dims.data(),
                                  aux_types.data(), aux_verIDs.data(),
                                  aux_dev_type.data(), aux_dev_id.data()))
        << "Error calling review_subgraph for '" << subgraph_prop << "'";

      if(num_attr > 0) {
        // set user specified attributes
        for (int i=0; i < num_attr; i++) {
          user_attrs[attr_keys[i]] = attr_vals[i];
          call_free_(attr_vals[i]);
          call_free_(attr_keys[i]);
        }
        // free memory used by custom op to allocate attributes
        call_free_(attr_vals);
        call_free_(attr_keys);
      }
    }

    if (accept) {
      nnvm::ObjectPtr n = nnvm::Node::Create();
      n->attrs.op = Op::Get(subgraph_op_name);
      n->attrs.name = "_op" + std::to_string(subgraph_id);
      n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));

      // set user specified attributes
      for(auto attr : user_attrs)
        n->attrs.dict[attr.first] = attr.second;

      return n;
    } else {
      return nullptr;
    }
  }
  // override CreateSubgraphSelector
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<CustomContainOpSelector>(supported_nodes);
  }

  std::string subgraph_prop;
  partCallSupportedOps_t call_supported_ops_;
  supportedOps_t supported_ops_;
  partCallReviewSubgraph_t call_review_subgraph_;
  reviewSubgraph_t review_subgraph_;
  opCallFree_t call_free_;
  std::unordered_set<std::string> supported_nodes;
  std::string subgraph_op_name;
  std::vector<std::pair<std::string, std::string>> options_map_;
  std::vector<const char*> opt_keys_, opt_vals_;
  std::vector<std::string> in_arg_names, in_aux_names;
  NDArray **in_args_ptr;
  NDArray **in_aux_ptr;
  std::vector<const char*> arg_names, aux_names;
  std::vector<void*> arg_data, aux_data;
  std::vector<const int64_t*> arg_shapes, aux_shapes;
  std::vector<int> arg_dims, aux_dims;
  std::vector<int> arg_types, aux_types;
  std::vector<size_t> arg_verIDs, aux_verIDs;
  std::vector<const char*> arg_dev_type, aux_dev_type;
  std::vector<int> arg_dev_id, aux_dev_id;
};
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_SUBGRAPH_PARTITIONER_CUSTOM_SUBGRAPH_PROPERTY_H_
