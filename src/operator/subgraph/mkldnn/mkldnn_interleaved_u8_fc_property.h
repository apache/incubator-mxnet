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


#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_INTERLEAVED_U8_FC_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_INTERLEAVED_U8_FC_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../common.h"
#include "mkldnn_subgraph_base-inl.h"

namespace mxnet {
namespace op {

class SgMKLDNNInterleavedu8FCSelector : public SubgraphSelector {
 public:
  explicit SgMKLDNNInterleavedu8FCSelector() {}

  bool Select(const nnvm::Node &n, const std::shared_ptr<NodeAttr>& node_attr) override {
    if (n.op() == Op::Get("_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt")) {
      found_ = false;
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if ((n.op() && n.op() == Op::Get("_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt")) &&
        (new_node.op() && new_node.op() == Op::Get("_sg_mkldnn_fully_connected"))) {
      found_ = true;
      return true;
    }
    return false;
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (!found_) {
      return std::vector<nnvm::Node *>(0);
    } else {
      return candidates;
    }
  }

 private:
  bool found_ = false;
};

class SgMKLDNNInterleavedu8FCProperty : public SubgraphProperty {
 public:
  SgMKLDNNInterleavedu8FCProperty() : SubgraphProperty(kAdjust){}

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN Interleavedu8FC optimization pass";
    auto property = std::make_shared<SgMKLDNNInterleavedu8FCProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_INTERLEAVED_U8_FC_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  void AdjustSubgraphNode(const std::vector<nnvm::Node*>& subgraph_nodes,
                          const SubgraphSelectorV2Ptr& subgraph_selector,
                          const int subgraph_id = 0) const {
    for (auto node : subgraph_nodes) {
      if (node->op() && node->op() == Op::Get("_sg_mkldnn_contrib_interleaved_matmul_selfatt_valatt")) {
        node->attrs.dict["shifted_output"] = "True";
        node->op()->attr_parser(&(node->attrs));
      } else if (node->op() && node->op() == Op::Get("_sg_mkldnn_fully_connected")) {
        node->attrs.dict["shift_value"] = "128";
        node->op()->attr_parser(&(node->attrs));
      }
    }
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNInterleavedu8FCSelector>();
    return selector;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_INTERLEAVED_U8_FC_PROPERTY_H_