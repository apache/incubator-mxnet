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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_fc_property.cc
 * \brief Partition gragph property for FullyConnected operator
 * \author Ciyong Chen
*/

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
#if MXNET_USE_ONEDNN == 1

#include <string>
#include <vector>
#include "../common.h"
#include "../../tensor/matrix_op-inl.h"
#include "mkldnn_subgraph_base-inl.h"
#include "mkldnn_fc-inl.h"

namespace mxnet {
namespace op {

class SgMKLDNNFCSelector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

 private:
  bool disable_fc_eltwise_;
  bool quantized_;
  SelectStatus status_;
  std::vector<const nnvm::Node *> matched_list_;

 public:
  explicit SgMKLDNNFCSelector(const bool dis_fc_eltwise, bool quantized) :
      disable_fc_eltwise_(dis_fc_eltwise),
      quantized_(quantized) {}

  bool Select(const nnvm::Node &n, const std::shared_ptr<NodeAttr>& node_attr) override {
    if (n.op() == Op::Get("FullyConnected") && SupportMKLDNNAttr(node_attr)) {
      status_ = disable_fc_eltwise_ ? kSuccess : kStart;
      matched_list_.clear();
      matched_list_.push_back(&n);
      return true;
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (status_ == kFail || status_ == kSuccess || new_node.is_variable())
      return false;

    // If n isn't the last matched node, then we encoutered a internal
    // branch, we should pop out the node behind n and stop fusion.
    if (matched_list_.back() != &n) {
      if (std::find(matched_list_.begin(), matched_list_.end(), &n) !=
        matched_list_.end()) {
        while (matched_list_.back() != &n) {
          matched_list_.pop_back();
        }
      }

      status_ = kSuccess;
      return false;
    }

    switch (status_) {
      case kStart:
        // Currently, For INT8 FC fusion, only supports relu/bounded_relu(clip)/abs.
        if (new_node.op() == Op::Get("Activation")) {
          const ActivationParam &param = nnvm::get<ActivationParam>(new_node.attrs.parsed);
          if ((quantized_ && SupportQuantizedMKLDNNAct(param)) ||
              (!quantized_ && SupportMKLDNNAct(param))) {
            matched_list_.push_back(&new_node);
            status_ = kSuccess;
            return true;
          }
        }
        if (new_node.op() == Op::Get("LeakyReLU")) {
          const LeakyReLUParam &param = nnvm::get<LeakyReLUParam>(new_node.attrs.parsed);
          if (param.act_type == leakyrelu::kLeakyReLU ||
              param.act_type == leakyrelu::kELU ||
              param.act_type == leakyrelu::kGELU) {
            matched_list_.push_back(&new_node);
            status_ = kSuccess;
            return true;
          }
        }
        if (!quantized_ && (new_node.op() == Op::Get("square") ||
            new_node.op() == Op::Get("sqrt") ||
            new_node.op() == Op::Get("exp"))) {
          matched_list_.push_back(&new_node);
          status_ = kSuccess;
          return true;
        }
        if (new_node.op() == Op::Get("abs")) {
          matched_list_.push_back(&new_node);
          status_ = kSuccess;
          return true;
        }
        if (new_node.op() == Op::Get("clip")) {
          const ClipParam &param = nnvm::get<ClipParam>(new_node.attrs.parsed);
          if (param.a_min == 0.f) {
            matched_list_.push_back(&new_node);
            status_ = kSuccess;
            return true;
          }
          status_ = kSuccess;
          return false;
        }
      default:
        status_ = kSuccess;
        return false;
    }
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status_ == kFail) {
      return std::vector<nnvm::Node *>(0);
    } else {
      std::vector<nnvm::Node *> ret;
      for (auto i : matched_list_) {
        auto non_const_i = const_cast<nnvm::Node *>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
            candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return candidates;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgMKLDNNFCSelector(disable_fc_eltwise_, quantized_);
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgMKLDNNFCProperty : public SubgraphProperty {
 public:
  SgMKLDNNFCProperty() {
    disable_fc_eltwise_ = dmlc::GetEnv("MXNET_DISABLE_ONEDNN_FUSE_FC_ELTWISE", false);
  }

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN FullyConnected optimization pass";
    auto property = std::make_shared<SgMKLDNNFCProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_ONEDNN_FC_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(last_node);
    std::ostringstream node_name;
    node_name << "sg_mkldnn_";
    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr &node) {
      if (node->is_variable()) return;
      auto &sub_name = node->op()->name;
      if (sub_name == "FullyConnected") {
        node_name << "fully_connected_";
      } else if (SupportMKLDNNFCEltwiseFusion(sub_name)) {
          node_name << "eltwise_";
          n->attrs.dict["with_eltwise"] = "True";
      }
    });
    node_name << std::to_string(subgraph_id);
    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_sg_mkldnn_fully_connected");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    bool quantized = HasAttr("quantize") ? GetAttr<bool>("quantize") : false;
    auto selector = std::make_shared<SgMKLDNNFCSelector>(disable_fc_eltwise_, quantized);
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::ObjectPtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }

 private:
  bool disable_fc_eltwise_;
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
