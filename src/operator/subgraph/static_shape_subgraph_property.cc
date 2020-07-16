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

#include "./common.h"
#include "./subgraph_property.h"
#include "../../imperative/cached_op.h"

namespace mxnet {
namespace op {

/*
 * This selects nodes for a subgraph that only contains static shape operators
 * and it visits nodes via both input and output links.
 */
class StaticShapeOpSelector: public SubgraphSelector {
 public:
  virtual bool Select(const nnvm::Node &seed_node) {
    const auto& infershape = nnvm::Op::GetAttr<mxnet::FInferShape>("FInferShape");
    return !seed_node.is_variable() && infershape.count(seed_node.op());
  }

  virtual bool SelectInput(const nnvm::Node &cur_node, const nnvm::Node &input_node) {
    const auto& infershape = nnvm::Op::GetAttr<mxnet::FInferShape>("FInferShape");
    return !input_node.is_variable() && infershape.count(input_node.op());
  }

  virtual bool SelectOutput(const nnvm::Node &cur_node, const nnvm::Node &output_node) {
    const auto& infershape = nnvm::Op::GetAttr<mxnet::FInferShape>("FInferShape");
    return !output_node.is_variable() && infershape.count(output_node.op());
  }
};

/*
 * This subgraph property finds a subgraph whose nodes have only static shape operators.
 * The operators in the subgraph will be executed by _CachedOp.
 */
class StaticShapeSubgraphProperty: public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() { return std::make_shared<StaticShapeSubgraphProperty>(); }

  // the criteria of selecting the subgraph nodes
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<StaticShapeOpSelector>();
  }

  // create an nnvm node for a given subgraph
  virtual nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                             const int subgraph_id = 0) const {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_CachedOp");
    n->attrs.name = "_CachedOp" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
    std::vector<std::pair<std::string, std::string> > flags{};
    n->attrs.parsed = CachedOpPtr(new CachedOp(sym, flags));
    return n;
  }
};

MXNET_REGISTER_SUBGRAPH_BACKEND(static_shape);
MXNET_REGISTER_SUBGRAPH_PROPERTY(static_shape, StaticShapeSubgraphProperty);

}  // namespace op
}  // namespace mxnet
