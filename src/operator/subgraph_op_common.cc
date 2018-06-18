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

#include "./subgraph_op_common.h"
#include "./operator_common.h"
#include "../imperative/imperative_utils.h"

namespace mxnet {
namespace op {

bool InferSubgraphDataType(const nnvm::Symbol &subgraph,
                           std::vector<int> *in_types,
                           std::vector<int> *out_types) {
  nnvm::Graph g;
  g.outputs = subgraph.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_types->size());
  CHECK_EQ(idx_g.outputs().size(), out_types->size());

  // Put the input and output data types to the dtype vector.
  nnvm::DTypeVector types(idx_g.num_node_entries(), -1);
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_types->size());
  for (size_t i = 0; i < in_types->size(); i++) {
    auto eid = idx_g.entry_id(input_nids[i], 0);
    types[eid] = in_types->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_types->size());
  for (size_t i = 0; i < out_types->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    types[eid] = out_types->at(i);
  }

  // Infer data type of the graph.
  g.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(types));
  g = exec::InferType(std::move(g));

  const auto& types1 = g.GetAttr<nnvm::DTypeVector>("dtype");
  // assign to in_types
  for (size_t i = 0; i < in_types->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    TYPE_ASSIGN_CHECK(*in_types, i, types1[eid]);
  }
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    TYPE_ASSIGN_CHECK(*out_types, i, types1[eid]);
  }
  // Check if we have inferred the dtypes correctly.
  return g.GetAttr<size_t>("dtype_num_unknown_nodes") == 0;
}

bool InferSubgraphStorage(const nnvm::Symbol &subgraph,
                          const int dev_mask,
                          DispatchMode* dispatch_mode,
                          std::vector<int> *in_stypes,
                          std::vector<int> *out_stypes) {
  nnvm::Graph g;
  g.outputs = subgraph.outputs;
  const auto& idx_g = g.indexed_graph();
  CHECK_EQ(idx_g.input_nodes().size(), in_stypes->size());
  CHECK_EQ(idx_g.outputs().size(), out_stypes->size());
  exec::DevMaskVector dev_masks(idx_g.num_node_entries(), dev_mask);

  // Put the input and output storages to the storage vector.
  nnvm::StorageVector stypes(idx_g.num_node_entries(), exec::kBadStorageID);
  const auto &input_nids = idx_g.input_nodes();
  CHECK_EQ(input_nids.size(), in_stypes->size());
  for (size_t i = 0; i < in_stypes->size(); i++) {
    auto eid = idx_g.entry_id(input_nids[i], 0);
    stypes[eid] = in_stypes->at(i);
  }
  CHECK_EQ(g.outputs.size(), out_stypes->size());
  for (size_t i = 0; i < out_stypes->size(); i++) {
    auto eid = idx_g.entry_id(g.outputs[i]);
    stypes[eid] = out_stypes->at(i);
  }

  // Infer storage type of the graph.
  bool dev_match = g.attrs.count("dev_mask") &&
                   g.GetAttr<exec::DevMaskVector>("dev_mask") == dev_masks;
  if (!dev_match) {
    g.attrs["dev_mask"] = std::make_shared<dmlc::any>(std::move(dev_masks));
  }
  g.attrs["storage_type"] = std::make_shared<dmlc::any>(std::move(stypes));
  g = exec::InferStorageType(std::move(g));

  const auto& stypes1 = g.GetAttr<StorageTypeVector>("storage_type");
  // assign to in_types
  for (size_t i = 0; i < in_stypes->size(); ++i) {
    const auto eid = idx_g.entry_id(input_nids[i], 0);
    STORAGE_TYPE_ASSIGN_CHECK(*in_stypes, i, stypes1[eid]);
  }

  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  // assign to out_types
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    const auto eid = idx_g.entry_id(g.outputs[i]);
    STORAGE_TYPE_ASSIGN_CHECK(*out_stypes, i, stypes1[eid]);
  }
  // Check if we have inferred the storages correctly.
  return g.GetAttr<size_t>("storage_type_num_unknown_nodes") == 0;
}

bool InferSubgraphBackwardStorage(const nnvm::Symbol &subgraph,
                                  const int dev_mask,
                                  DispatchMode* dispatch_mode,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  using namespace nnvm;
  // construct backward graph
  nnvm::Graph grad_graph;
  nnvm::Graph fwd_graph;
  std::vector<Node *> potential_nodes;
  {
    fwd_graph.outputs = subgraph.outputs;
    std::vector<nnvm::NodeEntry> ograd_entries;
    ograd_entries.reserve(fwd_graph.outputs.size());
    for (size_t i = 0; i < fwd_graph.outputs.size(); ++i) {
      ograd_entries.emplace_back(NodeEntry{Node::Create(), 0, 0});
    }

    std::vector<NodeEntry> xs;
    std::vector<NodePtr> args = subgraph.ListInputs(nnvm::Symbol::kReadOnlyArgs);
    xs.reserve(args.size());
    for (const auto& i : args)
      xs.emplace_back(NodeEntry{i, 0, 0});
    CHECK_GT(xs.size(), 0)
        << "There are no inputs in computation graph that require gradients.";

    static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
    grad_graph = pass::Gradient(
        fwd_graph, fwd_graph.outputs, xs, ograd_entries,
        exec::AggregateGradient, nullptr, nullptr,
        zero_ops, "_copy");
    potential_nodes.reserve(fwd_graph.outputs.size() + xs.size() + ograd_entries.size());
    for (auto e : ograd_entries)
      potential_nodes.push_back(e.node.get());
    for (auto e : xs)
      potential_nodes.push_back(e.node.get());
    for (auto e : fwd_graph.outputs)
      potential_nodes.push_back(e.node.get());
  }

  const auto& idx = grad_graph.indexed_graph();
  auto input_nodes = idx.input_nodes();
  StorageTypeVector storage_type_inputs(input_nodes.size());
  for (size_t i = 0; i < input_nodes.size(); i++) {
    auto node_id = input_nodes[i];
    const nnvm::IndexedGraph::Node &n = idx[node_id];
    auto it = std::find(potential_nodes.begin(), potential_nodes.end(), n.source);
    CHECK(it != potential_nodes.end());
    size_t idx = it - potential_nodes.begin();
    CHECK_LT(idx, in_attrs->size());
    storage_type_inputs[i] = in_attrs->at(idx);
  }
  CHECK_EQ(idx.outputs().size(), out_attrs->size());
  exec::DevMaskVector dev_masks(idx.num_nodes(), dev_mask);
  imperative::CheckAndInferStorageType(&grad_graph, std::move(dev_masks),
                                       std::move(storage_type_inputs), true);

  const auto& stypes = grad_graph.GetAttr<StorageTypeVector>("storage_type");
  DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
  auto &outputs = idx.outputs();
  CHECK(outputs.size() == out_attrs->size());
  for (size_t i = 0; i < out_attrs->size(); i++)
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, stypes[idx.entry_id(outputs[i])]);
  return true;
}

void LoopState::Forward(int iter_no,
                        std::vector<NDArray> cinputs,
                        const std::vector<OpReqType>& req,
                        std::vector<NDArray> coutputs,
                        bool is_recording) {
  using namespace nnvm;
  using namespace imperative;

  bool orig_is_record;
  if (is_recording)
    orig_is_record = Imperative::Get()->set_is_recording(true);
  else
    orig_is_record = Imperative::Get()->is_recording();

  std::vector<NDArray *> inputs(cinputs.size());
  std::vector<NDArray *> outputs(coutputs.size());
  for (size_t i = 0; i < inputs.size(); i++)
    inputs[i] = &cinputs[i];
  for (size_t i = 0; i < outputs.size(); i++)
    outputs[i] = &coutputs[i];

  std::vector<std::pair<std::string, std::string> > kwargs;
  kwargs.push_back(std::pair<std::string, std::string>("inline_limit", "0"));
  // We turn on static_alloc for two reasons.
  // It avoids the overhead of unnecessary memory allocation.
  // only static_alloc supports nested call of CachedOp.
  kwargs.push_back(std::pair<std::string, std::string>("static_alloc", "1"));
  CachedOpPtr op;
  if (is_recording && iter_ops.size() > (size_t) iter_no)
    op = iter_ops[iter_no];
  else if (!is_recording && iter_ops.size() == 1)
    op = iter_ops[0];

  // If we need to run backward and we don't have a cached op for this iteration,
  // we create one for this iteration.
  if (is_recording && op == nullptr) {
    op = std::make_shared<CachedOp>(subgraph_sym, kwargs);
    CHECK_EQ(iter_ops.size(), iter_no);
    iter_ops.push_back(op);
  } else if (op == nullptr) {
    // If we don't need to run backward and this is the first time of
    // running the iteration, we need to create a new cached op.
    op = std::make_shared<CachedOp>(subgraph_sym, kwargs);
    CHECK(iter_ops.empty());
    iter_ops.push_back(op);
  }
  OpStatePtr state = op->Forward(nullptr, inputs, outputs);

  if (is_recording) {
    all_inputs.push_back(cinputs);
    all_outputs.push_back(coutputs);
    all_states.push_back(state);
  }

  Imperative::Get()->set_is_recording(orig_is_record);
}

void LoopState::Backward(int iter_no,
                         std::vector<NDArray> ograds,
                         const std::vector<OpReqType> &req,
                         std::vector<NDArray> igrads) {
  using namespace nnvm;
  using namespace imperative;

  CHECK_GT(iter_ops.size(), iter_no)
      << "We didn't record the computation for iteration " << iter_no;
  auto op = iter_ops[iter_no];
  std::vector<NDArray *> inputs;
  std::vector<NDArray *> outputs;
  inputs.reserve(op->num_backward_inputs());
  outputs.reserve(op->num_inputs());
  for (size_t i = 0; i < ograds.size(); i++)
    inputs.push_back(&ograds[i]);

  const std::vector<bool> &save_inputs = op->save_inputs();
  const std::vector<bool> &save_outputs = op->save_outputs();
  CHECK_EQ(save_inputs.size(), all_inputs[iter_no].size());
  CHECK_EQ(op->num_outputs(), all_outputs[iter_no].size());
  for (size_t i = 0; i < all_inputs[iter_no].size(); i++) {
    if (save_inputs[i])
      inputs.push_back(&all_inputs[iter_no][i]);
  }
  for (size_t i = 0; i < all_outputs[iter_no].size(); i++) {
    if (save_outputs[i])
      inputs.push_back(&all_outputs[iter_no][i]);
  }
  CHECK_EQ(inputs.size(), op->num_backward_inputs());
  for (size_t i = 0; i < igrads.size(); i++)
    outputs.push_back(&igrads[i]);
  CHECK_EQ(outputs.size(), op->num_inputs());
  auto state = all_states[iter_no];
  op->Backward(false, state, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet
