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

#ifndef MXNET_IMPERATIVE_RUNTIME_H_
#define MXNET_IMPERATIVE_RUNTIME_H_

#include <mxnet/op_attr_types.h>
#include <mxnet/c_api.h>
#include <nnvm/symbolic.h>
#include <nnvm/op.h>
#include <nnvm/graph.h>
#include <vector>
#include <atomic>
#include <unordered_map>

#include "./ndarray.h"

namespace mxnet {
/*! \brief runtime functions for NDArray */
class ImperativeRuntime {
 public:
  class CachedOp {
   public:
    CachedOp(const nnvm::Symbol& sym);
    uint32_t num_inputs() {
      return fwd_graph_.indexed_graph().input_nodes().size();
    }
    uint32_t num_outputs() {
      return fwd_graph_.outputs.size();
    }
    void Forward(const std::vector<NDArray*>& inputs,
                 const std::vector<NDArray*>& outputs);
    void Backward(const OpStatePtr& state,
                  const std::vector<NDArray*>& inputs,
                  const std::vector<NDArray*>& outputs);
    nnvm::Graph GetForwardGraph(const std::vector<NDArray*>& inputs);
    nnvm::Graph GetBackwardGraph(const OpStatePtr& state,
                                 const std::vector<OpReqType>& reqs,
                                 const std::vector<NDArray*>& inputs);
    std::vector<nnvm::NodeEntry> Gradient(const nnvm::NodePtr& node,
                                          const std::vector<nnvm::NodeEntry>& ograds);
   private:
    struct CachedOpState {
      std::vector<NDArray> buff;
      std::vector<OpStatePtr> states;
    };
    std::mutex mutex_;
    nnvm::Graph fwd_graph_;
    nnvm::Graph grad_graph_;
    nnvm::Graph full_graph_;
    std::vector<bool> curr_grad_req_;
    std::vector<uint32_t> bwd_in_dep_, bwd_out_dep_, bwd_ograd_dep_;
    std::vector<uint32_t> bwd_input_eid_;
  };
  /*! \brief whether operator recording is on. */
  bool is_training() const {
    return is_train_;
  }
  /*! \brief turn on or turn off operator recording for autograd. */
  bool set_is_training(bool is_train) {
      bool old = is_train_;
      is_train_ = is_train;
      return old;
  }
  /*! \brief whether operator recording is on. */
  bool is_recording() const {
    return is_recording_;
  }
  /*! \brief turn on or turn off operator recording for autograd. */
  bool set_is_recording(bool is_recording) {
      bool old = is_recording_;
      is_recording_ = is_recording;
      return old;
  }
  /*! \brief to record operator, return corresponding node. */
  void RecordOp(nnvm::NodeAttrs&& attrs,
                const std::vector<NDArray*>& inputs,
                const std::vector<NDArray*>& outputs,
                const OpStatePtr& state = OpStatePtr(),
                std::vector<bool>* p_save_inputs = nullptr,
                std::vector<bool>* p_save_outputs = nullptr);
  /*! \brief */
  OpStatePtr Invoke(const Context& default_ctx,
                    const nnvm::NodeAttrs& attrs,
                    const std::vector<NDArray*>& inputs,
                    const std::vector<NDArray*>& outputs);
  /*! \brief */
  OpStatePtr InvokeOp(const Context& ctx,
                      const nnvm::NodeAttrs& attrs,
                      const std::vector<NDArray*>& inputs,
                      const std::vector<NDArray*>& outputs,
                      const std::vector<OpReqType>& req,
                      OpStatePtr state = OpStatePtr());
  /*! \brief mark variables for computing gradients. */
  void MarkVariables(const std::vector<NDArray*>& variables,
                     const std::vector<mx_uint>& grad_reqs,
                     const std::vector<NDArray*>& gradients);
  /*! \brief compute the gradient of outputs w.r.t variables. */
  std::vector<NDArray*> Backward(const std::vector<NDArray*>& outputs,
                                 const std::vector<NDArray*>& ograds,
                                 const std::vector<NDArray*>& variables,
                                 bool is_train, bool retain_graph,
                                 bool create_graph);
  /*! \return AutogradRuntime singleton */
  static ImperativeRuntime* Get();

 public:
  /*! \brief */
  class AGInfo {
   public:
    OpReqType grad_req;
    OpStatePtr state;
    std::vector<NDArray> outputs;
    std::vector<NDArray> out_grads;
    bool fresh_out_grad;

    AGInfo() :
      grad_req(kNullOp), fresh_out_grad(false) {}

    static void Clear(const nnvm::NodePtr& node) {
      if (node == nullptr || node->info.empty()) return;
      AGInfo& info = Get(node);
      if (info.grad_req != kNullOp) return;
      node->info.clear();
    }

    static AGInfo& Get(const nnvm::NodePtr& node) {
      return dmlc::get<AGInfo>(node->info);
    }

    static AGInfo& Create(const nnvm::NodePtr& node) {
      node->info.construct<AGInfo>();
      return Get(node);
    }

    static bool IsNone(const NDArray& arr) {
      return arr.entry_.node == nullptr || arr.entry_.node->info.empty();
    }

    static bool IsVariable(const nnvm::NodePtr& node) {
      AGInfo& info = Get(node);
      return info.grad_req != kNullOp && info.outputs.size() == 1
             && info.out_grads.size() == 1;
    }
  };
  /*! \brief make constructor protected. */
  ImperativeRuntime() {}
  /*! \brief find the input/output ndarrays that are needed for backward */
  void GetBackwardDependency(
      const nnvm::NodePtr& node,
      uint32_t num_inputs, uint32_t num_outputs,
      std::vector<bool> *p_save_inputs,
      std::vector<bool> *p_save_outputs);
  void RunGraph(
      const Context& default_ctx,
      const nnvm::IndexedGraph& idx,
      const std::vector<NDArray*> arrays,
      size_t node_start, size_t node_end,
      std::vector<OpReqType>&& array_reqs,
      std::vector<int>&& ref_count,
      std::vector<OpStatePtr> *p_states);
  /*! \brief indicate whether is training. */
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local bool is_train_;
  static thread_local bool is_recording_;
#else
  static MX_THREAD_LOCAL bool is_train_;
  static MX_THREAD_LOCAL bool is_recording_;
#endif
  /*! \brief node count used for naming */
  std::atomic<uint64_t> node_count_{0};
  /*! \brief variable count used for naming */
  std::atomic<uint64_t> variable_count_{0};
};

using CachedOpPtr = std::shared_ptr<ImperativeRuntime::CachedOp>;

}  // namespace mxnet
#endif  // MXNET_IMPERATIVE_RUNTIME_H_
