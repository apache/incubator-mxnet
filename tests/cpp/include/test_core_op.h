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
#ifndef TEST_CORE_OP_H_
#define TEST_CORE_OP_H_

#include <nnvm/node.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <string>
#include <map>
#include <queue>
#include <set>
#include "./test_op.h"
#include "profiler/vtune.h"
#include "../../../src/imperative/imperative_utils.h"
#include "../../../src/operator/operator_common.h"

namespace mxnet {
namespace test {
namespace op {

// Tried making this a struct w/constexpr, but getting undefined reference on gcc 5.4.1
#define COREOP_FWD_OP_NAME_KEY          "fwd_op_name"
#define COREOP_BWD_OP_NAME_KEY          "bwd_op_name"
#define COREOP_BWD_OP_NAME_VALUE_NONE   "[none]"

enum TimingDirection {
  kForward,
  kBackward
};

inline const char *TimingDirectionAsString(const TimingDirection td) {
  switch (td) {
    case kForward:
      return "Forward";
    case kBackward:
      return "Backward";
    default:
      CHECK(false) << "Unknown timing direction: " << static_cast<int>(td);
      return "<unknown>";
  }
}

/*!
 * Low-noise operator executor
 * @tparam DType Data type for the operator executions
 */
template<typename DType, typename AccReal = float>
class CoreOpExecutor : public test::op::OperatorDataInitializer<DType>
  , public test::op::OperatorExecutorTiming {
  /*! \brief Performance timing categories */
  /*!
   * \brief Parse additional arguments into NodeAttrs structure
   * \param op Pointer to operator object
   * \param args vector of string pairs representing argument key/value pairs
   * \return Constructed NodeAttrs structure
   */
  static nnvm::NodeAttrs ParseAttrs(const nnvm::Op *op, const kwargs_t& args) {
    const size_t count = args.size();
    std::vector<const char *> keys, values;
    keys.reserve(count);
    values.reserve(count);
    for (kwargs_t::const_iterator i_iter = args.begin(), e_iter = args.end();
         i_iter != e_iter; ++i_iter) {
      keys.emplace_back(i_iter->first.c_str());
      values.emplace_back(i_iter->second.c_str());
    }
    return imperative::ParseAttrs(op, op->num_inputs, count, keys.data(), values.data());
  }

  /*!
   * \brief Return vector of data blobs associated with anm array of NDArray objects
   * \param src vector of NDArrays
   * \param dest Vector to store pointers to the NDArrays' data blobs
   * \return Reference to the supplied vector of TBlob results
   */
  static inline std::vector<TBlob>& CollectBlobs(const std::vector<NDArray>& src,
                                                 std::vector<TBlob> *dest) {
    dest->resize(0);
    dest->reserve(dest->size() + src.size());
    for (size_t i = 0, n = src.size(); i < n; ++i) {
      dest->emplace_back(src[i].data());
    }
    return *dest;
  }

  /*!
   * \brief Create NDArray of random data
   * \param shape Shape of the tensor to be created
   * \param ctx Context to use when creating the array/tensor
   * \return The created NDArray
   */
  NDArray CreateRandArray(const mxnet::TShape& shape, const RunContext& run_ctx, int dtype) const {
    CHECK_GT(shape.Size(), 0);  // Check it's a valid shape
    NDArray array(shape, run_ctx.ctx, true, dtype);
    array.CheckAndAlloc();
    test::op::OperatorDataInitializer<DType>::FillRandom(run_ctx, array.data());
    return array;
  }

  /*!
   * \brief Create NDArray of zeros
   * \param shape Shape of the tensor to be created
   * \param ctx Context to use when creating the array/tensor
   * \return The created NDArray
   */
  NDArray CreateZeroArray(const mxnet::TShape& shape, const RunContext& run_ctx, int dtype) const {
    CHECK_GT(shape.Size(), 0);  // Check it's a valid shape
    NDArray array(shape, run_ctx.ctx, true, dtype);
    array.CheckAndAlloc();
    test::op::OperatorDataInitializer<DType>::FillZero(run_ctx, array.data());
    return array;
  }

  nnvm::ObjectPtr MakeNode() const {
    nnvm::ObjectPtr node = nnvm::Node::Create();
    node->attrs = attrs_;
    for (uint32_t i = 0; i < node->num_inputs(); ++i) {
      auto input = nnvm::Node::Create();
      input->attrs.name = "input_" + std::to_string(i);
      node->inputs.emplace_back(nnvm::NodeEntry(input, 0, 0));
    }
    return node;
  }

  /*!
   * \brief Get backward op executors
   * \return Vector of backward executors
   */
  std::vector<std::pair<std::shared_ptr<CoreOpExecutor>, std::string>> GetBackward() {
    std::vector<std::pair<std::shared_ptr<CoreOpExecutor>, std::string>> res;
    static auto gradient = nnvm::Op::GetAttr<nnvm::FGradient>("FGradient");
    nnvm::FGradient grad_fun = gradient.get(op_, nullptr);
    if (grad_fun) {
      auto n = MakeNode();
      uint32_t index = -1;
      std::map<std::string, uint32_t> name_to_indexs;
      std::vector<nnvm::NodeEntry> index_to_nodes;

      std::vector<nnvm::NodeEntry> out_grads(n->num_outputs());
      for (auto i = 0; i < out_grads.size(); ++i) {
        out_grads[i].node =  nnvm::Node::Create();
        out_grads[i].node->attrs.name =  "out_grad_" + std::to_string(i);
        name_to_indexs[out_grads[i].node->attrs.name] = ++index;
        out_grads[i].index = index;
        index_to_nodes.push_back(out_grads[i]);
        index_to_nodes[index].index = index;
        bwd_inputs_.emplace_back(CreateRandArray(output_shapes_[i], ctx_.run_ctx,
                output_types_[i]));
      }
      for (auto i = 0; i < n->num_inputs(); ++i) {
        name_to_indexs[n->inputs[i].node->attrs.name] = ++index;
        n->inputs[i].index = index;
        index_to_nodes.push_back(n->inputs[i]);
        index_to_nodes[index].index = index;
        bwd_outputs_.emplace_back(CreateZeroArray(input_shapes_[i], ctx_.run_ctx,
                input_types_[i]));
      }

      std::vector< nnvm::NodeEntry> entries = grad_fun(n, out_grads);
      CHECK_GE(entries.size(), 1U);

      std::map<uint32_t, uint32_t > bwd_output_nodes;
      std::queue<nnvm::NodeEntry*> queue;
      std::unordered_set<std::string> visited;
      for (auto i = 0; i < entries.size(); ++i) {
        nnvm::NodeEntry& node_entry = entries[i];
        queue.push(&node_entry);
        visited.insert(node_entry.node->attrs.name);
        name_to_indexs[node_entry.node->attrs.name] = ++index;
        node_entry.index = index;
        index_to_nodes.push_back(node_entry);
        index_to_nodes[index].index = index;
        bwd_output_nodes[node_entry.index] = i;
      }
      while (!queue.empty()) {
        auto* node_entry = queue.front();
        queue.pop();

        CHECK_NOTNULL(node_entry);
        CHECK_NOTNULL(node_entry->node.get());
        CHECK_NOTNULL(node_entry->node->op());
        CHECK_GT(node_entry->node->op()->name.size(), 0);

        auto it = name_to_indexs.find(node_entry->node->attrs.name);
        if (it == name_to_indexs.end()) {
          name_to_indexs[node_entry->node->attrs.name] = ++index;
          index_to_nodes.push_back(*node_entry);
        } else  {
          node_entry->index = it->second;
        }

        for (nnvm::NodeEntry& input : node_entry->node->inputs) {
            if (visited.find(input.node->attrs.name) != visited.end()) {
              continue;
            } else {
              input.index = name_to_indexs[input.node->attrs.name];
            }
            visited.insert(input.node->attrs.name);
            queue.push(&input);
        }
      }

      std::map<uint32_t, std::set<uint32_t>> node_outputs;
      std::queue<uint32_t > index_queue;
      std::unordered_set<uint32_t> visited_indexes;
      for (auto i = 0; i < entries.size(); ++i) {
        nnvm::NodeEntry& node_entry = entries[i];
        visited_indexes.insert(node_entry.index);
        index_queue.push(node_entry.index);
      }
      while (!index_queue.empty()) {
        auto node_index = index_queue.front();
        index_queue.pop();
        const auto& node_entry = index_to_nodes.at(node_index);

        for (nnvm::NodeEntry& input : node_entry.node->inputs) {
          node_outputs[input.index].insert(node_entry.index);
          if (visited_indexes.find(input.index) != visited_indexes.end()) {
            continue;
          }

          visited_indexes.insert(input.index);
          index_queue.push(input.index);
        }
      }

      visited_indexes.clear();
      std::map<uint32_t, mxnet::TShape> node_shapes;
      std::map<uint32_t, int> node_types;
      for (auto i = 0; i < n->num_inputs(); ++i) {
        index_queue.push(n->inputs[i].index);
        visited_indexes.insert(n->inputs[i].index);
        node_shapes[n->inputs[i].index] = input_shapes_[i];
        node_types[n->inputs[i].index] = input_types_[i];
      }
      for (auto i = 0; i < out_grads.size(); ++i) {
        index_queue.push(out_grads[i].index);
        visited_indexes.insert(out_grads[i].index);
        node_shapes[out_grads[i].index] = output_shapes_[i];
        node_types[out_grads[i].index] = output_types_[i];
      }
      bwd_output_executors_.resize(entries.size());
      std::map<uint32_t, CoreOpExecutor*> node_executors;
      // the order of nodes getting from button-to-up DFS can't be used
      // for executing CoreOpExecutors.
      while (!index_queue.empty()) {
        auto node_index = index_queue.front();
        index_queue.pop();
        const auto& node_entry = index_to_nodes.at(node_index);

        for (auto output_index : node_outputs[node_entry.index]) {
          if (visited_indexes.find(output_index) != visited_indexes.end()) {
            continue;
          }
          visited_indexes.insert(output_index);
          index_queue.push(output_index);
        }

        std::vector<mxnet::TShape> input_shapes;
        input_shapes.reserve(node_entry.node->num_inputs());
        for (nnvm::NodeEntry& input : node_entry.node->inputs) {
          CHECK(node_shapes.find(input.index) != node_shapes.end());
          input_shapes.push_back(node_shapes[input.index]);
        }

        if (node_entry.node->is_variable() || node_entry.node->op()->name.empty()) {
          continue;
        }

        if (verbose_) {
          std::cout << node_entry.node->op()->name << std::endl;
        }

        std::vector<mxnet::TShape> output_shapes;
        output_shapes.resize(node_entry.node->num_outputs());
        static auto& finfer_shape = Op::GetAttr<mxnet::FInferShape>("FInferShape");
        if (finfer_shape.count(node_entry.node->op())) {
          mxnet::FInferShape call_infer_shapes = finfer_shape[node_entry.node->op()];
          call_infer_shapes(node_entry.node->attrs, &input_shapes, &output_shapes);
          // how to handle #output >  1?
          CHECK_EQ(output_shapes.size(), 1U);
          node_shapes[node_entry.index] = output_shapes[0];
        } else {
          CHECK(false) << "can't find finfer_shape for op " << node_entry.node->op()->name;
        }

        std::shared_ptr<CoreOpExecutor> pOp = std::make_shared<CoreOpExecutor>(
                ctx().run_ctx.ctx.dev_type == Context::kGPU, input_shapes);

        node_executors[node_entry.index] = pOp.get();
        bwd_input_map_[pOp.get()];
        res.push_back({pOp, node_entry.node->op()->name});
        auto index_iter = bwd_output_nodes.find(node_entry.index);
        if (index_iter != bwd_output_nodes.end()) {
          bwd_output_executors_[index_iter->second] = pOp.get();
        }

        for (const auto& input : node_entry.node->inputs) {
          ExecutorInputSources executor_input_source;
          if (input.index < bwd_inputs_.size())  {
            executor_input_source.type  = ExecutorInputSources::GRAD;
            executor_input_source.index = input.index;
          } else if (input.index < bwd_inputs_.size() + inputs_.size()) {
            executor_input_source.type  = ExecutorInputSources::INPUT;
            executor_input_source.index = input.index - bwd_inputs_.size();
          } else {
            executor_input_source.type  = ExecutorInputSources::EXECUTOR;
            executor_input_source.index = 0;
            executor_input_source.executor = node_executors[input.index];
          }
          bwd_input_map_[pOp.get()].push_back(executor_input_source);
        }
      }
    }
    return res;
  }

  /*!
   * \brief Attach any temp or random resources required to perform the op's compute operation
   * \param ctx Operator context object
   * \param attrs NodeAttrs structure (node attributes)
   * \param op Pointer to nnvm Operator object
   */
  void AttachResources(OpContext *ctx, const nnvm::NodeAttrs& attrs, const nnvm::Op *op) {
    std::vector<ResourceRequest> reqs;
    std::vector<Resource>& requested = ctx->requested;
    static auto& fresource = nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");
    if (fresource.count(op) != 0) {
      reqs = fresource[op](attrs);
    } else {
      static auto& fresourceex = nnvm::Op::GetAttr<FResourceRequestEx>("FResourceRequestEx");
      if (fresourceex.count(op) != 0) {
        if (this->function_ || this->stateful_function_) {
          reqs = fresourceex[op](attrs, ctx->run_ctx.ctx.dev_mask(), DispatchMode::kFCompute);
        } else {
          reqs = fresourceex[op](attrs, ctx->run_ctx.ctx.dev_mask(), DispatchMode::kFComputeEx);
        }
      }
    }
    if (!reqs.empty()) {
      // Get the resource of temporal space.
      for (const ResourceRequest& req : reqs) {
        switch (req.type) {
          case ResourceRequest::kTempSpace: {
            requested.emplace_back(ResourceManager::Get()->Request(ctx->run_ctx.ctx, req));
            break;
          }
          case ResourceRequest::kRandom: {
            requested.emplace_back(ResourceManager::Get()->Request(ctx->run_ctx.ctx, req));
            break;
          }
          case ResourceRequest::kParallelRandom: {
            Resource rm = ResourceManager::Get()->Request(ctx->run_ctx.ctx, req);
            if (ctx->run_ctx.ctx.dev_mask() == Context::kCPU) {
              common::random::RandGenerator<cpu, DType>::AllocState(
                  rm.get_parallel_random<cpu, DType>());
            }
            requested.emplace_back(rm);
            break;
          }
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
          case ResourceRequest::kCuDNNDropoutDesc: {
            requested.emplace_back(ResourceManager::Get()->Request(ctx->run_ctx.ctx, req));
            break;
          }
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
          default:
            LOG(FATAL) << "resource type " << req.type << " is not yet supported";
        }
      }
    }
  }

 public:
  typedef DType   DataType;
  typedef AccReal AccRealType;

  /*! \brief Add 'fwd_op_name' to kwargs and return the new kwargs */
  static kwargs_t ArgsWithOpName(const kwargs_t& args,
                                 const std::string& fwd_op_name,
                                 const std::string& bwd_op_name = "") {
    CHECK(!fwd_op_name.empty());
    kwargs_t new_args;
    new_args.reserve(args.size() + 1);
    for (const auto& a : args) {
      if (a.first != COREOP_FWD_OP_NAME_KEY && a.first != COREOP_BWD_OP_NAME_KEY) {
        new_args.emplace_back(a);
      }
    }
    new_args.push_back({ COREOP_FWD_OP_NAME_KEY, fwd_op_name});
    if (!bwd_op_name.empty()) {
      new_args.push_back({ COREOP_BWD_OP_NAME_KEY, bwd_op_name});
    }
    return new_args;
  }

  /*! \brief Remove 'fwd_op_name' from kwargs and return the new kwargs */
  static kwargs_t ArgsSansOpName(const kwargs_t& args,
                                 std::string* fwd_op_name_ptr,
                                 std::string* bwd_op_name_ptr = nullptr) {
    CHECK_NOTNULL(fwd_op_name_ptr);
    CHECK_NOTNULL(bwd_op_name_ptr);
    bwd_op_name_ptr->resize(0);
    kwargs_t new_args;
    new_args.reserve(args.size());
    for (const auto& a : args) {
      if (a.first == COREOP_FWD_OP_NAME_KEY) {
        *fwd_op_name_ptr = a.second;
      } else if (a.first == COREOP_BWD_OP_NAME_KEY) {
        *bwd_op_name_ptr = a.second;
      } else {
        new_args.emplace_back(a);
      }
    }
    return new_args;
  }

  /*!
   * \brief Constructor
   * \param isGPU Is this going to be on the GPU?
   * \param shapes Array of input shapes
   */
  CoreOpExecutor(const bool isGPU, const mxnet::ShapeVector& shapes)
    : input_shapes_(shapes)
      , op_(nullptr)  {
    ctx_.is_train = true;
    ctx_.run_ctx.ctx.dev_id = 0;
    ctx_.run_ctx.stream = nullptr;
    ctx_.run_ctx.ctx.dev_type = Context::kCPU;
#if MXNET_USE_CUDA
    if (isGPU) {
      ctx_.run_ctx.ctx.dev_type = Context::kGPU;
      allocGPUStream_.reset(new GPUStreamScope(&ctx_));
    } else {
      ctx_.run_ctx.ctx.dev_type = Context::kCPU;
    }
#else
    CHECK(!isGPU);
    ctx_.run_ctx.ctx.dev_type = Context::kCPU;
#endif
  }

  /*!
   * \brief Get the operator context
   * \return Reference to this operator's context object
   */
  const OpContext& ctx() const {
    return ctx_;
  }

  static inline int default_dtype() {
    using foo = typename mshadow::DataType<DType>;
    return foo::kFlag;
  }

  nnvm::ObjectPtr GetBackwardDependency(const nnvm::ObjectPtr& node,
                                      std::map<int, const NDArray *>* index2array) const {
    index2array->clear();
    static auto& fgradient = nnvm::Op::GetAttr<nnvm::FGradient>("FGradient");

    const uint32_t num_inputs  = inputs().size();
    const uint32_t num_outputs = outputs().size();

    node->inputs.clear();
    node->inputs.reserve(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i) {
      node->inputs.emplace_back(nullptr, i, 0);
      (*index2array)[i] = &inputs()[i];
    }

    if (fgradient.count(node->op())) {
      std::vector<nnvm::NodeEntry> ograd_entries;
      ograd_entries.reserve(num_outputs);
      for (uint32_t i = 0; i < num_outputs; ++i) {
        const uint32_t index = num_inputs + i;
        ograd_entries.emplace_back(nullptr, index, 1);
        (*index2array)[index] = &outputs()[i];
      }
      const std::vector<nnvm::NodeEntry> igrad_entries = fgradient[node->op()](node, ograd_entries);

      if (!igrad_entries.empty()) {
        return igrad_entries[0].node;
      }
    }
    return nullptr;
  }

  nnvm::ObjectPtr CalcBackwardPass(std::map<int, const NDArray *> *index2array) const {
    nnvm::ObjectPtr node = nnvm::Node::Create();
    node->attrs = attrs_;
    return GetBackwardDependency(node, index2array);
  }

  /*!
   * \brief Initialize the execution objects and execution data (only occurs once)
   * \param args Parameter arguments
   * \param inputs Optional input data (otherwise, random data will be used as input)
   */
  void Init(const kwargs_t& in_args,
            const std::vector<NDArray>& inputs = {},
            const std::vector<NDArray>& outputs = {},
            const CoreOpExecutor *backward_for_op = nullptr,
            nnvm::ObjectPtr bwd_node_ptr = nullptr
  ) {
    if (!initialized_) {
      initialized_ = true;

      std::string op_name, bwd_op_name;
      kwargs_t args = ArgsSansOpName(in_args, &op_name, &bwd_op_name);
      CHECK(!op_name.empty());

      CHECK(!backward_for_op || bwd_op_name.empty())
        << "Backward op should not be supplied another backward operator";

      if (verbose_ && backward_for_op) {
        std::cout << "Backward op: " << op_name;
      }

      op_ = nnvm::Op::Get(op_name);
      CHECK_NOTNULL(op_);

      std::map<int, const NDArray *> index2array;
      if (backward_for_op) {
        bwd_node_ptr = backward_for_op->CalcBackwardPass(&index2array);
      }

      // Set up forward
      attrs_ = ParseAttrs(op_, args);

      int num_inputs = op_->num_inputs;
      if (op_->get_num_inputs) {
        num_inputs = op_->get_num_inputs(attrs_);
      } else if (backward_for_op) {
        if (bwd_node_ptr) {
          num_inputs = static_cast<int>(bwd_node_ptr->inputs.size());
        }
      }

      if (!inputs.empty()) {
        CHECK_EQ(inputs.size(), static_cast<size_t>(num_inputs));
      }

      int inferred_num_outputs /*, num_visible_outputs*/;

      if (op_->get_num_outputs) {
        inferred_num_outputs = op_->get_num_outputs(attrs_);
      } else {
        inferred_num_outputs = op_->num_outputs;
      }

      // Generic, all shapes the same. Probably this will need to be adjusted for more complex
      // operators such as dot
      std::vector<mxnet::TShape> input_shapes;
      if (!input_shapes_.empty()) {
        for (size_t i = 0, n = num_inputs; i < n; ++i) {
          input_shapes.emplace_back(i < input_shapes_.size() ? input_shapes_[i]
                                                             : input_shapes_[input_shapes_.size()
                                                                             - 1]);
        }
      }
      std::vector<NDArray *> inputs_p, outputs_p;

      if (!outputs.empty()) {
        CHECK_EQ(outputs.size(), static_cast<size_t>(inferred_num_outputs));
      }

      inputs_.reserve(num_inputs);
      inputs_p.reserve(num_inputs);
      outputs_.reserve(inferred_num_outputs);
      outputs_p.reserve(inferred_num_outputs);

      std::vector<int> input_types;
      input_types.reserve(num_inputs);
      std::vector<int> output_types;
      output_types.reserve(inferred_num_outputs);

      static auto& finfer_type = Op::GetAttr<nnvm::FInferType>("FInferType");
      if (finfer_type.count(op_)) {
        input_types.resize(num_inputs, -1);
        input_types[0] = default_dtype();  // Set first input to default type
        output_types.resize(inferred_num_outputs, -1);
        finfer_type[op_](attrs_, &input_types, &output_types);
        CHECK_EQ(input_types.size(), num_inputs);
        CHECK_EQ(output_types.size(), inferred_num_outputs);
      } else {
        if (backward_for_op) {
          if (bwd_node_ptr) {
            CHECK_EQ(bwd_node_ptr->inputs.size(), num_inputs);
            input_types.resize(bwd_node_ptr->inputs.size(), -1);
            for (int i = 0; i < num_inputs; ++i) {
              const int map_key = bwd_node_ptr->inputs[i].index;
              CHECK(index2array.find(map_key) != index2array.end());
              const int dtype = index2array[map_key]->dtype();
              input_types[i] = dtype;
            }
            for (const auto &fwd_inp : backward_for_op->inputs()) {
              const int dtype = fwd_inp.data().type_flag_;
              output_types.emplace_back(dtype);
            }
          } else {
            for (int x = 0; x < num_inputs; ++x) {
              input_types.emplace_back(default_dtype());
            }
            for (const auto &fwd_inp : backward_for_op->inputs()) {
              const int dtype = fwd_inp.data().type_flag_;
              output_types.emplace_back(dtype);
            }
          }
        } else {
          CHECK(false);  // above always true?
          for (int x = 0; x < num_inputs; ++x) {
            input_types.emplace_back(default_dtype());
          }
          for (int x = 0; x < inferred_num_outputs; ++x) {
            output_types.emplace_back(default_dtype());
          }
        }
      }
      input_types_ = input_types;

      // Output arrays
      if (outputs_.empty()) {
        std::vector<mxnet::TShape> output_shapes;
        static auto& finfer_shape = Op::GetAttr<mxnet::FInferShape>("FInferShape");
        if (finfer_shape.count(op_)) {
          mxnet::FInferShape call_infer_shapes = finfer_shape[op_];
          output_shapes.resize(inferred_num_outputs);
          call_infer_shapes(attrs_, &input_shapes, &output_shapes);
          input_shapes_ = input_shapes;
        } else {
          if (backward_for_op) {
            // BWD Input shapes
            if (bwd_node_ptr) {
              input_shapes.clear();
              CHECK_EQ(bwd_node_ptr->inputs.size(), num_inputs);
              for (int i = 0; i < num_inputs; ++i) {
                const int map_key = bwd_node_ptr->inputs[i].index;
                CHECK(index2array.find(map_key) != index2array.end());
                const mxnet::TShape &shp = index2array[map_key]->shape();
                input_shapes.push_back(shp);
                const mxnet::TShape ss = input_shapes[i];
              }
            } else {
              // TODO(cjolivier)
            }
            input_shapes_ = input_shapes;
            // BWD Output shapes
            output_shapes = backward_for_op->input_shapes_;
            output_shapes.resize(inferred_num_outputs);
          } else {
            output_shapes = input_shapes;
            output_shapes.resize(inferred_num_outputs);
          }
        }
        CHECK_EQ(output_shapes.size(), inferred_num_outputs);

        for (size_t i = 0; i < static_cast<size_t>(inferred_num_outputs); ++i) {
          // If supplied and valid, pass from the supplied outputs vector
          // Otherwise use empty for forward pass, or zero-filled for backward pass
          outputs_.emplace_back(i < outputs.size() ? outputs[i]
                                                   : (backward_for_op
                                                      ? CreateZeroArray(output_shapes[i],
                                                                        ctx_.run_ctx,
                                                                        output_types[i])
                                                      : NDArray()));
          outputs_p.emplace_back(&*outputs_.rbegin());
        }
        output_shapes_ = std::move(output_shapes);
      }
      output_types_ = std::move(output_types);

      for (size_t i = 0; i < static_cast<size_t>(num_inputs); ++i) {
        CHECK_LT(i, static_cast<int>(input_shapes.size()));
        inputs_.emplace_back(i < inputs.size()
                             ? inputs[i] : CreateRandArray(input_shapes[i],
                                                           ctx_.run_ctx,
                                                           input_types[i]));
        inputs_p.emplace_back(&*inputs_.rbegin());
      }

      if (!backward_for_op) {
        DispatchMode dispatch_mode = DispatchMode::kUndefined;
        imperative::SetShapeType(ctx_.run_ctx.ctx, attrs_, inputs_p, outputs_p, &dispatch_mode);
      }

      std::vector<OpReqType> req;
      imperative::SetWriteInplaceReq(inputs_p, outputs_p, &req_);

      CollectBlobs(inputs_, &blob_inputs_);
      CollectBlobs(outputs_, &blob_outputs_);

      function_ = common::GetFCompute<FCompute>(op_, "FCompute", ctx_.run_ctx.ctx);
      functionex_ = common::GetFCompute<FComputeEx>(op_, "FComputeEx", ctx_.run_ctx.ctx);
      stateful_function_ = common::GetFCompute<FStatefulCompute>(op_, "FStatefulCompute",
                                                                 ctx_.run_ctx.ctx);

      AttachResources(&ctx_, attrs_, op_);

      auto& is_layer_backward = Op::GetAttr<bool>("TIsLayerOpBackward");
      auto& createop = nnvm::Op::GetAttr<FCreateOpState>("FCreateOpState");
      if (createop.count(op_) || is_layer_backward.get(op_, false)) {
        if (backward_for_op) {
          state_ = backward_for_op->state_;
        }
        if (!state_) {
          if (!create_state_) {
            create_state_ = createop[op_];
          }
          state_ = create_state_(attrs_, ctx_.run_ctx.ctx, input_shapes_, input_types);
        }
      }

      if (!backward_for_op) {
        bool no_backward = false;
        // Set up backward
        std::vector<std::pair<std::shared_ptr<CoreOpExecutor>, std::string>> bwd;
        if (!bwd_op_name.empty()) {
          if (bwd_op_name != COREOP_BWD_OP_NAME_VALUE_NONE) {
            // Backward op was specified
            std::shared_ptr<CoreOpExecutor> pOp = std::make_shared<CoreOpExecutor>(
              ctx().run_ctx.ctx.dev_type == Context::kGPU, ShapesOf(this->outputs()));
            bwd.push_back({pOp, bwd_op_name});
          } else {
            no_backward = true;
          }
        } else {
          // Try to figure out backward op
          bwd = GetBackward();
        }
        if (!no_backward) {
          CHECK_GE(bwd.size(), 1U)
            << "Can't automatically determine backward op name. Please specify";
          for (std::pair<std::shared_ptr<CoreOpExecutor>, std::string> &bw_item : bwd) {
            bw_item.first->set_verbose(verbose_);
            backward_.emplace_back(bw_item.first);
            bw_item.first->Init(ArgsWithOpName(args, bw_item.second), {}, {}, this);
          }
        }
      }
    }
  }

  template<typename OpProp>
  inline bool initForward(const OpProp &opProp, std::vector<int> *in_type) {
    Init(opProp.GetArgs());
    resetForward();
    return true;
  }

  template<typename OpProp>
  inline bool initBackward(const OpProp &opProp, std::vector<int> *in_type) {
    resetBackward();
    return true;
  }

  inline void forward(const size_t count) {
    perf::TimingItem timeF(&OperatorExecutorTiming::GetTiming(), kForward, "Forward", count);
    mxnet::profiler::vtune::VTuneResume profile;
    if (stateful_function_) {
      for (size_t i = 0; i < count; ++i) {
        ExecuteStateful();
      }
    } else {
      for (size_t i = 0; i < count; ++i) {
        Execute();
      }
    }
  }

  inline void backward(const size_t count) {
    CHECK(HasBackward());
    perf::TimingItem timeF(&OperatorExecutorTiming::GetTiming(), kBackward, "Backward", count);
    mxnet::profiler::vtune::VTuneResume profile;
    if (stateful_function_) {
      for (size_t i = 0; i < count; ++i) {
        ExecuteBackwardStateful();
      }
    } else {
      for (size_t i = 0; i < count; ++i) {
        ExecuteBackward();
      }
    }
  }

  /*!
   * \brief Execute the operator for a dense tensor
   */
  void Execute() {
    CHECK_EQ(initialized_, true);
    CHECK_NOTNULL(function_);
    CollectBlobs(inputs_, &blob_inputs_);
    CollectBlobs(outputs_, &blob_outputs_);
    function_(attrs_, ctx_, blob_inputs_, req_, blob_outputs_);
  }

  /*!
   * \brief Execute the operator for a sparse tensor
   */
  void ExecuteEx() {
    CHECK_EQ(initialized_, true);
    CHECK_NOTNULL(functionex_);
    functionex_(attrs_, ctx_, inputs_, req_, outputs_);
  }

  /*!
   * \brief Execute the stateful operator
   */
  void ExecuteStateful() {
    CHECK_EQ(initialized_, true);
    CHECK(state_);
    CollectBlobs(inputs_, &blob_inputs_);
    CollectBlobs(outputs_, &blob_outputs_);
    stateful_function_(state_, ctx_, blob_inputs_, req_, blob_outputs_);
  }

  bool HasBackward() const {
    return !backward_.empty();
  }

  /*!
   * \brief Execute backward pass on operator
   */
  bool ExecuteBackward() {
    CHECK_EQ(initialized_, true);
    CHECK(HasBackward());
    if (!backward_.empty()) {
      // Avoid locked ref count here
      for (std::shared_ptr<CoreOpExecutor> &p : backward_) {
        if (bwd_input_map_.find(p.get()) !=  bwd_input_map_.end()) {
          p->inputs().clear();
          for (const auto &bwd_input_source : bwd_input_map_[p.get()]) {
            if (bwd_input_source.type == ExecutorInputSources::Type::EXECUTOR) {
              p->inputs().push_back(bwd_input_source.executor->outputs()[bwd_input_source.index]);
            } else if (bwd_input_source.type == ExecutorInputSources::Type::GRAD) {
              p->inputs().push_back(bwd_inputs_[bwd_input_source.index]);
            } else {
              p->inputs().push_back(inputs_[bwd_input_source.index]);
            }
          }
        }
        p->Execute();
      }
      if (!bwd_output_executors_.empty()) {
        bwd_outputs().clear();
        for (auto& p : bwd_output_executors_) {
          for (auto& output : p->outputs()) {
            bwd_outputs_.push_back(output);
          }
        }
      }
      return true;
    }
    return false;
  }

  /*!
   * \brief Execute backward pass on operator
   */
  bool ExecuteBackwardEx() {
    CHECK_EQ(initialized_, true);
    CHECK(HasBackward());
    if (!backward_.empty()) {
      // Avoid locked ref count here
      for (std::shared_ptr<CoreOpExecutor> &p : backward_) {
        p->ExecuteEx();
      }
      return true;
    }
    return false;
  }

  /*!
   * \brief Execute backward pass on stateful operator
   */
  bool ExecuteBackwardStateful() {
    CHECK_EQ(initialized_, true);
    CHECK(HasBackward());
    if (!backward_.empty()) {
      // Avoid locked ref count here
      for (std::shared_ptr<CoreOpExecutor> &p : backward_) {
        p->ExecuteStateful();
      }
      return true;
    }
    return false;
  }

  /*!
   * \brief Access input NDArray vector
   * \return reference to NDArray vector of forward inputs
   */
  std::vector<NDArray>& inputs() { return inputs_; }
  const std::vector<NDArray>& inputs() const { return inputs_; }
  std::vector<TBlob>& input_blobs() { return blob_inputs_; }
  const std::vector<TBlob>& input_blobs() const { return blob_inputs_; }

  /*!
   * \brief Access input NDArray vector
   * \return reference to NDArray vector of forward outputs
   */
  std::vector<NDArray>& outputs() { return outputs_; }
  const std::vector<NDArray>& outputs() const { return outputs_; }
  std::vector<TBlob>& output_blobs() { return blob_outputs_; }
  const std::vector<TBlob>& output_blobs() const { return blob_outputs_; }

  /*!
   * \brief Backward inputs (i.e. output grad)
   * \return reference to NDArray vector of backward inputs
   */
  std::vector<NDArray>& bwd_inputs() {
    return bwd_inputs_.empty() ? backward_[0]->inputs() : bwd_inputs_;
  }

  const std::vector<NDArray>& bwd_inputs() const {
    return bwd_inputs_.empty() ? backward_[0]->inputs() : bwd_inputs_;
  }

  /*!
   * \brief Backward outputs (i.e. input grad)
   * \return reference to NDArray vector of backward outputs
   */
  std::vector<NDArray>& bwd_outputs() {
    return bwd_outputs_.empty() ? backward_[0]->outputs() : bwd_outputs_;
  }

  const std::vector<NDArray>& bwd_outputs() const {
    return bwd_outputs_.empty() ? backward_[0]->outputs() : bwd_outputs_;
  }

  void set_verbose(bool verbose) {
    verbose_ = verbose;
  }

  virtual void resetForward() {}

  virtual void resetBackward() {}

 private:
  /*!
   * \brief Has the execution been initialized?
   */
  bool initialized_ = false;
  /*!
   * \brief Whether to print debug trace output
   */
  bool verbose_ = false;
  /*!
   * \brief This operator's context object
   */
  OpContext ctx_;

#if MXNET_USE_CUDA
  /*! \brief
   * Scoped GPU stream
   */
  std::unique_ptr<GPUStreamScope> allocGPUStream_;
#endif

  /*!
   * \brief Input data shape
   */

  mxnet::ShapeVector input_shapes_;
  /*!
   * \brief Input data type
   */
  std::vector<int> input_types_;

  /*!
   * \brief Output data shape
   */
  mxnet::ShapeVector output_shapes_;

  /*!
   * \brief Output data type
   */
  std::vector<int> output_types_;

  /*
   * \brief Pointer to the operator object
   */
  const nnvm::Op *op_;
  /*!
   * \brief Operator attributes
   */
  nnvm::NodeAttrs attrs_;
  /*!
   * \brief Input, output, bwd input and bwd output NDArray vectors
   */
  std::vector<NDArray> inputs_, outputs_, bwd_inputs_, bwd_outputs_;

  /*!
   * \brief Vectors of the TBlob objects associated with the NDArrays in inputs_ and outputs_
   */
  std::vector<TBlob> blob_inputs_, blob_outputs_;
  /*!
   * \brief Operator request type vector
   */
  std::vector<OpReqType> req_;
  /*!
   * \brief Operator's FCompute function (for dense tensors)
   */
  FCompute function_;
  /*!
   * \brief Operator's FCompute function (for sparse tensors)
   */
  FComputeEx functionex_;
  /*!
   * \brief Operator's FStatefulCompute function
   */
  FStatefulCompute stateful_function_;
  /*!
   * \brief Operator's FCreateOpState function
   */
  FCreateOpState create_state_;
  /*!
   * \brief Operator state
   */
  OpStatePtr state_;

  /*!
   * \brief Backward executors (if any)
   */
  std::vector<std::shared_ptr<CoreOpExecutor>> backward_;

  /*!
   * \brief The backwards which generated bwd output
   */
  std::vector<CoreOpExecutor*> bwd_output_executors_;

  struct ExecutorInputSources {
    enum Type {
      INPUT,
      GRAD,
      EXECUTOR
    };
    Type type;
    uint32_t index;
    // used for EXECUTOR
    CoreOpExecutor* executor = nullptr;
  };
  /*!
   * \brief The sources of input of backend CoreOpExecutor
   */
  std::map<CoreOpExecutor*, std::vector<ExecutorInputSources>> bwd_input_map_;
};

class CoreOpProp {
 public:
  virtual void Init(const kwargs_t& kwargs) { kwargs_ = kwargs; }
  const kwargs_t& GetArgs() const { return kwargs_; }
  virtual ~CoreOpProp() {}
 private:
  kwargs_t          kwargs_;
};

template<typename DType>
using CoreOperatorRunner = test::OperatorRunner<CoreOpProp, CoreOpExecutor<DType>>;


/*!
 * \brief Rune a core op forward and backward
 * \tparam DType Data type
 * \param isGPU true if operation is to be run on the GPU
 * \param op_kwargs Operator parameters
 * \param op_name Operator name as registered with nnvm
 * \param backward_op_name Backwards operator name as registered with nnvm
 *        If blank, the runner will attempt to determine the backwards operator. If it fails,
 *        an exception will be thrown.
 *        If the string is [none], then no backward operator will be created or executed
 */
template<typename DType = float>
inline void BasicRunCoreOpBidirectional(const bool isGPU,
                                        bool verbose,
                                        const kwargs_t& op_kwargs,
                                        const mxnet::ShapeVector& shapes,
                                        const char *op_name,
                                        const char *backward_op_name = "") {
  test::op::CoreOpExecutor<DType> op(isGPU, shapes);
  op.set_verbose(verbose);

  op.Init(op.ArgsWithOpName(op_kwargs, op_name, backward_op_name));

  if (verbose) {
    PRINT_NDARRAYS(op.ctx().run_ctx, op.inputs());
    PRINT_NDARRAYS(op.ctx().run_ctx, op.outputs());
  }
  op.Execute();
  if (verbose) {
    PRINT_NDARRAYS(op.ctx().run_ctx, op.outputs());
  }
  if (op.HasBackward()) {
    if (verbose) {
      PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_inputs());
      PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_outputs());
    }
    op.ExecuteBackward();
    if (verbose) {
      PRINT_NDARRAYS(op.ctx().run_ctx, op.bwd_outputs());
    }
  }
}

}  // namespace op
}  // namespace test
}  // namespace mxnet

#endif  // TEST_CORE_OP_H_
