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
 *  Copyright (c) 2019 by Contributors
 * \file node_op_util.h
 * \brief abstraction for commonly used nnvm::Node operations.
 */
#ifndef MXNET_NNVM_NODE_OP_UTIL_H_
#define MXNET_NNVM_NODE_OP_UTIL_H_
#include <mxnet/base.h>
#include <string>
#include <unordered_map>
#include "../operator/elemwise_op_common.h"

namespace mxnet {
namespace util {

class NodeOpGen {
 private:
    const nnvm::NodePtr &dependent_node;

 public:
    explicit NodeOpGen(const nnvm::NodePtr &dependent_node) : dependent_node{dependent_node} {}

    nnvm::NodeEntry mul(const nnvm::NodeEntry &lhs, const nnvm::NodeEntry &rhs) {
        return nnvm::NodeEntry{mxnet::op::MakeNode("elemwise_mul",
                                                   dependent_node->attrs.name + "_mul",
                                                   {lhs, rhs}, nullptr, &dependent_node)};
    }

    nnvm::NodeEntry mul(const nnvm::NodeEntry &x, double scalar) {
        const std::unordered_map<std::string, std::string> scalar_dict =
            {{"scalar", std::to_string(scalar)}};
        return nnvm::NodeEntry{mxnet::op::MakeNode("_mul_scalar",
                                                   dependent_node->attrs.name + "_mul_scalar",
                                                   {x}, &scalar_dict, &dependent_node)};
    }

    nnvm::NodeEntry mul(double scalar, const nnvm::NodeEntry &x) {
        return NodeOpGen::mul(x, scalar);
    }

    nnvm::NodeEntry div(const nnvm::NodeEntry &lhs, const nnvm::NodeEntry &rhs) {
        return nnvm::NodeEntry{mxnet::op::MakeNode("elemwise_div",
                                                   dependent_node->attrs.name + "_div",
                                                   {lhs, rhs}, nullptr, &dependent_node)};
    }

    nnvm::NodeEntry square(const nnvm::NodeEntry &x) {
        return nnvm::NodeEntry{mxnet::op::MakeNode("square",
                                                   dependent_node->attrs.name + "_square",
                                                   {x}, nullptr, &dependent_node)};
    }

    nnvm::NodeEntry reciprocal(const nnvm::NodeEntry &x) {
        return nnvm::NodeEntry{mxnet::op::MakeNode("reciprocal",
                                                   dependent_node->attrs.name + "_reciprocal",
                                                   {x}, nullptr, &dependent_node)};
    }

    nnvm::NodeEntry negative(const nnvm::NodeEntry &x) {
        return nnvm::NodeEntry{mxnet::op::MakeNode("negative",
                                                   dependent_node->attrs.name + "_negative",
                                                   {x}, nullptr, &dependent_node)};
    }

    nnvm::NodeEntry abs(const nnvm::NodeEntry &x) {
        return nnvm::NodeEntry{mxnet::op::MakeNode("abs",
                                                   dependent_node->attrs.name + "_abs",
                                                   {x}, nullptr, &dependent_node)};
    }
};

}  // namespace util
}  // namespace mxnet

#endif  // MXNET_NNVM_NODE_OP_UTIL_H_
