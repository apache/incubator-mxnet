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

#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_INL_H_
#if MXNET_USE_ONEDNN == 1

#include <string>
#include <utility>
#include <vector>
#include "mkldnn.hpp"
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"

namespace mxnet {
namespace op {

static inline bool SupportMKLDNNFCEltwiseFusion(const std::string op_name) {
  if (op_name == "Activation" ||
      op_name == "square" ||
      op_name == "sqrt" ||
      op_name == "exp" ||
      op_name == "abs" ||
      op_name == "clip" ||
      op_name == "LeakyReLU") {
    return true;
  } else {
    return false;
  }
}

static inline mkldnn::algorithm GetMKLDNNEltwiseAlgo(const std::string op_name) {
  if (op_name == "square")
    return mkldnn::algorithm::eltwise_square;
  else if (op_name == "sqrt")
    return mkldnn::algorithm::eltwise_sqrt;
  else if (op_name == "exp")
    return mkldnn::algorithm::eltwise_exp;
  else if (op_name == "abs")
    return mkldnn::algorithm::eltwise_abs;
  else
    LOG(FATAL) << "Unsupported eltwise fusion op: " << op_name;

  return mkldnn::algorithm::undef;
}

static inline bool IsOutputUint8(const MKLDNNFCFullParam& full_param) {
  auto alg = full_param.eltwise_param.alg;
  // TODO(ciyong): some alg doesn't support int8 so far.
  if (full_param.mkldnn_param.with_eltwise &&
      (alg == mkldnn::algorithm::eltwise_relu ||
       alg == mkldnn::algorithm::eltwise_logistic ||
       alg == mkldnn::algorithm::eltwise_soft_relu ||
       alg == mkldnn::algorithm::eltwise_bounded_relu ||
       alg == mkldnn::algorithm::eltwise_square ||
       alg == mkldnn::algorithm::eltwise_sqrt ||
       alg == mkldnn::algorithm::eltwise_exp ||
       alg == mkldnn::algorithm::eltwise_abs)) {
    return true;
  }

  return false;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_INL_H_
