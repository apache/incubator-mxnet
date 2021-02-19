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
 * \file mkldnn_adaptive_pooling.cc
 * \brief
 * \author Mateusz Ozga
*/

#if MXNET_USE_MKLDNN == 1

#include "./mkldnn_adaptive_pooling-inl.h"

namespace mxnet {
namespace op {
void MKLDNNAdaptivePoolingFwd::Init(
    const mxnet::NDArray &input, const mxnet::NDArray &output,
    const mkldnn::memory::dims &kernel, const mkldnn::memory::dims &strides,
    const mkldnn::memory::dims &pad_l, const mkldnn::memory::dims &pad_r,
    const bool is_train, const mkldnn::algorithm alg_kind) {
  const auto src_md = input.GetMKLDNNData()->get_desc();
  const auto dst_md = GetMemDesc(output);
  const mkldnn::engine engine = CpuEngine::Get()->get_engine();

  if (alg_kind != mkldnn::algorithm::pooling_avg &&
      alg_kind != mkldnn::algorithm::pooling_avg_include_padding &&
      alg_kind != mkldnn::algorithm::pooling_avg_exclude_padding) {
    LOG(FATAL) << "MKLDNN Adaptive Pooling: algorithm is not supported";
  }

  mkldnn::prop_kind prop = mkldnn::prop_kind::forward_scoring;
  if (is_train && alg_kind != mkldnn::algorithm::pooling_avg) {
    prop = mkldnn::prop_kind::forward_training;
  }
  if (is_train && prop == mkldnn::prop_kind::forward_scoring) {
    LOG(INFO) << "MKLDNN Pooling: training with prop_kind is forward_scoring";
  }

  const auto fwd_desc = mkldnn::pooling_forward::desc(
      prop, alg_kind, src_md, dst_md, strides, kernel, pad_l, pad_r);
  this->fwd_pd_.reset(
      new mkldnn::pooling_forward::primitive_desc(fwd_desc, engine));
  this->fwd_.reset(new mkldnn::pooling_forward(*(this->fwd_pd_)));
}

void MKLDNNAdaptivePoolingFwd::Execute(const NDArray &input,
                                       const OpReqType req,
                                       const NDArray &output,
                                       const NDArray *workspace) {
  NDArray in_buffer = input;
  if (input.IsView() && input.IsMKLDNNData()) {
    in_buffer = input.Reorder2Default();
  }

  auto input_mem = in_buffer.GetMKLDNNData();
  auto output_mem_t = CreateMKLDNNMem(output, this->fwd_pd_->dst_desc(), req);

  mkldnn_args_map_t args = {{MKLDNN_ARG_SRC, *input_mem},
                            {MKLDNN_ARG_DST, *(output_mem_t.second)}};

  if (this->with_workspace_) {
    auto engine = CpuEngine::Get()->get_engine();
    if (workspace == nullptr) {
      LOG(FATAL) << "MKLDNN Average Pooling: incorrect worskapce input";
    }
    auto ws = std::make_shared<mkldnn::memory>(
        (*(this->fwd_pd_)).workspace_desc(), engine,
        workspace->GetMKLDNNData()->get_data_handle());
    args[MKLDNN_ARG_WORKSPACE] = *ws;
  }
  if (this->fwd_) {
    MKLDNNStream::Get()->RegisterPrimArgs(*(this->fwd_), args);
    CommitOutput(output, output_mem_t);
    MKLDNNStream::Get()->Submit();
  } else {
    LOG(FATAL) << "MKLDNN Pooling: forward primitive is nullptr";
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
