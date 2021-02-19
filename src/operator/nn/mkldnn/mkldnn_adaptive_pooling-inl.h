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
 * \file mkldnn_adaptive_pooling-inl.h
 * \brief
 * \author Mateusz Ozga
*/
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ADAPTIVE_POOLING_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ADAPTIVE_POOLING_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>
#include <utility>
#include "../../operator_common.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

class MKLDNNAdaptivePoolingFwd {
 public:
  MKLDNNAdaptivePoolingFwd(const mxnet::NDArray &input,
                           const mxnet::NDArray &output,
                           const mkldnn::memory::dims &kernel,
                           const mkldnn::memory::dims &strides,
                           const mkldnn::memory::dims &pad_l,
                           const mkldnn::memory::dims &pad_r,
                           const mkldnn::algorithm alg_kind,
                           const bool with_workspace, const bool is_train)
      : with_workspace_(with_workspace), fwd_(nullptr) {
    Init(input, output, kernel, strides, pad_l, pad_r, is_train, alg_kind);
  }
  ~MKLDNNAdaptivePoolingFwd() = default;

 public:
  void Execute(const NDArray &input, const OpReqType req, const NDArray &output,
               const NDArray *workspace);

 private:
  bool with_workspace_;
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::pooling_forward> fwd_;

 private:
  void Init(const mxnet::NDArray &input, const mxnet::NDArray &output,
            const mkldnn::memory::dims &kernel,
            const mkldnn::memory::dims &strides,
            const mkldnn::memory::dims &pad_l,
            const mkldnn::memory::dims &pad_r, const bool is_train,
            const mkldnn::algorithm alg_kind);
};


template <typename T = mkldnn::memory::dims>
void updateAdaptivePaddingKernel(T *kernel, T *strides, T *pad_l, T *pad_r,
                                 const NDArray &in_data,
                                 const NDArray &out_data) {
  const int IH = in_data.shape()[2];
  const int IW = in_data.shape()[3];
  const int OH = out_data.shape()[2];
  const int OW = out_data.shape()[3];

  strides->at(0) = floor((IH << 1) / OH) - floor(IH / OH);
  strides->at(1) = floor((IW << 1) / OW) - floor(IW / OW);
  kernel->at(0) = ceil((IH << 1) / OH) - floor(IH / OH);
  kernel->at(1) = ceil((IW << 1) / OW) - floor(IW / OW);
  pad_l->at(0) = (strides->at(0) * (OH - 1) + kernel->at(0) - IH) >> 1;
  pad_l->at(1) = (strides->at(1) * (OW - 1) + kernel->at(1) - IW) >> 1;
}

template <typename T>
MKLDNNAdaptivePoolingFwd &GetPoolingFwd(const T &param, const bool is_train,
                                        const NDArray &input,
                                        const NDArray &output) {
  if (input.shape().ndim() != 4) {
    LOG(FATAL) << "MKLDNN Adaptive Avg Pool 2d: Expect only 2D input";
  }
  typedef ParamOpSign<T> MKLDNNPoolingSignature;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNPoolingSignature,
                                         MKLDNNAdaptivePoolingFwd, OpHash>
      pooling_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNPoolingSignature,
                                            MKLDNNAdaptivePoolingFwd, OpHash>
      pooling_fwds;
#endif
  bool with_workspace = is_train && true;
  MKLDNNPoolingSignature key(param);
  key.AddSign(is_train);
  key.AddSign(with_workspace);
  key.AddSign(input);
  key.AddSign(output);

  auto it = pooling_fwds.find(key);
  if (it == pooling_fwds.end()) {
    const int kernel_ndims = input.shape().ndim();

    mkldnn::memory::dims kernel(kernel_ndims);
    mkldnn::memory::dims strides(kernel_ndims);
    mkldnn::memory::dims pad_l(kernel_ndims);
    mkldnn::memory::dims pad_r(kernel_ndims);

    updateAdaptivePaddingKernel(&kernel, &strides, &pad_l, &pad_r, input, output);
    mkldnn::memory::validate_dims(kernel);
    mkldnn::memory::validate_dims(strides);
    mkldnn::memory::validate_dims(pad_l);
    mkldnn::memory::validate_dims(pad_r);

    mkldnn::algorithm kind = mkldnn::algorithm::pooling_avg;
    MKLDNNAdaptivePoolingFwd fwd(input, output, kernel, kernel, pad_l, pad_r,
                                 kind, false, false);
    it = AddToCache(&pooling_fwds, key, fwd);
  }
  return it->second;
}

template <typename T>
void MKLDNNAdaptivePoolingCompute(const OpContext &ctx, const T &param,
                                  const NDArray &in_data, const OpReqType req,
                                  const NDArray &out_data,
                                  const NDArray *workspace) {
  auto &fwd = GetPoolingFwd(param, ctx.is_train, in_data, out_data);
  fwd.Execute(in_data, req, out_data, workspace);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_ADAPTIVE_POOLING_INL_H_
