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
 * Copyright (c) 2020 by Contributors
 * \file mkldnn_dot-inl.h
 * \brief Common functions used by MKLDNN using Dot operator implementation
 * \author Bartosz Kuncer, bartosz.kuncer@intel.com
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_DOT_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_DOT_INL_H_

#if MXNET_USE_MKLDNN == 1
#include <vector>

#include "../../tensor/dot-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

inline mkldnn::memory::dims GetInputDims(const mxnet::TShape& shape,
                                         const bool axIsFirst,
                                         const bool isData) {
  const auto data_ndimM1 = shape.ndim() - 1;  // amount of dimensions decreased by 1

  const auto middleAxis = axIsFirst ? shape[0] : shape[data_ndimM1];
  auto combinedAxis = axIsFirst ? shape[data_ndimM1] : shape[0];
  for (size_t i = 1; i < data_ndimM1; ++i) {
    combinedAxis *= shape[i];
  }

  if (data_ndimM1 == 0) {
    // no transposition allowed for 1D tensors
    CHECK(axIsFirst != isData);
    combinedAxis = 1;
  }

  return isData ? mkldnn::memory::dims{combinedAxis, middleAxis}
                : mkldnn::memory::dims{middleAxis, combinedAxis};
}

mkldnn::matmul::primitive_desc GetDotFwdDesc(const DotParam &param,
                                             const NDArray &data,
                                             const NDArray &weight,
                                             const NDArray &output) {
  const mkldnn::memory::dims data_dims      =
                        GetInputDims(data.shape(), param.transpose_a, true),
                             weight_dims    =
                        GetInputDims(weight.shape(), !param.transpose_b, false);

  const mkldnn::memory::dims data_strides   = param.transpose_a ?
                                                mkldnn::memory::dims {1, data_dims[0]}
                                                                :
                                                mkldnn::memory::dims {data_dims[1], 1},
                             weight_strides = param.transpose_b ?
                                                mkldnn::memory::dims {1, weight_dims[0]}
                                                                :
                                                mkldnn::memory::dims {weight_dims[1], 1};

  mkldnn::memory::desc data_md{data_dims, get_mkldnn_type(data.dtype()), data_strides};
  mkldnn::memory::desc weight_md{weight_dims, get_mkldnn_type(weight.dtype()), weight_strides};
  mkldnn::memory::desc out_md{mkldnn::memory::dims{data_dims[0], weight_dims[1]},
                              get_mkldnn_type(output.dtype()),
                              mkldnn::memory::dims{weight_dims[1], 1}};
  mkldnn::matmul::desc fwd_desc(data_md, weight_md, out_md);

  return mkldnn::matmul::primitive_desc(fwd_desc, mxnet::CpuEngine::Get()->get_engine());
}

class MKLDNNDotFwd {
 public:
  mkldnn::matmul::primitive_desc fwd_pd;

  MKLDNNDotFwd(const DotParam &param,
               const NDArray &data,
               const NDArray &weight,
               const NDArray &output)
      : fwd_pd(GetDotFwdDesc(param, data, weight, output)) {
    fwd_ = std::make_shared<mkldnn::matmul>(fwd_pd);
  }

  ~MKLDNNDotFwd() {}

  inline mkldnn_output_t GetOutMem(const OpReqType &req,
                            const NDArray &output,
                            const NDArray &input);

  void Execute(const nnvm::NodeAttrs &attrs,
               const std::vector<NDArray> &inputs,
               const OpReqType &req,
               const NDArray &output);

  const mkldnn::matmul &GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<mkldnn::matmul> fwd_;
};

inline mkldnn_output_t MKLDNNDotFwd::GetOutMem(const OpReqType &req,
                                               const NDArray &output,
                                               const NDArray &input) {
  if (output.shape().ndim() > 7) {
    // MKLDNN does not support 8 and more dims
    auto out_shape = mxnet::TShape(2, fwd_pd.dst_desc().dims()[0]);
    out_shape[1] = fwd_pd.dst_desc().dims()[1];
    return CreateMKLDNNMem(output.MKLDNNDataReshape(out_shape), fwd_pd.dst_desc(), req, &(input));
  } else {
    return CreateMKLDNNMem(output, fwd_pd.dst_desc(), req, &(input));
  }
}

void MKLDNNDotFwd::Execute(const nnvm::NodeAttrs &attrs,
                           const std::vector<NDArray> &inputs,
                           const OpReqType &req,
                           const NDArray &output) {
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  mkldnn::memory data_mem(fwd_pd.src_desc(), cpu_engine,
                          reinterpret_cast<void*>(inputs[0].data().dptr_));
  mkldnn::memory weight_mem(fwd_pd.weights_desc(), cpu_engine,
                            reinterpret_cast<void*>(inputs[1].data().dptr_));
  mkldnn_output_t out_mem = GetOutMem(req, output, inputs[0]);

  mkldnn_args_map_t args = {
      {MKLDNN_ARG_SRC, data_mem},
      {MKLDNN_ARG_WEIGHTS, weight_mem},
      {MKLDNN_ARG_DST, *out_mem.second},
  };

  MKLDNNStream::Get()->RegisterPrimArgs(*fwd_, args);
  CommitOutput(output, out_mem);
  MKLDNNStream::Get()->Submit();
}

typedef ParamOpSign<DotParam> MKLDNNDotSignature;

static inline MKLDNNDotFwd &GetDotFwd(const DotParam &param,
                                      const NDArray &data,
                                      const NDArray &weight,
                                      const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNDotSignature,
              MKLDNNDotFwd, OpHash> dotFwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNDotSignature,
              MKLDNNDotFwd, OpHash> dotFwds;
#endif
  MKLDNNDotSignature key(param);
  key.AddSign(data);
  key.AddSign(weight);
  key.AddSign(output);

  auto it = dotFwds.find(key);
  if (it == dotFwds.end()) {
    MKLDNNDotFwd dotFwd(param, data, weight, output);
    it = AddToCache(&dotFwds, key, dotFwd);
  }
  return it->second;
}

void MKLDNNDotForward(const nnvm::NodeAttrs &attrs,
                      const OpContext &ctx,
                      const std::vector<NDArray> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<NDArray> &outputs) {
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  auto &fwd = GetDotFwd(param, inputs[0], inputs[1], outputs[0]);
  fwd.Execute(attrs, inputs, req[0], outputs[0]);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_DOT_INL_H_
