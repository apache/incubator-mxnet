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
 * \file mkldnn_reshape-inl.h
 * \brief Function definition of mkldnn reshape operator
 */

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RESHAPE_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RESHAPE_INL_H_

#if MXNET_USE_MKLDNN == 100
#include <vector>
#include "mkldnn_base-inl.h"
#include "../../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

class MKLDNNReshapeFwd {
 protected:
  std::shared_ptr<mkldnn::memory> out_;
  std::shared_ptr<mkldnn::memory> temp_;
  std::vector<mkldnn_args_map_t> args_map_;
  std::vector<mkldnn::primitive> prims_;
  bool needInvalidateInput = false;

 public:
  MKLDNNReshapeFwd(const OpReqType &req,
                   const NDArray &input,
                   const NDArray &output);
  int GetWorkspaceSize();
  void Execute(const NDArray &input,
               const NDArray &output,
               void* workspace = nullptr);
};

typedef ParamOpSign<ReshapeParam> MKLDNNReshapeSignature;

template<typename MKLDNNOpFwdType, typename ParamType, typename MKLDNNSigatureType>
MKLDNNOpFwdType &GetCachedForward(const ParamType& param,
                                  const OpReqType &req,
                                  const NDArray &input,
                                  const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNSigatureType,
                                         MKLDNNOpFwdType, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNSigatureType,
                                            MKLDNNOpFwdType, OpHash> fwds;
#endif
  MKLDNNSigatureType key(param);
  key.AddSign(req);
  key.AddSign(input);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNOpFwdType fwd(req, input, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

MKLDNNReshapeFwd &GetReshapeForward(const ReshapeParam& param,
                                    const OpReqType &req,
                                    const NDArray &input,
                                    const NDArray &output);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_RESHAPE_INL_H_
