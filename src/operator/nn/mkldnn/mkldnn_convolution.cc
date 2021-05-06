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
 * \file mkldnn_convolution.cc
 * \brief
 * \author Da Zheng
 */

#if MXNET_USE_MKLDNN == 1

#include "../convolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_convolution-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(MKLDNNConvParam);

bool SupportMKLDNNConv(const ConvolutionParam& params, const NDArray &input) {
  if ((params.kernel.ndim() != 1) &&
      (params.kernel.ndim() != 2) &&
      (params.kernel.ndim() != 3))
    return false;
  return SupportMKLDNNQuantize(input.dtype()) &&
         ((input.shape().ndim() == 3) ||
          (input.shape().ndim() == 4) ||
          (input.shape().ndim() == 5));
}

std::shared_ptr<mkldnn::convolution_forward::primitive_desc> GetConvFwdImpl(
                                                           const MKLDNNConvFullParam &param,
                                                           const bool is_train,
                                                           const NDArray &data,
                                                           const NDArray &weights,
                                                           const NDArray *bias,
                                                           const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weights, param.conv_param.num_group, param.mkldnn_param.quantized);
  auto out_md = GetMemDesc(output);
  auto bias_md =
      bias ? (param.mkldnn_param.quantized ? GetMemDesc(*bias, mshadow::kInt32) : GetMemDesc(*bias))
           : mkldnn::memory::desc{
             {}, mkldnn::memory::data_type::undef, mkldnn::memory::format_tag::any};
  auto bias_md_ptr = bias ? &bias_md : nullptr;

  mkldnn::memory::dims strides(param.conv_param.kernel.ndim());
  mkldnn::memory::dims padding(param.conv_param.kernel.ndim());
  if (param.conv_param.kernel.ndim() == 1) {
    CHECK_GE(param.conv_param.stride.ndim(), 1);
    CHECK_GE(param.conv_param.pad.ndim(), 1);
    CHECK_GE(param.conv_param.dilate.ndim(), 1);
    strides[0] = param.conv_param.stride[0];
    padding[0] = param.conv_param.pad[0];
  } else if (param.conv_param.kernel.ndim() == 2) {
    CHECK_GE(param.conv_param.stride.ndim(), 2);
    CHECK_GE(param.conv_param.pad.ndim(), 2);
    CHECK_GE(param.conv_param.dilate.ndim(), 2);
    strides[0] = param.conv_param.stride[0];
    strides[1] = param.conv_param.stride[1];
    padding[0] = param.conv_param.pad[0];
    padding[1] = param.conv_param.pad[1];
  } else if (param.conv_param.kernel.ndim() == 3) {
    CHECK_GE(param.conv_param.stride.ndim(), 3);
    CHECK_GE(param.conv_param.pad.ndim(), 3);
    CHECK_GE(param.conv_param.dilate.ndim(), 3);
    strides[0] = param.conv_param.stride[0];
    strides[1] = param.conv_param.stride[1];
    strides[2] = param.conv_param.stride[2];
    padding[0] = param.conv_param.pad[0];
    padding[1] = param.conv_param.pad[1];
    padding[2] = param.conv_param.pad[2];
  } else {
    LOG(FATAL) << "Unexpected MKL-DNN Conv kernel size "
               << param.conv_param.kernel.ndim() << ", supporting only 1 or 2 or 3.";
  }
  mkldnn::primitive_attr attr;
  mkldnn::post_ops ops;
  if (param.mkldnn_param.with_act) {
    const auto &act_param = param.act_param;
    ops.append_eltwise(act_param.scale, act_param.alg, act_param.alpha, act_param.beta);
  }
  if (param.mkldnn_param.with_sum) {
    ops.append_sum(param.sum_scale);
  }
  if (param.mkldnn_param.with_postsum_act) {
    const auto &act_param = param.postsum_act_param;
    ops.append_eltwise(act_param.scale, act_param.alg, act_param.alpha, act_param.beta);
  }
  attr.set_post_ops(ops);

  if (param.mkldnn_param.quantized && param.requantize_scales.size()) {
    int mask = (param.requantize_scales.size() > 1) ? 2 : 0;
    attr.set_output_scales(mask, param.requantize_scales);
  }
  auto GetConvFwdPd = [&param, &data, &weights, &output,
                       &attr](const mkldnn::convolution_forward::desc &desc) {
    auto engine = CpuEngine::Get()->get_engine();
    try {
      // MKL-DNN introduced padded formats since 0.15 which require more memory
      // compared to the actual size of the tensor. Currently, MKL-DNN operators
      // still reuse memory from memory planning, so here we need to select a
      // suboptimal kernel for computation that has the expected memory size requirements
      auto conv_pd =
          std::make_shared<mkldnn::convolution_forward::primitive_desc>(desc, attr, engine);
      while (conv_pd->dst_desc().get_size() != GetArraySize(output) ||
             conv_pd->src_desc().get_size() != GetArraySize(data) ||
             (!param.mkldnn_param.quantized &&
              conv_pd->weights_desc().get_size() != GetArraySize(weights))) {
        // next_impl() will visit desc and engine, please make sure they are still alive here.
        CHECK(conv_pd->next_impl()) << "No convolution implementation for this request.";
      }
      return conv_pd;
    } catch (mkldnn::error &e) {
      if (e.status == mkldnn_unimplemented && param.mkldnn_param.quantized) {
        LOG(ERROR) << "AVX512-BW support or Intel(R) MKL dependency is "
                      "required for int8 convolution";
      } else {
        LOG(ERROR) << e.message;
      }
      throw;
    }
  };

  if (param.conv_param.dilate.ndim() == 0 && bias_md_ptr == nullptr) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct, data_md,
                                           weight_md, out_md, strides, padding, padding);
    return GetConvFwdPd(desc);
  } else if (param.conv_param.dilate.ndim() == 0) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct, data_md,
                                           weight_md, *bias_md_ptr, out_md, strides, padding,
                                           padding);
    return GetConvFwdPd(desc);
  } else {
    mkldnn::memory::dims dilates(param.conv_param.kernel.ndim());
    if (param.conv_param.dilate.ndim() == 1) {
      dilates[0] = param.conv_param.dilate[0] - 1;
    } else if (param.conv_param.dilate.ndim() == 2) {
      dilates[0] = param.conv_param.dilate[0] - 1;
      dilates[1] = param.conv_param.dilate[1] - 1;
    } else if (param.conv_param.dilate.ndim() == 3) {
      dilates[0] = param.conv_param.dilate[0] - 1;
      dilates[1] = param.conv_param.dilate[1] - 1;
      dilates[2] = param.conv_param.dilate[2] - 1;
    } else {
      LOG(FATAL) << "Unexpected MKL-DNN Conv dilate size " << param.conv_param.dilate.ndim()
                 << ", supporting only 1 or 2 or 3.";
    }
    if (bias_md_ptr == nullptr) {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct, data_md,
                                             weight_md, out_md, strides, dilates, padding, padding);
      return GetConvFwdPd(desc);
    } else {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct, data_md,
                                             weight_md, *bias_md_ptr, out_md, strides, dilates,
                                             padding, padding);
      return GetConvFwdPd(desc);
    }
  }
}

static std::shared_ptr<mkldnn::convolution_backward_data::primitive_desc> GetConvBwdData(
    const ConvolutionParam &param, const NDArray &data, const NDArray &weight,
    const NDArray &output, const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weight, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides(param.kernel.ndim());
  mkldnn::memory::dims padding(param.kernel.ndim());
  if (param.kernel.ndim() == 1) {
    CHECK_GE(param.stride.ndim(), 1);
    CHECK_GE(param.pad.ndim(), 1);
    CHECK_GE(param.dilate.ndim(), 1);
    strides[0] = param.stride[0];
    padding[0] = param.pad[0];
  } else if (param.kernel.ndim() == 2) {
    CHECK_GE(param.stride.ndim(), 2);
    CHECK_GE(param.pad.ndim(), 2);
    CHECK_GE(param.dilate.ndim(), 2);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.kernel.ndim() == 3) {
    CHECK_GE(param.stride.ndim(), 3);
    CHECK_GE(param.pad.ndim(), 3);
    CHECK_GE(param.dilate.ndim(), 3);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    strides[2] = param.stride[2];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
    padding[2] = param.pad[2];
  } else {
    LOG(FATAL) << "Unexpected MKL-DNN Conv kernel size " << param.kernel.ndim()
               << ", supporting only 1 or 2 or 3.";
  }

  auto GetConvBwdDataPd = [&data, &weight, &output,
                           &fwd_pd](const mkldnn::convolution_backward_data::desc &desc) {
    auto engine = CpuEngine::Get()->get_engine();
    try {
      // MKL-DNN introduced padded formats since 0.15 which require more memory
      // compared to the actual size of the tensor. Currently, MKL-DNN operators
      // still reuse memory from memory planning, so here we need to select a
      // suboptimal kernel for computation that has the expected memory size requirements
      auto conv_pd =
          std::make_shared<mkldnn::convolution_backward_data::primitive_desc>(desc, engine, fwd_pd);
      while (conv_pd->diff_dst_desc().get_size() != GetArraySize(output) ||
             conv_pd->diff_src_desc().get_size() != GetArraySize(data) ||
             conv_pd->weights_desc().get_size() != GetArraySize(weight)) {
        // next_impl() will visit desc and engine, please make sure they are still alive here.
        CHECK(conv_pd->next_impl()) << "No convolution backward implementation for this request.";
      }
      return conv_pd;
    } catch (mkldnn::error &e) {
      LOG(ERROR) << e.message;
      throw;
    }
  };

  if (param.dilate.ndim() == 0) {
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct, data_md,
                                                 weight_md, out_md, strides, padding, padding);
    return GetConvBwdDataPd(desc);
  } else {
    mkldnn::memory::dims dilates(param.kernel.ndim());
    if (param.dilate.ndim() == 1) {
      dilates[0] = param.dilate[0] - 1;
    } else if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    } else if (param.dilate.ndim() == 3) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
      dilates[2] = param.dilate[2] - 1;
    } else {
      LOG(FATAL) << "Unexpected MKL-DNN Conv dilate size "
                 << param.dilate.ndim() << ", supporting only 1 or 2 or 3.";
    }
    mkldnn::convolution_backward_data::desc desc(mkldnn::algorithm::convolution_direct, data_md,
                                                 weight_md, out_md, strides, dilates, padding,
                                                 padding);
    return GetConvBwdDataPd(desc);
  }
}

static std::shared_ptr<mkldnn::convolution_backward_weights::primitive_desc> GetConvBwdWeights(
    const ConvolutionParam &param, const NDArray &data, const NDArray &weight, const NDArray *bias,
    const NDArray &output, const mkldnn::convolution_forward::primitive_desc &fwd_pd) {
  auto data_md = GetMemDesc(data);
  auto weight_md = GetWeightDesc(weight, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides(param.kernel.ndim());
  mkldnn::memory::dims padding(param.kernel.ndim());
  if (param.kernel.ndim() == 1) {
    CHECK_GE(param.stride.ndim(), 1);
    CHECK_GE(param.pad.ndim(), 1);
    CHECK_GE(param.dilate.ndim(), 1);
    strides[0] = param.stride[0];
    padding[0] = param.pad[0];
  } else if (param.kernel.ndim() == 2) {
    CHECK_GE(param.stride.ndim(), 2);
    CHECK_GE(param.pad.ndim(), 2);
    CHECK_GE(param.dilate.ndim(), 2);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  } else if (param.kernel.ndim() == 3) {
    CHECK_GE(param.stride.ndim(), 3);
    CHECK_GE(param.pad.ndim(), 3);
    CHECK_GE(param.dilate.ndim(), 3);
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    strides[2] = param.stride[2];
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
    padding[2] = param.pad[2];
  } else {
    LOG(FATAL) << "Unexpected MKL-DNN Conv kernel size " << param.kernel.ndim()
               << ", supporting only 1 or 2 or 3.";
  }

  auto GetConvBwdWeightsPd = [&data, &weight, &output,
                              &fwd_pd](const mkldnn::convolution_backward_weights::desc &desc) {
    auto engine = CpuEngine::Get()->get_engine();
    try {
      // MKL-DNN introduced padded formats since 0.15 which require more memory
      // compared to the actual size of the tensor. Currently, MKL-DNN operators
      // still reuse memory from memory planning, so here we need to select a
      // suboptimal kernel for computation that has the expected memory size requirements
      auto conv_pd = std::make_shared<mkldnn::convolution_backward_weights::primitive_desc>(
          desc, engine, fwd_pd);
      while (conv_pd->diff_dst_desc().get_size() != GetArraySize(output) ||
             conv_pd->src_desc().get_size() != GetArraySize(data) ||
             conv_pd->diff_weights_desc().get_size() != GetArraySize(weight)) {
        // next_impl() will visit desc and engine, please make sure they are still alive here.
        CHECK(conv_pd->next_impl()) << "No convolution backward implementation for this request.";
      }
      return conv_pd;
    } catch (mkldnn::error &e) {
      LOG(ERROR) << e.message;
      throw;
    }
  };

  if (param.dilate.ndim() == 0 && bias == nullptr) {
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct, data_md,
                                                    weight_md, out_md, strides, padding, padding);
    return GetConvBwdWeightsPd(desc);
  } else if (param.dilate.ndim() == 0) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct, data_md,
                                                    weight_md, bias_md, out_md, strides, padding,
                                                    padding);
    return GetConvBwdWeightsPd(desc);
  } else {
    mkldnn::memory::dims dilates(param.kernel.ndim());
    if (param.dilate.ndim() == 1) {
      dilates[0] = param.dilate[0] - 1;
    } else if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    } else if (param.dilate.ndim() == 3) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
      dilates[2] = param.dilate[2] - 1;
    } else {
      LOG(FATAL) << "Unexpected MKL-DNN Conv dilate size "
                 << param.dilate.ndim() << ", supporting only 1 or 2 or 3.";
    }
    if (bias == nullptr) {
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
                                                      data_md, weight_md, out_md, strides, dilates,
                                                      padding, padding);
      return GetConvBwdWeightsPd(desc);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_backward_weights::desc desc(mkldnn::algorithm::convolution_direct,
                                                      data_md, weight_md, bias_md, out_md, strides,
                                                      dilates, padding, padding);
      return GetConvBwdWeightsPd(desc);
    }
  }
}

MKLDNNConvForward::MKLDNNConvForward(const MKLDNNConvFullParam &param, const bool is_train,
                                     const NDArray &data, const NDArray &weight,
                                     const NDArray *bias, const NDArray &output)
    : pd_(GetConvFwdImpl(param, is_train, data, weight, bias, output)) {
  fwd_ = std::make_shared<mkldnn::convolution_forward>(GetPd());
}

MKLDNNConvForward &GetConvFwd(const MKLDNNConvFullParam &param, const bool is_train,
                              const NDArray &data, const NDArray &weight, const NDArray *bias,
                              const NDArray &output) {
  using conv_fwd_map = std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local conv_fwd_map fwds;
#else
  static MX_THREAD_LOCAL conv_fwd_map fwds;
#endif
  // TODO(zhennan): Hash conv_param for now, need to hash full param if we want to enable cache for
  // fused conv
  MKLDNNConvSignature key(param.conv_param);
  key.AddSign(is_train);
  // Here we can sign the conv op with NDArray because conv primitive will decide the right layout
  // for the, so we only need to get the shape and the data type of the arrays.
  key.AddSign(data);
  key.AddSign(weight);
  key.AddSign(output);
  if (bias) key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    auto fwd = MKLDNNConvForward(param, is_train, data, weight, bias, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNConvolutionForwardFullFeature(const MKLDNNConvFullParam &param, const OpContext &ctx,
                                         MKLDNNConvForward *fwd,
                                         const std::vector<NDArray> &in_data,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<NDArray> &out_data) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);

  auto &data = in_data[conv::kData];
  auto &weight = in_data[conv::kWeight];
  bool no_bias = param.conv_param.no_bias && !param.mkldnn_param.with_bn;

  auto data_mem = data.GetMKLDNNDataReorder(fwd->GetPd().src_desc());
  const mkldnn::memory *weight_mem;
  if (ctx.is_train) {
    // TODO(zhengda) kvstore doesn't handle MKLDNN correctly. Let's reorder it to the default format
    // for now.
    if (weight.IsMKLDNNData())
      // This asks the engine to change the layout of the weight array after it's used.
      weight.Reorder2DefaultAsync();
    weight_mem = GetWeights(weight, fwd->GetPd().weights_desc(), param.conv_param.num_group);
  } else {
    // For inference, we want to reorder the weight array so we don't need to reorder data every
    // time.
    if (weight.IsDefaultData()) {
      // We also need to modify the layout on the original weight array. The data conversion happens
      // after the weight array is used.
      weight.MKLDNNDataReorderAsync(fwd->GetPd().weights_desc());
      weight_mem = GetWeights(weight, fwd->GetPd().weights_desc(), param.conv_param.num_group);
    } else {
      weight_mem = weight.GetMKLDNNData();
    }
  }
  mkldnn_output_t out_mem;
  if (param.mkldnn_param.with_sum) {
    out_mem = mkldnn_output_t(OutDataOp::Noop,
                              const_cast<mkldnn::memory *>(out_data[conv::kOut].GetMKLDNNData()));
  } else {
    out_mem = CreateMKLDNNMem(out_data[conv::kOut], fwd->GetPd().dst_desc(), req[conv::kOut]);
  }

  mkldnn_args_map_t net_args;
  if (!no_bias) {
    const mkldnn::memory *bias_mem = in_data[conv::kBias].GetMKLDNNData();
    net_args.insert({MKLDNN_ARG_BIAS, *bias_mem});
  }

  net_args.insert({MKLDNN_ARG_SRC, *data_mem});
  net_args.insert({MKLDNN_ARG_WEIGHTS, *weight_mem});
  net_args.insert({MKLDNN_ARG_DST, *out_mem.second});
  MKLDNNStream::Get()->RegisterPrimArgs(fwd->GetFwd(), net_args);
  CommitOutput(out_data[conv::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
}

void MKLDNNConvolutionForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                              const std::vector<NDArray> &in_data,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &out_data) {
  MKLDNNConvFullParam param;
  param.conv_param = nnvm::get<ConvolutionParam>(attrs.parsed);
  param.mkldnn_param.Init(std::unordered_map<std::string, std::string>());
  auto &fwd =
      GetConvFwd(param, ctx.is_train, in_data[conv::kData], in_data[conv::kWeight],
                 param.conv_param.no_bias ? nullptr : &in_data[conv::kBias], out_data[conv::kOut]);
  MKLDNNConvolutionForwardFullFeature(param, ctx, &fwd, in_data, req, out_data);
}

MKLDNNConvBackward::MKLDNNConvBackward(const MKLDNNConvFullParam &param, const NDArray &data,
                                       const NDArray &weight, const NDArray *bias,
                                       const NDArray &output) {
  const auto fwd_pd = GetConvFwdImpl(param, true, data, weight, bias, output);
  bwd_data_pd_ = GetConvBwdData(param.conv_param, data, weight, output, *fwd_pd);
  bwd_weight_pd_ = GetConvBwdWeights(param.conv_param, data, weight, bias, output, *fwd_pd);
  bwd_data_ = std::make_shared<mkldnn::convolution_backward_data>(GetDataPd());
  bwd_weight_ = std::make_shared<mkldnn::convolution_backward_weights>(GetWeightsPd());
}

static inline MKLDNNConvBackward &GetConvBwd(const MKLDNNConvFullParam &param, const NDArray &data,
                                             const NDArray &weight, const NDArray *bias,
                                             const NDArray &output) {
  using mkldnn_conv_bwd_map = std::unordered_map<MKLDNNConvSignature, MKLDNNConvBackward, OpHash>;
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local mkldnn_conv_bwd_map bwds;
#else
  static MX_THREAD_LOCAL mkldnn_conv_bwd_map bwds;
#endif
  // TODO(zhennan): Hash conv_param for now, need to hash full param if we want to enable cache for
  // fused conv
  MKLDNNConvSignature key(param.conv_param);
  // Here we can sign the conv op with NDArray because conv primitive will decide the right layout
  // for the, so we only need to get the shape and the data type of the arrays.
  key.AddSign(data);
  key.AddSign(weight);
  key.AddSign(output);
  if (bias) key.AddSign(*bias);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto bwd = MKLDNNConvBackward(param, data, weight, bias, output);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNConvolutionBackward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const std::vector<NDArray> &in_grad = outputs;
  MKLDNNConvFullParam full_param;
  full_param.conv_param = nnvm::get<ConvolutionParam>(attrs.parsed);
  full_param.mkldnn_param.Init(std::unordered_map<std::string, std::string>());

  auto &data = inputs[conv::kData + 1];
  auto &weight = inputs[conv::kWeight + 1];
  const auto *bias = full_param.conv_param.no_bias ? nullptr : &inputs[conv::kBias + 1];
  auto &out_grad = inputs[conv::kOut];

  const ConvolutionParam &param = full_param.conv_param;

  CHECK_NE(req[conv::kWeight], kWriteInplace) << "cannot write weight inplace";
  MKLDNNConvBackward &convBwd = GetConvBwd(full_param, data, weight, bias, out_grad);
  auto out_grad_mem = out_grad.GetMKLDNNDataReorder(convBwd.GetDataPd().diff_dst_desc());
  if (req[conv::kData]) {
    auto weight_mem = GetWeights(weight, convBwd.GetDataPd().weights_desc(), param.num_group);
    auto in_grad_mem = CreateMKLDNNMem(in_grad[conv::kData], convBwd.GetDataPd().diff_src_desc(),
                                       req[conv::kData]);
    MKLDNNStream::Get()->RegisterPrimArgs(convBwd.GetBwdData(),
                                          {{MKLDNN_ARG_DIFF_DST, *out_grad_mem},
                                           {MKLDNN_ARG_WEIGHTS, *weight_mem},
                                           {MKLDNN_ARG_DIFF_SRC, *in_grad_mem.second}});
    CommitOutput(in_grad[conv::kData], in_grad_mem);
  }
  if (req[conv::kWeight] || req[conv::kBias]) {
    if (convBwd.GetDataPd().diff_dst_desc() != convBwd.GetWeightsPd().diff_dst_desc())
      out_grad_mem = out_grad.GetMKLDNNDataReorder(convBwd.GetWeightsPd().diff_dst_desc());
    auto data_mem = data.GetMKLDNNDataReorder(convBwd.GetWeightsPd().src_desc());
    auto in_grad_weight = CreateMKLDNNWeightGrad(
        in_grad[conv::kWeight], convBwd.GetWeightsPd().diff_weights_desc(), req[conv::kWeight]);

    mkldnn_args_map_t net_args = {{MKLDNN_ARG_DIFF_DST, *out_grad_mem},
                                  {MKLDNN_ARG_SRC, *data_mem},
                                  {MKLDNN_ARG_DIFF_WEIGHTS, *in_grad_weight.second}};
    mkldnn_output_t in_grad_bias;
    if (!param.no_bias) {
      in_grad_bias = CreateMKLDNNMem(in_grad[conv::kBias],
                                          convBwd.GetWeightsPd().diff_bias_desc(),
                                          req[conv::kBias]);
      net_args.insert({MKLDNN_ARG_DIFF_BIAS, *in_grad_bias.second});
    }
    MKLDNNStream::Get()->RegisterPrimArgs(convBwd.GetBwdWeights(), net_args);
    CommitOutput(in_grad[conv::kWeight], in_grad_weight);
    // CommitOutput Should run after RegisterPrimArgs for memory dependency
    if (!param.no_bias) {
      CommitOutput(in_grad[conv::kBias], in_grad_bias);
    }
  }
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
