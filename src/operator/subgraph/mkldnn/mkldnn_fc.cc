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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_fc.cc
 * \brief MKLDNN (Quantized) FullyConnected operator based on subgraph
 * \author Ciyong Chen
*/

#if MXNET_USE_ONEDNN == 1

#include <utility>
#include <vector>
#include <string>
#include "../common.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_fully_connected-inl.h"
#include "../../nn/mkldnn/mkldnn_act-inl.h"
#include "../../tensor/matrix_op-inl.h"
#include "../../quantization/quantization_utils.h"
#include "mkldnn_fc-inl.h"
#include "mkldnn_common.h"

namespace mxnet {
namespace op {

class SgMKLDNNFCOp {
 public:
  explicit SgMKLDNNFCOp(const nnvm::NodeAttrs &attrs)
    : subgraph_sym_(*attrs.subgraphs[0]),
      full_param_(nnvm::get<MKLDNNFCFullParam>(attrs.parsed)) {}

  void Forward(const OpContext &ctx,
               const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

  void Backward(const OpContext &ctx,
                const std::vector<NDArray> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<NDArray> &outputs) {
    LOG(FATAL) << "Not implemented: subgraph mkldnn fully connected only supports "
                  "inference computation.";
  }

 private:
  bool initialized_{false};
  bool channel_wise_runtime_{false};
  bool reorder_data_{false};
  nnvm::Symbol subgraph_sym_;
  MKLDNNFCFullParam full_param_;
  mkldnn_args_map_t args_;
  std::shared_ptr<MKLDNNFullyConnectedForward> fwd_;
  std::shared_ptr<mkldnn::memory> cached_data_mem_;
  std::shared_ptr<mkldnn::memory> cached_out_mem_;
  NDArray cached_weight_;
  NDArray cached_bias_;
  float cached_min_data_;
  float cached_max_data_;
  float cached_min_weight_;
  float cached_max_weight_;
  float cached_min_bias_;
  float cached_max_bias_;
  size_t weight_ver_;
  size_t bias_ver_;
  float cached_min_output_;
  float cached_max_output_;
  float data_scale_{0.0f};
  std::vector<float> weight_scales_;
  size_t total_num_inputs_;
  size_t total_num_outputs_;
};

void SgMKLDNNFCOp::Forward(const OpContext &ctx,
                           const std::vector<NDArray> &in_data,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &out_data) {
  auto &mkldnn_param = full_param_.mkldnn_param;
  auto &default_param = full_param_.default_param;
  bool has_bias = !default_param.no_bias;
  size_t base_num_inputs = has_bias ? 3 : 2;
  size_t base_num_outputs = 1;

  float min_data = 0.0f;
  float max_data = 0.0f;
  float min_weight = 0.0f;
  float max_weight = 0.0f;
  float min_bias = 0.0f;
  float max_bias = 0.0f;

  if (!initialized_) {
    if (mkldnn_param.channel_wise_quantize.has_value() &&
        mkldnn_param.channel_wise_quantize) {
      channel_wise_runtime_ = true;
    }

    total_num_inputs_ = base_num_inputs;
    total_num_outputs_ = base_num_outputs;
    if (mkldnn_param.quantized) {
      total_num_inputs_ = channel_wise_runtime_ ? (base_num_inputs + 2) : (base_num_inputs * 3);
      total_num_outputs_ =
        mkldnn_param.enable_float_output ? base_num_outputs : (base_num_outputs * 3);
    }
  }
  CHECK_EQ(in_data.size(), total_num_inputs_);
  CHECK_EQ(out_data.size(), total_num_outputs_);

  NDArray data = in_data[fullc::kData];
  const NDArray &weight = in_data[fullc::kWeight];
  const NDArray &output = out_data[fullc::kOut];

  if (mkldnn_param.quantized) {
    if (!channel_wise_runtime_) {
      min_weight = in_data[base_num_inputs + quantized_fullc::kWeightMin].data().dptr<float>()[0];
      max_weight = in_data[base_num_inputs + quantized_fullc::kWeightMax].data().dptr<float>()[0];
      if (has_bias) {
        min_bias = in_data[base_num_inputs + quantized_fullc::kBiasMin].data().dptr<float>()[0];
        max_bias = in_data[base_num_inputs + quantized_fullc::kBiasMax].data().dptr<float>()[0];
      }
    }
    min_data = in_data[base_num_inputs + quantized_fullc::kDataMin].data().dptr<float>()[0];
    max_data = in_data[base_num_inputs + quantized_fullc::kDataMax].data().dptr<float>()[0];
  }

  if (initialized_ && mkldnn_param.quantized &&
      dmlc::GetEnv("MXNET_ONEDNN_QFC_DYNAMIC_PARAMS", 0)) {
    if (channel_wise_runtime_) {
      if (cached_min_data_ != min_data || cached_max_data_ != max_data ||
          weight_ver_ != weight.version() ||
          (has_bias && (bias_ver_ != in_data[fullc::kBias].version()))) {
        initialized_ = false;
      }
    } else {
      if (cached_min_data_ != min_data || cached_max_data_ != max_data ||
          cached_min_weight_ != min_weight || cached_max_weight_ != max_weight ||
          (has_bias && (cached_min_bias_ != min_bias || cached_max_bias_ != max_bias))) {
        initialized_ = false;
      }
    }
  }

  if (!initialized_) {
    const auto nthreads = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    const auto engine = CpuEngine::Get()->get_engine();
    cached_min_data_ = min_data;
    cached_max_data_ = max_data;
    cached_min_weight_ = min_weight;
    cached_max_weight_ = max_weight;
    weight_ver_ = weight.version();
    cached_weight_ = weight;
    if (has_bias) {
      cached_min_bias_ = min_bias;
      cached_max_bias_ = max_bias;
      bias_ver_ = in_data[fullc::kBias].version();
      cached_bias_ = in_data[fullc::kBias];
    } else {
      cached_bias_ = NDArray();
    }
    const mxnet::TShape ishape = data.shape();
    const auto data_ndim = ishape.ndim();
    if (data.IsMKLDNNData()) {
      reorder_data_ = true;
      data = data.Reorder2Default();
    }
    if (data_ndim != 2) {
      if (!default_param.flatten) {
        data = data.MKLDNNDataReshape(
            Shape2(ishape.ProdShape(0, data_ndim - 1), ishape[data_ndim - 1]));
      } else {
        data = data.MKLDNNDataReshape(Shape2(ishape[0], ishape.ProdShape(1, data_ndim)));
      }
    }

    // create cached out_md
    const mxnet::TShape oshape = output.shape();
    mkldnn::memory::dims out_dims(2);
    if (oshape.ndim() == 2) {
      out_dims[0] = static_cast<int>(oshape[0]);
      out_dims[1] = static_cast<int>(oshape[1]);
    } else {
      if (!default_param.flatten) {
        out_dims[0] = static_cast<int>(oshape.ProdShape(0, oshape.ndim()-1));
        out_dims[1] = static_cast<int>(oshape[oshape.ndim()-1]);
      } else {
        out_dims[0] = static_cast<int>(static_cast<int>(oshape[0]));
        out_dims[1] = static_cast<int>(oshape.ProdShape(1, oshape.ndim()));
      }
    }
    mkldnn::memory::desc out_md = mkldnn::memory::desc(out_dims, get_mkldnn_type(output.dtype()),
      static_cast<mkldnn::memory::format_tag>(GetDefaultFormat(2)));
    cached_out_mem_ = std::make_shared<mkldnn::memory>(out_md, engine);

    bool support_channelwise_scale = false;
    if (mkldnn_param.quantized) {
      CHECK(data.dtype() == mshadow::kInt8 || data.dtype() == mshadow::kUint8);
      data_scale_ = GetQuantizeScale(data.dtype(), cached_min_data_, cached_max_data_);

      bool fuse_requantize = false;
      // Channelwise scaling is only supported when fusion is enabled (requantize or dequantize).
      if (mkldnn_param.min_calib_range.has_value() &&
          mkldnn_param.max_calib_range.has_value()) {
        cached_min_output_ = mkldnn_param.min_calib_range.value();
        cached_max_output_ = mkldnn_param.max_calib_range.value();
        support_channelwise_scale = true;
        fuse_requantize = true;
      }
      if (mkldnn_param.enable_float_output) {
        support_channelwise_scale = true;
      }
      // channel_wise  support_channelwise_scale  result
      // True          True                       True
      // True          False                      Error
      // False         True/False                 False
      if (channel_wise_runtime_ && !support_channelwise_scale) {
        LOG(FATAL)
          << "Currently, channel-wise quantization requires fuse requantize or dequantize."
          << " Please make sure the `min_calib_range` and `max_calib_range` are set when only"
          << " fuse requantize (outputs of FullyConnected are collected during calibration phase),"
          << " or the env var of `MXNET_DISABLE_ONEDNN_QFC_FLOAT_OUTPUT` and "
          << " `MXNET_DISABLE_ONEDNN_QFC_FUSE_ALL` are not set to true (default is false)";
      }
      support_channelwise_scale = support_channelwise_scale && channel_wise_runtime_;

      if (support_channelwise_scale) {
        MSHADOW_REAL_TYPE_SWITCH(cached_weight_.dtype(), DType, {
          weight_scales_ =
            GetWeightScales<DType>(cached_weight_, has_bias ? &cached_bias_ : nullptr,
                                   data_scale_, support_channelwise_scale);
        });
      } else {
        weight_scales_.resize(1);
        weight_scales_[0] =
          GetQuantizeScale(cached_weight_.dtype(), cached_min_weight_, cached_max_weight_);
        if (has_bias) {
          float bias_scale = GetQuantizeScale(mshadow::kInt8, cached_min_bias_, cached_max_bias_);
          float bias_int32_rescale = data_scale_ * weight_scales_[0] / bias_scale;
          // TODO(zhennan): mkldnn has bug to handle INT_MAX in bias, so set the maximum value
          // of bias to INT_MAX / 2.
          float bias_max_rescale =
              MaxValue<int32_t>() / 2 / MaxAbs(cached_min_bias_, cached_max_bias_) / bias_scale;
          if (bias_int32_rescale > bias_max_rescale) {
            // avoid overflow on bias
            bias_int32_rescale = bias_max_rescale;
            float weight_rescale =
              bias_int32_rescale * bias_scale / data_scale_ / weight_scales_[0];
            int8_t *weight_ptr = weight.data().dptr<int8_t>();
            size_t weight_size = weight.shape().Size();
            #pragma omp parallel for num_threads(nthreads)
            for (index_t i = 0; i < static_cast<index_t>(weight_size); ++i) {
              weight_ptr[i] = std::round(weight_ptr[i] * weight_rescale);
            }
            weight_scales_[0] *= weight_rescale;
          }
          NDArray bias = in_data[fullc::kBias];
          cached_bias_ =
              NDArray(bias.storage_type(), bias.shape(), bias.ctx(), true, mshadow::kInt32);
          int8_t *bias_ptr = bias.data().dptr<int8_t>();
          int32_t *quantized_bias_ptr = cached_bias_.data().dptr<int32_t>();
          size_t bias_size = bias.shape().Size();
          #pragma omp parallel for num_threads(nthreads)
          for (index_t i = 0; i < static_cast<index_t>(bias_size); ++i) {
            quantized_bias_ptr[i] = std::round(bias_ptr[i] * bias_int32_rescale);
          }
        }
      }

      size_t num_channel = cached_weight_.shape()[0];
      if (fuse_requantize || mkldnn_param.enable_float_output) {
        float tmp_scale_ = 1.0f;
        if (fuse_requantize) {
          if (mkldnn_param.with_eltwise) {
            tmp_scale_ = 1.0 / data_scale_;
            full_param_.eltwise_param.scale =
              GetQuantizeScale(output.dtype(), cached_min_output_, cached_max_output_);
          } else {
            tmp_scale_ =
              GetQuantizeScale(output.dtype(),
                               cached_min_output_,
                               cached_max_output_) / data_scale_;
          }
        } else {
          tmp_scale_ = 1.0 / data_scale_;
        }

        if (support_channelwise_scale) {
          full_param_.output_scales.resize(num_channel);
          #pragma omp parallel for num_threads(nthreads)
          for (index_t i = 0; i < static_cast<index_t>(num_channel); ++i) {
            full_param_.output_scales[i] = tmp_scale_ / weight_scales_[i];
          }
        } else {
          full_param_.output_scales.resize(1);
          full_param_.output_scales[0] = tmp_scale_ / weight_scales_[0];
        }
      } else {
        Stream<cpu> *s = ctx.get_stream<cpu>();
        if (data.dtype() == mshadow::kInt8) {
          mxnet_op::Kernel<QuantizationRangeForS8S8MultiplicationStruct, cpu>::Launch(
              s, 1, &cached_min_output_, &cached_max_output_, &min_data, &max_data, &min_weight,
              &max_weight);
        } else {
          mxnet_op::Kernel<QuantizationRangeForS8U8MultiplicationStruct, cpu>::Launch(
              s, 1, &cached_min_output_, &cached_max_output_, &min_data, &max_data, &min_weight,
              &max_weight);
        }
        full_param_.output_scales.resize(0);
      }
    }

    fwd_.reset(new MKLDNNFullyConnectedForward(full_param_, ctx.is_train, data, cached_weight_,
      (has_bias ? &cached_bias_ : nullptr), out_md));

    // convert weight and bias to the format that MKL-DNN requires
    if (!mkldnn_param.quantized || support_channelwise_scale) {
      mkldnn::memory::desc bias_md;
      if (has_bias) bias_md = fwd_->fwd_pd.bias_desc();
      ConvertWeightBias2MKLDNN(&cached_weight_, &cached_bias_, has_bias,
                              fwd_->fwd_pd.weights_desc(),
                              has_bias ? &bias_md : nullptr,
                              1, data_scale_, weight_scales_, false);
    } else {
      const auto def_weight_mem = weight.GetMKLDNNData();
      if (def_weight_mem->get_desc() != fwd_->fwd_pd.weights_desc()) {
        cached_weight_ = NDArray(fwd_->fwd_pd.weights_desc());
        auto cached_weight_mem = cached_weight_.GetMKLDNNData();
        std::unordered_map<int, mkldnn::memory> args(
          {{MKLDNN_ARG_FROM, *def_weight_mem},
          {MKLDNN_ARG_TO, *cached_weight_mem}});
        MKLDNNStream::Get()->RegisterPrimArgs(
          mkldnn::reorder(*def_weight_mem, *cached_weight_mem), args);
      }
    }

    const auto data_mem = data.GetMKLDNNData();
    cached_data_mem_ = std::make_shared<mkldnn::memory>(data_mem->get_desc(), engine);

    args_[MKLDNN_ARG_SRC] = *cached_data_mem_;
    args_[MKLDNN_ARG_WEIGHTS] = *cached_weight_.GetMKLDNNData();
    if (has_bias)
      args_[MKLDNN_ARG_BIAS] = *cached_bias_.GetMKLDNNData();
    args_[MKLDNN_ARG_DST] = *cached_out_mem_;
    initialized_ = true;
  }

  if (reorder_data_) {
    data = data.Reorder2Default();
  }
  MSHADOW_TYPE_SWITCH(data.dtype(), DType, {
    cached_data_mem_->set_data_handle(reinterpret_cast<void *>(data.data().dptr<DType>()));
  });
  MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
    cached_out_mem_->set_data_handle(reinterpret_cast<void *>(output.data().dptr<DType>()));
  });
  MKLDNNStream::Get()->RegisterPrimArgs(fwd_->GetFwd(), args_);
  MKLDNNStream::Get()->Submit();

  if (mkldnn_param.quantized && !mkldnn_param.enable_float_output) {
    float *min_output_ptr = out_data[quantized_fullc::kOutMin].data().dptr<float>();
    float *max_output_ptr = out_data[quantized_fullc::kOutMax].data().dptr<float>();
    *min_output_ptr = cached_min_output_;
    *max_output_ptr = cached_max_output_;
  }
}

static void SgMKLDNNFCParamParser(nnvm::NodeAttrs *attrs) {
  // For backward compatible, with_relu->with_eltwise
  auto legacy = attrs->dict.find("with_relu");
  if (legacy != attrs->dict.end()) {
    attrs->dict["with_eltwise"] = attrs->dict["with_relu"];
    attrs->dict.erase(legacy);
  }

  MKLDNNFCFullParam full_param;
  try {
    full_param.mkldnn_param.Init(attrs->dict);
  } catch (const dmlc::ParamError &e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto &k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  auto subgraph_sym = attrs->subgraphs[0];
  DFSVisit(subgraph_sym->outputs, [&](const nnvm::ObjectPtr &node) {
    if (node->is_variable()) return;
    auto &op_name = node->op()->name;
    if (op_name == "FullyConnected") {
      full_param.default_param =
          nnvm::get<FullyConnectedParam>(node->attrs.parsed);
    } else if (SupportMKLDNNFCEltwiseFusion(op_name)) {
      if (op_name == "Activation") {
        const ActivationParam act_param = nnvm::get<ActivationParam>(node->attrs.parsed);
        full_param.eltwise_param.alg = GetMKLDNNActAlgo(act_param);
      } else if (op_name == "LeakyReLU") {
        const auto act_param = nnvm::get<LeakyReLUParam>(node->attrs.parsed);
        full_param.eltwise_param.alpha = act_param.slope;
        full_param.eltwise_param.alg = GetMKLDNNActAlgo(act_param);
      } else if (op_name == "clip") {
        const ClipParam clip_param = nnvm::get<ClipParam>(node->attrs.parsed);
        full_param.eltwise_param.alg = mkldnn::algorithm::eltwise_bounded_relu;
        full_param.eltwise_param.alpha = clip_param.a_max;
      } else {
        full_param.eltwise_param.alg = GetMKLDNNEltwiseAlgo(op_name);
      }
    }
  });
  attrs->parsed = std::move(full_param);
}

static std::vector<std::string> SgMKLDNNFCListInputNames(const NodeAttrs &attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  std::vector<std::string> input_names = DefaultSubgraphOpListInputs(attrs);
  if (full_param.mkldnn_param.quantized) {
    bool channel_wise = false;
    if (full_param.mkldnn_param.channel_wise_quantize.has_value() &&
        full_param.mkldnn_param.channel_wise_quantize) {
      channel_wise = true;
    }
    input_names.emplace_back("min_data");
    input_names.emplace_back("max_data");
    if (!channel_wise) {
      input_names.emplace_back("min_weight");
      input_names.emplace_back("max_weight");
      if (!full_param.default_param.no_bias) {
        input_names.emplace_back("min_bias");
        input_names.emplace_back("max_bias");
      }
    }
  }
  return input_names;
}

static std::vector<std::string> SgMKLDNNFCListOutputNames(const NodeAttrs &attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    if (full_param.mkldnn_param.enable_float_output)
      return std::vector<std::string>{"output"};
    else
      return std::vector<std::string>{"output", "min_output", "max_output"};
  } else {
    return std::vector<std::string>{"output"};
  }
}

template <typename T>
static inline void FillBaseInputOutputInfo(const FullyConnectedParam &param,
                                           std::vector<T> *base_in_attrs,
                                           std::vector<T> *base_out_attrs,
                                           std::vector<T> *in_attrs,
                                           std::vector<T> *out_attrs) {
  auto base_num_inputs = param.no_bias ? 2 : 3;

  base_out_attrs->push_back(out_attrs->at(0));
  for (int i = 0; i < base_num_inputs; ++i) {
    base_in_attrs->push_back(in_attrs->at(i));
  }
}

static bool SgMKLDNNFCInferShape(const nnvm::NodeAttrs &attrs,
                                 mxnet::ShapeVector *in_shapes,
                                 mxnet::ShapeVector *out_shapes) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    mxnet::ShapeVector base_in_shapes;
    mxnet::ShapeVector base_out_shapes;
    FillBaseInputOutputInfo(full_param.default_param, &base_in_shapes, &base_out_shapes,
                            in_shapes, out_shapes);
    bool ret = DefaultSubgraphOpShape(attrs, &base_in_shapes, &base_out_shapes);

    for (size_t i = 0; i < in_shapes->size(); ++i) {
      if (i < base_in_shapes.size())
        in_shapes->at(i) = base_in_shapes[i];
      else
        SHAPE_ASSIGN_CHECK(*in_shapes, i, Shape1(1));
    }

    out_shapes->at(0) = base_out_shapes[0];
    if (!full_param.mkldnn_param.enable_float_output) {
      SHAPE_ASSIGN_CHECK(*out_shapes, 1, Shape1(1));
      SHAPE_ASSIGN_CHECK(*out_shapes, 2, Shape1(1));
    }
    return ret;
  } else {
    return DefaultSubgraphOpShape(attrs, in_shapes, out_shapes);
  }
}

static bool SgMKLDNNFCInferType(const nnvm::NodeAttrs &attrs,
                                std::vector<int> *in_types,
                                std::vector<int> *out_types) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    bool channel_wise = false;
    if (full_param.mkldnn_param.channel_wise_quantize.has_value() &&
        full_param.mkldnn_param.channel_wise_quantize) {
      channel_wise = true;
    }
    size_t base_num_inputs = full_param.default_param.no_bias ? 2 : 3;
    CHECK(in_types->at(0) == mshadow::kInt8 ||
          in_types->at(0) == mshadow::kUint8)
        << "QuantizedFullyConnected only supports int8/uint8 input, while "
        << in_types->at(0) << " is given.";
    for (size_t i = 1; i < in_types->size(); ++i) {
      if (channel_wise) {
        TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
      } else {
        if (i < base_num_inputs) {
          TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kInt8);
        } else {
          TYPE_ASSIGN_CHECK(*in_types, i, mshadow::kFloat32);
        }
      }
    }

    if (full_param.mkldnn_param.enable_float_output) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kFloat32);
    } else {
      if (full_param.mkldnn_param.min_calib_range.has_value() &&
          full_param.mkldnn_param.max_calib_range.has_value()) {
        if (IsOutputUint8(full_param)) {
          TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kUint8);
        } else {
          TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
        }
      } else {
        TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt32);
      }
      TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
      TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
    }
    return true;
  } else {
    return DefaultSubgraphOpType(attrs, in_types, out_types);
  }
}

static bool SgMKLDNNFCStorageType(const nnvm::NodeAttrs &attrs,
                                  const int dev_mask,
                                  DispatchMode *dispatch_mode,
                                  std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  if (full_param.mkldnn_param.quantized) {
    std::vector<int> base_in_attrs;
    std::vector<int> base_out_attrs;
    FillBaseInputOutputInfo(full_param.default_param, &base_in_attrs, &base_out_attrs,
                            in_attrs, out_attrs);
    bool ret = DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                            &base_in_attrs, &base_out_attrs);

    for (size_t i = 0; i < in_attrs->size(); ++i) {
      if (i < base_in_attrs.size())
        in_attrs->at(i) = base_in_attrs[i];
      else
        type_assign(&in_attrs->at(i), mxnet::kDefaultStorage);
    }

    out_attrs->at(0) = base_out_attrs[0];
    if (!full_param.mkldnn_param.enable_float_output) {
      type_assign(&out_attrs->at(1), mxnet::kDefaultStorage);
      type_assign(&out_attrs->at(2), mxnet::kDefaultStorage);
    }
    return ret;
  } else {
    return DefaultSubgraphOpStorageType(attrs, dev_mask, dispatch_mode,
                                        in_attrs, out_attrs);
  }
}

static OpStatePtr CreateSgMKLDNNFCState(const nnvm::NodeAttrs &attrs,
                                        Context ctx,
                                        const mxnet::ShapeVector &in_shapes,
                                        const std::vector<int> &in_types) {
  return OpStatePtr::Create<SgMKLDNNFCOp>(attrs);
}

static void SgMKLDNNFCForward(const OpStatePtr &state_pointer,
                              const OpContext &ctx,
                              const std::vector<NDArray> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &outputs) {
  SgMKLDNNFCOp &op = state_pointer.get_state<SgMKLDNNFCOp>();
  op.Forward(ctx, inputs, req, outputs);
}

nnvm::ObjectPtr SgMKLDNNFCQuantizedOp(const NodeAttrs& attrs) {
  nnvm::ObjectPtr node = nnvm::Node::Create();
  node->attrs.op = Op::Get("_sg_mkldnn_fully_connected");
  node->attrs.name = "quantized_" + attrs.name;
  node->attrs.dict = attrs.dict;
  node->attrs.dict["quantized"] = "True";
  node->attrs.subgraphs.reserve(attrs.subgraphs.size());
  for (auto sub : attrs.subgraphs) {
    node->attrs.subgraphs.push_back(sub);
  }
  node->op()->attr_parser(&(node->attrs));
  return node;
}

static bool SgMKLDNNAvoidFCQuantizeInput(const NodeAttrs& attrs, const size_t index_to_check,
                                         const std::string quantize_granularity) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  std::unordered_set<size_t> avoid_indexes;
  if (quantize_granularity == "channel-wise") {
    avoid_indexes.insert(fullc::kWeight);   // weight
    if (!full_param.default_param.no_bias) {
      avoid_indexes.insert(fullc::kBias);   // bias
    }
  }

  return avoid_indexes.count(index_to_check);
}

NNVM_REGISTER_OP(_sg_mkldnn_fully_connected)
.describe(R"code(_sg_mkldnn_fully_connected)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  auto num_inputs = full_param.default_param.no_bias ? 2 : 3;
  if (full_param.mkldnn_param.quantized) {
    if (full_param.mkldnn_param.channel_wise_quantize.has_value() &&
        full_param.mkldnn_param.channel_wise_quantize) {
      return num_inputs + 2;  // min_data, max_data
    } else {
      return num_inputs * 3;
    }
  } else {
    return num_inputs;
  }
})
.set_num_outputs([](const NodeAttrs& attrs) {
  auto const &full_param = nnvm::get<MKLDNNFCFullParam>(attrs.parsed);
  return (full_param.mkldnn_param.quantized &&
          !full_param.mkldnn_param.enable_float_output) ? 3 : 1;
})
.set_attr_parser(SgMKLDNNFCParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames", SgMKLDNNFCListInputNames)
.set_attr<nnvm::FListOutputNames>("FListOutputNames", SgMKLDNNFCListOutputNames)
.set_attr<mxnet::FInferShape>("FInferShape", SgMKLDNNFCInferShape)
.set_attr<nnvm::FInferType>("FInferType", SgMKLDNNFCInferType)
.set_attr<FInferStorageType>("FInferStorageType", SgMKLDNNFCStorageType)
.set_attr<FCreateOpState>("FCreateOpState", CreateSgMKLDNNFCState)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", SgMKLDNNFCForward)
.set_attr<bool>("TIsMKLDNN", true)
// TODO(Xinyu): a temp solution to enable GluonCV INT8 flow,
// will be reverted after the improvement of CachedOP is done.
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                               DefaultSubgraphOpMutableInputs)
.set_attr<std::string>("key_var_num_args", "num_args")
.set_attr<FQuantizable>("FQuantizable", [](const NodeAttrs& attrs) {
    return QuantizeType::kMust;
})
.set_attr<FQuantizedOp>("FQuantizedOp", SgMKLDNNFCQuantizedOp)
.set_attr<FNeedRequantize>("FNeedRequantize", [](const NodeAttrs& attrs) { return true; })
.set_attr<FAvoidQuantizeInput>("FAvoidQuantizeInput", SgMKLDNNAvoidFCQuantizeInput);

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
