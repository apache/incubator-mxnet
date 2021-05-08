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
 * Copyright (c) 2015 by Contributors
 * \file leaky_relu-inl.h
 * \brief leaky relu family operator
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_LEAKY_RELU_INL_H_
#define MXNET_OPERATOR_LEAKY_RELU_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/random_generator.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./random/sampler.h"
#include "./random/sample_op.h"
#include "./tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

namespace leakyrelu {
enum LeakyReLUOpInputs {kData, kGamma};
enum LeakyReLUOpOutputs {kOut, kMask};
enum LeakyReLUOpType {kLeakyReLU, kPReLU, kRReLU, kELU, kSELU, kGELU, kMish};
enum LeakyReLUOpResource {kRandom};
}  // namespace leakyrelu

struct LeakyReLUParam : public dmlc::Parameter<LeakyReLUParam> {
  // use int for enumeration
  int act_type;
  float slope;
  float lower_bound;
  float upper_bound;
  DMLC_DECLARE_PARAMETER(LeakyReLUParam) {
    DMLC_DECLARE_FIELD(act_type).set_default(leakyrelu::kLeakyReLU)
    .add_enum("rrelu", leakyrelu::kRReLU)
    .add_enum("leaky", leakyrelu::kLeakyReLU)
    .add_enum("prelu", leakyrelu::kPReLU)
    .add_enum("elu", leakyrelu::kELU)
    .add_enum("selu", leakyrelu::kSELU)
    .add_enum("gelu", leakyrelu::kGELU)
    .add_enum("mish", leakyrelu::kMish)
    .describe("Activation function to be applied.");
    DMLC_DECLARE_FIELD(slope).set_default(0.25f)
    .describe("Init slope for the activation. (For leaky and elu only)");
    DMLC_DECLARE_FIELD(lower_bound).set_default(0.125f)
    .describe("Lower bound of random slope. (For rrelu only)");
    DMLC_DECLARE_FIELD(upper_bound).set_default(0.334f)
    .describe("Upper bound of random slope. (For rrelu only)");
  }
  std::string ActType2String(int act_type) {
    switch (act_type) {
      case leakyrelu::kRReLU:
        return "rrelu";
      case leakyrelu::kLeakyReLU:
        return "leaky";
      case leakyrelu::kPReLU:
        return "prelu";
      case leakyrelu::kELU:
        return "elu";
      case leakyrelu::kSELU:
        return "selu";
      case leakyrelu::kGELU:
        return "gelu";
      default:
        LOG(FATAL) << "Unknown act_type enum " << act_type;
    }
    LOG(FATAL) << "should not reach here ";
    return "";
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream act_type_s, slope_s, lower_bound_s, upper_bound_s;
    act_type_s << act_type;
    slope_s << slope;
    lower_bound_s << lower_bound;
    upper_bound_s << upper_bound;
    (*dict)["act_type"] = ActType2String(act_type);
    (*dict)["slope"] = slope_s.str();
    (*dict)["lower_bound"] = lower_bound_s.str();
    (*dict)["upper_bound"] = upper_bound_s.str();
  }
};

template<typename xpu, typename DType>
class LeakyReLUOp : public Operator {
 public:
  explicit LeakyReLUOp(LeakyReLUParam param) {
    param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.act_type == leakyrelu::kPReLU ? 2 : 1;
    CHECK_EQ(in_data.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> data;
    Tensor<xpu, 3, DType> out;
    Tensor<xpu, 3, DType> mask;
    int n = in_data[leakyrelu::kData].shape_[0];
    int k = (in_data[leakyrelu::kData].ndim() > 1) ? in_data[leakyrelu::kData].shape_[1] : 1;
    Shape<3> dshape = Shape3(n, k, in_data[leakyrelu::kData].Size()/n/k);
    data = in_data[leakyrelu::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    out = out_data[leakyrelu::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    if (req[leakyrelu::kOut] == kNullOp) {
      return;
    }
    switch (param_.act_type) {
      case leakyrelu::kLeakyReLU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::xelu, Req>, xpu>::Launch(
            s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_, DType(param_.slope));
        });
        break;
      }
      case leakyrelu::kPReLU: {
        mxnet::TShape gshape = expand_shape(in_data[leakyrelu::kGamma].shape_,
                                     in_data[leakyrelu::kData].shape_);
        mxnet::TShape new_lshape, new_rshape, new_oshape;
        const int ndim = op::BinaryBroadcastShapeCompact(in_data[leakyrelu::kData].shape_,
                                                         gshape,
                                                         out_data[leakyrelu::kOut].shape_,
                                                         &new_lshape, &new_rshape, &new_oshape);
        if (!ndim) {
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
            const size_t size = (minthree(out_data[leakyrelu::kOut].Size(),
                                          in_data[leakyrelu::kData].Size(),
                                          in_data[leakyrelu::kGamma].Size())
            + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::xelu, Req>, xpu>::Launch(
                s, size, out_data[leakyrelu::kOut].dptr<DType>(),
            in_data[leakyrelu::kData].dptr<DType>(), in_data[leakyrelu::kGamma].dptr<DType>());
          });
        } else {
          BROADCAST_NDIM_SWITCH(ndim, NDim, {
            mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
            mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
            mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
            mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, mshadow_op::xelu>, xpu>::
            template LaunchEx(s, new_oshape.Size(), req[leakyrelu::kOut], lstride, rstride, oshape,
            in_data[leakyrelu::kData].dptr<DType>(), in_data[leakyrelu::kGamma].dptr<DType>(),
            out_data[leakyrelu::kOut].dptr<DType>());
          });
        }
        break;
      }
      case leakyrelu::kRReLU: {
        if (ctx.is_train) {
          mask = out_data[leakyrelu::kMask].get_with_shape<xpu, 3, DType>(dshape, s);
          mxnet::op::UniformSampler<xpu> sampler;
          Tensor<xpu, 1, DType> low, high;
          mxnet::op::GetSamplingTempData<xpu, DType>(DType(0.0f), DType(1.0f), ctx, &low, &high);
          mxnet::common::random::RandGenerator<xpu, DType> *pgen =
            ctx.requested[0].get_parallel_random<xpu, DType>();
          Tensor<xpu, 1, DType> out = mask.FlatTo1D();
          sampler.Sample(low, high, out, pgen, s);
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kMask], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
              s, mask.size(0) * mask.size(1) * mask.size(2), mask.dptr_, mask.dptr_,
              DType(param_.upper_bound - param_.lower_bound));
          });
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kMask], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::plus, Req>, xpu>::Launch(
              s, mask.size(0) * mask.size(1) * mask.size(2), mask.dptr_, mask.dptr_,
              DType(param_.lower_bound));
          });
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::xelu, Req>, xpu>::Launch(
              s, mask.size(0) * mask.size(1) * mask.size(2), out.dptr_, data.dptr_, mask.dptr_);
          });
        } else {
          const float slope = (param_.lower_bound + param_.upper_bound) / 2.0f;
          MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::xelu, Req>, xpu>::Launch(
              s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_, DType(slope));
          });
        }
        break;
      }
      case leakyrelu::kELU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::elu, Req>, xpu>::Launch(
            s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_,
            DType(param_.slope));
        });
        break;
      }
      case leakyrelu::kSELU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::selu, Req>, xpu>::Launch(
            s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_);
        });
        break;
      }
      case leakyrelu::kGELU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::gelu, Req>, xpu>::Launch(
            s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_);
        });
        break;
      }
      case leakyrelu::kMish: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mish, Req>, xpu>::Launch(
            s, out.size(0) * out.size(1) * out.size(2), out.dptr_, data.dptr_);
        });
        break;
      }
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

  virtual void Backward(const OpContext & ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    size_t expected = param_.act_type == leakyrelu::kPReLU ? 2 : 1;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data.size(), expected);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> output;
    Tensor<xpu, 3, DType> data;
    Tensor<xpu, 3, DType> gdata;
    Tensor<xpu, 3, DType> grad;
    Tensor<xpu, 3, DType> mask;
    int n = out_grad[leakyrelu::kOut].shape_[0];
    int k = (out_grad[leakyrelu::kOut].ndim() > 1) ? out_grad[leakyrelu::kOut].shape_[1] : 1;
    Shape<3> dshape = Shape3(n, k, out_grad[leakyrelu::kOut].Size()/n/k);
    grad = out_grad[leakyrelu::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    gdata = in_grad[leakyrelu::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    output = out_data[leakyrelu::kOut].get_with_shape<xpu, 3, DType>(dshape, s);
    if (param_.act_type == leakyrelu::kRReLU) {
      mask = out_data[leakyrelu::kMask].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    if (   param_.act_type == leakyrelu::kPReLU 
        || param_.act_type == leakyrelu::kGELU
        || param_.act_type == leakyrelu::kMish) {
      data = in_data[leakyrelu::kData].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    switch (param_.act_type) {
      case leakyrelu::kLeakyReLU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<
            mxnet_op::backward_grad_tuned<mshadow_op::xelu_grad>, Req>, xpu>::Launch(
              s, gdata.size(0) * gdata.size(1) * gdata.size(2), gdata.dptr_, grad.dptr_,
              output.dptr_, DType(param_.slope));
        });
        break;
      }
      case leakyrelu::kPReLU: {
        mxnet::TShape gshape = expand_shape(in_grad[leakyrelu::kGamma].shape_,
                                     in_grad[leakyrelu::kData].shape_);
        mxnet::TShape new_lshape, new_rshape, new_oshape;
        const bool need_bc = BinaryBroadcastShapeCompact(in_grad[leakyrelu::kData].shape_,
                                                         gshape,
                                                         out_grad[leakyrelu::kOut].shape_,
                                                         &new_lshape,
                                                         &new_rshape,
                                                         &new_oshape) != 0;
        if (!need_bc) {
#if !defined(__CUDACC__)
          ElemwiseBinaryOp::BackwardUseIn<xpu,
                                          mshadow_op::xelu_grad,
                                          mshadow_op::prelu_grad>(
            nnvm::NodeAttrs(), ctx, {out_grad[leakyrelu::kOut],
                                     in_data[leakyrelu::kData],
                                     in_data[leakyrelu::kGamma]}, req, in_grad);
#else
          ElemwiseBinaryRTCBwdUseIn {"xelu_grad", "prelu_grad"}(
            nnvm::NodeAttrs(), ctx, {out_grad[leakyrelu::kOut],
                                     in_data[leakyrelu::kData],
                                     in_data[leakyrelu::kGamma]}, req, in_grad);
#endif  // !defined(__CUDACC__)
        } else {
#if !defined(__CUDACC__)
          BROADCAST_NDIM_SWITCH(new_oshape.ndim(), NDim, {
            BinaryBroadcastBackwardUseInImpl<xpu, NDim, DType,
              mshadow_op::xelu_grad, mshadow_op::prelu_grad>(
                ctx, {out_grad[leakyrelu::kOut],
                      in_data[leakyrelu::kData],
                      in_data[leakyrelu::kGamma]}, req, in_grad,
                new_lshape, new_rshape, new_oshape);
          });
#else
          std::vector<TBlob> new_in_grad(2);
          new_in_grad[leakyrelu::kData] = in_grad[leakyrelu::kData];
          new_in_grad[leakyrelu::kGamma] = in_grad[leakyrelu::kGamma].reshape(gshape);
          BinaryBroadcastRTCBackwardUseIn {"xelu_grad", "prelu_grad"}(
              nnvm::NodeAttrs(), ctx, {out_grad[leakyrelu::kOut],
                                       in_data[leakyrelu::kData],
                                       in_data[leakyrelu::kGamma]},
                req, new_in_grad);
#endif  // !defined(__CUDACC__)
        }
        break;
      }
      case leakyrelu::kRReLU: {
        Assign(gdata, req[leakyrelu::kData], F<mshadow_op::xelu_grad>(output, mask) * grad);
        break;
      }
      case leakyrelu::kELU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<
            mxnet_op::backward_grad_tuned<mshadow_op::elu_grad>, Req>, xpu>::Launch(
              s, gdata.size(0) * gdata.size(1) * gdata.size(2), gdata.dptr_, grad.dptr_,
              output.dptr_, DType(param_.slope));
        });
        break;
      }
      case leakyrelu::kSELU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<
            mxnet_op::backward_grad_tuned<mshadow_op::selu_grad>, Req>, xpu>::Launch(
              s, gdata.size(0) * gdata.size(1) * gdata.size(2), gdata.dptr_, grad.dptr_,
              output.dptr_);
        });
        break;
      }
      case leakyrelu::kGELU: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<
            mxnet_op::backward_grad_tuned<mshadow_op::gelu_grad>, Req>, xpu>::Launch(
              s, gdata.size(0) * gdata.size(1) * gdata.size(2), gdata.dptr_, grad.dptr_,
              data.dptr_);
        });
        break;
      }
      case leakyrelu::kMish: {
        MXNET_ASSIGN_REQ_SWITCH(req[leakyrelu::kData], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<
            mxnet_op::backward_grad_tuned<mshadow_op::mish_grad>, Req>, xpu>::Launch(
              s, gdata.size(0) * gdata.size(1) * gdata.size(2), gdata.dptr_, grad.dptr_,
              data.dptr_, output.dptr_);
        });
        break;
      }
      default:
        LOG(FATAL) << "Not implmented";
    }
  }

 private:
  /*! \brief Minimum of three */
  static MSHADOW_XINLINE size_t minthree(const size_t a, const size_t b, const size_t c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
  }
  static inline mxnet::TShape expand_shape(const mxnet::TShape& src, const mxnet::TShape& dst) {
    mxnet::TShape result(dst.ndim(), -1);
    int s = src.ndim() - 1;
    for (int i = dst.ndim() - 1; i >= 0; i--) {
      if (s >= 0 && i <= 1 && (dst[i] == src[s] || src[s] == 1)) {
        result[i] = src[s];
        s--;
      } else {
        result[i] = 1;
      }
    }
    CHECK(s == -1) << "Cannot broadcast gamma to data. gamma: " << src << ", data: " << dst;
    return result;
  }
  LeakyReLUParam param_;
};  // class LeakyReLUOp

template<typename xpu>
void LeakyReLUCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  if (inputs[0].Size() == 0U) return;
  const LeakyReLUParam &param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  const std::vector<TBlob> no_use_but_adapt_origin_api;
  size_t expected = param.act_type == leakyrelu::kPReLU ? 2 : 1;
  CHECK_EQ(inputs.size(), expected);

  MSHADOW_REAL_TYPE_SWITCH(inputs[leakyrelu::kData].type_flag_, DType, {
    LeakyReLUOp<xpu, DType> op(param);
    op.Forward(ctx, inputs, req, outputs, no_use_but_adapt_origin_api);
  });
}

template<typename xpu>
void LeakyReLUGradCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  if (inputs[0].Size() == 0U) return;
  const LeakyReLUParam& param = nnvm::get<LeakyReLUParam>(attrs.parsed);
  const std::vector<TBlob> no_use_but_adapt_origin_api;
  // inputs: out_grad, input_data, input_gamma, output, output_mask
  size_t expected_in = param.act_type == leakyrelu::kPReLU ? 2 : 1;
  size_t expected_out = param.act_type == leakyrelu::kRReLU ? 2 : 1;

  CHECK_GE(inputs.size(), 1 + expected_in + expected_out);
  std::vector<TBlob> out_grad{inputs[0]};
  std::vector<TBlob> in_data(inputs.begin() + 1,
                             inputs.begin() + 1 + expected_in);
  std::vector<TBlob> out_data(inputs.begin() + 1 + expected_in,
                              inputs.begin() + 1 + expected_in + expected_out);

  CHECK_EQ(req.size(), outputs.size());
  int dtype = inputs[0].type_flag_;
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    LeakyReLUOp<xpu, DType> op(param);
    op.Backward(ctx, out_grad, in_data, out_data, req, in_grad, no_use_but_adapt_origin_api);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_LEAKY_RELU_INL_H_

