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
 * \file softmax_output-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
#define MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace softmaxout_enum {
enum SoftmaxOutputOpInputs {kData, kLabel};
enum SoftmaxOutputOpOutputs {kOut};
enum SoftmaxOutputNormType {kNull, kBatch, kValid};
enum SoftmaxOutputOpResource {kTempSpace};
}  // namespace softmaxout_enum

struct SoftmaxOutputParam : public dmlc::Parameter<SoftmaxOutputParam> {
  float grad_scale;
  float ignore_label;
  bool multi_output;
  bool use_ignore;
  bool preserve_shape;
  int normalization;
  bool out_grad;
  float smooth_alpha;
  DMLC_DECLARE_PARAMETER(SoftmaxOutputParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scales the gradient by a float factor.");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("The instances whose `labels` == `ignore_label` will be ignored "
              "during backward, if `use_ignore` is set to ``true``).");
    DMLC_DECLARE_FIELD(multi_output).set_default(false)
    .describe("If set to ``true``, the softmax function will be computed along "
              "axis ``1``. This is applied when the shape "
              "of input array differs from the shape of label array.");
    DMLC_DECLARE_FIELD(use_ignore).set_default(false)
    .describe("If set to ``true``, the `ignore_label` value will not contribute "
              "to the backward gradient.");
    DMLC_DECLARE_FIELD(preserve_shape).set_default(false)
    .describe("If set to ``true``, the softmax function will be computed along "
              "the last axis (``-1``).");
    DMLC_DECLARE_FIELD(normalization)
    .add_enum("null", softmaxout_enum::kNull)
    .add_enum("batch", softmaxout_enum::kBatch)
    .add_enum("valid", softmaxout_enum::kValid)
    .set_default(softmaxout_enum::kNull)
    .describe("Normalizes the gradient.");
    DMLC_DECLARE_FIELD(out_grad)
    .set_default(false)
    .describe("Multiplies gradient with output gradient element-wise.");
    DMLC_DECLARE_FIELD(smooth_alpha)
    .set_default(0.0f)
    .set_range(0.0f, 1.0f)
    .describe("Constant for computing a label smoothed version of cross-entropy"
              "for the backwards pass.  This constant gets subtracted from the"
              "one-hot encoding of the gold label and distributed uniformly to"
              "all other labels.");
  };

  bool operator==(const SoftmaxOutputParam& other) const {
    return this->grad_scale == other.grad_scale &&
           this->ignore_label == other.ignore_label &&
           this->multi_output == other.multi_output &&
           this->use_ignore == other.use_ignore &&
           this->preserve_shape == other.preserve_shape &&
           this->normalization == other.normalization &&
           this->out_grad == other.out_grad &&
           this->smooth_alpha == other.smooth_alpha;
  }
};

template<typename xpu, typename DType>
class SoftmaxOutputOp : public Operator {
 public:
  explicit SoftmaxOutputOp(SoftmaxOutputParam param) : param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U) << "SoftmaxOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 1U) << "SoftmaxOutput Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (param_.multi_output) {
      index_t n = in_data[softmaxout_enum::kData].size(0);
      index_t k = in_data[softmaxout_enum::kData].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<index_t>(in_data[softmaxout_enum::kData].Size()/n/k));
      Tensor<xpu, 3, DType> data =
          in_data[softmaxout_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);
      Tensor<xpu, 3, DType> out =
          out_data[softmaxout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
      Softmax(out, data);
    } else {
      if (param_.preserve_shape) {
        Tensor<xpu, 2, DType> data = in_data[softmaxout_enum::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> out = out_data[softmaxout_enum::kOut].FlatTo2D<xpu, DType>(s);
        Softmax(out, data);
      } else {
        index_t n = in_data[softmaxout_enum::kData].size(0);
        index_t k = in_data[softmaxout_enum::kData].Size()/n;
        Shape<2> s2 = Shape2(n, k);
        Tensor<xpu, 2, DType> data =
            in_data[softmaxout_enum::kData].get_with_shape<xpu, 2, DType>(s2, s);
        Tensor<xpu, 2, DType> out =
            out_data[softmaxout_enum::kOut].get_with_shape<xpu, 2, DType>(s2, s);
        Softmax(out, data);
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_GE(in_grad.size(), 1U);
    CHECK_GE(req.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    if (out_data[softmaxout_enum::kOut].shape_ ==
        in_data[softmaxout_enum::kLabel].shape_) {
      // use probability as label
      Tensor<xpu, 2, DType> label = in_data[softmaxout_enum::kLabel].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> out = out_data[softmaxout_enum::kOut].FlatTo2D<xpu, DType>(s);
      Tensor<xpu, 2, DType> grad = in_grad[softmaxout_enum::kData].FlatTo2D<xpu, DType>(s);
      if (param_.out_grad) {
        Tensor<xpu, 2, DType> ograd = out_grad[softmaxout_enum::kOut].FlatTo2D<xpu, DType>(s);
        grad = scalar<DType>(param_.grad_scale) * (out - label) * ograd;
      } else {
        grad = (out - label) * scalar<DType>(param_.grad_scale);
      }
    } else if (param_.multi_output) {
      index_t n = out_data[softmaxout_enum::kOut].size(0);
      index_t k = out_data[softmaxout_enum::kOut].size(1);
      Shape<3> s3 = Shape3(n, k, static_cast<index_t>(out_data[softmaxout_enum::kOut].Size()/n/k));
      Shape<2> s2 = Shape2(s3[0], s3[2]);
      Tensor<xpu, 2, DType> label =
          in_data[softmaxout_enum::kLabel].get_with_shape<xpu, 2, DType>(s2, s);
      Tensor<xpu, 3, DType> out =
          out_data[softmaxout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
      Tensor<xpu, 3, DType> grad =
          in_grad[softmaxout_enum::kData].get_with_shape<xpu, 3, DType>(s3, s);

      index_t valid_cnt = label.shape_.Size();
      if (param_.use_ignore) {
          SoftmaxGrad(grad, out, label, static_cast<DType>(param_.ignore_label));
      } else {
          SoftmaxGrad(grad, out, label);
      }
      if (param_.normalization == softmaxout_enum::kBatch) {
        valid_cnt = label.size(0);
      } else if (param_.normalization == softmaxout_enum::kValid) {
        int i_label = static_cast<int>(param_.ignore_label);
        Tensor<cpu, 2, DType> workspace =
          ctx.requested[softmaxout_enum::kTempSpace].get_host_space_typed<2, DType>(
          label.shape_);
        Copy(workspace, label, label.stream_);
        for (index_t i = 0; i < workspace.size(0); ++i) {
          for (index_t j = 0; j < workspace.size(1); ++j) {
            if (static_cast<int>(workspace[i][j]) == i_label) {
              valid_cnt--;
            }
          }
        }
        valid_cnt = valid_cnt == 0 ? 1 : valid_cnt;
      } else {
        valid_cnt = 1;
      }
      grad *= DType(param_.grad_scale /
                    (param_.normalization == softmaxout_enum::kValid ? 1 : s3[2]) /
                    valid_cnt);
      if (param_.out_grad) {
        Tensor<xpu, 3, DType> ograd =
          out_grad[softmaxout_enum::kOut].get_with_shape<xpu, 3, DType>(s3, s);
        grad *= ograd;
      }
    } else {
      Shape<1> label_shape = Shape1(in_data[softmaxout_enum::kLabel].Size());
      Shape<2> data_shape;
      if (param_.preserve_shape) {
        data_shape = out_data[softmaxout_enum::kOut].shape_.FlatTo2D();
//        Tensor<xpu, 1, DType> label = in_data[softmaxout_enum::kLabel].FlatTo1D<xpu, DType>(s);
//        Tensor<xpu, 2, DType> out = out_data[softmaxout_enum::kOut].FlatTo2D<xpu, DType>(s);
//        Tensor<xpu, 2, DType> grad = in_grad[softmaxout_enum::kData].FlatTo2D<xpu, DType>(s);
      } else {
        index_t n = out_data[softmaxout_enum::kOut].size(0);
        data_shape = Shape2(n, out_data[softmaxout_enum::kOut].Size()/n);
      }
      Tensor<xpu, 1, DType> label = in_data[softmaxout_enum::kLabel].get_with_shape<xpu, 1, DType>(
          label_shape, s);
      Tensor<xpu, 2, DType> out =
          out_data[softmaxout_enum::kOut].get_with_shape<xpu, 2, DType>(data_shape, s);
      Tensor<xpu, 2, DType> grad =
          in_grad[softmaxout_enum::kData].get_with_shape<xpu, 2, DType>(data_shape, s);
      index_t valid_cnt = label.shape_.Size();
      if (param_.use_ignore) {
        if (param_.smooth_alpha == 0.0f) {
          SoftmaxGrad(grad, out, label, static_cast<DType>(param_.ignore_label));
        } else {
          SmoothSoftmaxGrad(grad, out, label, static_cast<DType>(param_.ignore_label),
                            param_.smooth_alpha);
        }
      } else {
        if (param_.smooth_alpha == 0.0f) {
          SoftmaxGrad(grad, out, label);
        } else {
          SmoothSoftmaxGrad(grad, out, label, param_.smooth_alpha);
        }
      }
      if (param_.normalization == softmaxout_enum::kBatch) {
        valid_cnt = label.size(0);
      } else if (param_.normalization == softmaxout_enum::kValid) {
        int i_label = static_cast<int>(param_.ignore_label);
        Tensor<cpu, 1, DType> workspace =
          ctx.requested[softmaxout_enum::kTempSpace].get_host_space_typed<1, DType>(
          label.shape_);
        Copy(workspace, label, label.stream_);
        for (index_t i = 0; i < label.size(0); ++i) {
          if (static_cast<int>(workspace[i]) == i_label) {
            valid_cnt--;
          }
        }
        valid_cnt = valid_cnt == 0 ? 1 : valid_cnt;
      } else {
        valid_cnt = 1;
      }
      grad *= DType(param_.grad_scale / valid_cnt);
      if (param_.out_grad) {
        Tensor<xpu, 2, DType> ograd =
          out_grad[softmaxout_enum::kOut].get_with_shape<xpu, 2, DType>(data_shape, s);
        grad *= ograd;
      }
    }
  }

 private:
  SoftmaxOutputParam param_;
};  // class SoftmaxOutputOp

template<typename xpu>
void SoftmaxOutputCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  const SoftmaxOutputParam &param = nnvm::get<SoftmaxOutputParam>(attrs.parsed);
  const std::vector<TBlob> no_use_but_adapt_origin_api;
  CHECK_EQ(inputs.size(), 2U);

  MSHADOW_REAL_TYPE_SWITCH(inputs[softmaxout_enum::kData].type_flag_, DType, {
    SoftmaxOutputOp<xpu, DType> op(param);
    op.Forward(ctx, inputs, req, outputs, no_use_but_adapt_origin_api);
  });
}

template<typename xpu>
void SoftmaxOutputGradCompute(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  const SoftmaxOutputParam& param = nnvm::get<SoftmaxOutputParam>(attrs.parsed);
  const std::vector<TBlob> no_use_but_adapt_origin_api;
  CHECK_EQ(inputs.size(), 2U);

  std::vector<TBlob> out_grad{inputs[0]};
  std::vector<TBlob> out_data{inputs[0]};
  std::vector<TBlob> in_data(inputs.begin(), inputs.end());
  int dtype = inputs[0].type_flag_;
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    SoftmaxOutputOp<xpu, DType> op(param);
    op.Backward(ctx, out_grad, in_data, out_data, req, in_grad, no_use_but_adapt_origin_api);
  });
}


#if DMLC_USE_CXX11
class SoftmaxOutputProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(mxnet::ShapeVector *in_shape,
                  mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
    const mxnet::TShape &dshape = in_shape->at(0);
    if (!shape_is_known(dshape)) return false;

    // label.shape == data.shape: use probability as label
    if (dshape != (*in_shape)[softmaxout_enum::kLabel]) {
      if (param_.multi_output) {
        mxnet::TShape lshape1 = Shape2(dshape[0], dshape.Size()/dshape[0]/dshape[1]);
        mxnet::TShape lshape2(dshape.ndim() - 1, -1);
        lshape2[0] = dshape[0];
        for (int i = 2; i < dshape.ndim(); ++i)
          lshape2[i-1] = dshape[i];
        mxnet::TShape lshape3 = dshape;
        lshape3[1] = 1;
        if (!mxnet::ndim_is_known(in_shape->at(softmaxout_enum::kLabel))) {
          in_shape->at(softmaxout_enum::kLabel) = lshape1;
        } else if (in_shape->at(softmaxout_enum::kLabel) == lshape1) {
        } else if (in_shape->at(softmaxout_enum::kLabel) == lshape2) {
        } else if (in_shape->at(softmaxout_enum::kLabel) == lshape3) {
        } else {
          std::ostringstream os;
          os << "Expecting " << lshape1 << " or " << lshape2
             << ". But got " << in_shape->at(softmaxout_enum::kLabel);
          throw InferShapeError(os.str(), softmaxout_enum::kLabel);
        }
      } else {
        mxnet::TShape label_shape(dshape.ndim() - 1, -1);
        for (int i = 0; i + 1 < dshape.ndim(); ++i)
          label_shape[i] = dshape[i];
        SHAPE_ASSIGN_CHECK(*in_shape, softmaxout_enum::kLabel, label_shape);
      }
    }
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (size_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new SoftmaxOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "SoftmaxOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    if (param_.out_grad) {
      return {in_data[softmaxout_enum::kLabel], out_data[softmaxout_enum::kOut],
              out_grad[softmaxout_enum::kOut]};
    } else {
      return {in_data[softmaxout_enum::kLabel], out_data[softmaxout_enum::kOut]};
    }
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_data[softmaxout_enum::kOut], in_grad[softmaxout_enum::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[softmaxout_enum::kData], out_data[softmaxout_enum::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return nullptr;
  }

  Operator* CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                             std::vector<int> *in_type) const override;

 protected:
  SoftmaxOutputParam param_;
};  // class SoftmaxOutputProp

class DeprecatedSoftmaxProp : public SoftmaxOutputProp {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    LOG(INFO) << "Softmax symbol is renamed to SoftmaxOutput. "
      << "This API will be deprecated in Dec, 2015";
    SoftmaxOutputProp::param_.Init(kwargs);
  }

  std::string TypeString() const override {
    return "Softmax";
  }
};
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::SoftmaxOutputParam> {
  size_t operator()(const mxnet::op::SoftmaxOutputParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.grad_scale);
    ret = dmlc::HashCombine(ret, val.ignore_label);
    ret = dmlc::HashCombine(ret, val.multi_output);
    ret = dmlc::HashCombine(ret, val.use_ignore);
    ret = dmlc::HashCombine(ret, val.preserve_shape);
    ret = dmlc::HashCombine(ret, val.normalization);
    ret = dmlc::HashCombine(ret, val.out_grad);
    ret = dmlc::HashCombine(ret, val.smooth_alpha);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_SOFTMAX_OUTPUT_INL_H_
