/*!
 * Copyright (c) 2015 by Contributors
 * \file instance_norm-inl.h
 * \brief
 * \author Sebastian Bodenstein
*/
#ifndef MXNET_OPERATOR_INSTANCE_NORM_INL_H_
#define MXNET_OPERATOR_INSTANCE_NORM_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace instance_norm {
enum InstanceNormInputs { kData, kWeight, kBias };
enum InstanceNormOutputs { kOut, kMean, kVar };
}  // namespace instance_norm

struct InstanceNormParam : public dmlc::Parameter<InstanceNormParam> {
  float eps;
  DMLC_DECLARE_PARAMETER(InstanceNormParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f).describe(
        "Epsilon to prevent div 0");
  }
};  // struct InstanceNormParam

template <typename xpu>
class InstanceNormOp : public Operator {
 public:
  explicit InstanceNormOp(InstanceNormParam param) { param_ = param; }
  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);

    CHECK_GE(in_data[instance_norm::kData].Size(), 3)
        << "InstanceNorm only supports input tensors of rank > 2.";

    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[instance_norm::kData].size(0);
    int c = in_data[instance_norm::kData].size(1);
    int rest_dim =
        static_cast<int>(in_data[instance_norm::kData].Size() / n / c);
    Shape<2> s2 = Shape2(n * c, rest_dim);
    const real_t scale = static_cast<real_t>(1) / static_cast<real_t>(rest_dim);
    // Get Inputs
    Tensor<xpu, 2> data =
        in_data[instance_norm::kData].get_with_shape<xpu, 2, real_t>(s2, s);
    Tensor<xpu, 1> weight =
        in_data[instance_norm::kWeight].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[instance_norm::kBias].get<xpu, 1, real_t>(s);
    // Get Outputs
    Tensor<xpu, 2> out =
        out_data[instance_norm::kOut].get_with_shape<xpu, 2, real_t>(s2, s);
    Tensor<xpu, 1> var = out_data[instance_norm::kVar].FlatTo1D<xpu, real_t>(s);
    Tensor<xpu, 1> mean =
        out_data[instance_norm::kMean].FlatTo1D<xpu, real_t>(s);
    // Calculate mean + var
    mean = scale * sumall_except_dim<0>(data);
    var = scale * sumall_except_dim<0>(F<mshadow_op::square>(
                      data - broadcast<0>(mean, data.shape_)));
    Assign(
        out, req[instance_norm::kOut],
        broadcast<0>(reshape(repmat(weight, n), Shape1(n * c)), out.shape_) *
                (data - broadcast<0>(mean, data.shape_)) /
                F<mshadow_op::square_root>(
                    broadcast<0>(var + param_.eps, data.shape_)) +
            broadcast<0>(reshape(repmat(bias, n), Shape1(n * c)), out.shape_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);

    CHECK_GE(in_data[instance_norm::kData].Size(), 3)
        << "InstanceNorm only supports input tensors of rank > 2.";

    Stream<xpu> *s = ctx.get_stream<xpu>();
    int n = in_data[instance_norm::kData].size(0);
    int c = in_data[instance_norm::kData].size(1);
    int rest_dim =
        static_cast<int>(in_data[instance_norm::kData].Size() / n / c);
    Shape<2> s2 = Shape2(n * c, rest_dim);
    // Get Inputs
    Tensor<xpu, 2> data =
        in_data[instance_norm::kData].get_with_shape<xpu, 2, real_t>(s2, s);
    Tensor<xpu, 2> gdata =
        in_grad[instance_norm::kData].get_with_shape<xpu, 2, real_t>(s2, s);
    Tensor<xpu, 1> weight =
        in_data[instance_norm::kWeight].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gweight =
        in_grad[instance_norm::kWeight].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[instance_norm::kBias].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[instance_norm::kBias].get<xpu, 1, real_t>(s);
    // Get Outputs
    Tensor<xpu, 2> out =
        out_data[instance_norm::kOut].get_with_shape<xpu, 2, real_t>(s2, s);
    Tensor<xpu, 2> gout =
        out_grad[instance_norm::kOut].get_with_shape<xpu, 2, real_t>(s2, s);
    Tensor<xpu, 1> var = out_data[instance_norm::kVar].FlatTo1D<xpu, real_t>(s);
    Tensor<xpu, 1> mean =
        out_data[instance_norm::kMean].FlatTo1D<xpu, real_t>(s);

    // Calculate grads
    Assign(gbias, req[instance_norm::kBias], sumall_except_dim<0>(gout));

    Assign(gweight, req[instance_norm::kOut],
           sumall_except_dim<0>((data - broadcast<0>(mean, data.shape_)) /
                                F<mshadow_op::square_root>(broadcast<0>(
                                    var + param_.eps, data.shape_))));
    Assign(gdata, req[instance_norm::kOut],
           gout * broadcast<0>(reshape(repmat(weight, n), Shape1(n * c)), out.shape_) /
                F<mshadow_op::square_root>(
                    broadcast<0>(var + param_.eps, data.shape_)));
  }

 private:
  InstanceNormParam param_;
};  // class InstanceNormOp

template <typename xpu>
Operator *CreateOp(InstanceNormParam param, int dtype);

#if DMLC_USE_CXX11
class InstanceNormProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs)
      override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;

    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);
    out_shape->push_back(Shape2(dshape[0], dshape[1]));
    out_shape->push_back(Shape2(dshape[0], dshape[1]));
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new InstanceNormProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "InstanceNorm"; }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[instance_norm::kOut],   out_data[instance_norm::kMean],
            out_data[instance_norm::kVar],   in_data[instance_norm::kData],
            in_data[instance_norm::kWeight], in_data[instance_norm::kBias]};
  }

  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 3; }

  std::vector<std::string> ListArguments() const override {
    return {"data", "weight", "bias"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "mean", "var"};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  InstanceNormParam param_;
};      // InstanceNormProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_INSTANCE_NORM_INL_H_
