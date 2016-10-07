/*!
 *  Copyright (c) 2015 by Contributors
 * \file broadcast_reduce_op-inl.h
 * \brief Function defintion of matrix related operators
 */
#ifndef MXNET_OPERATOR_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_MATRIX_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct TransposeParam : public dmlc::Parameter<TransposeParam> {
  TShape axes;
  DMLC_DECLARE_PARAMETER(TransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(TShape())
    .describe("Target axis order. By default the axes will be inverted.");
  }
};

template<typename xpu>
void TransposeImpl(RunContext ctx,
                   const TBlob& src,
                   const TBlob& ret,
                   const TShape& axes) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(src.type_flag_, ret.type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret.type_flag_, DType, {
    switch (axes.ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = src.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = ret.get<xpu, 1, DType>(s);
      Copy(out, in, s);
      break;
     }
     case 2: {
      mshadow::Tensor<xpu, 2, DType> in = src.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> out = ret.FlatTo2D<xpu, DType>(s);
      if (axes[0] == 1 && axes[1] == 0) {
        out = in.T();
      } else {
        Copy(out, in, s);
      }
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = src.get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = ret.get<xpu, 3, DType>(s);
      out = transpose(in, axes.get<3>());
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = src.get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = ret.get<xpu, 4, DType>(s);
      out = transpose(in, axes.get<4>());
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = src.get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = ret.get<xpu, 5, DType>(s);
      out = transpose(in, axes.get<5>());
      break;
     }
     default:
      LOG(FATAL) << "Transpose support at most 5 dimensions";
      break;
    }
  });
}

// matrix transpose
template<typename xpu>
void Transpose(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(req[0], kWriteTo) << "Transpose does not support inplace";
  if (param.axes.ndim() == 0) {
    TShape axes = TShape(inputs[0].ndim());
    for (index_t i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], axes);
  } else {
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], outputs[0], param.axes);
  }
}

inline bool TransposeShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& shp = (*in_attrs)[0];
  CHECK_LE(shp.ndim(), 5) << "Transpose support at most 5 dimensions";
  TShape ret(shp.ndim());
  if (param.axes.ndim() == 0) {
    for (index_t i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim()-1-i];
    }
  } else {
    CHECK_EQ(shp.ndim(), param.axes.ndim());
    for (index_t i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < shp.ndim());
      ret[i] = shp[param.axes[i]];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}


struct ExpandDimParam : public dmlc::Parameter<ExpandDimParam> {
  index_t axis;
  DMLC_DECLARE_PARAMETER(ExpandDimParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("Position (amongst axes) where new axis is to be inserted.");
  }
};


inline bool ExpandDimShape(const nnvm::NodeAttrs& attrs,
                           std::vector<TShape> *in_attrs,
                           std::vector<TShape> *out_attrs) {
  const ExpandDimParam& param = nnvm::get<ExpandDimParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& shp = (*in_attrs)[0];
  CHECK_LE(param.axis, shp.ndim())
      << "axis exceeds the dimension of the array";
  TShape ret(shp.ndim() + 1);
  for (index_t i = 0; i < param.axis; ++i) ret[i] = shp[i];
  ret[param.axis] = 1;
  for (index_t i = param.axis+1; i < ret.ndim(); ++i) ret[i] = shp[i-1];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}

template<typename xpu>
void DotForward_(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
      << "dot only support 32 bit float so far";

  if (inputs[0].ndim() == 2 && inputs[1].ndim() == 2) {
    mshadow::Tensor<xpu, 2, real_t> out = outputs[0].FlatTo2D<xpu, real_t>(s);
    ASSIGN_DISPATCH(out, req[0],
                    dot(inputs[0].get<xpu, 2, real_t>(s),
                        inputs[1].get<xpu, 2, real_t>(s)));
  } else {
    CHECK_NE(req[0], kAddTo) << "AddTo not yet suported";
    mshadow::Tensor<xpu, 1, real_t> out = outputs[0].get<xpu, 1, real_t>(s);
    mshadow::VectorDot(out,
                       inputs[0].get<xpu, 1, real_t>(s),
                       inputs[1].get<xpu, 1, real_t>(s));
  }
}

template<typename xpu>
void DotBackward_(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(req[0], kWriteInplace);
  CHECK_NE(req[1], kWriteInplace);

  if (inputs[1].ndim() == 2 && inputs[2].ndim() == 2) {
    mshadow::Tensor<xpu, 2, real_t> mout_grad = inputs[0].get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mlhs_data = inputs[1].get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mrhs_data = inputs[2].get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mlhs_grad = outputs[0].get<xpu, 2, real_t>(s);
    mshadow::Tensor<xpu, 2, real_t> mrhs_grad = outputs[1].get<xpu, 2, real_t>(s);
    ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mlhs_data.T(), mout_grad));
    ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mout_grad, mrhs_data.T()));
  } else {
    mshadow::Tensor<xpu, 1, real_t> mout_grad = inputs[0].get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mlhs_data = inputs[1].get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mrhs_data = inputs[2].get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mlhs_grad = outputs[0].get<xpu, 1, real_t>(s);
    mshadow::Tensor<xpu, 1, real_t> mrhs_grad = outputs[1].get<xpu, 1, real_t>(s);
    ASSIGN_DISPATCH(mrhs_grad, req[1],
                    broadcast_scalar(mout_grad, mlhs_data.shape_) * mlhs_data);
    ASSIGN_DISPATCH(mlhs_grad, req[0],
                    broadcast_scalar(mout_grad, mlhs_data.shape_) * mrhs_data);
  }
}

inline bool DotShape(const nnvm::NodeAttrs& attrs,
                     std::vector<TShape> *in_attrs,
                     std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& lshape = (*in_attrs)[0];
  TShape& rshape = (*in_attrs)[1];
  if (lshape.ndim() == 2 && rshape.ndim() == 2) {
    CHECK_EQ(lshape[1], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(lshape[0], rshape[1]));
  } else if (lshape.ndim() == 1 && rshape.ndim() == 1) {
    CHECK_EQ(lshape[0], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
  } else {
    LOG(FATAL) << "dot currently only support 2D*2D array or 1D*1D array"
               << lshape << " v.s. " << rshape;
    return false;
  }
  return true;
}

template<typename xpu>
void BatchDotForward_(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32)
      << "dot only support 32 bit float so far";

  mshadow::Tensor<xpu, 3, real_t> out = outputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mlhs = inputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mrhs = inputs[1].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 1, real_t*> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, real_t*>(mshadow::Shape1(3 * out.size(0)), s);
  if (kNullOp != req[0]) {
    mshadow::BatchGEMM<false, false>(out, mlhs, mrhs, 1.0f,
                                     (kAddTo == req[0]) ? 1.0f : 0.0f,
                                     workspace);
  }
}

template<typename xpu>
void BatchDotBackward_(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(req[1], kWriteInplace);
  CHECK_NE(req[0], kWriteInplace);

  mshadow::Tensor<xpu, 3, real_t> mout_grad = inputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mlhs_data = inputs[1].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mrhs_data = inputs[2].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mlhs_grad = outputs[0].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 3, real_t> mrhs_grad = outputs[1].get<xpu, 3, real_t>(s);
  mshadow::Tensor<xpu, 2, real_t*> workspace =
    ctx.requested[0].get_space_typed<xpu, 2, real_t*>(
      mshadow::Shape2(2, 3 * mout_grad.size(0)), s);
  mshadow::Tensor<xpu, 1, real_t*> rhs_workspace = workspace[0];
  mshadow::Tensor<xpu, 1, real_t*> lhs_workspace = workspace[1];
  if (kNullOp != req[1]) {
    mshadow::BatchGEMM<true, false>(mrhs_grad, mlhs_data, mout_grad, 1.0f,
                                    (kAddTo == req[1]) ? 1.0f : 0.0f,
                                    rhs_workspace);
  }
  if (kNullOp != req[0]) {
    mshadow::BatchGEMM<false, true>(mlhs_grad, mout_grad, mrhs_data, 1.0f,
                                    (kAddTo == req[0]) ? 1.0f : 0.0f,
                                    lhs_workspace);
  }
}

inline bool BatchDotShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& lshape = (*in_attrs)[0];
  TShape& rshape = (*in_attrs)[1];
  if (lshape.ndim() == 3 && rshape.ndim() == 3) {
    CHECK(lshape[0] == rshape[0] && lshape[2] == rshape[1])
      << "batch_dot shape error: " << lshape << " X " << rshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape3(lshape[0], lshape[1], rshape[2]));
  } else {
    LOG(FATAL) << "batch_dot currently only support 3D*3D array"
               << lshape << " v.s. " << rshape;
  }
  return true;
}


struct SimpleCropParam : public dmlc::Parameter<SimpleCropParam> {
  TShape begin, end;
  DMLC_DECLARE_PARAMETER(SimpleCropParam) {
    DMLC_DECLARE_FIELD(begin)
    .describe("starting coordinates");
    DMLC_DECLARE_FIELD(end)
    .describe("ending coordinates");
  }
};

// matrix crop for multi dimensional cropping: see also slice
template<typename xpu>
void Crop(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const SimpleCropParam& param = nnvm::get<SimpleCropParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    switch (inputs[0].ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = inputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
      out = slice(in, param.begin.get<1>(), param.end.get<1>());
      break;
     }
     case 2: {
      Tensor<xpu, 2, DType> in = inputs[0].get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = outputs[0].get<xpu, 2, DType>(s);
      out = slice(in, param.begin.get<2>(), param.end.get<2>());
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = inputs[0].get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
      out = slice(in, param.begin.get<3>(), param.end.get<3>());
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = inputs[0].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = outputs[0].get<xpu, 4, DType>(s);
      out = slice(in, param.begin.get<4>(), param.end.get<4>());
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = inputs[0].get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = outputs[0].get<xpu, 5, DType>(s);
      out = slice(in, param.begin.get<5>(), param.end.get<5>());
      break;
     }
     default:
      LOG(FATAL) << "crop supports at most 5 dimensions";
      break;
    }
  });
}

inline bool CropShape(const nnvm::NodeAttrs& attrs,
                      std::vector<TShape> *in_attrs,
                      std::vector<TShape> *out_attrs) {
  const SimpleCropParam& param = nnvm::get<SimpleCropParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& shp = (*in_attrs)[0];
  CHECK_EQ(shp.ndim(), param.begin.ndim());
  CHECK_EQ(shp.ndim(), param.end.ndim());
  TShape ret(shp.ndim());
  for (index_t i = 0; i < shp.ndim(); ++i) {
    CHECK(param.begin[i] <= shp[i]
          && param.end[i] <= shp[i]
          && param.begin[i] < param.end[i]);
    ret[i] = param.end[i] - param.begin[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return true;
}


struct SliceParam : public dmlc::Parameter<SliceParam> {
  int axis;
  int begin;
  int end;
  DMLC_DECLARE_PARAMETER(SliceParam) {
    DMLC_DECLARE_FIELD(axis).set_lower_bound(0)
      .describe("The axis to be sliced");
    DMLC_DECLARE_FIELD(begin).set_lower_bound(0)
      .describe("The beginning index to be sliced");
    DMLC_DECLARE_FIELD(end).set_lower_bound(0)
      .describe("The end index to be sliced");
  }
};

inline bool SliceShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape> *in_attrs,
                       std::vector<TShape> *out_attrs) {
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  TShape& ishape = (*in_attrs)[0];
  CHECK(param.axis < static_cast<int>(ishape.ndim())) <<
    "axis must be smaller than the source ndim! Recieved axis=" <<
      param.axis << ", src_ndim=" << ishape.ndim();
  int axis_size = static_cast<int>(ishape[param.axis]);
  CHECK_LE(param.end, axis_size);
  CHECK_LT(param.begin, param.end);

  TShape shape(ishape.ndim());
  for (index_t i = 0; i < ishape.ndim(); ++i) {
    if (static_cast<int>(i) == param.axis) {
      shape[i] = static_cast<index_t>(param.end - param.begin);
    } else {
      shape[i] = ishape[i];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  return true;
}


template<typename xpu>
void Slice(const nnvm::NodeAttrs& attrs,
           const OpContext& ctx,
           const std::vector<TBlob>& inputs,
           const std::vector<OpReqType>& req,
           const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int ndim = static_cast<int>(outputs[0].ndim());

  if (param.axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> in =
            inputs[0].FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> out =
            outputs[0].FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(out, req[0], slice<1>(in, param.begin, param.end));
      });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> in =
            inputs[0].FlatTo3D<xpu, DType>(param.axis, s);
        mshadow::Tensor<xpu, 3, DType> out =
            outputs[0].FlatTo3D<xpu, DType>(param.axis, s);
        ASSIGN_DISPATCH(out, req[0], slice<1>(in, param.begin, param.end));
      });
  }
}

// Backward pass of broadcast over the given axis
template<typename xpu>
void SliceGrad_(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  using namespace mshadow::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int ndim = static_cast<int>(outputs[0].shape_.ndim());

  if (param.axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> ograd =
            inputs[0].FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> igrad =
            outputs[0].FlatTo2D<xpu, DType>(s);
        if (req[0] == kAddTo) {
          slice<1>(igrad, param.begin, param.end) += F<identity>(ograd);
        } else if (req[0] == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, param.begin, param.end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req[0], kNullOp);
        }
      });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> ograd =
            inputs[0].FlatTo3D<xpu, DType>(param.axis, s);
        mshadow::Tensor<xpu, 3, DType> igrad =
            outputs[0].FlatTo3D<xpu, DType>(param.axis, s);
        if (req[0] == kAddTo) {
          slice<1>(igrad, param.begin, param.end) += F<identity>(ograd);
        } else if (req[0] == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, param.begin, param.end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req[0], kNullOp);
        }
      });
  }
}

struct FlipParam : public dmlc::Parameter<FlipParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(FlipParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("The dimension to flip");
  }
};

// matrix crop
template<typename xpu>
void Flip(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const FlipParam& param = nnvm::get<FlipParam>(attrs.parsed);
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    switch (inputs[0].shape_.ndim()) {
     case 0:
      break;
     case 1: {
      Tensor<xpu, 1, DType> in = inputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 2: {
      Tensor<xpu, 2, DType> in = inputs[0].get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = outputs[0].get<xpu, 2, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = inputs[0].get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = outputs[0].get<xpu, 3, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = inputs[0].get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = outputs[0].get<xpu, 4, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = inputs[0].get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = outputs[0].get<xpu, 5, DType>(s);
      out = flip(in, param.axis);
      break;
     }
     default:
      LOG(FATAL) << "flip supports at most 5 dimensions";
      break;
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_MATRIX_OP_INL_H_
