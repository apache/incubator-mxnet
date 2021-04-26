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
 *  Copyright (c) 2015 by Contributors
 * \file broadcast_reduce_op.h
 * \brief Function definition of broadcast and reduce operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_

#include <algorithm>
#include <vector>
#include <string>
#include "../../common/utils.h"
#include "../nn/moments-inl.h"
#include "../tensor/broadcast_reduce_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../../api/operator/op_utils.h"

namespace mxnet {
namespace op {

struct NumpyReduceAxesParam : public dmlc::Parameter<NumpyReduceAxesParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  dmlc::optional<int> dtype;
  bool keepdims;
  dmlc::optional<double> initial;
  DMLC_DECLARE_PARAMETER(NumpyReduceAxesParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("int8", mshadow::kInt8)
      .add_enum("int32", mshadow::kInt32)
      .add_enum("int64", mshadow::kInt64)
      .add_enum("bool", mshadow::kBool)
      .set_default(dmlc::optional<int>())
      .describe("The type of the returned array and of the accumulator in which the elements are "
                "summed. The dtype of a is used by default unless a has an integer dtype of less "
                "precision than the default platform integer. In that case, if a is signed then "
                "the platform integer is used while if a is unsigned then an unsigned integer of "
                "the same precision as the platform integer is used.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(initial).set_default(dmlc::optional<double>())
      .describe("Starting value for the sum.");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, dtype_s, keepdims_s, initial_s;
    axis_s << axis;
    dtype_s << dtype;
    keepdims_s << keepdims;
    initial_s << initial;
    (*dict)["axis"] = axis_s.str();
    if (dtype.has_value()) {
      (*dict)["dtype"] = MXNetTypeWithBool2String(dtype.value());
    } else {
      (*dict)["dtype"] = dtype_s.str();
    }
    (*dict)["keepdims"] = keepdims_s.str();
    (*dict)["initial"] = initial_s.str();
  }
};

struct NumpyReduceAxesNoDTypeParam : public dmlc::Parameter<NumpyReduceAxesNoDTypeParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  bool keepdims;
  dmlc::optional<double> initial;
  DMLC_DECLARE_PARAMETER(NumpyReduceAxesNoDTypeParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(initial).set_default(dmlc::optional<double>())
      .describe("Starting value for the sum.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, keepdims_s, initial_s;
    axis_s << axis;
    keepdims_s << keepdims;
    initial_s << initial;
    (*dict)["axis"] = axis_s.str();
    (*dict)["keepdims"] = keepdims_s.str();
    (*dict)["initial"] = initial_s.str();
  }
};

struct NumpyReduceAxesBoolParam : public dmlc::Parameter<NumpyReduceAxesBoolParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  bool keepdims;
  DMLC_DECLARE_PARAMETER(NumpyReduceAxesBoolParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, keepdims_s;
    axis_s << axis;
    keepdims_s << keepdims;
    (*dict)["axis"] = axis_s.str();
    (*dict)["keepdims"] = keepdims_s.str();
  }
};

inline TShape NumpyReduceAxesShapeImpl(const TShape& ishape,
                                       const dmlc::optional<mxnet::Tuple<int>>& axis,
                                       bool keepdims) {
  // If input is a scalar, output should be a scalar too
  if (ishape.ndim() == 0) {
    if (axis.has_value()) {
      const mxnet::Tuple<int>& axes = axis.value();
      if (axes.ndim() > 0) {
        CHECK_EQ(axes.ndim(), 1);
        CHECK(axes[0] == 0 || axes[0] == -1);
      }
    }
    return TShape(0, -1);
  }

  // axis=None, do global reduction
  if (!axis.has_value()) {
    if (keepdims) {
      return TShape(ishape.ndim(), 1);
    } else {
      return TShape(0, -1);
    }
  }

  // axis = (), will return identity(input)
  if (axis.value().ndim() == 0) {
    return ishape;
  }

  // axis has value
  mxnet::Tuple<int> axes(axis.value());
  for (index_t i = 0; i < axes.ndim(); i++) {
    if (axes[i] < 0) {
      axes[i] += ishape.ndim();
    }
  }
  std::sort(axes.begin(), axes.end());

  for (index_t i = 1; i < axes.ndim(); i++) {
    CHECK_LT(axes[i-1], axes[i])
        << "Reduction axes have duplicates "
        << axes;
  }
  CHECK_LT(axes[axes.ndim()-1], ishape.ndim())
      << "Reduction axis " << axes[axes.ndim()-1]
      << " Exceeds input dimensions " << ishape;
  CHECK_GE(axes[0], 0)
      << "Reduction axis " << axis.value()
      << " Exceeds input dimensions " << ishape;

  TShape oshape;
  if (keepdims) {
    oshape = TShape(ishape);
  } else {
    oshape = TShape(ishape.ndim() - axes.ndim(), -1);
  }

  if (keepdims) {
    for (index_t i = 0; i < axes.ndim(); ++i) {
      oshape[axes[i]] = 1;
    }
  } else {
    for (index_t i = 0, j = 0, k = 0; i < ishape.ndim(); ++i) {
      if (j < axes.ndim() && i == axes[j]) {
        ++j;
        continue;
      }
      oshape[k++] = ishape[i];
    }
  }
  return oshape;
}

inline bool NumpyReduceAxesShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims));
  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyReduceAxesBoolShape(const nnvm::NodeAttrs& attrs,
                                     std::vector<TShape> *in_attrs,
                                     std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyReduceAxesBoolParam& param = nnvm::get<NumpyReduceAxesBoolParam>(attrs.parsed);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims));
  return shape_is_known(out_attrs->at(0));
}

inline bool NumpyReduceAxesNoDTypeShape(const nnvm::NodeAttrs& attrs,
                                        std::vector<TShape> *in_attrs,
                                        std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const NumpyReduceAxesNoDTypeParam& param = nnvm::get<NumpyReduceAxesNoDTypeParam>(attrs.parsed);
  // check the case where the reduction axis should not be zero
  bool is_all_reducded_axes_not_zero = true;
  const TShape& ishape = (*in_attrs)[0];
  if (param.axis.has_value()) {
    const mxnet::Tuple<int>& axes = param.axis.value();
    for (int i = 0; i < axes.ndim(); ++i) {
      if ((axes[i] >= 0) && (ishape[axes[i]] == 0)) {
        is_all_reducded_axes_not_zero = false;
        break;
      }
    }
  } else {
    if (ishape.Size() == 0) {
      // global reduction should excuted only when input have size more than 0
      is_all_reducded_axes_not_zero = false;
    }
  }
  CHECK(is_all_reducded_axes_not_zero)
    << "zero-size array to reduction operation maximum which has no identity";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl((*in_attrs)[0], param.axis, param.keepdims));
  return shape_is_known(out_attrs->at(0));
}

template<bool safe_acc_hint = false>
inline bool NeedSafeAcc(int itype, int otype) {
  bool rule = (itype != otype) || (itype != mshadow::kFloat32 && itype != mshadow::kFloat64);
  return safe_acc_hint && rule;
}

namespace mxnet_op {
struct set_to_nan {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out) {
    out[i] = DType(nanf(""));
  }
};

}  // namespace mxnet_op

void TVMOpReduce(const OpContext& ctx, const TBlob& input,
                 const dmlc::optional<mxnet::Tuple<int>>& axis,
                 const TBlob& output, const OpReqType req, const std::string& reducer_name);

template<typename xpu, typename reducer, bool safe_acc_hint = false, bool normalize = false,
         typename OP = op::mshadow_op::identity>
void NumpyReduceAxesCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  if (req[0] == kNullOp) return;
  const auto& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  Stream<xpu>* s = ctx.get_stream<xpu>();
  if (outputs[0].shape_.Size() == 0) return;
  if (inputs[0].shape_.Size() == 0 && outputs[0].shape_.Size() != 0) {
    using namespace mxnet_op;
    if (normalize) {
      LOG(WARNING) << "WARNING: Mean of empty slice.";
      if (mxnet::common::is_float(outputs[0].type_flag_)) {
        MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
          Kernel<set_to_nan, xpu>::Launch(s, outputs[0].shape_.Size(),
                                          outputs[0].dptr<DType>());
        });
      } else {
        LOG(WARNING) << "WARNING: nan is outside the range of"<<
                        "representable values of type 'int'";
        MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
          Kernel<set_zero, xpu>::Launch(s, outputs[0].shape_.Size(),
                                        outputs[0].dptr<DType>());
        });
      }
    } else if (std::is_same<reducer, mshadow_op::sum>::value) {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Kernel<set_zero, xpu>::Launch(s, outputs[0].shape_.Size(),
                                      outputs[0].dptr<DType>());
      });
    } else {
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Kernel<set_one, xpu>::Launch(s, outputs[0].shape_.Size(),
                                     outputs[0].dptr<DType>());
      });
    }
    return;
  }
  CHECK_NE(req[0], kWriteInplace) << "Reduce does not support write in-place";
#if MXNET_USE_TVM_OP
  // If boolean ndarray, use the kernel generated by TVM
  if (inputs[0].type_flag_ == mshadow::kBool) {
    std::string reducer_name;
    if (std::is_same<reducer, mshadow_op::sum>::value) {
      reducer_name = "sum";
    } else {
      LOG(FATAL) << "Only reduce op: `sum` is supported for boolean ndarrays";
    }
    TVMOpReduce(ctx, inputs[0], param.axis, outputs[0], req[0], reducer_name);
    if (normalize) {
      using namespace mshadow::expr;
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
        auto out = outputs[0].FlatTo2D<xpu, OType>(s);
        out /= scalar<OType>(inputs[0].Size()/outputs[0].Size());
      });
    }
    return;
  }
#endif
  if (param.axis.has_value() && param.axis.value().ndim() == 0) {
    UnaryOp::IdentityCompute<xpu>(attrs, ctx, inputs, req, outputs);
  }
  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }

  if (NeedSafeAcc<safe_acc_hint>(inputs[0].type_flag_, outputs[0].type_flag_)) {
    ReduceAxesComputeImpl<xpu, reducer, true, normalize, OP>(ctx, inputs, req, outputs, small);
  } else {
    ReduceAxesComputeImpl<xpu, reducer, false, normalize, OP>(ctx, inputs, req, outputs, small);
  }
}

template<typename xpu, typename reducer, typename OP = op::mshadow_op::identity>
void NumpyReduceAxesNoDTypeCompute(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  const NumpyReduceAxesNoDTypeParam& param = nnvm::get<NumpyReduceAxesNoDTypeParam>(attrs.parsed);
  if (param.initial.has_value()) {
    LOG(FATAL) << "initial is not supported yet";
  }
  if (inputs[0].shape_.Size() == 0U || outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  if (param.axis.has_value() && param.axis.value().ndim() == 0) {
    UnaryOp::IdentityCompute<xpu>(attrs, ctx, inputs, req, outputs);
  }
  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }
  ReduceAxesComputeImpl<xpu, reducer, false, false, OP>(ctx, inputs, req, outputs, small);
}

template<typename xpu, typename reducer, typename OP = op::mshadow_op::NonZero, int init>
void NumpyReduceAxesBoolCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  const NumpyReduceAxesBoolParam& param = nnvm::get<NumpyReduceAxesBoolParam>(attrs.parsed);
  mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
  if (outputs[0].shape_.Size() == 0) return;
  if (inputs[0].shape_.Size() == 0 && outputs[0].shape_.Size() != 0) {
    using namespace mxnet_op;
    if (init == 0) {
      Kernel<set_false, xpu>::Launch(s, outputs[0].shape_.Size(), outputs[0].dptr<bool>());
    } else {
      Kernel<set_true, xpu>::Launch(s, outputs[0].shape_.Size(), outputs[0].dptr<bool>());
    }
    return;
  }
  if (param.axis.has_value() && param.axis.value().ndim() == 0) {
    UnaryOp::IdentityCompute<xpu>(attrs, ctx, inputs, req, outputs);
  }
  TShape small;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(inputs[0].shape_, param.axis, true);
  }
  ReduceAxesComputeBoolImpl<xpu, reducer, false, false, OP>(ctx, inputs, req, outputs, small);
}

template<typename xpu, typename reducer>
void NumpySearchAxisCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const ReduceAxisParam& param = nnvm::get<ReduceAxisParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int axis = inputs[0].ndim();
  TBlob input = inputs[0];
  if (param.axis.has_value()) {
    axis = param.axis.value();
  } else {
    // If global reduction, reshape the input tensor into 2D shape (1, inputs[0].shape_.Size())
    // and search on axis = 1.
    mxnet::TShape shape_2d(2, 1);
    shape_2d[1] = input.shape_.Size();
    input = TBlob(input.dptr_, shape_2d, input.dev_mask(), input.type_flag_, input.dev_id());
    axis = 1;
  }
  axis = CheckAxis(axis, input.shape_.ndim());
  if (inputs[0].shape_.ndim() != 0) {
    if (param.axis.has_value()) {
      // cannot do argmax in an empty dimension
      CHECK_NE(inputs[0].shape_[axis], 0)
          << "searching input tensor of shape " << inputs[0].shape_
          << " along axis = " << axis << " of zero dim-size is not allowed";
    } else {
      // cannot do argmax on an empty array
      CHECK_NE(inputs[0].shape_.Size(), 0U) << "attempt to search an empty sequence";
    }
  }
  if (input.shape_.Size() == 0U) return;  // zero-size tensor
  mxnet::TShape shape = AxisShapeCompact(input.shape_, &axis, false);
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, int64_t> out =
      outputs[0].get_with_shape<xpu, 2, int64_t>(Shape2(shape[0], shape[2]), s);
    Tensor<xpu, 3, DType> in =
      input.get_with_shape<xpu, 3, DType>(shape.get<3>(), s);
    CHECK(req[0] != kAddTo) << "AddTo is not supported";
    ASSIGN_DISPATCH(out, req[0], tcast<int64_t>(reduce_with_axis<reducer, true>(in, 1)));
  });
}

struct arg_min_max_parse {
  template <typename DType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  OType* out_data,
                                  const DType* in_data) {
    out_data[i] = in_data[i].idx;
  }
};

template <typename Reducer, int NDim, typename DType, typename OType>
void NumpyArgMinMaxReduce(mshadow::Stream<cpu> *s, const TBlob& in_data, const TBlob& out_data,
                          const mshadow::Tensor<cpu, 1, char>& workspace) {
  using namespace mshadow;
  Shape<NDim> rshape, rstride;
  broadcast::diff<NDim>(out_data.shape_.get<NDim>(), in_data.shape_.get<NDim>(), &rshape, &rstride);
  size_t N = out_data.shape_.Size(), M = rshape.Size();
  broadcast::seq_reduce_compute<Reducer, NDim, OType, DType, OType,
    mxnet::op::mshadow_op::identity,
    mxnet::op::mshadow_op::arg_min_max_set_index<OType, index_t>> (
    N, M, false, in_data.dptr<DType>(), static_cast<OType*>(out_data.dptr_),
    in_data.shape_.get<NDim>(), out_data.shape_.get<NDim>(), rshape, rstride);
}

template<typename Reducer, typename xpu, typename IType>
void NumpyArgMinMaxCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[0] == kNullOp) return;
  // parse param
  const auto& param = nnvm::get<ReduceAxisParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  TBlob out = outputs[0];
  TBlob in = inputs[0];
  // do some shape checks
  if (in.shape_.ndim() != 0) {
    if (param.axis.has_value()) {
      // cannot do argmax in an empty dimension
      int axis = param.axis.value();
      axis = CheckAxis(axis, in.shape_.ndim());
      CHECK_NE(in.shape_[axis], 0)
          << "searching input tensor of shape " << inputs[0].shape_
          << " along axis = " << axis << " of zero dim-size is not allowed";
    } else {
      // cannot do argmax on an empty array
      CHECK_NE(in.shape_.Size(), 0U) << "attempt to search an empty sequence";
    }
  }
  if (in.shape_.Size() == 0U) return;  // zero-size tensor
  // prepare shape
  dmlc::optional<mxnet::Tuple<int>> axes;
  if (param.axis.has_value()) {
    mxnet::Tuple<int> t({param.axis.value()});
    axes = dmlc::optional<mxnet::Tuple<int>>(t);
  }
  TShape small;
  small = NumpyReduceAxesShapeImpl(in.shape_, axes, true);
  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(in.shape_, small, &src_shape, &dst_shape);
  const TBlob in_data = in.reshape(src_shape);
  // request a work space
  size_t workspace_size = broadcast::ReduceWorkspaceSize(s, dst_shape, req[0], src_shape);
  MSHADOW_TYPE_SWITCH_WITH_BOOL(in.type_flag_, DType, {
    // define OType
    typedef mxnet::op::mshadow_op::IndexedNum<IType, DType> OType;
    // switch dim
    BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
      constexpr size_t align_size = 1024;
      const size_t aligned_first_workspace_size = ((workspace_size + align_size - 1) / align_size)
                                                  * align_size;
      workspace_size = aligned_first_workspace_size +
                       sizeof(OType) * out.shape_.Size();
      Tensor<xpu, 1, char> workspace =
                ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      // set up intermediate output
      TBlob intermediate = out;
      intermediate.dptr_ = reinterpret_cast<int64_t*>(workspace.dptr_ +
                                                      aligned_first_workspace_size);
      // reshape the input and intermediate output tensor
      const TBlob intermediate_out_data = intermediate.reshape(dst_shape);
      NumpyArgMinMaxReduce<Reducer, NDim, DType, OType>(s, in_data,
        intermediate_out_data, workspace);
      // parse the indices from the intermediate tensor back to the actual output tensor
      using namespace mxnet_op;
      Kernel<arg_min_max_parse, xpu>::Launch(
          s, out.shape_.Size(), outputs[0].dptr<int64_t>(),
          static_cast<OType*>(intermediate_out_data.dptr_));
    });
  });
}

#if MXNET_USE_CUDA

struct NumpyArgMinMaxRTCCompute {
  std::string reducer;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

#endif

template<typename xpu, bool normalize = false>
inline void NumpyReduceAxesBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                           const OpContext& ctx,
                                           const std::vector<TBlob>& inputs,
                                           const std::vector<OpReqType>& req,
                                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_NE(outputs[0].type_flag_, kBool) << "reduce operators do not support gradient calculation "
                                            "for input tensors of boolean type.";
  if (outputs[0].shape_.Size() == 0) return;
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
  }

  BroadcastComputeImpl<xpu>(attrs, ctx, inputs, req, outputs, small);

  if (normalize) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, IType, {
      Tensor<xpu, 1, IType> igrad = outputs[0].FlatTo1D<xpu, IType>(s);
      igrad /= scalar<IType>(outputs[0].Size()/inputs[0].Size());
    });
  }
}

template<typename xpu, typename OP, bool normalize = false>
void NumpyReduceAxesBackwardUseInOut(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const NumpyReduceAxesParam& param = nnvm::get<NumpyReduceAxesParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
  }
  ReduceAxesBackwardUseInOutImpl<xpu, OP, normalize>(ctx, small, inputs, req, outputs);
}

struct NumpyMomentsParam : public dmlc::Parameter<NumpyMomentsParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  dmlc::optional<int> dtype;
  bool keepdims;
  int ddof;
  DMLC_DECLARE_PARAMETER(NumpyMomentsParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a sum is performed. The default, axis=None, will sum "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(dtype)
      .add_enum("float16", mshadow::kFloat16)
      .add_enum("float32", mshadow::kFloat32)
      .add_enum("float64", mshadow::kFloat64)
      .add_enum("int8", mshadow::kInt8)
      .add_enum("int32", mshadow::kInt32)
      .add_enum("int64", mshadow::kInt64)
      .set_default(dmlc::optional<int>())
      .describe("The type of the returned array and of the accumulator in which the elements are "
                "summed. The dtype of a is used by default unless a has an integer dtype of less "
                "precision than the default platform integer. In that case, if a is signed then "
                "the platform integer is used while if a is unsigned then an unsigned integer of "
                "the same precision as the platform integer is used.");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(ddof).set_default(0)
      .describe("Starting value for the sum.");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, dtype_s, keepdims_s, ddof_s;
    axis_s << axis;
    keepdims_s << keepdims;
    ddof_s << ddof;
    (*dict)["axis"] = axis_s.str();
    dtype_s << dtype;
    if (dtype.has_value()) {
      (*dict)["dtype"] = MXNetTypeWithBool2String(dtype.value());
    } else {
      (*dict)["dtype"] = dtype_s.str();
    }
    (*dict)["keepdims"] = keepdims_s.str();
    (*dict)["ddof"] = ddof_s.str();
  }
};

struct NumpyWeightedAverageParam : public dmlc::Parameter<NumpyWeightedAverageParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  bool returned;
  bool weighted;

  DMLC_DECLARE_PARAMETER(NumpyWeightedAverageParam) {
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("Axis or axes along which a average is performed. "
                "The default, axis=None, will average "
                "all of the elements of the input array. If axis is negative it counts from the "
                "last to the first axis.");
    DMLC_DECLARE_FIELD(returned)
      .set_default(false)
      .describe("If True, the tuple (average, sum_of_weights) is returned,"
                "otherwise only the average is returned."
                "If weights=None, sum_of_weights is equivalent to"
                "the number of elements over which the average is taken.");
    DMLC_DECLARE_FIELD(weighted)
      .set_default(true)
      .describe("Auxiliary flag to deal with none weights.");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, returned_s, weighted_s;
    axis_s << axis;
    returned_s << returned;
    weighted_s << weighted;
    (*dict)["axis"] = axis_s.str();
    (*dict)["returned"] = returned_s.str();
    (*dict)["weighted"] = weighted_s.str();
  }
};

inline bool NumpyWeightedAverageShape(const nnvm::NodeAttrs& attrs,
                                      std::vector<TShape> *in_attrs,
                                      std::vector<TShape> *out_attrs) {
  const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), (param.weighted ? 2U : 1U));
  CHECK_EQ(out_attrs->size(), 2U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }

  const TShape& a_shape = (*in_attrs)[0];
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl(a_shape, param.axis, false));

  if (param.weighted) {
    const TShape& w_shape = (*in_attrs)[1];
    if (w_shape.ndim() != a_shape.ndim()) {
      CHECK_EQ(w_shape.ndim(), 1U)
        << "1D weights expected when shapes of a and weights differ.";
      CHECK_EQ(param.axis.has_value(), true)
        << "Axis must be specified when shapes of a and weights differ.";
      mxnet::Tuple<int> axes(param.axis.value());
      CHECK_EQ(axes.ndim(), 1U) << "Axis must be int when shapes of a and weights differ.";
      int red_axis = axes[0] < 0 ? axes[0] + a_shape.ndim() : axes[0];
      CHECK_EQ(a_shape[red_axis], w_shape[0])
        << "Length of weights not compatible with specified axis.";
      SHAPE_ASSIGN_CHECK(*out_attrs, 1,
                         NumpyReduceAxesShapeImpl(
                           w_shape, dmlc::optional<mxnet::Tuple<int>>(), false));
    } else {
      for (int i = 0; i < w_shape.ndim(); i++) {
        CHECK_EQ(w_shape[i], a_shape[i]);
      }
      SHAPE_ASSIGN_CHECK(*out_attrs, 1,
                         NumpyReduceAxesShapeImpl(w_shape, param.axis, false));
    }
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape(0, -1));
  }

  return shape_is_known(out_attrs->at(0)) && shape_is_known(out_attrs->at(1));
}

template<int req, int NDim, bool onedim = false>
struct avg_grad_a_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out,
                                  const DType* w,
                                  const DType* scl,
                                  const DType* ograd,
                                  mshadow::Shape<NDim> small,
                                  mshadow::Shape<NDim> big) {
    // partial a = w / sum(w)
    size_t big_idx = i;
    size_t small_idx = i;
    size_t big_stride = 1;
    size_t small_stride = 1;
    size_t red_axis_idx = 0;
    for (int axis = NDim-1; axis >= 0; --axis) {
      size_t axis_idx = big_idx % big[axis];
      small_idx -= axis_idx * big_stride;
      if (small[axis] != 1) {
        small_idx += axis_idx * small_stride;
      } else if (onedim && small[axis] != big[axis]) {
        red_axis_idx = axis_idx;
      }
      big_idx /= big[axis];
      big_stride *= big[axis];
      small_stride *= small[axis];
    }
    if (onedim) {
      KERNEL_ASSIGN(out[i], req, (ograd[small_idx] * (w[red_axis_idx] / *scl)));
    } else {
      KERNEL_ASSIGN(out[i], req, (ograd[small_idx] * (w[i] / scl[small_idx])));
    }
  }
};

template<int req, int NDim>
struct avg_grad_w_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out,
                                  const DType* a,
                                  const DType* scl,
                                  const DType* sum_of_wa,
                                  const DType* ograd,
                                  mshadow::Shape<NDim> small,
                                  mshadow::Shape<NDim> big) {
    // partial w = (a * sum(w) - sum(a*w)) / (sum(w) * sum(w))
    size_t big_idx = i;
    size_t small_idx = i;
    size_t big_stride = 1;
    size_t small_stride = 1;
    for (int axis = NDim-1; axis >= 0; --axis) {
      size_t axis_idx = big_idx % big[axis];
      small_idx -= axis_idx * big_stride;
      if (small[axis] != 1) {
        small_idx += axis_idx * small_stride;
      }
      big_idx /= big[axis];
      big_stride *= big[axis];
      small_stride *= small[axis];
    }
    DType ret = ograd[small_idx] *
      (((a[i] * scl[small_idx] - sum_of_wa[small_idx]) / scl[small_idx]) / scl[small_idx]);
    KERNEL_ASSIGN(out[i], req, ret);
  }
};

template<int req, int NDim>
struct avg_grad_w_1D_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out,
                                  const DType* a,
                                  const DType* scl,
                                  const DType* sum_of_wa,
                                  const DType* ograd,
                                  mshadow::Shape<NDim> big,
                                  const int red_axis) {
    DType scl_val = *scl;
    size_t tail = 1;
    size_t head = 1;
    for (int axis = NDim-1; axis > red_axis; --axis) {
      tail *= big[axis];
    }
    for (int axis = 0; axis < red_axis; ++axis) {
      head *= big[axis];
    }
    DType ret = 0;
    for (size_t j = 0; j < head; ++j) {
      for (size_t k = 0; k < tail; ++k) {
        size_t a_idx = j*(tail*big[red_axis]) + i * tail + k;
        size_t small_idx = j*tail + k;
        ret += (ograd[small_idx] *
          (((a[a_idx] * scl_val - sum_of_wa[small_idx]) / scl_val) / scl_val));
      }
    }
    KERNEL_ASSIGN(out[i], req, ret);
  }
};

template<typename xpu, bool back = false>
void NumpyWeightedAverageComputeImpl(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs,
                                     const dmlc::optional<mxnet::Tuple<int>>& axis) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  TShape small1 = NumpyReduceAxesShapeImpl(data.shape_, axis, true);
  // Reshape weights
  TShape small2 = small1;
  TBlob weights = inputs[1];

  bool one_dim = weights.shape_.ndim() != data.shape_.ndim();

  int red_axis = -1;

  if (one_dim) {
    CHECK_EQ(weights.shape_.ndim(), 1U)
      << "1D weights expected when shapes of a and weights differ.";
    CHECK_EQ(axis.has_value(), true)
      << "Axis must be specified when shapes of a and weights differ.";
    Tuple<int> axes(axis.value());
    CHECK_EQ(axes.ndim(), 1U)
      << "Axis must be int when shapes of a and weights differ.";
    red_axis = axes[0] < 0 ? axes[0] + data.shape_.ndim() : axes[0];
    CHECK_EQ(weights.shape_[0], data.shape_[red_axis])
      << "Length of weights not compatible with specified axis.";
    TShape new_w_shape(data.shape_.ndim(), 1);
    new_w_shape[red_axis] = weights.shape_[0];
    weights = weights.reshape(new_w_shape);
    small2 = TShape(new_w_shape.ndim(), 1);
  }
  TBlob wa;
  TBlob sum_of_wa;
  Tensor<xpu, 1, char> workspace;
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    // Get temp space
    size_t temp_data_size = data.shape_.Size() * sizeof(DType);
    size_t temp_sum_size = small1.Size() * sizeof(DType);
    TShape src_shape, dst_shape;
    BroadcastReduceShapeCompact(data.shape_, small1, &src_shape, &dst_shape);
    size_t workspace_size = 0;
    workspace_size = broadcast::ReduceWorkspaceSize(
      s, dst_shape, {kWriteTo}, src_shape);
    size_t temp_mem_size = temp_data_size + temp_sum_size + workspace_size;
    Tensor<xpu, 1, char> temp_mem =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
    auto *temp_data_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
    auto *temp_sum_ptr = reinterpret_cast<DType*>(temp_mem.dptr_ + temp_data_size);
    char *workspace_ptr = temp_mem.dptr_ + temp_data_size + temp_sum_size;
    workspace = Tensor<xpu, 1, char>(workspace_ptr, Shape1(workspace_size), s);

    // Compute weighted data
    wa = TBlob(temp_data_ptr, data.shape_, xpu::kDevMask);
    sum_of_wa = TBlob(temp_sum_ptr, small1, xpu::kDevMask);
  });
#if !defined(__CUDACC__)
  BinaryBroadcastCompute<xpu, mshadow_op::mul>(
    attrs, ctx, {data, weights}, {kWriteTo}, {wa});

  // Compute sum of weighted data
  ReduceAxesComputeImpl<xpu, mshadow_op::sum, true>(
    ctx, {wa}, {kWriteTo}, {sum_of_wa}, small1, &workspace);
#else
  BinaryBroadcastRTCCompute {"mul"}(attrs, ctx, {data, weights}, {kWriteTo}, {wa});

  // Compute sum of weighted data
  ReduceAxesRTCComputeImpl(ctx, {wa}, {kWriteTo}, {sum_of_wa}, small1, "red::sum{}",
                           &workspace, false, "identity");
#endif
  if (!back) {
    const TBlob& avg = outputs[0];
    const TBlob& sum_of_weights = outputs[1];
    // Compute sum of weight
    TBlob scl = sum_of_weights.reshape(small2);
#if !defined(__CUDACC__)
    ReduceAxesComputeImpl<xpu, mshadow_op::sum, true>(
      ctx, {weights}, {kWriteTo}, {scl}, small2, &workspace);
    // Compute avg and assign output
    BinaryBroadcastCompute<xpu, mshadow_op::div>(
      attrs, ctx, {sum_of_wa, scl}, req, {avg.reshape(small1)});
#else
    ReduceAxesRTCComputeImpl(ctx, {weights}, {kWriteTo}, {scl}, small2, "red::sum{}",
                             &workspace, false, "identity");
    // Compute avg and assign output
    BinaryBroadcastRTCCompute {"div"}(
      attrs, ctx, {sum_of_wa, scl}, req, {avg.reshape(small1)});
#endif
  } else {
    MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
      // Compute and assign the derivatives of a and weights
      const TBlob& igrad_a = outputs[0];
      const TBlob& igrad_w = outputs[1];
      const TBlob& scl = inputs[2];
      const TBlob& ograd = inputs[3];
      MXNET_NDIM_SWITCH(igrad_a.shape_.ndim(), NDim, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_a, {
          if (one_dim) {
            // 1D weights
            Kernel<avg_grad_a_kernel<req_a, NDim, true>, xpu>::Launch(
                s, igrad_a.shape_.Size(), igrad_a.dptr<DType>(),
                weights.dptr<DType>(), scl.dptr<DType>(), ograd.dptr<DType>(),
                small1.get<NDim>(),
                igrad_a.shape_.get<NDim>());
          } else {
            Kernel<avg_grad_a_kernel<req_a, NDim, false>, xpu>::Launch(
                s, igrad_a.shape_.Size(), igrad_a.dptr<DType>(),
                weights.dptr<DType>(), scl.dptr<DType>(), ograd.dptr<DType>(),
                small1.get<NDim>(),
                igrad_a.shape_.get<NDim>());
          }
        });
        MXNET_ASSIGN_REQ_SWITCH(req[1], req_w, {
          if (one_dim) {
            Kernel<avg_grad_w_1D_kernel<req_w, NDim>, xpu>::Launch(
                s, igrad_w.shape_.Size(), igrad_w.dptr<DType>(),
                data.dptr<DType>(), scl.dptr<DType>(), sum_of_wa.dptr<DType>(), ograd.dptr<DType>(),
                data.shape_.get<NDim>(),
                red_axis);
          } else {
            Kernel<avg_grad_w_kernel<req_w, NDim>, xpu>::Launch(
                s, igrad_w.shape_.Size(), igrad_w.dptr<DType>(),
                data.dptr<DType>(), scl.dptr<DType>(), sum_of_wa.dptr<DType>(), ograd.dptr<DType>(),
                small1.get<NDim>(),
                igrad_w.shape_.get<NDim>());
          }
        });
      })
    });
  }
}

template<typename xpu>
void NumpyWeightedAverageForward(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kWriteInplace) << "Average does not support write in-place";
  const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
  const TBlob& data = inputs[0];
  TShape small;
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    if (!param.weighted) {
      small = NumpyReduceAxesShapeImpl(data.shape_, param.axis, true);
      // Compute sum of weights which equals to the product of sizes of reduced axes
      Stream<xpu>* s = ctx.get_stream<xpu>();
      auto ret = outputs[1].FlatTo1D<xpu, DType>(s);
      ret = scalar<DType>(data.shape_.Size()/small.Size());
    }
  });
  if (!param.weighted) {
    // Compute mean
#if !defined(__CUDACC__)
    ReduceAxesComputeImpl<xpu, mshadow_op::sum, true, true>(
      ctx, inputs, req, {outputs[0]}, small);
#else
    ReduceAxesRTCComputeImpl(ctx, inputs, req, {outputs[0]}, small,
                             "red::sum{}", nullptr, true);
#endif
  } else {
    NumpyWeightedAverageComputeImpl<xpu>(
      attrs, ctx, inputs, req, outputs, param.axis);
  }
}

template<typename xpu>
void NumpyWeightedAverageBackward(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const auto& param = nnvm::get<NumpyWeightedAverageParam>(attrs.parsed);
  if (req[0] == kNullOp && !param.weighted) return;
  CHECK_EQ(inputs.size(), (param.weighted ? 6U : 5U));
  CHECK_EQ(outputs.size(), (param.weighted ? 2U : 1U));
  const TBlob& ograd = inputs[0];
  const TBlob& data = inputs[2];
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    if (!param.weighted) {
      TShape small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
      Stream<xpu>* s = ctx.get_stream<xpu>();
      auto ograd_tensor = ograd.FlatTo1D<xpu, DType>(s);
      ograd_tensor /= scalar<DType>(data.shape_.Size()/small.Size());
      BroadcastComputeImpl<xpu>(attrs, ctx, {ograd}, req, {outputs[0]}, small);
    } else {
      const TBlob& weights = inputs[3];
      const TBlob& scl = inputs[5];
      NumpyWeightedAverageComputeImpl<xpu, true>(
        attrs, ctx, {data, weights, scl, ograd}, req, outputs, param.axis);
    }
  });
}

template<typename xpu, bool sqrt>
void NumpyMomentsForward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  using namespace mxnet_op;

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(req.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);

  const NumpyMomentsParam& param = nnvm::get<NumpyMomentsParam>(attrs.parsed);

  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& data = inputs[0];
  const TBlob& moment = outputs[0];
  const TBlob& mean = outputs[1];

  mxnet::TShape small;
  if (param.keepdims) {
    small = moment.shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(data.shape_, param.axis, true);
  }

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(data.shape_, small, &src_shape, &dst_shape);

  // Get workspace and temp space for data - mean
  size_t workspace_size = broadcast::ReduceWorkspaceSize(s, dst_shape, req[0], src_shape);
  size_t temp_data_size = data.shape_.Size() * common::mshadow_type_info(inputs[0].type_flag_).size;
  size_t temp_mem_size = temp_data_size + workspace_size;
  Tensor<xpu, 1, char> temp_mem =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
  char *workspace_ptr = temp_mem.dptr_ + temp_data_size;
  Tensor<xpu, 1, char> workspace(workspace_ptr, Shape1(workspace_size), s);
  // Compute mean
#if !defined(__CUDACC__)
  ReduceAxesComputeImpl<xpu, mshadow_op::sum, true, true>(
    ctx, inputs, {kWriteTo}, {mean}, small, &workspace);
#else
  ReduceAxesRTCComputeImpl(ctx, inputs, {kWriteTo}, {mean}, small, "red::sum{}",
                           &workspace, true, "identity");
#endif
  // Compute data - mean
  Shape<6> data_shape, mean_shape;
  for (int i = 0; i < 6; ++i) {
    data_shape[i] = (i < data.shape_.ndim()) ? data.shape_[i] : 1;
    mean_shape[i] = (i < small.ndim()) ? small[i] : 1;
  }
#if !defined(__CUDACC__)
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      DType *temp_data_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
      Kernel<VarBroadcastKernel, xpu>::Launch(s, data_shape.Size(), temp_data_ptr,
        data.dptr<DType>(), mean.dptr<DType>(), data_shape, mean_shape);
      Tensor<xpu, 1, DType> temp_data_tensor(temp_data_ptr, Shape1(data.shape_.Size()), s);
      TBlob temp_data_blob = TBlob(temp_data_tensor).reshape(data.shape_);
      ReduceAxesComputeImpl<xpu, mshadow_op::sum, true, true>(
        ctx, {temp_data_blob}, {req[0]}, {moment}, small, &workspace, param.ddof);
      if (sqrt && req[0] != kNullOp) {
        Tensor<xpu, 1, OType> moment_tensor = moment.FlatTo1D<xpu, OType>(s);
        moment_tensor = F<mshadow_op::square_root>(moment_tensor);
      }
    });
  });
#else
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      DType *temp_data_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
      Kernel<VarBroadcastKernel, xpu>::Launch(s, data_shape.Size(), temp_data_ptr,
        data.dptr<DType>(), mean.dptr<DType>(), data_shape, mean_shape);
      Tensor<xpu, 1, DType> temp_data_tensor(temp_data_ptr, Shape1(data.shape_.Size()), s);
      TBlob temp_data_blob = TBlob(temp_data_tensor).reshape(data.shape_);
      ReduceAxesRTCComputeImpl(ctx, {temp_data_blob}, {req[0]}, {moment}, small,
                               "red::sum{}", &workspace, true, "identity", param.ddof);
      if (sqrt && req[0] != kNullOp) {
        UnaryRTCCompute {"sqrt"}({}, ctx, {moment}, {kWriteInplace}, {moment});
      }
  });
#endif
}

template<typename xpu>
void NumpyBroadcastToForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  TShape expanded_ishape(outputs[0].shape_.ndim(), 1);
  const TShape& ishape = inputs[0].shape_;
  CHECK_LE(ishape.ndim(), expanded_ishape.ndim()) << "output ndim cannot be less than input ndim";
  const int ndim_delta = expanded_ishape.ndim() - ishape.ndim();
  for (int i = 0; i < ishape.ndim(); ++i) {
    expanded_ishape[i + ndim_delta] = ishape[i];
  }
  BroadcastComputeImpl<xpu>(attrs, ctx, {inputs[0].reshape(expanded_ishape)},
                            req, outputs, expanded_ishape);
}

template<typename xpu>
void NumpyBroadcastToBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  if (inputs[0].shape_.Size() == 0U) return;  // zero-size ograd
  TShape expanded_igrad_shape(inputs[0].shape_.ndim(), 1);
  const TShape& igrad_shape = outputs[0].shape_;
  CHECK_LE(igrad_shape.ndim(), expanded_igrad_shape.ndim())
      << "output ndim cannot be less than input ndim";
  const int ndim_delta = expanded_igrad_shape.ndim() - igrad_shape.ndim();
  for (int i = 0; i < igrad_shape.ndim(); ++i) {
    expanded_igrad_shape[i + ndim_delta] = igrad_shape[i];
  }
#if !defined(__CUDACC__)
  if (NeedSafeAcc<true>(inputs[0].type_flag_, outputs[0].type_flag_)) {
    ReduceAxesComputeImpl<xpu, mshadow_op::sum, true>(
        ctx, inputs, req, {outputs[0].reshape(expanded_igrad_shape)}, expanded_igrad_shape);
  } else {
    ReduceAxesComputeImpl<xpu, mshadow_op::sum, false>(
        ctx, inputs, req, {outputs[0].reshape(expanded_igrad_shape)}, expanded_igrad_shape);
  }
#else
  ReduceAxesRTCComputeImpl(ctx, inputs, req, {outputs[0].reshape(expanded_igrad_shape)},
                           expanded_igrad_shape, "red::sum{}", nullptr, false);
#endif
}

template<typename xpu, typename OP>
void NumpyReduceAxesNoDTypeBackward(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const NumpyReduceAxesNoDTypeParam& param = nnvm::get<NumpyReduceAxesNoDTypeParam>(attrs.parsed);
  TShape small;
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = NumpyReduceAxesShapeImpl(outputs[0].shape_, param.axis, true);
  }
  ReduceAxesBackwardUseInOutImpl<xpu, OP, false>(ctx, small, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NUMPY_NP_BROADCAST_REDUCE_OP_H_
