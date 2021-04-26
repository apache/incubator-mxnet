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
 * Copyright (c) 2017 by Contributors
 * \file np_where_op.cc
 * \brief Function definition of numpy operator where
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_WHERE_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_WHERE_OP_INL_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "../../common/utils.h"
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct NumpyWhereScalarParam : public dmlc::Parameter<NumpyWhereScalarParam> {
  double scalar;
  DMLC_DECLARE_PARAMETER(NumpyWhereScalarParam) {
    DMLC_DECLARE_FIELD(scalar)
    .set_default(0.0)
    .describe("The scalar value of x/y.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream scalar_s;
    scalar_s << scalar;
    (*dict)["scalar"] = scalar_s.str();
  }
};

struct NumpyWhereScalar2Param : public dmlc::Parameter<NumpyWhereScalar2Param> {
  double x, y;
  DMLC_DECLARE_PARAMETER(NumpyWhereScalar2Param) {
    DMLC_DECLARE_FIELD(x)
    .set_default(0.0)
    .describe("The scalar value of x.");
    DMLC_DECLARE_FIELD(y)
    .set_default(0.0)
    .describe("The scalar value of y.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream x_s, y_s;
    x_s << x;
    y_s << y;
    (*dict)["x"] = x_s.str();
    (*dict)["y"] = y_s.str();
  }
};

template<int ndim>
struct numpy_where_kernel {
  template<typename CType, typename DType>
  MSHADOW_XINLINE static void Map(index_t base, OpReqType req, const Shape<ndim> &cstride,
                                  const Shape<ndim> &xstride, const Shape<ndim> &ystride,
                                  const Shape<ndim> &oshape, CType *datac, DType *datax,
                                  DType *datay, DType *out) {
    Shape<ndim> coord = mxnet_op::unravel(base, oshape);
    auto cidx = static_cast<index_t>(mxnet_op::dot(coord, cstride));
    auto xidx = static_cast<index_t>(mxnet_op::dot(coord, xstride));
    auto yidx = static_cast<index_t>(mxnet_op::dot(coord, ystride));
    KERNEL_ASSIGN(out[base], req, datac[cidx] != CType(0) ? datax[xidx] : datay[yidx]);
  }
};

template<int ndim, bool is_left>
struct numpy_where_backward_kernel {
  template<typename CType, typename DType>
  MSHADOW_XINLINE static void Map(index_t base, OpReqType req,
                                  const Shape<ndim> &cstride, const Shape<ndim> &oshape,
                                  CType *datac, DType *datao, DType *grad) {
    Shape<ndim> coord = mxnet_op::unravel(base, oshape);
    auto cidx = static_cast<index_t>(mxnet_op::dot(coord, cstride));
    if (is_left) {
      KERNEL_ASSIGN(grad[base], req, datac[cidx] != CType(0) ? datao[base] : DType(0));
    } else {
      KERNEL_ASSIGN(grad[base], req, datac[cidx] == CType(0) ? datao[base] : DType(0));
    }
  }
};

template<int ndim, bool is_left>
struct numpy_where_scalar_kernel {
  template<typename CType, typename DType>
  MSHADOW_XINLINE static void Map(index_t base, OpReqType req, const Shape<ndim> &cstride,
                                  const Shape<ndim> &ystride, const Shape<ndim> &oshape,
                                  CType *datac, DType datax, DType *datay, DType *out) {
    Shape<ndim> coord = mxnet_op::unravel(base, oshape);
    auto cidx = static_cast<index_t>(mxnet_op::dot(coord, cstride));
    auto yidx = static_cast<index_t>(mxnet_op::dot(coord, ystride));
    if (is_left) {
      KERNEL_ASSIGN(out[base], req, datac[cidx] != CType(0) ? datax : datay[yidx]);
    } else {
      KERNEL_ASSIGN(out[base], req, datac[cidx] != CType(0) ? datay[yidx] : datax);
    }
  }
};

struct numpy_where_scalar2_kernel {
  template<typename DType, typename CType>
  MSHADOW_XINLINE static void Map(index_t i, OpReqType req, DType* out, const CType* cond,
                                  const DType x, const DType y) {
    KERNEL_ASSIGN(out[i], req, (CType(0) != cond[i]? x : y));
  }
};

template<typename xpu>
inline void NumpyWhereOpForward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  CHECK_LE(outputs[0].shape_.ndim(), broadcast::MAX_DIM);

  const TBlob& cond = inputs[0];
  const TBlob& x = inputs[1];
  const TBlob& y = inputs[2];
  const TBlob& out = outputs[0];
  Stream<xpu> *s = ctx.get_stream<xpu>();
  std::vector<Shape<broadcast::MAX_DIM>> in_strides;
  in_strides.resize(3);
  for (int i = 0; i < 3; ++i) {
    TShape expanded_ishape(broadcast::MAX_DIM, 1);
    const TShape& ishape = inputs[i].shape_;
    const int ndim_delta = expanded_ishape.ndim() - ishape.ndim();
    for (int j = 0; j < ishape.ndim(); ++j) {
      expanded_ishape[j + ndim_delta] = ishape[j];
    }
    in_strides[i] = mxnet_op::calc_stride(expanded_ishape.get<broadcast::MAX_DIM>());
  }
  TShape expanded_oshape(broadcast::MAX_DIM, 1);
  const int ndim_delta = expanded_oshape.ndim() - out.shape_.ndim();
  for (int j = 0; j < out.shape_.ndim(); ++j) {
    expanded_oshape[j + ndim_delta] = (out.shape_)[j];
  }
  Shape<broadcast::MAX_DIM> oshape = expanded_oshape.get<broadcast::MAX_DIM>();
  MSHADOW_TYPE_SWITCH_WITH_BOOL(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(cond.type_flag_, CType, {
      mxnet_op::Kernel<numpy_where_kernel<broadcast::MAX_DIM>, xpu>::Launch(
        s, out.Size(), req[0],
        in_strides[0], in_strides[1], in_strides[2], oshape,
        cond.dptr<CType>(), x.dptr<DType>(),
        y.dptr<DType>(), out.dptr<DType>());
    });
  });
}

#if !defined(__CUDACC__)
#define NP_WHERE_REDUCE_AXES(safe_acc, ...) \
  ReduceAxesComputeImpl<xpu, mshadow_op::sum, safe_acc>(__VA_ARGS__)
#else
#define NP_WHERE_REDUCE_AXES(safe_acc, ...) ReduceAxesRTCComputeImpl(__VA_ARGS__, "red::sum{}")
#endif

template<typename xpu>
inline void NumpyWhereOpBackward(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK(common::is_float(inputs[0].type_flag_)) << "Backward only supports float types!";
  if (inputs[0].shape_.Size() == 0U) return;  // zero-size tensor

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& ograd = inputs[0];
  const TBlob& cond = inputs[1];
  const TBlob& dx = outputs[0];
  const TBlob& dy = outputs[1];
  // get expanded oshape
  TShape expanded_oshape(broadcast::MAX_DIM, 1);
  int ndim_delta = expanded_oshape.ndim() - ograd.shape_.ndim();
  for (int j = 0; j < ograd.shape_.ndim(); ++j) {
    expanded_oshape[j + ndim_delta] = (ograd.shape_)[j];
  }
  Shape<broadcast::MAX_DIM> oshape = expanded_oshape.get<broadcast::MAX_DIM>();
  // get cond stride
  TShape expanded_cshape(broadcast::MAX_DIM, 1);
  ndim_delta = expanded_cshape.ndim() - cond.shape_.ndim();
  for (int j = 0; j < cond.shape_.ndim(); ++j) {
    expanded_cshape[j + ndim_delta] = (cond.shape_)[j];
  }
  Shape<broadcast::MAX_DIM> cstride =
      mxnet_op::calc_stride(expanded_cshape.get<broadcast::MAX_DIM>());
  // get expanded lshape
  TShape expanded_lshape(broadcast::MAX_DIM, 1);
  ndim_delta = expanded_lshape.ndim() - dx.shape_.ndim();
  for (int j = 0; j < dx.shape_.ndim(); ++j) {
    expanded_lshape[j + ndim_delta] = (dx.shape_)[j];
  }
  // get expanded rshape
  TShape expanded_rshape(broadcast::MAX_DIM, 1);
  ndim_delta = expanded_rshape.ndim() - dy.shape_.ndim();
  for (int j = 0; j < dy.shape_.ndim(); ++j) {
    expanded_rshape[j + ndim_delta] = (dy.shape_)[j];
  }

  MSHADOW_TYPE_SWITCH_WITH_BOOL(ograd.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(cond.type_flag_, CType, {
      Tensor<xpu, 1, char> largespace;
      Tensor<xpu, broadcast::MAX_DIM, DType> workspace;
      size_t ws_size = 0;
      if (ograd.shape_ != dx.shape_ || ograd.shape_ != dy.shape_) {
        size_t ws_size1 = broadcast::ReduceWorkspaceSize(
            s, expanded_lshape, req[0], expanded_oshape);
        size_t ws_size2 = broadcast::ReduceWorkspaceSize(
            s, expanded_rshape, req[1], expanded_oshape);
        ws_size = std::max(ws_size1, ws_size2);
      }
      // process left output
      if (ograd.shape_ == dx.shape_) {
        mxnet_op::Kernel<numpy_where_backward_kernel<broadcast::MAX_DIM, true>, xpu>::Launch(
          s, ograd.Size(), req[0], cstride, oshape,
          cond.dptr<CType>(), ograd.dptr<DType>(), dx.dptr<DType>());
      } else {
        largespace = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(ograd.shape_.Size() * sizeof(DType) + ws_size), s);
        workspace = Tensor<xpu, broadcast::MAX_DIM, DType>(
            reinterpret_cast<DType*>(largespace.dptr_ + ws_size),
            expanded_oshape.get<broadcast::MAX_DIM>(), s);
        mxnet_op::Kernel<numpy_where_backward_kernel<broadcast::MAX_DIM, true>, xpu>::Launch(
          s, ograd.Size(), req[0], cstride, oshape,
          cond.dptr<CType>(), ograd.dptr<DType>(), workspace.dptr_);
        if (NeedSafeAcc<true>(dx.type_flag_, dx.type_flag_)) {
          NP_WHERE_REDUCE_AXES(true, ctx, {TBlob(workspace)}, {req[0]},
              {dx.reshape(expanded_lshape)}, expanded_lshape);
        } else {
          NP_WHERE_REDUCE_AXES(false, ctx, {TBlob(workspace)}, {req[0]},
              {dx.reshape(expanded_lshape)}, expanded_lshape);
        }
      }
      // process right output
      if (ograd.shape_ == dy.shape_) {
        mxnet_op::Kernel<numpy_where_backward_kernel<broadcast::MAX_DIM, false>, xpu>::Launch(
          s, ograd.Size(), req[1], cstride, oshape,
          cond.dptr<CType>(), ograd.dptr<DType>(), dy.dptr<DType>());
      } else {
        largespace = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(ograd.shape_.Size() * sizeof(DType) + ws_size), s);
        workspace = Tensor<xpu, broadcast::MAX_DIM, DType>(
            reinterpret_cast<DType*>(largespace.dptr_ + ws_size),
            expanded_oshape.get<broadcast::MAX_DIM>(), s);
        mxnet_op::Kernel<numpy_where_backward_kernel<broadcast::MAX_DIM, false>, xpu>::Launch(
          s, ograd.Size(), req[1], cstride, oshape,
          cond.dptr<CType>(), ograd.dptr<DType>(), workspace.dptr_);
        if (NeedSafeAcc<true>(dy.type_flag_, dy.type_flag_)) {
          NP_WHERE_REDUCE_AXES(true, ctx, {TBlob(workspace)}, {req[1]},
              {dy.reshape(expanded_rshape)}, expanded_rshape);
        } else {
          NP_WHERE_REDUCE_AXES(false, ctx, {TBlob(workspace)}, {req[1]},
              {dy.reshape(expanded_rshape)}, expanded_rshape);
        }
      }
    });
  });
}

template<typename xpu, bool is_left>
inline void NumpyWhereScalarOpForward(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  CHECK_LE(outputs[0].shape_.ndim(), broadcast::MAX_DIM);

  const NumpyWhereScalarParam& param = nnvm::get<NumpyWhereScalarParam>(attrs.parsed);
  const TBlob& cond = inputs[0];
  const TBlob& y = inputs[1];
  const TBlob& out = outputs[0];
  Stream<xpu> *s = ctx.get_stream<xpu>();
  std::vector<Shape<broadcast::MAX_DIM>> in_strides;
  in_strides.resize(2);
  for (int i = 0; i < 2; ++i) {
    TShape expanded_ishape(broadcast::MAX_DIM, 1);
    const TShape& ishape = inputs[i].shape_;
    const int ndim_delta = expanded_ishape.ndim() - ishape.ndim();
    for (int j = 0; j < ishape.ndim(); ++j) {
      expanded_ishape[j + ndim_delta] = ishape[j];
    }
    in_strides[i] = mxnet_op::calc_stride(expanded_ishape.get<broadcast::MAX_DIM>());
  }
  TShape expanded_oshape(broadcast::MAX_DIM, 1);
  const int ndim_delta = expanded_oshape.ndim() - out.shape_.ndim();
  for (int j = 0; j < out.shape_.ndim(); ++j) {
    expanded_oshape[j + ndim_delta] = (out.shape_)[j];
  }
  Shape<broadcast::MAX_DIM> oshape = expanded_oshape.get<broadcast::MAX_DIM>();
  MSHADOW_TYPE_SWITCH_WITH_BOOL(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(cond.type_flag_, CType, {
      mxnet_op::Kernel<numpy_where_scalar_kernel<broadcast::MAX_DIM, is_left>, xpu>::Launch(
        s, out.Size(), req[0],
        in_strides[0], in_strides[1], oshape,
        cond.dptr<CType>(), DType(param.scalar),
        y.dptr<DType>(), out.dptr<DType>());
    });
  });
}

template<typename xpu, bool is_lscalar>
inline void NumpyWhereScalarOpBackward(const nnvm::NodeAttrs& attrs,
                                       const OpContext& ctx,
                                       const std::vector<TBlob>& inputs,
                                       const std::vector<OpReqType>& req,
                                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK(common::is_float(inputs[0].type_flag_)) << "Backward only supports float types!";
  if (inputs[0].shape_.Size() == 0U) return;  // zero-size tensor

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& ograd = inputs[0];
  const TBlob& cond = inputs[1];
  const TBlob& dx = outputs[0];
  // get expanded oshape
  TShape expanded_oshape(broadcast::MAX_DIM, 1);
  int ndim_delta = expanded_oshape.ndim() - ograd.shape_.ndim();
  for (int j = 0; j < ograd.shape_.ndim(); ++j) {
    expanded_oshape[j + ndim_delta] = (ograd.shape_)[j];
  }
  Shape<broadcast::MAX_DIM> oshape = expanded_oshape.get<broadcast::MAX_DIM>();
  // get cond stride
  TShape expanded_cshape(broadcast::MAX_DIM, 1);
  ndim_delta = expanded_cshape.ndim() - cond.shape_.ndim();
  for (int j = 0; j < cond.shape_.ndim(); ++j) {
    expanded_cshape[j + ndim_delta] = (cond.shape_)[j];
  }
  Shape<broadcast::MAX_DIM> cstride =
      mxnet_op::calc_stride(expanded_cshape.get<broadcast::MAX_DIM>());
  // get expanded lshape
  TShape expanded_lshape(broadcast::MAX_DIM, 1);
  ndim_delta = expanded_lshape.ndim() - dx.shape_.ndim();
  for (int j = 0; j < dx.shape_.ndim(); ++j) {
    expanded_lshape[j + ndim_delta] = (dx.shape_)[j];
  }

  MSHADOW_TYPE_SWITCH_WITH_BOOL(ograd.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(cond.type_flag_, CType, {
      Tensor<xpu, 1, char> largespace;
      Tensor<xpu, broadcast::MAX_DIM, DType> workspace;
      size_t ws_size = 0;
      if (ograd.shape_ != dx.shape_) {
        ws_size = broadcast::ReduceWorkspaceSize(s, expanded_lshape, req[0],
                                                 expanded_oshape);
      }
      // If lscalar, then process right output, `is_left` should be false
      if (ograd.shape_ == dx.shape_) {
        mxnet_op::Kernel<numpy_where_backward_kernel<broadcast::MAX_DIM, !is_lscalar>, xpu>::Launch(
          s, ograd.Size(), req[0], cstride, oshape,
          cond.dptr<CType>(), ograd.dptr<DType>(), dx.dptr<DType>());
      } else {
        largespace = ctx.requested[0].get_space_typed<xpu, 1, char>(
            Shape1(ograd.shape_.Size() * sizeof(DType) + ws_size), s);
        workspace = Tensor<xpu, broadcast::MAX_DIM, DType>(
            reinterpret_cast<DType*>(largespace.dptr_ + ws_size),
            expanded_oshape.get<broadcast::MAX_DIM>(), s);
        mxnet_op::Kernel<numpy_where_backward_kernel<broadcast::MAX_DIM, !is_lscalar>, xpu>::Launch(
          s, ograd.Size(), req[0], cstride, oshape,
          cond.dptr<CType>(), ograd.dptr<DType>(), workspace.dptr_);
        if (NeedSafeAcc<true>(dx.type_flag_, dx.type_flag_)) {
          NP_WHERE_REDUCE_AXES(true, ctx, {TBlob(workspace)}, {req[0]},
              {dx.reshape(expanded_lshape)}, expanded_lshape);
        } else {
          NP_WHERE_REDUCE_AXES(false, ctx, {TBlob(workspace)}, {req[0]},
              {dx.reshape(expanded_lshape)}, expanded_lshape);
        }
      }
    });
  });
}

#undef NP_WHERE_REDUCE_AXES

template<typename xpu>
inline void NumpyWhereScalar2OpForward(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  if (outputs[0].shape_.Size() == 0U) return;  // zero-size tensor
  using namespace mxnet_op;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const NumpyWhereScalar2Param& param = nnvm::get<NumpyWhereScalar2Param>(attrs.parsed);
  const TBlob& cond = inputs[0];
  const TBlob& out = outputs[0];
  MSHADOW_TYPE_SWITCH_WITH_BOOL(out.type_flag_, DType, {
    MSHADOW_TYPE_SWITCH_WITH_BOOL(cond.type_flag_, CType, {
      Kernel<numpy_where_scalar2_kernel, xpu>::Launch(s, out.Size(), req[0],
          out.dptr<DType>(), cond.dptr<CType>(), DType(param.x), DType(param.y));
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_WHERE_OP_INL_H_
