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
 * \file np_norm-inl.h
 * \brief norm
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_NORM_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_NORM_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include <string>
#include <sstream>
#include <cmath>
#include "../../tensor/la_op.h"
#include "../../tensor/la_op-inl.h"
#include "../../tensor/init_op.h"
#include "./broadcast_reduce_op_customized.h"
#include "./np_gesvd-inl.h"
#include "../np_matrix_op-inl.h"

namespace mxnet {
namespace op {

namespace mshadow_op {
/*! \brief Lp-norm power reducer */

struct nrmlp {
  double lp;
  MSHADOW_XINLINE nrmlp(): lp(2) {}
  MSHADOW_XINLINE nrmlp(double l): lp(l) {}

  /* \brief power for Lp norm */
  MSHADOW_XINLINE static double lp_power(volatile double src, volatile double p) {
    if (p != 0.0) {
      if (src == 0.0) {
        return src;
      } else {
        return power::Map(src, p);
      }
    } else {  // 0-norm, sparsity
      return static_cast<double>(src != 0);
    }
  }

  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE void Reduce(volatile AType& sum_of_powers, volatile DType src) { // NOLINT(*)
    if (src != 0) {
      sum_of_powers += AType(lp_power(static_cast<double>(src), lp));
    }
  }

  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  MSHADOW_XINLINE void Reduce(volatile AType& sum_of_powers, volatile DType src, volatile DType& scale) { // NOLINT(*)
#pragma GCC diagnostic push
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif
    if (src != 0) {
      DType src_abs = abs::Map(src);
      if (scale < src_abs) {
        sum_of_powers = sum_of_powers * AType(lp_power(static_cast<double>(scale / src_abs), lp));
        sum_of_powers = sum_of_powers + 1;
        scale = src_abs;
      } else {
        sum_of_powers = sum_of_powers + AType(lp_power(static_cast<double>(src_abs / scale), lp));
      }
    }
#pragma GCC diagnostic pop
  }

  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_val, volatile DType& src_val) { // NOLINT(*)
    dst_val += src_val;
  }

  /*! \brief combine the results of two reducers */
  template<typename DType>
  MSHADOW_XINLINE static void Merge(volatile DType& dst_ssq, volatile DType& dst_scale, volatile DType& src_ssq, volatile DType& src_scale) { // NOLINT(*)
    if (dst_scale != 0 && dst_scale >= src_scale) {
      dst_ssq = dst_ssq + src_ssq * DType(lp_power(static_cast<double>(src_scale / dst_scale), 2));
    } else if (src_scale != 0 && dst_scale < src_scale) {
      dst_ssq = src_ssq + dst_ssq * DType(lp_power(static_cast<double>(dst_scale / src_scale), 2));
      dst_scale = src_scale;
    }
  }

  /*! \brief finalize reduction result */
  template<typename DType>
  MSHADOW_XINLINE void Finalize(volatile DType& sum_of_powers) { // NOLINT(*)
    if (lp != 0.0) {
      sum_of_powers = DType(lp_power(static_cast<double>(sum_of_powers), 1.0 / lp));
    }
  }

  /*! \brief finalize reduction result */
  template<typename DType>
  MSHADOW_XINLINE void Finalize(volatile DType& sum_of_powers, volatile DType& scale) { // NOLINT(*)
#pragma GCC diagnostic push
#if __GNUC__ >= 7
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif
    if (lp != 0.0) {
      sum_of_powers = scale * DType(lp_power(static_cast<double>(sum_of_powers), 1.0 / lp));
    }
#pragma GCC diagnostic pop
  }

  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &sum_of_powers) { // NOLINT(*)
    sum_of_powers = 0;
  }

  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  MSHADOW_XINLINE static void SetInitValue(DType &sum_of_powers, DType &scale) { // NOLINT(*)
    SetInitValue(sum_of_powers);
    scale = 0;
  }
};

/*! \brief Elementwise gradient of Lp-norm, does not handle p = 1 */
struct nrmlp_grad : public mxnet_op::tunable {
  double lp;
  MSHADOW_XINLINE nrmlp_grad(): lp(2) {}
  MSHADOW_XINLINE nrmlp_grad(double l): lp(l) {}

  /* \brief elementwise gradient of lp norm */
  template<typename DType>
  MSHADOW_XINLINE DType Map(DType a, DType b) {
    DType ret;
    if (lp != 0.0) {  // dx_i = (|x_i| / y) ^ (p - 1) * sgn(x_i)
      DType abs_a = DType(abs::Map(a));
      DType sgn_a = DType(sign::Map(a));
      ret = power::Map(DType(abs_a / b), DType(lp - 1)) * sgn_a;
    } else {  // L0 norm is elementwise constant and non-differentiable
      ret = 0;
    }
    return ret;
  }
};

/*! \brief Gradient for abs-min/max */
struct abs_grad : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    DType sgn = DType(sign::Map(a));
    DType grad = DType(abs::Map(a)) == DType(abs::Map(b)) ?
                 DType(1.0) : DType(0.0);
    return sgn * grad;
  }
};

/*! \brief Sign */
struct abs_sign : public mxnet_op::tunable {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(sign::Map(a));
  }
};

}  // namespace mshadow_op

inline bool NumpyLpNormShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs);

inline bool NumpyMatrixNormShape(const nnvm::NodeAttrs& attrs,
                                   mxnet::ShapeVector *in_attrs,
                                   mxnet::ShapeVector *out_attrs);

inline void assign_svd_empty(mxnet::ShapeVector *out_attrs);

bool NumpyNormShape(const nnvm::NodeAttrs& attrs,
                    mxnet::ShapeVector *in_attrs,
                    mxnet::ShapeVector *out_attrs);

bool NumpyNormType(const nnvm::NodeAttrs& attrs,
                   std::vector<int>* in_attrs,
                   std::vector<int>* out_attrs);

TShape swapMatDims(const TShape &shape, const TShape &axis);

TShape inverseTranspose(const TShape &axes);

struct NumpyNormParam : public dmlc::Parameter<NumpyNormParam> {
  double ord;
  dmlc::optional<mxnet::TShape> axis;
  bool keepdims;
  int flag;
  DMLC_DECLARE_PARAMETER(NumpyNormParam) {
    DMLC_DECLARE_FIELD(ord).set_default(2)
    .describe("Order of the norm. inf means numpy’s inf object.");
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<mxnet::TShape>())
    .describe(R"code(If axis is an integer, it specifies the axis of x along
     which to compute the vector norms. If axis is a 2-tuple,
     it specifies the axes that hold 2-D matrices, and the matrix norms of
     these matrices are computed. If axis is None then either a vector norm (when x is 1-D)
     or a matrix norm (when x is 2-D) is returned.If axis is an integer,
     it specifies the axis of x along which to compute the vector norms.
     If axis is a 2-tuple, it specifies the axes that hold 2-D matrices,
     and the matrix norms of these matrices are computed. If axis is None then either a
     vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned.)code");
    DMLC_DECLARE_FIELD(keepdims).set_default(false)
    .describe("If this is set to `True`, the reduced axis is left "
    "in the result as dimension with size one.");
    DMLC_DECLARE_FIELD(flag).set_default(-1)
    .describe("Mapping relations between ord and flag."
    "ord:  None,  'fro', 'nuc', 'inf'  '-inf'."
    "flag:  0 ,    1,      2,    3,      4. ");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream oss_axis, oss_ord, oss_keepdims, oss_flag;
    if (axis.has_value()) {
      oss_axis << axis.value();
    } else {
      oss_axis << axis;
    }
    (*dict)["axis"] = oss_axis.str();
    oss_ord << ord;
    (*dict)["ord"] = oss_ord.str();
    oss_keepdims << keepdims;
    (*dict)["keepdims"] = oss_keepdims.str();
    oss_flag << flag;
    (*dict)["flag"] = oss_flag.str();
  }
};

template<typename xpu>
void NumpyLpNormCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);
  double ord = param.ord;

  if (req[0] == kNullOp) return;

  mxnet::TShape small;
  mxnet::TShape out_shape = outputs[0].shape_;
  if (param.keepdims) {
    small = outputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(inputs[0].shape_, param.axis, true, false);
    const_cast<std::vector<TBlob>&>(outputs)[0] = outputs[0].reshape(small);
  }
  bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
  if (!safe_acc && inputs[0].type_flag_ == mshadow::kFloat16) {
    common::LogOnce("MXNET_SAFE_ACCUMULATION=1 is recommended for LpNorm with float16 inputs. "
                    "See https://mxnet.apache.org/api/faq/env_var "
                    "for more details.");
  }
  if (param.axis.value().ndim() != 2) {  // elementwise Lp-norm
    if (ord == -std::numeric_limits<double>::infinity()) {  // -inf norm
      LOG(FATAL) << "-inf norm handled in front-end.";
    } else if (param.ord == std::numeric_limits<double>::infinity()) {  // inf norm
      LOG(FATAL) << "inf norm handled in front-end.";
    } else {
#ifndef __CUDACC__
      mshadow_op::nrmlp host_reducer(param.ord);
      mshadow_op::nrmlp *reducer_instance = nullptr;
      reducer_instance = &host_reducer;
      if (safe_acc) {
        ReduceAxesComputeImplWithReducer<xpu, mshadow_op::nrmlp, true, mshadow_op::abs>(
          ctx, inputs, req, outputs, small, reducer_instance);
      } else {
        ReduceAxesComputeImplWithReducer<xpu, mshadow_op::nrmlp, false, mshadow_op::abs>(
          ctx, inputs, req, outputs, small, reducer_instance);
      }
#else
      ReduceAxesRTCComputeImpl(
        ctx, inputs, req, outputs, small, "red::nrmlp{" + std::to_string(param.ord) + "}",
        nullptr, false, "abs");
#endif
    }
  }
  if (!param.keepdims) {
    const_cast<std::vector<TBlob>&>(outputs)[0] = outputs[0].reshape(out_shape);
  }
}

template<typename xpu>
void NumpyLpNormGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);
  double ord = param.ord;
  mxnet::TShape small;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (param.keepdims) {
    small = inputs[0].shape_;
  } else {
    small = ReduceAxesShapeImpl(outputs[0].shape_, param.axis, true, false);
  }

  if (param.axis.value().ndim() != 2) {  // Elementwise Lp norm
    if (ord == -std::numeric_limits<double>::infinity()) {  // -inf norm
      LOG(FATAL) << "-inf norm handled in front-end.";
    } else if (ord == std::numeric_limits<double>::infinity()) {  // inf norm
      LOG(FATAL) << "inf norm handled in front-end.";
    } else if (ord == 1) {  // nrmlp_grad does not handle p = 1, legacy code from tensor
      mxnet::TShape src_shape, dst_shape;
      BroadcastReduceShapeCompact(outputs[0].shape_, small, &src_shape, &dst_shape);
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> in_shape;
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> out_shape;
      for (int i = 0; i < MXNET_SPECIAL_MAX_NDIM; ++i) {
        if (i < dst_shape.ndim()) {
          in_shape[i] = src_shape[i];
          out_shape[i] = dst_shape[i];
        } else {
          in_shape[i] = 1;
          out_shape[i] = 1;
        }
      }
      // refer to NumpyNormType()
      CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        if (dst_shape.ndim() == 2) {
          Tensor<xpu, 2, DType> ograd =
            inputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
          Tensor<xpu, 2, DType> igrad =
            outputs[0].get_with_shape<xpu, 2, DType>(src_shape.get<2>(), s);
          Tensor<xpu, 2, DType> data =
            inputs[1].get_with_shape<xpu, 2, DType>(src_shape.get<2>(), s);
          MXNET_REQ_TYPE_SWITCH(req[0], Req, {
            Kernel<norm_backward_broadcast<Req>, xpu>::Launch(
              s, igrad.shape_.Size(), igrad.dptr_, ograd.dptr_, data.dptr_,
              in_shape, out_shape, src_shape.ndim());
          });
        } else {
          const int ndim = MXNET_SPECIAL_MAX_NDIM;
          Tensor<xpu, ndim, DType> igrad =
            outputs[0].get_with_shape<xpu, ndim, DType>(src_shape.get<ndim>(), s);
          Tensor<xpu, ndim, DType> ograd =
            inputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
          Tensor<xpu, ndim, DType> data =
            inputs[1].get_with_shape<xpu, ndim, DType>(src_shape.get<ndim>(), s);
          MXNET_REQ_TYPE_SWITCH(req[0], Req, {
            Kernel<norm_backward_broadcast<Req>, xpu>::Launch(
              s, igrad.shape_.Size(), igrad.dptr_, ograd.dptr_, data.dptr_,
              in_shape, out_shape, src_shape.ndim());
          });
        }
      });
    } else {  // Elementwise Lp
      mshadow_op::nrmlp_grad host_mapper(ord);
      mshadow_op::nrmlp_grad *mapper_instance = nullptr;
#ifdef __CUDACC__
      cudaStream_t copy_stream = mshadow::Stream<gpu>::GetStream(s);
      cudaMalloc(reinterpret_cast<void**>(&mapper_instance), sizeof(mshadow_op::nrmlp_grad));
      cudaMemcpyAsync(mapper_instance, &host_mapper, sizeof(mshadow_op::nrmlp_grad),
                      cudaMemcpyHostToDevice, copy_stream);
      cudaStreamSynchronize(copy_stream);
#else
      mapper_instance = &host_mapper;
#endif
      MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        if (req[0] == kAddTo) {
          TBlob workspace = TBlob(ctx.requested[0].get_space_typed<xpu, 1, DType>(
                              Shape1(outputs[0].shape_.Size()), s));
          std::vector<TBlob> temp({workspace.reshape(outputs[0].shape_)});
          ReduceAxesBackwardUseInOutImplWithMapper<xpu, mshadow_op::nrmlp_grad, false>(
            ctx, small, inputs, req, temp, mapper_instance);
          Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
          out += workspace.FlatTo1D<xpu, DType>(s);
        } else {
          ReduceAxesBackwardUseInOutImplWithMapper<xpu, mshadow_op::nrmlp_grad, false>(
            ctx, small, inputs, req, outputs, mapper_instance);
        }
      });
#ifdef __CUDACC__
      cudaFree(mapper_instance);
#endif
    }
  } else {  // matrix norm should switch to matrix norm op
    LOG(FATAL) << "Case handled in matrix norm compute.";
  }
}

template<typename xpu>
void NumpyMatrixNormCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  if (req[0] == kNullOp) return;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);
  double ord = param.ord;

  TShape reduced_shape;
  if (param.keepdims) {
    reduced_shape = outputs[0].shape_;
  } else {
    reduced_shape = ReduceAxesShapeImpl(inputs[0].shape_, param.axis, true, false);
  }

  if (param.flag == 1) {  // Frobenius norm
#if !defined(__CUDACC__)
    ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, false, mshadow_op::identity>(
      ctx, inputs, req, outputs, reduced_shape);
#else
    ReduceAxesRTCComputeImpl(
      ctx, inputs, req, outputs, reduced_shape, "red::nrm2{}", nullptr, false, "identity");
#endif
    return;
  }

  TShape mat_axis = param.axis.value();

  if (param.ord != 2 && param.ord != -2) {  // row norm or col norm
    TShape sum_shape = inputs[0].shape_;
    sum_shape[mat_axis[!(param.ord == 1 || param.ord == -1)]] = 1;
    TBlob temp = outputs[1].reshape(sum_shape);
    std::vector<TBlob> sum_output({temp});
#if !defined(__CUDACC__)
    ReduceAxesComputeImpl<xpu, mshadow::red::sum, false, false, mshadow_op::abs>(
      ctx, inputs, req, sum_output, sum_shape);
    if (param.ord > 0) {
      ReduceAxesComputeImpl<xpu, mshadow::red::maximum, false, false, mshadow_op::identity>(
        ctx, sum_output, req, outputs, reduced_shape);
    } else {
      ReduceAxesComputeImpl<xpu, mshadow::red::minimum, false, false, mshadow_op::identity>(
        ctx, sum_output, req, outputs, reduced_shape);
    }
#else
    ReduceAxesRTCComputeImpl(ctx, inputs, req, sum_output, sum_shape,
                             "red::sum{}", nullptr, false, "abs");
    if (param.ord > 0) {
      ReduceAxesRTCComputeImpl(ctx, sum_output, req, outputs, reduced_shape,
                               "red::maximum{}", nullptr, false);
    } else {
      ReduceAxesRTCComputeImpl(ctx, sum_output, req, outputs, reduced_shape,
                               "red::minimum{}", nullptr, false);
    }
#endif
    return;
  }

  if (inputs[0].type_flag_ == mshadow::kFloat16) {
    LOG(FATAL) << "Matrix +/- 2-norm does not support float 16 due to SVD implementation.";
  }

  // spectral norms
  TShape old_shape = inputs[0].shape_;
  TShape svd_in_shape = inputs[0].shape_;
  TShape axes(old_shape.ndim(), 1);
  for (int i = 0; i < old_shape.ndim(); ++i) {
    axes[i] = i;
  }

  svd_in_shape = swapMatDims(svd_in_shape, mat_axis);
  axes = swapMatDims(axes, mat_axis);
  TShape reduce_axes = inverseTranspose(axes);

  int row_dim = svd_in_shape[svd_in_shape.ndim() - 2];
  int col_dim = svd_in_shape[svd_in_shape.ndim() - 1];
  int svd_dim = row_dim <= col_dim ? row_dim : col_dim;
  int batch_dim = svd_in_shape.ProdShape(0, svd_in_shape.ndim() - 2);

  TShape L_shape = svd_in_shape;
  TShape L_trans = inputs[0].shape_;
  if (row_dim > col_dim) {
    L_shape[L_shape.ndim() - 2] = 1;
    L_trans[mat_axis[0]] = 1;
  } else {
    L_shape[L_shape.ndim() - 1] = 1;
    L_trans[mat_axis[1]] = 1;
  }

  std::vector<TBlob> eigen;
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 3, DType> UT =
      outputs[1].get_with_shape<xpu, 3, DType>(Shape3(batch_dim, row_dim, row_dim), s);
    Tensor<xpu, 2, DType> L =
      outputs[2].get_with_shape<xpu, 2, DType>(Shape2(batch_dim, svd_dim), s);
    Tensor<xpu, 3, DType> V =
      outputs[3].get_with_shape<xpu, 3, DType>(Shape3(batch_dim, row_dim, col_dim), s);

    size_t svd_space = linalg_gesvd_workspace_query(UT[0], L[0], V[0], s);
    size_t space = svd_in_shape.Size() + svd_space;
    space += space & 1;
    size_t offset = svd_in_shape.Size() + (1 & svd_in_shape.Size());

    TBlob temp = TBlob(ctx.requested[0].get_space_typed<xpu, 1, DType>(
                        Shape1(space), s));
    TBlob workspace(reinterpret_cast<DType*>(temp.dptr_), svd_in_shape,
                     temp.dev_mask(), temp.dev_id());
    TBlob svd_workspace(reinterpret_cast<DType*>(temp.dptr_) + offset, TShape(1, svd_space),
                        temp.dev_mask(), temp.dev_id());
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], workspace, axes);
    Tensor<xpu, 3, DType> svd_input =
      workspace.get_with_shape<xpu, 3, DType>(Shape3(batch_dim, row_dim, col_dim), s);
    gesvd::op(svd_input, UT, L, V, ctx, attrs, &svd_workspace);
    TBlob workspace0(reinterpret_cast<DType*>(temp.dptr_), L_trans,
                     temp.dev_mask(), temp.dev_id());
    TransposeImpl<xpu>(ctx.run_ctx, TBlob(L).reshape(L_shape), workspace0, reduce_axes);
    eigen.emplace_back(workspace0);
  });

#if !defined(__CUDACC__)
  if (param.flag == 2) {  // nuclear norm
    ReduceAxesComputeImpl<xpu, mshadow::red::sum, false, false, mshadow_op::identity>(
      ctx, eigen, req, outputs, reduced_shape);
  } else if (dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true)) {
    if (ord == 2) {
      ReduceAxesComputeImpl<xpu, mshadow::red::maximum, true, false, mshadow_op::abs>(
        ctx, eigen, req, outputs, reduced_shape);
    } else if (ord == -2) {
      ReduceAxesComputeImpl<xpu, mshadow::red::minimum, true, false, mshadow_op::abs>(
        ctx, eigen, req, outputs, reduced_shape);
    }
  } else {
    if (ord == 2) {
      ReduceAxesComputeImpl<xpu, mshadow::red::maximum, false, false, mshadow_op::abs>(
        ctx, eigen, req, outputs, reduced_shape);
    } else if (ord == -2) {
      ReduceAxesComputeImpl<xpu, mshadow::red::minimum, false, false, mshadow_op::abs>(
        ctx, eigen, req, outputs, reduced_shape);
    }
  }
#else
  if (param.flag == 2) {  // nuclear norm
    ReduceAxesRTCComputeImpl(ctx, eigen, req, outputs, reduced_shape, "red::sum{}", nullptr, false);
  } else {
    if (ord == 2) {
      ReduceAxesRTCComputeImpl(ctx, eigen, req, outputs, reduced_shape,
                               "red::maximum{}", nullptr, false, "abs");
    } else if (ord == -2) {
      ReduceAxesRTCComputeImpl(ctx, eigen, req, outputs, reduced_shape,
                               "red::minimum{}", nullptr, false, "abs");
    }
  }
#endif
}

template<typename xpu>
void NumpyMatrixNormGradCompute(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (req[0] == kNullOp) return;

  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);

  TShape reduced_shape;
  TShape old_shape_ = inputs[0].shape_;
  if (param.keepdims) {
    reduced_shape = inputs[0].shape_;
  } else {
    reduced_shape = ReduceAxesShapeImpl(outputs[0].shape_, param.axis, true, false);
  }

  std::vector<TBlob> map_inputs;
  std::vector<TBlob> map_outputs;

  if (param.flag == 1) {  // frob norm
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      map_inputs = std::vector<TBlob>({inputs[0], inputs[4], inputs[5]});
      if (req[0] == kAddTo) {
        TBlob workspace = TBlob(ctx.requested[0].get_space_typed<xpu, 1, DType>(
                            Shape1(outputs[0].shape_.Size()), s));
        std::vector<TBlob> temp({workspace.reshape(outputs[0].shape_)});
        ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(
          ctx, reduced_shape, map_inputs, req, temp);
        Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
        out += workspace.FlatTo1D<xpu, DType>(s);
      } else {
        ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(
          ctx, reduced_shape, map_inputs, req, outputs);
      }
    });
    return;
  }

  TShape mat_axis = param.axis.value();

  if (param.ord != 2 && param.ord != -2) {  // row norm or col norm
    TShape sum_shape = outputs[0].shape_;
    TShape out_shape = outputs[0].shape_;
    int sum_dim = mat_axis[!(param.ord == 1 || param.ord == -1)];
    sum_shape[sum_dim] = 1;
    TShape small(3, 1), squeezed(3, outputs[0].shape_[sum_dim]);
    squeezed[0] = small[0] = sum_shape.ProdShape(0, sum_dim);
    squeezed[2] = small[2] = sum_shape.ProdShape(sum_dim + 1, sum_shape.ndim());
    map_inputs = std::vector<TBlob>({ inputs[0], inputs[6], inputs[5] });

    size_t sum_size = sum_shape.Size();
    size_t ws_offset = sum_size + (sum_size & 1);
    size_t ws_size = ws_offset + (req[0] == kAddTo ? outputs[0].shape_.Size() : 0);
    ws_size += ws_size & 1;

    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      TBlob workspace = TBlob(ctx.requested[0].get_space_typed<xpu, 1, DType>(
                                Shape1(ws_size), s));
      TBlob temp0 = TBlob(reinterpret_cast<DType*>(workspace.dptr_),
                          sum_shape, workspace.dev_mask(), workspace.dev_id());
      std::vector<TBlob> map_outputs({ temp0 });
      ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::abs_grad, false>(
        ctx, reduced_shape, map_inputs, req, map_outputs);
      temp0 = temp0.reshape(small);
      map_outputs = std::vector<TBlob>({temp0, inputs[4], inputs[6]});
      if (req[0] == kAddTo) {
        TBlob out_temp = TBlob(reinterpret_cast<DType*>(workspace.dptr_) + ws_offset,
                               outputs[0].shape_, workspace.dev_mask(), workspace.dev_id());
        std::vector<TBlob> tmp_outputs({ out_temp });
        ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::abs_sign, false>(
          ctx, sum_shape, map_outputs, req, tmp_outputs);
        out_temp = out_temp.reshape(squeezed);
        Tensor<xpu, 3, DType> tmp_out =
          out_temp.get_with_shape<xpu, 3, DType>(Shape3(squeezed[0], squeezed[1], squeezed[2]), s);
        Tensor<xpu, 3, DType> mask =
          temp0.get_with_shape<xpu, 3, DType>(Shape3(small[0], small[1], small[2]), s);
        tmp_out = tmp_out * broadcast_to(mask, squeezed);
        TBlob final_output = outputs[0].reshape(squeezed);
        Tensor<xpu, 3, DType> out =
          final_output.get_with_shape<xpu, 3, DType>(
            Shape3(squeezed[0], squeezed[1], squeezed[2]), s);
        out += tmp_out;
      } else {
        ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::abs_sign, false>(
          ctx, sum_shape, map_outputs, req, outputs);
        TBlob final_output = outputs[0].reshape(squeezed);
        Tensor<xpu, 3, DType> out =
          final_output.get_with_shape<xpu, 3, DType>(
            Shape3(squeezed[0], squeezed[1], squeezed[2]), s);
        Tensor<xpu, 3, DType> mask =
          temp0.get_with_shape<xpu, 3, DType>(Shape3(small[0], small[1], small[2]), s);
        out = out * broadcast_to(mask, squeezed);
      }
    });
    return;
  }

  if (!param.keepdims) {
    const_cast<std::vector<TBlob>&>(inputs)[0] = inputs[0].reshape(reduced_shape);
    const_cast<std::vector<TBlob>&>(inputs)[5] = inputs[5].reshape(reduced_shape);
  }

  map_inputs.push_back(inputs[0]);
  TBlob L_reduced = inputs[5];
  TBlob L_irreduced = inputs[7];

  TShape old_shape = inputs[4].shape_;
  TShape svd_in_shape = old_shape;
  TShape axes(old_shape.ndim(), 1);
  for (int i = 0; i < old_shape.ndim(); ++i) {
    axes[i] = i;
  }
  svd_in_shape = swapMatDims(svd_in_shape, mat_axis);
  axes = swapMatDims(axes, mat_axis);
  TShape reduce_axes = inverseTranspose(axes);

  int row_dim = svd_in_shape[svd_in_shape.ndim() - 2];
  int col_dim = svd_in_shape[svd_in_shape.ndim() - 1];
  int batch_dim = svd_in_shape.ProdShape(0, svd_in_shape.ndim() - 2);

  TShape L_shape = svd_in_shape;
  TShape L_trans = old_shape;
  if (row_dim > col_dim) {
    L_shape[L_shape.ndim() - 2] = 1;
    L_trans[mat_axis[0]] = 1;
  } else {
    L_shape[L_shape.ndim() - 1] = 1;
    L_trans[mat_axis[1]] = 1;
  }
  L_irreduced = L_irreduced.reshape(L_shape);
  int kmn = outputs[0].shape_.Size();
  int kmm = inputs[1].shape_.Size();
  int km = inputs[2].shape_.Size();
  size_t workspace_size = svd_in_shape.ProdShape(0, svd_in_shape.ndim()) * 2
                            + km + kmn + 5;
  workspace_size += req[0] == kAddTo? kmn : kmm;
  size_t workspace_offset1 = svd_in_shape.ProdShape(0, svd_in_shape.ndim());
  workspace_offset1 += workspace_offset1 & 1;
  size_t workspace_offset2 = workspace_offset1 * 2;
  size_t workspace_offset3 = workspace_offset2;
  if (req[0] == kAddTo) {
    workspace_offset3 += kmn + (kmn & 1);
  } else {
    workspace_offset3 += kmm + (kmm & 1);
  }
  size_t workspace_offset4 = workspace_offset3 + km + (km & 1);

  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    TBlob workspace = TBlob(ctx.requested[0].get_space_typed<xpu, 1, DType>(
                        Shape1(workspace_size), s));
    TBlob workspace0(reinterpret_cast<DType*>(workspace.dptr_), L_trans,
                     workspace.dev_mask(), workspace.dev_id());
    TBlob workspace1(reinterpret_cast<DType*>(workspace.dptr_) + workspace_offset1, L_trans,
                     workspace.dev_mask(), workspace.dev_id());
    TBlob tempM(reinterpret_cast<DType*>(workspace.dptr_) + workspace_offset2, inputs[1].shape_,
                     workspace.dev_mask(), workspace.dev_id());
    TBlob tempMd(reinterpret_cast<DType*>(workspace.dptr_) + workspace_offset3, inputs[2].shape_,
                     workspace.dev_mask(), workspace.dev_id());
    TBlob temp(reinterpret_cast<DType*>(workspace.dptr_) + workspace_offset4, inputs[3].shape_,
                     workspace.dev_mask(), workspace.dev_id());
    TransposeImpl<xpu>(ctx.run_ctx, L_irreduced.reshape(L_shape), workspace0, reduce_axes);
    map_inputs.push_back(workspace0);
    map_inputs.push_back(L_reduced);
    if (param.flag == 2) {  // nuclear norm
      mxnet::op::Fill<false, DType, xpu>(s, workspace1, req[0], DType(1.0));
    } else {
      std::vector<TBlob> reduce_output({ workspace1 });
      ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::abs_grad, false>(
        ctx, reduced_shape, map_inputs, req, reduce_output);
    }
    workspace1 = workspace1.reshape(L_shape);
    gesvd_backward::op(inputs[1].FlatToKD<xpu, 3, DType>(s),
                       workspace1.reshape(inputs[2].shape_).FlatToKD<xpu, 2, DType>(s),
                       inputs[3].FlatToKD<xpu, 3, DType>(s),
                       inputs[6].FlatToKD<xpu, 3, DType>(s),
                       inputs[7].FlatToKD<xpu, 2, DType>(s),
                       inputs[8].FlatToKD<xpu, 3, DType>(s),
                       temp.get_with_shape<xpu, 3, DType>(Shape3(batch_dim, row_dim, col_dim)),
                       tempM.FlatToKD<xpu, 3, DType>(s),
                       tempMd.FlatToKD<xpu, 2, DType>(s),
                       s, attrs);
    Tensor<xpu, 3, DType> temp_flat = temp.FlatToKD<xpu, 3, DType>(s);
    TBlob in_grad_trans(reinterpret_cast<DType*>(workspace0.dptr_),
                        swapMatDims(inputs[0].shape_, mat_axis),
                        workspace.dev_mask(), workspace.dev_id());
    TransposeImpl<xpu>(ctx.run_ctx, inputs[0], in_grad_trans, axes);
    Tensor<xpu, 3, DType> trans_in_grad = in_grad_trans.FlatToKD<xpu, 3, DType>(s);
    temp_flat = temp_flat * broadcast_to(trans_in_grad, temp.shape_);
    if (req[0] == kAddTo) {
      TBlob ograd(reinterpret_cast<DType*>(tempM.dptr_), outputs[0].shape_,
                  workspace.dev_mask(), workspace.dev_id());
      TransposeImpl<xpu>(ctx.run_ctx, temp.reshape(svd_in_shape), ograd, reduce_axes);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      out += ograd.FlatTo1D<xpu, DType>(s);
    } else {
      TransposeImpl<xpu>(ctx.run_ctx, temp.reshape(svd_in_shape), outputs[0], reduce_axes);
    }
  });
  if (!param.keepdims) {
    const_cast<std::vector<TBlob>&>(inputs)[0] = inputs[0].reshape(old_shape_);
    const_cast<std::vector<TBlob>&>(inputs)[5] = inputs[5].reshape(old_shape_);
  }
}

template<typename xpu>
void NumpyNormComputeForward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (inputs[0].shape_.Size() == 0U) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      mxnet::op::Fill<false, DType, xpu>(s, outputs[0], req[0], DType(0.0));
    });
    return;
  }
  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);

  if (param.flag == -2) {  // flattened L2 norm
    std::vector<TBlob> flat_inputs({
      inputs[0].reshape(TShape(1, inputs[0].shape_.Size()))
    });
    std::vector<TBlob> flat_outputs({
      outputs[0].reshape(TShape(1, 1))
    });
#if !defined(__CUDACC__)
    ReduceAxesComputeImpl<xpu, mshadow_op::nrm2, false, false, mshadow_op::identity>(
      ctx, flat_inputs, req, flat_outputs, TShape(1, 1));
#else
    ReduceAxesRTCComputeImpl(
      ctx, flat_inputs, req, flat_outputs, TShape(1, 1), "red::nrm2{}", nullptr, false, "identity");
#endif
    return;
  }

  if (param.axis.value().ndim() == 2) {
    NumpyMatrixNormCompute<xpu>(attrs, ctx, inputs, req, outputs);
  } else {
    NumpyLpNormCompute<xpu>(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu>
void NumpyNormComputeBackward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (inputs[0].shape_.Size() == 0U) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      mxnet::op::Fill<false, DType, xpu>(s, outputs[0], req[0], DType(0.0));
    });
    return;
  }
  if (!common::is_float(outputs[0].type_flag_)) {
    LOG(FATAL) << "Computing gradient for integer inputs is not well-undefined behavior.";
  }
  const NumpyNormParam& param = nnvm::get<NumpyNormParam>(attrs.parsed);

  if (param.flag == -2) {  // flattened L2 norm
    std::vector<TBlob> flat_inputs({
      inputs[0].reshape(TShape(1, 1)),
      inputs[4].reshape(TShape(1, outputs[0].shape_.Size())),
      inputs[5].reshape(TShape(1, 1))
    });
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      if (req[0] == kAddTo) {
        TBlob workspace = TBlob(ctx.requested[0].get_space_typed<xpu, 1, DType>(
                            Shape1(outputs[0].shape_.Size()), s));
        std::vector<TBlob> temp({ workspace });
        ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(
          ctx, TShape(1, 1), flat_inputs, req, temp);
        Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
        out += workspace.FlatTo1D<xpu, DType>(s);
      } else {
        std::vector<TBlob> flat_outputs({
          outputs[0].reshape(TShape(1, outputs[0].shape_.Size()))
        });
        ReduceAxesBackwardUseInOutImpl<xpu, mshadow_op::div, false>(
          ctx, TShape(1, 1), flat_inputs, req, flat_outputs);
      }
    });
    return;
  }

  // need to infer shape again in backward
  std::vector<TShape> in_attrs({
    inputs.size() == 9 ? inputs[4].shape_ : inputs[1].shape_
  });
  std::vector<TShape> out_attrs({
    inputs.size() == 9 ? inputs[5].shape_ : inputs[2].shape_,
    TShape(), TShape(), TShape()
  });
  NumpyNormShape(attrs, &in_attrs, &out_attrs);

  if (param.axis.value().ndim() == 2) {
    NumpyMatrixNormGradCompute<xpu>(attrs, ctx, inputs, req, outputs);
  } else {
    std::vector<TBlob> grad_inputs({inputs[0], inputs[4], inputs[5]});
    NumpyLpNormGradCompute<xpu>(attrs, ctx, grad_inputs, req, outputs);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_NORM_INL_H_
