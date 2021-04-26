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
 * \file np_tensordot_op-inl.h
 * \brief CPU Implementation of numpy-compatible tensordot
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_TENSORDOT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_TENSORDOT_OP_INL_H_

#include <vector>
#include <string>
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct TensordotParam : public dmlc::Parameter<TensordotParam> {
  mxnet::Tuple<int> a_axes_summed, b_axes_summed;
  DMLC_DECLARE_PARAMETER(TensordotParam) {
    DMLC_DECLARE_FIELD(a_axes_summed);
    DMLC_DECLARE_FIELD(b_axes_summed);
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream a_axes_summed_s, b_axes_summed_s;
    a_axes_summed_s << a_axes_summed;
    b_axes_summed_s << b_axes_summed;
    (*dict)["a_axes_summed"] = a_axes_summed_s.str();
    (*dict)["b_axes_summed"] = b_axes_summed_s.str();
  }
};

/**
 * deals with negative axes.
 */
inline void ShiftAxes(Tuple<int>* axes_summed, const int ndim) {
  for (auto& i : *axes_summed) {
    i = (i + ndim) % ndim;
  }
}

/**
 * Gets matrix dimensions of a and b after transpose and reshape.
 */
inline void GetMatrixDimensions(index_t* ad1,
                                index_t* ad2,
                                index_t* bd1,
                                index_t* bd2,
                                const mxnet::Tuple<int>& a_axes_remained,
                                const mxnet::Tuple<int>& a_axes_summed,
                                const mxnet::Tuple<int>& b_axes_remained,
                                const mxnet::Tuple<int>& b_axes_summed,
                                const mxnet::TShape& a_shape,
                                const mxnet::TShape& b_shape) {
  *ad1 = 1;
  *ad2 = 1;
  *bd1 = 1;
  *bd2 = 1;

  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    *ad1 *= a_shape[a_axes_remained[i]];
  }
  for (int i = 0; i < a_axes_summed.ndim(); i++) {
    *ad2 *= a_shape[a_axes_summed[i]];
  }
  for (int i = 0; i < b_axes_summed.ndim(); i++) {
    *bd1 *= b_shape[b_axes_summed[i]];
  }
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    *bd2 *= b_shape[b_axes_remained[i]];
  }
}

/**
 * gets new axes of a and b after transpose and reshape.
 */
inline void GetReorderedAxes(const mxnet::Tuple<int>& a_axes_summed,
                             mxnet::Tuple<int>* a_axes_remained,
                             mxnet::Tuple<int>* a_axes,
                             const mxnet::Tuple<int>& b_axes_summed,
                             mxnet::Tuple<int>* b_axes_remained,
                             mxnet::Tuple<int>* b_axes,
                             const mxnet::TShape& a_shape,
                             const mxnet::TShape& b_shape) {
  std::vector<int> a_axes_remained_vector;
  for (int i = 0; i < a_shape.ndim(); i++) {
    a_axes_remained_vector.push_back(i);
  }
  for (auto& i : a_axes_summed) {
    a_axes_remained_vector.erase(std::find(a_axes_remained_vector.begin(),
      a_axes_remained_vector.end(), i));
  }
  *a_axes_remained = mxnet::Tuple<int>(a_axes_remained_vector);

  std::vector<int> a_axes_vector(a_axes_remained_vector);
  for (auto& i : a_axes_summed) {
    a_axes_vector.push_back(i);
  }
  *a_axes = mxnet::Tuple<int>(a_axes_vector);

  std::vector<int> b_axes_remained_vector;
  for (int i = 0; i < b_shape.ndim(); i++) {
    b_axes_remained_vector.push_back(i);
  }
  for (auto& i : b_axes_summed) {
    b_axes_remained_vector.erase(std::find(b_axes_remained_vector.begin(),
                                           b_axes_remained_vector.end(), i));
  }
  *b_axes_remained = mxnet::Tuple<int>(b_axes_remained_vector);

  std::vector<int> b_axes_vector;
  for (auto& i : b_axes_summed) {
    b_axes_vector.push_back(i);
  }
  for (auto& i : b_axes_remained_vector) {
    b_axes_vector.push_back(i);
  }
  *b_axes = mxnet::Tuple<int>(b_axes_vector);
}

/**
 * gets shapes of a and b after transpose and reshape.
 */
inline mxnet::TShape GetReorderedShape(const mxnet::TShape& shape, const mxnet::Tuple<int>& axes) {
  mxnet::TShape new_shape(shape);
  for (int i = 0; i < axes.ndim(); i++) {
    new_shape[i] = shape[axes[i]];
  }
  return new_shape;
}

/**
 * gets matrix dot. Reshapes tensor a as ad1-by-ad2 matrix, tensor b as bd1-by-bd2 matrix, then
 * calculates matrix dot a * b and stores in tensor out.
 */
template<typename xpu>
void MatrixDot(const OpContext& ctx,
               const TBlob& a,
               const TBlob& b,
               const TBlob& out,
               const OpReqType req,
               const index_t ad1,
               const index_t ad2,
               const index_t bd1,
               const index_t bd2,
               const bool aT = false,
               const bool bT = false) {
  using namespace mshadow;
  using namespace mshadow_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    Tensor<xpu, 2, DType> a_tensor = a.get_with_shape<xpu, 2, DType>(Shape2(ad1, ad2), s);
    Tensor<xpu, 2, DType> b_tensor = b.get_with_shape<xpu, 2, DType>(Shape2(bd1, bd2), s);

    if (aT && bT) {
      CHECK_EQ(ad1, bd2);
      Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad2, bd1), s);
      ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor.T(), b_tensor.T()));
    } else if (aT && !bT) {
      CHECK_EQ(ad1, bd1);
      Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad2, bd2), s);
      ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor.T(), b_tensor));
    } else if (!aT && bT) {
      CHECK_EQ(ad2, bd2);
      Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad1, bd1), s);
      ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor, b_tensor.T()));
    } else {
      CHECK_EQ(ad2, bd1);
      Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad1, bd2), s);
      ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor, b_tensor));
    }
  });
}

/**
 * Scalar multiply.
 */
template<int req>
struct scalar_mul_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType *out, const DType* tensor, const DType *scalar) {
    KERNEL_ASSIGN(out[i], req, tensor[i] * scalar[0]);
  }
};

/**
 * Calculates tensordot.
 */
template<typename xpu>
void TensordotImpl(const Tuple<int>& a_axes_summed,
                   const Tuple<int>& b_axes_summed,
                   const OpContext& ctx,
                   const TBlob& a,
                   const TBlob& b,
                   const TBlob& out,
                   const std::vector<OpReqType>& req,
                   const Tensor<xpu, 1, char>& workspace) {
  if (req[0] == kNullOp) {
    return;
  }

  if (out.shape_.Size() == 0U) {
    return;  // zero-size output, no need to launch kernel
  }

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(out.type_flag_, a.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(out.type_flag_, b.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(out.type_flag_ == kFloat32 || out.type_flag_ == kFloat64 ||
       (out.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
      << "Tensordot only supports float32/float64 for CPU, and float16/float32/float64 for GPU";

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    if (a_shape.Size() == 0U || b_shape.Size() == 0U) {
      // 0-size input
      if (req[0] != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
    } else if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
      // Both 0-D scalars, equivalent to multiply
      Tensor<xpu, 1, DType> a_data = a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_data = b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(out_data, req[0], a_data * b_data);
    } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
      // Either of them is a scalar, just scale by one of them
      const DType* tensor = (a_shape.ndim() == 0) ? b.dptr<DType>() : a.dptr<DType>();
      const DType* scalar = (a_shape.ndim() == 0) ? a.dptr<DType>() : b.dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        mxnet_op::Kernel<scalar_mul_kernel<Req>, xpu>::Launch(
          s, out.Size(), out.dptr<DType>(), tensor, scalar);
      });
    } else {
      // Two tensors of at least 1 dimensions.
      Tuple<int> a_axes_remained;
      Tuple<int> b_axes_remained;
      Tuple<int> a_axes;
      Tuple<int> b_axes;
      GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
                       &b_axes, a_shape, b_shape);

      index_t ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
      GetMatrixDimensions(&ad1, &ad2, &bd1, &bd2, a_axes_remained, a_axes_summed,
                          b_axes_remained, b_axes_summed, a_shape, b_shape);

      mxnet::TShape a_temp_shape = GetReorderedShape(a_shape, a_axes);
      mxnet::TShape b_temp_shape = GetReorderedShape(b_shape, b_axes);

      DType* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      DType* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.Size() * sizeof(DType));
      TBlob a_res = TBlob(a_ptr, a_temp_shape, xpu::kDevMask);
      TBlob b_res = TBlob(b_ptr, b_temp_shape, xpu::kDevMask);

      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a, a_res,
                                    mxnet::TShape(a_axes.begin(), a_axes.end()));
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b, b_res,
                                    mxnet::TShape(b_axes.begin(), b_axes.end()));

      MatrixDot<xpu>(ctx, a_res, b_res, out, req[0], ad1, ad2, bd1, bd2);
    }
  });
}

/**
 * Calculates workspace size of tensordot.
 */
template<typename xpu>
size_t TensordotWorkspaceSize(const Tuple<int>& a_axes_summed,
                              const Tuple<int>& b_axes_summed,
                              const TBlob& a,
                              const TBlob& b,
                              const TBlob& out,
                              const std::vector<OpReqType>& req) {
  if (req[0] == kNullOp) {
    return 0U;
  }

  if (out.shape_.Size() == 0U) {
    return 0U;  // zero-size output, no need to launch kernel
  }

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    if (a_shape.Size() == 0U || b_shape.Size() == 0U) {
      // 0-size input
      return 0U;
    } else if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
      // Both 0-D scalars, equivalent to multiply
      return 0U;
    } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
      // Either of them is a scalar, just scale by one of them
      return 0U;
    } else {
      // Two tensors of at least 1 dimensions.
      return (a.Size() + b.Size()) * sizeof(DType);
    }
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

/**
 * forward function
 */
template<typename xpu>
void TensordotOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;

  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);
  Tuple<int> a_axes_summed = param.a_axes_summed;
  Tuple<int> b_axes_summed = param.b_axes_summed;
  ShiftAxes(&a_axes_summed, a_shape.ndim());
  ShiftAxes(&b_axes_summed, b_shape.ndim());

  size_t workspace_size = TensordotWorkspaceSize<xpu>(a_axes_summed, b_axes_summed, a, b, out, req);
  Tensor<xpu, 1, char> workspace = ctx.requested[0].get_space_typed<xpu, 1, char>(
    Shape1(workspace_size), ctx.get_stream<xpu>());
  TensordotImpl<xpu>(a_axes_summed, b_axes_summed, ctx, a, b, out, req, workspace);
}

/**
 * gets shapes for inverse transpose.
 */
inline mxnet::TShape GetReverseShape(const mxnet::Tuple<int>& shape) {
  mxnet::TShape shape2(shape.begin(), shape.end());
  for (int i = 0; i < shape.ndim(); i++) {
    shape2[shape[i]] = i;
  }
  return shape2;
}

#if !defined(__CUDACC__)
#define NP_TENSORDOT_REDUCE_AXES(safe_acc, ...) \
  ReduceAxesComputeImpl<xpu, mshadow_op::sum, safe_acc>(__VA_ARGS__)
#else
#define NP_TENSORDOT_REDUCE_AXES(safe_acc, ...) \
  ReduceAxesRTCComputeImpl(__VA_ARGS__, "red::sum{}")
#endif

/**
 * calculates tensordot derivative.
 */
template<typename xpu>
void TensordotBackwardImpl(const Tuple<int>& a_axes_summed,
                           const Tuple<int>& b_axes_summed,
                           const OpContext& ctx,
                           const TBlob& out_grad,
                           const TBlob& a,
                           const TBlob& b,
                           const TBlob& grad_a,
                           const TBlob& grad_b,
                           const std::vector<OpReqType>& req,
                           const Tensor<xpu, 1, char>& workspace) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  if ((a_shape.Size() == 0U) || (b_shape.Size() == 0U)) {
    return;  // zero-size output, no need to launch kernel
  }
  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
      // Both 0-D scalars, equivalent to multiply
      Tensor<xpu, 1, DType> out_grad_data = out_grad.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> a_data = a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_data = b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> grad_a_data = grad_a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> grad_b_data = grad_b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(grad_a_data, req[0], b_data * out_grad_data);
      ASSIGN_DISPATCH(grad_b_data, req[1], a_data * out_grad_data);
    } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
      // Either of them is a scalar, just scale by one of them
      const TBlob& tensor = (a_shape.ndim() == 0) ? b : a;
      const TBlob& tensor_grad = (a_shape.ndim() == 0) ? grad_b : grad_a;
      const TBlob& scalar = (a_shape.ndim() == 0) ? a : b;
      const TBlob& scalar_grad = (a_shape.ndim() == 0) ? grad_a : grad_b;
      Tensor<xpu, 1, DType> scalar_ = scalar.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> scalar_grad_ = scalar_grad.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> tensor_ = tensor.FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> tensor_grad_ = tensor_grad.FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out_grad_ = out_grad.FlatTo1D<xpu, DType>(s);
      const OpReqType& tensor_req = (a_shape.ndim() == 0) ? req[1] : req[0];
      const OpReqType& scalar_req = (a_shape.ndim() == 0) ? req[0] : req[1];
      ASSIGN_DISPATCH(tensor_grad_, tensor_req,
                      broadcast_scalar(scalar_, tensor_grad_.shape_) * out_grad_);
      Tensor<xpu, 1, DType> dtypespace =
        Tensor<xpu, 1, DType>(reinterpret_cast<DType*>(workspace.dptr_),
                              workspace.shape_,
                              workspace.stride_,
                              workspace.stream_);
      ASSIGN_DISPATCH(dtypespace, kWriteTo, tensor_ * out_grad_);

      NP_TENSORDOT_REDUCE_AXES(true, ctx, {TBlob(dtypespace)}, {scalar_req},
                               {TBlob(scalar_grad_)}, scalar_grad_.shape_);
    } else {
      // Two tensors of at least 1 dimensions.
      Tuple<int> a_axes_remained;
      Tuple<int> b_axes_remained;
      Tuple<int> a_axes;
      Tuple<int> b_axes;
      GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
                      &b_axes, a_shape, b_shape);

      index_t ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
      GetMatrixDimensions(&ad1, &ad2, &bd1, &bd2, a_axes_remained, a_axes_summed,
                          b_axes_remained, b_axes_summed, a_shape, b_shape);

      std::vector<int> a_T_axes;
      for (int i = 0; i < a_axes_summed.ndim(); i++) {
        a_T_axes.push_back(a_axes_summed[i]);
      }
      for (int i = 0; i < a_axes_remained.ndim(); i++) {
        a_T_axes.push_back(a_axes_remained[i]);
      }
      mxnet::TShape a_temp_shape(GetReorderedShape(a_shape, a_axes));
      mxnet::TShape a_T_temp_shape(GetReorderedShape(a_shape, a_T_axes));

      std::vector<int> b_T_axes;
      for (int i = 0; i < b_axes_remained.ndim(); i++) {
        b_T_axes.push_back(b_axes_remained[i]);
      }
      for (int i = 0; i < b_axes_summed.ndim(); i++) {
        b_T_axes.push_back(b_axes_summed[i]);
      }
      mxnet::TShape b_temp_shape(GetReorderedShape(b_shape, b_axes));
      mxnet::TShape b_T_temp_shape(GetReorderedShape(b_shape, b_T_axes));

      DType* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
      DType* a_ptr2 = reinterpret_cast<DType*>(workspace.dptr_ + a.Size() * sizeof(DType));
      DType* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + 2 * a.Size() * sizeof(DType));
      DType* b_ptr2 = reinterpret_cast<DType*>(workspace.dptr_ +
                                               (2 * a.Size() +
                                               b.Size()) *
                                               sizeof(DType));

      TBlob a_res = TBlob(a_ptr, a_temp_shape, xpu::kDevMask);
      TBlob b_res = TBlob(b_ptr, b_temp_shape, xpu::kDevMask);
      TBlob a_res2 = TBlob(a_ptr2, a_T_temp_shape, xpu::kDevMask);
      TBlob b_res2 = TBlob(b_ptr2, b_T_temp_shape, xpu::kDevMask);

      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a, a_res2,
                                    mxnet::TShape(a_T_axes.begin(), a_T_axes.end()));
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b, b_res2,
                                    mxnet::TShape(b_T_axes.begin(), b_T_axes.end()));
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, grad_a, a_res,
                                    mxnet::TShape(a_axes.begin(), a_axes.end()));
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, grad_b, b_res,
                                    mxnet::TShape(b_axes.begin(), b_axes.end()));
      MatrixDot<xpu>(ctx, a_res2, out_grad, b_res, req[1], ad2, ad1, ad1, bd2);
      MatrixDot<xpu>(ctx, out_grad, b_res2, a_res, req[0], ad1, bd2, bd2, bd1);

      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a_res, grad_a, GetReverseShape(a_axes));
      mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b_res, grad_b, GetReverseShape(b_axes));
    }
  });
}

/**
 * Calculates workspace size of tensordot backward.
 */
template<typename xpu>
size_t TensordotBackwardWorkspaceSize(const Tuple<int>& a_axes_summed,
                                      const Tuple<int>& b_axes_summed,
                                      const TBlob& out_grad,
                                      const TBlob& a,
                                      const TBlob& b,
                                      const TBlob& grad_a,
                                      const TBlob& grad_b,
                                      const std::vector<OpReqType>& req) {
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  if ((a_shape.Size() == 0U) || (b_shape.Size() == 0U)) {
    return 0U;  // zero-size output, no need to launch kernel
  }
  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
      // Both 0-D scalars, equivalent to multiply
      return 0U;
    } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
      // Either of them is a scalar, just scale by one of them
      return out_grad.shape_.Size() * sizeof(DType);
    } else {
      return (a.Size() + b.Size()) * 2 * sizeof(DType);
    }
  });
  LOG(FATAL) << "InternalError: cannot reach here";
  return 0U;
}

/**
 * backward function.
 */
template<typename xpu>
void TensordotOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);

  const TBlob& out_grad = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;

  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);
  Tuple<int> a_axes_summed = param.a_axes_summed;
  Tuple<int> b_axes_summed = param.b_axes_summed;
  ShiftAxes(&a_axes_summed, a_shape.ndim());
  ShiftAxes(&b_axes_summed, b_shape.ndim());

  size_t workspace_size = TensordotBackwardWorkspaceSize<xpu>(a_axes_summed, b_axes_summed,
                                                              out_grad, a, b, grad_a,
                                                              grad_b, req);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size),
                                                   ctx.get_stream<xpu>());
  TensordotBackwardImpl<xpu>(a_axes_summed, b_axes_summed, ctx, out_grad, a, b, grad_a,
                             grad_b, req, workspace);
}

struct TensordotIntAxesParam : public dmlc::Parameter<TensordotIntAxesParam> {
  int axes;
  DMLC_DECLARE_PARAMETER(TensordotIntAxesParam) {
    DMLC_DECLARE_FIELD(axes);
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axes_s;
    axes_s << axes;
    (*dict)["axes"] = axes_s.str();
  }
};

/**
 * gets summed axes of a and b from parameter axes.
 */
inline void GetSummedAxes(mxnet::Tuple<int>* a_axes_summed_ptr,
                          mxnet::Tuple<int>* b_axes_summed_ptr,
                          const int axes,
                          const mxnet::TShape& a_shape) {
  std::vector<int> a_axes_summed_vector;
  for (int i = 0; i < axes; i++) {
    a_axes_summed_vector.push_back(a_shape.ndim() - axes + i);
  }
  *a_axes_summed_ptr = mxnet::Tuple<int>(a_axes_summed_vector);

  std::vector<int> b_axes_summed_vector;
  for (int i = 0; i < axes; i++) {
    b_axes_summed_vector.push_back(i);
  }
  *b_axes_summed_ptr = mxnet::Tuple<int>(b_axes_summed_vector);
}

/**
 * Calculates tensordot.
 */
template<typename xpu>
void TensordotIntAxesImpl(const int axes,
                          const OpContext& ctx,
                          const TBlob& a,
                          const TBlob& b,
                          const TBlob& out,
                          const OpReqType req) {
  if (req == kNullOp) {
    return;
  }

  if (out.shape_.Size() == 0U) {
    return;  // zero-size output, no need to launch kernel
  }

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(out.type_flag_, a.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(out.type_flag_, b.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(out.type_flag_ == kFloat32 || out.type_flag_ == kFloat64 ||
       (out.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
      << "Tensordot only supports float32/float64 for CPU, and float16/float32/float64 for GPU";

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    if (a_shape.Size() == 0U || b_shape.Size() == 0U) {
      // 0-size input
      if (req != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
    } else if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
      // Both 0-D scalars, equivalent to multiply
      Tensor<xpu, 1, DType> a_data = a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_data = b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(out_data, req, a_data * b_data);
    } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
      // Either of them is a scalar, just scale by one of them
      const DType* tensor = (a_shape.ndim() == 0) ? b.dptr<DType>() : a.dptr<DType>();
      const DType* scalar = (a_shape.ndim() == 0) ? a.dptr<DType>() : b.dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        mxnet_op::Kernel<scalar_mul_kernel<Req>, xpu>::Launch(
          s, out.Size(), out.dptr<DType>(), tensor, scalar);
      });
    } else {
      // Two tensors of at least 1 dimensions.
      Tuple<int> a_axes_summed;
      Tuple<int> b_axes_summed;
      GetSummedAxes(&a_axes_summed, &b_axes_summed, axes, a_shape);

      Tuple<int> a_axes_remained;
      Tuple<int> b_axes_remained;
      Tuple<int> a_axes;
      Tuple<int> b_axes;
      GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
                      &b_axes, a_shape, b_shape);

      index_t ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
      GetMatrixDimensions(&ad1, &ad2, &bd1, &bd2, a_axes_remained, a_axes_summed,
                          b_axes_remained, b_axes_summed, a_shape, b_shape);
      MatrixDot<xpu>(ctx, a, b, out, req, ad1, ad2, bd1, bd2);
    }
  });
}

/**
 * forward function
 */
template<typename xpu>
void TensordotIntAxesOpForward(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];

  const TensordotIntAxesParam& param = nnvm::get<TensordotIntAxesParam>(attrs.parsed);
  const int axes = param.axes;

  TensordotIntAxesImpl<xpu>(axes, ctx, a, b, out, req[0]);
}

template<typename xpu>
void TensordotIntAxesBackwardImpl(const int axes,
                                  const OpContext& ctx,
                                  const TBlob& out_grad,
                                  const TBlob& a,
                                  const TBlob& b,
                                  const TBlob& grad_a,
                                  const TBlob& grad_b,
                                  const std::vector<OpReqType>& req) {
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  if ((a_shape.Size() == 0U) || (b_shape.Size() == 0U)) {
    return;  // zero-size output, no need to launch kernel
  }

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
      // Both 0-D scalars, equivalent to multiply
      Tensor<xpu, 1, DType> out_grad_data = out_grad.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> a_data = a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_data = b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> grad_a_data = grad_a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> grad_b_data = grad_b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(grad_a_data, req[0], b_data * out_grad_data);
      ASSIGN_DISPATCH(grad_b_data, req[1], a_data * out_grad_data);
    } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
      // Either of them is a scalar, just scale by one of them
      const TBlob& tensor = (a_shape.ndim() == 0) ? b : a;
      const TBlob& tensor_grad = (a_shape.ndim() == 0) ? grad_b : grad_a;
      const TBlob& scalar = (a_shape.ndim() == 0) ? a : b;
      const TBlob& scalar_grad = (a_shape.ndim() == 0) ? grad_a : grad_b;
      Tensor<xpu, 1, DType> scalar_ = scalar.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> scalar_grad_ = scalar_grad.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> tensor_ = tensor.FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> tensor_grad_ = tensor_grad.FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out_grad_ = out_grad.FlatTo1D<xpu, DType>(s);
      const OpReqType& tensor_req = (a_shape.ndim() == 0) ? req[1] : req[0];
      const OpReqType& scalar_req = (a_shape.ndim() == 0) ? req[0] : req[1];
      ASSIGN_DISPATCH(tensor_grad_, tensor_req,
                      broadcast_scalar(scalar_, tensor_grad_.shape_) * out_grad_);
      Tensor<xpu, 1, DType> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, DType>(Shape1(out_grad.shape_.Size()), s);
      ASSIGN_DISPATCH(workspace, kWriteTo, tensor_ * out_grad_);

      NP_TENSORDOT_REDUCE_AXES(true, ctx, {TBlob(workspace)}, {scalar_req},
                               {TBlob(scalar_grad_)}, scalar_grad_.shape_);
    } else {
      // Two tensors of at least 1 dimensions.
      Tuple<int> a_axes_summed;
      Tuple<int> b_axes_summed;
      GetSummedAxes(&a_axes_summed, &b_axes_summed, axes, a_shape);

      Tuple<int> a_axes_remained;
      Tuple<int> b_axes_remained;
      Tuple<int> a_axes;
      Tuple<int> b_axes;
      GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
                      &b_axes, a_shape, b_shape);

      index_t ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
      GetMatrixDimensions(&ad1, &ad2, &bd1, &bd2, a_axes_remained, a_axes_summed,
                          b_axes_remained, b_axes_summed, a_shape, b_shape);

      MatrixDot<xpu>(ctx, a, out_grad, grad_b, req[1], ad1, ad2, ad1, bd2, true, false);
      MatrixDot<xpu>(ctx, out_grad, b, grad_a, req[0], ad1, bd2, bd1, bd2, false, true);
    }
  });
}

#undef NP_TENSORDOT_REDUCE_AXES

/**
 * backward function.
 */
template<typename xpu>
void TensordotIntAxesOpBackward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);

  const TBlob& out_grad = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];

  const TensordotIntAxesParam& param = nnvm::get<TensordotIntAxesParam>(attrs.parsed);
  const int axes = param.axes;

  TensordotIntAxesBackwardImpl<xpu>(axes, ctx, out_grad, a, b, grad_a, grad_b, req);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_TENSORDOT_OP_INL_H_
