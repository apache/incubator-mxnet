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
 * \file np_kron-inl.h
 * \brief Function definition of matrix numpy-compatible kron operator
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_KRON_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_KRON_INL_H_

#include <vector>
#include "np_tensordot_op-inl.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

template<int ndim, int req>
struct kron {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out,
                                  const DType* a, const DType* b,
                                  mshadow::Shape<ndim> ashape,
                                  mshadow::Shape<ndim> bshape,
                                  mshadow::Shape<ndim> oshape) {
    using namespace mxnet_op;

    auto k = unravel(i, oshape);
    Shape<ndim> ia;
    Shape<ndim> jb;
    for (int q = 0; q < ndim; q++) {
      ia[q] = static_cast<int>(k[q] / bshape[q]);
      jb[q] = k[q] % bshape[q];
    }
    auto idx_a = ravel(ia, ashape);
    auto idx_b = ravel(jb, bshape);

    KERNEL_ASSIGN(out[i], req, a[idx_a] * b[idx_b]);
  }
};

template<int ndim, int req>
struct kron_back_a {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* agrad,
                                  const DType* b, const DType* ograd,
                                  mshadow::Shape<ndim> ashape,
                                  mshadow::Shape<ndim> bshape,
                                  mshadow::Shape<ndim> oshape) {
    using namespace mxnet_op;

    auto ia = unravel(i, ashape);
    Shape<ndim> k;
    DType temp_agrad = 0;

    for (index_t idx_b = 0; idx_b < bshape.Size(); idx_b++) {
      auto jb = unravel(idx_b, bshape);
      for (int q = 0; q < ndim; q++) {
        k[q] = ia[q]*bshape[q] + jb[q];
      }
      auto idx_o = ravel(k, oshape);
      temp_agrad += b[idx_b]*ograd[idx_o];
    }
    KERNEL_ASSIGN(agrad[i], req, temp_agrad);
  }
};

template<int ndim, int req>
struct kron_back_b {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, const DType* a,
                                  DType* bgrad, const DType* ograd,
                                  mshadow::Shape<ndim> ashape,
                                  mshadow::Shape<ndim> bshape,
                                  mshadow::Shape<ndim> oshape) {
    using namespace mxnet_op;

    auto jb = unravel(i, bshape);
    Shape<ndim> k;
    DType temp_bgrad = 0;

    for (index_t idx_a = 0; idx_a < ashape.Size(); idx_a++) {
      auto ia = unravel(idx_a, ashape);
      for (int q = 0; q < ndim; q++) {
        k[q] = ia[q] * bshape[q] + jb[q];
      }
      auto idx_o = ravel(k, oshape);
      temp_bgrad += a[idx_a]*ograd[idx_o];
    }
    KERNEL_ASSIGN(bgrad[i], req, temp_bgrad);
  }
};

template<typename xpu>
void KronOpForwardImpl(const OpContext& ctx,
                       OpReqType req,
                       const TBlob& a,
                       const TBlob& b,
                       const TBlob& out
                       ) {
  using namespace mshadow;

  if (req == kNullOp) {
    return;
  }

  if (out.shape_.Size() == 0U) {
    return;  // zero-size output, no need to launch kernel
  }

  const mxnet::TShape& ashape = a.shape_;
  const mxnet::TShape& bshape = b.shape_;
  const mxnet::TShape& oshape = out.shape_;


  // TensordotIntAxesImpl<xpu>(0, ctx, a, b, out, req[0]);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
    if (ashape.Size() == 0U || bshape.Size() == 0U) {
      // 0-size input
      if (req != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
    } else if (ashape.ndim() == 0 && bshape.ndim() == 0) {
      // Both 0-D scalars, equivalent to multiply
      Tensor<xpu, 1, DType> a_data = a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_data = b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(out_data, req, a_data * b_data);
    } else if (ashape.ndim() == 0 || bshape.ndim() == 0) {
      // Either of them is a scalar, just scale by one of them
      const DType* tensor = (ashape.ndim() == 0) ? b.dptr<DType>() : a.dptr<DType>();
      const DType* scalar = (ashape.ndim() == 0) ? a.dptr<DType>() : b.dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        mxnet_op::Kernel<scalar_mul_kernel<Req>, xpu>::Launch(
          s, out.Size(), out.dptr<DType>(), tensor, scalar);
      });
    } else {
      MXNET_NDIM_SWITCH(oshape.ndim(), ndim, {
        Shape<ndim> ashape_ = oshape.get<ndim>();
        Shape<ndim> bshape_ = oshape.get<ndim>();
        Shape<ndim> oshape_ = oshape.get<ndim>();
        int temp = ashape.ndim()-bshape.ndim();
        int s_dim = (temp > 0)?bshape.ndim():ashape.ndim();
        for (int i = 0; i < s_dim; i++) {
          ashape_[ndim - i - 1] = ashape[ashape.ndim() - i - 1];
          bshape_[ndim - i - 1] = bshape[bshape.ndim() - i - 1];
          oshape_[ndim - i - 1] = oshape[oshape.ndim() - i - 1];
        }
        if (temp > 0) {
          for (int i = s_dim; i < ndim; i++) {
            ashape_[ndim - i - 1] = ashape[ashape.ndim() - i - 1];
            bshape_[ndim - i - 1] = 1;
            oshape_[ndim - i - 1] = oshape[oshape.ndim() - i - 1];
          }
        } else {
          for (int i = s_dim; i < ndim; i++) {
            ashape_[ndim - i - 1] = 1;
            bshape_[ndim - i - 1] = bshape[bshape.ndim() - i - 1];
            oshape_[ndim - i - 1] = oshape[oshape.ndim() - i - 1];
          }
        }
        MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
          mxnet_op::Kernel<kron<ndim, req_type>, xpu>::Launch(
            s, out.Size(), out.dptr<DType>(), a.dptr<DType>(), b.dptr<DType>(),
            ashape_, bshape_, oshape_);
        });
      });
    }
  });
}

#if !defined(__CUDACC__)
#define NP_KRON_REDUCE_AXES(safe_acc, workspace, ...) \
  ReduceAxesComputeImpl<xpu, mshadow_op::sum, safe_acc>(__VA_ARGS__, &workspace)
#else
#define NP_KRON_REDUCE_AXES(safe_acc, workspace, ...) \
  ReduceAxesRTCComputeImpl(__VA_ARGS__, "red::sum{}", &workspace)
#endif

template<typename xpu>
void KronOpBackwardImpl(const OpContext& ctx,
                        const std::vector<OpReqType>& req,
                        const TBlob& a,
                        const TBlob& b,
                        const TBlob& ograd,
                        const TBlob& agrad,
                        const TBlob& bgrad) {
  const mxnet::TShape& ashape = a.shape_;
  const mxnet::TShape& bshape = b.shape_;
  const mxnet::TShape& oshape = ograd.shape_;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
    if (ashape.ndim() == 0 && bshape.ndim() == 0) {
      // Both 0-D scalars, equivalent to multiply
      Tensor<xpu, 1, DType> ograd_data = ograd.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> a_data = a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_data = b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> agrad_data = agrad.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> bgrad_data = bgrad.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(agrad_data, req[0], b_data * ograd_data);
      ASSIGN_DISPATCH(bgrad_data, req[1], a_data * ograd_data);
    } else if (ashape.ndim() == 0 || bshape.ndim() == 0) {
      // Either of them is a scalar, just scale by one of them
      const TBlob& tensor = (ashape.ndim() == 0) ? b : a;
      const TBlob& tensor_grad = (ashape.ndim() == 0) ? bgrad : agrad;
      const TBlob& scalar = (ashape.ndim() == 0) ? a : b;
      const TBlob& scalar_grad = (ashape.ndim() == 0) ? agrad : bgrad;
      Tensor<xpu, 1, DType> scalar_ = scalar.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> scalar_grad_ = scalar_grad.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> tensor_ = tensor.FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> tensor_grad_ = tensor_grad.FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> ograd_ = ograd.FlatTo1D<xpu, DType>(s);
      const OpReqType& tensor_req = (ashape.ndim() == 0) ? req[1] : req[0];
      const OpReqType& scalar_req = (ashape.ndim() == 0) ? req[0] : req[1];
      ASSIGN_DISPATCH(tensor_grad_, tensor_req,
                      broadcast_scalar(scalar_, tensor_grad_.shape_) * ograd_);
      TShape src_shape, dst_shape;
      BroadcastReduceShapeCompact(ograd.shape_, scalar_grad_.shape_, &src_shape, &dst_shape);
      size_t workspace_size = broadcast::ReduceWorkspaceSize(s, dst_shape,
                                                             {scalar_req}, src_shape);
      constexpr size_t align_size = 1024;
      const size_t aligned_first_workspace_size = ((workspace_size + align_size - 1) / align_size)
                                                  * align_size;
      workspace_size = aligned_first_workspace_size + ograd.shape_.Size() * sizeof(DType);
      Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      Tensor<xpu, 1, DType> temp(reinterpret_cast<DType*>(workspace.dptr_ +
                                                          aligned_first_workspace_size),
                                 Shape1(ograd.shape_.Size()), s);
      ASSIGN_DISPATCH(temp, kWriteTo, tensor_ * ograd_);

      NP_KRON_REDUCE_AXES(true, workspace, ctx, {TBlob(temp)}, {scalar_req},
                          {TBlob(scalar_grad_)}, scalar_grad_.shape_);
    } else {
      MXNET_NDIM_SWITCH(oshape.ndim(), ndim, {
        Shape<ndim> ashape_ = oshape.get<ndim>();
        Shape<ndim> bshape_ = oshape.get<ndim>();
        Shape<ndim> oshape_ = oshape.get<ndim>();
        int temp = ashape.ndim()-bshape.ndim();
        int s_dim =  (temp > 0)?bshape.ndim():ashape.ndim();
        for (int i = 0; i < s_dim; i++) {
          ashape_[ndim - i - 1] = ashape[ashape.ndim() - i - 1];
          bshape_[ndim - i - 1] = bshape[bshape.ndim() - i - 1];
          oshape_[ndim - i - 1] = oshape[oshape.ndim() - i - 1];
        }
        if (temp > 0) {
          for (int i = s_dim; i < ndim; i++) {
            ashape_[ndim - i - 1] = ashape[ashape.ndim() - i - 1];
            bshape_[ndim - i - 1] = 1;
            oshape_[ndim - i - 1] = oshape[oshape.ndim() - i - 1];
          }
        } else {
          for (int i = s_dim; i < ndim; i++) {
            ashape_[ndim - i - 1] = 1;
            bshape_[ndim - i - 1] = bshape[bshape.ndim() - i - 1];
            oshape_[ndim - i - 1] = oshape[oshape.ndim() - i - 1];
          }
        }
        MSHADOW_TYPE_SWITCH(agrad.type_flag_, DType, {
          MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
            mxnet_op::Kernel<kron_back_a<ndim, req_type>, xpu>::Launch(
              s, agrad.Size(), agrad.dptr<DType>(), b.dptr<DType>(), ograd.dptr<DType>(),
              ashape_, bshape_, oshape_);
          });
        });
        MSHADOW_TYPE_SWITCH(bgrad.type_flag_, DType, {
          MXNET_ASSIGN_REQ_SWITCH(req[1], req_type, {
            mxnet_op::Kernel<kron_back_b<ndim, req_type>, xpu>::Launch(
              s, bgrad.Size(), a.dptr<DType>(), bgrad.dptr<DType>(), ograd.dptr<DType>(),
              ashape_, bshape_, oshape_);
          });
        });
      });
    }
  });
}

#undef NP_KRON_REDUCE_AXES

template<typename xpu>
inline void KronOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];

  KronOpForwardImpl<xpu>(ctx, req[0], a, b, out);
}


template<typename xpu>
inline void KronOpBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  using namespace mshadow;

  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& ograd = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];

  KronOpBackwardImpl<xpu>(ctx, req, a, b, ograd, grad_a, grad_b);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_KRON_INL_H_
