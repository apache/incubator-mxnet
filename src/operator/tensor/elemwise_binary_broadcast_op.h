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
 * \file elementwise_binary_broadcast_op.h
 * \brief Function definition of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./elemwise_binary_op.h"
#include "../operator_common.h"
#include "broadcast_reduce-inl.h"

namespace mxnet {
namespace op {
inline bool BinaryBroadcastShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  TShape& lhs = (*in_attrs)[0];
  TShape& rhs = (*in_attrs)[1];

  // avoid pre-mature shape inference.
  if (lhs.ndim() == 0 || rhs.ndim() == 0) return false;

  if (lhs == rhs) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, lhs);
    return true;
  }
  TShape out(std::max(lhs.ndim(), rhs.ndim()));
  index_t bl = out.ndim() - lhs.ndim();
  index_t br = out.ndim() - rhs.ndim();
  for (index_t i = 0; i < out.ndim(); ++i) {
    index_t l = 1, r = 1;
    if (i >= bl) l = lhs[i-bl];
    if (i >= br) r = rhs[i-br];
    if (l != r) {
      if (l == 0 || r == 0) {
        out[i] = 0;
      } else {
        CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes " << lhs << " " << rhs;
        out[i] = std::max(l, r);
      }
    } else {
      out[i] = l;
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
  return true;
}

#define BROADCAST_NDIM_SWITCH(ndim, NDim, ...)  \
  if (ndim <= 2) {                    \
    const int NDim = 2;               \
    {__VA_ARGS__}                     \
  } else if (ndim <= 4) {             \
    const int NDim = 4;               \
    {__VA_ARGS__}                     \
  } else if (ndim <= broadcast::MAX_DIM) {  \
    const int NDim = broadcast::MAX_DIM;    \
    {__VA_ARGS__}                     \
  } else {                            \
    LOG(FATAL) << "NDim too large ";  \
  }

inline int BinaryBroadcastShapeCompact(const TShape& lshape, const TShape& rshape,
                                       const TShape& oshape, TShape *new_lshape,
                                       TShape *new_rshape, TShape *new_oshape) {
  if (lshape == rshape) return 0;
  index_t odim = std::max<index_t>(oshape.ndim(), broadcast::MAX_DIM);
  *new_lshape = TShape(odim);
  *new_rshape = TShape(odim);
  *new_oshape = TShape(odim);
  index_t bl = oshape.ndim() - lshape.ndim();
  index_t br = oshape.ndim() - rshape.ndim();
  index_t j = 0, lprod = 1, rprod = 1, oprod = 1;
  for (index_t i = 0; i < oshape.ndim(); ++i) {
    index_t l = 1, r = 1, o = oshape[i];
    if (i >= bl) l = lshape[i-bl];
    if (i >= br) r = rshape[i-br];
    if ((lprod != rprod || l != r) &&
        lprod*l > 1 && rprod*r > 1) {
      (*new_lshape)[j] = lprod;
      (*new_rshape)[j] = rprod;
      (*new_oshape)[j] = oprod;
      lprod = rprod = oprod = 1; ++j;
    }
    lprod *= l;
    rprod *= r;
    oprod *= o;
  }
  if (lprod > 1 || rprod > 1) {
    (*new_lshape)[j] = lprod;
    (*new_rshape)[j] = rprod;
    (*new_oshape)[j] = oprod;
    ++j;
  }
  if (j <= broadcast::MAX_DIM) {
    BROADCAST_NDIM_SWITCH(j, NDim, {
      new_lshape->assign(&(*new_lshape)[0], &(*new_lshape)[NDim]);
      new_rshape->assign(&(*new_rshape)[0], &(*new_rshape)[NDim]);
      new_oshape->assign(&(*new_oshape)[0], &(*new_oshape)[NDim]);
    });
  } else {
    LOG(FATAL) << "Too many broadcast dimensions with operands " << lshape << " " << rshape;
  }
  return j;
}

inline bool BinaryBroadcastStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  using namespace common;
  bool dispatched = false;
  if (!dispatched && ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, dns -> dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  const auto lhs_stype = in_attrs->at(0);
  const auto rhs_stype = in_attrs->at(1);
  if (!dispatched && rhs_stype == kCSRStorage && lhs_stype == kDefaultStorage) {
    // dns, csr -> dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }

  return dispatched;
}

namespace mxnet_op {
template<int ndim, typename DType, typename OP>
struct binary_broadcast_kernel {
  /*! \brief Map function for binary_broadcast_kernel */
  MSHADOW_XINLINE static void Map(int base, int length, OpReqType req,
                                  const Shape <ndim> &lstride, const Shape <ndim> &rstride,
                                  const Shape <ndim> &oshape, DType *lhs, DType *rhs,
                                  DType *out) {
    Shape <ndim> coord = unravel(base, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto ridx = static_cast<index_t>(dot(coord, rstride));
    KERNEL_ASSIGN(out[base], req, OP::Map(lhs[lidx], rhs[ridx]));
    // starts from 1 to avoid extra inc at end of loop
    for (int i = 1; i < length; ++i) {
      inc(&coord, oshape, &lidx, lstride, &ridx, rstride);
      // When tuning, don't actually run the op, since it's not going to be tuned against
      // the actual op we'll eventually be using
      KERNEL_ASSIGN(out[base + i], req, OP::Map(lhs[lidx], rhs[ridx]));
    }
  }
};
}  // namespace mxnet_op

template<typename xpu, typename OP>
void BinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryOp::Compute<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    if (req[0] != kNullOp) {
      mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
      MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        BROADCAST_NDIM_SWITCH(ndim, NDim, {
          mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
          mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
          mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, DType, OP>, xpu>::
          template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
          inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), outputs[0].dptr<DType>());
        });
      });
    }
  }
}

template<typename xpu, typename OP, bool sparse_kernel>
void BinaryBroadcastComputeEx(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace common;
  TShape new_lshape, new_rshape, new_oshape;
  NDArray lhs = inputs[0];
  NDArray rhs = inputs[1];
  NDArray out  = outputs[0];
  int ndim = BinaryBroadcastShapeCompact(lhs.shape(), rhs.shape(),
    out.shape(), &new_lshape, &new_rshape, &new_oshape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (!ndim && lhs.storage_type() == kDefaultStorage &&
      rhs.storage_type() == kCSRStorage &&
      out.storage_type() == kDefaultStorage) {
    const TShape dshape = lhs.shape();
    const nnvm::dim_t num_rows = dshape[0];
    const nnvm::dim_t row_length = dshape[1];
    // dns, csr -> dns
    CHECK_EQ(rhs.aux_type(csr::kIdx), rhs.aux_type(csr::kIndPtr));
    MSHADOW_IDX_TYPE_SWITCH(rhs.aux_type(csr::kIdx), IType, {
      MSHADOW_TYPE_SWITCH(out.dtype(), DType, {
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        const IType* csr_indptr = rhs.aux_data(csr::kIndPtr).dptr<IType>();
        const IType* csr_idx = rhs.aux_data(csr::kIdx).dptr<IType>();
        const DType* csr_data = rhs.data().dptr<DType>();
        const DType* data_ptr = lhs.data().dptr<DType>();
        DType* out_ptr = out.data().dptr<DType>();
        if (sparse_kernel) {
          if (req[0] != kWriteInplace) {
            Kernel<op_with_req<mshadow_op::identity, Req>, xpu>::Launch(s,
              dshape.Size(), out_ptr, data_ptr);
          }
          Kernel<DnsCsrSparseKernel<OP, Req>, xpu>::Launch(s, num_rows,
            out_ptr, data_ptr, csr_data, csr_idx, csr_indptr, row_length);
        } else {
          Kernel<DnsCsrKernel<OP, Req>, xpu>::Launch(s, num_rows,
            out_ptr, data_ptr, csr_data, csr_idx, csr_indptr, row_length);
        }
      });
      });
    });
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace broadcast;
  TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryOp::BackwardUseNone<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      const TBlob lhs = outputs[0].reshape(new_lshape);
      const TBlob rhs = outputs[1].reshape(new_rshape);
      const TBlob out = inputs[0].reshape(new_oshape);
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        // Request temporary storage
        size_t workspace_size_l = ReduceWorkspaceSize<NDim, DType>(s, lhs, req[0], out);
        size_t workspace_size_r = ReduceWorkspaceSize<NDim, DType>(s, rhs, req[1], out);
        size_t workspace_size = std::max(workspace_size_l, workspace_size_r);
        Tensor<xpu, 1, char> workspace =
          ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
        Reduce<red::sum, NDim, DType, LOP>(s, lhs, req[0], workspace, out);
        Reduce<red::sum, NDim, DType, ROP>(s, rhs, req[1], workspace, out);
      });
    });
  }
}

template<typename xpu, int ndim, typename DType, typename LOP, typename ROP>
inline void BinaryBroadcastBackwardUseInImpl(const OpContext& ctx,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<OpReqType>& req,
                                             const std::vector<TBlob>& outputs,
                                             const TShape& new_lshape,
                                             const TShape& new_rshape,
                                             const TShape& new_oshape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob lgrad = outputs[0].reshape(new_lshape);
  const TBlob rgrad = outputs[1].reshape(new_rshape);
  const TBlob ograd = inputs[0].reshape(new_oshape);
  const TBlob lhs = inputs[1].reshape(new_lshape);
  const TBlob rhs = inputs[2].reshape(new_rshape);
  size_t workspace_size_l = ReduceWorkspaceSize<ndim, DType>(s, lgrad, req[0], ograd, lhs, rhs);
  size_t workspace_size_r = ReduceWorkspaceSize<ndim, DType>(s, rgrad, req[1], ograd, lhs, rhs);
  size_t workspace_size = std::max(workspace_size_l, workspace_size_r);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
  Reduce<red::sum, ndim, DType, op::mshadow_op::mul, LOP>(s, lgrad, req[0], workspace,
    ograd, lhs, rhs);
  Reduce<red::sum, ndim, DType, op::mshadow_op::mul, ROP>(s, rgrad, req[1], workspace,
    ograd, lhs, rhs);
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseIn(const nnvm::NodeAttrs& attrs,
                                  const OpContext& ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs) {
  TShape new_lshape, new_rshape, new_oshape;
  const bool need_bc = BinaryBroadcastShapeCompact(outputs[0].shape_,
                                                   outputs[1].shape_, inputs[0].shape_,
                                                   &new_lshape, &new_rshape, &new_oshape) != 0;
  if (!need_bc) {
    ElemwiseBinaryOp::BackwardUseIn<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(new_oshape.ndim(), NDim, {
        BinaryBroadcastBackwardUseInImpl<xpu, NDim, DType, LOP, ROP>(
          ctx, inputs, req, outputs, new_lshape, new_rshape, new_oshape);
      });
    });
  }
}

#define MXNET_OPERATOR_REGISTER_BINARY_BROADCAST(name)                \
  NNVM_REGISTER_OP(name)                                              \
  .set_num_inputs(2)                                                  \
  .set_num_outputs(1)                                                 \
  .set_attr<nnvm::FListInputNames>("FListInputNames",                 \
    [](const NodeAttrs& attrs) {                                      \
      return std::vector<std::string>{"lhs", "rhs"};                  \
    })                                                                \
  .set_attr<nnvm::FInferShape>("FInferShape", BinaryBroadcastShape)   \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)       \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                   \
    [](const NodeAttrs& attrs){                                       \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};       \
    })                                                                \
  .add_argument("lhs", "NDArray-or-Symbol", "First input to the function")                      \
  .add_argument("rhs", "NDArray-or-Symbol", "Second input to the function")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_
