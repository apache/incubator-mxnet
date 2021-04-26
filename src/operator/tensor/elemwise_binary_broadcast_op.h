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
 * \file elemwise_binary_broadcast_op.h
 * \brief Function definition of elementwise binary broadcast operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_BROADCAST_OP_H_

#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
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
                                 mxnet::ShapeVector *in_attrs,
                                 mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& lhs = (*in_attrs)[0];
  mxnet::TShape& rhs = (*in_attrs)[1];

  // avoid pre-mature shape inference.
  if (!mxnet::ndim_is_known(lhs) || !mxnet::ndim_is_known(rhs)) return false;

  if (lhs == rhs) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, lhs);
    return shape_is_known(lhs);
  }
  mxnet::TShape out(std::max(lhs.ndim(), rhs.ndim()), -1);
  const int bl = out.ndim() - lhs.ndim();
  const int br = out.ndim() - rhs.ndim();
  for (int i = 0; i < out.ndim(); ++i) {
    dim_t l = 1, r = 1;
    if (i >= bl) l = lhs[i-bl];
    if (i >= br) r = rhs[i-br];
    if (!mxnet::dim_size_is_known(l) || !mxnet::dim_size_is_known(r)) continue;
    if (l != r) {
      // Make it compatible with NumPy.
      // For example, (2, 3) cannot broadcast to (2, 0, 3), but (1, 3) can broadcast to (2, 0, 3).
      CHECK(l == 1 || r == 1)
        << "operands could not be broadcast together with shapes " << lhs << " " << rhs;
      out[i] = (l == 1 ? r : l);
    } else {
      out[i] = l;
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, out);
  return shape_is_known(lhs) && shape_is_known(rhs) && shape_is_known(out);
}

inline bool BinaryBroadcastMulStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int lhs_stype = in_attrs->at(0);
  const int rhs_stype = in_attrs->at(1);
  int& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage) {
    dispatched = storage_type_assign(&out_stype, kCSRStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

inline bool BinaryBroadcastAddStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int>* in_attrs,
                                          std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int lhs_stype = in_attrs->at(0);
  const int rhs_stype = in_attrs->at(1);
  int& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && ((lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage) ||
                      (lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage))) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
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

inline int BinaryBroadcastShapeCompact(const mxnet::TShape& lshape, const mxnet::TShape& rshape,
                                       const mxnet::TShape& oshape, mxnet::TShape *new_lshape,
                                       mxnet::TShape *new_rshape, mxnet::TShape *new_oshape) {
  if (lshape == rshape) return 0;
  const int odim = std::max(oshape.ndim(), broadcast::MAX_DIM);
  *new_lshape = mxnet::TShape(odim, 1);
  *new_rshape = mxnet::TShape(odim, 1);
  *new_oshape = mxnet::TShape(odim, 1);
  int bl = oshape.ndim() - lshape.ndim();
  int br = oshape.ndim() - rshape.ndim();
  int j = 0;
  index_t lprod = 1, rprod = 1, oprod = 1;
  for (int i = 0; i < oshape.ndim(); ++i) {
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
      new_lshape->assign(new_lshape->begin(), new_lshape->begin() + NDim);
      new_rshape->assign(new_rshape->begin(), new_rshape->begin() + NDim);
      new_oshape->assign(new_oshape->begin(), new_oshape->begin() + NDim);
    });
  } else {
    LOG(FATAL) << "Too many broadcast dimensions with operands " << lshape << " " << rshape;
  }

  return j;
}

template<typename xpu, typename OP>
void BinaryBroadcastIntCompute(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0U) return;
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryOp::ComputeInt<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    if (req[0] == kNullOp) return;
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    if (outputs[0].type_flag_ == mshadow::kBool) {
      LOG(FATAL) << "Operator " << attrs.op->name << " does not support boolean type";
    }
    MXNET_INT_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
        mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
        mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::
        template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), outputs[0].dptr<DType>());
      });
    });
  }
}

template<typename xpu, typename OP>
void BinaryBroadcastCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0U) return;
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryOp::Compute<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    if (req[0] == kNullOp) return;
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    if (outputs[0].type_flag_ == mshadow::kBool) {
      LOG(FATAL) << "Operator " << attrs.op->name << " does not support boolean type";
    }
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        broadcast::BinaryBroadcastComputeImpl<NDim, DType, OP>(s, req[0],
                                                               inputs[0].reshape(new_lshape),
                                                               inputs[1].reshape(new_rshape),
                                                               outputs[0].reshape(new_oshape));
      });
    });
  }
}

#if MXNET_USE_CUDA

struct BinaryBroadcastRTCCompute {
  std::string OP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

struct BinaryBroadcastRTCBackwardUseNone {
  std::string LOP;
  std::string ROP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

struct BinaryBroadcastRTCBackwardUseIn {
  std::string LOP;
  std::string ROP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

#endif  // MXNET_USE_CUDA

template<typename xpu, typename OP>
void BinaryBroadcastComputeWithBool(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0U) return;
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryOp::ComputeWithBool<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    if (req[0] == kNullOp) return;
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, DType, {
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
        mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
        mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
        mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::
        template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
        inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), outputs[0].dptr<DType>());
      });
    });
  }
}

template<typename xpu, typename OP>
void BinaryBroadcastComputeLogic(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0U) return;
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  const TBlob& lhs = inputs[0];
  const TBlob& rhs = inputs[1];
  const TBlob& out = outputs[0];
  int ndim = BinaryBroadcastShapeCompact(lhs.shape_, rhs.shape_, out.shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryOp::ComputeLogic<xpu, OP>(attrs, ctx, inputs, req, outputs);
  } else {
    if (req[0] == kNullOp) return;
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH_WITH_BOOL(lhs.type_flag_, DType, {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(rhs.type_flag_, EType, {
         BROADCAST_NDIM_SWITCH(ndim, NDim, {
          mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
          mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
          mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
          mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, OP>, xpu>::
          template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
                            lhs.dptr<DType>(), rhs.dptr<EType>(),
                            out.dptr<bool>());
        });
      });
    });
  }
}

template<typename xpu, typename OP>
void BinaryBroadcastCsrDnsCsrImpl(const OpContext& ctx,
                                  const NDArray& csr,
                                  const NDArray& dns,
                                  const OpReqType req,
                                  const NDArray& output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  CHECK(req != kAddTo && req != kWriteInplace);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  bool col_vec;
  if (dns.shape().ndim() == 1) {
    col_vec = false;
  } else {
    col_vec = (dns.shape()[0] == csr.shape()[0])? true : false;
  }

  if (csr.storage_initialized()) {
    const nnvm::dim_t nnz = csr.storage_shape()[0];
    const nnvm::dim_t num_rows = output.shape()[0];
    output.CheckAndAlloc({Shape1(num_rows + 1), Shape1(nnz)});

    MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
      MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), CType, {
        MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIndPtr), RType, {
          MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
            // broadcast_mul/div between csr and a scalar case
            if ((dns.shape().ndim() == 2 && dns.shape()[0] == 1 && dns.shape()[1] == 1) ||
                (dns.shape().ndim() == 1 && dns.shape()[0] == 1)) {
              Kernel<csr_dns_csr_broadcast_kernel<req_type, OP, false>, xpu>::Launch(
                s, nnz, csr.data().dptr<DType>(), dns.data().dptr<DType>(),
                output.data().dptr<DType>(), nnz);
            } else {
              // broadcast_mul/div between csr and column vector
              if (col_vec) {
                Kernel<csr_dns_csr_broadcast_kernel<req_type, OP, true>, xpu>::Launch(
                  s, num_rows, csr.data().dptr<DType>(), csr.aux_data(kIdx).dptr<CType>(),
                  csr.aux_data(kIndPtr).dptr<RType>(), dns.data().dptr<DType>(),
                  output.data().dptr<DType>());
              // broadcast_mul/div between csr and row vector
              } else {
                Kernel<csr_dns_csr_broadcast_kernel<req_type, OP, false>, xpu>::Launch(
                  s, num_rows, csr.data().dptr<DType>(), csr.aux_data(kIdx).dptr<CType>(),
                  csr.aux_data(kIndPtr).dptr<RType>(), dns.data().dptr<DType>(),
                  output.data().dptr<DType>());
              }
            }
            Copy(output.aux_data(kIdx).FlatTo1D<xpu, CType>(),
                 csr.aux_data(kIdx).FlatTo1D<xpu, CType>(), s);
            Copy(output.aux_data(kIndPtr).FlatTo1D<xpu, RType>(),
                 csr.aux_data(kIndPtr).FlatTo1D<xpu, RType>(), s);
          });
        });
      });
    });
  // If input csr is an empty matrix, fill zeros and return
  } else {
    FillZerosCsrImpl(s, output);
    return;
  }
}

template<typename xpu, typename OP>
void BinaryBroadcastCsrDnsDnsImpl(const OpContext& ctx,
                                  const NDArray& csr,
                                  const NDArray& dns,
                                  const OpReqType req,
                                  const NDArray& output,
                                  const mxnet::TShape& new_csrshape,
                                  const mxnet::TShape& new_dnsshape,
                                  const mxnet::TShape& new_oshape,
                                  const int ndim,
                                  const bool reverse) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  CHECK(req == kWriteTo) << "Only kWriteTo supported for broadcast(csr, dns) = dns";
  const bool legal_op = std::is_same<OP, mshadow_op::plus>::value ||
                        std::is_same<OP, mshadow_op::minus>::value;
  CHECK(legal_op) << "Only add/sub are supported for broadcast(csr, dns) = dns";
  CHECK_EQ(csr.shape()[0], output.shape()[0]);
  CHECK_EQ(csr.shape()[1], output.shape()[1]);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const nnvm::dim_t num_rows = output.shape()[0];
  const nnvm::dim_t num_cols = output.shape()[1];
  const TBlob& csr_data = csr.data();
  const TBlob& csr_indices = csr.aux_data(kIdx);
  const TBlob& csr_indptr = csr.aux_data(kIndPtr);
  TBlob dns_data = dns.data();
  TBlob out_data = output.data();

  MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
    BROADCAST_NDIM_SWITCH(ndim, NDim, {
      Shape<NDim> oshape = new_oshape.get<NDim>();
      Shape<NDim> lstride = calc_stride(new_csrshape.get<NDim>());
      Shape<NDim> rstride = calc_stride(new_dnsshape.get<NDim>());
      if (reverse && std::is_same<OP, mshadow_op::minus>::value) {
        Kernel<binary_broadcast_kernel<NDim, mshadow_op::plus>, xpu>::
        template LaunchEx(s, new_oshape.Size(), req, lstride, rstride, oshape,
        DType(0), dns_data.dptr<DType>(), out_data.dptr<DType>());
      } else {
        Kernel<binary_broadcast_kernel<NDim, OP>, xpu>::
        template LaunchEx(s, new_oshape.Size(), req, lstride, rstride, oshape,
        DType(0), dns_data.dptr<DType>(), out_data.dptr<DType>());
      }
    });
  });
  if (csr.storage_initialized()) {
    MSHADOW_TYPE_SWITCH(csr.dtype(), DType, {
      MSHADOW_IDX_TYPE_SWITCH(csr.aux_type(kIdx), CType, {
        MSHADOW_IDX_TYPE_SWITCH(csr.aux_type(kIndPtr), RType, {
          MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
            if (reverse && std::is_same<OP, mshadow_op::minus>::value) {
              Kernel<csr_dns_map_kernel<req_type, mshadow_op::minus, true>, xpu>::Launch(
                s, num_rows, csr_data.dptr<DType>(), csr_indices.dptr<CType>(),
                csr_indptr.dptr<RType>(), out_data.dptr<DType>(), num_rows, num_cols);
            } else {
              Kernel<csr_dns_map_kernel<req_type, mshadow_op::plus>, xpu>::Launch(
                s, num_rows, csr_data.dptr<DType>(), csr_indices.dptr<CType>(),
                csr_indptr.dptr<RType>(), out_data.dptr<DType>(), num_rows, num_cols);
            }
          });
        });
      });
    });
  }
}

template<typename xpu, typename OP>
void BinaryBroadcastComputeSparseEx(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_LE(inputs[1].shape().ndim(), 2)
    << "input dense matrix should have less than or equal to 2 dimensions";
  if (req[0] == kNullOp) return;
  const NDArray& lhs = inputs[0];
  const NDArray& rhs = inputs[1];
  const NDArray& out = outputs[0];
  const auto lhs_stype = lhs.storage_type();
  const auto rhs_stype = rhs.storage_type();
  const auto out_stype = out.storage_type();
  // If the input is a matrix with the same shape, should be elemwise
  if ((rhs.shape().ndim() != 1U) && (rhs.shape()[0] != 1) && (rhs.shape()[1] != 1)) {
    if (lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage && out_stype == kCSRStorage) {
      const bool supported_op = std::is_same<OP, mshadow_op::mul>::value;
      CHECK(supported_op)
        << "Please use elemwise_div for division between csr and dense of the same shape";
      ElemwiseBinaryOp::DnsCsrCsrOp<xpu, mshadow_op::mul>(attrs, ctx, rhs, lhs, req[0], out, true);
    }
  } else {
    // broadcast(CSR, Dense(1D)) = CSR
    if (lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage && out_stype == kCSRStorage) {
      BinaryBroadcastCsrDnsCsrImpl<xpu, OP>(ctx, lhs, rhs, req[0], out);
    } else {
      LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
    }
  }
}

template<typename xpu, typename OP>
void BinaryBroadcastComputeDenseEx(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK_LE(inputs[1].shape().ndim(), 2)
    << "input dense matrix should have less than or equal to 2 dimensions";
  if (req[0] == kNullOp) return;
  const NDArray& lhs = inputs[0];
  const NDArray& rhs = inputs[1];
  const NDArray& out = outputs[0];
  const auto lhs_stype = lhs.storage_type();
  const auto rhs_stype = rhs.storage_type();
  const auto out_stype = out.storage_type();
  bool reverse = (lhs_stype == kDefaultStorage);
  const NDArray& dns = (reverse) ? lhs : rhs;
  const NDArray& csr = (reverse) ? rhs : lhs;
  mxnet::TShape new_csrshape, new_dnsshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(csr.shape(), dns.shape(), out.shape(),
                                         &new_csrshape, &new_dnsshape, &new_oshape);

  if (((lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage) ||
       (lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage)) &&
      out_stype == kDefaultStorage) {
    // If the input is a matrix with the same shape, should be elemwise
    if (!ndim) {
      mshadow::Stream<xpu>* s = ctx.get_stream<xpu>();
      ElemwiseBinaryOp::DnsCsrDnsOp<OP>(s, attrs, ctx, dns, csr, req[0], outputs[0], !reverse);
    } else {
      // broadcast(CSR, Dense(1D)) = CSR
      BinaryBroadcastCsrDnsDnsImpl<xpu, OP>(ctx, csr, dns, req[0], out,
                                            new_csrshape, new_dnsshape, new_oshape,
                                            ndim, reverse);
    }
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu, typename LOP, typename ROP>
inline typename std::enable_if<std::is_same<xpu, cpu>::value, void>::type
BinaryBroadcastBackwardUseNone(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  using namespace broadcast;
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryOp::BackwardUseNone<cpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Stream<cpu> *s = ctx.get_stream<cpu>();
      const TBlob lhs = outputs[0].reshape(new_lshape);
      const TBlob rhs = outputs[1].reshape(new_rshape);
      const TBlob out = inputs[0].reshape(new_oshape);
      BROADCAST_NDIM_SWITCH(ndim, NDim, {
        // Request temporary storage
        size_t workspace_size = new_oshape.Size();
        Tensor<cpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<cpu, 1, char>(
                Shape1(workspace_size * sizeof(index_t)), s);
        ReduceWithExtraMem<red::sum, NDim, DType, LOP>(s, lhs, req[0], workspace, out);
        ReduceWithExtraMem<red::sum, NDim, DType, ROP>(s, rhs, req[1], workspace, out);
      });
    });
  }
}

template<typename xpu, int ndim, typename DType, typename LOP, typename ROP>
void BinaryBroadcastBackwardUseInImplWithWorkspace(const OpContext& ctx,
                                                   const std::vector<TBlob>& inputs,
                                                   const std::vector<OpReqType>& req,
                                                   const std::vector<TBlob>& outputs,
                                                   const mshadow::Tensor<xpu, 1, char>& workspace,
                                                   const mxnet::TShape& new_lshape,
                                                   const mxnet::TShape& new_rshape,
                                                   const mxnet::TShape& new_oshape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob lgrad = outputs[0].reshape(new_lshape);
  const TBlob rgrad = outputs[1].reshape(new_rshape);
  const TBlob ograd = inputs[0].reshape(new_oshape);
  const TBlob lhs = inputs[1].reshape(new_lshape);
  const TBlob rhs = inputs[2].reshape(new_rshape);
  if (ograd.Size() != 0) {
    Reduce<red::sum, ndim, DType, op::mshadow_op::mul, LOP>(s, lgrad, req[0], workspace,
      ograd, lhs, rhs);
    Reduce<red::sum, ndim, DType, op::mshadow_op::mul, ROP>(s, rgrad, req[1], workspace,
      ograd, lhs, rhs);
  }
}

template<typename xpu, int ndim, typename DType, typename LOP, typename ROP>
inline void BinaryBroadcastBackwardUseInImpl(const OpContext& ctx,
                                             const std::vector<TBlob>& inputs,
                                             const std::vector<OpReqType>& req,
                                             const std::vector<TBlob>& outputs,
                                             const mxnet::TShape& new_lshape,
                                             const mxnet::TShape& new_rshape,
                                             const mxnet::TShape& new_oshape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob lgrad = outputs[0].reshape(new_lshape);
  const TBlob rgrad = outputs[1].reshape(new_rshape);
  const TBlob ograd = inputs[0].reshape(new_oshape);
  const TBlob lhs = inputs[1].reshape(new_lshape);
  const TBlob rhs = inputs[2].reshape(new_rshape);
  size_t workspace_size_l = ReduceWorkspaceSize(
      s, lgrad.shape_, req[0], ograd.shape_, lhs.shape_, rhs.shape_);
  size_t workspace_size_r = ReduceWorkspaceSize(
      s, rgrad.shape_, req[1], ograd.shape_, lhs.shape_, rhs.shape_);
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
  // skip kernel launch for zero-size tensors
  if (inputs[0].shape_.Size() == 0U) {
    return;
  }
  mxnet::TShape new_lshape, new_rshape, new_oshape;
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
  .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)  \
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
