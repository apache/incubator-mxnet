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
 *  Copyright (c) 2020 by Contributors
 * \file np_cross-inl.h
 * \brief Function definition of cross product of two (arrays of) vectors
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_CROSS_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_CROSS_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include <string>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/broadcast_reduce_op.h"
#include "../tensor/elemwise_binary_broadcast_op.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct NumpyCrossParam : public dmlc::Parameter<NumpyCrossParam> {
  int axisa, axisb, axisc;
  DMLC_DECLARE_PARAMETER(NumpyCrossParam) {
    DMLC_DECLARE_FIELD(axisa)
    .set_default(-1)
    .describe("Axis of `a` that defines the vector(s).  By default, the last axis.");
    DMLC_DECLARE_FIELD(axisb)
    .set_default(-1)
    .describe("Axis of `b` that defines the vector(s).  By default, the last axis.");
    DMLC_DECLARE_FIELD(axisc)
    .set_default(-1)
    .describe("Axis of `c` containing the cross product vector(s)."
              "Ignored if both input vectors have dimension 2, as the return is scalar."
              "By default, the last axis.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axisa_s, axisb_s, axisc_s;
    axisa_s << axisa;
    axisb_s << axisb;
    axisc_s << axisc;
    (*dict)["axisa"] = axisa_s.str();
    (*dict)["axisb"] = axisb_s.str();
    (*dict)["axisc"] = axisc_s.str();
  }
};

#define SUM_NDIM_SWITCH(ndim, NDim, ...)  \
  if (ndim == 1) {                    \
    const int NDim = 1;               \
    {__VA_ARGS__}                     \
  } else if (ndim == 2) {             \
    const int NDim = 2;               \
    {__VA_ARGS__}                     \
  } else if (ndim == 3) {             \
    const int NDim = 3;               \
    {__VA_ARGS__}                     \
  } else if (ndim == 4) {             \
    const int NDim = 4;               \
    {__VA_ARGS__}                     \
  } else if (ndim <= broadcast::MAX_DIM) {  \
    const int NDim = broadcast::MAX_DIM;    \
    {__VA_ARGS__}                     \
  } else {                            \
    LOG(FATAL) << "NDim too large ";  \
  }

struct CrossInAssign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, const DType *in_ptr, DType *out_ptr,
                                  const index_t stride, const index_t index, const size_t msize) {
    if (index < stride && i * stride + index < msize) {
      out_ptr[i] = in_ptr[i * stride + index];
    }
  }
};

template<int req>
struct CrossOutAssign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, const DType *in_ptr, DType *out_ptr,
                                  const int positive, const index_t stride,
                                  const index_t index, const size_t msize) {
    if (index < stride && i * stride + index < msize) {
      KERNEL_ASSIGN(out_ptr[i * stride + index], req, positive == 1 ? in_ptr[i] : -in_ptr[i]);
    }
  }
};

template<int req>
struct ResAssign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, const DType *in_data, DType *out_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

struct DeleteAssign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const DType *in_data, DType *out_data,
                                  const int in_stride, const int out_stride) {
    const DType *in_ptr = in_data + i * in_stride;
    DType *out_ptr = out_data + i * out_stride;
    if (in_stride == out_stride + 1) {
      for (int idx = 0; idx < out_stride; ++idx) {
        out_ptr[idx] = in_ptr[idx];
      }
    }
  }
};

// Get moveaxis index.
inline mxnet::Tuple<int> GetMoveaxisIndex(const int& source,
                                          const int& destination,
                                          const mxnet::TShape& shape) {
  const int ndim = shape.ndim();
  const int src_axis = CheckAxis(source, ndim);
  const int dest_axis = CheckAxis(destination, ndim);
  std::vector<int> moveaxis_index_vec;
  for (int i = 0; i < ndim; ++i) {
    if (i != src_axis) { moveaxis_index_vec.push_back(i); }
  }
  moveaxis_index_vec.insert(moveaxis_index_vec.begin() + dest_axis, src_axis);
  return mxnet::Tuple<int>(moveaxis_index_vec);
}

// Get moveaxis shape.
inline mxnet::TShape GetMoveaxisShape(const Tuple<int>& moveaxis_index,
                                      const mxnet::TShape& org_shape) {
  const int ndim = org_shape.ndim();
  if (ndim == 0) { return mxnet::TShape(0, 0); }
  CHECK_EQ(moveaxis_index.ndim(), org_shape.ndim()) << "moveaxis index dismatch original shape.";
  std::vector<size_t> moveaxis_shape_vec(ndim, -1);
  for (int i = 0; i < ndim; ++i) {
    moveaxis_shape_vec[i] = org_shape[moveaxis_index[i]];
  }
  return mxnet::TShape(moveaxis_shape_vec.begin(), moveaxis_shape_vec.end());
}

// Get or check broadcast shape for cross product.
inline void GetOrCheckLRShape(const nnvm::NodeAttrs& attrs,
                              const mxnet::TShape& a_moveaxis_shape,
                              const mxnet::TShape& b_moveaxis_shape,
                              mxnet::TShape *c_shape_ptr = nullptr) {
  const int a_ndim = a_moveaxis_shape.ndim();
  const int b_ndim = b_moveaxis_shape.ndim();
  mxnet::TShape a_cutoff_shape(a_ndim - 1, -1);
  mxnet::TShape b_cutoff_shape(b_ndim - 1, -1);
  for (int i = 0; i < a_ndim - 1; ++i) {
    a_cutoff_shape[i] = a_moveaxis_shape[i];
  }
  for (int i = 0; i < b_ndim - 1; ++i) {
    b_cutoff_shape[i] = b_moveaxis_shape[i];
  }
  mxnet::ShapeVector in_shape_vec({ a_cutoff_shape, b_cutoff_shape});
  mxnet::ShapeVector out_shape_vec({ mxnet::TShape() });
  mxnet::op::BinaryBroadcastShape(attrs, &in_shape_vec, &out_shape_vec);
  if (c_shape_ptr && (a_moveaxis_shape[a_ndim - 1] == 3 || b_moveaxis_shape[b_ndim - 1] == 3)) {
    mxnet::TShape c_shape(out_shape_vec[0].ndim() + 1, -1);
    for (int i = 0; i < c_shape.ndim() - 1; ++i) {
      c_shape[i] = out_shape_vec[0][i];
    }
    c_shape[c_shape.ndim() - 1] = 3;
    *c_shape_ptr = c_shape;
  } else {
    *c_shape_ptr = out_shape_vec[0];
  }
}

// Get data[..., 0] shape.
inline mxnet::TShape GetCutoffShape(const mxnet::TShape& shape) {
  if (shape.ndim() == 0 || !ndim_is_known(shape)) { return mxnet::TShape(0, 0); }
  mxnet::TShape cutoff_shape(shape.ndim() - 1, -1);
  for (int i = 0; i < shape.ndim() - 1; ++i) { cutoff_shape[i] = shape[i]; }
  return cutoff_shape;
}

template<typename DType>
inline size_t AligndWorkspaceSize(const size_t& offset,
                                  const size_t& add_size) {
  size_t m32 = 32;
  size_t wks = offset + add_size;
  return ((wks * sizeof(DType) + m32 - 1) / m32 * m32 + sizeof(DType) - 1) / sizeof(DType);
}

// Calculate workspace size for numpy cross forward op.
template<typename xpu, typename DType>
inline size_t NumpyCrossWorkspaceSize(const mxnet::TShape& a_moveaxis_shape,
                                      const mxnet::TShape& b_moveaxis_shape,
                                      const mxnet::TShape& c_moveaxis_shape,
                                      const int& a_axis,
                                      const int& b_axis,
                                      const OpContext& ctx,
                                      const std::vector<OpReqType>& req) {
  if (kNullOp == req[0]) { return 0U; }
  // Zero-size input, no need to launch kernel
  if (0U == a_moveaxis_shape.Size() || 0U == b_moveaxis_shape.Size()) { return 0U; }

  size_t workspace_size = 0;
  const int a_ndim = a_moveaxis_shape.ndim();
  const int b_ndim = b_moveaxis_shape.ndim();
  const int c_ndim = c_moveaxis_shape.ndim();

  if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
    if (a_moveaxis_shape[a_ndim - 1] == 2 && b_moveaxis_shape[b_ndim - 1] == 2) {
      // Case 1: a.shape[-1] == 2 and b.shape[-1] == 2, param.axisc is ignored.
      workspace_size += a_moveaxis_shape.ProdShape(0, a_ndim - 1);
      workspace_size += b_moveaxis_shape.ProdShape(0, b_ndim - 1);
      workspace_size += c_moveaxis_shape.Size();
    } else {
      // Case 2, 3, 4: a.shape[-1] == 3 or b.shape[-1] == 3, param.axisc is not ignored.
      workspace_size += a_moveaxis_shape.ProdShape(0, a_ndim - 1);
      workspace_size += b_moveaxis_shape.ProdShape(0, b_ndim - 1);
      workspace_size += c_moveaxis_shape.Size();
      workspace_size += 3 * c_moveaxis_shape.ProdShape(0, c_ndim - 1);
    }
    if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
      workspace_size += a_moveaxis_shape.Size();
      workspace_size += b_moveaxis_shape.Size();
    }
  } else {
    if (a_moveaxis_shape[a_ndim - 1] == 2 && b_moveaxis_shape[b_ndim - 1] == 2) {
      // Case 1: a.shape[-1] == 2 and b.shape[-1] == 2, param.axisc is ignored.
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  a_moveaxis_shape.ProdShape(0, a_ndim - 1));
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  b_moveaxis_shape.ProdShape(0, b_ndim - 1));
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  c_moveaxis_shape.Size());
    } else {
      // Case 2, 3, 4: a.shape[-1] == 3 or b.shape[-1] == 3, param.axisc is not ignored.
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  a_moveaxis_shape.ProdShape(0, a_ndim - 1));
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  b_moveaxis_shape.ProdShape(0, b_ndim - 1));
      for (int i = 0; i < 3; ++i) {
        workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                    c_moveaxis_shape.ProdShape(0, c_ndim - 1));
      }
      workspace_size = AligndWorkspaceSize<DType>(workspace_size,
                                                  c_moveaxis_shape.Size());
    }
    if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
      workspace_size = AligndWorkspaceSize<DType>(workspace_size, a_moveaxis_shape.Size());
      workspace_size = AligndWorkspaceSize<DType>(workspace_size, b_moveaxis_shape.Size());
    }
  }
  return workspace_size;
}

template<typename xpu, typename DType, int a_dim, int b_dim>
struct NumpyCrossForwardImpl {
  static void op(const TBlob& a, const TBlob& b, const TBlob& c,
                 const std::vector<Tuple<int> >& moveaxis_index_vec,
                 const std::vector<mxnet::TShape>& moveaxis_shape_vec,
                 const int a_axis, const int b_axis, const int c_axis,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<OpReqType>& req,
                 const Tensor<xpu, 1, DType>& workspace) {
    CHECK(a_dim == 3 || b_dim == 3)
      << "no specialized NumpyCrossOp defined for template parameters.";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const Tuple<int>& a_moveaxis_index = moveaxis_index_vec[0];
    const Tuple<int>& b_moveaxis_index = moveaxis_index_vec[1];
    const mxnet::TShape& a_moveaxis_shape = moveaxis_shape_vec[0];
    const mxnet::TShape& b_moveaxis_shape = moveaxis_shape_vec[1];
    const mxnet::TShape& c_moveaxis_shape = moveaxis_shape_vec[2];
    const int a_ndim = a_moveaxis_shape.ndim();
    const int b_ndim = b_moveaxis_shape.ndim();
    const int c_ndim = b_moveaxis_shape.ndim();
    CHECK_EQ(c_moveaxis_shape[c_ndim - 1], 3)
      << "no specialized NumpyCrossOp defined for template parameters.";

    TBlob aw_data, bw_data, c_data, cw_data, a_data, b_data;
    std::vector<TBlob> cw_data_vec;
    if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
      // Allocate workspace in cpu, no need to align address.
      DType *aw_ptr = workspace.dptr_;
      DType *bw_ptr = aw_ptr + a_moveaxis_shape.ProdShape(0, a_ndim - 1);
      DType *cw_ptr = bw_ptr + b_moveaxis_shape.ProdShape(0, b_ndim - 1);
      DType *c_ptr = cw_ptr + 3 * c_moveaxis_shape.ProdShape(0, c_ndim - 1);
      a_data = a;
      b_data = b;
      if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
        DType *a_ptr = c_ptr + c_moveaxis_shape.Size();
        DType *b_ptr = a_ptr + a_moveaxis_shape.Size();
        a_data = TBlob(a_ptr, a_moveaxis_shape, a.dev_mask(), a.dev_id());
        b_data = TBlob(b_ptr, b_moveaxis_shape, b.dev_mask(), b.dev_id());
        TransposeImpl<xpu>(ctx.run_ctx, a, a_data,
                           mxnet::TShape(a_moveaxis_index.begin(), a_moveaxis_index.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_data,
                           mxnet::TShape(b_moveaxis_index.begin(), b_moveaxis_index.end()));
      }
      aw_data = TBlob(aw_ptr, GetCutoffShape(a_moveaxis_shape), a.dev_mask(), a.dev_id());
      bw_data = TBlob(bw_ptr, GetCutoffShape(b_moveaxis_shape), b.dev_mask(), b.dev_id());
      cw_data = TBlob(cw_ptr, c_moveaxis_shape, c.dev_mask(), c.dev_id());
      c_data = TBlob(c_ptr, c_moveaxis_shape, c.dev_mask(), c.dev_id());
      for (int i = 0; i < 3; ++i) {
        cw_data_vec.push_back(TBlob(cw_ptr + i * c_moveaxis_shape.ProdShape(0, c_ndim - 1),
                                    GetCutoffShape(c_moveaxis_shape), c.dev_mask(), c.dev_id()));
      }
    } else {
      // Allocate workspace in gpu, need to align address.
      size_t offset = 0;
      aw_data = TBlob(workspace.dptr_ + offset, GetCutoffShape(a_moveaxis_shape),
                      a.dev_mask(), a.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, aw_data.shape_.Size());

      bw_data = TBlob(workspace.dptr_ + offset, GetCutoffShape(b_moveaxis_shape),
                      b.dev_mask(), b.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, bw_data.shape_.Size());

      cw_data = TBlob(workspace.dptr_ + offset, c_moveaxis_shape, c.dev_mask(), c.dev_id());
      for (int i = 0; i < 3; ++i) {
        cw_data_vec.push_back(TBlob(workspace.dptr_ + offset,
                                    GetCutoffShape(c_moveaxis_shape), c.dev_mask(), c.dev_id()));
        offset = AligndWorkspaceSize<DType>(offset, cw_data_vec[i].shape_.Size());
      }
      c_data = TBlob(workspace.dptr_ + offset, c_moveaxis_shape, c.dev_mask(), c.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, c_data.shape_.Size());

      a_data = a;
      b_data = b;
      if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
        a_data = TBlob(workspace.dptr_ + offset, a_moveaxis_shape,
                       a.dev_mask(), a.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, a_data.shape_.Size());

        b_data = TBlob(workspace.dptr_ + offset, b_moveaxis_shape,
                       b.dev_mask(), b.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, b_data.shape_.Size());

        TransposeImpl<xpu>(ctx.run_ctx, a, a_data,
                          mxnet::TShape(a_moveaxis_index.begin(), a_moveaxis_index.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_data,
                          mxnet::TShape(b_moveaxis_index.begin(), b_moveaxis_index.end()));
      }
    }
    std::vector<int> positive_vec;
    std::vector<int> a_index_vec, b_index_vec, c_index_vec;
    std::vector<OpReqType> req_vec;
    if (a_dim == 2 && b_dim == 3) {
      a_index_vec = {1, 0, 0, 1};
      b_index_vec = {2, 2, 1, 0};
      c_index_vec = {0, 1, 2, 2};
      positive_vec = {1, 0, 1, 0};
      req_vec = { kWriteTo, kWriteTo, kWriteTo, kAddTo };
    } else if (a_dim == 3 && b_dim == 2) {
      a_index_vec = {2, 2, 0, 1};
      b_index_vec = {1, 0, 1, 0};
      c_index_vec = {0, 1, 2, 2};
      positive_vec = {0, 1, 1, 0};
      req_vec = { kWriteTo, kWriteTo, kWriteTo, kAddTo };
    } else {
      a_index_vec = {1, 2, 2, 0, 0, 1};
      b_index_vec = {2, 1, 0, 2, 1, 0};
      c_index_vec = {0, 0, 1, 1, 2, 2};
      positive_vec = {1, 0, 1, 0, 1, 0};
      req_vec = { kWriteTo, kAddTo, kWriteTo, kAddTo, kWriteTo, kAddTo};
    }
    for (size_t i = 0; i < a_index_vec.size(); ++i) {
      int idx = c_index_vec[i];
      mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, aw_data.Size(), a_data.dptr<DType>(),
                                                   aw_data.dptr<DType>(), a_data.size(a_ndim - 1),
                                                   a_index_vec[i], a_data.Size());
      mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, bw_data.Size(), b_data.dptr<DType>(),
                                                   bw_data.dptr<DType>(), b_data.size(b_ndim - 1),
                                                   b_index_vec[i], b_data.Size());
#if !defined(__CUDACC__)
      BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { aw_data, bw_data },
                                                       { kWriteTo }, { cw_data_vec[idx] });
#else
      BinaryBroadcastRTCCompute {"mul"}(attrs, ctx, { aw_data, bw_data },
                                        { kWriteTo }, { cw_data_vec[idx] });
#endif  // !defined(__CUDACC__)
      MXNET_ASSIGN_REQ_SWITCH(req_vec[i], req_type, {
        mxnet_op::Kernel<CrossOutAssign<req_type>, xpu>::Launch(s, cw_data_vec[idx].Size(),
                                                                cw_data_vec[idx].dptr<DType>(),
                                                                c_data.dptr<DType>(),
                                                                positive_vec[i],
                                                                c_data.size(c_ndim - 1),
                                                                idx, c_data.Size());
      });
    }
    cw_data = cw_data.reshape(c.shape_);
    const DType *res_ptr = c_data.dptr<DType>();
    if (c_axis != c_ndim -1) {
      const Tuple<int> c_axis_index = GetMoveaxisIndex(-1, c_axis, c_moveaxis_shape);
      TransposeImpl<xpu>(ctx.run_ctx, c_data, cw_data,
                         mxnet::TShape(c_axis_index.begin(), c_axis_index.end()));
      res_ptr = cw_data.dptr<DType>();
    }
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
        s, c.Size(), res_ptr, c.dptr<DType>());
    });
  }
};

template<typename xpu, typename DType>
struct NumpyCrossForwardImpl<xpu, DType, 2, 2> {
  static void op(const TBlob& a, const TBlob& b, const TBlob& c,
                 const std::vector<Tuple<int> >& moveaxis_index_vec,
                 const std::vector<mxnet::TShape>& moveaxis_shape_vec,
                 const int a_axis, const int b_axis, const int c_axis,
                 const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<OpReqType>& req,
                 const Tensor<xpu, 1, DType>& workspace) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const Tuple<int>& a_moveaxis_index = moveaxis_index_vec[0];
    const Tuple<int>& b_moveaxis_index = moveaxis_index_vec[1];
    const mxnet::TShape& a_moveaxis_shape = moveaxis_shape_vec[0];
    const mxnet::TShape& b_moveaxis_shape = moveaxis_shape_vec[1];
    const mxnet::TShape& c_shape = c.shape_;
    const int a_ndim = a_moveaxis_shape.ndim();
    const int b_ndim = b_moveaxis_shape.ndim();

    TBlob aw_data, bw_data, cw_data, a_data, b_data;
    if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
      // Allocate workspace in cpu, no need to align address.
      DType *aw_ptr = workspace.dptr_;
      DType *bw_ptr = aw_ptr + a_moveaxis_shape.ProdShape(0, a_ndim - 1);
      DType *cw_ptr = bw_ptr + b_moveaxis_shape.ProdShape(0, b_ndim - 1);
      aw_data = TBlob(aw_ptr, GetCutoffShape(a_moveaxis_shape), a.dev_mask(), a.dev_id());
      bw_data = TBlob(bw_ptr, GetCutoffShape(b_moveaxis_shape), b.dev_mask(), b.dev_id());
      cw_data = TBlob(cw_ptr, c_shape, c.dev_mask(), c.dev_id());
      a_data = a;
      b_data = b;
      if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
        DType *a_ptr = cw_ptr + c_shape.Size();
        DType *b_ptr = a_ptr + a_moveaxis_shape.Size();
        a_data = TBlob(a_ptr, a_moveaxis_shape, a.dev_mask(), a.dev_id());
        b_data = TBlob(b_ptr, b_moveaxis_shape, b.dev_mask(), b.dev_id());
        TransposeImpl<xpu>(ctx.run_ctx, a, a_data,
                           mxnet::TShape(a_moveaxis_index.begin(), a_moveaxis_index.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_data,
                           mxnet::TShape(b_moveaxis_index.begin(), b_moveaxis_index.end()));
      }
    } else {
      // Allocate workspace in cpu, need to align address.
      size_t offset = 0;
      aw_data = TBlob(workspace.dptr_ + offset, GetCutoffShape(a_moveaxis_shape),
                      a.dev_mask(), a.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, aw_data.shape_.Size());

      bw_data = TBlob(workspace.dptr_ + offset, GetCutoffShape(b_moveaxis_shape),
                      b.dev_mask(), b.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, bw_data.shape_.Size());

      cw_data = TBlob(workspace.dptr_ + offset, c_shape,
                      c.dev_mask(), c.dev_id());
      offset = AligndWorkspaceSize<DType>(offset, cw_data.shape_.Size());
      a_data = a;
      b_data = b;
      if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
        a_data = TBlob(workspace.dptr_ + offset, a_moveaxis_shape,
                       a.dev_mask(), a.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, a_data.shape_.Size());

        b_data = TBlob(workspace.dptr_ + offset, b_moveaxis_shape,
                       b.dev_mask(), b.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, b_data.shape_.Size());

        TransposeImpl<xpu>(ctx.run_ctx, a, a_data,
                           mxnet::TShape(a_moveaxis_index.begin(), a_moveaxis_index.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_data,
                           mxnet::TShape(b_moveaxis_index.begin(), b_moveaxis_index.end()));
      }
    }
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, aw_data.Size(), a_data.dptr<DType>(),
                                                 aw_data.dptr<DType>(), a_data.size(a_ndim - 1),
                                                 0, a_data.Size());
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, bw_data.Size(), b_data.dptr<DType>(),
                                                 bw_data.dptr<DType>(), b_data.size(b_ndim - 1),
                                                 1, b_data.Size());
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { aw_data, bw_data },
                                                     { req[0] }, { c });
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx, { aw_data, bw_data },
                                      { req[0] }, { c });
#endif  // !defined(__CUDACC__)
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, aw_data.Size(), a_data.dptr<DType>(),
                                                 aw_data.dptr<DType>(), a_data.size(a_ndim - 1),
                                                 1, a_data.Size());
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, bw_data.Size(), b_data.dptr<DType>(),
                                                 bw_data.dptr<DType>(), b_data.size(b_ndim - 1),
                                                 0, b_data.Size());
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { aw_data, bw_data },
                                                     { kWriteTo }, { cw_data });
    BinaryBroadcastCompute<xpu, op::mshadow_op::minus>(attrs, ctx, { c, cw_data },
                                                       { kWriteTo }, { c });
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx, { aw_data, bw_data },
                                      { kWriteTo }, { cw_data });
    BinaryBroadcastRTCCompute {"sub"}(attrs, ctx, { c, cw_data },
                                      { kWriteTo }, { c });
#endif  // !defined(__CUDACC__)
  }
};

template<typename xpu>
void NumpyCrossForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& c = outputs[0];

  if (kNullOp == req[0]) { return; }
  // Zero-size output, no need to launch kernel
  if (0U == a.Size() || 0U == b.Size()) { return; }

  const NumpyCrossParam& param = nnvm::get<NumpyCrossParam>(attrs.parsed);
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;
  const mxnet::TShape& c_shape = c.shape_;
  const int a_ndim = a_shape.ndim();
  const int b_ndim = b_shape.ndim();
  const int c_ndim = c_shape.ndim();
  Tuple<int> a_moveaxis_index = GetMoveaxisIndex(param.axisa, -1, a_shape);
  Tuple<int> b_moveaxis_index = GetMoveaxisIndex(param.axisb, -1, b_shape);
  Tuple<int> c_moveaxis_index = GetMoveaxisIndex(param.axisc, -1, c_shape);
  mxnet::TShape a_moveaxis_shape = GetMoveaxisShape(a_moveaxis_index, a_shape);
  mxnet::TShape b_moveaxis_shape = GetMoveaxisShape(b_moveaxis_index, b_shape);
  mxnet::TShape c_moveaxis_shape = GetMoveaxisShape(c_moveaxis_index, c_shape);
  const int a_axis = CheckAxis(param.axisa, a_ndim);
  const int b_axis = CheckAxis(param.axisb, b_ndim);
  const int c_axis = CheckAxis(param.axisc, c_ndim);
  const std::vector<mxnet::TShape > shape_vec({ a_moveaxis_shape, b_moveaxis_shape,
                                                c_moveaxis_shape });
  const std::vector<Tuple<int> > index_vec({ a_moveaxis_index, b_moveaxis_index,
                                             c_moveaxis_index });

  MSHADOW_SGL_DBL_TYPE_SWITCH(c.type_flag_, DType, {
    // Calculate workspace.
    size_t workspace_size = NumpyCrossWorkspaceSize<xpu, DType>(a_moveaxis_shape,
                                                                b_moveaxis_shape,
                                                                c_moveaxis_shape,
                                                                a_axis, b_axis, ctx, req);
    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>(
      Shape1(workspace_size), s);

    if (a_moveaxis_shape[a_ndim - 1] == 2) {
      if (b_moveaxis_shape[b_ndim - 1] == 2) {
        // Case 1: a.shape[-1] == 2 and b.shape[-1] == 2, param.axisc is ignored.
        NumpyCrossForwardImpl<xpu, DType, 2, 2>::op(a, b, c, index_vec, shape_vec,
                                                    a_axis, b_axis, c_axis,
                                                    attrs, ctx, req, workspace);
      } else {
        // Case 2: a.shape[-1] == 2 and b.shape[-1] == 3, param.axisc is not ignored.
        NumpyCrossForwardImpl<xpu, DType, 2, 3>::op(a, b, c, index_vec, shape_vec,
                                                    a_axis, b_axis, c_axis,
                                                    attrs, ctx, req, workspace);
      }
    } else {
      if (b_moveaxis_shape[b_ndim - 1] == 2) {
        // Case 3: a.shape[-1] == 3 and b.shape[-1] == 2, param.axisc is not ignored.
        NumpyCrossForwardImpl<xpu, DType, 3, 2>::op(a, b, c, index_vec, shape_vec,
                                                    a_axis, b_axis, c_axis,
                                                    attrs, ctx, req, workspace);
      } else {
        // Case 4: a.shape[-1] == 3 and b.shape[-1] == 3, param.axisc is not ignored.
        NumpyCrossForwardImpl<xpu, DType, 3, 3>::op(a, b, c, index_vec, shape_vec,
                                                    a_axis, b_axis, c_axis,
                                                    attrs, ctx, req, workspace);
      }
    }
  });
}

inline bool CheckUseBroadcast(const mxnet::TShape& a_move_shape,
                              const mxnet::TShape& b_move_shape) {
  return !(GetCutoffShape(a_move_shape) == GetCutoffShape(b_move_shape));
}

inline mxnet::TShape GetOriShape(const mxnet::TShape& move_shape,
                                 const int axis) {
  Tuple<int> origin_index = GetMoveaxisIndex(-1, axis, move_shape);
  return GetMoveaxisShape(origin_index, move_shape);
}

inline std::vector<int> GetReduceAxis(const mxnet::TShape& move_shape,
                                      const mxnet::TShape& broad_move_shape) {
  std::vector<int> axis_idx;
  if (move_shape.ndim() == broad_move_shape.ndim() ||
      move_shape.ndim() == broad_move_shape.ndim() + 1) {
    for (int i = 0; i < move_shape.ndim() - 1; ++i) {
      if (move_shape[i] != broad_move_shape[i]) { axis_idx.push_back(i); }
    }
  }
  return axis_idx;
}

template<typename xpu, typename DType, int a_dim, int b_dim>
inline void CrossImplWrap(const std::vector<TBlob>& inputs,
                          const std::vector<TBlob>& outputs,
                          const std::vector<int>& axises,
                          const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const OpReqType& req,
                          const Tensor<xpu, 1, DType>& workspace) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(axises.size(), 3U);
  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& c = outputs[0];
  if (kNullOp == req) { return; }
  if (0U == a.Size() || 0U == b.Size()) { return; }

  const int a_axis = CheckAxis(axises[0], a.ndim());
  const int b_axis = CheckAxis(axises[1], b.ndim());
  const int c_axis = CheckAxis(axises[2], c.ndim());
  Tuple<int> a_move_index = GetMoveaxisIndex(a_axis, -1, a.shape_);
  Tuple<int> b_move_index = GetMoveaxisIndex(b_axis, -1, b.shape_);
  Tuple<int> c_move_index = GetMoveaxisIndex(c_axis, -1, c.shape_);
  mxnet::TShape a_move_shape = GetMoveaxisShape(a_move_index, a.shape_);
  mxnet::TShape b_move_shape = GetMoveaxisShape(b_move_index, b.shape_);
  mxnet::TShape c_move_shape = GetMoveaxisShape(c_move_index, c.shape_);
  // Check workspace size.
  size_t workspace_size = NumpyCrossWorkspaceSize<xpu, DType>(a_move_shape, b_move_shape,
                                                              c_move_shape, a_axis, b_axis,
                                                              ctx, { req });
  CHECK_GE(workspace.MSize(), workspace_size)
    << "Not enough working space size for cross product(should >= " << workspace_size << ")";
  NumpyCrossForwardImpl<xpu, DType, a_dim, b_dim>::op(a, b, c,
                                                      { a_move_index, b_move_index},
                                                      { a_move_shape, b_move_shape, c_move_shape},
                                                      a_axis, b_axis, c_axis,
                                                      attrs, ctx, { req }, workspace);
}

template<typename xpu, typename DType>
struct ReduceImplWrap {
  static size_t wks(const mxnet::TShape& out_shape,
                    const mxnet::TShape& out_move_shape,
                    const mxnet::TShape& in_shape,
                    const mxnet::TShape& in_move_shape,
                    const OpContext& ctx, const OpReqType& req) {
    size_t ws_reduce = 0U;
    std::vector<int> reduce_axis = GetReduceAxis(out_move_shape, in_move_shape);
    if (reduce_axis.empty() || req == kNullOp) { return 0U; }
    ws_reduce = broadcast::ReduceWorkspaceSize(ctx.get_stream<xpu>(),
                                               out_shape, req, in_shape);
    return ws_reduce;
  }

  static void op(const TBlob& work_in,
                 const TBlob& work_out,
                 const TBlob& out_data,
                 const OpContext& ctx,
                 const OpReqType& out_req,
                 const Tensor<xpu, 1, char> workspace_tensor) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // Reduce work_in to work_out.
#if !defined(__CUDACC__)
    SUM_NDIM_SWITCH(work_out.ndim(), NDim, {
      op::broadcast::Reduce<mshadow_op::sum, NDim, DType, op::mshadow_op::identity, false>(
        s, work_out, kWriteTo, workspace_tensor, work_in);
    });
#else
    SUM_NDIM_SWITCH(work_out.ndim(), NDim, {
      op::broadcast::RTCReduce(ctx, work_out, kWriteTo, workspace_tensor, work_in,
                               "red::sum{}", NDim, "identity");
    });
#endif
    // Copy work_out to out_data.
    MXNET_ASSIGN_REQ_SWITCH(out_req, req_type, {
      mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
        s, out_data.Size(), work_out.dptr<DType>(), out_data.dptr<DType>());
    });
  }
};

template<typename xpu, typename DType, int a_dim, int b_dim>
struct NumpyCrossBackwardImpl {
  static std::vector<size_t>
  CrossBwkWorkspaceSize(const bool use_broadcast,
                        const mxnet::TShape& a_shape,
                        const mxnet::TShape& b_shape,
                        const mxnet::TShape& c_shape,
                        const mxnet::TShape& a_move_shape,
                        const mxnet::TShape& b_move_shape,
                        const mxnet::TShape& c_move_shape,
                        const int& a_axis, const int& b_axis, const int& c_axis,
                        const OpContext& ctx,
                        const std::vector<OpReqType>& req) {
    CHECK((a_dim == 2 && b_dim == 3) || (a_dim == 3 && b_dim == 2))
      << "no specialized NumpyCrossOp defined for template parameters.";
    std::vector<size_t> workspace_size(3, 0U);
    if (use_broadcast) {
      size_t ws_a = 0U, ws_b = 0U, rws_a = 0U, rws_b = 0U;
      size_t ws1 = NumpyCrossWorkspaceSize<xpu, DType>(b_move_shape, c_move_shape, c_move_shape,
                                                       b_axis, c_axis, ctx, { kWriteTo });
      size_t ws2 = NumpyCrossWorkspaceSize<xpu, DType>(b_move_shape, c_move_shape, a_move_shape,
                                                       b_axis, c_axis, ctx, { kWriteTo });
      ws_a = std::max(ws1, ws2);
      size_t ws3 = NumpyCrossWorkspaceSize<xpu, DType>(c_move_shape, a_move_shape, c_move_shape,
                                                       c_axis, a_axis, ctx, { kWriteTo });
      size_t ws4 = NumpyCrossWorkspaceSize<xpu, DType>(c_move_shape, a_move_shape, b_move_shape,
                                                       c_axis, a_axis, ctx, { kWriteTo });
      ws_b = std::max(ws3, ws4);
      // Get delete result shape.
      mxnet::TShape c_move_dshape = c_move_shape;
      c_move_dshape[c_move_shape.ndim() - 1] = 2;
      if (a_dim == 2) {
        mxnet::TShape c_dshape = GetOriShape(c_move_dshape, a_axis);
        // Calculate workspace size used in sum(grad_a).
        size_t rws1 = ReduceImplWrap<xpu, DType>::wks(c_dshape, c_move_dshape,
                                                      c_shape, c_move_shape, ctx, req[0]);
        // Calculate workspace size used in sum(grad_b).
        size_t rws2 = ReduceImplWrap<xpu, DType>::wks(b_shape, b_move_shape,
                                                      c_shape, c_move_shape, ctx, req[1]);
        rws_a = std::max(rws1, rws2);
      }
      if (b_dim == 2) {
        mxnet::TShape c_dshape = GetOriShape(c_move_dshape, b_axis);
        // Calculate workspace size used in sum(grad_a).
        size_t rws1 = ReduceImplWrap<xpu, DType>::wks(a_shape, a_move_shape,
                                                      c_shape, c_move_shape, ctx, req[0]);
        // Calculate workspace size used in sum(grad_b).
        size_t rws2 = ReduceImplWrap<xpu, DType>::wks(c_dshape, c_move_dshape,
                                                      c_shape, c_move_shape, ctx, req[1]);
        rws_b = std::max(rws1, rws2);
      }
      size_t rws = (std::max(rws_a, rws_b) + sizeof(DType) - 1) / sizeof(DType);
      workspace_size[0] += std::max(ws_a, ws_b);  // For cross product workspace.
      workspace_size[1] += c_move_shape.Size();   // For cross product result.
      workspace_size[1] += c_move_shape.Size();   // For delete result shape.
      workspace_size[2] += rws;                   // For reduce workspace.
    } else {
      mxnet::TShape a_moveaxis_shape = (a_dim == 2 ? c_move_shape : a_move_shape);
      mxnet::TShape b_moveaxis_shape = (b_dim == 2 ? c_move_shape : b_move_shape);
      size_t ws1 = NumpyCrossWorkspaceSize<xpu, DType>(b_moveaxis_shape, c_move_shape,
                                                       a_moveaxis_shape, b_axis, c_axis, ctx,
                                                       { kWriteTo });
      size_t ws2 = NumpyCrossWorkspaceSize<xpu, DType>(c_move_shape, a_moveaxis_shape,
                                                       b_moveaxis_shape, c_axis, a_axis, ctx,
                                                       { kWriteTo });
      workspace_size[0] += std::max(ws1, ws2);  // For cross product workspace.
      if (a_dim == 2 && b_dim == 3) {
        workspace_size[1] += a_moveaxis_shape.Size();  // For cross product result.
        workspace_size[1] += a_move_shape.Size();      // For delete kernel result.
      }
      if (a_dim == 3 && b_dim == 2) {
        workspace_size[1] += b_moveaxis_shape.Size();  // For cross product result.
        workspace_size[1] += b_move_shape.Size();      // For delete kernel result.
      }
    }
    return workspace_size;
  }

  static void op(const bool use_broadcast,
                 const TBlob& grad_c, const TBlob& a, const TBlob& b,
                 const TBlob& grad_a, const TBlob& grad_b,
                 const std::vector<mxnet::TShape>& moveaxis_shape_vec,
                 const int a_axis, const int b_axis, const int c_axis,
                 const nnvm::NodeAttrs &attrs,
                 const OpContext& ctx,
                 const std::vector<OpReqType>& req) {
    CHECK((a_dim == 2 && b_dim == 3) || (a_dim == 3 && b_dim == 2))
      << "no specialized NumpyCrossOp defined for template parameters.";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const mxnet::TShape& a_move_shp = moveaxis_shape_vec[0];
    const mxnet::TShape& b_move_shp = moveaxis_shape_vec[1];
    const mxnet::TShape& c_move_shp = moveaxis_shape_vec[2];
    std::vector<int> a_reduce_axis = GetReduceAxis(a_move_shp, c_move_shp);
    std::vector<int> b_reduce_axis = GetReduceAxis(b_move_shp, c_move_shp);
    const int c_ndim = c_move_shp.ndim();
    // Get delete result shape.
    mxnet::TShape c_move_dshp = c_move_shp;
    c_move_dshp[c_move_shp.ndim() - 1] = 2;
    std::vector<size_t> wk_size = CrossBwkWorkspaceSize(use_broadcast,
                                                        a.shape_, b.shape_, grad_c.shape_,
                                                        a_move_shp, b_move_shp, c_move_shp,
                                                        a_axis, b_axis, c_axis, ctx, req);
    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>(
      Shape1(wk_size[0] + wk_size[1] + wk_size[2]), s);
    if (use_broadcast) {
      // Use broadcast in forward, need reduce in backward.
      DType *w0_ptr = workspace.dptr_;
      DType *w1_ptr = w0_ptr + wk_size[0];
      DType *w2_ptr = w1_ptr + c_move_shp.Size();
      char *w3_ptr = reinterpret_cast<char*>(w2_ptr + c_move_shp.Size());
      TBlob w0_data(w0_ptr, Shape1(wk_size[0]), grad_c.dev_mask(), grad_c.dev_id());
      Tensor<xpu, 1, char> w3_tensor(w3_ptr, Shape1(wk_size[2] * sizeof(DType)), s);
      if (a_dim == 2) {  // a_dim == 2, b_dim == 3
        TBlob w1_data(w1_ptr, c_move_shp, grad_c.dev_mask(), grad_c.dev_id());
        TBlob w2_data(w2_ptr, c_move_dshp, grad_c.dev_mask(), grad_c.dev_id());
        // Calculate grad_a = cross(b, grad_c).
        CrossImplWrap<xpu, DType, 3, 3>({ b, grad_c }, { w1_data }, { b_axis, c_axis, -1 },
                                        attrs, ctx, kWriteTo, w0_data.get<xpu, 1, DType>(s));
        // Copy w1_data to w2_data with delete.
        mxnet_op::Kernel<DeleteAssign, xpu>::Launch(s, c_move_dshp.ProdShape(0, c_ndim - 1),
                                                    w1_data.dptr<DType>(),
                                                    w2_data.dptr<DType>(), 3, 2);
        // Transpose w2_data to w1_data.
        if (a_axis != grad_a.ndim() - 1) {
          const Tuple<int> axis_idx = GetMoveaxisIndex(-1, a_axis, c_move_dshp);
          mxnet::TShape c_dshp = GetMoveaxisShape(axis_idx, c_move_dshp);
          w1_data = TBlob(w1_ptr, c_dshp, grad_c.dev_mask(), grad_c.dev_id());
          TransposeImpl<xpu>(ctx.run_ctx, w2_data, w1_data,
                             mxnet::TShape(axis_idx.begin(), axis_idx.end()));
          w2_data = TBlob(w2_ptr, grad_a.shape_, grad_c.dev_mask(), grad_c.dev_id());
        } else {
          // If no transpose, exchange the pointer.
          w1_data = TBlob(w2_ptr, c_move_dshp, grad_c.dev_mask(), grad_c.dev_id());
          w2_data = TBlob(w1_ptr, grad_a.shape_, grad_c.dev_mask(), grad_c.dev_id());
        }
        // Reduce w1_data to w2_data.
        if (a_reduce_axis.empty()) {
          // No need Reduce w1_data, Copy w1_data to grad_a.
          MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
            mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
              s, grad_a.Size(), w1_data.dptr<DType>(), grad_a.dptr<DType>());
          });
        } else {
          // Need Reduce w1_data to w2_data and Copy w2_data to grad_a.
          ReduceImplWrap<xpu, DType>::op(w1_data, w2_data, grad_a, ctx, req[0], w3_tensor);
        }
        // Calculate grad_b = cross(grad_c, a).
        if (b_reduce_axis.empty()) {
          CrossImplWrap<xpu, DType, 3, 2>({ grad_c, a }, { grad_b }, { c_axis, a_axis, b_axis },
                                          attrs, ctx, req[1], w0_data.get<xpu, 1, DType>(s));
        } else {
          mxnet::TShape c_shp = GetOriShape(c_move_shp, b_axis);
          w1_data = TBlob(w1_ptr, c_shp, grad_c.dev_mask(), grad_c.dev_id());
          w2_data = TBlob(w2_ptr, grad_b.shape_, grad_c.dev_mask(), grad_c.dev_id());
          CrossImplWrap<xpu, DType, 3, 2>({ grad_c, a }, { w1_data }, { c_axis, a_axis, b_axis },
                                          attrs, ctx, req[1], w0_data.get<xpu, 1, DType>(s));
          // Need Reduce w1_data to w2_data and Copy w2_data to grad_b.
          ReduceImplWrap<xpu, DType>::op(w1_data, w2_data, grad_b, ctx, req[1], w3_tensor);
        }
      }  // End of a_dim == 2
      if (b_dim == 2) {  // a_dim == 3, b_dim == 2
        TBlob w1_data(w1_ptr, c_move_shp, grad_c.dev_mask(), grad_c.dev_id());
        TBlob w2_data(w2_ptr, c_move_dshp, grad_c.dev_mask(), grad_c.dev_id());
        // Calculate grad_b = cross(grad_c, a).
        CrossImplWrap<xpu, DType, 3, 3>({ grad_c, a }, { w1_data }, { c_axis, a_axis, -1 },
                                        attrs, ctx, kWriteTo, w0_data.get<xpu, 1, DType>(s));
        // Copy w1_data to w2_data with delete.
        mxnet_op::Kernel<DeleteAssign, xpu>::Launch(s, c_move_dshp.ProdShape(0, c_ndim - 1),
                                                    w1_data.dptr<DType>(),
                                                    w2_data.dptr<DType>(), 3, 2);
        // Transpose w2_data to w1_data.
        if (b_axis != grad_b.ndim() - 1) {
          const Tuple<int> axis_idx = GetMoveaxisIndex(-1, b_axis, c_move_dshp);
          mxnet::TShape c_dshp = GetMoveaxisShape(axis_idx, c_move_dshp);
          w1_data = TBlob(w1_ptr, c_dshp, grad_c.dev_mask(), grad_c.dev_id());
          TransposeImpl<xpu>(ctx.run_ctx, w2_data, w1_data,
                             mxnet::TShape(axis_idx.begin(), axis_idx.end()));
          w2_data = TBlob(w2_ptr, grad_b.shape_, grad_c.dev_mask(), grad_c.dev_id());
        } else {
          // If no transpose, exchange the pointer.
          w1_data = TBlob(w2_ptr, c_move_dshp, grad_c.dev_mask(), grad_c.dev_id());
          w2_data = TBlob(w1_ptr, grad_b.shape_, grad_c.dev_mask(), grad_c.dev_id());
        }
        // Reduce w1_data to w2_data.
        if (b_reduce_axis.empty()) {
          // No need Reduce w1_data, Copy w1_data to grad_b.
          MXNET_ASSIGN_REQ_SWITCH(req[1], req_type, {
            mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
              s, grad_b.Size(), w1_data.dptr<DType>(), grad_b.dptr<DType>());
          });
        } else {
          // Need Reduce w1_data to w2_data and Copy w2_data to grad_a.
          ReduceImplWrap<xpu, DType>::op(w1_data, w2_data, grad_b, ctx, req[1], w3_tensor);
        }
        // Calculate grad_a = cross(b, grad_c).
        if (a_reduce_axis.empty()) {
          CrossImplWrap<xpu, DType, 2, 3>({ b, grad_c }, { grad_a }, { b_axis, c_axis, a_axis },
                                          attrs, ctx, req[0], workspace);
        } else {
          mxnet::TShape c_shp = GetOriShape(c_move_shp, a_axis);
          w1_data = TBlob(w1_ptr, c_shp, grad_c.dev_mask(), grad_c.dev_id());
          w2_data = TBlob(w2_ptr, grad_a.shape_, grad_c.dev_mask(), grad_c.dev_id());
          CrossImplWrap<xpu, DType, 2, 3>({ b, grad_c }, { w1_data }, { b_axis, c_axis, a_axis },
                                          attrs, ctx, req[0], w0_data.get<xpu, 1, DType>(s));
          // Need Reduce w1_data to w2_data and Copy w2_data to grad_b.
          ReduceImplWrap<xpu, DType>::op(w1_data, w2_data, grad_a, ctx, req[0], w3_tensor);
        }
      }  // End of b_dim == 3
    } else {
      // No use broadcast in forward, not need reduce in backward.
      DType *w0_ptr = workspace.dptr_;
      DType *w1_ptr = w0_ptr + wk_size[0];
      DType *w2_ptr = w1_ptr + c_move_shp.Size();
      TBlob w0_data(w0_ptr, Shape1(wk_size[0]), grad_c.dev_mask(), grad_c.dev_id());
      if (a_dim == 2) {  // a_dim == 2, b_dim == 3
        TBlob w1_data(w1_ptr, c_move_shp, grad_c.dev_mask(), grad_c.dev_id());
        TBlob w2_data(w2_ptr, a_move_shp, grad_c.dev_mask(), grad_c.dev_id());
        // Calculate w1_data = cross(b, grad_c).
        CrossImplWrap<xpu, DType, 3, 3>({ b, grad_c }, { w1_data }, { b_axis, c_axis, -1 },
                                        attrs, ctx, kWriteTo, w0_data.get<xpu, 1, DType>(s));
        // Calculate grad_b = cross(grad_c, a).
        CrossImplWrap<xpu, DType, 3, 2>({ grad_c, a }, { grad_b }, { c_axis, a_axis, b_axis },
                                        attrs, ctx, req[1], w0_data.get<xpu, 1, DType>(s));
        // Copy w1_data to w2_data with delete.
        mxnet_op::Kernel<DeleteAssign, xpu>::Launch(s, a_move_shp.ProdShape(0, a.ndim() - 1),
                                                    w1_data.dptr<DType>(),
                                                    w2_data.dptr<DType>(), 3, 2);
        DType *res_ptr = w2_data.dptr<DType>();
        if (a_axis != grad_a.ndim() - 1) {
          // Transpose w2_data to w1_data.
          const Tuple<int> grad_a_axis_idx = GetMoveaxisIndex(-1, a_axis, a_move_shp);
          w1_data = TBlob(w1_ptr, grad_a.shape_, grad_c.dev_mask(), grad_c.dev_id());
          TransposeImpl<xpu>(ctx.run_ctx, w2_data, w1_data,
                             mxnet::TShape(grad_a_axis_idx.begin(), grad_a_axis_idx.end()));
          res_ptr = w1_data.dptr<DType>();
        }
        // Copy w1_data to grad_a.
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
            s, grad_a.Size(), res_ptr, grad_a.dptr<DType>());
        });
      }  // End of a_dim == 2
      if (b_dim == 2) {  // a_dim == 3, b_dim == 2
        TBlob w1_data(w1_ptr, c_move_shp, grad_c.dev_mask(), grad_c.dev_id());
        TBlob w2_data(w2_ptr, b_move_shp, grad_c.dev_mask(), grad_c.dev_id());
        // Calculate grad_a = cross(b, grad_c).
        CrossImplWrap<xpu, DType, 2, 3>({ b, grad_c }, { grad_a }, { b_axis, c_axis, a_axis },
                                        attrs, ctx, req[0], w0_data.get<xpu, 1, DType>(s));
        // Calculate w1_data = cross(grad_c, a).
        CrossImplWrap<xpu, DType, 3, 3>({ grad_c, a }, { w1_data }, { c_axis, a_axis, -1 },
                                        attrs, ctx, kWriteTo, w0_data.get<xpu, 1, DType>(s));
        // Copy w1_data to w2_data with delete.
        mxnet_op::Kernel<DeleteAssign, xpu>::Launch(s, b_move_shp.ProdShape(0, b.ndim() - 1),
                                                    w1_data.dptr<DType>(),
                                                    w2_data.dptr<DType>(), 3, 2);
        DType *res_ptr = w2_data.dptr<DType>();
        if (b_axis != grad_b.ndim() - 1) {
          // Transpose w2_data to w1_data.
          const Tuple<int> grad_b_axis_idx = GetMoveaxisIndex(-1, b_axis, b_move_shp);
          w1_data = TBlob(w1_ptr, grad_b.shape_, grad_c.dev_mask(), grad_c.dev_id());
          TransposeImpl<xpu>(ctx.run_ctx, w2_data, w1_data,
                             mxnet::TShape(grad_b_axis_idx.begin(), grad_b_axis_idx.end()));
          res_ptr = w1_data.dptr<DType>();
        }
        // Copy w1_data to grad_b.
        MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
          mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
            s, grad_b.Size(), res_ptr, grad_b.dptr<DType>());
        });
      }  // End of b_dim == 2
    }  // End of use_broadcast
  }
};

template<typename xpu, typename DType>
struct NumpyCrossBackwardImpl<xpu, DType, 3, 3> {
  static std::vector<size_t>
  CrossBwkWorkspaceSize(const bool use_broadcast,
                        const mxnet::TShape& a_shape,
                        const mxnet::TShape& b_shape,
                        const mxnet::TShape& c_shape,
                        const mxnet::TShape& a_move_shape,
                        const mxnet::TShape& b_move_shape,
                        const mxnet::TShape& c_move_shape,
                        const int& a_axis, const int& b_axis, const int& c_axis,
                        const OpContext& ctx,
                        const std::vector<OpReqType>& req) {
    std::vector<size_t> workspace_size(3, 0U);
    if (use_broadcast) {
      // Calculate workspace size used in cross(b_move, grad_c_move).
      size_t ws1 = NumpyCrossWorkspaceSize<xpu, DType>(b_move_shape, c_move_shape, c_move_shape,
                                                       b_axis, c_axis, ctx, { kWriteTo });
      // Calculate workspace size used in cross(grad_c_move, a_move).
      size_t ws2 = NumpyCrossWorkspaceSize<xpu, DType>(c_move_shape, a_move_shape, c_move_shape,
                                                       c_axis, a_axis, ctx, { kWriteTo });
      // Calculate workspace size used in sum(grad_a).
      size_t rws1 = ReduceImplWrap<xpu, DType>::wks(a_shape, a_move_shape,
                                                    c_shape, c_move_shape, ctx, req[0]);
      // Calculate workspace size used in sum(grad_b).
      size_t rws2 = ReduceImplWrap<xpu, DType>::wks(b_shape, b_move_shape,
                                                    c_shape, c_move_shape, ctx, req[1]);
      // For cross product workspace.
      workspace_size[0] += std::max(ws1, ws2);
      // For reduce workspace.
      workspace_size[1] += (std::max(rws1, rws2) + sizeof(DType) - 1) / sizeof(DType);
      // For cross result and reduce result.
      workspace_size[2] += c_move_shape.Size();
      workspace_size[2] += std::max(a_move_shape.Size(), b_move_shape.Size());
    } else {
      size_t ws1 = NumpyCrossWorkspaceSize<xpu, DType>(b_move_shape, c_move_shape, a_move_shape,
                                                       b_axis, c_axis, ctx, { req[0] });
      size_t ws2 = NumpyCrossWorkspaceSize<xpu, DType>(c_move_shape, a_move_shape, b_move_shape,
                                                       c_axis, a_axis, ctx, { req[1] });
      workspace_size[0] += std::max(ws1, ws2);  // For cross product workspace.
    }
    return workspace_size;
  }

  static void op(const bool use_broadcast,
                 const TBlob& grad_c, const TBlob& a, const TBlob& b,
                 const TBlob& grad_a, const TBlob& grad_b,
                 const std::vector<mxnet::TShape>& moveaxis_shape_vec,
                 const int a_axis, const int b_axis, const int c_axis,
                 const nnvm::NodeAttrs &attrs,
                 const OpContext& ctx,
                 const std::vector<OpReqType>& req) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const mxnet::TShape& a_move_shp = moveaxis_shape_vec[0];
    const mxnet::TShape& b_move_shp = moveaxis_shape_vec[1];
    const mxnet::TShape& c_move_shp = moveaxis_shape_vec[2];
    std::vector<size_t> wk_size = CrossBwkWorkspaceSize(use_broadcast,
                                                        a.shape_, b.shape_, grad_c.shape_,
                                                        a_move_shp, b_move_shp, c_move_shp,
                                                        a_axis, b_axis, c_axis, ctx, req);
    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>(
      Shape1(wk_size[0] + wk_size[1] + wk_size[2]), s);
    if (use_broadcast) {
      // Use broadcast in forward, need reduce in backward.
      std::vector<int> a_reduce_axis = GetReduceAxis(a_move_shp, c_move_shp);
      std::vector<int> b_reduce_axis = GetReduceAxis(b_move_shp, c_move_shp);
      // Allocate workspace.
      DType *w0_ptr = workspace.dptr_;
      char *w1_ptr = reinterpret_cast<char*>(w0_ptr + wk_size[0]);
      TBlob w0_data(w0_ptr, Shape1(wk_size[0]), grad_c.dev_mask(), grad_c.dev_id());
      Tensor<xpu, 1, char> w1_tensor(w1_ptr, Shape1(wk_size[1] * sizeof(DType)), s);
      if (a_reduce_axis.empty()) {
        // Calculate grad_a = cross(b, grad_c).
        CrossImplWrap<xpu, DType, 3, 3>({ b, grad_c }, { grad_a }, { b_axis, c_axis, a_axis },
                                        attrs, ctx, req[0], w0_data.get<xpu, 1, DType>(s));
      } else {
        mxnet::TShape c_shp = GetOriShape(c_move_shp, a_axis);
        DType *w2_ptr = w0_ptr + wk_size[0] + wk_size[1];
        DType *w3_ptr = w2_ptr + c_move_shp.Size();
        TBlob w2_data(w2_ptr, c_shp, grad_c.dev_mask(), grad_c.dev_id());
        TBlob w3_data(w3_ptr, grad_a.shape_, grad_c.dev_mask(), grad_c.dev_id());
        // Calculate w2_data = cross(b, grad_c).
        CrossImplWrap<xpu, DType, 3, 3>({ b, grad_c }, { w2_data }, { b_axis, c_axis, a_axis },
                                        attrs, ctx, kWriteTo, w0_data.get<xpu, 1, DType>(s));
        // Reduce w2_data to w3_data and Copy w3_data to grad_a.
        ReduceImplWrap<xpu, DType>::op(w2_data, w3_data, grad_a, ctx, req[0], w1_tensor);
      }
      if (b_reduce_axis.empty()) {
        // Calculate grad_b = cross(grad_c, a).
        CrossImplWrap<xpu, DType, 3, 3>({ grad_c, a }, { grad_b }, { c_axis, a_axis, b_axis },
                                        attrs, ctx, req[1], w0_data.get<xpu, 1, DType>(s));
      } else {
        mxnet::TShape c_shp = GetOriShape(c_move_shp, b_axis);
        DType *w2_ptr = w0_ptr + wk_size[0] + wk_size[1];
        DType *w3_ptr = w2_ptr + c_move_shp.Size();
        TBlob w2_data(w2_ptr, c_shp, grad_c.dev_mask(), grad_c.dev_id());
        TBlob w3_data(w3_ptr, grad_b.shape_, grad_c.dev_mask(), grad_c.dev_id());
        // Calculate w2_data = cross(grad_c, a).
        CrossImplWrap<xpu, DType, 3, 3>({ grad_c, a }, { w2_data }, { c_axis, a_axis, b_axis },
                                        attrs, ctx, kWriteTo, w0_data.get<xpu, 1, DType>(s));
        // Reduce w2_data to w3_data and Copy w3_data to grad_b.
        ReduceImplWrap<xpu, DType>::op(w2_data, w3_data, grad_b, ctx, req[1], w1_tensor);
      }
    } else {
      CrossImplWrap<xpu, DType, 3, 3>({ b, grad_c }, { grad_a }, { b_axis, c_axis, a_axis },
                                      attrs, ctx, req[0], workspace);
      CrossImplWrap<xpu, DType, 3, 3>({ grad_c, a }, { grad_b }, { c_axis, a_axis, b_axis },
                                      attrs, ctx, req[1], workspace);
    }
  }
};

template<typename xpu, typename DType>
struct NumpyCrossBackwardImpl<xpu, DType, 2, 2> {
  static std::vector<size_t>
  CrossBwkWorkspaceSize(const bool use_broadcast,
                        const mxnet::TShape& a_shape,
                        const mxnet::TShape& b_shape,
                        const mxnet::TShape& c_shape,
                        const mxnet::TShape& a_move_shape,
                        const mxnet::TShape& b_move_shape,
                        const mxnet::TShape& c_move_shape,
                        const int& a_axis, const int& b_axis, const int& c_axis,
                        const OpContext& ctx,
                        const std::vector<OpReqType>& req) {
    std::vector<size_t> workspace_size(3, 0U);
    const int a_ndim = a_move_shape.ndim();
    const int b_ndim = b_move_shape.ndim();
    const int c_ndim = c_move_shape.ndim();
    mxnet::TShape grad_move_shape(c_ndim + 1, 2);
    for (int i = 0; i < c_ndim; ++i) { grad_move_shape[i] = c_move_shape[i]; }

    workspace_size[0] += grad_move_shape.Size();                 // For grad_a_move or grad_b_move.
    workspace_size[0] += a_move_shape.ProdShape(0, a_ndim - 1);  // For a_move work data.
    workspace_size[0] += b_move_shape.ProdShape(0, b_ndim - 1);  // For b_move work data.
    workspace_size[0] =                                          // For c_move work data.
      AligndWorkspaceSize<DType>(workspace_size[0], c_move_shape.Size());

    if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
      if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
        workspace_size[1] += a_move_shape.Size();     // For a_move size.
        workspace_size[1] += b_move_shape.Size();     // For b_move size.
        workspace_size[1] += grad_move_shape.Size();  // For grad_a_move or grad_b_move trans.
      } else {
        workspace_size[1] = AligndWorkspaceSize<DType>(workspace_size[1], a_move_shape.Size());
        workspace_size[1] = AligndWorkspaceSize<DType>(workspace_size[1], b_move_shape.Size());
        workspace_size[1] = AligndWorkspaceSize<DType>(workspace_size[1], grad_move_shape.Size());
      }
    }
    if (use_broadcast) {
      mxnet::TShape grad_a_dshape = GetOriShape(grad_move_shape, a_axis);
      mxnet::TShape grad_b_dshape = GetOriShape(grad_move_shape, b_axis);
      size_t rws1 = ReduceImplWrap<xpu, DType>::wks(a_shape, a_move_shape,
                                                    grad_a_dshape, grad_move_shape, ctx, req[0]);
      size_t rws2 = ReduceImplWrap<xpu, DType>::wks(b_shape, b_move_shape,
                                                    grad_b_dshape, grad_move_shape, ctx, req[1]);
      size_t rws = (std::max(rws1, rws2) + sizeof(DType) - 1) / sizeof(DType);
      workspace_size[2] += std::max(a_shape.Size(), b_shape.Size());  // For reduce result.
      workspace_size[2] += rws;                                       // For reduce workspace.
    }
    return workspace_size;
  }

  static void op(const bool use_broadcast,
                 const TBlob& grad_c, const TBlob& a, const TBlob& b,
                 const TBlob& grad_a, const TBlob& grad_b,
                 const std::vector<mxnet::TShape>& moveaxis_shape_vec,
                 const int a_axis, const int b_axis, const int c_axis,
                 const nnvm::NodeAttrs &attrs,
                 const OpContext& ctx,
                 const std::vector<OpReqType>& req) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tuple<int> a_move_idx = GetMoveaxisIndex(a_axis, -1, a.shape_);
    Tuple<int> b_move_idx = GetMoveaxisIndex(b_axis, -1, b.shape_);
    const mxnet::TShape& a_move_shp = moveaxis_shape_vec[0];
    const mxnet::TShape& b_move_shp = moveaxis_shape_vec[1];
    const mxnet::TShape& c_move_shp = grad_c.shape_;
    std::vector<int> a_reduce_axis = GetReduceAxis(a_move_shp, c_move_shp);
    std::vector<int> b_reduce_axis = GetReduceAxis(b_move_shp, c_move_shp);
    const int a_ndim = a_move_shp.ndim();
    const int b_ndim = b_move_shp.ndim();
    const int c_ndim = c_move_shp.ndim();
    mxnet::TShape grad_move_shp(c_ndim + 1, 2);
    for (int i = 0; i < c_ndim; ++i) { grad_move_shp[i] = c_move_shp[i]; }
    // Calculate workspace size.
    std::vector<size_t> wk_size = CrossBwkWorkspaceSize(use_broadcast,
                                                        a.shape_, b.shape_, grad_c.shape_,
                                                        a_move_shp, b_move_shp, c_move_shp,
                                                        a_axis, b_axis, c_axis, ctx, req);
    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>(
      Shape1(wk_size[0] + wk_size[1] + wk_size[2]), s);

    // Allocate workspace in cpu, no need to align address.
    DType *grad_ptr = workspace.dptr_;
    DType *aw_ptr = grad_ptr + grad_move_shp.Size();
    DType *bw_ptr = aw_ptr + a_move_shp.ProdShape(0, a_ndim - 1);
    DType *cw_ptr = bw_ptr + b_move_shp.ProdShape(0, b_ndim - 1);
    TBlob grad_move_data(grad_ptr, grad_move_shp, grad_c.dev_mask(), grad_c.dev_id());
    TBlob aw_data(aw_ptr, GetCutoffShape(a_move_shp), grad_c.dev_mask(), grad_c.dev_id());
    TBlob bw_data(bw_ptr, GetCutoffShape(b_move_shp), grad_c.dev_mask(), grad_c.dev_id());
    TBlob cw_data(cw_ptr, c_move_shp, grad_c.dev_mask(), grad_c.dev_id());
    TBlob a_move_data = a;
    TBlob b_move_data = b;
    TBlob grad_data = grad_move_data;
    size_t offset = 0;
    if (a_axis != a_ndim -1 || b_axis != b_ndim - 1) {
      if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
        DType *a_ptr = workspace.dptr_ + wk_size[0];
        DType *b_ptr = a_ptr + a_move_shp.Size();
        a_move_data = TBlob(a_ptr, a_move_shp, a.dev_mask(), a.dev_id());
        b_move_data = TBlob(b_ptr, b_move_shp, b.dev_mask(), b.dev_id());
        TransposeImpl<xpu>(ctx.run_ctx, a, a_move_data,
                           mxnet::TShape(a_move_idx.begin(), a_move_idx.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_move_data,
                           mxnet::TShape(b_move_idx.begin(), b_move_idx.end()));
      } else {
        DType *w1_ptr = workspace.dptr_ + wk_size[0];
        a_move_data = TBlob(w1_ptr + offset, a_move_shp, a.dev_mask(), a.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, a_move_shp.Size());

        b_move_data = TBlob(w1_ptr + offset, b_move_shp, a.dev_mask(), a.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, a_move_shp.Size());

        TransposeImpl<xpu>(ctx.run_ctx, a, a_move_data,
                           mxnet::TShape(a_move_idx.begin(), a_move_idx.end()));
        TransposeImpl<xpu>(ctx.run_ctx, b, b_move_data,
                           mxnet::TShape(b_move_idx.begin(), b_move_idx.end()));
      }
    }
    // Copy b_move_data[..., 1] to bw_data.
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, bw_data.Size(),
                                                 b_move_data.dptr<DType>(),
                                                 bw_data.dptr<DType>(),
                                                 b_move_data.size(b_ndim - 1),
                                                 1, b_move_data.Size());
    // cw_data = grad_c_move * b_move_data[..., 1].
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { grad_c, bw_data },
                                                     { kWriteTo }, { cw_data });
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx, { grad_c, bw_data },
                                      { kWriteTo }, { cw_data });
#endif  // !defined(__CUDACC__)
    // Copy cw_data to grad_move_data[..., 0].
    mxnet_op::Kernel<CrossOutAssign<kWriteTo>, xpu>::Launch(s, cw_data.Size(),
                                                            cw_data.dptr<DType>(),
                                                            grad_move_data.dptr<DType>(),
                                                            true, grad_move_data.size(c_ndim),
                                                            0, grad_move_data.Size());
    // Copy b_move_data[..., 0] to bw_data.
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, bw_data.Size(),
                                                 b_move_data.dptr<DType>(),
                                                 bw_data.dptr<DType>(),
                                                 b_move_data.size(b_ndim - 1),
                                                 0, b_move_data.Size());
    // cw_data = grad_c_move * b_move_data[..., 0].
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { grad_c, bw_data },
                                                     { kWriteTo }, { cw_data });
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx, { grad_c, bw_data },
                                      { kWriteTo }, { cw_data });
#endif  // !defined(__CUDACC__)
    // Copy -cw_data to grad_move_data[..., 1].
    mxnet_op::Kernel<CrossOutAssign<kWriteTo>, xpu>::Launch(s, cw_data.Size(),
                                                            cw_data.dptr<DType>(),
                                                            grad_move_data.dptr<DType>(),
                                                            false, grad_move_data.size(c_ndim),
                                                            1, grad_move_data.Size());
    // Transpose grad_move_data according to a_axis.
    grad_data = grad_move_data;
    if (a_axis != a_ndim - 1) {
      mxnet::TShape grad_shp = GetOriShape(grad_move_shp, a_axis);
      if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
        DType *grad_ptr = workspace.dptr_ + wk_size[0] + a_move_shp.Size() + b_move_shp.Size();
        grad_data = TBlob(grad_ptr, grad_shp, grad_c.dev_mask(), grad_c.dev_id());
      } else {
        DType *w1_ptr = workspace.dptr_ + wk_size[0];
        grad_data = TBlob(w1_ptr + offset, grad_shp, grad_c.dev_mask(), grad_c.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, grad_shp.Size());
      }
      const Tuple<int> axis_idx = GetMoveaxisIndex(-1, a_axis, grad_move_shp);
      TransposeImpl<xpu>(ctx.run_ctx, grad_move_data, grad_data,
                         mxnet::TShape(axis_idx.begin(), axis_idx.end()));
    }
    if (!a_reduce_axis.empty()) {
      size_t interval = std::max(grad_a.Size(), grad_b.Size());
      DType *grad_delete_ptr = workspace.dptr_ + wk_size[0] + wk_size[1];
      char *dw_ptr = reinterpret_cast<char*>(grad_delete_ptr + interval);
      TBlob grad_delete_data(grad_delete_ptr, grad_a.shape_, grad_c.dev_mask(), grad_c.dev_id());
      Tensor<xpu, 1, char> dw_tensor(dw_ptr, Shape1((wk_size[2] - interval) * sizeof(DType)), s);
      // Reduce grad_data to grad_delete_data and copy to grad_a.
      ReduceImplWrap<xpu, DType>::op(grad_data, grad_delete_data, grad_a, ctx, req[0], dw_tensor);
    } else {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
          s, grad_a.Size(), grad_data.dptr<DType>(), grad_a.dptr<DType>());
      });
    }

    // Copy a_move_data[..., 1] to aw_data.
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, aw_data.Size(),
                                                 a_move_data.dptr<DType>(),
                                                 aw_data.dptr<DType>(),
                                                 a_move_data.size(a_ndim - 1),
                                                 1, a_move_data.Size());
    // cw_data = grad_c_move * a_move_data[..., 1].
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { grad_c, aw_data },
                                                     { kWriteTo }, { cw_data });
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx, { grad_c, aw_data },
                                      { kWriteTo }, { cw_data });
#endif  // !defined(__CUDACC__)
    // Copy -cw_data to grad_move_data[..., 0].
    mxnet_op::Kernel<CrossOutAssign<kWriteTo>, xpu>::Launch(s, cw_data.Size(),
                                                            cw_data.dptr<DType>(),
                                                            grad_move_data.dptr<DType>(),
                                                            false, grad_move_data.size(c_ndim),
                                                            0, grad_move_data.Size());
    // Copy a_move_data[..., 0] to aw_data.
    mxnet_op::Kernel<CrossInAssign, xpu>::Launch(s, aw_data.Size(),
                                                 a_move_data.dptr<DType>(),
                                                 aw_data.dptr<DType>(),
                                                 a_move_data.size(a_ndim - 1),
                                                 0, a_move_data.Size());
    // cw_data = grad_c_move * a_move_data[..., 0].
#if !defined(__CUDACC__)
    BinaryBroadcastCompute<xpu, op::mshadow_op::mul>(attrs, ctx, { grad_c, aw_data },
                                                     { kWriteTo }, { cw_data });
#else
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx, { grad_c, aw_data },
                                      { kWriteTo }, { cw_data });
#endif  // !defined(__CUDACC__)
    // Copy cw_data to grad_move_data[..., 1].
    mxnet_op::Kernel<CrossOutAssign<kWriteTo>, xpu>::Launch(s, cw_data.Size(),
                                                            cw_data.dptr<DType>(),
                                                            grad_move_data.dptr<DType>(),
                                                            true, grad_move_data.size(c_ndim),
                                                            1, grad_move_data.Size());
    // Transpose grad_move_data according to b_axis.
    grad_data = grad_move_data;
    if (b_axis != b_ndim - 1) {
      mxnet::TShape grad_shp = GetOriShape(grad_move_shp, b_axis);
      if (ctx.run_ctx.get_ctx().dev_mask() == cpu::kDevMask) {
        DType *grad_ptr = workspace.dptr_ + wk_size[0] + a_move_shp.Size() + b_move_shp.Size();
        grad_data = TBlob(grad_ptr, grad_shp, grad_c.dev_mask(), grad_c.dev_id());
      } else {
        DType *w1_ptr = workspace.dptr_ + wk_size[0];
        grad_data = TBlob(w1_ptr + offset, grad_shp, grad_c.dev_mask(), grad_c.dev_id());
        offset = AligndWorkspaceSize<DType>(offset, grad_shp.Size());
      }
      const Tuple<int> axis_idx = GetMoveaxisIndex(-1, b_axis, grad_move_shp);
      TransposeImpl<xpu>(ctx.run_ctx, grad_move_data, grad_data,
                         mxnet::TShape(axis_idx.begin(), axis_idx.end()));
    }
    if (!b_reduce_axis.empty()) {
      size_t interval = std::max(grad_a.Size(), grad_b.Size());
      DType *grad_delete_ptr = workspace.dptr_ + wk_size[0] + wk_size[1];
      char *dw_ptr = reinterpret_cast<char*>(grad_delete_ptr + interval);
      TBlob grad_delete_data(grad_delete_ptr, grad_b.shape_, grad_c.dev_mask(), grad_c.dev_id());
      Tensor<xpu, 1, char> dw_tensor(dw_ptr, Shape1((wk_size[2] - interval) * sizeof(DType)), s);
      // Reduce grad_data to grad_delete_data and copy to grad_a.
      ReduceImplWrap<xpu, DType>::op(grad_data, grad_delete_data, grad_b, ctx, req[1], dw_tensor);
    } else {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        mxnet_op::Kernel<ResAssign<req_type>, xpu>::Launch(
          s, grad_b.Size(), grad_data.dptr<DType>(), grad_b.dptr<DType>());
      });
    }
  }
};

template <typename xpu>
void NumpyCrossBackward(const nnvm::NodeAttrs &attrs,
                        const OpContext &ctx,
                        const std::vector<TBlob> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  const TBlob& grad_c = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];

  if (kNullOp == req[0] && kNullOp == req[1]) { return; }
  // Zero-size output, no need to launch kernel
  if (0U == grad_c.Size()) { return; }

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;
  const mxnet::TShape& c_shape = grad_c.shape_;
  const int a_ndim = a_shape.ndim();
  const int b_ndim = b_shape.ndim();
  const int c_ndim = c_shape.ndim();
  const NumpyCrossParam& param = nnvm::get<NumpyCrossParam>(attrs.parsed);
  Tuple<int> a_moveaxis_index = GetMoveaxisIndex(param.axisa, -1, a_shape);
  Tuple<int> b_moveaxis_index = GetMoveaxisIndex(param.axisb, -1, b_shape);
  Tuple<int> c_moveaxis_index = GetMoveaxisIndex(param.axisc, -1, c_shape);
  mxnet::TShape a_moveaxis_shape = GetMoveaxisShape(a_moveaxis_index, a_shape);
  mxnet::TShape b_moveaxis_shape = GetMoveaxisShape(b_moveaxis_index, b_shape);
  mxnet::TShape c_moveaxis_shape = GetMoveaxisShape(c_moveaxis_index, c_shape);
  const int a_axis = CheckAxis(param.axisa, a_ndim);
  const int b_axis = CheckAxis(param.axisb, b_ndim);
  const int c_axis = CheckAxis(param.axisc, c_ndim);
  std::vector<mxnet::TShape> move_shp_vec({ a_moveaxis_shape, b_moveaxis_shape, c_moveaxis_shape });

  MSHADOW_SGL_DBL_TYPE_SWITCH(grad_c.type_flag_, DType, {
    bool use_broadcast = CheckUseBroadcast(a_moveaxis_shape, b_moveaxis_shape);
    if (a_moveaxis_shape[a_ndim - 1] == 2) {
      if (b_moveaxis_shape[b_ndim - 1] == 2) {
        // Case 1: a.shape[-1] == 2 and b.shape[-1] == 2, param.axisc is ignored.
        NumpyCrossBackwardImpl<xpu, DType, 2, 2>::op(use_broadcast, grad_c, a, b, grad_a, grad_b,
                                                     move_shp_vec, a_axis, b_axis, c_axis,
                                                     attrs, ctx, req);
      } else {
        // Case 2: a.shape[-1] == 2 and b.shape[-1] == 3, param.axisc is not ignored.
        NumpyCrossBackwardImpl<xpu, DType, 2, 3>::op(use_broadcast, grad_c, a, b, grad_a, grad_b,
                                                     move_shp_vec, a_axis, b_axis, c_axis,
                                                     attrs, ctx, req);
      }
    } else {
      if (b_moveaxis_shape[b_ndim - 1] == 2) {
        // Case 3: a.shape[-1] == 3 and b.shape[-1] == 2, param.axisc is not ignored.
        NumpyCrossBackwardImpl<xpu, DType, 3, 2>::op(use_broadcast, grad_c, a, b, grad_a, grad_b,
                                                     move_shp_vec, a_axis, b_axis, c_axis,
                                                     attrs, ctx, req);
      } else {
        // Case 4: a.shape[-1] == 3 and b.shape[-1] == 3, param.axisc is not ignored.
        NumpyCrossBackwardImpl<xpu, DType, 3, 3>::op(use_broadcast, grad_c, a, b, grad_a, grad_b,
                                                     move_shp_vec, a_axis, b_axis, c_axis,
                                                     attrs, ctx, req);
      }
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_CROSS_INL_H_
