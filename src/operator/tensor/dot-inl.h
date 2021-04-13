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
 * \file dot-inl.h
 * \brief Function definition of matrix dot operator
 */

#ifndef MXNET_OPERATOR_TENSOR_DOT_INL_H_
#define MXNET_OPERATOR_TENSOR_DOT_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <type_traits>

#include "./util/tensor_util-inl.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "./init_op.h"
#include "../mxnet_op.h"
#ifdef __CUDACC__
#include "./dot-inl.cuh"
#endif  // __CUDACC__

namespace mxnet {
namespace op {

struct DotParam : public dmlc::Parameter<DotParam> {
  bool transpose_a;
  bool transpose_b;
  dmlc::optional<int> forward_stype;
  DMLC_DECLARE_PARAMETER(DotParam) {
    DMLC_DECLARE_FIELD(transpose_a)
      .describe("If true then transpose the first input before dot.")
      .set_default(false);
    DMLC_DECLARE_FIELD(transpose_b)
      .describe("If true then transpose the second input before dot.")
      .set_default(false);
    DMLC_DECLARE_FIELD(forward_stype)
      .describe("The desired storage type of the forward output given by user, if the"
                "combination of input storage types and this hint does not match"
                "any implemented ones, the dot operator will perform fallback operation"
                "and still produce an output of the desired storage type.")
      .add_enum("default", kDefaultStorage)
      .add_enum("row_sparse", kRowSparseStorage)
      .add_enum("csr", kCSRStorage)
      .set_default(dmlc::optional<int>());
  }
  bool operator==(const DotParam& other) const {
    return this->transpose_a == other.transpose_a &&
           this->transpose_b == other.transpose_b &&
           this->forward_stype == other.forward_stype;
  }
};

template<typename xpu>
void DotForward_(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(outputs[0].type_flag_ == kFloat32 || outputs[0].type_flag_ == kFloat64 ||
      (outputs[0].type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
      << "dot only supports float32/float64 for CPU, and float16/float32/float64 for GPU";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    // VectorDot() with fp16 is not supported in mshadow. Dispatch to dot() instead.
    if (inputs[0].ndim() == 1 && inputs[1].ndim() == 1 && inputs[0].type_flag_ != kFloat16) {
      CHECK_NE(req[0], kAddTo) << "AddTo not yet supported";
      Tensor<xpu, 1, DType> out = outputs[0].get<xpu, 1, DType>(s);
      VectorDot(out,
                inputs[0].get<xpu, 1, DType>(s),
                inputs[1].get<xpu, 1, DType>(s));
    } else {
      index_t ma, na, mb, nb, m, n;
      if (param.transpose_a) {
        ma = inputs[0].size(0);
        na = inputs[0].Size()/ma;
        m = na;
      } else {
        na = inputs[0].size(inputs[0].ndim()-1);
        ma = inputs[0].Size()/na;
        m = ma;
      }
      if (param.transpose_b) {
        nb = inputs[1].size(inputs[1].ndim()-1);
        mb = inputs[1].Size()/nb;
        n = mb;
      } else {
        mb = inputs[1].size(0);
        nb = inputs[1].Size()/mb;
        n = nb;
      }
      Tensor<xpu, 2, DType> input0 =
      inputs[0].get_with_shape<xpu, 2, DType>(Shape2(ma, na), s);
      Tensor<xpu, 2, DType> input1 =
      inputs[1].get_with_shape<xpu, 2, DType>(Shape2(mb, nb), s);
      Tensor<xpu, 2, DType> out =
      outputs[0].get_with_shape<xpu, 2, DType>(Shape2(m, n), s);
      if (param.transpose_a && param.transpose_b) {
        ASSIGN_DISPATCH(out, req[0], dot(input0.T(), input1.T()));
      } else if (!param.transpose_a && param.transpose_b) {
        ASSIGN_DISPATCH(out, req[0], dot(input0, input1.T()));
      } else if (param.transpose_a && !param.transpose_b) {
        ASSIGN_DISPATCH(out, req[0], dot(input0.T(), input1));
      } else {
        ASSIGN_DISPATCH(out, req[0], dot(input0, input1));
      }
    }
  });
}

template<typename xpu>
void DotBackward_(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(req[0], kWriteInplace);
  CHECK_NE(req[1], kWriteInplace);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if (inputs[1].ndim() == 1 && inputs[2].ndim() == 1) {
      Tensor<xpu, 1, DType> mout_grad = inputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> mlhs_data = inputs[1].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> mrhs_data = inputs[2].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> mlhs_grad = outputs[0].get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> mrhs_grad = outputs[1].get<xpu, 1, DType>(s);
      ASSIGN_DISPATCH(mrhs_grad, req[1],
                      broadcast_scalar(mout_grad, mlhs_data.shape_) * mlhs_data);
      ASSIGN_DISPATCH(mlhs_grad, req[0],
                      broadcast_scalar(mout_grad, mlhs_data.shape_) * mrhs_data);
    } else {
      int ma, na, mb, nb, m, n;
      if (param.transpose_a) {
        ma = outputs[0].size(0);
        na = outputs[0].Size()/ma;
        m = na;
      } else {
        na = outputs[0].size(outputs[0].ndim()-1);
        ma = outputs[0].Size()/na;
        m = ma;
      }
      if (param.transpose_b) {
        nb = outputs[1].size(outputs[1].ndim()-1);
        mb = outputs[1].Size()/nb;
        n = mb;
      } else {
        mb = outputs[1].size(0);
        nb = outputs[1].Size()/mb;
        n = nb;
      }
      Tensor<xpu, 2, DType> mout_grad =
      inputs[0].get_with_shape<xpu, 2, DType>(Shape2(m, n), s);
      Tensor<xpu, 2, DType> mlhs_data =
      inputs[1].get_with_shape<xpu, 2, DType>(Shape2(ma, na), s);
      Tensor<xpu, 2, DType> mrhs_data =
      inputs[2].get_with_shape<xpu, 2, DType>(Shape2(mb, nb), s);
      Tensor<xpu, 2, DType> mlhs_grad =
      outputs[0].get_with_shape<xpu, 2, DType>(Shape2(ma, na), s);
      Tensor<xpu, 2, DType> mrhs_grad =
      outputs[1].get_with_shape<xpu, 2, DType>(Shape2(mb, nb), s);
      if (param.transpose_a && param.transpose_b) {
        // Gradient of z = dot(x.T, y.T)
        // dy = dot(x, dz).T = dot(dz.T, x.T)
        // dx = dot(dz, y).T = dot(y.T, dz.T)
        ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mout_grad.T(), mlhs_data.T()));
        ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mrhs_data.T(), mout_grad.T()));
      } else if (!param.transpose_a && param.transpose_b) {
        // Gradient of z = dot(x, y.T)
        // dy = dot(x.T, dz).T = dot(dz.T, x)
        // dx = dot(dz, y)
        ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mout_grad.T(), mlhs_data));
        ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mout_grad, mrhs_data));
      } else if (param.transpose_a && !param.transpose_b) {
        // Gradient of z = dot(x.T, y)
        // dy = dot(x, dz)
        // dx = dot(dz, y.T).T = dot(y, dz.T)
        ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mlhs_data, mout_grad));
        ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mrhs_data, mout_grad.T()));
      } else {
        // Gradient of z = dot(x, y)
        // dy = dot(x.T, dz)
        // dx = dot(dz, y.T)
        ASSIGN_DISPATCH(mrhs_grad, req[1], dot(mlhs_data.T(), mout_grad));
        ASSIGN_DISPATCH(mlhs_grad, req[0], dot(mout_grad, mrhs_data.T()));
      }
    }
  });
}

inline bool DotForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int>* in_attrs,
                                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  // csr has many zero columns, so the result of dot(csr.T, matrix) should be
  // rsp
  const auto& lhs_stype = in_attrs->at(0);
  const auto& rhs_stype = in_attrs->at(1);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  bool only_lhs_transpose = param.transpose_a && !param.transpose_b;
  bool rhs_rsp_or_dns =
      rhs_stype == kRowSparseStorage || rhs_stype == kDefaultStorage;
  bool hint_has_value = param.forward_stype.has_value();
  NDArrayStorageType target_stype = hint_has_value ?
                                    static_cast<NDArrayStorageType>(param.forward_stype.value()) :
                                    kUndefinedStorage;
  if (!dispatched && lhs_stype == kDefaultStorage &&
      rhs_stype == kDefaultStorage) {
    // dns, dns -> dns
    target_stype = hint_has_value ? target_stype : kDefaultStorage;
    if (target_stype == kDefaultStorage) {
      dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
#if MXNET_USE_MKLDNN == 1
                                       DispatchMode::kFComputeEx);
#else
                                       DispatchMode::kFCompute);
#endif
    }
  }
  if (!dispatched && lhs_stype == kCSRStorage && only_lhs_transpose && rhs_rsp_or_dns) {
    // csr.T, rsp/dns -> rsp
    target_stype = hint_has_value ? target_stype : kRowSparseStorage;
    if (target_stype == kRowSparseStorage) {
      dispatched = storage_type_assign(&out_stype, kRowSparseStorage,
                                       dispatch_mode, DispatchMode::kFComputeEx);
    // csr.T, rsp/dns -> dns
    } else if (target_stype == kDefaultStorage) {
      dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                       DispatchMode::kFComputeEx);
    }
  }
  if (!dispatched && lhs_stype == kCSRStorage && rhs_rsp_or_dns &&
      !param.transpose_a && !param.transpose_b) {
    // csr, rsp/dns -> dns
    target_stype = hint_has_value ? target_stype : kDefaultStorage;
    if (target_stype == kDefaultStorage) {
      dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                       DispatchMode::kFComputeEx);
    }
  }
  if (!dispatched && lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage &&
      !param.transpose_a) {
    target_stype = hint_has_value ? target_stype : kCSRStorage;
    if (dev_mask == mshadow::cpu::kDevMask) {
      // dns, csr -> csr on CPU
      if (target_stype == kCSRStorage && !param.transpose_b) {
        dispatched = storage_type_assign(&out_stype, kCSRStorage, dispatch_mode,
                                         DispatchMode::kFComputeEx);
      // dns, csr/csr.T -> dns on CPU
      } else if (target_stype == kDefaultStorage) {
        dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                         DispatchMode::kFComputeEx);
      }
    // dns, csr/csr.T -> dns on GPU
    } else if (dev_mask == mshadow::gpu::kDevMask) {
      if (target_stype == kDefaultStorage) {
        dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                         DispatchMode::kFComputeEx);
      }
    }
  }
  if (!dispatched) {
    target_stype = (target_stype == kUndefinedStorage)? kDefaultStorage : target_stype;
    dispatched = storage_type_assign(&out_stype, target_stype, dispatch_mode,
                                     DispatchMode::kFComputeFallback);
  }
  return dispatched;
}

inline bool DotBackwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                        const int dev_mask,
                                        DispatchMode* dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 2U);
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  const auto& ograd_stype = in_attrs->at(0);
  const auto& lhs_stype = in_attrs->at(1);
  const auto& rhs_stype = in_attrs->at(2);
  const bool no_transpose = !param.transpose_a && !param.transpose_b;
  auto& lhs_grad_stype = out_attrs->at(0);
  auto& rhs_grad_stype = out_attrs->at(1);
  bool dispatched = false;
  if (!dispatched && lhs_stype == kDefaultStorage && rhs_stype == kDefaultStorage &&
      ograd_stype == kDefaultStorage) {
    if (type_assign(&lhs_grad_stype, kDefaultStorage) &&
        type_assign(&rhs_grad_stype, kDefaultStorage)) {
      DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFCompute);
      dispatched = true;
    }
  }
  if (!dispatched && no_transpose && lhs_stype == kCSRStorage &&
      (ograd_stype == kRowSparseStorage || ograd_stype == kDefaultStorage)) {
    // backward: csr.T, rsp/dns -> rsp, dns.T, rsp/dns -> dns
    if (type_assign(&rhs_grad_stype, kRowSparseStorage) &&
        type_assign(&lhs_grad_stype, kDefaultStorage)) {
      DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
      dispatched = true;
    }
  }
  if (!dispatched && param.transpose_a && !param.transpose_b && lhs_stype == kCSRStorage &&
      (ograd_stype == kRowSparseStorage || ograd_stype == kDefaultStorage)) {
    // backward: csr, rsp/dns -> dns, dns, rsp/dns -> dns
    if (type_assign(&rhs_grad_stype, kDefaultStorage) &&
        type_assign(&lhs_grad_stype, kDefaultStorage)) {
      DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
      dispatched = true;
    }
  }
  if (!dispatched && !param.transpose_a &&
      lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage &&
      ograd_stype == kDefaultStorage) {
    if (type_assign(&lhs_grad_stype, kDefaultStorage) &&
        type_assign(&rhs_grad_stype, kDefaultStorage)) {
      DISPATCH_MODE_ASSIGN_CHECK(dispatch_mode, 0, DispatchMode::kFComputeEx);
      dispatched = true;
    }
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

/*!
 * \brief CPU Kernel of dot(csr, dns1) = dns2
 * Parallelization by row blocks
 */
struct DotCsrDnsDnsByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const nnvm::dim_t seg_len,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = std::min(seg_start + seg_len, num_rows);
    for (dim_t j = seg_start; j < seg_end; ++j) {
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_out = j * num_cols;
      for (IType k = indptr_l[j]; k < indptr_l[j+1]; ++k) {
        const DType val = data_l[k];
        const dim_t offset_r = col_idx_l[k] * num_cols;
        for (dim_t l = 0; l < num_cols; ++l) {
          out[offset_out+l] += data_r[offset_r+l] * val;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(csr.T(), dns1) = dns2
 * Parallelization by row blocks
 */
struct DotCsrTransDnsDnsByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const nnvm::dim_t seg_len,
                                  const nnvm::dim_t num_rows_l,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = (i + 1) * seg_len;
    for (dim_t j = 0; j < num_rows_l; ++j) {
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_r = j * num_cols;
      for (IType k = indptr_l[j]; k < indptr_l[j+1]; ++k) {
        const CType col_idx = col_idx_l[k];
        if (col_idx < seg_start || col_idx >= seg_end) continue;
        const dim_t offset_out = col_idx * num_cols;
        const DType val = data_l[k];
        for (dim_t l = 0; l < num_cols; ++l) {
          out[offset_out+l] += data_r[offset_r+l] * val;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(csr.T(), dns) = rsp
 * Parallelization by row blocks which evenly partition the non-zero rows.
 */
struct DotCsrTransDnsRspByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   */
  template<typename DType, typename IType, typename CType, typename RType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  nnvm::dim_t* row_flg_sum,
                                  RType* row_idx,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const nnvm::dim_t seg_len,
                                  const nnvm::dim_t num_rows_l,
                                  const nnvm::dim_t nnr,
                                  const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= nnr) return;
    const dim_t seg_end = (i + 1) * seg_len;
    const dim_t col_start = row_idx[seg_start];
    const dim_t col_end = seg_end >= nnr ? (row_idx[nnr-1] + 1) : row_idx[seg_end];
    for (dim_t j = 0; j < num_rows_l; ++j) {
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_r = j * num_cols;
      for (IType k = indptr_l[j]; k < indptr_l[j+1]; ++k) {
        const CType col_idx = col_idx_l[k];
        if (col_idx < col_start || col_idx >= col_end) continue;

        const nnvm::dim_t rsp_row = row_flg_sum[col_idx] - 1;
        const nnvm::dim_t offset_out = rsp_row * num_cols;
        const DType val = data_l[k];

        for (dim_t l = 0; l < num_cols; ++l) {
          out[offset_out+l] += data_r[offset_r+l] * val;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(csr, rsp) = dns
 * Parallelization by row blocks
 */
struct DotCsrRspDnsByRowBlocks {
  /*!
   * \brief
   * \param i         the i-th thread
   * \param nnr_r     storage_shape[0] of the rsp
   * \param num_rows  dns.shape[0]
   * \param num_cols  dns.shape[1]
   */
  template<typename DType, typename IType, typename CType, typename RType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const RType* row_idx_r,
                                  const nnvm::dim_t nnr_r,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols,
                                  const nnvm::dim_t seg_len) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = std::min(seg_start + seg_len, num_rows);
    for (dim_t j = seg_start; j < seg_end; ++j) {
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_out = j * num_cols;
      // Use binary search to find the lower_bound of val in row_idx array
      const RType* first = row_idx_r;
      const RType* last = row_idx_r + nnr_r;
      const CType val = col_idx_l[indptr_l[j]];
      const RType* it;
      int count = last - first, step;
      while (count > 0) {
        it = first;
        step = count / 2;
        it += step;
        if (*it < val) {
          first = ++it;
          count -= step + 1;
        } else {
          count = step;
        }
      }
      const RType* row_idx_ptr = first;
      // end of binary search
      if (row_idx_ptr == row_idx_r+nnr_r || *row_idx_ptr > col_idx_l[indptr_l[j+1]-1]) continue;
      for (IType k = indptr_l[j]; k < indptr_l[j+1] && row_idx_ptr != row_idx_r+nnr_r;) {
        if (col_idx_l[k] == *row_idx_ptr) {
          const dim_t offset_r = (row_idx_ptr - row_idx_r) * num_cols;
          for (dim_t l = 0; l < num_cols; ++l) {
            out[offset_out+l] += data_l[k] * data_r[offset_r+l];
          }
          ++k;
          ++row_idx_ptr;
        } else if (col_idx_l[k] < *row_idx_ptr) {
          ++k;
        } else {
          ++row_idx_ptr;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(csr.T(), rsp1) = rsp2, with row_idx marked for non-zero rows
 * Parallelization by row blocks
 */
struct DotCsrTransRspRspByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   * \param num_rows_l number of rows of lhs matrix
   * \param nnr_r number of non-zero rows of rhs matrix
   * \param num_rows number of rows of out matrix
   * \param num_cols number of cols of out matrix
   */
  template<typename DType, typename IType, typename CType, typename RType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  RType* row_idx_out,
                                  const DType* data_l,
                                  const IType* indptr_l,
                                  const CType* col_idx_l,
                                  const DType* data_r,
                                  const RType* row_idx_r,
                                  const nnvm::dim_t num_rows_l,
                                  const nnvm::dim_t nnr_r,
                                  const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols,
                                  const nnvm::dim_t seg_len) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows) return;
    const dim_t seg_end = (i + 1) * seg_len;
    for (dim_t rid = 0; rid < nnr_r; ++rid) {
      const RType j = row_idx_r[rid];
      if (indptr_l[j] == indptr_l[j+1]) continue;
      const dim_t offset_r = rid * num_cols;
      for (IType k = indptr_l[j]; k < indptr_l[j+1]; ++k) {
        const CType col_idx = col_idx_l[k];
        if (col_idx < seg_start || col_idx >= seg_end) continue;
        row_idx_out[col_idx] = 1;  // mark nonzero row as 1
        const dim_t offset_out = col_idx * num_cols;
        for (dim_t l = 0; l < num_cols; ++l) {
          out[offset_out+l] += data_r[offset_r+l] * data_l[k];
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of PopulateCsrForNNC
 * Parallelization by individual rows
 * Populates the indptr and indices array
 * based on number of non zero columns
 */
struct PopulateCsrForNNC {
  /*!
   * \brief
   * \param i the i-th thread
   * \param nnc_idx all non zero column indexes
   * \param indptr_out indptr array for output
   * \param col_idx_out column indices for output
   * \param nnc number of non zero columns in the output
   * \param num_rows_l number of rows in lhs
   */
  template <typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i, const CType* nnc_idx,
                                  IType* indptr_out, CType* col_idx_out,
                                  const nnvm::dim_t nnc,
                                  const nnvm::dim_t num_rows_l) {
    const CType start_idx = i * nnc;
    nnvm::dim_t cur = 0;
    indptr_out[i] = start_idx;
    if (static_cast<nnvm::dim_t>(i) == (num_rows_l - 1)) indptr_out[i + 1] = indptr_out[i] + nnc;
    for (IType idx = start_idx; idx < (start_idx + nnc); idx++) {
      col_idx_out[idx] = nnc_idx[cur++];
    }
  }
};

/*!
 * \brief CPU Impl of dot(dns, csr) = csr
 */
struct DotDnsCsrCsrByRowBlocks {
  /*!
   * \brief
   * \param i the i-th thread
   * \param num_rows_r number of rows in rhs
   * \param num_rows_l number of rows in lhs
   * \param num_cols number of columns in output
   * \param nnc number of non zero columns
   */

  template <typename DType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(
      int i, DType* out, const DType* data_l, const IType* indptr_r,
      const CType* col_idx_r, const DType* data_r, const nnvm::dim_t seg_len,
      const IType num_rows_r, const IType num_rows_l,
      const nnvm::dim_t num_cols, const nnvm::dim_t nnc,
      const CType* prefix_sum) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows_l) return;
    const dim_t seg_end = std::min(seg_start + seg_len, num_rows_l);

    for (dim_t j = seg_start; j < seg_end; j++) {
      for (dim_t k = 0; k < num_rows_r; k++) {
        const dim_t working_idx = j * num_rows_r + k;
        const DType val = data_l[working_idx];
        if (indptr_r[k] == indptr_r[k + 1]) continue;
        const dim_t row_start = j * nnc;
        for (dim_t cur = indptr_r[k]; cur < indptr_r[k + 1]; cur++) {
          dim_t cur_col_idx_r = col_idx_r[cur];
          const dim_t out_idx = row_start + prefix_sum[cur_col_idx_r] - 1;
          out[out_idx] += val * data_r[cur];
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(dns1, csr) = dns2
 * Parallelization by row blocks
 */
struct DotDnsCsrDnsByRowBlocks {
  /*!
   * \brief
   * \param i           the i-th thread
   * \param out         output matrix
   * \param data_l      data of lhs
   * \param data_r      values of csr
   * \param indptr_r    row offsets of csr
   * \param col_idx_r   column indices of csr
   * \param seg_len     workload of this thread
   * \param num_rows_l  number of rows in lhs
   * \param num_cols_l  number of columns in lhs
   * \param num_rows_r  number of rows in rhs
   * \param num_cols_r  number of columns in rhs
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  const DType* data_l,
                                  const DType* data_r,
                                  const IType* indptr_r,
                                  const CType* col_idx_r,
                                  const nnvm::dim_t seg_len,
                                  const nnvm::dim_t num_rows_l,
                                  const nnvm::dim_t num_cols_l,
                                  const nnvm::dim_t num_rows_r,
                                  const nnvm::dim_t num_cols_r) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows_l) return;
    const dim_t seg_end = std::min(seg_start + seg_len, num_rows_l);
    for (dim_t j = 0; j < num_rows_r; ++j) {
      if (indptr_r[j] == indptr_r[j+1]) continue;
      for (IType k = indptr_r[j]; k < indptr_r[j+1]; ++k) {
        const CType col_idx = col_idx_r[k];
        const DType val = data_r[k];
        for (dim_t r = seg_start; r < seg_end; ++r) {
          out[r*num_cols_r+col_idx] += data_l[r*num_cols_l+j] * val;
        }
      }
    }
  }
};

/*!
 * \brief CPU Kernel of dot(dns1, csr.T) = dns2
 * Parallelization by row blocks
 */
struct DotDnsCsrTransDnsByRowBlocks {
  /*!
   * \brief
   * \param i           the i-th thread
   * \param out         output matrix
   * \param data_l      data of lhs
   * \param data_r      values of csr
   * \param indptr_r    row offsets of csr
   * \param col_idx_r   column indices of csr
   * \param seg_len     workload of this thread
   * \param num_rows_l  number of rows in lhs
   * \param num_cols_l  number of columns in lhs
   * \param num_rows_r  number of rows in rhs
   * \param num_cols_r  number of columns in rhs
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_CINLINE static void Map(int i,
                                  DType* out,
                                  const DType* data_l,
                                  const DType* data_r,
                                  const IType* indptr_r,
                                  const CType* col_idx_r,
                                  const nnvm::dim_t seg_len,
                                  const nnvm::dim_t num_rows_l,
                                  const nnvm::dim_t num_cols_l,
                                  const nnvm::dim_t num_rows_r,
                                  const nnvm::dim_t num_cols_r) {
    using nnvm::dim_t;
    const dim_t seg_start = i * seg_len;
    if (seg_start >= num_rows_l) return;
    const dim_t seg_end = std::min(seg_start + seg_len, num_rows_l);
    for (dim_t j = 0; j < num_rows_r; ++j) {
      if (indptr_r[j] == indptr_r[j+1]) continue;
      for (IType k = indptr_r[j]; k < indptr_r[j+1]; ++k) {
        const CType col_idx = col_idx_r[k];
        const DType val = data_r[k];
        for (dim_t r = seg_start; r < seg_end; ++r) {
          out[r*num_rows_r+j] += data_l[r*num_cols_l+col_idx] * val;
        }
      }
    }
  }
};

/*!
 * \brief CPU Impl of dot(csr, dns1) = dns2 and dot(csr.T, dns1) = dns2
 */
inline void DotCsrDnsDnsImpl(const OpContext& ctx,
                             const cpu& cpu_dev,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  if (!lhs.storage_initialized()) {
    Fill(s, *ret, req, 0);
    return;
  }

  using nnvm::dim_t;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob& data_r = rhs;
  const TBlob data_out = *ret;

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        dim_t num_threads;
        if (kWriteTo == req) {
          num_threads = data_out.Size();
          mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(
              s, num_threads, data_out.dptr<DType>());
        }
        num_threads = mxnet_op::get_num_threads<cpu>(data_out.shape_[0]);
        bool dynamic = false;
        const dim_t large_matrix_threshold = 1024 * 10;
        if (data_out.shape_[0] > large_matrix_threshold) {
          dynamic = true;
          // each unit of work processes at least 1024 elements in the output
          const dim_t unit_work_per_thread = 1024;
          num_threads = data_out.Size() / unit_work_per_thread;
        }
        dim_t seg_len = (data_out.shape_[0] + num_threads - 1) / num_threads;
        if (trans_lhs) {
          mxnet_op::Kernel<DotCsrTransDnsDnsByRowBlocks, cpu>::Launch(s, num_threads,
              data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
              col_idx_l.dptr<CType>(), data_r.dptr<DType>(), seg_len,
              lhs.shape()[0], data_out.shape_[0], data_out.shape_[1]);
        } else {
          if (dynamic) {
            mxnet_op::Kernel<DotCsrDnsDnsByRowBlocks, cpu>::LaunchDynamic(s, num_threads,
                data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                col_idx_l.dptr<CType>(), data_r.dptr<DType>(), seg_len,
                data_out.shape_[0], data_out.shape_[1]);
          } else {
            mxnet_op::Kernel<DotCsrDnsDnsByRowBlocks, cpu>::Launch(s, num_threads,
                data_out.dptr<DType>(), data_l.dptr<DType>(), indptr_l.dptr<IType>(),
                col_idx_l.dptr<CType>(), data_r.dptr<DType>(), seg_len,
                data_out.shape_[0], data_out.shape_[1]);
          }
        }
      });
    });
  });
}

/*!
 * \brief CPU Impl of dot(csr.T, dns) = rsp
 */
inline void DotCsrDnsRspImpl(const OpContext& ctx,
                             const cpu& cpu_dev,
                             const NDArray& lhs,
                             const TBlob& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  if (kNullOp == req) return;
  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(ret->storage_type(), kRowSparseStorage);
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  if (!lhs.storage_initialized()) {
    FillZerosRspImpl(s, *ret);
    return;
  }
  CHECK_EQ(req, kWriteTo);

  using namespace mxnet_op;
  using nnvm::dim_t;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob& data_r = rhs;

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(ret->aux_type(rowsparse::kIdx), RType, {  // row idx type
          const dim_t num_rows = lhs.shape()[1];
          size_t workspace_size = num_rows * sizeof(dim_t);
          mshadow::Tensor<cpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<cpu, 1, char>(
            mshadow::Shape1(workspace_size), s);
          dim_t* row_flg = reinterpret_cast<dim_t*>(workspace.dptr_);
          // prefix sum array re-uses the row_flg array temp space
          dim_t* prefix_sum = row_flg;
          Kernel<set_zero, cpu>::Launch(s, num_rows, row_flg);
          Kernel<MarkRowFlgKernel, cpu>::Launch(s, lhs.aux_shape(csr::kIdx)[0], row_flg,
            col_idx_l.dptr<CType>());

          prefix_sum[0] = row_flg[0];
          for (nnvm::dim_t i = 1; i < num_rows; i++) {
            prefix_sum[i] = prefix_sum[i - 1] + row_flg[i];
          }
          dim_t nnr = prefix_sum[num_rows - 1];

          if (nnr == 0) {
            FillZerosRspImpl(s, *ret);
            return;
          }

          ret->CheckAndAlloc({mshadow::Shape1(nnr)});
          const TBlob& data_out = ret->data();
          const TBlob& row_idx = ret->aux_data(rowsparse::kIdx);

          dim_t num_threads = data_out.Size();
          mxnet_op::Kernel<set_zero, cpu>::Launch(s, num_threads, data_out.dptr<DType>());
          RType* row_idx_out = row_idx.dptr<RType>();

          mxnet_op::Kernel<FillRspRowIdxKernel, cpu>::Launch(s, num_rows,
            row_idx_out, prefix_sum, num_rows);

          num_threads = mxnet_op::get_num_threads<cpu>(nnr);
          dim_t seg_len = (nnr + num_threads - 1) / num_threads;
          if (trans_lhs) {
            mxnet_op::Kernel<DotCsrTransDnsRspByRowBlocks, cpu>::Launch(s, num_threads,
              data_out.dptr<DType>(), prefix_sum, row_idx_out, data_l.dptr<DType>(),
              indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(), data_r.dptr<DType>(),
              seg_len, lhs.shape()[0], nnr, ret->shape()[1]);
          } else {
            LOG(FATAL) << "DotCsrDnsRspImpl has not implemented dot(csr, dns)=rsp yet.";
          }
        });
      });
    });
  });
}

/*!
 * \brief CPU Impl of dot(csr, rsp) = dns
 */
inline void DotCsrRspDnsImpl(const OpContext& ctx,
                             const cpu& cpu_dev,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             TBlob* ret) {
  if (kNullOp == req) return;
  // reuse csr dns implementation when storage_shape == shape for rhs
  if (rhs.storage_shape()[0] == rhs.shape()[0]) {  // if rsp is actually dense
    DotCsrDnsDnsImpl(ctx, cpu_dev, lhs, rhs.data(), req, trans_lhs, ret);
    return;
  }

  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(rhs.storage_type(), kRowSparseStorage);
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  if (!lhs.storage_initialized() || !rhs.storage_initialized()) {
    if (kWriteTo == req) {
      MSHADOW_SGL_DBL_TYPE_SWITCH(ret->type_flag_, DType, {  // data type
        mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(
            s, ret->Size(), ret->dptr<DType>());
      });
    }
    return;
  }
  using nnvm::dim_t;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob data_r = rhs.data();
  const TBlob row_idx_r = rhs.aux_data(rowsparse::kIdx);

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(row_idx_r.type_flag_, RType, {  // row idx type
          dim_t num_threads;
          if (kWriteTo == req) {
            num_threads = ret->Size();
            mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, num_threads,
                                                              ret->dptr<DType>());
          }
          num_threads = mxnet_op::get_num_threads<cpu>(ret->shape_[0]);
          dim_t seg_len = (ret->shape_[0] + num_threads - 1) / num_threads;
          if (trans_lhs) {
            LOG(FATAL) << "DotCsrRspDnsImpl has not implemented dot(csr.T, rsp) = dns yet";
          } else {
            mxnet_op::Kernel<DotCsrRspDnsByRowBlocks, cpu>::Launch(s, num_threads,
                ret->dptr<DType>(), data_l.dptr<DType>(),
                indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(), data_r.dptr<DType>(),
                row_idx_r.dptr<RType>(), rhs.storage_shape()[0],
                ret->shape_[0], ret->shape_[1], seg_len);
          }
        });
      });
    });
  });
}

/*!
 * \brief CPU Impl of dot(csr.T, rsp1) = rsp2
 */
inline void DotCsrRspRspImpl(const OpContext& ctx,
                             const cpu& cpu_dev,
                             const NDArray& lhs,
                             const NDArray& rhs,
                             const OpReqType req,
                             const bool trans_lhs,
                             NDArray* ret) {
  if (kNullOp == req) return;
  // reuse csr dns implementation when storage_shape == shape for rhs
  if (rhs.storage_shape()[0] == rhs.shape()[0]) {  // if rsp is actually dense
    DotCsrDnsRspImpl(ctx, cpu_dev, lhs, rhs.data(), req, trans_lhs, ret);
    return;
  }

  CHECK_EQ(lhs.storage_type(), kCSRStorage);
  CHECK_EQ(rhs.storage_type(), kRowSparseStorage);
  CHECK_EQ(ret->storage_type(), kRowSparseStorage);
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  if (!lhs.storage_initialized() || !rhs.storage_initialized()) {
    FillZerosRspImpl(s, *ret);
    return;
  }
  CHECK_EQ(req, kWriteTo);

  using mxnet_op::set_zero;
  using nnvm::dim_t;

  const TBlob data_l = lhs.data();
  const TBlob indptr_l = lhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_l = lhs.aux_data(csr::kIdx);
  const TBlob data_r = rhs.data();
  const TBlob row_idx_r = rhs.aux_data(rowsparse::kIdx);

  // pre-allocate spaces for ret using the dense dimension size
  if (ret->storage_type() == kRowSparseStorage) {
    ret->CheckAndAlloc({mshadow::Shape1(lhs.shape()[1])});
  }
  const TBlob data_out = ret->data();
  const TBlob row_idx_out = ret->aux_data(rowsparse::kIdx);

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_l.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_l.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_l.type_flag_, CType, {  // col idx type
        MSHADOW_IDX_TYPE_SWITCH(row_idx_r.type_flag_, RType, {  // row idx type
          dim_t num_threads = data_out.Size();
          mxnet_op::Kernel<set_zero, cpu>::Launch(s, num_threads, data_out.dptr<DType>());
          num_threads = mxnet_op::get_num_threads<cpu>(data_out.shape_[0]);
          dim_t seg_len = (data_out.shape_[0] + num_threads - 1) / num_threads;
          if (trans_lhs) {
            RType* row_idx = row_idx_out.dptr<RType>();
            num_threads = row_idx_out.Size();
            mxnet_op::Kernel<set_zero, cpu>::Launch(s, num_threads, row_idx);
            mxnet_op::Kernel<DotCsrTransRspRspByRowBlocks, cpu>::Launch(s, num_threads,
                data_out.dptr<DType>(), row_idx, data_l.dptr<DType>(),
                indptr_l.dptr<IType>(), col_idx_l.dptr<CType>(), data_r.dptr<DType>(),
                row_idx_r.dptr<RType>(), lhs.shape()[0], rhs.storage_shape()[0],
                ret->shape()[0], ret->shape()[1], seg_len);
            dim_t nnr = 0;
            nnr = mxnet::common::ParallelAccumulate(row_idx, ret->shape()[0], nnr);
            if (0 == nnr) {
              FillZerosRspImpl(s, *ret);
              return;
            }
            ret->set_aux_shape(rowsparse::kIdx, mshadow::Shape1(nnr));
            mshadow::Tensor<cpu, 2, DType> rsp_data = data_out.FlatTo2D<cpu, DType>(s);
            dim_t idx = 0;
            for (index_t i = 0; i < ret->shape()[0]; ++i) {
              if (row_idx[i] > 0) {
                row_idx[idx] = i;
                mshadow::Copy(rsp_data[idx], rsp_data[i], s);
                ++idx;
              }
            }
          } else {
            LOG(FATAL) << "DotCsrRspRspImpl has not implemented dot(csr, rsp) = rsp2 yet";
          }
        });
      });
    });
  });
}

/*
 * \brief Impl of dot(dns, csr) = csr
 */
inline void DotDnsCsrCsrImpl(const OpContext& ctx, const cpu& cpu_dev,
                             const TBlob& lhs, const NDArray& rhs,
                             const OpReqType req, NDArray* ret) {
  if (kNullOp == req) return;

  CHECK_EQ(req, kWriteTo);
  CHECK_EQ(rhs.storage_type(), kCSRStorage);

  using namespace mshadow;
  using namespace mshadow::expr;
  using nnvm::dim_t;

  /* Initialize data structures */
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  const NDArray& out = *ret;
  const TBlob data_l = lhs;
  const TBlob data_r = rhs.data();
  const TBlob indptr_r = rhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_r = rhs.aux_data(csr::kIdx);
  if (!rhs.storage_initialized()) {
    FillZerosCsrImpl(s, *ret);
    return;
  }

  MSHADOW_SGL_DBL_TYPE_SWITCH(data_r.type_flag_, DType, {     // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_r.type_flag_, IType, {     // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_r.type_flag_, CType, {  // colidx type
        /* Allocate workspace */
        CType num_cols_out = out.shape()[1];
        CType rhs_data_size = static_cast<CType>(col_idx_r.shape_.Size());
        size_t workspace_size = 2 * num_cols_out * sizeof(CType);
        Tensor<cpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<cpu, 1, char>(
                Shape1(workspace_size), s);
        CType* col_flg = reinterpret_cast<dim_t*>(workspace.dptr_);

        CType* prefix_sum = col_flg;
        CType* nnc_idx = prefix_sum + num_cols_out;

        /* Set the column flags for nnz columns */
        mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, num_cols_out,
                                                          col_flg);
        mxnet_op::Kernel<MarkRowFlgKernel, cpu>::Launch(
            s, rhs_data_size, col_flg, col_idx_r.dptr<CType>());

        /* 1. Calculate prefix sum from col flgs
         * 2. Storage all non zero column indexes in nnc_idx
         */
        CType cur = 0;
        prefix_sum[0] = col_flg[0];
        if (prefix_sum[0]) nnc_idx[cur++] = 0;
        for (CType i = 1; i < num_cols_out; i++) {
          prefix_sum[i] = prefix_sum[i - 1] + col_flg[i];
          if (prefix_sum[i] > prefix_sum[i - 1]) nnc_idx[cur++] = i;
        }

        /* Allocate aux data for out */
        IType num_rows_l = lhs.shape_[0];
        dim_t nnc = prefix_sum[num_cols_out - 1];
        dim_t nnz = nnc * num_rows_l;
        out.CheckAndAllocAuxData(csr::kIndPtr, Shape1(num_rows_l + 1));
        out.CheckAndAllocAuxData(csr::kIdx, Shape1(nnz));
        out.CheckAndAllocData(Shape1(nnz));

        /* Set csr indptr and index according to nnc_idx*/
        IType* indptr_out = out.aux_data(csr::kIndPtr).dptr<IType>();
        CType* col_idx_out = out.aux_data(csr::kIdx).dptr<CType>();
        DType* data_out = out.data().dptr<DType>();
        mxnet_op::Kernel<PopulateCsrForNNC, cpu>::Launch(
            s, num_rows_l, nnc_idx, indptr_out, col_idx_out, nnc, num_rows_l);
        mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(s, nnz, data_out);

        const dim_t num_threads = mxnet_op::get_num_threads<cpu>(num_rows_l);
        const dim_t seg_len = (num_rows_l + num_threads - 1) / num_threads;

        IType num_rows_r = rhs.shape()[0];
        mxnet_op::Kernel<DotDnsCsrCsrByRowBlocks, cpu>::Launch(
            s, num_threads, data_out, data_l.dptr<DType>(),
            indptr_r.dptr<IType>(), col_idx_r.dptr<CType>(),
            data_r.dptr<DType>(), seg_len, num_rows_r, num_rows_l, num_cols_out,
            nnc, prefix_sum);
      });
    });
  });
}

/*
 * \brief Impl of dot(dns, csr) = dns and dot(dns, csr.T) = dns
 */
inline void DotDnsCsrDnsImpl(const OpContext& ctx, const cpu& cpu_dev,
  const TBlob& dns, const NDArray& rhs,
  const OpReqType req, NDArray* ret,
  const bool transpose_b) {
  if (req == kNullOp) return;
  CHECK_EQ(rhs.storage_type(), kCSRStorage);
  mshadow::Stream<cpu>* s = ctx.get_stream<cpu>();
  if (!rhs.storage_initialized()) {
    Fill(s, ret->data(), req, 0);
    return;
  }

  using nnvm::dim_t;

  const TBlob data_r = rhs.data();
  const TBlob indptr_r = rhs.aux_data(csr::kIndPtr);
  const TBlob col_idx_r = rhs.aux_data(csr::kIdx);
  const TBlob& data_l = dns;
  const TBlob data_out = ret->data();

  MSHADOW_REAL_TYPE_SWITCH(data_r.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(indptr_r.type_flag_, IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(col_idx_r.type_flag_, CType, {  // col idx type
        dim_t num_threads;
        if (req == kWriteTo || req == kWriteInplace) {
          num_threads = data_out.Size();
          mxnet_op::Kernel<mxnet_op::set_zero, cpu>::Launch(
              s, num_threads, data_out.dptr<DType>());
        }
        num_threads = mxnet_op::get_num_threads<cpu>(data_out.shape_[0]);
        // seg by output row
        dim_t seg_len = (data_out.shape_[0] + num_threads - 1) / num_threads;
        if (transpose_b) {
          mxnet_op::Kernel<DotDnsCsrTransDnsByRowBlocks, cpu>::Launch(s, num_threads,
              data_out.dptr<DType>(), data_l.dptr<DType>(),
              data_r.dptr<DType>(), indptr_r.dptr<IType>(),
              col_idx_r.dptr<CType>(), seg_len,
              dns.shape_[0], dns.shape_[1],
              rhs.shape()[0], rhs.shape()[1]);
        } else {
          mxnet_op::Kernel<DotDnsCsrDnsByRowBlocks, cpu>::Launch(s, num_threads,
              data_out.dptr<DType>(), data_l.dptr<DType>(),
              data_r.dptr<DType>(), indptr_r.dptr<IType>(),
              col_idx_r.dptr<CType>(), seg_len,
              dns.shape_[0], dns.shape_[1],
              rhs.shape()[0], rhs.shape()[1]);
        }
      });
    });
  });
}

inline bool DotShape(const nnvm::NodeAttrs& attrs,
                     mxnet::ShapeVector *in_attrs,
                     mxnet::ShapeVector *out_attrs) {
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& lshape = (*in_attrs)[0];
  mxnet::TShape& rshape = (*in_attrs)[1];
  if (!ndim_is_known(lshape) || !ndim_is_known(rshape)) return false;
  CHECK_GT(lshape.ndim(), 0) << "scalar tensor is not supported by this operator.";
  CHECK_GT(rshape.ndim(), 0) << "scalar tensor is not supported by this operator.";
  if (lshape.ndim() == 1 && rshape.ndim() == 1) {
    CHECK(!param.transpose_a && !param.transpose_b) << "Cannot transpose vectors";
    CHECK_EQ(lshape[0], rshape[0]) << "dot shape error: " << lshape << " X " << rshape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
  } else {
    bool Ta = param.transpose_a, Tb = param.transpose_b;
    mxnet::TShape L[2], R[2];
    if (Ta) {
      L[0] = mshadow::Shape1(lshape[0]);
      L[1] = lshape.ndim() > 1 ?
             mxnet::TShape(&lshape[1], lshape.end()) : mxnet::TShape(1, 1);
    } else {
      L[0] = lshape.ndim() > 1 ?
             mxnet::TShape(&lshape[0], &lshape[lshape.ndim()-1]) : mxnet::TShape(1, 1);
      L[1] = mshadow::Shape1(lshape[lshape.ndim()-1]);
    }
    if (Tb) {
      R[0] = rshape.ndim() > 1 ?
             mxnet::TShape(&rshape[0], &rshape[rshape.ndim()-1]) : mxnet::TShape(1, 1);
      R[1] = mshadow::Shape1(rshape[rshape.ndim()-1]);
    } else {
      R[0] = mshadow::Shape1(rshape[0]);
      R[1] = rshape.ndim() > 1 ?
             mxnet::TShape(&rshape[1], rshape.end()) : mxnet::TShape(1, 1);
    }

    if (L[!Ta].Size() != 0 && R[Tb].Size() != 0) {
      CHECK_EQ(L[!Ta].Size(), R[Tb].Size())
        << "dot shape error: " << lshape << " X " << rshape;
    }
    std::vector<index_t> buf;
    if (lshape.ndim() > 1) buf.insert(buf.end(), &L[Ta][0], L[Ta].end());
    if (rshape.ndim() > 1) buf.insert(buf.end(), &R[!Tb][0], R[!Tb].end());
    mxnet::TShape oshape(buf.begin(), buf.end());
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  }
  // return true if output shape is fully inferred
  return shape_is_known((*out_attrs)[0]);
}

#if MXNET_USE_MKLDNN == 1
inline void DotForwardExMKLDNN(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs);
#endif

template<typename xpu>
void DotForwardEx(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
#if MXNET_USE_MKLDNN == 1
  if (common::ContainsOnlyStorage(inputs, kDefaultStorage) &&
      common::ContainsOnlyStorage(outputs, kDefaultStorage)) {
    if (std::is_same<xpu, cpu>::value) {
      DotForwardExMKLDNN(attrs, ctx, inputs, req, outputs);
    } else {
      FallBackCompute(DotForward_<gpu>, attrs, ctx, inputs, req, outputs);
    }
    return;
  }
#endif
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK_EQ(inputs[0].shape().ndim(), 2) << "sparse dot only supports 2 dimensional lhs";
  CHECK_EQ(inputs[1].shape().ndim(), 2) << "sparse dot only supports 2 dimensional rhs";
  auto lhs_stype = inputs[0].storage_type();
  auto rhs_stype = inputs[1].storage_type();
  auto out_stype = outputs[0].storage_type();
  if (lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage &&
      out_stype == kDefaultStorage && !param.transpose_b) {
    TBlob ret = outputs[0].data();
    DotCsrDnsDnsImpl(ctx, xpu(), inputs[0], inputs[1].data(), req[0], param.transpose_a, &ret);
  } else if (lhs_stype == kCSRStorage && rhs_stype == kRowSparseStorage
      && out_stype == kDefaultStorage && !param.transpose_b) {
    TBlob ret = outputs[0].data();
    DotCsrRspDnsImpl(ctx, xpu(), inputs[0], inputs[1], req[0], param.transpose_a, &ret);
  } else if (lhs_stype == kCSRStorage && rhs_stype == kDefaultStorage
      && out_stype == kRowSparseStorage && !param.transpose_b) {
    NDArray out = outputs[0];
    DotCsrDnsRspImpl(ctx, xpu(), inputs[0], inputs[1].data(), req[0], param.transpose_a, &out);
  } else if (lhs_stype == kCSRStorage && rhs_stype == kRowSparseStorage
      && out_stype == kRowSparseStorage && !param.transpose_b) {
    NDArray ret = outputs[0];
    DotCsrRspRspImpl(ctx, xpu(), inputs[0], inputs[1], req[0], param.transpose_a, &ret);
  } else if (lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage &&
             out_stype == kCSRStorage &&
             !(param.transpose_a || param.transpose_b)) {
    NDArray ret = outputs[0];
    DotDnsCsrCsrImpl(ctx, xpu(), inputs[0].data(), inputs[1], req[0], &ret);
  } else if (lhs_stype == kDefaultStorage && rhs_stype == kCSRStorage &&
             out_stype == kDefaultStorage && !(param.transpose_a)) {
    NDArray ret = outputs[0];
    DotDnsCsrDnsImpl(ctx, xpu(), inputs[0].data(), inputs[1], req[0], &ret, param.transpose_b);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu>
void DotBackwardEx(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<NDArray>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  CHECK(!(req[0] != kNullOp && outputs[0].storage_type() == kCSRStorage))
    << "sparse dot does not support computing the gradient of csr";
  CHECK(!(req[1] != kNullOp && outputs[1].storage_type() == kCSRStorage))
    << "sparse dot does not support computing the gradient of csr";
  CHECK_NE(req[1], kWriteInplace) << "DotBackwardEx does not support WriteInplace";

  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK_EQ(inputs[0].shape().ndim(), 2) << "sparse dot only supports 2 dimensional lhs";
  CHECK_EQ(inputs[1].shape().ndim(), 2) << "sparse dot only supports 2 dimensional rhs";
  const auto ograd_stype = inputs[0].storage_type();
  const auto lhs_stype = inputs[1].storage_type();
  const auto rhs_stype = inputs[2].storage_type();
  const auto grad_rhs_stype = outputs[1].storage_type();
  if (ograd_stype == kDefaultStorage  // ograd dns format
      && lhs_stype == kCSRStorage  // csr input lhs of the op
      && grad_rhs_stype == kDefaultStorage && !param.transpose_b) {  // grad(rhs) dns format
    TBlob ret = outputs[1].data();
    DotCsrDnsDnsImpl(ctx, xpu(), inputs[1], inputs[0].data(), req[1], !param.transpose_a, &ret);
  } else if (ograd_stype == kDefaultStorage && lhs_stype == kCSRStorage
      && grad_rhs_stype == kRowSparseStorage && !param.transpose_b) {
    NDArray ret = outputs[1];
    DotCsrDnsRspImpl(ctx, xpu(), inputs[1], inputs[0].data(), req[1], !param.transpose_a, &ret);
  } else if (ograd_stype == kRowSparseStorage && lhs_stype == kCSRStorage
      && grad_rhs_stype == kRowSparseStorage && !param.transpose_b) {
    NDArray ret = outputs[1];
    DotCsrRspRspImpl(ctx, xpu(), inputs[1], inputs[0], req[1], !param.transpose_a, &ret);
  } else if (ograd_stype == kRowSparseStorage && lhs_stype == kCSRStorage
      && grad_rhs_stype == kDefaultStorage && !param.transpose_b) {
    TBlob ret = outputs[1].data();
    DotCsrRspDnsImpl(ctx, xpu(), inputs[1], inputs[0], req[1], !param.transpose_a, &ret);
  } else if (ograd_stype == kDefaultStorage &&  // ograd dns format
             lhs_stype == kDefaultStorage &&  // lhs dns format
             rhs_stype == kCSRStorage && !param.transpose_a) {
    NDArray ret = outputs[0];
    DotDnsCsrDnsImpl(ctx, xpu(), inputs[0].data(), inputs[2], req[0], &ret, !param.transpose_b);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu>
void BatchDotForward_(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (req[0] == kNullOp) return;
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  CHECK_EQ(outputs[0].type_flag_, inputs[0].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(outputs[0].type_flag_, inputs[1].type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(outputs[0].type_flag_ == kFloat32 || outputs[0].type_flag_ == kFloat64 ||
    (outputs[0].type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
    << "dot only supports float32/float64 for CPU, and float16/float32/float64 for GPU";
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    int ndim = outputs[0].ndim();
    if (outputs[0].shape_.Size() == 0 || inputs[0].shape_.Size() == 0
                                      || inputs[1].shape_.Size() == 0) {
      if (outputs[0].shape_.Size() != 0 && req[0] != kAddTo) {
        mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(s, outputs[0].shape_.Size(),
                                                          outputs[0].dptr<DType>());
      }
      return;
    }
    size_t batch_size = outputs[0].shape_.ProdShape(0, ndim - 2);
    mshadow::Tensor<xpu, 3, DType> out =
        outputs[0].get_with_shape<xpu, 3, DType>(Shape3(batch_size,
                                                        outputs[0].shape_[ndim - 2],
                                                        outputs[0].shape_[ndim - 1]), s);
    mshadow::Tensor<xpu, 3, DType> mlhs =
        inputs[0].get_with_shape<xpu, 3, DType>(Shape3(batch_size,
                                                       inputs[0].shape_[ndim - 2],
                                                       inputs[0].shape_[ndim - 1]), s);
    mshadow::Tensor<xpu, 3, DType> mrhs =
        inputs[1].get_with_shape<xpu, 3, DType>(Shape3(batch_size,
                                                       inputs[1].shape_[ndim - 2],
                                                       inputs[1].shape_[ndim - 1]), s);
    mshadow::Tensor<xpu, 1, DType*> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, DType*>(mshadow::Shape1(3 * out.size(0)), s);
    if (param.transpose_a && param.transpose_b) {
      mshadow::BatchGEMM<true, true>(out, mlhs, mrhs, (DType)1.0f,
                                     (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                     workspace);
    } else if (!param.transpose_a && param.transpose_b) {
      mshadow::BatchGEMM<false, true>(out, mlhs, mrhs, (DType)1.0f,
                                     (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                     workspace);
    } else if (param.transpose_a && !param.transpose_b) {
      mshadow::BatchGEMM<true, false>(out, mlhs, mrhs, (DType)1.0f,
                                     (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                     workspace);
    } else {
      mshadow::BatchGEMM<false, false>(out, mlhs, mrhs, (DType)1.0f,
                                     (kAddTo == req[0]) ? (DType)1.0f : (DType)0.0f,
                                     workspace);
    }
  });
}

inline bool BatchDotShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const DotParam& param = nnvm::get<DotParam>(attrs.parsed);
  mxnet::TShape& lshape = (*in_attrs)[0];
  mxnet::TShape& rshape = (*in_attrs)[1];
  // return false if lhs and rhs both have fully unknown shape
  if (!ndim_is_known(lshape) || !ndim_is_known(rshape)) return false;
  if (lshape.ndim() >= 3 && rshape.ndim() >= 3 && lshape.ndim() == rshape.ndim()) {
    int ndim = lshape.ndim();
    // only partially infer shape if last dim of lhs and second dim of rhs is known
    bool last_dim_known = dim_size_is_known(lshape, ndim - 1);
    bool second_dim_known = dim_size_is_known(rshape, ndim - 2);
    if ( !last_dim_known || !second_dim_known) return false;
    for (int i = 0; i < ndim - 2; i++) {
      CHECK_EQ(lshape[i], rshape[i])
        << "batch_dot shape error (the leading batch dimensions must be equal): "
        << lshape << " X " << rshape
        << " trans_a=" << param.transpose_a << " trans_b=" << param.transpose_b;
    }
    dim_t out_m = param.transpose_a ? lshape[ndim - 1] : lshape[ndim - 2];
    dim_t lshape_k = param.transpose_a ? lshape[ndim - 2] : lshape[ndim - 1];
    dim_t out_n = param.transpose_b ? rshape[ndim - 2] : rshape[ndim - 1];
    dim_t rshape_k = param.transpose_b ? rshape[ndim - 1] : rshape[ndim - 2];
    CHECK_EQ(lshape_k, rshape_k)
      << "batch_dot shape error (shape mismatch): " << lshape << " X " << rshape
      << " trans_a=" << param.transpose_a << " trans_b=" << param.transpose_b;
    std::vector<dim_t> out_shape_vec;
    for (int i = 0; i < ndim - 2; i++) {
      out_shape_vec.push_back(lshape[i]);
    }
    out_shape_vec.push_back(out_m);
    out_shape_vec.push_back(out_n);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(out_shape_vec));
  } else {
    LOG(FATAL) << "batch_dot currently only support N-D*N-D array (N >= 3)"
               << lshape << " v.s. " << rshape;
  }
  // return true if output shape is fully inferred
  return shape_is_known((*out_attrs)[0]);
}

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::DotParam> {
  size_t operator()(const mxnet::op::DotParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.transpose_a);
    ret = dmlc::HashCombine(ret, val.transpose_b);
    ret = dmlc::HashCombine(ret, val.forward_stype);
    return ret;
  }
};
}  // namespace std

#endif  // MXNET_OPERATOR_TENSOR_DOT_INL_H_
