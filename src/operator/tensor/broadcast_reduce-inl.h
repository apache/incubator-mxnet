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
 *  Copyright (c) 2015-2017 by Contributors
 * \file broadcast_reduce-inl.h
 * \brief CPU-specific Function definition of broadcast and reduce operators
 */
#ifndef MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_
#define MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {
namespace mxnet_op {
template<int ndim, typename OP>
struct binary_broadcast_kernel {
  /*! \brief Map function for binary_broadcast_kernel */
  template<typename IType, typename DType>
  MSHADOW_XINLINE static void Map(index_t base, index_t length, OpReqType req,
                                  const Shape <ndim> &lstride, const Shape <ndim> &rstride,
                                  const Shape <ndim> &oshape, IType *lhs, IType *rhs,
                                  DType *out) {
    Shape <ndim> coord = unravel(base, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto ridx = static_cast<index_t>(dot(coord, rstride));
    KERNEL_ASSIGN(out[base], req, OP::Map(lhs[lidx], rhs[ridx]));
    // starts from 1 to avoid extra inc at end of loop
    for (index_t i = 1; i < length; ++i) {
      inc(&coord, oshape, &lidx, lstride, &ridx, rstride);
      // When tuning, don't actually run the op, since it's not going to be tuned against
      // the actual op we'll eventually be using
      KERNEL_ASSIGN(out[base + i], req, OP::Map(lhs[lidx], rhs[ridx]));
    }
  }

  /*! \brief Map function for binary_broadcast_kernel */
  template<typename LType, typename RType, typename OType>
  MSHADOW_XINLINE static void Map(index_t base, index_t length, OpReqType req,
                                  const Shape <ndim> &lstride, const Shape <ndim> &rstride,
                                  const Shape <ndim> &oshape, LType *lhs, RType *rhs,
                                  OType *out) {
    Shape <ndim> coord = unravel(base, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto ridx = static_cast<index_t>(dot(coord, rstride));
    KERNEL_ASSIGN(out[base], req, OP::Map(lhs[lidx], rhs[ridx]));
    // starts from 1 to avoid extra inc at end of loop
    for (index_t i = 1; i < length; ++i) {
      inc(&coord, oshape, &lidx, lstride, &ridx, rstride);
      // When tuning, don't actually run the op, since it's not going to be tuned against
      // the actual op we'll eventually be using
      KERNEL_ASSIGN(out[base + i], req, OP::Map(lhs[lidx], rhs[ridx]));
    }
  }

  /*! \brief Map function for binary_broadcast_kernel */
  template<typename IType, typename DType>
  MSHADOW_XINLINE static void Map(index_t base, index_t length, OpReqType req,
                                  const Shape <ndim> &lstride, const Shape <ndim> &rstride,
                                  const Shape <ndim> &oshape, IType lhs, IType *rhs,
                                  DType *out) {
    Shape <ndim> coord = unravel(base, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto ridx = static_cast<index_t>(dot(coord, rstride));
    KERNEL_ASSIGN(out[base], req, OP::Map(lhs, rhs[ridx]));
    // starts from 1 to avoid extra inc at end of loop
    for (index_t i = 1; i < length; ++i) {
      inc(&coord, oshape, &lidx, lstride, &ridx, rstride);
      // When tuning, don't actually run the op, since it's not going to be tuned against
      // the actual op we'll eventually be using
      KERNEL_ASSIGN(out[base + i], req, OP::Map(lhs, rhs[ridx]));
    }
  }

  /*! \brief Map function for binary_broadcast_kernel */
  /* used for mixed type binary ops */
  template<typename IType, typename DType,
           typename std::enable_if<!std::is_same<IType, DType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t base, index_t length, OpReqType req,
                                  const Shape <ndim> &lstride, const Shape <ndim> &rstride,
                                  const Shape <ndim> &oshape, IType *lhs, DType *rhs,
                                  DType *out) {
    Shape <ndim> coord = unravel(base, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto ridx = static_cast<index_t>(dot(coord, rstride));
    KERNEL_ASSIGN(out[base], req, OP::Map(lhs[lidx], rhs[ridx]));
    // starts from 1 to avoid extra inc at end of loop
    for (index_t i = 1; i < length; ++i) {
      inc(&coord, oshape, &lidx, lstride, &ridx, rstride);
      // When tuning, don't actually run the op, since it's not going to be tuned against
      // the actual op we'll eventually be using
      KERNEL_ASSIGN(out[base + i], req, OP::Map(lhs[lidx], rhs[ridx]));
    }
  }

  /*! \brief Map function for binary_broadcast_kernel */
  /* used for mixed type binary ops */
  template<typename IType, typename DType,
           typename std::enable_if<!std::is_same<IType, DType>::value &&
                                   !std::is_pointer<IType>::value, int>::type = 0>
  MSHADOW_XINLINE static void Map(index_t base, index_t length, OpReqType req,
                                  const Shape <ndim> &lstride, const Shape <ndim> &rstride,
                                  const Shape <ndim> &oshape, IType lhs, DType *rhs,
                                  DType *out) {
    Shape <ndim> coord = unravel(base, oshape);
    auto lidx = static_cast<index_t>(dot(coord, lstride));
    auto ridx = static_cast<index_t>(dot(coord, rstride));
    KERNEL_ASSIGN(out[base], req, OP::Map(lhs, rhs[ridx]));
    // starts from 1 to avoid extra inc at end of loop
    for (index_t i = 1; i < length; ++i) {
      inc(&coord, oshape, &lidx, lstride, &ridx, rstride);
      // When tuning, don't actually run the op, since it's not going to be tuned against
      // the actual op we'll eventually be using
      KERNEL_ASSIGN(out[base + i], req, OP::Map(lhs, rhs[ridx]));
    }
  }
};

template<int req, typename OP, bool col_vec>
struct csr_dns_csr_broadcast_kernel {
  /*!
   * \brief Map function for broadcast between csr and 1D vector
   * \param row          global thread id/assigned row id
   * \param csr_data     ptr to data buffer of csr matrix
   * \param csr_indices  ptr to indices buffer of csr matrix
   * \param csr_indptr   ptr to indptr buffer of csr matrix
   * \param dns          ptr to data buffer of the dense vector
   * \param out          ptr to the data buffer of the result csr matrix
   */
  template<typename DType, typename CType, typename RType>
  MSHADOW_XINLINE static void Map(index_t row, const DType *csr_data, const CType *csr_indices,
                                  const RType *csr_indptr, const DType *dns, DType *out) {
    const nnvm::dim_t curr_row_i = csr_indptr[row];
    const nnvm::dim_t next_row_i = csr_indptr[row + 1];
    for (nnvm::dim_t iter = curr_row_i; iter < next_row_i; iter++) {
      KERNEL_ASSIGN(out[iter], req, OP::Map(csr_data[iter],
                    (col_vec)? dns[row] : dns[csr_indices[iter]]));
    }
  }

  /*!
   * \brief Map function for broadcast between csr and a scalar
   * \param i           global thread id
   * \param csr_data    ptr to data buffer of csr matrix
   * \param scalar_ptr  ptr to data buffer of the scalar tensor, only the 0-th element is used
   * \param out         ptr to the data buffer of output csr matrix
   * \param nnz         number of non-zero elements in input csr matrix
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, const DType *csr_data, const DType* scalar_ptr,
                                  DType *out, const nnvm::dim_t nnz) {
    const DType scale = scalar_ptr[0];
    if (i < nnz) {
      KERNEL_ASSIGN(out[i], req, OP::Map(csr_data[i], scale));
    }
  }
};

template<int req, typename OP, bool reverse = false>
struct csr_dns_map_kernel {
  template <typename DType, typename CType, typename RType>
  MSHADOW_XINLINE static void Map(index_t row, const DType *csr_data, const CType *csr_indices,
                                  const RType *csr_indptr, DType *out, const nnvm::dim_t num_rows,
                                  const nnvm::dim_t num_cols) {
    if (row < num_rows) {
      const nnvm::dim_t curr_row_i = csr_indptr[row];
      const nnvm::dim_t next_row_i = csr_indptr[row + 1];
      for (nnvm::dim_t iter = curr_row_i; iter < next_row_i; iter++) {
        const nnvm::dim_t target = row * num_cols + csr_indices[iter];
        KERNEL_ASSIGN(out[target], req,
                      reverse ? OP::Map(out[target], csr_data[iter]) :
                                OP::Map(csr_data[iter], out[target]));
      }
    }
  }
};

}  // namespace mxnet_op

namespace broadcast {
using namespace mshadow;

const int MAX_DIM = 5;

template<int ndim>
MSHADOW_XINLINE void unravel_dot(const index_t idx, const Shape<ndim>& shape,
  const Shape<ndim>& stridej, const Shape<ndim>& stridek, index_t* j, index_t* k) {
  *j = 0;
  *k = 0;
  #pragma unroll
  for (index_t i = ndim-1, idx_t = idx; i >=0; --i) {
    const auto tmp = idx_t / shape[i];
    const auto coord = idx_t - tmp*shape[i];
    *j += coord*stridej[i];
    *k += coord*stridek[i];
    idx_t = tmp;
  }
}

template<int ndim>
MSHADOW_XINLINE int diff(const Shape<ndim>& small,
                         const Shape<ndim>& big,
                         Shape<ndim>* dims,
                         Shape<ndim>* stride) {
  int mdim = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i) {
    mdim += small[i] != big[i];
    (*dims)[i] = (*stride)[i] = 1;
  }

  index_t s = 1;
  #pragma unroll
  for (int i = ndim - 1, j = mdim; i >= 0; --i) {
    if (small[i] != big[i]) {
      --j;
      (*stride)[j] = s;
      (*dims)[j] = big[i];
    }
    s *= big[i];
  }
  return mdim;
}

template<typename DType>
MSHADOW_XINLINE void assign(DType* dst, const bool addto, const DType src) {
  if (addto) {
    *dst += src;
  } else {
    *dst = src;
  }
}

template<int ndim, typename DType, typename OP>
MSHADOW_XINLINE void binary_broadcast_assign(const index_t idx, const bool addto,
                                             const DType* __restrict lhs,
                                             const DType* __restrict rhs, DType* out,
                                             const Shape<ndim>& lshape, const Shape<ndim>& rshape,
                                             const Shape<ndim>& oshape) {
  const Shape<ndim> coord = mxnet_op::unravel(idx, oshape);
  const index_t j = mxnet_op::ravel(coord, lshape);
  const index_t k = mxnet_op::ravel(coord, rshape);
  assign(&out[idx], addto, OP::Map(lhs[j], rhs[k]));
}

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP,
         typename IndexOP = mxnet::op::mshadow_op::set_index_no_op<AType, index_t>>
MSHADOW_XINLINE std::pair<AType, AType> seq_reduce_assign_block(size_t start, size_t len,
                                              size_t j,
                                              const DType* __restrict big,
                                              const Shape<ndim>& rshape,
                                              const Shape<ndim>& rstride) {
  Shape<ndim> coord;
  AType val, residual;
  Reducer::SetInitValue(val, residual);
  for (size_t k = start; k < start + len; ++k) {
    coord = mxnet_op::unravel(k, rshape);
    AType temp = OP::Map(big[j + mxnet_op::dot(coord, rstride)]);
    if (IndexOP::do_op)
      IndexOP::Op(&temp, k);
    Reducer::Reduce(val, temp, residual);
  }
  return std::make_pair(val, residual);
}

template<typename Reducer, int ndim, typename AType, typename DType, typename OType,
         typename OP, typename IndexOP = mxnet::op::mshadow_op::set_index_no_op<AType, index_t>>
MSHADOW_XINLINE void seq_reduce_assign(const index_t idx, const size_t M, const bool addto,
                                       const DType* __restrict big, OType *small,
                                       const Shape<ndim>& bshape, const Shape<ndim>& sshape,
                                       const Shape<ndim>& rshape, const Shape<ndim>& rstride,
                                       const bool use_omp = false) {
  Shape<ndim> coord = mxnet_op::unravel(idx, sshape);
  index_t j = mxnet_op::ravel(coord, bshape);
  AType val, residual;
  Reducer::SetInitValue(val, residual);
  if (!use_omp) {
    for (size_t k = 0; k < M; ++k) {
      coord = mxnet_op::unravel(k, rshape);
      AType temp = OP::Map(big[j + mxnet_op::dot(coord, rstride)]);
      // argmin/max, set IndexedNum.idx
      if (IndexOP::do_op)
        IndexOP::Op(&temp, k);
      Reducer::Reduce(val, temp, residual);
    }
  } else {
    const int thread_count = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    auto pairs = std::make_unique<std::pair<AType, AType>[]>(thread_count);
    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < thread_count; ++i) {
      pairs[i] = seq_reduce_assign_block<Reducer, ndim, AType, DType, OType, OP, IndexOP>
          (i * (M / thread_count),
          i < (thread_count - 1) ? (M / thread_count) : (M / thread_count) + M % thread_count,
          j, big, rshape, rstride);
    }
    for (int i = 0; i < thread_count; ++i) {
      Reducer::Merge(val, residual, pairs[i].first, pairs[i].second);
    }
  }
  Reducer::Finalize(val, residual);
  assign(&small[idx], addto, OType(val));
}

namespace {

// Returns the stride with which the fastest dimension is moving.
// Used to detect memory access scatter.
inline int fastest_stride(const TShape &small, const TShape &big,
                                   const TShape &big_stride) {
  const int ndim = small.ndim();
  for (int i = ndim-1; i >= 0; --i) {
    if (big[i] != 1) {
      return (small[i] == big[i]) ? 1 : big_stride[i];
    }
  }
  return 1;
}

}  // namespace

template<int ndim, typename DType, typename OP>
void BinaryBroadcastComputeImpl(Stream<cpu> *s, const OpReqType req,
                                const TBlob& lhs, const TBlob& rhs, const TBlob& out) {
  mshadow::Shape<ndim> oshape = out.shape_.get<ndim>();
  mshadow::Shape<ndim> lstride = mxnet_op::calc_stride(lhs.shape_.get<ndim>());
  mshadow::Shape<ndim> rstride = mxnet_op::calc_stride(rhs.shape_.get<ndim>());
  mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<ndim, OP>, cpu>::
  template LaunchEx(s, out.shape_.Size(), req, lstride, rstride, oshape,
                    lhs.dptr<DType>(), rhs.dptr<DType>(), out.dptr<DType>());
}

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP,
         typename IndexOP = mxnet::op::mshadow_op::set_index_no_op<AType, index_t>>
void seq_reduce_compute(const size_t N, const size_t M, const bool addto,
                        const DType *big, OType *small, const Shape<ndim> bshape,
                        const Shape<ndim> sshape, const Shape<ndim> rshape,
                        const Shape<ndim> rstride) {
  const int thread_count = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
  #pragma omp parallel for num_threads(thread_count) if (N >= thread_count)
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    seq_reduce_assign<Reducer, ndim, AType, DType, OType, OP, IndexOP>
        (idx, M, addto, big, small, bshape, sshape, rshape, rstride, N < thread_count);
  }
}

template <typename Reducer, int ndim, typename DType, typename OP>
void seq_reduce_compute_extra_mem(const size_t N, const size_t M, const bool addto,
                                  const DType* big, DType* small,
                                  const Shape<ndim> bshape,
                                  const Shape<ndim> sshape,
                                  const Shape<ndim> rshape,
                                  const Shape<ndim> rstride,
                                  const index_t* ws_dptr) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    Shape<ndim> coord = mxnet_op::unravel(idx, sshape);
    index_t j = mxnet_op::ravel(coord, bshape);
    DType val, residual;
    Reducer::SetInitValue(val, residual);
    for (size_t k = 0; k < M; ++k) {
      Reducer::Reduce(val, OP::Map(big[j + ws_dptr[k]]), residual);
    }
    assign(&small[idx], addto, val);
  }
}

template <typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void Reduce(Stream<cpu>* s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size(), M = rshape.Size();
  if (!safe_acc) {
    seq_reduce_compute<Reducer, ndim, DType, DType, DType, OP>(
      N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(),
      big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride);
  } else {
    MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH_WITH_BOOL(small.type_flag_, OType, {
        typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
        seq_reduce_compute<Reducer, ndim, AccType, DataType, OutType, OP>(
          N, M, req == kAddTo, big.dptr<DataType>(), small.dptr<OutType>(),
          big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride);
      });
    });
  }
}

template <typename Reducer, int ndim, typename DType, typename OP>
void ReduceBool(Stream<cpu>* s, const TBlob& small, const OpReqType req,
                const Tensor<cpu, 1, char>& workspace, const TBlob& big) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size(), M = rshape.Size();
  seq_reduce_compute<Reducer, ndim, bool, DType, bool, OP>(
    N, M, req == kAddTo, big.dptr<DType>(), small.dptr<bool>(),
    big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride);
}

template <typename Reducer, int ndim, typename DType, typename OP>
void ReduceWithExtraMem(Stream<cpu>* s, const TBlob& small, const OpReqType req,
                        const Tensor<cpu, 1, char>& workspace, const TBlob& big) {
  using namespace mxnet_op;
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  index_t* ws_dptr = reinterpret_cast<index_t*>(workspace.dptr_);
  size_t N = small.shape_.Size(), M = rshape.Size();
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t k = 0; k < static_cast<index_t>(M); k++) {
    Shape<ndim> coord = mxnet_op::unravel(k, rshape);
    ws_dptr[k] = mxnet_op::dot(coord, rstride);
  }

  seq_reduce_compute_extra_mem<Reducer, ndim, DType, OP>(
    N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(), big.shape_.get<ndim>(),
    small.shape_.get<ndim>(), rshape, rstride, ws_dptr);
}

inline size_t ReduceWorkspaceSize(Stream<cpu> *s, const mxnet::TShape& small, const OpReqType req,
                                  const mxnet::TShape& big) {
  return 0;
}

inline size_t ReduceWorkspaceSize(Stream<cpu> *s, const mxnet::TShape& small, const OpReqType req,
                                  const mxnet::TShape& big, const mxnet::TShape& lhs,
                                  const mxnet::TShape& rhs) {
  return 0;
}

#if MXNET_USE_CUDA

namespace {

constexpr int warpSize = 32;
constexpr int unroll_reduce = 2;

// Returns a/b integer division rounded up
template<typename Type>
Type ceil_idiv(const Type a, const Type b) {
  return (a + b - 1)/b;
}

uint64_t calc_num_load(const int X, const int Y, const int* strides) {
  // Number of full warps
  uint64_t num_full_warp = X / warpSize;
  // Length of the partial warp i.e. number of threads that are performing loads
  uint64_t len_part_warp = X % warpSize;

  uint64_t num_load_full = (std::min(warpSize, strides[0]) +
    std::min(warpSize, strides[1]) +
    std::min(warpSize, strides[2]))*num_full_warp;

  uint64_t num_load_part =
  (std::min(len_part_warp, ceil_idiv<uint64_t>(len_part_warp*strides[0], warpSize)) +
    std::min(len_part_warp, ceil_idiv<uint64_t>(len_part_warp*strides[1], warpSize)) +
    std::min(len_part_warp, ceil_idiv<uint64_t>(len_part_warp*strides[2], warpSize)))*
  (len_part_warp != 0);

  uint64_t num_load = (num_load_full + num_load_part)*(uint64_t)Y;
  return num_load;
}

inline int diff(const TShape& small, const TShape& big,
                TShape* dims, TShape* stride) {
  int ndim = small.ndim();
  int mdim = 0;
  #pragma unroll
  for (int i = 0; i < ndim; ++i) {
    mdim += small[i] != big[i];
    (*dims)[i] = (*stride)[i] = 1;
  }

  index_t s = 1;
  #pragma unroll
  for (int i = ndim - 1, j = mdim; i >= 0; --i) {
    if (small[i] != big[i]) {
      --j;
      (*stride)[j] = s;
      (*dims)[j] = big[i];
    }
    s *= big[i];
  }
  return mdim;
}

constexpr int nthread_reduce = 512;
constexpr index_t kBaseGridNum = 1024;

}  // namespace

// Configuration for ReduceImpl()
struct ReduceImplConfig {
  index_t N;
  index_t M;
  index_t Mnext;
  struct {
    dim3 blockDim;
    dim3 gridDim;
    int shMemSize;
    bool do_transpose;
  } kernel_1;
  struct {
    int blockSize;
    int gridSize;
  } kernel_2;
  size_t workspace_size;

  TShape rshape, rstride;
  TShape lhs_shape, lhs_stride;
  TShape rhs_shape, rhs_stride;

  inline ReduceImplConfig(const ::mxnet::TShape& small, const ::mxnet::TShape& big,
                          const ::mxnet::TShape* lhs,
                          const ::mxnet::TShape* rhs) :
    rshape(small.ndim(), 1), rstride(small.ndim(), 1),
    lhs_shape(small.ndim(), 1), lhs_stride(small.ndim(), 1),
    rhs_shape(small.ndim(), 1), rhs_stride(small.ndim(), 1) {
    // The largest reduction type currently is (index_t, double) struct
    // aligned to 16B
    constexpr size_t max_type_size = 2 * sizeof(double);
    constexpr int maxLoopPerTB = 64;
    int ndim = small.ndim();

    diff(small, big, &rshape, &rstride);
    N = small.Size();

    M = rshape[0];
    for (int i = 1; i < ndim; ++i) {
      M *= rshape[i];
    }

    bool multiOp = false;
    if (lhs != nullptr) {
      CHECK_NOTNULL(rhs);
      diff(small, *lhs, &lhs_shape, &lhs_stride);
      diff(small, *rhs, &rhs_shape, &rhs_stride);
      multiOp = true;
    }

    workspace_size = 0;
    kernel_1.shMemSize = 0;
    kernel_1.do_transpose = false;

    if (M == 1) {
      kernel_1.blockDim.x = nthread_reduce;
      kernel_1.gridDim.x = std::min(kBaseGridNum,
          static_cast<index_t>((N + kernel_1.blockDim.x - 1)/kernel_1.blockDim.x));
    } else {
      int reduce_strides[3];
      reduce_strides[0] = fastest_stride(small, big, big);
      reduce_strides[1] = (multiOp) ? fastest_stride(small, *lhs, *lhs) : 1;
      reduce_strides[2] = (multiOp) ? fastest_stride(small, *rhs, *rhs) : 1;

      int reduce_strides_transp[3];
      reduce_strides_transp[0] = fastest_stride(small, rshape, rstride);
      reduce_strides_transp[1] = (multiOp) ?
        fastest_stride(small, lhs_shape, lhs_stride) : 1;
      reduce_strides_transp[2] = (multiOp) ?
        fastest_stride(small, rhs_shape, rhs_stride) : 1;

      uint64_t num_load = calc_num_load(N, M, reduce_strides);
      uint64_t num_load_transp = calc_num_load(M, N, reduce_strides_transp);

      Mnext = 1;
      kernel_1.do_transpose = (num_load > num_load_transp);

      kernel_1.blockDim.x = 0;
      kernel_1.blockDim.y = 0;

      if (kernel_1.do_transpose) {
        // Fastest thread ID goes through M
        // Loop over N has step size kernel_1.blockDim.y
        if (N < 8) {
          kernel_1.blockDim.y = 1;
        } else if (N < 256) {
          kernel_1.blockDim.y = 4;
        } else {
          if (M < 8) {
            kernel_1.blockDim.x = 1;
          } else if (M < 256) {
            kernel_1.blockDim.x = 4;
          } else {
            kernel_1.blockDim.x = warpSize;
          }
        }
      } else {
        // Fastest thread ID goes through N
        // Loop over M has step size kernel_1.blockDim.y
        if (M < 8) {
          kernel_1.blockDim.y = 1;
        } else if (M < 256) {
          kernel_1.blockDim.y = 4;
        } else {
          if (N < 8) {
            kernel_1.blockDim.x = 1;
          } else if (N < 256) {
            kernel_1.blockDim.x = 4;
          } else {
            kernel_1.blockDim.x = warpSize;
          }
        }
      }

      if (kernel_1.blockDim.x == 0 && kernel_1.blockDim.y == 0) {
        LOG(FATAL) << "Unable to set blockDim";
      } else if (kernel_1.blockDim.x == 0) {
        kernel_1.blockDim.x = nthread_reduce / kernel_1.blockDim.y;
      } else if (kernel_1.blockDim.y == 0) {
        kernel_1.blockDim.y = nthread_reduce / kernel_1.blockDim.x;
      }

      if (kernel_1.do_transpose) {
        // Fastest thread ID goes through M
        kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
            ceil_idiv<unsigned int>(N, kernel_1.blockDim.y));
        kernel_1.gridDim.y = std::min(kBaseGridNum, Mnext);
        int by = kernel_1.blockDim.y;
        if (kernel_1.blockDim.y % warpSize == 0) {
          // Fix shared memory bank conflict
          by++;
        }
        kernel_1.shMemSize = (kernel_1.blockDim.x > 1) ?
          kernel_1.blockDim.x*by*max_type_size * 2 : 0;
        // Maximum number of times we want TB to loop in M
        // Max size of M-block each TB can handle
        int maxMblock = kernel_1.blockDim.x*maxLoopPerTB;
        Mnext = (M + maxMblock - 1) / maxMblock;
      } else {
        // Fastest thread ID goes through N
        kernel_1.gridDim.x = std::min((unsigned int)kBaseGridNum,
            ceil_idiv<unsigned int>(N, kernel_1.blockDim.x));
        kernel_1.gridDim.y = std::min(kBaseGridNum, Mnext);
        kernel_1.shMemSize = (kernel_1.blockDim.y > 1) ?
          kernel_1.blockDim.x*kernel_1.blockDim.y*max_type_size * 2 : 0;
        // Maximum number of times we want TB to loop in M
        // Max size of M-block each TB can handle
        int maxMblock = kernel_1.blockDim.y*maxLoopPerTB;
        Mnext = (M + maxMblock - 1) / maxMblock;
      }

      if (Mnext > 1) {
        // small_dptr[] is N*Mnext*type_size bytes
        workspace_size += N * Mnext * max_type_size;
        // Set gridDim.y to Mnext
        kernel_1.gridDim.y = std::min(kBaseGridNum, Mnext);
      }

      if (Mnext > 1) {
        kernel_2.blockSize = nthread_reduce;
        kernel_2.gridSize = std::min(kBaseGridNum,
            static_cast<index_t>((N + kernel_2.blockSize - 1)/kernel_2.blockSize));
      }
    }
  }
};

inline size_t ReduceWorkspaceSize(Stream<gpu> *s, const ::mxnet::TShape& small, const OpReqType req,
                                  const ::mxnet::TShape& big) {
  if (req == kNullOp) return 0;
  ReduceImplConfig config(small, big, nullptr, nullptr);
  return config.workspace_size;
}

inline size_t ReduceWorkspaceSize(Stream<gpu> *s, const ::mxnet::TShape& small, const OpReqType req,
                                  const ::mxnet::TShape& big, const ::mxnet::TShape& lhs,
                                  const ::mxnet::TShape& rhs) {
  if (req == kNullOp) return 0;
  ReduceImplConfig config(small, big, &lhs, &rhs);
  return config.workspace_size;
}

#endif  // MXNET_USE_CUDA

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
MSHADOW_XINLINE void seq_reduce_assign(const index_t idx, const size_t M, const bool addto,
                                       const DType* __restrict big, const DType* __restrict lhs,
                                       const DType* __restrict rhs, DType *small,
                                       const Shape<ndim>& big_shape, const Shape<ndim>& lhs_shape0,
                                       const Shape<ndim>& rhs_shape0,
                                       const Shape<ndim>& small_shape, const Shape<ndim>& rshape,
                                       const Shape<ndim>& lhs_shape, const Shape<ndim>& rhs_shape,
                                       const Shape<ndim>& rstride, const Shape<ndim>& lhs_stride,
                                       const Shape<ndim>& rhs_stride) {
  Shape<ndim> coord = mxnet_op::unravel(idx, small_shape);
  const index_t idx_big0 = mxnet_op::ravel(coord, big_shape);
  const index_t idx_lhs0 = mxnet_op::ravel(coord, lhs_shape0);
  const index_t idx_rhs0 = mxnet_op::ravel(coord, rhs_shape0);
  DType val, residual;
  Reducer::SetInitValue(val, residual);
  for (size_t k = 0; k < M; ++k) {
    Shape<ndim> coord_big = mxnet_op::unravel(k, rshape);
    index_t idx_big = idx_big0 + mxnet_op::dot(coord_big, rstride);

    Shape<ndim> coord_lhs = mxnet_op::unravel(k, lhs_shape);
    index_t idx_lhs = idx_lhs0 + mxnet_op::dot(coord_lhs, lhs_stride);

    Shape<ndim> coord_rhs = mxnet_op::unravel(k, rhs_shape);
    index_t idx_rhs = idx_rhs0 + mxnet_op::dot(coord_rhs, rhs_stride);

    Reducer::Reduce(val, OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs])), residual);
  }
  Reducer::Finalize(val, residual);
  assign(&small[idx], addto, val);
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void seq_reduce_compute(const size_t N, const size_t M, const bool addto,
                        const DType *big, const DType *lhs, const DType *rhs, DType *small,
                        const Shape<ndim> big_shape, const Shape<ndim> small_shape,
                        const Shape<ndim> rshape, const Shape<ndim> rstride,
                        const Shape<ndim> lhs_shape, const Shape<ndim> lhs_stride,
                        const Shape<ndim> rhs_shape, const Shape<ndim> rhs_stride,
                        const Shape<ndim>& lhs_shape0, const Shape<ndim>& rhs_shape0) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    seq_reduce_assign<Reducer, ndim, DType, OP1, OP2>(idx, M, addto, big, lhs, rhs, small,
      big_shape, lhs_shape0, rhs_shape0, small_shape, rshape, lhs_shape, rhs_shape, rstride,
      lhs_stride, rhs_stride);
  }
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void Reduce(Stream<cpu> *s, const TBlob& small, const OpReqType req,
            const Tensor<cpu, 1, char>& workspace, const TBlob& big, const TBlob& lhs,
            const TBlob& rhs) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size();
  size_t M = rshape.Size();

  Shape<ndim> lhs_shape, lhs_stride;
  diff(small.shape_.get<ndim>(), lhs.shape_.get<ndim>(), &lhs_shape, &lhs_stride);

  Shape<ndim> rhs_shape, rhs_stride;
  diff(small.shape_.get<ndim>(), rhs.shape_.get<ndim>(), &rhs_shape, &rhs_stride);

  seq_reduce_compute<Reducer, ndim, DType, OP1, OP2>(
    N, M, req == kAddTo,
    big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(), small.dptr<DType>(),
    big.shape_.get<ndim>(), small.shape_.get<ndim>(),
    rshape, rstride,
    lhs_shape, lhs_stride,
    rhs_shape, rhs_stride,
    lhs.shape_.get<ndim>(), rhs.shape_.get<ndim>());
}

#if MXNET_USE_CUDA

void RTCReduce(const OpContext& ctx,
               const TBlob& small,
               const OpReqType req,
               const Tensor<gpu, 1, char>& workspace,
               const TBlob& big,
               const std::string& reducer,
               int ndim,
               const std::string& OP,
               const bool use_index = false);

void RTCReduce(const OpContext& ctx,
               const TBlob& small,
               const OpReqType req,
               const Tensor<gpu, 1, char>& workspace,
               const TBlob& big,
               const TBlob &lhs,
               const TBlob &rhs,
               const std::string& reducer,
               int ndim,
               const std::string& OP1,
               const std::string& OP2);

#endif

}  // namespace broadcast
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_BROADCAST_REDUCE_INL_H_
