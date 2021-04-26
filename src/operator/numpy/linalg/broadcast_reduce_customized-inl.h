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
 * \file broadcast_reduce_customized-inl.h
 * \brief CPU-specific Function definition of broadcast and reduce operators
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_BROADCAST_REDUCE_CUSTOMIZED_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_BROADCAST_REDUCE_CUSTOMIZED_INL_H_

#include "../../tensor/broadcast_reduce-inl.h"

namespace mxnet {
namespace op {
namespace broadcast {
using namespace mshadow;
using mxnet_op::unravel;
using mxnet_op::ravel;
using mxnet_op::dot;
using mxnet_op::unravel_dot;

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
MSHADOW_XINLINE void seq_reduce_assign_wr(const index_t idx, const size_t M, const bool addto,
                                          const DType* __restrict big, OType *small,
                                          const Shape<ndim>& bshape, const Shape<ndim>& sshape,
                                          const Shape<ndim>& rshape, const Shape<ndim>& rstride,
                                          Reducer* reducer) {
  Shape<ndim> coord = unravel(idx, sshape);
  index_t j = ravel(coord, bshape);
  AType val, residual;
  reducer->SetInitValue(val, residual);
  for (size_t k = 0; k < M; ++k) {
    coord = unravel(k, rshape);
    reducer->Reduce(val, AType(OP::Map(big[j + dot(coord, rstride)])), residual);
  }
  reducer->Finalize(val, residual);
  assign(&small[idx], addto, OType(val));
}

template<typename Reducer, int ndim, typename AType, typename DType, typename OType, typename OP>
void seq_reduce_compute_wr(const size_t N, const size_t M, const bool addto,
                           const DType *big, OType *small, const Shape<ndim> bshape,
                           const Shape<ndim> sshape, const Shape<ndim> rshape,
                           const Shape<ndim> rstride,
                           Reducer* reducer) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    seq_reduce_assign_wr<Reducer, ndim, AType, DType, OType, OP>(idx, M, addto, big, small,
        bshape, sshape, rshape, rstride, reducer);
  }
}

template <typename Reducer, int ndim, typename DType, typename OP, bool safe_acc = false>
void ReduceWithReducer(Stream<cpu>* s, const TBlob& small, const OpReqType req,
                       const Tensor<cpu, 1, char>& workspace, const TBlob& big,
                       Reducer* reducer) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size(), M = rshape.Size();
  if (!safe_acc) {
    seq_reduce_compute_wr<Reducer, ndim, DType, DType, DType, OP>(
      N, M, req == kAddTo, big.dptr<DType>(), small.dptr<DType>(),
      big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride, reducer);
  } else {
    MXNET_ACC_TYPE_SWITCH(mshadow::DataType<DType>::kFlag, DataType, AType, {
      typedef typename std::conditional<safe_acc, AType, DataType>::type AccType;
      MSHADOW_TYPE_SWITCH_WITH_BOOL(small.type_flag_, OType, {
        typedef typename std::conditional<safe_acc, OType, DataType>::type OutType;
        seq_reduce_compute_wr<Reducer, ndim, AccType, DataType, OutType, OP>(
          N, M, req == kAddTo, big.dptr<DataType>(), small.dptr<OutType>(),
          big.shape_.get<ndim>(), small.shape_.get<ndim>(), rshape, rstride, reducer);
      });
    });
  }
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
MSHADOW_XINLINE void seq_reduce_assign_wr(const index_t idx, const size_t M, const bool addto,
                                          const DType* __restrict big, const DType* __restrict lhs,
                                          const DType* __restrict rhs, DType *small,
                                          const Shape<ndim>& big_shape,
                                          const Shape<ndim>& lhs_shape0,
                                          const Shape<ndim>& rhs_shape0,
                                          const Shape<ndim>& small_shape, const Shape<ndim>& rshape,
                                          const Shape<ndim>& lhs_shape,
                                          const Shape<ndim>& rhs_shape,
                                          const Shape<ndim>& rstride, const Shape<ndim>& lhs_stride,
                                          const Shape<ndim>& rhs_stride,
                                          Reducer* reducer) {
  Shape<ndim> coord = unravel(idx, small_shape);
  const index_t idx_big0 = ravel(coord, big_shape);
  const index_t idx_lhs0 = ravel(coord, lhs_shape0);
  const index_t idx_rhs0 = ravel(coord, rhs_shape0);
  DType val, residual;
  reducer->SetInitValue(val, residual);
  for (size_t k = 0; k < M; ++k) {
    Shape<ndim> coord_big = unravel(k, rshape);
    index_t idx_big = idx_big0 + dot(coord_big, rstride);

    Shape<ndim> coord_lhs = unravel(k, lhs_shape);
    index_t idx_lhs = idx_lhs0 + dot(coord_lhs, lhs_stride);

    Shape<ndim> coord_rhs = unravel(k, rhs_shape);
    index_t idx_rhs = idx_rhs0 + dot(coord_rhs, rhs_stride);

    reducer->Reduce(val, OP1::Map(big[idx_big], OP2::Map(lhs[idx_lhs], rhs[idx_rhs])), residual);
  }
  reducer->Finalize(val, residual);
  assign(&small[idx], addto, val);
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void seq_reduce_compute_wr(const size_t N, const size_t M, const bool addto,
                           const DType *big, const DType *lhs, const DType *rhs, DType *small,
                           const Shape<ndim> big_shape, const Shape<ndim> small_shape,
                           const Shape<ndim> rshape, const Shape<ndim> rstride,
                           const Shape<ndim> lhs_shape, const Shape<ndim> lhs_stride,
                           const Shape<ndim> rhs_shape, const Shape<ndim> rhs_stride,
                           const Shape<ndim>& lhs_shape0, const Shape<ndim>& rhs_shape0,
                           Reducer* reducer) {
  #pragma omp parallel for num_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount())
  for (index_t idx = 0; idx < static_cast<index_t>(N); ++idx) {
    seq_reduce_assign_wr<Reducer, ndim, DType, OP1, OP2>(idx, M, addto, big, lhs, rhs, small,
      big_shape, lhs_shape0, rhs_shape0, small_shape, rshape, lhs_shape, rhs_shape, rstride,
      lhs_stride, rhs_stride, reducer);
  }
}

template<typename Reducer, int ndim, typename DType, typename OP1, typename OP2>
void ReduceWithReducer(Stream<cpu> *s, const TBlob& small, const OpReqType req,
                       const Tensor<cpu, 1, char>& workspace, const TBlob& big, const TBlob& lhs,
                       const TBlob& rhs, Reducer* reducer) {
  if (req == kNullOp) return;
  Shape<ndim> rshape, rstride;
  diff(small.shape_.get<ndim>(), big.shape_.get<ndim>(), &rshape, &rstride);
  size_t N = small.shape_.Size();
  size_t M = rshape.Size();

  Shape<ndim> lhs_shape, lhs_stride;
  diff(small.shape_.get<ndim>(), lhs.shape_.get<ndim>(), &lhs_shape, &lhs_stride);

  Shape<ndim> rhs_shape, rhs_stride;
  diff(small.shape_.get<ndim>(), rhs.shape_.get<ndim>(), &rhs_shape, &rhs_stride);

  seq_reduce_compute_wr<Reducer, ndim, DType, OP1, OP2>(
    N, M, req == kAddTo,
    big.dptr<DType>(), lhs.dptr<DType>(), rhs.dptr<DType>(), small.dptr<DType>(),
    big.shape_.get<ndim>(), small.shape_.get<ndim>(),
    rshape, rstride,
    lhs_shape, lhs_stride,
    rhs_shape, rhs_stride,
    lhs.shape_.get<ndim>(), rhs.shape_.get<ndim>(),
    reducer);
}

}  // namespace broadcast
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_BROADCAST_REDUCE_CUSTOMIZED_INL_H_
