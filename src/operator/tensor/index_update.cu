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
 * \file index_update.cu
 * \brief GPU implementation of index_update operator
 */

#include <cub/cub.cuh>
#include "./index_update-inl.h"
#include "../tensor/util/tensor_util-inl.cuh"
#include "../tensor/util/tensor_util-inl.h"


namespace mxnet {
namespace op {

template<typename DType>
struct IndexUpdateForwardGPUKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* out,
                                  const DType* val,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_pre_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                                  const int a_tail_size, const int ind_num,
                                  const int ind_ndim, const int* ind,
                                  const int a_ndim, const int seg) {
    index_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += a_pre_stride[seg + dim] * ind[dim * ind_num + i];
    }
    id *= a_tail_size;
    for (int _i = 0; _i < a_tail_size; ++_i) {
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_id = mxnet_op::unravel(_i, a_tail_shape);
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_id;
      for (int _j = 0; _j < seg; ++_j) {
        val_id[_j] = 0;
      }
      for (int _j = seg; _j < seg + a_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : a_tail_id[_j];
      }
      val_id[seg + ind_ndim - 1] = (val_shape[seg + ind_ndim - 1] == 1) ? 0 : i;
      index_t val_dest = mxnet_op::dot(val_id, val_stride);
      out[id + _i] = val[val_dest];
    }
  }
};

template<typename xpu, typename DType>
void IndexUpdateForwardCalc(mshadow::Stream<xpu> *s,
                            const int ind_num, DType* out,
                            const DType* val,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_tail_shape,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_pre_stride,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                            const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> a_shape,
                            const int a_tail_size,
                            const int ind_ndim, const int* ind,
                            const int a_ndim) {
  using namespace mxnet_op;
  using namespace mshadow;
  int seg = MXNET_SPECIAL_MAX_NDIM - a_ndim;
  Kernel<IndexUpdateForwardGPUKernel<DType>, xpu>::Launch(
    s, ind_num, out, val, a_tail_shape, a_pre_stride,
    val_stride, val_shape, a_tail_size, ind_num,
    ind_ndim, ind, a_ndim, seg);
}


struct IndexUpdateBackwardValGPUKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(size_t i, DType* grad_val,
                                  const DType* ograd, const int* ind_vec,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_tail_shape,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_pre_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                                  const int ograd_tail_size, const int ind_num,
                                  const int ind_ndim, const int out_ndim, const int seg) {
    index_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += ograd_pre_stride[seg + dim] * ind_vec[dim * ind_num + i];
    }
    id *= ograd_tail_size;
    for (int _i = 0; _i < ograd_tail_size; ++_i) {
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_tail_id =
        mxnet_op::unravel(_i, ograd_tail_shape);
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_id;
      for (int _j = 0; _j < seg; ++_j) {
        val_id[_j] = 0;
      }
      for (int _j = seg; _j < seg + out_ndim; ++_j) {
        val_id[_j] = (val_shape[_j] == 1) ? 0 : ograd_tail_id[_j];
      }
      val_id[seg + ind_ndim - 1] = (val_shape[seg + ind_ndim - 1] == 1) ? 0 : i;
      index_t val_dest = mxnet_op::dot(val_id, val_stride);
      atomicAdd(&grad_val[val_dest], ograd[id + _i]);
    }
  }
};

template<>
void IndexUpdateOpBackwardValImpl<gpu>(const OpContext& ctx,
                                const TBlob& grad_val,
                                const TBlob& ograd,
                                const TBlob& t_ind,
                                const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_tail_shape,
                                const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> ograd_pre_stride,
                                const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_stride,
                                const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> val_shape,
                                const int tail_size, const int ind_num, const int ind_ndim,
                                const int ndim) {
  using namespace mshadow;
  using namespace mxnet_op;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  int seg = MXNET_SPECIAL_MAX_NDIM - ndim;
  MXNET_NO_INT16_TYPE_SWITCH(grad_val.type_flag_, DType, {
    Kernel<IndexUpdateBackwardValGPUKernel, gpu>::Launch(
        s, ind_num, grad_val.dptr<DType>(), ograd.dptr<DType>(), t_ind.dptr<int>(),
        ograd_tail_shape, ograd_pre_stride, val_stride, val_shape, tail_size,
        ind_num, ind_ndim, ndim, seg);
  });
}

template<typename DType>
struct IndexUpdateBackwardAGPUKernel {
  MSHADOW_XINLINE static void Map(size_t i, DType* out_grad,
                                  const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> grada_pre_stride,
                                  const int tail_size, const int ind_num, const int ind_ndim,
                                  const int32_t* ind, const int seg,
                                  const int req) {
    index_t id = 0;
    for (int dim = 0; dim < ind_ndim; ++dim) {
      id += grada_pre_stride[seg + dim] * ind[dim * ind_num + i];
    }
    id *= tail_size;
    for (int _i = 0; _i < tail_size; ++_i) {
      out_grad[id + _i] = 0;
    }
  }
};

template<>
void IndexUpdateOpBackwardAImpl<gpu>(const OpContext& ctx,
                                     const TBlob& grad_a,
                                     const TBlob& ograd,
                                     const TBlob& ind,
                                     const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> grada_pre_stride,
                                     const int tail_size, const int ind_num, const int ind_ndim,
                                     const int seg, const int req) {
  using namespace mxnet_op;
  using namespace mshadow;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  MXNET_NO_INT16_TYPE_SWITCH(grad_a.type_flag_, DType, {
    size_t alignment = std::max(sizeof(DType), sizeof(int32_t));
    size_t id_size = PadBytes(sizeof(int32_t) * ind.Size(), alignment);
    size_t ograd_size = PadBytes(sizeof(DType) * ograd.Size(), alignment);
    size_t temp_mem_size = id_size + ograd_size;
    Tensor<gpu, 1, char> temp_mem =
      ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_mem_size), s);
    TBlob t_ograd = TBlob(temp_mem.dptr_, ograd.shape_, ograd.dev_mask(),
                          ograd.type_flag_, ograd.dev_id());
    TBlob t_ind = TBlob(temp_mem.dptr_ + ograd_size, ind.shape_, ind.dev_mask(),
                        mshadow::kInt32, ind.dev_id());
    mxnet_op::copy(s, t_ograd, ograd);
    mxnet_op::copy(s, t_ind, ind);
    Kernel<IndexUpdateBackwardAGPUKernel<DType>, gpu>::Launch(s, ind_num, t_ograd.dptr<DType>(),
                                                           grada_pre_stride, tail_size,
                                                           ind_num, ind_ndim,
                                                           t_ind.dptr<int32_t>(), seg, req);
    Kernel<ReqCopy<DType>, gpu>::Launch(s, grad_a.shape_.Size(), grad_a.dptr<DType>(),
                                        t_ograd.dptr<DType>(), req);
  });
}

NNVM_REGISTER_OP(_npx_index_update)
.set_attr<FCompute>("FCompute<gpu>", IndexUpdateOpForward<gpu>);

NNVM_REGISTER_OP(_backward_index_update_val)
.set_attr<FCompute>("FCompute<gpu>", IndexUpdateOpBackwardVal<gpu>);

NNVM_REGISTER_OP(_backward_index_update_a)
.set_attr<FCompute>("FCompute<gpu>", IndexUpdateOpBackwardA<gpu>);

}  // namespace op
}  // namespace mxnet
