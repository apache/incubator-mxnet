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
 * \file index_add.cu
 * \brief GPU implementation of index_add operator
 */

#include <cub/cub.cuh>
#include "./index_add-inl.h"
#include "../tensor/util/tensor_util-inl.cuh"
#include "../tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

struct IndexAddBackwardValGPUKernel {
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
void IndexAddOpBackwardValImpl<gpu>(const OpContext& ctx,
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
    Kernel<IndexAddBackwardValGPUKernel, gpu>::Launch(
    s, ind_num, grad_val.dptr<DType>(), ograd.dptr<DType>(), t_ind.dptr<int>(),
    ograd_tail_shape, ograd_pre_stride, val_stride, val_shape, tail_size,
    ind_num, ind_ndim, ndim, seg);
  });
}

NNVM_REGISTER_OP(_backward_index_add_val)
.set_attr<FCompute>("FCompute<gpu>", IndexAddOpBackwardVal<gpu>);

}  // namespace op
}  // namespace mxnet

