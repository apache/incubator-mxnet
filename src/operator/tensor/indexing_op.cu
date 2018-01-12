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
 * Copyright (c) 2017 by Contributors
 * \file indexing_op.cu
 * \brief
 * \author Siyi Li, Chi Zhang
*/

#include "./indexing_op.h"
#include "./util/tensor_util-inl.cuh"

namespace mxnet {
namespace op {

/*! \brief If there are out-of-bound indices, out will be assigned to 1.
 */

struct is_valid_check {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, int32_t* out, const DType* data,
                                  const DType min, const DType max) {
    if (data[i] < min || data[i] > max) *out = 1;
  }
};


struct AddTakeGradRspGPUKernel {
  template<typename DType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* out,
                                             const nnvm::dim_t* prefix_sum,
                                             const nnvm::dim_t* sorted_data,
                                             const nnvm::dim_t data_size,
                                             const nnvm::dim_t* original_idx,
                                             const DType* ograd,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    if (tid == 0 || sorted_data[tid - 1] != sorted_data[tid]) {
      do {
        dim_t data = sorted_data[tid];
        dim_t idx = original_idx[tid];
        dim_t row_id = prefix_sum[data] - 1;
        dim_t ograd_offset = idx * row_length;
        dim_t out_offset = row_id * row_length;
        for (int i = 0; i < row_length; i++) {
          out[out_offset + i] += ograd[ograd_offset + i];
        }
        tid++;
      } while (tid < data_size && sorted_data[tid - 1] == sorted_data[tid]);
    }
  }
};

template<>
void SparseEmbeddingOpForwardRspImpl<gpu>(const OpContext& ctx,
                                          const TBlob& data,
                                          const NDArray& weight,
                                          const OpReqType req,
                                          const TBlob& output) {
  if (req == kNullOp) return;
  using namespace rowsparse;
  using namespace mxnet_op;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  // zeros weight
  if (req == kWriteTo && !weight.storage_initialized()) {
    size_t out_size = output.shape_.Size();
    MSHADOW_TYPE_SWITCH(output.type_flag_, DType, {
      Fill<false>(s, TBlob(output.dptr<DType>(), mshadow::Shape1(out_size),
          gpu::kDevMask), kWriteTo, 0);
    })
    return;
  }
  // check out-of-bound indices
  int32_t is_valid = 0;
  MSHADOW_TYPE_SWITCH(data.type_flag_, DType, {
    DType min = 0;
    DType max = static_cast<DType>(weight.shape()[0] - 1);
    DType* data_ptr = data.dptr<DType>();
    size_t data_size = data.shape_.Size();
    Tensor<gpu, 1, char> workspace = ctx.requested[0]
        .get_space_typed<gpu, 1, char>(Shape1(sizeof(int32_t)), s);
    int32_t* is_valid_ptr = reinterpret_cast<int32_t*>(workspace.dptr_);
    Kernel<set_zero, gpu>::Launch(s, 1, is_valid_ptr);
    Kernel<is_valid_check, gpu>::Launch(s, data_size, is_valid_ptr, data_ptr, min, max);
    CUDA_CALL(cudaMemcpy(&is_valid, is_valid_ptr, sizeof(int32_t),
              cudaMemcpyDeviceToHost));
  })
  CHECK_EQ(is_valid, 0) << "SparseEmbedding input contains data out of bound";
  // the weight is actually dense
  if (weight.aux_shape(kIdx)[0] == weight.shape()[0]) {
    EmbeddingOpForwardDnsImpl<gpu>(s, data, weight.data(), req, output);
  } else {
    EmbeddingOpForwardRspImpl<gpu>(s, data, weight, req, output);
  }
}


template<>
inline void SparseEmbeddingOpBackwardRspImpl<gpu>(const OpContext& ctx,
                                                  const TBlob& ograd,
                                                  const TBlob& data,
                                                  const OpReqType req,
                                                  const NDArray& output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace mshadow::expr;
  using namespace rowsparse;
  using nnvm::dim_t;
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "SparseEmbedding layer doesn't support "
                          << "weight gradient calculation with req != write";

  // Request temporary storage for marking non-zero rows and prefix sum
  Stream<gpu> *s = ctx.get_stream<gpu>();
  dim_t num_rows = output.shape()[0];
  dim_t row_length = output.shape()[1];
  dim_t data_size = static_cast<dim_t>(data.shape_.Size());
  dim_t num_threads;
  if (data_size == 0) {
    FillZerosRspImpl(s, output);
    return;
  }

  MSHADOW_TYPE_SWITCH(data.type_flag_, IType, {
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), RType, {
        dim_t* prefix_sum = NULL;
        void* temp_storage = NULL;
        dim_t* sorted_data = NULL;
        dim_t* original_idx = NULL;
        // calculate resource bytes
        size_t row_flg_storage_bytes = num_rows * sizeof(dim_t);
        size_t sorted_data_storage_bytes = data_size * sizeof(dim_t);
        size_t original_idx_storage_bytes = data_size * sizeof(dim_t);
        size_t sum_workspace_bytes = 0;
        size_t sort_workspace_size = SortByKeyWorkspaceSize<dim_t, dim_t, gpu>(data_size);
        cub::DeviceScan::InclusiveSum(temp_storage,
                                      sum_workspace_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      Stream<gpu>::GetStream(s));
        // temp_workspace is shared by inclusive sum and sort
        size_t temp_workspace_bytes = std::max(sum_workspace_bytes, sort_workspace_size);
        size_t total_storage_bytes = row_flg_storage_bytes + sorted_data_storage_bytes +
                                     original_idx_storage_bytes + temp_workspace_bytes;

        // request resource and split it. layout =
        // row_flg/prefixsum, sorted_data, original_idx, temp_storage
        Tensor<gpu, 1, char> workspace = ctx.requested[0]
            .get_space_typed<gpu, 1, char>(Shape1(total_storage_bytes), s);
        prefix_sum = reinterpret_cast<dim_t*>(workspace.dptr_);
        sorted_data = reinterpret_cast<dim_t*>(workspace.dptr_ + row_flg_storage_bytes);
        original_idx = reinterpret_cast<dim_t*>(workspace.dptr_ + row_flg_storage_bytes +
                                                sorted_data_storage_bytes);
        temp_storage = workspace.dptr_ + total_storage_bytes - temp_workspace_bytes;
        // compute row flags and prefix sum
        num_threads = num_rows;
        Fill<false>(s, TBlob(prefix_sum, Shape1(num_threads), gpu::kDevMask), kWriteTo, 0);
        Kernel<MarkRowFlgKernel, gpu>::Launch(s, data_size, prefix_sum, data.dptr<IType>());
        cub::DeviceScan::InclusiveSum(temp_storage,
                                      temp_workspace_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      mshadow::Stream<gpu>::GetStream(s));
        // retrieve nnr and allocate output
        dim_t nnr = 0;
        CUDA_CALL(cudaMemcpy(&nnr, &prefix_sum[num_rows-1], sizeof(dim_t),
            cudaMemcpyDeviceToHost));
        output.CheckAndAlloc({Shape1(nnr)});
        // fill row_idx array of output matrix, using the row_flg values
        RType* grad_row_idx = output.aux_data(kIdx).dptr<RType>();
        Kernel<FillRspRowIdxKernel, gpu>::Launch(s, num_rows,
            grad_row_idx, prefix_sum, num_rows);

        // make a copy of the data, to be sorted
        TBlob sorted_data_blob(sorted_data, Shape1(data_size), gpu::kDevMask);
        auto sorted_data_tensor = sorted_data_blob.FlatTo1D<gpu, dim_t>(s);
        mxnet_op::copy(s, sorted_data_blob, data);

        // generate original idx
        Tensor<gpu, 1, dim_t> original_idx_tensor(original_idx, Shape1(data_size), s);
        Kernel<range_fwd, gpu>::Launch(s, data_size, 1, static_cast<dim_t>(0), static_cast<dim_t>(1),
                                       kWriteTo, original_idx);
        // sort data with its original idx
        int num_bits = ilog2(num_rows - 1);
        char* temp_storage_ptr = reinterpret_cast<char*>(temp_storage);
        Tensor<gpu, 1, char> temp_storage_tensor(temp_storage_ptr,
                                                 Shape1(sort_workspace_size), s);
        SortByKey(sorted_data_tensor, original_idx_tensor, true,
                  &temp_storage_tensor, 0, num_bits);

        // accumulate gradients
        DType* grad_data = output.data().dptr<DType>();
        Fill<false>(s, TBlob(grad_data, Shape1(nnr * row_length), gpu::kDevMask),
            kWriteTo, 0);
        num_threads = data_size;
        Kernel<AddTakeGradRspGPUKernel, gpu>::Launch(s, num_threads, grad_data, prefix_sum,
               sorted_data, data_size, original_idx, ograd.dptr<DType>(), row_length);
      });
    });
  });
}

NNVM_REGISTER_OP(Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpForward<gpu>);

NNVM_REGISTER_OP(_contrib_SparseEmbedding)
.set_attr<FComputeEx>("FComputeEx<gpu>", SparseEmbeddingOpForwardEx<gpu>);

NNVM_REGISTER_OP(_backward_Embedding)
.set_attr<FCompute>("FCompute<gpu>", EmbeddingOpBackward<gpu>);

NNVM_REGISTER_OP(_backward_SparseEmbedding)
.set_attr<FComputeEx>("FComputeEx<gpu>", SparseEmbeddingOpBackwardEx<gpu>);

NNVM_REGISTER_OP(take)
.set_attr<FCompute>("FCompute<gpu>", TakeOpForward<gpu>);

NNVM_REGISTER_OP(_backward_take)
.set_attr<FCompute>("FCompute<gpu>", TakeOpBackward<gpu>);

NNVM_REGISTER_OP(batch_take)
.set_attr<FCompute>("FCompute<gpu>", BatchTakeOpForward<gpu>);

NNVM_REGISTER_OP(one_hot)
.set_attr<FCompute>("FCompute<gpu>", OneHotOpForward<gpu>);

NNVM_REGISTER_OP(gather_nd)
.set_attr<FCompute>("FCompute<gpu>", GatherNDForward<gpu>);

NNVM_REGISTER_OP(scatter_nd)
.set_attr<FCompute>("FCompute<gpu>", ScatterNDForward<gpu>);

NNVM_REGISTER_OP(_scatter_set_nd)
.set_attr<FCompute>("FCompute<gpu>", ScatterSetNDForward<gpu>);
}  // namespace op
}  // namespace mxnet
