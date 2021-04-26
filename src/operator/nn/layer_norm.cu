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
 * Copyright (c) 2015 by Contributors
 * \file layer_norm.cu
 * \brief Implements Ba et. al, Layer Normalization (https://arxiv.org/abs/1607.06450).
*/
#include "./layer_norm-inl.h"

using namespace mshadow::cuda;

namespace mxnet {
namespace op {

template <>
void LayerNormGradComputeGeneralImpl<gpu>(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const TBlob& ograd,
                                          const TBlob& data,
                                          const TBlob& gamma,
                                          const TBlob& mean,
                                          const TBlob& std,
                                          const TBlob& normalized_data,
                                          const TBlob& ograd_mult,
                                          const TBlob& red_out,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<TBlob>& outputs,
                                          const mshadow::Tensor<gpu, 1, char>& workspace,
                                          const mxnet::TShape& red_dst_shape,
                                          const mxnet::TShape& red_src_shape,
                                          const mxnet::TShape& red_exclude_dst_shape,
                                          const mxnet::TShape& red_exclude_src_shape,
                                          const int channel_size) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  // Compute normalized_data = (data - mean) / std
  BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                    {data, mean},
                                    {kWriteTo}, {normalized_data});
  BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                    {normalized_data, std},
                                    {kWriteTo}, {normalized_data});
  // Calculate grad_beta
  if (req[2] != kNullOp) {
    BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, outputs[2].reshape(red_exclude_dst_shape), req[2], workspace,
                           ograd.reshape(red_exclude_src_shape), "red::sum{}", NDim, "identity");
    });
  }
  // Calculate grad_gamma, it will be sum(ograd * normalized_data, exclude_axis)
  ElemwiseBinaryRTCCompute {"mul"}(attrs, ctx, {normalized_data, ograd},
                                   {kWriteTo}, {ograd_mult});
  if (req[1] != kNullOp) {
    BROADCAST_NDIM_SWITCH(red_exclude_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, outputs[1].reshape(red_exclude_dst_shape), req[1], workspace,
                           ograd_mult.reshape(red_exclude_src_shape), "red::sum{}", NDim,
                           "identity");
    });
  }
  // Calculate grad_data:
  //   ograd_mult = ograd * gamma / std
  //   grad_data = ograd_mult - mean(ograd_mult, axis)
  //               + normalized_data * (-mean(normalized_data * ograd_mult, axis))
  if (req[0] != kNullOp) {
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {ograd, gamma},
                                      {kWriteTo}, {ograd_mult});
    BinaryBroadcastRTCCompute {"div"}(attrs, ctx,
                                      {ograd_mult, std},
                                      {kWriteTo}, {ograd_mult});
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
        broadcast::RTCReduce(ctx, red_out.reshape(red_dst_shape), kWriteTo, workspace,
                             ograd_mult.reshape(red_src_shape), "red::sum{}", NDim, "identity");
    });
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<gpu, 1, DType> red_out_tensor = red_out.FlatTo1D<gpu, DType>(s);
      red_out_tensor /= scalar<DType>(channel_size);
    });
    BinaryBroadcastRTCCompute {"sub"}(attrs, ctx,
                                      {ograd_mult, red_out},
                                      {req[0]}, {outputs[0]});
    ElemwiseBinaryRTCCompute {"mul"}(attrs, ctx, {ograd_mult, normalized_data},
                                     {kWriteTo}, {ograd_mult});
    BROADCAST_NDIM_SWITCH(red_dst_shape.ndim(), NDim, {
      broadcast::RTCReduce(ctx, red_out.reshape(red_dst_shape), kWriteTo, workspace,
                           ograd_mult.reshape(red_src_shape), "red::sum{}", NDim, "identity");
    });
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<gpu, 1, DType> red_out_tensor = red_out.FlatTo1D<gpu, DType>(s);
      red_out_tensor /=  scalar<DType>(- channel_size);
    });
    BinaryBroadcastRTCCompute {"mul"}(attrs, ctx,
                                      {normalized_data, red_out},
                                      {kAddTo}, {outputs[0]});
  }
}
template <typename DType>
__device__ __forceinline__ DType warp_shfl(DType value, int src_lane,
                                           int width = 32, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(mask, value, src_lane, width);
#else
  return __shfl(value, src_lane, width);
#endif
}

template <typename DType>
__device__ __forceinline__ DType warp_shfl_xor(DType value, int laneMask,
                                               int width = 32, unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}


/* A single updating step of the Welford's online algorithm to calculate the mean and variance.
 * The value 'curr' will be accumulated to the (mean, sigma2, count) triplet.
 *
 */
template<typename DType, typename IType>
__device__ __forceinline__ void StepWelfordOnlineSum(const DType curr,
                                                     DType& mean,         //NOLINT
                                                     DType& sigma2,       //NOLINT
                                                     IType& count) {      //NOLINT
  count += IType(1);
  DType delta = curr - mean;
  mean += delta / count;
  sigma2 += delta * (curr - mean);
}

/* Merge the mean/variance of two partitions. It's the key step of the Chan's parallel algorithm.
 * The (lhs_mean, lhs_sigma2, lhs_count) will be merged into (rhs_mean, rhs_sigma2, rhs_count)
 *
 * See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance for more details.
 *
 *  TODO(sxjscience) Explore the possibility of int lhs_count and rhs_count
 */
template<typename DType, typename IType>
__device__ __inline__ void ChanMergePartition(const DType lhs_mean,
                                              const DType lhs_sigma2,
                                              const IType lhs_count,
                                              DType& rhs_mean,         //NOLINT
                                              DType& rhs_sigma2,       //NOLINT
                                              IType& rhs_count) {      //NOLINT
  DType delta = rhs_mean - lhs_mean;
  DType nA = static_cast<DType>(lhs_count);
  DType nB = static_cast<DType>(rhs_count);
  rhs_count = nA + nB;
  if (rhs_count > DType(0)) {
    nA = nA / rhs_count;
    nB = nB / rhs_count;
    rhs_mean = nA * lhs_mean + nB * rhs_mean;
    rhs_sigma2 = rhs_sigma2 + lhs_sigma2 + delta * delta * nA * nB * rhs_count;
  } else {
    rhs_mean = DType(0);
    rhs_sigma2 = DType(0);
  }
}

/* Split the input column into multiple partitions and compute the mean/sigma of each partition.
 * Each thread will keep a mean/sigma2. The mean/sigma2 can be further merged to get the mean and
 * sigma2 of the column.
 */
template<typename AType, typename DType, typename IType>
__device__ __forceinline__ void BlockWelfordOnlineSum(const DType* __restrict__ col_vals,
                                                      const int nchannel,
                                                      AType& mean,         //NOLINT
                                                      AType& sigma2,       //NOLINT
                                                      IType& count) {      //NOLINT
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int nthread = blockDim.x * blockDim.y;
  // Each thread takes charge of 4 consecutive numbers. This should optimize the loading speed using
  // vectorized types like float4.
  // Also, to minimize branch divergence, we split the for-loop into two parts.
  int l = 4 * tid;
  for (; l + 3 < nchannel; l += 4 * nthread) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      StepWelfordOnlineSum(static_cast<AType>(col_vals[l + i]), mean, sigma2, count);
    }
  }
  for (; l < nchannel; ++l) {
    StepWelfordOnlineSum(static_cast<AType>(col_vals[l]), mean, sigma2, count);
  }
}

template<>
__device__ __forceinline__
void BlockWelfordOnlineSum<float, mshadow::half::half_t, int>
                                          (const mshadow::half::half_t* __restrict__ col_vals,
                                           const int nchannel,
                                           float& mean,                    //NOLINT
                                           float& sigma2,                  //NOLINT
                                           int& count) {                 //NOLINT
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int nthread = blockDim.x * blockDim.y;
  // We cast the input half pointer to half2 to optimize the loading speed.
  // Here, we need to notice that CUDA forces memory alignment, i.e.,
  // ASSERT static_cast<size_t>(ptr) % sizeof(dtype) == 0.
  // Thus, we need to shift the address of the half pointer to be aligned by half2.
  int align_shift = (reinterpret_cast<size_t>(col_vals) % 4) != 0;
  int padding = (nchannel - align_shift) % 2;
  int half2_size = (nchannel - align_shift) / 2;
  const __half2* half2_col_vals = reinterpret_cast<const __half2*>(col_vals + align_shift);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (align_shift) {
      StepWelfordOnlineSum(__half2float(col_vals[0].cuhalf_), mean, sigma2, count);
    }
    if (padding) {
      StepWelfordOnlineSum(__half2float(col_vals[nchannel - 1].cuhalf_), mean, sigma2, count);
    }
  }

  for (int l = tid; l < half2_size; l += nthread) {
    float2 ele_val =  __half22float2(half2_col_vals[l]);
    StepWelfordOnlineSum(ele_val.x, mean, sigma2, count);
    StepWelfordOnlineSum(ele_val.y, mean, sigma2, count);
  }
}

/* Fused CUDA kernel for the forward pass of layer normalization.
 * It computes the LayerNorm when axis=-1, i.e., contiguous reduction scenario.
 * Shape of the input tensors:
 *      in_data = (nbatch, nchannel)
 *      gamma = (nchannel,)
 *      beta = (nchannel,)
 *      out_data = (nchannel,)
 *      mean_data = (nbatch,)
 *      var_data = (nbatch,)
 *  It's always launched with (blockDim.x, blockDim.y) = (WARP_SIZE, blockDim.y)
 *  Also, when blockDim.y > 1, it requires shared memory that has size:
 *      sizeof(AType) * blockDim.y + sizeof(int) * blockDim.y / 2
 */
template<typename AType, typename DType, typename IType>
__global__ void LayerNormFusedForwardKernelContig(const int nbatch,
                                                  const int nchannel,
                                                  const AType eps,
                                                  const DType* __restrict__ in_data,
                                                  const DType* __restrict__ gamma,
                                                  const DType* __restrict__ beta,
                                                  DType* __restrict__ out_data,
                                                  DType* __restrict__ mean_data,
                                                  DType* __restrict__ std_data) {
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthread = blockDim.x * blockDim.y;
  IType count = 0;
  AType mean = 0;
  AType sigma2 = 0;

  if (bid < nbatch) {
    extern __shared__ char buf[];  // Shared memory
    const DType* col_vals = in_data + bid * nchannel;
    BlockWelfordOnlineSum(col_vals, nchannel, mean, sigma2, count);

    // Merge the mean/sigma2 within a warp
    // Use the Chan's Parallel Algorithm to merge all (mean, sigma2, counts)
    // within a warp of threads.
    // After calling the function, threadIdx.x == 0 will store the result of
    // the aggregated (mean, sigma2, counts).
    for (int mask = blockDim.x / 2; mask > 0; mask >>= 1) {
      AType meanB = warp_shfl_xor(mean, mask);
      AType sigma2B = warp_shfl_xor(sigma2, mask);
      IType countB = warp_shfl_xor(count, mask);
      ChanMergePartition(meanB, sigma2B, countB, mean, sigma2, count);
    }
    if (blockDim.y > 1) {
      // Inter-warp reduction. Copy the upper-half of the warps to shared memory
      // and merge with the lower-half warp
      AType* mean_buf = reinterpret_cast<AType*>(buf);
      AType* sigma2_buf =
        reinterpret_cast<AType*>(buf + sizeof(AType) * blockDim.y / 2 * blockDim.x);
      IType* count_buf = reinterpret_cast<IType*>(buf + sizeof(AType) * blockDim.y * blockDim.x);
      for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          mean_buf[idx] = mean;
          sigma2_buf[idx] = sigma2;
          count_buf[idx] = count;
        }
        __syncthreads();
        if (threadIdx.y < offset) {
          const int idx = threadIdx.y * blockDim.x + threadIdx.x;
          ChanMergePartition(mean_buf[idx], sigma2_buf[idx], count_buf[idx], mean, sigma2, count);
        }
        __syncthreads();
      }
      // Broadcast the result to all threads
      if (threadIdx.y == 0) {
        mean_buf[threadIdx.x] = mean;
        sigma2_buf[threadIdx.x] = sigma2;
      }
      __syncthreads();
      mean = mean_buf[threadIdx.x];
      sigma2 = sigma2_buf[threadIdx.x] / nchannel;
    } else {
      sigma2 /= nchannel;
    }
    // Calculate the out_data: gamma * (x - mean) / sqrt(var + eps) + beta
    AType std_eps = sqrt(sigma2 + eps);
    AType invstd_eps = DType(1.0) / std_eps;
    DType* out_col_val = out_data + bid * nchannel;

    if (gamma != nullptr && beta != nullptr) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_col_val[i] = gamma[i] * static_cast<DType>(invstd_eps *
                                                       (static_cast<AType>(col_vals[i]) - mean))
                                                         + beta[i];
      }
    } else if (gamma == nullptr && beta != nullptr) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_col_val[i] = static_cast<DType>(invstd_eps * (static_cast<AType>(col_vals[i]) - mean))
                                                       + beta[i];
      }
    } else if (gamma != nullptr && beta == nullptr) {
      for (int i = tid; i < nchannel; i += nthread) {
        out_col_val[i] = gamma[i] * static_cast<DType>(invstd_eps *
                                                       (static_cast<AType>(col_vals[i]) - mean));
      }
    } else {
      for (int i = tid; i < nchannel; i += nthread) {
        out_col_val[i] = static_cast<DType>(invstd_eps * (static_cast<AType>(col_vals[i]) - mean));
      }
    }
    // Write the out_data and var_data
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean_data[bid] = static_cast<DType>(mean);
      std_data[bid] = static_cast<DType>(std_eps);
    }
  }
}

template<bool safe_acc = false>
void LayerNormGPUContig(const LayerNormParam param,
                        const OpContext& ctx, const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 3U);
  mxnet::TShape data_shape(2, 0);
  mxnet::TShape mean_shape(1, 0);
  size_t in_ndim = inputs[layernorm::kData].ndim();
  data_shape[0] = mean_shape[0] = inputs[layernorm::kData].shape_.ProdShape(0, in_ndim - 1);
  data_shape[1] = inputs[layernorm::kData].shape_[in_ndim - 1];
  const TBlob in_data = inputs[layernorm::kData].reshape(data_shape);
  const TBlob gamma = inputs[layernorm::kGamma];
  const TBlob beta = inputs[layernorm::kBeta];
  const TBlob out_data = outputs[layernorm::kOut].reshape(data_shape);
  const TBlob mean_data = outputs[layernorm::kMean].reshape(mean_shape);
  const TBlob std_data = outputs[layernorm::kStd].reshape(mean_shape);
  // Make sure the inputs are contiguous
  CHECK_EQ(in_data.CheckContiguous(), true);
  CHECK_EQ(gamma.CheckContiguous(), true);
  CHECK_EQ(beta.CheckContiguous(), true);
  CHECK_EQ(out_data.CheckContiguous(), true);
  CHECK_EQ(mean_data.CheckContiguous(), true);
  CHECK_EQ(std_data.CheckContiguous(), true);

  // Lauch the kernel. The dynamic shared memory size is
  // sizeof(DType) * blockDim.y * blockDim.x + sizeof(DType) * blockDim.y / 2 * blockDim.x
  int nbatch = data_shape[0];
  int nchannel = data_shape[1];
  float eps = param.eps;
  int ngrid_x = (nbatch > kMaxGridDim) ? (nbatch + kBaseGridNum - 1) / kBaseGridNum : nbatch;
  int ngrid_y = (nbatch > kMaxGridDim) ? kBaseGridNum : 1;
  int nthread_y;
  const dim3 dimGrid(ngrid_x, ngrid_y);
  if (nchannel <= 128) {
    nthread_y = 1;
  } else if (nchannel <= 512) {
    nthread_y = 2;
  } else {
    nthread_y = 4;
  }
  cudaStream_t stream = Stream<gpu>::GetStream(ctx.get_stream<gpu>());
  const dim3 dimBlock(32, nthread_y);
  MXNET_REAL_ACC_TYPE_SWITCH(in_data.type_flag_, DType, AccType, {
    typedef typename std::conditional<safe_acc, AccType, DType>::type AType;
    int nshared = nthread_y > 1 ? nthread_y * 32 * sizeof(AType)
                                  + (nthread_y / 2) * 32 * sizeof(int) : 0;
    CheckLaunchParam(dimGrid, dimBlock);
    LayerNormFusedForwardKernelContig<AType, DType, int> <<<dimGrid, dimBlock, nshared, stream>>>
     (nbatch, nchannel, static_cast<AType>(eps),
      in_data.dptr<DType>(), gamma.dptr<DType>(), beta.dptr<DType>(),
      out_data.dptr<DType>(), mean_data.dptr<DType>(), std_data.dptr<DType>());
  });
  MSHADOW_CUDA_POST_KERNEL_CHECK(LayerNormFusedForwardKernelContig);
}

template<>
void LayerNormCompute<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  if (req[0] == kNullOp) return;
  CHECK_NE(req[0], kAddTo);
  int axis = param.axis;
  if (axis < 0) {
    axis += static_cast<int>(inputs[0].ndim());
  }
  CHECK(axis >= 0 && axis < inputs[0].ndim()) << "Channel axis out of range: " << param.axis;
  if (axis == inputs[0].ndim() - 1) {
    // Try to use the accelerated CUDA kernels
    bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
    if (!safe_acc && inputs[0].type_flag_ == mshadow::kFloat16) {
      common::LogOnce("MXNET_SAFE_ACCUMULATION=1 is recommended for LayerNorm with float16 inputs. "
                      "See https://mxnet.apache.org/api/faq/env_var "
                      "for more details.");
    }
    if (safe_acc) {
      return LayerNormGPUContig<true>(param, ctx, inputs, req, outputs);
    } else {
      return LayerNormGPUContig<false>(param, ctx, inputs, req, outputs);
    }
  }
  return LayerNormComputeGeneral<gpu>(attrs, ctx, inputs, req, outputs);
}


/* Fused CUDA kernel for calculating the gradient w.r.t gamma/beta in LayerNorm when axis=-1
 * (Contiguous case).
 * The gradient of gamma and beta are:
 *   d_gamma = sum(out_grad * (x - mean) / std, axis=0)
 *   d_beta = sum(out_grad, axis=0)
 *
 * We compute the gradient (mainly reduction over a non-contiguous axis) using two steps to
 * improve the parallelism.
 *
 * In the first step, we divide the rows uniformly into K parts. K independent threadblocks are used
 * to calculate the partial reduction result of each part. Illustrated below:
 *
 *      1st Block          2nd Block          3rd Block              k-th Block
 * | --------------- | ---------------- | --------------- | ... | ---------------- |
 * | --------------- | ---------------- | --------------- | ... | ---------------- |
 * | --------------- | ---------------- | --------------- | ... | ---------------- |
 * | --------------- | ---------------- | --------------- | ... | ---------------- |
 *     part_gamma[0]     part_gamma[1]      part_gamma[2]           part_gamma[k-1]
 *     part_beta[0]      part_beta[1]       part_beta[2]            part_beta[k-1]
 *
 *
 * In the second step, we sum up the row-values in part_gamma and part_beta.
 *
 * This `LayerNormFusedBackwardKernel_PartGammaBeta` function implements the first step and
 * `LayerNormFusedBackwardKernel_GammaBeta` implements the second step.
 */
template<typename AType, typename DType>
__global__ void LayerNormFusedBackwardKernel_PartGammaBeta(const int nbatch,
                                                           const int nchannel,
                                                           const DType* __restrict__ in_data,
                                                           const DType* __restrict__ out_grad,
                                                           const DType* __restrict__ mean_data,
                                                           const DType* __restrict__ std_data,
                                                           AType* __restrict__ part_gamma_grad,
                                                           AType* __restrict__ part_beta_grad) {
  extern __shared__ char buf[];
  AType* d_buf = reinterpret_cast<AType*>(buf);
  const int npart = gridDim.y;
  const int block_row_num = (nbatch + npart - 1) / npart;
  // The rows are divided into `npart` parts. Each threadblock calculates the reduction result
  // within the corresponding row ranges.
  int row_stride = blockDim.x + 1;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r_begin = blockIdx.y * block_row_num;
  int r_end = min((blockIdx.y + 1) * block_row_num, nbatch);
  AType* buf_gamma_grad = d_buf;
  AType* buf_beta_grad = d_buf + blockDim.y * row_stride;
  AType local_gamma_grad = 0;
  AType local_beta_grad = 0;

  if (c < nchannel) {
    for (int r_b = r_begin; r_b < r_end; r_b += blockDim.y) {
      int r = r_b + threadIdx.y;
      if (r < r_end) {
        AType local_mean = static_cast<AType>(mean_data[r]);
        AType local_std = static_cast<AType>(std_data[r]);
        int read_idx = r * nchannel + c;
        AType local_in_data = static_cast<AType>(in_data[read_idx]);
        AType local_out_grad = static_cast<AType>(out_grad[read_idx]);
        local_gamma_grad += (local_in_data - local_mean) / local_std * local_out_grad;
        local_beta_grad += local_out_grad;
      }
    }
  }
  buf_gamma_grad[threadIdx.y * row_stride + threadIdx.x] = local_gamma_grad;
  buf_beta_grad[threadIdx.y * row_stride + threadIdx.x] = local_beta_grad;
  __syncthreads();
  for (int offset = blockDim.y/2;  offset > 1;  offset >>= 1) {
    if (threadIdx.y < offset) {
      int idx1 = threadIdx.y * row_stride + threadIdx.x;
      int idx2 = (threadIdx.y + offset) * row_stride + threadIdx.x;
      buf_gamma_grad[idx1] += buf_gamma_grad[idx2];
      buf_beta_grad[idx1] += buf_beta_grad[idx2];
    }
    __syncthreads();
  }
  if (threadIdx.y == 0 && c < nchannel) {
    part_gamma_grad[blockIdx.y * nchannel + c] = buf_gamma_grad[threadIdx.x]
                                                   + buf_gamma_grad[threadIdx.x + row_stride];
    part_beta_grad[blockIdx.y * nchannel + c] = buf_beta_grad[threadIdx.x]
                                                   + buf_beta_grad[threadIdx.x + row_stride];
  }
}

template<bool gamma_addto, bool beta_addto, typename AType, typename DType>
__global__ void LayerNormFusedBackwardKernel_GammaBeta(const int nbatch,
                                                       const int nchannel,
                                                       const int npart,
                                                       const AType* __restrict__ part_gamma_grad,
                                                       const AType* __restrict__ part_beta_grad,
                                                       DType* gamma_grad,
                                                       DType* beta_grad) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (c < nchannel) {
    extern __shared__ char buf[];
    AType* buf_gamma_grad = reinterpret_cast<AType*>(buf);
    AType* buf_beta_grad = reinterpret_cast<AType*>(buf) + blockDim.x * blockDim.y;
    buf_gamma_grad[tid] = 0;
    buf_beta_grad[tid] = 0;
    for (int r = threadIdx.y; r < npart; r += blockDim.y) {
      buf_gamma_grad[tid] += part_gamma_grad[r * nchannel + c];
      buf_beta_grad[tid] += part_beta_grad[r * nchannel + c];
    }
    __syncthreads();
    // Begin for inter-warp reduce
    if (npart > 1) {
      for (int offset = blockDim.y/2; offset > 0; offset >>= 1) {
        if (threadIdx.y < offset) {
          int idx1 = tid;
          int idx2 = tid + offset * blockDim.x;
          buf_gamma_grad[idx1] += buf_gamma_grad[idx2];
          buf_beta_grad[idx1] += buf_beta_grad[idx2];
        }
        __syncthreads();
      }
    }
    if (threadIdx.y == 0) {
      if (gamma_grad) {
        if (gamma_addto) {
          gamma_grad[c] += static_cast<DType>(buf_gamma_grad[threadIdx.x]);
        } else {
          gamma_grad[c] = static_cast<DType>(buf_gamma_grad[threadIdx.x]);
        }
      }
      if (beta_grad) {
        if (beta_addto) {
          beta_grad[c] += static_cast<DType>(buf_beta_grad[threadIdx.x]);
        } else {
          beta_grad[c] = static_cast<DType>(buf_beta_grad[threadIdx.x]);
        }
      }
    }
  }
}

/*
 *
 *
 */
template<int LOAD_UNROLL, bool data_addto, typename AType, typename DType>
__global__ void LayerNormFusedBackwardKernel_Data(const int nbatch,
                                                  const int nchannel,
                                                  const DType* __restrict__ in_data,
                                                  const DType* __restrict__ out_grad,
                                                  const DType* __restrict__ mean_data,
                                                  const DType* __restrict__ std_data,
                                                  const DType* __restrict__ gamma,
                                                  DType* data_grad) {
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  const int nthread = blockDim.x * blockDim.y;
  if (bid < nbatch) {
    // Shared memory with size blockDim.y * blockDim.x * sizeof(DType)
    extern __shared__ char buf[];
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    // 1. Calculate: mean(out_grad * gamma / std, axis=-1)
    //               mean(out_grad * gamma / std * (x - mean) / std, axis=-1)
    AType sum_val0 = 0;  // Stores mean(out_grad * gamma / std, axis=-1)
    AType sum_val1 = 0;  // Stores mean(out_grad * gamma / std * (x - mean) / std, axis=-1)
    AType mean = static_cast<AType>(mean_data[bid]);
    AType invstd_eps = AType(1) / static_cast<AType>(std_data[bid]);
    int l = LOAD_UNROLL * tid;
    for (; l + LOAD_UNROLL - 1 < nchannel; l += nthread * LOAD_UNROLL) {
#pragma unroll
      for (int i = 0; i < LOAD_UNROLL; ++i) {
        AType ele_og = static_cast<AType>(out_grad[bid * nchannel + l + i]);
        AType ele_x = static_cast<AType>(in_data[bid * nchannel + l + i]);
        AType ele_gamma = static_cast<AType>(gamma[l + i]);
        sum_val0 += ele_og * ele_gamma * invstd_eps;
        sum_val1 += ele_og * ele_gamma * (ele_x - mean) * invstd_eps * invstd_eps;
      }
    }
    for (; l < nchannel; ++l) {
      AType ele_og = static_cast<AType>(out_grad[bid * nchannel + l]);
      AType ele_x = static_cast<AType>(in_data[bid * nchannel + l]);
      AType ele_gamma = static_cast<AType>(gamma[l]);
      sum_val0 += ele_og * ele_gamma * invstd_eps;
      sum_val1 += ele_og * ele_gamma * (ele_x - mean) * invstd_eps * invstd_eps;
    }
    // Intra-warp reduction (all-reduce)
    for (int mask = blockDim.x / 2; mask > 0; mask >>= 1) {
      sum_val0 += warp_shfl_xor(sum_val0, mask);
      sum_val1 += warp_shfl_xor(sum_val1, mask);
    }
    // Inter-warp reduction (all-reduce)
    if (blockDim.y > 1) {
      AType* sum_val0_buf = reinterpret_cast<AType*>(buf);
      AType* sum_val1_buf =
        reinterpret_cast<AType*>(buf + blockDim.y / 2 * blockDim.x * sizeof(AType));
      for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          sum_val0_buf[idx] = sum_val0;
          sum_val1_buf[idx] = sum_val1;
        }
        __syncthreads();
        if (threadIdx.y < offset) {
          const int idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_val0 += sum_val0_buf[idx];
          sum_val1 += sum_val1_buf[idx];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        sum_val0_buf[threadIdx.x] = sum_val0;
        sum_val1_buf[threadIdx.x] = sum_val1;
      }
      __syncthreads();
      sum_val0 = sum_val0_buf[threadIdx.x];
      sum_val1 = sum_val1_buf[threadIdx.x];
    }
    sum_val0 /= nchannel;
    sum_val1 /= nchannel;
    // 2. Calculate the gradient as
    //      out_grad * gamma / std - sum_val0 - (x - mean) / std * sum_val1
    for (int l = tid; l < nchannel; l += nthread) {
      AType ele_out_grad = static_cast<AType>(out_grad[bid * nchannel + l]);
      AType ele_x = static_cast<AType>(in_data[bid * nchannel + l]);
      AType ele_gamma = static_cast<AType>(gamma[l]);
      if (data_addto) {
        data_grad[bid * nchannel + l] +=
          static_cast<DType>(ele_out_grad * ele_gamma * invstd_eps
                               - sum_val0 - (ele_x - mean) * invstd_eps * sum_val1);
      } else {
        data_grad[bid * nchannel + l] =
          static_cast<DType>(ele_out_grad * ele_gamma * invstd_eps - sum_val0
                                               - (ele_x - mean) * invstd_eps * sum_val1);
      }
    }
  }
}

void GetGammaBetaGradKernelParams(const int nbatch, const int nchannel,
                                  dim3* part_grad_block_dim, dim3* part_grad_grid_dim,
                                  dim3* gb_block_dim, dim3* gb_grid_dim,
                                  int* npart) {
  *npart = 16;
  *part_grad_block_dim = dim3(32, 16);
  *part_grad_grid_dim = dim3((nchannel + 32 - 1) / 32, *npart);
  *gb_block_dim = dim3(32, *npart);
  *gb_grid_dim = dim3((nchannel + 32 - 1) / 32);
  CheckLaunchParam(*part_grad_grid_dim, *part_grad_block_dim);
  CheckLaunchParam(*gb_grid_dim, *gb_block_dim);
}

template<bool safe_acc = false>
void LayerNormGradGPUContig(const LayerNormParam param,
                            const OpContext& ctx, const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 5U);
  const TBlob out_grad = inputs[0];
  const TBlob in_data = inputs[1];
  const TBlob gamma = inputs[2];
  const TBlob mean_data = inputs[3];
  const TBlob std_data = inputs[4];
  const TBlob data_grad = outputs[0];
  const TBlob gamma_grad = outputs[1];
  const TBlob beta_grad = outputs[2];

  // Make sure the inputs are contiguous
  CHECK_EQ(out_grad.CheckContiguous(), true);
  CHECK_EQ(in_data.CheckContiguous(), true);
  CHECK_EQ(gamma.CheckContiguous(), true);
  CHECK_EQ(mean_data.CheckContiguous(), true);
  CHECK_EQ(std_data.CheckContiguous(), true);
  int nbatch = in_data.shape_.ProdShape(0, in_data.ndim() - 1);
  int nchannel = in_data.shape_[in_data.ndim() - 1];
  int data_grad_req = req[0];
  int gamma_grad_req = req[1];
  int beta_grad_req = req[2];
  CHECK_NE(data_grad_req, kWriteInplace);
  CHECK_NE(gamma_grad_req, kWriteInplace);
  CHECK_NE(beta_grad_req, kWriteInplace);
  Stream<gpu> *s = ctx.get_stream<gpu>();
  cudaStream_t stream = Stream<gpu>::GetStream(s);

  // Calculate the gradient for gamma/beta
  CHECK_EQ(gamma_grad.CheckContiguous(), true);
  CHECK_EQ(beta_grad.CheckContiguous(), true);
  dim3 part_grad_block_dim, part_grad_grid_dim, gb_block_dim, gb_grid_dim;
  int npart;
  GetGammaBetaGradKernelParams(nbatch, nchannel, &part_grad_block_dim, &part_grad_grid_dim,
                               &gb_block_dim, &gb_grid_dim, &npart);
  if (gamma_grad_req != kNullOp || beta_grad_req != kNullOp) {
    MXNET_REAL_ACC_TYPE_SWITCH(in_data.type_flag_, DType, AccType, {
      typedef typename std::conditional<safe_acc, AccType, DType>::type AType;
      Tensor<gpu, 1, AType> workspace =
        ctx.requested[0].get_space_typed<gpu, 1, AType>(Shape1(2 * npart * nchannel), s);
      AType* part_gamma_grad_ptr = workspace.dptr_;
      AType* part_beta_grad_ptr = workspace.dptr_ + npart * nchannel;
      const int nshared_K1 = 2 * (part_grad_block_dim.x + 1)
                               * part_grad_block_dim.y * sizeof(AType);
      const int nshared_K2 = 2 * gb_block_dim.x * gb_block_dim.y * sizeof(AType);
      DType* gamma_grad_ptr = (gamma_grad_req != kNullOp) ? gamma_grad.dptr<DType>() : nullptr;
      DType* beta_grad_ptr = (beta_grad_req != kNullOp) ? beta_grad.dptr<DType>() : nullptr;
      LayerNormFusedBackwardKernel_PartGammaBeta
        <<<part_grad_grid_dim, part_grad_block_dim, nshared_K1, stream>>>
        (nbatch, nchannel, in_data.dptr<DType>(), out_grad.dptr<DType>(),
         mean_data.dptr<DType>(), std_data.dptr<DType>(), part_gamma_grad_ptr, part_beta_grad_ptr);
      MSHADOW_CUDA_POST_KERNEL_CHECK(LayerNormFusedBackwardKernel_PartGammaBeta);
      if (gamma_grad_req == kAddTo && beta_grad_req != kAddTo) {
        LayerNormFusedBackwardKernel_GammaBeta<true, false>
          <<<gb_grid_dim, gb_block_dim, nshared_K2, stream>>>
          (nbatch, nchannel, npart, part_gamma_grad_ptr, part_beta_grad_ptr,
           gamma_grad_ptr, beta_grad_ptr);
      } else if (gamma_grad_req != kAddTo && beta_grad_req == kAddTo) {
        LayerNormFusedBackwardKernel_GammaBeta<false, true>
          <<<gb_grid_dim, gb_block_dim, nshared_K2, stream>>>
          (nbatch, nchannel, npart, part_gamma_grad_ptr, part_beta_grad_ptr,
            gamma_grad_ptr, beta_grad_ptr);
      } else if (gamma_grad_req == kAddTo && beta_grad_req == kAddTo) {
        LayerNormFusedBackwardKernel_GammaBeta<true, true>
          <<<gb_grid_dim, gb_block_dim, nshared_K2, stream>>>
          (nbatch, nchannel, npart, part_gamma_grad_ptr, part_beta_grad_ptr,
            gamma_grad_ptr, beta_grad_ptr);
      } else {
        LayerNormFusedBackwardKernel_GammaBeta<false, false>
          <<<gb_grid_dim, gb_block_dim, nshared_K2, stream>>>
          (nbatch, nchannel, npart, part_gamma_grad_ptr, part_beta_grad_ptr,
            gamma_grad_ptr, beta_grad_ptr);
      }
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(LayerNormFusedBackwardKernel_GammaBeta);
  }

  // Calculate the gradient for data
  CHECK_EQ(data_grad.CheckContiguous(), true);
  int ngrid_x = (nbatch > kMaxGridDim) ? (nbatch + kBaseGridNum - 1) / kBaseGridNum : nbatch;
  int ngrid_y = (nbatch > kMaxGridDim) ? kBaseGridNum : 1;
  const dim3 data_grid_dim(ngrid_x, ngrid_y);
  int nthread_y;
  if (nchannel <= 32) {
    nthread_y = 1;
  } else if (nchannel <= 128) {
    nthread_y = 2;
  } else if (nchannel <= 512) {
    nthread_y = 4;
  } else {
    nthread_y = 8;
  }
  const dim3 data_block_dim(32, nthread_y);
  const int LOAD_UNROLL = 4;
  if (data_grad_req != kNullOp) {
    MXNET_REAL_ACC_TYPE_SWITCH(in_data.type_flag_, DType, AccType, {
      typedef typename std::conditional<safe_acc, AccType, DType>::type AType;
      int nshared = data_block_dim.y > 1 ? data_block_dim.y * data_block_dim.x * sizeof(AType) : 0;
      CheckLaunchParam(data_grid_dim, data_block_dim);
      if (data_grad_req == kAddTo) {
        LayerNormFusedBackwardKernel_Data<LOAD_UNROLL, true, AType>
          <<<data_grid_dim, data_block_dim, nshared, stream>>>
          (nbatch, nchannel, in_data.dptr<DType>(), out_grad.dptr<DType>(), mean_data.dptr<DType>(),
           std_data.dptr<DType>(), gamma.dptr<DType>(), data_grad.dptr<DType>());
      } else {
        LayerNormFusedBackwardKernel_Data<LOAD_UNROLL, false, AType>
          <<<data_grid_dim, data_block_dim, nshared, stream>>>
          (nbatch, nchannel, in_data.dptr<DType>(), out_grad.dptr<DType>(), mean_data.dptr<DType>(),
           std_data.dptr<DType>(), gamma.dptr<DType>(), data_grad.dptr<DType>());
      }
    });
    MSHADOW_CUDA_POST_KERNEL_CHECK(LayerNormFusedBackwardKernel_Data);
  }
}

template<>
void LayerNormGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  const LayerNormParam& param = nnvm::get<LayerNormParam>(attrs.parsed);
  int axis = param.axis;
  if (axis < 0) {
    axis += static_cast<int>(inputs[0].ndim());
  }
  CHECK(axis >= 0 && axis < inputs[0].ndim()) << "Channel axis out of range: " << param.axis;
  if (axis == inputs[0].ndim() - 1) {
    // Use the accelerated CUDA kernels
    bool safe_acc = dmlc::GetEnv("MXNET_SAFE_ACCUMULATION", true);
    if (safe_acc) {
      return LayerNormGradGPUContig<true>(param, ctx, inputs, req, outputs);
    } else {
      return LayerNormGradGPUContig<false>(param, ctx, inputs, req, outputs);
    }
  }
  return LayerNormGradComputeGeneral<gpu>(attrs, ctx, inputs, req, outputs);
}


NNVM_REGISTER_OP(LayerNorm)
.set_attr<FCompute>("FCompute<gpu>", LayerNormCompute<gpu>);

NNVM_REGISTER_OP(_backward_LayerNorm)
.set_attr<FCompute>("FCompute<gpu>", LayerNormGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
