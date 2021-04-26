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

#include "broadcast_reduce-inl.h"
#include "elemwise_unary_op.h"

#if MXNET_USE_CUDA
#include "../../common/cuda/rtc.h"
#endif  // MXNET_USE_CUDA

using namespace mshadow;

namespace mxnet {
namespace op {
namespace broadcast {

#if MXNET_USE_CUDA

namespace {

struct reduce_kernel_params {
  index_t big_shape[MAX_DIM];
  index_t small_shape[MAX_DIM];
  index_t lhs_shape0[MAX_DIM];
  index_t rhs_shape0[MAX_DIM];
  index_t rshape[MAX_DIM];
  index_t rstride[MAX_DIM];
  index_t lhs_stride[MAX_DIM];
  index_t rhs_stride[MAX_DIM];
  index_t lhs_shape[MAX_DIM];
  index_t rhs_shape[MAX_DIM];
};

const char reduce_function_code[] = R"code(
#define FUNC OP(IType0::from(big[idx_big[u]]))
using AType = typename AccType<InputType0>::type;
)code";

const char reduce_function_use_input_code[] = R"code(
#define FUNC OP1(IType0::from(big[idx_big[u]]),     \
                 OP2(IType1::from(lhs[idx_lhs[u]]), \
                     IType2::from(rhs[idx_rhs[u]])))
using AType = typename AccType<InputType0>::type;
)code";

const char reduce_function_index_code[] = R"code(
#define FUNC AType(OP(IType0::from(big[idx_big[u]])), index)

template <typename T>
struct AccIndex {
  index_t idx;
  T num;

  __device__ inline AccIndex<T>() {}
  __device__ inline AccIndex<T>(const T& val, const index_t idx) : num(val), idx(idx) {}

  __device__ inline operator index_t() const volatile {
    return idx;
  }

  __device__ inline AccIndex<T>& operator=(const AccIndex<T>& other) {
    idx = other.idx;
    num = other.num;
    return *this;
  }
};

using AType = AccIndex<typename AccType<InputType0>::type>;
)code";

const char reduce_kernel_code[] = R"code(
struct reduce_kernel_params {
  index_t big_shape[util::MAX_DIM];
  index_t small_shape[util::MAX_DIM];
  index_t lhs_shape0[util::MAX_DIM];
  index_t rhs_shape0[util::MAX_DIM];
  index_t rshape[util::MAX_DIM];
  index_t rstride[util::MAX_DIM];
  index_t lhs_stride[util::MAX_DIM];
  index_t rhs_stride[util::MAX_DIM];
  index_t lhs_shape[util::MAX_DIM];
  index_t rhs_shape[util::MAX_DIM];
};

inline __device__ AType reduce(const index_t idx, const int tidx,
                               const int tidy, const int N,
                               const index_t Mstart, const index_t Mend,
                               const InputType0* __restrict big,
                               const InputType1* __restrict lhs,
                               const InputType2* __restrict rhs,
                               const reduce_kernel_params& params) {
  extern __shared__ char shTileChar[];
  using IType0 = AccType<InputType0>;
  using IType1 = AccType<InputType1>;
  using IType2 = AccType<InputType2>;
  using OType = AccType<OutputType0>;
  AType* shTile = (AType*)(shTileChar);
  const int bx = (do_transpose) ? blockDim.y : blockDim.x;
  const int by = (do_transpose) ? blockDim.x : blockDim.y;
  index_t coord[ndim];
  util::unravel(idx, params.small_shape, coord);
  index_t idx_big0, idx_lhs0, idx_rhs0;
  idx_big0 = util::ravel(coord, params.big_shape);
  if (use_input) {
    idx_lhs0 = util::ravel(coord, params.lhs_shape0);
    idx_rhs0 = util::ravel(coord, params.rhs_shape0);
  }

  AType val, residual;
  REDUCER.SetInitValue(val, residual);
  if (idx < N) {
    for (index_t k = tidy + Mstart; k < Mend; k += by*UNROLL) {
      index_t idx_big[UNROLL];
      index_t idx_lhs[UNROLL];
      index_t idx_rhs[UNROLL];
      #pragma unroll
      for (int u=0;u < UNROLL;u++) {
        idx_big[u] = idx_big0 + util::unravel_dot<ndim>(k + u*by, params.rshape,
                                                        params.rstride);
        if (use_input) {
          idx_lhs[u] = idx_lhs0 + util::unravel_dot<ndim>(k + u*by, params.lhs_shape,
                                                          params.lhs_stride);
          idx_rhs[u] = idx_rhs0 + util::unravel_dot<ndim>(k + u*by, params.rhs_shape,
                                                          params.rhs_stride);
        }
      }
      AType tmp[UNROLL];
      #pragma unroll
      for (int u=0;u < UNROLL;u++) {
        if (k + u*by < Mend) {
          const index_t index = k + u*by;
          tmp[u] = FUNC;
        }
      }
      #pragma unroll
      for (int u=0;u < UNROLL;u++) {
        if (k + u*by < Mend) REDUCER.Reduce(val, tmp[u], residual);
      }
    }
  }

  // Shared memory block bx * by. Reduction is along by. Final result is in tidy=0
  if (by > 1) {
    // Fix bx to avoid bank conflicts. Assumes warpSize number of banks
    const int fbx = (do_transpose && ((bx & (warpSize - 1)) == 0)) ? (bx + 1) : bx;
    const int it0 = tidx + tidy*fbx;
    shTile[it0 * 2] = val;
    shTile[it0 * 2 + 1] = residual;
    __syncthreads();
    for (int t=1;t < by;t <<= 1) {
      AType tmp, tmp_residual;
      REDUCER.SetInitValue(tmp, tmp_residual);
      if (tidy + t < by) {
        tmp = shTile[(it0 + t*fbx) * 2];
        tmp_residual = shTile[(it0 + t*fbx) * 2 + 1];
      }
      __syncthreads();
      REDUCER.Merge(shTile[it0 * 2], shTile[it0 * 2 + 1], tmp, tmp_residual);
      __syncthreads();
    }
    if (idx < N && tidy == 0) {
      REDUCER.Finalize(shTile[tidx * 2], shTile[tidx * 2 + 1]);
      return shTile[tidx * 2];
    } else {
      return AType();
    }
  } else {
    if (idx < N) {
      REDUCER.Finalize(val, residual);
      return val;
    } else {
      return AType();
    }
  }
}

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void reduce_kernel_single(const int N, const int M,
                                     const InputType0* __restrict big,
                                     const InputType1* __restrict lhs,
                                     const InputType2* __restrict rhs,
                                     OutputType0 *small,
                                     const reduce_kernel_params params,
                                     const int Mnext) {
  using OType = AccType<OutputType0>;
  const int tid = threadIdx.x + threadIdx.y*blockDim.x;
  const int bx = (do_transpose) ? blockDim.y : blockDim.x;
  const int by = (do_transpose) ? blockDim.x : blockDim.y;
  const int tidx = (do_transpose) ? tid / by : threadIdx.x;
  const int tidy = (do_transpose) ? tid % by : threadIdx.y;
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const index_t Mstart = (index_t)((int64)M*(int64)m0/(int64)Mnext);
    const index_t Mend   = (index_t)((int64)M*(int64)(m0 + 1)/(int64)Mnext);
    for (index_t idx0 = blockIdx.x*bx; idx0 < N; idx0 += bx*gridDim.x) {
      const index_t idx = idx0 + tidx;
      AType val = reduce(idx, tidx, tidy, N, Mstart, Mend, big, lhs, rhs, params);
      if (idx < N && (by == 1 || tidy == 0)) {
        if (req == OpReqType::kAddTo) {
          small[idx + m0 * N] = OType::to(op::add(OType::from(small[idx + m0 * N]),
                                                  static_cast<typename OType::type>(val)));
        } else {
          small[idx + m0 * N] = OType::to(val);
        }
      }
    }
  }
}

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void reduce_kernel_multi(const int N, const int M,
                                    const InputType0* __restrict big,
                                    const InputType1* __restrict lhs,
                                    const InputType2* __restrict rhs,
                                    AType *small,
                                    const reduce_kernel_params params,
                                    const int Mnext) {
  const int tid = threadIdx.x + threadIdx.y*blockDim.x;
  const int bx = (do_transpose) ? blockDim.y : blockDim.x;
  const int by = (do_transpose) ? blockDim.x : blockDim.y;
  const int tidx = (do_transpose) ? tid / by : threadIdx.x;
  const int tidy = (do_transpose) ? tid % by : threadIdx.y;
  for (int m0 = blockIdx.y; m0 < Mnext; m0 += gridDim.y) {
    // This TB handles M range [Mstart, ...., Mend - 1]
    const index_t Mstart = (index_t)((int64)M*(int64)m0/(int64)Mnext);
    const index_t Mend   = (index_t)((int64)M*(int64)(m0 + 1)/(int64)Mnext);
    for (index_t idx0 = blockIdx.x*bx; idx0 < N; idx0 += bx*gridDim.x) {
      const index_t idx = idx0 + tidx;
      AType val = reduce(idx, tidx, tidy, N, Mstart, Mend, big, lhs, rhs, params);
      if (idx < N && (by == 1 || tidy == 0)) {
        small[idx + m0 * N] = val;
      }
    }
  }
}

)code";

const char reduce_lines_kernel_code[] = R"code(
__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void reduce_lines_kernel(const index_t N, const index_t M,
                                    const index_t small_in_stride,
                                    const AType* __restrict small_in,
                                    OutputType0 *small_out) {
  using OType = AccType<OutputType0>;
  for (index_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    AType val, residual;
    REDUCER.SetInitValue(val, residual);
    for (int k = 0; k < M; k++) {
      REDUCER.Reduce(val,
        small_in[idx + k*small_in_stride],
        residual);
    }

    if (idx < N) {
      REDUCER.Finalize(val, residual);
      if (req == OpReqType::kAddTo) {
        small_out[idx] = OType::to(op::add(OType::from(small_out[idx]),
                                           static_cast<typename OType::type>(val)));
      } else {
        small_out[idx] = OType::to(val);
      }
    }
  }
}
)code";

void RTCReduceImpl(Stream<gpu> *s, const TBlob& small, const bool addto,
                const TBlob& big, const Tensor<gpu, 1, char>& workspace,
                const ReduceImplConfig& config, const int ndim,
                const std::string &common_code, int dev_id,
                const TBlob *lhs = nullptr, const TBlob *rhs = nullptr,
                const bool use_index = false) {
  using namespace common::cuda::rtc;
  void* small_dptr = small.dptr_;
  if (config.Mnext > 1) {
    // small_dptr[] is N*Mnext*sizeof(DType) bytes
    small_dptr = workspace.dptr_;
    // Check that the workspace is contigiuous
    CHECK_EQ(workspace.CheckContiguous(), true);
    // Check that we have enough storage
    CHECK_GE(workspace.size(0), config.workspace_size);
  }

  const int by = (config.kernel_1.do_transpose) ?
    config.kernel_1.blockDim.x : config.kernel_1.blockDim.y;
  const bool do_unroll = (config.M / (by*config.Mnext) >= unroll_reduce);
  std::string code = common_code +
                     "#define UNROLL " +
                     (do_unroll ? std::to_string(unroll_reduce) : "1") +
                     "\n"
                     "const bool do_transpose = " +
                     (config.kernel_1.do_transpose ? "true" : "false") +
                     ";\n"
                     "using InputType0 = " +
                     common::mshadow_type_info(big.type_flag_).name +
                     ";\n"
                     "using OutputType0 = " +
                     common::mshadow_type_info(small.type_flag_).name +
                     ";\n"
                     "using InputType1 = " +
                     ((lhs != nullptr)
                     ? common::mshadow_type_info(lhs->type_flag_).name
                     : "float32") +
                     ";\n"
                     "using InputType2 = " +
                     ((rhs != nullptr)
                     ? common::mshadow_type_info(rhs->type_flag_).name
                     : "float32") +
                     ";\n";
  if (lhs != nullptr) {
    code += "const bool use_input = true;";
  } else {
    code += "const bool use_input = false;";
  }

  reduce_kernel_params param {};
  for (int i = 0; i < ndim; ++i) {
    param.big_shape[i] = big.shape_[i];
    param.small_shape[i] = small.shape_[i];
    param.rshape[i] = config.rshape[i];
    param.rstride[i] = config.rstride[i];
    if (lhs != nullptr) {
      param.lhs_shape0[i] = lhs->shape_[i];
      param.rhs_shape0[i] = rhs->shape_[i];
      param.lhs_shape[i] = config.lhs_shape[i];
      param.rhs_shape[i] = config.rhs_shape[i];
      param.lhs_stride[i] = config.lhs_stride[i];
      param.rhs_stride[i] = config.rhs_stride[i];
    }
  }

  void *null_ptr = nullptr;
  std::vector<const void*> args;
  args.emplace_back(&config.N);
  args.emplace_back(&config.M);
  args.emplace_back(&big.dptr_);
  if (lhs != nullptr) {
    args.emplace_back(&(lhs->dptr_));
    args.emplace_back(&(rhs->dptr_));
  } else {
    args.emplace_back(&(null_ptr));
    args.emplace_back(&(null_ptr));
  }
  args.emplace_back(&small_dptr);
  args.emplace_back(&param);
  args.emplace_back(&config.Mnext);

  const auto &function_code = (lhs == nullptr)
                            ? (use_index ? reduce_function_index_code : reduce_function_code)
                            : reduce_function_use_input_code;
  const auto& kernel_name = (config.Mnext > 1) ? "reduce_kernel_multi" : "reduce_kernel_single";
  auto reduce_kernel_func = get_function(code + function_code,
                                         kernel_name,
                                         reduce_kernel_code,
                                         dev_id);
  launch(reduce_kernel_func, config.kernel_1.gridDim,
         config.kernel_1.blockDim,
         config.kernel_1.shMemSize, s, &args);

  if (config.Mnext > 1) {
    args.resize(0);
    args.emplace_back(&config.N);
    args.emplace_back(&config.Mnext);
    args.emplace_back(&config.N);
    args.emplace_back(&small_dptr);
    args.emplace_back(&small.dptr_);

    auto reduce_lines_kernel_func = get_function(code + function_code,
                                                 "reduce_lines_kernel",
                                                 reduce_lines_kernel_code,
                                                 dev_id);
    launch(reduce_lines_kernel_func, config.kernel_2.gridSize,
           config.kernel_2.blockSize, 0, s, &args);
  }
}

struct reduce_kernel_M1_params {
  index_t big_shape[MAX_DIM];
  index_t lhs_shape[MAX_DIM];
  index_t rhs_shape[MAX_DIM];
  index_t small_shape[MAX_DIM];
};

const char reduce_kernel_M1_code[] = R"code(
struct reduce_kernel_M1_params {
  index_t big_shape[util::MAX_DIM];
  index_t lhs_shape[util::MAX_DIM];
  index_t rhs_shape[util::MAX_DIM];
  index_t small_shape[util::MAX_DIM];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void reduce_kernel_M1(const int N,
                                 const InputType0* __restrict big,
                                 const InputType1* __restrict lhs,
                                 const InputType2* __restrict rhs,
                                 OutputType0 *small,
                                 const reduce_kernel_M1_params params) {
  using IType0 = AccType<InputType0>;
  using IType1 = AccType<InputType1>;
  using IType2 = AccType<InputType2>;
  using OType = AccType<OutputType0>;
  for (index_t index = threadIdx.x + blockIdx.x*blockDim.x;
       index < N;
       index += blockDim.x*gridDim.x) {
    index_t coord[ndim];
    util::unravel(index, params.small_shape, coord);
    index_t idx_big[1];
    idx_big[0] = util::ravel(coord, params.big_shape);
    index_t idx_lhs[1], idx_rhs[1];
    if (use_input) {
      idx_lhs[0] = util::ravel(coord, params.lhs_shape);
      idx_rhs[0] = util::ravel(coord, params.rhs_shape);
    }
    AType val, residual;
    REDUCER.SetInitValue(val, residual);
    const int u = 0;
    REDUCER.Reduce(val, FUNC, residual);
    REDUCER.Finalize(val, residual);
    if (req == OpReqType::kAddTo) {
      const auto temp = op::add(static_cast<typename OType::type>(val),
                                OType::from(small[index]));
      small[index] = OType::to(temp);
    } else {
      small[index] = OType::to(static_cast<typename OType::type>(val));
    }
  }
}
)code";

void RTCReduceM1Impl(Stream<gpu> *s, const TBlob &small, const TBlob &big,
                     const TBlob *lhs, const TBlob *rhs,
                     const ReduceImplConfig &config, const int ndim,
                     const std::string &common_code, int dev_id,
                     const bool use_index = false) {
  using namespace common::cuda::rtc;

  std::string code = common_code +
                     "using InputType0 = " +
                     common::mshadow_type_info(big.type_flag_).name +
                     ";\n"
                     "using InputType1 = " +
                     ((lhs != nullptr)
                     ? common::mshadow_type_info(lhs->type_flag_).name
                     : "float32") +
                     ";\n"
                     "using InputType2 = " +
                     ((rhs != nullptr)
                     ? common::mshadow_type_info(rhs->type_flag_).name
                     : "float32") +
                     ";\n"
                     "using OutputType0 = " +
                     common::mshadow_type_info(small.type_flag_).name +
                     ";\n";
  if (lhs != nullptr) {
    code += "const bool use_input = true;";
  } else {
    code += "const bool use_input = false;";
  }

  reduce_kernel_M1_params param {};
  for (int i = 0; i < ndim; ++i) {
    param.big_shape[i] = big.shape_[i];
    param.small_shape[i] = small.shape_[i];
    if (lhs != nullptr) {
      param.lhs_shape[i] = lhs->shape_[i];
      param.rhs_shape[i] = rhs->shape_[i];
    }
  }

  void *null_ptr = nullptr;
  std::vector<const void*> args;
  args.emplace_back(&config.N);
  args.emplace_back(&big.dptr_);
  if (lhs != nullptr) {
    args.emplace_back(&(lhs->dptr_));
    args.emplace_back(&(rhs->dptr_));
  } else {
    args.emplace_back(&(null_ptr));
    args.emplace_back(&(null_ptr));
  }
  args.emplace_back(&small.dptr_);
  args.emplace_back(&param);

  const auto &function_code = (lhs == nullptr)
                            ? (use_index ? reduce_function_index_code : reduce_function_code)
                            : reduce_function_use_input_code;
  auto reduce_kernel_M1_func = get_function(code + function_code,
                                            "reduce_kernel_M1",
                                            reduce_kernel_M1_code,
                                            dev_id);
  launch(reduce_kernel_M1_func, config.kernel_1.gridDim,
         config.kernel_1.blockDim,
         0, s, &args);
}

}  // namespace

void RTCReduce(const OpContext& ctx,
               const TBlob& small,
               const OpReqType req,
               const Tensor<gpu, 1, char>& workspace,
               const TBlob& big,
               const std::string& reducer,
               int ndim,
               const std::string& OP,
               const bool use_index) {
  using namespace mxnet::common::cuda::rtc;
  if (req == kNullOp) return;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  ReduceImplConfig config(small.shape_, big.shape_, nullptr, nullptr);
  std::string common_code = std::string("const OpReqType req = ") +
                            util::to_string(req) +
                            ";\n"
                            "#define OP op::" +
                            OP +
                            "\n"
                            "#define REDUCER " +
                            reducer +
                            "\n"
                            "const int ndim = " +
                            std::to_string(ndim) +
                            ";\n";
  if (config.M == 1) {
    RTCReduceM1Impl(s, small, big, nullptr, nullptr, config,
                    ndim, common_code, ctx.run_ctx.ctx.dev_id,
                    use_index);
  } else {
    RTCReduceImpl(s, small, req == kAddTo, big, workspace, config,
                  ndim, common_code, ctx.run_ctx.ctx.dev_id,
                  nullptr, nullptr, use_index);
  }
}

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
               const std::string& OP2) {
  using namespace mxnet::common::cuda::rtc;
  if (req == kNullOp) return;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  ReduceImplConfig config(small.shape_, big.shape_, &lhs.shape_, &rhs.shape_);
  std::string common_code = std::string("const OpReqType req = ") +
                            util::to_string(req) +
                            ";\n"
                            "#define OP1 op::" +
                            OP1 +
                            "\n"
                            "#define OP2 op::" +
                            OP2 +
                            "\n"
                            "#define REDUCER " +
                            reducer +
                            "\n"
                            "const int ndim = " +
                            std::to_string(ndim) +
                            ";\n";
  if (config.M == 1) {
    RTCReduceM1Impl(s, small, big, &lhs, &rhs, config, ndim, common_code, ctx.run_ctx.ctx.dev_id);
  } else {
    RTCReduceImpl(s, small, req == kAddTo, big, workspace, config,
                  ndim, common_code, ctx.run_ctx.ctx.dev_id, &lhs, &rhs);
  }
}

#endif  // MXNET_USE_CUDA

}  // namespace broadcast
}  // namespace op
}  // namespace mxnet
