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
)code";

const char reduce_function_use_input_code[] = R"code(
#define FUNC OP1(IType0::from(big[idx_big[u]]),     \
                 OP2(IType1::from(lhs[idx_lhs[u]]), \
                     IType2::from(rhs[idx_rhs[u]])))
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

__global__ void reduce_kernel(const int N, const int M, const bool addto,
                              const InputType0* __restrict big,
                              const InputType1* __restrict lhs,
                              const InputType2* __restrict rhs,
                              OutputType0 *small,
                              const reduce_kernel_params params,
                              const int Mnext) {
  extern __shared__ char shTileChar[];
  using IType0 = AccType<InputType0>;
  using IType1 = AccType<InputType1>;
  using IType2 = AccType<InputType2>;
  using OType = AccType<OutputType0>;
  using AType = typename IType0::type;
  AType* shTile = (AType*)(shTileChar);
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
      int idx = idx0 + tidx;
      index_t coord[ndim];
      util::unravel(idx, params.small_shape, coord);
      index_t idx_big0, idx_lhs0, idx_rhs0;
      idx_big0 = util::ravel(coord, params.big_shape);
      if (use_input) {
        idx_lhs0 = util::ravel(coord, params.lhs_shape0);
        idx_rhs0 = util::ravel(coord, params.rhs_shape0);
      }

      AType val, residual;
      REDUCER::SetInitValue(val, residual);
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
          typename OType::type tmp[UNROLL];
          #pragma unroll
          for (int u=0;u < UNROLL;u++) {
            if (k + u*by < Mend) {
              tmp[u] = FUNC;
            }
          }
          #pragma unroll
          for (int u=0;u < UNROLL;u++) {
            if (k + u*by < Mend) REDUCER::Reduce(val, tmp[u], residual);
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
          REDUCER::SetInitValue(tmp, tmp_residual);
          if (tidy + t < by) {
            tmp = shTile[(it0 + t*fbx) * 2];
            tmp_residual = shTile[(it0 + t*fbx) * 2 + 1];
          }
          __syncthreads();
          REDUCER::Merge(shTile[it0 * 2], shTile[it0 * 2 + 1], tmp, tmp_residual);
          __syncthreads();
        }
        if (idx < N && tidy == 0) {
          if (addto) {
            small[idx + m0 * N] = OType::to(op::add(OType::from(small[idx + m0 * N]),
                                                    shTile[tidx * 2]));
          } else {
            small[idx + m0 * N] = OType::to(shTile[tidx * 2]);
          }
        }
      } else {
        if (idx < N) {
          if (addto) {
            small[idx + m0 * N] = OType::to(op::add(OType::from(small[idx + m0 * N]),
                                                    val));
          } else {
            small[idx + m0 * N] = OType::to(val);
          }
        }
      }
    }
  }
}
)code";

const char reduce_lines_kernel_code[] = R"code(
__global__ void reduce_lines_kernel(const index_t N, const index_t M,
                                    const index_t small_in_stride,
                                    const OutputType0* __restrict small_in,
                                    OutputType0 *small_out) {
  using OType = AccType<OutputType0>;
  for (index_t idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    typename OType::type val, residual;
    REDUCER::SetInitValue(val, residual);
    for (int k = 0; k < M; k++) {
      REDUCER::Reduce(val,
        OType::from(reinterpret_cast<const OutputType0*>(small_in)[idx + k*small_in_stride]),
        residual);
    }

    if (idx < N) {
      if (req == OpReqType::kAddTo) {
        small_out[idx] = OType::to(op::add(OType::from(small_out[idx]), val));
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
                const TBlob *lhs = nullptr, const TBlob *rhs = nullptr) {
  using namespace common::cuda::rtc;
  void* small_dptr = small.dptr_;
  bool first_kernel_addto = addto;
  if (config.Mnext > 1) {
    // small_dptr[] is N*Mnext*sizeof(DType) bytes
    small_dptr = workspace.dptr_;
    first_kernel_addto = false;
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
  args.emplace_back(&first_kernel_addto);
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
                            ? reduce_function_code
                            : reduce_function_use_input_code;
  auto reduce_kernel_func = get_function(code + function_code,
                                         "reduce_kernel",
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

    auto reduce_lines_kernel_func = get_function(code,
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
  for (int idx = threadIdx.x + blockIdx.x*blockDim.x; idx < N; idx += blockDim.x*gridDim.x) {
    index_t coord[ndim];
    util::unravel(idx, params.small_shape, coord);
    const index_t idx_big = util::ravel(coord, params.big_shape);
    const index_t idx_lhs = util::ravel(coord, params.lhs_shape);
    const index_t idx_rhs = util::ravel(coord, params.rhs_shape);
    const typename OType::type val =
      OP1(IType0::from(big[idx_big]), OP2(IType1::from(lhs[idx_lhs]),
                                          IType2::from(rhs[idx_rhs])));
    if (req == OpReqType::kAddTo) {
      const auto temp = op::add(val, OType::from(small[idx]));
      small[idx] = OType::to(temp);
    } else {
      small[idx] = OType::to(val);
    }
  }
}
)code";

void RTCReduceM1Impl(Stream<gpu> *s, const TBlob &small, const TBlob &big,
                     const TBlob &lhs, const TBlob &rhs,
                     const ReduceImplConfig &config, const int ndim,
                     const std::string &common_code, int dev_id) {
  using namespace common::cuda::rtc;

  std::string code = common_code +
                     "using InputType0 = " +
                     common::mshadow_type_info(big.type_flag_).name +
                     ";\n"
                     "using InputType1 = " +
                     common::mshadow_type_info(lhs.type_flag_).name +
                     ";\n"
                     "using InputType2 = " +
                     common::mshadow_type_info(rhs.type_flag_).name +
                     ";\n"
                     "using OutputType0 = " +
                     common::mshadow_type_info(small.type_flag_).name +
                     ";\n";
  reduce_kernel_M1_params param {};
  for (int i = 0; i < ndim; ++i) {
    param.big_shape[i] = big.shape_[i];
    param.small_shape[i] = small.shape_[i];
    param.lhs_shape[i] = lhs.shape_[i];
    param.rhs_shape[i] = rhs.shape_[i];
  }

  std::vector<const void*> args;
  args.emplace_back(&config.N);
  args.emplace_back(&big.dptr_);
  args.emplace_back(&lhs.dptr_);
  args.emplace_back(&rhs.dptr_);
  args.emplace_back(&small.dptr_);
  args.emplace_back(&param);

  auto reduce_kernel_M1_func = get_function(code,
                                            "reduce_kernel_M1",
                                            reduce_kernel_M1_code,
                                            dev_id);
  launch(reduce_kernel_M1_func, config.kernel_1.gridDim,
         config.kernel_1.blockDim,
         0, s, &args);
}

}  // namespace

void RTCReduce(const NodeAttrs& attrs,
               const OpContext& ctx,
               const TBlob& small,
               const OpReqType req,
               const Tensor<gpu, 1, char>& workspace,
               const TBlob& big,
               const std::string& reducer,
               int ndim,
               const std::string& OP) {
  using namespace mxnet::common::cuda::rtc;
  if (req == kNullOp) return;
  Stream<gpu> *s = ctx.get_stream<gpu>();
  size_t type_size = common::mshadow_type_info(small.type_flag_).size;
  if (small.type_flag_ == mshadow::kFloat16) {
    type_size = sizeof(float);
  }
  ReduceImplConfig config(small.shape_, big.shape_, nullptr, nullptr, type_size);
  if (config.M == 1) {
    // With M == 1 result is just (possibly reshaped) OP(big)
    UnaryRTCCompute {OP} (attrs, ctx, {big}, {req}, {small});
  } else {
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
    RTCReduceImpl(s, small, req == kAddTo, big, workspace, config,
                  ndim, common_code, ctx.run_ctx.ctx.dev_id);
  }
}

void RTCReduce(const NodeAttrs& attrs,
               const OpContext& ctx,
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
  size_t type_size = common::mshadow_type_info(small.type_flag_).size;
  if (small.type_flag_ == mshadow::kFloat16) {
    type_size = sizeof(float);
  }
  ReduceImplConfig config(small.shape_, big.shape_, &lhs.shape_, &rhs.shape_, type_size);
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
    RTCReduceM1Impl(s, small, big, lhs, rhs, config, ndim, common_code, ctx.run_ctx.ctx.dev_id);
  } else {
    RTCReduceImpl(s, small, req == kAddTo, big, workspace, config,
                  ndim, common_code, ctx.run_ctx.ctx.dev_id, &lhs, &rhs);
  }
}

#endif  // MXNET_USE_CUDA

}  // namespace broadcast
}  // namespace op
}  // namespace mxnet