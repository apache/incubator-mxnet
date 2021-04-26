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

#include <string>

#if MXNET_USE_CUDA
#include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA

#include "broadcast_reduce-inl.h"
#include "elemwise_binary_broadcast_op.h"

#if MXNET_USE_CUDA
#include "../../common/cuda/rtc/vectorization-inl.h"
#include "../../common/cuda/rtc.h"
#endif  // MXNET_USE_CUDA

namespace mxnet {
namespace op {

#if MXNET_USE_CUDA

struct binary_broadcast_params {
  const void* inputs[2];
  void* outputs[1];
  index_t stride[2][broadcast::MAX_DIM];
  index_t oshape[broadcast::MAX_DIM];
  index_t size[2];
};

const char broadcast_kernel_fwd[] = R"code(
struct binary_broadcast_params {
  const void* inputs[2];
  void* outputs[1];
  index_t stride[2][util::MAX_DIM];
  index_t oshape[util::MAX_DIM];
  index_t size[2];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void binary_broadcast_kernel(
    const binary_broadcast_params param,
    const index_t lead_dim,
    const index_t other_dim,
    const index_t N,
    const index_t num_aligned_elements) {
  using namespace vector;
  const index_t M = num_aligned_elements * other_dim;

  VectorizedLoader<InputType0, nvec, aligned> lloader(
    reinterpret_cast<const InputType0*>(param.inputs[0]), param.size[0]);
  VectorizedLoader<InputType1, nvec, aligned> rloader(
    reinterpret_cast<const InputType1*>(param.inputs[1]), param.size[1]);

  using IType0 = AccType<InputType0>;
  using IType1 = AccType<InputType1>;
  using OType = AccType<OutputType0>;


  for (index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < M;
       idx += gridDim.x * blockDim.x) {
    OutputType0 * current_output_pointer;
    index_t output_size;
    index_t output_idx;
    if (aligned) {
      // Simplified case
      index_t lindex, rindex;
      util::unravel_dot<ndim>(idx * nvec, param.oshape,
                              param.stride[0], param.stride[1],
                              &lindex, &rindex);
      lloader.load(lindex / nvec, param.size[0]);
      rloader.load(rindex / nvec, param.size[1]);
      current_output_pointer = reinterpret_cast<OutputType0*>(param.outputs[0]);
      output_size = N;
      output_idx = idx;
    } else {
      const index_t row = idx / num_aligned_elements;
      const index_t lead_dim_idx = idx - row * num_aligned_elements;

      index_t lindex, rindex;
      const index_t original_idx = max(lead_dim_idx * nvec - lloader.alignment(),
                                       static_cast<index_t>(0)) +
                                   row * lead_dim;
      util::unravel_dot<ndim>(original_idx, param.oshape,
                              param.stride[0], param.stride[1],
                              &lindex, &rindex);
      lloader.load((lindex + lloader.alignment()) / nvec, param.size[0]);
      rloader.load((rindex + lloader.alignment()) / nvec, param.size[1]);
      current_output_pointer = reinterpret_cast<OutputType0*>(param.outputs[0]) + row * lead_dim;
      output_size = lead_dim;
      output_idx = lead_dim_idx;
    }
    VectorizedStorer<OutputType0, nvec, aligned> storer(current_output_pointer, output_size);

    if (req == OpReqType::kAddTo) {
      storer.load(output_idx, output_size);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const auto temp = OP(IType0::from(lloader.separate()[i]),
                           IType1::from(rloader.separate()[i]));

      if (req == OpReqType::kAddTo) {
        const auto temp2 = op::add(temp, OType::from(storer.separate()[i]));
        storer.separate()[i] = OType::to(temp2);
      } else {
        storer.separate()[i] = OType::to(temp);
      }
    }
    storer.store(output_idx, output_size);
  }
}
)code";

const char single_side_broadcast_kernel_fwd[] = R"code(
struct binary_broadcast_params {
  const void* inputs[2];
  void* outputs[1];
  index_t stride[2][util::MAX_DIM];
  index_t oshape[util::MAX_DIM];
  index_t size[2];
};

__launch_bounds__(kRTCMaxThreadsPerBlock)
__global__ void single_side_binary_broadcast_kernel(
    const binary_broadcast_params param,
    const index_t lead_dim,
    const index_t other_dim,
    const index_t N,
    const index_t num_aligned_elements) {
  using namespace vector;
  const index_t M = num_aligned_elements * other_dim;
  constexpr int other_side = 1 - side;

  VectorizedLoader<DType, nvec, aligned> lloader(
    reinterpret_cast<const DType*>(param.inputs[side]), param.size[side]);

  using IType = AccType<DType>;
  using IType2 = AccType<DType2>;
  using OType = AccType<OutputType0>;


  for (index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < M;
       idx += gridDim.x * blockDim.x) {
    index_t original_idx;
    OutputType0 * current_output_pointer;
    index_t output_size;
    index_t output_idx;
    if (aligned) {
      // Simplified case
      original_idx = idx * nvec;
      const index_t lindex = util::unravel_dot<ndim>(original_idx, param.oshape,
                                                     param.stride[side]);
      lloader.load(lindex / nvec, param.size[side]);
      current_output_pointer = reinterpret_cast<OutputType0*>(param.outputs[0]);
      output_size = N;
      output_idx = idx;
    } else {
      const index_t row = idx / num_aligned_elements;
      const index_t lead_dim_idx = idx - row * num_aligned_elements;
      original_idx = lead_dim_idx * nvec -
                     lloader.alignment() + row * lead_dim;
      const index_t original_idx_clamped = max(lead_dim_idx * nvec - lloader.alignment(),
                                               static_cast<index_t>(0)) +
                                           row * lead_dim;
      const index_t lindex = util::unravel_dot<ndim>(original_idx_clamped, param.oshape,
                                                     param.stride[side]);
      lloader.load((lindex + lloader.alignment()) / nvec, param.size[side]);
      current_output_pointer = reinterpret_cast<OutputType0*>(param.outputs[0]) + row * lead_dim;
      output_size = lead_dim;
      output_idx = lead_dim_idx;
    }
    VectorizedStorer<OutputType0, nvec, aligned> storer(current_output_pointer, output_size);

    if (req == OpReqType::kAddTo) {
      storer.load(output_idx, output_size);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const index_t rindex = min(max(util::unravel_dot<ndim>(original_idx + i,
                                                             param.oshape,
                                                             param.stride[other_side]),
                                     static_cast<index_t>(0)),
                                 param.size[other_side] - 1);
      const auto rinput = IType2::from(
                            reinterpret_cast<const DType2*>(param.inputs[other_side])
                            [rindex]);

      typename OType::type temp;
      if (side == 0) {
        // Left side is vectorized
        temp = OP(IType::from(lloader.separate()[i]),
                  rinput);
      } else {
        // Right side is vectorized
        temp = OP(rinput,
                  IType::from(lloader.separate()[i]));
      }

      if (req == OpReqType::kAddTo) {
        const auto temp2 = op::add(temp, OType::from(storer.separate()[i]));
        storer.separate()[i] = OType::to(temp2);
      } else {
        storer.separate()[i] = OType::to(temp);
      }
    }
    storer.store(output_idx, output_size);
  }
}
)code";
namespace {

std::vector<index_t> calc_stride(const mxnet::TShape& shape, int ndim) {
  CHECK_EQ(ndim, shape.ndim());
  std::vector<index_t> stride(ndim);
  index_t cumprod = 1;
  for (int i = shape.ndim() - 1; i >= 0; --i) {
    stride[i] = (shape[i] > 1) ? cumprod : 0;
    cumprod *= shape[i];
  }
  return stride;
}

}  // namespace

void BinaryBroadcastRTCCompute::operator()(const nnvm::NodeAttrs& attrs,
                                           const OpContext& ctx,
                                           const std::vector<TBlob>& inputs,
                                           const std::vector<OpReqType>& req,
                                           const std::vector<TBlob>& outputs) {
  using namespace mxnet::common::cuda::rtc;
  if (outputs[0].shape_.Size() == 0U) return;
  if (req[0] == kNullOp) return;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(inputs[0].shape_, inputs[1].shape_, outputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  // Pad the ndim
  BROADCAST_NDIM_SWITCH(ndim, NDim, {
      if (ndim != 0) {
        ndim = NDim;
      }
  });

  if (!ndim) {
    ElemwiseBinaryRTCCompute {OP}(attrs, ctx, inputs, req, outputs);
  } else {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    const TBlob& lhs = inputs[0].reshape(new_lshape);
    const TBlob& rhs = inputs[1].reshape(new_rshape);
    const TBlob& output = outputs[0].reshape(new_oshape);

    const auto& lstride = calc_stride(lhs.shape_, ndim);
    const auto& rstride = calc_stride(rhs.shape_, ndim);

    size_t output_type_size = common::mshadow_type_info(outputs[0].type_flag_).size;
    const int nvec = output_type_size <= sizeof(uint64_t)
                       ? (sizeof(uint64_t) / output_type_size)
                       : 1;
    binary_broadcast_params params{};
    params.inputs[0] = lhs.dptr_;
    params.inputs[1] = rhs.dptr_;
    params.outputs[0] = output.dptr_;
    for (int i = 0; i < ndim; ++i) {
      params.stride[0][i] = lstride[i];
      params.stride[1][i] = rstride[i];
      params.oshape[i] = new_oshape[i];
    }
    params.size[0] = lhs.shape_.Size();
    params.size[1] = rhs.shape_.Size();

    index_t lead_dim = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      /* Find the first non-1 dimension
         to check the alignment
      */
      if (params.oshape[i] != 1) {
        lead_dim = params.oshape[i];
        break;
      }
    }
    const index_t other_dim = output.shape_.Size() / lead_dim;

    int first_different = -1;
    int common_shape = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      if (params.stride[0][i] == params.stride[1][i]) {
        common_shape *= params.oshape[i];
      } else {
        first_different = i;
        break;
      }
    }

    int lead_input_num = 0;
    std::string code = std::string("const OpReqType req = ") +
                       util::to_string(req[0]) +
                       ";\n"
                       "#define OP op::" +
                       OP +
                       "\n"
                       "const int ndim = " +
                       std::to_string(ndim) +
                       ";\n";
    if (common_shape != 1) {
      VectorizedKernelRTCLauncher(code, "binary_broadcast_kernel",
                                  broadcast_kernel_fwd, nvec,
                                  lead_dim, other_dim, s, params,
                                  inputs, outputs,
                                  ctx.run_ctx.get_ctx().dev_id,
                                  lead_input_num);
    } else {
      if (params.stride[0][first_different] == 0) {
        lead_input_num = 1;
        code += "const int side = 1;\n"
                "using DType = InputType1;\n"
                "using DType2 = InputType0;\n";
      } else {
        code += "const int side = 0;\n"
                "using DType = InputType0;\n"
                "using DType2 = InputType1;\n";
      }
      VectorizedKernelRTCLauncher(code, "single_side_binary_broadcast_kernel",
                                  single_side_broadcast_kernel_fwd, nvec,
                                  lead_dim, other_dim, s, params,
                                  inputs, outputs,
                                  ctx.run_ctx.get_ctx().dev_id,
                                  lead_input_num);
    }
  }
}

void BinaryBroadcastRTCBackwardUseNone::operator()(const nnvm::NodeAttrs& attrs,
                                                   const OpContext& ctx,
                                                   const std::vector<TBlob>& inputs,
                                                   const std::vector<OpReqType>& req,
                                                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  int ndim = BinaryBroadcastShapeCompact(outputs[0].shape_, outputs[1].shape_, inputs[0].shape_,
                                         &new_lshape, &new_rshape, &new_oshape);
  if (!ndim) {
    ElemwiseBinaryRTCBwdUseNone {LOP, ROP}(attrs, ctx, inputs, req, outputs);
  } else {
    Stream<gpu> *s = ctx.get_stream<gpu>();
    const TBlob lhs = outputs[0].reshape(new_lshape);
    const TBlob rhs = outputs[1].reshape(new_rshape);
    const TBlob out = inputs[0].reshape(new_oshape);
    BROADCAST_NDIM_SWITCH(ndim, NDim, {
      // Request temporary storage
      size_t workspace_size = new_oshape.Size();
      Tensor<gpu, 1, char> workspace =
          ctx.requested[0].get_space_typed<gpu, 1, char>(
              Shape1(workspace_size * sizeof(index_t)), s);
      if (out.shape_.Size() != 0) {
        broadcast::RTCReduce(ctx, lhs, req[0],
                             workspace, out,
                             "red::sum{}", NDim, LOP);
        broadcast::RTCReduce(ctx, rhs, req[1],
                             workspace, out,
                             "red::sum{}", NDim, ROP);
      } else {
        using namespace common::cuda::rtc::util;
        if (lhs.shape_.Size() != 0) {
          cudaMemsetAsync(lhs.dptr_, 0,
                          lhs.shape_.Size() * common::mshadow_type_info(lhs.type_flag_).size,
                          Stream<gpu>::GetStream(s));
        }
        if (rhs.shape_.Size() != 0) {
          cudaMemsetAsync(rhs.dptr_, 0,
                          rhs.shape_.Size() * common::mshadow_type_info(rhs.type_flag_).size,
                          Stream<gpu>::GetStream(s));
        }
      }
    });
  }
}

void BinaryBroadcastRTCBackwardUseIn::operator()(const nnvm::NodeAttrs& attrs,
                                                 const OpContext& ctx,
                                                 const std::vector<TBlob>& inputs,
                                                 const std::vector<OpReqType>& req,
                                                 const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  // skip kernel launch for zero-size tensors
  if (inputs[0].shape_.Size() == 0U) {
    return;
  }
  mxnet::TShape new_lshape, new_rshape, new_oshape;
  const bool need_bc = BinaryBroadcastShapeCompact(outputs[0].shape_,
                                                   outputs[1].shape_, inputs[0].shape_,
                                                   &new_lshape, &new_rshape, &new_oshape) != 0;
  if (!need_bc) {
    ElemwiseBinaryRTCBwdUseIn {LOP, ROP}(attrs, ctx, inputs, req, outputs);
  } else {
    BROADCAST_NDIM_SWITCH(new_oshape.ndim(), NDim, {
        using namespace mshadow;
        Stream<gpu> *s = ctx.get_stream<gpu>();
        const TBlob lgrad = outputs[0].reshape(new_lshape);
        const TBlob rgrad = outputs[1].reshape(new_rshape);
        const TBlob ograd = inputs[0].reshape(new_oshape);
        const TBlob lhs = inputs[1].reshape(new_lshape);
        const TBlob rhs = inputs[2].reshape(new_rshape);
        size_t workspace_size_l = broadcast::ReduceWorkspaceSize(
            s, lgrad.shape_, req[0], ograd.shape_, lhs.shape_,
            rhs.shape_);
        size_t workspace_size_r = broadcast::ReduceWorkspaceSize(
            s, rgrad.shape_, req[1], ograd.shape_, lhs.shape_,
            rhs.shape_);
        size_t workspace_size = std::max(workspace_size_l, workspace_size_r);
        Tensor<gpu, 1, char> workspace =
            ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), s);
        if (req[0] != kNullOp) {
          broadcast::RTCReduce(ctx, lgrad, req[0], workspace,
                               ograd, lhs, rhs, "red::sum{}", NDim,
                               "mul", LOP);
        }
        if (req[1] != kNullOp) {
          broadcast::RTCReduce(ctx, rgrad, req[1], workspace,
                               ograd, lhs, rhs, "red::sum{}", NDim,
                               "mul", ROP);
        }
    });
  }
}

#endif  // MXNET_USE_CUDA

}  // namespace op
}  // namespace mxnet
