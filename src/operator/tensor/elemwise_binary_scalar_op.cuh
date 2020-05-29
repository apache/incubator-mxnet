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
 *  Copyright (c) 2020 by Contributors
 * \file elemwise_binary_scalar_op.cuh
 * \brief GPU helpers for binary elementwise operators with scalar
 */

#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_CUH_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_CUH_

#include <cuda_runtime.h>
#include "../operator_common.h"
#include "../../common/cuda/vectorization.cuh"
#include "elemwise_unary_op.h"

#include <vector>

#if MXNET_USE_CUDA

namespace mxnet {
namespace op {

namespace binary_scalar {

using common::cuda::VectorizedKernelLauncher;
using common::cuda::VectorizedLoader;
using common::cuda::VectorizedStorer;

template <typename DType, int NumInputs, int NumOutputs>
struct VectorizedKernelParams {
  const DType* inputs[NumInputs];
  DType* outputs[NumOutputs];
  DType scalar;
};

template <bool aligned, typename DType, typename LType, typename OP, int req>
__global__ void VectorizedBinaryScalarKernelBwd(const VectorizedKernelParams<DType, 2, 1> params,
                                                const index_t N) {
  VectorizedLoader<DType, LType, aligned> ograd_loader(params.inputs[0], N);
  VectorizedLoader<DType, LType, aligned> input_loader(params.inputs[1], N);
  VectorizedStorer<DType, LType, aligned> storer(params.outputs[0], N);

  const index_t M = ograd_loader.num_aligned_elements();

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    ograd_loader.load(tid, N);
    input_loader.load(tid, N);
    if (req == kAddTo) {
      storer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < ograd_loader.nvec(); ++i) {
      DType ograd = ograd_loader.separate()[i];
      DType temp = ograd * OP::Map(input_loader.separate()[i],
                                   params.scalar);

      if (req == kAddTo) {
        storer.separate()[i] += temp;
      } else {
        storer.separate()[i] = temp;
      }
    }
    storer.store(tid, N);
  }
}

template <typename DType, typename OP, int req>
class VectorizedBinaryScalarBwd {
 public:
  using ParamType = VectorizedKernelParams<DType, 2, 1>;

  template <bool aligned, typename LType>
  static void Launch(const index_t blocks, const index_t threads,
                     cudaStream_t stream,
                     const ParamType params, const index_t lead_dim,
                     const index_t /* other_dim */) {
    VectorizedBinaryScalarKernelBwd<aligned, DType, LType, OP, req>
      <<<blocks, threads, 0, stream>>>(params, lead_dim);
  }
};

}  // namespace binary_scalar

struct binary_scalar_kernel_params {
  const void *inputs[1];
  void *outputs[1];
  double scalar;
};

const char binary_scalar_kernel_fwd[] = R"code(

struct binary_scalar_kernel_params {
  const void *inputs[1];
  void *outputs[1];
  double scalar;
};

__global__ void binary_scalar_kernel(const binary_scalar_kernel_params params,
                                     const index_t lead_dim,
                                     const index_t other_dim,
                                     const index_t N,
                                     const index_t num_aligned_elements) {
  using namespace vector;
  VectorizedLoader<InputType0, nvec, aligned> loader(
    reinterpret_cast<const InputType0*>(params.inputs[0]), N);
  VectorizedStorer<OutputType0, nvec, aligned> storer(
    reinterpret_cast<OutputType0*>(params.outputs[0]), N);

  using IType = AccType<InputType0>;
  using OType = AccType<OutputType0>;

  const index_t M = num_aligned_elements;

  for (index_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    loader.load(tid, N);
    if (req == OpReqType::kAddTo) {
      storer.load(tid, N);
    }
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const auto input = IType::from(loader.separate()[i]);
      // enables returning different type
      const auto temp = OP(input, static_cast<typename IType::type>(params.scalar));

      if (req == OpReqType::kAddTo) {
        // temp2 may have a wider type than either temp
        // or OType
        const auto temp2 = op::add(temp, OType::from(storer.separate()[i]));
        storer.separate()[i] = OType::to(temp2);
      } else {
        storer.separate()[i] = OType::to(temp);
      }
    }
    storer.store(tid, N);
  }
}

)code";

struct BinaryScalarRTCCompute {
  std::string OP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
    using namespace mxnet::common::cuda::rtc;
    if (req[0] == kNullOp) return;
    mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
    const double alpha = param.scalar;

    const std::string code = std::string("const OpReqType req = ") +
                             util::to_string(req[0]) +
                             ";\n" +
                             "#define OP op::" +
                             OP +
                             "\n" +
                             binary_scalar_kernel_fwd;
    const int nvec = outputs[0].type_flag_ == mshadow::kFloat64 ? 2 : 4;

    const index_t size = outputs[0].Size();
    binary_scalar_kernel_params params = { {inputs[0].dptr_},
                                           {outputs[0].dptr_},
                                           alpha };

    VectorizedKernelRTCLauncher(code, "binary_scalar_kernel", nvec,
                                size, 1, s, params,
                                inputs, outputs,
                                ctx.run_ctx.get_ctx().dev_id);
  }

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs) {
    if (req[0] == kNullOp) {
      return;
    }
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    InitStorageGeometry<1, 1>(attrs, inputs, outputs);
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage)
      << "This function works only for sparse types.";
    CHECK_EQ(inputs[0].storage_type(), outputs[0].storage_type())
      << "The storage type of both inputs and outputs needs to be the same.";
    AllocateGeometry(&outputs[0], req[0], &inputs[0]);
    CopyGeometryBlobs<gpu>(ctx.get_stream<gpu>(), &outputs[0], req[0], inputs[0]);
    outputs[0].CheckAndAllocData(inputs[0].storage_shape());
    if (inputs[0].storage_shape().Size()) {
      std::vector<TBlob> in_blobs, out_blobs;
      in_blobs.reserve(inputs.size());
      out_blobs.reserve(outputs.size());
      for (auto &input : inputs) {
        in_blobs.emplace_back(input.data());
      }
      for (auto &output : outputs) {
        out_blobs.emplace_back(output.data());
      }
      this->operator()(attrs, ctx, in_blobs, req, out_blobs);
    }
  }
};

template <typename OP>
void BinaryScalarOp::Backward_(const nnvm::NodeAttrs &attrs,
                               mshadow::Stream<gpu>* s,
                               const std::vector<TBlob> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<TBlob> &outputs) {
  using namespace binary_scalar;
  if (req[0] == kNullOp) return;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  const NumpyBinaryScalarParam& param = nnvm::get<NumpyBinaryScalarParam>(attrs.parsed);
  const double alpha = param.scalar;
  MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      using LType = uint4;
      using Kernel = VectorizedBinaryScalarBwd<DType, OP, Req>;

      const index_t size = outputs[0].Size();
      typename Kernel::ParamType params;
      params.inputs[0] = inputs[0].dptr<DType>();
      params.inputs[1] = inputs[1].dptr<DType>();
      params.outputs[0] = outputs[0].dptr<DType>();
      params.scalar = (DType)alpha;

      VectorizedKernelLauncher<DType, LType, Kernel>(size, 1, s, params);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_CUDA
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_CUH_
