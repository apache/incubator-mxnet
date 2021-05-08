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
 *  Copyright (c) 2019 by Contributors
 * \file np_bicount_op.cu
 * \brief numpy compatible bincount operator GPU registration
 */

#include "./np_bincount_op-inl.h"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include "../tensor/util/tensor_util-inl.cuh"
#include "../tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

struct BincountFusedKernel {
  template<typename DType, typename OType>
  static MSHADOW_XINLINE void Map(int i, const DType* data, OType* out) {
    int idx = data[i];
    atomicAdd(&out[idx], 1);
  }

  template<typename DType, typename OType>
  static MSHADOW_XINLINE void Map(int i, const DType* data, const OType* weights,
                                  OType* out) {
    int idx = data[i];
    atomicAdd(&out[idx], weights[i]);
  }
};

struct is_valid_check {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, char* invalid_ptr, const DType* data) {
    if (data[i] < 0) *invalid_ptr = 1;
  }
};

template<typename DType>
bool CheckInvalidInput(mshadow::Stream<gpu> *s, const DType *data, const size_t& data_size,
                       char* is_valid_ptr) {
  using namespace mxnet_op;
  int32_t is_valid = 0;
  Kernel<set_zero, gpu>::Launch(s, 1, is_valid_ptr);
  Kernel<is_valid_check, gpu>::Launch(s, data_size, is_valid_ptr, data);
  CUDA_CALL(cudaMemcpyAsync(&is_valid, is_valid_ptr, sizeof(char),
                            cudaMemcpyDeviceToHost, mshadow::Stream<gpu>::GetStream(s)));
  CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
  return is_valid == 0;
}

template<>
void NumpyBincountForwardImpl<gpu>(const OpContext &ctx,
                                   const NDArray &data,
                                   const NDArray &weights,
                                   const NDArray &out,
                                   const size_t &data_n,
                                   const int &minlength) {
  using namespace mxnet_op;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();

  MXNET_NO_FLOAT16_TYPE_SWITCH(data.dtype(), DType, {
    DType* h_ptr;
    DType* d_ptr;
    int bin = minlength;
    d_ptr = data.data().dptr<DType>();
    Tensor<gpu, 1, char> workspace = ctx.requested[0]
            .get_space_typed<gpu, 1, char>(Shape1(1), s);
    char* is_valid_ptr = reinterpret_cast<char*>(workspace.dptr_);
    bool is_valid = CheckInvalidInput(s, d_ptr, data_n, is_valid_ptr);
    CHECK(is_valid) << "Input should be nonnegative number";   // check invalid input

    h_ptr = reinterpret_cast<DType*>(malloc(data_n*sizeof(DType)));
    CUDA_CALL(cudaMemcpyAsync(h_ptr, d_ptr, data_n*sizeof(DType), cudaMemcpyDeviceToHost,
                              mshadow::Stream<gpu>::GetStream(s)));
    CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
    for (size_t i = 0; i < data_n; i++) {
      if (h_ptr[i] + 1 > bin) bin = h_ptr[i] + 1;
    }
    free(h_ptr);
    mxnet::TShape s(1, bin);
    const_cast<NDArray &>(out).Init(s);  // set the output shape forcefully
  });

  MXNET_NO_INT16_TYPE_SWITCH(data.dtype(), DType, {
      MXNET_NO_INT16_TYPE_SWITCH(weights.dtype(), OType, {
        size_t out_size = out.shape().Size();
        Kernel<set_zero, gpu>::Launch(s, out_size, out.data().dptr<OType>());
        Kernel<BincountFusedKernel, gpu>::Launch(
          s, data_n, data.data().dptr<DType>(), weights.data().dptr<OType>(),
          out.data().dptr<OType>());
      });
    });
}

template<>
void NumpyBincountForwardImpl<gpu>(const OpContext &ctx,
                                   const NDArray &data,
                                   const NDArray &out,
                                   const size_t &data_n,
                                   const int &minlength) {
  using namespace mxnet_op;
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();

  MXNET_NO_FLOAT16_TYPE_SWITCH(data.dtype(), DType, {
    DType* h_ptr;
    DType* d_ptr;
    int bin = minlength;
    d_ptr = data.data().dptr<DType>();
    Tensor<gpu, 1, char> workspace = ctx.requested[0]
            .get_space_typed<gpu, 1, char>(Shape1(1), s);
    char* is_valid_ptr = reinterpret_cast<char*>(workspace.dptr_);
    bool is_valid = CheckInvalidInput(s, d_ptr, data_n, is_valid_ptr);
    CHECK(is_valid) << "Input should be nonnegative number";   // check invalid input

    h_ptr = reinterpret_cast<DType*>(malloc(data_n*sizeof(DType)));
    CUDA_CALL(cudaMemcpyAsync(h_ptr, d_ptr, data_n*sizeof(DType), cudaMemcpyDeviceToHost,
                              mshadow::Stream<gpu>::GetStream(s)));
    CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));
    for (size_t i = 0; i < data_n; i++) {
      if (h_ptr[i] + 1 > bin) bin = h_ptr[i] + 1;
    }
    free(h_ptr);
    mxnet::TShape s(1, bin);
    const_cast<NDArray &>(out).Init(s);  // set the output shape forcefully
  });

  MXNET_NO_INT16_TYPE_SWITCH(data.dtype(), DType, {
    MXNET_NO_INT16_TYPE_SWITCH(out.dtype(), OType, {
      size_t out_size = out.shape().Size();
      Kernel<set_zero, gpu>::Launch(s, out_size, out.data().dptr<OType>());
      Kernel<BincountFusedKernel, gpu>::Launch(
        s, data_n, data.data().dptr<DType>(), out.data().dptr<OType>());
    });
  });
}

NNVM_REGISTER_OP(_npi_bincount)
.set_attr<FComputeEx>("FComputeEx<gpu>", NumpyBincountForward<gpu>);

}  // namespace op
}  // namespace mxnet
