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
 *  Copyright (c) 2016 by Contributors
 * \file optimizer_op.cu
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#include "./optimizer_op-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

template<int req>
struct SGDMomStdDnsRspDnsKernel<req, gpu> {
  template<typename DType, typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i, index_t row_length, DType* out_data,
    DType* mom_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const RType* prefix_sum, const DType clip_gradient,
    const DType momentum, const DType lr, const DType wd, const DType rescale_grad) {
    using nnvm::dim_t;
    const dim_t row_id = i / row_length;
    const dim_t col_id = i % row_length;
    const dim_t nnr = prefix_sum[row_id];
    const bool non_zero = (row_id == 0) ? prefix_sum[0] > 0
                                        : nnr > prefix_sum[row_id - 1];
    const RType grad_i = (nnr - 1) * row_length + col_id;
    const DType grad = non_zero ? grad_data[grad_i]
                                : static_cast<DType>(0);
    DType grad_rescaled = rescale_grad * grad;
    if (clip_gradient >= 0.0f) {
      grad_rescaled = mshadow_op::clip::Map(grad_rescaled, clip_gradient);
    }
    grad_rescaled += wd * weight_data[i];
    mom_data[i] *= momentum;
    mom_data[i] -= grad_rescaled;
    KERNEL_ASSIGN(out_data[i], req, weight_data[i] + lr * mom_data[i]);
  }
};

template<>
void SGDMomStdUpdateDnsRspDnsImpl<gpu>(const SGDMomParam& param,
                                       const OpContext& ctx,
                                       const TBlob& weight,
                                       const NDArray& grad,
                                       const TBlob& mom,
                                       const OpReqType& req,
                                       TBlob *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  using namespace mshadow;
  Stream<gpu>* s = ctx.get_stream<gpu>();
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteInplace) << "kWriteInplace is expected for sparse sgd_mom_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(mom.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(kIdx), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        DType* weight_data = weight.dptr<DType>();
        IType* grad_idx = grad.aux_data(kIdx).dptr<IType>();
        DType* grad_val = grad.data().dptr<DType>();
        DType* mom_data = mom.dptr<DType>();
        DType* out_data = out->dptr<DType>();
        nnvm::dim_t num_rows = weight.shape_[0];
        nnvm::dim_t row_length = weight.shape_.ProdShape(1, weight.ndim());

        nnvm::dim_t* prefix_sum = nullptr;
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      Stream<gpu>::GetStream(s));
        Tensor<gpu, 1, char> workspace = ctx.requested[0]
          .get_space_typed<gpu, 1, char>(Shape1(num_rows * sizeof(nnvm::dim_t) +
                                         temp_storage_bytes), s);
        prefix_sum = reinterpret_cast<nnvm::dim_t*>(workspace.dptr_);
        d_temp_storage = workspace.dptr_ + num_rows*sizeof(nnvm::dim_t);
        // mark row flags
        Fill<false>(s, TBlob(prefix_sum, Shape1(num_rows), gpu::kDevMask), kWriteTo, 0);
        if (grad.storage_initialized()) {
          Kernel<MarkRowFlgKernel, gpu>::Launch(s, grad.aux_shape(kIdx)[0],
            prefix_sum, grad_idx);
          // calculate inclusive prefix sum
          cub::DeviceScan::InclusiveSum(d_temp_storage,
                                        temp_storage_bytes,
                                        prefix_sum,
                                        prefix_sum,
                                        num_rows,
                                        mshadow::Stream<gpu>::GetStream(s));
        }
        size_t num_threads = num_rows * row_length;
        Kernel<SGDMomStdDnsRspDnsKernel<req_type, gpu>, gpu>::Launch(s, num_threads, row_length,
          out_data, mom_data, weight_data, grad_idx, grad_val, prefix_sum,
          static_cast<DType>(param.clip_gradient), static_cast<DType>(param.momentum),
          static_cast<DType>(param.lr), static_cast<DType>(param.wd),
          static_cast<DType>(param.rescale_grad));
      });
    });
  });
}

template<int req>
struct AdamStdDnsRspDnsKernel<req, gpu> {
  template<typename DType, typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i, const nnvm::dim_t row_length, DType* out_data,
    DType* mean_data, DType* var_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const RType* prefix_sum, const DType clip_gradient,
    const DType beta1, const DType beta2, const DType lr, const DType wd,
    const DType epsilon, const DType rescale_grad) {
    using namespace mshadow_op;
    using nnvm::dim_t;
    const dim_t row_id = i / row_length;
    const dim_t col_id = i % row_length;
    const bool non_zero = (row_id == 0) ? prefix_sum[0] > 0
                          : prefix_sum[row_id] > prefix_sum[row_id - 1];
    const RType grad_offset = (prefix_sum[row_id] - 1) * row_length + col_id;
    DType grad_rescaled = non_zero ? static_cast<DType>(grad_data[grad_offset] * rescale_grad)
                                   : static_cast<DType>(0);
    if (clip_gradient >= 0.0f) {
      grad_rescaled = clip::Map(grad_rescaled, clip_gradient);
    }
    grad_rescaled += weight_data[i] * wd;
    mean_data[i] = beta1 * mean_data[i] + (1.f - beta1) * grad_rescaled;
    var_data[i] = beta2 * var_data[i] +
                  (1.f - beta2) * square::Map(grad_rescaled);
    KERNEL_ASSIGN(out_data[i], req, weight_data[i] - lr * mean_data[i] /
                  (square_root::Map(var_data[i]) + epsilon));
  }
};

template<>
void AdamStdUpdateDnsRspDnsImpl<gpu>(const AdamParam& param,
                                     const OpContext& ctx,
                                     const TBlob& weight,
                                     const NDArray& grad,
                                     const TBlob& mean,
                                     const TBlob& var,
                                     const OpReqType& req,
                                     TBlob *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  using namespace mshadow;
  Stream<gpu>* s = ctx.get_stream<gpu>();
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteInplace) << "kWriteInplace is expected for sparse adam_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(mean.shape_.Size(), 0);
  CHECK_GT(var.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(kIdx), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        const DType* weight_data = weight.dptr<DType>();
        const IType* grad_idx = grad.aux_data(kIdx).dptr<IType>();
        const DType* grad_val = grad.data().dptr<DType>();
        DType* mean_data = mean.dptr<DType>();
        DType* var_data = var.dptr<DType>();
        DType* out_data = out->dptr<DType>();
        const nnvm::dim_t num_rows = weight.shape_[0];
        const nnvm::dim_t row_length = weight.shape_.ProdShape(1, weight.ndim());
        nnvm::dim_t* prefix_sum = nullptr;
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      prefix_sum,
                                      prefix_sum,
                                      num_rows,
                                      Stream<gpu>::GetStream(s));
        Tensor<gpu, 1, char> workspace = ctx.requested[0]
          .get_space_typed<gpu, 1, char>(Shape1(num_rows * sizeof(nnvm::dim_t) +
                                         temp_storage_bytes), s);
        prefix_sum = reinterpret_cast<nnvm::dim_t*>(workspace.dptr_);
        d_temp_storage = workspace.dptr_ + num_rows*sizeof(nnvm::dim_t);
        // mark row flags
        Fill<false>(s, TBlob(prefix_sum, Shape1(num_rows), gpu::kDevMask), kWriteTo, 0);
        if (grad.storage_initialized()) {
          Kernel<MarkRowFlgKernel, gpu>::Launch(s, grad.aux_shape(kIdx)[0],
            prefix_sum, grad_idx);
          // calculate inclusive prefix sum
          cub::DeviceScan::InclusiveSum(d_temp_storage,
                                        temp_storage_bytes,
                                        prefix_sum,
                                        prefix_sum,
                                        num_rows,
                                        Stream<gpu>::GetStream(s));
        }

        Kernel<AdamStdDnsRspDnsKernel<req_type, gpu>, gpu>::Launch(s, weight.shape_.Size(),
          row_length, out_data, mean_data, var_data, weight_data, grad_idx, grad_val, prefix_sum,
          static_cast<DType>(param.clip_gradient), static_cast<DType>(param.beta1),
          static_cast<DType>(param.beta2), static_cast<DType>(param.lr),
          static_cast<DType>(param.wd), static_cast<DType>(param.epsilon),
          static_cast<DType>(param.rescale_grad));
      });
    });
  });
}

NNVM_REGISTER_OP(signsgd_update)
.set_attr<FCompute>("FCompute<gpu>", SignSGDUpdate<gpu>);

NNVM_REGISTER_OP(signum_update)
.set_attr<FCompute>("FCompute<gpu>", SignumUpdate<gpu>);

NNVM_REGISTER_OP(sgd_update)
.set_attr<FCompute>("FCompute<gpu>", SGDUpdate<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SGDUpdateEx<gpu>);

NNVM_REGISTER_OP(sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", SGDMomUpdate<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SGDMomUpdateEx<gpu>);

NNVM_REGISTER_OP(mp_sgd_update)
.set_attr<FCompute>("FCompute<gpu>", MP_SGDUpdate<gpu>);

NNVM_REGISTER_OP(mp_sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", MP_SGDMomUpdate<gpu>);

NNVM_REGISTER_OP(multi_sgd_update)
.set_attr<FCompute>("FCompute<gpu>", MultiSGDUpdate<gpu, type_identity, 2>);
NNVM_REGISTER_OP(multi_sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", MultiSGDMomUpdate<gpu, type_identity, 3>);
NNVM_REGISTER_OP(multi_mp_sgd_update)
.set_attr<FCompute>("FCompute<gpu>", MultiSGDUpdate<gpu, single_precision, 3>);
NNVM_REGISTER_OP(multi_mp_sgd_mom_update)
.set_attr<FCompute>("FCompute<gpu>", MultiSGDMomUpdate<gpu, single_precision, 4>);

NNVM_REGISTER_OP(nag_mom_update)
.set_attr<FCompute>("FCompute<gpu>", NAGMomUpdate<gpu>);

NNVM_REGISTER_OP(mp_nag_mom_update)
.set_attr<FCompute>("FCompute<gpu>", MP_NAGMomUpdate<gpu>);

NNVM_REGISTER_OP(ftml_update)
.set_attr<FCompute>("FCompute<gpu>", FTMLUpdate<gpu>);

NNVM_REGISTER_OP(adam_update)
.set_attr<FCompute>("FCompute<gpu>", AdamUpdate<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", AdamUpdateEx<gpu>);

NNVM_REGISTER_OP(rmsprop_update)
.set_attr<FCompute>("FCompute<gpu>", RMSPropUpdate<gpu>);

NNVM_REGISTER_OP(rmspropalex_update)
.set_attr<FCompute>("FCompute<gpu>", RMSPropAlexUpdate<gpu>);

NNVM_REGISTER_OP(ftrl_update)
.set_attr<FCompute>("FCompute<gpu>", FtrlUpdate<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", FtrlUpdateEx<gpu>);

NNVM_REGISTER_OP(_sparse_adagrad_update)
.set_attr<FComputeEx>("FComputeEx<gpu>", AdagradUpdateEx<gpu>);

NNVM_REGISTER_OP(lamb_update_phase1)
.set_attr<FCompute>("FCompute<gpu>", LambUpdatePhaseOne<gpu>);

NNVM_REGISTER_OP(lamb_update_phase2)
.set_attr<FCompute>("FCompute<gpu>", LambUpdatePhaseTwo<gpu>);

NNVM_REGISTER_OP(mp_lamb_update_phase1)
.set_attr<FCompute>("FCompute<gpu>", MPLambUpdatePhaseOne<gpu>);

NNVM_REGISTER_OP(mp_lamb_update_phase2)
.set_attr<FCompute>("FCompute<gpu>", MPLambUpdatePhaseTwo<gpu>);

}  // namespace op
}  // namespace mxnet
