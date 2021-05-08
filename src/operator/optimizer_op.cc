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
 * \file optimizer_op.cc
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#include "./optimizer_op-inl.h"
#include "./elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SGDParam);
DMLC_REGISTER_PARAMETER(SGDMomParam);
DMLC_REGISTER_PARAMETER(MultiSGDParam);
DMLC_REGISTER_PARAMETER(MultiSGDMomParam);
DMLC_REGISTER_PARAMETER(FTMLParam);
DMLC_REGISTER_PARAMETER(AdamParam);
DMLC_REGISTER_PARAMETER(NAGParam);
DMLC_REGISTER_PARAMETER(NAGMomParam);
DMLC_REGISTER_PARAMETER(RMSPropParam);
DMLC_REGISTER_PARAMETER(RMSPropAlexParam);
DMLC_REGISTER_PARAMETER(FtrlParam);
DMLC_REGISTER_PARAMETER(SignSGDParam);
DMLC_REGISTER_PARAMETER(SignumParam);
DMLC_REGISTER_PARAMETER(AdagradParam);
DMLC_REGISTER_PARAMETER(LambUpdatePhaseOneParam);
DMLC_REGISTER_PARAMETER(LambUpdatePhaseTwoParam);

NNVM_REGISTER_OP(signsgd_update)
.describe(R"code(Update function for SignSGD optimizer.

.. math::

 g_t = \nabla J(W_{t-1})\\
 W_t = W_{t-1} - \eta_t \text{sign}(g_t)

It updates the weights using::

 weight = weight - learning_rate * sign(gradient)

.. note::
   - sparse ndarray not supported for this optimizer yet.
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SignSGDParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FCompute>("FCompute<cpu>", SignSGDUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_arguments(SignSGDParam::__FIELDS__());


NNVM_REGISTER_OP(signum_update)
.describe(R"code(SIGN momentUM (Signum) optimizer.

.. math::

 g_t = \nabla J(W_{t-1})\\
 m_t = \beta m_{t-1} + (1 - \beta) g_t\\
 W_t = W_{t-1} - \eta_t \text{sign}(m_t)

It updates the weights using::
 state = momentum * state + (1-momentum) * gradient
 weight = weight - learning_rate * sign(state)

Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

.. note::
   - sparse ndarray not supported for this optimizer yet.
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SignumParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FCompute>("FCompute<cpu>", SignumUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_arguments(SignumParam::__FIELDS__());

template<int req>
struct SGDMomStdDnsRspDnsKernel<req, cpu> {
  template<typename DType, typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i, index_t row_length, DType* out_data,
    DType* mom_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const RType* prefix_sum, const DType clip_gradient,
    const DType momentum, const DType lr, const DType wd, const DType rescale_grad) {
    const bool non_zero = (i == 0) ? prefix_sum[0] > 0
                                   : prefix_sum[i] > prefix_sum[i-1];

    const index_t row_i = i * row_length;
    const RType grad_i = (prefix_sum[i]-1) * row_length;
    for (index_t j = 0; j < row_length; j++) {
      const index_t data_i = row_i + j;
      const DType grad = non_zero ? grad_data[grad_i + j]
                                  : static_cast<DType>(0);
      DType grad_rescaled = rescale_grad * grad;
      if (clip_gradient >= 0.0f) {
        grad_rescaled = mshadow_op::clip::Map(grad_rescaled, clip_gradient);
      }
      grad_rescaled += wd * weight_data[data_i];
      mom_data[data_i] *= momentum;
      mom_data[data_i] -= grad_rescaled;
      KERNEL_ASSIGN(out_data[data_i], req, weight_data[data_i] + lr * mom_data[data_i]);
    }
  }
};

/*
 * \brief standard momentum update for dense weight on cpu.
 *        state is expected to be dense, while grad is expected to be row_sparse.
 */
template<>
void SGDMomStdUpdateDnsRspDnsImpl<cpu>(const SGDMomParam& param,
                                       const OpContext& ctx,
                                       const TBlob& weight,
                                       const NDArray& grad,
                                       const TBlob& mom,
                                       const OpReqType& req,
                                       TBlob *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  using namespace mshadow;
  Stream<cpu>* s = ctx.get_stream<cpu>();
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteInplace) << "kWriteInplace is expected for sparse sgd_mom_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(mom.shape_.Size(), 0);
  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(kIdx), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        DType* weight_data = weight.dptr<DType>();
        const IType* grad_idx = grad.aux_data(kIdx).dptr<IType>();
        const DType* grad_val = grad.data().dptr<DType>();
        DType* mom_data = mom.dptr<DType>();
        DType* out_data = out->dptr<DType>();
        const nnvm::dim_t num_rows = weight.shape_[0];
        const auto row_length = weight.shape_.ProdShape(1, weight.ndim());
        Tensor<cpu, 1, char> workspace = ctx.requested[0]
          .get_space_typed<cpu, 1, char>(Shape1(num_rows * sizeof(nnvm::dim_t)), s);

        nnvm::dim_t* prefix_sum = reinterpret_cast<nnvm::dim_t*>(workspace.dptr_);
        // mark row flags
        Kernel<set_zero, cpu>::Launch(s, num_rows, prefix_sum);
        if (grad.storage_initialized()) {
          Kernel<MarkRowFlgKernel, cpu>::Launch(s, grad.aux_shape(kIdx)[0],
            prefix_sum, grad_idx);
          // calculate inclusive prefix sum
          for (nnvm::dim_t i = 1; i < num_rows; i++) {
            prefix_sum[i] += prefix_sum[i - 1];
          }
        }
        Kernel<SGDMomStdDnsRspDnsKernel<req_type, cpu>, cpu>::Launch(s, num_rows, row_length,
          out_data, mom_data, weight_data, grad_idx, grad_val, prefix_sum,
          static_cast<DType>(param.clip_gradient), static_cast<DType>(param.momentum),
          static_cast<DType>(param.lr), static_cast<DType>(param.wd),
          static_cast<DType>(param.rescale_grad));
      });
    });
  });
}

template<int req>
struct AdamStdDnsRspDnsKernel<req, cpu> {
  template<typename DType, typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i, const nnvm::dim_t row_length, DType* out_data,
    DType* mean_data, DType* var_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const RType* prefix_sum, const DType clip_gradient,
    const DType beta1, const DType beta2, const DType lr, const DType wd,
    const DType epsilon, const DType rescale_grad) {
    using namespace mshadow_op;
    const bool non_zero = (i == 0) ? prefix_sum[0] > 0
                                   : prefix_sum[i] > prefix_sum[i-1];

    const index_t row_i = i * row_length;
    const RType grad_i = (prefix_sum[i]-1) * row_length;
    for (index_t j = 0; j < row_length; j++) {
      const index_t data_i = row_i + j;
      DType grad_rescaled = non_zero ? static_cast<DType>(
                                         grad_data[grad_i + j] * rescale_grad)
                                     : static_cast<DType>(0);
      if (clip_gradient >= 0.0f) {
        grad_rescaled = clip::Map(grad_rescaled, clip_gradient);
      }
      grad_rescaled += weight_data[data_i] * wd;
      mean_data[data_i] = beta1 * mean_data[data_i] + (1.f - beta1) * grad_rescaled;
      var_data[data_i] = beta2 * var_data[data_i] +
                         (1.f - beta2) * square::Map(grad_rescaled);
      KERNEL_ASSIGN(out_data[data_i], req, weight_data[data_i] - lr * mean_data[data_i] /
                    (square_root::Map(var_data[data_i]) + epsilon));
    }
  }
};


template<>
void AdamStdUpdateDnsRspDnsImpl<cpu>(const AdamParam& param,
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
  Stream<cpu>* s = ctx.get_stream<cpu>();
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
        nnvm::dim_t num_rows = weight.shape_[0];
        nnvm::dim_t row_length = weight.shape_.ProdShape(1, weight.ndim());
        Tensor<cpu, 1, char> workspace = ctx.requested[0]
          .get_space_typed<cpu, 1, char>(Shape1(num_rows * sizeof(nnvm::dim_t)), s);

        nnvm::dim_t* prefix_sum = reinterpret_cast<nnvm::dim_t*>(workspace.dptr_);
        // mark row flags
        Kernel<set_zero, cpu>::Launch(s, num_rows, prefix_sum);
        if (grad.storage_initialized()) {
          Kernel<MarkRowFlgKernel, cpu>::Launch(s, grad.aux_shape(kIdx)[0],
            prefix_sum, grad_idx);
          // calculate inclusive prefix sum
          for (nnvm::dim_t i = 1; i < num_rows; i++) {
            prefix_sum[i] += prefix_sum[i - 1];
          }
        }

        Kernel<AdamStdDnsRspDnsKernel<req_type, cpu>, cpu>::Launch(s, num_rows, row_length,
          out_data, mean_data, var_data, weight_data, grad_idx, grad_val, prefix_sum,
          static_cast<DType>(param.clip_gradient), static_cast<DType>(param.beta1),
          static_cast<DType>(param.beta2), static_cast<DType>(param.lr),
          static_cast<DType>(param.wd), static_cast<DType>(param.epsilon),
          static_cast<DType>(param.rescale_grad));
      });
    });
  });
}

/*!
 * \brief Storge type inference function for SGD.
 */
inline bool SGDStorageType(const nnvm::NodeAttrs& attrs,
                           const int dev_mask,
                           DispatchMode* dispatch_mode,
                           std::vector<int>* in_attrs,
                           std::vector<int>* out_attrs) {
  using namespace common;
  const SGDParam& param = nnvm::get<SGDParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int weight_stype = in_attrs->at(0);
  const int grad_stype = in_attrs->at(1);
  bool dispatched = false;
  if (!dispatched && ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, ... -> dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && grad_stype == kRowSparseStorage &&
      (weight_stype == kRowSparseStorage || weight_stype == kDefaultStorage)) {
    // grad's stype = rsp
    dispatched = storage_type_assign(out_attrs, static_cast<NDArrayStorageType>(weight_stype),
                                     dispatch_mode, DispatchMode::kFComputeEx);
    // warn users if lazy_update is turned on
    if (dispatched && param.wd != 0 && param.lazy_update) LogLazyUpdate();
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

NNVM_REGISTER_OP(multi_sgd_update)
.describe(R"code(Update function for Stochastic Gradient Descent (SDG) optimizer.

It updates the weights using::

 weight = weight - learning_rate * (gradient + wd * weight)

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const MultiSGDParam& param = dmlc::get<MultiSGDParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights * 2);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const MultiSGDParam& param = dmlc::get<MultiSGDParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights);
  })
.set_attr_parser(ParamParser<MultiSGDParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MultiSGDShape<MultiSGDParam, 2>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, -1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<MultiSGDParam>(attrs.parsed).num_weights;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("weight_") + std::to_string(i));
      ret.push_back(std::string("grad_") + std::to_string(i));
    }
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", MultiSGDUpdate<cpu, type_identity, 2>)
.add_argument("data", "NDArray-or-Symbol[]", "Weights")
.add_arguments(MultiSGDParam::__FIELDS__());

NNVM_REGISTER_OP(multi_sgd_mom_update)
.describe(R"code(Momentum update function for Stochastic Gradient Descent (SGD) optimizer.

Momentum update has better convergence rates on neural networks. Mathematically it looks
like below:

.. math::

  v_1 = \alpha * \nabla J(W_0)\\
  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  W_t = W_{t-1} + v_t

It updates the weights using::

  v = momentum * v - learning_rate * gradient
  weight += v

Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const MultiSGDMomParam& param = dmlc::get<MultiSGDMomParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights * 3);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const MultiSGDMomParam& param = dmlc::get<MultiSGDMomParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights);
  })
.set_attr_parser(ParamParser<MultiSGDMomParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MultiSGDShape<MultiSGDMomParam, 3>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<-1, -1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<MultiSGDParam>(attrs.parsed).num_weights;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("weight_") + std::to_string(i));
      ret.push_back(std::string("grad_") + std::to_string(i));
      ret.push_back(std::string("mom_") + std::to_string(i));
    }
    return ret;
  })
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const MultiSGDMomParam& param = dmlc::get<MultiSGDMomParam>(attrs.parsed);
    ret.reserve(param.num_weights);
    for (int i = 0; i < param.num_weights; ++i) {
      ret.push_back(i * 3 + 2);
    }
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", MultiSGDMomUpdate<cpu, type_identity, 3>)
.add_argument("data", "NDArray-or-Symbol[]", "Weights, gradients and momentum")
.add_arguments(MultiSGDMomParam::__FIELDS__());

NNVM_REGISTER_OP(multi_mp_sgd_update)
.describe(R"code(Update function for multi-precision Stochastic Gradient Descent (SDG) optimizer.

It updates the weights using::

 weight = weight - learning_rate * (gradient + wd * weight)

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const MultiSGDParam& param = dmlc::get<MultiSGDParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights * 3);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const MultiSGDParam& param = dmlc::get<MultiSGDParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights);
  })
.set_attr_parser(ParamParser<MultiSGDParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MultiSGDShape<MultiSGDParam, 3>)
.set_attr<nnvm::FInferType>("FInferType", MP_MultiSGD_InferType<MultiSGDParam, 3, 1>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<MultiSGDParam>(attrs.parsed).num_weights;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("weight_") + std::to_string(i));
      ret.push_back(std::string("grad_") + std::to_string(i));
      ret.push_back(std::string("weight32_") + std::to_string(i));
    }
    return ret;
  })
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const MultiSGDParam& param = dmlc::get<MultiSGDParam>(attrs.parsed);
    ret.reserve(param.num_weights);
    for (int i = 0; i < param.num_weights; ++i) {
      ret.push_back(i * 3 + 2);
    }
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", MultiSGDUpdate<cpu, single_precision, 3>)
.add_argument("data", "NDArray-or-Symbol[]", "Weights")
.add_arguments(MultiSGDParam::__FIELDS__());

NNVM_REGISTER_OP(multi_mp_sgd_mom_update)
.describe(R"code(Momentum update function for multi-precision Stochastic Gradient Descent (SGD) optimizer.

Momentum update has better convergence rates on neural networks. Mathematically it looks
like below:

.. math::

  v_1 = \alpha * \nabla J(W_0)\\
  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  W_t = W_{t-1} + v_t

It updates the weights using::

  v = momentum * v - learning_rate * gradient
  weight += v

Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

)code" ADD_FILELINE)
.set_num_inputs([](const nnvm::NodeAttrs& attrs) {
    const MultiSGDMomParam& param = dmlc::get<MultiSGDMomParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights * 4);
  })
.set_num_outputs([](const nnvm::NodeAttrs& attrs) {
    const MultiSGDMomParam& param = dmlc::get<MultiSGDMomParam>(attrs.parsed);
    return static_cast<uint32_t>(param.num_weights);
  })
.set_attr_parser(ParamParser<MultiSGDMomParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MultiSGDShape<MultiSGDMomParam, 4>)
.set_attr<nnvm::FInferType>("FInferType", MP_MultiSGD_InferType<MultiSGDMomParam, 4, 2>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    uint32_t num_args = dmlc::get<MultiSGDMomParam>(attrs.parsed).num_weights;
    std::vector<std::string> ret;
    for (uint32_t i = 0; i < num_args; ++i) {
      ret.push_back(std::string("weight_") + std::to_string(i));
      ret.push_back(std::string("grad_") + std::to_string(i));
      ret.push_back(std::string("mom_") + std::to_string(i));
      ret.push_back(std::string("weight32_") + std::to_string(i));
    }
    return ret;
  })
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    std::vector<uint32_t> ret;
    const MultiSGDMomParam& param = dmlc::get<MultiSGDMomParam>(attrs.parsed);
    for (int i = 0; i < param.num_weights; ++i) {
      ret.push_back(i * 4 + 2);
      ret.push_back(i * 4 + 3);
    }
    return ret;
  })
.set_attr<FCompute>("FCompute<cpu>", MultiSGDMomUpdate<cpu, single_precision, 4>)
.add_argument("data", "NDArray-or-Symbol[]", "Weights")
.add_arguments(MultiSGDMomParam::__FIELDS__());

NNVM_REGISTER_OP(sgd_update)
MXNET_ADD_SPARSE_OP_ALIAS(sgd_update)
.describe(R"code(Update function for Stochastic Gradient Descent (SGD) optimizer.

It updates the weights using::

 weight = weight - learning_rate * (gradient + wd * weight)

However, if gradient is of ``row_sparse`` storage type and ``lazy_update`` is True,
only the row slices whose indices appear in grad.indices are updated::

 for row in gradient.indices:
     weight[row] = weight[row] - learning_rate * (gradient[row] + wd * weight[row])

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", SGDStorageType)
.set_attr<FCompute>("FCompute<cpu>", SGDUpdate<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SGDUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_arguments(SGDParam::__FIELDS__());

NNVM_REGISTER_OP(sgd_mom_update)
MXNET_ADD_SPARSE_OP_ALIAS(sgd_mom_update)
.describe(R"code(Momentum update function for Stochastic Gradient Descent (SGD) optimizer.

Momentum update has better convergence rates on neural networks. Mathematically it looks
like below:

.. math::

  v_1 = \alpha * \nabla J(W_0)\\
  v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  W_t = W_{t-1} + v_t

It updates the weights using::

  v = momentum * v - learning_rate * gradient
  weight += v

Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.

However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and weight's storage
type is the same as momentum's storage type,
only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::

  for row in gradient.indices:
      v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
      weight[row] += v[row]

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDMomParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<FInferStorageType>("FInferStorageType", StdOptStorageType<1, SGDMomParam>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FResourceRequestEx>("FResourceRequestEx",
  [](const NodeAttrs& attrs, const int dev_mask, const DispatchMode dispatch_mode) {
    std::vector<ResourceRequest> request;
    if (dispatch_mode == DispatchMode::kFComputeEx) {
      request.emplace_back(ResourceRequest::kTempSpace);
    }
    return request;
  })
.set_attr<FCompute>("FCompute<cpu>", SGDMomUpdate<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SGDMomUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_arguments(SGDMomParam::__FIELDS__());

NNVM_REGISTER_OP(mp_sgd_update)
.describe("Updater function for multi-precision sgd optimizer")
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", MP_InferType<2, 1, 3>)
.set_attr<FCompute>("FCompute<cpu>", MP_SGDUpdate<cpu>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "gradient")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(SGDParam::__FIELDS__());

NNVM_REGISTER_OP(mp_sgd_mom_update)
.describe("Updater function for multi-precision sgd optimizer")
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<SGDMomParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", MP_InferType<2, 1, 4>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", MP_SGDMomUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(SGDMomParam::__FIELDS__());

NNVM_REGISTER_OP(ftml_update)
.describe(R"code(The FTML optimizer described in
*FTML - Follow the Moving Leader in Deep Learning*,
available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.

.. math::

 g_t = \nabla J(W_{t-1})\\
 v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 d_t = \frac{ 1 - \beta_1^t }{ \eta_t } (\sqrt{ \frac{ v_t }{ 1 - \beta_2^t } } + \epsilon)
 \sigma_t = d_t - \beta_1 d_{t-1}
 z_t = \beta_1 z_{ t-1 } + (1 - \beta_1^t) g_t - \sigma_t W_{t-1}
 W_t = - \frac{ z_t }{ d_t }

)code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<FTMLParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<5, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<5, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3, 4};
  })
.set_attr<FCompute>("FCompute<cpu>", FTMLUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("d", "NDArray-or-Symbol", "Internal state ``d_t``")
.add_argument("v", "NDArray-or-Symbol", "Internal state ``v_t``")
.add_argument("z", "NDArray-or-Symbol", "Internal state ``z_t``")
.add_arguments(FTMLParam::__FIELDS__());

NNVM_REGISTER_OP(adam_update)
MXNET_ADD_SPARSE_OP_ALIAS(adam_update)
.describe(R"code(Update function for Adam optimizer. Adam is seen as a generalization
of AdaGrad.

Adam update consists of the following steps, where g represents gradient and m, v
are 1st and 2nd order moment estimates (mean and variance).

.. math::

 g_t = \nabla J(W_{t-1})\\
 m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
 v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
 W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }

It updates the weights using::

 m = beta1*m + (1-beta1)*grad
 v = beta2*v + (1-beta2)*(grad**2)
 w += - learning_rate * m / (sqrt(v) + epsilon)

However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and the storage
type of weight is the same as those of m and v,
only the row slices whose indices appear in grad.indices are updated (for w, m and v)::

 for row in grad.indices:
     m[row] = beta1*m[row] + (1-beta1)*grad[row]
     v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
     w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdamParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<FInferStorageType>("FInferStorageType", StdOptStorageType<2, AdamParam>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", AdamUpdate<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", AdamUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_arguments(AdamParam::__FIELDS__());


NNVM_REGISTER_OP(nag_mom_update)
.describe(R"code(Update function for Nesterov Accelerated Gradient( NAG) optimizer.
It updates the weights using the following formula,

.. math::
  v_t = \gamma v_{t-1} + \eta * \nabla J(W_{t-1} - \gamma v_{t-1})\\
  W_t = W_{t-1} - v_t

Where 
:math:`\eta` is the learning rate of the optimizer
:math:`\gamma` is the decay rate of the momentum estimate
:math:`\v_t` is the update vector at time step `t`
:math:`\W_t` is the weight vector at time step `t`

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NAGMomParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FCompute>("FCompute<cpu>", NAGMomUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_arguments(NAGMomParam::__FIELDS__());


NNVM_REGISTER_OP(mp_nag_mom_update)
.describe(R"code(Update function for multi-precision Nesterov Accelerated Gradient( NAG) optimizer.
)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<NAGMomParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", MP_InferType<2, 1, 4>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", MP_NAGMomUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mom", "NDArray-or-Symbol", "Momentum")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(NAGMomParam::__FIELDS__());


NNVM_REGISTER_OP(rmsprop_update)
.describe(R"code(Update function for `RMSProp` optimizer.

`RMSprop` is a variant of stochastic gradient descent where the gradients are
divided by a cache which grows with the sum of squares of recent gradients?

`RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate for
each parameter monotonically over the course of training.
While this is analytically motivated for convex optimizations, it may not be ideal
for non-convex problems. `RMSProp` deals with this heuristically by allowing the
learning rates to rebound as the denominator decays over time.

Define the Root Mean Square (RMS) error criterion of the gradient as
:math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
gradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.

The :math:`E[g^2]_t` is given by:

.. math::
  E[g^2]_t = \rho * E[g^2]_{t-1} + (1-\rho) * g_t^2

The update step is

.. math::
  \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t

The RMSProp code follows the version in
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
Tieleman & Hinton, 2012.

Hinton suggests the momentum term :math:`\rho` to be 0.9 and the learning rate
:math:`\eta` to be 0.001.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RMSPropParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs &attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FCompute>("FCompute<cpu>", RMSPropUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("n", "NDArray-or-Symbol", "n")
.add_arguments(RMSPropParam::__FIELDS__());

NNVM_REGISTER_OP(rmspropalex_update)
.describe(R"code(Update function for RMSPropAlex optimizer.

`RMSPropAlex` is non-centered version of `RMSProp`.

Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
:math:`E[g]_t` is the decaying average over past gradient.

.. math::
  E[g^2]_t = \rho * E[g^2]_{t-1} + (1 - \rho) * g_t^2\\
  E[g]_t = \rho * E[g]_{t-1} + (1 - \rho) * g_t\\
  momentum_t = \gamma * momentum_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\

The update step is

.. math::
  \theta_{t+1} = \theta_t + momentum_t

The RMSPropAlex code follows the version in
http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.

Graves suggests the momentum term :math:`\rho` to be 0.95, :math:`\gamma`
to be 0.9 and the learning rate :math:`\eta` to be 0.0001.
)code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RMSPropAlexParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<5, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<5, 1>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3, 4};
  })
.set_attr<FCompute>("FCompute<cpu>", RMSPropAlexUpdate<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("n", "NDArray-or-Symbol", "n")
.add_argument("g", "NDArray-or-Symbol", "g")
.add_argument("delta", "NDArray-or-Symbol", "delta")
.add_arguments(RMSPropAlexParam::__FIELDS__());

NNVM_REGISTER_OP(ftrl_update)
MXNET_ADD_SPARSE_OP_ALIAS(ftrl_update)
.describe(R"code(Update function for Ftrl optimizer.
Referenced from *Ad Click Prediction: a View from the Trenches*, available at
http://dl.acm.org/citation.cfm?id=2488200.

It updates the weights using::

 rescaled_grad = clip(grad * rescale_grad, clip_gradient)
 z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
 n += rescaled_grad**2
 w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)

If w, z and n are all of ``row_sparse`` storage type,
only the row slices whose indices appear in grad.indices are updated (for w, z and n)::

 for row in grad.indices:
     rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
     z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
     n[row] += rescaled_grad[row]**2
     w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<FtrlParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<4, 1, false, true, false>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.set_attr<FCompute>("FCompute<cpu>", FtrlUpdate<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", FtrlUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("z", "NDArray-or-Symbol", "z")
.add_argument("n", "NDArray-or-Symbol", "Square of grad")
.add_arguments(FtrlParam::__FIELDS__());

NNVM_REGISTER_OP(_sparse_adagrad_update)
.describe(R"code(Update function for AdaGrad optimizer.

Referenced from *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*,
and available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.

Updates are applied by::

    rescaled_grad = clip(grad * rescale_grad, clip_gradient)
    history = history + square(rescaled_grad)
    w = w - learning_rate * rescaled_grad / sqrt(history + epsilon)

Note that non-zero values for the weight decay option are not supported.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<AdagradParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<3, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<FInferStorageType>("FInferStorageType", AdagradStorageType)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FComputeEx>("FComputeEx<cpu>", AdagradUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("history", "NDArray-or-Symbol", "History")
.add_arguments(AdagradParam::__FIELDS__());

NNVM_REGISTER_OP(lamb_update_phase1)
.describe(R"code(Phase I of lamb update it performs the following operations and returns g:.

Link to paper: https://arxiv.org/pdf/1904.00962.pdf

.. math::
    \begin{gather*}
    grad = grad * rescale_grad
    if (grad < -clip_gradient)
    then
         grad = -clip_gradient
    if (grad > clip_gradient)
    then
         grad = clip_gradient

    mean = beta1 * mean + (1 - beta1) * grad;
    variance = beta2 * variance + (1. - beta2) * grad ^ 2;

    if (bias_correction)
    then
         mean_hat = mean / (1. - beta1^t);
         var_hat = var / (1 - beta2^t);
         g = mean_hat / (var_hat^(1/2) + epsilon) + wd * weight;
    else
         g = mean / (var_data^(1/2) + epsilon) + wd * weight;
    \end{gather*}

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LambUpdatePhaseOneParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<FCompute>("FCompute<cpu>", LambUpdatePhaseOne<cpu>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_arguments(LambUpdatePhaseOneParam::__FIELDS__());

NNVM_REGISTER_OP(lamb_update_phase2)
.describe(R"code(Phase II of lamb update it performs the following operations and updates grad.

Link to paper: https://arxiv.org/pdf/1904.00962.pdf

.. math::
    \begin{gather*}
    if (lower_bound >= 0)
    then
         r1 = max(r1, lower_bound)
    if (upper_bound >= 0)
    then
         r1 = max(r1, upper_bound)

    if (r1 == 0 or r2 == 0)
    then
         lr = lr
    else
         lr = lr * (r1/r2)
    weight = weight - lr * g
    \end{gather*}

)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LambUpdatePhaseTwoParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LambUpdatePhaseTwoShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<FCompute>("FCompute<cpu>", LambUpdatePhaseTwo<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("g", "NDArray-or-Symbol", "Output of lamb_update_phase 1")
.add_argument("r1", "NDArray-or-Symbol", "r1")
.add_argument("r2", "NDArray-or-Symbol", "r2")
.add_arguments(LambUpdatePhaseTwoParam::__FIELDS__());

NNVM_REGISTER_OP(mp_lamb_update_phase1)
.describe(R"code(Mixed Precision version of Phase I of lamb update 
it performs the following operations and returns g:.

          Link to paper: https://arxiv.org/pdf/1904.00962.pdf

          .. math::
              \begin{gather*}
              grad32 = grad(float16) * rescale_grad
              if (grad < -clip_gradient)
              then
                   grad = -clip_gradient
              if (grad > clip_gradient)
              then
                   grad = clip_gradient

              mean = beta1 * mean + (1 - beta1) * grad;
              variance = beta2 * variance + (1. - beta2) * grad ^ 2;

              if (bias_correction)
              then
                   mean_hat = mean / (1. - beta1^t);
                   var_hat = var / (1 - beta2^t);
                   g = mean_hat / (var_hat^(1/2) + epsilon) + wd * weight32;
              else
                   g = mean / (var_data^(1/2) + epsilon) + wd * weight32;
              \end{gather*}

          )code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LambUpdatePhaseOneParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<5, 1>)
.set_attr<nnvm::FInferType>("FInferType", MPLambPhaseOneType<2, 1, 5>)
.set_attr<FCompute>("FCompute<cpu>", MPLambUpdatePhaseOne<cpu>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2, 3};
  })
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("mean", "NDArray-or-Symbol", "Moving mean")
.add_argument("var", "NDArray-or-Symbol", "Moving variance")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(LambUpdatePhaseOneParam::__FIELDS__());

NNVM_REGISTER_OP(mp_lamb_update_phase2)
.describe(R"code(Mixed Precision version Phase II of lamb update 
it performs the following operations and updates grad.

          Link to paper: https://arxiv.org/pdf/1904.00962.pdf

          .. math::
              \begin{gather*}
              if (lower_bound >= 0)
              then
                   r1 = max(r1, lower_bound)
              if (upper_bound >= 0)
              then
                   r1 = max(r1, upper_bound)

              if (r1 == 0 or r2 == 0)
              then
                   lr = lr
              else
                   lr = lr * (r1/r2)
              weight32 = weight32 - lr * g
              weight(float16) = weight32
              \end{gather*}

          )code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LambUpdatePhaseTwoParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MPLambUpdatePhaseTwoShape)
.set_attr<nnvm::FInferType>("FInferType", MP_InferType<1, 1, 5>)
.set_attr<FCompute>("FCompute<cpu>", MPLambUpdatePhaseTwo<cpu>)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{4};
  })
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("g", "NDArray-or-Symbol", "Output of mp_lamb_update_phase 1")
.add_argument("r1", "NDArray-or-Symbol", "r1")
.add_argument("r2", "NDArray-or-Symbol", "r2")
.add_argument("weight32", "NDArray-or-Symbol", "Weight32")
.add_arguments(LambUpdatePhaseTwoParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
