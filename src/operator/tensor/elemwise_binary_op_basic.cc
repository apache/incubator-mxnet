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
 * \file elemwise_binary_op_basic.cc
 * \brief CPU Implementation of basic elementwise binary broadcast operators
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op-inl.h"
#include "../nn/mkldnn/mkldnn_ops-inl.h"
#include "../nn/mkldnn/mkldnn_base-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNSum(const NDArray& input) {
  int ndim = input.shape().ndim();
  return input.dtype() == mshadow::kFloat32 && (ndim >= 1 && ndim <= 4) &&
         input.storage_type() == kDefaultStorage;
}

static void ElemwiseAddEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
#if MXNET_USE_MKLDNN == 1
  if (SupportMKLDNNSum(inputs[0]) && SupportMKLDNNSum(inputs[1])) {
    MKLDNNSumForward(attrs, ctx, inputs, req[0], outputs[0]);
    return;
  } else if (inputs[0].storage_type() == kDefaultStorage
             && inputs[1].storage_type() == kDefaultStorage) {
    FallBackCompute(ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::plus>,
                    attrs, ctx, inputs, req, outputs);
    return;
  }
#endif
  ElemwiseBinaryOp::ComputeEx<cpu, op::mshadow_op::plus>(attrs, ctx, inputs,
                                                         req, outputs);
}

static inline bool ElemwiseAddStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int> *in_attrs,
                                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  bool ret = ElemwiseBinaryOp::PreferDenseStorageType<true, true, true>(
               attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && !MKLDNNEnvSet()) {
    *dispatch_mode = DispatchMode::kFComputeFallback;
  } else if (dev_mask == mshadow::cpu::kDevMask
      && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)
      && out_attrs->at(0) == kDefaultStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  return ret;
}

MXNET_OPERATOR_REGISTER_BINARY(elemwise_add)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddStorageType)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::plus>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
#endif
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseAddEx)
.set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */
                            [](const NodeAttrs& attrs) {
                            return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};})
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_add)
.add_alias("_add").add_alias("_plus").add_alias("_Plus")
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.describe(R"code(Adds arguments element-wise.

The storage type of ``elemwise_add`` output depends on storage types of inputs

   - elemwise_add(row_sparse, row_sparse) = row_sparse
   - elemwise_add(csr, csr) = csr
   - elemwise_add(default, csr) = default
   - elemwise_add(csr, default) = default
   - elemwise_add(default, rsp) = default
   - elemwise_add(rsp, default) = default
   - otherwise, ``elemwise_add`` generates output with default storage

)code")
.set_attr<nnvm::FGradient>("FGradient", CloneGradient{"_backward_add"});

// specialized gradient add function to do add to optimization
// this must differ from elemwise_add to prevent add to optimization in forward pass.
MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_grad_add, op::mshadow_op::plus);

static void _backward_ElemwiseAddEx(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
#if MXNET_USE_MKLDNN == 1
  if (inputs[0].IsMKLDNNData()) {
    MKLDNNCopy(attrs, ctx, inputs[0], req[0], outputs[0]);
    MKLDNNCopy(attrs, ctx, inputs[0], req[1], outputs[1]);
    return;
  } else if (common::ContainsOnlyStorage(inputs, kDefaultStorage)) {
    FallBackCompute(
        ElemwiseBinaryOp::BackwardUseNone<cpu, mshadow_op::identity, mshadow_op::identity>,
            attrs, ctx, inputs, req, outputs);
    return;
  }
#endif
  ElemwiseBinaryOp::BackwardUseNoneEx<cpu, mshadow_op::identity, mshadow_op::identity>(
      attrs, ctx, inputs, req, outputs);
}

static inline bool ElemwiseAddBackwardStorageType(const nnvm::NodeAttrs& attrs,
                                                  const int dev_mask,
                                                  DispatchMode* dispatch_mode,
                                                  std::vector<int> *in_attrs,
                                                  std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  bool ret = ElemwiseStorageType<1, 2, true, true, true>(attrs, dev_mask, dispatch_mode,
                                                         in_attrs, out_attrs);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && !MKLDNNEnvSet()) {
    *dispatch_mode = DispatchMode::kFComputeFallback;
  } else if (dev_mask == mshadow::cpu::kDevMask) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  return ret;
}

NNVM_REGISTER_OP(_backward_add)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 0},
                                                                           {0, 1}};
                                })
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<bool>("TIsMKLDNN", true)
#endif
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseNone<
  cpu, mshadow_op::identity, mshadow_op::identity>)
.set_attr<FComputeEx>("FComputeEx<cpu>", _backward_ElemwiseAddEx)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseAddBackwardStorageType);

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_PD(elemwise_sub, op::mshadow_op::minus)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_sub)
.add_alias("_sub").add_alias("_minus").add_alias("_Minus")
.describe(R"code(Subtracts arguments element-wise.

The storage type of ``elemwise_sub`` output depends on storage types of inputs

   - elemwise_sub(row_sparse, row_sparse) = row_sparse
   - elemwise_sub(csr, csr) = csr
   - elemwise_sub(default, csr) = default
   - elemwise_sub(csr, default) = default
   - elemwise_sub(default, rsp) = default
   - elemwise_sub(rsp, default) = default
   - otherwise, ``elemwise_sub`` generates output with default storage

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sub"});

NNVM_REGISTER_OP(_backward_sub)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 0},
                                                                           {0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseNone<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BackwardUseNoneEx<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FInferStorageType>("FInferStorageType",
                             ElemwiseStorageType<1, 2, true, true, true>);

MXNET_OPERATOR_REGISTER_BINARY(elemwise_mul)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_mul)
.describe(R"code(Multiplies arguments element-wise.

The storage type of ``elemwise_mul`` output depends on storage types of inputs

   - elemwise_mul(default, default) = default
   - elemwise_mul(row_sparse, row_sparse) = row_sparse
   - elemwise_mul(default, row_sparse) = row_sparse
   - elemwise_mul(row_sparse, default) = row_sparse
   - elemwise_mul(csr, csr) = csr
   - otherwise, ``elemwise_mul`` generates output with default storage

)code")
.set_attr<FInferStorageType>("FInferStorageType",
                             ElemwiseBinaryOp::PreferSparseStorageType)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, op::mshadow_op::mul>)
.set_attr<FComputeEx>("FComputeEx<cpu>",
                      ElemwiseBinaryOp::ComputeDnsLRValueEx<cpu, op::mshadow_op::mul, true, true>)
.set_attr<FResourceRequest>("FResourceRequest",  /* For Sparse CSR */
                              [](const NodeAttrs& attrs) {
                                return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                              })
.add_alias("_mul").add_alias("_Mul")
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    // z = x * y
    // f(x, y) = x * y
    // dx = z_grad * y, dy = z_grad * x
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_x",
                              {ograds[0], n->inputs[1]}, nullptr, &n));
    ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_y",
                              {ograds[0], n->inputs[0]}, nullptr, &n));
    return ret;
});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(elemwise_div, op::mshadow_op::div)
MXNET_ADD_SPARSE_OP_ALIAS(elemwise_div)
.describe(R"code(Divides arguments element-wise.

The storage type of ``elemwise_div`` output is always dense

)code")
.add_alias("_div").add_alias("_Div")
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    // z = x / y
    // f(x, y) = x / y
    // dx = z_grad / y, dy = z_grad * x * -1/y^2
    auto r_y = MakeNode("reciprocal", n->attrs.name + "_reciprocal_y",
                        {n->inputs[1]}, nullptr, &n);

    // Order Matters : First should be z_grad followed by y.
    auto grad_r_y = MakeNode("_backward_reciprocal", n->attrs.name + "_backward_reciprocal_y",
                        {ograds[0], n->inputs[1]}, nullptr, &n);

    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_x",
                              {ograds[0], nnvm::NodeEntry{r_y}}, nullptr, &n));
    ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_y",
                              {n->inputs[0], nnvm::NodeEntry{grad_r_y}}, nullptr, &n));
    return ret;
});

MXNET_OPERATOR_REGISTER_BINARY(_mod)
.add_alias("_Mod")
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod"});

NNVM_REGISTER_OP(_backward_mod)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<std::pair<int, int> >{{0, 1}};
                                })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BackwardUseIn<
  cpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
