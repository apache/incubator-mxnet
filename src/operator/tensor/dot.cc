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
 * \file dot.cc
 * \brief CPU Implementation of matrix dot
 */

#include "./dot-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "../nn/mkldnn/mkldnn_dot-inl.h"
#endif

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(DotParam);

NNVM_REGISTER_OP(dot)
MXNET_ADD_SPARSE_OP_ALIAS(dot)
.describe(R"doc(Dot product of two arrays.

``dot``'s behavior depends on the input array dimensions:

- 1-D arrays: inner product of vectors
- 2-D arrays: matrix multiplication
- N-D arrays: a sum product over the last axis of the first input and the first
  axis of the second input

  For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
  result array will have shape `(n,m,r,s)`. It is computed by::

    dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])

  Example::

    x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
    y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
    dot(x,y)[0,0,1,1] = 0
    sum(x[0,0,:]*y[:,1,1]) = 0

The storage type of ``dot`` output depends on storage types of inputs, transpose option and
forward_stype option for output storage type. Implemented sparse operations include:

- dot(default, default, transpose_a=True/False, transpose_b=True/False) = default
- dot(csr, default, transpose_a=True) = default
- dot(csr, default, transpose_a=True) = row_sparse
- dot(csr, default) = default
- dot(csr, row_sparse) = default
- dot(default, csr) = csr (CPU only)
- dot(default, csr, forward_stype='default') = default
- dot(default, csr, transpose_b=True, forward_stype='default') = default

If the combination of input storage types and forward_stype does not match any of the
above patterns, ``dot`` will fallback and generate output with default storage.

.. Note::

    If the storage type of the lhs is "csr", the storage type of gradient w.r.t rhs will be
    "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
    and Adam. Note that by default lazy updates is turned on, which may perform differently
    from standard updates. For more details, please check the Optimization API at:
    https://mxnet.incubator.apache.org/api/python/optimization/optimization.html

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", DotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", DotForwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", DotForward_<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", DotForwardEx<cpu>)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
#endif
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_dot"})
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.add_arguments(DotParam::__FIELDS__());

#if MXNET_USE_MKLDNN == 1
inline bool SupportMKLDNNDot(const std::vector<NDArray>& inputs,
                             const std::vector<NDArray>& outputs) {
  const auto in_dtype = inputs[0].dtype();
  return in_dtype == inputs[1].dtype() &&
         in_dtype == outputs[0].dtype() &&
         (in_dtype == mshadow::kBfloat16 || in_dtype == mshadow::kFloat32);
}

inline void DotForwardExMKLDNN(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  if (SupportMKLDNNDot(inputs, outputs)) {
    MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
    MKLDNNRun(MKLDNNDotForward, attrs, ctx, inputs, req, outputs);
    MKLDNN_OPCHECK_RUN(DotForward_<cpu>, attrs, ctx, inputs, req, outputs);
  } else {
    FallBackCompute(DotForward_<cpu>, attrs, ctx, inputs, req, outputs);
  }
}
#endif

NNVM_REGISTER_OP(_backward_dot)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", DotBackwardInferStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", DotBackward_<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", DotBackwardEx<cpu>)
.add_arguments(DotParam::__FIELDS__());

NNVM_REGISTER_OP(batch_dot)
.add_alias("_npx_batch_dot")
.describe(R"doc(Batchwise dot product.

``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
``y`` are data in batch, namely N-D (N >= 3) arrays in shape of `(B0, ..., B_i, :, :)`.

For example, given ``x`` with shape `(B_0, ..., B_i, N, M)` and ``y`` with shape
`(B_0, ..., B_i, M, K)`, the result array will have shape `(B_0, ..., B_i, N, K)`,
which is computed by::

   batch_dot(x,y)[b_0, ..., b_i, :, :] = dot(x[b_0, ..., b_i, :, :], y[b_0, ..., b_i, :, :])

)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<DotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", BatchDotShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", BatchDotForward_<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::ObjectPtr& n,
       const std::vector<nnvm::NodeEntry>& ograds) {
  const DotParam& param = nnvm::get<DotParam>(n->attrs.parsed);
  nnvm::ObjectPtr lhs_grad;
  nnvm::ObjectPtr rhs_grad;
  std::string lhs_gnode_name = n->attrs.name + "_backward_lhs";
  std::string rhs_gnode_name = n->attrs.name + "_backward_rhs";
  if (param.transpose_a && param.transpose_b) {
    // Gradient of z = dot(x.T, y.T)
    // dx = dot(dz, y).T = dot(y.T, dz.T)
    // dy = dot(x, dz).T = dot(dz.T, x.T)
    lhs_grad = MakeNode("batch_dot", lhs_gnode_name,
                        {n->inputs[1], ograds[0]}, &(n->attrs.dict), &n);
    rhs_grad = MakeNode("batch_dot", rhs_gnode_name,
                        {ograds[0], n->inputs[0]}, &(n->attrs.dict), &n);
  } else if (!param.transpose_a && param.transpose_b) {
    // Gradient of z = dot(x, y.T)
    // dx = dot(dz, y)
    // dy = dot(x.T, dz).T = dot(dz.T, x)
    auto lhs_attrs_dict = n->attrs.dict;
    auto rhs_attrs_dict = n->attrs.dict;
    lhs_attrs_dict["transpose_a"] = "false";
    lhs_attrs_dict["transpose_b"] = "false";
    rhs_attrs_dict["transpose_a"] = "true";
    rhs_attrs_dict["transpose_b"] = "false";
    lhs_grad = MakeNode("batch_dot", lhs_gnode_name,
                        {ograds[0], n->inputs[1]}, &lhs_attrs_dict, &n);
    rhs_grad = MakeNode("batch_dot", rhs_gnode_name,
                        {ograds[0], n->inputs[0]}, &rhs_attrs_dict, &n);
  } else if (param.transpose_a && !param.transpose_b) {
    // Gradient of z = dot(x.T, y)
    // dx = dot(dz, y.T).T = dot(y, dz.T)
    // dy = dot(x, dz)
    auto lhs_attrs_dict = n->attrs.dict;
    auto rhs_attrs_dict = n->attrs.dict;
    lhs_attrs_dict["transpose_a"] = "false";
    lhs_attrs_dict["transpose_b"] = "true";
    rhs_attrs_dict["transpose_a"] = "false";
    rhs_attrs_dict["transpose_b"] = "false";
    lhs_grad = MakeNode("batch_dot", lhs_gnode_name,
                        {n->inputs[1], ograds[0]}, &lhs_attrs_dict, &n);
    rhs_grad = MakeNode("batch_dot", rhs_gnode_name,
                        {n->inputs[0], ograds[0]}, &rhs_attrs_dict, &n);
  } else {
    // Gradient of z = dot(x, y)
    // dx = dot(dz, y.T)
    // dy = dot(x.T, dz)
    auto lhs_attrs_dict = n->attrs.dict;
    auto rhs_attrs_dict = n->attrs.dict;
    lhs_attrs_dict["transpose_a"] = "false";
    lhs_attrs_dict["transpose_b"] = "true";
    rhs_attrs_dict["transpose_a"] = "true";
    rhs_attrs_dict["transpose_b"] = "false";
    lhs_grad = MakeNode("batch_dot", lhs_gnode_name,
                        {ograds[0], n->inputs[1]}, &lhs_attrs_dict, &n);
    rhs_grad = MakeNode("batch_dot", rhs_gnode_name,
                        {n->inputs[0], ograds[0]}, &rhs_attrs_dict, &n);
  }
  std::vector<nnvm::NodeEntry> ret;
  ret.emplace_back(nnvm::NodeEntry{lhs_grad, 0, 0});
  ret.emplace_back(nnvm::NodeEntry{rhs_grad, 0, 0});
  return ret;
})
.add_argument("lhs", "NDArray-or-Symbol", "The first input")
.add_argument("rhs", "NDArray-or-Symbol", "The second input")
.add_arguments(DotParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
