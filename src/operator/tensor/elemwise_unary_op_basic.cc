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
 * \file elemwise_unary_op_basic.cc
 * \brief CPU Implementation of elementwise unary function.
 */
#include <mxnet/base.h>
#include "elemwise_unary_op.h"
#include "./elemwise_binary_op-inl.h"
#include "../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nnvm/node_op_util.h"

namespace mxnet {
namespace op {

// infer storage function for _identity_with_attr_like_rhs op
static bool IdentityAttrLikeRhsStorageType(const nnvm::NodeAttrs& attrs,
                                           const int dev_mask,
                                           DispatchMode* dispatch_mode,
                                           std::vector<int> *in_attrs,
                                           std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const auto& rhs_stype = in_attrs->at(1);
  auto& lhs_stype = in_attrs->at(0);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;

  CHECK_NE(rhs_stype, kUndefinedStorage);
  type_assign(&out_stype, rhs_stype);
  type_assign(&lhs_stype, rhs_stype);
  if (!dispatched && lhs_stype == kDefaultStorage && rhs_stype == kDefaultStorage &&
      out_stype == kDefaultStorage) {
    // dns, dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && (lhs_stype == kRowSparseStorage || lhs_stype == kCSRStorage) &&
      (lhs_stype == out_stype)) {
    // rsp, _ -> rsp, or csr, _ -> csr
    dispatched = storage_type_assign(&out_stype, static_cast<NDArrayStorageType>(out_stype),
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched && (lhs_stype == kRowSparseStorage || lhs_stype == kCSRStorage) &&
      (out_stype == kDefaultStorage)) {
    // rsp/csr, _ -> dns
    dispatched = storage_type_assign(&out_stype, static_cast<NDArrayStorageType>(out_stype),
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

// relu
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(relu, cpu, mshadow_op::relu)
.describe(R"code(Computes rectified linear activation.

.. math::
   max(features, 0)

The storage type of ``relu`` output depends upon the input storage type:

   - relu(default) = default
   - relu(row_sparse) = row_sparse
   - relu(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_relu"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_relu, unary_bwd<mshadow_op::relu_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      std::vector<nnvm::NodeEntry> ret;
      // ograds[0]: dL/dxgrad
      // inputs[0]: dL/dy
      // inputs[1]: y
      // f(x) -> relu(x)
      // f'(x) = 1 if x > 0 else 0
      // f''(x) = 0
      auto dydx = MakeNode("_greater", n->attrs.name + "_dydx",
          {n->inputs[1], nnvm::NodeEntry{
            MakeNode("zeros_like", n->attrs.name + "tmp", {n->inputs[1]}, nullptr, &n)
          }}, nullptr, &n);
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry(dydx)}, nullptr, &n));
      ret.emplace_back(MakeNode("zeros_like", n->attrs.name + "_backward_grad_grad_in",
                                {n->inputs[1]}, nullptr, &n));
      return ret;
    });

// sigmoid
MXNET_OPERATOR_REGISTER_UNARY(sigmoid)
MXNET_ADD_SPARSE_OP_ALIAS(sigmoid)
.describe(R"code(Computes sigmoid of x element-wise.

.. math::
   y = 1 / (1 + exp(-x))

The storage type of ``sigmoid`` output is always dense

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::sigmoid>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sigmoid"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_sigmoid,
                                               unary_bwd<mshadow_op::sigmoid_grad>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // n->inputs[0] : y_grad
      // n->inputs[1] : f(x) = sigmoid(x)
      // ograds[0] : head_grads
      // f''(x) = f'(x) * (1 - 2*f(x))
      // NodeEntry{n} : y_grad * f'(x)
      auto ones = MakeNode("ones_like", n->attrs.name + "_grad_ones", {n->inputs[1]}, nullptr, &n);
      const std::unordered_map<std::string, std::string> args = {{"scalar", "2.0"}};
      auto two_y = MakeNode("_mul_scalar", n->attrs.name + "_mul_two", {n->inputs[1]}, &args, &n);
      auto one_minus_two_y = MakeNode("elemwise_sub", n->attrs.name + "_grad_sub",
                                    {nnvm::NodeEntry{ones}, nnvm::NodeEntry{two_y}}, nullptr, &n);
      auto grad_grad_mid = MakeNode("elemwise_mul", n->attrs.name + "_grad_mul",
                                    {n->inputs[0], nnvm::NodeEntry{one_minus_two_y}}, nullptr, &n);
      auto dydx = MakeNode("elemwise_div", n->attrs.name + "_grad_div",
                           {nnvm::NodeEntry{n}, n->inputs[0]}, nullptr, &n);

      // when building gradient graph, the backward node of n->inputs[1] will be
      // added to the graph again, therefore f`(x) will be multiplied
      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry{dydx}}, nullptr, &n));
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "backward_grad_grad_in",
                                {ograds[0], nnvm::NodeEntry{grad_grad_mid}}, nullptr, &n));
      return ret;
    });



DMLC_REGISTER_PARAMETER(HardSigmoidParam);
MXNET_OPERATOR_REGISTER_UNARY(hard_sigmoid)
.describe(R"code(Computes hard sigmoid of x element-wise.

.. math::
   y = max(0, min(1, alpha * x + beta))

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<HardSigmoidParam>)
.set_attr<FCompute>("FCompute<cpu>", HardSigmoidForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_hard_sigmoid"})
.add_arguments(HardSigmoidParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_hard_sigmoid)
.set_attr_parser(ParamParser<HardSigmoidParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", HardSigmoidBackward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// softsign
MXNET_OPERATOR_REGISTER_UNARY(softsign)
.describe(R"code(Computes softsign of x element-wise.

.. math::
   y = x / (1 + abs(x))

The storage type of ``softsign`` output is always dense

)code" ADD_FILELINE)
  .set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::softsign>)
  .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_softsign"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_softsign)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Compute<cpu,
  unary_bwd<mshadow_op::softsign_grad> >)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // NodeEntry{n} : dL/dy * f'(x)
      // n->inputs[0] : dL/dy
      // n->inputs[1] : x (ElemwiseGradUseIn)
      // ograds[0] : head_grads (dL/dx_grad)
      // y = f(x) = softsign(x)
      // f'(x) = dy/dx = 1/(1 + |x|)^2
      // f''(x) = (-2/|x|) * softsign(x) * f'(x) = -2*x/(|x|*(1 + |x|)^3)
      auto dydx = n->inputs[0];
      auto x = n->inputs[1];
      auto dldy_mul_dydx = nnvm::NodeEntry{n};
      auto op = mxnet::util::NodeOpGen{n};

      auto dldy = op.div(dldy_mul_dydx, dydx);
      auto abs_x = op.abs(x);
      auto r_abs_x = op.reciprocal(abs_x);
      auto neg_two_r_abs_x = op.mul(-2.0, r_abs_x);
      auto softsign_x = MakeNode("softsign", n->attrs.name + "_softsign_x",
                              {nnvm::NodeEntry{x}}, nullptr, &n);
      auto softsign_mul_dydx = op.mul(nnvm::NodeEntry{softsign_x}, dldy_mul_dydx);
      auto grad_grad_x = op.mul(softsign_mul_dydx, neg_two_r_abs_x);

      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(op.mul(ograds[0], dldy));
      ret.emplace_back(op.mul(ograds[0], grad_grad_x));
      return ret;
  });

// copy
static void CopyEx(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<NDArray>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
#if MXNET_USE_MKLDNN == 1
  const auto in_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  if (inputs[0].IsMKLDNNData()) {
    MKLDNNRun(MKLDNNCopy, attrs, ctx, inputs[0], req[0], outputs[0]);
    return;
  } else if (in_stype == kDefaultStorage && out_stype == kDefaultStorage) {
    if (req[0] != kNullOp && req[0] != kWriteInplace)
      FallBackCompute(UnaryOp::IdentityCompute<cpu>, attrs, ctx, inputs, req, outputs);
    return;
  }
#endif  // MXNET_USE_MKLDNN == 1
  UnaryOp::IdentityComputeEx<cpu>(attrs, ctx, inputs, req, outputs);
}

static inline bool CopyStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int> *in_attrs,
                                   std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  bool ret = ElemwiseStorageType<1, 1, false, true, true>(attrs, dev_mask, dispatch_mode,
                                                          in_attrs, out_attrs);
#if MXNET_USE_MKLDNN == 1
  // We have to make sure all inputs are default layouts. Otherwise, we might
  // want to fallback.
  if (dev_mask == mshadow::cpu::kDevMask
      && in_attrs->at(0) == kDefaultStorage
      && out_attrs->at(0) == kDefaultStorage) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif  // MXNET_USE_MKLDNN == 1
  return ret;
}

MXNET_OPERATOR_REGISTER_UNARY(_copy)
.MXNET_DESCRIBE("Returns a copy of the input.")
.add_alias("identity")
.set_attr<FInferStorageType>("FInferStorageType", CopyStorageType)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", CopyEx)
#if MXNET_USE_MKLDNN == 1
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<bool>("TIsMKLDNN", true)
#endif  // MXNET_USE_MKLDNN == 1
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

NNVM_REGISTER_OP(_backward_copy)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FInferStorageType>("FInferStorageType", CopyStorageType)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", CopyEx)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})  // MXNET_USE_MKLDNN == 1
#endif
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  });

NNVM_REGISTER_OP(_backward_reshape)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                [](const NodeAttrs& attrs){
                                  return std::vector<std::pair<int, int> >{{0, 0}};
                                })
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
                                  [](const NodeAttrs& attrs){
                                    return std::vector<bool>{true};
                                  });

MXNET_OPERATOR_REGISTER_UNARY(BlockGrad)
MXNET_ADD_SPARSE_OP_ALIAS(stop_gradient)
.add_alias("stop_gradient")
.describe(R"code(Stops gradient computation.

Stops the accumulated gradient of the inputs from flowing through this operator
in the backward direction. In other words, this operator prevents the contribution
of its inputs to be taken into account for computing gradients.

Example::

  v1 = [1, 2]
  v2 = [0, 1]
  a = Variable('a')
  b = Variable('b')
  b_stop_grad = stop_gradient(3 * b)
  loss = MakeLoss(b_stop_grad + a)

  executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
  executor.forward(is_train=True, a=v1, b=v2)
  executor.outputs
  [ 1.  5.]

  executor.backward()
  executor.grad_arrays
  [ 0.  0.]
  [ 1.  1.]

)code" ADD_FILELINE)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1, false, true, true>)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeEx<cpu>)
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_UNARY(make_loss)
MXNET_ADD_SPARSE_OP_ALIAS(make_loss)
  .describe(R"code(Make your own loss function in network construction.

This operator accepts a customized loss function symbol as a terminal loss and
the symbol should be an operator with no backward dependency.
The output of this function is the gradient of loss with respect to the input data.

For example, if you are a making a cross entropy loss function. Assume ``out`` is the
predicted output and ``label`` is the true label, then the cross entropy can be defined as::

  cross_entropy = label * log(out) + (1 - label) * log(1 - out)
  loss = make_loss(cross_entropy)

We will need to use ``make_loss`` when we are creating our own loss function or we want to
combine multiple loss functions. Also we may want to stop some variables' gradients
from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.

The storage type of ``make_loss`` output depends upon the input storage type:

   - make_loss(default) = default
   - make_loss(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"loss"};
  })
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1, false, true, true>)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeEx<cpu>)
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(MakeNode("ones_like", n->attrs.name + "_backward",
                     &(n->inputs), nullptr, &n));
    return ret;
  });

// identity output as first input, but attributes (shape and type) are constrained to be like rhs
// storage type attribute is not constrained to be like rhs if it is already defined
// for internal use only
NNVM_REGISTER_OP(_identity_with_attr_like_rhs)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) { return std::vector<std::string>{"lhs", "rhs"}; })
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption", [](const NodeAttrs& attrs) {
      return std::vector<std::pair<int, int> >{{0, 0}};
    })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
    [](const NodeAttrs& attrs){ return std::vector<bool>{true}; })
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 1); })
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeFirstItemEx<cpu>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", IdentityAttrLikeRhsStorageType)
.set_attr<nnvm::FGradient>(
    "FGradient",  [](const nnvm::NodePtr& n,
                     const std::vector<nnvm::NodeEntry>& ograds) {
      if (CheckGradAllZero(ograds)) return MakeZeroGradNodes(n, ograds);
      std::vector<nnvm::NodeEntry> lhs = MakeGradNode("_backward_copy", n, ograds,
                              std::unordered_map<std::string, std::string>());
      lhs.emplace_back(MakeNode("zeros_like", n->attrs.name + "_rhs_backward",
                         {n->inputs[1]}, nullptr, &n));
      return lhs;
    })
.add_argument("lhs", "NDArray-or-Symbol", "First input.")
.add_argument("rhs", "NDArray-or-Symbol", "Second input.");

void ReshapeLikeRangeCanonicalize(int ndims, const char *side,
                                  const dmlc::optional<int> &begin,
                                  const dmlc::optional<int> &end, int *cbegin,
                                  int *cend) {
  *cbegin = begin.has_value() ? begin.value() : 0;
  if (*cbegin < 0)
    *cbegin += ndims;

  if (!end.has_value()) {
    *cend = ndims;
  } else {
    *cend = end.value();
    if (*cend < 0) {
      *cend += ndims;
    }
  }
  CHECK(*cend <= ndims) << "Invalid end for " << side << "_end=" << end
                        << " as dimension number is " << ndims;
  CHECK((*cbegin < *cend)) << "Invalid begin, end, get " << side
                           << "_begin=" << begin << ", " << side
                           << "_end=" << end;

  CHECK(*cend >= 0) << "Invalid end for " << side << "_end=" << end;
  CHECK(*cbegin >= 0) << "Invalid begin for " << side << "_begin=" << begin;
}

void GetReshapeLikeParams(const ReshapeLikeParam &param, const mxnet::TShape &lshape,
                          const mxnet::TShape &rshape, int *lhs_begin, int *lhs_end,
                          int *rhs_begin, int *rhs_end) {
  // LHS params
  ReshapeLikeRangeCanonicalize(lshape.ndim(), "lhs", param.lhs_begin,
                               param.lhs_end, lhs_begin, lhs_end);
  // RHS params
  ReshapeLikeRangeCanonicalize(rshape.ndim(), "rhs", param.rhs_begin,
                               param.rhs_end, rhs_begin, rhs_end);
}

bool ReshapeLikeShapeCompute(const nnvm::NodeAttrs &attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  const ReshapeLikeParam &param = nnvm::get<ReshapeLikeParam>(attrs.parsed);
  const mxnet::TShape &lshape = (*in_attrs)[0];
  const mxnet::TShape &rshape = (*in_attrs)[1];
  int lhs_begin, lhs_end, rhs_begin, rhs_end;
  GetReshapeLikeParams(param, lshape, rshape, &lhs_begin, &lhs_end, &rhs_begin,
                       &rhs_end);

  int lhsrank = lshape.ndim();
  int orank = lhsrank + (rhs_end - rhs_begin) - (lhs_end - lhs_begin);
  mxnet::TShape oshape(orank, -1);

  for (int i = 0; i < lhs_begin; ++i)
    oshape[i] = lshape[i];

  int opos = lhs_begin;
  for (int i = rhs_begin; i < rhs_end; ++i) {
    oshape[opos] = rshape[i];
    opos += 1;
  }

  for (int i = lhs_end; i < lhsrank; ++i) {
    oshape[opos] = lshape[i];
    opos += 1;
  }

  CHECK_EQ((*in_attrs)[0].Size(), oshape.Size())
      << "Cannot reshape lhs with shape " << (*in_attrs)[0] << "to new "
      << "shape " << oshape << " because they have different "
      << "size.";
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(oshape);
}

DMLC_REGISTER_PARAMETER(ReshapeLikeParam);
NNVM_REGISTER_OP(reshape_like)
.describe(R"code(Reshape some or all dimensions of `lhs` to have the same shape as some or all dimensions of `rhs`.

Returns a **view** of the `lhs` array with a new shape without altering any data.

Example::

  x = [1, 2, 3, 4, 5, 6]
  y = [[0, -4], [3, 2], [2, 2]]
  reshape_like(x, y) = [[1, 2], [3, 4], [5, 6]]

More precise control over how dimensions are inherited is achieved by specifying \
slices over the `lhs` and `rhs` array dimensions. Only the sliced `lhs` dimensions \
are reshaped to the `rhs` sliced dimensions, with the non-sliced `lhs` dimensions staying the same.

  Examples::

  - lhs shape = (30,7), rhs shape = (15,2,4), lhs_begin=0, lhs_end=1, rhs_begin=0, rhs_end=2, output shape = (15,2,7)
  - lhs shape = (3, 5), rhs shape = (1,15,4), lhs_begin=0, lhs_end=2, rhs_begin=1, rhs_end=2, output shape = (15)

Negative indices are supported, and `None` can be used for either `lhs_end` or `rhs_end` to indicate the end of the range.

  Example::

  - lhs shape = (30, 12), rhs shape = (4, 2, 2, 3), lhs_begin=-1, lhs_end=None, rhs_begin=1, rhs_end=None, output shape = (30, 2, 2, 3)

)code" ADD_FILELINE)
.add_alias("_npx_reshape_like")
.set_num_inputs(2)
.set_attr_parser(ParamParser<ReshapeLikeParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) { return std::vector<std::string>{"lhs", "rhs"}; })
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption", [](const NodeAttrs& attrs) {
      return std::vector<std::pair<int, int> >{{0, 0}};
    })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
    [](const NodeAttrs& attrs){ return std::vector<bool>{true}; })
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 1); })
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<mxnet::FInferShape>("FInferShape", ReshapeLikeShapeCompute)
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
                                             std::vector<int> *in_attrs,
                                             std::vector<int> *out_attrs) {
    CHECK_EQ(in_attrs->size(), 2) << " in operator " << attrs.name;
    std::vector<int> checked_in_attrs = { (*in_attrs)[0] };
    bool ret = !type_is_none((*in_attrs)[1]) &&
               ElemwiseType<1, 1>(attrs, &checked_in_attrs, out_attrs);
    (*in_attrs)[0] = checked_in_attrs[0];
    return ret;
  })
.set_attr<nnvm::FGradient>(
    "FGradient",  [](const nnvm::NodePtr& n,
                     const std::vector<nnvm::NodeEntry>& ograds) {
      if (CheckGradAllZero(ograds)) return MakeZeroGradNodes(n, ograds);
      std::vector<nnvm::NodeEntry> lhs = MakeGradNode("_backward_copy", n, ograds,
                              std::unordered_map<std::string, std::string>());
      lhs.emplace_back(MakeNode("zeros_like", n->attrs.name + "_rhs_backward",
                         {n->inputs[1]}, nullptr, &n));
      return lhs;
    })
.add_argument("lhs", "NDArray-or-Symbol", "First input.")
.add_argument("rhs", "NDArray-or-Symbol", "Second input.")
.add_arguments(ReshapeLikeParam::__FIELDS__());

void ShapeComputeCPU(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  size_t type_size = mshadow::mshadow_sizeof(out_data.type_flag_);
  memcpy(out_data.dptr_, in_data.shape_.data(), in_data.ndim() * type_size);
}

NNVM_REGISTER_OP(shape_array)
.add_alias("_npx_shape_array")
.describe(R"code(Returns a 1D int64 array containing the shape of data.

Example::

  shape_array([[1,2,3,4], [5,6,7,8]]) = [2,4]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FCompute>("FCompute<cpu>", ShapeComputeCPU)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     mxnet::ShapeVector *in_attrs,
     mxnet::ShapeVector *out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);
    mxnet::TShape target_shape(1, -1);
    target_shape[0] = in_attrs->at(0).ndim();
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
    return !shape_is_none(out_attrs->at(0));
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int>* in_attrs,
     std::vector<int>* out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
    return out_attrs->at(0) != -1;
  })
.add_argument("data", "NDArray-or-Symbol", "Input Array.");

void SizeComputeCPU(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  size_t type_size = mshadow::mshadow_sizeof(out_data.type_flag_);
  const index_t size_var = in_data.Size();
  memcpy(out_data.dptr_, &size_var, type_size);
}

NNVM_REGISTER_OP(size_array)
.describe(R"code(Returns a 1D int64 array containing the size of data.

Example::

  size_array([[1,2,3,4], [5,6,7,8]]) = [8]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<FCompute>("FCompute<cpu>", SizeComputeCPU)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     mxnet::ShapeVector *in_attrs,
     mxnet::ShapeVector *out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(1, 1));
    return !shape_is_none(out_attrs->at(0));
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int>* in_attrs,
     std::vector<int>* out_attrs) {
    CHECK_EQ(in_attrs->size(), 1U);
    CHECK_EQ(out_attrs->size(), 1U);
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt64);
    return out_attrs->at(0) != -1;
  })
.add_argument("data", "NDArray-or-Symbol", "Input Array.");

DMLC_REGISTER_PARAMETER(CastParam);
NNVM_REGISTER_OP(Cast)
.add_alias("cast")
.add_alias("_npi_cast")
.add_alias("_npx_cast")
.describe(R"code(Casts all elements of the input to a new type.

.. note:: ``Cast`` is deprecated. Use ``cast`` instead.

Example::

   cast([0.9, 1.3], dtype='int32') = [0, 1]
   cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
   cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<CastParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", CastType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", CastCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_cast"})
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(CastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_cast)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", CastCompute<cpu>);

// negative
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(negative, cpu, mshadow_op::negation)
.describe(R"code(Numerical negative of the argument, element-wise.

The storage type of ``negative`` output depends upon the input storage type:

   - negative(default) = default
   - negative(row_sparse) = row_sparse
   - negative(csr) = csr

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

// abs
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(abs, cpu, mshadow_op::abs)
.describe(R"code(Returns element-wise absolute value of the input.

Example::

   abs([-2, 0, 3]) = [2, 0, 3]

The storage type of ``abs`` output depends upon the input storage type:

   - abs(default) = default
   - abs(row_sparse) = row_sparse
   - abs(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_abs"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_abs, unary_bwd<mshadow_op::sign>)
.set_attr<nnvm::FGradient>("FGradient",
    [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
      // ograds[0]: dL/dxgrad
      // inputs[0]: dL/dy
      // inputs[1]: x
      // f(x) -> abs(x)
      // f'(x) = 1 if x > 0 else -1
      // f''(x) = 0
      auto dydx = MakeNode("elemwise_div", n->attrs.name + "_dydx",
                           {nnvm::NodeEntry{n}, n->inputs[0]}, nullptr, &n);

      std::vector<nnvm::NodeEntry> ret;
      ret.emplace_back(MakeNode("elemwise_mul", n->attrs.name + "_backward_grad_grad",
                                {ograds[0], nnvm::NodeEntry(dydx)}, nullptr, &n));
      ret.emplace_back(MakeNode("zeros_like", n->attrs.name + "_backward_grad_grad_in",
                                {n->inputs[1]}, nullptr, &n));
      return ret;
    });


// sign
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(sign, cpu, mshadow_op::sign)
.describe(R"code(Returns element-wise sign of the input.

Example::

   sign([-2, 0, 3]) = [-1, 0, 1]

The storage type of ``sign`` output depends upon the input storage type:

   - sign(default) = default
   - sign(row_sparse) = row_sparse
   - sign(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_sign"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU(_backward_sign, unary_bwd<mshadow_op::sign_grad>);

// round
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(round, cpu, mshadow_op::round)
.describe(R"code(Returns element-wise rounded value to the nearest integer of the input.

Example::

   round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]

The storage type of ``round`` output depends upon the input storage type:

  - round(default) = default
  - round(row_sparse) = row_sparse
  - round(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// rint
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(rint, cpu, mshadow_op::rint)
.describe(R"code(Returns element-wise rounded value to the nearest integer of the input.

.. note::
   - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
   - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.

Example::

   rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]

The storage type of ``rint`` output depends upon the input storage type:

   - rint(default) = default
   - rint(row_sparse) = row_sparse
   - rint(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// ceil
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(ceil, cpu, mshadow_op::ceil)
.describe(R"code(Returns element-wise ceiling of the input.

The ceil of the scalar x is the smallest integer i, such that i >= x.

Example::

   ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]

The storage type of ``ceil`` output depends upon the input storage type:

   - ceil(default) = default
   - ceil(row_sparse) = row_sparse
   - ceil(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// floor
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(floor, cpu, mshadow_op::floor)
.describe(R"code(Returns element-wise floor of the input.

The floor of the scalar x is the largest integer i, such that i <= x.

Example::

   floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]

The storage type of ``floor`` output depends upon the input storage type:

   - floor(default) = default
   - floor(row_sparse) = row_sparse
   - floor(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// trunc
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(trunc, cpu, mshadow_op::trunc)
.describe(R"code(Return the element-wise truncated value of the input.

The truncated value of the scalar x is the nearest integer i which is closer to
zero than x is. In short, the fractional part of the signed number x is discarded.

Example::

   trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]

The storage type of ``trunc`` output depends upon the input storage type:

   - trunc(default) = default
   - trunc(row_sparse) = row_sparse
   - trunc(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// fix
MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(fix, cpu, mshadow_op::fix)
.describe(R"code(Returns element-wise rounded value to the nearest \
integer towards zero of the input.

Example::

   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]

The storage type of ``fix`` output depends upon the input storage type:

   - fix(default) = default
   - fix(row_sparse) = row_sparse
   - fix(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// erf
MXNET_OPERATOR_REGISTER_UNARY(erf)
.add_alias("_npx_erf")
.describe(R"code(Returns element-wise gauss error function of the input.

Example::

   erf([0, -1., 10.]) = [0., -0.8427, 1.]

)code" ADD_FILELINE)
#if MSHADOW_USE_MKL == 1
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::MKL_Compute<mshadow_op::erf, mkl_func::erf>)
#else
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::erf>)
#endif    // MSHADOW_USE_MKL == 1
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_erf"});


MXNET_OPERATOR_REGISTER_BINARY(_backward_erf)
.set_attr<FCompute>("FCompute<cpu>",
                    ElemwiseBinaryOp::Compute<cpu, unary_bwd<mshadow_op::erf_grad>>);

// erfinv
MXNET_OPERATOR_REGISTER_UNARY(erfinv)
.add_alias("_npx_erfinv")
.describe(R"code(Returns element-wise inverse gauss error function of the input.

Example::

   erfinv([0, 0.5., -1.]) = [0., 0.4769, -inf]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::erfinv>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_erfinv"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_erfinv)
.set_attr<FCompute>("FCompute<cpu>",
                    ElemwiseBinaryOp::Compute<cpu, unary_bwd<mshadow_op::erfinv_grad>>);

// gamma
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(gamma, cpu, mshadow_op::gamma)
MXNET_ADD_SPARSE_OP_ALIAS(gamma)
.add_alias("_npx_gamma")
.describe(R"code(Returns the gamma function (extension of the factorial function \
to the reals), computed element-wise on the input array.

The storage type of ``gamma`` output is always dense

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gamma"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_gamma,
                                                  unary_bwd<mshadow_op::gamma_grad>);

// gammaln
MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(gammaln, cpu, mshadow_op::gammaln)
.add_alias("_npx_gammaln")
MXNET_ADD_SPARSE_OP_ALIAS(gammaln)
.describe(R"code(Returns element-wise log of the absolute value of the gamma function \
of the input.

The storage type of ``gammaln`` output is always dense

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gammaln"});

MXNET_OPERATOR_REGISTER_BINARY_WITH_SPARSE_CPU_DR(_backward_gammaln,
                                                  unary_bwd<mshadow_op::gammaln_grad>);

MXNET_OPERATOR_REGISTER_UNARY(logical_not)
.describe(R"code(Returns the result of logical NOT (!) function

Example:
  logical_not([-2., 0., 1.]) = [0., 1., 0.]

)code")
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::nt>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

}  // namespace op
}  // namespace mxnet
