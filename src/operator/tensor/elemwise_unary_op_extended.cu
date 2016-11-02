/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_unary_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
// sin
NNVM_REGISTER_OP(sin)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::sin>);

NNVM_REGISTER_OP(_backward_sin)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::sin_grad> >);

// cos
NNVM_REGISTER_OP(cos)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::cos>);

NNVM_REGISTER_OP(_backward_cos)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::cos_grad> >);

// tan
NNVM_REGISTER_OP(tan)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::tan>);

NNVM_REGISTER_OP(_backward_tan)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::tan_grad> >);

// arcsin
NNVM_REGISTER_OP(arcsin)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::arcsin>);

NNVM_REGISTER_OP(_backward_arcsin)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::arcsin_grad> >);

// arccos
NNVM_REGISTER_OP(arccos)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::arccos>);

NNVM_REGISTER_OP(_backward_arccos)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::arccos_grad> >);

// arctan
NNVM_REGISTER_OP(arctan)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::arctan>);

NNVM_REGISTER_OP(_backward_arctan)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::arctan_grad> >);

// degrees
NNVM_REGISTER_OP(degrees)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::degrees>);

NNVM_REGISTER_OP(_backward_degrees)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::degrees_grad> >);

// cosh
NNVM_REGISTER_OP(cosh)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::cosh>);

NNVM_REGISTER_OP(_backward_cosh)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::cosh_grad> >);

// sinh
NNVM_REGISTER_OP(sinh)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::sinh>);

NNVM_REGISTER_OP(_backward_sinh)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::sinh_grad> >);

// tanh
NNVM_REGISTER_OP(tanh)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::tanh>);

NNVM_REGISTER_OP(_backward_tanh)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::tanh_grad> >);

// arcsinh
NNVM_REGISTER_OP(arcsinh)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::arcsinh>);

NNVM_REGISTER_OP(_backward_arcsinh)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::arcsinh_grad> >);

// arccosh
NNVM_REGISTER_OP(arccosh)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::arccosh>);

NNVM_REGISTER_OP(_backward_arccosh)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::arccosh_grad> >);

// arctanh
NNVM_REGISTER_OP(arctanh)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::arctanh>);

NNVM_REGISTER_OP(_backward_arctanh)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::arctanh_grad> >);

// gamma
NNVM_REGISTER_OP(gamma)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::gamma>);

NNVM_REGISTER_OP(_backward_gamma)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::gamma_grad> >);

// gammaln
NNVM_REGISTER_OP(gammaln)
.set_attr<FCompute>("FCompute<gpu>", UnaryCompute<gpu, mshadow_op::gammaln>);

NNVM_REGISTER_OP(_backward_gammaln)
.set_attr<FCompute>("FCompute<gpu>", BinaryCompute<gpu, unary_bwd<mshadow_op::gammaln_grad> >);

}  // namespace op
}  // namespace mxnet
