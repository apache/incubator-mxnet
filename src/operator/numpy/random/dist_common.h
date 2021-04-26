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
 *  Copyright (c) 2015 by Contributors
 * \file dist_common.h
 * \brief Function definition of common functions for distributions
 * \with two parameters.
 */

#ifndef MXNET_OPERATOR_NUMPY_RANDOM_DIST_COMMON_H_
#define MXNET_OPERATOR_NUMPY_RANDOM_DIST_COMMON_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <string>
#include <vector>
#include "../../elemwise_op_common.h"
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

template <typename xpu>
void _copy(mshadow::Stream<xpu> *s, float *dst, float*src);

template <typename xpu>
void _copy(mshadow::Stream<xpu> *s, double *dst, double*src);


inline int FillShape(const mxnet::TShape &lshape, const mxnet::TShape &rshape,
                     const mxnet::TShape &oshape, mxnet::TShape *new_lshape,
                     mxnet::TShape *new_rshape, mxnet::TShape *new_oshape) {
  const int odim = std::max(oshape.ndim(), broadcast::MAX_DIM);
  *new_lshape = mxnet::TShape(odim, 1);
  *new_rshape = mxnet::TShape(odim, 1);
  *new_oshape = mxnet::TShape(odim, 1);
  int bl = oshape.ndim() - lshape.ndim();
  int br = oshape.ndim() - rshape.ndim();
  int j = 0;
  dim_t lprod = 1, rprod = 1, oprod = 1;
  for (int i = 0; i < oshape.ndim(); ++i) {
    dim_t l = 1;
    dim_t r = 1;
    dim_t o = oshape[i];
    if (i >= bl) l = lshape[i - bl];
    if (i >= br) r = rshape[i - br];
    if ((lprod != rprod || lprod != oprod || l != r || l != o) &&
        (lprod * l > 1 || rprod * r > 1 || oprod * o > 1)) {
      (*new_lshape)[j] = lprod;
      (*new_rshape)[j] = rprod;
      (*new_oshape)[j] = oprod;
      lprod = rprod = oprod = 1;
      ++j;
    }
    lprod *= l;
    rprod *= r;
    oprod *= o;
  }
  if (lprod > 1 || rprod > 1 || oprod > 1) {
    (*new_lshape)[j] = lprod;
    (*new_rshape)[j] = rprod;
    (*new_oshape)[j] = oprod;
    ++j;
  }
  if (j <= broadcast::MAX_DIM) {
    BROADCAST_NDIM_SWITCH(j, NDim, {
      new_lshape->assign(new_lshape->begin(), new_lshape->begin() + NDim);
      new_rshape->assign(new_rshape->begin(), new_rshape->begin() + NDim);
      new_oshape->assign(new_oshape->begin(), new_oshape->begin() + NDim);
    });
  } else {
    LOG(FATAL) << "Too many broadcast dimensions with operands " << lshape
               << " " << rshape;
  }
  return j;
}

inline void CheckBroadcastable(const mxnet::TShape &from,
                               const mxnet::TShape &to) {
  const int bl = to.ndim() - from.ndim();
  const int br = 0;
  for (int i = 0; i < to.ndim(); ++i) {
    dim_t l = 1, r = 1;
    if (i >= bl) l = from[i - bl];
    if (i >= br) r = to[i - br];
    if (!mxnet::dim_size_is_known(l) || !mxnet::dim_size_is_known(r)) continue;
    if (l != r) {
      // Make it compatible with NumPy.
      // For example, (2, 3) cannot broadcast to (2, 0, 3), but (1, 3) can
      // broadcast to (2, 0, 3).
      CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes " << from
          << " " << to;
    }
  }
}

inline void InferBroadcastShape(const mxnet::TShape &lhs,
                                const mxnet::TShape &rhs,
                                mxnet::TShape *out_ptr) {
  mxnet::TShape &out = (*out_ptr);
  const int bl = out.ndim() - lhs.ndim();
  const int br = out.ndim() - rhs.ndim();
  for (int i = 0; i < out.ndim(); ++i) {
    dim_t l = 1, r = 1;
    if (i >= bl) l = lhs[i - bl];
    if (i >= br) r = rhs[i - br];
    if (!mxnet::dim_size_is_known(l) || !mxnet::dim_size_is_known(r)) continue;
    if (l != r) {
      // Make it compatible with NumPy.
      // For example, (2, 3) cannot broadcast to (2, 0, 3), but (1, 3) can
      // broadcast to (2, 0, 3).
      CHECK(l == 1 || r == 1)
          << "operands could not be broadcast together with shapes " << lhs
          << " " << rhs;
      out[i] = (l == 1 ? r : l);
    } else {
      out[i] = l;
    }
  }
}

template <typename DistParam>
inline bool TwoparamsDistOpShape(const nnvm::NodeAttrs &attrs,
                                 std::vector<TShape> *in_attrs,
                                 std::vector<TShape> *out_attrs) {
  // The inferShape function for sampling Ops has two modes: Concat/Broadcast,
  // if size[0] == -2, the Concat schema will be selected:
  // output_size = (size[1:],) + broadcast(param1.shape, param2.shape)
  // otherwise output_size = broadcast(param1.shape, param2.shape, size)
  const DistParam &param = nnvm::get<DistParam>(attrs.parsed);
  // Variable indicating the mode.
  bool concat_mode = false;
  // Variable storing the info from `size` parameter.
  std::vector<dim_t> oshape_vec;
  if (param.size.has_value()) {
    // Size declared.
    const auto &size = param.size.value();
    index_t head = size[0];
    if (head == -2) {
      concat_mode = true;
    } else {
      oshape_vec.emplace_back(head);
    }
    for (int i = 1; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
    // If under the broadcast mode, `size` is equivalent to the final output_size.
    if (!concat_mode) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
    for (size_t input_idx = 0; input_idx < in_attrs->size(); input_idx++) {
      CheckBroadcastable((*in_attrs)[input_idx], (*out_attrs)[0]);
      }
    }
  }
  // Under concat mode, or `size` is not declared.
  if (concat_mode || (!param.size.has_value())) {
    // broadcast(param1.shape, param2.shape).
    mxnet::TShape param_broadcast_shape;
    if (in_attrs->size() == 2U) {
      // Both params from ndarray.
      mxnet::TShape &param1 = (*in_attrs)[0];
      mxnet::TShape &param2 = (*in_attrs)[1];
      mxnet::TShape out(std::max(param1.ndim(), param2.ndim()), -1);
      InferBroadcastShape(param1, param2, &out);
      param_broadcast_shape = out;
    } else if (in_attrs->size() == 1U) {
      // One param from ndarray.
      param_broadcast_shape = in_attrs->at(0);
    } else if (in_attrs->size() == 0) {
      // Two scalar case.
      param_broadcast_shape = TShape(0, -1);
    }
    if (concat_mode) {
      for (int i = 0; i < param_broadcast_shape.ndim(); ++i) {
        oshape_vec.emplace_back(param_broadcast_shape[i]);
      }
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
    } else {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, param_broadcast_shape);
    }
  }
  if (out_attrs->size() == 2U) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, out_attrs->at(0));
  }
  return true;
}

template <typename DistParam>
inline bool UnaryDistOpShape(const nnvm::NodeAttrs &attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  const DistParam &param = nnvm::get<DistParam>(attrs.parsed);
  if (param.size.has_value()) {
    // Size declared.
    std::vector<dim_t> oshape_vec;
    const auto &size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
    for (size_t input_idx = 0; input_idx < in_attrs->size(); input_idx++) {
      CheckBroadcastable((*in_attrs)[input_idx], (*out_attrs)[0]);
    }
  } else {
    if (in_attrs->size() == 1U) {
      // One param from ndarray.
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0))
    } else {
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(0, -1))
    }
  }
  return shape_is_known(out_attrs->at(0));
}


// Infer Shape function for sample_n Op.
// i.e. output_shape = (shape,) + broadcast(param1.shape, param2.shape)
template <typename DistParam>
inline bool TwoparamsDistOpConcatShape(const nnvm::NodeAttrs &attrs,
                                       std::vector<TShape> *in_attrs,
                                       std::vector<TShape> *out_attrs) {
  const DistParam &param = nnvm::get<DistParam>(attrs.parsed);
  // broadcast(param1.shape, param2.shape).
  mxnet::TShape param_broadcast_shape;
  if (in_attrs->size() == 2U) {
      // Both params from ndarray.
      mxnet::TShape &param1 = (*in_attrs)[0];
      mxnet::TShape &param2 = (*in_attrs)[1];
      mxnet::TShape out(std::max(param1.ndim(), param2.ndim()), -1);
      InferBroadcastShape(param1, param2, &out);
      param_broadcast_shape = out;
    } else if (in_attrs->size() == 1U) {
      // One param from ndarray.
      param_broadcast_shape = in_attrs->at(0);
    } else if (in_attrs->size() == 0) {
      // Two scalar case.
      param_broadcast_shape = TShape(0, -1);
    }
  if (param.size.has_value()) {
    // Size declared.
    std::vector<dim_t> oshape_vec;
    const auto &size = param.size.value();
    for (int i = 0; i < size.ndim(); ++i) {
      oshape_vec.emplace_back(size[i]);
    }
    for (int i = 0; i < param_broadcast_shape.ndim(); ++i) {
      oshape_vec.emplace_back(param_broadcast_shape[i]);
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape(oshape_vec));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, param_broadcast_shape);
  }
  if (out_attrs->size() == 2U) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, out_attrs->at(0));
  }
  return true;
}

template<typename xpu, int ndim, typename DType>
inline void CommonReparamBackwardImpl(const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs,
                                      const mxnet::TShape& new_lshape,
                                      const mxnet::TShape& new_rshape,
                                      const mxnet::TShape& new_oshape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob lgrad = outputs[0].reshape(new_lshape);
  const TBlob rgrad = outputs[1].reshape(new_rshape);
  const TBlob ograd = inputs[0].reshape(new_oshape);
  // Mean
  const TBlob lhs = inputs[2].reshape(new_lshape);
  // Scale
  const TBlob rhs = inputs[3].reshape(new_rshape);
  const TBlob samples = inputs[4].reshape(new_oshape);
  const TBlob noise = inputs[5].reshape(new_oshape);
  size_t workspace_size_l = ReduceWorkspaceSize(
    s, lgrad.shape_, req[0], ograd.shape_, lhs.shape_, rhs.shape_);
  size_t workspace_size_r = ReduceWorkspaceSize(
    s, rgrad.shape_, req[1], ograd.shape_, lhs.shape_, rhs.shape_);
  size_t workspace_size = std::max(workspace_size_l, workspace_size_r);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
#if !defined(__CUDACC__)
  Reduce<red::sum, ndim, DType, op::mshadow_op::identity>(
    s, lgrad, req[0], workspace, ograd);
  Reduce<red::sum, ndim, DType, op::mshadow_op::mul, op::mshadow_op::left>(
    s, rgrad, req[1], workspace, ograd, noise, rhs);
#else
  RTCReduce(ctx, lgrad, req[0], workspace, ograd, "red::sum{}", ndim, "identity");
  RTCReduce(ctx, rgrad, req[1], workspace, ograd, noise, rhs, "red::sum{}", ndim, "mul", "left");
#endif
}

template<typename xpu, int ndim, typename DType>
inline void CommonScalarReparamBackwardImpl(const OpContext& ctx,
                                            const std::vector<TBlob>& inputs,
                                            const std::vector<OpReqType>& req,
                                            const std::vector<TBlob>& outputs,
                                            const mxnet::TShape& new_ishape,
                                            const mxnet::TShape& new_oshape,
                                            const bool loc_is_tensor = false) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace broadcast;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob igrad = outputs[0].reshape(new_ishape);
  // inputs: [grad_from_samples, grad_from_noise(invisible), input_tensor,
  //          samples, noise]
  const TBlob ograd = inputs[0].reshape(new_oshape);
  const TBlob itensor = inputs[2].reshape(new_ishape);
  const TBlob samples = inputs[3].reshape(new_oshape);
  const TBlob noise = inputs[4].reshape(new_oshape);
  size_t workspace_size =
    ReduceWorkspaceSize(s, igrad.shape_, req[0], ograd.shape_);
  Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
#if !defined(__CUDACC__)
  if (loc_is_tensor) {
    Reduce<red::sum, ndim, DType, op::mshadow_op::identity>(s, igrad, req[0],
                                                            workspace, ograd);
  } else {
    Reduce<red::sum, ndim, DType, op::mshadow_op::mul, op::mshadow_op::left>(
      s, igrad, req[0], workspace, ograd, noise, noise);
  }
#else
  if (loc_is_tensor) {
    RTCReduce(ctx, igrad, req[0], workspace, ograd, "red::sum{}", ndim, "identity");
  } else {
    RTCReduce(ctx, igrad, req[0], workspace, ograd, noise, noise, "red::sum{}",
              ndim, "mul", "left");
  }
#endif
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_RANDOM_DIST_COMMON_H_ */
