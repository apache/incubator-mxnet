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
 * \file broadcast_reduce_op_customized.h
 * \brief Function definition of broadcast and reduce operators
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_BROADCAST_REDUCE_OP_CUSTOMIZED_H_
#define MXNET_OPERATOR_NUMPY_LINALG_BROADCAST_REDUCE_OP_CUSTOMIZED_H_

#include "../../tensor/broadcast_reduce_op.h"
#include "./broadcast_reduce_customized-inl.h"
#include <vector>

namespace mxnet {
namespace op {

template<typename xpu, typename Reducer, bool safe_acc,
         typename OP = op::mshadow_op::identity>
void ReduceAxesComputeImplWithReducer(const OpContext& ctx,
                                      const std::vector<TBlob>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<TBlob>& outputs,
                                      const mxnet::TShape& small,
                                      Reducer* reducer = nullptr) {
  using namespace mshadow;
  using namespace mshadow::expr;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(inputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, OType, {
    const TBlob in_data = inputs[0].reshape(src_shape);
    const TBlob out_data = outputs[0].reshape(dst_shape);
    BROADCAST_NDIM_SWITCH(dst_shape.ndim(), NDim, {
      size_t workspace_size = broadcast::ReduceWorkspaceSize(
          s, out_data.shape_, req[0], in_data.shape_);
      Tensor<xpu, 1, char> workspace =
          ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
      broadcast::ReduceWithReducer<Reducer, NDim, OType, OP, safe_acc>(
          s, out_data, req[0], workspace, in_data, reducer);
      // no normalization
    });
  });
}

template<int req, typename Mapper>
struct reduce_axes_backward_broadcast_wm {
  template<typename DType, typename OType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType *data,
                                  OType *out,
                                  DType *igrad,
                                  OType *ograd,
                                  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> in_shape,
                                  mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> out_shape,
                                  const uint32_t ndim,
                                  Mapper* OP = nullptr) {
    size_t in_stride = 1;
    size_t out_stride = 1;
    index_t idx = i;
    index_t out_idx = i;
    bool need_clean = !OP;
    for (int iter = ndim - 1; iter >= 0; --iter) {
      size_t dim_idx = idx % in_shape[iter];
      out_idx -= dim_idx * in_stride;
      if (out_shape[iter] != 1) {
        out_idx += dim_idx * out_stride;
      }
      idx /= in_shape[iter];
      in_stride *= in_shape[iter];
      out_stride *= out_shape[iter];
    }
    OP = OP ? OP : new Mapper();
    KERNEL_ASSIGN(igrad[i], req, DType(ograd[out_idx]) * OP->Map(data[i], DType(out[out_idx])));
    if (need_clean) {
      delete OP;
    }
  }
};

template<typename xpu, typename Mapper, bool normalize = false>
void ReduceAxesBackwardUseInOutImplWithMapper(const OpContext& ctx,
                                              const mxnet::TShape &small,
                                              const std::vector<TBlob>& inputs,
                                              const std::vector<OpReqType>& req,
                                              const std::vector<TBlob>& outputs,
                                              Mapper* OP = nullptr) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;

  mxnet::TShape src_shape, dst_shape;
  BroadcastReduceShapeCompact(outputs[0].shape_, small, &src_shape, &dst_shape);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, OType, {
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> in_shape;
      mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> out_shape;
      for (int i = 0; i < MXNET_SPECIAL_MAX_NDIM; ++i) {
        if (i < dst_shape.ndim()) {
          in_shape[i] = src_shape[i];
          out_shape[i] = dst_shape[i];
        } else {
          in_shape[i] = 1;
          out_shape[i] = 1;
        }
      }
      if (dst_shape.ndim() == 2) {
        Tensor<xpu, 2, OType> igrad =
          outputs[0].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
        Tensor<xpu, 2, DType> ograd =
          inputs[0].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
        Tensor<xpu, 2, OType> data =
          inputs[1].get_with_shape<xpu, 2, OType>(src_shape.get<2>(), s);
        Tensor<xpu, 2, DType> out =
          inputs[2].get_with_shape<xpu, 2, DType>(dst_shape.get<2>(), s);
        MXNET_REQ_TYPE_SWITCH(req[0], Req, {
          Kernel<reduce_axes_backward_broadcast_wm<Req, Mapper>, xpu>::Launch(
            s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
            in_shape, out_shape, src_shape.ndim(), OP);
        });
        if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
      } else {
        const int ndim = MXNET_SPECIAL_MAX_NDIM;
        Tensor<xpu, ndim, OType> igrad =
          outputs[0].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> ograd =
          inputs[0].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
        Tensor<xpu, ndim, OType> data =
          inputs[1].get_with_shape<xpu, ndim, OType>(src_shape.get<ndim>(), s);
        Tensor<xpu, ndim, DType> out =
          inputs[2].get_with_shape<xpu, ndim, DType>(dst_shape.get<ndim>(), s);
        MXNET_REQ_TYPE_SWITCH(req[0], Req, {
          Kernel<reduce_axes_backward_broadcast_wm<Req, Mapper>, xpu>::Launch(
            s, outputs[0].shape_.Size(), data.dptr_, out.dptr_, igrad.dptr_, ograd.dptr_,
            in_shape, out_shape, src_shape.ndim(), OP);
        });
        if (normalize) igrad /= scalar<OType>(src_shape.Size()/dst_shape.Size());
      }
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_BROADCAST_REDUCE_OP_CUSTOMIZED_H_
