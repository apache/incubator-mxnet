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
 * \file elementwise_unary_op.h
 * \brief Function definition of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <climits>
#include <string>
#include "./cast_storage-inl.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../elemwise_op_common.h"
#include "../../common/utils.h"
#include "../../ndarray/ndarray_function.h"

#if MSHADOW_USE_MKL == 1
#include "../mkl_functions-inl.h"
#endif  // MSHADOW_USE_MKL == 1

namespace mxnet {
namespace op {

namespace {

/*! \brief Infer the output storage geometry
 * \return boolean signifying whether the proper storage geometry was initialized
 */
template<int n_in, int n_out>
bool InitStorageGeometry(const nnvm::NodeAttrs& attrs,
                                const std::vector<NDArray>& inputs,
                                const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), static_cast<size_t>(n_in))
    << " in operator " << attrs.name;
  CHECK_EQ(outputs.size(), static_cast<size_t>(n_out))
    << " in operator " << attrs.name;
  static_assert(n_in > 0 && n_out > 0, "Invalid input and/or output count values");
  const mxnet::TShape& isshape = inputs[0].storage_shape();
  if (!shape_is_none(isshape)) {
    NDArray *output = nullptr;
    for (size_t i = 0, n = inputs.size(); i < n; ++i) {
      const NDArray &input = inputs[i];
      if (i < n_out) {
        output = const_cast<NDArray *>(&outputs[i]);
      }
      CHECK_EQ(output->shape(), inputs[i].shape());
      CHECK_EQ(output->storage_type(), input.storage_type());
      CHECK_EQ(output->aux_shapes().size(), input.aux_shapes().size());
      mxnet::ShapeVector aux_shapes;
      const size_t aux_shape_count = input.aux_shapes().size();
      aux_shapes.reserve(aux_shape_count);
      for (size_t j = 0; j < aux_shape_count; ++j) {
        aux_shapes.emplace_back(input.aux_shape(j));
      }
      output->CheckAndAlloc(aux_shapes);
      DCHECK_EQ(output->storage_shape(), input.storage_shape());
    }
    return true;
  }
  if (isshape.ndim() > 0 && !isshape.Size()
    && inputs[0].storage_type() != kDefaultStorage) {
    return true;  // 0% density
  } else {
    CHECK(false);  // implement when necessary
  }
  return false;
}

/*! \brief Copy blob data */
template<typename xpu>
void inline CopyBlob(mshadow::Stream<xpu> *s,
                            const TBlob *dest_blob,
                            const OpReqType reqi,
                            const TBlob& src_blob) {
  CHECK_EQ(src_blob.type_flag_, dest_blob->type_flag_);
  CHECK_EQ(src_blob.shape_, dest_blob->shape_);
  MSHADOW_TYPE_SWITCH(src_blob.type_flag_, DType, {
    // Check if the pointers are the same (in-place operation needs no copy)
    if (reqi != kNullOp && src_blob.dptr<DType>() != dest_blob->dptr<DType>()) {
      mshadow::Copy(dest_blob->FlatTo1D<xpu, DType>(s), src_blob.FlatTo1D<xpu, DType>(s), s);
    }
  });
}

/*! \brief Allocate geometry-related blob data for sparse tensors
 * \param dest Destination sparse NDArray
 * \param clone_from sparse NDArray from which to clone storage attributes
 */
void inline AllocateGeometry(const NDArray *dest,
                             const OpReqType req,
                             const NDArray* clone_from = nullptr) {
  if (req != kNullOp) {
    if (clone_from) {
      const mxnet::TShape& ishape = clone_from->storage_shape();
      dest->CheckAndAllocData(ishape);
      CHECK_EQ(dest->storage_type(), clone_from->storage_type());
      for (size_t i = 0, n = clone_from->aux_shapes().size(); i < n; ++i) {
        dest->CheckAndAllocAuxData(i, clone_from->aux_shape(i));
      }
      DCHECK_EQ(dest->aux_shapes().size(), clone_from->aux_shapes().size());
    } else {
      for (size_t i = 0, n = dest->aux_shapes().size(); i < n; ++i) {
        dest->CheckAndAllocAuxData(i, dest->aux_shape(i));
      }
      dest->CheckAndAllocData(dest->storage_shape());
    }
  }
}

/*! \brief Copy the geometry-related blobs (row sparse indexes, etc.) */
template<typename xpu>
inline void CopyGeometryBlobs(mshadow::Stream<xpu> *s,
                                     const NDArray *dest,
                                     const OpReqType reqi,
                                     const NDArray &src) {
  CHECK_EQ(src.aux_shapes().size(), dest->aux_shapes().size());
  // My assumption is that the geometry blobs are not large enough to justify an omp loop here,
  // since the thread synchronization calls for each fork will take longer
  // than copying a few floats
  for (size_t i = 0, n = src.aux_shapes().size(); i < n; ++i) {
    const TBlob src_blob = src.aux_data(i);
    const TBlob dest_blob = dest->aux_data(i);
    CopyBlob<xpu>(s, &dest_blob, reqi, src_blob);
  }
}

}  // namespace

class OpBase {
 protected:
  /*! \brief simple kernel to set to a scalar value of arbitrary type */
  template<int req>
  using set_to_scalar = mxnet_op::op_with_req<mshadow_op::identity, req>;


  /*! \brief Generic copy NDArray */
  template<typename xpu>
  static inline void CopyNDArray(mshadow::Stream<xpu> *s,
                                 const NDArray *dest,
                                 const OpReqType reqi,
                                 const NDArray& src) {
    DCHECK_EQ(dest->storage_type(), src.storage_type());
    AllocateGeometry(dest, reqi, &src);
    CopyGeometryBlobs(s, dest, reqi, src);
    CopyBlob(s, &dest->data(), reqi, src.data());
  }

  /*! \brief Map NDArray vectors to TBlob vectors and pass to compute function */
  template<typename xpu, typename FComputer>
  static inline void MapToFCompute(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs,
                                   FComputer computer) {
    std::vector<TBlob> in_blobs, out_blobs;
    in_blobs.reserve(inputs.size());
    out_blobs.reserve(outputs.size());
    for (size_t i = 0, n = inputs.size(); i < n; ++i) {
      in_blobs.emplace_back(inputs[i].data());
    }
    for (size_t i = 0, n = outputs.size(); i < n; ++i) {
      out_blobs.emplace_back(outputs[i].data());
    }
    computer(attrs, ctx, in_blobs, req, out_blobs);
  }

  /*! \brief Keep row shape[0] dimension and gather the remaining dimensions in location shape[1] */
  template<typename DType, typename xpu>
  static inline mshadow::Tensor<xpu, 2, DType> AsRowise2D(mshadow::Stream<xpu> *s,
                                                          const TBlob& blob) {
    const size_t dim = blob.shape_.ndim();
    if (dim) {
      mxnet::TShape shape({blob.shape_[0], 1});
      for (size_t i = 1; i < dim; ++i) {
        shape[1] *= blob.shape_[i];
      }
      return mshadow::Tensor<xpu, 2, DType>(
        blob.dptr<DType>(), mshadow::Shape2(shape[0], shape[1]), s);
    }
    return mshadow::Tensor<xpu, 2, DType>();
  }

  /*! \brief Fill dense output block with a single scalar value */
  template<typename DType>
  static inline void FillDense(mshadow::Stream<cpu> *s,
                               const size_t size,
                               const DType val,
                               const OpReqType req,
                               DType *out) {
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      mxnet_op::Kernel<OpBase::set_to_scalar<Req>, cpu>::Launch(s, size, out, val);
    });
  }
};  // OpBase

/*! \brief Unary operator class */
class UnaryOp : public OpBase {
 public:
  /*! \brief Map NDArray vectors to TBlob vectors and pass to compute function */
  template<typename xpu, typename FComputer>
  static inline void MapToFCompute(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs,
                                   FComputer computer) {
    InitStorageGeometry<1, 1>(attrs, inputs, outputs);
    CHECK_EQ(inputs.size(), outputs.size());  // need to figure out what to do for binary type
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage);
    CHECK_EQ(inputs[0].storage_type(), outputs[0].storage_type());
    AllocateGeometry(&outputs[0], req[0], &inputs[0]);
    CopyGeometryBlobs<xpu>(ctx.get_stream<xpu>(), &outputs[0], req[0], inputs[0]);
    outputs[0].CheckAndAllocData(inputs[0].storage_shape());
    if (inputs[0].storage_shape().Size()) {
      OpBase::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, computer);
    }
  }

  template<typename OP>
  static void Compute_(const nnvm::NodeAttrs& attrs,
                       mshadow::Stream<cpu>* s,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        if (inputs[0].Size() != 0) {
          mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, cpu>::Launch(
            s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
        }
      });
    });
  }

  template<typename xpu, typename OP>
  static void Compute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    Compute_<OP>(attrs, s, inputs, req, outputs);
  }

  template<typename xpu, typename OP>
  static void ComputeMixedType(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

    if (mxnet::common::is_float(inputs[0].type_flag_)) {
      UnaryOp::Compute<xpu, OP>(attrs, ctx, inputs, req, outputs);
    } else {
      MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        MXNET_INT_TYPE_SWITCH(inputs[0].type_flag_, IType, {
          MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
            if (inputs[0].Size() != 0) {
              mxnet_op::Kernel<mxnet_op::mixed_type_unary_op<OP, Req>, xpu>::Launch(
                s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<IType>());
            }
          });
        });
      });
    }
  }

  template<typename xpu, typename OP>
  static void ComputeInt(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    MXNET_INT_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        if (inputs[0].Size() != 0) {
          mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
            s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
        }
      });
    });
  }

  template<typename xpu, typename OP>
  static void ComputeLogic(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        if (inputs[0].Size() != 0) {
          mxnet_op::Kernel<mxnet_op::op_with_req<OP, Req>, xpu>::Launch(
              s, inputs[0].Size(), outputs[0].dptr<bool>(), inputs[0].dptr<DType>());
        }
      });
    });
  }

  template<typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a sparse input storage type";
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a sparse output storage type";
    if (inputs[0].storage_shape().Size()) {
      MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
    }
  }

#if MSHADOW_USE_MKL == 1
  template<typename OP, typename MKL_OP>
  static void MKL_Compute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
    if (req[0] == kNullOp)  return;
    auto type_flag = inputs[0].type_flag_;
    size_t input_size = inputs[0].Size();
    if ((req[0] == kWriteTo || req[0] == kWriteInplace) &&
        mkl_func::check_size(input_size) &&
        mkl_func::check_type(type_flag)) {
      // set DType as float or double according to type_flag
      MSHADOW_SGL_DBL_TYPE_SWITCH(type_flag, DType, {
        MKL_OP::Vectorize(input_size, inputs[0].dptr<DType>(), outputs[0].dptr<DType>());
      });
    } else {
      Compute<cpu, OP>(attrs, ctx, inputs, req, outputs);
    }
  }

  template<typename OP, typename MKL_OP>
  static void MKL_ComputeEx(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U)
      << "Invalid input, only one input is allowed";
    CHECK_EQ(outputs.size(), 1U)
      << "Invalid output, only one output is allowed";
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a sparse input storage type";
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a sparse output storage type";
    if (inputs[0].storage_shape().Size()) {
      MapToFCompute<cpu>(attrs, ctx, inputs, req, outputs, MKL_Compute<OP, MKL_OP>);
    }
  }
#endif

  template<typename xpu>
  static void IdentityCompute(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    switch (req[0]) {
      case kWriteTo:
        CHECK_EQ(outputs[0].dev_mask(), inputs[0].dev_mask());
        mxnet_op::copy(ctx.get_stream<xpu>(), outputs[0], inputs[0]);
        break;
      case kAddTo: {
          Stream<xpu> *s = ctx.get_stream<xpu>();
          MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, DType, {
            mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, kAddTo>, xpu>::Launch(
              s, inputs[0].Size(), outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
          });
        }
        break;
      case kWriteInplace:
// cannot check if ptrs are the same for MKLDNN because we may have
// created copies of input when reordering. WriteInPlace will still write to original array
#if MXNET_USE_ONEDNN == 0
        CHECK_EQ(inputs[0].dptr_, outputs[0].dptr_);
#endif
        break;
      case kNullOp:
        break;
    }
  }

  template<typename xpu>
  static void IdentityComputeEx(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    const auto in_stype = inputs[0].storage_type();
    const auto out_stype = outputs[0].storage_type();
    if (in_stype == out_stype && (in_stype == kRowSparseStorage || in_stype == kCSRStorage)) {
      MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, IdentityCompute<xpu>);
    } else {
      LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
    }
  }

  template<typename xpu>
  static void IdentityComputeFirstItemEx(const nnvm::NodeAttrs& attrs,
                                         const OpContext& ctx,
                                         const std::vector<NDArray>& inputs,
                                         const std::vector<OpReqType>& req,
                                         const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    const auto lhs_stype = inputs[0].storage_type();
    const auto out_stype = outputs[0].storage_type();
    bool supported_stype = lhs_stype == kRowSparseStorage || lhs_stype == kCSRStorage;
    if (supported_stype && lhs_stype == out_stype) {
      // csr, _ -> csr, or rsp, _ -> rsp
      OpBase::CopyNDArray(ctx.get_stream<xpu>(), &outputs[0], req[0], inputs[0]);
    } else if (supported_stype && out_stype == kDefaultStorage) {
     // csr/rsp, _ -> dns
      CastStorageComputeImpl<xpu>(ctx, inputs[0], outputs[0]);
    } else {
      LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
    }
  }
};

/*! \brief Map legacy unary_bwd to backward_grad */
template<typename GRAD_OP>
using unary_bwd = ::mxnet::op::mxnet_op::backward_grad_tuned<GRAD_OP>;

struct CastParam : public dmlc::Parameter<CastParam> {
  // use int for enumeration
  int dtype;
  DMLC_DECLARE_PARAMETER(CastParam) {
    DMLC_DECLARE_FIELD(dtype)
    MXNET_ADD_ALL_TYPES_WITH_BOOL
    .describe("Output data type.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream dtype_s;
    dtype_s << dtype;
    (*dict)["dtype"] = MXNetTypeWithBool2String(dtype);
  }
};

inline bool CastType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  const CastParam& param = nnvm::get<CastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  return (*in_attrs)[0] != -1;
}

template<typename xpu>
void CastCompute(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, DstDType, {
    Tensor<xpu, 1, DstDType> out = outputs[0].FlatTo1D<xpu, DstDType>(s);
    MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, SrcDType, {
      Tensor<xpu, 1, SrcDType> data = inputs[0].FlatTo1D<xpu, SrcDType>(s);
      if ((outputs[0].type_flag_ != inputs[0].type_flag_ ||
          req[0] != kWriteInplace) && outputs[0].Size() != 0) {
        Assign(out, req[0], tcast<DstDType>(data));
      }
    });
  });
}

struct HardSigmoidParam : public dmlc::Parameter<HardSigmoidParam> {
  real_t alpha;
  real_t beta;
  DMLC_DECLARE_PARAMETER(HardSigmoidParam) {
    DMLC_DECLARE_FIELD(alpha)
    .set_default(0.2)
    .describe("Slope of hard sigmoid");
    DMLC_DECLARE_FIELD(beta)
    .set_default(0.5)
    .describe("Bias of hard sigmoid.");
  }
};

template<int req>
struct hard_sigmoid_forward {
  template<typename DType>
    MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const DType* in_data,
                                    const real_t alpha, const real_t beta) {
      DType result = DType(alpha * in_data[i] + beta);
      result = (DType(1) < result) ? DType(1) : result;
      result = (DType(0) > result) ? DType(0) : result;
      KERNEL_ASSIGN(out_data[i], req, result);
    }
};

template<int req>
struct hard_sigmoid_backward {
  template<typename DType>
    MSHADOW_XINLINE static void Map(index_t i, DType* in_grad, const DType* in_data,
                                    const DType* out_grad, const real_t alpha, const real_t beta) {
      DType out_val = DType(alpha) * in_data[i] + DType(beta);
      DType grad = (out_val > DType(0) && out_val < DType(1)) ?
                   (out_grad[i] * DType(alpha)) : DType(0);
      KERNEL_ASSIGN(in_grad[i], req, grad);
    }
};


template<typename xpu>
void HardSigmoidForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  CHECK(req[0] != kNullOp);
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const HardSigmoidParam& param = nnvm::get<HardSigmoidParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_REAL_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<hard_sigmoid_forward<req_type>, xpu>::Launch(
        s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
        param.alpha, param.beta);
    });
  });
}

template<typename xpu>
void HardSigmoidBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data = inputs[1];
  const TBlob& in_grad = outputs[0];
  const HardSigmoidParam& param = nnvm::get<HardSigmoidParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_REAL_TYPE_SWITCH(in_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<hard_sigmoid_backward<req_type>, xpu>::Launch(
        s, in_grad.Size(), in_grad.dptr<DType>(), in_data.dptr<DType>(),
        out_grad.dptr<DType>(), param.alpha, param.beta);
    });
  });
}

struct ReshapeLikeParam : public dmlc::Parameter<ReshapeLikeParam> {
  dmlc::optional<int> lhs_begin, rhs_begin, lhs_end, rhs_end;
  DMLC_DECLARE_PARAMETER(ReshapeLikeParam) {
    DMLC_DECLARE_FIELD(lhs_begin)
        .set_default(dmlc::optional<int>())
        .describe(
            "Defaults to 0. "
            "The beginning index along which the lhs dimensions are to be "
            "reshaped. Supports negative indices.");
    DMLC_DECLARE_FIELD(lhs_end)
        .set_default(dmlc::optional<int>())
        .describe("Defaults to None. "
                  "The ending index along which the lhs dimensions are to be "
                  "used for reshaping. Supports negative indices.");
    DMLC_DECLARE_FIELD(rhs_begin)
        .set_default(dmlc::optional<int>())
        .describe("Defaults to 0. "
                  "The beginning index along which the rhs dimensions are to "
                  "be used for "
                  "reshaping. Supports negative indices.");
    DMLC_DECLARE_FIELD(rhs_end)
        .set_default(dmlc::optional<int>())
        .describe("Defaults to None. "
                  "The ending index along which the rhs dimensions are to be "
                  "used for reshaping. Supports negative indices.");
  }
};

struct AroundParam : public dmlc::Parameter<AroundParam> {
  int decimals;
  DMLC_DECLARE_PARAMETER(AroundParam) {
    DMLC_DECLARE_FIELD(decimals)
      .set_default(0)
      .describe("Number of decimal places to round to.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream decimals_s;
    decimals_s << decimals;
    (*dict)["decimal"] = decimals_s.str();
  }
};

template<int req>
struct around_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const DType* in_data,
                                  const int decimals) {
    int d = 0;
    DType temp = in_data[i];
    DType roundtemp;
    while (d != decimals) {
      if (decimals > 0) {
        d++;
        temp *= 10;
      } else {
        d--;
        temp /= 10;
      }
    }
    roundtemp = (DType)round(static_cast<double>(temp));
    // If temp is x.5 and roundtemp is odd number, decrease or increase roundtemp by 1.
    // For example, in numpy, around(0.5) should be 0 but in c, round(0.5) is 1.
    if (roundtemp - temp == 0.5 && (static_cast<index_t>(roundtemp)) % 2 != 0) {
      roundtemp -= 1;
    } else if (temp - roundtemp == 0.5 && (static_cast<index_t>(roundtemp)) % 2 != 0) {
      roundtemp += 1;
    }
    while (d != 0) {
      if (roundtemp == 0) {
        break;
      }
      if (decimals > 0) {
        d--;
        roundtemp /= 10;
      } else {
        d++;
        roundtemp *= 10;
      }
    }
    KERNEL_ASSIGN(out_data[i], req, roundtemp);
  }
};

template<typename xpu>
void AroundOpForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const AroundParam& param = nnvm::get<AroundParam>(attrs.parsed);
  using namespace mxnet_op;
  // if the type is uint8, int8, int32 or int64 and decimals is greater than 0
  // we simply return the number back.
  if (in_data.type_flag_ >= mshadow::kUint8 && in_data.type_flag_ <= mshadow::kInt64 \
     && param.decimals > 0) {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      Kernel<mshadow_op::identity_with_cast, xpu>::Launch(
        s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>());
    });
  } else {
    MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<around_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          param.decimals);
      });
    });
  }
}

struct NumpyNanToNumParam : public dmlc::Parameter<NumpyNanToNumParam> {
  bool copy;
  double nan;
  dmlc::optional<double> posinf, neginf;
  DMLC_DECLARE_PARAMETER(NumpyNanToNumParam) {
    DMLC_DECLARE_FIELD(copy)
    .set_default(true)
    .describe("Whether to create a copy of `x` (True) or to replace values"
              "in-place (False). The in-place operation only occurs if"
              "casting to an array does not require a copy."
              "Default is True.");
    DMLC_DECLARE_FIELD(nan)
    .set_default(0.0)
    .describe("Value to be used to fill NaN values. If no value is passed"
              "then NaN values will be replaced with 0.0.");
    DMLC_DECLARE_FIELD(posinf)
    .set_default(dmlc::optional<double>())
    .describe("Value to be used to fill positive infinity values."
              "If no value is passed then positive infinity values will be"
              "replaced with a very large number.");
    DMLC_DECLARE_FIELD(neginf)
    .set_default(dmlc::optional<double>())
    .describe("Value to be used to fill negative infinity values."
              "If no value is passed then negative infinity values"
              "will be replaced with a very small (or negative) number.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream copy_s, nan_s, posinf_s, neginf_s;
    copy_s << copy;
    nan_s << nan;
    posinf_s << posinf;
    neginf_s << neginf;
    (*dict)["copy"] = copy_s.str();
    (*dict)["nan"] = nan_s.str();
    (*dict)["posinf"] = posinf_s.str();
    (*dict)["neginf"] = neginf_s.str();
  }
};

template<int req>
struct nan_to_num_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* out_data,
                                  const DType* in_data,
                                  const DType nan,
                                  const DType posinf,
                                  const DType neginf) {
    DType val = in_data[i];
    if (mshadow_op::IsNan<DType>(val))  val = nan;
    if (val > 0 && mshadow_op::IsInf(val))  val = posinf;
    if (val < 0 && mshadow_op::IsInf(val))  val = neginf;
    KERNEL_ASSIGN(out_data[i], req, val);
  }
};

template<typename xpu>
void NumpyNanToNumOpForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mxnet;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const NumpyNanToNumParam& param = nnvm::get<NumpyNanToNumParam>(attrs.parsed);
  using namespace mxnet_op;

  if (!common::is_float(in_data.type_flag_) && req[0] == kWriteInplace) return;
  if (!common::is_float(in_data.type_flag_)) {
    copy(s, out_data, in_data);
    return;
  }

  MSHADOW_REAL_TYPE_SWITCH(out_data.type_flag_, DType, {
    DType defaultnan = static_cast<DType>(param.nan);
    DType posinf;
    DType neginf;
    if (param.posinf.has_value()) {
      posinf = static_cast<DType>(param.posinf.value());
    } else {
      posinf = mshadow::red::limits::MaxValue<DType>();
    }
    if (param.neginf.has_value()) {
      neginf = static_cast<DType>(param.neginf.value());
    } else {
      neginf = mshadow::red::limits::MinValue<DType>();
    }
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<nan_to_num_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          defaultnan, posinf, neginf);
    });
  });
}

template<int req>
struct nan_to_num_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i,
                                  DType* in_grad,
                                  const DType* out_grad,
                                  const DType* in_data) {
    DType val = out_grad[i];
    if (mshadow_op::IsNan(in_data[i]))  val = 0;
    if (val > 0 && mshadow_op::IsInf(in_data[i]))  val = 0;
    if (val < 0 && mshadow_op::IsInf(in_data[i]))  val = 0;
    KERNEL_ASSIGN(in_grad[i], req, val);
  }
};

template<typename xpu>
void NumpyNanToNumOpBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_NE(req[0], kWriteInplace);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data = inputs[1];
  const TBlob& in_grad = outputs[0];
  CHECK_EQ(common::is_float(in_data.type_flag_), true);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<nan_to_num_backward<req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),
          in_data.dptr<DType>());
    });
  });
}

/*! \brief Unary compute */
#define MXNET_OPERATOR_REGISTER_UNARY(__name$)                      \
  NNVM_REGISTER_OP(__name$)                                         \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>) \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "The input array.")

#if MSHADOW_USE_MKL == 1
  /*! \bried MKL Unary compute.
   *  With this macro means mxnet compile with MKL to accelerate math function with mkl.
   *  Will Register FCompute with UnaryOp::MKL_Compute() to compelet the math function.
  */
  #define MXNET_MKL_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(__name$, __xpu$,                          \
                                                         __kernel$, __mkl_kernel$)                 \
    MXNET_OPERATOR_REGISTER_UNARY(__name$)                                                         \
    MXNET_ADD_SPARSE_OP_ALIAS(__name$)                                                             \
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1,                    \
                                 false, true, true>)                                               \
    .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::MKL_Compute<__kernel$, __mkl_kernel$>)   \
    .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::ComputeEx<__xpu$, __kernel$>)

  /*! \bried MKL Unary compute.
   *  With this macro means mxnet compile with MKL to accelerate math function with mkl.
   *  Will Register FCompute with UnaryOp::MKL_Compute() to compelet the math function.
  */
  #define MXNET_MKL_OPERATOR_REGISTER_UNARY_WITH_RSP(__name$, __xpu$, __kernel$, __mkl_kernel$)    \
    MXNET_OPERATOR_REGISTER_UNARY(__name$)                                                         \
    MXNET_ADD_SPARSE_OP_ALIAS(__name$)                                                             \
    .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1,                    \
                                 false, true, false>)                                              \
    .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::MKL_Compute<__kernel$, __mkl_kernel$>)   \
    .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::MKL_ComputeEx<__xpu$, __kernel$>)

  #define MXNET_MKL_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(__name$, __xpu$, __kernel$,             \
                                                           __mkl_kernel$)                          \
    MXNET_OPERATOR_REGISTER_UNARY(__name$)                                                         \
    .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::MKL_Compute<__kernel$, __mkl_kernel$>)
#endif

/*! \brief Unary compute, with FComputeEx for csr and rsp available  */
#define MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP_CSR(__name$, __xpu$, __kernel$)                     \
  MXNET_OPERATOR_REGISTER_UNARY(__name$)                                                           \
  MXNET_ADD_SPARSE_OP_ALIAS(__name$)                                                               \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1, false, true, true>)  \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Compute<__xpu$, __kernel$>)                \
  .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::ComputeEx<__xpu$, __kernel$>)

/*! \brief Unary compute, with FComputeEx for rsp available  */
#define MXNET_OPERATOR_REGISTER_UNARY_WITH_RSP(__name$, __xpu$, __kernel$)                         \
  MXNET_OPERATOR_REGISTER_UNARY(__name$)                                                           \
  MXNET_ADD_SPARSE_OP_ALIAS(__name$)                                                               \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1, false, true, false>) \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Compute<__xpu$, __kernel$>)                \
  .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::ComputeEx<__xpu$, __kernel$>)

/*! \brief Unary compute, dense result.
 *  FInferStorageType attr is not set using this macro. By default DefaultStorageType is used.
 */
#define MXNET_OPERATOR_REGISTER_UNARY_WITH_SPARSE_DR(__name$, __xpu$, __kernel$)        \
  MXNET_OPERATOR_REGISTER_UNARY(__name$)                                                \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Compute<__xpu$, __kernel$>)

#if MXNET_USE_CUDA

struct UnaryRTCCompute {
  std::string OP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<NDArray>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<NDArray>& outputs);
};

struct UnaryBwdInOutRTCCompute {
  std::string OP;

  void operator()(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs);
};

#endif  // MXNET_USE_CUDA

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
