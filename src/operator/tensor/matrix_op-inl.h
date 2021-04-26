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
 * \file matrix_op-inl.h
 * \brief Function definition of matrix related operators
 */
#ifndef MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <type_traits>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../channel_op_common.h"
#include "../mxnet_op.h"
#include "broadcast_reduce_op.h"
#include "./init_op.h"
#include "../../common/static_array.h"
#include "./slice-inl.h"

#if MXNET_USE_CUDA
#include <thrust/device_vector.h>
#endif

#ifdef __CUDACC__
#include "./pseudo2DTranspose_op-inl.cuh"
#endif


namespace mxnet {
namespace op {

struct ReshapeParam : public dmlc::Parameter<ReshapeParam> {
  mxnet::TShape target_shape;
  bool keep_highest;
  mxnet::Tuple<int> shape;
  bool reverse;
  DMLC_DECLARE_PARAMETER(ReshapeParam) {
    DMLC_DECLARE_FIELD(shape)
    .set_default(mxnet::Tuple<int>())
    .describe("The target shape");
    DMLC_DECLARE_FIELD(reverse)
    .set_default(false)
    .describe("If true then the special values are inferred from right to left");
    DMLC_DECLARE_FIELD(target_shape)
    .set_default(mxnet::TShape(0, -1))
    .describe("(Deprecated! Use ``shape`` instead.) "
              "Target new shape. One and only one dim can be 0, "
              "in which case it will be inferred from the rest of dims");
    DMLC_DECLARE_FIELD(keep_highest).set_default(false)
    .describe("(Deprecated! Use ``shape`` instead.) Whether keep the highest dim unchanged."
              "If set to true, then the first dim in target_shape is ignored,"
              "and always fixed as input");
  }

  bool operator==(const ReshapeParam &other) const {
    return this->target_shape == other.target_shape &&
           this->keep_highest == other.keep_highest &&
           this->shape == other.shape &&
           this->reverse == other.reverse;
  }
};

template<typename IType>
inline mxnet::TShape InferReshapeShape(const mxnet::Tuple<IType>& shape,
                                       const mxnet::TShape& dshape, bool reverse) {
  std::vector<IType> dshape_vec;
  std::vector<IType> param_shape_vec(shape.begin(), shape.end());
  for (int i = 0; i < dshape.ndim(); ++i) {
    dshape_vec.push_back(dshape[i]);
  }
  std::vector<IType> tmp;
  size_t src_idx = 0;
  int inf_idx = -1;
  if (reverse) {
    std::reverse(dshape_vec.begin(), dshape_vec.end());
    std::reverse(param_shape_vec.begin(), param_shape_vec.end());
  }
  auto dshape_len = dshape_vec.size();
  auto params_len = param_shape_vec.size();
  for (size_t i = 0; i < params_len; ++i) {
    IType proposed_dim = param_shape_vec[i];
    if (proposed_dim == 0) {
      // keep same
      CHECK_LT(src_idx, dshape_len);
      tmp.push_back(dshape_vec[src_idx++]);
    } else if (proposed_dim == -1) {
      // infer
      CHECK_LT(inf_idx, 0) << "One and only one dim can be inferred";
      inf_idx = i;
      tmp.push_back(1);
      src_idx++;
    } else if (proposed_dim == -2) {
      // copy all remaining dims from source
      while (src_idx < dshape_len) {
        const int dn = dshape_vec[src_idx++];
        tmp.push_back(dn);
      }
    } else if (proposed_dim == -3) {
      // merge two dims from source
      CHECK_LT(src_idx, dshape_len-1);
      const int d1 = dshape_vec[src_idx++];
      const int d2 = dshape_vec[src_idx++];
      if (!mxnet::dim_size_is_known(d1) || !mxnet::dim_size_is_known(d2)) {
        tmp.push_back(-1);
      } else {
        tmp.push_back(d1 * d2);
      }
    } else if (proposed_dim == -4) {
      // split the source dim s into two dims
      // read the left dim and then the right dim (either can be -1)
      CHECK_LT(i + 2, params_len);
      CHECK_LT(src_idx, dshape_len);
      const int d0 = dshape_vec[src_idx++];
      IType d1 = param_shape_vec[++i];
      IType d2 = param_shape_vec[++i];
      CHECK(d1 != -1 || d2 != -1) << "Split dims cannot both be -1.";
      if (d1 == -1 && d0 >= 0) d1 = d0 / d2;  // d0 must be known to do this
      if (d2 == -1 && d0 >= 0) d2 = d0 / d1;  // d0 must be known to do this
      CHECK(d1 * d2 == static_cast<IType>(d0) || static_cast<IType>(d0) == IType(-1)) <<
        "Split dims " << d1 << ", " << d2 << " do not divide original dim " << d0;
      tmp.push_back(d1);
      tmp.push_back(d2);
    } else {
      // greater than 0, new shape
      tmp.push_back(proposed_dim);
      src_idx++;
    }
  }

  if (inf_idx >= 0) {
    if (shape_is_known(dshape)) {
      IType new_size = 1;
      for (IType x : tmp) new_size *= x;
      tmp[inf_idx] = dshape.Size() / new_size;
    } else {
      tmp[inf_idx] = -1;
    }
  }
  if (reverse) {
    std::reverse(param_shape_vec.begin(), param_shape_vec.end());
    std::reverse(dshape_vec.begin(), dshape_vec.end());
    std::reverse(tmp.begin(), tmp.end());
  }
  mxnet::TShape oshape(tmp.begin(), tmp.end());
  return oshape;
}

inline bool ReverseReshapeInferShape(mxnet::TShape *in, const mxnet::TShape& out) {
  if (shape_is_known(*in) && shape_is_known(out)) {
    return true;
  } else if (!shape_is_known(out)) {
    return false;
  } else {
    int zero_axis = -1;
    int known_dim_size_prod = 1;
    for (int i = 0; i < in->ndim(); i++) {
      if (!mxnet::dim_size_is_known(*in, i)) {
        if (zero_axis != -1)
          return false;  // more than 1 zero found.
        else
          zero_axis = i;
      } else {
        known_dim_size_prod *= (*in)[i];
      }
    }
    (*in)[zero_axis] = out.Size() / known_dim_size_prod;
    return true;
  }
}

inline bool ReshapeShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const ReshapeParam& param_ = nnvm::get<ReshapeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape &dshape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(dshape)) return false;
  mxnet::TShape oshape;
  if (param_.shape.ndim() != 0) {
    oshape = InferReshapeShape(param_.shape, dshape, param_.reverse);
  } else if (param_.target_shape.ndim() != -1) {
    LOG(INFO) << "Using target_shape will be deprecated.";
    oshape = param_.target_shape;
    int neg_count = 0;
    index_t inf_idx = 0;
    index_t start_idx = param_.keep_highest ? 1 : 0;
    if (param_.keep_highest) {
      oshape[0] = dshape[0];
    }
    for (int i = start_idx; i < oshape.ndim(); ++i) {
      if (oshape[i] == 0) {
        neg_count++;
        inf_idx = i;
      }
    }
    if (neg_count == 1) {
      oshape[inf_idx] = 1;
      oshape[inf_idx] = dshape.Size() / oshape.Size();
    }
  } else {
    return shape_is_known((*out_attrs)[0])
           && ReverseReshapeInferShape(&(*in_attrs)[0], (*out_attrs)[0]);
  }
  ReverseReshapeInferShape(&dshape, oshape);
#if 0
  CHECK_EQ(oshape.Size(), dshape.Size())
    << "Target shape size is different to source. "
    << "Target: " << oshape
    << "\nSource: " << dshape;
#endif
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return ReverseReshapeInferShape(&(*in_attrs)[0], (*out_attrs)[0]);
}

inline bool FlattenShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape &dshape = (*in_attrs)[0];
  if (!shape_is_known(dshape)) return false;
  size_t target_dim = 1;
  for (int i = 1; i < dshape.ndim(); ++i) {
    target_dim *= dshape[i];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape2(dshape[0], target_dim));
  return true;
}

struct TransposeParam : public dmlc::Parameter<TransposeParam> {
  mxnet::TShape axes;
  DMLC_DECLARE_PARAMETER(TransposeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(mxnet::TShape(0, -1))
    .describe("Target axis order. By default the axes will be inverted.");
  }

  bool operator==(const TransposeParam &other) const {
    return this->axes == other.axes;
  }
};


/*!
 * \brief This function performs transpose operation on a 2D matrix by utilizing the L1 cache
 * \param in  input tensor
 * \param out output tensor
 * \param row shape of dim 0 of input
 * \param col shape of dim 1 of input
 * \tparam DType Data type
 * \tparam is_addto
 */
template<typename DType, bool is_addto>
MSHADOW_XINLINE void Transpose2D(const DType *in, DType *out, index_t row, index_t col) {
  // ensure cache line hits and prevent cache miss for any configuration
  // L1 cache size to be utilized = 32kb = 2^15
  // Largest size of a single unit of any dtype <= 8 byte = 2^3
  // Number of elements - (2^15/2^3) = 2^12
  // Block-size - 2^6 v 2^6 (64 v 64)

  // But we could leverage unrolling of for loops (for parallelization)
  // Block-size - 2^5 v 2^5 (32 v 32) with potential 4 pragma for loop unrolled
  // blocksize * blocksize * num_threads = cache_size / dtype_size
  // Instead of explicit unroll, let compiler figure out optimal unroll factor
  const index_t blocksize = 32;

  // collapse 2 parallelizes 2 for loops
  // inner 2 for loops aren't parallelized to prevent cache miss

  // Microsoft Visual C++ compiler does not support omp collapse
  #ifdef _MSC_VER
    #pragma omp parallel for
  #else
    #pragma omp parallel for collapse(2)
  #endif  // _MSC_VER

  for (index_t i = 0; i < row; i += blocksize) {
    for (index_t j = 0; j < col; j += blocksize) {
      // transpose the block
      for (index_t a = j; (a < blocksize + j) && (a < col); ++a) {
        for (index_t b = i; (b < blocksize + i) && (b < row); ++b) {
          if (!is_addto) {
            out[a * row + b] = in[b * col + a];
          } else {
            out[a * row + b] += in[b * col + a];
          }
        }
      }
    }
  }
}

inline bool IsIdentityTranspose(const TShape& axes) {
  for (dim_t i = 0; i < axes.ndim(); i++) {
    if (axes[i] != i) return false;
  }
  return true;
}

template<typename xpu, bool is_addto = false>
bool TransposeCommonImpl(RunContext ctx,
                   const TBlob& src,
                   const TBlob& ret,
                   const mxnet::TShape& axes) {
  // return true when running successfully, otherwise false
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(src.type_flag_, ret.type_flag_);
  // zero-size tensor, no need to compute
  if (src.shape_.Size() == 0U) return true;
  Stream<xpu> *s = ctx.get_stream<xpu>();
#ifdef __CUDACC__
  // This transpose can be used only if there exist n and m such that:
  // params = (0, ..., n-1, n+m, ..., params.size, n, ..., n+m-1)
  // Example: (0, 2, 3, 1) or (0, 3, 1, 2), but not (0, 2, 1, 3).
  if (isPseudo2DTranspose(axes)) {
    MSHADOW_TYPE_SWITCH(ret.type_flag_, DType, {
      transpose_pseudo2D<DType, is_addto>(ret, src, axes, s);
    });
    return true;
  }
#endif
  // Special handle the identity case
  if (IsIdentityTranspose(axes)) {
    MSHADOW_TYPE_SWITCH(ret.type_flag_, DType, {
      Tensor<xpu, 1, DType> in = src.get_with_shape<xpu, 1, DType>(mshadow::Shape1(src.Size()), s);
      Tensor<xpu, 1, DType> out = ret.get_with_shape<xpu, 1, DType>(mshadow::Shape1(ret.Size()), s);
      if (!is_addto) {
        // Use memcpy to accelerate the speed
        Copy(out, in, s);
      } else {
        mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, kAddTo>, xpu>::Launch(
            s, ret.Size(), out.dptr_, in.dptr_);
      }
    });
    return true;
  }
  // Handle the general transpose case
  MSHADOW_TYPE_SWITCH(ret.type_flag_, DType, {
    switch (axes.ndim()) {
     case 2: {
      Tensor<xpu, 2, DType> in = src.get<xpu, 2, DType>(s);
      Tensor<xpu, 2, DType> out = ret.get<xpu, 2, DType>(s);
      if (ctx.get_ctx().dev_mask() == cpu::kDevMask) {
        Transpose2D<DType, is_addto>(in.dptr_, out.dptr_, in.shape_[0], in.shape_[1]);
      } else {
        LOG(FATAL) << "Not Implemented. We should never reach here because the 2D case "
                      "in GPU has been covered by transpose_pseudo2D."
                      " Report an issue in Github.";
      }
      break;
     }
     case 3: {
      Tensor<xpu, 3, DType> in = src.get<xpu, 3, DType>(s);
      Tensor<xpu, 3, DType> out = ret.get<xpu, 3, DType>(s);
      if (!is_addto) {
        out = transpose(in, axes.get<3>());
      } else {
        out += transpose(in, axes.get<3>());
      }
      break;
     }
     case 4: {
      Tensor<xpu, 4, DType> in = src.get<xpu, 4, DType>(s);
      Tensor<xpu, 4, DType> out = ret.get<xpu, 4, DType>(s);
      if (!is_addto) {
        out = transpose(in, axes.get<4>());
      } else {
        out += transpose(in, axes.get<4>());
      }
      break;
     }
     case 5: {
      Tensor<xpu, 5, DType> in = src.get<xpu, 5, DType>(s);
      Tensor<xpu, 5, DType> out = ret.get<xpu, 5, DType>(s);
      if (!is_addto) {
        out = transpose(in, axes.get<5>());
      } else {
        out += transpose(in, axes.get<5>());
      }
      break;
     }
     case 6: {
      Tensor<xpu, 6, DType> in = src.get<xpu, 6, DType>(s);
      Tensor<xpu, 6, DType> out = ret.get<xpu, 6, DType>(s);
      if (!is_addto) {
        out = transpose(in, axes.get<6>());
      } else {
        out += transpose(in, axes.get<6>());
      }
      break;
     }
     default:
      // return false when dimensions > 6
      return false;
      break;
    }
  });
  return true;
}

template<typename xpu, bool is_addto = false>
void TransposeImpl(RunContext ctx,
                   const TBlob& src,
                   const TBlob& ret,
                   const mxnet::TShape& axes) {
  CHECK_LE(axes.ndim(), 6) << "TransposeImpl supports at most 6 dimensions";
  CHECK((TransposeCommonImpl<xpu, is_addto>(ctx, src, ret, axes))) <<
    "Failed to execute TransposeImpl Operator";
}

template <bool is_addto>
struct TransposeExKernel {
  /*!
   * \brief
   * \param tid      global thread id
   * \param out_data output data
   * \param in_data  input data
   * \param strides  input strides and output strides
   * \param ndim     the number of dimension
   */
  template <typename DType>
  MSHADOW_XINLINE static void Map(index_t tid,
      DType *out_data,
      const DType *in_data,
      const dim_t *strides,
      const int ndim
      ) {
    // tid is the index of input data
    const dim_t* const out_strides = strides + ndim;
    index_t k = tid;
    index_t out_id = 0;
    for (int i = 0; i < ndim; ++i) {
      out_id += (k / strides[i]) * out_strides[i];
      k %= strides[i];
    }
    if (is_addto)
      out_data[out_id] += in_data[tid];
    else
      out_data[out_id] = in_data[tid];
  }
};

template<typename xpu, bool is_addto = false>
void TransposeExImpl(RunContext ctx,
                   const TBlob& src,
                   const TBlob& ret,
                   const mxnet::TShape& axes,
                   mshadow::Tensor<xpu, 1, dim_t>& strides_xpu
                   ) {
  /*
   * If ndim <= 6, it is not necessary to allocate any space for `strides_xpu`
   * If ndim > 6, `strides_xpu` should be allocated `ndim * 2` elements
   */
  using namespace mshadow;
  using namespace mshadow::expr;
  if (TransposeCommonImpl<xpu, is_addto>(ctx, src, ret, axes)) return;
  CHECK_GT(axes.ndim(), 6) <<
    "Failed to execute TransposeExImpl when axes.ndim() <= 6";
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(ret.type_flag_, DType, {
      CHECK_EQ(strides_xpu.MSize(), axes.ndim() * 2) << \
        "If ndim > 6, `strides_xpu` should be allocated `ndim * 2` elements";

      const mxnet::TShape &in_shape = src.shape_;
      // strides: in_strides and out_strides
      const int ndim = axes.ndim();
      std::vector<dim_t> strides(ndim * 2);
      // compute in_strides
      strides[ndim - 1] = 1;
      for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * in_shape[i + 1];
      }
      // compute out_strides
      std::vector<dim_t> tmp_strides(ndim);
      tmp_strides[ndim - 1] = 1;
      for (int i = ndim - 2; i >= 0; --i) {
        tmp_strides[i] = tmp_strides[i + 1] * in_shape[axes[i + 1]];
      }
      // reorder tmp_strides to out_strides
      dim_t * const out_strides = &strides[ndim];
      for (int i = 0; i < ndim; ++i) {
        out_strides[axes[i]] = tmp_strides[i];
      }
      Shape<1> strides_shape;
      strides_shape[0] = ndim * 2;
      Tensor<cpu, 1, dim_t> strides_cpu(strides.data(), strides_shape);
      // copy arguments into xpu context
      Copy(strides_xpu, strides_cpu, s);
      const DType *in = src.dptr<DType>();
      DType *out = ret.dptr<DType>();
      if (is_addto) {
        mxnet_op::Kernel<TransposeExKernel<true>, xpu>::Launch(s,
            in_shape.Size(), out, in, strides_xpu.dptr_, ndim);
      } else {
        mxnet_op::Kernel<TransposeExKernel<false>, xpu>::Launch(s,
            in_shape.Size(), out, in, strides_xpu.dptr_, ndim);
      }
  });
}

template<typename xpu>
mshadow::Tensor<xpu, 1, dim_t> GetTransposeExWorkspace(
      const OpContext& ctx,
      const mxnet::TShape& axes
    ) {
  if (axes.ndim() > 6) {
    // allocate workspace when axes.ndim() > 6
    mshadow::Shape<1> strides_shape;
    strides_shape[0] = axes.ndim() * 2;
    return ctx.requested[0].get_space_typed<xpu, 1, dim_t>(
        strides_shape, ctx.get_stream<xpu>());
  }
  return {};
}

// matrix transpose
template<typename xpu>
void Transpose(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
  if (req[0] == kNullOp) {
    return;
  }
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK(req[0] == kWriteTo || req[0] == kAddTo)
       << "Transpose only supports kNullOp, kWriteTo and kAddTo";
  mxnet::TShape axes;
  if (param.axes.ndim() == 0) {
    axes = mxnet::TShape(inputs[0].ndim(), -1);
    for (int i = 0; i < axes.ndim(); ++i) {
      axes[i] = axes.ndim() - 1 - i;
    }
  } else {
    axes = common::CanonicalizeAxes(param.axes);
  }

  mshadow::Tensor<xpu, 1, dim_t> workspace =
    GetTransposeExWorkspace<xpu>(ctx, axes);
  if (req[0] == kAddTo) {
    TransposeExImpl<xpu, true>(ctx.run_ctx, inputs[0], outputs[0],
        axes, workspace);
  } else {
    TransposeExImpl<xpu, false>(ctx.run_ctx, inputs[0], outputs[0],
        axes, workspace);
  }
}

inline bool TransposeShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector *in_attrs,
                             mxnet::ShapeVector *out_attrs) {
  const TransposeParam& param = nnvm::get<TransposeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& shp = (*in_attrs)[0];
  mxnet::TShape& out_shp = (*out_attrs)[0];
  if (!mxnet::ndim_is_known(shp) && !mxnet::ndim_is_known(out_shp))
    return false;  // none of the shapes is known
  if (out_shp.ndim() >= 0 && shp.ndim() >= 0)
    CHECK_EQ(out_shp.ndim(), shp.ndim());
  mxnet::TShape get(std::max(shp.ndim(), out_shp.ndim()), -1);
  mxnet::TShape ret(std::max(shp.ndim(), out_shp.ndim()), -1);
  if (param.axes.ndim() == 0) {
    for (int i = 0; i < shp.ndim(); ++i) {
      ret[i] = shp[shp.ndim()-1-i];
    }
    for (int i = 0; i < out_shp.ndim(); ++i) {
      get[shp.ndim()-1-i] = out_shp[i];
    }
  } else {
    CHECK_EQ(std::max(shp.ndim(), out_shp.ndim()), param.axes.ndim());
    for (int i = 0; i < shp.ndim(); ++i) {
      CHECK(param.axes[i] < static_cast<int64_t>(shp.ndim()));
      ret[i] = shp[param.axes[i]];
    }
    for (int i = 0; i < out_shp.ndim(); ++i) {
      get[param.axes[i]] = out_shp[i];
    }
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, get);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);
  return shape_is_known(ret);
}


struct ExpandDimParam : public dmlc::Parameter<ExpandDimParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(ExpandDimParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("Position where new axis is to be inserted. Suppose that "
              "the input `NDArray`'s dimension is `ndim`, the range of "
              "the inserted axis is `[-ndim, ndim]`");
  }

  bool operator==(const ExpandDimParam &other) const {
    return this->axis == other.axis;
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s;
    axis_s << axis;
    (*dict)["axis"] = axis_s.str();
  }
};


inline bool ExpandDimShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  const ExpandDimParam& param = nnvm::get<ExpandDimParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& ishape = (*in_attrs)[0];
  mxnet::TShape& oshape = (*out_attrs)[0];
  if (!mxnet::ndim_is_known(ishape) && !mxnet::ndim_is_known(oshape)) {
    return false;
  }

  int indim = ishape.ndim();
  bool unknown_ishape = false;
  if (-1 == indim) {
    indim = oshape.ndim() - 1;
    unknown_ishape = true;
  }

  int axis = param.axis;
  if (axis < 0) {
    axis += indim + 1;
  }
  CHECK(axis >= 0 && axis <= indim)
      << "axis must be in the range [" << -indim << ", " << indim << "] ("
      << param.axis << " provided)";
  mxnet::TShape ret(indim + 1, -1);
  for (int i = 0; i < axis; ++i) {
    ret[i] = (unknown_ishape? -1 : ishape[i]);
  }
  ret[axis] = 1;
  for (int i = axis+1; i < indim+1; ++i) {
    ret[i] = (unknown_ishape? -1 : ishape[i-1]);
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, ret);

  ret = mxnet::TShape(indim, -1);
  for (int i = 0; i < axis; ++i) ret[i] = oshape[i];
  for (int i = axis+1; i < indim+1; ++i) ret[i-1] = oshape[i];
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, ret);
  return shape_is_known(in_attrs->at(0)) && shape_is_known(out_attrs->at(0));
}

// Currently MKLDNN only supports step = 1 or step has no value
inline bool SupportMKLDNNSlice(const SliceParam& param) {
  if (param.step.ndim() == 0U) return true;
  for (int i = 0; i < param.step.ndim(); ++i) {
    if (param.step[i].has_value() && param.step[i].value() != 1)
      return false;
  }
  return true;
}

inline bool SliceForwardInferStorageType(const nnvm::NodeAttrs& attrs,
                                         const int dev_mask,
                                         DispatchMode* dispatch_mode,
                                         std::vector<int>* in_attrs,
                                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 1);
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  const auto& in_stype = in_attrs->at(0);
  auto& out_stype = out_attrs->at(0);
  bool dispatched = false;
  const auto dispatch_ex = DispatchMode::kFComputeEx;
  // If step = 1, no need to fallback; otherwise fallback to dense
  bool trivial_step = false;
  if (param.step.ndim() == 0U) {
    trivial_step = true;
  } else if (param.step.ndim() == 1U
      && (!param.step[0].has_value() || param.step[0].value() == 1)) {
    trivial_step = true;
  }

  if (in_stype == kDefaultStorage) {
#if MXNET_USE_ONEDNN == 1
    if (dev_mask == Context::kCPU && MKLDNNEnvSet()
        && SupportMKLDNNSlice(param)) {
      dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                       dispatch_mode, dispatch_ex);
    }
#endif
    if (!dispatched) {
      dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                       dispatch_mode, DispatchMode::kFCompute);
    }
  }

  if (!dispatched && in_stype == kCSRStorage && trivial_step) {
    dispatched = storage_type_assign(&out_stype, kCSRStorage,
                                     dispatch_mode, dispatch_ex);
  }

  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }

  return dispatched;
}

// slice the indptr of a csr
struct SliceCsrIndPtr {
  template<typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* out, const IType* in, const IType* base) {
    KERNEL_ASSIGN(out[i], kWriteTo, in[i] - *base);
  }
};

/*
 * a wrapper to launch SliceCsrIndPtr kernel.
 * slice [src[begin] .. src[end]) and store in dst[0, end - begin)
 */
template<typename xpu, typename IType>
void SliceCsrIndPtrImpl(const int begin, const int end, RunContext ctx,
                        const IType* src, IType* dst) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int indptr_len = end - begin + 1;
  Kernel<SliceCsrIndPtr, xpu>::Launch(s, indptr_len, dst, src + begin, src + begin);
}

/*
 * Slice a CSR NDArray for first dimension
 */
template<typename xpu>
void SliceDimOneCsrImpl(const mxnet::TShape &begin, const mxnet::TShape &end, const OpContext& ctx,
                        const NDArray &in, const NDArray &out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  nnvm::dim_t begin_row = begin[0];
  nnvm::dim_t end_row = end[0];
  nnvm::dim_t indptr_len = end_row - begin_row + 1;
  out.CheckAndAllocAuxData(kIndPtr, Shape1(indptr_len));
  // assume idx indptr share the same type
  MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIdx), IType, {
      MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
        RType* in_indptr = in.aux_data(kIndPtr).dptr<RType>();
        RType* out_indptr = out.aux_data(kIndPtr).dptr<RType>();
        SliceCsrIndPtrImpl<xpu, RType>(begin_row, end_row, ctx.run_ctx, in_indptr, out_indptr);

        Stream<xpu> *s = ctx.get_stream<xpu>();

        RType nnz = 0;
        mshadow::Copy(Tensor<cpu, 1, RType>(&nnz, Shape1(1)),
                      Tensor<xpu, 1, RType>(out_indptr + indptr_len - 1, Shape1(1), s));
        // return csr zeros if nnz = 0
        if (nnz == 0) {
          out.set_aux_shape(kIdx, Shape1(0));
          return;
        }
        // copy indices and values
        out.CheckAndAllocAuxData(kIdx, Shape1(nnz));
        out.CheckAndAllocData(Shape1(nnz));
        IType* in_idx = in.aux_data(kIdx).dptr<IType>();
        IType* out_idx = out.aux_data(kIdx).dptr<IType>();
        DType* in_data = in.data().dptr<DType>();
        DType* out_data = out.data().dptr<DType>();

        RType offset = 0;
        mshadow::Copy(Tensor<cpu, 1, RType>(&offset, Shape1(1)),
                      Tensor<xpu, 1, RType>(in_indptr + begin_row, Shape1(1), s));

        mshadow::Copy(Tensor<xpu, 1, IType>(out_idx, Shape1(nnz), s),
                      Tensor<xpu, 1, IType>(in_idx + offset, Shape1(nnz), s), s);
        mshadow::Copy(Tensor<xpu, 1, DType>(out_data, Shape1(nnz), s),
                      Tensor<xpu, 1, DType>(in_data + offset, Shape1(nnz), s), s);
      });
    });
  });
}

/*!
 * \brief slice a CSRNDArray for two dimensions
 */
struct SliceDimTwoCsrAssign {
  /*!
   * \brief This function slices a CSRNDArray on axis one between begin_col and end_col
   * \param i           loop index
   * \param out_idx     output csr ndarray column indices
   * \param out_data    output csr ndarray data
   * \param out_indptr  output csr ndarray row index pointer
   * \param in_idx      input csr ndarray column indices
   * \param in_data     input csr ndarray data
   * \param in_indptr   input csr ndarray row index pointer
   * \param begin_col   begin column indice
   * \param end_col     end column indice
   */
  template<typename IType, typename RType, typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  IType* out_idx, DType* out_data,
                                  const RType* out_indptr,
                                  const IType* in_idx, const DType* in_data,
                                  const RType* in_indptr,
                                  const int begin_col, const int end_col) {
    RType ind = out_indptr[i];
    for (RType j = in_indptr[i]; j < in_indptr[i+1]; j++) {
      // indices of CSRNDArray are in ascending order per row
      if (in_idx[j] >= end_col) {
        break;
      } else if (in_idx[j] >= begin_col) {
        out_idx[ind] = in_idx[j] - begin_col;
        out_data[ind] = in_data[j];
        ind++;
      }
    }
  }
};

/*
 * Slice a CSR NDArray for two dimensions
 */
template<typename xpu>
void SliceDimTwoCsrImpl(const mxnet::TShape &begin, const mxnet::TShape &end, const OpContext& ctx,
                        const NDArray &in, const NDArray &out);


template<typename xpu>
void SliceCsrImpl(const SliceParam &param, const OpContext& ctx,
                  const NDArray &in, OpReqType req, const NDArray &out) {
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "kAddTo for Slice on CSR input is not supported";
  CHECK_NE(req, kWriteInplace) << "kWriteInplace for Slice on CSR input is not supported";

  const mxnet::TShape ishape = in.shape();
  const mxnet::TShape oshape = out.shape();

  int N = ishape.ndim();
  mxnet::TShape begin(N, -1), end(N, -1);
  for (int i = 0; i < N; ++i) {
    int s = 0;
    if (i < param.begin.ndim() && param.begin[i]) {
      s = *param.begin[i];
      if (s < 0) s += ishape[i];
    }
    begin[i] = s;
    end[i] = s + oshape[i];
  }
  switch (N) {
    case 1: {
      SliceDimOneCsrImpl<xpu>(begin, end, ctx, in, out);
      break;
    }
    case 2: {
      SliceDimTwoCsrImpl<xpu>(begin, end, ctx, in, out);
      break;
    }
    default:
      LOG(FATAL) << "CSR is only for 2-D shape";
      break;
  }
}

template<typename xpu>
void SliceEx(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<NDArray>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  auto in_stype = inputs[0].storage_type();
  if (in_stype == kCSRStorage) {
    SliceCsrImpl<xpu>(param, ctx, inputs[0], req[0], outputs[0]);
  } else {
    LOG(FATAL) << "Slice not implemented for storage type" << in_stype;
  }
}

template<int ndim>
inline bool GetIndexRange(const mxnet::TShape& dshape,
                          const mxnet::Tuple<dmlc::optional<index_t>>& param_begin,
                          const mxnet::Tuple<dmlc::optional<index_t>>& param_end,
                          const mxnet::Tuple<dmlc::optional<index_t>>& param_step,
                          common::StaticArray<index_t, ndim>* begin,
                          common::StaticArray<index_t, ndim>* end,
                          common::StaticArray<index_t, ndim>* step) {
  // Function returns false if output is zero-sized, true otherwise.
  bool zero_size_shape = false;
  CHECK_NE(dshape.ndim(), 0U);
  CHECK_LE(param_begin.ndim(), dshape.ndim())
    << "Slicing axis exceeds data dimensions";
  CHECK_LE(param_end.ndim(), dshape.ndim())
    << "Slicing axis exceeds data dimensions";
  CHECK_EQ(param_begin.ndim(), param_end.ndim())
    << "begin and end must have the same length";
  CHECK_EQ(ndim, dshape.ndim())
    << "Static array size=" << ndim
    << " is not equal to data shape ndim=" << dshape.ndim();

  if (param_step.ndim() > 0) {
    CHECK_EQ(param_step.ndim(), param_begin.ndim())
      << "step and begin must have the same length";
  }

  for (int i = 0; i < param_begin.ndim(); ++i) {
    index_t s = param_step.ndim() > 0 && param_step[i].has_value() ? param_step[i].value() : 1;
    CHECK_NE(s, 0) << "slice op step[" << i << "] cannot be 0";

    index_t b = 0, e = 0;
    const index_t len = dshape[i];
    if (len > 0) {
      b = param_begin[i].has_value() ? param_begin[i].value() : (s < 0 ? len - 1 : 0);
      e = param_end[i].has_value() ? param_end[i].value() : (s < 0 ? -1 : len);

      if (b < 0) {
        b += len;
      }
      if (e < 0 && param_end[i].has_value()) {
        e += len;
      }

      // move the begin and end to correct position for calculating dim size
      b = (b < 0 && s > 0) ? 0 : b;
      b = (b > len - 1 && s < 0) ? len - 1 : b;
      // if the start value lead to empty tensor under step s, use -1 for indication
      b = (b < 0 || b > len - 1) ? -1 : b;
      e = e > -1 ? e : -1;
      e = e > len ? len : e;
    } else if (len == 0) {
      b = 0;
      e = 0;
    }

    (*begin)[i] = b;
    (*end)[i] = e;
    (*step)[i] = s;
    // checking begin==end
    if (b == e) {
      zero_size_shape = true;
    }
  }

  for (int i = param_begin.ndim(); i < dshape.ndim(); ++i) {
    (*begin)[i] = 0;
    (*end)[i] = dshape[i];
    (*step)[i] = 1;
  }

  return zero_size_shape;
}

inline void SetSliceOpOutputDimSize(const mxnet::TShape& dshape,
                                    const index_t i, const index_t b,
                                    const index_t e, const index_t s,
                                    mxnet::TShape* oshape) {
  if (!mxnet::dim_size_is_known(dshape, i)) {
    (*oshape)[i] = -1;
    return;
  }
  if (e != b && b >= 0) {
    if (s > 0) {
      (*oshape)[i] = e > b ? (e - b - 1) / s + 1 : 0;
    } else {
      (*oshape)[i] = e < b ? (b - e - 1) / (-s) + 1 : 0;
    }
  } else {
      (*oshape)[i] = 0;
  }
}

inline bool SliceOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(dshape)) return false;
  CHECK_GT(dshape.ndim(), 0) << "slice only works for ndim > 0";
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  mxnet::TShape oshape = dshape;

  MXNET_NDIM_SWITCH(dshape.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    GetIndexRange(dshape, param.begin, param.end, param.step, &begin, &end, &step);
    for (int i = 0; i < param.begin.ndim(); ++i) {
      const index_t b = begin[i], e = end[i], s = step[i];
      SetSliceOpOutputDimSize(dshape, i, b, e, s, &oshape);
    }
  })

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(dshape) && shape_is_known(oshape);
}

template<int ndim, int req, typename xpu>
struct slice_forward;

template<int ndim, int req>
struct slice_forward<ndim, req, gpu> {
  // i is the i-th row after flattening out into 2D tensor
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data,
                                  const mshadow::Shape<ndim> dshape,
                                  const mshadow::Shape<ndim> oshape,
                                  const common::StaticArray<index_t, ndim> begin,
                                  const common::StaticArray<index_t, ndim> step) {
    const index_t data_last_dim_size = dshape[ndim-1];
    const index_t out_last_dim_size = oshape[ndim-1];
    const index_t step_last_dim = step[ndim-1];
    const index_t begin_last_dim = begin[ndim-1];
    const index_t j = i % out_last_dim_size;
    index_t irow = 0;  // row id of flattend 2D data
    index_t stride = 1;
    index_t idx = i / out_last_dim_size;
    #pragma unroll
    for (int k = ndim - 2; k >= 0; --k) {
      irow += stride * ((idx % oshape[k]) * step[k] + begin[k]);
      idx /= oshape[k];
      stride *= dshape[k];
    }
    KERNEL_ASSIGN(out[i], req,
                  data[irow * data_last_dim_size + j * step_last_dim + begin_last_dim]);
  }
};

template<int ndim, int req>
struct slice_forward<ndim, req, cpu> {
  // i is the i-th row after flattening out into 2D tensor
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* data,
                                  const mshadow::Shape<ndim> dshape,
                                  const mshadow::Shape<ndim> oshape,
                                  const common::StaticArray<index_t, ndim> begin,
                                  const common::StaticArray<index_t, ndim> step) {
    const index_t data_last_dim_size = dshape[ndim-1];
    const index_t out_last_dim_size = oshape[ndim-1];
    const index_t step_last_dim = step[ndim-1];
    const index_t begin_last_dim = begin[ndim-1];
    index_t out_offset = i * out_last_dim_size;
    for (index_t j = 0; j < out_last_dim_size; ++j) {
      index_t irow = 0;  // row id of flattend 2D data
      index_t stride = 1;
      index_t idx = i;
      #pragma unroll
      for (int k = ndim - 2; k >= 0; --k) {
        irow += stride * ((idx % oshape[k]) * step[k] + begin[k]);
        idx /= oshape[k];
        stride *= dshape[k];
      }
      KERNEL_ASSIGN(out[out_offset++], req,
                    data[irow * data_last_dim_size + j * step_last_dim + begin_last_dim]);
    }
  }
};

template<typename xpu>
void SliceOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  if (out.Size() == 0) return;
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    GetIndexRange(data.shape_, param.begin, param.end, param.step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH_WITH_BOOL(out.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        size_t num_threads = out.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= out.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_forward<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
            out.dptr<DType>(), data.dptr<DType>(),
            data.shape_.get<ndim>(), out.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

template<int ndim, int req, typename xpu>
struct slice_assign;

template<int ndim, int req>
struct slice_assign<ndim, req, cpu> {
  // i is the i-th row after flattening out into 2D tensor
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* val,
                                  const mshadow::Shape<ndim> oshape,
                                  const mshadow::Shape<ndim> vshape,
                                  const common::StaticArray<index_t, ndim> begin,
                                  const common::StaticArray<index_t, ndim> step) {
    const index_t data_last_dim_size = oshape[ndim-1];
    const index_t out_last_dim_size = vshape[ndim-1];
    const index_t step_last_dim = step[ndim-1];
    const index_t begin_last_dim = begin[ndim-1];
    index_t offset = i * out_last_dim_size;
    for (index_t j = 0; j < out_last_dim_size; ++j) {
      index_t irow = 0;  // row id of flattend 2D out
      index_t stride = 1;
      index_t idx = i;
      #pragma unroll
      for (int k = ndim - 2; k >= 0; --k) {
        irow += stride * ((idx % vshape[k]) * step[k] + begin[k]);
        idx /= vshape[k];
        stride *= oshape[k];
      }
      KERNEL_ASSIGN(out[irow * data_last_dim_size + j * step_last_dim + begin_last_dim],
                    req, val[offset++]);
    }
  }
};

template<int ndim, int req>
struct slice_assign<ndim, req, gpu> {
  // i is the i-th row after flattening out into 2D tensor
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* val,
                                  const mshadow::Shape<ndim> oshape,
                                  const mshadow::Shape<ndim> vshape,
                                  const common::StaticArray<index_t, ndim> begin,
                                  const common::StaticArray<index_t, ndim> step) {
    const index_t data_last_dim_size = oshape[ndim-1];
    const index_t out_last_dim_size = vshape[ndim-1];
    const index_t step_last_dim = step[ndim-1];
    const index_t begin_last_dim = begin[ndim-1];
    const index_t j = i % out_last_dim_size;
    index_t irow = 0;  // row id of flattend 2D out
    index_t stride = 1;
    index_t idx = i / out_last_dim_size;
    #pragma unroll
    for (int k = ndim - 2; k >= 0; --k) {
      irow += stride * ((idx % vshape[k]) * step[k] + begin[k]);
      idx /= vshape[k];
      stride *= oshape[k];
    }
    KERNEL_ASSIGN(out[irow * data_last_dim_size + j * step_last_dim + begin_last_dim],
                  req, val[i]);
  }
};

template<typename xpu>
void SliceOpBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  if (req[0] == kNullOp) return;
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  const TBlob& ograd = inputs[0];
  const TBlob& igrad = outputs[0];
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  if (req[0] == kWriteTo) {
    Fill(s, igrad, req[0], 0);
  } else if (req[0] == kWriteInplace) {
    LOG(FATAL) << "_slice_backward does not support kWriteInplace";
  }
  if (ograd.Size() == 0) return;
  MXNET_NDIM_SWITCH(ograd.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    GetIndexRange(igrad.shape_, param.begin, param.end, param.step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      int num_threads = ograd.shape_.FlatTo2D()[0];
      if (std::is_same<xpu, gpu>::value) {
        num_threads *= ograd.shape_.get<ndim>()[ndim - 1];
      }
      mxnet_op::Kernel<slice_assign<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
          igrad.dptr<DType>(), ograd.dptr<DType>(),
          igrad.shape_.get<ndim>(), ograd.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

inline bool SliceAssignOpShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector *in_attrs,
                               mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(dshape)) return false;
  mxnet::TShape vshape = dshape;  // vshape is the value shape on the right hand side
  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(dshape.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    GetIndexRange(dshape, param.begin, param.end, param.step, &begin, &end, &step);
    for (int i = 0; i < param.begin.ndim(); ++i) {
      const index_t b = begin[i], e = end[i], s = step[i];
      SetSliceOpOutputDimSize(dshape, i, b, e, s, &vshape);
    }
  })
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, vshape);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  return true;
}

template<typename xpu>
void SliceAssignOpForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), 2U);  // data[index] = val, data and val are two inputs
  CHECK_EQ(outputs.size(), 1U);
  if (req[0] == kNullOp) return;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  const TBlob& val = inputs[1];
  const TBlob& out = outputs[0];
  if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> in = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Copy(out, in, s);
    });
  } else if (req[0] != kWriteInplace) {
    LOG(FATAL) << "_slice_assign only supports kWriteTo and kWriteInplace";
  }

  const SliceParam& param = nnvm::get<SliceParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    bool zero_size_shape = GetIndexRange(data.shape_, param.begin, param.end, param.step,
                                        &begin, &end, &step);
    if (zero_size_shape) {
      return;  // slice_assign of zero-sized subspace needs no operation.
    }
    MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        index_t num_threads = val.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= val.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_assign<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
            out.dptr<DType>(), val.dptr<DType>(),
            out.shape_.get<ndim>(), val.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

struct SliceAssignScalarParam : public dmlc::Parameter<SliceAssignScalarParam> {
  double scalar;
  mxnet::Tuple<dmlc::optional<index_t>> begin, end;
  mxnet::Tuple<dmlc::optional<index_t>> step;
  DMLC_DECLARE_PARAMETER(SliceAssignScalarParam) {
    DMLC_DECLARE_FIELD(scalar)
    .set_default(0)
    .describe("The scalar value for assignment.");
    DMLC_DECLARE_FIELD(begin)
    .describe("starting indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(end)
    .describe("ending indices for the slice operation, supports negative indices.");
    DMLC_DECLARE_FIELD(step)
    .set_default(mxnet::Tuple<dmlc::optional<index_t>>())
    .describe("step for the slice operation, supports negative values.");
  }
};

inline bool SliceAssignScalarOpShape(const nnvm::NodeAttrs& attrs,
                                    mxnet::ShapeVector *in_attrs,
                                    mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = (*in_attrs)[0];
  if (!shape_is_known(dshape)) return false;
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, dshape);
  return true;
}

template<int ndim>
struct slice_assign_scalar {
  // i is the i-th row after flattening out into 2D tensor
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType val,
                                  const OpReqType req,
                                  const mshadow::Shape<ndim> oshape,
                                  const mshadow::Shape<ndim> vshape,
                                  const common::StaticArray<index_t, ndim> begin,
                                  const common::StaticArray<index_t, ndim> step) {
    const index_t data_last_dim_size = oshape[ndim-1];
    const index_t out_last_dim_size = vshape[ndim-1];
    const index_t step_last_dim = step[ndim-1];
    const index_t begin_last_dim = begin[ndim-1];
    for (index_t j = 0; j < out_last_dim_size; ++j) {
      index_t irow = 0;  // row id of flattend 2D out
      index_t stride = 1;
      index_t idx = i;
      #pragma unroll
      for (int k = ndim - 2; k >= 0; --k) {
        irow += stride * ((idx % vshape[k]) * step[k] + begin[k]);
        idx /= vshape[k];
        stride *= oshape[k];
      }
      KERNEL_ASSIGN(out[irow * data_last_dim_size + j * step_last_dim + begin_last_dim], req, val);
    }
  }
};

template<typename xpu>
void SliceAssignScalarOpForward(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  if (req[0] == kWriteTo) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> in = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Copy(out, in, s);
    });
  } else if (req[0] != kWriteInplace) {
    LOG(FATAL) << "_crop_assign_scalar only supports kWriteTo and kWriteInplace";
  }

  mxnet::TShape vshape = data.shape_;
  const SliceAssignScalarParam& param = nnvm::get<SliceAssignScalarParam>(attrs.parsed);
  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    bool zero_size_shape = GetIndexRange(data.shape_, param.begin, param.end, param.step,
                                        &begin, &end, &step);
    if (zero_size_shape) {
      return;  // slice_assign of zero-sized subspaced needs no operation.
    }
    for (index_t i = 0; i < param.begin.ndim(); ++i) {
      const index_t b = begin[i], e = end[i], s = step[i];
      SetSliceOpOutputDimSize(data.shape_, i, b, e, s, &vshape);
    }
    MSHADOW_TYPE_SWITCH_WITH_BOOL(out.type_flag_, DType, {
      mxnet_op::Kernel<slice_assign_scalar<ndim>, xpu>::Launch(s, vshape.FlatTo2D()[0],
          out.dptr<DType>(), static_cast<DType>(param.scalar), req[0],
          out.shape_.get<ndim>(), vshape.get<ndim>(), begin, step);
    })
  })
}

struct SliceAxisParam : public dmlc::Parameter<SliceAxisParam> {
  int axis;
  index_t begin;
  dmlc::optional<index_t> end;
  DMLC_DECLARE_PARAMETER(SliceAxisParam) {
    DMLC_DECLARE_FIELD(axis)
      .describe("Axis along which to be sliced, supports negative indexes.");
    DMLC_DECLARE_FIELD(begin)
      .describe("The beginning index along the axis to be sliced, "
                " supports negative indexes.");
    DMLC_DECLARE_FIELD(end)
      .describe("The ending index along the axis to be sliced, "
                " supports negative indexes.");
  }
};

inline void GetSliceAxisParams(const SliceAxisParam& param, const mxnet::TShape& ishape,
                           int* axis, index_t* begin, index_t* end) {
  *axis = param.axis;
  if (*axis < 0) {
    *axis += ishape.ndim();
  }
  CHECK(*axis < ishape.ndim() && *axis >= 0) <<
    "Transformed axis must be smaller than the source ndim and larger than zero! Recieved axis=" <<
    param.axis << ", src_ndim=" << ishape.ndim() << ", transformed axis=" << *axis;
  index_t axis_size = static_cast<index_t>(ishape[*axis]);
  *begin = param.begin;
  *end = -1;
  if (*begin < 0) {
    *begin += axis_size;
  }
  if (axis_size > 0) {
    if (!static_cast<bool>(param.end)) {
      *end = axis_size;
    } else {
      *end = param.end.value();
      if (*end < 0) {
        *end += axis_size;
      }
    }
    CHECK(*end <= axis_size) << "Invalid end for end=" << *end << " as axis_size is " << axis_size;
    CHECK((*begin < *end))
      << "Invalid begin, end, get begin=" << param.begin << ", end=" << param.end;
  } else {
    *begin = 0;
    *end = 0;
  }
  CHECK(*end >= 0)
    << "Invalid begin, end, get begin=" << param.begin << ", end=" << param.end;
  CHECK(*begin >= 0) << "Invalid begin for begin=" << param.begin;
}

inline bool SliceAxisShape(const nnvm::NodeAttrs& attrs,
                       mxnet::ShapeVector *in_attrs,
                       mxnet::ShapeVector *out_attrs) {
  const SliceAxisParam& param = nnvm::get<SliceAxisParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(ishape)) return false;
  int axis;
  index_t begin, end;
  GetSliceAxisParams(param, ishape, &axis, &begin, &end);
  if (!mxnet::dim_size_is_known(ishape, axis)) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, ishape);
    return false;
  }
  mxnet::TShape shape(ishape.ndim(), -1);
  for (int i = 0; i < ishape.ndim(); ++i) {
    if (i == axis) {
      shape[i] = static_cast<index_t>(end - begin);
    } else {
      shape[i] = ishape[i];
    }
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  return shape_is_known(shape);
}


template<typename xpu>
void SliceAxis(const nnvm::NodeAttrs& attrs,
           const OpContext& ctx,
           const std::vector<TBlob>& inputs,
           const std::vector<OpReqType>& req,
           const std::vector<TBlob>& outputs) {
  using namespace mshadow::expr;
  const SliceAxisParam& param = nnvm::get<SliceAxisParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int axis;
  index_t begin, end;
  GetSliceAxisParams(param, inputs[0].shape_, &axis, &begin, &end);
  int ndim = outputs[0].ndim();

  if (axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> in =
            inputs[0].FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> out =
            outputs[0].FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(out, req[0], slice<1>(in, begin, end));
      });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> in =
            inputs[0].FlatTo3D<xpu, DType>(axis, s);
        mshadow::Tensor<xpu, 3, DType> out =
            outputs[0].FlatTo3D<xpu, DType>(axis, s);
        ASSIGN_DISPATCH(out, req[0], slice<1>(in, begin, end));
      });
  }
}

// Backward pass of broadcast over the given axis
template<typename xpu>
void SliceAxisGrad_(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob>& inputs,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& outputs) {
  if (outputs[0].shape_.Size() == 0) {
    return;
  }
  const SliceAxisParam& param = nnvm::get<SliceAxisParam>(attrs.parsed);
  using namespace mshadow::op;
  using namespace mshadow::expr;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  int axis;
  index_t begin, end;
  GetSliceAxisParams(param, outputs[0].shape_, &axis, &begin, &end);
  int ndim = outputs[0].shape_.ndim();
  if (axis + 1 == ndim) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 2, DType> ograd =
            inputs[0].FlatTo2D<xpu, DType>(s);
        mshadow::Tensor<xpu, 2, DType> igrad =
            outputs[0].FlatTo2D<xpu, DType>(s);
        if (req[0] == kAddTo) {
          slice<1>(igrad, begin, end) += F<identity>(ograd);
        } else if (req[0] == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, begin, end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req[0], kNullOp);
        }
      });
  } else {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        mshadow::Tensor<xpu, 3, DType> ograd =
            inputs[0].FlatTo3D<xpu, DType>(axis, s);
        mshadow::Tensor<xpu, 3, DType> igrad =
            outputs[0].FlatTo3D<xpu, DType>(axis, s);
        if (req[0] == kAddTo) {
          slice<1>(igrad, begin, end) += F<identity>(ograd);
        } else if (req[0] == kWriteTo) {
          igrad = 0.0f;
          slice<1>(igrad, begin, end) = F<identity>(ograd);
        } else {
          CHECK_EQ(req[0], kNullOp);
        }
      });
  }
}

struct SliceLikeParam : public dmlc::Parameter<SliceLikeParam> {
  mxnet::Tuple<int> axes;
  DMLC_DECLARE_PARAMETER(SliceLikeParam) {
    DMLC_DECLARE_FIELD(axes).set_default(mxnet::Tuple<int>())
    .describe("List of axes on which input data will be sliced according to the "
              "corresponding size of the second input. By default will slice on "
              "all axes. Negative axes are supported.");
  }
};

inline bool SliceLikeShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_attrs,
                           mxnet::ShapeVector *out_attrs) {
  const SliceLikeParam& param = nnvm::get<SliceLikeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  mxnet::TShape& ishape = (*in_attrs)[0];
  mxnet::TShape& from_shape = (*in_attrs)[1];
  if (!mxnet::ndim_is_known(ishape) || !mxnet::ndim_is_known(from_shape)) {
    return false;
  }
  if (param.axes.ndim() == 0) {
    CHECK_EQ(ishape.ndim(), from_shape.ndim())
      << "By default slice_axis performs slice on all axes, but ndim mismatch "
         "for inputs: " << ishape.ndim() << " vs. " << from_shape.ndim();
    for (int i = 0; i < ishape.ndim(); ++i) {
      CHECK_GE(ishape[i], from_shape[i])
        << "Slice axis " << i << " with size " << from_shape[i]
        << "exceeds limit of input with size " << ishape[i];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, from_shape);
  } else {
    mxnet::TShape shape(ishape);
    for (int i = 0; i < param.axes.ndim(); ++i) {
      int axis = param.axes[i];
      if (axis < 0) {
        axis += ishape.ndim();
      }
      CHECK_GE(axis, 0)
        << "Slice axis: " << param.axes[i] << " too small";
      CHECK_GT(ishape.ndim(), axis)
        << "Slice axis: " << axis << " exceeds first input: " << ishape.ndim();
      CHECK_GT(from_shape.ndim(), axis)
        << "Slice axis: " << axis << " exceeds second input: " << from_shape.ndim();
      shape[axis] = from_shape[axis];
      CHECK_GE(ishape[axis], from_shape[axis])
        << "Slice axis " << axis << " with size " << from_shape[axis]
        << "exceeds limit of input with size " << ishape[axis];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  }
  return true;
}

inline void SliceLikeInferRanges(const mxnet::TShape& dshape,
                                 const mxnet::TShape& fshape,
                                 const mxnet::Tuple<int>& axes,
                                 mxnet::Tuple<dmlc::optional<index_t>>* param_begin,
                                 mxnet::Tuple<dmlc::optional<index_t>>* param_end,
                                 mxnet::Tuple<dmlc::optional<index_t>>* param_step) {
  std::vector<dmlc::optional<index_t>> pb(dshape.ndim());
  std::vector<dmlc::optional<index_t>> pe(dshape.ndim());
  std::vector<dmlc::optional<index_t>> ps(dshape.ndim());
  if (axes.ndim() == 0) {
    for (int i = 0; i < dshape.ndim(); ++i) {
      pb[i] = 0;
      pe[i] = fshape[i];
      ps[i] = 1;
    }
  } else {
    for (int i = 0; i < axes.ndim(); ++i) {
      int axis = axes[i];
      if (axis < 0) {
        axis += dshape.ndim();
      }
      CHECK_GE(axis, 0)
        << "Slice axis: " << axes[i] << " too small";
      CHECK_LT(axis, dshape.ndim())
        << "Slice axis: " << axis << " exceeds first input: " << dshape.ndim();
      CHECK_LT(axis, fshape.ndim())
        << "Slice axis: " << axis << " exceeds first input: " << fshape.ndim();
      pb[axis] = 0;
      pe[axis] = fshape[axis];
      ps[axis] = 1;
    }
  }
  *param_begin = mxnet::Tuple<dmlc::optional<index_t>>(pb.begin(), pb.end());
  *param_end = mxnet::Tuple<dmlc::optional<index_t>>(pe.begin(), pe.end());
  *param_step = mxnet::Tuple<dmlc::optional<index_t>>(ps.begin(), ps.end());
}

template<typename xpu>
void SliceLikeForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mshadow::expr;
  const SliceLikeParam& param = nnvm::get<SliceLikeParam>(attrs.parsed);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& data = inputs[0];
  const TBlob& out = outputs[0];
  const mxnet::TShape& ishape = data.shape_;
  const mxnet::TShape& from_shape = inputs[1].shape_;
  mxnet::Tuple<dmlc::optional<index_t>> param_begin;
  mxnet::Tuple<dmlc::optional<index_t>> param_end;
  mxnet::Tuple<dmlc::optional<index_t>> param_step;
  SliceLikeInferRanges(ishape, from_shape, param.axes, &param_begin, &param_end, &param_step);

  MXNET_NDIM_SWITCH(data.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    GetIndexRange(data.shape_, param_begin, param_end, param_step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(out.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        int num_threads = out.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= out.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_forward<ndim, Req, xpu>, xpu>::Launch(s,
            num_threads, out.dptr<DType>(), data.dptr<DType>(),
            data.shape_.get<ndim>(), out.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

template<typename xpu>
void SliceLikeBackward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  if (req[1] != kNullOp && req[1] != kAddTo) {
    Fill(s, outputs[1], req[1], 0);  // Second input not relavant to gradients.
  }
  if (req[0] == kNullOp) return;
  const TBlob& ograd = inputs[0];
  const TBlob& igrad = outputs[0];
  const SliceLikeParam& param = nnvm::get<SliceLikeParam>(attrs.parsed);
  if (req[0] == kWriteTo) {
    Fill(s, igrad, req[0], 0);
  } else if (req[0] == kWriteInplace) {
    LOG(FATAL) << "_slice_like_backward does not support kWriteInplace";
  }

  const mxnet::TShape& ishape = ograd.shape_;
  const mxnet::TShape& from_shape = outputs[1].shape_;
  mxnet::Tuple<dmlc::optional<index_t>> param_begin;
  mxnet::Tuple<dmlc::optional<index_t>> param_end;
  mxnet::Tuple<dmlc::optional<index_t>> param_step;
  SliceLikeInferRanges(ishape, from_shape, param.axes, &param_begin, &param_end, &param_step);

  MXNET_NDIM_SWITCH(ograd.ndim(), ndim, {
    common::StaticArray<index_t, ndim> begin, end, step;
    GetIndexRange(ograd.shape_, param_begin, param_end, param_step, &begin, &end, &step);
    MSHADOW_TYPE_SWITCH(ograd.type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        int num_threads = ograd.shape_.FlatTo2D()[0];
        if (std::is_same<xpu, gpu>::value) {
          num_threads *= ograd.shape_.get<ndim>()[ndim - 1];
        }
        mxnet_op::Kernel<slice_assign<ndim, Req, xpu>, xpu>::Launch(s, num_threads,
            igrad.dptr<DType>(), ograd.dptr<DType>(),
            igrad.shape_.get<ndim>(), ograd.shape_.get<ndim>(), begin, step);
      })
    })
  })
}

struct ClipParam : public dmlc::Parameter<ClipParam> {
  real_t a_min, a_max;
  DMLC_DECLARE_PARAMETER(ClipParam) {
    DMLC_DECLARE_FIELD(a_min)
    .describe("Minimum value");
    DMLC_DECLARE_FIELD(a_max)
    .describe("Maximum value");
  }

  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream a_min_s, a_max_s;
    a_min_s << a_min;
    a_max_s << a_max;
    (*dict)["a_min"] = a_min_s.str();
    (*dict)["a_max"] = a_max_s.str();
  }
};


struct clip {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* datas,
                                  const float a_min, const float a_max) {
    DType data = datas[i];
    if (data > a_max) {
      out[i] = a_max;
    } else if (data < a_min) {
      out[i] = a_min;
    } else {
      out[i] = data;
    }
  }
};


struct clip_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out, const DType* grad, const DType* datas,
                                  const float a_min, const float a_max) {
    DType data = datas[i];
    if (data > a_max) {
      out[i] = 0;
    } else if (data < a_min) {
      out[i] = 0;
    } else {
      out[i] = grad[i];
    }
  }
};


template<typename xpu>
void Clip(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  const ClipParam& param = nnvm::get<ClipParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    mxnet_op::Kernel<mxnet::op::clip, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
                                                   inputs[0].dptr<DType>(),
                                                   param.a_min, param.a_max);
  });
}

template<typename xpu>
void ClipEx(const nnvm::NodeAttrs& attrs,
            const OpContext& ctx,
            const std::vector<NDArray>& inputs,
            const std::vector<OpReqType>& req,
            const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs[0].dtype(), outputs[0].dtype());
  CHECK_EQ(inputs[0].storage_type(), outputs[0].storage_type());
  CHECK_NE(inputs[0].storage_type(), kDefaultStorage);
  UnaryOp::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Clip<xpu>);
}

template<typename xpu>
void ClipGrad_(const nnvm::NodeAttrs& attrs,
               const OpContext& ctx,
               const std::vector<TBlob>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const ClipParam& param = nnvm::get<ClipParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<clip_grad, xpu>::Launch(s, outputs[0].Size(), outputs[0].dptr<DType>(),
    inputs[0].dptr<DType>(), inputs[1].dptr<DType>(), param.a_min, param.a_max);
  });
}

/*!
 * \brief The parameters of the repeat operator include
 * the number of repeating time and axis (optional).
 * The parameters will be later used to deduce the
 * output ndarray shape in bool RepeatShape() function.
 */
struct RepeatParam : public dmlc::Parameter<RepeatParam> {
  int repeats = 1;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(RepeatParam) {
    DMLC_DECLARE_FIELD(repeats)
      .describe("The number of repetitions for each element.");
    DMLC_DECLARE_FIELD(axis)
      .set_default(dmlc::optional<int>())
      .describe("The axis along which to repeat values."
                " The negative numbers are interpreted counting from the backward."
                " By default, use the flattened input array,"
                " and return a flat output array.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream repeats_s, axis_s;
    repeats_s << repeats;
    axis_s << axis;
    (*dict)["repeats"] = repeats_s.str();
    (*dict)["axis"] = axis_s.str();
  }
};

/*!
 * \brief Helper function for getting user input params for the operator repeat.
 * Sanity check the user input values.
 */
inline void GetRepeatParams(const RepeatParam& param, const mxnet::TShape& ishape,
                            int* repeats, dmlc::optional<int>* axisOpt) {
  *repeats = param.repeats;
  CHECK_GE(*repeats, 0) << "repeats cannot be a negative number";
  *axisOpt = param.axis;
  if (static_cast<bool>(*axisOpt)) {
    int ndims = ishape.ndim();
    int axis = axisOpt->value();
    if (axis < 0) {
      axis += ndims;
    }
    CHECK(axis >= 0 && axis < ndims) << "axis = " << axisOpt->value() << " out of bounds";
  }
}

inline bool RepeatOpShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector *in_attrs,
                        mxnet::ShapeVector *out_attrs) {
  const RepeatParam& param = nnvm::get<RepeatParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }
  int repeats = 0;
  dmlc::optional<int> axisOpt;
  GetRepeatParams(param, ishape, &repeats, &axisOpt);
  // If 0 repeats, return an empty 1-dim, 0-size array
  if (0 == repeats) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(1, 0));
    return true;
  }

  // If repeats > 0, multiply the size of the corresponding axis by repeats
  if (static_cast<bool>(axisOpt)) {
    int ndims = ishape.ndim();
    int axis = axisOpt.value();
    if (axis < 0) {
      axis += ndims;
    }
    mxnet::TShape shape(ishape.ndim(), -1);
    for (int i = 0; i < ishape.ndim(); ++i) {
      if (i == axis) {
        shape[i] = repeats * ishape[i];
      } else {
        shape[i] = ishape[i];
      }
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  } else {  // If axis is not input by user, return a flat 1D array of size = in.size*repeats
    mxnet::TShape shape(1, ishape.Size() * repeats);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, shape);
  }
  return shape_is_known(out_attrs->at(0));
}

inline bool RepeatOpType(const nnvm::NodeAttrs& attrs,
                         std::vector<int>* in_attrs,
                         std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  if ((*in_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  } else if ((*out_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  }
  return true;
}

/*!
 * \brief Reshape the input and output tensors for
 * using broadcast_to to achieve the funcitonality
 * of operator repeat.
 * \return a pair of mxnet::TShape's, first is the reshaped
 * input shape, second is the reshaped output shape.
 */
inline std::pair<mxnet::TShape, mxnet::TShape> ReshapeInputOutputForRepeatOp(
  const mxnet::TShape& ishape,
  const dmlc::optional<int>& axisOpt,
  const int repeats) {
  if (static_cast<bool>(axisOpt)) {
    int axis = axisOpt.value();
    int ndim = ishape.ndim();
    if (axis < 0)  {
      axis += ndim;
    }
    CHECK(axis >= 0 && axis < ishape.ndim()) << "Invalid input of axis";

    // reshape the input tensor by adding a dim at the (axis+1)-th dim
    mxnet::TShape rshape(ishape.ndim()+1, 1);
    // the shape we want to broadcast to
    mxnet::TShape bshape(rshape.ndim(), 1);
    int i = 0;
    while (i <= axis) {
      rshape[i] = bshape[i] = ishape[i];
      ++i;
    }
    rshape[i] = 1;
    bshape[i] = repeats;
    while (i < ishape.ndim()) {
      rshape[i+1] = ishape[i];
      bshape[i+1] = ishape[i];
      ++i;
    }
    return std::make_pair(rshape, bshape);
  } else {
    // axis is not input by user
    // reshape the tensor into shape (ishape.Size(), 1)
    // then add one dim at axis = 1 and broadcast to
    // shape (ishape.Size(), repeats)
    mxnet::TShape rshape(2, 1);
    rshape[0] = ishape.Size();
    rshape[1] = 1;

    mxnet::TShape bshape(2, 1);
    bshape[0] = rshape[0];
    bshape[1] = repeats;
    return std::make_pair(rshape, bshape);
  }
}

template<typename xpu>
void RepeatOpForward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  const TBlob& iTBlob = inputs[0];
  const mxnet::TShape& ishape = iTBlob.shape_;
  if (!shape_is_known(ishape)) return;

  int repeats = 0;
  dmlc::optional<int> axisOpt;
  const RepeatParam& param = nnvm::get<RepeatParam>(attrs.parsed);
  GetRepeatParams(param, ishape, &repeats, &axisOpt);
  if (0 == repeats) return;

  std::pair<mxnet::TShape, mxnet::TShape> rshapes = \
    ReshapeInputOutputForRepeatOp(ishape, axisOpt, repeats);

  // reshaped input tblob
  TBlob iblob(inputs[0].dptr_, rshapes.first, inputs[0].dev_mask(),
    inputs[0].type_flag_, inputs[0].dev_id());
  std::vector<TBlob> newInputs = {iblob};

  // reshaped output tblob
  TBlob oblob(outputs[0].dptr_, rshapes.second, outputs[0].dev_mask(),
    outputs[0].type_flag_, outputs[0].dev_id());
  std::vector<TBlob> newOutputs = {oblob};

  BroadcastCompute<xpu>(attrs, ctx, newInputs, req, newOutputs);
}

/*!
 * \brief Compute the gradient of the loss function
 * with respect to the input of the operator.
 * Backpropagation is employed to implement the
 * chain rule.
 * \param inputs the gradient of the loss function
 * with respect to the outputs of the operator
 * \param outputs the gradient of the loss function
 * with respect to the inputs of the operator
 */
template<typename xpu>
void RepeatOpBackward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  const mxnet::TShape& oshape = outputs[0].shape_;
  if (!shape_is_known(oshape)) return;

  int repeats = 0;
  dmlc::optional<int> axisOpt;
  const RepeatParam& param = nnvm::get<RepeatParam>(attrs.parsed);
  GetRepeatParams(param, oshape, &repeats, &axisOpt);
  if (0 == repeats) return;

  std::pair<mxnet::TShape, mxnet::TShape> rshapes =
    ReshapeInputOutputForRepeatOp(oshape, axisOpt, repeats);

  // reshaped output grad tblob
  TBlob oblob(outputs[0].dptr_, rshapes.first, outputs[0].dev_mask(),
    outputs[0].type_flag_, outputs[0].dev_id());
  std::vector<TBlob> newOutputs = {oblob};

  // reshaped input grad tblob
  TBlob iblob(inputs[0].dptr_, rshapes.second, inputs[0].dev_mask(),
    inputs[0].type_flag_, inputs[0].dev_id());
  std::vector<TBlob> newInputs = {iblob};

#if !defined(__CUDACC__)
  ReduceAxesComputeImpl<xpu, mshadow::red::sum, false, false>(
      ctx, newInputs, req, newOutputs, rshapes.first);
#else
  ReduceAxesRTCComputeImpl(ctx, newInputs, req, newOutputs, rshapes.first,
                           "red::sum{}", nullptr, false);
#endif
}

struct TileParam : public dmlc::Parameter<TileParam> {
  mxnet::Tuple<int> reps;
  DMLC_DECLARE_PARAMETER(TileParam) {
    DMLC_DECLARE_FIELD(reps)
      .describe("The number of times for repeating the tensor a. Each dim size of reps"
                " must be a positive integer."
                " If reps has length d, the result will have dimension of max(d, a.ndim);"
                " If a.ndim < d, a is promoted to be d-dimensional by prepending new axes."
                " If a.ndim > d, reps is promoted to a.ndim by pre-pending 1's to it.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream reps_s;
    reps_s << reps;
    (*dict)["reps"] = reps_s.str();
  }
};

inline bool TileOpShape(const nnvm::NodeAttrs& attrs,
                        mxnet::ShapeVector *in_attrs,
                        mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const TileParam& param = nnvm::get<TileParam>(attrs.parsed);
  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!shape_is_known(ishape)) {
    return false;
  }
  const mxnet::Tuple<int>& reps = param.reps;
  // If reps is empty, return a identical input array
  if (reps.ndim() == 0) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, ishape);
    return true;
  }
  mxnet::TShape oshape(std::max(ishape.ndim(), reps.ndim()), -1);
  int i1 = ishape.ndim() - 1;
  int i2 = reps.ndim() - 1;
  for (int i = oshape.ndim() - 1; i >= 0; --i) {
    if (i1 >= 0 && i2 >= 0) {
      oshape[i] = ishape[i1--] * reps[i2--];
    } else if (i1 >= 0) {
      oshape[i] = ishape[i1--];
    } else if (i2 >= 0) {
      oshape[i] = reps[i2--];
    }
  }
  // If reps contains 0s, oshape is a zero-size shape.
  // Need to distinguish between np_shape mode and legacy mode.
  if (!Imperative::Get()->is_np_shape()) {
    common::ConvertToNumpyShape(&oshape);
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  return shape_is_known(oshape);
}

inline bool TileOpType(const nnvm::NodeAttrs& attrs,
                       std::vector<int>* in_attrs,
                       std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  if ((*in_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, (*in_attrs)[0]);
  } else if ((*out_attrs)[0] != -1) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, (*out_attrs)[0]);
  }
  return true;
}

/*!
 * \brief Reshape the input and output tensors for
 * using broadcast_to to achieve the functionality
 * of operator tile.
 * \return a pair of mxnet::TShape's, first is the reshaped
 * input shape, second is the reshaped output shape.
 */
inline std::pair<mxnet::TShape, mxnet::TShape> ReshapeInputOutputForTileOp(
  const mxnet::TShape& ishape,
  const mxnet::Tuple<int>& reps) {
  if (reps.ndim() == 0) {
    return std::make_pair(ishape, ishape);
  }

  // The shape we want to broadcast to
  mxnet::TShape bshape(std::max(ishape.ndim(), reps.ndim()) * 2, 1);

  // The shape of the input tensor after adding new axes before each dim
  mxnet::TShape rshape(bshape.ndim(), 1);

  int i1 = ishape.ndim() - 1;
  int i2 = reps.ndim() - 1;
  for (int i = bshape.ndim() - 1; i >= 0; --i) {
    if (0 == (i & 1)) {
      bshape[i] = (i2 >= 0? reps[i2--] : 1);
      rshape[i] = 1;
    } else {
      rshape[i] = bshape[i] = (i1 >= 0? ishape[i1--] : 1);
    }
  }

  return std::make_pair(rshape, bshape);
}

/*!
 * \brief Implementation of tiling the input tensor a based
 * on the user-input shape, reps.
 * If a.ndim < reps.ndim, new axes are pre-pended to a. For example,
 * the input tensor has shape (3,), and the reps is (2, 4); the input
 * tensor would be reshaped to (1, 3).
 * If a.ndim > reps.ndim, pre-pending 1's to reps. For example,
 * the input tensor has shape (2, 3, 4, 5), and reps is (2, 2);
 * the reps would be changed to (1, 1, 2, 2).
 * Suppose we have a.ndim = reps.ndim now. To achieve tiling,
 * we utilize the operator broadcast_to. For example, for a tensor
 * of shape (2, 3, 4, 5) and reps (2, 8, 9, 3), we first reshape
 * the tensor to the shape (1, 2, 1, 3, 1, 4, 1, 5) by adding
 * one axis before each dimension. Then, we want to broadcast
 * the new tensor to shape (2, 2, 8, 3, 9, 4, 3, 5). The final
 * output tensor would have shape (2*2, 8*3, 9*4, 3*5).
 */
template<typename xpu>
void TileOpForward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  if (inputs[0].Size() == 0) return;
  const mxnet::TShape& ishape = inputs[0].shape_;
  const mxnet::Tuple<int>& reps = nnvm::get<TileParam>(attrs.parsed).reps;

  // If any one of the number in reps is zero, return immediately
  for (int i = 0; i < reps.ndim(); ++i) {
    if (0 == reps[i]) return;
  }

  std::pair<mxnet::TShape, mxnet::TShape> rshapes = ReshapeInputOutputForTileOp(ishape, reps);

  // reshaped input tblob
  TBlob iblob(inputs[0].dptr_, rshapes.first, inputs[0].dev_mask(),
    inputs[0].type_flag_, inputs[0].dev_id());
  std::vector<TBlob> newInputs = {iblob};
  // reshaped output tblob
  TBlob oblob(outputs[0].dptr_, rshapes.second, outputs[0].dev_mask(),
    outputs[0].type_flag_, outputs[0].dev_id());
  std::vector<TBlob> newOutputs = {oblob};

  BroadcastCompute<xpu>(attrs, ctx, newInputs, req, newOutputs);
}

/*!
 * \brief Compute the gradient of the loss function
 * with respect to the input of the operator.
 * Backpropagation is employed to implement the
 * chain rule.
 * \param inputs the gradient of the loss function
 * with respect to the outputs of the operator
 * \param outputs the gradient of the loss function
 * with respect to the inputs of the operator
 */
template<typename xpu>
void TileOpBackward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);

  if (inputs[0].Size() == 0) return;
  const mxnet::TShape& oshape = outputs[0].shape_;
  const mxnet::Tuple<int>& reps = nnvm::get<TileParam>(attrs.parsed).reps;

  // If any one of the number in reps is zero, return immediately
  for (int i = 0; i < reps.ndim(); ++i) {
    if (0 == reps[i]) return;
  }

  std::pair<mxnet::TShape, mxnet::TShape> rshapes = ReshapeInputOutputForTileOp(oshape, reps);

  // reshaped output grad tblob
  TBlob oblob(outputs[0].dptr_, rshapes.first, outputs[0].dev_mask(),
    outputs[0].type_flag_, outputs[0].dev_id());
  std::vector<TBlob> newOutputs = {oblob};
  // reshaped input grad tblob
  TBlob iblob(inputs[0].dptr_, rshapes.second, inputs[0].dev_mask(),
    inputs[0].type_flag_, inputs[0].dev_id());
  std::vector<TBlob> newInputs = {iblob};

#if !defined(__CUDACC__)
  ReduceAxesComputeImpl<xpu, mshadow::red::sum, false, false>(
      ctx, newInputs, req, newOutputs, rshapes.first);
#else
  ReduceAxesRTCComputeImpl(ctx, newInputs, req, newOutputs, rshapes.first,
                           "red::sum{}", nullptr, false);
#endif
}

struct ReverseParam : public dmlc::Parameter<ReverseParam> {
  mxnet::Tuple<int> axis;
  DMLC_DECLARE_PARAMETER(ReverseParam) {
    DMLC_DECLARE_FIELD(axis)
    .describe("The axis which to reverse elements.");
  }
};


#define REVERSE_MAX_DIM 10U

struct reverse {
  MSHADOW_XINLINE static index_t ReverseIndex(index_t idx,
                                              index_t nreversedim,
                                              const index_t * stride_,
                                              const index_t * trailing_) {
    index_t outputIndex = idx;
    for (index_t i = 0; i < nreversedim; ++i) {
      const index_t low = outputIndex % trailing_[i];
      index_t high = outputIndex / trailing_[i];
      const index_t x = high%stride_[i];
      high /= stride_[i];
      outputIndex = (high*stride_[i] + stride_[i] - 1 - x)*trailing_[i] + low;
    }
    return outputIndex;
  }
#ifdef __CUDACC__
  template<typename DType>
  __device__  static void Map(index_t index, index_t nreversedim, const DType *src, DType *dst,
                              const index_t * stride_,
                              const index_t * trailing_) {
    __shared__ index_t stride_share[REVERSE_MAX_DIM];
    __shared__ index_t trailing_share[REVERSE_MAX_DIM];
    if (threadIdx.x < REVERSE_MAX_DIM) {
      stride_share[threadIdx.x] = stride_[threadIdx.x];
      trailing_share[threadIdx.x] = trailing_[threadIdx.x];
    }
    __syncthreads();
    index_t new_idx = ReverseIndex(index, nreversedim, stride_share, trailing_share);
    dst[new_idx] = src[index];
  }
#else
  template<typename DType>
  MSHADOW_XINLINE  static void Map(index_t index, index_t nreversedim, const DType *src, DType *dst,
                                   const index_t * stride_,
                                   const index_t * trailing_) {
    index_t new_idx = ReverseIndex(index, nreversedim, stride_, trailing_);
    dst[new_idx] = src[index];
  }
#endif
};


template<typename xpu>
void ReverseOpForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  const ReverseParam& param = nnvm::get<ReverseParam>(attrs.parsed);
  CHECK_EQ(inputs[0].type_flag_, outputs[0].type_flag_);
  CHECK_LT(param.axis.ndim(), REVERSE_MAX_DIM);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const mxnet::TShape& ishape = inputs[0].shape_;

  std::vector<index_t> stride_(param.axis.ndim());
  std::vector<index_t>  trailing_(param.axis.ndim());
  index_t reverse_index = 0;
  for (int axis : param.axis) {
    CHECK_LT(axis, ishape.ndim());
    stride_[reverse_index] = ishape[axis];
    trailing_[reverse_index] = 1;
    for (int i2 = axis + 1; i2 < ishape.ndim(); ++i2) {
      trailing_[reverse_index] *= ishape[i2];
    }
    reverse_index++;
  }

#ifdef __CUDACC__
  mshadow::Tensor<xpu, 1, uint8_t> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, uint8_t>(
      mshadow::Shape1(reverse_index * sizeof(index_t) * 2), s);

  auto stride_workspace = workspace.dptr_;
  auto trailing_workspace = workspace.dptr_ + reverse_index * sizeof(index_t);

  cudaMemcpyAsync(stride_workspace, thrust::raw_pointer_cast(stride_.data()),
                  stride_.size() * sizeof(index_t),
                  cudaMemcpyHostToDevice, mshadow::Stream<gpu>::GetStream(s));
  cudaMemcpyAsync(trailing_workspace, thrust::raw_pointer_cast(trailing_.data()),
                  trailing_.size() * sizeof(index_t),
                  cudaMemcpyHostToDevice, mshadow::Stream<gpu>::GetStream(s));

#endif

#ifdef __CUDACC__
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<reverse, xpu>::Launch(s, inputs[0].Size(), reverse_index,
    inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
    reinterpret_cast<index_t*>(stride_workspace), reinterpret_cast<index_t*>(trailing_workspace));
  });
#else
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<reverse, xpu>::Launch(s, inputs[0].Size(), reverse_index,
    inputs[0].dptr<DType>(), outputs[0].dptr<DType>(),
    stride_.data(), trailing_.data());
  });
#endif
}


struct StackParam : public dmlc::Parameter<StackParam> {
  int axis;
  int num_args;
  DMLC_DECLARE_PARAMETER(StackParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_default(0)
    .describe("The axis in the result array along which the input arrays are stacked.");
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be stacked.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s, num_args_s;
    axis_s << axis;
    num_args_s << num_args;
    (*dict)["axis"] = axis_s.str();
    (*dict)["num_args"] = num_args_s.str();
  }
};


inline bool StackOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const StackParam& param = dmlc::get<StackParam>(attrs.parsed);

  mxnet::TShape dshape;
  for (const mxnet::TShape& i : (*in_attrs)) {
    shape_assign(&dshape, i);
  }
  if (!shape_is_known(dshape)) return false;

  mxnet::TShape oshape(dshape.ndim() + 1, -1);
  int axis = CheckAxis(param.axis, oshape.ndim());
  for (int i = 0; i < axis; ++i) {
    oshape[i] = dshape[i];
  }
  oshape[axis] = param.num_args;
  for (index_t i = axis + 1; i < oshape.ndim(); ++i) {
    oshape[i] = dshape[i-1];
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(oshape);
}


template<typename xpu>
void StackOpForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, outputs[0].ndim());

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_BOOL(outputs[0].type_flag_, DType, {
    std::vector<Tensor<xpu, 3, DType> > data(inputs.size());
    Tensor<xpu, 3, DType> out;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < axis; ++i) {
      leading *= outputs[0].shape_[i];
    }
    for (int i = axis + 1; i < outputs[0].ndim(); ++i) {
      trailing *= outputs[0].shape_[i];
    }
    size_t mid = outputs[0].shape_[axis];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    out = outputs[0].get_with_shape<xpu, 3, DType>(oshape, s);

    for (size_t i = 0; i < inputs.size(); ++i) {
      Shape<3> dshape = Shape3(leading, 1, trailing);
      data[i] = inputs[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Concatenate(data, &out, 1, req[0]);
  })
}

template<typename xpu>
void StackOpBackward(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const StackParam& param = dmlc::get<StackParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, inputs[0].ndim());

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH_WITH_BOOL(inputs[0].type_flag_, DType, {
    std::vector<Tensor<xpu, 3, DType> > grad_in(outputs.size());
    Tensor<xpu, 3, DType> grad;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < axis; ++i) {
      leading *= inputs[0].shape_[i];
    }
    for (int i = axis + 1; i < inputs[0].ndim(); ++i) {
      trailing *= inputs[0].shape_[i];
    }
    size_t mid = inputs[0].shape_[axis];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    grad = inputs[0].get_with_shape<xpu, 3, DType>(oshape, s);

    for (size_t i = 0; i < outputs.size(); ++i) {
      Shape<3> dshape = Shape3(leading, 1, trailing);
      grad_in[i] = outputs[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Split(grad, &grad_in, 1, req);
  })
}

struct SqueezeParam : public dmlc::Parameter<SqueezeParam> {
  dmlc::optional<mxnet::Tuple<int>> axis;
  DMLC_DECLARE_PARAMETER(SqueezeParam) {
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<mxnet::Tuple<int>>())
    .describe("Selects a subset of the single-dimensional entries in the shape."
              " If an axis is selected with shape entry greater than one, an error is raised.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream axis_s;
    axis_s << axis;
    (*dict)["axis"] = axis_s.str();
  }
};

// Given a shape that may have dim size equal to 0,
// move all the zeros to the last of the shape array
// and keep the relative order of the non-zero values.
// Returns the new shape size after moving all zeros to the end.
inline size_t SqueezeShapeHelper(mxnet::TShape* shape) {
  CHECK(shape != nullptr);
  size_t count = 0;
  for (int i = 0; i < shape->ndim(); ++i) {
    if ((*shape)[i] == -1) {
      ++count;
    } else {
      std::swap((*shape)[i], (*shape)[i-count]);
    }
  }
  return shape->ndim() - count;
}

inline bool SqueezeShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const SqueezeParam& param = nnvm::get<SqueezeParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U) << "Input: [data]";
  CHECK_EQ(out_attrs->size(), 1U);
  const mxnet::TShape& dshape = in_attrs->at(0);
  const int dndim = dshape.ndim();
  if (!shape_is_known(dshape)) return false;
  mxnet::TShape oshape = dshape;
  if (param.axis.has_value()) {
    // preprocess axis
    mxnet::Tuple<int> axes = param.axis.value();
    for (int i = 0; i < axes.ndim(); ++i) {
      if (axes[i] < 0) {
        axes[i] += dndim;
        CHECK_GE(axes[i], 0)
          << "axis " << axes[i] - dndim << " is out of bounds for array of dimension " << dndim;
      }
      CHECK_LT(axes[i], dndim)
        << "axis " << axes[i] << " is out of bounds for array of dimension " << dndim;
      CHECK_EQ(dshape[axes[i]], 1)
        << "cannot select an axis to squeeze out which has size="
        << dshape[axes[i]] << " not equal to one";
      CHECK_NE(oshape[axes[i]], -1) << "duplicate value in axis";
      oshape[axes[i]] = -1;
    }
  } else {
    for (int i = 0; i < oshape.ndim(); ++i) {
      if (oshape[i] == 1) oshape[i] = -1;
    }
  }
  size_t oshape_size = SqueezeShapeHelper(&oshape);
  if (oshape_size == 0) {  // corner case when dshape is (1, 1, 1, 1)
    oshape[0] = 1;
    oshape_size = 1;
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(oshape.data(), oshape.data()+oshape_size));
  return true;
}

struct DepthToSpaceParam : public dmlc::Parameter<DepthToSpaceParam> {
  int block_size;
  DMLC_DECLARE_PARAMETER(DepthToSpaceParam) {
    DMLC_DECLARE_FIELD(block_size)
      .describe("Blocks of [block_size. block_size] are moved");
  }
};

inline bool DepthToSpaceOpShape(const nnvm::NodeAttrs& attrs,
                                mxnet::ShapeVector* in_attrs,
                                mxnet::ShapeVector* out_attrs) {
  const DepthToSpaceParam& param = nnvm::get<DepthToSpaceParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ(in_attrs->at(0).ndim(), 4) << "Operation Depth To Space requires exactly 4D tensor";

  mxnet::TShape expected_out(4, -1);

  mxnet::TShape& in_shape = in_attrs->at(0);
  if (!mxnet::ndim_is_known(in_shape)) {
    return false;
  }
  int block = param.block_size;
  CHECK_NE(block, 0) << "block_size must be a positive integer value";
  CHECK_NE(in_shape[1], 0) << "Depth dimension:1 cannot be 0";
  CHECK_EQ(in_shape[1] % (block * block), 0)
    << "Cannot perform Depth To Space operation on the specified tensor."
       " Dimension:1(depth dimension) should be a multiple of 'block^2'";
  CHECK_NE(in_shape[0], 0)
    << "Operation requires a 4D tensor. Size of dimension:0 cannot be 0";
  CHECK_NE(in_shape[2], 0)
    << "Operation requires a 4D tensor. Size of dimension:2 cannot be 0";
  CHECK_NE(in_shape[3], 0)
    << "Operation requires a 4D tensor. Size of dimension:3 cannot be 0";

  expected_out[0] = in_shape[0];
  expected_out[1] = in_shape[1] / (block * block);
  int i = 2;
  while (i < expected_out.ndim()) {
    expected_out[i] = in_shape[i] * block;
    ++i;
  }

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, expected_out);
  return shape_is_known(expected_out);
}

inline bool DepthToSpaceOpType(const nnvm::NodeAttrs& attrs,
                               std::vector<int>* in_attrs,
                               std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

/*!
 * \brief This function updates the value of input index from where the data element
 * needs to be fetched and written out to the ith location in output tensor
 * \param index_position    index within offset array to get offset of given dimension
 * \param dim_size          size of current dimension
 * \param idx               output tensor index
 * \param inp_index         index within input tensor from where value is retrieved
 * \param offset_arr        array containing the linear offset of input tensor
 */
MSHADOW_XINLINE void update_index(index_t index_position, index_t dim_size, index_t *idx,
                                  index_t *inp_index, const index_t* offset_arr) {
  index_t next_idx_val = *idx / dim_size;
  *inp_index += (*idx - next_idx_val * dim_size) * offset_arr[index_position];
  *idx = next_idx_val;
}

/*!
 * \brief This function performs the tensor transpose (0, 1, 2, 3, 4, 5) ->
 * (0, 3, 4, 1, 5, 2) by computing linear index within input tensor to be mapped
 * to the ith index of output tensor
 * \param i           tensor index
 * \param out_data    output tensor
 * \param in_data     input tensor
 * \param block       size of chunks to be moved out of depth dimension
 * \param size        array containing the size of each dimension of input tensor
 * \param offset_arr  array containing the linear offset of input tensor
 */
template<int req>
struct depth_to_space_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const DType* in_data,
                                  const int block, const index_t* size, const index_t* offset_arr) {
    index_t inp_index = 0, idx = i, dim_size;
    dim_size = block;
    update_index(2, dim_size, &idx, &inp_index, offset_arr);
    dim_size = size[3];
    update_index(5, dim_size, &idx, &inp_index, offset_arr);
    dim_size = block;
    update_index(1, dim_size, &idx, &inp_index, offset_arr);
    dim_size = size[2];
    update_index(4, dim_size, &idx, &inp_index, offset_arr);
    dim_size = size[1] / (block * block);
    update_index(3, dim_size, &idx, &inp_index, offset_arr);
    dim_size = size[0];
    update_index(0, dim_size, &idx, &inp_index, offset_arr);
    KERNEL_ASSIGN(out_data[i], req, in_data[inp_index]);
  }
};

/*!
 * \brief This function calculates the linear offset for each dimension of
 * input tensor and stores them in an array, which is later used in
 * performing depth_to_space operation
 * \param i           global thread id
 * \param offset_arr  array to be populated with offset values
 * \param size        array to be populated with size of each dimension of input tensor
 * \param block       size of chunks to be moved out of depth dimension
 * \param size0       size of Dim 0 of input tensor
 * \param size1       size of Dim 1 of input tensor
 * \param size2       size of Dim 2 of input tensor
 * \param size3       size of Dim 3 of input tensor
 */
template<int req>
struct compute_offset_for_depth_to_space {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* offset_arr, DType* size, const int block,
                                  const index_t size0, const index_t size1, const index_t size2,
                                  const index_t size3) {
    size[0] = size0;
    size[1] = size1;
    size[2] = size2;
    size[3] = size3;

    offset_arr[5] = 1;
    offset_arr[4] = offset_arr[5] * size[3];
    offset_arr[3] = offset_arr[4] * size[2];
    offset_arr[2] = offset_arr[3] * size[1] / (block * block);
    offset_arr[1] = offset_arr[2] * block;
    offset_arr[0] = offset_arr[1] * block;
  }
};

template<typename xpu>
void DepthToSpaceOpForward(const nnvm::NodeAttrs& attrs,
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
  const DepthToSpaceParam& param = nnvm::get<DepthToSpaceParam>(attrs.parsed);
  using namespace mxnet_op;
  int block = param.block_size;

  mshadow::Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(mshadow::Shape1(sizeof(index_t) * 10), s);
  char* workspace_curr_ptr = workspace.dptr_;
  index_t* offset_arr = reinterpret_cast<index_t*>(workspace_curr_ptr);
  index_t* size = reinterpret_cast<index_t*>(workspace_curr_ptr + sizeof(index_t) * 6);

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<compute_offset_for_depth_to_space<req_type>, xpu>::Launch(
          s, 1, offset_arr, size, block, in_data.shape_[0], in_data.shape_[1],
          in_data.shape_[2], in_data.shape_[3]);

      Kernel<depth_to_space_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          block, size, offset_arr);
    });
  });
}

inline bool SpaceToDepthOpShape(const nnvm::NodeAttrs& attrs,
                                mxnet::ShapeVector* in_attrs,
                                mxnet::ShapeVector* out_attrs) {
  const DepthToSpaceParam& param = nnvm::get<DepthToSpaceParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_EQ(in_attrs->at(0).ndim(), 4) << "Operation Space To Depth requires exactly 4D tensor";

  mxnet::TShape expected_out(in_attrs->at(0).ndim(), -1);

  mxnet::TShape& in_shape = in_attrs->at(0);
  if (!mxnet::ndim_is_known(in_shape)) {
    return false;
  }
  int block = param.block_size;
  CHECK_NE(block, 0) << "block_size must be a positive integer value";
  CHECK_NE(in_shape[0], 0)
    << "Operation requires a 4D tensor. Size of dimension:0 cannot be 0";
  CHECK_NE(in_shape[1], 0) << "Depth dimension:1 cannot be 0";
  CHECK_NE(in_shape[2], 0)
    << "Operation requires a 4D tensor. Size of dimension:2 cannot be 0";
  CHECK_EQ(in_shape[2] % block, 0)
    << "Cannot perform Depth To Space operation on the specified tensor."
       " Dimension:2(1st Space dimension) should be a multiple of 'block' ";
  CHECK_NE(in_shape[3], 0)
    << "Operation requires a 4D tensor. Size of dimension:3 cannot be 0";
  CHECK_EQ(in_shape[3] % block, 0)
    << "Cannot perform Depth To Space operation on the specified tensor."
       " Dimension:3(2nd space dimension) should be a multiple of 'block' ";

  expected_out[0] = in_shape[0];
  expected_out[1] = in_shape[1] * block * block;
  int i = 2;
  while (i < expected_out.ndim()) {
    expected_out[i] = in_shape[i] / block;
    ++i;
  }

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, expected_out);
  return shape_is_known(expected_out);
}

inline bool SpaceToDepthOpType(const nnvm::NodeAttrs& attrs,
                               std::vector<int>* in_attrs,
                               std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

/*!
 * \brief This function preforms the tensor transpose (0, 1, 2, 3, 4, 5) ->
 * (0, 3, 5, 1, 2, 4) by computing linear index within input tensor to be mapped
 * to the ith index of output tensor
 * \param i           tensor index
 * \param out_data    output tensor
 * \param in_data     input tensor
 * \param block       size of chunks to be moved out of depth dimension
 * \param size        array containing the size of each dimension of input tensor
 * \param offset_arr  array containing the linear offset of input tensor
 */
template<int req>
struct space_to_depth_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* out_data, const DType* in_data, const int block,
                                  const index_t* size, const index_t* offset_arr) {
    index_t inp_index = 0, idx = i, dim_size;
    dim_size = size[3] / block;
    update_index(4, dim_size, &idx, &inp_index, offset_arr);
    dim_size = size[2] / block;
    update_index(2, dim_size, &idx, &inp_index, offset_arr);
    dim_size = size[1];
    update_index(1, dim_size, &idx, &inp_index, offset_arr);
    dim_size = block;
    update_index(5, dim_size, &idx, &inp_index, offset_arr);
    dim_size = block;
    update_index(3, dim_size, &idx, &inp_index, offset_arr);
    dim_size = size[0];
    update_index(0, dim_size, &idx, &inp_index, offset_arr);
    KERNEL_ASSIGN(out_data[i], req, in_data[inp_index]);
  }
};

/*!
 * \brief This function calculates the linear offset for each dimension of
 * input tensor and stores them in an array, which is later used in
 * performing space_to_depth operation
 * \param i           global thread id
 * \param offset_arr  array to be populated with offset values
 * \param size        array to be populated with size of each dimension of input tensor
 * \param block       size of chunks to be moved out of depth dimension
 * \param size0       size of Dim 0 of input tensor
 * \param size1       size of Dim 1 of input tensor
 * \param size2       size of Dim 2 of input tensor
 * \param size3       size of Dim 3 of input tensor
 */
template<int req>
struct compute_offset_for_space_to_depth {
  template<typename DType>
  MSHADOW_XINLINE static void Map(index_t i, DType* offset_arr, DType* size, const int block,
                                  const index_t size0, const index_t size1,
                                  const index_t size2, const index_t size3) {
    size[0] = size0;
    size[1] = size1;
    size[2] = size2;
    size[3] = size3;

    offset_arr[5] = 1;
    offset_arr[4] = offset_arr[5] * block;
    offset_arr[3] = offset_arr[4] * size[3] / block;
    offset_arr[2] = offset_arr[3] * block;
    offset_arr[1] = offset_arr[2] * size[2] / block;
    offset_arr[0] = offset_arr[1] * size[1];
  }
};

template<typename xpu>
void SpaceToDepthOpForward(const nnvm::NodeAttrs& attrs,
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
  const DepthToSpaceParam& param = nnvm::get<DepthToSpaceParam>(attrs.parsed);
  using namespace mxnet_op;
  int block = param.block_size;

  mshadow::Tensor<xpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, char>(mshadow::Shape1(sizeof(index_t) * 10), s);
  char* workspace_curr_ptr = workspace.dptr_;
  index_t* offset_arr = reinterpret_cast<index_t*>(workspace_curr_ptr);
  index_t* size = reinterpret_cast<index_t*>(workspace_curr_ptr + sizeof(index_t) * 6);

  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<compute_offset_for_space_to_depth<req_type>, xpu>::Launch(
          s, 1, offset_arr, size, block, in_data.shape_[0], in_data.shape_[1],
          in_data.shape_[2], in_data.shape_[3]);
      Kernel<space_to_depth_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          block, size, offset_arr);
    });
  });
}

namespace split_enum {
enum SplitOpInputs {kData};
}  // namespace split_enum

struct SplitParam : public dmlc::Parameter<SplitParam> {
  mxnet::TShape indices;
  int axis;
  bool squeeze_axis;
  int sections;
  DMLC_DECLARE_PARAMETER(SplitParam) {
    DMLC_DECLARE_FIELD(indices)
    .describe("Indices of splits. The elements should denote the boundaries of at which split"
              " is performed along the `axis`.");
    DMLC_DECLARE_FIELD(axis).set_default(1)
    .describe("Axis along which to split.");
    DMLC_DECLARE_FIELD(squeeze_axis).set_default(0)
    .describe("If true, Removes the axis with length 1 from the shapes of the output arrays."
              " **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1"
              " only along the `axis` which it is split."
              " Also `squeeze_axis` can be set to ``true``"
              " only if ``input.shape[axis] == num_outputs``.");
    DMLC_DECLARE_FIELD(sections).set_default(0)
    .describe("Number of sections if equally splitted. Default to 0 which means split by indices.");
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream indices_s, axis_s, squeeze_axis_s, sections_s;
    indices_s << indices;
    axis_s << axis;
    squeeze_axis_s << squeeze_axis;
    sections_s << sections;
    (*dict)["indices"] = indices_s.str();
    (*dict)["axis"] = axis_s.str();
    (*dict)["squeeze_axis"] = squeeze_axis_s.str();
    (*dict)["sections"] = sections_s.str();
  }
};  // struct SplitParam

inline mxnet::TShape GetSplitIndices(const mxnet::TShape& ishape, int axis, int sections) {
  mxnet::TShape indices(sections+1, -1);
  indices[0] = 0;
  int64_t section_size_b = (int64_t) (ishape[axis] / sections);
  int64_t section_size_a = section_size_b + 1;
  int section_a = ishape[axis] % sections;
  for (int i = 0; i < sections; ++i) {
    if ( i < section_a ) {
      indices[i+1] = section_size_a * (i + 1);
    } else {
      indices[i+1] = section_size_b + indices[i];
    }
  }
  return indices;
}

inline bool SplitOpType(const nnvm::NodeAttrs& attrs,
                        std::vector<int>* in_attrs,
                        std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  int dtype = (*in_attrs)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  out_attrs->clear();
  int num_outputs = (param.sections > 0) ? param.sections : param.indices.ndim();
  for (int i = 0; i < num_outputs; ++i) {
    out_attrs->push_back(dtype);
  }
  return true;
}

inline bool SplitOpShapeImpl(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs,
                             const int real_axis) {
  using namespace mshadow;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  mxnet::TShape dshape = in_attrs->at(split_enum::kData);
  mxnet::TShape ishape = in_attrs->at(split_enum::kData);
  const mxnet::TShape indices =
    (param.sections > 0) ? GetSplitIndices(ishape, real_axis, param.sections) : param.indices;
  int num_outputs = (param.sections > 0) ? indices.ndim() - 1 : indices.ndim();
  // Pre-compute squeezed output shape for future usage
  mxnet::TShape squeezed_dshape = dshape;
  for (int d = real_axis; d < squeezed_dshape.ndim() - 1; ++d) {
    squeezed_dshape[d] = squeezed_dshape[d+1];
  }
  squeezed_dshape = mxnet::TShape(&squeezed_dshape[0], &squeezed_dshape[squeezed_dshape.ndim()-1]);
  // Assign shape to every output
  for (int i = 0; i < num_outputs; ++i) {
    index_t start = indices[i];
    index_t end = (i < num_outputs - 1) ? indices[i + 1] : ishape[real_axis];
    if (ishape[real_axis] == 0U) {
      end = start;
    } else {
      CHECK(start <= end)
        << "start " << start << " is not less than end " << end << "for subarray " << i;
      CHECK(end <= ishape[real_axis])
        << "end " << end << " is no less than the size of the axis " << ishape[real_axis];
    }
    dshape[real_axis] = (end - start);
    if (param.squeeze_axis) {
      CHECK_EQ(end - start, 1U) << "expected axis size of 1 but got " << end - start;
      SHAPE_ASSIGN_CHECK(*out_attrs, i, squeezed_dshape);
    } else {
      SHAPE_ASSIGN_CHECK(*out_attrs, i, dshape);
    }
  }
  mxnet::TShape back_calculate_dshape = ishape;
  back_calculate_dshape[real_axis] = 0;
  for (int d = 0; d < real_axis; ++d) {
    back_calculate_dshape[d] = (*out_attrs)[0][d];
  }
  if (param.squeeze_axis) {
    back_calculate_dshape[real_axis] = num_outputs;
  } else {
    for (int i = 0; i < num_outputs; ++i) {
      back_calculate_dshape[real_axis] += (*out_attrs)[i][real_axis];
    }
  }
  for (int d = real_axis + 1; d < ishape.ndim(); ++d) {
    if (param.squeeze_axis) {
      back_calculate_dshape[d] = (*out_attrs)[0][d - 1];
    } else {
      back_calculate_dshape[d] = (*out_attrs)[0][d];
    }
  }
  SHAPE_ASSIGN_CHECK(*in_attrs, split_enum::kData, back_calculate_dshape);
  return true;
}

inline bool SplitOpShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector* in_attrs,
                         mxnet::ShapeVector* out_attrs) {
  using namespace mshadow;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  mxnet::TShape dshape = in_attrs->at(split_enum::kData);
  if (!mxnet::ndim_is_known(dshape)) return false;
  if (param.axis >= 0) {
    CHECK_LT(param.axis, dshape.ndim());
  } else {
    CHECK_LT(param.axis + dshape.ndim(), dshape.ndim());
  }
  int real_axis = param.axis;
  if (real_axis < 0) {
    real_axis += dshape.ndim();
  }
  return SplitOpShapeImpl(attrs, in_attrs, out_attrs, real_axis);
}

struct SplitKernel {
  /*!
   * \brief Map function for forward split_v2 operator
   * \param i              global thread id
   * \param in_data        ptr to input buffer
   * \param out_data       ptr to ptr of outputs buffer
   * \param indices        ptr to indices buffer
   * \param num_sections   # of sections after split
   * \param axis_size      size of axis to be splitted on
   * \param trailing_size  step size within the data buffer of the axis to be splitted on
   */
  template<typename DType>
  static MSHADOW_XINLINE void Map(size_t i,
                                  const DType *in_data, DType** out_data, const size_t* indices,
                                  const size_t num_sections, const size_t axis_size,
                                  const size_t trailing_size) {
    size_t idx = i / trailing_size % axis_size;
    size_t target = 0;
    for (size_t section = 0;
         section < num_sections && indices[section] <= idx;
         target = section++) {}
    DType* target_data = out_data[target];
    const size_t mid_idx = idx - indices[target];
    const size_t head_idx = i / (trailing_size * axis_size);
    const size_t tail_idx = i % trailing_size;
    const size_t section_size = indices[target + 1] - indices[target];
    const size_t target_idx =
      head_idx * trailing_size * section_size + mid_idx * trailing_size + tail_idx;
    target_data[target_idx] = in_data[i];
  }
};

struct ConcatenateKernel {
  /*!
   * \brief Map function for backward split_v2 operator
   * \param i              global thread id
   * \param out_grad       ptr to ptr of out grads buffer
   * \param in_grad        ptr to input grad buffer
   * \param indices        ptr to indices buffer
   * \param num_sections   # of sections after split
   * \param axis_size      size of axis to be splitted on
   * \param trailing_size  step size within the data buffer of the axis to be splitted on
   */
  template<typename DType>
  static MSHADOW_XINLINE void Map(size_t i,
                                  DType** out_grad, DType* in_grad, const size_t* indices,
                                  const size_t num_sections, const size_t axis_size,
                                  const size_t trailing_size) {
    size_t idx = i / trailing_size % axis_size;
    size_t src = 0;
    for (size_t section = 0;
         section < num_sections && indices[section] <= idx;
         src = section++) {}
    DType* src_grad = out_grad[src];
    const size_t mid_idx = idx - indices[src];
    const size_t head_idx = i / (trailing_size * axis_size);
    const size_t tail_idx = i % trailing_size;
    const size_t section_size = indices[src + 1] - indices[src];
    const size_t src_idx =
      head_idx * trailing_size * section_size + mid_idx * trailing_size + tail_idx;
    in_grad[i] = src_grad[src_idx];
  }
};

template<typename xpu>
inline void SplitOpForwardImpl(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx,
                               const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs,
                               const int real_axis) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& input_data = inputs[split_enum::kData];
  size_t leading = 1, trailing = 1;
  CHECK_LT(real_axis, input_data.ndim());
  size_t mid = input_data.shape_[real_axis];
  for (int i = 0; i < real_axis; ++i) {
    leading *= input_data.shape_[i];
  }
  for (int i = real_axis + 1; i < input_data.ndim(); ++i) {
    trailing *= input_data.shape_[i];
  }

  size_t workspace_size = 0;
  const mxnet::TShape& ishape = input_data.shape_;
  const mxnet::TShape split_pts =
    (param.sections > 0) ? GetSplitIndices(ishape, real_axis, param.sections) : param.indices;
  std::vector<size_t> indices;
  for (const auto& section : split_pts) {
    indices.push_back(section);
  }
  if (param.sections == 0) {
    indices.push_back(ishape[real_axis]);
  }
  workspace_size += indices.size() * sizeof(size_t);
  MSHADOW_TYPE_SWITCH(input_data.type_flag_, DType, {
    std::vector<DType*> output_data;
    for (const TBlob& data : outputs) {
      output_data.push_back(data.dptr<DType>());
    }
    workspace_size += output_data.size() * sizeof(DType*);
    Tensor<xpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
    Tensor<cpu, 1, size_t> indices_cpu_tensor(indices.data(), Shape1(indices.size()));
    Tensor<xpu, 1, size_t> indices_xpu_tensor(
      reinterpret_cast<size_t*>(workspace.dptr_), Shape1(indices.size()));
    Tensor<cpu, 1, DType*> ptrs_cpu_tensor(output_data.data(), Shape1(output_data.size()));
    Tensor<xpu, 1, DType*> ptrs_xpu_tensor(
      reinterpret_cast<DType**>(workspace.dptr_ + indices.size() * sizeof(size_t)),
      Shape1(output_data.size()));
    mshadow::Copy(indices_xpu_tensor, indices_cpu_tensor, s);
    mshadow::Copy(ptrs_xpu_tensor, ptrs_cpu_tensor, s);
    Kernel<SplitKernel, xpu>::Launch(
      s, input_data.Size(), input_data.dptr<DType>(), ptrs_xpu_tensor.dptr_,
      indices_xpu_tensor.dptr_, indices.size() - 1, mid, trailing);
  });
}

template<typename xpu>
inline void SplitOpForward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), (param.sections > 0) ? param.sections : param.indices.ndim());
  const TBlob& input_data = inputs[split_enum::kData];
  int real_axis = param.axis;
  if (real_axis < 0) {
    real_axis += input_data.ndim();
  }
  SplitOpForwardImpl<xpu>(attrs, ctx, inputs, req, outputs, real_axis);
}

template<typename xpu>
inline void SplitOpBackwardImpl(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs,
                                const int real_axis) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  TBlob input_grad = outputs[split_enum::kData];
  size_t leading = 1, trailing = 1;
  CHECK_LT(real_axis, input_grad.ndim());
  size_t mid = input_grad.shape_[real_axis];
  for (int i = 0; i < real_axis; ++i) {
    leading *= input_grad.shape_[i];
  }
  for (int i = real_axis + 1; i < input_grad.ndim(); ++i) {
    trailing *= input_grad.shape_[i];
  }

  size_t workspace_size = 0;
  const mxnet::TShape& ishape = input_grad.shape_;
  const mxnet::TShape split_pts =
    (param.sections > 0) ? GetSplitIndices(ishape, real_axis, param.sections) : param.indices;
  std::vector<size_t> indices;
  for (const auto& section : split_pts) {
    indices.push_back(section);
  }
  if (param.sections == 0) {
    indices.push_back(ishape[real_axis]);
  }
  workspace_size += indices.size() * sizeof(size_t);
  MSHADOW_TYPE_SWITCH(input_grad.type_flag_, DType, {
    std::vector<DType*> out_grads;
    for (const TBlob& output_grad : inputs) {
      out_grads.push_back(output_grad.dptr<DType>());
    }
    workspace_size += out_grads.size() * sizeof(DType*);
    Tensor<xpu, 1, char> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(workspace_size), s);
    Tensor<cpu, 1, size_t> indices_cpu_tensor(indices.data(), Shape1(indices.size()));
    Tensor<xpu, 1, size_t> indices_xpu_tensor(
      reinterpret_cast<size_t*>(workspace.dptr_), Shape1(indices.size()));
    Tensor<cpu, 1, DType*> ptrs_cpu_tensor(out_grads.data(), Shape1(inputs.size()));
    Tensor<xpu, 1, DType*> ptrs_xpu_tensor(
      reinterpret_cast<DType**>(workspace.dptr_ + indices.size() * sizeof(size_t)),
      Shape1(inputs.size()));
    mshadow::Copy(indices_xpu_tensor, indices_cpu_tensor, s);
    mshadow::Copy(ptrs_xpu_tensor, ptrs_cpu_tensor, s);
    Kernel<ConcatenateKernel, xpu>::Launch(
      s, input_grad.Size(), ptrs_xpu_tensor.dptr_, input_grad.dptr<DType>(),
      indices_xpu_tensor.dptr_, indices.size() - 1, mid, trailing);
  });
}

template<typename xpu>
inline void SplitOpBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), (param.sections > 0) ? param.sections : param.indices.ndim())
    << "out grad vector size mush match the output size";
  CHECK_EQ(outputs.size(), 1U);
  int real_axis = param.axis;
  if (real_axis < 0) {
      real_axis += outputs[split_enum::kData].ndim();
  }
  SplitOpBackwardImpl<xpu>(attrs, ctx, inputs, req, outputs, real_axis);
}

inline uint32_t SplitNumOutputs(const NodeAttrs& attrs) {
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  return (param.sections > 0) ? param.sections : param.indices.ndim();
}

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::TransposeParam> {
  size_t operator()(const mxnet::op::TransposeParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.axes);
    return ret;
  }
};

template<>
struct hash<mxnet::op::ReshapeParam> {
  size_t operator()(const mxnet::op::ReshapeParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.target_shape);
    ret = dmlc::HashCombine(ret, val.keep_highest);
    ret = dmlc::HashCombine(ret, val.shape);
    ret = dmlc::HashCombine(ret, val.reverse);
    return ret;
  }
};

template<>
struct hash<mxnet::op::ExpandDimParam> {
  size_t operator()(const mxnet::op::ExpandDimParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.axis);
    return ret;
  }
};

}  // namespace std

#endif  // MXNET_OPERATOR_TENSOR_MATRIX_OP_INL_H_
