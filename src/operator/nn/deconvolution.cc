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
 * Copyright (c) 2015 by Contributors
 * \file deconvolution.cc
 * \brief
 * \author Wei Wu, Da Zheng
*/

#include "./deconvolution-inl.h"
#include "../operator_common.h"
#include "../../common/utils.h"
#if MXNET_USE_MKLDNN == 1
#include "./mkldnn/mkldnn_base-inl.h"
#include "./mkldnn/mkldnn_ops-inl.h"
#endif  // MXNET_USE_MKLDNN

namespace mxnet {
namespace op {

#if MXNET_USE_MKLDNN == 1
static void DeconvolutionComputeExCPU(const nnvm::NodeAttrs& attrs,
                                      const OpContext& ctx,
                                      const std::vector<NDArray>& inputs,
                                      const std::vector<OpReqType>& req,
                                      const std::vector<NDArray>& outputs) {
  const DeconvolutionParam& params = nnvm::get<DeconvolutionParam>(attrs.parsed);
  if (SupportMKLDNNDeconv(params, inputs[0])) {
    if (params.kernel.ndim() == 3) {
      // we cannot check the output, as 3D deconvolution is not natively supported yet
      MKLDNNRun(MKLDNNDeconvolutionForward, attrs, ctx, inputs, req, outputs);
    } else {
      MKLDNN_OPCHECK_INIT(false, outputs.size(), inputs, outputs);
      MKLDNNRun(MKLDNNDeconvolutionForward, attrs, ctx, inputs, req, outputs);
      MKLDNN_OPCHECK_RUN(DeconvolutionCompute<cpu>, attrs, ctx, inputs, req, outputs);
    }
    return;
  }
  FallBackCompute(DeconvolutionCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

static void DeconvolutionGradComputeExCPU(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
  const DeconvolutionParam& params = nnvm::get<DeconvolutionParam>(attrs.parsed);
  if (SupportMKLDNNDeconv(params, inputs[0])) {
    if (params.kernel.ndim() == 3) {
      // we cannot check the output, as 3D deconvolution is not natively supported yet
      MKLDNNRun(MKLDNNDeconvolutionBackward, attrs, ctx, inputs, req, outputs);
    } else {
      MKLDNN_OPCHECK_INIT(true, outputs.size(), inputs, outputs);
      MKLDNNRun(MKLDNNDeconvolutionBackward, attrs, ctx, inputs, req, outputs);
      MKLDNN_OPCHECK_RUN(DeconvolutionGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
    }
    return;
  }
  FallBackCompute(DeconvolutionGradCompute<cpu>, attrs, ctx, inputs, req, outputs);
}

inline static bool DeconvStorageType(const nnvm::NodeAttrs& attrs,
                                     const int dev_mask,
                                     DispatchMode* dispatch_mode,
                                     std::vector<int>* in_attrs,
                                     std::vector<int>* out_attrs) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(in_attrs->size(), in_expected);
  CHECK_EQ(out_attrs->size(), 1);

  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs,
                           out_attrs);
}

inline static bool BackwardDeconvStorageType(const nnvm::NodeAttrs& attrs,
                                             const int dev_mask,
                                             DispatchMode* dispatch_mode,
                                             std::vector<int> *in_attrs,
                                             std::vector<int> *out_attrs) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 3 : 4;
  uint32_t out_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(in_attrs->size(), in_expected);
  CHECK_EQ(out_attrs->size(), out_expected);

  return MKLDNNStorageType(attrs, dev_mask, true, dispatch_mode, in_attrs,
                           out_attrs);
}
#endif

static bool DeconvolutionShape(const nnvm::NodeAttrs& attrs,
                               mxnet::ShapeVector *in_shape,
                               mxnet::ShapeVector *out_shape) {
  const DeconvolutionParam& param_ = nnvm::get<DeconvolutionParam>(attrs.parsed);
#if MXNET_USE_CUDNN == 0 && MXNET_USE_MKLDNN == 0
  if (param_.kernel.ndim() > 2) {
    LOG(FATAL) << "If not using CUDNN or MKLDNN, only 1D or 2D Deconvolution is supported";
    return false;
  }
#endif

  using namespace mshadow;
  if (!param_.no_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  out_shape->resize(1, mxnet::TShape());
  const mxnet::TShape &dshape = (*in_shape)[deconv::kData];
  if (!mxnet::ndim_is_known(dshape)) return false;

  if (param_.kernel.ndim() == 1) {
    // 1d conv
    CHECK_EQ(dshape.ndim(), 3U) << "Input data should be 3D in batch-num_filter-x";
    Shape<3> dshape_ncw = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
    Shape<3> wshape = Shape3(dshape_ncw[1], param_.num_filter / param_.num_group,
        param_.kernel[0]);
    wshape = ConvertLayout(wshape, kNCW, param_.layout.value());
    SHAPE_ASSIGN_CHECK(*in_shape, deconv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, deconv::kBias, Shape1(param_.num_filter));
    }

    const index_t dilated_ksize_x = param_.DilatedKernelSize(0);

    index_t o_pad[1];
    index_t o_adj[1];
    param_.InferPad(dshape_ncw, o_pad, o_adj);

    CHECK_EQ(dshape_ncw[1] % param_.num_group, 0U) \
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;

    CHECK_GE(param_.stride[0]-1, o_adj[0]) << "adj(x) must be samller than stride[0]";

    Shape<3> oshape;
    oshape[0] = dshape_ncw[0];
    oshape[1] = param_.num_filter;
    if (mxnet::dim_size_is_known(dshape_ncw[2])) {
      oshape[2] = param_.stride[0] * (dshape_ncw[2] - 1) +
          dilated_ksize_x - 2 * o_pad[0] + o_adj[0];
    } else {
      oshape[2] = -1;
    }

    if (param_.target_shape.ndim() > 0) {
      if (param_.target_shape[0] > 0) {
        CHECK_EQ(param_.target_shape[0], oshape[2]) \
          << "param_.target_shape[0] was not reasonable, please set it carefully";
      }
    }

    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCW, param_.layout.value()));

    return true;
  } else if (param_.kernel.ndim() == 2) {
    // 2d conv
    CHECK_EQ(dshape.ndim(), 4U) \
      << "Input data should be 4D in batch-num_filter-y-x";
    Shape<4> dshape_nchw = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
    Shape<4> wshape = Shape4(dshape_nchw[1],
        param_.num_filter / param_.num_group,
        param_.kernel[0], param_.kernel[1]);
    wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
    SHAPE_ASSIGN_CHECK(*in_shape, deconv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, deconv::kBias, Shape1(param_.num_filter));
    }

    const index_t dilated_ksize_y = param_.DilatedKernelSize(0);
    const index_t dilated_ksize_x = param_.DilatedKernelSize(1);

    index_t o_pad[2];
    index_t o_adj[2];
    param_.InferPad(dshape_nchw, o_pad, o_adj);

    CHECK_EQ(dshape_nchw[1] % param_.num_group, 0U) \
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;

    CHECK_GE(param_.stride[0]-1, o_adj[0]) << "adj(y) must be samller than stride[0]";
    CHECK_GE(param_.stride[1]-1, o_adj[1]) << "adj(x) must be samller than stride[1]";

    Shape<4> oshape;
    oshape[0] = dshape_nchw[0];
    oshape[1] = param_.num_filter;
    if (mxnet::dim_size_is_known(dshape_nchw[2])) {
      oshape[2] = param_.stride[0] * (dshape_nchw[2] - 1) +
          dilated_ksize_y - 2 * o_pad[0] + o_adj[0];
    } else {
      oshape[2] = -1;
    }
    if (mxnet::dim_size_is_known(dshape_nchw[3])) {
      oshape[3] = param_.stride[1] * (dshape_nchw[3] - 1) +
          dilated_ksize_x - 2 * o_pad[1] + o_adj[1];
    } else {
      oshape[3] = -1;
    }

    if (param_.target_shape.ndim() > 1) {
      if (param_.target_shape[0] > 0) {
        CHECK_EQ(param_.target_shape[0], oshape[2]) \
          << "param_.target_shape[0] was not reasonable, please set it carefully";
      }
      if (param_.target_shape[1] > 0) {
        CHECK_EQ(param_.target_shape[1], oshape[3]) \
          << "param_.target_shape[1] was not reasonable, please set it carefully";
      }
    }

    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));

    return true;
  } else if (param_.kernel.ndim() == 3) {
    // 3d conv
    CHECK_EQ(dshape.ndim(), 5U) \
      << "Input data should be 5D in batch-num_filter-depth-y-x";
    Shape<5> dshape_ncdhw = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW);
    Shape<5> wshape = Shape5(dshape_ncdhw[1], param_.num_filter / param_.num_group,
        param_.kernel[0], param_.kernel[1], param_.kernel[2]);
    wshape = ConvertLayout(wshape, kNCDHW, param_.layout.value());
    SHAPE_ASSIGN_CHECK(*in_shape, deconv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, deconv::kBias, Shape1(param_.num_filter));
    }

    // Note: 3D dilation currently not supported.
    // Calculations below done to preserve symmetry with 1D/2D code.
    const index_t dilated_ksize_d = param_.DilatedKernelSize(0);
    const index_t dilated_ksize_y = param_.DilatedKernelSize(1);
    const index_t dilated_ksize_x = param_.DilatedKernelSize(2);

    index_t o_pad[3];
    index_t o_adj[3];
    param_.InferPad(dshape_ncdhw, o_pad, o_adj);

    CHECK_EQ(dshape_ncdhw[1] % param_.num_group, 0U) \
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    CHECK_EQ(param_.dilate.Size(), 1U)
      << "Dilate is not supported in 3d deconvolution";

    CHECK_GE(param_.stride[0]-1, o_adj[0]) << "adj(d) must be samller than stride[0]";
    CHECK_GE(param_.stride[1]-1, o_adj[1]) << "adj(y) must be samller than stride[1]";
    CHECK_GE(param_.stride[2]-1, o_adj[2]) << "adj(x) must be samller than stride[2]";

    Shape<5> oshape;
    oshape[0] = dshape_ncdhw[0];
    oshape[1] = param_.num_filter;
    if (mxnet::dim_size_is_known(dshape_ncdhw[2])) {
      oshape[2] = param_.stride[0] * (dshape_ncdhw[2] - 1) +
          dilated_ksize_d - 2 * o_pad[0] + o_adj[0];
    } else {
      oshape[2] = -1;
    }
    if (mxnet::dim_size_is_known(dshape_ncdhw[3])) {
      oshape[3] = param_.stride[1] * (dshape_ncdhw[3] - 1) +
          dilated_ksize_y - 2 * o_pad[1] + o_adj[1];
    } else {
      oshape[3] = -1;
    }
    if (mxnet::dim_size_is_known(dshape_ncdhw[4])) {
      oshape[4] = param_.stride[2] * (dshape_ncdhw[4] - 1) +
          dilated_ksize_x - 2 * o_pad[2] + o_adj[2];
    } else {
      oshape[4] = -1;
    }

    if (param_.target_shape.ndim() > 2) {
      if (param_.target_shape[0] > 0) {
        CHECK_EQ(param_.target_shape[0], oshape[2]) \
          << "param_.target_shape[0] was not reasonable, please set it carefully";
      }
      if (param_.target_shape[1] > 0) {
        CHECK_EQ(param_.target_shape[1], oshape[3]) \
          << "param_.target_shape[1] was not reasonable, please set it carefully";
      }
      if (param_.target_shape[2] > 0) {
        CHECK_EQ(param_.target_shape[2], oshape[4]) \
          << "param_.target_shape[2] was not reasonable, please set it carefully";
      }
    }

    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCDHW, param_.layout.value()));

    return true;
  } else {
    LOG(FATAL) << "Unknown convolution type";
    return false;
  }
}

static inline std::vector<std::string> ListArguments(const DeconvolutionParam& param_) {
  if (!param_.no_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

static bool DeconvolutionType(const nnvm::NodeAttrs& attrs,
                              std::vector<int> *in_type, std::vector<int> *out_type) {
  const DeconvolutionParam& param_ = nnvm::get<DeconvolutionParam>(attrs.parsed);
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  if (type_is_none(dtype)) {
    // Input type is undefined, we try backward inference
    if (out_type->size() == 0 || type_is_none((*out_type)[0])) {
       // Neither the input nor the output are defined,
       // types cannot be infered for this op
      return false;
    } else {
       // Input type is undefined but output type is: backward inference
      dtype = (*out_type)[0];
    }
  } else {
    // Input type is defined but output type is not: forward inference
    out_type->clear();
    out_type->push_back(dtype);
  }
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param_)[i]);
    }
  }
  return true;
}

static void DeconvolutionParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  DeconvolutionParam param_;
  param_.Init(attrs->dict);
  if (param_.kernel.ndim() == 1) {
    param_.layout = param_.layout? param_.layout.value() : mshadow::kNCW;
    if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
    if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
    if (param_.adj.ndim() == 0) param_.adj = Shape1(0);
  } else if (param_.kernel.ndim() == 2) {
    param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
    if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    if (param_.adj.ndim() == 0) param_.adj = Shape2(0, 0);
  } else {
    CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D deconvolution not supported";
    param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
    if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
    if (param_.adj.ndim() == 0) param_.adj = Shape3(0, 0, 0);
  }
  CHECK_EQ(param_.kernel.ndim(), param_.stride.ndim())
    << "Stride must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while stride is "
    << param_.stride;
  CHECK_EQ(param_.kernel.ndim(), param_.dilate.ndim())
    << "Dilate must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while dilate is "
    << param_.dilate;
  CHECK_EQ(param_.kernel.ndim(), param_.pad.ndim())
    << "Padding must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while padding is "
    << param_.pad;
  CHECK_EQ(param_.kernel.ndim(), param_.adj.ndim())
    << "Adjustment must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while adjustment is "
    << param_.adj;
  attrs->parsed = std::move(param_);
}

struct DeconvolutionGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::ObjectPtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[deconv::kData]);
    heads.push_back(n->inputs[deconv::kWeight]);
    const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(n->attrs.parsed);
    if (!param.no_bias)
      heads.push_back(n->inputs[deconv::kBias]);
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

DMLC_REGISTER_PARAMETER(DeconvolutionParam);

NNVM_REGISTER_OP(Deconvolution)
.add_alias("_npx_deconvolution")
.describe("Computes 1D or 2D transposed convolution (aka fractionally strided convolution) of the "
    "input tensor. This operation can be seen as the gradient of Convolution operation with "
    "respect to its input. Convolution usually reduces the size of the input. Transposed "
    "convolution works the other way, going from a smaller input to a larger output while "
    "preserving the connectivity pattern.")
.set_num_inputs([](const NodeAttrs& attrs) {
  const DeconvolutionParam& params = nnvm::get<DeconvolutionParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_num_outputs(1)
.set_attr_parser(DeconvolutionParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return ListArguments(nnvm::get<DeconvolutionParam>(attrs.parsed));
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output"};
})
.set_attr<mxnet::FInferShape>("FInferShape", DeconvolutionShape)
.set_attr<nnvm::FInferType>("FInferType", DeconvolutionType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.set_attr<FCompute>("FCompute<cpu>", DeconvolutionCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", DeconvolutionGrad{"_backward_Deconvolution"})
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FInferStorageType>("FInferStorageType", DeconvStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", DeconvolutionComputeExCPU)
#endif
.add_argument("data", "NDArray-or-Symbol", "Input tensor to the deconvolution operation.")
.add_argument("weight", "NDArray-or-Symbol", "Weights representing the kernel.")
.add_argument("bias", "NDArray-or-Symbol", "Bias added to the result after the deconvolution "
    "operation.")
.add_arguments(DeconvolutionParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Deconvolution)
.set_num_inputs([](const NodeAttrs& attrs) {
  const DeconvolutionParam& params = nnvm::get<DeconvolutionParam>(attrs.parsed);
  return params.no_bias ? 3 : 4;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const DeconvolutionParam& params = nnvm::get<DeconvolutionParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(DeconvolutionParamParser)
#if MXNET_USE_MKLDNN == 1
.set_attr<bool>("TIsMKLDNN", true)
.set_attr<FInferStorageType>("FInferStorageType", BackwardDeconvStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", DeconvolutionGradComputeExCPU)
#endif
.set_attr<FCompute>("FCompute<cpu>", DeconvolutionGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
