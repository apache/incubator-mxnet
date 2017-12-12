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
 * Copyright (c) 2017 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu, Jun Wu, Da Zheng
*/

#include "./convolution-inl.h"
#include "../elemwise_op_common.h"
#include "./mkldnn/mkldnn_ops-inl.h"
#include "./mkldnn/mkldnn_base-inl.h"
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ConvolutionParam);

static inline index_t AddPad(index_t dsize, index_t pad) {
  return dsize + 2 * pad;
}

static inline std::vector<std::string> ListArguments(const ConvolutionParam& param_) {
  if (!param_.no_bias) {
    return {"data", "weight", "bias"};
  } else {
    return {"data", "weight"};
  }
}

static void ConvolutionCompute_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req, const std::vector<NDArray>& outputs) {
#if MXNET_USE_MKLDNN == 1
  if (SupportMKLDNNConv(inputs[0])) {
    MKLDNNConvolution_Forward(attrs, ctx, inputs, req, outputs);
    return;
  }
#endif
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++)
    in_blobs[i] = inputs[i].data();
  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  ConvolutionCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
}

static void ConvolutionGradCompute_CPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx, const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req, const std::vector<NDArray>& outputs) {
#if MXNET_USE_MKLDNN == 1
  if (SupportMKLDNNConv(inputs[0])) {
    MKLDNNConvolution_Backward(attrs, ctx, inputs, req, outputs);
    return;
  }
#endif
  std::vector<TBlob> in_blobs(inputs.size());
  for (size_t i = 0; i < in_blobs.size(); i++)
    in_blobs[i] = inputs[i].data();
  std::vector<TBlob> out_blobs(outputs.size());
  for (size_t i = 0; i < out_blobs.size(); i++)
    out_blobs[i] = outputs[i].data();
  ConvolutionGradCompute<cpu>(attrs, ctx, in_blobs, req, out_blobs);
}

static bool ConvolutionShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  using namespace mshadow;
  const ConvolutionParam& param_ = nnvm::get<ConvolutionParam>(attrs.parsed);
  if (!param_.no_bias) {
    CHECK_EQ(in_shape->size(), 3U) << "Input:[data, weight, bias]";
  } else {
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, weight]";
  }
  // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
  out_shape->resize(1, TShape());
  const TShape &dshp = (*in_shape)[conv::kData];
  if (dshp.ndim() ==  0) return false;

  if (param_.kernel.ndim() == 1) {
    // 1d conv
    CHECK_EQ(dshp.ndim(), 3U) << "Input data should be 3D in batch-num_filter-x";
    Shape<3> dshape = ConvertLayout(dshp.get<3>(), param_.layout.value(), kNCW);
    Shape<3> wshape = Shape3(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
        param_.kernel[0]);
    wshape = ConvertLayout(wshape, kNCW, param_.layout.value());
    wshape[0] *= param_.num_group;
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }

    const index_t dilated_ksize_x = param_.DilatedKernelSize(0);
    CHECK_EQ(dshape[1] % param_.num_group, 0U) \
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    Shape<3> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = dshape[2] ?
      (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_x) / param_.stride[0] + 1 : 0;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<3>(), param_.layout.value(), kNCW);
    dshape[0] = oshape[0];
    if (oshape[2] && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_x - 1 - 2 * param_.pad[0];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
        ConvertLayout(dshape, kNCW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != 0) {
      CHECK_LE(dilated_ksize_x, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
    }
    return true;
  } else if (param_.kernel.ndim() == 2) {
    // 2d conv
    CHECK_EQ(dshp.ndim(), 4U) \
      << "Input data should be 4D in batch-num_filter-y-x";
    Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
    Shape<4> wshape = Shape4(param_.num_filter / param_.num_group,
        dshape[1] / param_.num_group,
        param_.kernel[0], param_.kernel[1]);
    wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
    wshape[0] *= param_.num_group;
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }

    const index_t dilated_ksize_y = param_.DilatedKernelSize(0);
    const index_t dilated_ksize_x = param_.DilatedKernelSize(1);
    CHECK_EQ(dshape[1] % param_.num_group, 0U) \
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    Shape<4> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = dshape[2] ?
      (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_y) / param_.stride[0] + 1 : 0;
    oshape[3] = dshape[3] ?
      (AddPad(dshape[3], param_.pad[1]) - dilated_ksize_x) / param_.stride[1] + 1 : 0;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
    dshape[0] = oshape[0];
    if (oshape[2] && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param_.pad[0];
    }
    if (oshape[3] && param_.stride[1] == 1) {
      dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param_.pad[1];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
        ConvertLayout(dshape, kNCHW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != 0) {
      CHECK_LE(dilated_ksize_y, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
    }
    if (dshape[3] != 0) {
      CHECK_LE(dilated_ksize_x, AddPad(dshape[3], param_.pad[1])) << "kernel size exceed input";
    }
    return true;
  } else if (param_.kernel.ndim() == 3) {
    // 3d conv
    CHECK_EQ(dshp.ndim(), 5U) \
      << "Input data should be 5D in batch-num_filter-depth-y-x";
    Shape<5> dshape = ConvertLayout(dshp.get<5>(), param_.layout.value(), kNCDHW);
    Shape<5> wshape = Shape5(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
        param_.kernel[0], param_.kernel[1], param_.kernel[2]);
    wshape = ConvertLayout(wshape, kNCDHW, param_.layout.value());
    wshape[0] *= param_.num_group;
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
    if (!param_.no_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
    }

    // Note: 3D dilation currently not supported.
    // Calculations below done to preserve symmetry with 1D/2D code.
    const index_t dilated_ksize_d = param_.DilatedKernelSize(0);
    const index_t dilated_ksize_y = param_.DilatedKernelSize(1);
    const index_t dilated_ksize_x = param_.DilatedKernelSize(2);
    CHECK_EQ(dshape[1] % param_.num_group, 0U)
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0U)
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    CHECK_EQ(param_.dilate.Size(), 1U)
      << "Dilate is not supported in 3d convolution";
    Shape<5> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = dshape[2] ?
      (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_d) / param_.stride[0] + 1 : 0;
    oshape[3] = dshape[3] ?
      (AddPad(dshape[3], param_.pad[1]) - dilated_ksize_y) / param_.stride[1] + 1 : 0;
    oshape[4] = dshape[4] ?
      (AddPad(dshape[4], param_.pad[2]) - dilated_ksize_x) / param_.stride[2] + 1 : 0;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCDHW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input depth/height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<5>(), param_.layout.value(), kNCDHW);
    dshape[0] = oshape[0];
    if (oshape[2] && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_d - 1 - 2 * param_.pad[0];
    }
    if (oshape[3] && param_.stride[1] == 1) {
      dshape[3] = oshape[3] + dilated_ksize_y - 1 - 2 * param_.pad[1];
    }
    if (oshape[4] && param_.stride[2] == 1) {
      dshape[4] = oshape[4] + dilated_ksize_x - 1 - 2 * param_.pad[2];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
        ConvertLayout(dshape, kNCDHW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != 0) {
      CHECK_LE(dilated_ksize_d, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
    }
    if (dshape[3] != 0) {
      CHECK_LE(dilated_ksize_y, AddPad(dshape[3], param_.pad[1])) << "kernel size exceed input";
    }
    if (dshape[4] != 0) {
      CHECK_LE(dilated_ksize_x, AddPad(dshape[4], param_.pad[2])) << "kernel size exceed input";
    }
    return true;
  } else {
    LOG(FATAL) << "Unknown convolution type";
    return false;
  }
}

static bool ConvolutionType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type, std::vector<int> *out_type) {
  const ConvolutionParam& param_ = nnvm::get<ConvolutionParam>(attrs.parsed);
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  for (index_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param_)[i]);
    }
  }
  out_type->clear();
  out_type->push_back(dtype);
  return true;
}

inline static bool ConvStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int> *in_attrs,
                                   std::vector<int> *out_attrs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(in_attrs->size(), in_expected);
  CHECK_EQ(out_attrs->size(), 1);

#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask
      // We should allow MKLDNN conv to apply to the default storage as well.
      // Even with format conversion, MKLDNN conv should still be faster than
      // the native implementation.
      && (in_attrs->at(0) == kMKLDNNStorage
          || in_attrs->at(0) == kDefaultStorage)) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    (*out_attrs)[0] = kMKLDNNStorage;
    return true;
  }
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  (*out_attrs)[0] = kDefaultStorage;
  return true;
}

inline static bool backward_ConvStorageType(const nnvm::NodeAttrs& attrs,
                                          const int dev_mask,
                                          DispatchMode* dispatch_mode,
                                          std::vector<int> *in_attrs,
                                          std::vector<int> *out_attrs) {
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 3 : 4;
  uint32_t out_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(in_attrs->size(), in_expected);
  CHECK_EQ(out_attrs->size(), out_expected);

#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask
      // We should allow MKLDNN conv to apply to the default storage as well.
      // Even with format conversion, MKLDNN conv should still be faster than
      // the native implementation.
      && (in_attrs->at(0) == kMKLDNNStorage
          || in_attrs->at(0) == kDefaultStorage)) {
    *dispatch_mode = DispatchMode::kFComputeEx;
    (*out_attrs)[conv::kData] = kMKLDNNStorage;
    // We don't want the parameter gradients are stored in MKLDNN storage.
    // These will be sent to the KVstore to update the global parameters.
    // We should convert storage inside an operator so that we can take
    // advantage of TempSpace.
    (*out_attrs)[conv::kWeight] = kDefaultStorage;
    if (!param.no_bias)
      (*out_attrs)[conv::kBias] = kDefaultStorage;
    for (size_t i = 0; i < out_attrs->size(); i++)
    return true;
  }
#endif
  *dispatch_mode = DispatchMode::kFCompute;
  for (size_t i = 0; i < out_attrs->size(); i++)
    (*out_attrs)[i] = kDefaultStorage;
  return true;
}

static void ConvolutionParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  ConvolutionParam param_;
  try {
    param_.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }

  if (param_.kernel.ndim() == 1) {
    param_.layout = param_.layout? param_.layout.value() : mshadow::kNCW;
    if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
    if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
  } else if (param_.kernel.ndim() == 2) {
    param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
    if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
  } else {
    CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim() << "D convolution not supported";
    param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
    if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
  }
  attrs->parsed = std::move(param_);
}

struct ConvolutionGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    const ConvolutionParam& param = nnvm::get<ConvolutionParam>(n->attrs.parsed);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[conv::kData]);
    heads.push_back(n->inputs[conv::kWeight]);
    if (!param.no_bias)
      heads.push_back(n->inputs[conv::kBias]);
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

NNVM_REGISTER_OP(Convolution)
.describe(R"code(Compute *N*-D convolution on *(N+2)*-D input.

In the 2-D convolution, given input data with shape *(batch_size,
channel, height, width)*, the output is computed by

.. math::

   out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
   weight[i,j,:,:]

where :math:`\star` is the 2-D cross-correlation operator.

For general 2-D convolution, the shapes are

- **data**: *(batch_size, channel, height, width)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_height, out_width)*.

Define::

  f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

then we have::

  out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
  out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
width)*. We can choose other layouts such as *NHWC*.

If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
evenly into *g* parts along the channel axis, and also evenly split ``weight``
along the first dimension. Next compute the convolution on the *i*-th part of
the data with the *i*-th weight part. The output is obtained by concatenating all
the *g* results.

1-D convolution does not have *height* dimension but only *width* in space.

- **data**: *(batch_size, channel, width)*
- **weight**: *(num_filter, channel, kernel[0])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_width)*.

3-D convolution adds an additional *depth* dimension besides *height* and
*width*. The shapes are

- **data**: *(batch_size, channel, depth, height, width)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.

Both ``weight`` and ``bias`` are learnable parameters.

There are other options to tune the performance.

- **cudnn_tune**: enable this option leads to higher startup time but may give
  faster speed. Options are

  - **off**: no tuning
  - **limited_workspace**:run test and pick the fastest algorithm that doesn't
    exceed workspace limit.
  - **fastest**: pick the fastest algorithm and ignore workspace limit.
  - **None** (default): the behavior is determined by environment variable
    ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
    (default), 2 for fastest.

- **workspace**: A large number leads to more (GPU) memory usage but may improve
  the performance.

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ConvolutionParam& params = nnvm::get<ConvolutionParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_num_outputs(1)
.set_attr_parser(ConvolutionParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const ConvolutionParam& params = nnvm::get<ConvolutionParam>(attrs.parsed);
  if (params.no_bias)
    return std::vector<std::string>{"data", "weight"};
  else
    return std::vector<std::string>{"data", "weight", "bias"};
})
.set_attr<nnvm::FInferShape>("FInferShape", ConvolutionShape)
.set_attr<nnvm::FInferType>("FInferType", ConvolutionType)
.set_attr<FInferStorageType>("FInferStorageType", ConvStorageType)
.set_attr<FCompute>("FCompute<cpu>", ConvolutionCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ConvolutionCompute_CPU)
.set_attr<nnvm::FGradient>("FGradient", ConvolutionGrad{"_backward_Convolution"})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(ConvolutionParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Convolution)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ConvolutionParam& params = nnvm::get<ConvolutionParam>(attrs.parsed);
  return params.no_bias ? 2 : 3;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", backward_ConvStorageType)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(ConvolutionParamParser)
.set_attr<FCompute>("FCompute<cpu>", ConvolutionGradCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ConvolutionGradCompute_CPU);

}  // namespace op
}  // namespace mxnet
