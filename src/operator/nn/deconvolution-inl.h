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
 * \file deconvolution-inl.h
 * \brief
 * \author Wei Wu, Da Zheng
*/
#ifndef MXNET_OPERATOR_NN_DECONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_DECONVOLUTION_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <mshadow/tensor.h>
#include "../operator_common.h"
#include "../linalg.h"
#include "./im2col.h"


namespace mxnet {
namespace op {

namespace deconv {
  enum DeconvolutionOpInputs {kData, kWeight, kBias};
  enum DeconvolutionOpOutputs {kOut};
  enum DeconvolutionOpResource {kTempSpace};
  enum DeconvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

struct DeconvolutionParam : public dmlc::Parameter<DeconvolutionParam> {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  mxnet::TShape adj;
  mxnet::TShape target_shape;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(DeconvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Deconvolution kernel size: (w,), (h, w) or (d, h, w). "
                  "This is same as the kernel size used for the corresponding convolution");
    DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, 0))
        .describe("The stride used for the corresponding convolution: (w,), (h, w) or (d, h, w). "
                  "Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, 0))
        .describe("Dilation factor for each dimension of the input: (w,), (h, w) or (d, h, w). "
                  "Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, 0))
        .describe("The amount of implicit zero padding added during convolution for each "
                  "dimension of the input: "
                  "(w,), (h, w) or (d, h, w). "
                  "``(kernel-1)/2`` is usually a good choice. "
                  "If `target_shape` is set, "
                  "`pad` will be ignored and a padding that will generate the target shape "
                  "will be used. Defaults to no padding.");
    DMLC_DECLARE_FIELD(adj).set_default(mxnet::TShape(0, 0))
        .describe("Adjustment for output shape: (w,), (h, w) or (d, h, w). "
                  "If `target_shape` is set, "
                  "`adj` will be ignored and computed accordingly.");
    DMLC_DECLARE_FIELD(target_shape).set_default(mxnet::TShape(0, 0))
        .describe("Shape of the output tensor: (w,), (h, w) or (d, h, w).");
    DMLC_DECLARE_FIELD(num_filter).set_lower_bound(1)
        .describe("Number of output filters.");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
        .describe("Number of groups partition.");
    DMLC_DECLARE_FIELD(workspace).set_default(512).set_lower_bound(0)
        .describe("Maximum temporary workspace allowed (MB) in deconvolution."
                  "This parameter has two usages. When CUDNN is not used, it determines the "
                  "effective batch size of the deconvolution kernel. When CUDNN is used, "
                  "it controls the maximum temporary storage used for tuning "
                  "the best CUDNN kernel when `limited_workspace` strategy is used.");
    DMLC_DECLARE_FIELD(no_bias).set_default(true)
        .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
      .add_enum("off", deconv::kOff)
      .add_enum("limited_workspace", deconv::kLimited)
      .add_enum("fastest", deconv::kFastest)
      .set_default(dmlc::optional<int>())
      .describe("Whether to pick convolution algorithm by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
      .add_enum("NCW", mshadow::kNCW)
      .add_enum("NCHW", mshadow::kNCHW)
      .add_enum("NCDHW", mshadow::kNCDHW)
      .add_enum("NHWC", mshadow::kNHWC)
      .add_enum("NDHWC", mshadow::kNDHWC)
      .set_default(dmlc::optional<int>())
      .describe("Set layout for input, output and weight. Empty for "
                "default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d."
                "NHWC and NDHWC are only supported on GPU.");
  }

  template<size_t ndim>
  void InferPad(const TShape &input, index_t (&o_pad)[ndim], index_t (&o_adj)[ndim]) const {
    // Modified by Li.bs
    // Use tag to control the calculation of pad
    bool bCal = false;
    if (target_shape.ndim() != 0) {
      for (index_t i = 0; i < target_shape.ndim(); i++) {
        if (target_shape[i] != 0) bCal = true;
      }
    }

    if (bCal) {
      size_t input_ndim = input.ndim();

      for (size_t i = 0; i < ndim; i++) {
        // input.ndim() can be larger than ndim, in case that the complete input
        // shape was passed and not only the ndim last ones
        if (mxnet::dim_size_is_known(input, input_ndim - ndim + i)) {
          o_pad[i] = stride[i] * (input[(input_ndim - ndim) + i] - 1) + DilatedKernelSize(i);
          CHECK_GE(o_pad[i], target_shape[i]) << "too big target shape";
          o_pad[i] -= target_shape[i];
          o_adj[i] = o_pad[i] % 2;
          o_pad[i] = (o_pad[i] + 1) / 2;
        }
      }
    } else {
      for (int i = 0; i < static_cast<int>(ndim); i++) {
        o_pad[i] = i < pad.ndim() ? pad[i] : 0;
        o_adj[i] = i < adj.ndim() ? adj[i] : 0;
      }
    }
  }

  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const DeconvolutionParam& other) const {
    return this->kernel == other.kernel &&
           this->stride == other.stride &&
           this->dilate == other.dilate &&
           this->pad == other.pad &&
           this->adj == other.adj &&
           this->target_shape == other.target_shape &&
           this->num_filter == other.num_filter &&
           this->num_group == other.num_group &&
           this->workspace == other.workspace &&
           this->no_bias == other.no_bias &&
           this->cudnn_tune == other.cudnn_tune &&
           this->cudnn_off == other.cudnn_off &&
           this->layout == other.layout;
  }
};

typedef ParamOpSign<DeconvolutionParam> DeconvSignature;

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::DeconvolutionParam> {
  size_t operator()(const mxnet::op::DeconvolutionParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.dilate);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.adj);
    ret = dmlc::HashCombine(ret, val.target_shape);
    ret = dmlc::HashCombine(ret, val.num_filter);
    ret = dmlc::HashCombine(ret, val.num_group);
    ret = dmlc::HashCombine(ret, val.workspace);
    ret = dmlc::HashCombine(ret, val.no_bias);
    ret = dmlc::HashCombine(ret, val.cudnn_tune);
    ret = dmlc::HashCombine(ret, val.cudnn_off);
    ret = dmlc::HashCombine(ret, val.layout);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class DeconvolutionOp {
 public:
  void Init(DeconvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;

    if (param_.kernel.ndim() > 2) {
      LOG(FATAL) << "If not using CUDNN, only 1D or 2D Deconvolution is supported";
    }

    CHECK_EQ(req[deconv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    LayerSetUp(in_data[deconv::kData].shape_, out_data[deconv::kData].shape_);
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init cuBLAS handle in stream";
#endif
    auto in_data_shape = in_data[deconv::kData].shape_;
    // G: num of groups
    // N: num of batches
    // C: num of channels
    // IH: input height
    // IW: input width
    // KH: kernel height
    // KW: kernel width
    // OH: output width
    // OW: output height
    // OC: num of output channels

    // input_4d: (N, C, IH, IW)
    // output_4d: (N, OC, OH, OW)
    Tensor<xpu, 4, DType> input_4d = TBlobTo4DTensor(in_data[deconv::kData], s);
    Tensor<xpu, 4, DType> output_4d = TBlobTo4DTensor(out_data[deconv::kOut], s);
    index_t o_pad[2], o_adj[2];
    if (param_.kernel.ndim() == 2) {
      param_.InferPad(mxnet::TShape({in_data_shape[2], in_data_shape[3]}), o_pad, o_adj);
    } else {
      param_.InferPad({in_data_shape[2]}, o_pad, o_adj);
    }

    auto stride = param_.kernel.ndim() == 2 ? param_.stride : TShape({1, param_.stride[0]});
    auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : TShape({1, param_.dilate[0]});
    auto padding = param_.kernel.ndim() == 2 ? TShape({o_pad[0], o_pad[1]}) : TShape({0, o_pad[0]});

    // weight_3d: (G, C/G, OC/G * KH * KW)
    Tensor<xpu, 3, DType> weight_3d = in_data[deconv::kWeight].get_with_shape<xpu, 3, DType>(
        Shape3(group_, conv_in_channels_ / group_, kernel_dim_), s);

    Tensor<xpu, 1, DType> workspace = ctx.requested[deconv::kTempSpace]
        .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_ + in_data[deconv::kData].shape_.Size()), s);

    mxnet::TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
    col_buffer_shape[0] = conv_out_channels_ * param_.kernel.Size();
    for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = in_data[deconv::kData].shape_[i + 1];
    }

    // create a colum buffer to hold the matrix product between weight_3d(T) and input_data
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    // col_buffer_3d : (G, OC/G * KH * KW, IH * IW)
    Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
        Shape3(group_, kernel_dim_, conv_in_spatial_dim_), s);

    for (index_t i = 0; i < num_; ++i) {
      // Tensor<xpu, 3, DType> data_3d = input_4d[i];
      Tensor<xpu, 3, DType> data_3d = Tensor<xpu, 3, DType>(
          workspace.dptr_ + col_buffer_size_,
          Shape3(group_, input_4d.shape_[1] / group_, conv_in_spatial_dim_), s);

      // data_3d : (G, IC/G, IH * IW)
      data_3d = reshape(swapaxis<1, 0>(input_4d.Slice(i, i + 1)), data_3d.shape_);
      /*
          std::cout << "data_3d: " << std::endl;
          DType *tmp_data = new DType[data_3d.shape_.Size()];
          if (ctx.run_ctx.get_ctx().dev_mask() == gpu::kDevMask) {
            std::cout << "running on GPU " << std::endl;
            NDArray data(data_3d, ctx.run_ctx.get_ctx().dev_id);
            data.SyncCopyToCPU(tmp_data, data_3d.shape_.Size());
            std::cout << "complete " << std::endl;
          } else {
            tmp_data = static_cast<DType *>(data_3d[0].dptr_);
          }

          for (auto j = 0; j < data_3d.shape_[1]; ++j) {
            for (auto k = 0; k < data_3d.shape_[2]; ++k) {
                std::cout << *(tmp_data + j * data_3d.shape_[2] + k) << " ";
            }
            std::cout << std::endl;
          }
      */
      for (int g = 0; g < group_; ++g) {
        // Legacy approach shown here for comparison:
        // col_buffer_3d[g] = dot(weight_3d[g].T(), data_3d[g]);
        linalg_gemm(weight_3d[g], data_3d[g], col_buffer_3d[g], true, false, s);
      }


      // TODO: (lnyuan) remove debugging code
      /*
            std::cout << "col buffer: " << std::endl;
            DType *tmp_col = new DType[col_buffer_size_];
            if (ctx.run_ctx.get_ctx().dev_mask() == gpu::kDevMask) {
              std::cout << "running on GPU " << std::endl;
              NDArray col_data(col_buffer, ctx.run_ctx.get_ctx().dev_id);
              col_data.SyncCopyToCPU(tmp_col, col_buffer_size_);
              std::cout << "complete " << std::endl;
            } else {
              tmp_col = static_cast<DType *>(col_buffer_3d[0].dptr_);
            }

            for (auto j = 0; j < col_buffer_3d.shape_[1]; ++j) {
              for (auto k = 0; k < col_buffer_3d.shape_[2]; ++k) {
                std::cout << *(tmp_col + j * col_buffer_3d.shape_[2] + k) << " ";
              }
              std::cout << std::endl;
            }
      */
      col2im(s, col_buffer.dptr<DType>(), out_data[deconv::kOut].shape_, col_buffer.shape_,
          param_.kernel, padding, stride, dilate,
          out_data[deconv::kOut].dptr<DType>() + i * output_dim_, req[deconv::kOut]);
    }

    if (bias_term_) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1, DType> bias = in_data[deconv::kBias].get<xpu, 1, DType>(s);
      output_4d += mshadow::expr::broadcast<1>(bias, output_4d.shape_);
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(in_grad.size(), expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[deconv::kWeight].CheckContiguous(), true);

    LayerSetUp(in_grad[deconv::kData].shape_, out_grad[deconv::kOut].shape_);
    Stream<xpu> *s = ctx.get_stream<xpu>();
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init cuBLAS handle in stream";
#endif

    auto in_data_shape = in_data[deconv::kData].shape_;
    Tensor<xpu, 4, DType> data_4d = TBlobTo4DTensor(in_data[deconv::kData], s);
    Tensor<xpu, 4, DType> grad = TBlobTo4DTensor(out_grad[deconv::kOut], s);
    Tensor<xpu, 4, DType> gdata = TBlobTo4DTensor(in_grad[deconv::kData], s);
    index_t o_pad[2], o_adj[2];
    if (param_.kernel.ndim() == 2) {
      param_.InferPad(mxnet::TShape({in_data_shape[2], in_data_shape[3]}), o_pad, o_adj);
    } else {
      param_.InferPad({in_data_shape[2]}, o_pad, o_adj);
    }
    auto stride = param_.kernel.ndim() == 2 ? param_.stride : TShape({1, param_.stride[0]});
    auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : TShape({1, param_.dilate[0]});
    auto padding = param_.kernel.ndim() == 2 ? TShape({o_pad[0], o_pad[1]}) : TShape({0, o_pad[0]});

    // weight_3d: (G, C/G, OC * KH * KW)
    Tensor<xpu, 3, DType> weight_3d = in_data[deconv::kWeight]
        .get_with_shape<xpu, 3, DType>(Shape3(group_, conv_in_channels_ / group_, kernel_dim_), s);

    // dweight_3d: (G, C/G, OC * KH * KW)
    Tensor<xpu, 3, DType> dweight_3d = in_grad[deconv::kWeight]
        .get_with_shape<xpu, 3, DType>(Shape3(group_, conv_in_channels_ / group_, kernel_dim_), s);

    Tensor<xpu, 1, DType> workspace = ctx.requested[deconv::kTempSpace]
        .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_ + data_4d.shape_.Size()), s);

    // calculate shape of col_buffer
    TShape col_buffer_shape(num_spatial_axes_ + 1, 1);
    col_buffer_shape[0] = conv_out_channels_ * param_.kernel.Size();
    for (int i = 1; i < col_buffer_shape.ndim(); ++i) {
      col_buffer_shape[i] = in_data[deconv::kData].shape_[i+1];
    }

    // create a column buffer to store ograd
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    // col_buffer_3d: (G, OC/G * KH * KW, IH * IW)
    Tensor<xpu, 3, DType> col_buffer_3d = col_buffer.get_with_shape<xpu, 3, DType>(
        Shape3(group_, kernel_dim_, conv_in_spatial_dim_), s);

    for (index_t i = 0; i < num_; ++i) {
      // Tensor<xpu, 3, DType> data_3d = input_4d[i];
      Tensor<xpu, 3, DType> data_3d = Tensor<xpu, 3, DType>(
          workspace.dptr_ + col_buffer_size_,
          Shape3(group_, data_4d.shape_[1] / group_, conv_in_spatial_dim_), s);

      // data_3d : (G, C/G, IH * IW)
      data_3d = reshape(swapaxis<1, 0>(data_4d.Slice(i, i + 1)), data_3d.shape_);

      // convert output gradient array to column buffer
      im2col(s, out_grad[deconv::kOut].dptr<DType>() + i * output_dim_, out_grad[deconv::kOut].shape_,
          col_buffer.shape_, param_.kernel, padding, stride, dilate, col_buffer.dptr<DType>());

      for (int g = 0; g < group_; ++g) {
        auto request = (i == 0) ? req[deconv::kWeight] : kAddTo;
        // Legacy approach shown here for comparison:
        // dweight_3d[gid] += dot(temp_dst[gid], tmpc.T());
        linalg_gemm(data_3d[g], col_buffer_3d[g], dweight_3d[g], false, true, s, request);
      }
      if (req[deconv::kData] == kWriteTo ||
          req[deconv::kData] == kWriteInplace ||
          req[deconv::kData] == kAddTo) {
        for (int g = 0; g < group_; ++g) {
          // Legacy approach shown here for comparison:
          // temp_dst[gid] = dot(weight_3d[gid], tmpc);
          linalg_gemm(weight_3d[g], col_buffer_3d[g], data_3d[g], false, false, s);
        }
        Assign(gdata.Slice(i, i + 1),
               req[deconv::kData],
               (swapaxis<1, 0>(reshape(data_3d,
                                       Shape4(gdata.shape_[1],
                                              1,
                                              gdata.size(2),
                                              gdata.size(3))))));
      }
    }
    if (!param_.no_bias) {
      Tensor<xpu, 1, DType> gbias = in_grad[deconv::kBias].get<xpu, 1, DType>(s);
      Assign(gbias, req[deconv::kBias], sumall_except_dim<1>(grad));
    }
  }

 private:

  inline Tensor<xpu, 4, DType> TBlobTo4DTensor(const TBlob &tb, Stream<xpu> *s) {
    using namespace mshadow;
    if (param_.kernel.ndim() == 2)
      return tb.get<xpu, 4, DType>(s);
    else
      return tb.get_with_shape<xpu, 4, DType>(
          Shape4(tb.shape_[0], tb.shape_[1], 1, tb.shape_[2]), s);
  }

  void LayerSetUp(const mxnet::TShape& ishape, const mxnet::TShape& oshape) {
    channel_axis_ = 1;  // hard code channel axis
    const index_t first_spatial_axis = channel_axis_ + 1;
    const int num_axes = param_.kernel.ndim() + 2;
    num_spatial_axes_ = num_axes - first_spatial_axis;

    // batch size
    num_ = ishape[0];
    // number of input channels
    channels_ = ishape[1];
    group_ = param_.num_group;
    conv_out_channels_ = param_.num_filter;
    conv_in_channels_ = channels_;
    bias_term_ = !param_.no_bias;
    kernel_dim_ = conv_out_channels_ / group_ * param_.kernel.Size();
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    conv_out_spatial_dim_ = oshape.ProdShape(2, oshape.ndim());
    conv_in_spatial_dim_ = ishape.ProdShape(2, ishape.ndim());
    // size of the column buffer used for storing im2col-ed pixels
    col_buffer_size_ = kernel_dim_ * group_ * conv_in_spatial_dim_;
    // input/output image size (#channels * height * width)
    input_dim_ = ishape.ProdShape(1, ishape.ndim());
    output_dim_ = oshape.ProdShape(1, oshape.ndim());
  }

private:
  DeconvolutionParam param_;
  index_t channel_axis_;  // channel axis of the input
  index_t channels_;  // number of channels of input image
  index_t num_spatial_axes_;  // number of spatial axes
  index_t num_;  // batch size
  index_t group_;  // number of groups
  index_t conv_out_channels_;  // number of output channels (num_filter)
  index_t conv_out_spatial_dim_;  // number of pixels of output images per channel
  index_t conv_in_spatial_dim_; // number of pixels of input images per channel
  index_t conv_in_channels_;  // number of input channels
  index_t kernel_dim_;  // number of input channels per group * kernel size
  index_t weight_offset_;  // number of output channels per group * kernel_dim_
  index_t col_buffer_size_;
  index_t input_dim_;
  index_t output_dim_;
  bool bias_term_;  // has bias term?
};  // class DeconvolutionOp

template<typename xpu>
void _DeconvolutionCompute(const DeconvolutionParam& param,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  MSHADOW_REAL_TYPE_SWITCH(inputs[deconv::kData].type_flag_, DType, {
    DeconvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void DeconvolutionCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  _DeconvolutionCompute<xpu>(param, ctx, inputs, req, outputs);
}

template<typename xpu>
void _DeconvolutionGradCompute(const DeconvolutionParam& param,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    DeconvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}


template<typename xpu>
void DeconvolutionGradCompute(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx, const std::vector<TBlob>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<TBlob>& outputs) {
  const DeconvolutionParam& param = nnvm::get<DeconvolutionParam>(attrs.parsed);
  _DeconvolutionGradCompute<xpu>(param, ctx, inputs, req, outputs);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_DECONVOLUTION_INL_H_
