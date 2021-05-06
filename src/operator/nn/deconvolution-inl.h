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
#include "../operator_common.h"
#include "../linalg.h"


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
  void InferPad(mxnet::TShape input, index_t (&o_pad)[ndim], index_t (&o_adj)[ndim] ) const {
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
      LOG(FATAL) << "Only 1D or 2D Deconvolution is natively supported. "
                 << ((MXNET_USE_MKLDNN || MXNET_USE_CUDNN)
                         ? "Fallback to native implementation (if occurred) is therefore "
                           "impossible for 3D Deconvolution."
                         : "");
    }

    CHECK_EQ(req[deconv::kOut], kWriteTo);
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    auto in_data_shape = in_data[deconv::kData].shape_;
    Tensor<xpu, 4, DType> data = TBlobTo4DTensor(in_data[deconv::kData], s);
    Tensor<xpu, 4, DType> out = TBlobTo4DTensor(out_data[deconv::kOut], s);
    index_t o_pad[2], o_adj[2];
    if (param_.kernel.ndim() == 2) {
      param_.InferPad(mxnet::TShape({in_data_shape[2], in_data_shape[3]}), o_pad, o_adj);
    } else {
      index_t o_pad_1D[1], o_adj_1D[1];
      param_.InferPad({in_data_shape[2]}, o_pad_1D, o_adj_1D);
      o_pad[0] = 0;
      o_pad[1] = o_pad_1D[0];
      o_adj[0] = 0;
      o_adj[1] = o_adj_1D[0];
    }
    auto stride = param_.kernel.ndim() == 2 ? param_.stride : mxnet::TShape({1, param_.stride[0]});
    auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : mxnet::TShape({1, param_.dilate[0]});
    auto kernel = param_.kernel.ndim() == 2 ? param_.kernel : mxnet::TShape({1, param_.kernel[0]});
    auto kernel_size = kernel.Size();

    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               data.shape_[1] / param_.num_group,
               param_.num_filter / param_.num_group * kernel_size);
    Tensor<xpu, 3, DType> wmat =
        in_data[deconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    const index_t nbatch = data.size(0);
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[deconv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(out.shape_, data.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(
                                            workspace.dptr_,
                                            Shape2(shape_colunit_[0],
                                            shape_colunit_[1] * step), s);
      Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
                                           workspace.dptr_ + temp_col.shape_.Size(),
                                           Shape3(shape_dstunit_[0],
                                           shape_dstunit_[1],
                                           shape_dstunit_[2] * step), s);
      temp_dst = reshape(swapaxis<1, 0>(data.Slice(i, i + step)), temp_dst.shape_);
      if (o_pad[0] == 0 && o_pad[1] == 0) {
        temp_col = unpack_patch2col(out.Slice(i, i + step),
                                    kernel[0],
                                    kernel[1],
                                    stride[0],
                                    stride[1],
                                    dilate[0],
                                    dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(out.Slice(i, i + step),
                                        o_pad[0], o_pad[1]),
                                    kernel[0],
                                    kernel[1],
                                    stride[0],
                                    stride[1],
                                    dilate[0],
                                    dilate[1]);
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        mshadow::Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid,
                                              gstride * (gid + 1));
        // Legacy approach shown here for comparison:
        //   tmpc = dot(wmat[gid].T(), temp_dst[gid]);
        linalg_gemm(wmat[gid], temp_dst[gid], tmpc, true, false, s);
      }
      if (o_pad[0] == 0 && o_pad[1] == 0) {
        out.Slice(i, i + step) = pack_col2patch(temp_col,
                                   out.Slice(i, i + step).shape_,
                                   kernel[0],
                                   kernel[1],
                                   stride[0],
                                   stride[1],
                                   dilate[0],
                                   dilate[1]);
      } else {
        Shape<4> pshape = out.Slice(i, i + step).shape_;
        pshape[2] += 2 * o_pad[0];
        pshape[3] += 2 * o_pad[1];
        out.Slice(i, i + step) = crop(pack_col2patch(temp_col,
                                        pshape,
                                        kernel[0],
                                        kernel[1],
                                        stride[0],
                                        stride[1],
                                        dilate[0],
                                        dilate[1]),
                                        out[i][0].shape_);
      }
    }
    if (!param_.no_bias) {
      // add bias, broadcast bias to dim 1: channel
      Tensor<xpu, 1, DType> bias = in_data[deconv::kBias].get<xpu, 1, DType>(s);
      out += mshadow::expr::broadcast<1>(bias, out.shape_);
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &in_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    // TODO(bing): check the BLAS Handle, be careful
    CHECK_EQ(out_grad.size(), 1U);
    size_t expected = param_.no_bias == 0 ? 3 : 2;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(in_grad.size(), expected);
    CHECK_EQ(req.size(), expected);
    CHECK_EQ(in_data[deconv::kWeight].CheckContiguous(), true);
    // get data
    Stream<xpu> *s = ctx.get_stream<xpu>();
    auto in_data_shape = in_data[deconv::kData].shape_;
    Tensor<xpu, 4, DType> data = TBlobTo4DTensor(in_data[deconv::kData], s);
    Tensor<xpu, 4, DType> grad = TBlobTo4DTensor(out_grad[deconv::kOut], s);
    Tensor<xpu, 4, DType> gdata = TBlobTo4DTensor(in_grad[deconv::kData], s);

    index_t o_pad[2], o_adj[2];
    if (param_.kernel.ndim() == 2) {
      param_.InferPad(mxnet::TShape({in_data_shape[2], in_data_shape[3]}), o_pad, o_adj);
    } else {
      index_t o_pad_1D[1], o_adj_1D[1];
      param_.InferPad({in_data_shape[2]}, o_pad_1D, o_adj_1D);
      o_pad[0] = 0;
      o_pad[1] = o_pad_1D[0];
      o_adj[0] = 0;
      o_adj[1] = o_adj_1D[0];
    }
    auto stride = param_.kernel.ndim() == 2 ? param_.stride : mxnet::TShape({1, param_.stride[0]});
    auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : mxnet::TShape({1, param_.dilate[0]});
    auto kernel = param_.kernel.ndim() == 2 ? param_.kernel : mxnet::TShape({1, param_.kernel[0]});
    auto kernel_size = kernel.Size();

    Shape<3> wmat_shape =
        Shape3(param_.num_group,
               data.shape_[1] / param_.num_group,
               param_.num_filter / param_.num_group * kernel_size);
    Tensor<xpu, 3, DType> wmat =
        in_data[deconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
    Tensor<xpu, 3, DType> gwmat =
        in_grad[deconv::kWeight].get_with_shape<xpu, 3, DType>(wmat_shape, s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif

    const index_t nbatch = data.size(0);
    Tensor<xpu, 1, DType> workspace =
        ctx.requested[deconv::kTempSpace].get_space_typed<xpu, 1, DType>(
            Shape1(this->InitTemp(grad.shape_, data.shape_)), s);
    for (index_t i = 0; i < nbatch; i += nstep_) {
      const index_t step = std::min(nstep_, nbatch - i);
      Tensor<xpu, 2, DType> temp_col = Tensor<xpu, 2, DType>(
                                           workspace.dptr_,
                                           Shape2(shape_colunit_[0],
                                           shape_colunit_[1] * step), s);
      Tensor<xpu, 3, DType> temp_dst = Tensor<xpu, 3, DType>(
                                           workspace.dptr_ + temp_col.shape_.Size(),
                                           Shape3(shape_dstunit_[0],
                                           shape_dstunit_[1],
                                           shape_dstunit_[2] * step), s);
      temp_dst = reshape(swapaxis<1, 0>(data.Slice(i, i + step)), temp_dst.shape_);
      if (o_pad[0] == 0 && o_pad[1] == 0) {
        temp_col = unpack_patch2col(grad.Slice(i, i + step),
                                     kernel[0],
                                     kernel[1],
                                     stride[0],
                                     stride[1],
                                     dilate[0],
                                     dilate[1]);
      } else {
        temp_col = unpack_patch2col(pad(grad.Slice(i, i + step), o_pad[0], o_pad[1]),
                                     kernel[0],
                                     kernel[1],
                                     stride[0],
                                     stride[1],
                                     dilate[0],
                                     dilate[1]);
      }
      const index_t gstride = temp_col.size(0) / param_.num_group;
      for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
        Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
        if (i == 0) {
          Tensor<xpu, 2, DType> tmp_gwmat = gwmat[gid];
          // Legacy approach shown here for comparison:
          //   Assign(tmp_gwmat, req[deconv::kWeight], dot(temp_dst[gid], tmpc.T()));
          linalg_gemm(temp_dst[gid], tmpc, tmp_gwmat, false, true, s, req[deconv::kWeight]);
        } else {
          // Legacy approach shown here for comparison:
          //   gwmat[gid] += dot(temp_dst[gid], tmpc.T());
          linalg_gemm(temp_dst[gid], tmpc, gwmat[gid], false, true, s, kAddTo);
        }
      }
      if (req[deconv::kData] == kWriteTo ||
          req[deconv::kData] == kWriteInplace ||
          req[deconv::kData] == kAddTo) {
        for (uint32_t gid = 0; gid < param_.num_group; ++gid) {
          Tensor<xpu, 2, DType> tmpc = temp_col.Slice(gstride * gid, gstride * (gid + 1));
          // Legacy approach shown here for comparison:
          //   temp_dst[gid] = dot(wmat[gid], tmpc);
          linalg_gemm(wmat[gid], tmpc, temp_dst[gid], false, false, s);
        }
        Assign(gdata.Slice(i, i + step),
               req[deconv::kData],
               (swapaxis<1, 0>(reshape(temp_dst,
                                      mshadow::Shape4(gdata.shape_[1],
                                                      step,
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
  inline index_t InitTemp(const mshadow::Shape<4> &ishape,
                          const mshadow::Shape<4> &oshape) {
    const int ksize = param_.kernel.Size();
    shape_colunit_ = mshadow::Shape2(ishape[1] * ksize,
                                     oshape[2] * oshape[3]);
    shape_dstunit_ = mshadow::Shape3(param_.num_group,
                                     oshape[1] / param_.num_group,
                                     oshape[2] * oshape[3]);
    // See convolution for workspace calculations. nstep_ will be the effective batch size
    nstep_ = std::max<index_t>(
        std::min<index_t>(param_.workspace /
          (shape_colunit_.Size() + shape_dstunit_.Size()), ishape[0]),
      1);

    mshadow::Shape<2> scol = mshadow::Shape2(shape_colunit_[0],
                                             shape_colunit_[1] * nstep_);
    mshadow::Shape<3> sdst = mshadow::Shape3(shape_dstunit_[0],
                                             shape_dstunit_[1],
                                             shape_dstunit_[2] * nstep_);
    index_t required_size = scol.Size() + sdst.Size();
    return required_size;
  }

  inline Tensor<xpu, 4, DType> TBlobTo4DTensor(const TBlob &tb, Stream<xpu> *s) {
    using namespace mshadow;
    if (param_.kernel.ndim() == 2)
      return tb.get<xpu, 4, DType>(s);
    else
      return tb.get_with_shape<xpu, 4, DType>(
          Shape4(tb.shape_[0], tb.shape_[1], 1, tb.shape_[2]), s);
  }

  DeconvolutionParam param_;
  mshadow::Shape<2> shape_colunit_;
  mshadow::Shape<3> shape_dstunit_;
  index_t nstep_;
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
