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
 * \file dropout-inl.h
 * \brief
 * \author Bing Xu, Da Zheng, Hang Zhang
*/

#ifndef MXNET_OPERATOR_NN_DROPOUT_INL_H_
#define MXNET_OPERATOR_NN_DROPOUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../random/sampler.h"
#include "../tensor/elemwise_binary_broadcast_op.h"

#if (MSHADOW_USE_MKL == 1) && defined(_OPENMP) && !defined(__CUDACC__)
#define MXNET_USE_MKL_DROPOUT 1
#endif

#if MXNET_USE_MKL_DROPOUT
#include <omp.h>

#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#endif  // MXNET_USE_MKL_DROPOUT

#define MXNET_USE_CUDNN_DROPOUT MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7

namespace dropout {
enum DropoutOpInputs {kData};
enum DropoutOpOutputs {kOut, kMask};
enum DropoutOpForwardResource {kRandom};
enum DropoutOpMode {kTraining, kAlways};
}  // namespace dropout

namespace mxnet {
namespace op {

const int MAX_DIM = 5;

struct DropoutParam : public dmlc::Parameter<DropoutParam> {
  float p;
  int mode;
  mxnet::TShape axes;
  dmlc::optional<bool> cudnn_off;
  DMLC_DECLARE_PARAMETER(DropoutParam) {
    DMLC_DECLARE_FIELD(p).set_default(0.5)
    .set_range(0, 1)
    .describe("Fraction of the input that gets dropped out during training time.");
    DMLC_DECLARE_FIELD(mode)
    .add_enum("training", dropout::kTraining)
    .add_enum("always", dropout::kAlways)
    .set_default(dropout::kTraining)
    .describe("Whether to only turn on dropout during training or to also turn on for inference.");
    DMLC_DECLARE_FIELD(axes).set_default(mxnet::TShape(0, 0))
    .describe("Axes for variational dropout kernel. Same dropout will be applied to elements "
              "along the specified axis.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(dmlc::optional<bool>(false))
    .describe("Whether to turn off cuDNN in dropout operator. "
              "This option is ignored if axes is specified.");
  }
  std::string Mode2String(int mode) {
    switch (mode) {
      case dropout::kTraining:
        return "training";
      case dropout::kAlways:
        return "always";
      default:
        LOG(FATAL) << "Unknown mode enum " << mode;
    }
    LOG(FATAL) << "should not reach here ";
    return "";
  }
  void SetAttrDict(std::unordered_map<std::string, std::string>* dict) {
    std::ostringstream p_s, mode_s, axes_s, cudnn_off_s;
    p_s << p;
    mode_s << mode;
    axes_s << axes;
    cudnn_off_s << cudnn_off;
    (*dict)["p"] = p_s.str();
    (*dict)["mode"] = Mode2String(mode);
    (*dict)["axes"] = axes_s.str();
    (*dict)["cudnn_off"] = cudnn_off_s.str();
  }
};  // struct DropoutParam

template<typename xpu, typename DType>
class DropoutOp {
#if MXNET_USE_MKL_DROPOUT
  static void BernoulliGenerate(common::random::RandGenerator<cpu, DType> gen,
                                int n, double p, int* r) {
    typename RandGenerator<xpu, DType>::Impl genImpl(&gen, 1);
    const int seed = 17 + abs(genImpl.rand() % 4096);
    CHECK_GE(seed, 0);
    const int nthr = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
#pragma omp parallel num_threads(nthr)
    {
      const int ithr = omp_get_thread_num();
      const int avg_amount = (n + nthr - 1) / nthr;
      const int my_offset = ithr * avg_amount;
      const int my_amount = std::min(my_offset + avg_amount, n) - my_offset;
      if (my_amount > 0) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MCG31, seed);
        vslSkipAheadStream(stream, my_offset);
        viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, stream, my_amount, r + my_offset, p);
        vslDeleteStream(&stream);
      }
    }
  }
  static inline bool MKLAvailable() {
    // TODO(lnyuan): how to let user enable/disable MKL Dropout
    return true;
  }

  // MKL forward pass
  inline void MKLForward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<TBlob> &out_data) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    RandGenerator<xpu, DType> *pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
    CHECK_NOTNULL(pgen);
    Tensor<xpu, 1, uint8_t> mask = out_data[dropout::kMask].FlatTo1D<xpu, uint8_t>(s);
    Tensor<xpu, 2, DType> data = in_data[dropout::kData].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = out_data[dropout::kOut].FlatTo2D<xpu, DType>(s);
    DType *outptr = out.dptr_;
    DType *dataptr = data.dptr_;

    index_t count = data.shape_[0] * data.shape_[1];
    // allocating buffer for MKL routine to calculate int32 based maskptr
    Tensor<xpu, 1, int> temp_space =
      ctx.requested[1].get_space_typed<xpu, 1, int>(Shape1(count), s);
    auto mkl_mask = temp_space.dptr_;

    BernoulliGenerate(*pgen, count, this->pkeep_, mkl_mask);
    const float pk_1 = 1.0f / this->pkeep_;
    const int nthr = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();
    const int blk_size = 64;
    const int nblk = count / blk_size;

    #pragma omp parallel num_threads(nthr)
    {
      #pragma omp for
      for (index_t b = 0; b < nblk; ++b) {
        for (index_t k = 0; k < blk_size; ++k) {
          const index_t i = b * blk_size + k;
          outptr[i] = dataptr[i] * mkl_mask[i] * pk_1;
          auto mask_idx = i >> 3;  // div 8
          uint8_t mask_offset = i & 7;  // mod 8
          if (mkl_mask[i]) {
            // set bit
            mask.dptr_[mask_idx] |= 1U << mask_offset;
          } else {
            // clear bit
            mask.dptr_[mask_idx] &= ~(1U << mask_offset);
          }
        }
      }
    }

    // tail
    for (index_t i = nblk * blk_size; i < count; ++i) {
      outptr[i] = dataptr[i] * mkl_mask[i] * pk_1;
      auto mask_idx = i >> 3;  // div 8
      uint8_t mask_offset = i & 7;  // mod 8
      if (mkl_mask[i]) {
        // set bit
        mask.dptr_[mask_idx] |= 1U << mask_offset;
      } else {
        // clear bit
        mask.dptr_[mask_idx] &= ~(1U << mask_offset);
      }
    }
  }

  // MKL backward pass
  inline void MKLBackward(const OpContext &ctx,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &out_data,
                          const std::vector<TBlob> &out_grad) {
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> grad = out_grad[dropout::kOut].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 1, uint8_t> mask = out_data[dropout::kMask].FlatTo1D<xpu, uint8_t>(s);
    Tensor<xpu, 2, DType> gdata = in_grad[dropout::kData].FlatTo2D<xpu, DType>(s);
    DType *ingradptr = gdata.dptr_;
    const DType *outgradptr = grad.dptr_;
    const uint8_t *maskptr = mask.dptr_;
    const index_t count = grad.shape_[0] * grad.shape_[1];
    const float pk_1 = 1.0f / this->pkeep_;
    const int nthr = engine::OpenMP::Get()->GetRecommendedOMPThreadCount();

#pragma omp parallel for num_threads(nthr)
    for (index_t i = 0; i < count; ++i) {
      auto mask_idx = i >> 3;  // div 8;
      uint8_t mask_offset = i & 7;  // mod 8
      bool mask_val = maskptr[mask_idx] & (1U << mask_offset);
      ingradptr[i] = outgradptr[i] * mask_val * pk_1;
    }
  }
#endif  // #if MXNET_USE_MKL_DROPOUT

 public:
  /*!
   * \brief Dropout kernel, compute dropout tensor
   */
  struct DropoutKernel {
    /*!
     * \brief Dropout kernel function
     * \param id Thread number (0-based representing count)
     * \param gen Random number generator
     * \param N Total number of items in the output
     * \param step Step between items, related to parallelism
     * \param dropout_out Output dropout values
     * \param mask_out  Output mask with one bit for each element
     * \param input_data Input data to perform the dropout on
     * \param pkeep Dropout rate (keep when the generated random number is less than this value)
     */
    MSHADOW_XINLINE static void Map(index_t id,
                                    RandGenerator<xpu, DType> gen,
                                    const index_t N,
                                    const index_t step,
                                    DType *dropout_out,
                                    uint8_t *mask_out,
                                    const DType *input_data,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        // mask_out is set per bit position
        // therefore bitwise shift need to be performed here
        auto mask_idx = i >> 3;  // div 8;
        uint8_t mask_offset = i & 7;  // mod 8
        bool mask_val = mshadow_op::threshold_eq::Map<real_t>(rand_num, pkeep);
        const float pk_1 = 1.0f / pkeep;
        if (mask_val) {
          // set bit
          mask_out[mask_idx] |= 1U << mask_offset;
        } else {
          // clear bit
          mask_out[mask_idx] &= ~(1U << mask_offset);
        }
        dropout_out[i] = mask_val * input_data[i] * pk_1;
      })
    }
  };

  struct DropoutBackwardKernel {
    MSHADOW_XINLINE static void Map(index_t i,
                                    OpReqType req,
                                    DType *igrad,
                                    DType *ograd,
                                    const uint8_t *mask,
                                    const real_t pkeep) {
      auto mask_idx = i >> 3;  // div 8;
      uint8_t mask_offset = i & 7;  // mod 8
      bool mask_val = mask[mask_idx] & (1U << mask_offset);
      const float pk_1 = 1.0f / pkeep;
      KERNEL_ASSIGN(igrad[i], req, mask_val * ograd[i] * pk_1);
    }
  };

  struct BernoulliKernel {
    /*! \brief Bernoulli kernel for generating mask */
    MSHADOW_XINLINE static void Map(index_t id,
                                    RandGenerator<xpu, DType> gen,
                                    const index_t N,
                                    const index_t step,
                                    DType *dropout_out,
                                    uint8_t *mask_out,
                                    const real_t pkeep) {
      RNG_KERNEL_LOOP(xpu, DType, id, gen, N, step, {
        const real_t rand_num = static_cast<real_t>(genImpl.uniform());
        // mask_out is set per bit position
        // therefore bitwise shift need to be performed here
        auto mask_idx = i >> 3;  // div 8;
        uint8_t mask_offset = i & 7;  // mod 8
        bool mask_val = mshadow_op::threshold_eq::Map<real_t>(rand_num, pkeep);
        const float pk_1 = 1.0f / pkeep;
        if (mask_val) {
          // set bit
          mask_out[mask_idx] |= 1U << mask_offset;
        } else {
          // clear bit
          mask_out[mask_idx] &= ~(1U << mask_offset);
        }
        dropout_out[i] = mask_val * pk_1;
      })
    }
  };

  template<int ndim>
  struct BernoulliBackwardKernel {
    MSHADOW_XINLINE static void Map(index_t base,
                                    index_t length,
                                    OpReqType req,
                                    const Shape<ndim> &lstride,
                                    const Shape<ndim> &rstride,
                                    const Shape<ndim> &oshape,
                                    DType *igrad,
                                    DType *ograd,
                                    const uint8_t *mask,
                                    const real_t pkeep) {
      Shape <ndim> coord = unravel(base, oshape);
      auto lidx = static_cast<index_t>(dot(coord, lstride));
      auto ridx = static_cast<index_t>(dot(coord, rstride));
      auto mask_idx = ridx >> 3;  // div 8;
      uint8_t mask_offset = ridx & 7;  // mod 8
      bool mask_val = mask[mask_idx] & (1U << mask_offset);
      const float pk_1 = 1.0f / pkeep;
      KERNEL_ASSIGN(igrad[base], req, mask_val * ograd[lidx] * pk_1);
      // starts from 1 to avoid extra inc at end of loop
      for (index_t i = 1; i < length; ++i) {
        inc(&coord, oshape, &lidx, lstride, &ridx, rstride);
        mask_idx = ridx >> 3;  // div 8
        mask_offset = ridx & 7;  // mod 8
        mask_val = mask[mask_idx] & (1U << mask_offset);
        KERNEL_ASSIGN(igrad[base + i], req, mask_val * ograd[lidx] * pk_1);
      }
    }
  };

  explicit DropoutOp(const DropoutParam &param, Context ctx) {
    this->pkeep_ = 1.0f - param.p;
    this->mode_ = static_cast<dropout::DropoutOpMode>(param.mode);
    this->axes_ = param.axes;
    this->dropout_passthrough_ = true;
#if MXNET_USE_CUDNN_DROPOUT
    this->cudnn_off_ = param.cudnn_off && param.cudnn_off.value();
    this->ctx_ = ctx;
    if (ctx.dev_type == kGPU && this->pkeep_ > 0 && !this->cudnn_off_) {
      dtype_ = mshadow::DataType<DType>::kCudnnFlag;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc_));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc_));
      CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc_));
    }
#endif  // MXNET_USE_CUDNN_DROPOUT
  }

  ~DropoutOp() {
#if MXNET_USE_CUDNN_DROPOUT
    if (this->ctx_.dev_type == kGPU && this->pkeep_ > 0 && !this->cudnn_off_) {
      CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc_));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc_));
      CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc_));
    }
#endif  // MXNET_USE_CUDNN_DROPOUT
  }

#if MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
  inline bool CuDNNAvailable() {
    return this->pkeep_ > 0 && !this->cudnn_off_;
  }

  inline void CuDNNForward(const OpContext &ctx,
                           const TBlob &in,
                           const TBlob &mask,
                           const TBlob &out) {
      Stream<xpu> *s = ctx.get_stream<xpu>();

      // set dropout state.
      ctx.requested[0].get_cudnn_dropout_desc(&dropout_desc_, s, 1.0f - this->pkeep_);

      // describe input/output tensor
      int dim[4], stride[4];
      dim[0] = 1;
      dim[1] = 1;
      dim[2] = 1;
      dim[3] = out.Size();
      stride[0] = out.Size();
      stride[1] = out.Size();
      stride[2] = out.Size();
      stride[3] = 1;
      CUDNN_CALL(cudnnSetTensorNdDescriptor(x_desc_,
                                            dtype_,
                                            4,
                                            dim,
                                            stride));
      CUDNN_CALL(cudnnSetTensorNdDescriptor(y_desc_,
                                            dtype_,
                                            4,
                                            dim,
                                            stride));

      // perform dropout with cudnn
      CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(x_desc_, &dropout_reserve_byte_));
      // cudnn uses bits to record the positions that are dropped, so reserve bytes is always
      // 1/8 of input size.
      CHECK_GE(mask.Size() * sizeof(uint8_t), dropout_reserve_byte_) <<
        "The size of the mask space is smaller than the required cudnn reserved space.";
      CUDNN_CALL(cudnnDropoutForward(s->dnn_handle_,
                                     dropout_desc_,
                                     x_desc_,
                                     in.dptr<DType>(),
                                     y_desc_,
                                     out.dptr<DType>(),
                                     mask.dptr<uint8_t>(),
                                     dropout_reserve_byte_));
  }

  inline void CuDNNBackward(const OpContext &ctx,
                            const TBlob &out_grad,
                            const TBlob &mask,
                            const TBlob &in_grad) {
      Stream<xpu> *s = ctx.get_stream<xpu>();

      // describe input/output tensor
      int dim[4], stride[4];
      dim[0] = 1;
      dim[1] = 1;
      dim[2] = 1;
      dim[3] = in_grad.Size();
      stride[0] = in_grad.Size();
      stride[1] = in_grad.Size();
      stride[2] = in_grad.Size();
      stride[3] = 1;
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dy_desc_,
                                            dtype_,
                                            4,
                                            dim,
                                            stride));
      CUDNN_CALL(cudnnSetTensorNdDescriptor(dx_desc_,
                                            dtype_,
                                            4,
                                            dim,
                                            stride));

      // perform dropout with cudnn
      CUDNN_CALL(cudnnDropoutBackward(s->dnn_handle_,
                                      dropout_desc_,
                                      dy_desc_,
                                      out_grad.dptr<DType>(),
                                      dx_desc_,
                                      in_grad.dptr<DType>(),
                                      mask.dptr<uint8_t>(),
                                      dropout_reserve_byte_));
  }
#endif  // MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    this->dropout_passthrough_ = true;
    if (req[dropout::kOut] != kNullOp) {
      CHECK_EQ(in_data.size(), 1U);
      if (ctx.is_train) {
        CHECK_EQ(out_data.size(), 2U);
      }
      Stream<xpu> *s = ctx.get_stream<xpu>();
      const TBlob &in = in_data[dropout::kData];
      const TBlob &out = out_data[dropout::kOut];
      const TBlob &mask = out_data[dropout::kMask];
      CHECK_EQ(mask.type_flag_, mshadow::kUint8);

      if (this->pkeep_ < 1 && (ctx.is_train || this->mode_ == dropout::kAlways)) {
        this->dropout_passthrough_ = false;
        if (this->axes_.ndim() == 0) {
          CHECK_EQ((out.Size() + 7) / 8, mask.Size());
#if MXNET_USE_MKL_DROPOUT
          if (MKLAvailable()) {
            MKLForward(ctx, in_data, out_data);
            return;
          }
#endif  // MXNET_USE_MKL_DROPOUT
#if MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
          if (CuDNNAvailable()) {
            CuDNNForward(ctx, in, mask, out);
            return;
          }
#endif  // MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
          RandGenerator<xpu, DType> *pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
          CHECK_NOTNULL(pgen);
          CHECK(req[dropout::kOut] != kAddTo);
          // Use batch size 8 to avoid race condition on mask
          LaunchRNGBatch<DropoutKernel, xpu>(s, pgen, out.Size(), 64 /* batch_size */,
                                             out.dptr<DType>(),
                                             mask.dptr<uint8_t>(),
                                             in.dptr<DType>(),
                                             this->pkeep_);
          return;
        } else {
          // allocating temp buffer to store masked output
          TShape temp_shape = out.shape_;
          for (int i = 0; i < this->axes_.ndim(); ++i) {
            temp_shape[this->axes_[i]] = 1;
          }
          CHECK_EQ((temp_shape.Size() + 7) / 8, mask.Size());
          Tensor<xpu, 1, DType> temp =
              ctx.requested[1].get_space_typed<xpu, 1, DType>(Shape1(temp_shape.Size()), s);
          RandGenerator<xpu, DType> *pgen = ctx.requested[0].get_parallel_random<xpu, DType>();
          CHECK_NOTNULL(pgen);
          // initialize the mask
          // Use batch size 8 to avoid race condition on mask
          LaunchRNGBatch<BernoulliKernel, xpu>(s, pgen, temp_shape.Size(), 64 /* batch_size */,
                                               temp.dptr_,
                                               mask.dptr<uint8_t>(),
                                               this->pkeep_);
          // broadcast mul
          TShape new_lshape, new_rshape, new_oshape;
          int ndim = BinaryBroadcastShapeCompact(in.shape_,
                                                 temp_shape, out.shape_,
                                                 &new_lshape, &new_rshape, &new_oshape);
          if (!ndim) {
            MXNET_ASSIGN_REQ_SWITCH(req[dropout::kOut], Req, {
              mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::mul, Req>, xpu>::Launch(
                s, out.Size(), out.dptr<DType>(), in.dptr<DType>(),
                temp.dptr_);
            });
          } else {
            BROADCAST_NDIM_SWITCH(ndim, NDim, {
              mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
              mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
              mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
              mxnet_op::Kernel<mxnet_op::binary_broadcast_kernel<NDim, mshadow_op::mul>, xpu>::
              template LaunchEx(s, new_oshape.Size(), req[dropout::kOut],
                                lstride, rstride, oshape, in.dptr<DType>(),
                                temp.dptr_, out.dptr<DType>());
            });
          }
        }
      } else {
        if (req[dropout::kOut] == kWriteInplace) return;

        MXNET_ASSIGN_REQ_SWITCH(req[dropout::kOut], Req, {
          mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
            s, out.Size(), out.dptr<DType>(), in.dptr<DType>());
        });
      }
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    if (!this->dropout_passthrough_) {
      this->dropout_passthrough_ = true;
      const TBlob &gdata = in_grad[dropout::kData];
      const TBlob &grad = out_grad[dropout::kOut];
      const TBlob &mask = out_data[dropout::kMask];
      CHECK_EQ(mask.type_flag_, mshadow::kUint8);
      CHECK_EQ((grad.Size() + 7) / 8, mask.Size());

      if (this->axes_.ndim() == 0) {
#if MXNET_USE_MKL_DROPOUT
        if (MKLAvailable()) {
          MKLBackward(ctx, in_grad, out_data, out_grad);
          return;
        }
#endif  // MXNET_USE_MKL_DROPOUT
#if MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
        if (CuDNNAvailable()) {
          CuDNNBackward(ctx, grad, mask, gdata);
          return;
        }
#endif  // MXNET_USE_CUDNN_DROPOUT && defined(__CUDACC__)
        // standard case for dropout
        MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
          mxnet_op::Kernel<DropoutBackwardKernel, xpu>::Launch(
              s, gdata.Size(), Req, gdata.dptr<DType>(), grad.dptr<DType>(),
              mask.dptr<uint8_t>(), pkeep_);
        })
        return;
      } else {
        TShape temp_shape = grad.shape_;
        for (int i = 0; i < this->axes_.ndim(); ++i) {
          temp_shape[this->axes_[i]] = 1;
        }
        // broardcast mul
        TShape new_lshape, new_rshape, new_oshape;
        int ndim = BinaryBroadcastShapeCompact(grad.shape_,
                                               temp_shape, gdata.shape_,
                                               &new_lshape, &new_rshape, &new_oshape);
        if (!ndim) {
          MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
            mxnet_op::Kernel<DropoutBackwardKernel, xpu>::Launch(
              s, gdata.Size(), Req, gdata.dptr<DType>(), grad.dptr<DType>(),
              mask.dptr<uint8_t >(), pkeep_);
          });
        } else {
          BROADCAST_NDIM_SWITCH(ndim, NDim, {
            mshadow::Shape<NDim> oshape = new_oshape.get<NDim>();
            mshadow::Shape<NDim> lstride = mxnet_op::calc_stride(new_lshape.get<NDim>());
            mshadow::Shape<NDim> rstride = mxnet_op::calc_stride(new_rshape.get<NDim>());
            mxnet_op::Kernel<BernoulliBackwardKernel<NDim>, xpu>::
            template LaunchEx(s, new_oshape.Size(), req[0], lstride, rstride, oshape,
            gdata.dptr<DType>(), grad.dptr<DType>(), mask.dptr<uint8_t>(), pkeep_);
          });
        }
      }
    } else {
      const TBlob& gdata = in_grad[dropout::kData];
      const TBlob& grad = out_grad[dropout::kOut];
      MXNET_ASSIGN_REQ_SWITCH(req[dropout::kData], Req, {
        mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
          s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>());
      });
    }
  }

 private:
  /*! \brief Dropout rate (keep when the generated random number is less than this value) */
  real_t pkeep_;
  /*! \brief Dropout mode */
  dropout::DropoutOpMode mode_;
  /*! \brief Axes on which dropout mask is shared in the form of broadcast multiply */
  mxnet::TShape axes_;
  /*! \brief Flag to record whether forward is executed in pass-through mode */
  bool dropout_passthrough_;
#if MXNET_USE_CUDNN_DROPOUT
  bool cudnn_off_;
  Context ctx_;
  cudnnDataType_t dtype_;
  cudnnDropoutDescriptor_t dropout_desc_;
  size_t dropout_reserve_byte_;
  cudnnTensorDescriptor_t x_desc_, y_desc_, dx_desc_, dy_desc_;
#endif  // MXNET_USE_CUDNN_DROPOUT
};  // class DropoutOp

template<typename xpu>
void DropoutCompute(const OpStatePtr& state,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropoutOp<xpu, DType>& op = state.get_state<DropoutOp<xpu, DType>>();
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void DropoutGradCompute(const OpStatePtr& state,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1);
  CHECK_EQ(req.size(), 1);
  std::vector<TBlob> out_grads(2);
  std::vector<TBlob> out_data(2);
  out_grads[dropout::kOut] = inputs[0];
  out_data[dropout::kMask] = inputs[1];

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DropoutOp<xpu, DType>& op = state.get_state<DropoutOp<xpu, DType>>();
    op.Backward(ctx, out_grads, out_data, req, outputs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_DROPOUT_INL_H_
