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
 * Copyright (c) 2018 by Contributors
 * \file pdf_op.h
 * \brief Operators for computing the pdf of random distributions.
 */
#ifndef MXNET_OPERATOR_RANDOM_PDF_OP_H_
#define MXNET_OPERATOR_RANDOM_PDF_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../special_functions-inl.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

template<typename DType>
MSHADOW_XINLINE static DType ceph_psi(DType val) { return special_functions::cephes::psi(val); }
template<>
MSHADOW_XINLINE mshadow::half::half_t ceph_psi(mshadow::half::half_t val) {
    return special_functions::cephes::psi<float>(val);
}

template<bool logpdf>
struct PDF_Uniform {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size,
                                  DType *out, IType1 *sample, IType2 *lower, IType2 *upper) {
    const index_t index(start / sample_size);
    const DType l(lower[index]), h(upper[index]);
    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        // No check whether sample is in the support.
        out[i] = logpdf ? -DType(log(h - l)) : DType(1.0) / (h - l);
    }
  }
};

template<bool logpdf>
struct PDF_Uniform_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size, OpReqType req,
                  DType *out, IType1 *sample, IType2 *lower, IType2 *upper,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lower, IType2 *grad_upper) {
    const index_t index(start / sample_size);
    const DType l(lower[index]), h(upper[index]);

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
        grad_lower[i]  = scaling / (h - l);
        grad_upper[i]  = scaling / (l - h);
        KERNEL_ASSIGN(grad_sample[i], req, 0);
    }
  }
};

template<bool logpdf>
struct PDF_Normal {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size,
                                  DType *out, IType1 *sample, IType2 *loc, IType2 *scale) {
    const index_t index(start / sample_size);
    const DType u(loc[index]), s(scale[index]), sq(s * s);
    const DType normalizer(sqrt(2.0 * mxnet_op::PI) * s);

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        const DType exponent((DType(-0.5) * (x - u) * (x - u)) / (sq));
        out[i] = logpdf ? exponent - log(normalizer) : exp(exponent) / normalizer;
    }
  }
};

template<bool logpdf>
struct PDF_Normal_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size, OpReqType req,
                  DType *out, IType1 *sample, IType2 *loc, IType2 *scale,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_loc, IType2 *grad_scale) {
    const index_t index(start / sample_size);
    const DType u(loc[index]), s(scale[index]), s_squared(s * s), s_cubed(s_squared * s);

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
        grad_loc[i]    = scaling * (x - u) / s_squared;
        grad_scale[i]  = scaling * ((x - u) * (x - u) - s_squared) / s_cubed;
        KERNEL_ASSIGN(grad_sample[i], req, scaling * (u - x) / s_squared);
    }
  }
};

template<bool logpdf>
struct PDF_Gamma {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size,
                                  DType *out, IType1 *sample, IType2 *alpha, IType2 *beta) {
    const index_t index(start / sample_size);
    const DType a(alpha[index]), b(beta[index]), lgamma_a(lgamma(a)), a_log_b(a * log(b));

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        const DType lpdf(a_log_b + (a - 1) * log(x) - b * x - lgamma_a);
        out[i] = logpdf ? lpdf : DType(exp(lpdf));
    }
  }
};

template<bool logpdf>
struct PDF_Gamma_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size, OpReqType req,
                  DType *out, IType1 *sample, IType2 *alpha, IType2 *beta,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_alpha, IType2 *grad_beta) {
    const index_t index(start / sample_size);
    const DType a(alpha[index]), b(beta[index]), log_b(log(b)), ceph_psi_a(ceph_psi(a));

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
        grad_alpha[i]  = scaling * (log_b + log(x) - ceph_psi_a);
        grad_beta[i]   = scaling * (a / b - x);
        KERNEL_ASSIGN(grad_sample[i], req, scaling * ((a - 1) / x - b));
    }
  }
};

template<bool logpdf>
struct PDF_Exponential {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size,
                                  DType *out, IType1 *sample, IType2 *lambda) {
    const index_t index(start / sample_size);
    const DType l(lambda[index]), log_l(log(l));

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        out[i] = logpdf ? log_l - l * x : l * exp(-l * x);
    }
  }
};

template<bool logpdf>
struct PDF_Exponential_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size, OpReqType req,
                  DType *out, IType1 *sample, IType2 *lambda,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lambda) {
    const index_t index(start / sample_size);
    const DType l(lambda[index]);

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
        grad_lambda[i] = scaling * (DType(1) / l - x);
        KERNEL_ASSIGN(grad_sample[i], req, -scaling * l);
    }
  }
};

template<bool logpdf>
struct PDF_Poisson {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size,
                                  DType *out, IType1 *sample, IType2 *lambda) {
    const index_t index(start / sample_size);
    const DType l(lambda[index]), log_l(log(l));

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        const DType lpdf((x * log_l - lgamma(x + 1)) - l);
        out[i] = logpdf ? lpdf  : DType(exp(lpdf));
    }
  }
};

template<bool logpdf>
struct PDF_Poisson_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size, OpReqType req,
                  DType *out, IType1 *sample, IType2 *lambda,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_lambda) {
    const index_t index(start / sample_size);
    const DType l(lambda[index]);

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
        grad_lambda[i] = scaling * (x / l - DType(1));
        KERNEL_ASSIGN(grad_sample[i], req, 0);
    }
  }
};


template<bool logpdf>
struct PDF_NegativeBinomial {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size,
                                  DType *out, IType1 *sample, IType2 *limit, IType2 *prob) {
    const index_t index(start / sample_size);
    const DType l(limit[index]), p(prob[index]), lgamma_l(lgamma(l));

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType x(sample[i]);
        const DType lpdf((lgamma(x + l) - lgamma(x + 1) - lgamma_l) + l * log(p) + x * log(1 - p));
        out[i] = logpdf ? lpdf : DType(exp(lpdf));
    }
  }

  template<typename DType>
  MSHADOW_XINLINE static DType LPDF(DType l, DType p, DType x) {
    // Note that "p" is the failure and not the success probability.
    return (lgamma(x + l) - lgamma(x + 1) - lgamma(l)) + l * log(p) + x * log(1 - p);
  }
};

template<bool logpdf>
struct PDF_NegativeBinomial_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size, OpReqType req,
                  DType *out, IType1 *sample, IType2 *limit, IType2 *prob,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_limit, IType2 *grad_prob) {
    const index_t index(start / sample_size);
    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        DType grad_l(0), grad_p(0);
        LPDF_GRAD(DType(limit[index]), DType(prob[index]),
                  DType(sample[i]), out[i],
                  grad_out[i], &grad_l, &grad_p);
        grad_limit[i]  = grad_l;
        grad_prob[i]   = grad_p;
        KERNEL_ASSIGN(grad_sample[i], req, 0);
    }
  }

  template<typename DType>
  MSHADOW_XINLINE static void LPDF_GRAD(DType l, DType p, DType x, DType o, DType grad_o,
                                        DType* grad_l, DType* grad_p) {
    const DType scaling(grad_o * (logpdf ? DType(1) : o));
    *grad_l = scaling * ((ceph_psi(x + l) - ceph_psi(l)) + log(p));
    *grad_p = scaling * (l / p - x / (1 - p));
  }
};

template<bool logpdf>
struct PDF_GeneralizedNegativeBinomial {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size,
                                  DType *out, IType1 *sample, IType2 *mu, IType2 *alpha) {
    const index_t index(start / sample_size);

    // Reparameterize with limit = 1 / alpha, prob = 1 / (mu * alpha + 1)
    const DType limit(1.0 / alpha[index]), prob(1.0 / (mu[index]*alpha[index]+1.0));

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const DType lpdf(PDF_NegativeBinomial<logpdf>::LPDF(limit, prob, DType(sample[i])));
        out[i] = logpdf ? lpdf : DType(exp(lpdf));
    }
  }
};

template<bool logpdf>
struct PDF_GeneralizedNegativeBinomial_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size, OpReqType req,
                  DType *out, IType1 *sample, IType2 *mu, IType2 *alpha,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_mu, IType2 *grad_alpha) {
    const index_t index(start / sample_size);
    const DType fmu(mu[index]), falpha(alpha[index]), den(fmu * falpha + 1.0);

    // Reparameterize with limit = 1 / alpha, prob = 1 / (mu * alpha + 1)
    const DType limit(1.0 / falpha), prob(1.0 / (fmu * falpha + 1.0));

    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        // Grad returned as d_limit, d_prob
        DType grad_l(0), grad_p(0);
        PDF_NegativeBinomial_Grad<logpdf>::LPDF_GRAD(limit, prob,
            DType(sample[i]), out[i],
            grad_out[i], &grad_l, &grad_p);
        grad_mu[i]     = -grad_p * falpha / (den * den);
        grad_alpha[i]  = -grad_l / (falpha * falpha) - grad_p * fmu / (den * den);
        KERNEL_ASSIGN(grad_sample[i], req, 0);
    }
  }
};

template<bool logpdf>
struct PDF_Dirichlet {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size, index_t k,
                                  DType *out, IType1 *sample, IType2 *alpha) {
    const index_t index(start / sample_size);
    const index_t end = start + length;
    for (index_t i = start; i < end; ++i) {
        const IType1 *cur_sample = sample + i * k;
        const IType2 *cur_alpha  = alpha + index * k;
        DType sum_alpha(0), sum_lgamma(0), sum_sample(0);
        for (index_t j = 0; j < k; ++j) {
          sum_alpha  += cur_alpha[j];
          sum_lgamma += lgamma(cur_alpha[j]);
          sum_sample += (cur_alpha[j]-1) * log(cur_sample[j]);
        }
        DType lpdf(sum_sample + (lgamma(sum_alpha) - sum_lgamma));
        out[i] = logpdf ? lpdf : DType(exp(lpdf));
    }
  }
};


template<bool logpdf>
struct PDF_Dirichlet_Grad {
  template<typename DType, typename IType1, typename IType2>
  MSHADOW_XINLINE static void Map(index_t start, index_t length, index_t sample_size,
                  OpReqType req, index_t k,
                  DType *out, IType1 *sample, IType2 *alpha,
                  DType *grad_out, IType1 *grad_sample, IType2 *grad_alpha
                  ) {
    const index_t index(start / sample_size);
    const index_t end = start + length;

    for (index_t i = start; i < end; ++i) {
        // Digamma function
        const IType1 *cur_sample = sample + i * k;
        const IType2 *cur_alpha = alpha + index * k;

        const DType scaling(grad_out[i]*(logpdf ? DType(1) : out[i]));
        DType sum_alpha(0);
        for (index_t j = 0; j < k; ++j) {
          sum_alpha += cur_alpha[j];
        }
        const DType psi_sum(ceph_psi(sum_alpha));

        for (index_t j = 0; j < k; ++j) {
          size_t grad_alpha_index = i%sample_size + sample_size * (j + k * index);
          size_t grad_sample_index = i * k + j;

          // order grad_alpha differently to allow efficient reduction at the end.
          grad_alpha[grad_alpha_index] =
            scaling * (log(cur_sample[j]) + (psi_sum - ceph_psi(cur_alpha[j])));
          KERNEL_ASSIGN(grad_sample[grad_sample_index],
            req, scaling * (cur_alpha[j]-1) / cur_sample[j]);
        }
    }
  }
};

struct PdfParam : public dmlc::Parameter<PdfParam> {
  bool is_log;
  DMLC_DECLARE_PARAMETER(PdfParam) {
    DMLC_DECLARE_FIELD(is_log).set_default(false)
    .describe("If set, compute the density of the log-probability instead of the probability.");
  }
};

template<bool vparm = false>
inline bool PdfOpShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape>* in_attrs,
                       std::vector<TShape>* out_attrs) {
  CHECK_GT(in_attrs->size(), 1)
    << "pdf operator takes at least 2 arguments (" << in_attrs->size() << " given)";
  CHECK_EQ(out_attrs->size(), 1);
  // All inputs must be defined in order to infer output shape.
  if ( std::all_of((*in_attrs).begin(),
                   (*in_attrs).end(),
                   [](const TShape& s){ return s.ndim() > 0; }) ) {
    // Tensors of distribution parameters must have same shape.
    for (size_t i = 2; i < in_attrs->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*in_attrs, i, (*in_attrs)[i - 1]);
    }
    // Tensors of distribution parameters must match leftmost subshape of samples.
    CHECK_LE((*in_attrs)[1].ndim(), (*in_attrs)[0].ndim())
      << "dimension of input samples (" << (*in_attrs)[0].ndim()
      << ") must be at least dimension of distribution parameters ("<< (*in_attrs)[1].ndim() << ")";
    TShape tshape((*in_attrs)[0].begin(), (*in_attrs)[0].begin() + (*in_attrs)[1].ndim());
    if (vparm) {
      *(tshape.end() - 1) = *((*in_attrs)[0].end() - 1);
    }
    for (size_t i = 1; i < in_attrs->size(); ++i) {
      SHAPE_ASSIGN_CHECK(*in_attrs, i, tshape);
    }
    // Output shape must equal input tensor of samples except for last dimension if we are
    // dealing with samples that are itself vectors. Be aware of the special case where we
    // are dealing with a single vector sample.
    if (vparm && ((*in_attrs)[0].ndim() == 1)) {
      // Special case where we are dealing with a single vector sample.
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::Shape1(1));
    } else {
      TShape oshape((*in_attrs)[0].begin(), (*in_attrs)[0].end() - (vparm ? 1 : 0));
      SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
    }
    return true;
  }
  return false;
}

template<typename OP>
struct LaunchExWrapper {
  template<typename ...Args>
  MSHADOW_XINLINE static void Map(const index_t start, const index_t length,
                                  const index_t sample_size, Args... args) {
    // Apply the operator to the sample in strides of sample_size, so that
    // the operators can assume that their distribution parameters are constant.
    index_t i = start;

    // Get aligned
    const index_t align_step = sample_size - (i % sample_size);
    const index_t first_stride = length > align_step ? align_step : length;
    OP::Map(i, first_stride, sample_size, args...);
    i += first_stride;

    const index_t end = start + length - sample_size;
    for (; i < end; i += sample_size) {
      OP::Map(i, sample_size, sample_size, args...);
    }

    // Last stride might not be aligned either
    const index_t last_stride = start + length - i;
    if (last_stride > 0) {  // Don't overstep even if length <= sample_size
      OP::Map(i, last_stride, sample_size, args...);
    }
  }
};

template<typename xpu, typename DType, typename pdf, int pnum, bool vparm = false>
struct PdfCaller;

template<typename xpu, typename DType, typename pdf>
struct PdfCaller<xpu, DType, pdf, 1, false> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 mshadow::Stream<xpu> *s) {
    CHECK_EQ(inputs[0].Size()%inputs[1].Size(), 0);
    CHECK_EQ(inputs[0].Size()%outputs[0].Size(), 0);
    index_t num_samples(inputs[0].Size() / inputs[1].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdf>, xpu>::LaunchEx(s, outputs[0].Size(), num_samples,
                outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
  }
};

template<typename xpu, typename DType, typename pdf>
struct PdfCaller<xpu, DType, pdf, 1, true> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 mshadow::Stream<xpu> *s) {
    CHECK_EQ(inputs[0].Size()%inputs[1].Size(), 0);
    CHECK_EQ(inputs[0].Size()%outputs[0].Size(), 0);
    index_t num_samples(inputs[0].Size() / inputs[1].Size());
    index_t sample_size(inputs[0].Size() / outputs[0].Size());

    // Covers distributions parametrized by a vector of parameters (Dirichlet distribution).
    mxnet_op::Kernel<LaunchExWrapper<pdf>, xpu>::LaunchEx(s, outputs[0].Size(),
                num_samples, sample_size,
                outputs[0].dptr<DType>(), inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
  }
};

template<typename xpu, typename DType, typename pdf>
struct PdfCaller<xpu, DType, pdf, 2, false> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<TBlob>& outputs,
                 mshadow::Stream<xpu> *s) {
    CHECK_EQ(inputs[0].Size()%inputs[1].Size(), 0);
    CHECK_EQ(inputs[0].Size(), outputs[0].Size());
    index_t num_samples(inputs[0].Size() / inputs[1].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdf>, xpu>::LaunchEx(s, outputs[0].Size(), num_samples,
                outputs[0].dptr<DType>(), inputs[0].dptr<DType>(),
                inputs[1].dptr<DType>(), inputs[2].dptr<DType>());
  }
};

template<typename xpu, template<bool> class pdf, int pnum, bool vparm>
void PdfOpForward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_NE(req[0], kAddTo);
  CHECK_EQ(inputs.size(), pnum + 1);
  CHECK_EQ(outputs.size(), 1);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const PdfParam& param = nnvm::get<PdfParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    if ( param.is_log ) {
      PdfCaller<xpu, DType, pdf<true>, pnum, vparm>::op(inputs, outputs, s);
    } else {
      PdfCaller<xpu, DType, pdf<false>, pnum, vparm>::op(inputs, outputs, s);
    }
  });
}


template<typename xpu, typename DType, typename pdfgrad, int pnum, int vparm = false>
struct PdfGradCaller;

template<typename xpu, typename DType, typename pdfgrad>
struct PdfGradCaller<xpu, DType, pdfgrad, 1, false> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& grads,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(inputs[1].Size() / inputs[2].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdfgrad>, xpu>::LaunchEx(s, inputs[0].Size(),
                num_samples, req[0],
                inputs[3].dptr<DType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(),
                inputs[0].dptr<DType>(), grads[0].dptr<DType>(), grads[1].dptr<DType>());
  }
};

template<typename xpu, typename DType, typename pdfgrad>
struct PdfGradCaller<xpu, DType, pdfgrad, 1, true> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& grads,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(inputs[1].Size() / inputs[2].Size());
    index_t sample_size(inputs[1].Size() / inputs[0].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdfgrad>, xpu>::LaunchEx(s, inputs[0].Size(), num_samples,
                req[0], sample_size,
                inputs[3].dptr<DType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(),
                inputs[0].dptr<DType>(), grads[0].dptr<DType>(), grads[1].dptr<DType>());
  }
};

template<typename xpu, typename DType, typename pdfgrad>
struct PdfGradCaller<xpu, DType, pdfgrad, 2, false> {
  static void op(const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& grads,
                 mshadow::Stream<xpu> *s) {
    index_t num_samples(inputs[1].Size() / inputs[2].Size());
    mxnet_op::Kernel<LaunchExWrapper<pdfgrad>, xpu>::LaunchEx(s, inputs[0].Size(),
                num_samples, req[0],
                inputs[4].dptr<DType>(), inputs[1].dptr<DType>(), inputs[2].dptr<DType>(),
                inputs[3].dptr<DType>(), inputs[0].dptr<DType>(),
                grads[0].dptr<DType>(), grads[1].dptr<DType>(), grads[2].dptr<DType>());
  }
};

template<typename xpu, template<bool> class pdfgrad, int pnum, bool vparm>
void PdfOpBackward(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  CHECK_EQ(inputs.size(), pnum + 3);
  CHECK_EQ(outputs.size(), pnum + 1);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const PdfParam& param = nnvm::get<PdfParam>(attrs.parsed);
  const size_t N(outputs[1].Size());
  const TShape src_shape(Shape2(N, outputs[0].Size() / N)), dst_shape(Shape2(N, 1));
  const size_t red_work_size(broadcast::ReduceWorkspaceSize(
          s, dst_shape, kAddTo, src_shape));
#if !defined(__CUDACC__)
  // Inputs to PdfOpBackward: grad, samples, parm1, parm2, pdf.
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    const size_t tmp_size(outputs[0].Size() * pnum * sizeof(DType) + red_work_size);
    Tensor<xpu, 1, char> tmp_space =
            ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(tmp_size), s);
    std::vector<TBlob> grads = {outputs[0]};
    grads.push_back(TBlob(tmp_space.dptr_, outputs[0].shape_,
                          outputs[1].dev_mask(), outputs[1].type_flag_, outputs[1].dev_id()));
    if (pnum == 2) {
      grads.push_back(TBlob(tmp_space.dptr_ + outputs[0].Size() * sizeof(DType), outputs[0].shape_,
                            outputs[2].dev_mask(), outputs[2].type_flag_, outputs[2].dev_id()));
    }
    if (param.is_log) {
      PdfGradCaller<xpu, DType, pdfgrad<true>, pnum, vparm>::op(inputs, req, grads, s);
    } else {
      PdfGradCaller<xpu, DType, pdfgrad<false>, pnum, vparm>::op(inputs, req, grads, s);
    }
    Tensor<xpu, 1, char> red_work(
            tmp_space.dptr_ + pnum * outputs[0].Size() * sizeof(DType), Shape1(red_work_size), s);
    broadcast::Reduce<red::sum, 2, DType, op::mshadow_op::identity>(
       s, outputs[1].reshape(dst_shape), req[1], red_work, grads[1].reshape(src_shape));
    if (pnum == 2) {
      broadcast::Reduce<red::sum, 2, DType, op::mshadow_op::identity>(
       s, outputs[2].reshape(dst_shape), req[2], red_work, grads[2].reshape(src_shape));
    }
  });
#else
  // Inputs to PdfOpBackward: grad, samples, parm1, parm2, pdf.
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    const size_t tmp_size(outputs[0].Size() * pnum * sizeof(DType) + red_work_size);
    Tensor<xpu, 1, char> tmp_space =
            ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(tmp_size), s);
    std::vector<TBlob> grads = {outputs[0]};
    grads.push_back(TBlob(tmp_space.dptr_, outputs[0].shape_,
                          outputs[1].dev_mask(), outputs[1].type_flag_, outputs[1].dev_id()));
    if (pnum == 2) {
      grads.push_back(TBlob(tmp_space.dptr_ + outputs[0].Size() * sizeof(DType), outputs[0].shape_,
                            outputs[2].dev_mask(), outputs[2].type_flag_, outputs[2].dev_id()));
    }
    if (param.is_log) {
      PdfGradCaller<xpu, DType, pdfgrad<true>, pnum, vparm>::op(inputs, req, grads, s);
    } else {
      PdfGradCaller<xpu, DType, pdfgrad<false>, pnum, vparm>::op(inputs, req, grads, s);
    }
    Tensor<xpu, 1, char> red_work(
            tmp_space.dptr_ + pnum * outputs[0].Size() * sizeof(DType), Shape1(red_work_size), s);
    broadcast::RTCReduce(ctx, outputs[1].reshape(dst_shape), req[1], red_work,
                         grads[1].reshape(src_shape), "red::sum{}", 2, "identity");
    if (pnum == 2) {
      broadcast::RTCReduce(ctx, outputs[2].reshape(dst_shape), req[2], red_work,
                           grads[2].reshape(src_shape), "red::sum{}", 2, "identity");
    }
  });

#endif
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_PDF_OP_H_
