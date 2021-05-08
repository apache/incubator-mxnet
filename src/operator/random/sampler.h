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
 * \file sampler.h
 * \brief implementations of random sampling functors.
 */

#ifndef MXNET_OPERATOR_RANDOM_SAMPLER_H_
#define MXNET_OPERATOR_RANDOM_SAMPLER_H_

#include <algorithm>

using namespace mshadow;
using namespace mxnet::op::mxnet_op;
using namespace mxnet::common::random;

namespace mxnet {
namespace op {

/*!
 * \brief Launch a generic kernel with parallel random generator.
 * \tparam gen random generator
 * \tparam N Number of iterations
 * \tparam Args Varargs type to eventually pass to the OP::Map() function
 */
template<typename OP, typename xpu, typename GType, typename ...Args>
inline static void LaunchRNG(mshadow::Stream<xpu> *s,
                             common::random::RandGenerator<xpu, GType> *gen,
                             const index_t N, Args... args) {
  // minimal check to avoid division by zero, below.
  // if `N` is zero the map operation is a no-op in any case.
  if (N <= 0) {
    return;
  }
  int num_threads = (N + RandGenerator<xpu>::kMinNumRandomPerThread - 1) /
      RandGenerator<xpu>::kMinNumRandomPerThread;
  num_threads = std::min(num_threads, RandGenerator<xpu>::kNumRandomStates);
  index_t num_steps_per_thread = std::max<index_t>((N + num_threads - 1) / num_threads,
      RandGenerator<xpu>::kMinNumRandomPerThread);
  Kernel<OP, xpu>::Launch(s, num_threads, *gen, N, num_steps_per_thread, args...);
}

/*!
 * \brief Launch a generic kernel with parallel random generator.
 * Each thread will perform a batch of iterations sequentially.
 * \tparam gen random generator
 * \tparam N Number of iterations
 * \tparam batch_size number of iterations to be performed in a batch per thread
 * \tparam Args Varargs type to eventually pass to the OP::Map() function
 */
template<typename OP, typename xpu, typename GType, typename ...Args>
inline static void LaunchRNGBatch(mshadow::Stream<xpu> *s,
                                  common::random::RandGenerator<xpu, GType> *gen,
                                  const index_t N, const int batch_size, Args... args) {
  // minimal check to avoid division by zero, below.
  // if `N` is zero the map operation is a no-op in any case.
  if (N <= 0) {
    return;
  }
  int num_threads = (N + RandGenerator<xpu>::kMinNumRandomPerThread - 1) /
      RandGenerator<xpu>::kMinNumRandomPerThread;
  num_threads = std::min(num_threads, RandGenerator<xpu>::kNumRandomStates);
  index_t num_steps_per_thread = std::max<index_t>((N + num_threads - 1) / num_threads,
      RandGenerator<xpu>::kMinNumRandomPerThread);
  num_steps_per_thread = (num_steps_per_thread + batch_size - 1) / batch_size * batch_size;
  Kernel<OP, xpu>::Launch(s, num_threads, *gen, N, num_steps_per_thread, args...);
}

#define RNG_KERNEL_LOOP(xpu, GType, thread_id, gen, N, step, ...)        \
  const index_t start = thread_id * step;                                    \
  const index_t end = start + step;                                          \
  typename RandGenerator<xpu, GType>::Impl genImpl(&gen, thread_id);     \
  for (index_t i = start; i < end && i < N; ++i) {                           \
    {__VA_ARGS__}                                                        \
  }

template<typename xpu>
struct SampleUniformKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id, RandGenerator<xpu, OType> gen,
                                  const index_t N, const index_t step,
                                  index_t nParm, index_t nSample,
                                  const IType *lower, const IType *upper, OType *out) {
    RNG_KERNEL_LOOP(xpu, OType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(lower[i / nBatch] +
                     (upper[i / nBatch] - lower[i / nBatch]) * genImpl.uniform());
    });
  }
};

template<typename xpu>
struct UniformSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lower,
                                   const Tensor<xpu, 1, IType>& upper,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    LaunchRNG<SampleUniformKernel<xpu>, xpu>(s, pgen, out.size(0), lower.size(0), out.size(0),
                                             lower.dptr_, upper.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleRandIntKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id, RandGenerator<xpu, OType> gen,
                                  const index_t N, const index_t step,
                                  index_t nParm, index_t nSample,
                                  const IType *lower, const IType *upper, OType *out) {
    RNG_KERNEL_LOOP(xpu, OType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      if (sizeof(IType) == sizeof(int64_t))
        out[i] = OType(lower[i / nBatch] +
                     genImpl.rand_int64() % (upper[i / nBatch] - lower[i / nBatch]));
      else
        out[i] = OType(lower[i / nBatch] +
                     genImpl.rand() % (upper[i / nBatch] - lower[i / nBatch]));
    });
  }
};

template<typename xpu>
struct RandIntSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lower,
                                   const Tensor<xpu, 1, IType>& upper,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    LaunchRNG<SampleRandIntKernel<xpu>, xpu>(s, pgen, out.size(0), lower.size(0), out.size(0),
                                             lower.dptr_, upper.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleNormalKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id, RandGenerator<xpu, OType> gen,
                                  const index_t N, const index_t step,
                                  index_t nParm, index_t nSample,
                                  const IType *mean, const IType *std, OType *out) {
    RNG_KERNEL_LOOP(xpu, OType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(genImpl.normal() * std[i / nBatch] + mean[i / nBatch]);
    });
  }
};

template<typename xpu>
struct NormalSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& mean,
                                   const Tensor<xpu, 1, IType>& std,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    LaunchRNG<SampleNormalKernel<xpu>, xpu>(s, pgen, out.size(0), mean.size(0), out.size(0),
                                            mean.dptr_, std.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleExponentialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id, RandGenerator<xpu, OType> gen,
                                  const index_t N, const index_t step,
                                  index_t nParm, index_t nSample,
                                  const IType *lambda, OType *out) {
    RNG_KERNEL_LOOP(xpu, OType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(-log(1.0 - genImpl.uniform()) / lambda[i / nBatch]);
    });
  }
};

template<typename xpu>
struct ExponentialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lambda,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    LaunchRNG<SampleExponentialKernel<xpu>, xpu>(s, pgen, out.size(0),
                                                 lambda.size(0), out.size(0),
                                                 lambda.dptr_, out.dptr_);
  }
};

template<typename xpu, typename IType, typename OType>
MSHADOW_XINLINE OType SampleGamma(IType a, IType b, typename RandGenerator<xpu, OType>::Impl *gen) {
  // Generate one sample of the gamma distribution
  OType sample;
  OType d = a < 1 ? a + 2.0 / 3.0 : a - 1.0 / 3.0;
  OType k = sqrt(9.0 * d);
  OType c = 1.0 / k;
  while (1) {
    OType Z = gen->normal();
    if (Z > -k) {
      OType x = 1.0 + c * Z;
      OType V = x * x * x;
      if (log(1.0-gen->uniform()) < 0.5 * Z * Z + d * (1.0 - V + log(V))) {
        sample = d * V * b;
        break;
      }
    }
  }
  return a < 1 ? sample * pow(gen->uniform(), OType(1.0 / a)) : sample;
}

template<typename xpu>
struct SampleGammaKernel {
  template<typename IType, typename OType, typename FType>
  MSHADOW_XINLINE static void Map(index_t id, RandGenerator<xpu, FType> gen,
                                  const index_t N, const index_t step,
                                  index_t nParm, index_t nSample,
                                  const IType *alpha, const IType *beta, OType *out) {
    RNG_KERNEL_LOOP(xpu, FType, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(SampleGamma<xpu, IType, FType>(alpha[i / nBatch],
                                                    beta[i / nBatch], &genImpl));
    });
  }
};

template<typename xpu>
struct GammaSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& alpha,
                                   const Tensor<xpu, 1, IType>& beta,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    typedef typename std::conditional<std::is_floating_point<OType>::value,
                                      OType, float>::type FType;
    RandGenerator<xpu, FType> *gen = reinterpret_cast<RandGenerator<xpu, FType> *>(pgen);
    LaunchRNG<SampleGammaKernel<xpu>, xpu>(s, gen, out.size(0), alpha.size(0), out.size(0),
                                           alpha.dptr_, beta.dptr_, out.dptr_);
  }
};

template<typename xpu>
MSHADOW_XINLINE int SamplePoisson(float lambda, typename RandGenerator<xpu, float>::Impl *gen) {
  // Generate one sample of the poisson distribution. Intentionally written
  // towards a specific type (float) for internal computation which is sufficient
  // for accurate enough computation.
  if ( lambda < 12.0 ) {
    float t = expf(-lambda);
    int x = 0;
    for ( float prod = gen->uniform(); prod > t; prod *= gen->uniform() ) { x += 1; }
    return x;
  } else {
    // Approximation for high lambda according to:
    // Numerical Recipes in C: The Art of Scientific Computing
    // Cambridge University Press
    const float pi(3.1415926);
    const float sq(sqrt(2.0*lambda));
    const float loglambda(log(lambda));
    const float g(lambda*loglambda-lgammaf(lambda+1.0));
    float em(0), t(0), y(0);
    do {
      do {
        y = tanf(pi * gen->uniform());
        em = sq * y + lambda;
      } while (em < 0.0);
      em = floorf(em);
      t = 0.9 * (1.0 + y * y) * expf(em * loglambda - lgammaf(em + 1.0) - g);
    } while (gen->uniform() > t);
    return static_cast<int>(em);
  }
}

template<typename xpu>
struct SamplePoissonKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id, RandGenerator<xpu, float> gen,
                                  const index_t N, const index_t step,
                                  index_t nParm, index_t nSample,
                                  const IType *lambda, OType *out) {
    RNG_KERNEL_LOOP(xpu, float, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      out[i] = OType(SamplePoisson<xpu>(lambda[i / nBatch], &genImpl));
    });
  }
};

template<typename xpu>
struct PoissonSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lambda,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    RandGenerator<xpu, float> *gen = reinterpret_cast<RandGenerator<xpu, float> *>(pgen);
    LaunchRNG<SamplePoissonKernel<xpu>, xpu>(s, gen, out.size(0), lambda.size(0), out.size(0),
                                             lambda.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleNegativeBinomialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id, RandGenerator<xpu, float> gen,
                                  const index_t N, const index_t step,
                                  index_t nParm, index_t nSample,
                                  const IType *k, const IType *p, OType *out) {
    RNG_KERNEL_LOOP(xpu, float, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      float alpha = k[i / nBatch];
      float prob = p[i / nBatch];
      float beta = (1.0 - prob) / prob;
      float lambda = SampleGamma<xpu, IType, float>(alpha, beta, &genImpl);
      out[i] = OType(SamplePoisson<xpu>(lambda, &genImpl));
    });
  }
};

template<typename xpu>
struct NegativeBinomialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& k,
                                   const Tensor<xpu, 1, IType>& p,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    RandGenerator<xpu, float> *gen = reinterpret_cast<RandGenerator<xpu, float> *>(pgen);
    LaunchRNG<SampleNegativeBinomialKernel<xpu>, xpu>(s, gen, out.size(0), k.size(0), out.size(0),
                                                      k.dptr_, p.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleGeneralizedNegativeBinomialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(index_t id, RandGenerator<xpu, float> gen,
                                  const index_t N, const index_t step,
                                  index_t nParm, index_t nSample,
                                  const IType *mu, const IType *alpha, OType *out) {
    RNG_KERNEL_LOOP(xpu, float, id, gen, N, step, {
      index_t nBatch(1 + (nSample - 1) / nParm);
      float lambda = alpha[i / nBatch] == 0 ?
                     static_cast<float>(mu[i / nBatch]) :
                     SampleGamma<xpu, IType, float>(IType(1) / alpha[i / nBatch],
                                                    alpha[i / nBatch] * mu[i / nBatch], &genImpl);
      out[i] = OType(SamplePoisson<xpu>(lambda, &genImpl));
    });
  }
};

template<typename xpu>
struct GeneralizedNegativeBinomialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& mu,
                                   const Tensor<xpu, 1, IType>& alpha,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGenerator<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    RandGenerator<xpu, float> *gen = reinterpret_cast<RandGenerator<xpu, float> *>(pgen);
    LaunchRNG<SampleGeneralizedNegativeBinomialKernel<xpu>, xpu>(s, gen, out.size(0),
                                                                 mu.size(0), out.size(0),
                                                                 mu.dptr_, alpha.dptr_, out.dptr_);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_SAMPLER_H_
