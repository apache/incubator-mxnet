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

#ifndef MXNET_COMMON_CUDA_RTC_SPECIAL_FUNCTIONS_INL_H_
#define MXNET_COMMON_CUDA_RTC_SPECIAL_FUNCTIONS_INL_H_

#include <cfloat>
#include <string>

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

// This code is based on the Cephes Library availible at http://www.netlib.org/cephes
// The original author, Stephen Moshier, has kindly given permission to use this code
// in mxnet.  (See email below).
//
//     Date: Tue, 13 Sep 2016 09:28:20 -0400
//     From: Stephen Moshier
//     To: Flunkert, Valentin
//     Subject: Re: cephes code in mxnet
//
//     Hello Valentin,
//
//     Thank you for writing.  You are welcome to use and modify the Cephes code
//     and distribute it under the Apache license.
//
//     Good luck with your project,
//     Steve Moshier
//
// Cephes Math Library Release 2.2:  June, 1992
// Copyright 1984, 1987, 1992 by Stephen L. Moshier
// Direct inquiries to 30 Frost Street, Cambridge, MA 02140
//
const char special_functions_definitions[] = R"code(
namespace op {

namespace special_functions {

template<typename DType>
__device__ inline static DType trigamma(DType x);

template<>
__device__ inline double trigamma<double>(double x) {
  double PI(3.14159265358979323846);
  double sign = +1;
  double result = 0;
  if (x < 0.5) {
    sign = -1;
    const double sin_pi_x = sin(PI * x);
    result -= (PI * PI) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const double ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (1./6 - ixx * (1./30 - ixx * (1./42)))) / x;
  return sign * result;
}

template<>
__device__ inline float trigamma<float>(float x) {
  float PI(3.14159265358979323846);
  float sign = +1;
  float result = 0;
  if (x < 0.5f) {
    sign = -1;
    const float sin_pi_x = sinf(PI * x);
    result -= (PI * PI) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const float ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (1.f/6 - ixx * (1.f/30 - ixx * (1.f/42)))) / x;
  return sign * result;
}

struct cephes {
  /*
   * Helper to evaluate a polynomial given an array of coefficients.
   */
  template <typename DType>
  __device__ inline static DType polevl(DType x, const DType coef[], int N) {
    DType ans;
    DType const *p;
    int i;

    p = coef;
    ans = *p++;

    i = N;
    do {
      ans = ans * x  +  *p++;
    } while ( --i );

    return( ans );
  }


  /*
   * Helper function for psi that handles double/float specific differences
   * in the algorithm.
   */
  template<typename DType>
  __device__ inline static DType psi_helper(DType s);

  /*
   *
   *	Psi (digamma) function
   *
   *
   * SYNOPSIS:
   *
   * float x, y, psif();
   *
   * y = psif( x );
   *
   *
   * DESCRIPTION:
   *
   *              d      -
   *   psi(x)  =  -- ln | (x)
   *              dx
   *
   * is the logarithmic derivative of the gamma function.
   * For integer x,
   *                   n-1
   *                    -
   * psi(n) = -EUL  +   >  1/k.
   *                    -
   *                   k=1
   *
   * This formula is used for 0 < n <= 10.  If x is negative, it
   * is transformed to a positive argument by the reflection
   * formula  psi(1-x) = psi(x) + pi cot(pi x).
   * For general positive x, the argument is made greater than 10
   * using the recurrence  psi(x+1) = psi(x) + 1/x.
   * Then the following asymptotic expansion is applied:
   *
   *                           inf.   B
   *                            -      2k
   * psi(x) = log(x) - 1/2x -   >   -------
   *                            -        2k
   *                           k=1   2k x
   *
   * where the B2k are Bernoulli numbers.
   *
   * ACCURACY:
   *    Absolute error,  relative when |psi| > 1 :
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      -33,0        30000      8.2e-7      1.2e-7
   *    IEEE      0,33        100000      7.3e-7      7.7e-8
   *
   * ERROR MESSAGES:
   *     message         condition      value returned
   * psi singularity    x integer <=0      MAXNUMF
   */
  template<typename DType>
  __device__ inline static DType psi(DType x) {
    DType p, q, nz, s, w, y;
    int i, n, negative;

    DType EUL(0.57721566490153286061);
    DType PI(3.14159265358979323846);

    negative = 0;
    nz = 0.0;

    if ( x <= 0.0 ) {
      negative = 1;
      q = x;
      p = ::floor(q);
      if ( p == q ) {
        return DBL_MAX;
      }
      /* Remove the zeros of tan(PI x)
       * by subtracting the nearest integer from x
       */
      nz = q - p;
      if ( nz != 0.5 ) {
        if ( nz > 0.5 ) {
          p += 1.0;
          nz = q - p;
        }
        nz = PI/::tan(PI*nz);
      } else {
        nz = 0.0;
      }
      x = 1.0 - x;
    }

    /* check for positive integer up to 10 */
    if ( (x <= 10.0) && (x == ::floor(x)) ) {
      y = 0.0;
      n = x;
      for ( i = 1; i < n; i++ ) {
        w = i;
        y += 1.0/w;
      }
      y -= EUL;
      goto done;
    }

    s = x;
    w = 0.0;
    while ( s < 10.0 ) {
      w += 1.0/s;
      s += 1.0;
    }

    y = psi_helper(s);

    y = logf(s)  -  (0.5/s)  -  y  -  w;

done:

    if ( negative ) {
      y -= nz;
    }

    return(y);
  }
};


template<>
__device__ inline double cephes::psi_helper<double>(double s) {
  double z;
  const double A[] = {
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2
  };

  if ( s < 1.0e17 ) {
    z = 1.0/(s * s);
    return z * cephes::polevl<double>(z, A, 6);
  } else {
    return 0.0;
  }
}

template<>
__device__ inline float cephes::psi_helper<float>(float s) {
  float z;
  const float A[] = {
    -4.16666666666666666667E-3f,
    3.96825396825396825397E-3f,
    -8.33333333333333333333E-3f,
    8.33333333333333333333E-2f
  };

  if ( s < 1.0e8 ) {
    z = 1.0/(s * s);
    return z * cephes::polevl<float>(z, A, 3);
  } else {
    return 0.0;
  }
}
}  // namespace special_functions
}  // namespace op
)code";

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_COMMON_CUDA_RTC_SPECIAL_FUNCTIONS_INL_H_
