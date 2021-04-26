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

#ifndef MXNET_COMMON_CUDA_RTC_REDUCER_INL_H_
#define MXNET_COMMON_CUDA_RTC_REDUCER_INL_H_

#if MXNET_USE_CUDA

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {


const char reducer[] = R"code(
namespace red {

struct sum {
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst,  volatile DType2 src) {
    dst = op::add(dst, src);
  }

  /*! \brief do stable reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst,  volatile DType2 src,
                                       volatile DType& residual) {
    DType y = op::sub(src, residual);
    DType t = dst + y;
    if (util::isinf(t)) {
      residual = 0;
    } else {
      residual = (t - dst) - y;
    }
    dst = t;
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& dst_residual,
                                      volatile DType& src_val, volatile DType& src_residual) {
    DType t1 = dst_val + src_val;
    if (util::isinf(t1)) {
      dst_val = t1;
      dst_residual = 0;
    } else {
      DType e = t1 - dst_val;
      DType t2 = ((src_val - e) + (dst_val - (t1 - e))) + dst_residual + src_residual;
      dst_val = t1 + t2;
      dst_residual = t2 - (dst_val - t1);
    }
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType& none) {}
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv) {
    initv = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &residual) {
    SetInitValue(initv);
    residual = 0;
  }
};

struct product {
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst, volatile DType2 src) {
    dst = op::mul(dst, src);
  }
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst, volatile DType2 src,
                                       volatile DType& none) {
    Reduce(dst, src);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& dst_residual,
                                      volatile DType& src_val, volatile DType& src_residual) {
    Reduce(dst_val, src_val);
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType& none) {}
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv) {
    initv = 1;
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &none) {
    SetInitValue(initv);
  }
};

struct nansum {
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst, volatile DType2 src) {
    if (util::isnan(src)) return;
    dst = op::add(dst, src);
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  __device__ inline static void Reduce(volatile DType& dst, volatile DType src,
                                       volatile DType& residual) {
    if (util::isnan(src)) return;
    DType y = src - residual;
    DType t = dst + y;
    residual = (t - dst) - y;
    dst = t;
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& dst_residual,
                                      volatile DType& src_val, volatile DType& src_residual) {
    DType t1 = dst_val + src_val;
    DType e = t1 - src_val;
    DType t2 = ((src_val - e) + (dst_val - (t1 - e))) + dst_residual + src_residual;
    dst_val = t1 + t2;
    dst_residual = t2 - (dst_val - t1);
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType& none) {}
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  __device__ inline static void SetInitValue(DType & initv) {
      initv = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &residual) {
    SetInitValue(initv);
    residual = 0;
  }
};

struct nanprod {
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst, volatile DType2 src) {
    if (util::isnan(src)) return;
    dst = op::mul(dst, src);
  }
  /*! \brief do reduction into dst */
  template<typename DType>
  __device__ inline static void Reduce(volatile DType& dst, volatile DType src,
                                       volatile DType& none) {
    Reduce(dst, src);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& dst_residual,
                                      volatile DType& src_val, volatile DType& src_residual) {
    Reduce(dst_val, src_val);
  }
  /*! \brief finalize reduction */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType& none) {}
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  __device__ inline static void SetInitValue(DType & initv) {
    initv = 1;
  }
  /*!
  *\brief set the initial value during reduction
  */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &none) {
    SetInitValue(initv);
  }
};

struct nrm2 {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  __device__ inline static void Reduce(volatile AType& sum_of_squares, volatile DType src) {
    sum_of_squares = op::add(sum_of_square, src * src);
  }
  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  __device__ inline static void Reduce(volatile AType& sum_of_squares,
                                       volatile DType src, volatile DType& scale) {
    if (src != 0) {
      DType abs = op::abs(src);
      if (scale < abs) {
        sum_of_squares = 1 + sum_of_squares * (scale / abs) * (scale / abs);
        scale = abs;
      } else {
        sum_of_squares = sum_of_squares + (abs / scale) * (abs / scale);
      }
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    dst_val = op::add(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_ssq, volatile DType& dst_scale,
                                      volatile DType& src_ssq, volatile DType& src_scale) {
    if (dst_scale != 0 && dst_scale >= src_scale) {
      dst_ssq = dst_ssq + src_ssq * (src_scale / dst_scale) * (src_scale / dst_scale);
    } else if (src_scale != 0 && dst_scale < src_scale) {
      dst_ssq = src_ssq + dst_ssq * (dst_scale / src_scale) * (dst_scale / src_scale);
      dst_scale = src_scale;
    }
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& sum_of_squares) {
    sum_of_squares = op::sqrt(sum_of_squares);
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& sum_of_squares, volatile DType& scale) {
    sum_of_squares = scale * op::sqrt(sum_of_squares);
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &sum_of_squares) {
    sum_of_squares = 0;
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &sum_of_squares, DType &scale) {
    SetInitValue(sum_of_squares);
    scale = 0;
  }
};

struct nrmlp {
  double lp;
  /* \brief power for Lp norm */
  __device__ inline static double lp_power(volatile double src, volatile double p) {
    if (p != 0.0) {
      if (src == 0.0) {
        return src;
      } else {
        return op::power(src, p);
      }
    } else {  // 0-norm, sparsity
      return static_cast<double>(src != 0);
    }
  }

  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  __device__ inline void Reduce(volatile AType& sum_of_powers, volatile DType src) {
    if (src != 0) {
      sum_of_powers += AType(lp_power(static_cast<double>(src), lp));
    }
  }

  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  __device__ inline void Reduce(volatile AType& sum_of_powers, volatile DType src,
                                volatile DType& scale) {
    if (src != 0) {
      DType src_abs = op::abs(src);
      if (scale < src_abs) {
        sum_of_powers = sum_of_powers * AType(lp_power(static_cast<double>(scale / src_abs), lp));
        sum_of_powers = sum_of_powers + 1;
        scale = src_abs;
      } else {
        sum_of_powers = sum_of_powers + AType(lp_power(static_cast<double>(src_abs / scale), lp));
      }
    }
  }

  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    dst_val = dst_val + src_val;
  }

  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_ssq, volatile DType& dst_scale,
                                      volatile DType& src_ssq, volatile DType& src_scale) {
    if (dst_scale != 0 && dst_scale >= src_scale) {
      dst_ssq = dst_ssq + src_ssq * DType(lp_power(static_cast<double>(src_scale / dst_scale), 2));
    } else if (src_scale != 0 && dst_scale < src_scale) {
      dst_ssq = src_ssq + dst_ssq * DType(lp_power(static_cast<double>(dst_scale / src_scale), 2));
      dst_scale = src_scale;
    }
  }

  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline void Finalize(volatile DType& sum_of_powers) {
    if (lp != 0.0) {
      sum_of_powers = DType(lp_power(static_cast<double>(sum_of_powers), 1.0 / lp));
    }
  }

  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline void Finalize(volatile DType& sum_of_powers, volatile DType& scale) {
    if (lp != 0.0) {
      sum_of_powers = scale * DType(lp_power(static_cast<double>(sum_of_powers), 1.0 / lp));
    }
  }

  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &sum_of_powers) {
    sum_of_powers = 0;
  }

  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &sum_of_powers, DType &scale) {
    SetInitValue(sum_of_powers);
    scale = 0;
  }
};

}  // namespace red
)code";

const char logic_reducer[] = R"code(
namespace red {

struct maximum {
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst,  volatile DType2 src) { // NOLINT(*)
    if (!util::isnan(dst)) {
      if (!(dst >= src)) dst = src;
    }
  }
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst,  volatile DType2 src,
                                       volatile DType& none) {
    Reduce(dst, src);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& dst_residual,
                                      volatile DType& src_val, volatile DType& src_residual) {
    Reduce(dst_val, src_val);
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType& none) {}
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv) {
    initv = limits::NegInfValue<DType>();
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &none) {
    SetInitValue(initv);
  }
};

struct minimum {
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst,  volatile DType2 src) {
    if (!util::isnan(dst)) {
      if (!(dst <= src)) dst = src;
    }
  }
  /*! \brief do reduction into dst */
  template<typename DType, typename DType2>
  __device__ inline static void Reduce(volatile DType& dst,  volatile DType2 src,
                                       volatile DType& none) {
    Reduce(dst, src);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& src_val) {
    Reduce(dst_val, src_val);
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst_val, volatile DType& dst_residual,
                                      volatile DType& src_val, volatile DType& src_residual) {
    Reduce(dst_val, src_val);
  }
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction result */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType& none) {}
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv) {
    initv = limits::PosInfValue<DType>();
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &none) {
    SetInitValue(initv);
  }
};

struct argmax {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  __device__ inline static void Reduce(volatile AType& dst,  volatile DType src) {
    if (dst.num < src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  __device__ inline static void Reduce(volatile AType& dst,  volatile DType src,
                                       volatile DType&) {
    if (dst.num < src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst, volatile DType& src) {
    if (dst.num < src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst, volatile DType&,
                                      volatile DType& src, volatile DType&) {
    if (dst.num < src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief finalize reduction */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType&) {}
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv) {
    initv.num = limits::NegInfValue<decltype(initv.num)>();
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &) {
    initv.num = limits::NegInfValue<decltype(initv.num)>();
  }
};

struct argmin {
  /*! \brief do reduction into dst */
  template<typename AType, typename DType>
  __device__ inline static void Reduce(volatile AType& dst,  volatile DType src) {
    if (dst.num > src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief do stable reduction into dst */
  template<typename AType, typename DType>
  __device__ inline static void Reduce(volatile AType& dst,  volatile DType src,
                                       volatile DType& residual) {
    if (dst.num > src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst, volatile DType& src) {
    if (dst.num > src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief combine the results of two reducers */
  template<typename DType>
  __device__ inline static void Merge(volatile DType& dst, volatile DType&,
                                      volatile DType& src, volatile DType&) {
    if (dst.num > src.num || (dst.num == src.num && dst.idx > src.idx)) {
      dst.num = src.num;
      dst.idx = src.idx;
    }
  }
  /*! \brief finalize reduction */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst) {}
  /*! \brief finalize reduction */
  template<typename DType>
  __device__ inline static void Finalize(volatile DType& dst, volatile DType& residual) {}
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv) {
    initv.num = limits::PosInfValue<decltype(initv.num)>();
  }
  /*!
   *\brief set the initial value during reduction
   */
  template<typename DType>
  __device__ inline static void SetInitValue(DType &initv, DType &residual) {
    initv.num = limits::PosInfValue<decltype(initv.num)>();
  }
};
}  // namespace red
)code";
}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_REDUCER_INL_H_

