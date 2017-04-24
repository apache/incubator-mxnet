/*!
 * Copyright (c) 2017 by Contributors
 * \file batch_norm.cu
 * \brief CUDA Batch Normalization code
 * \author Chris Olivier, Bing Xu
 * Adapted from Torch
*/

#include "batch_norm-inl.h"

#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn_batch_norm-inl.h"
#endif

#include <mshadow/cuda/tensor_gpu-inl.cuh>
#include "../common/cuda_utils.h"

namespace mxnet {
namespace op {
namespace cuda {

static const unsigned WARP_SIZE = 32;

// The maximum number of threads in a block
static const unsigned MAX_BLOCK_SIZE = 512;

template<typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ __forceinline__ Out to(const In v) { return (Out) v; }
};

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static unsigned getNumThreads(int nElem) {
  unsigned threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template<typename DType, typename Acctype>
struct Float2 {
  Acctype v1, v2;
  __device__ Float2() {}
  __device__ Float2(DType v1, DType v2)
    : v1(ScalarConvert<DType, Acctype>::to(v1))
      , v2(ScalarConvert<DType, Acctype>::to(v2)) {}
  __device__ Float2(DType v)
    : v1(ScalarConvert<DType, Acctype>::to(v))
      , v2(ScalarConvert<DType, Acctype>::to(v)) {}
  __device__ Float2(int v)
    : v1(ScalarConvert<int, Acctype>::to(v))
      , v2(ScalarConvert<int, Acctype>::to(v)) {}
  __device__ Float2 &operator+=(const Float2 &a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template<typename DType, typename Acctype, typename DeviceTensor3>
struct SumOp {
  __device__ SumOp(const DeviceTensor3 t) : tensor(t) {}
  __device__ __forceinline__ Acctype operator()(int batch, int plane, int n) {
    return ScalarConvert<DType, Acctype>::to(tensor(batch, plane, n));
  }
  const DeviceTensor3 tensor;
};

template<typename DType, typename Acctype, typename DeviceTensor3>
struct VarOp {
  __device__ VarOp(Acctype m, const DeviceTensor3 t)
    : mean(m)
      , tensor(t) {
  }
  __device__ __forceinline__ Acctype operator()(int batch, int plane, int n) {
    DType val = tensor(batch, plane, n);
    return (val - mean) * (val - mean);
  }
  const Acctype mean;
  const DeviceTensor3 tensor;
};

template<typename DType, typename Acctype, typename DeviceTensor3>
struct GradOp {
  __device__ GradOp(Acctype m, const DeviceTensor3 i, const DeviceTensor3 g)
    : mean(m), input(i), gradOutput(g) {}
  __device__ __forceinline__ Float2<DType, Acctype> operator()(int batch, int plane, int n) {
    DType g = gradOutput(batch, plane, n);
    DType c = ScalarConvert<Acctype, DType>::to(input(batch, plane, n) - mean);
    return Float2<DType, Acctype>(g, g * c);
  }
  const Acctype mean;
  const DeviceTensor3 input;
  const DeviceTensor3 gradOutput;
};

// Sum across all threads within a warp
template<typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += __shfl_xor(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ T values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

template<typename DType, typename Acctype>
static __device__ __forceinline__ Float2<DType, Acctype> warpSum(Float2<DType, Acctype> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
template<typename T, typename Op, typename DeviceTensor3>
static __device__ T reduce(Op op, DeviceTensor3 tensor, int plane) {
  T sum = (T) 0;
  for (int batch = 0; batch < tensor.getSize(0); ++batch) {
    for (int x = threadIdx.x; x < tensor.getSize(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T) 0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}

template <typename DType, typename Acctype, typename DeviceTensor1, typename DeviceTensor3>
__global__ void BatchNormalizationUpdateOutputInferenceKernel(
  DeviceTensor3 input,
  DeviceTensor3 output,
  DeviceTensor1 runningMean,
  DeviceTensor1 runningVar,
  DeviceTensor1 saveMean,
  DeviceTensor1 saveStd,
  DeviceTensor1 weight,
  DeviceTensor1 bias,
  DType epsilon) {
  int plane = blockIdx.x;

  Acctype invstd = Acctype(1) / sqrt(runningVar[plane] + epsilon);
  Acctype mean = ScalarConvert<DType, Acctype>::to(runningMean[plane]);
  Acctype gamma = weight.numElements() > 0 ? ScalarConvert<DType, Acctype>::to(weight[plane])
                                           : Acctype(1);
  Acctype beta = bias.numElements() > 0 ? ScalarConvert<DType, Acctype>::to(bias[plane])
                                        : Acctype(0);
  if (threadIdx.x == 0) {
    saveMean[plane] = runningMean[plane];
    saveStd[plane]  = runningVar[plane];
  }
  // Write normalized and update the output
  for (int batch = 0, nbatch = input.getSize(0); batch < nbatch; ++batch) {
    for (int x = threadIdx.x, nx = input.getSize(2); x < nx; x += blockDim.x) {
      const DType inp = input(batch, plane, x);
      output(batch, plane, x) =
        ScalarConvert<Acctype, DType>::to(gamma * (inp - mean) * invstd + beta);
    }
  }
}

template <typename DType, typename Acctype, typename DeviceTensor1, typename DeviceTensor3>
__global__ void BatchNormalizationUpdateOutputKernel(
  DeviceTensor3 input,
  DeviceTensor3 output,
  DeviceTensor1 weight,
  DeviceTensor1 bias,
  Acctype epsilon,
  Acctype momentum,
  DeviceTensor1 runningMean,
  DeviceTensor1 runningVar,
  DeviceTensor1 saveMean,
  DeviceTensor1 saveStd) {
  int plane = blockIdx.x;
  int N = input.getSize(0) * input.getSize(2);

  Acctype norm = Acctype(1) / N;

  // Compute the mean and variance across (batch, x/y/z)
  Acctype mean = reduce<Acctype>(SumOp<DType, Acctype, DeviceTensor3>(input), input, plane) * norm;
  __syncthreads();
  Acctype varN = reduce<Acctype>(VarOp<DType, Acctype, DeviceTensor3>(mean, input), input, plane);
  Acctype invStd = 0;
  if (varN != Acctype(0) || epsilon != Acctype(0)) {
    invStd = 1.0 / sqrt(varN * norm + epsilon);
  }

  // Save the mean, variance, and moving averages
  if (threadIdx.x == 0) {
    // For one item (0th) per plane (channel), write the per-channel data (ie mean, variance, etc)
    // Momentum based writeback
    saveMean[plane] = ScalarConvert<Acctype, DType>::to(mean);
    saveStd[plane] = ScalarConvert<Acctype, DType>::to(INVSTD_TO_VARIANCE(invStd, epsilon));
  }

  // Write normalized and update the output
  Acctype gamma = weight.numElements() > 0 ? ScalarConvert<DType, Acctype>::to(weight[plane])
                                           : ScalarConvert<int, Acctype>::to(1);
  Acctype beta = bias.numElements() > 0 ? ScalarConvert<DType, Acctype>::to(bias[plane])
                                        : ScalarConvert<int, Acctype>::to(0);
  for (int batch = 0, nbatch = input.getSize(0); batch < nbatch; ++batch) {
    for (int x = threadIdx.x, nx = input.getSize(2); x < nx; x += blockDim.x) {
      const DType inp = input(batch, plane, x);
      output(batch, plane, x) =
        ScalarConvert<Acctype, DType>::to(gamma * (inp - mean) * invStd + beta);
    }
  }
}

template <typename DType, typename Acctype, typename DeviceTensor1, typename DeviceTensor3>
static __global__ void BatchNormalizationBackwardKernel(
  const DeviceTensor3 input,
  const DeviceTensor3 gradOutput,
  DeviceTensor3 gradInput,
  DeviceTensor1 gradWeight,
  DeviceTensor1 gradBias,
  const DeviceTensor1 weight,
  const DeviceTensor1 runningMean,
  const DeviceTensor1 runningVar,
  const DeviceTensor1 saveMean,
  const DeviceTensor1 saveStd,
  const bool train,
  const bool fix_gamma,
  Acctype scale,
  Acctype momentum,
  double eps) {
  int plane = blockIdx.x;
  int N = gradOutput.getSize(0) * gradOutput.getSize(2);

  Acctype mean, stdVal;
  if (train) {
    mean = ScalarConvert<DType, Acctype>::to(saveMean[plane]);
    stdVal = ScalarConvert<DType, Acctype>::to(VARIANCE_TO_INVSTD(saveStd[plane], eps));
  } else {
    mean = ScalarConvert<DType, Acctype>::to(runningMean[plane]);
    stdVal = 1 / sqrt(runningVar[plane] + eps);
  }

  Acctype weightVal = weight.numElements() > 0 ?
                      ScalarConvert<DType, Acctype>::to(weight[plane]) : Acctype(1);
  Acctype norm = Acctype(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(gradOutput)
  // 2. DotProduct(input - mean, gradOutput)
  GradOp<DType, Acctype, DeviceTensor3> g(mean, input, gradOutput);
  Float2<DType, Acctype> res = reduce<Float2<DType, Acctype>,
    GradOp<DType, Acctype, DeviceTensor3>, DeviceTensor3>(g, gradOutput, plane);
  Acctype gradOutputSum = res.v1;
  Acctype dotP = res.v2;

  Acctype gradMean = gradOutputSum * norm;
  Acctype projScale = dotP * norm * stdVal * stdVal;
  Acctype gradScale = stdVal * weightVal;

  if (threadIdx.x == 0 && train) {
    const DType variance = saveStd[plane];
    const DType   mean   = saveMean[plane];

    // update running averages
    runningMean[plane] = runningMean[plane] * momentum + mean * (DType(1) - momentum);
    runningVar[plane] = runningVar[plane] * momentum + variance * (DType(1) - momentum);
  }

  if (gradInput.numElements() > 0) {
    for (int batch = 0, nbatch = gradOutput.getSize(0); batch < nbatch; ++batch) {
      for (int x = threadIdx.x, nx = gradOutput.getSize(2); x < nx; x += blockDim.x) {
        DType gradOut = gradOutput(batch, plane, x);
        if (train) {
          DType inp = input(batch, plane, x);
          Acctype proj = (inp - mean) * projScale;
          gradInput(batch, plane, x) =
            ScalarConvert<Acctype, DType>::to((gradOut - proj - gradMean) * gradScale);
        } else {
          gradInput(batch, plane, x) = ScalarConvert<Acctype, DType>::to(gradOut * gradScale);
        }
      }
    }
  }

  if (gradWeight.numElements() > 0) {
    if (threadIdx.x == 0) {
      if (!fix_gamma) {
        gradWeight[plane] += ScalarConvert<Acctype, DType>::to(scale * dotP * stdVal);
      } else {
        gradWeight[plane] += DType(0);
      }
    }
  }

  if (gradBias.numElements() > 0) {
    if (threadIdx.x == 0) {
      gradBias[plane] += ScalarConvert<Acctype, DType>::to(scale * gradOutputSum);
    }
  }
}

template<typename DType, int Dim>
struct DeviceTensor {
 public:
  inline DeviceTensor(DType *p, const int *size)
    : dptr_(p) {
    for (int i = 0; i < Dim; ++i) {
      size_[i] = size ? size[i] : 0;
    }
  }

  __host__ __device__ __forceinline__ unsigned getSize(const int i) const {
    return size_[i];
  }

  __host__ __device__ __forceinline__ int numElements() const {
    int n = 1;
    for (int i = 0; i < Dim; ++i) {
      n *= size_[i];
    }
    return n;
  }

  __host__ __device__ __forceinline__ DType& operator ()(const size_t batch,
                                                         const size_t plane,
                                                         const size_t x) const {
    int offset = 0;

    offset *= size_[0];
    offset += batch;

    offset *= size_[1];
    offset += plane;

    offset *= size_[2];
    offset += x;

    return *(const_cast<DType *>(dptr_ + offset));
  }

  __host__ __device__ __forceinline__ DType& operator[](const size_t x) const {
    return *(dptr_ + x);
  }

  DType *dptr_;
  int size_[Dim];
};

template <typename DType, int Dim>
static DeviceTensor<DType, Dim> devicetensor(const TBlob& blob) {
  DType *data = blob.dptr<DType>();
  const int inDim = blob.shape_.ndim();
  if (inDim == Dim) {
    DeviceTensor<DType, Dim> tensor(data, nullptr);
    for (int i = 0; i < Dim; ++i) {
      tensor.size_[i] = blob.size(i);
    }
    return tensor;
  }

  // View in which the last dimensions are collapsed or expanded as needed
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = blob.size(i);
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= blob.size(i);
    }
  }
  return DeviceTensor<DType, Dim>(data, &size[0]);
}

#define DeviceTensor1 DeviceTensor<DType, 1>
#define DeviceTensor3 DeviceTensor<DType, 3>

template <typename DType, typename accreal>
static void BatchNormalizationUpdateOutput(mshadow::Stream<gpu> *s,
                                            const OpContext &ctx,
                                            const std::vector<TBlob> &in_data,
                                            const std::vector<OpReqType> &req,
                                            const std::vector<TBlob> &out_data,
                                            const std::vector<TBlob> &aux_states,
                                            bool train,
                                            const bool fix_gamma,
                                            double momentum,
                                            double eps) {
  DeviceTensor3 input = devicetensor<DType, 3>(in_data[batchnorm::kData]);
  DeviceTensor3 output = devicetensor<DType, 3>(out_data[batchnorm::kOut]);
  DeviceTensor1 weight = fix_gamma ? DeviceTensor1(nullptr, nullptr)
                                   : devicetensor<DType, 1>(in_data[batchnorm::kGamma]);
  DeviceTensor1 bias = devicetensor<DType, 1>(in_data[batchnorm::kBeta]);
  DeviceTensor1 runningMean = devicetensor<DType, 1>(aux_states[batchnorm::kMovingMean]);
  DeviceTensor1 runningVar = devicetensor<DType, 1>(aux_states[batchnorm::kMovingVar]);
  DeviceTensor1 saveMean = devicetensor<DType, 1>(out_data[batchnorm::kMean]);
  DeviceTensor1 saveStd = devicetensor<DType, 1>(out_data[batchnorm::kVar]);

  DCHECK(!fix_gamma || weight.numElements() == 0);

  if (!train) {
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    BatchNormalizationUpdateOutputInferenceKernel<DType, accreal, DeviceTensor1, DeviceTensor3>
      <<<blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
      input, output, runningMean, runningVar, saveMean, saveStd, weight, bias, eps);
  } else {
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    BatchNormalizationUpdateOutputKernel<DType, accreal, DeviceTensor1, DeviceTensor3>
      <<<blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
      input, output, weight, bias, eps, momentum, runningMean, runningVar,
        saveMean, saveStd);
  }
  MSHADOW_CUDA_POST_KERNEL_CHECK(BatchNormalizationUpdateOutput);
}

template<typename DType, typename accreal>
static void BatchNormalizationBackward(mshadow::Stream<gpu> *s,
                                        const OpContext &ctx,
                                        const std::vector<TBlob> &out_grad,
                                        const std::vector<TBlob> &in_data,
                                        const std::vector<TBlob> &out_data,
                                        const std::vector<OpReqType> &req,
                                        const std::vector<TBlob> &in_grad,
                                        const std::vector<TBlob> &aux_states,
                                        const bool train,
                                        const bool fix_gamma,
                                        const DType scale,
                                        double momentum,
                                        double eps) {
  DeviceTensor3 input = devicetensor<DType, 3>(in_data[batchnorm::kData]);
  DeviceTensor3 gradOutput = devicetensor<DType, 3>(out_grad[batchnorm::kOut]);
  DeviceTensor3 gradInput = devicetensor<DType, 3>(in_grad[batchnorm::kData]);
  DeviceTensor1 gradWeight = devicetensor<DType, 1>(in_grad[batchnorm::kGamma]);
  DeviceTensor1 gradBias = devicetensor<DType, 1>(in_grad[batchnorm::kBeta]);
  DeviceTensor1 weight = fix_gamma ? DeviceTensor1(nullptr, nullptr)
                                   : devicetensor<DType, 1>(in_data[batchnorm::kGamma]);
  DeviceTensor1 runningMean = devicetensor<DType, 1>(aux_states[batchnorm::kMovingMean]);
  DeviceTensor1 runningVar = devicetensor<DType, 1>(aux_states[batchnorm::kMovingVar]);
  DeviceTensor1 saveMean = devicetensor<DType, 1>(out_data[batchnorm::kMean]);
  DeviceTensor1 saveVar = devicetensor<DType, 1>(out_data[batchnorm::kVar]);

  DCHECK(!fix_gamma || weight.numElements() == 0);

  dim3 blocks(gradOutput.getSize(1));
  dim3 threads(getNumThreads(gradOutput.getSize(2)));
  BatchNormalizationBackwardKernel<DType,  accreal,  DeviceTensor1, DeviceTensor3>
    <<<blocks, threads, 0, mshadow::Stream<gpu>::GetStream(s)>>>(
    input, gradOutput, gradInput, gradWeight, gradBias, weight, runningMean, runningVar,
      saveMean, saveVar, train, fix_gamma, scale, momentum, eps);
  MSHADOW_CUDA_POST_KERNEL_CHECK(BatchNormalizationBackward);
}

}  // namespace cuda

/*! \brief Forward batch-norm pass on GPU */
template<typename xpu, typename DType, typename AccReal>
void BatchNormOp<xpu, DType, AccReal>::DoForward(mshadow::Stream<gpu> *stream,
                                                 const OpContext &ctx,
                                                 const std::vector<TBlob> &in_data,
                                                 const std::vector<OpReqType> &req,
                                                 const std::vector<TBlob> &out_data,
                                                 const std::vector<TBlob> &aux_states) {
  cuda::BatchNormalizationUpdateOutput<DType, AccReal>(stream,
                                                        ctx,
                                                        in_data,
                                                        req,
                                                        out_data,
                                                        aux_states,
                                                        ctx.is_train && !param_.use_global_stats,
                                                        param_.fix_gamma,
                                                        param_.momentum,
                                                        param_.eps);
  MSHADOW_CUDA_POST_KERNEL_CHECK(BatchNormOp_doForward_gpu);
}

/*! \brief Backward batch-norm pass on GPU */
template<typename xpu, typename DType, typename AccReal>
void BatchNormOp<xpu, DType, AccReal>::DoBackward(mshadow::Stream<gpu> *stream,
                                                  const OpContext &ctx,
                                                  const std::vector<TBlob> &out_grad,
                                                  const std::vector<TBlob> &in_data,
                                                  const std::vector<TBlob> &out_data,
                                                  const std::vector<OpReqType> &req,
                                                  const std::vector<TBlob> &in_grad,
                                                  const std::vector<TBlob> &aux_states) {
  cuda::BatchNormalizationBackward<DType, AccReal>(stream,
                                                    ctx,
                                                    out_grad,
                                                    in_data,
                                                    out_data,
                                                    req,
                                                    in_grad,
                                                    aux_states,
                                                    ctx.is_train,
                                                    param_.fix_gamma,
                                                    1.0,
                                                    param_.momentum,
                                                    param_.eps);
  MSHADOW_CUDA_POST_KERNEL_CHECK(BatchNormOp_doBackward_gpu);
}

/*! \brief Create GPU operator for batch normalization */
template<>
Operator *CreateOp<gpu>(BatchNormParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  if (!param.use_global_stats) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new CuDNNBatchNormOp<DType>(param);
    })
  } else {
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
      op = new BatchNormOp<gpu, DType, AccReal>(param);
    })
  }
#else
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype,
                              DType,
                              AccReal,
                              { op = new BatchNormOp<gpu, DType, AccReal>(param); });
#endif
  return op;
}

}  // namespace op
}  // namespace mxnet
