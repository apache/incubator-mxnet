#include "./seg_op.h"
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>


namespace mxnet {
namespace op {

namespace seg_op {
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define IND2(x, y, sy) ((x) * (sy) + (y))
#define IND3(x, y, z, sy, sz) (IND2(x, y, sy) * (sz) + (z))
#define SEG_CUDA_POST_KERNEL_CHECK(x) \
  do { \
    cudaError err = cudaPeekAtLastError(); \
    if(err != cudaSuccess) {std::cout << "Name: " << #x << " Line: " << __LINE__ << " ErrStr:" << cudaGetErrorString(err) << std::endl;exit(0);} \
  } while (0)

std::pair<dim3, dim3> KernelLauchParamB1G1(int total_count) {
  const int thread_num = 256;
  dim3 dimBlock(thread_num);
  int grid_size = CEIL_DIV(total_count, thread_num);
  int grid_dim_x = grid_size > cuda::kMaxGridNum ? cuda::kMaxGridNum : grid_size;
  dim3 dimGrid(grid_dim_x);
  return std::make_pair(dimBlock, dimGrid);
}


template<int UNROLL_X, typename DType>
__device__ void SumSharedMem(volatile DType* data) {
  if (UNROLL_X == 1) {
    if (threadIdx.x < 16) {
      data[threadIdx.x] += data[threadIdx.x + 16];
      data[threadIdx.x] += data[threadIdx.x + 8];
      data[threadIdx.x] += data[threadIdx.x + 4];
      data[threadIdx.x] += data[threadIdx.x + 2];
      data[threadIdx.x] += data[threadIdx.x + 1];
    }
  }
  else {
    //TODO Enable arbitrary UNROLL_X
    if (UNROLL_X >= 8) {
      data[threadIdx.x] += data[threadIdx.x + 128];
      data[threadIdx.x + 32] += data[threadIdx.x + 128 + 32];
      data[threadIdx.x + 64] += data[threadIdx.x + 128 + 64];
      data[threadIdx.x + 96] += data[threadIdx.x + 128 + 96];
    }
    if (UNROLL_X >= 4) {
      data[threadIdx.x] += data[threadIdx.x + 64];
      data[threadIdx.x + 32] += data[threadIdx.x + 96];
    }
    data[threadIdx.x] += data[threadIdx.x + 32];
    data[threadIdx.x] += data[threadIdx.x + 16];
    data[threadIdx.x] += data[threadIdx.x + 8];
    data[threadIdx.x] += data[threadIdx.x + 4];
    data[threadIdx.x] += data[threadIdx.x + 2];
    data[threadIdx.x] += data[threadIdx.x + 1];
  }
}


/*Fills the selected position in the destination with the index
Important! Here we assume that the sel is sorted!

for i = 0 to seg_num - 1
if(ind_ptr[idx] == ind_ptr[idx + 1]) dst[sel[i]] = i
*/
__global__ void FillSegStartIndex(int* dst, const int* ind_ptr, int seg_num) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < seg_num; idx += blockDim.x * gridDim.x) {
    if (ind_ptr[idx] != ind_ptr[idx + 1]) {
      dst[ind_ptr[idx]] = idx;
    }
  }
}

template<typename DType>
__global__ void EwiseSet(DType* dst, const DType val, int size) {
  for(int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    dst[idx] = val;
  }
}

template<typename DType>
__global__ void EwiseMul(DType* dst, const DType* lhs, const DType* rhs, int size) {
  for(int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    dst[idx] = lhs[idx] * rhs[idx];
  }
}

struct GetSegId {
  static size_t get_temp_bytes(int nnz) {
    size_t temp_storage_bytes = 0;
    cub::Max max_op;
    void* d_temp_storage = nullptr;
    int* d_in = nullptr;
    int* d_out = nullptr;
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, max_op, nnz);
    return temp_storage_bytes;
  }

  static void compute(int* seg_ids, const int* ind_ptr, int seg_num, int nnz, char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
    cudaMemsetAsync(seg_ids, 0, sizeof(int) * nnz, stream);
    std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(seg_num);
    FillSegStartIndex << <block_grid_dim3.second, block_grid_dim3.first, 0, stream >> > (seg_ids, ind_ptr, seg_num);
    cub::Max max_op;
    cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes, seg_ids, seg_ids, max_op, nnz, stream);
    return;
  }
};

struct MinOpCUB {
  cub::Min cub_op;
  template<typename DType>
  __forceinline__ DType init_value() const {
    return std::numeric_limits<DType>::max();
  }
};

struct MaxOpCUB {
  cub::Max cub_op;
  template<typename DType>
  __forceinline__ DType init_value() const {
    return std::numeric_limits<DType>::lowest();
  }
};

struct SumOpCUB {
  cub::Sum cub_op;
  template<typename DType>
  __forceinline__ DType init_value() const {
    return static_cast<DType>(0);
  }
};


/*! \brief minus_exp operator */
struct minus_exp {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return exp(a - b);
  }
};

/*! \brief diff_square operator */
struct diff_square {
  /*! \brief map a, b to result using defined operation */
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return (a - b) * (a - b);
  }
};
/*Expand the indptr to make it repeat k times and increase the values of the indices during the expansion

for i = 0 to repeat_num - 1
for j = 0 to seg_num - 1
dst[i * seg_num + j] = i * nnz + ind_ptr[j];
dst[repeat_num * seg_num] = repeat_num * nnz
*/
__global__ void ExpandIndptr(int* dst, const int* ind_ptr, int seg_num, int nnz, int repeat_num) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx <= seg_num * repeat_num; idx += blockDim.x * gridDim.x) {
    if (idx < seg_num * repeat_num) {
      int seg_id = idx % seg_num;
      int repeat_id = idx / seg_num;
      dst[idx] = ind_ptr[seg_id] + repeat_id * nnz;
    }
    else {
      dst[idx] = repeat_num * nnz;
    }
  }
}

template<typename ReduceOp, typename DType>
struct SegReduceContigCUDA {
  static size_t get_temp_bytes(int batch_num, int nnz, int seg_num) {
    size_t temp_storage_bytes = 0;
    ReduceOp op;
    void* d_temp_storage = nullptr;
    DType* d_in = nullptr;
    DType* d_out = nullptr;
    int* d_begin_offsets = nullptr;
    int* d_end_offsets = nullptr;
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out,
      batch_num * seg_num, d_begin_offsets, d_end_offsets,
      op.cub_op, op.template init_value<DType>());
    temp_storage_bytes += sizeof(int) * (seg_num * batch_num + 1); // Size for the new expaned ind_ptr
    return temp_storage_bytes;
  }

  static void compute(DType* dst, const DType* src, const int* ind_ptr, int batch_num, int nnz, int seg_num, char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
    int expanded_ind_ptr_size = seg_num * batch_num + 1;
    CHECK_GE(temp_storage_bytes, sizeof(int) * expanded_ind_ptr_size);
    ReduceOp op;
    int* expanded_ind_ptr = reinterpret_cast<int*>(temp_storage);
    std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(expanded_ind_ptr_size);
    ExpandIndptr <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>> (expanded_ind_ptr, ind_ptr, seg_num, nnz, batch_num);
    SEG_CUDA_POST_KERNEL_CHECK(ExpandIndptr);
    temp_storage_bytes -= sizeof(int) * expanded_ind_ptr_size;
    char* cub_temp_storage = temp_storage + sizeof(int) * expanded_ind_ptr_size;
    cub::DeviceSegmentedReduce::Reduce(cub_temp_storage,
      temp_storage_bytes,
      src, dst, batch_num * seg_num,
      expanded_ind_ptr, expanded_ind_ptr + 1, op.cub_op, op.template init_value<DType>(), stream);
  }
};

template<int reduce_type>
void SegReduceImpl(const Tensor<gpu, 2, float> &dst,
                   const Tensor<gpu, 2, float> &data,
                   const Tensor<gpu, 1, int> &indptr,
                   const OpReqType req,
                   const OpContext& ctx,
                   mshadow::Stream<gpu>* s) {
  using namespace mxnet_op;
  if (req == kNullOp) return;

  int batch_num = data.shape_[0];
  int nnz = data.shape_[1];
  int seg_num = indptr.shape_[0] - 1;
  cudaStream_t stream = Stream<gpu>::GetStream(s);

  size_t temp_storage_bytes = 0;
  if(reduce_type == SegReduceType::kSum) {
    temp_storage_bytes = SegReduceContigCUDA<SumOpCUB, float>::get_temp_bytes(batch_num, nnz, seg_num);
  } else if(reduce_type == SegReduceType::kMax) {
    temp_storage_bytes = SegReduceContigCUDA<MaxOpCUB, float>::get_temp_bytes(batch_num, nnz, seg_num);
  } else if(reduce_type == SegReduceType::kMin) {
    temp_storage_bytes = SegReduceContigCUDA<MinOpCUB, float>::get_temp_bytes(batch_num, nnz, seg_num);
  } else {
    LOG(FATAL) << "Unsupported!";
  }
  
  Tensor<gpu, 1, char> workspace;
  float* reduce_dst_ptr;
  char* workspace_ptr;
  if(req == kAddTo) {
    int dst_size = batch_num * seg_num;
     workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(
       Shape1(temp_storage_bytes + sizeof(float) * dst_size), s);
     reduce_dst_ptr = reinterpret_cast<float*>(workspace.dptr_);
     workspace_ptr = workspace.dptr_ + sizeof(float) * dst_size;
  } else {
    workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
    reduce_dst_ptr = dst.dptr_;
    workspace_ptr = workspace.dptr_;
  }
  if(reduce_type == SegReduceType::kSum) {
    SegReduceContigCUDA<SumOpCUB, float>::compute(
      reduce_dst_ptr, data.dptr_, indptr.dptr_, batch_num, nnz, seg_num,
      workspace_ptr, temp_storage_bytes, Stream<gpu>::GetStream(s));
  } else if(reduce_type == SegReduceType::kMax) {
    SegReduceContigCUDA<MaxOpCUB, float>::compute(
      reduce_dst_ptr, data.dptr_, indptr.dptr_, batch_num, nnz, seg_num,
      workspace_ptr, temp_storage_bytes, Stream<gpu>::GetStream(s));
  } else if(reduce_type == SegReduceType::kMin) {
    SegReduceContigCUDA<MinOpCUB, float>::compute(
      reduce_dst_ptr, data.dptr_, indptr.dptr_, batch_num, nnz, seg_num,
      workspace_ptr, temp_storage_bytes, Stream<gpu>::GetStream(s));
  }
  // Handle the AddTo case
  if(req == kAddTo) {
    if (reduce_type == SegReduceType::kSum) {
      Kernel<op_with_req<mshadow::op::identity, kAddTo>, gpu>::Launch(s, batch_num * nnz, dst.dptr_, reduce_dst_ptr);
    } else {
      LOG(FATAL) << "Unsupported!";
    }
  }
}

/* Compute broadcast rhs and apply the binary OP between lhs and rhs. Add the result to dst

dst : Shape (batch_num, nnz)
lhs : Shape (batch_num, nnz)
rhs : Shape (batch_num, seg_num)
seg_ids: Shape(nnz,)

for i = 0 to batch_num - 1
for j = 0 to nnz - 1
dst[i, j] += OP::Map(lhs[i, j], rhs[i, seg_ids[j]])
*/
template<typename OP, bool add_to, typename DType>
__global__ void SegBroadcastBinaryContigKernel(DType* dst,
                                               const DType* lhs,
                                               const DType* rhs,
                                               const int* seg_ids,
                                               int batch_num, int nnz, int seg_num) {
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_num * nnz; idx += blockDim.x * gridDim.x) {
    int batch_id = idx / nnz;
    int ele_id = idx % nnz;
    DType res = OP::Map(lhs[idx], rhs[batch_id * seg_num + seg_ids[ele_id]]);
    if (add_to) {
      dst[idx] += res;
    }
    else {
      dst[idx] = res;
    }
  }
}

struct BatchSegBroadcastBinaryCUDA {
  static size_t get_temp_bytes(int nnz) {
    size_t temp_storage_bytes = GetSegId::get_temp_bytes(nnz);
    temp_storage_bytes += sizeof(int) * nnz; // Size of temp seg_ids
    return temp_storage_bytes;
  }

  template<typename OP, bool add_to, typename DType>
  static void compute(DType* dst, const DType* lhs, const DType* rhs, const int* ind_ptr, int batch_num, int nnz, int seg_num,
                      char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
    int* seg_ids = reinterpret_cast<int*>(temp_storage);
    GetSegId::compute(seg_ids, ind_ptr, seg_num, nnz,
      temp_storage + sizeof(int) * nnz, temp_storage_bytes - sizeof(int) * nnz, stream);
    std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(batch_num * nnz);
    SegBroadcastBinaryContigKernel<OP, add_to> << <block_grid_dim3.second, block_grid_dim3.first, 0, stream >> >
      (dst, lhs, rhs, seg_ids, batch_num, nnz, seg_num);
    SEG_CUDA_POST_KERNEL_CHECK(SegBroadcastBinaryContigKernel);
  }
};

template<typename OP>
void SegBroadcastBinaryImpl(const Tensor<gpu, 2, float> &dst,
                            const Tensor<gpu, 2, float> &lhs,
                            const Tensor<gpu, 2, float> &rhs,
                            const Tensor<gpu, 1, int> &indptr,
                            const OpReqType req,
                            const OpContext& ctx,
                            Stream<gpu>* s) {
  if (req == kNullOp) return;
  int batch_num = lhs.shape_[0];
  int nnz = lhs.shape_[1];
  int seg_num = rhs.shape_[1];
  cudaStream_t stream = Stream<gpu>::GetStream(s);

  size_t temp_storage_bytes = BatchSegBroadcastBinaryCUDA::get_temp_bytes(nnz);
  Tensor<gpu, 1, char> workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
  if(req == kAddTo) {
    BatchSegBroadcastBinaryCUDA::compute<OP, true>(dst.dptr_, lhs.dptr_, rhs.dptr_, indptr.dptr_, batch_num, nnz, seg_num, workspace.dptr_, temp_storage_bytes, stream);
  } else {
    cudaMemsetAsync(dst.dptr_, 0, sizeof(float) * batch_num * nnz, stream);
    BatchSegBroadcastBinaryCUDA::compute<OP, false>(dst.dptr_, lhs.dptr_, rhs.dptr_, indptr.dptr_, batch_num, nnz, seg_num, workspace.dptr_, temp_storage_bytes, stream);
  }
}

struct SegSoftmaxContigCUDA {
  static size_t get_temp_sum_bytes(int batch_num, int nnz, int seg_num) {
    size_t temp_sum_storage_bytes = 0;
    SumOpCUB sum_op;
    void* d_temp_storage = nullptr;
    float* d_in = nullptr;
    float* d_out = nullptr;
    int* d_begin_offsets = nullptr;
    int* d_end_offsets = nullptr;
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_sum_storage_bytes, d_in, d_out,
      batch_num * seg_num, d_begin_offsets, d_end_offsets,
      sum_op.cub_op, sum_op.template init_value<float>());
    return temp_sum_storage_bytes;
  }

  static size_t get_temp_max_bytes(int batch_num, int nnz, int seg_num) {
    size_t temp_max_storage_bytes = 0;
    MaxOpCUB max_op;
    void* d_temp_storage = nullptr;
    float* d_in = nullptr;
    float* d_out = nullptr;
    int* d_begin_offsets = nullptr;
    int* d_end_offsets = nullptr;
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_max_storage_bytes, d_in, d_out,
      batch_num * seg_num, d_begin_offsets, d_end_offsets,
      max_op.cub_op, max_op.template init_value<float>());
    return temp_max_storage_bytes;
  }

  static size_t get_temp_bytes(int batch_num, int nnz, int seg_num) {
    size_t temp_space_bytes = 0;
    temp_space_bytes = std::max(temp_space_bytes, get_temp_max_bytes(batch_num, nnz, seg_num));  // Temp space for max reduce
    temp_space_bytes = std::max(temp_space_bytes, get_temp_sum_bytes(batch_num, nnz, seg_num));  // Temp space for sum reduce
    temp_space_bytes = std::max(temp_space_bytes, GetSegId::get_temp_bytes(nnz));  // Temp space for GetSegId
    return temp_space_bytes +
           sizeof(int) * (seg_num * batch_num + 1) +  // Size of the new expaned ind_ptr
           sizeof(int) * nnz +  // Size of the seg_ids
           sizeof(float) * batch_num * seg_num;  // Size of the temp holder for the max value / sum value
  }

  static void compute(float* dst, const float* src, const int* ind_ptr, int batch_num, int nnz, int seg_num, char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
    int expanded_ind_ptr_size = seg_num * batch_num + 1;
    CHECK_GE(temp_storage_bytes, get_temp_bytes(batch_num, nnz, seg_num));
    int* expanded_ind_ptr = reinterpret_cast<int*>(temp_storage);
    temp_storage += sizeof(int) * expanded_ind_ptr_size;
    int* seg_ids = reinterpret_cast<int*>(temp_storage);
    temp_storage += sizeof(int) * nnz;
    float* temp_reduce_holder = reinterpret_cast<float*>(temp_storage);
    temp_storage += sizeof(float) * batch_num * seg_num;
    // Set the destination to be all -1
    std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(batch_num * nnz);
    EwiseSet <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>> (dst, 0.0f, batch_num * nnz);
    // Get the expanded indptr
    block_grid_dim3 = KernelLauchParamB1G1(expanded_ind_ptr_size);
    ExpandIndptr <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>> (expanded_ind_ptr, ind_ptr, seg_num, nnz, batch_num);
    SEG_CUDA_POST_KERNEL_CHECK(ExpandIndptr);
    // Get the seg ids
    GetSegId::compute(seg_ids, ind_ptr, seg_num, nnz, temp_storage, GetSegId::get_temp_bytes(nnz), stream);
    // Calculate the maximum value
    MaxOpCUB max_op;
    size_t temp_max_bytes = get_temp_max_bytes(batch_num, nnz, seg_num);
    cub::DeviceSegmentedReduce::Reduce(temp_storage,
      temp_max_bytes,
      src, temp_reduce_holder, batch_num * seg_num,
      expanded_ind_ptr, expanded_ind_ptr + 1, max_op.cub_op, max_op.template init_value<float>(), stream);
    // Use minus_exp to subtract the maximum value and take the exp
    block_grid_dim3 = KernelLauchParamB1G1(batch_num * nnz);
    SegBroadcastBinaryContigKernel<minus_exp, false> <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>>
      (dst, src, temp_reduce_holder, seg_ids, batch_num, nnz, seg_num);
    SEG_CUDA_POST_KERNEL_CHECK(SegBroadcastBinaryContigKernel);
    // Calculate the sum
    SumOpCUB sum_op;
    size_t temp_sum_bytes = get_temp_sum_bytes(batch_num, nnz, seg_num);
    cub::DeviceSegmentedReduce::Reduce(temp_storage,
      temp_sum_bytes,
      dst, temp_reduce_holder, batch_num * seg_num,
      expanded_ind_ptr, expanded_ind_ptr + 1, sum_op.cub_op, sum_op.template init_value<float>(), stream);
    // Use broadcast_div to divide the sum
    block_grid_dim3 = KernelLauchParamB1G1(batch_num * nnz);
    SegBroadcastBinaryContigKernel<mshadow::op::div, false> <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>>
      (dst, dst, temp_reduce_holder, seg_ids, batch_num, nnz, seg_num);
    SEG_CUDA_POST_KERNEL_CHECK(SegBroadcastBinaryContigKernel);
  }
};

void SegSoftmaxImpl(const Tensor<gpu, 2, float> &dst,
                    const Tensor<gpu, 2, float> &data,
                    const Tensor<gpu, 1, int> &indptr,
                    const OpReqType req,
                    const OpContext& ctx,
                    Stream<gpu>* s) {
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "Forward AddTo for seg_softmax is currently not supported!";
  int batch_num = data.shape_[0];
  int nnz = data.shape_[1];
  int seg_num = indptr.shape_[0] - 1;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  size_t temp_storage_bytes = SegSoftmaxContigCUDA::get_temp_bytes(batch_num, nnz, seg_num);
  Tensor<gpu, 1, char> workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
  SegSoftmaxContigCUDA::compute(dst.dptr_, data.dptr_, indptr.dptr_, batch_num, nnz, seg_num, workspace.dptr_, temp_storage_bytes, stream);
}

struct SegSoftmaxContigBackwardCUDA {
  static size_t get_temp_sum_bytes(int batch_num, int nnz, int seg_num) {
    size_t temp_sum_storage_bytes = 0;
    SumOpCUB sum_op;
    void* d_temp_storage = nullptr;
    float* d_in = nullptr;
    float* d_out = nullptr;
    int* d_begin_offsets = nullptr;
    int* d_end_offsets = nullptr;
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_sum_storage_bytes, d_in, d_out,
      batch_num * seg_num, d_begin_offsets, d_end_offsets,
      sum_op.cub_op, sum_op.template init_value<float>());
    return temp_sum_storage_bytes;
  }

  static size_t get_temp_bytes(int batch_num, int nnz, int seg_num) {
    size_t temp_space_bytes = 0;
    temp_space_bytes = std::max(temp_space_bytes, get_temp_sum_bytes(batch_num, nnz, seg_num));  // Temp space for summation
    temp_space_bytes = std::max(temp_space_bytes, GetSegId::get_temp_bytes(nnz));  // Temp space for GetSegId
    return temp_space_bytes +  // Temp space
           sizeof(int) * (seg_num * batch_num + 1) +  // Size of the new expaned ind_ptr
           sizeof(int) * nnz +  // Size of seg_ids
           sizeof(float) * batch_num * seg_num;  // Size of the temp holder for the sum value
  }

  static void compute(float* dst, const float *ograd, const float* val, const int* ind_ptr,
                      int batch_num, int nnz, int seg_num,
                      char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
    CHECK_GE(temp_storage_bytes, get_temp_bytes(batch_num, nnz, seg_num));
    int expanded_ind_ptr_size = seg_num * batch_num + 1;
    int* expanded_ind_ptr = reinterpret_cast<int*>(temp_storage);
    temp_storage += sizeof(int) * expanded_ind_ptr_size;
    int* seg_ids = reinterpret_cast<int*>(temp_storage);
    temp_storage += sizeof(int) * nnz;
    float* temp_reduce_holder = reinterpret_cast<float*>(temp_storage);
    temp_storage += sizeof(float) * batch_num * seg_num;
    // Set the destination to be all 0
    cudaMemsetAsync(dst, 0, sizeof(float) * batch_num * nnz, stream);
    // Calculate expanded_indptr
    std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(expanded_ind_ptr_size);
    ExpandIndptr <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>> (expanded_ind_ptr, ind_ptr, seg_num, nnz, batch_num);
    SEG_CUDA_POST_KERNEL_CHECK(ExpandIndptr);
    // Get the seg ids
    GetSegId::compute(seg_ids, ind_ptr, seg_num, nnz, temp_storage, GetSegId::get_temp_bytes(nnz), stream);
    // Perform EwiseMul and store the result
    block_grid_dim3 = KernelLauchParamB1G1(batch_num * nnz);
    EwiseMul <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>> (dst, ograd, val, batch_num * nnz);
    SEG_CUDA_POST_KERNEL_CHECK(EwiseMul);
    // Calculate the sum
    SumOpCUB sum_op;
    size_t temp_sum_bytes = get_temp_sum_bytes(batch_num, nnz, seg_num);
    cub::DeviceSegmentedReduce::Reduce(temp_storage,
      temp_sum_bytes,
      dst, temp_reduce_holder, batch_num * seg_num,
      expanded_ind_ptr, expanded_ind_ptr + 1, sum_op.cub_op,
      sum_op.template init_value<float>(), stream);
    // Use minus_exp to subtract the sum value
    block_grid_dim3 = KernelLauchParamB1G1(batch_num * nnz);
    SegBroadcastBinaryContigKernel<mshadow::op::minus, false>
      <<<block_grid_dim3.second, block_grid_dim3.first, 0, stream>>>
      (dst, ograd, temp_reduce_holder, seg_ids, batch_num, nnz, seg_num);
    SEG_CUDA_POST_KERNEL_CHECK(SegBroadcastBinaryContigKernel);
    // Use EwiseMul to multiply the value
    block_grid_dim3 = KernelLauchParamB1G1(batch_num * nnz);
    EwiseMul << <block_grid_dim3.second, block_grid_dim3.first, 0, stream >> > (dst, dst, val, batch_num * nnz);
    SEG_CUDA_POST_KERNEL_CHECK(EwiseMul);
  }
};

void SegSoftmaxBackwardImpl(const Tensor<gpu, 2, float> &dst,
                            const Tensor<gpu, 2, float> &ograd,
                            const Tensor<gpu, 2, float> &val,
                            const Tensor<gpu, 1, int> &indptr,
                            const OpReqType req,
                            const OpContext& ctx,
                            Stream<gpu>* s) {
  using namespace mxnet_op;
  if (req == kNullOp) return;
  int batch_num = ograd.shape_[0];
  int nnz = ograd.shape_[1];
  int seg_num = indptr.shape_[1] - 1;
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  size_t temp_storage_bytes = SegSoftmaxContigBackwardCUDA::get_temp_bytes(batch_num, nnz, seg_num);
  Tensor<gpu, 1, char> workspace;
  float* backward_dst_ptr;
  char* workspace_ptr;
  if(req == kAddTo) {
    workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(
       Shape1(temp_storage_bytes + sizeof(float) * batch_num * nnz), s);
    backward_dst_ptr = reinterpret_cast<float*>(workspace.dptr_);
    workspace_ptr = workspace.dptr_ + sizeof(float) * batch_num * nnz;
  } else {
    workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(
      Shape1(temp_storage_bytes), s);
    backward_dst_ptr = dst.dptr_;
    workspace_ptr = workspace.dptr_;
  }
  SegSoftmaxContigBackwardCUDA::compute(
    backward_dst_ptr, ograd.dptr_, val.dptr_, indptr.dptr_, batch_num, nnz, seg_num,
    workspace_ptr, temp_storage_bytes, stream);
  // Handle the AddTo case
  if (req == kAddTo) {
      Kernel<op_with_req<mshadow::op::identity, kAddTo>, gpu>::Launch(
        s, batch_num * nnz, dst.dptr_, backward_dst_ptr);
  }
}

/*For all the nodes, computes the inner product between the node and it's neighborhoods and add to dst.

dst: Shape (K, nnz)
embed1: Shape (K, node_num, feat_dim)
embed2: Shape (K, neighbor_node_num, feat_dim)
neighbor_ids: Shape (nnz, )
neighbor_ind_ptr: Shape(node_num + 1, )
rev_node_ids : Shape(nnz, ), The reverse mapping from 0->nnz-1 to node_ids


use mul to compute the inner-product and use squared_diff to compute the squared distance.

for k = 0 to K-1
    for i = 0  to node_num - 1
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            neighbor_id = neighbor_ids[j]
            dst[k, j] += InnerProduct(embed1[k, i], embed2[k, neighbor_id]) or ||embed1[k, i] - embed2[k, neighbor_id]||^2_2

*/
template<typename OP, int UNROLL_NODE = 1, int TY_SZ = 16, int UNROLL_Y = 4, int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(TY_SZ * WARP_SZ)
SegTakeKCorrKernel(float* dst,
                   const float* embed1,
                   const float* embed2,
                   const int* neighbor_ids,
                   const int* neighbor_ind_ptr,
                   const int* rev_node_ids,
                   int K, int node_num, int neighbor_node_num,
                   int nnz, int feat_dim) {
    int k = blockIdx.y;
    __shared__ float embed1_shared[UNROLL_NODE * TY_SZ][WARP_SZ * UNROLL_X];
    __shared__ float embed2_shared[TY_SZ][WARP_SZ * UNROLL_X];
    __shared__ int rev_node_ids_shared[UNROLL_Y * TY_SZ];
    __shared__ int neighbor_ids_shared[UNROLL_Y * TY_SZ];
    __shared__ float dst_shared[UNROLL_Y * TY_SZ]; // Shared variable to store the result that should be saved to dst
    for(int c_begin = 0; c_begin < feat_dim; c_begin += WARP_SZ * UNROLL_X) { // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
        for(int b_nid = UNROLL_NODE * TY_SZ * blockIdx.x; b_nid < node_num; b_nid += UNROLL_NODE * TY_SZ * gridDim.x) {
            int e_nid = min(b_nid + UNROLL_NODE * TY_SZ, node_num);
            int b_neighbor_ind = neighbor_ind_ptr[b_nid];
            int e_neighbor_ind = neighbor_ind_ptr[e_nid];
            // 1. Load embed1 to shared memory
            #pragma unroll
            for(int j = 0; j < UNROLL_NODE; j++) {
                int nid_delta = j * TY_SZ + threadIdx.y;
                #pragma unroll
                for(int i = 0; i < UNROLL_X; i++) {
                    int c_delta = i * WARP_SZ + threadIdx.x;
                    if(c_begin + c_delta < feat_dim && b_nid + nid_delta < e_nid) {
                        embed1_shared[nid_delta][c_delta] = embed1[IND3(k, b_nid + nid_delta, c_begin + c_delta, node_num, feat_dim)];
                    } else {
                        embed1_shared[nid_delta][c_delta] = 0.0f;
                    }
                }
            }
            // 2. Compute the inner product between embed1 and embed2
            for(int b_ind_inner = b_neighbor_ind; b_ind_inner < e_neighbor_ind; b_ind_inner += UNROLL_Y * TY_SZ) {
                int e_ind_inner = min(b_ind_inner + UNROLL_Y * TY_SZ, e_neighbor_ind);
                // 2.1 Initilaize the shared dst variables to zero.
                #pragma unroll
                for(int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                    if(threadIdx.y == 0) dst_shared[i * WARP_SZ + threadIdx.x] = 0.0f;
                }
                // 2.2 Load the rev_node_ids and neighbor_node_ids to shared memory
                if(threadIdx.y == 0) {
                    #pragma unroll
                    for(int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                        if (b_ind_inner + i * WARP_SZ + threadIdx.x < e_ind_inner && i * WARP_SZ + threadIdx.x < UNROLL_Y * TY_SZ) {
                            rev_node_ids_shared[i * WARP_SZ + threadIdx.x] = rev_node_ids[b_ind_inner + i * WARP_SZ + threadIdx.x] - b_nid;
                            neighbor_ids_shared[i * WARP_SZ + threadIdx.x] = neighbor_ids[b_ind_inner + i * WARP_SZ + threadIdx.x];
                        }
                    }
                }
                __syncthreads();
                // 2.3 Load embed2 to shared memory and do the computation
                #pragma unroll
                for(int j = 0; j < UNROLL_Y; j++) {
                    int ind_inner_delta = j * TY_SZ + threadIdx.y;
                    // 2.3.1 Perform the loading
                    #pragma unroll
                    for(int i = 0; i < UNROLL_X; i++) {
                        int c_delta = i * WARP_SZ + threadIdx.x;
                        if(c_delta + c_begin < feat_dim && b_ind_inner + ind_inner_delta < e_ind_inner) {
                            // Load and perform the binary operator
                            // TODO(sxjscience) potential overflow problem, consider use size_t instead
                            embed2_shared[threadIdx.y][c_delta] = OP::Map(embed2[IND3(k, neighbor_ids_shared[ind_inner_delta], c_delta + c_begin, neighbor_node_num, feat_dim)],
                                                                            embed1_shared[rev_node_ids_shared[ind_inner_delta]][c_delta]);
                        } else {
                            embed2_shared[threadIdx.y][c_delta] = 0.0f;
                        }
                    }
                    // 2.3.2 Perform the reduction
                    SumSharedMem<UNROLL_X>(embed2_shared[threadIdx.y]);
                    // 2.3.3 Accumulate the result to the local dst variable
                    if(threadIdx.x == 0) dst_shared[j * TY_SZ + threadIdx.y] += embed2_shared[threadIdx.y][0];
                    __syncthreads();
                }
                // 2.4 Write the shared variable back to the global memory
                if(threadIdx.y == 0) {
                    #pragma unroll
                    for (int i = 0; i < CEIL_DIV(UNROLL_Y * TY_SZ, WARP_SZ); i++) {
                        if (b_ind_inner + i * WARP_SZ + threadIdx.x < e_ind_inner && i * WARP_SZ + threadIdx.x < UNROLL_Y * TY_SZ) {
                            dst[k * nnz + b_ind_inner + i * WARP_SZ + threadIdx.x] += dst_shared[i * WARP_SZ + threadIdx.x];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}


/*Compute the backward pass of SegTakeKCorr w.r.t embed1 when inner product is used.

dst: Shape (K, node_num, feat_dim)
g_out: Shape (K, nnz)
embed2: Shape (K, neighbor_node_num, feat_dim)
neighbor_ids: Shape (nnz, )
neighbor_ind_ptr: Shape(node_num + 1, )


for k = 0 to K-1
    for i = 0  to node_num - 1
        dst[k, i, :] = 0
        for j = ind_ptr[i] to ind_ptr[i+1] - 1
            dst[k, i, :] += g_out[k, j] * embed2[k, neighbor_ids[j], :]
*/
template<int UNROLL_X = 4, int TX_SZ = 32>
__global__ void
__launch_bounds__(TX_SZ)
SegTakeKCorrBackwardEmbed1Kernel(float* dst,
                                 const float* g_out,
                                 const float* embed2,
                                 const int* neighbor_ids,
                                 const int* neighbor_ind_ptr,
                                 int K, int node_num, int neighbor_node_num,
                                 int nnz, int feat_dim) {
    int k = blockIdx.z;
    int c_begin = blockIdx.y * TX_SZ * UNROLL_X; // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
    float dst_local[UNROLL_X];
    for (int nid = blockIdx.x; nid < node_num; nid += gridDim.x) {
        int b_neighbor_ind = neighbor_ind_ptr[nid];
        int e_neighbor_ind = neighbor_ind_ptr[nid + 1];
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c = c_begin + i * TX_SZ + threadIdx.x;
            if(c < feat_dim) {
                dst_local[i] = dst[IND3(k, nid, c, node_num, feat_dim)];
            }
        }
        for(int j = b_neighbor_ind; j < e_neighbor_ind; j++) {
            #pragma unroll
            for(int i = 0; i < UNROLL_X; i++) {
                int c = c_begin + i * TX_SZ + threadIdx.x;
                if (c < feat_dim) {
                    dst_local[i] += g_out[k * nnz + j] * embed2[IND3(k, neighbor_ids[j], c, neighbor_node_num, feat_dim)];
                }
            }
        }
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c = c_begin + i * TX_SZ + threadIdx.x;
            if(c < feat_dim) {
                dst[IND3(k, nid, c, node_num, feat_dim)] = dst_local[i];
            }
        }
    }
}


__global__ void IdxArrayKernel(int* dst, int size) {
  for(int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
    dst[idx] = idx;
  }
}

/*Compute the backward pass of SegTakeKCorr w.r.t embed2 when inner product is used.

dst: Shape (K, neighbor_node_num, feat_dim)
g_out: Shape (K, nnz)
embed1: Shape (K, node_num, feat_dim)
sorted_neighbor_ids: Shape (nnz, )
original_inds: Shape(nnz, )
rev_node_ids : Shape(nnz, ), The reverse mapping from 0->nnz-1 to node_ids

for k = 0 to K - 1
    for i = 0  to nnz - 1
      dst_ind = sorted_neighbor_ids[i]
      src_ind = sorted_inds[i]
      dst[k, dst_ind, :] += g_out[k, src_ind] * embed1[k, rev_node_ids[src_ind], :]

*/
template<int UNROLL_X = 4, int TX_SZ = 32, int TY_SZ = 1>
__global__ void
__launch_bounds__(TX_SZ * TY_SZ)
SegTakeKCorrBackwardEmbed2Kernel(float* dst,
                                 const float* g_out,
                                 const float* embed1,
                                 const int* sorted_neighbor_ids,
                                 const int* sorted_inds,
                                 const int* rev_node_ids,
                                 int K, int node_num, int neighbor_node_num,
                                 int nnz, int feat_dim) {
  int k = blockIdx.z;
  int c_begin = blockIdx.y * TX_SZ * UNROLL_X; // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
  float dst_local[UNROLL_X];
  int idx = blockIdx.x * TY_SZ + threadIdx.y;
  if (idx < nnz && (idx == 0 || sorted_neighbor_ids[idx] != sorted_neighbor_ids[idx - 1])) {
    const int dst_ind = sorted_neighbor_ids[idx];
    #pragma unroll
    for (int i = 0; i < UNROLL_X; i++) {
      int c = c_begin + i * TX_SZ + threadIdx.x;
      if (c < feat_dim) {
        dst_local[i] = dst[IND3(k, dst_ind, c, neighbor_node_num, feat_dim)];
      }
    }
    do {
      const int src_ind = sorted_inds[idx];
      #pragma unroll
      for(int i = 0; i < UNROLL_X; i++) {
          int c = c_begin + i * TX_SZ + threadIdx.x;
          if (c < feat_dim) {
              dst_local[i] += g_out[k * nnz + src_ind] * embed1[IND3(k, rev_node_ids[src_ind], c, node_num, feat_dim)];
          }
      }
      idx++;
    } while (idx < nnz && (sorted_neighbor_ids[idx] == sorted_neighbor_ids[idx - 1]));
    #pragma unroll
    for (int i = 0; i < UNROLL_X; i++) {
      int c = c_begin + i * TX_SZ + threadIdx.x;
      if (c < feat_dim) {
        dst[IND3(k, dst_ind, c, neighbor_node_num, feat_dim)] = dst_local[i];
      }
    }
  }
}

struct SegTakeKCorrCUDA {
    static size_t get_temp_bytes(int nnz) {
        size_t temp_storage_bytes = GetSegId::get_temp_bytes(nnz);
        temp_storage_bytes += sizeof(int) * nnz; // Size of temp seg_ids
        return temp_storage_bytes;
    }

    static size_t get_sort_temp_bytes(int nnz, int node_num) {
      size_t temp_storage_bytes = 0;
      void* d_temp_storage = nullptr;
      int* d_keys_in = nullptr;
      int* d_keys_out = nullptr;
      int* d_values_in = nullptr;
      int* d_values_out = nullptr;
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, nnz);
      return temp_storage_bytes;
    }

    static size_t get_temp_bytes_backward_embed2(int nnz, int seg_num) {
      size_t temp_storage_bytes = GetSegId::get_temp_bytes(nnz);
      temp_storage_bytes += get_sort_temp_bytes(nnz, seg_num);
      temp_storage_bytes += sizeof(int) * nnz; // Size of temp seg_ids
      temp_storage_bytes += sizeof(int) * nnz; // Size of temp sorted_neighbor_ids
      temp_storage_bytes += sizeof(int) * nnz; // Size of temp value_in
      temp_storage_bytes += sizeof(int) * nnz; // Size of temp value_out
      return temp_storage_bytes;
    }

    template<bool add_to>
    static void compute(float* dst, const float* embed1, const float* embed2,
                        const int* neighbor_ids, const int* neighbor_ind_ptr,
                        int K, int node_num, int neighbor_node_num, int nnz, int feat_dim, int type,
                        char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        long long K_ll = static_cast<long long>(K);
        long long node_num_ll = static_cast<long long>(node_num);
        long long neighbor_node_num_ll = static_cast<long long>(neighbor_node_num);
        long long feat_dim_ll = static_cast<long long>(feat_dim);
        long long int_max_ll = static_cast<long long>(std::numeric_limits<int>::max());
        CHECK_LT(K_ll * node_num_ll * feat_dim_ll, int_max_ll);
        CHECK_LT(K_ll * neighbor_node_num_ll * feat_dim_ll, int_max_ll);
        if(!add_to) {
            cudaMemsetAsync(dst, 0, sizeof(float) * K * nnz, stream);
        }
        int* rev_node_ids = reinterpret_cast<int*>(temp_storage);
        GetSegId::compute(rev_node_ids, neighbor_ind_ptr, node_num, nnz,
                          temp_storage + sizeof(int) * nnz, temp_storage_bytes - sizeof(int) * nnz, stream);
        static const int UNROLL_NODE = 1;
        static const int TY_SZ = 16;
        static const int UNROLL_Y = 4;
        static const int UNROLL_X = 4;
        static const int WARP_SZ = 32;
        dim3 dimBlock(WARP_SZ, TY_SZ);
        dim3 dimGrid(CEIL_DIV(node_num, UNROLL_NODE * TY_SZ), K);
        if (type == SegTakeKCorrType::kInnerProduct) {
            SegTakeKCorrKernel<mshadow::op::mul, UNROLL_NODE, TY_SZ, UNROLL_Y, UNROLL_X, WARP_SZ> <<<dimGrid, dimBlock, 0, stream >>>
                (dst, embed1, embed2, neighbor_ids, neighbor_ind_ptr, rev_node_ids, K, node_num, neighbor_node_num, nnz, feat_dim);
        } else if (type == SegTakeKCorrType::kEuclidean) {
            SegTakeKCorrKernel<diff_square, UNROLL_NODE, TY_SZ, UNROLL_Y, UNROLL_X, WARP_SZ> <<<dimGrid, dimBlock, 0, stream >>>
                (dst, embed1, embed2, neighbor_ids, neighbor_ind_ptr, rev_node_ids, K, node_num, neighbor_node_num, nnz, feat_dim);
        } else {
            LOG(FATAL) << "Unsupported correlation type!";
        }
        SEG_CUDA_POST_KERNEL_CHECK(SegTakeKCorrKernel);
    }

    template<bool add_to>
    static void compute_grad_embed1(float* dst, const float* g_out, const float* embed2,
                                    const int* neighbor_ids, const int* neighbor_ind_ptr,
                                    int K, int node_num, int neighbor_node_num, int nnz, int feat_dim, int type, cudaStream_t stream) {
        CHECK_EQ(type, SegTakeKCorrType::kInnerProduct);
        long long K_ll = static_cast<long long>(K);
        long long node_num_ll = static_cast<long long>(node_num);
        long long neighbor_node_num_ll = static_cast<long long>(neighbor_node_num);
        long long feat_dim_ll = static_cast<long long>(feat_dim);
        long long int_max_ll = static_cast<long long>(std::numeric_limits<int>::max());
        CHECK_LT(K_ll * node_num_ll * feat_dim_ll, int_max_ll);
        CHECK_LT(K_ll * neighbor_node_num_ll * feat_dim_ll, int_max_ll);
        if(!add_to) {
            cudaMemsetAsync(dst, 0, sizeof(float) * K * node_num * feat_dim, stream);
        }
        static const int UNROLL_X = 4;
        static const int TX_SZ = 32;
        dim3 dimBlock(TX_SZ);
        dim3 dimGrid(node_num, CEIL_DIV(feat_dim, TX_SZ * UNROLL_X), K);
        SegTakeKCorrBackwardEmbed1Kernel<UNROLL_X, TX_SZ> <<<dimGrid, dimBlock, 0, stream >>>
                (dst, g_out, embed2, neighbor_ids, neighbor_ind_ptr, K, node_num,
                 neighbor_node_num, nnz, feat_dim);
        SEG_CUDA_POST_KERNEL_CHECK(SegTakeKCorrBackwardEmbed1Kernel);
    }

    template<bool add_to>
    static void compute_grad_embed2(float* dst, const float* g_out, const float* embed1,
                                    const int* neighbor_ids, const int* neighbor_ind_ptr,
                                    int K, int node_num, int neighbor_node_num, int nnz, int feat_dim, int type,
                                    char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        CHECK_EQ(type, SegTakeKCorrType::kInnerProduct);
        long long K_ll = static_cast<long long>(K);
        long long node_num_ll = static_cast<long long>(node_num);
        long long neighbor_node_num_ll = static_cast<long long>(neighbor_node_num);
        long long feat_dim_ll = static_cast<long long>(feat_dim);
        long long int_max_ll = static_cast<long long>(std::numeric_limits<int>::max());
        CHECK_LT(K_ll * node_num_ll * feat_dim_ll, int_max_ll);
        CHECK_LT(K_ll * neighbor_node_num_ll * feat_dim_ll, int_max_ll);
        if (!add_to) {
            cudaMemsetAsync(dst, 0, sizeof(float) * K * neighbor_node_num * feat_dim, stream);
        }
        int* rev_node_ids = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* sorted_neighbor_ids = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* temp_ind_in = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* sorted_ind = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        // 1. Sort the neighbor_ids
        std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(nnz);
        IdxArrayKernel << <block_grid_dim3.second, block_grid_dim3.first, 0, stream >> > (temp_ind_in, nnz);
        size_t temp_sort_bytes = get_sort_temp_bytes(nnz, node_num);
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_sort_bytes,
          neighbor_ids, sorted_neighbor_ids,
          temp_ind_in, sorted_ind, nnz, 0, sizeof(int) * 8, stream);
        temp_storage += get_sort_temp_bytes(nnz, node_num);
        // 2. Compute the rev mapping
        GetSegId::compute(rev_node_ids, neighbor_ind_ptr, node_num, nnz, temp_storage, GetSegId::get_temp_bytes(nnz), stream);
        // 3. Run the kernel
        static const int UNROLL_X = 4;
        static const int TX_SZ = 32;
        static const int TY_SZ = 1;
        dim3 dimBlock(TX_SZ, TY_SZ);
        dim3 dimGrid(CEIL_DIV(nnz, TY_SZ), CEIL_DIV(feat_dim, TX_SZ * UNROLL_X), K);
        SegTakeKCorrBackwardEmbed2Kernel<UNROLL_X, TX_SZ, TY_SZ> << <dimGrid, dimBlock, 0, stream >> >
          (dst, g_out, embed1, sorted_neighbor_ids, sorted_ind, rev_node_ids, K, node_num,
            neighbor_node_num, nnz, feat_dim);
        SEG_CUDA_POST_KERNEL_CHECK(SegTakeKCorrBackwardEmbed2Kernel);
    }
};

void SegTakeKCorrImpl(const Tensor<gpu, 2, float> &dst,
                      const Tensor<gpu, 3, float> &embed1,
                      const Tensor<gpu, 3, float> &embed2,
                      const Tensor<gpu, 1, int> &neighbor_ids,
                      const Tensor<gpu, 1, int> &neighbor_ind_ptr,
                      const OpReqType req,
                      const OpContext& ctx,
                      Stream<gpu>* s) {
  using namespace mxnet_op;
  if (req == kNullOp) return;
  int K = embed1.shape_[0];
  int node_num = embed1.shape_[1];
  int feat_dim = embed1.shape_[2];
  int neighbor_node_num = embed2.shape_[1];
  int nnz = neighbor_ids.shape_[0];
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  size_t temp_storage_bytes = SegTakeKCorrCUDA::get_temp_bytes(nnz);
  Tensor<gpu, 1, char> workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
  if(req == kAddTo) {
    SegTakeKCorrCUDA::compute<true>(dst.dptr_, embed1.dptr_, embed2.dptr_, neighbor_ids.dptr_ , neighbor_ind_ptr.dptr_, K, node_num, neighbor_node_num, nnz, feat_dim,
                                    SegTakeKCorrType::kInnerProduct, workspace.dptr_, temp_storage_bytes, stream);
  } else {
    SegTakeKCorrCUDA::compute<false>(dst.dptr_, embed1.dptr_, embed2.dptr_, neighbor_ids.dptr_ , neighbor_ind_ptr.dptr_, K, node_num, neighbor_node_num, nnz, feat_dim,
                                     SegTakeKCorrType::kInnerProduct, workspace.dptr_, temp_storage_bytes, stream);
  }
}

void SegTakeKCorrBackwardEmbed1Impl(const Tensor<gpu, 3, float> &dst,
                                    const Tensor<gpu, 2, float> &ograd,
                                    const Tensor<gpu, 3, float> &embed2,
                                    const Tensor<gpu, 1, int> &neighbor_ids,
                                    const Tensor<gpu, 1, int> &neighbor_ind_ptr,
                                    const OpReqType req,
                                    const OpContext& ctx,
                                    Stream<gpu>* s) {
  using namespace mxnet_op;
  if (req == kNullOp) return;
  int K = ograd.shape_[0];
  int node_num = neighbor_ind_ptr.shape_[0] - 1;
  int feat_dim = embed2.shape_[2];
  int neighbor_node_num = embed2.shape_[1];
  int nnz = neighbor_ids.shape_[0];
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  if(req == kAddTo) {
    SegTakeKCorrCUDA::compute_grad_embed1<true>(dst.dptr_, ograd.dptr_, embed2.dptr_, neighbor_ids.dptr_, neighbor_ind_ptr.dptr_,
                                                K, node_num, neighbor_node_num, nnz, feat_dim, SegTakeKCorrType::kInnerProduct, stream);
  } else {
    SegTakeKCorrCUDA::compute_grad_embed1<false>(dst.dptr_, ograd.dptr_, embed2.dptr_, neighbor_ids.dptr_, neighbor_ind_ptr.dptr_,
                                                 K, node_num, neighbor_node_num, nnz, feat_dim, SegTakeKCorrType::kInnerProduct, stream);
  }
}

void SegTakeKCorrBackwardEmbed2Impl(const Tensor<gpu, 3, float> &dst,
                                    const Tensor<gpu, 2, float> &ograd,
                                    const Tensor<gpu, 3, float> &embed1,
                                    const Tensor<gpu, 1, int> &neighbor_ids,
                                    const Tensor<gpu, 1, int> &neighbor_ind_ptr,
                                    const OpReqType req,
                                    const OpContext& ctx,
                                    Stream<gpu>* s) {
  using namespace mxnet_op;
  if (req == kNullOp) return;
  int K = ograd.shape_[0];
  int node_num = embed1.shape_[1];
  int feat_dim = embed1.shape_[2];
  int neighbor_node_num = dst.shape_[1];
  int nnz = neighbor_ids.shape_[0];
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  size_t temp_storage_bytes = SegTakeKCorrCUDA::get_temp_bytes_backward_embed2(nnz, node_num);
  Tensor<gpu, 1, char> workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
  if(req == kAddTo) {
    SegTakeKCorrCUDA::compute_grad_embed2<true>(dst.dptr_, ograd.dptr_, embed1.dptr_, neighbor_ids.dptr_, neighbor_ind_ptr.dptr_,
      K, node_num, neighbor_node_num, nnz, feat_dim, SegTakeKCorrType::kInnerProduct, workspace.dptr_, temp_storage_bytes, stream);
  } else {
    SegTakeKCorrCUDA::compute_grad_embed2<false>(dst.dptr_, ograd.dptr_, embed1.dptr_, neighbor_ids.dptr_, neighbor_ind_ptr.dptr_,
      K, node_num, neighbor_node_num, nnz, feat_dim, SegTakeKCorrType::kInnerProduct, workspace.dptr_, temp_storage_bytes, stream);
  }
}

/*Divide the elements in a segmentation by the length of the segmentation.
If the length of the segmentation is zero, no division will take place.

data: Shape(batch_num, seg_num, feat_dim)
indptr: Shape(seg_num + 1,)
*/
template<int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(WARP_SZ)
BatchDivSegLength(float* data, const int* indptr, int batch_num, int seg_num, int feat_dim) {
    int batch_id = blockIdx.z;
    int c_begin = blockIdx.y * UNROLL_X * WARP_SZ;
    for (int seg_id = blockIdx.x; seg_id < seg_num; seg_id += gridDim.x) {
        int ind_end = indptr[seg_id + 1];
        int ind_begin = indptr[seg_id];
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c = c_begin + i * WARP_SZ + threadIdx.x;
            if (c < feat_dim && ind_end > ind_begin) {
                data[IND3(batch_id, seg_id, c, seg_num, feat_dim)] /= (ind_end - ind_begin);
            }
        }
    }
}

/*Take the sum/mean/max/min of the data within the segments

dst_value: Shape (batch_num, seg_num, feat_dim)
dst_index: Shape (batch_num, seg_num, feat_dim)
data: Shape (batch_num, total_ind_num, feat_dim)
indices: Shape (nnz, )
indptr: Shape (seg_num + 1, )

for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        if max or min:
            initialize dst_index to -1
        for j = indptr[i]  to indptr[i+1] - 1
            if sum:
                dst_value[k, i, :] += data[k, indices[j], :]
            else if max:
                if(dst_value[k, i, :] > data[k, indices[j], :]
                    dst_value[k, i, :] = data[k, indices[j], :]
                    dst_index[k, i, :] = indices[j]
            else if min:
                if(dst_value[k, i, :] < data[k, indices[j], :]
                    dst_value[k, i, :] = data[k, indices[j], :]
                    dst_index[k, i, :] = indices[j]
*/
template<int reduce_type, int UNROLL_X = 4, int WARP_SZ = 32>
__global__ void
__launch_bounds__(WARP_SZ)
SegPoolKernel(float* dst_value, int* dst_index,
                              const float* data, const int* indices, const int* indptr,
                              int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz) {
    int batch_id = blockIdx.z;
    int c_begin = blockIdx.y * UNROLL_X * WARP_SZ;
    float dst_value_local[UNROLL_X];
    int dst_index_local[UNROLL_X];
    for(int seg_id = blockIdx.x; seg_id < seg_num; seg_id += gridDim.x) {
        int ind_begin = indptr[seg_id];
        int ind_end = indptr[seg_id + 1];
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            if(reduce_type == SegReduceType::kSum || reduce_type == SegReduceType::kMean) {
                dst_value_local[i] = 0;
            } else if(reduce_type == SegReduceType::kMax) {
                if(ind_end == ind_begin) {
                    dst_value_local[i] = 0;
                } else {
                    dst_value_local[i] = -FLT_MAX;
                }
                dst_index_local[i] = -1;
            } else if(reduce_type == SegReduceType::kMin) {
                if(ind_end == ind_begin) {
                    dst_value_local[i] = 0;
                } else {
                    dst_value_local[i] = FLT_MAX;
                }
                dst_index_local[i] = -1;
            }
        }
        for(int j = ind_begin; j < ind_end; j++) {
            int data_ind = indices[j];
            // Perform the reduction       
            #pragma unroll
            for(int i = 0; i < UNROLL_X; i++) {
                int c = c_begin + i * WARP_SZ + threadIdx.x;
                if(c < feat_dim) {
                    float data_val = data[IND3(batch_id, data_ind, c, total_ind_num, feat_dim)];
                    if (reduce_type == SegReduceType::kSum || reduce_type == SegReduceType::kMean) {
                        dst_value_local[i] += data_val;
                    } else if (reduce_type == SegReduceType::kMax) {
                        if(data_val > dst_value_local[i]) {
                            dst_value_local[i] = data_val;
                            dst_index_local[i] = j;
                        }
                    } else if (reduce_type == SegReduceType::kMin) {
                        if (data_val < dst_value_local[i]) {
                            dst_value_local[i] = data_val;
                            dst_index_local[i] = j;
                        }
                    }
                }
            }
        }
        if(reduce_type == SegReduceType::kMean) {
            #pragma unroll
            for(int i = 0; i < UNROLL_X; i++) {
                int c = c_begin + i * WARP_SZ + threadIdx.x;
                if (c < feat_dim && ind_end - ind_begin > 0) {
                    dst_value_local[i] /= (ind_end - ind_begin);
                }
            }
        }
        #pragma unroll
        for(int i = 0; i < UNROLL_X; i++) {
            int c = c_begin + i * WARP_SZ + threadIdx.x;
            if(c < feat_dim) {
                int dst_ind = IND3(batch_id, seg_id, c, seg_num, feat_dim);
                dst_value[dst_ind] = dst_value_local[i];
                if(reduce_type == SegReduceType::kMax || reduce_type == SegReduceType::kMin) {
                    dst_index[dst_ind] = dst_index_local[i];
                }
            }
        }
    }
}

/*Backward pass of the SegPool operator when sum is used

dst: Shape (batch_num, total_ind_num, feat_dim)
g_out: Shape(batch_num, seg_num, feat_dim)
out_index: Shape (batch_num, seg_num, feat_dim)
sorted_indices : Shape (nnz,)
sorted_orig_inds: Shape (nnz, )
seg_ids: Shape(nnz,)
indptr: Shape (seg_num + 1, )

for k = 0 to batch_num - 1
    for i = 0 to seg_num - 1
        for j = indptr[i]  to indptr[i+1] - 1
            if sum:
                dst[k, indices[j], :] += g_out[k, i, :]
            elif mean:
                dst[k, indices[j], :] += g_out[k, i, :] / (indptr[i+1] - indptr[i])
            else:
                dst[k, indices[j], :] += g_out[k, i, :] * (out_index[k, i, :] == indices[j])
Sorted Case ==>
for k = 0 to batch_num - 1
    for i = 0 to nnz - 1
        dst_ind = sorted_indices[i] --> indices[j]
        orig_ind = sorted_orig_inds[i] --> j
        seg_id = seg_ids[orig_ind] --> i
        if sum:
            dst[k, dst_ind, :] += g_out[k, seg_id, :]
        elif mean:
            dst[k, dst_ind, :] += g_out[k, seg_id, :] / (indptr[seg_id + 1] - indptr[seg_id])
        else:
            orig_ind = sorted_orig_inds[i]  --> j
            dst[k, dst_ind, :] += g_out[k, seg_id, :] * (out_index[k, seg_id, :] == orig_ind)

*/
template<int reduce_type, int UNROLL_X = 4, int TX_SZ = 32>
__global__ void
__launch_bounds__(TX_SZ)
SegPoolBackwardKernel(float* dst, const float* g_out, const int* out_index, const int* sorted_indices, const int* sorted_orig_inds, const int* seg_ids,
                      const int* indptr, int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz) {
  int k = blockIdx.z;
  int c_begin = blockIdx.y * TX_SZ * UNROLL_X; // We deal with a bunch of channels and scan through the neighboring nodes to write to the destination
  float dst_local[UNROLL_X];
  int idx = blockIdx.x;
  if (idx < nnz && (idx == 0 || sorted_indices[idx] != sorted_indices[idx - 1])) {
    const int dst_ind = sorted_indices[idx];
    #pragma unroll
    for (int i = 0; i < UNROLL_X; i++) {
      int c = c_begin + i * TX_SZ + threadIdx.x;
      if (c < feat_dim) {
        dst_local[i] = dst[IND3(k, dst_ind, c, total_ind_num, feat_dim)];
      }
    }
    do {
      const int orig_ind = sorted_orig_inds[idx];
      const int seg_id = seg_ids[orig_ind];
      #pragma unroll
      for(int i = 0; i < UNROLL_X; i++) {
          int c = c_begin + i * TX_SZ + threadIdx.x;
          if (c < feat_dim) {
            if(reduce_type == SegReduceType::kSum) {
              dst_local[i] += g_out[IND3(k, seg_id, c, seg_num, feat_dim)];
            } else if(reduce_type == SegReduceType::kMean) {
              dst_local[i] += g_out[IND3(k, seg_id, c, seg_num, feat_dim)] / (indptr[seg_id + 1] - indptr[seg_id]);
            } else {
              dst_local[i] += g_out[IND3(k, seg_id, c, seg_num, feat_dim)] * (out_index[IND3(k, seg_id, c, seg_num, feat_dim)] == orig_ind);
            }
          }
      }
      idx++;
    } while (idx < nnz && (sorted_indices[idx] == sorted_indices[idx - 1]));
    #pragma unroll
    for (int i = 0; i < UNROLL_X; i++) {
      int c = c_begin + i * TX_SZ + threadIdx.x;
      if (c < feat_dim) {
        dst[IND3(k, dst_ind, c, total_ind_num, feat_dim)] = dst_local[i];
      }
    }
  }
}

struct SegPoolCUDA {
    static size_t get_sort_temp_bytes(int nnz) {
      size_t temp_storage_bytes = 0;
      void* d_temp_storage = nullptr;
      int* d_keys_in = nullptr;
      int* d_keys_out = nullptr;
      int* d_values_in = nullptr;
      int* d_values_out = nullptr;
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, nnz);
      return temp_storage_bytes;
    }

    static size_t get_temp_bytes_backward(int nnz) {
      size_t temp_storage_bytes = get_sort_temp_bytes(nnz); // Tempspace for sorting
      temp_storage_bytes += GetSegId::get_temp_bytes(nnz);
      temp_storage_bytes += sizeof(int) * nnz; // seg_ids
      temp_storage_bytes += sizeof(int) * nnz; // sorted_indices
      temp_storage_bytes += sizeof(int) * nnz; // temp_ind_in
      temp_storage_bytes += sizeof(int) * nnz; // sorted_orig_inds
      return temp_storage_bytes;
    }
  
    template<int reduce_type>
    static void compute(float* dst_value, int* dst_index, const float* data, const int* indices, const int* indptr,
                        int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz, cudaStream_t stream) {
        static const int UNROLL_X = 4;
        static const int WARP_SZ = 32;
        dim3 dimBlock(WARP_SZ);
        dim3 dimGrid(seg_num, CEIL_DIV(feat_dim, WARP_SZ * UNROLL_X), batch_num);
        SegPoolKernel<reduce_type, UNROLL_X, WARP_SZ> <<<dimGrid, dimBlock, 0, stream >>> (dst_value, dst_index, data, indices, indptr, batch_num, seg_num, feat_dim, total_ind_num, nnz);
        SEG_CUDA_POST_KERNEL_CHECK(SegPoolKernel);
    }

    template<int reduce_type, bool add_to>
    static void compute_grad_data(float* dst, const float* g_out, const int* out_index, const int* indices, const int* indptr,
                                  int batch_num, int seg_num, int feat_dim, int total_ind_num, int nnz,
                                  char* temp_storage, size_t temp_storage_bytes, cudaStream_t stream) {
        if(!add_to) {
            cudaMemsetAsync(dst, 0, sizeof(float) * batch_num * total_ind_num * feat_dim, stream);
        }
        int* seg_ids = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* sorted_indices = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* temp_ind_in = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        int* sorted_orig_inds = reinterpret_cast<int*>(temp_storage);
        temp_storage += sizeof(int) * nnz;
        // 1. Sort the indices
        std::pair<dim3, dim3> block_grid_dim3 = KernelLauchParamB1G1(nnz);
        IdxArrayKernel << <block_grid_dim3.second, block_grid_dim3.first, 0, stream >> > (temp_ind_in, nnz);
        size_t temp_sort_bytes = get_sort_temp_bytes(nnz);
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_sort_bytes,
          indices, sorted_indices,
          temp_ind_in, sorted_orig_inds, nnz, 0, sizeof(int) * 8, stream);
        temp_storage += get_sort_temp_bytes(nnz);
        // 2. Compute the rev mapping
        GetSegId::compute(seg_ids, indptr, seg_num, nnz, temp_storage, GetSegId::get_temp_bytes(nnz), stream);
        // 3. Run the kernel
        static const int UNROLL_X = 4;
        static const int TX_SZ = 32;
        dim3 dimBlock(TX_SZ);
        dim3 dimGrid(nnz, CEIL_DIV(feat_dim, TX_SZ * UNROLL_X), batch_num);
        SegPoolBackwardKernel<reduce_type, UNROLL_X, TX_SZ> << <dimGrid, dimBlock, 0, stream >> > (dst, g_out, out_index, sorted_indices,
          sorted_orig_inds, seg_ids, indptr, batch_num, seg_num, feat_dim, total_ind_num, nnz);
        SEG_CUDA_POST_KERNEL_CHECK(SegPoolBackwardKernel);
    }
};

template<int pool_type>
void SegPoolImpl(const Tensor<gpu, 3, float> &dst_value,
                 const Tensor<gpu, 3, int> &pool_indices,
                 const Tensor<gpu, 3, float> &data,
                 const Tensor<gpu, 1, int> &indices,
                 const Tensor<gpu, 1, int> &indptr,
                 const OpReqType req,
                 const OpContext &ctx,
                 Stream<gpu>* s) {
  using namespace mxnet_op;
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "Not supported!";
  int batch_num = data.shape_[0];
  int total_ind_num = data.shape_[1];
  int feat_dim = data.shape_[2];
  int seg_num = dst_value.shape_[1];
  int nnz = indices.shape_[0];
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  if(pool_type == SegReduceType::kSum) {
    SegPoolCUDA::compute<SegReduceType::kSum>(dst_value.dptr_, pool_indices.dptr_, data.dptr_, indices.dptr_, indptr.dptr_,
                                              batch_num, seg_num, feat_dim, total_ind_num, nnz, stream);
  } else if(pool_type == SegReduceType::kMean) {
    SegPoolCUDA::compute<SegReduceType::kMean>(dst_value.dptr_, pool_indices.dptr_, data.dptr_, indices.dptr_, indptr.dptr_,
                                               batch_num, seg_num, feat_dim, total_ind_num, nnz, stream);
  } else if(pool_type == SegReduceType::kMax) {
    SegPoolCUDA::compute<SegReduceType::kMax>(dst_value.dptr_, pool_indices.dptr_, data.dptr_, indices.dptr_, indptr.dptr_,
                                              batch_num, seg_num, feat_dim, total_ind_num, nnz, stream);
  } else {
    LOG(FATAL) << "Unsupported!";
  }
}

template<int pool_type>
void SegPoolBackwardImpl(const Tensor<gpu, 3, float> &dst,
                         const Tensor<gpu, 3, float> &ograd,
                         const Tensor<gpu, 3, int> &out_index,
                         const Tensor<gpu, 1, int> &indices,
                         const Tensor<gpu, 1, int> &indptr,
                         const OpReqType req,
                         const OpContext &ctx,
                         Stream<gpu>* s) {
  using namespace mxnet_op;
  if (req == kNullOp) return;
  int batch_num = dst.shape_[0];
  int total_ind_num = dst.shape_[1];
  int feat_dim = dst.shape_[2];
  int nnz = indices.shape_[0];
  int seg_num = ograd.shape_[1];
  cudaStream_t stream = Stream<gpu>::GetStream(s);
  size_t temp_storage_bytes = SegPoolCUDA::get_temp_bytes_backward(nnz);
  Tensor<gpu, 1, char> workspace = ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
  const int* out_index_ptr = nullptr;
  if(pool_type == SegReduceType::kMax) {
    out_index_ptr = out_index.dptr_;
  }
  if (req == kAddTo) {
    SegPoolCUDA::compute_grad_data<pool_type, true>(dst.dptr_, ograd.dptr_, out_index_ptr, indices.dptr_, indptr.dptr_,
                                                    batch_num, seg_num, feat_dim, total_ind_num, nnz,
                                                    workspace.dptr_, temp_storage_bytes, stream);
  } else {
    SegPoolCUDA::compute_grad_data<pool_type, false>(dst.dptr_, ograd.dptr_, out_index_ptr, indices.dptr_, indptr.dptr_,
                                                    batch_num, seg_num, feat_dim, total_ind_num, nnz,
                                                    workspace.dptr_, temp_storage_bytes, stream);
  }
}
}  // namespace seg_op

NNVM_REGISTER_OP(_contrib_seg_sum)
.set_attr<FCompute>("FCompute<gpu>", SegReduceForward<gpu, seg_op::SegReduceType::kSum>);

NNVM_REGISTER_OP(_contrib__backward_seg_sum)
.set_attr<FCompute>("FCompute<gpu>", SegBroadcastToForward<gpu>);

NNVM_REGISTER_OP(_contrib_seg_broadcast_add)
.set_attr<FCompute>("FCompute<gpu>", SegBroadcastBinaryForward<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_contrib_seg_broadcast_mul)
.set_attr<FCompute>("FCompute<gpu>", SegBroadcastBinaryForward<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_contrib_seg_broadcast_to)
.set_attr<FCompute>("FCompute<gpu>", SegBroadcastToForward<gpu>);

NNVM_REGISTER_OP(_contrib_seg_softmax)
.set_attr<FCompute>("FCompute<gpu>", SegSoftmaxForward<gpu>);

NNVM_REGISTER_OP(_contrib__backward_seg_softmax)
.set_attr<FCompute>("FCompute<gpu>", SegSoftmaxBackward<gpu>);

NNVM_REGISTER_OP(_contrib_seg_take_k_corr)
.set_attr<FCompute>("FCompute<gpu>", SegTakeKCorrForward<gpu>);

NNVM_REGISTER_OP(_contrib_seg_weighted_pool)
.set_attr<FCompute>("FCompute<gpu>", SegWeightedPoolForward<gpu>);

NNVM_REGISTER_OP(_contrib__backward_seg_take_k_corr_embed2)
.set_attr<FCompute>("FCompute<gpu>", SegTakeKCorrBackwardEmbed2<gpu>);

NNVM_REGISTER_OP(_contrib_seg_pool)
.set_attr<FCompute>("FCompute<gpu>", SegPoolForward<gpu>);

NNVM_REGISTER_OP(_contrib__backward_seg_sum_mean_pool)
.set_attr<FCompute>("FCompute<gpu>", SegSumMeanPoolBackward<gpu>);

NNVM_REGISTER_OP(_contrib__backward_seg_max_pool)
.set_attr<FCompute>("FCompute<gpu>", SegMaxPoolBackward<gpu>);
}  // namespace op
}  // namespace mxnet
