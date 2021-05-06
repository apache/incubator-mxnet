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
 * \file pooled_storage_manager.h
 * \brief Storage manager with a memory pool.
 */
#ifndef MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
#define MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_

#if MXNET_USE_CUDA
  #include <cuda_runtime.h>
#endif  // MXNET_USE_CUDA

#include <mxnet/base.h>
#include <mxnet/storage.h>
#include <unordered_map>
#include <algorithm>
#include <atomic>
#include <vector>
#include <mutex>
#include <new>
#include "./storage_manager.h"
#include "../common/cuda_utils.h"
#include "../common/utils.h"


namespace mxnet {
namespace storage {

#if MXNET_USE_CUDA
/*!
 * \brief Storage manager with a memory pool on gpu. Memory chunks are reused based on exact size
 * match.
 */
class GPUPooledStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   *
   * \param initial_context context used by this Storage Manager
   */
  explicit GPUPooledStorageManager(Context initial_context) :
  free_list_size_(0), initial_context_(initial_context) {
    reserve_ = dmlc::GetEnv<double>("MXNET_GPU_MEM_POOL_RESERVE", 5.0);
    page_size_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_PAGE_SIZE", 4096);
    large_alloc_round_size_ = dmlc::GetEnv("MXNET_GPU_MEM_LARGE_ALLOC_ROUND_SIZE", 2 * 1024 * 1024);
    memory_limit_percentage_ = dmlc::GetEnv<double>("MXNET_GPU_MEM_POOL_LIMIT", 100.0);
    if (large_alloc_round_size_ <= 0)
      LOG(FATAL) << "MXNET_GPU_MEM_LARGE_ALLOC_ROUND_SIZE cannot be set to a value <= 0, found: "
                 << large_alloc_round_size_;

    if (page_size_ < NDEV)
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_PAGE_SIZE cannot be set to a value smaller than " << NDEV \
                 << ". Got " << page_size_ << ".";

    if (memory_limit_percentage_ <= 0 || memory_limit_percentage_ > 100)
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_LIMIT invalid: " << memory_limit_percentage_
                 << " must be 0 <= x < 100" << std::endl;
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledStorageManager() {
    ReleaseAll();
  }

  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override;
  size_t GetMemoryInUseInBytes() override;

  void DirectFree(Storage::Handle handle) override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    DirectFreeNoLock(handle);
  }

  void ReleaseAll() override;

 private:
  void DirectFreeNoLock(Storage::Handle handle) {
    mxnet::common::cuda::DeviceStore device_store(handle.ctx.real_dev_id(), true);
    cudaError_t err = cudaFree(handle.dptr);
    size_t size = RoundAllocSize(handle.size);
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
    used_memory_ -= size;
  }

  // Round a value 'x' up to the next multiple of 'multiple'
  size_t RoundToMultiple(size_t x, size_t multiple) {
    size_t retVal = ((x + multiple - 1) / multiple) * multiple;
    return retVal;
  }

  size_t RoundAllocSize(size_t size) {
    // Round up small allocs to multiple of page_size_ to consolidate the pool lookups
    size = RoundToMultiple(size, page_size_);
    // To ensure proper freeing under some driver variants, make sure
    // large allocs entirely occupy their slabs, which cannot then be
    // locked by smaller permanent allocations sharing the slab.
    if (size > large_alloc_round_size_)
      size = RoundToMultiple(size, large_alloc_round_size_);
    return size;
  }

 private:
  // used memory
  size_t used_memory_ = 0;
  // page size
  size_t page_size_;
  // size that large allocations should be rounded to, for proper freeing.
  size_t large_alloc_round_size_;
  // amount of free memory in pool
  size_t free_list_size_;
  // percentage of GPU memory to use
  double memory_limit_percentage_;
  // percentage of reserved memory
  double reserve_;
  // number of devices
  const size_t NDEV = 32;
  // context used by this Storage Manager
  const Context initial_context_;
  // memory pool
  std::unordered_map<size_t, std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledStorageManager);
};  // class GPUPooledStorageManager

void GPUPooledStorageManager::Alloc(Storage::Handle* handle) {
  // Set dptr to nullptr when handle size is 0.
  if (handle->size == 0) {
    handle->dptr = nullptr;
    return;
  }

  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t size = RoundAllocSize(handle->size);
  // try to find array in the pool
  auto&& reuse_it = memory_pool_.find(size);
  if (reuse_it == memory_pool_.end() || reuse_it->second.size() == 0) {
    // if cant find array of right size in the pool...
    mxnet::common::cuda::DeviceStore device_store(handle->ctx.real_dev_id(), true);
    // get memory info from CUDA
    size_t real_free, total;
    cudaMemGetInfo(&real_free, &total);
    // compute artificial limit
    double mem_limit_in_bytes = total * memory_limit_percentage_ / 100.0;
    size_t calc_free = mem_limit_in_bytes - used_memory_;
    // amount of free memory is from calc if there is more real free memory
    size_t free = (calc_free < real_free) ? calc_free : real_free;
    // release pool memory if theres not enough to alloc this array
    if (free <= mem_limit_in_bytes * reserve_ / 100.0 ||
        size > free - mem_limit_in_bytes * reserve_ / 100.0)
      ReleaseAll();

    // enforce artificial memory limit
    if (used_memory_ + size > mem_limit_in_bytes)
      LOG(FATAL) << "memory limit reached, used: " << used_memory_ << "  limit: "
                 << mem_limit_in_bytes << " (" << memory_limit_percentage_ << "% of "
                 << total << ")";

    // try allocating a new array with CUDA
    void* ret = nullptr;
    cudaError_t e = cudaMalloc(&ret, size);
    if (e != cudaSuccess) {
      if (e == cudaErrorMemoryAllocation) {
        ReleaseAll();
        e = cudaMalloc(&ret, size);
        if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
          LOG(FATAL) << "cudaMalloc retry failed: " << cudaGetErrorString(e);
        }
      } else if (e != cudaErrorCudartUnloading) {
        LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
      }
    }
    used_memory_ += size;
    handle->dptr = ret;
  } else {
    // reuse array from pool
    auto&& reuse_pool = reuse_it->second;
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    // decrement the free list size as memory pool is getting used.
    free_list_size_ -= size;
    handle->dptr = ret;
  }
}

void GPUPooledStorageManager::Free(Storage::Handle handle) {
  // This method doesn't actually free array
  // but instead adds to the pool for re-use later


  // Do nothing if dptr is nullptr. Otherwise, nullptr may be reused
  // which can cause illegal memory access error.
  if (handle.dptr == nullptr) return;

  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  size_t size = RoundAllocSize(handle.size);
  auto&& reuse_pool = memory_pool_[size];
  reuse_pool.push_back(handle.dptr);
  // increment free list size as memory pool is growing
  free_list_size_ += size;
}

void GPUPooledStorageManager::ReleaseAll() {
  for (auto&& i : memory_pool_) {
    for (auto&& j : i.second) {
      Storage::Handle handle;
      handle.dptr = j;
      handle.size = i.first;
      handle.ctx = initial_context_;
      DirectFreeNoLock(handle);
    }
  }
  memory_pool_.clear();
  // clear list size as memory pool is now empty
  free_list_size_ = 0;
}

/*!
 * \brief Storage manager with a memory pool, with rounded size, on gpu.
 *
 * This GPU mem pool uses a mixture of nearest pow2 (exponential) rounding and
 * nearest multiple (linear) rounding to help alleviate the memory allocation stress
 * in which the default naive exact-size-match pool falls short, such as in variable-length
 * input/output cases like RNN workloads.
 *
 * \param cutoff the cutoff at which rounding is switched from exponential to linear. It's set
 * through MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF environment variable. Must be between 20 (1 MB)
 * and 34 (16 GB).
 * Suppose the cutoff is X, the memory size buckets look like this:
 * exp2(0), exp2(1), ..., exp2(X), 2*exp2(X), 3*exp2(X), ...
 */
class GPUPooledRoundedStorageManager final : public StorageManager {
 public:
  /*!
   * \brief Default constructor.
   *
   * \param initial_context context used by this Storage Manager
   */
  explicit GPUPooledRoundedStorageManager(Context initial_context) :
    initial_context_(initial_context) {
    reserve_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_RESERVE", 5);
    page_size_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_PAGE_SIZE", 4096);
    cut_off_ = dmlc::GetEnv("MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF", 24);
    if (page_size_ < 32) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_PAGE_SIZE cannot be set to a value smaller than 32. " \
                 << "Got: " << page_size_ << ".";
    }
    if (page_size_ != 1ul << common::ilog2ul(page_size_ - 1)) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_PAGE_SIZE must be a power of 2. Got: " << page_size_ << ".";
    }
    page_size_ = common::ilog2ul(page_size_ - 1);
    if (cut_off_ < 20 || cut_off_ > LOG2_MAX_MEM) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF cannot be set to a value " \
                 << "smaller than 20 or greater than " << LOG2_MAX_MEM << ". Got: " \
                 << cut_off_ << ".";
    }
    if (cut_off_ < page_size_) {
      LOG(FATAL) << "MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF cannot be set to a value " \
                 << "smaller than log2 of MXNET_GPU_MEM_POOL_PAGE_SIZE. Got: " \
                 << cut_off_ << " vs " << page_size_ << ".";
    }
    memory_pool_ = std::vector<std::vector<void*>>((1ul << (LOG2_MAX_MEM - cut_off_)) + cut_off_);
  }
  /*!
   * \brief Default destructor.
   */
  ~GPUPooledRoundedStorageManager() {
    ReleaseAll();
  }

  void Alloc(Storage::Handle* handle) override;
  void Free(Storage::Handle handle) override;

  void DirectFree(Storage::Handle handle) override {
    std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
    DirectFreeNoLock(handle);
  }

  void ReleaseAll() override;

 private:
  inline int div_pow2_round_up(size_t s, int divisor_log2) {
    // (1025, 10) -> 2
    // (2048, 10) -> 2
    // (2049, 10) -> 3
    size_t result = s >> divisor_log2;
    return static_cast<int>(result + (s > (result << divisor_log2) ? 1 : 0));
  }
  inline int get_bucket(size_t s) {
    int log_size = common::ilog2ul(s - 1);
    if (log_size > static_cast<int>(cut_off_))
      return div_pow2_round_up(s, cut_off_) - 1 + cut_off_;
    else
      return std::max(log_size, static_cast<int>(page_size_));
  }
  inline size_t get_size(int bucket) {
    if (bucket <= static_cast<int>(cut_off_))
      return 1ul << bucket;
    else
      return (bucket - cut_off_ + 1) * (1ul << cut_off_);
  }

  void DirectFreeNoLock(Storage::Handle handle) {
    mxnet::common::cuda::DeviceStore device_store(handle.ctx.real_dev_id(), true);
    cudaError_t err = cudaFree(handle.dptr);
    size_t size = get_size(get_bucket(handle.size));
    // ignore unloading error, as memory has already been recycled
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
      LOG(FATAL) << "CUDA: " << cudaGetErrorString(err);
    }
    used_memory_ -= size;
  }

 private:
  // number of devices
  const int NDEV = 32;
  // log2 of maximum page size. 16GB
  const size_t LOG2_MAX_MEM = 34;
  // address width in bits
  static const int addr_width = sizeof(size_t) * 8;
  // used memory
  size_t used_memory_ = 0;
  // page size
  size_t page_size_;
  // log2 of memory size before switching to exponential mode to linear mode
  size_t cut_off_;
  // percentage of reserved memory
  int reserve_;
  // context used by this Storage Manager
  const Context initial_context_;
  // memory pool
  std::vector<std::vector<void*>> memory_pool_;
  DISALLOW_COPY_AND_ASSIGN(GPUPooledRoundedStorageManager);
};  // class GPUPooledRoundedStorageManager

void GPUPooledRoundedStorageManager::Alloc(Storage::Handle* handle) {
  // Set dptr to nullptr when handle size is 0.
  if (handle->size == 0) {
    handle->dptr = nullptr;
    return;
  }

  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  int bucket = get_bucket(handle->size);
  size_t size = get_size(bucket);
  auto&& reuse_pool = memory_pool_[bucket];
  if (reuse_pool.size() == 0) {
    mxnet::common::cuda::DeviceStore device_store(handle->ctx.real_dev_id(), true);
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    if (free <= total * reserve_ / 100 || size > free - total * reserve_ / 100)
      ReleaseAll();

    void* ret = nullptr;
    cudaError_t e = cudaMalloc(&ret, size);
    if (e != cudaSuccess) {
      if (e == cudaErrorMemoryAllocation) {
        ReleaseAll();
        e = cudaMalloc(&ret, size);
        if (e != cudaSuccess && e != cudaErrorCudartUnloading) {
          LOG(FATAL) << "cudaMalloc retry failed: " << cudaGetErrorString(e);
        }
      } else if (e != cudaErrorCudartUnloading) {
        LOG(FATAL) << "cudaMalloc failed: " << cudaGetErrorString(e);
      }
    }
    used_memory_ += size;
    handle->dptr = ret;
  } else {
    auto ret = reuse_pool.back();
    reuse_pool.pop_back();
    handle->dptr = ret;
  }
}

void GPUPooledRoundedStorageManager::Free(Storage::Handle handle) {
  // Do nothing if dptr is nullptr. Otherwise, nullptr may be reused
  // which can cause illegal memory access error.
  if (handle.dptr == nullptr) return;

  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  int bucket = get_bucket(handle.size);
  auto&& reuse_pool = memory_pool_[bucket];
  reuse_pool.push_back(handle.dptr);
}

void GPUPooledRoundedStorageManager::ReleaseAll() {
  for (size_t i = 0; i < memory_pool_.size(); i++) {
    int size = get_size(i);
    for (auto& j : memory_pool_[i]) {
      Storage::Handle handle;
      handle.size = size;
      handle.dptr = j;
      handle.ctx = initial_context_;
      DirectFreeNoLock(handle);
    }
    memory_pool_[i].clear();
  }
}

size_t GPUPooledStorageManager::GetMemoryInUseInBytes() {
  std::lock_guard<std::mutex> lock(Storage::Get()->GetMutex(Context::kGPU));
  return used_memory_ - free_list_size_;
}

#endif  // MXNET_USE_CUDA

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_POOLED_STORAGE_MANAGER_H_
