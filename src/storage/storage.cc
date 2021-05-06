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
 */
#include <mxnet/storage.h>
#include "./storage_manager.h"
#include "./naive_storage_manager.h"
#include "./pooled_storage_manager.h"
#include "./cpu_shared_storage_manager.h"
#include "./cpu_device_storage.h"
#include "./gpu_device_storage.h"
#include "./pinned_memory_storage.h"
#include "../common/lazy_alloc_array.h"
#include "../profiler/storage_profiler.h"

namespace mxnet {

// consider change storage as a pure abstract class
class StorageImpl : public Storage {
 public:
  void Alloc(Handle* handle) override;
  void Free(Handle handle) override;
  void DirectFree(Handle handle) override;
  void ReleaseAll(Context ctx) override;
  size_t GetMemoryInUseInBytes(const Context &ctx) override;
  void SharedIncrementRefCount(Handle handle) override;
  StorageImpl() {}
  virtual ~StorageImpl() = default;

 private:
  static constexpr size_t kMaxNumberOfDevices = Context::kMaxDevType + 1;
#if MXNET_USE_CUDA
  static int num_gpu_device;
#endif  // MXNET_USE_CUDA

  // internal storage managers
  std::array<common::LazyAllocArray<storage::StorageManager>,
             kMaxNumberOfDevices> storage_managers_;
  storage::DeviceStorageProfiler profiler_;
};  // struct Storage::Impl
#if MXNET_USE_CUDA
int StorageImpl::num_gpu_device = 0;
#endif  // MXNET_USE_CUDA

void StorageImpl::Alloc(Storage::Handle* handle) {
  // space already recycled, ignore request
  auto&& device = storage_managers_.at(handle->ctx.dev_type);
  std::shared_ptr<storage::StorageManager> manager = device.Get(
      handle->ctx.real_dev_id(), [handle]() {
        storage::StorageManager *ptr = nullptr;
        switch (handle->ctx.dev_type) {
          case Context::kCPU: {
            ptr = new storage::NaiveStorageManager<storage::CPUDeviceStorage>();
            break;
          }
          case Context::kCPUShared: {
#if !defined(ANDROID) && !defined(__ANDROID__)
            ptr = new storage::CPUSharedStorageManager();
#else
            LOG(FATAL) << "Unimplemented device";
#endif  // !defined(ANDROID) && !defined(__ANDROID__)
            break;
          }
          case Context::kCPUPinned: {
#if MXNET_USE_CUDA
            num_gpu_device = 0;
            cudaError_t e = cudaGetDeviceCount(&num_gpu_device);
            if (e != cudaSuccess) {
              num_gpu_device = 0;
            }
            if (num_gpu_device > 0) {
              ptr = new storage::NaiveStorageManager<storage::PinnedMemoryStorage>();
            } else {
              ptr = new storage::NaiveStorageManager<storage::CPUDeviceStorage>();
            }
#else
            ptr = new storage::NaiveStorageManager<storage::CPUDeviceStorage>();
#endif  // MXNET_USE_CUDA
            break;
          }
          case Context::kGPU: {
#if MXNET_USE_CUDA
            CUDA_CALL(cudaGetDeviceCount(&num_gpu_device));
            CHECK_GT(num_gpu_device, 0) << "GPU usage requires at least 1 GPU";

            const char *type = getenv("MXNET_GPU_MEM_POOL_TYPE");
            const bool default_pool = (type == nullptr);
            if (default_pool) type = "Naive";
            std::string strategy = type;

            if (strategy == "Round") {
              ptr = new storage::GPUPooledRoundedStorageManager(handle->ctx);
              LOG(INFO) << "Using GPUPooledRoundedStorageManager.";
            } else if (strategy == "Naive") {
              ptr = new storage::GPUPooledStorageManager(handle->ctx);
            } else if (strategy == "Unpooled") {
              ptr = new storage::NaiveStorageManager<storage::GPUDeviceStorage>();
            } else {
              LOG(FATAL) << "Unknown memory pool strategy specified: " << strategy << ".";
            }
#else
            LOG(FATAL) << "Compile with USE_CUDA=1 to enable GPU usage";
#endif  // MXNET_USE_CUDA
            break;
          }
          default: LOG(FATAL) <<  "Unimplemented device " << handle->ctx.dev_type;
        }
        return ptr;
      });

  manager->Alloc(handle);
  profiler_.OnAlloc(*handle);
}

void StorageImpl::Free(Storage::Handle handle) {
  // Do nothing if dtpr is nullptr because the handle may have already
  // been freed or have not been allocated memory yet.
  if (handle.dptr == nullptr) return;

  const Context &ctx = handle.ctx;
  auto&& device = storage_managers_.at(ctx.dev_type);
  std::shared_ptr<storage::StorageManager> manager = device.Get(
      ctx.real_dev_id(), []() {
        LOG(FATAL) <<  "Cannot Free space to a device you have not allocated";
        return nullptr;
      });

  manager->Free(handle);
  profiler_.OnFree(handle);
}

size_t StorageImpl::GetMemoryInUseInBytes(const Context &ctx) {
  auto &&device = storage_managers_.at(ctx.dev_type);
  std::shared_ptr<storage::StorageManager> manager = device.Get(
    ctx.real_dev_id(), []() {
      LOG(WARNING) << "Cannot get memory usage for a device you have not allocated";
        return nullptr;
      });
  if (nullptr == manager)
    return 0;
  else
    return manager->GetMemoryInUseInBytes();
}

void StorageImpl::DirectFree(Storage::Handle handle) {
  // Do nothing if dtpr is nullptr because the handle may have already
  // been freed or have not been allocated memory yet.
  if (handle.dptr == nullptr) return;

  const Context &ctx = handle.ctx;
  auto&& device = storage_managers_.at(ctx.dev_type);
  std::shared_ptr<storage::StorageManager> manager = device.Get(
      ctx.real_dev_id(), []() {
        LOG(FATAL) <<  "Cannot Free space to a device you have not allocated";
        return nullptr;
      });

  manager->DirectFree(handle);
  profiler_.OnFree(handle);
}

void StorageImpl::ReleaseAll(Context ctx) {
  auto&& device = storage_managers_.at(ctx.dev_type);
  std::shared_ptr<storage::StorageManager> manager = device.Get(
    ctx.real_dev_id(), []() {
    LOG(FATAL) << "Cannot Free space to a device you have not allocated";
    return nullptr;
  });
  manager->ReleaseAll();
}

void StorageImpl::SharedIncrementRefCount(Storage::Handle handle) {
  CHECK_EQ(handle.ctx.dev_type, Context::kCPUShared);
  auto&& device = storage_managers_.at(Context::kCPUShared);
  auto manager = device.Get(0, []() {
      LOG(FATAL) << "Cannot increment ref count before allocating any shared memory.";
      return nullptr;
    });
#if defined(ANDROID) || defined(__ANDROID__)
  LOG(FATAL) << "Shared memory not implemented on Android";
#else
  dynamic_cast<storage::CPUSharedStorageManager*>(manager.get())->IncrementRefCount(handle);
#endif  // defined(ANDROID) || defined(__ANDROID__)
}

std::shared_ptr<Storage> Storage::_GetSharedRef() {
#ifdef __MXNET_JS__
  // dummy code needed for emscripten code to pass
  // do not know why, the new will be NULLPTR
  static int *q = new int();
#endif
  static std::shared_ptr<Storage> inst(new StorageImpl());
  return inst;
}

Storage* Storage::Get() {
  static Storage *ptr = _GetSharedRef().get();
  return ptr;
}
}  // namespace mxnet
