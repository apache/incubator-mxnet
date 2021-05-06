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
 * \file storage.h
 * \brief Storage manager across multiple devices.
 */
#ifndef MXNET_STORAGE_H_
#define MXNET_STORAGE_H_

#include <memory>
#include "./base.h"

namespace mxnet {

/*!
 * \brief Storage manager across multiple devices.
 */
class Storage {
 public:
  /*!
   * \brief Storage handle.
   */
  struct Handle {
    /*!
     * \brief Pointer to the data.
     */
    void* dptr{nullptr};
    /*!
     * \brief Size of the storage.
     */
    size_t size{0};
    /*!
     * \brief Context information about device and ID.
     */
    Context ctx;
    /*!
     * \brief Id for IPC shared memory
     */
    int shared_pid{-1};
    int shared_id{-1};
  };
  /*!
   * \brief Allocate a new contiguous memory for a given size.
   * \param size Total size of memory in bytes.
   * \param ctx Context information about the device and ID.
   * \return Handle struct.
   */
  Handle Alloc(size_t size, Context ctx) {
    Handle hd;
    hd.size = size;
    hd.ctx = ctx;
    this->Alloc(&hd);
    return hd;
  }

  virtual size_t GetMemoryInUseInBytes(const Context &ctx) = 0;

  /*!
   * \brief Allocate a new contiguous memory for a given size.
   * \param handle handle initialized with size and ctx
   */
  virtual void Alloc(Handle* handle) = 0;
  /*!
   * \brief Increase ref counter on shared memory.
   * \param handle handle to shared memory.
   */
  virtual void SharedIncrementRefCount(Handle handle) = 0;
  /*!
   * \brief Free storage.
   * \param handle Handle struct.
   */
  virtual void Free(Handle handle) = 0;
  /*!
   * \brief Free storage directly, without putting it into memory pool.
   *  This can synchronization of all previous runned device functions.
   *
   *  This function is suitable for conatiner structure with requirement on upsizing
   *  in the beginning phase of the iteration.
   *
   * \param handle Handle struct.
   */
  virtual void DirectFree(Handle handle) = 0;
  /*!
  * \brief Release all memory from device if using a pooled storage manager
  *
  * This release all memory from pool storage managers such as
  * GPUPooledStorageManager and GPUPooledRoundedStorageManager.
  * For non-pool memory managers this has no effect.
  */
  virtual void ReleaseAll(Context ctx) = 0;
  /*!
   * \brief Destructor.
   */
  virtual ~Storage() {}
  /*!
   * \brief Returns mutex used by storage manager
   */
  std::mutex& GetMutex(Context::DeviceType dev) {
    if (dev == Context::kCPU) {
      return cpu_mutex_;
    } else {
      return gpu_mutex_;
    }
  }
  /*!
   * \return Storage singleton.
   */
  static Storage* Get();
  /*!
   * \brief Get shared pointer reference to storage singleton.
   *  Most user should not call this function.
   *  This function is called by another singleton X who requires
   *  Storage to be destructed after X.
   *
   * \return A shared pointer to Storage singleton.
   */
  static std::shared_ptr<Storage> _GetSharedRef();

 private:
  std::mutex cpu_mutex_;
  std::mutex gpu_mutex_;
};  // class Storage
}  // namespace mxnet
#endif  // MXNET_STORAGE_H_
