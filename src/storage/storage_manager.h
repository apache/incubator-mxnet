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
 * \file storage_manager.h
 * \brief Storage manager.
 */

#ifndef MXNET_STORAGE_STORAGE_MANAGER_H_
#define MXNET_STORAGE_STORAGE_MANAGER_H_

#include <mxnet/storage.h>
#include <cstddef>

namespace mxnet {
namespace storage {

/*!
 * \brief Storage manager interface.
 */
class StorageManager {
 public:
  /*!
   * \brief Allocation.
   * \param handle Handle struct.
   */
  virtual void Alloc(Storage::Handle* handle) = 0;
  /*!
   * \brief Deallocation.
   * \param handle Handle struct.
   */
  virtual void Free(Storage::Handle handle) = 0;
  /*!
   * \brief Direct deallocation.
   * \param handle Handle struct.
   */
  virtual void DirectFree(Storage::Handle handle) = 0;

  virtual size_t GetMemoryInUseInBytes() {
    LOG(FATAL) << "GetMemoryInUseInBytes API is not implemented";
    return 0;
  }
  /*!
  * \brief Release all memory if using a pool storage manager
  *
  * This release all memory from pool storage managers such as
  * GPUPooledStorageManager and GPUPooledRoundedStorageManager.
  * For non-pool memory managers this has no effect.
  */
  virtual void ReleaseAll() {}
  /*!
   * \brief Destructor.
   */
  virtual ~StorageManager() = default;
};  // namespace StorageManager

}  // namespace storage
}  // namespace mxnet

#endif  // MXNET_STORAGE_STORAGE_MANAGER_H_
