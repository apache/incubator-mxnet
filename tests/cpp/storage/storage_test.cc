/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * Copyright (c) 2017 by Contributors
 * \file storage_test.cc
 * \brief cpu/gpu storage tests
*/
#include <stdlib.h>
#include <gtest/gtest.h>
#include <dmlc/logging.h>
#include <mxnet/storage.h>
#include <cstdio>
#include "test_util.h"

TEST(Storage, Basic_CPU) {
  constexpr size_t kSize = 1024;
  auto&& storage = mxnet::Storage::Get();
  mxnet::Context context_cpu{};
  auto&& handle = storage->Alloc(kSize, context_cpu);
  EXPECT_EQ(handle.ctx, context_cpu);
  EXPECT_EQ(handle.size, kSize);
  storage->Free(handle);

  handle = storage->Alloc(kSize, context_cpu);
  EXPECT_EQ(handle.ctx, context_cpu);
  EXPECT_EQ(handle.size, kSize);
  storage->Free(handle);

  handle = storage->Alloc(0, context_cpu);
  EXPECT_EQ(handle.dptr, nullptr);
  storage->Free(handle);
}

#if MXNET_USE_CUDA
TEST(Storage_GPU, Basic_GPU) {
  if (mxnet::test::unitTestsWithCuda) {
    putenv("MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF=20");
    putenv("MXNET_GPU_MEM_POOL_TYPE=Round");

    auto &&storage = mxnet::Storage::Get();
    mxnet::Context context_gpu = mxnet::Context::GPU(0);
    auto &&handle = storage->Alloc(32, context_gpu);
    auto &&handle2 = storage->Alloc(2097153, context_gpu);
    EXPECT_EQ(handle.ctx, context_gpu);
    EXPECT_EQ(handle.size, 32);
    EXPECT_EQ(handle2.ctx, context_gpu);
    EXPECT_EQ(handle2.size, 2097153);
    auto ptr = handle.dptr;
    auto ptr2 = handle2.dptr;
    storage->Free(handle);
    storage->Free(handle2);

    handle = storage->Alloc(4095, context_gpu);
    EXPECT_EQ(handle.ctx, context_gpu);
    EXPECT_EQ(handle.size, 4095);
    EXPECT_EQ(handle.dptr, ptr);
    storage->Free(handle);

    handle2 = storage->Alloc(3145728, context_gpu);
    EXPECT_EQ(handle2.ctx, context_gpu);
    EXPECT_EQ(handle2.size, 3145728);
    EXPECT_EQ(handle2.dptr, ptr2);
    storage->Free(handle2);

    handle = storage->Alloc(0, context_gpu);
    EXPECT_EQ(handle.dptr, nullptr);
    storage->Free(handle);

    unsetenv("MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF");
    unsetenv("MXNET_GPU_MEM_POOL_TYPE");
  }
  if (mxnet::test::unitTestsWithCuda) {
    constexpr size_t kSize = 1024;
    mxnet::Context context_gpu = mxnet::Context::GPU(0);
    auto &&storage = mxnet::Storage::Get();
    size_t used_mem_in_bytes = storage->GetMemoryInUseInBytes(context_gpu);
    auto &&handle = storage->Alloc(kSize, context_gpu);
    assert(handle.ctx == context_gpu);
    assert(handle.size == kSize);
    assert(storage->GetMemoryInUseInBytes(context_gpu) >= used_mem_in_bytes + kSize);
    auto ptr = handle.dptr;
    storage->Free(handle);

    handle = storage->Alloc(kSize, context_gpu);
    EXPECT_EQ(handle.ctx, context_gpu);
    EXPECT_EQ(handle.size, kSize);
    EXPECT_EQ(handle.dptr, ptr);
    storage->Free(handle);

    handle = storage->Alloc(0, context_gpu);
    EXPECT_EQ(handle.dptr, nullptr);
    storage->Free(handle);
  }
}
#endif  // MXNET_USE_CUDA

