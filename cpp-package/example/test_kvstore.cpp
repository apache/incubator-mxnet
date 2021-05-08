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
#include "mxnet/c_api.h"  // MXGetGPUCount()
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

static bool test_single_key(const Context &context, const std::string &context_str) {
  std::string key = "singlekeytest-" + context_str;

  NDArray result(Shape(4), context);
  NDArray result_cpu;

  // initialize data
  NDArray data_cpu({0.f, 233.f, -0.12f, 9.f}, Shape(4), Context::cpu());
  NDArray data = data_cpu.Copy(context);
  NDArray::WaitAll();

  KVStore::Init(key, data);
  NDArray::WaitAll();

  // retrieve result
  KVStore::Pull(key, &result);
  NDArray::WaitAll();

  result_cpu = result.Copy(Context::cpu());
  NDArray::WaitAll();

  // compare
  for (size_t j=0; j < result_cpu.Size(); j++) {
    if (result_cpu.GetData()[j] != data_cpu.GetData()[j]) {
      LG << "Error: wrong initialized data in singlekeytest-" << context_str
          << ", expect " << data_cpu.GetData()[j]
          << " got " << result_cpu.GetData()[j];
      return false;
    }
  }

  // push gradient
  NDArray grad_cpu({0.1f, -2.f, -4.4f, 0.f}, Shape(4), Context::cpu());
  NDArray grad = grad_cpu.Copy(context);
  NDArray::WaitAll();

  KVStore::Push(key, grad);
  NDArray::WaitAll();

  // retrieve result
  KVStore::Pull(key, &result);
  NDArray::WaitAll();

  result_cpu = result.Copy(Context::cpu());
  NDArray::WaitAll();

  // compare
  for (size_t j=0; j < result_cpu.Size(); j++) {
    if (result_cpu.GetData()[j] != grad_cpu.GetData()[j]) {
      LG << "Error: wrong gradient data in singlekeytest-" << context_str
          << ", expect " << grad_cpu.GetData()[j]
          << " got " << result_cpu.GetData()[j];
      return false;
    }
  }

  return true;
}

static bool test_multiple_key(const Context &context, const std::string &context_str) {
  std::vector<std::string> keys(2);
  keys[0] = "multikeytest-0-" + context_str;
  keys[1] = "multikeytest-1-" + context_str;

  std::vector<NDArray> results(2);
  results[0] = NDArray(Shape(4), context);
  results[1] = NDArray(Shape(4), context);
  std::vector<NDArray> results_cpu(2);

  // initialize data
  std::vector<NDArray> data_cpu(2);
  data_cpu[0] = NDArray({0.f, 2.f, -3.12f, 4.f}, Shape(4), Context::cpu());
  data_cpu[1] = NDArray({0.8f, -2.f, 6.6f, 77.f}, Shape(4), Context::cpu());
  std::vector<NDArray> data(2);
  data[0] = data_cpu[0].Copy(context);
  data[1] = data_cpu[1].Copy(context);
  NDArray::WaitAll();

  KVStore::Init(keys, data);
  NDArray::WaitAll();

  // retrieve result
  KVStore::Pull(keys, &results);
  NDArray::WaitAll();

  results_cpu[0] = results[0].Copy(Context::cpu());
  results_cpu[1] = results[1].Copy(Context::cpu());
  NDArray::WaitAll();

  // compare
  for (size_t i=0; i < results_cpu.size(); i++) {
    for (size_t j=0; j < results_cpu[i].Size(); j++) {
      if (results_cpu[i].GetData()[j] != data_cpu[i].GetData()[j]) {
        LG << "Error: wrong initialized data in multikeytest-" << context_str
            << ", expect " << data_cpu[i].GetData()[j]
            << " got " << results_cpu[i].GetData()[j];
        return false;
      }
    }
  }

  // push gradient, reduce for the second
  std::vector<std::string> push_keys(3);
  push_keys[0] = "multikeytest-0-" + context_str;
  push_keys[1] = "multikeytest-1-" + context_str;
  push_keys[2] = "multikeytest-1-" + context_str;

  std::vector<NDArray> grads_cpu(3);
  grads_cpu[0] = NDArray({0.2f, -0.3f, -1.1f, 0.0f}, Shape(4), Context::cpu());
  grads_cpu[1] = NDArray({2.f, 4.f, -4.f, -5.f}, Shape(4), Context::cpu());
  grads_cpu[2] = NDArray({-3.f, -0.2f, 12.f, -9.f}, Shape(4), Context::cpu());
  std::vector<NDArray> grads(3);
  grads[0] = grads_cpu[0].Copy(context);
  grads[1] = grads_cpu[1].Copy(context);
  grads[2] = grads_cpu[2].Copy(context);
  NDArray::WaitAll();

  KVStore::Push(push_keys, grads);
  NDArray::WaitAll();

  // retrieve result
  KVStore::Pull(keys, &results);
  NDArray::WaitAll();

  results_cpu[0] = results[0].Copy(Context::cpu());
  results_cpu[1] = results[1].Copy(Context::cpu());
  NDArray::WaitAll();

  // compare the first
  for (size_t j=0; j < results_cpu[0].Size(); j++) {
    if (results_cpu[0].GetData()[j] != grads_cpu[0].GetData()[j]) {
      LG << "Error: wrong gradient data in multikeytest-" << context_str
          << ", expect " << grads_cpu[0].GetData()[j]
          << " got " << results_cpu[0].GetData()[j];
      return false;
    }
  }

  // compare the second
  for (size_t j=0; j < results_cpu[1].Size(); j++) {
    if (results_cpu[1].GetData()[j] != (grads_cpu[1].GetData()[j] + grads_cpu[2].GetData()[j])) {
      LG << "Error: wrong reduced gradient data in multikeytest-" << context_str
          << ", expect " << (grads_cpu[1].GetData()[j] + grads_cpu[2].GetData()[j])
          << " got " << results_cpu[1].GetData()[j];
      return false;
    }
  }

  return true;
}

int main(int argc, char** argv) {
  KVStore::SetType("local");

  bool success1 = test_single_key(Context::cpu(), "cpu");
  bool success2 = test_multiple_key(Context::cpu(), "cpu");

  bool success3 = true;
  bool success4 = true;

  int gpu_count = 0;
  if (MXGetGPUCount(&gpu_count) != 0) {
    LG << "Error: MXGetGPUCount";

    MXNotifyShutdown();
    return 1;
  }

  if (gpu_count > 0) {
    success3 = test_single_key(Context::gpu(), "gpu");
    success4 = test_multiple_key(Context::gpu(), "gpu");
  }

  int ret = (success1 && success2 && success3 && success4) ? 0 : 1;

  MXNotifyShutdown();
  return ret;
}
