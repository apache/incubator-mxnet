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
 * \file kvstore.hpp
 * \brief implementation of kvstore
 * \author Xin Li
 */

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "mxnet-cpp/kvstore.h"
#include "mxnet-cpp/optimizer.h"

#ifndef MXNET_CPP_KVSTORE_HPP_
#define MXNET_CPP_KVSTORE_HPP_

namespace mxnet {
namespace cpp {

inline void KVStore::Controller(int head, const char* body, void* controller_handle) {
  if (head == 0) {
    std::map<std::string, std::string> params;
    std::istringstream sin(body);
    std::string line;
    while (getline(sin, line)) {
      size_t n = line.find('=');
      params.emplace(line.substr(0, n), line.substr(n+1));
    }
    std::unique_ptr<Optimizer> opt(OptimizerRegistry::Find(params.at("opt_type")));
    params.erase("opt_type");
    for (const auto& pair : params) {
      opt->SetParam(pair.first, pair.second);
    }
    get_kvstore()->SetOptimizer(std::move(opt), true);
  }
}

inline KVStoreHandle& KVStore::get_handle() {
  static KVStoreHandle handle_ = nullptr;
  return handle_;
}

inline std::unique_ptr<Optimizer>& KVStore::get_optimizer() {
  static std::unique_ptr<Optimizer> optimizer_;
  return optimizer_;
}

inline KVStore*& KVStore::get_kvstore() {
  static KVStore* kvstore_ = new KVStore;
  return kvstore_;
}

inline KVStore::KVStore() {}

inline void KVStore::SetType(const std::string& type) {
  CHECK_EQ(MXKVStoreCreate(type.c_str(), &(get_kvstore()->get_handle())), 0);
}

inline void KVStore::RunServer() {
  CHECK_NE(GetRole(), "worker");
  CHECK_EQ(MXKVStoreRunServer(get_kvstore()->get_handle(), &Controller, 0), 0);
}

inline void KVStore::Init(int key, const NDArray& val) {
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStoreInit(get_kvstore()->get_handle(), 1, &key, &val_handle), 0);
}

inline void KVStore::Init(const std::string& key, const NDArray& val) {
  const char* key_handle = key.c_str();
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStoreInitEx(get_kvstore()->get_handle(), 1, &key_handle, &val_handle), 0);
}

inline void KVStore::Init(const std::vector<int>& keys, const std::vector<NDArray>& vals) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<NDArrayHandle> val_handles(vals.size());
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStoreInit(get_kvstore()->get_handle(), keys.size(), keys.data(),
      val_handles.data()), 0);
}

inline void KVStore::Init(const std::vector<std::string>& keys, const std::vector<NDArray>& vals) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<const char*> key_handles(keys.size());
  std::transform(keys.cbegin(), keys.cend(), key_handles.begin(),
      [](const std::string& key) {
        return key.c_str();
      });
  std::vector<NDArrayHandle> val_handles(vals.size());
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStoreInitEx(get_kvstore()->get_handle(), key_handles.size(), key_handles.data(),
      val_handles.data()), 0);
}

inline void KVStore::Push(int key, const NDArray& val, int priority) {
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStorePush(get_kvstore()->get_handle(), 1, &key, &val_handle, priority), 0);
}

inline void KVStore::Push(const std::string& key, const NDArray& val, int priority) {
  const char* key_handle = key.c_str();
  NDArrayHandle val_handle = val.GetHandle();
  CHECK_EQ(MXKVStorePushEx(get_kvstore()->get_handle(), 1, &key_handle, &val_handle, priority), 0);
}

inline void KVStore::Push(const std::vector<int>& keys,
                          const std::vector<NDArray>& vals, int priority) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<NDArrayHandle> val_handles(vals.size());
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePush(get_kvstore()->get_handle(), keys.size(), keys.data(),
      val_handles.data(), priority), 0);
}

inline void KVStore::Push(const std::vector<std::string>& keys,
                          const std::vector<NDArray>& vals, int priority) {
  CHECK_EQ(keys.size(), vals.size());
  std::vector<const char*> key_handles(keys.size());
  std::transform(keys.cbegin(), keys.cend(), key_handles.begin(),
      [](const std::string& key) {
        return key.c_str();
      });
  std::vector<NDArrayHandle> val_handles(vals.size());
  std::transform(vals.cbegin(), vals.cend(), val_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePushEx(get_kvstore()->get_handle(), key_handles.size(), key_handles.data(),
      val_handles.data(), priority), 0);
}

inline void KVStore::Pull(int key, NDArray* out, int priority) {
  NDArrayHandle out_handle = out->GetHandle();
  CHECK_EQ(MXKVStorePull(get_kvstore()->get_handle(), 1, &key, &out_handle, priority), 0);
}

inline void KVStore::Pull(const std::string& key, NDArray* out, int priority) {
  const char* key_handle = key.c_str();
  NDArrayHandle out_handle = out->GetHandle();
  CHECK_EQ(MXKVStorePullEx(get_kvstore()->get_handle(), 1, &key_handle, &out_handle, priority), 0);
}

inline void KVStore::Pull(const std::vector<int>& keys,
                          std::vector<NDArray>* outs, int priority) {
  CHECK_EQ(keys.size(), outs->size());

  std::vector<NDArrayHandle> out_handles(keys.size());
  std::transform(outs->cbegin(), outs->cend(), out_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePull(get_kvstore()->get_handle(), keys.size(), keys.data(),
      out_handles.data(), priority), 0);
}

inline void KVStore::Pull(const std::vector<std::string>& keys,
                          std::vector<NDArray>* outs, int priority) {
  CHECK_EQ(keys.size(), outs->size());

  std::vector<const char*> key_handles(keys.size());
  std::transform(keys.cbegin(), keys.cend(), key_handles.begin(),
      [](const std::string& key) {
        return key.c_str();
      });
  std::vector<NDArrayHandle> out_handles(keys.size());
  std::transform(outs->cbegin(), outs->cend(), out_handles.begin(),
      [](const NDArray& val) {
        return val.GetHandle();
      });

  CHECK_EQ(MXKVStorePullEx(get_kvstore()->get_handle(), key_handles.size(), key_handles.data(),
      out_handles.data(), priority), 0);
}

inline void KVStore::Updater(int key, NDArrayHandle recv, NDArrayHandle local,
                             void* handle_) {
  Optimizer *opt = static_cast<Optimizer*>(handle_);
  opt->Update(key, NDArray(local), NDArray(recv));
}

inline void KVStore::SetOptimizer(std::unique_ptr<Optimizer> optimizer, bool local) {
  if (local) {
    get_kvstore()->get_optimizer() = std::move(optimizer);
    CHECK_EQ(MXKVStoreSetUpdater(get_kvstore()->get_handle(),
                                 &Updater, get_kvstore()->get_optimizer().get()), 0);
  } else {
    CHECK_EQ(MXKVStoreSendCommmandToServers(get_kvstore()->get_handle(), 0,
                                            (*optimizer).Serialize().c_str()), 0);
  }
}

inline std::string KVStore::GetType() {
  const char *type;
  CHECK_EQ(MXKVStoreGetType(get_kvstore()->get_handle(), &type), 0);
  return type;
}

inline int KVStore::GetRank() {
  int rank;
  CHECK_EQ(MXKVStoreGetRank(get_kvstore()->get_handle(), &rank), 0);
  return rank;
}

inline int KVStore::GetNumWorkers() {
  int num_workers;
  CHECK_EQ(MXKVStoreGetGroupSize(get_kvstore()->get_handle(), &num_workers), 0);
  return num_workers;
}

inline void KVStore::Barrier() {
  CHECK_EQ(MXKVStoreBarrier(get_kvstore()->get_handle()), 0);
}

inline std::string KVStore::GetRole() {
  int ret;
  CHECK_EQ(MXKVStoreIsSchedulerNode(&ret), 0);
  if (ret) {
    return "scheduler";
  }
  CHECK_EQ(MXKVStoreIsServerNode(&ret), 0);
  if (ret) {
    return "server";
  }
  CHECK_EQ(MXKVStoreIsWorkerNode(&ret), 0);
  CHECK(ret);
  return "worker";
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_KVSTORE_HPP_
