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
*  Copyright (c) 2016 by Contributors
* \file kvstore.h
* \brief definition of kvstore
* \author Chuntao Hong
*/

#ifndef MXNET_CPP_KVSTORE_H_
#define MXNET_CPP_KVSTORE_H_

#include <string>
#include <vector>
#include "mxnet-cpp/ndarray.h"

namespace mxnet {
namespace cpp {

class KVStore {
 public:
  static void SetType(const std::string& type);
  static void RunServer();
  static void Init(int key, const NDArray& val);
  static void Init(const std::string& key, const NDArray& val);
  static void Init(const std::vector<int>& keys, const std::vector<NDArray>& vals);
  static void Init(const std::vector<std::string>& keys, const std::vector<NDArray>& vals);
  static void Push(int key, const NDArray& val, int priority = 0);
  static void Push(const std::string& key, const NDArray& val, int priority = 0);
  static void Push(const std::vector<int>& keys,
                   const std::vector<NDArray>& vals, int priority = 0);
  static void Push(const std::vector<std::string>& keys,
                   const std::vector<NDArray>& vals, int priority = 0);
  static void Pull(int key, NDArray* out, int priority = 0);
  static void Pull(const std::string& key, NDArray* out, int priority = 0);
  static void Pull(const std::vector<int>& keys,
                   std::vector<NDArray>* outs, int priority = 0);
  static void Pull(const std::vector<std::string>& keys,
                   std::vector<NDArray>* outs, int priority = 0);
  // TODO(lx): put lr in optimizer or not?
  static void SetOptimizer(std::unique_ptr<Optimizer> optimizer, bool local = false);
  static std::string GetType();
  static int GetRank();
  static int GetNumWorkers();
  static void Barrier();
  static std::string GetRole();

 private:
  KVStore();
  static KVStoreHandle& get_handle();
  static std::unique_ptr<Optimizer>& get_optimizer();
  static KVStore*& get_kvstore();
  static void Controller(int head, const char* body, void* controller_handle);
  static void Updater(int key, NDArrayHandle recv, NDArrayHandle local, void* handle_);
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_KVSTORE_H_
