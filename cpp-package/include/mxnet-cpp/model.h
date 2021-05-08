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
* \file model.h
* \brief MXNET.cpp model module
* \author Zhang Chen
*/

#ifndef MXNET_CPP_MODEL_H_
#define MXNET_CPP_MODEL_H_

#include <string>
#include <vector>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/symbol.h"
#include "mxnet-cpp/ndarray.h"

namespace mxnet {
namespace cpp {

struct FeedForwardConfig {
  Symbol symbol;
  std::vector<Context> ctx = {Context::cpu()};
  int num_epoch = 0;
  int epoch_size = 0;
  std::string optimizer = "sgd";
  // TODO(zhangchen-qinyinghua) More implement
  // initializer=Uniform(0.01),
  // numpy_batch_size=128,
  // arg_params=None, aux_params=None,
  // allow_extra_params=False,
  // begin_epoch=0,
  // **kwargs):
  FeedForwardConfig(const FeedForwardConfig &other) {}
  FeedForwardConfig() {}
};
class FeedForward {
 public:
  explicit FeedForward(const FeedForwardConfig &conf) : conf_(conf) {}
  void Predict();
  void Score();
  void Fit();
  void Save();
  void Load();
  static FeedForward Create();

 private:
  void InitParams();
  void InitPredictor();
  void InitIter();
  void InitEvalIter();
  FeedForwardConfig conf_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_MODEL_H_

