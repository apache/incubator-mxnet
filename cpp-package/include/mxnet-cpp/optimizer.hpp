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
* \file optimizer.hpp
* \brief implementation of optimizer
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_CPP_OPTIMIZER_HPP_
#define MXNET_CPP_OPTIMIZER_HPP_

#include <dmlc/strtonum.h>
#include <algorithm>
#include <utility>
#include <numeric>
#include <map>
#include <cmath>
#include <string>
#include <vector>
#include "mxnet-cpp/optimizer.h"
#include "mxnet-cpp/op.h"
#include "mxnet-cpp/op_map.h"

namespace {

// TODO(lx75249): Add imperative operators to op.h under ndarray namespace
inline void _clip(mxnet::cpp::NDArray &data, float limit) {
  data = mxnet::cpp::Operator("clip")
    .SetParam("a_min", -limit)
    .SetParam("a_max", limit)
    .SetInput("data", data)
    .Invoke()[0];
}
inline mxnet::cpp::NDArray _sqrt(mxnet::cpp::NDArray data) {
  return mxnet::cpp::Operator("sqrt")
    .SetInput("data", data)
    .Invoke()[0];
}

}  // namespace

namespace mxnet {
namespace cpp {
inline Optimizer::Optimizer(unsigned begin_num_update)
  : begin_num_update_(begin_num_update),
    num_update_(begin_num_update_) {
  params_["lr"] = "0.01f";
  params_["wd"] = "0.f";
}

inline std::map<std::string, OptimizerCreator>& OptimizerRegistry::cmap() {
  static std::map<std::string, OptimizerCreator> cmap_;
  return cmap_;
}

inline OpMap*& Optimizer::op_map() {
  static OpMap *op_map_ = new OpMap();
  return op_map_;
}

inline Optimizer::~Optimizer() {}

inline void Optimizer::CreateState_(int index, NDArray weight) {
}

inline std::string Optimizer::Serialize() const {
  using ValueType = std::map<std::string, std::string>::value_type;
  auto params = params_;
  params.emplace("opt_type", GetType());
  return std::accumulate(params.cbegin(), params.cend(), std::string(""),
    [](const std::string& sum, const ValueType& i) {
      return sum + '\n' + i.first + '=' + i.second;
    }).substr(1);
}

inline const std::vector<const char*> Optimizer::GetParamKeys_() const {
  std::vector<const char*> keys;
  for (auto& iter : params_)
    keys.push_back(iter.first.c_str());
  return keys;
}

inline const std::vector<const char*> Optimizer::GetParamValues_() const {
  std::vector<const char*> values;
  for (auto& iter : params_)
    values.push_back(iter.second.c_str());
  return values;
}

inline unsigned Optimizer::UpdateCount_(int index) {
  if (count_.count(index) == 0) {
    count_.emplace(index, begin_num_update_);
  }
  unsigned new_count = ++count_[index];
  num_update_ = std::max(num_update_, new_count);
  return new_count;
}

inline float Optimizer::GetLR_(int index) {
  if (nullptr != lrScheduler_) {
    return lrScheduler_->GetLR(num_update_);
  }
  return dmlc::stof(params_["lr"]);
}

inline float Optimizer::GetWD_(int index) {
  float wd = dmlc::stof(params_["wd"]);
  return wd;
}

inline Optimizer* OptimizerRegistry::Find(const std::string& name) {
  if (cmap().empty()) {
    // Optimizers should only be registered once
    MXNETCPP_REGISTER_OPTIMIZER(sgd, SGDOptimizer);
    MXNETCPP_REGISTER_OPTIMIZER(ccsgd, SGDOptimizer);  // For backward compatibility
    MXNETCPP_REGISTER_OPTIMIZER(rmsprop, RMSPropOptimizer);
    MXNETCPP_REGISTER_OPTIMIZER(adam, AdamOptimizer);
    MXNETCPP_REGISTER_OPTIMIZER(adagrad, AdaGradOptimizer);
    MXNETCPP_REGISTER_OPTIMIZER(adadelta, AdaDeltaOptimizer);
    MXNETCPP_REGISTER_OPTIMIZER(signum, SignumOptimizer);
  }
  auto it = cmap().find(name);
  if (it == cmap().end())
    return nullptr;
  return it->second();
}

inline int OptimizerRegistry::__REGISTER__(const std::string& name, OptimizerCreator creator) {
  CHECK_EQ(cmap().count(name), 0) << name << " already registered";
  cmap().emplace(name, std::move(creator));
  return 0;
}

inline SGDOptimizer::SGDOptimizer(unsigned begin_num_update)
  : Optimizer(begin_num_update) {
  update_handle_ = op_map()->GetSymbolCreator("sgd_update");
  mom_update_handle_ = op_map()->GetSymbolCreator("sgd_mom_update");
}

inline std::string SGDOptimizer::GetType() const {
  return "sgd";
}

inline SGDOptimizer::~SGDOptimizer() {
  for (auto &it : states_) {
    delete it.second;
  }
}

inline void SGDOptimizer::Update(int index, NDArray weight, NDArray grad) {
  if (states_.count(index) == 0) {
    CreateState_(index, weight);
  }

  params_["lr"] = std::to_string(GetLR_(index));
  params_["wd"] = std::to_string(GetWD_(index));
  UpdateCount_(index);
  auto keys = GetParamKeys_();
  auto values = GetParamValues_();
  CHECK_EQ(keys.size(), values.size());

  NDArrayHandle inputs[3];
  inputs[0] = weight.GetHandle();
  inputs[1] = grad.GetHandle();

  int num_outputs = 1;
  NDArrayHandle output = weight.GetHandle();
  NDArrayHandle *outputs = &output;

  if (states_[index] == nullptr) {
    MXImperativeInvoke(update_handle_, 2, inputs,
        &num_outputs, &outputs,
        keys.size(), keys.data(), values.data(), nullptr);
  } else {
    inputs[2] = states_[index]->GetHandle();
    MXImperativeInvoke(mom_update_handle_, 3, inputs,
        &num_outputs, &outputs,
        keys.size(), keys.data(), values.data(), nullptr);
  }
}

inline void SGDOptimizer::CreateState_(int index, NDArray weight) {
  if (params_.count("momentum") == 0) {
    states_[index] = nullptr;
  } else {
    states_[index] = new NDArray(weight.GetShape(), weight.GetContext());
    *states_[index] = 0;
  }
}

// inplementing Signum optimizer

inline SignumOptimizer::SignumOptimizer(unsigned begin_num_update)
  : Optimizer(begin_num_update) {
  update_handle_ = op_map()->GetSymbolCreator("signsgd_update");
  mom_update_handle_ = op_map()->GetSymbolCreator("signum_update");
}

inline std::string SignumOptimizer::GetType() const {
  return "signum";
}

inline SignumOptimizer::~SignumOptimizer() {
  for (auto &it : states_) {
    delete it.second;
  }
}

inline void SignumOptimizer::Update(int index, NDArray weight, NDArray grad) {
  if (states_.count(index) == 0) {
    CreateState_(index, weight);
  }

  params_["lr"] = std::to_string(GetLR_(index));
  params_["wd"] = std::to_string(GetWD_(index));
  UpdateCount_(index);
  auto keys = GetParamKeys_();
  auto values = GetParamValues_();
  CHECK_EQ(keys.size(), values.size());

  NDArrayHandle inputs[3];
  inputs[0] = weight.GetHandle();
  inputs[1] = grad.GetHandle();

  int num_outputs = 1;
  NDArrayHandle output = weight.GetHandle();
  NDArrayHandle *outputs = &output;

  if (states_[index] == nullptr) {
    MXImperativeInvoke(update_handle_, 2, inputs,
        &num_outputs, &outputs,
        keys.size(), keys.data(), values.data(), nullptr);
  } else {
    inputs[2] = states_[index]->GetHandle();
    MXImperativeInvoke(mom_update_handle_, 3, inputs,
        &num_outputs, &outputs,
        keys.size(), keys.data(), values.data(), nullptr);
  }
}

inline void SignumOptimizer::CreateState_(int index, NDArray weight) {
  if (params_.count("momentum") == 0) {
    states_[index] = nullptr;
  } else {
    states_[index] = new NDArray(weight.GetShape(), weight.GetContext());
    *states_[index] = 0;
  }
}

// finish implementing Signum



inline RMSPropOptimizer::RMSPropOptimizer(unsigned begin_num_update)
  : Optimizer(begin_num_update) {
  update_handle_ = op_map()->GetSymbolCreator("rmsprop_update");
  alex_update_handle_ = op_map()->GetSymbolCreator("rmspropalex_update");
  SetParam("gamma1", 0.9f);
  SetParam("gamma2", 0.9f);
  SetParam("epsilon", 1e-8);
}

inline std::string RMSPropOptimizer::GetType() const {
  return "rmsprop";
}

inline RMSPropOptimizer::~RMSPropOptimizer() {
  for (auto &it : n_) {
    delete it.second;
  }
  for (auto &it : g_) {
    delete it.second;
  }
  for (auto &it : delta_) {
    delete it.second;
  }
}

inline void RMSPropOptimizer::Update(int index, NDArray weight, NDArray grad) {
  if (n_.count(index) == 0) {
    CreateState_(index, weight);
  }

  params_["lr"] = std::to_string(GetLR_(index));
  params_["wd"] = std::to_string(GetWD_(index));
  UpdateCount_(index);
  auto keys = GetParamKeys_();
  auto values = GetParamValues_();
  CHECK_EQ(keys.size(), values.size());

  NDArrayHandle inputs[5];
  inputs[0] = weight.GetHandle();
  inputs[1] = grad.GetHandle();
  inputs[2] = n_[index]->GetHandle();
  inputs[3] = g_[index]->GetHandle();
  inputs[4] = delta_[index]->GetHandle();

  int num_outputs = 1;
  NDArrayHandle output = weight.GetHandle();
  NDArrayHandle *outputs = &output;

  MXImperativeInvoke(alex_update_handle_, 5, inputs,
      &num_outputs, &outputs,
      keys.size(), keys.data(), values.data(), nullptr);
}

inline void RMSPropOptimizer::CreateState_(int index, NDArray weight) {
  n_[index] = new NDArray(weight.GetShape(), weight.GetContext());
  *n_[index] = 0;
  g_[index] = new NDArray(weight.GetShape(), weight.GetContext());
  *g_[index] = 0;
  delta_[index] = new NDArray(weight.GetShape(), weight.GetContext());
  *delta_[index] = 0;
}

inline AdamOptimizer::AdamOptimizer(unsigned begin_num_update)
  : Optimizer(begin_num_update) {
  update_handle_ = op_map()->GetSymbolCreator("adam_update");
  SetParam("beta1", 0.9f);
  SetParam("beta2", 0.999f);
  SetParam("epsilon", 1e-8);
}

inline std::string AdamOptimizer::GetType() const {
  return "adam";
}

inline AdamOptimizer::~AdamOptimizer() {
  for (auto &it : mean_) {
    delete it.second;
  }
  for (auto &it : var_) {
    delete it.second;
  }
}

inline void AdamOptimizer::Update(int index, NDArray weight, NDArray grad) {
  if (mean_.count(index) == 0) {
    CreateState_(index, weight);
  }

  params_["lr"] = std::to_string(GetLR_(index));
  params_["wd"] = std::to_string(GetWD_(index));
  UpdateCount_(index);
  auto keys = GetParamKeys_();
  auto values = GetParamValues_();
  CHECK_EQ(keys.size(), values.size());

  float lr = dmlc::stof(params_["lr"]);
  float b1 = dmlc::stof(params_["beta1"]);
  float b2 = dmlc::stof(params_["beta2"]);
  float t = count_[index];
  float coef1 = 1.0f - std::pow(b1, t);
  float coef2 = 1.0f - std::pow(b2, t);
  lr *= std::sqrt(coef2) / coef1;

  NDArrayHandle inputs[4];
  inputs[0] = weight.GetHandle();
  inputs[1] = grad.GetHandle();

  int num_outputs = 1;
  NDArrayHandle output = weight.GetHandle();
  NDArrayHandle *outputs = &output;

  inputs[2] = mean_[index]->GetHandle();
  inputs[3] = var_[index]->GetHandle();

  MXImperativeInvoke(update_handle_, 4, inputs,
    &num_outputs, &outputs,
    keys.size(), keys.data(), values.data(), nullptr);
}

inline void AdamOptimizer::CreateState_(int index, NDArray weight) {
  mean_[index] = new NDArray(weight.GetShape(), weight.GetContext());
  *mean_[index] = 0;
  var_[index] = new NDArray(weight.GetShape(), weight.GetContext());
  *var_[index] = 0;
}

inline AdaGradOptimizer::AdaGradOptimizer(unsigned begin_num_update)
  : Optimizer(begin_num_update) {
  SetParam("eps", 1e-7);
}

inline std::string AdaGradOptimizer::GetType() const {
  return "adagrad";
}

inline void AdaGradOptimizer::Update(int index, NDArray weight, NDArray grad) {
  if (history_.count(index) == 0) {
    CreateState_(index, weight);
  }

  float eps = dmlc::stof(params_["eps"]);
  float lr = GetLR_(index);
  float wd = GetWD_(index);
  UpdateCount_(index);
  if (params_.count("rescale_grad") > 0) {
    grad *= dmlc::stof(params_["rescale_grad"]);
  }
  if (params_.count("clip_gradient") > 0) {
    _clip(grad, dmlc::stof(params_["clip_gradient"]));
  }
  auto& history = *history_[index];
  history += grad * grad;
  weight -= (grad / _sqrt(history + eps) + weight * wd) * lr;
}

inline AdaGradOptimizer::~AdaGradOptimizer() {
  for (auto& it : history_) {
    delete it.second;
  }
}

inline void AdaGradOptimizer::CreateState_(int index, NDArray weight) {
  history_[index] = new NDArray(weight.GetShape(), weight.GetContext());
  *history_[index] = 0;
}

inline AdaDeltaOptimizer::AdaDeltaOptimizer(unsigned begin_num_update)
  : Optimizer(begin_num_update) {
  SetParam("rho", 0.90f);
  SetParam("epsilon", 1e-5);
}

inline std::string AdaDeltaOptimizer::GetType() const {
  return "adadelta";
}

inline void AdaDeltaOptimizer::Update(int index, NDArray weight, NDArray grad) {
  if (acc_g_.count(index) == 0) {
    CreateState_(index, weight);
  }

  float rho = dmlc::stof(params_["rho"]);
  float epsilon = dmlc::stof(params_["epsilon"]);
  float wd = GetWD_(index);
  UpdateCount_(index);

  if (params_.count("rescale_grad") > 0) {
    grad *= dmlc::stof(params_["rescale_grad"]);
  }
  if (params_.count("clip_gradient") > 0) {
    _clip(grad, dmlc::stof(params_["clip_gradient"]));
  }

  auto& acc_g = *acc_g_[index];
  auto& acc_delta = *acc_delta_[index];
  acc_g *= rho;
  acc_g += grad * grad * (1.0f - rho);

  auto delta = _sqrt(acc_delta + epsilon) / _sqrt(acc_g + epsilon) * grad;
  acc_delta *= rho;
  acc_delta += delta * delta * (1.0f - rho);
  weight *= 1.0f - wd;
  weight -= delta;
}

inline AdaDeltaOptimizer::~AdaDeltaOptimizer() {
  for (auto& it : acc_g_) {
    delete it.second;
  }
  for (auto& it : acc_delta_) {
    delete it.second;
  }
}

inline void AdaDeltaOptimizer::CreateState_(int index, NDArray weight) {
  acc_g_[index] = new NDArray(weight.GetShape(), weight.GetContext());
  *acc_g_[index] = 0;
  acc_delta_[index] = new NDArray(weight.GetShape(), weight.GetContext());
  *acc_delta_[index] = 0;
}

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_OPTIMIZER_HPP_
