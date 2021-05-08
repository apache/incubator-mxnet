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
 * \file executor.hpp
 * \brief implementation of the executor
 * \author Zhang Chen, Chuntao Hong
 */

#ifndef MXNET_CPP_EXECUTOR_HPP_
#define MXNET_CPP_EXECUTOR_HPP_

#include <vector>
#include <map>
#include <string>
#include "mxnet-cpp/executor.h"
#include "mxnet-cpp/optimizer.h"


namespace mxnet {
namespace cpp {
inline Executor::Executor(const Symbol &symbol, Context context,
                          const std::vector<NDArray> &arg_arrays,
                          const std::vector<NDArray> &grad_arrays,
                          const std::vector<OpReqType> &grad_reqs,
                          const std::vector<NDArray> &aux_arrays,
                          const std::map<std::string, Context> &group_to_ctx,
                          Executor *shared_exec) {
  this->arg_arrays = arg_arrays;
  this->grad_arrays = grad_arrays;
  this->aux_arrays = aux_arrays;
  this->symbol_ = symbol;
  this->device_type = context.GetDeviceType();
  this->device_id = context.GetDeviceId();

  std::vector<NDArrayHandle> arg_handles;
  std::vector<NDArrayHandle> grad_handles;

  CHECK_EQ(arg_arrays.size(), grad_arrays.size())
      << "Number of input arg_arrays is different from the number of input grad_arrays";
  for (int i = 0; i < arg_arrays.size(); i++) {
    if (grad_arrays[i].GetShape().size() != 0) {
      grad_handles.push_back(grad_arrays[i].GetHandle());
      arg_handles.push_back(arg_arrays[i].GetHandle());
    }
  }

  this->require_grad = false;
  std::vector<mx_uint> grad_reqs_uint;
  for (auto s : grad_reqs) {
    if (s != OpReqType::kNullOp) {
      this->require_grad = true;
    }
    grad_reqs_uint.push_back(s);
  }
  CHECK_EQ(MXAutogradMarkVariables(arg_handles.size(), arg_handles.data(),
                                   grad_reqs_uint.data(), grad_handles.data()),0);

  std::map<std::string, NDArray> arg_map = arg_dict();
  std::map<std::string, NDArray> aux_map = aux_dict();
  const auto input_name_list = symbol_.ListInputs();
  std::vector<NDArray> combined_arrays;
  for (size_t i = 0; i < input_name_list.size(); ++i) {
    const auto &input_name = input_name_list[i];
    auto iter_arg = arg_map.find(input_name);
    if (iter_arg != arg_map.end()) {
      combined_arrays.push_back(iter_arg->second);
    } else {
      auto iter_aux = aux_map.find(input_name);
      CHECK(iter_aux != aux_map.end())
          << "Can not find name in args array and aux array";
      combined_arrays.push_back(iter_aux->second);
    }
  }
  this->combined_arrays = combined_arrays;

  CHECK_EQ(MXCreateCachedOp(symbol.GetHandle(), 0, nullptr, nullptr, &handle_, false), 0);
}


}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_EXECUTOR_HPP_
