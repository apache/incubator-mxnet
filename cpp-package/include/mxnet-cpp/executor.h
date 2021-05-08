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
* \file executor.h
* \brief executor definition
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_CPP_EXECUTOR_H_
#define MXNET_CPP_EXECUTOR_H_

#include <vector>
#include <map>
#include <set>
#include <string>
#include "mxnet-cpp/base.h"
#include "mxnet-cpp/symbol.h"

namespace mxnet {
namespace cpp {

class Optimizer;

/*!
* \brief Executor interface
*/
class Executor {
 public:
  Executor(const Symbol &symbol, Context context,
           const std::vector<NDArray> &arg_arrays,
           const std::vector<NDArray> &grad_arrays,
           const std::vector<OpReqType> &grad_reqs,
           const std::vector<NDArray> &aux_arrays,
           const std::map<std::string, Context> &group_to_ctx =
               std::map<std::string, Context>(),
           Executor *shared_exec = nullptr);
  explicit Executor(const CachedOpHandle &h) { handle_ = h; }
  /*!
  * \brief Perform a Forward operation of Operator
  *  After this operation, user can get the result by using function head.
  */
  void Forward(bool is_train) {
    std::vector<NDArrayHandle> arg_handles;
    for (const auto &array : combined_arrays) {
      arg_handles.push_back(array.GetHandle());
    }
    int prev_is_record = 0;
    int prev_train_mode = 0;
    CHECK_EQ(MXAutogradSetIsRecording(1, &prev_is_record), 0);
    if (is_train == true) {
      CHECK_EQ(MXAutogradSetIsTraining(1, &prev_train_mode), 0);
    }
    std::vector<NDArrayHandle> output_handles;
    std::transform(outputs.begin(), outputs.end(),
        std::back_inserter(output_handles), [](NDArray& a) {
          return a.GetHandle();
        });
    int out_size = 0;
    NDArrayHandle *out_array = nullptr;
    CHECK_EQ(MXInvokeCachedOp(handle_, arg_handles.size(), arg_handles.data(),
                              device_type, device_id, &out_size, &out_array, nullptr),
             0);
    outputs.clear();
    outputs.reserve(out_size);
    for (mx_uint i = 0; i < out_size; ++i) {
      outputs.push_back(NDArray(out_array[i]));
    }
    int cur_train_mode = prev_train_mode;
    int cur_is_record = prev_is_record;
    if (is_train == true) {
      CHECK_EQ(MXAutogradSetIsTraining(cur_train_mode, &prev_train_mode), 0);
    }
    CHECK_EQ(MXAutogradSetIsRecording(cur_is_record, &prev_is_record), 0);
  }
  /*!
  * \brief Perform a Backward operation of the Operator.
  *  This must be called after Forward.
  *  After this operation, NDArrays specified by grad_in_args_store will be
  *updated accordingly.
  *  User is allowed to pass in an empty Array if the head node is
  *  loss function and head gradeitn is not needed.
  *
  * \param head_grads the gradient of head nodes to be backproped.
  */
  void Backward(const std::vector<NDArray> &head_grads =
                    std::vector<NDArray>()) {
    if (require_grad == true) {
      if (outputs.size() == 0) {
        Forward(false);
      }
      std::vector<NDArrayHandle> out_handles;
      for (const auto &array : outputs) {
        out_handles.push_back(array.GetHandle());
      }
                          
      std::vector<NDArrayHandle> head_grads_;
      for (auto d : head_grads) {
        head_grads_.push_back(d.GetHandle());
      }
      if (head_grads_.size() > 0) {
        CHECK_EQ(MXAutogradBackwardEx(out_handles.size(), out_handles.data(),
                                      head_grads_.data(), 0, nullptr, 0, 0, 1,
                                      nullptr, nullptr), 0);
      } else {
        CHECK_EQ(MXAutogradBackwardEx(out_handles.size(), out_handles.data(),
                                      nullptr, 0, nullptr, 0, 0, 1,
                                      nullptr, nullptr), 0);
      }
      grad_arrays.clear();
      grad_arrays.reserve(arg_arrays.size());
      for (const auto &array : arg_arrays) {
        NDArrayHandle grad;
        CHECK_EQ(MXNDArrayGetGrad(array.GetHandle(), &grad),0);
        grad_arrays.push_back(NDArray(grad));
      }
    }
  }
  // TODO(zhangchen-qinyinghua)
  // To implement reshape function
  void Reshape();
  /*!
  * \brief destructor, free the handle
  */
  ~Executor() { MXFreeCachedOp(handle_); }
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<NDArray> aux_arrays;
  std::vector<NDArray> combined_arrays;
  int device_type;
  int device_id;
  bool require_grad;
  /*!
  * \brief arrays store the outputs of forward
  */
  std::vector<NDArray> outputs;
  std::map<std::string, NDArray> arg_dict() {
    return GetDict(symbol_.ListArguments(), arg_arrays);
  }
  std::map<std::string, NDArray> grad_dict() {
    return GetDict(symbol_.ListArguments(), grad_arrays);
  }
  std::map<std::string, NDArray> aux_dict() {
    return GetDict(symbol_.ListAuxiliaryStates(), aux_arrays);
  }

 private:
  Executor(const Executor &e);
  Executor &operator=(const Executor &e);
  CachedOpHandle handle_;
  Symbol symbol_;
  std::map<std::string, NDArray> GetDict(const std::vector<std::string> &names,
                                         const std::vector<NDArray> &arrays) {
    std::map<std::string, NDArray> ret;
    std::set<std::string> name_set;
    for (const auto &s : names) {
      CHECK(name_set.find(s) == name_set.end()) << "Duplicate names detected, "
                                                << s;
      name_set.insert(s);
    }
    CHECK_EQ(name_set.size(), arrays.size())
        << "names size not equal to arrays size";
    for (size_t i = 0; i < names.size(); ++i) {
      ret[names[i]] = arrays[i];
    }
    return ret;
  }
};
}  // namespace cpp
}  // namespace mxnet
#endif  // MXNET_CPP_EXECUTOR_H_
