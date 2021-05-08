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
* \file optimizer.h
* \brief definition of optimizer
* \author Chuntao Hong, Zhang Chen
*/

#ifndef MXNET_CPP_OPTIMIZER_H_
#define MXNET_CPP_OPTIMIZER_H_

#include <dmlc/strtonum.h>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "mxnet-cpp/base.h"
#include "dmlc/logging.h"
#include "mxnet-cpp/ndarray.h"
#include "mxnet-cpp/op_map.h"
#include "mxnet-cpp/lr_scheduler.h"

namespace mxnet {
namespace cpp {

/*!
* \brief Optimizer interface
*/
class Optimizer {
 public:
  /*!
  * \brief constructor
  * \param beign_num_update The initial number of updates
  */
  explicit Optimizer(unsigned begin_num_update);
  /*!
  * \brief get optimizer type
  * \return string of optimizer type
  */
  virtual std::string GetType() const = 0;
  /*!
  * \brief destructor
  */
  virtual ~Optimizer();
  /*!
  * \brief set config parameters
  * \param name name of the config parameter
  * \param value value of the config parameter
  * \return reference of self
  */
  template <typename T>
  Optimizer *SetParam(const std::string &name, const T &value) {
    std::string value_str;
    std::stringstream ss;
    ss << value;
    ss >> value_str;

    params_[name] = value_str;
    return this;
  }
  /*!
  * \bried set the lr scheduler
  * \param lrScheduler lr scheduler used for this optimizer
  * \return reference if self
  */
  Optimizer *SetLRScheduler(std::unique_ptr<LRScheduler> lrScheduler) {
    CHECK(lrScheduler);
    lrScheduler_ = std::move(lrScheduler);
    lrScheduler_->SetLR(dmlc::stof(params_["lr"]));
    return this;
  }
  /*!
  *  \brief Update a weight with gradient.
  *  \param index the unique index for the weight.
  *  \param weight the weight to update.
  *  \param grad gradient for the weight.
  */
  virtual void Update(int index, NDArray weight, NDArray grad) = 0;
  // TODO(zhangcheng-qinyinghua)
  // implement Update a list of arrays, maybe in the form of map
  // void Update(int index, std::vector<NDArray> weights, std::vector<NDArray>
  // grad, mx_float lr);

  /*!
  *  \brief Serialize the optimizer parameters to a string.
  *  \return serialization
  */
  std::string Serialize() const;

 protected:
  std::map<std::string, std::string> params_;
  static OpMap*& op_map();
  const std::vector<const char*> GetParamKeys_() const;
  const std::vector<const char*> GetParamValues_() const;
  std::map<int, unsigned> count_;
  unsigned begin_num_update_, num_update_;
  unsigned UpdateCount_(int index);
  float GetLR_(int index);
  float GetWD_(int index);
  virtual void CreateState_(int index, NDArray weight);
  std::unique_ptr<LRScheduler> lrScheduler_ = nullptr;
};

typedef std::function<Optimizer*()> OptimizerCreator;

class OptimizerRegistry {
 public:
  static Optimizer* Find(const std::string& name);
  static int __REGISTER__(const std::string& name, OptimizerCreator creator);
 private:
  static std::map<std::string, OptimizerCreator>& cmap();
  OptimizerRegistry() = delete;
  ~OptimizerRegistry() = delete;
};
#define MXNETCPP_REGISTER_OPTIMIZER(Name, OptimizerType)\
  OptimizerRegistry::__REGISTER__(#Name, [](){return new OptimizerType();})

class SGDOptimizer : public Optimizer {
 public:
  explicit SGDOptimizer(unsigned begin_num_update = 0);
  std::string GetType() const override;
  void Update(int index, NDArray weight, NDArray grad) override;
 private:
  virtual ~SGDOptimizer();
  void CreateState_(int index, NDArray weight) override;
  std::map<int, NDArray*> states_;
  AtomicSymbolCreator update_handle_;
  AtomicSymbolCreator mom_update_handle_;
};

class SignumOptimizer : public Optimizer {
 public:
  explicit SignumOptimizer(unsigned begin_num_update = 0);
  std::string GetType() const override;
  void Update(int index, NDArray weight, NDArray grad) override;
 private:
  virtual ~SignumOptimizer();
  void CreateState_(int index, NDArray weight) override;
  std::map<int, NDArray*> states_;
  AtomicSymbolCreator update_handle_;
  AtomicSymbolCreator mom_update_handle_;
};


class RMSPropOptimizer : public Optimizer {
 public:
  explicit RMSPropOptimizer(unsigned begin_num_update = 0);
  std::string GetType() const override;
  void Update(int index, NDArray weight, NDArray grad) override;
 private:
  virtual ~RMSPropOptimizer();
  void CreateState_(int index, NDArray weight) override;
  std::map<int, NDArray*> n_, g_, delta_;
  AtomicSymbolCreator update_handle_;
  AtomicSymbolCreator alex_update_handle_;
};

class AdamOptimizer : public Optimizer {
 public:
  explicit AdamOptimizer(unsigned begin_num_update = 0);
  std::string GetType() const override;
  void Update(int index, NDArray weight, NDArray grad) override;
 private:
  virtual ~AdamOptimizer();
  void CreateState_(int index, NDArray weight) override;
  std::map<int, NDArray*> mean_;
  std::map<int, NDArray*> var_;
  AtomicSymbolCreator update_handle_;
};

class AdaGradOptimizer : public Optimizer {
 public:
  explicit AdaGradOptimizer(unsigned begin_num_update = 0);
  std::string GetType() const override;
  void Update(int index, NDArray weight, NDArray grad) override;
 private:
  virtual ~AdaGradOptimizer();
  void CreateState_(int index, NDArray weight) override;
  std::map<int, NDArray*> history_;
};

class AdaDeltaOptimizer : public Optimizer {
 public:
  explicit AdaDeltaOptimizer(unsigned begin_num_update = 0);
  std::string GetType() const override;
  void Update(int index, NDArray weight, NDArray grad) override;
 private:
  virtual ~AdaDeltaOptimizer();
  void CreateState_(int index, NDArray weight) override;
  std::map<int, NDArray*> acc_g_, acc_delta_;
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_OPTIMIZER_H_
