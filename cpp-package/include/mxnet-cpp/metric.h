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
* \file base.h
* \brief metrics defined
* \author Zhang Chen
*/

#ifndef MXNET_CPP_METRIC_H_
#define MXNET_CPP_METRIC_H_

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include "mxnet-cpp/ndarray.h"
#include "dmlc/logging.h"

namespace mxnet {
namespace cpp {

class EvalMetric {
 public:
  explicit EvalMetric(const std::string& name, int num = 0)
      : name(name), num(num) {}
  virtual void Update(NDArray labels, NDArray preds) = 0;
  void Reset() {
    num_inst = 0;
    sum_metric = 0.0f;
  }
  float Get() { return sum_metric / num_inst; }
  void GetNameValue();

 protected:
  std::string name;
  int num;
  float sum_metric = 0.0f;
  int num_inst = 0;

  static void CheckLabelShapes(NDArray labels, NDArray preds,
                               bool strict = false) {
    if (strict) {
      CHECK_EQ(Shape(labels.GetShape()), Shape(preds.GetShape()));
    } else {
      CHECK_EQ(labels.Size(), preds.Size());
    }
  }
};

class Accuracy : public EvalMetric {
 public:
  Accuracy() : EvalMetric("accuracy") {}

  void Update(NDArray labels, NDArray preds) override {
    CHECK_EQ(labels.GetShape().size(), 1);
    mx_uint len = labels.GetShape()[0];
    std::vector<mx_float> pred_data(len);
    std::vector<mx_float> label_data(len);
    preds.ArgmaxChannel().SyncCopyToCPU(&pred_data, len);
    labels.SyncCopyToCPU(&label_data, len);
    for (mx_uint i = 0; i < len; ++i) {
      sum_metric += (pred_data[i] == label_data[i]) ? 1 : 0;
      num_inst += 1;
    }
  }
};

class LogLoss : public EvalMetric {
 public:
  LogLoss() : EvalMetric("logloss") {}

  void Update(NDArray labels, NDArray preds) override {
    static const float epsilon = 1e-15;
    mx_uint len = labels.GetShape()[0];
    mx_uint m = preds.GetShape()[1];
    std::vector<mx_float> pred_data(len * m);
    std::vector<mx_float> label_data(len);
    preds.SyncCopyToCPU(&pred_data, pred_data.size());
    labels.SyncCopyToCPU(&label_data, len);
    for (mx_uint i = 0; i < len; ++i) {
      sum_metric +=
          -std::log(std::max(pred_data[i * m + label_data[i]], epsilon));
      num_inst += 1;
    }
  }
};

class MAE : public EvalMetric {
 public:
  MAE() : EvalMetric("mae") {}

  void Update(NDArray labels, NDArray preds) override {
    CheckLabelShapes(labels, preds);

    std::vector<mx_float> pred_data;
    preds.SyncCopyToCPU(&pred_data);
    std::vector<mx_float> label_data;
    labels.SyncCopyToCPU(&label_data);

    size_t len = preds.Size();
    mx_float sum = 0;
    for (size_t i = 0; i < len; ++i) {
      sum += std::abs(pred_data[i] - label_data[i]);
    }
    sum_metric += sum / len;
    ++num_inst;
  }
};

class MSE : public EvalMetric {
 public:
  MSE() : EvalMetric("mse") {}

  void Update(NDArray labels, NDArray preds) override {
    CheckLabelShapes(labels, preds);

    std::vector<mx_float> pred_data;
    preds.SyncCopyToCPU(&pred_data);
    std::vector<mx_float> label_data;
    labels.SyncCopyToCPU(&label_data);

    size_t len = preds.Size();
    mx_float sum = 0;
    for (size_t i = 0; i < len; ++i) {
      mx_float diff = pred_data[i] - label_data[i];
      sum += diff * diff;
    }
    sum_metric += sum / len;
    ++num_inst;
  }
};

class RMSE : public EvalMetric {
 public:
  RMSE() : EvalMetric("rmse") {}

  void Update(NDArray labels, NDArray preds) override {
    CheckLabelShapes(labels, preds);

    std::vector<mx_float> pred_data;
    preds.SyncCopyToCPU(&pred_data);
    std::vector<mx_float> label_data;
    labels.SyncCopyToCPU(&label_data);

    size_t len = preds.Size();
    mx_float sum = 0;
    for (size_t i = 0; i < len; ++i) {
      mx_float diff = pred_data[i] - label_data[i];
      sum += diff * diff;
    }
    sum_metric += std::sqrt(sum / len);
    ++num_inst;
  }
};

class PSNR : public EvalMetric {
 public:
  PSNR() : EvalMetric("psnr") {
  }

  void Update(NDArray labels, NDArray preds) override {
    CheckLabelShapes(labels, preds);

    std::vector<mx_float> pred_data;
    preds.SyncCopyToCPU(&pred_data);
    std::vector<mx_float> label_data;
    labels.SyncCopyToCPU(&label_data);

    size_t len = preds.Size();
    mx_float sum = 0;
    for (size_t i = 0; i < len; ++i) {
      mx_float diff = pred_data[i] - label_data[i];
      sum += diff * diff;
    }
    mx_float mse = sum / len;
    if (mse > 0) {
      sum_metric += 10 * std::log(255.0f / mse) / log10_;
    } else {
      sum_metric += 99.0f;
    }
    ++num_inst;
  }

 private:
  mx_float log10_ = std::log(10.0f);
};

}  // namespace cpp
}  // namespace mxnet

#endif  // MXNET_CPP_METRIC_H_

