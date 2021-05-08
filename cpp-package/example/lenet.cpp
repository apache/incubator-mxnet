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
 */
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>
#include "mxnet-cpp/MxNetCpp.h"
#include "utils.h"

using namespace mxnet::cpp;

class Lenet {
 public:
  Lenet()
      : ctx_cpu(Context(DeviceType::kCPU, 0)),
#if MXNET_USE_CPU
        ctx_dev(Context(DeviceType::kCPU, 0))
#else
        ctx_dev(Context(DeviceType::kGPU, 0))
#endif
        {}

  void Run(int max_epoch) {
    /*
     * LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
     * "Gradient-based learning applied to document recognition."
     * Proceedings of the IEEE (1998)
     * */

    /*define the symbolic net*/
    Symbol data = Symbol::Variable("data");
    Symbol data_label = Symbol::Variable("data_label");
    Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
    Symbol conv2_w("conv2_w"), conv2_b("conv2_b");
    Symbol conv3_w("conv3_w"), conv3_b("conv3_b");
    Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
    Symbol fc2_w("fc2_w"), fc2_b("fc2_b");

    Symbol conv1 =
        Convolution("conv1", data, conv1_w, conv1_b, Shape(5, 5), 20);
    Symbol tanh1 = Activation("tanh1", conv1, ActivationActType::kTanh);
    Symbol pool1 = Pooling("pool1", tanh1, Shape(2, 2), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

    Symbol conv2 = Convolution("conv2", pool1, conv2_w, conv2_b,
      Shape(5, 5), 50);
    Symbol tanh2 = Activation("tanh2", conv2, ActivationActType::kTanh);
    Symbol pool2 = Pooling("pool2", tanh2, Shape(2, 2), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

    Symbol conv3 = Convolution("conv3", pool2, conv3_w, conv3_b,
      Shape(2, 2), 500);
    Symbol tanh3 = Activation("tanh3", conv3, ActivationActType::kTanh);
    Symbol pool3 = Pooling("pool3", tanh3, Shape(2, 2), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(1, 1));

    Symbol flatten = Flatten("flatten", pool3);
    Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 500);
    Symbol tanh4 = Activation("tanh4", fc1, ActivationActType::kTanh);
    Symbol fc2 = FullyConnected("fc2", tanh4, fc2_w, fc2_b, 10);

    Symbol lenet = SoftmaxOutput("softmax", fc2, data_label);

    for (auto s : lenet.ListArguments()) {
      LG << s;
    }

    /*setup basic configs*/
    int val_fold = 1;
    int W = 28;
    int H = 28;
    int batch_size = 42;
    float learning_rate = 1e-4;
    float weight_decay = 1e-4;

    /*prepare the data*/
    std::vector<float> data_vec, label_vec;
    size_t data_count = GetData(&data_vec, &label_vec);
    const float *dptr = data_vec.data();
    const float *lptr = label_vec.data();
    NDArray data_array = NDArray(Shape(data_count, 1, W, H), ctx_cpu,
                                 false);  // store in main memory, and copy to
    // device memory while training
    NDArray label_array =
      NDArray(Shape(data_count), ctx_cpu,
                false);  // it's also ok if just store them all in device memory
    data_array.SyncCopyFromCPU(dptr, data_count * W * H);
    label_array.SyncCopyFromCPU(lptr, data_count);
    data_array.WaitToRead();
    label_array.WaitToRead();

    size_t train_num = data_count * (1 - val_fold / 10.0);
    train_data = data_array.Slice(0, train_num);
    train_label = label_array.Slice(0, train_num);
    val_data = data_array.Slice(train_num, data_count);
    val_label = label_array.Slice(train_num, data_count);

    LG << "here read fin";

    /*init some of the args*/
    // map<string, NDArray> args_map;
    args_map["data"] = data_array.Slice(0, batch_size).Copy(ctx_dev);
    args_map["data_label"] = label_array.Slice(0, batch_size).Copy(ctx_dev);
    NDArray::WaitAll();

    LG << "here slice fin";
    /*
     * we can also feed in some of the args other than the input all by
     * ourselves,
     * fc2-w , fc1-b for example:
     * */
    // args_map["fc2_w"] =
    // NDArray(mshadow::Shape2(500, 4 * 4 * 50), ctx_dev, false);
    // NDArray::SampleGaussian(0, 1, &args_map["fc2_w"]);
    // args_map["fc1_b"] = NDArray(mshadow::Shape1(10), ctx_dev, false);
    // args_map["fc1_b"] = 0;

    lenet.InferArgsMap(ctx_dev, &args_map, args_map);
    Optimizer* opt = OptimizerRegistry::Find("ccsgd");
    opt->SetParam("momentum", 0.9)
       ->SetParam("rescale_grad", 1.0)
       ->SetParam("clip_gradient", 10)
       ->SetParam("lr", learning_rate)
       ->SetParam("wd", weight_decay);

    Executor *exe = lenet.SimpleBind(ctx_dev, args_map);
    auto arg_names = lenet.ListArguments();

    for (int ITER = 0; ITER < max_epoch; ++ITER) {
      size_t start_index = 0;
      while (start_index < train_num) {
        if (start_index + batch_size > train_num) {
          start_index = train_num - batch_size;
        }
        args_map["data"] =
            train_data.Slice(start_index, start_index + batch_size)
                .Copy(ctx_dev);
        args_map["data_label"] =
            train_label.Slice(start_index, start_index + batch_size)
                .Copy(ctx_dev);
        start_index += batch_size;
        NDArray::WaitAll();

        exe->Forward(true);
        exe->Backward();
        // Update parameters
        for (size_t i = 0; i < arg_names.size(); ++i) {
          if (arg_names[i] == "data" || arg_names[i] == "data_label") continue;
          opt->Update(i, exe->arg_arrays[i], exe->grad_arrays[i]);
        }
      }

      LG << "Iter " << ITER
         << ", accuracy: " << ValAccuracy(batch_size * 10, lenet);
    }
    delete exe;
    delete opt;
  }

 private:
  Context ctx_cpu;
  Context ctx_dev;
  std::map<std::string, NDArray> args_map;
  NDArray train_data;
  NDArray train_label;
  NDArray val_data;
  NDArray val_label;

  size_t GetData(std::vector<float> *data, std::vector<float> *label) {
    const char *train_data_path = "./data/mnist_data/mnist_train.csv";
    std::ifstream inf(train_data_path);
    std::string line;
    inf >> line;  // ignore the header
    size_t _N = 0;
    while (inf >> line) {
      for (auto &c : line) c = (c == ',') ? ' ' : c;
      std::stringstream ss;
      ss << line;
      float _data;
      ss >> _data;
      label->push_back(_data);
      while (ss >> _data) data->push_back(_data / 256.0);
      _N++;
    }
    inf.close();
    return _N;
  }

  float ValAccuracy(int batch_size, Symbol lenet) {
    size_t val_num = val_data.GetShape()[0];

    size_t correct_count = 0;
    size_t all_count = 0;

    size_t start_index = 0;
    while (start_index < val_num) {
      if (start_index + batch_size > val_num) {
        start_index = val_num - batch_size;
      }
      args_map["data"] =
          val_data.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
      args_map["data_label"] =
          val_label.Slice(start_index, start_index + batch_size).Copy(ctx_dev);
      start_index += batch_size;
      NDArray::WaitAll();

      Executor *exe = lenet.SimpleBind(ctx_dev, args_map);
      exe->Forward(false);

      const auto &out = exe->outputs;
      NDArray out_cpu = out[0].Copy(ctx_cpu);
      NDArray label_cpu =
          val_label.Slice(start_index - batch_size, start_index).Copy(ctx_cpu);

      NDArray::WaitAll();

      const mx_float *dptr_out = out_cpu.GetData();
      const mx_float *dptr_label = label_cpu.GetData();
      for (int i = 0; i < batch_size; ++i) {
        float label = dptr_label[i];
        int cat_num = out_cpu.GetShape()[1];
        float p_label = 0, max_p = dptr_out[i * cat_num];
        for (int j = 0; j < cat_num; ++j) {
          float p = dptr_out[i * cat_num + j];
          if (max_p < p) {
            p_label = j;
            max_p = p;
          }
        }
        if (label == p_label) correct_count++;
      }
      all_count += batch_size;

      delete exe;
    }
    return correct_count * 1.0 / all_count;
  }
};

int main(int argc, char const *argv[]) {
  TRY
  Lenet lenet;
  lenet.Run(argc > 1 ? strtol(argv[1], nullptr, 10) : 100000);
  MXNotifyShutdown();
  CATCH
  return 0;
}
