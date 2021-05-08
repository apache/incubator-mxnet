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
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Symbol LenetSymbol() {
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

  Symbol conv1 = Convolution("conv1", data, conv1_w, conv1_b, Shape(5, 5), 20);
  Symbol tanh1 = Activation("tanh1", conv1, ActivationActType::kTanh);
  Symbol pool1 = Pooling("pool1", tanh1, Shape(2, 2), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  Symbol conv2 = Convolution("conv2", pool1, conv2_w, conv2_b, Shape(5, 5), 50);
  Symbol tanh2 = Activation("tanh2", conv2, ActivationActType::kTanh);
  Symbol pool2 = Pooling("pool2", tanh2, Shape(2, 2), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  Symbol flatten = Flatten("flatten", pool2);
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, 500);
  Symbol tanh3 = Activation("tanh3", fc1, ActivationActType::kTanh);
  Symbol fc2 = FullyConnected("fc2", tanh3, fc2_w, fc2_b, 10);

  Symbol lenet = SoftmaxOutput("softmax", fc2, data_label);

  return lenet;
}

NDArray ResizeInput(NDArray data, const Shape new_shape) {
  NDArray pic = data.Reshape(Shape(0, 1, 28, 28));
  NDArray output;
  Operator("_contrib_BilinearResize2D")
    .SetParam("height", new_shape[2])
    .SetParam("width", new_shape[3])
    (pic).Invoke(output);
  return output;
}

int main(int argc, char const *argv[]) {
  /*setup basic configs*/
  int W = 28;
  int H = 28;
  int batch_size = 128;
  int max_epoch = argc > 1 ? strtol(argv[1], nullptr, 10) : 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  auto dev_ctx = Context::cpu();
  int num_gpu;
  MXGetGPUCount(&num_gpu);
#if !MXNET_USE_CPU
  if (num_gpu > 0) {
    dev_ctx = Context::gpu();
  }
#endif

  TRY
  auto lenet = LenetSymbol();
  std::map<std::string, NDArray> args_map;

  const Shape data_shape = Shape(batch_size, 1, H, W),
              label_shape = Shape(batch_size);
  args_map["data"] = NDArray(data_shape, dev_ctx);
  args_map["data_label"] = NDArray(label_shape, dev_ctx);
  lenet.InferArgsMap(dev_ctx, &args_map, args_map);

  args_map["fc1_w"] = NDArray(Shape(500, 4 * 4 * 50), dev_ctx);
  NDArray::SampleGaussian(0, 1, &args_map["fc1_w"]);
  args_map["fc2_b"] = NDArray(Shape(10), dev_ctx);
  args_map["fc2_b"] = 0;

  std::vector<std::string> data_files = { "./data/mnist_data/train-images-idx3-ubyte",
                                          "./data/mnist_data/train-labels-idx1-ubyte",
                                          "./data/mnist_data/t10k-images-idx3-ubyte",
                                          "./data/mnist_data/t10k-labels-idx1-ubyte"
                                        };

  auto train_iter =  MXDataIter("MNISTIter");
  if (!setDataIter(&train_iter, "Train", data_files, batch_size)) {
    return 1;
  }

  auto val_iter = MXDataIter("MNISTIter");
  if (!setDataIter(&val_iter, "Label", data_files, batch_size)) {
    return 1;
  }

  Optimizer* opt = OptimizerRegistry::Find("sgd");
  opt->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0)
     ->SetParam("clip_gradient", 10)
     ->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);


  auto *exec = lenet.SimpleBind(dev_ctx, args_map);
  auto arg_names = lenet.ListArguments();

  // Create metrics
  Accuracy train_acc, val_acc;

  for (int iter = 0; iter < max_epoch; ++iter) {
      int samples = 0;
      train_iter.Reset();
      train_acc.Reset();

      auto tic = std::chrono::system_clock::now();

     while (train_iter.Next()) {
      samples += batch_size;
      auto data_batch = train_iter.GetDataBatch();

      ResizeInput(data_batch.data, data_shape).CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();

      // Compute gradients
      exec->Forward(true);
      exec->Backward();

      // Update parameters
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "data" || arg_names[i] == "data_label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }

      // Update metric
      train_acc.Update(data_batch.label, exec->outputs[0]);
    }

     // one epoch of training is finished
     auto toc = std::chrono::system_clock::now();
     float duration = std::chrono::duration_cast<std::chrono::milliseconds>
                      (toc - tic).count() / 1000.0;
     LG << "Epoch[" << iter << "] " << samples / duration \
         << " samples/sec " << "Train-Accuracy=" << train_acc.Get();;

      val_iter.Reset();
      val_acc.Reset();

    Accuracy acu;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      ResizeInput(data_batch.data, data_shape).CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();

      // Only forward pass is enough as no gradient is needed when evaluating
      exec->Forward(false);
      NDArray::WaitAll();
      acu.Update(data_batch.label, exec->outputs[0]);
      val_acc.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Epoch[" << iter << "] Val-Accuracy=" << val_acc.Get();
  }

  delete exec;
  delete opt;
  MXNotifyShutdown();
  CATCH
  return 0;
}
