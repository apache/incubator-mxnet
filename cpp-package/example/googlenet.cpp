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
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Symbol ConvFactory(Symbol data, int num_filter,
                   Shape kernel,
                   Shape stride = Shape(1, 1),
                   Shape pad = Shape(0, 0),
                   const std::string & name = "",
                   const std::string & suffix = "") {
  Symbol conv_w("conv_" + name + suffix + "_w"), conv_b("conv_" + name + suffix + "_b");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1, 1), pad);
  return Activation("relu_" + name + suffix, conv, "relu");
}

Symbol InceptionFactory(Symbol data, int num_1x1, int num_3x3red,
                        int num_3x3, int num_d5x5red, int num_d5x5,
                        PoolingPoolType pool, int proj, const std::string & name) {
  Symbol c1x1 = ConvFactory(data, num_1x1, Shape(1, 1),
                            Shape(1, 1), Shape(0, 0), name + "_1x1");

  Symbol c3x3r = ConvFactory(data, num_3x3red, Shape(1, 1),
                             Shape(1, 1), Shape(0, 0), name + "_3x3", "_reduce");

  Symbol c3x3 = ConvFactory(c3x3r, num_3x3, Shape(3, 3),
                            Shape(1, 1), Shape(1, 1), name + "_3x3");

  Symbol cd5x5r = ConvFactory(data, num_d5x5red, Shape(1, 1),
                              Shape(1, 1), Shape(0, 0), name + "_5x5", "_reduce");

  Symbol cd5x5 = ConvFactory(cd5x5r, num_d5x5, Shape(5, 5),
                             Shape(1, 1), Shape(2, 2), name + "_5x5");

  Symbol pooling = Pooling(name + "_pool", data, Shape(3, 3), pool,
                           false, false, PoolingPoolingConvention::kValid,
                           Shape(1, 1), Shape(1, 1));

  Symbol cproj = ConvFactory(pooling, proj, Shape(1, 1),
                             Shape(1, 1), Shape(0, 0), name + "_proj");

  std::vector<Symbol> lst;
  lst.push_back(c1x1);
  lst.push_back(c3x3);
  lst.push_back(cd5x5);
  lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol GoogleNetSymbol(int num_classes) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol conv1 = ConvFactory(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::kMax,
                         false, false, PoolingPoolingConvention::kValid, Shape(2, 2));
  Symbol conv2 = ConvFactory(pool1, 64, Shape(1, 1), Shape(1, 1),
                             Shape(0, 0), "conv2");
  Symbol conv3 = ConvFactory(conv2, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv3");
  Symbol pool3 = Pooling("pool3", conv3, Shape(3, 3), PoolingPoolType::kMax,
                         false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  Symbol in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, PoolingPoolType::kMax, 32, "in3a");
  Symbol in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, PoolingPoolType::kMax, 64, "in3b");
  Symbol pool4 = Pooling("pool4", in3b, Shape(3, 3), PoolingPoolType::kMax,
                         false, false, PoolingPoolingConvention::kValid, Shape(2, 2));
  Symbol in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, PoolingPoolType::kMax, 64, "in4a");
  Symbol in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, PoolingPoolType::kMax, 64, "in4b");
  Symbol in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, PoolingPoolType::kMax, 64, "in4c");
  Symbol in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, PoolingPoolType::kMax, 64, "in4d");
  Symbol in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, PoolingPoolType::kMax, 128, "in4e");
  Symbol pool5 = Pooling("pool5", in4e, Shape(3, 3), PoolingPoolType::kMax,
                         false, false, PoolingPoolingConvention::kValid, Shape(2, 2));
  Symbol in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, PoolingPoolType::kMax, 128, "in5a");
  Symbol in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, PoolingPoolType::kMax, 128, "in5b");
  Symbol pool6 = Pooling("pool6", in5b, Shape(7, 7), PoolingPoolType::kAvg,
                         false, false, PoolingPoolingConvention::kValid, Shape(1, 1));

  Symbol flatten = Flatten("flatten", pool6);

  Symbol fc1_w("fc1_w"), fc1_b("fc1_b");
  Symbol fc1 = FullyConnected("fc1", flatten, fc1_w, fc1_b, num_classes);

  return SoftmaxOutput("softmax", fc1, data_label);
}

int main(int argc, char const *argv[]) {
  int batch_size = 50;
  int max_epoch = argc > 1 ? strtol(argv[1], nullptr, 10) : 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  auto ctx = Context::gpu();
#if MXNET_USE_CPU
  ctx = Context::cpu();;
#endif

  TRY
  auto googlenet = GoogleNetSymbol(10);
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

  args_map["data"] = NDArray(Shape(batch_size, 3, 256, 256), ctx);
  args_map["data_label"] = NDArray(Shape(batch_size), ctx);
  googlenet.InferArgsMap(ctx, &args_map, args_map);

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
     ->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("clip_gradient", 10)
     ->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);


  auto *exec = googlenet.SimpleBind(ctx, args_map);
  auto arg_names = googlenet.ListArguments();

  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    train_iter.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();
      exec->Forward(true);
      exec->Backward();
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "data" || arg_names[i] == "data_label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }
    }

    Accuracy acu;
    val_iter.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();
      exec->Forward(false);
      NDArray::WaitAll();
      acu.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Accuracy: " << acu.Get();
  }

  delete exec;
  delete opt;
  MXNotifyShutdown();
  CATCH
  return 0;
}
