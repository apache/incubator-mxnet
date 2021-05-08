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
#include <fstream>
#include <vector>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Symbol ConvFactoryBN(Symbol data, int num_filter,
                     Shape kernel, Shape stride, Shape pad,
                     const std::string & name,
                     const std::string & suffix = "") {
  Symbol conv_w("conv_" + name + suffix + "_w"), conv_b("conv_" + name + suffix + "_b");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1, 1), pad);
  std::string name_suffix = name + suffix;
  Symbol gamma(name_suffix + "_gamma");
  Symbol beta(name_suffix + "_beta");
  Symbol mmean(name_suffix + "_mmean");
  Symbol mvar(name_suffix + "_mvar");
  Symbol bn = BatchNorm("bn_" + name + suffix, conv, gamma, beta, mmean, mvar);
  return Activation("relu_" + name + suffix, bn, "relu");
}

Symbol InceptionFactoryA(Symbol data, int num_1x1, int num_3x3red,
                         int num_3x3, int num_d3x3red, int num_d3x3,
                         PoolingPoolType pool, int proj,
                         const std::string & name) {
  Symbol c1x1 = ConvFactoryBN(data, num_1x1, Shape(1, 1), Shape(1, 1),
                              Shape(0, 0), name + "1x1");
  Symbol c3x3r = ConvFactoryBN(data, num_3x3red, Shape(1, 1), Shape(1, 1),
                               Shape(0, 0), name + "_3x3r");
  Symbol c3x3 = ConvFactoryBN(c3x3r, num_3x3, Shape(3, 3), Shape(1, 1),
                              Shape(1, 1), name + "_3x3");
  Symbol cd3x3r = ConvFactoryBN(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                                Shape(0, 0), name + "_double_3x3", "_reduce");
  Symbol cd3x3 = ConvFactoryBN(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                               Shape(1, 1), name + "_double_3x3_0");
  cd3x3 = ConvFactoryBN(data = cd3x3, num_d3x3, Shape(3, 3), Shape(1, 1),
                        Shape(1, 1), name + "_double_3x3_1");
  Symbol pooling = Pooling(name + "_pool", data,
                           Shape(3, 3), pool, false, false,
                           PoolingPoolingConvention::kValid,
                           Shape(1, 1), Shape(1, 1));
  Symbol cproj = ConvFactoryBN(pooling, proj, Shape(1, 1), Shape(1, 1),
                               Shape(0, 0), name + "_proj");
  std::vector<Symbol> lst;
  lst.push_back(c1x1);
  lst.push_back(c3x3);
  lst.push_back(cd3x3);
  lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol InceptionFactoryB(Symbol data, int num_3x3red, int num_3x3,
                         int num_d3x3red, int num_d3x3, const std::string & name) {
  Symbol c3x3r = ConvFactoryBN(data, num_3x3red, Shape(1, 1),
                               Shape(1, 1), Shape(0, 0),
                               name + "_3x3", "_reduce");
  Symbol c3x3 = ConvFactoryBN(c3x3r, num_3x3, Shape(3, 3), Shape(2, 2),
                              Shape(1, 1), name + "_3x3");
  Symbol cd3x3r = ConvFactoryBN(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                                Shape(0, 0), name + "_double_3x3", "_reduce");
  Symbol cd3x3 = ConvFactoryBN(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                               Shape(1, 1), name + "_double_3x3_0");
  cd3x3 = ConvFactoryBN(cd3x3, num_d3x3, Shape(3, 3), Shape(2, 2),
                        Shape(1, 1), name + "_double_3x3_1");
  Symbol pooling = Pooling("max_pool_" + name + "_pool", data,
                           Shape(3, 3), PoolingPoolType::kMax,
                           false, false, PoolingPoolingConvention::kValid,
                           Shape(2, 2), Shape(1, 1));
  std::vector<Symbol> lst;
  lst.push_back(c3x3);
  lst.push_back(cd3x3);
  lst.push_back(pooling);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
}

Symbol InceptionSymbol(int num_classes) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  // stage 1
  Symbol conv1 = ConvFactoryBN(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  // stage 2
  Symbol conv2red = ConvFactoryBN(pool1, 64, Shape(1, 1), Shape(1, 1),  Shape(0, 0), "conv2red");
  Symbol conv2 = ConvFactoryBN(conv2red, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv2");
  Symbol pool2 = Pooling("pool2", conv2, Shape(3, 3), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  // stage 3
  Symbol in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, PoolingPoolType::kAvg, 32, "3a");
  Symbol in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, PoolingPoolType::kAvg, 64, "3b");
  Symbol in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, "3c");

  // stage 4
  Symbol in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, PoolingPoolType::kAvg, 128, "4a");
  Symbol in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128,  PoolingPoolType::kAvg, 128, "4b");
  Symbol in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, PoolingPoolType::kAvg, 128, "4c");
  Symbol in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192,  PoolingPoolType::kAvg, 128, "4d");
  Symbol in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, "4e");

  // stage 5
  Symbol in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, PoolingPoolType::kAvg, 128, "5a");
  Symbol in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, PoolingPoolType::kMax, 128, "5b");

  // average pooling
  Symbol avg = Pooling("global_pool", in5b, Shape(7, 7), PoolingPoolType::kAvg);

  // classifier
  Symbol flatten = Flatten("flatten", avg);
  Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
  Symbol fc1 = FullyConnected("fc1", flatten, conv1_w, conv1_b, num_classes);
  return SoftmaxOutput("softmax", fc1, data_label);
}

NDArray ResizeInput(NDArray data, const Shape new_shape) {
  NDArray pic = data.Reshape(Shape(0, 1, 28, 28));
  NDArray pic_1channel;
  Operator("_contrib_BilinearResize2D")
    .SetParam("height", new_shape[2])
    .SetParam("width", new_shape[3])
    (pic).Invoke(pic_1channel);
  NDArray output;
  Operator("tile")
    .SetParam("reps", Shape(1, 3, 1, 1))
    (pic_1channel).Invoke(output);
  return output;
}

int main(int argc, char const *argv[]) {
  int batch_size = 40;
  int max_epoch = argc > 1 ? strtol(argv[1], nullptr, 10) : 100;
  float learning_rate = 1e-2;
  float weight_decay = 1e-4;

  /*context*/
  auto ctx = Context::cpu();
  int num_gpu;
  MXGetGPUCount(&num_gpu);
#if !MXNET_USE_CPU
  if (num_gpu > 0) {
    ctx = Context::gpu();
  }
#endif

  TRY
  auto inception_bn_net = InceptionSymbol(10);
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

  const Shape data_shape = Shape(batch_size, 3, 224, 224),
              label_shape = Shape(batch_size);
  args_map["data"] = NDArray(data_shape, ctx);
  args_map["data_label"] = NDArray(label_shape, ctx);
  inception_bn_net.InferArgsMap(ctx, &args_map, args_map);

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

  // initialize parameters
  auto initializer = Uniform(0.07);
  for (auto& arg : args_map) {
    initializer(arg.first, &arg.second);
  }

  Optimizer* opt = OptimizerRegistry::Find("sgd");
  opt->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("clip_gradient", 10)
     ->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);

  auto *exec = inception_bn_net.SimpleBind(ctx, args_map);
  auto arg_names = inception_bn_net.ListArguments();

  // Create metrics
  Accuracy train_acc, val_acc;
  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    train_iter.Reset();
    train_acc.Reset();
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      ResizeInput(data_batch.data, data_shape).CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();

      exec->Forward(true);
      exec->Backward();
      // Update parameters
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "data" || arg_names[i] == "data_label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }

      NDArray::WaitAll();
      train_acc.Update(data_batch.label, exec->outputs[0]);
    }

    val_iter.Reset();
    val_acc.Reset();
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      ResizeInput(data_batch.data, data_shape).CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();
      exec->Forward(false);
      NDArray::WaitAll();
      val_acc.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Train Accuracy: " << train_acc.Get();
    LG << "Validation Accuracy: " << val_acc.Get();
  }
  delete exec;
  delete opt;
  MXNotifyShutdown();
  CATCH
  return 0;
}
