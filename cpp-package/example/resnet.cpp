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
#include <cstdlib>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

Symbol ConvolutionNoBias(const std::string& symbol_name,
                         Symbol data,
                         Symbol weight,
                         Shape kernel,
                         int num_filter,
                         Shape stride = Shape(1, 1),
                         Shape dilate = Shape(1, 1),
                         Shape pad = Shape(0, 0),
                         int num_group = 1,
                         int64_t workspace = 512) {
  return Operator("Convolution")
      .SetParam("kernel", kernel)
      .SetParam("num_filter", num_filter)
      .SetParam("stride", stride)
      .SetParam("dilate", dilate)
      .SetParam("pad", pad)
      .SetParam("num_group", num_group)
      .SetParam("workspace", workspace)
      .SetParam("no_bias", true)
      .SetInput("data", data)
      .SetInput("weight", weight)
      .CreateSymbol(symbol_name);
}

Symbol getConv(const std::string & name, Symbol data,
               int  num_filter,
               Shape kernel, Shape stride, Shape pad,
               bool with_relu,
               mx_float bn_momentum) {
  Symbol conv_w(name + "_w");
  Symbol conv = ConvolutionNoBias(name, data, conv_w,
                                  kernel, num_filter, stride, Shape(1, 1),
                                  pad, 1, 512);

  Symbol gamma(name + "_gamma");
  Symbol beta(name + "_beta");
  Symbol mmean(name + "_mmean");
  Symbol mvar(name + "_mvar");

  Symbol bn = BatchNorm(name + "_bn", conv, gamma,
                        beta, mmean, mvar, 2e-5, bn_momentum, false);

  if (with_relu) {
    return Activation(name + "_relu", bn, "relu");
  } else {
    return bn;
  }
}

Symbol makeBlock(const std::string & name, Symbol data, int num_filter,
                 bool dim_match, mx_float bn_momentum) {
  Shape stride;
  if (dim_match) {
    stride = Shape(1, 1);
  } else {
    stride = Shape(2, 2);
  }

  Symbol conv1 = getConv(name + "_conv1", data, num_filter,
                         Shape(3, 3), stride, Shape(1, 1),
                         true, bn_momentum);

  Symbol conv2 = getConv(name + "_conv2", conv1, num_filter,
                         Shape(3, 3), Shape(1, 1), Shape(1, 1),
                         false, bn_momentum);

  Symbol shortcut;

  if (dim_match) {
    shortcut = data;
  } else {
    Symbol shortcut_w(name + "_proj_w");
    shortcut = ConvolutionNoBias(name + "_proj", data, shortcut_w,
                                 Shape(2, 2), num_filter,
                                 Shape(2, 2), Shape(1, 1), Shape(0, 0),
                                 1, 512);
  }

  Symbol fused = shortcut + conv2;
  return Activation(name + "_relu", fused, "relu");
}

Symbol getBody(Symbol data, int num_level, int num_block, int num_filter, mx_float bn_momentum) {
  for (int level = 0; level < num_level; level++) {
    for (int block = 0; block < num_block; block++) {
      data = makeBlock("level" + std::to_string(level + 1) + "_block" + std::to_string(block + 1),
                       data, num_filter * (std::pow(2, level)),
                       (level == 0 || block > 0), bn_momentum);
    }
  }
  return data;
}

Symbol ResNetSymbol(int num_class, int num_level = 3, int num_block = 9,
                    int num_filter = 16, mx_float bn_momentum = 0.9,
                    mxnet::cpp::Shape pool_kernel = mxnet::cpp::Shape(8, 8)) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  Symbol gamma("gamma");
  Symbol beta("beta");
  Symbol mmean("mmean");
  Symbol mvar("mvar");

  Symbol zscore = BatchNorm("zscore", data, gamma,
                            beta, mmean, mvar, 0.001, bn_momentum);

  Symbol conv = getConv("conv0", zscore, num_filter,
                        Shape(3, 3), Shape(1, 1), Shape(1, 1),
                        true, bn_momentum);

  Symbol body = getBody(conv, num_level, num_block, num_filter, bn_momentum);

  Symbol pool = Pooling("pool", body, pool_kernel, PoolingPoolType::kAvg);

  Symbol flat = Flatten("flatten", pool);

  Symbol fc_w("fc_w"), fc_b("fc_b");
  Symbol fc = FullyConnected("fc", flat, fc_w, fc_b, num_class);

  return SoftmaxOutput("softmax", fc, data_label);
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
  int max_epoch = argc > 1 ? strtol(argv[1], nullptr, 10) : 100;
  float learning_rate = 1e-4;
  float weight_decay = 1e-4;

  TRY
  auto resnet = ResNetSymbol(10);
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

  /*context*/
  auto ctx = Context::cpu();
  int num_gpu;
  MXGetGPUCount(&num_gpu);
  int batch_size = 8;
#if !MXNET_USE_CPU
  if (num_gpu > 0) {
    ctx = Context::gpu();
    batch_size = 32;
  }
#endif

  const Shape data_shape = Shape(batch_size, 3, 224, 224),
              label_shape = Shape(batch_size);
  args_map["data"] = NDArray(data_shape, ctx);
  args_map["data_label"] = NDArray(label_shape, ctx);
  resnet.InferArgsMap(ctx, &args_map, args_map);

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
  opt->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay)
     ->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("clip_gradient", 10);

  auto *exec = resnet.SimpleBind(ctx, args_map);
  auto arg_names = resnet.ListArguments();

  // Create metrics
  Accuracy train_acc, val_acc;
  LogLoss logloss_train, logloss_val;
  for (int epoch = 0; epoch < max_epoch; ++epoch) {
    LG << "Epoch: " << epoch;
    train_iter.Reset();
    train_acc.Reset();
    int iter = 0;
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      ResizeInput(data_batch.data, data_shape).CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();

      exec->Forward(true);
      exec->Backward();

      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == "data" || arg_names[i] == "data_label") continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }
      NDArray::WaitAll();
      train_acc.Update(data_batch.label, exec->outputs[0]);
      logloss_train.Reset();
      logloss_train.Update(data_batch.label, exec->outputs[0]);
      ++iter;
      LG << "EPOCH: " << epoch << " ITER: " << iter
         << " Train Accuracy: " << train_acc.Get()
         << " Train Loss: " << logloss_train.Get();
    }
    LG << "EPOCH: " << epoch << " Train Accuracy: " << train_acc.Get();

    val_iter.Reset();
    val_acc.Reset();
    iter = 0;
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      ResizeInput(data_batch.data, data_shape).CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();
      exec->Forward(false);
      NDArray::WaitAll();
      val_acc.Update(data_batch.label, exec->outputs[0]);
      LG << "EPOCH: " << epoch << " ITER: " << iter << " Val Accuracy: " << val_acc.Get();
      ++iter;
    }
    LG << "Validation Accuracy: " << val_acc.Get();
  }
  delete exec;
  delete opt;
  MXNotifyShutdown();
  CATCH
  return 0;
}
