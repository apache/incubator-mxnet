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

/*
 * Example: mlp_csv
 * Description:
 * The following example demonstrates how to use CSVIter. This example creates
 * mlp (multi-layer perceptron) model and trains the MNIST data which is in
 * CSV format.
 */
#include <chrono>
#include <string>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

/*
 * Implementing the mlp symbol with given hidden units configuration.
 */
Symbol mlp(const std::vector<int> &hidden_units) {
    auto data = Symbol::Variable("data");
    auto label = Symbol::Variable("label");

    std::vector<Symbol> weights(hidden_units.size());
    std::vector<Symbol> biases(hidden_units.size());
    std::vector<Symbol> outputs(hidden_units.size());

    for (size_t i = 0; i < hidden_units.size(); ++i) {
        weights[i] = Symbol::Variable("w" + std::to_string(i));
        biases[i] = Symbol::Variable("b" + std::to_string(i));
        Symbol fc = FullyConnected(
                                   i == 0? data : outputs[i-1],  // data
                                   weights[i],
                                   biases[i],
                                   hidden_units[i]);
        outputs[i] = i == hidden_units.size()-1 ? fc : Activation(fc, ActivationActType::kRelu);
    }
    return SoftmaxOutput(outputs.back(), label);
}

/*
 * Convert the input string of number of hidden units into the vector of integers.
 */
std::vector<int> getLayers(const std::string& hidden_units_string) {
    std::vector<int> hidden_units;
    char *pNext;
    int num_unit = strtol(hidden_units_string.c_str(), &pNext, 10);
    hidden_units.push_back(num_unit);
    while (*pNext) {
        num_unit = strtol(pNext, &pNext, 10);
        hidden_units.push_back(num_unit);
    }
    return hidden_units;
}

void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "mlp_csv --train mnist_training_set.csv --test mnist_test_set.csv --epochs 10 "
    << "--batch_size 100 --hidden_units \"128 64 64\" --gpu" << std::endl;
    std::cout << "The example uses mnist data in CSV format. The MNIST data in CSV format assumes "
    << "the column 0 to be label and the rest 784 column to be data." << std::endl;
    std::cout << "By default, the example uses 'cpu' context. If '--gpu' is specified, "
    << "program uses 'gpu' context." <<std::endl;
}

int main(int argc, char** argv) {
    const int image_size = 28;
    const int num_mnist_features = image_size * image_size;
    int batch_size = 100;
    int max_epoch = 10;
    const float learning_rate = 0.1;
    const float weight_decay = 1e-2;
    bool isGpu = false;

    std::string training_set;
    std::string test_set;
    std::string hidden_units_string;
    int index = 1;
    while (index < argc) {
        if (strcmp("--train", argv[index]) == 0) {
            index++;
            training_set = argv[index];
        } else if (strcmp("--test", argv[index]) == 0) {
            index++;
            test_set = argv[index];
        } else if (strcmp("--epochs", argv[index]) == 0) {
            index++;
            max_epoch = strtol(argv[index], nullptr, 10);
        } else if (strcmp("--batch_size", argv[index]) == 0) {
            index++;
            batch_size = strtol(argv[index], nullptr, 10);
        } else if (strcmp("--hidden_units", argv[index]) == 0) {
            index++;
            hidden_units_string = argv[index];
        } else if (strcmp("--gpu", argv[index]) == 0) {
            isGpu = true;
            index++;
        } else if (strcmp("--help", argv[index]) == 0) {
            printUsage();
            return 0;
        }
        index++;
    }

    if (training_set.empty() || test_set.empty() || hidden_units_string.empty()) {
        std::cout << "ERROR: The mandatory arguments such as path to training and test data or "
        << "number of hidden units for mlp are not specified." << std::endl << std::endl;
        printUsage();
        return 1;
    }

    std::vector<int> hidden_units = getLayers(hidden_units_string);

    if (hidden_units.empty()) {
        std::cout << "ERROR: Number of hidden units are not provided in correct format."
        << "The numbers need to be separated by ' '." << std::endl << std::endl;
        printUsage();
        return 1;
    }

    /*
     * The MNIST data in CSV format has 785 columns.
     * The first column is "Label" and rest of the columns contain data.
     * The mnist_train.csv has 60000 records and mnist_test.csv has
     * 10000 records.
     */
    auto train_iter = MXDataIter("CSVIter")
    .SetParam("data_csv", training_set)
    .SetParam("data_shape", Shape(num_mnist_features + 1, 1))
    .SetParam("batch_size", batch_size)
    .SetParam("flat", 1)
    .SetParam("shuffle", 0)
    .CreateDataIter();

    auto val_iter = MXDataIter("CSVIter")
    .SetParam("data_csv", test_set)
    .SetParam("data_shape", Shape(num_mnist_features + 1, 1))
    .SetParam("batch_size", batch_size)
    .SetParam("flat", 1)
    .SetParam("shuffle", 0)
    .CreateDataIter();

    TRY
    auto net = mlp(hidden_units);

    Context ctx = Context::cpu();
    if (isGpu) {
        ctx = Context::gpu();
    }

    std::map<std::string, NDArray> args;
    args["data"] = NDArray(Shape(batch_size, num_mnist_features), ctx);
    args["label"] = NDArray(Shape(batch_size), ctx);
    // Let MXNet infer shapes other parameters such as weights
    net.InferArgsMap(ctx, &args, args);

    // Initialize all parameters with uniform distribution U(-0.01, 0.01)
    auto initializer = Uniform(0.01);
    for (auto& arg : args) {
        // arg.first is parameter name, and arg.second is the value
        initializer(arg.first, &arg.second);
    }

    // Create sgd optimiz er
    Optimizer* opt = OptimizerRegistry::Find("sgd");
    opt->SetParam("rescale_grad", 1.0/batch_size)
    ->SetParam("lr", learning_rate)
    ->SetParam("wd", weight_decay);

    // Create executor by binding parameters to the model
    auto *exec = net.SimpleBind(ctx, args);
    auto arg_names = net.ListArguments();

    // Start training
    for (int iter = 0; iter < max_epoch; ++iter) {
        int samples = 0;
        train_iter.Reset();

        auto tic = std::chrono::system_clock::now();
        while (train_iter.Next()) {
            samples += batch_size;
            auto data_batch = train_iter.GetDataBatch();

            /*
             * The shape of data_batch.data is (batch_size, (num_mnist_features + 1))
             * Need to reshape this data so that label column can be extracted from this data.
             */
            NDArray reshapedData = data_batch.data.Reshape(Shape((num_mnist_features + 1),
                                                                 batch_size));

            /*
             * Extract the label data by slicing the first column of the data and
             * copy it to "label" arg.
             */
            reshapedData.Slice(0, 1).Reshape(Shape(batch_size)).CopyTo(&args["label"]);

            /*
             * Extract the feature data by slicing the columns 1 to 785 of the data and
             * copy it to "data" arg.
             */
            reshapedData.Slice(1, (num_mnist_features + 1)).Reshape(Shape(batch_size,
                                                                         num_mnist_features))
                                                           .CopyTo(&args["data"]);

            exec->Forward(true);

            // Compute gradients
            exec->Backward();
            // Update parameters
            for (size_t i = 0; i < arg_names.size(); ++i) {
                if (arg_names[i] == "data" || arg_names[i] == "label") continue;
                opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
            }
        }
        auto toc = std::chrono::system_clock::now();

        Accuracy acc;
        val_iter.Reset();
        while (val_iter.Next()) {
            auto data_batch = val_iter.GetDataBatch();

            /*
             * The shape of data_batch.data is (batch_size, (num_mnist_features + 1))
             * Need to reshape this data so that label column can be extracted from this data.
             */
            NDArray reshapedData = data_batch.data.Reshape(Shape((num_mnist_features + 1),
                                                                 batch_size));

            /*
             * Extract the label data by slicing the first column of the data and
             * copy it to "label" arg.
             */
            NDArray labelData = reshapedData.Slice(0, 1).Reshape(Shape(batch_size));
            labelData.CopyTo(&args["label"]);

            /*
             * Extract the feature data by slicing the columns 1 to 785 of the data and
             * copy it to "data" arg.
             */
            reshapedData.Slice(1, (num_mnist_features + 1)).Reshape(Shape(batch_size,
                                                                         num_mnist_features))
                                                                   .CopyTo(&args["data"]);

            // Forward pass is enough as no gradient is needed when evaluating
            exec->Forward(false);
            acc.Update(labelData, exec->outputs[0]);
        }
        float duration = std::chrono::duration_cast<std::chrono::milliseconds>
        (toc - tic).count() / 1000.0;
        LG << "Epoch[" << iter << "]  " << samples/duration << " samples/sec Accuracy: "
        << acc.Get();
    }

    delete exec;
    delete opt;
    MXNotifyShutdown();
    CATCH
    return 0;
}
