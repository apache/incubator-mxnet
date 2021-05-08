<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# MXNet C++ Package Examples

## Building C++ examples

The examples in this folder demonstrate the **training** workflow. The **inference workflow** related examples can be found in [inference](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inference>) folder.
Please build the MXNet C++ Package as explained in the [README](<https://github.com/apache/incubator-mxnet/tree/master/cpp-package#building-c-package>) File before building these examples manually.
The examples in this folder are built while building the MXNet library and cpp-package from source. However, they can be built manually as follows

From cpp-package/examples directory

-  Build all examples in release mode: **make all**
-  Build all examples in debug mode: **make debug**

By default, the examples are built to be run on GPU. To build examples to run on CPU:

-  Release: **make all MXNET\_USE\_CPU=1**
-  Debug: **make debug MXNET\_USE\_CPU=1**

The examples that are built to be run on GPU may not work on the non-GPU machines.
The makefile will also download the necessary data files and store in a data folder. (The download will take couple of minutes, but will be done only once on a fresh installation.)


## Examples demonstrating training workflow

This directory contains following examples. In order to run the examples, ensure that the path to the MXNet shared library is added to the OS specific environment variable viz. **LD\_LIBRARY\_PATH** for Linux, Mac and Ubuntu OS and **PATH** for Windows OS. For example `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/ubuntu/incubator-mxnet/lib` on ubuntu using gpu.

### [alexnet.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/alexnet.cpp>)

The example implements the C++ version of AlexNet. The networks trains on MNIST data. The number of epochs can be specified as a command line argument. For example to train with 10 epochs use the following:

```
build/alexnet 10
```

### [googlenet.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/googlenet.cpp>)

The code implements a GoogLeNet/Inception network using the C++ API. The example uses MNIST data to train the network. By default, the example trains the model for 100 epochs. The number of epochs can also be specified in the command line. For example, to train the model for 10 epochs use the following:

```
build/googlenet 10
```

### [mlp.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp.cpp>)

The code implements a multilayer perceptron from scratch. The example creates its own dummy data to train the model. The example does not require command line parameters. It trains the model for 20,000 epochs.
To run the example use the following command:

```
build/mlp
```

### [mlp_cpu.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp_cpu.cpp>)

The code implements a multilayer perceptron to train the MNIST data. The code demonstrates the use of "SimpleBind"  C++ API and MNISTIter. The example is designed to work on CPU. The example does not require command line parameters.
To run the example use the following command:

```
build/mlp_cpu
```

### [mlp_gpu.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp_gpu.cpp>)

The code implements a multilayer perceptron to train the MNIST data. The code demonstrates the use of the "SimpleBind"  C++ API and MNISTIter. The example is designed to work on GPU. The example does not require command line arguments. To run the example execute following command:

```
build/mlp_gpu
```

### [mlp_csv.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp_csv.cpp>)

The code implements a multilayer perceptron to train the MNIST data. The code demonstrates the use of the "SimpleBind"  C++ API and CSVIter. The CSVIter can iterate data that is in CSV format. The example can be run on CPU or GPU. The example usage is as follows:

```
build/mlp_csv --train data/mnist_data/mnist_train.csv --test data/mnist_data/mnist_test.csv --epochs 10 --batch_size 100 --hidden_units "128 64 64" --gpu
```
* To get the `mnist_training_set.csv` and `mnist_test_set.csv` please run the following command:
```python
# in incubator-mxnet/cpp-package/example directory
python mnist_to_csv.py ./data/mnist_data/train-images-idx3-ubyte ./data/mnist_data/train-labels-idx1-ubyte ./data/mnist_data/mnist_train.csv 60000
python mnist_to_csv.py ./data/mnist_data/t10k-images-idx3-ubyte ./data/mnist_data/t10k-labels-idx1-ubyte ./data/mnist_data/mnist_test.csv 10000
```

### [resnet.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/resnet.cpp>)

The code implements a resnet model using the C++ API. The model is used to train MNIST data. The number of epochs for training the model can be specified on the command line. By default, model is trained for 100 epochs. For example, to train with 10 epochs use the following command:

```
build/resnet 10
```

### [lenet.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/lenet.cpp>)

The code implements a lenet model using the C++ API. It uses MNIST training data in CSV format to train the network. The example does not use built-in CSVIter to read the data from CSV file. The number of epochs can be specified on the command line. By default, the mode is trained for 100,000 epochs. For example, to train with 10 epochs use the following command:

```
build/lenet 10
```
### [lenet\_with\_mxdataiter.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/mlp_cpu.cpp>)

The code implements a lenet model using the C++ API. It uses MNIST training data to train the network. The example uses built-in MNISTIter to read the data. The number of epochs can be specified on the command line. By default, the mode is trained for 100 epochs. For example, to train with 10 epochs use the following command:

```
build/lenet_with_mxdataiter 10
```

In addition, there is `run_lenet_with_mxdataiter.sh` that downloads the mnist data and run `lenet_with_mxdataiter` example.

### [inception_bn.cpp](<https://github.com/apache/incubator-mxnet/blob/master/cpp-package/example/inception_bn.cpp>)

The code implements an Inception network using the C++ API with batch normalization. The example uses MNIST data to train the network. The model trains for 100 epochs. The example can be run by executing the following command:

```
build/inception_bn
```
