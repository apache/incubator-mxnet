---
layout: page_api
title: Basics
action: Get Started
action_url: /get_started
permalink: /api/cpp/docs/tutorials/basics.html
is_tutorial: true
tag: cpp
---
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

Basics
======

This tutorial provides basic usages of the C++ package through the classical handwritten digits
identification database--[MNIST](http://yann.lecun.com/exdb/mnist/).

The following contents assume that the working directory is `/path/to/mxnet/cpp-package/example`.

Load Data
--------
Before going into codes, we need to fetch MNIST data. You can either use the script `/path/to/mxnet/cpp-package/example/get_data.sh`,
or download mnist data by yourself from Lecun's [website](http://yann.lecun.com/exdb/mnist/)
and decompress them into `data/mnist_data` folder.

Except linking the MXNet shared library, the C++ package itself is a header-only package,
which means all you need to do is to include the header files. Among the header files,
`op.h` is special since it is generated dynamically. The generation should be done when
[building the C++ package]({{'/api/cpp/'|relative_url}}).
It is important to note that you need to **copy the shared library** (`libmxnet.so` in Linux and MacOS,
`libmxnet.dll` in Windows) from `/path/to/mxnet/lib` to the working directory.
We do not recommend you to use pre-built binaries because MXNet is under heavy development,
the operator definitions in `op.h` may be incompatible with the pre-built version.

In order to use functionalities provides by the C++ package, first we include the general
header file `MxNetCpp.h` and specify the namespaces.

```c++
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;
```

Next we can use the data iter to load MNIST data (separated to training sets and validation sets).
The digits in MNIST are 2-dimension arrays, so we should set `flat` to true to flatten the data.

```c++
auto train_iter = MXDataIter("MNISTIter")
    .SetParam("image", "./data/mnist_data/train-images-idx3-ubyte")
    .SetParam("label", "./data/mnist_data/train-labels-idx1-ubyte")
    .SetParam("batch_size", batch_size)
    .SetParam("flat", 1)
    .CreateDataIter();
auto val_iter = MXDataIter("MNISTIter")
    .SetParam("image", "./data/mnist_data/t10k-images-idx3-ubyte")
    .SetParam("label", "./data/mnist_data/t10k-labels-idx1-ubyte")
    .SetParam("batch_size", batch_size)
    .SetParam("flat", 1)
    .CreateDataIter();
```

The data have been successfully loaded. We can now easily construct various models to identify
the digits with the help of C++ package.

GPU Support
-----------
It's worth noting that changing context from `Context::cpu()` to `Context::gpu()` is not enough,
because the data read by data iter are stored in memory, we cannot assign it directly to the
parameters. To bridge this gap, NDArray provides data synchronization functionalities between
GPU and CPU. We will illustrate it by making the mlp code run on GPU.

In the previous code, data are used like

```c++
args["X"] = data_batch.data;
args["label"] = data_batch.label;
```

It will be problematic if other parameters are created in the context of GPU. We can use
`NDArray::CopyTo` to solve this problem.

```c++
// Data provided by DataIter are stored in memory, should be copied to GPU first.
data_batch.data.CopyTo(&args["X"]);
data_batch.label.CopyTo(&args["label"]);
// CopyTo is imperative, need to wait for it to complete.
NDArray::WaitAll();
```

By replacing the former code to the latter one, we successfully port the code to GPU.
You can find the complete code in `mlp_gpu.cpp`. Compilation is similar to the cpu version.
Note that the shared library must be built with GPU support enabled.
