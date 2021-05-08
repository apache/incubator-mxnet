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


# Gotchas using NumPy in Apache MXNet

The goal of this tutorial is to explain some common misconceptions about using [NumPy](http://www.numpy.org/) arrays in Apache MXNet. We are going to explain why you need to minimize or completely remove usage of NumPy from your Apache MXNet code. We also going to show how to minimize NumPy performance impact, when you have to use NumPy.

Warning: The latest MXNet offers NumPy-compatible array class `mx.np.ndarray` and NDArray is now a legacy array class in MXNet 1.x. This tutorial is just for reference for the legacy NDArray.

## Asynchronous and non-blocking nature of Apache MXNet

Instead of using NumPy arrays Apache MXNet offers its own array implementation named [NDArray](../../../../api/legacy/ndarray/ndarray.rst). `NDArray API` was intentionally designed to be similar to `NumPy`, but there are differences.

One key difference is in the way calculations are executed. Every `NDArray` manipulation in Apache MXNet is done in asynchronous, non-blocking way. That means, that when we write code like `c = a * b`, where both `a` and `b` are `NDArrays`, the function is pushed to the [Execution Engine](https://mxnet.apache.org/api/architecture/overview.html#execution-engine), which starts the calculation. The function immediately returns back, and the  user thread can continue execution, despite the fact that the calculation may not have been completed yet.

`Execution Engine` builds the computation graph which may reorder or combine some calculations, but it honors dependency order: if there are other manipulation with `c` done later in the code, the `Execution Engine` will start doing them once the result of `c` is available. We don't need to write callbacks to start execution of subsequent code - the `Execution Engine` is going to do it for us.

To get the result of the computation we only need to access the resulting variable, and the flow of the code will be blocked until the computation results are assigned to the resulting variable. This behavior allows to increase code performance while still supporting imperative programming mode.

Refer to the [intro tutorial to NDArray](./index.ipynb), if you are new to Apache MXNet and would like to learn more how to manipulate NDArrays.

## Converting NDArray to NumPy Array blocks calculation

Many people are familiar with NumPy and flexible doing tensor manipulations using it. `NDArray API` offers  a convinient [.asnumpy() method](../../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.NDArray.asnumpy) to cast `nd.array` to `np.array`. However, by doing this cast and using `np.array` for calculation, we cannot use all the goodness of `Execution Engine`. All manipulations done on `np.array` are blocking. Moreover, the cast to `np.array` itself is a blocking operation (same as [.asscalar()](../../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.NDArray.asscalar), [.wait_to_read()](../../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.NDArray.wait_to_read) and [.waitall()](../../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.waitall)).

That means that if we have a long computation graph and, at some point, we want to cast the result to `np.array`, it may feel like the casting takes a lot of time. But what really takes this time is `Execution Engine`, which finishes all the async calculations we have pushed into it to get the final result, which then will be converted to `np.array`.

Because of the blocking nature of [.asnumpy() method](../../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.NDArray.asnumpy), using it reduces the execution performance, especially if the calculations are done on GPU: Apache MXNet has to copy data from GPU to CPU to return `np.array`.

The best solution is to **make manipulations directly on NDArrays by methods provided in [NDArray API](../../../../api/legacy/ndarray/ndarray.rst)**.

## NumPy operators vs. NDArray operators

Despite the fact that [NDArray API](../../../../api/legacy/ndarray/ndarray.rst) was specifically designed to be similar to `NumPy`, sometimes it is not easy to replace existing `NumPy` computations. The main reason is that not all operators, that are available in `NumPy`, are available in `NDArray API`. The list of currently available operators is available on [NDArray class page](../../../../api/legacy/ndarray/ndarray.rst).

If a required operator is missing from `NDArray API`, there are few things you can do.

### Combine a higher level operator using a few lower level operators

There are a situation, when you can assemble a higher level operator using existing operators. An example for that is the [np.full_like()](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.full_like.html) operator. This operator doesn't exist in `NDArray API`, but can be easily replaced with a combination of existing operators.


```{.python .input}
from mxnet import nd
import numpy as np

# NumPy has full_like() operator
np_y = np.full_like(a=np.arange(6, dtype=int), fill_value=10)

# NDArray doesn't have it, but we can replace it with
# creating an array of ones and then multiplying by fill_value
nd_y = nd.ones(shape=(6,)) * 10

# To compare results we had to convert NDArray to NumPy
# But this is okay for that particular case
np.array_equal(np_y, nd_y.asnumpy())
```

```True``` <!--notebook-skip-line-->

### Find similar operator with different name and/or signature

Some operators may have slightly different name, but are similar in terms of functionality. For example [nd.ravel_multi_index()](../../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.ravel_multi_index) is similar to [np.ravel()](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ma.ravel.html#numpy.ma.ravel). In other cases some operators may have similar names, but different signatures. For example [np.split()](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.split.html#numpy.split) and [nd.split()](../../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.split) are similar, but the former works with indices and the latter requires the number of splits to be provided.

One particular example of different input requirements is [nd.pad()](../../../../api/legacy/ndarray/ndarray.rst#mxnet.ndarray.pad). The trick is that it can only work with 4-dimensional tensors. If your input has less dimensions, then you need to expand its number before using `nd.pad()` as it is shown in the code block below:


```{.python .input}
def pad_array(data, max_length):
    # expand dimensions to 4, because nd.pad can work only with 4 dims
    data_expanded = data.reshape(1, 1, 1, data.shape[0])

    # pad all 4 dimensions with constant value of 0
    data_padded = nd.pad(data_expanded,
                             mode='constant',
                             pad_width=[0, 0, 0, 0, 0, 0, 0, max_length - data.shape[0]],
                             constant_value=0)

    # remove temporary dimensions
    data_reshaped_back = data_padded.reshape(max_length)
    return data_reshaped_back

pad_array(nd.array([1, 2, 3]), max_length=10)
```

`[ 1.  2.  3.  0.  0.  0.  0.  0.  0.  0.]` <!--notebook-skip-line-->


`<NDArray 10 @cpu(0)>` <!--notebook-skip-line-->


### Search for an operator on [GitHub](https://github.com/apache/incubator-mxnet/labels/Operator)

Apache MXNet community is responsive to requests, and everyone is welcomed to contribute new operators. Have in mind, that there is always a lag between new operators being merged into the codebase and release of a next stable version. For example, [nd.diag()](https://github.com/apache/incubator-mxnet/pull/11643) operator was recently introduced to Apache MXNet, but on the moment of writing this tutorial, it is not in any stable release. You can always get all latest implementations by installing the [master version](https://mxnet.apache.org/get_started?version=master&platform=linux&language=python&environ=pip&processor=cpu#) of Apache MXNet.

## How to minimize the impact of blocking calls

There are cases, when you have to use either `.asnumpy()` or `.asscalar()` methods. As it is explained before, this will force Apache MXNet to block the execution until the result can be retrieved. One common use case is printing a metric or a value of a loss function.

You can minimize the impact of a blocking call by calling `.asnumpy()` or `.asscalar()` in the moment, when you think the calculation of this value is already done. In the example below, we introduce the `LossBuffer` class. It is used to cache the previous value of a loss function. By doing so, we delay printing by one iteration in hope that the `Execution Engine` would finish the previous iteration and blocking time would be minimized.


```{.python .input}
from __future__ import print_function

import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.ndarray import NDArray
from mxnet.gluon import HybridBlock
import numpy as np

class LossBuffer(object):
    """
    Simple buffer for storing loss value
    """
    def __init__(self):
        self._loss = None

    def new_loss(self, loss):
        ret = self._loss
        self._loss = loss
        return ret

    @property
    def loss(self):
        return self._loss


net = gluon.nn.Dense(10)
ce = gluon.loss.SoftmaxCELoss()
net.initialize()

data = nd.random.uniform(shape=(1024, 100))
label = nd.array(np.random.randint(0, 10, (1024,)), dtype='int32')
train_dataset = gluon.data.ArrayDataset(data, label)
train_data = gluon.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

trainer = gluon.Trainer(net.collect_params(), optimizer='sgd')
loss_buffer = LossBuffer()

for data, label in train_data:
    with autograd.record():
        out = net(data)
        # This call saves new loss and returns previous loss
        prev_loss = loss_buffer.new_loss(ce(out, label))

    loss_buffer.loss.backward()
    trainer.step(data.shape[0])

    if prev_loss is not None:
        print("Loss: {}".format(np.mean(prev_loss.asnumpy())))
```

```text
    Loss: 2.310760974884033 <!--notebook-skip-line-->

    Loss: 2.334498643875122 <!--notebook-skip-line-->

    Loss: 2.3244147300720215 <!--notebook-skip-line-->

    Loss: 2.332686424255371 <!--notebook-skip-line-->

    Loss: 2.321366310119629 <!--notebook-skip-line-->

    Loss: 2.3236165046691895 <!--notebook-skip-line-->

    Loss: 2.3178648948669434 <!--notebook-skip-line-->
```

## Conclusion

For performance reasons, it is better to use native `NDArray API` methods and avoid using NumPy altogether. In case when you must use NumPy, you can use convenient method `.asnumpy()` on `NDArray` to get NumPy representation. By doing so, you block the whole computational process, and force data to be synced between CPU and GPU. If it is a necessary evil to do that, try to minimize the blocking time by calling `.asnumpy()` in time, when you expect the value to be already computed.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
