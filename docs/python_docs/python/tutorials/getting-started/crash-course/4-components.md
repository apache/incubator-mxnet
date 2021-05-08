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

# Step 4: Necessary components that are not in the network

Data and models are not the only components that
you need to train a deep learning model. In this notebook, you will
learn about the common components involved in training deep learning models. 
Here is a list of components necessary for training models in MXNet.

1. Initialization
2. Loss functions
    1. Built-in
    2. Custom
3. Optimizers
4. Metrics

```{.python .input}
from mxnet import np, npx,gluon
import mxnet as mx
from mxnet.gluon import nn
npx.set_np()

ctx = mx.cpu()
```

## Initialization

In a previous notebook, you used `net.initialize()` to initialize the network
before a forward pass. Now, you will learn about initialization in a little more
detail.

First, define and initialize the `sequential` network from earlier.
After you initialize it, print the parameters using `collect_params()` method.

```{.python .input}
net = nn.Sequential()

net.add(nn.Dense(5, in_units=3, activation="relu"),
        nn.Dense(25, activation="relu"),
        nn.Dense(2)
       )

net
```

```{.python .input}
net.initialize()
params = net.collect_params()

for key, value in params.items():
    print(key, value)


```

Next, you will print shape and params after the first forward pass.

```{.python .input}
x = np.random.uniform(-1, 1, (10, 3))
net(x)  # Forward computation

params = net.collect_params()
for key, value in params.items():
    print(key, value)


```

#### Built-in Initialization

MXNet makes it easy to initialize by providing many common initializers. A subset that you will be using in the following sections include:

- Constant
- Normal

For more information, see
[Initializers](../../../api/initializer/index.rst)

When you use `net.intialize()`, MXNet, by default, initializes the weight matrices uniformly
by drawing random values with a uniform-distribution between −0.07 and 0.07 and
updates the bias parameters by setting them all to 0.

To initialize your network using different built-in types, you have to use the
`init` keyword argument in the `initialize()` method. If you have
already initialized the weight but want to reinitialize the weight, set the
`force_reinit` flag to `True`. Here is an example using
`constant` and `normal` initialization. 

```{.python .input}
from mxnet import init

# Constant init initializes the weights to be a constant value for all the params
net.initialize(init=init.Constant(3), force_reinit=True, ctx=ctx)
print(net[0].weight.data()[0])
```

If you use Normal to initialize your weights then you will use a normal
distribution with a mean of zero and standard deviation of sigma. 

```{.python .input}
net.initialize(init=init.Normal(sigma=0.2), force_reinit=True, ctx=ctx)
print(net[0].weight.data()[0])
```

## Components used in a training loop

Till now you have seen how to create an algorithm and how to initialize it using mxnet
APIs; additionally you have learned the basics of using mxnet. When you start training the
ML algorithm, how do you actually teach the algorithm to learn or train?

There are three main components for training an algorithm.

1. Loss function: calculates how far the model is from the true distribution
2. Autograd: the mxnet auto differentiation tool that calculates the gradients to
optimize the parameters
3. Optimizer: updates the parameters based on an optimization algorithm

You have already learned about autograd in the previous notebook. In this
notebook, you will learn more about loss functions and optimizers.

## Loss function

Loss functions are used to train neural networks and help the algorithm learn
from the data. The loss function computes the difference between the
output from the neural network and ground truth. This output is used to
update the neural network weights during training. Next, you will look at a
simple example.

Suppose you have a neural network `net` and the data is stored in a variable
`data`. The data consists of 5 total records (rows) and two features (columns)
and the output from the neural network after the first epoch is given by the
variable `nn_output`.

```{.python .input}
net = gluon.nn.Dense(1)
net.initialize()

nn_input = np.array([[1.2, 0.56],
                     [3.0, 0.72],
                     [0.89, 0.9],
                     [0.89, 2.3],
                     [0.99, 0.52]])

nn_output = net(nn_input)
nn_output
```

The ground truth value of the data is stored in `groundtruth_label` is

```{.python .input}
groundtruth_label = np.array([[0.0083],
                             [0.00382],
                             [0.02061],
                             [0.00495],
                             [0.00639]]).reshape(5, 1)
```

For this problem, you will use the L2 Loss. L2Loss, also called Mean Squared Error, is a
regression loss function that computes the squared distances between the target
values and the output of the neural network. It is defined as:

$$L = \frac{1}{2N}\sum_i{|label_i − pred_i|)^2}$$

The L2 loss function creates larger gradients for loss values which are farther apart due to the
square operator and it also smooths the loss function space. 

```{.python .input}
def L2Loss(output_values, true_values):
    return np.mean((output_values - true_values) ** 2, axis=1) / 2

L2Loss(nn_output, groundtruth_label)
```

Now, you can do the same thing using the mxnet API

```{.python .input}
from mxnet.gluon import nn, loss as gloss
loss = gloss.L2Loss()

loss(nn_output, groundtruth_label)
```

A network can improve by iteratively updating its weights to minimise the loss.
Some tasks use a combination of multiple loss functions, but often you will just
use one. MXNet Gluon provides a number of the most commonly used loss functions.
The choice of your loss function will depend on your network and task. Some
common tasks and loss function pairs include:

- regression: L1Loss, L2Loss

- classification: SigmoidBinaryCrossEntropyLoss, SoftmaxCrossEntropyLoss

- embeddings: HingeLoss

#### Customizing your Loss functions

You can also create custom loss functions using **Loss Blocks**.

You can inherit the base `Loss` class and write your own `forward` method. The
backward propagation will be automatically computed by autograd. However, that
only holds true if you can build your loss from existing mxnet operators.

```{.python .input}
from mxnet.gluon.loss import Loss

class custom_L1_loss(Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(custom_L1_loss, self).__init__(weight, batch_axis, **kwargs)

    def forward(self, pred, label):
        l = np.abs(label - pred)
        l = l.reshape(len(l),)
        return l
    
L1 = custom_L1_loss()
L1(nn_output, groundtruth_label)
```

```{.python .input}
l1=gloss.L1Loss()
l1(nn_output, groundtruth_label)
```

## Optimizer

The loss function determines how much to change the parameters based on how far the
model is from the groundtruth. Optimizer determines how the model
weights or parameters are updated based on the loss function. In Gluon, this
optimization step is performed by the `gluon.Trainer`.

Here is a basic example of how to call the `gluon.Trainer` method.

```{.python .input}
from mxnet import optimizer
```

```{.python .input}
trainer = gluon.Trainer(net.collect_params(),
                        optimizer="Adam",
                        optimizer_params={
                            "learning_rate":0.1,
                            "wd":0.001
                        })
```

When creating a **Gluon Trainer**, you must provide the trainer object with
1. A collection of parameters that need to be learnt. The collection of
parameters will be the weights and biases of your network that you are training.
2. An Optimization algorithm (optimizer) that you want to use for training. This
algorithm will be used to update the parameters every training iteration when
`trainer.step` is called. For more information, see
[optimizers](../../../api/optimizer/index.rst)

```{.python .input}
curr_weight = net.weight.data()
print(curr_weight)
```

```{.python .input}
batch_size = len(nn_input)
trainer.step(batch_size)
print(net.weight.data())
```

```{.python .input}
print(curr_weight - net.weight.grad() * 1 / 5)
```

## Metrics

MXNet includes a `metrics` API that you can use to evaluate how your model is
performing. This is typically used during training to monitor performance on the
validation set. MXNet includes many commonly used metrics, a few are listed below:

- [Accuracy](../../../api/gluon/metric/index.rst#mxnet.gluon.metric.Accuracy)
- [CrossEntropy](../../../api/gluon/metric/index.rst#mxnet.gluon.metric.CrossEntropy)
- [Mean squared error](../../../api/gluon/metric/index.rst#mxnet.gluon.metric.MSE)
- [Root mean squared error (RMSE)](../../../api/gluon/metric/index.rst#mxnet.gluon.metric.RMSE)

Now, you will define two arrays for a dummy binary classification example.

```{.python .input}
# Vector of likelihoods for all the classes
pred = np.array([[0.1, 0.9], [0.05, 0.95], [0.83, 0.17], [0.63, 0.37]])

labels = np.array([1, 1, 0, 1])
```

Before you can calculate the accuracy of your model, the metric (accuracy)
should be instantiated before the training loop

```{.python .input}
from mxnet.gluon.metric import Accuracy

acc = Accuracy()
```

To run and calculate the updated accuracy for each batch or epoch, you can call
the `update()` method. This method uses labels and predictions which can be
either class indexes or a vector of likelihoods for all of the classes.

```{.python .input}
acc.update(labels=labels, preds=pred)
```

#### Creating custom metrics

In addition to built-in metrics, if you want to create a custom metric, you can
use the following skeleton code. This code inherits from the `EvalMetric` base
class.

```{.python .input}
def MyCustomMetric(EvalMetric):
    def __init__(self):
        super().init()

    def update(self, labels, preds):
        pass

```

Here is an example using the Precision metric. First, define the two values
`labels` and `preds`.

```{.python .input}
labels = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1])
preds = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
```

Next, define the custom metric class `precision` and instantiate it

```{.python .input}
from mxnet.gluon.metric import EvalMetric

class precision(EvalMetric):
    def __init__(self):
        super().__init__(name="Precision")
        
    def update(self,labels, preds):
        tp_labels = (labels == 1)
        true_positives = sum(preds[tp_labels] == 1)
        fp_labels = (labels == 0)
        false_positives = sum(preds[fp_labels] == 1)
        return true_positives / (true_positives + false_positives)
        
p = precision()
```

And finally, call the `update` method to return the results of `precision` for your data

```{.python .input}
p.update(np.array(y_true), np.array(y_pred))
```

## Next steps

Now that you have learned all the components required to train a neural network,
you will see how to load your data using the Gluon API in [Step 5: Gluon
Datasets and DataLoader](./5-datasets.ipynb)
