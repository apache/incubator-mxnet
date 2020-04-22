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

# MXNet Gluon Fit API

In this tutorial, you will learn how to use the [Gluon Fit API](https://cwiki.apache.org/confluence/display/MXNET/Gluon+Fit+API+-+Tech+Design) which is the easiest way to train deep learning models using the [Gluon API](/api/python/docs/tutorials/packages/gluon/index.html) in Apache MXNet.

With the Fit API, you can train a deep learning model with a minimal amount of code. Just specify the network, loss function and the data you want to train on. You don't need to worry about the boiler plate code to loop through the dataset in batches (often called as 'training loop'). Advanced users can train with bespoke training loops, and many of these use cases will be covered by the Fit API.

To demonstrate the Fit API, you will train an image classification model using the [ResNet-18](https://arxiv.org/abs/1512.03385) neural network architecture. The model will be trained using the [Fashion-MNIST dataset](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/).

## Prerequisites

To complete this tutorial, you will need:

- [MXNet](/get_started) (The version of MXNet will be >= 1.5.0, you can use `pip install mxnet` to get 1.5.0 release pip package or build from source with master, refer to [MXNet installation](/get_started?version=master&platform=linux&language=python&environ=pip&processor=cpu)
- [Jupyter Notebook](https://jupyter.org/index.html) (For interactively running the provided .ipynb file)


```python
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.contrib.estimator.event_handler import TrainBeginMixin, TrainEndMixin, EpochEndMixin, CheckpointHandler

gpu_count = mx.context.num_gpus()
ctx = [mx.gpu(i) for i in range(gpu_count)] if gpu_count > 0 else mx.cpu()
```

## Dataset

[Fashion-MNIST](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/) dataset consists of fashion items divided into ten categories: t-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot.

- It has 60,000 grayscale images of size 28 * 28 for training.
- It has 10,000 grayscale images of size 28 * 28 for testing/validation.

We will use the ```gluon.data.vision``` package to directly import the Fashion-MNIST dataset and perform pre-processing on it.


```python
# Get the training data
fashion_mnist_train = gluon.data.vision.FashionMNIST(train=True)

# Get the validation data
fashion_mnist_val = gluon.data.vision.FashionMNIST(train=False)
```


```python
transforms = [gluon.data.vision.transforms.Resize(224), # We pick 224 as the model we use takes an input of size 224.
                gluon.data.vision.transforms.ToTensor()]

# Now we will stack all these together.
transforms = gluon.data.vision.transforms.Compose(transforms)
```


```python
# Apply the transformations
fashion_mnist_train = fashion_mnist_train.transform_first(transforms)
fashion_mnist_val = fashion_mnist_val.transform_first(transforms)
```


```python
batch_size = 256 # Batch size of the images
num_workers = 4 # The number of parallel workers for loading the data using Data Loaders.

train_data_loader = gluon.data.DataLoader(fashion_mnist_train, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
val_data_loader = gluon.data.DataLoader(fashion_mnist_val, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers)
```

## Model and Optimizers

Let's load the resnet-18 model architecture from [Gluon Model Zoo](https://mxnet.apache.org/api/python/gluon/model_zoo.html) and initialize its parameters. The Gluon Model Zoo contains a repository of pre-trained models as well the model architecture definitions. We are using the model architecture from the model zoo in order to train it from scratch.


```python
resnet_18_v1 = vision.resnet18_v1(pretrained=False, classes = 10)
resnet_18_v1.initialize(init = mx.init.Xavier(), ctx=ctx)
```

We will be using `SoftmaxCrossEntropyLoss` as the loss function since this is a multi-class classification problem. We will be using `sgd` (Stochastic Gradient Descent) as the optimizer.
You can experiment with a [different loss](/api/python/docs/api/gluon/loss/index.html) or [optimizer](/api/python/docs/api/optimizer/index.html) as well.


```python
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
```

Let's define the trainer object for training the model.


```python
learning_rate = 0.04 # You can experiment with your own learning rate here
num_epochs = 2 # You can run training for more epochs
trainer = gluon.Trainer(resnet_18_v1.collect_params(),
                        'sgd', {'learning_rate': learning_rate})
```

## Train using Fit API

As stated earlier, the Fit API greatly simplifies the boiler plate code and complexity for training using MXNet Gluon.

In the basic usage example, with just 2 lines of code, we will set up our model for training.

### Basic Usage


```python
train_acc = mx.metric.Accuracy() # Metric to monitor

# Define the estimator, by passing to it the model, loss function, metrics, trainer object and context
est = estimator.Estimator(net=resnet_18_v1,
                          loss=loss_fn,
                          metrics=train_acc,
                          trainer=trainer,
                          context=ctx)

# ignore warnings for nightly test on CI only
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Magic line
    est.fit(train_data=train_data_loader,
        epochs=num_epochs)
```

```text
    Training begin: using optimizer SGD with current learning rate 0.0400 <!--notebook-skip-line-->
    Train for 2 epochs. <!--notebook-skip-line-->

    [Epoch 0] finished in 25.110s: train_accuracy : 0.7877 train_softmaxcrossentropyloss0 : 0.5905 <!--notebook-skip-line-->

    [Epoch 1] finished in 23.595s: train_accuracy : 0.8823 train_softmaxcrossentropyloss0 : 0.3197 <!--notebook-skip-line-->
    Train finished using total 48s at epoch 1. train_accuracy : 0.8823 train_softmaxcrossentropyloss0 : 0.3197 <!--notebook-skip-line-->
```

### Advanced Usage

The Fit API is also customizable with several `Event Handlers` which give a fine grained control over the steps in training and exposes callback methods that provide control over the stages involved in training. Available callback methods are: `train_begin`, `train_end`, `batch_begin`, `batch_end`, `epoch_begin` and `epoch_end`.

You can use built-in event handlers such as `LoggingHandler`, `CheckpointHandler` or `EarlyStoppingHandler` to log and save the model at certain time-steps during training. You can also stop the training when the model's performance plateaus.
There are also some default utility handlers that will be added to your estimator by default. For example, `StoppingHandler` is used to control when the training ends, based on number of epochs or number of batches trained.
`MetricHandler` is used to calculate training metrics at end of each batch and epoch.
`ValidationHandler` is used to validate your model on test data at each epoch's end and then calculate validation metrics.
You can create these utility handlers with different configurations and pass to estimator. This will override the default handler configuration.
You can create a custom handler by inheriting one or multiple
[base event handlers](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/contrib/estimator/event_handler.py#L32)
 including: `TrainBeginMixin`, `TrainEndMixin`, `EpochBeginMixin`, `EpochEndMixin`, `BatchBeginMixin`, `BatchEndMixin`.


### Custom Event Handler

Here we will showcase an example custom event handler that inherits features from a few base handler classes.
Our custom event handler is a simple one: record the loss values at the end of every epoch in our training phase.

Note: For each of the method, the `Estimator` object is passed along, so you can access training metrics.

```python
class LossRecordHandler(TrainBeginMixin, TrainEndMixin, EpochEndMixin):
    def __init__(self):
        super(LossRecordHandler, self).__init__()
        self.loss_history = {}

    def train_begin(self, estimator, *args, **kwargs):
        print("Training begin")

    def train_end(self, estimator, *args, **kwargs):
        # Print all the losses at the end of training
        print("Training ended")
        for loss_name in self.loss_history:
            for i, loss_val in enumerate(self.loss_history[loss_name]):
                print("Epoch: {}, Loss name: {}, Loss value: {}".format(i, loss_name, loss_val))

    def epoch_end(self, estimator, *args, **kwargs):
        for metric in estimator.train_metrics:
            # look for train Loss in training metrics
            # we wrapped loss value as a metric to record it
            if isinstance(metric, mx.metric.Loss):
                loss_name, loss_val = metric.get()
                # append loss value for this epoch
                self.loss_history.setdefault(loss_name, []).append(loss_val)
```


```python
# Let's reset the model, trainer and accuracy objects from above

resnet_18_v1.initialize(force_reinit=True, init = mx.init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(resnet_18_v1.collect_params(),
                        'sgd', {'learning_rate': learning_rate})
train_acc = mx.metric.Accuracy()
```


```python
# Define the estimator, by passing to it the model, loss function, metrics, trainer object and context
est = estimator.Estimator(net=resnet_18_v1,
                          loss=loss_fn,
                          metrics=train_acc,
                          trainer=trainer,
                          context=ctx)

# Define the handlers, let's say in built Checkpointhandler
checkpoint_handler = CheckpointHandler(model_dir='./',
                                       model_prefix='my_model',
                                       monitor=train_acc,  # Monitors a metric
                                       save_best=True)  # Save the best model in terms of
# Let's instantiate another handler which we defined above
loss_record_handler = LossRecordHandler()
# ignore warnings for nightly test on CI only
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Magic line
    est.fit(train_data=train_data_loader,
            val_data=val_data_loader,
            epochs=num_epochs,
            event_handlers=[checkpoint_handler, loss_record_handler]) # Add the event handlers
```

```text
    Training begin: using optimizer SGD with current learning rate 0.0400 <!--notebook-skip-line-->
    Train for 2 epochs. <!--notebook-skip-line-->

    [Epoch 0] finished in 25.236s: train_accuracy : 0.7917 train_softmaxcrossentropyloss0 : 0.5741 val_accuracy : 0.6612 val_softmaxcrossentropyloss0 : 0.8627 <!--notebook-skip-line-->

    [Epoch 1] finished in 24.892s: train_accuracy : 0.8826 train_softmaxcrossentropyloss0 : 0.3229 val_accuracy : 0.8474 val_softmaxcrossentropyloss0 : 0.4262 <!--notebook-skip-line-->

    Train finished using total 50s at epoch 1. train_accuracy : 0.8826 train_softmaxcrossentropyloss0 : 0.3229 val_accuracy : 0.8474 val_softmaxcrossentropyloss0 : 0.4262 <!--notebook-skip-line-->

    Training begin <!--notebook-skip-line-->
    Epoch 1, loss 0.5741 <!--notebook-skip-line-->
    Epoch 2, loss 0.3229 <!--notebook-skip-line-->
```

You can load the saved model, by using the `load_parameters` API in Gluon. For more details refer to the [Loading model parameters from file tutorial](/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html#saving-model-parameters-to-file)


```python
resnet_18_v1 = vision.resnet18_v1(pretrained=False, classes=10)
resnet_18_v1.load_parameters('./my_model-best.params', ctx=ctx)
```

## Next Steps

- For more hands on learning about deep learning, check out [Dive into Deep Learning](https://d2l.ai)
