# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

''' Unit tests for Gluon Estimator '''

import sys
import unittest
import warnings
import pytest

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.contrib.estimator import *
from mxnet.gluon.contrib.estimator.event_handler import *

mx.reset_np()

def _get_test_network(params=None):
    net = nn.Sequential()
    net.add(nn.Dense(4, activation='relu', flatten=False))
    net.share_parameters(params)
    return net

def _get_test_data():
    batch_size = 4
    in_data = mx.nd.random.uniform(shape=(10, 3))
    out_data = mx.nd.random.uniform(shape=(10, 4))
    # Input dataloader
    dataset = gluon.data.dataset.ArrayDataset(in_data, out_data)
    dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size)
    dataiter = mx.io.NDArrayIter(data=in_data, label=out_data, batch_size=batch_size)
    return dataloader, dataiter


def test_fit():
    ''' test estimator with different train data types '''
    net = _get_test_network()
    dataloader, dataiter = _get_test_data()
    num_epochs = 1
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    acc = mx.gluon.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=acc,
                    trainer=trainer,
                    context=ctx)

    est.fit(train_data=dataloader,
            epochs=num_epochs)

    with pytest.raises(ValueError):
        est.fit(train_data=dataiter,
                epochs=num_epochs)

    # Input NDArray
    with pytest.raises(ValueError):
        est.fit(train_data=[mx.nd.ones(shape=(10, 3))],
                epochs=num_epochs)


def test_validation():
    ''' test different validation data types'''
    net = _get_test_network()
    dataloader, dataiter = _get_test_data()
    num_epochs = 1
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    acc = mx.gluon.metric.Accuracy()
    val_loss = gluon.loss.L1Loss()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=acc,
                    trainer=trainer,
                    context=ctx,
                    val_loss=val_loss)
    # Input dataloader
    est.fit(train_data=dataloader,
            val_data=dataloader,
            epochs=num_epochs)

    # using validation handler
    train_metrics = est.train_metrics
    val_metrics = est.val_metrics
    validation_handler = ValidationHandler(val_data=dataloader, eval_fn=est.evaluate)

    with pytest.raises(ValueError):
        est.fit(train_data=dataiter,
                val_data=dataiter,
                epochs=num_epochs)
    # Input NDArray
    with pytest.raises(ValueError):
        est.fit(train_data=[mx.nd.ones(shape=(10, 3))],
                val_data=[mx.nd.ones(shape=(10, 3))],
                epochs=num_epochs)


def test_initializer():
    ''' test with no initializer, inconsistent initializer '''
    net = _get_test_network()
    train_data, _ = _get_test_data()
    num_epochs = 1
    ctx = mx.cpu()

    loss = gluon.loss.L2Loss()
    acc = mx.gluon.metric.Accuracy()
    # no initializer
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=acc,
                    context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs)

    # different initializer for net and estimator
    net = _get_test_network()
    net.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # catch reinit warning
    with warnings.catch_warnings(record=True) as w:
        est = Estimator(net=net,
                        loss=loss,
                        train_metrics=acc,
                        initializer=mx.init.MSRAPrelu(),
                        trainer=trainer,
                        context=ctx)
        assert 'Network already fully initialized' in str(w[-1].message)
    # net partially initialized, fine tuning use case
    net = gluon.model_zoo.vision.resnet18_v1(pretrained=True, ctx=ctx)
    net.output = gluon.nn.Dense(10) #last layer not initialized
    est = Estimator(net, loss=loss, train_metrics=acc, context=ctx)
    dataset =  gluon.data.ArrayDataset(mx.nd.zeros((10, 3, 224, 224)), mx.nd.zeros((10, 10)))
    train_data = gluon.data.DataLoader(dataset=dataset, batch_size=5)
    est.fit(train_data=train_data,
            epochs=num_epochs)


def test_trainer():
    ''' test with no trainer and invalid trainer '''
    net = _get_test_network()
    train_data, _ = _get_test_data()
    num_epochs = 1
    ctx = mx.cpu()

    loss = gluon.loss.L2Loss()
    acc = mx.gluon.metric.Accuracy()
    net.initialize(ctx=ctx)
    # input no trainer
    with warnings.catch_warnings(record=True) as w:
        est = Estimator(net=net,
                        loss=loss,
                        train_metrics=acc,
                        context=ctx)
        assert 'No trainer specified' in str(w[-1].message)
    est.fit(train_data=train_data,
            epochs=num_epochs)

    # input invalid trainer
    trainer = 'sgd'
    with pytest.raises(ValueError):
        est = Estimator(net=net,
                        loss=loss,
                        train_metrics=acc,
                        trainer=trainer,
                        context=ctx)


def test_metric():
    ''' test with no metric, list of metrics, invalid metric '''
    net = _get_test_network()
    train_data, _ = _get_test_data()
    num_epochs = 1
    ctx = mx.cpu()

    loss = gluon.loss.L2Loss()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # input no metric
    est = Estimator(net=net,
                    loss=loss,
                    trainer=trainer,
                    context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs)
    # input list of metrics
    metrics = [mx.gluon.metric.Accuracy(), mx.gluon.metric.Accuracy()]
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=metrics,
                    trainer=trainer,
                    context=ctx)
    est.fit(train_data=train_data,
            epochs=num_epochs)
    # input invalid metric
    with pytest.raises(ValueError):
        est = Estimator(net=net,
                        loss=loss,
                        train_metrics='acc',
                        trainer=trainer,
                        context=ctx)
    # test default metric
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    est = Estimator(net=net,
                    loss=loss,
                    trainer=trainer,
                    context=ctx)
    assert isinstance(est.train_metrics[0], mx.gluon.metric.Accuracy)


def test_loss():
    ''' test with invalid loss '''
    net = _get_test_network()
    ctx = mx.cpu()
    acc = mx.gluon.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    # input invalid loss
    with pytest.raises(ValueError):
        est = Estimator(net=net,
                        loss='mse',
                        train_metrics=acc,
                        trainer=trainer,
                        context=ctx)


def test_context():
    ''' test with no context, list of context, invalid context '''
    net = _get_test_network()
    loss = gluon.loss.L2Loss()
    metrics = mx.gluon.metric.Accuracy()
    # input no context
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=metrics)
    # input list of context
    gpus = mx.context.num_gpus()
    ctx = [mx.gpu(i) for i in range(gpus)] if gpus > 0 else [mx.cpu()]
    net = _get_test_network()
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=metrics,
                    context=ctx)
    # input invalid context
    with pytest.raises(ValueError):
        est = Estimator(net=net,
                        loss=loss,
                        train_metrics=metrics,
                        context='cpu')

    with pytest.raises(AssertionError):
        est = Estimator(net=net,
                        loss=loss,
                        train_metrics=metrics,
                        context=[mx.gpu(0), mx.gpu(100)])


def test_categorize_handlers():
    class CustomHandler1(TrainBegin):

        def train_begin(self):
            print("custom train begin")

    class CustomHandler2(EpochBegin, BatchBegin, TrainEnd):

        def epoch_begin(self):
            print("custom epoch begin")

        def batch_begin(self):
            print("custom batch begin")

        def train_end(self):
            print("custom train end")

    class CustomHandler3(EpochBegin, BatchBegin, BatchEnd, TrainEnd):

        def epoch_begin(self):
            print("custom epoch begin")

        def batch_begin(self):
            print("custom batch begin")

        def batch_end(self):
            print("custom batch end")

        def train_end(self):
            print("custom train end")

    net = nn.Sequential()
    net.add(nn.Dense(10))
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    est = Estimator(net, loss=loss)
    event_handlers = [CustomHandler1(), CustomHandler2(), CustomHandler3()]
    train_begin, epoch_begin, batch_begin, \
    batch_end, epoch_end, train_end = est._categorize_handlers(event_handlers)
    assert len(train_begin) == 1
    assert len(epoch_begin) == 2
    assert len(batch_begin) == 2
    assert len(batch_end) == 1
    assert len(train_end) == 2


def test_default_handlers():
    net = _get_test_network()
    train_data, _ = _get_test_data()

    num_epochs = 1
    ctx = mx.cpu()

    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})

    train_acc = mx.gluon.metric.RMSE()
    loss = gluon.loss.L2Loss()

    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=train_acc,
                    trainer=trainer,
                    context=ctx)
    # no handler(all default handlers), no warning
    with warnings.catch_warnings(record=True) as w:
        est.fit(train_data=train_data, epochs=num_epochs)

    # handler with prepared loss and metrics
    # use mix of default and user defined handlers
    train_metrics = est.train_metrics
    val_metrics = est.val_metrics
    logging = LoggingHandler(metrics=train_metrics)
    est.fit(train_data=train_data, epochs=num_epochs, event_handlers=[logging])

    # handler with all user defined metrics
    # use mix of default and user defined handlers
    metric = MetricHandler(metrics=[train_acc])
    logging = LoggingHandler(metrics=[train_acc])
    est.fit(train_data=train_data, epochs=num_epochs, event_handlers=[metric, logging])

    # handler with mixed metrics, some handler use metrics prepared by estimator
    # some handler use metrics user prepared
    logging = LoggingHandler(metrics=[mx.gluon.metric.RMSE("val acc")])
    with pytest.raises(ValueError):
        est.fit(train_data=train_data, epochs=num_epochs, event_handlers=[logging])

    # test handler order
    train_metrics = est.train_metrics
    val_metrics = est.val_metrics
    early_stopping = EarlyStoppingHandler(monitor=val_metrics[0])
    handlers = est._prepare_default_handlers(val_data=None, event_handlers=[early_stopping])
    assert len(handlers) == 5
    assert isinstance(handlers[0], GradientUpdateHandler)
    assert isinstance(handlers[1], MetricHandler)
    assert isinstance(handlers[4], LoggingHandler)

def test_val_net():
    ''' test estimator with different training and validation networks '''
    net = _get_test_network()
    val_net = _get_test_network(params=net.collect_params())
    dataloader, dataiter = _get_test_data()
    num_epochs = 1
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    val_loss = gluon.loss.L2Loss()
    acc = mx.gluon.metric.Accuracy()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=acc,
                    trainer=trainer,
                    context=ctx,
                    val_loss=val_loss,
                    val_net=val_net)

    est.fit(train_data=dataloader,
            val_data=dataloader,
            epochs=num_epochs)

    ''' test partial weight sharing of two resnets '''
    net = gluon.model_zoo.vision.resnet18_v1(pretrained=False, ctx=ctx)
    net.output = gluon.nn.Dense(10)
    val_net = gluon.model_zoo.vision.resnet18_v1(pretrained=False, ctx=ctx)
    val_net.output = net.output
    dataset = gluon.data.ArrayDataset(mx.nd.zeros((10, 3, 224, 224)), mx.nd.zeros((10, 10)))
    dataloader = gluon.data.DataLoader(dataset=dataset, batch_size=5)
    net.initialize(ctx=ctx)
    val_net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=acc,
                    trainer=trainer,
                    context=ctx,
                    val_loss=val_loss,
                    val_net=val_net)

    est.fit(train_data=dataloader,
            val_data=dataloader,
            epochs=num_epochs)

def test_val_handlers():
    net = _get_test_network()
    train_data, _ = _get_test_data()
    val_data, _ = _get_test_data()

    num_epochs = 1
    ctx = mx.cpu()
    net.initialize(ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})

    train_acc = mx.gluon.metric.RMSE()
    loss = gluon.loss.L2Loss()

    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=train_acc,
                    trainer=trainer,
                    context=ctx)

    with warnings.catch_warnings(record=True) as w:
        est.fit(train_data=train_data, epochs=num_epochs)
        est.evaluate(val_data=val_data)

    logging = LoggingHandler(log_interval=1, metrics=est.val_metrics)
    est.evaluate(val_data=val_data, event_handlers=[logging])

