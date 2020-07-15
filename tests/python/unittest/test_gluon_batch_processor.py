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

''' Unit tests for Gluon Batch Processor '''

import sys
import unittest
import warnings

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.contrib.estimator import *
from mxnet.gluon.contrib.estimator.event_handler import *
from mxnet.gluon.contrib.estimator.batch_processor import BatchProcessor
import pytest

mx.reset_np()

def _get_test_network():
    net = nn.Sequential()
    net.add(nn.Dense(4, activation='relu', flatten=False))
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

def test_batch_processor_fit():
    ''' test estimator with different train data types '''
    net = _get_test_network()
    dataloader, dataiter = _get_test_data()
    num_epochs = 1
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    acc = mx.gluon.metric.Accuracy()
    net.initialize(ctx=ctx)
    processor = BatchProcessor()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=acc,
                    trainer=trainer,
                    context=ctx,
                    batch_processor=processor)

    est.fit(train_data=dataloader,
            epochs=num_epochs)

    with pytest.raises(ValueError):
        est.fit(train_data=dataiter,
                epochs=num_epochs)

    # Input NDArray
    with pytest.raises(ValueError):
        est.fit(train_data=[mx.nd.ones(shape=(10, 3))],
                epochs=num_epochs)


def test_batch_processor_validation():
    ''' test different validation data types'''
    net = _get_test_network()
    dataloader, dataiter = _get_test_data()
    num_epochs = 1
    ctx = mx.cpu()
    loss = gluon.loss.L2Loss()
    acc = mx.gluon.metric.Accuracy()
    val_loss = gluon.loss.L1Loss()
    net.initialize(ctx=ctx)
    processor = BatchProcessor()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})
    est = Estimator(net=net,
                    loss=loss,
                    train_metrics=acc,
                    trainer=trainer,
                    context=ctx,
                    val_loss=val_loss,
                    batch_processor=processor)
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

