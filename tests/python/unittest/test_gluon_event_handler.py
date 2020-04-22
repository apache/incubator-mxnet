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

import os
import logging
import sys
import re

import mxnet as mx
from common import TemporaryDirectory
from mxnet import nd
from mxnet.gluon import nn, loss
from mxnet.gluon.contrib.estimator import estimator, event_handler
from mxnet.gluon.contrib.estimator.event_handler import LoggingHandler
from mxnet.gluon.contrib.estimator.event_handler import ValidationHandler
from mxnet.gluon.contrib.estimator import EpochEnd
from mxnet.gluon.data.dataset import Dataset
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

class TestAxisArrayDataset(Dataset):
    def __init__(self, * args):
        self._length = len(args[1])
        self._data = []
        for _, data in enumerate(args):
            self._data.append(data)

    def __getitem__(self, idx):
        if len(self._data) == 1:
            return self._data[idx][0]
        else:
            return tuple(data[:, idx] for data in self._data)

    def __len__(self):
        return self._length

class TestHandler(EpochEndMixin):
    def __init__(self):
        pass

    def epoch_end(self, estimator, *args, **kwargs):
        estimator.run_test_handler = True

def _get_test_network(net=nn.Sequential()):
    net.add(nn.Dense(128, activation='relu', flatten=False),
            nn.Dense(64, activation='relu'),
            nn.Dense(10, activation='relu'))
    return net


def _get_test_data(in_size=32):
    data = nd.ones((in_size, 100))
    label = nd.zeros((in_size, 1))
    data_arr = mx.gluon.data.dataset.ArrayDataset(data, label)
    return mx.gluon.data.DataLoader(data_arr, batch_size=8)

def _get_batch_axis_test_data(in_size=32):
    data = nd.ones((100, in_size))
    label = nd.zeros((1, in_size))
    data_arr = TestAxisArrayDataset(data, label)
    return mx.gluon.data.DataLoader(data_arr, batch_size=8)

def test_checkpoint_handler():
    with TemporaryDirectory() as tmpdir:
        model_prefix = 'test_epoch'
        file_path = os.path.join(tmpdir, model_prefix)
        test_data = _get_test_data()

        net = _get_test_network()
        ce_loss = loss.SoftmaxCrossEntropyLoss()
        acc = mx.metric.Accuracy()
        est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)
        checkpoint_handler = event_handler.CheckpointHandler(model_dir=tmpdir,
                                                             model_prefix=model_prefix,
                                                             monitor=acc,
                                                             save_best=True,
                                                             epoch_period=1)
        est.fit(test_data, event_handlers=[checkpoint_handler], epochs=1)
        assert checkpoint_handler.current_epoch == 1
        assert checkpoint_handler.current_batch == 4
        assert os.path.isfile(file_path + '-best.params')
        assert os.path.isfile(file_path + '-best.states')
        assert os.path.isfile(file_path + '-epoch0batch4.params')
        assert os.path.isfile(file_path + '-epoch0batch4.states')

        model_prefix = 'test_batch'
        file_path = os.path.join(tmpdir, model_prefix)
        net = _get_test_network(nn.HybridSequential())
        net.hybridize()
        est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)
        checkpoint_handler = event_handler.CheckpointHandler(model_dir=tmpdir,
                                                             model_prefix=model_prefix,
                                                             epoch_period=None,
                                                             batch_period=2,
                                                             max_checkpoints=2)
        est.fit(test_data, event_handlers=[checkpoint_handler], batches=10)
        assert checkpoint_handler.current_batch == 10
        assert checkpoint_handler.current_epoch == 3
        assert not os.path.isfile(file_path + 'best.params')
        assert not os.path.isfile(file_path + 'best.states')
        assert not os.path.isfile(file_path + '-epoch0batch0.params')
        assert not os.path.isfile(file_path + '-epoch0batch0.states')
        assert os.path.isfile(file_path + '-symbol.json')
        assert os.path.isfile(file_path + '-epoch1batch7.params')
        assert os.path.isfile(file_path + '-epoch1batch7.states')
        assert os.path.isfile(file_path + '-epoch2batch9.params')
        assert os.path.isfile(file_path + '-epoch2batch9.states')

def test_resume_checkpoint():
    with TemporaryDirectory() as tmpdir:
        model_prefix = 'test_net'
        file_path = os.path.join(tmpdir, model_prefix)
        test_data = _get_test_data()

        net = _get_test_network()
        ce_loss = loss.SoftmaxCrossEntropyLoss()
        acc = mx.metric.Accuracy()
        est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)
        checkpoint_handler = event_handler.CheckpointHandler(model_dir=tmpdir,
                                                             model_prefix=model_prefix,
                                                             monitor=acc,
                                                             max_checkpoints=1)
        est.fit(test_data, event_handlers=[checkpoint_handler], epochs=2)
        assert os.path.isfile(file_path + '-epoch1batch8.params')
        assert os.path.isfile(file_path + '-epoch1batch8.states')
        checkpoint_handler = event_handler.CheckpointHandler(model_dir=tmpdir,
                                                             model_prefix=model_prefix,
                                                             monitor=acc,
                                                             max_checkpoints=1,
                                                             resume_from_checkpoint=True)
        est.fit(test_data, event_handlers=[checkpoint_handler], epochs=5)
        # should only continue to train 3 epochs and last checkpoint file is epoch4
        assert est.max_epoch == 3
        assert os.path.isfile(file_path + '-epoch4batch20.states')


def test_early_stopping():
    test_data = _get_test_data()

    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)
    early_stopping = event_handler.EarlyStoppingHandler(monitor=acc,
                                                        patience=0,
                                                        mode='min')
    est.fit(test_data, event_handlers=[early_stopping], epochs=5)
    assert early_stopping.current_epoch == 2
    assert early_stopping.stopped_epoch == 1

    early_stopping = event_handler.EarlyStoppingHandler(monitor=acc,
                                                        patience=2,
                                                        mode='auto')
    est.fit(test_data, event_handlers=[early_stopping], epochs=1)
    assert early_stopping.current_epoch == 1


def test_logging():
    with TemporaryDirectory() as tmpdir:
        test_data = _get_test_data()
        file_name = 'test_log'
        output_dir = os.path.join(tmpdir, file_name)

        net = _get_test_network()
        ce_loss = loss.SoftmaxCrossEntropyLoss()
        acc = mx.metric.Accuracy()
        est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)

        est.logger.addHandler(logging.FileHandler(output_dir))

        train_metrics = est.train_metrics
        val_metrics = est.val_metrics
        logging_handler = event_handler.LoggingHandler(metrics=train_metrics)
        est.fit(test_data, event_handlers=[logging_handler], epochs=3)
        assert logging_handler.batch_index == 0
        assert logging_handler.current_epoch == 3
        assert os.path.isfile(output_dir)
        del est  # Clean up estimator and logger before deleting tmpdir


def test_custom_handler():
    class CustomStopHandler(event_handler.TrainBeginMixin,
                            event_handler.BatchEndMixin,
                            event_handler.EpochEndMixin):
        def __init__(self, batch_stop=None, epoch_stop=None):
            self.batch_stop = batch_stop
            self.epoch_stop = epoch_stop
            self.num_batch = 0
            self.num_epoch = 0
            self.stop_training = False

        def train_begin(self, estimator, *args, **kwargs):
            self.num_batch = 0
            self.num_epoch = 0

        def batch_end(self, estimator, *args, **kwargs):
            self.num_batch += 1
            if self.num_batch == self.batch_stop:
                self.stop_training = True
            return self.stop_training

        def epoch_end(self, estimator, *args, **kwargs):
            self.num_epoch += 1
            if self.num_epoch == self.epoch_stop:
                self.stop_training = True
            return self.stop_training

    # total data size is 32, batch size is 8
    # 4 batch per epoch
    test_data = _get_test_data()
    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)
    custom_handler = CustomStopHandler(3, 2)
    est.fit(test_data, event_handlers=[custom_handler], epochs=3)
    assert custom_handler.num_batch == 3
    assert custom_handler.num_epoch == 1
    custom_handler = CustomStopHandler(100, 5)
    est.fit(test_data, event_handlers=[custom_handler], epochs=10)
    assert custom_handler.num_batch == 5 * 4
    assert custom_handler.num_epoch == 5

def test_logging_interval():
    ''' test different options for logging handler '''
    ''' test case #1: log interval is 1 '''
    batch_size = 8
    data_size = 100
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    log_interval = 1
    net = _get_test_network()
    dataloader = _get_test_data(in_size=data_size)
    num_epochs = 1
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    logging = LoggingHandler(metrics=[acc], log_interval=log_interval)
    est = estimator.Estimator(net=net,
                              loss=ce_loss,
                              train_metrics=acc)

    est.fit(train_data=dataloader,
            epochs=num_epochs,
            event_handlers=[logging])

    sys.stdout = old_stdout
    log_info_list = mystdout.getvalue().splitlines()
    info_len = 0
    for info in log_info_list:
        match = re.match(
            '(\[Epoch \d+\]\[Batch \d+\]\[Samples \d+\] time\/interval: \d+.\d+s' +
            ' training accuracy: \d+.\d+)', info)
        if match:
            info_len += 1

    assert(info_len == int(data_size/batch_size/log_interval) + 1)
    ''' test case #2: log interval is 5 '''
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    acc = mx.metric.Accuracy()
    log_interval = 5
    logging = LoggingHandler(metrics=[acc], log_interval=log_interval)
    est = estimator.Estimator(net=net,
                              loss=ce_loss,
                              train_metrics=acc)
    est.fit(train_data=dataloader,
            epochs=num_epochs,
            event_handlers=[logging])
    sys.stdout = old_stdout
    log_info_list = mystdout.getvalue().splitlines()
    info_len = 0
    for info in log_info_list:
        match = re.match(
            '(\[Epoch \d+\]\[Batch \d+\]\[Samples \d+\] time\/interval: \d+.\d+s' +
            ' training accuracy: \d+.\d+)', info)
        if match:
            info_len += 1

    assert(info_len == int(data_size/batch_size/log_interval) + 1)

def test_validation_handler_batch_axis():
    # test case #1: test batch_axis=0
    test_data = _get_test_data()
    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)
    est.fit(test_data, epochs=3)

    #test case #2: test batch_axis=1
    test_data = _get_batch_axis_test_data()
    val_data = _get_batch_axis_test_data(in_size=30)
    est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)
    est.fit(test_data, val_data=val_data,
            epochs=3, batch_axis=1)

def test_validation_handler():
    test_data = _get_test_data()

    net = _get_test_network()
    ce_loss = loss.SoftmaxCrossEntropyLoss()
    acc = mx.metric.Accuracy()
    est = estimator.Estimator(net, loss=ce_loss, train_metrics=acc)
    val_handler = ValidationHandler(val_data=test_data,
                                    eval_fn=est.evaluate,
                                    event_handlers=TestHandler())

    est.fit(train_data=test_data, val_data=test_data,
            event_handlers=[val_handler], epochs=2)
    assert est.run_test_handler == True
