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

# coding: utf-8
# pylint: disable=wildcard-import, unused-argument, too-many-ancestors
"""Gluon Batch Processor for Estimators"""

from ...utils import split_and_load
from .... import autograd

__all__ = ['BatchProcessor']

class BatchProcessor(object):
    """BatchProcessor Class for plug and play fit_batch & evaluate_batch

    During training or validation, data are divided into minibatches for processing. This
    class aims at providing hooks of training or validating on a minibatch of data. Users
    may provide customized fit_batch() and evaluate_batch() methods by inheriting from
    this class and overriding class methods.

    :py:class:`BatchProcessor` can be used to replace fit_batch() and evaluate_batch()
    in the base estimator class
    """

    def __init__(self):
        pass

    def _get_data_and_label(self, batch, ctx, batch_axis=0):
        if not isinstance(batch, list):
            raise ValueError("Default batch processor only supports batch of [data, label]"
                             "Alternatively, you can write customized batch processor for"
                             "your estimator.")
        data = batch[0]
        label = batch[1]
        data = split_and_load(data, ctx_list=ctx, batch_axis=batch_axis)
        label = split_and_load(label, ctx_list=ctx, batch_axis=batch_axis)
        return data, label

    def evaluate_batch(self, estimator,
                       val_batch,
                       batch_axis=0):
        """Evaluate the estimator model on a batch of validation data.

        Parameters
        ----------
        estimator : Estimator
            Reference to the estimator
        val_batch : tuple
            Data and label of a batch from the validation data loader.
        batch_axis : int, default 0
            Batch axis to split the validation data into devices.
        """
        data, label = self._get_data_and_label(val_batch, estimator.context, batch_axis)
        pred = [estimator.val_net(x) for x in data]
        loss = [estimator.val_loss(y_hat, y) for y_hat, y in zip(pred, label)]

        return data, label, pred, loss

    def fit_batch(self, estimator,
                  train_batch,
                  batch_axis=0):
        """Trains the estimator model on a batch of training data.

        Parameters
        ----------
        estimator : Estimator
            Reference to the estimator
        train_batch : tuple
            Data and label of a batch from the training data loader.
        batch_axis : int, default 0
            Batch axis to split the training data into devices.

        Returns
        -------
        data: List of NDArray
            Sharded data from the batch. Data is sharded with
            `gluon.split_and_load`.
        label: List of NDArray
            Sharded label from the batch. Labels are sharded with
            `gluon.split_and_load`.
        pred: List of NDArray
            Prediction on each of the sharded inputs.
        loss: List of NDArray
            Loss on each of the sharded inputs.
        """
        data, label = self._get_data_and_label(train_batch, estimator.context, batch_axis)

        with autograd.record():
            pred = [estimator.net(x) for x in data]
            loss = [estimator.loss(y_hat, y) for y_hat, y in zip(pred, label)]

        for l in loss:
            l.backward()

        return data, label, pred, loss
