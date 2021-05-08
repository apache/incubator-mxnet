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

"""
Description : Trainer module for SNAIL
"""

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
from tqdm import trange, tqdm
from mxboard import SummaryWriter
from models import SNAIL
from utils import batch_for_few_shot
from data_loader import loader
# pylint: disable=invalid-name, too-many-locals, too-many-instance-attributes
def setting_ctx(GPU_COUNT):
    """
    Description : set gpu count
    """
    if GPU_COUNT > 0:
        ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
    else:
        ctx = [mx.cpu()]
    return ctx

class Train():
    """
    Description : Train module for SNAIL
    """
    def __init__(self, config):
        ##setting hyper-parameters
        self.config = config
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.N = config.N
        self.K = config.K
        self.input_dims = config.input_dims
        self.GPU_COUNT = config.GPU_COUNT
        self.ctx = setting_ctx(self.GPU_COUNT)
        self.build_model()
        self.writer = SummaryWriter(logdir=self.config.logdir, filename_suffix="_SNAIL")

    def build_model(self):
        """
        Description : build network
        """
        self.net = SNAIL(N=self.N, K=self.K, input_dims=self.input_dims, ctx=self.ctx)
        self.net.collect_params().initialize(ctx=self.ctx)
        self.loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        self.trainer = gluon.Trainer(self.net.collect_params(),\
                                     optimizer='Adam', optimizer_params={'learning_rate':0.0001})

    def save_model(self, epoch, tr_acc, te_acc):
        """
        Description : save network
        """
        filename = 'models/best_perf_epoch_'+str(epoch)+"_tr_acc_"+\
        str(tr_acc)+"_te_acc_"+str(te_acc)
        self.net.save_params(filename)

    def train(self):
        """
        Description : train network
        """
        tr_dataloader, te_dataloader = loader(self.config, self.ctx)
        tr_acc_per_epoch = list()
        te_acc_per_epoch = list()
        train_acc = mx.metric.Accuracy()
        test_acc = mx.metric.Accuracy()
        global_va_acc = 0.0
        for epoch in trange(self.epochs):
            tr_acc = list()
            te_acc = list()
            tr_iter = iter(tr_dataloader)
            te_iter = iter(te_dataloader)
            for batch in tqdm(tr_iter):
                x, y = batch
                x, y, last_targets = batch_for_few_shot(self.N, self.K, self.batch_size, x, y)
                with autograd.record():
                    x_split = gluon.utils.split_and_load(x, self.ctx)
                    y_split = gluon.utils.split_and_load(y, self.ctx)
                    last_targets_split = gluon.utils.split_and_load(last_targets, self.ctx)
                    last_model = [self.net(X, Y)[:, -1, :] for X, Y in zip(x_split, y_split)]
                    loss_val = [self.loss_fn(X, Y) for X, Y in zip(last_model, last_targets_split)]
                    for l in loss_val:
                        l.backward()
                    for pred, target in zip(last_model, last_targets_split):
                        train_acc.update(preds=nd.argmax(pred, 1), labels=target)
                        tr_acc.append(train_acc.get()[1])

                self.trainer.step(self.batch_size, ignore_stale_grad=True)


            for batch in tqdm(te_iter):
                x, y = batch
                x, y, last_targets = batch_for_few_shot(self.N, self.K,\
                                                        int(self.batch_size / len(self.ctx)), x, y)
                x = x.copyto(self.ctx[0])
                y = y.copyto(self.ctx[0])
                last_targets = last_targets.copyto(self.ctx[0])
                model_output = self.net(x, y)
                last_model = model_output[:, -1, :]
                test_acc.update(preds=nd.argmax(last_model, 1), labels=last_targets)
                te_acc.append(test_acc.get()[1])
            current_va_acc = np.mean(te_acc)
            if global_va_acc < current_va_acc:
                self.save_model(epoch, round(np.mean(tr_acc), 2), round(np.mean(te_acc), 2))
                global_va_acc = current_va_acc
            print("epoch {e}  train_acc:{ta} test_acc:{tea} ".format(e=epoch,\
                                                                     ta=np.mean(tr_acc),\
                                                                     tea=np.mean(te_acc)))
            self.writer.add_scalar(tag="train_accuracy", value=np.mean(tr_acc), global_step=epoch)
            self.writer.add_scalar(tag="test_accuracy", value=np.mean(te_acc), global_step=epoch)
            tr_acc_per_epoch.append(np.mean(tr_acc))
            te_acc_per_epoch.append(np.mean(te_acc))
