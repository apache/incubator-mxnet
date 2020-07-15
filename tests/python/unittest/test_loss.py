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

import mxnet as mx
import numpy as np
from mxnet import gluon, autograd
from mxnet.test_utils import assert_almost_equal, default_context
from common import setup_module, with_seed, teardown_module, xfail_when_nonstandard_decimal_separator
import unittest

mx.reset_np()

@xfail_when_nonstandard_decimal_separator
@with_seed()
def test_loss_ndarray():
    output = mx.nd.array([1, 2, 3, 4])
    label = mx.nd.array([1, 3, 5, 7])
    weighting = mx.nd.array([0.5, 1, 0.5, 1])

    loss = gluon.loss.L1Loss()
    assert mx.nd.sum(loss(output, label)).asscalar() == 6.
    loss = gluon.loss.L1Loss(weight=0.5)
    assert mx.nd.sum(loss(output, label)).asscalar() == 3.
    loss = gluon.loss.L1Loss()
    assert mx.nd.sum(loss(output, label, weighting)).asscalar() == 5.

    loss = gluon.loss.L2Loss()
    assert mx.nd.sum(loss(output, label)).asscalar() == 7.
    loss = gluon.loss.L2Loss(weight=0.25)
    assert mx.nd.sum(loss(output, label)).asscalar() == 1.75
    loss = gluon.loss.L2Loss()
    assert mx.nd.sum(loss(output, label, weighting)).asscalar() == 6

    output = mx.nd.array([[0, 2], [1, 4]])
    label = mx.nd.array([0, 1])
    weighting = mx.nd.array([[0.5], [1.0]])

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    L = loss(output, label).asnumpy()
    assert_almost_equal(L, np.array([ 2.12692809,  0.04858733]), rtol=1e-3, atol=1e-4)

    L = loss(output, label, weighting).asnumpy()
    assert_almost_equal(L, np.array([ 1.06346405,  0.04858733]), rtol=1e-3, atol=1e-4)


def get_net(num_hidden, flatten=True):
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=128, flatten=flatten)
    act1 = mx.symbol.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(act1, name = 'fc2', num_hidden = 64, flatten=flatten)
    act2 = mx.symbol.Activation(fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=num_hidden, flatten=flatten)
    return fc3


@with_seed()
def test_bce_equal_ce2():
    N = 100
    loss1 = gluon.loss.SigmoidBCELoss(from_sigmoid=True)
    loss2 = gluon.loss.SoftmaxCELoss(from_logits=True)
    out1 = mx.random.uniform(0.1, 0.9, shape=(N, 1))
    out2 = mx.nd.log(mx.nd.concat(1-out1, out1, dim=1) + 1e-8)
    label = mx.nd.round(mx.random.uniform(0, 1, shape=(N, 1)))
    assert_almost_equal(loss1(out1, label).asnumpy(), loss2(out2, label).asnumpy())

def test_logistic_loss_equal_bce():
    N = 100
    loss_binary = gluon.loss.LogisticLoss(label_format='binary')
    loss_signed = gluon.loss.LogisticLoss(label_format='signed')
    loss_bce = gluon.loss.SigmoidBCELoss(from_sigmoid=False)
    data = mx.random.uniform(-10, 10, shape=(N, 1))
    label = mx.nd.round(mx.random.uniform(0, 1, shape=(N, 1)))
    assert_almost_equal(loss_binary(data, label), loss_bce(data, label), atol=1e-6)
    assert_almost_equal(loss_signed(data, 2 * label - 1), loss_bce(data, label), atol=1e-6)


@with_seed()
def test_ctc_loss():
    loss = gluon.loss.CTCLoss()
    l = loss(mx.nd.ones((2,20,4)), mx.nd.array([[1,0,-1,-1],[2,1,1,-1]]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss(layout='TNC')
    l = loss(mx.nd.ones((20,2,4)), mx.nd.array([[1,0,-1,-1],[2,1,1,-1]]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss(layout='TNC', label_layout='TN')
    l = loss(mx.nd.ones((20,2,4)), mx.nd.array([[1,0,-1,-1],[2,1,1,-1]]).T)
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    l = loss(mx.nd.ones((2,20,4)), mx.nd.array([[2,1,2,2],[3,2,2,2]]), None, mx.nd.array([2,3]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    l = loss(mx.nd.ones((2,25,4)), mx.nd.array([[2,1,-1,-1],[3,2,2,-1]]), mx.nd.array([20,20]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))

    loss = gluon.loss.CTCLoss()
    l = loss(mx.nd.ones((2,25,4)), mx.nd.array([[2,1,3,3],[3,2,2,3]]), mx.nd.array([20,20]), mx.nd.array([2,3]))
    assert_almost_equal(l, np.array([18.82820702, 16.50581741]))


@xfail_when_nonstandard_decimal_separator
@with_seed()
def test_sdml_loss():

    N = 5 # number of samples
    DIM = 10 # Dimensionality
    EPOCHS = 20

    # Generate randomized data and 'positive' samples
    data = mx.random.uniform(-1, 1, shape=(N, DIM))
    pos = data + mx.random.uniform(-0.1, 0.1, shape=(N, DIM)) # correlated paired data
    data_iter = mx.io.NDArrayIter({'data' : data, 'pos' : pos}, batch_size=N)

    # Init model and trainer
    sdml_loss = gluon.loss.SDMLLoss()
    model = gluon.nn.Dense(DIM, activation='tanh') # Simple NN encoder
    model.initialize(mx.init.Xavier(), ctx=mx.current_context())
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate' : 0.1})

    for i in range(EPOCHS): # Training loop
        data_iter.reset()
        for iter_batch in data_iter:
            batch = [datum.as_in_context(mx.current_context()) for datum in iter_batch.data]
            with autograd.record():
                data, pos = batch
                z_data, z_pos = model(data), model(pos)
                loss = sdml_loss(z_data, z_pos)
                loss.backward()
            trainer.step(1)

    # After training euclidean distance between aligned pairs should be lower than all non-aligned pairs
    avg_loss = loss.sum()/len(loss)
    assert(avg_loss < 0.05)

@with_seed()
def test_cosine_loss():
    #Generating samples
    input1 = mx.nd.random.randn(3, 2)
    input2 = mx.nd.random.randn(3, 2)
    label = mx.nd.sign(mx.nd.random.randn(input1.shape[0]))
    #Calculating loss from cosine embedding loss function in Gluon
    Loss = gluon.loss.CosineEmbeddingLoss()
    loss = Loss(input1, input2, label)

    # Calculating the loss Numpy way
    numerator = mx.nd.sum(input1 * input2, keepdims=True, axis=1)
    denominator = mx.nd.sqrt(mx.nd.sum(input1**2, axis=1, keepdims=True)) \
    * mx.nd.sqrt(mx.nd.sum(input2**2, axis=1, keepdims=True))
    numpy_loss = mx.nd.where(label == 1, 1-numerator/denominator, \
    mx.nd.broadcast_maximum(mx.nd.array([0]), numerator/denominator, axis=1))
    assert_almost_equal(loss.asnumpy(), numpy_loss.asnumpy(), rtol=1e-3, atol=1e-5)

@xfail_when_nonstandard_decimal_separator
def test_poisson_nllloss():
    shape=(3, 4)
    not_axis0 = tuple(range(1, len(shape)))
    pred = mx.nd.random.normal(shape=shape)
    min_pred = mx.nd.min(pred)
    #This is necessary to ensure only positive random values are generated for prediction,
    # to avoid ivalid log calculation
    pred[:] = pred + mx.nd.abs(min_pred)
    target = mx.nd.random.normal(shape=shape)
    min_target = mx.nd.min(target)
    #This is necessary to ensure only positive random values are generated for prediction,
    # to avoid ivalid log calculation
    target[:] += mx.nd.abs(min_target)

    Loss = gluon.loss.PoissonNLLLoss(from_logits=True)
    Loss_no_logits = gluon.loss.PoissonNLLLoss(from_logits=False)
    #Calculating by brute formula for default value of from_logits = True

    # 1) Testing for flag logits = True
    brute_loss = np.mean(np.exp(pred.asnumpy()) - target.asnumpy() * pred.asnumpy())
    loss_withlogits = Loss(pred, target)
    assert_almost_equal(brute_loss, loss_withlogits.asscalar())

    #2) Testing for flag logits = False
    loss_no_logits = Loss_no_logits(pred, target)
    np_loss_no_logits = np.mean(pred.asnumpy() - target.asnumpy() * np.log(pred.asnumpy() + 1e-08))
    if np.isnan(loss_no_logits.asscalar()):
        assert_almost_equal(np.isnan(np_loss_no_logits), np.isnan(loss_no_logits.asscalar()))
    else:
        assert_almost_equal(np_loss_no_logits, loss_no_logits.asscalar())

    #3) Testing for Sterling approximation
    shape=(2, 3)
    np_pred = np.random.uniform(1, 5, shape)
    np_target = np.random.uniform(1, 5, shape)
    np_compute_full = np.mean((np_pred - np_target * np.log(np_pred + 1e-08)) + ((np_target * np.log(np_target)-\
     np_target + 0.5 * np.log(2 * np_target * np.pi))*(np_target > 1)))
    Loss_compute_full = gluon.loss.PoissonNLLLoss(from_logits=False, compute_full=True)
    loss_compute_full = Loss_compute_full(mx.nd.array(np_pred), mx.nd.array(np_target))
    assert_almost_equal(np_compute_full, loss_compute_full.asscalar())

