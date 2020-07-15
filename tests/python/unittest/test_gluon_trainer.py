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
import unittest
import os
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal
from common import setup_module, with_seed, assertRaises, xfail_when_nonstandard_decimal_separator
from copy import deepcopy
import pytest

mx.reset_np()

def dict_equ(a, b):
    assert set(a) == set(b)
    for k in a:
        assert (a[k].asnumpy() == b[k].asnumpy()).all()

@with_seed()
def test_multi_trainer():
    x = gluon.Parameter('x', shape=(10,), stype='row_sparse')
    x.initialize()
    # test set trainer
    trainer0 = gluon.Trainer([x], 'sgd')
    assert(x._trainer() is trainer0)
    # test unset trainer
    x._set_trainer(None)
    assert(x._trainer is None)
    x._set_trainer(trainer0)
    with pytest.raises(RuntimeError):
        # multiple trainers for a sparse Parameter is not allowed
        trainer1 = gluon.Trainer([x], 'sgd')

@with_seed()
def test_trainer_with_sparse_grad_on_single_context():
    x = gluon.Parameter('x', shape=(10,), grad_stype='row_sparse')
    x.initialize(ctx=[mx.cpu(0)], init='zeros')
    trainer = gluon.Trainer([x], 'sgd', {'learning_rate': 1.0, 'momentum': 0.5})
    with mx.autograd.record():
        for w in x.list_data():
            y = w + 1
            y.backward()
    trainer.step(1)

    assert trainer._update_on_kvstore is None
    assert trainer._kvstore is None  # No kvstore created for single-device training
    assert (x.data(mx.cpu(0)).asnumpy() == -1).all()

@with_seed()
def test_trainer_with_teststore():
    x = gluon.Parameter('x', shape=(10,))
    x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
    kv = mx.kv.create('teststore')
    trainer = gluon.Trainer([x], 'sgd', {'learning_rate': 1.0, 'momentum': 0.5}, kvstore=kv)
    with mx.autograd.record():
        for w in x.list_data():
            y = w + 1
            y.backward()
    trainer.step(1)

    assert trainer._update_on_kvstore == False
    assert (x.data(mx.cpu(1)).asnumpy() == -2).all()
    # Expect exceptions if update_on_kvstore is set to True,
    # because TestStore does not support that
    invalid_trainer = gluon.Trainer([x], 'sgd', kvstore=kv, update_on_kvstore=True)
    pytest.raises(ValueError, invalid_trainer._init_kvstore)

@with_seed()
def test_trainer():
    x = gluon.Parameter('x', shape=(10,))
    x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
    trainer = gluon.Trainer([x], 'sgd', {'learning_rate': 1.0, 'momentum': 0.5})
    with mx.autograd.record():
        for w in x.list_data():
            y = w + 1
            y.backward()
    trainer.step(1)

    assert trainer._optimizer.param_dict == trainer._optimizer.param_dict
    assert (x.data(mx.cpu(1)).asnumpy() == -2).all()

    x.lr_mult = 0.5
    with mx.autograd.record():
        for w in x.list_data():
            y = w + 1
            y.backward()
    trainer.step(1)
    assert (x.data(mx.cpu(1)).asnumpy() == -4).all()

    trainer.save_states('test_trainer.states')
    states = deepcopy(trainer._kvstore._updater.states) if trainer._update_on_kvstore \
             else deepcopy(trainer._updaters[0].states)
    trainer.load_states('test_trainer.states')
    if trainer._update_on_kvstore:
        dict_equ(trainer._kvstore._updater.states, states)
        assert trainer._optimizer == trainer._kvstore._updater.optimizer
        # invalid usage of update and allreduce_grads if update_on_kvstore
        pytest.raises(AssertionError, trainer.update, 1)
        pytest.raises(AssertionError, trainer.allreduce_grads)
    else:
        for updater in trainer._updaters:
            dict_equ(updater.states, states)
        assert trainer._optimizer == trainer._updaters[0].optimizer

    x = gluon.Parameter('x', shape=(10,))
    x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
    trainer2 = gluon.Trainer([x], 'sgd', {'learning_rate': 1.0, 'momentum': 0.5},
                             update_on_kvstore=False)
    with mx.autograd.record():
        for i, w in enumerate(x.list_data()):
            y = i*w
            y.backward()
    assert (x.grad(mx.cpu(0)).asnumpy() != x.grad(mx.cpu(1)).asnumpy()).all()
    trainer2.allreduce_grads()
    assert (x.grad(mx.cpu(0)).asnumpy() == x.grad(mx.cpu(1)).asnumpy()).all()
    trainer2.update(1)

    assert (x.data(mx.cpu(1)).asnumpy() == -1).all(), x.data(mx.cpu(1)).asnumpy()

@with_seed()
def test_trainer_save_load():
    previous_update_on_kvstore = os.getenv('MXNET_UPDATE_ON_KVSTORE', "1")
    os.putenv('MXNET_UPDATE_ON_KVSTORE', '1')

    x = gluon.Parameter('x', shape=(10,), lr_mult=1.0)
    x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
    trainer = gluon.Trainer([x], 'sgd', {'learning_rate': 0.1})
    with mx.autograd.record():
        for w in x.list_data():
            y = w + 1
            y.backward()
    trainer.step(1)
    assert trainer._kvstore._updater.optimizer._get_lr(0) == 0.1
    trainer.save_states('test_trainer_save_load.states')
    trainer.load_states('test_trainer_save_load.states')
    x.lr_mult = 2.0
    # check if parameter dict is correctly associated with optimizer after load_state
    assert trainer._kvstore._updater.optimizer._get_lr(0) == 0.2
    os.putenv('MXNET_UPDATE_ON_KVSTORE', previous_update_on_kvstore)

@with_seed()
def test_trainer_sparse_save_load():
    x = gluon.Parameter('x', shape=(10, 1), lr_mult=1.0,
                        stype='row_sparse', grad_stype='row_sparse')
    x.initialize(ctx=[mx.cpu(0)], init='zeros')
    trainer = gluon.Trainer([x], 'sgd', {'learning_rate': 0.1})
    all_rows = mx.nd.arange(0, 10, ctx=mx.cpu(0))
    with mx.autograd.record():
        for w in x.list_row_sparse_data(all_rows):
            y = w * 1
            y.backward()
    trainer.step(1)
    assert trainer._kvstore._updater.optimizer._get_lr(0) == 0.1
    trainer.save_states('test_trainer_sparse_save_load.states')
    trainer.load_states('test_trainer_sparse_save_load.states')
    x.lr_mult = 2.0
    # check if parameter dict is correctly associated with optimizer after load_state
    assert trainer._kvstore._updater.optimizer._get_lr(0) == 0.2

@with_seed()
def test_trainer_multi_layer_init():
    class Net(gluon.Block):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            # sparse param
            self.embed_weight = gluon.Parameter('embed_weight', stype='row_sparse',
                                                shape=(4,3), grad_stype='row_sparse')
            # dense param from a hybrid block
            self.dense0 = nn.Dense(2)

        def forward(self, x):
            embed_weight = self.embed_weight.row_sparse_data(x)
            embed = mx.nd.Embedding(data=x, weight=embed_weight,
                                    input_dim=4, output_dim=3, sparse_grad=True)
            return self.dense0(embed)

    def check_init(ctxes):
        net = Net()
        net.initialize(mx.init.One(), ctx=ctxes)
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 1})
        data = mx.nd.array([[0,2], [1,2]])
        xs = gluon.utils.split_and_load(data, ctxes)
        ys = []
        with mx.autograd.record():
            for x in xs:
                y = net(x)
                ys.append(y)
        for y in ys:
            y.backward()
        trainer.step(1)
        # all parameters should be initialized
        assert not trainer._params_to_init
        all_rows = mx.nd.arange(0, 4, ctx=mx.cpu(1))
        # check the updated weights
        weight = net.embed_weight.row_sparse_data(all_rows).asnumpy()
        assert (weight[0] == -1).all()
        assert (weight[1] == -1).all()
        assert (weight[2] == -3).all()
        assert (weight[3] == 1).all()

    check_init([mx.cpu(1), mx.cpu(2)])
    check_init([mx.cpu(1)])

@xfail_when_nonstandard_decimal_separator
@with_seed()
def test_trainer_reset_kv():
    def check_trainer_reset_kv(kv):
        x = gluon.Parameter('x', shape=(10,), lr_mult=1.0)
        params = {'x': x}
        x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
        trainer = gluon.Trainer(params, 'sgd', {'learning_rate': 0.1}, kvstore=kv)
        mx.nd.save('test_trainer_reset_kv.params', {k: v._reduce() for k, v in params.items()})
        with mx.autograd.record():
            for w in x.list_data():
                y = w + 1
                y.backward()
        trainer.step(1)
        assert trainer._kvstore.type == kv
        # load would reset kvstore
        mx.nd.waitall()
        params = mx.nd.load('test_trainer_reset_kv.params')
        x._load_init(params['x'], None)
        if trainer._update_on_kvstore:
            # drop kvstore state if new parameters are loaded
            assert trainer._kvstore is None
            assert trainer._kv_initialized is False
        with mx.autograd.record():
            for w in x.list_data():
                y = w + 1
                y.backward()
        trainer.step(1)
        # the updated parameter should be based on the loaded checkpoint
        assert (x.data(mx.cpu()) == -0.2).asnumpy().all()

    kvs = ['local', 'device']
    for kv in kvs:
        check_trainer_reset_kv(kv)

@xfail_when_nonstandard_decimal_separator
@with_seed()
def test_trainer_sparse_kv():
    def check_trainer_sparse_kv(kv, stype, grad_stype, update_on_kv, expected):
        x = mx.gluon.Parameter('x', shape=(10,1), lr_mult=1.0, stype=stype, grad_stype=grad_stype)
        x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
        trainer = gluon.Trainer([x], 'sgd', {'learning_rate': 0.1},
                                kvstore=kv, update_on_kvstore=update_on_kv)
        all_rows = mx.nd.arange(0, 10, ctx=mx.cpu(0))
        try:
            ws = x.list_data() if stype == 'default' else x.list_row_sparse_data(all_rows)
            with mx.autograd.record():
                for w in ws:
                    y = w + 1
                    y.backward()
            trainer.step(1)
            assert trainer._kvstore.type == kv
            assert trainer._kv_initialized
            assert trainer._update_on_kvstore is expected
            # the updated parameter should be based on the loaded checkpoint
            mx.nd.waitall()
            updated_w = x.data(mx.cpu(0)) if stype == 'default' else x.row_sparse_data(all_rows)
            assert (updated_w == -0.2).asnumpy().all(), updated_w
        except Exception as err:
            assert isinstance(err, expected)

    kvs = ['local', 'device']
    global_update_on_kvstore = bool(int(os.getenv('MXNET_UPDATE_ON_KVSTORE', "1")))
    for kv in kvs:
        check_trainer_sparse_kv(kv, 'default', 'default', True, True)
        check_trainer_sparse_kv(kv, 'default', 'default', False, False)
        check_trainer_sparse_kv(kv, 'default', 'default', None, global_update_on_kvstore)
        check_trainer_sparse_kv(kv, 'default', 'row_sparse', None, False)
        check_trainer_sparse_kv(kv, 'default', 'row_sparse', True, True)
        check_trainer_sparse_kv(kv, 'default', 'row_sparse', False, False)
        check_trainer_sparse_kv(kv, 'row_sparse', 'row_sparse', None, True)
        check_trainer_sparse_kv(kv, 'row_sparse', 'row_sparse', False, ValueError)

@with_seed()
def test_trainer_lr_sched():
    x = gluon.Parameter('x', shape=(10,))
    x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
    freq = 2
    factor = 0.1
    lr = 1
    lr_sched = mx.lr_scheduler.FactorScheduler(freq, factor=factor, base_lr=lr)
    trainer = gluon.Trainer([x], 'sgd', {'learning_rate': lr, 'lr_scheduler': lr_sched})
    for i in range(10):
        with mx.autograd.record():
            for w in x.list_data():
                y = w + 1
                y.backward()
        trainer.step(1)
        if i % freq == 0:
            assert trainer.learning_rate == lr, (lr, trainer.learning_rate, i)
            lr *= factor
    mx.nd.waitall()

    # Update on kvstore = False
    x = gluon.Parameter('x', shape=(10,))
    x.initialize(ctx=[mx.cpu(0), mx.cpu(1)], init='zeros')
    freq = 2
    factor = 0.1
    lr = 1
    lr_sched = mx.lr_scheduler.FactorScheduler(freq, factor=factor, base_lr=lr)
    trainer = gluon.Trainer([x], 'sgd', {'learning_rate': lr, 'lr_scheduler': lr_sched},
                            update_on_kvstore=False)
    for i in range(10):
        with mx.autograd.record():
            for w in x.list_data():
                y = w + 1
                y.backward()
        trainer.step(1)
        if i % freq == 0:
            assert trainer.learning_rate == lr, (lr, trainer.learning_rate, i)
            lr *= factor
    mx.nd.waitall()

@with_seed()
def test_gluon_trainer_param_order():
    net = mx.gluon.nn.Sequential()
    # layers may be added in a random order for all workers
    layers = {'ones_': 1, 'zeros_': 0}
    for name, init in layers.items():
        net.add(mx.gluon.nn.Dense(10, in_units=10, weight_initializer=mx.init.Constant(init),
                                  use_bias=False))
    net.initialize()
    params = net.collect_params()
    trainer = gluon.Trainer(params, 'sgd')
    for name, init in layers.items():
        expected_idx = 0 if name == 'ones_' else 1
        expected_name = '{}.weight'.format(expected_idx)
        assert trainer._params[expected_idx].name == params[expected_name].name
