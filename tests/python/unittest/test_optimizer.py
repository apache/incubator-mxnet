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

import itertools
import numpy as np
import itertools
import mxnet as mx
import mxnet.lr_scheduler as lr_scheduler
from mxnet import gluon
import unittest
import pytest
import math
from mxnet.test_utils import *
from common import retry, xfail_when_nonstandard_decimal_separator

def test_learning_rate():
    o1 = mx.optimizer.Optimizer(learning_rate=0.01)
    o1.set_learning_rate(0.2)
    assert o1.learning_rate == 0.2

    lr_s = lr_scheduler.FactorScheduler(step=1)
    o2 = mx.optimizer.Optimizer(lr_scheduler=lr_s, learning_rate=0.3)
    assert o2.learning_rate == 0.3
    o2.lr_scheduler.base_lr = 0.4
    assert o2.learning_rate == 0.4

    lr_s = lr_scheduler.FactorScheduler(step=1, base_lr=1024)
    o3 = mx.optimizer.Optimizer(lr_scheduler=lr_s)
    assert o3.learning_rate == 1024


def test_learning_rate_expect_user_warning():
    lr_s = lr_scheduler.FactorScheduler(step=1)
    o = mx.optimizer.Optimizer(lr_scheduler=lr_s, learning_rate=0.3)

    with pytest.raises(UserWarning):
        o.set_learning_rate(0.5)


@xfail_when_nonstandard_decimal_separator
def test_sgd():
    opt1 = mx.optimizer.SGD
    opt2 = mx.optimizer.SGD
    shapes = [(3, 4, 5), (10, 4), (7,)]
    mom_options = [{}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]

    for dtype in [np.float16, np.float32]:
        for params in itertools.product(mom_options, cg_options, rg_options,
                                        wd_options, mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            if dtype == np.float16:
                compare_optimizer(opt1(use_fused_step=False, **kwarg),
                                  opt2(use_fused_step=True, **kwarg),
                                  shapes, dtype, rtol=1e-3, atol=1e-4)
            else:
                compare_optimizer(opt1(use_fused_step=False, **kwarg),
                                  opt2(use_fused_step=True, **kwarg),
                                  shapes, dtype)
            # test operator fallback on cpu
            if dtype != np.float16:
                compare_optimizer(opt1(use_fused_step=False, **kwarg),
                                  opt2(use_fused_step=True, **kwarg),
                                  [shapes[0][:2], shapes[1]],
                                  dtype, w_stype='csr', g_stype='csr')


class PySparseSGD(mx.optimizer.Optimizer):
    """python reference implemenation of sgd"""
    def __init__(self, learning_rate=0.1, momentum=0.0, **kwargs):
        super(PySparseSGD, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        """Create additional optimizer state: momentum

        Parameters
        ----------
        weight : NDArray
        The weight data

        """
        if self.momentum == 0.0:
            return None
        else:
            return mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype)

    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            lr = self._get_lr(index)
            wd = self._get_wd(index)
            self._update_count(index)
            num_rows = weight.shape[0]
            if self.momentum == 0.0:
                # Update on a per row basis, skip all-zero rows
                for row in range(num_rows):
                    grad_row = grad[row].asnumpy()
                    all_zeros = mx.test_utils.almost_equal(grad_row, np.zeros_like(grad_row))
                    if all_zeros:
                        continue
                    grad[row] *= self.rescale_grad
                    if self.clip_gradient is not None:
                        grad[row] = mx.nd.clip(grad[row], -self.clip_gradient, self.clip_gradient)
                    grad[row] += wd * weight[row]
                    weight[row] -= lr * grad[row]
            else:
                mom = state
                for row in range(num_rows):
                    grad_row = grad[row].asnumpy()
                    all_zeros = mx.test_utils.almost_equal(grad_row, np.zeros_like(grad_row))
                    if all_zeros:
                        continue
                    grad[row] *= self.rescale_grad
                    if self.clip_gradient is not None:
                        grad[row] = mx.nd.clip(grad[row], -self.clip_gradient, self.clip_gradient)
                    grad[row] += wd * weight[row]
                    mom[row] *= self.momentum
                    mom[row] -= grad[row]
                    weight[row] += lr * mom[row]


@xfail_when_nonstandard_decimal_separator
def test_sparse_sgd():
    opt1 = PySparseSGD
    opt2 = mx.optimizer.SGD
    shapes = [(3, 4, 5), (10, 4), (7,)]
    mom_options = [{}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float32]:
        for params in itertools.product(mom_options, cg_options, rg_options,
                                        wd_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            compare_optimizer(opt1(**kwarg),
                              opt2(use_fused_step=True, lazy_update=True, **kwarg), shapes, dtype,
                              w_stype='row_sparse', g_stype='row_sparse')
            compare_optimizer(opt1(**kwarg),
                              opt2(use_fused_step=True, lazy_update=True, **kwarg), shapes, dtype,
                              w_stype='default', g_stype='row_sparse')


@xfail_when_nonstandard_decimal_separator
def test_std_sparse_sgd():
    opt1 = mx.optimizer.SGD
    opt2 = mx.optimizer.SGD
    shapes = [(3, 4, 5), (10, 4), (7,)]
    mom_options = [{}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]

    for dtype in [np.float32]:
        for params in itertools.product(mom_options, cg_options, rg_options,
                                        wd_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, lazy_update=False, **kwarg), shapes, dtype,
                              w_stype='row_sparse', g_stype='row_sparse')
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, lazy_update=False, **kwarg), shapes, dtype,
                              w_stype='default', g_stype='row_sparse')


@xfail_when_nonstandard_decimal_separator
def test_nag():
    opt1 = mx.optimizer.NAG
    opt2 = mx.optimizer.NAG
    shapes = [(3, 4, 5), (10, 4), (7,)]
    mom_options = [{}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]

    for dtype in [np.float16, np.float32]:
        for params in itertools.product(mom_options, cg_options, rg_options,
                                        wd_options, mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg),
                              shapes, dtype, rtol=1e-3, atol=1e-4)


@xfail_when_nonstandard_decimal_separator
def test_lars():
    opt1 = mx.optimizer.LARS
    opt2 = mx.optimizer.LARS
    shapes = [(3, 4, 5), (10, 4), (7,)]
    eta_options = [{}, {'eta': 0.002}, {'eta': 0.01}]
    mom_options = [{'momentum': 0.0}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}]
    rg_options = [{}, {'rescale_grad': 0.14}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(eta_options, mom_options, cg_options, rg_options,
                                        wd_options, mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg),
                              shapes, dtype, rtol=1e-3, atol=1e-3)


@xfail_when_nonstandard_decimal_separator
def test_lamb():
    opt1 = mx.optimizer.LAMB
    opt2 = mx.optimizer.LAMB

    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}]
    beta2_options = [{}, {'beta2': 0.8}]
    cg_options = [{}, {'clip_gradient': 0.4}]
    rg_options = [{}, {'rescale_grad': 0.14}]
    wd_options = [{}, {'wd': 0.03}]
    bc_options = [{'bias_correction': False}, {'bias_correction': True}]
    lb_options = [{'lower_bound': None}, {'lower_bound': 1e-3}]
    ub_options = [{'upper_bound': None}, {'upper_bound': 10}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options, rg_options,
                                        wd_options, bc_options, lb_options, ub_options,
                                        mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg),
                              shapes, dtype, rtol=1e-3, atol=1e-3)


@xfail_when_nonstandard_decimal_separator
def test_lans():
    opt1 = mx.optimizer.LANS
    opt2 = mx.optimizer.LANS

    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}]
    beta2_options = [{}, {'beta2': 0.8}]
    cg_options = [{}, {'clip_gradient': 0.4}]
    rg_options = [{}, {'rescale_grad': 0.14}]
    wd_options = [{}, {'wd': 0.03}]
    lb_options = [{'lower_bound': None}, {'lower_bound': 1e-3}]
    ub_options = [{'upper_bound': None}, {'upper_bound': 10}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options, rg_options,
                                        wd_options, lb_options, ub_options,
                                        mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg),
                              shapes, dtype, rtol=1e-3, atol=1e-3)


def test_sgld():
    opt1 = mx.optimizer.SGLD
    opt2 = mx.optimizer.SGLD
    shapes = [(3, 4, 5), (10, 4), (7,)]
    ns_options = [1234, 42]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]

    for seed in ns_options:
        for dtype in [np.float16, np.float32]:
            for params in itertools.product(cg_options, wd_options, mp_options, agg_options):
                kwarg = {k: v for param in params for k, v in param.items()}
                if (dtype == np.float16 and ('multi_precision' not in kwarg
                                             or not kwarg['multi_precision'])):
                    continue
                atol = 1e-2 if dtype == np.float16 else 1e-3
                rtol = 1e-4 if dtype == np.float16 else 1e-5
                compare_optimizer_noise_seeded(opt1(**kwarg),
                                               opt2(**kwarg),
                                               shapes, dtype, seed, atol=atol, rtol=rtol)


@xfail_when_nonstandard_decimal_separator
def test_ftml():
    opt1 = mx.optimizer.FTML
    opt2 = mx.optimizer.FTML
    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}, {'beta1': 0.7}]
    beta2_options = [{}, {'beta2': 0.8}, {'beta2': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]

    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options,
                                        rg_options, wd_options, mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg),
                              shapes, dtype, rtol=1e-3, atol=1e-4)


# Sparse ADAM
class PySparseAdam(mx.optimizer.Optimizer):
    """python reference implemenation of sparse adam"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 lazy_update=False, **kwargs):
        super(PySparseAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
        The weight data

        """
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)
            t = self._index_update_count[index]

            mean, variance = state
            num_rows = weight.shape[0]

            coef1 = 1. - self.beta1 ** t
            coef2 = 1. - self.beta2 ** t
            lr *= math.sqrt(coef2) / coef1

            for row in range(num_rows):
                # check row slices of all zeros
                all_zeros = mx.test_utils.almost_equal(grad[row].asnumpy(),
                                                       np.zeros_like(grad[row].asnumpy()))
                # skip zeros during lazy update
                if all_zeros and self.lazy_update:
                    continue
                grad[row] *= self.rescale_grad
                # clip gradients
                if self.clip_gradient is not None:
                    mx.nd.clip(grad[row], -self.clip_gradient, self.clip_gradient, out=grad[row])
                grad[row] += wd * weight[row]
                # update mean
                mean[row] *= self.beta1
                mean[row] += grad[row] * (1. - self.beta1)
                # update variance
                variance[row] *= self.beta2
                variance[row] += (1 - self.beta2) * mx.nd.square(grad[row], out=grad[row])
                # update weight
                weight[row] -= lr * mean[row] / (mx.nd.sqrt(variance[row]) + self.epsilon)


@xfail_when_nonstandard_decimal_separator
def test_adam():
    opt1 = mx.optimizer.Adam
    opt2 = mx.optimizer.Adam
    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}, {'beta1': 0.7}]
    beta2_options = [{}, {'beta2': 0.8}, {'beta2': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options,
                                        rg_options, wd_options, mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            # atol 2e-5 needed to pass with seed 1248389097
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg), shapes, dtype,
                              rtol=1e-4, atol=2e-5)


@xfail_when_nonstandard_decimal_separator
def test_sparse_adam():
    opt1 = PySparseAdam
    opt2 = mx.optimizer.Adam
    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}]
    beta2_options = [{}, {'beta2': 0.8}]
    cg_options = [{}, {'clip_gradient': 0.4}]
    rg_options = [{}, {'rescale_grad': 0.14}]
    wd_options = [{}, {'wd': 0.03}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options,
                                        rg_options, wd_options, mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            # atol 2e-5 needed to pass with seed 1248389097
            compare_optimizer(opt1(lazy_update=False, **kwarg),
                              opt2(use_fused_step=True, lazy_update=False, **kwarg), shapes, dtype,
                              rtol=1e-4, atol=2e-5)
            # atol 2e-5 needed to pass with seed 781809840
            compare_optimizer(opt1(lazy_update=True, **kwarg),
                              opt2(use_fused_step=True, lazy_update=True, **kwarg), shapes,
                              dtype, w_stype='row_sparse', g_stype='row_sparse',
                              rtol=1e-4, atol=2e-5)
            compare_optimizer(opt1(lazy_update=False, **kwarg),
                              opt2(use_fused_step=True, lazy_update=False, **kwarg), shapes,
                              dtype, w_stype='row_sparse', g_stype='row_sparse',
                              rtol=1e-4, atol=2e-5)
            compare_optimizer(opt1(lazy_update=True, **kwarg),
                              opt2(use_fused_step=True, lazy_update=True, **kwarg), shapes,
                              dtype, w_stype='default', g_stype='row_sparse',
                              rtol=1e-4, atol=2e-5)
            compare_optimizer(opt1(lazy_update=False, **kwarg),
                              opt2(use_fused_step=True, lazy_update=False, **kwarg), shapes,
                              dtype, w_stype='default', g_stype='row_sparse',
                              rtol=1e-4, atol=2e-5)


@xfail_when_nonstandard_decimal_separator
@pytest.mark.skip(reason="Flaky test https://github.com/apache/incubator-mxnet/issues/18400")
def test_adamax():
    opt1 = mx.optimizer.Adamax
    opt2 = mx.optimizer.Adamax
    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}, {'beta1': 0.7}]
    beta2_options = [{}, {'beta2': 0.8}, {'beta2': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options,
                                        rg_options, wd_options, mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and
                    ('multi_precision' not in kwarg or not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shapes, dtype)


@xfail_when_nonstandard_decimal_separator
def test_signum():
    opt1 = mx.optimizer.Signum
    opt2 = mx.optimizer.Signum
    shapes = [(3, 4, 5), (10, 4), (7,)]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    wd_lh_options = [{}, {'wd_lh': 0.015}, {'wd_lh': 0.0}]
    mom_options = [{}, {'momentum': 0.9}]
    lr_options = [{'learning_rate': 0.05},{'learning_rate': 0.01}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(cg_options, rg_options, wd_options,
                                        wd_lh_options, mom_options, lr_options,
                                        mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and
                    ('multi_precision' not in kwarg or not kwarg['multi_precision'])):
                continue
            rtol, atol = (1e-3, 1e-4) if dtype is np.float16 else (1e-4, 1e-5)
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg), shapes, dtype,
                                   rtol=rtol, atol=atol)


@xfail_when_nonstandard_decimal_separator
def test_rms():
    opt1 = mx.optimizer.RMSProp
    opt2 = mx.optimizer.RMSProp
    shapes = [(3, 4, 5), (10, 4), (7,)]
    rho_options = [{}, {'rho': 0.5}]
    cg_options = [{}, {'clip_gradient': 0.4}]
    cw_options = [{}, {'clip_weights': 0.01}]
    center_options = [{'centered': False}, {'centered': True}]
    rg_options = [{}, {'rescale_grad': 0.14}]
    wd_options = [{}, {'wd': 0.03}]
    mom_options = [{'momentum': 0.0}, {'momentum': 0.9}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        # Reduce foating point compare tolerance to avoid flaky test failure.
        rtol, atol = (1e-1, 1e-1) if dtype is np.float16 else (1e-2, 1e-2)

        for params in itertools.product(rho_options, cg_options, cw_options,
                                        center_options, rg_options, wd_options,
                                        mom_options, mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and
                    ('multi_precision' not in kwarg or not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg), shapes, dtype,
                              rtol=rtol, atol=atol)
            if default_context() == mx.cpu():
                compare_optimizer(opt1(use_fused_step=False, **kwarg),
                                  opt2(use_fused_step=True, **kwarg),
                                  shapes, dtype, g_stype='row_sparse', rtol=rtol, atol=atol)


class PySparseFtrl(mx.optimizer.Optimizer):
    """python reference implemenation of sparse Ftrl optimizer.

    Referenced from *Ad Click Prediction: a View from the Trenches*, available at
    http://dl.acm.org/citation.cfm?id=2488200.

    Parameters
    ----------
    lamda1 : float, optional
        L1 regularization coefficient.
    learning_rate : float, optional
        The initial learning rate.
    beta : float, optional
        Per-coordinate learning rate correlation parameter.
    eta :
        .. math::
           \\eta_{t,i} = \\frac{learningrate}{\\beta+\\sqrt{\\sum_{s=1}^tg_{s,i}^t}}
    """

    def __init__(self, lamda1=0.01, learning_rate=0.1, beta=1, **kwargs):
        super(PySparseFtrl, self).__init__(**kwargs)
        self.lamda1 = lamda1
        self.beta = beta
        self.lr = learning_rate

    def create_state(self, index, weight):
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # z
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # n

    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            self._update_count(index)
            wd = self._get_wd(index)
            lr = self._get_lr(index)
            num_rows = weight.shape[0]

            z, n = state
            for row in range(num_rows):
                all_zeros = mx.test_utils.almost_equal(grad[row].asnumpy(), np.zeros_like(grad[row].asnumpy()))
                if all_zeros:
                    continue
                grad[row] *= self.rescale_grad
                if self.clip_gradient is not None:
                    mx.nd.clip(grad[row], -self.clip_gradient, self.clip_gradient, out=grad[row])

                # update z[row], n[row]
                sigma = - mx.nd.sqrt(n[row])
                n[row] += mx.nd.square(grad[row])
                denom = mx.nd.sqrt(n[row])
                sigma += denom
                sigma /= lr
                z[row] += grad[row] - sigma * weight[row]

                # update weight
                denom += self.beta
                denom /= lr
                denom += wd
                d = mx.nd.sign(z[row]) * mx.nd.maximum(mx.nd.abs(z[row]) - self.lamda1, 0)
                weight[row] = - d / denom


@xfail_when_nonstandard_decimal_separator
@retry(3)
def test_ftrl():
    opt1 = mx.optimizer.Ftrl
    opt2 = mx.optimizer.Ftrl
    shapes = [(3, 4, 5), (10, 4), (7,)]
    lamda1_options = [{'lamda1': 0.}, {'lamda1': 0.1}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(lamda1_options, cg_options,
                                        rg_options, wd_options,
                                        mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and
                    ('multi_precision' not in kwarg or not kwarg['multi_precision'])):
                continue
            rtol, atol = (1e-3, 1e-3) if dtype is np.float16 else (1e-4, 1e-4)
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg), shapes, dtype,
                              rtol=rtol, atol=atol)


@xfail_when_nonstandard_decimal_separator
def test_sparse_ftrl():
    opt1 = PySparseFtrl
    opt2 = mx.optimizer.Ftrl
    shapes = [(3, 4, 5), (10, 4), (7,)]
    lamda1_options = [{'lamda1': 0.}, {'lamda1': 0.1}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(lamda1_options, cg_options,
                                        rg_options, wd_options,
                                        mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and
                    ('multi_precision' not in kwarg or not kwarg['multi_precision'])):
                continue
            rtol, atol = (1e-3, 1e-3) if dtype is np.float16 else (1e-4, 1e-4)
            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shapes,
                              dtype, w_stype='row_sparse', g_stype='row_sparse',
                              rtol=rtol, atol=atol)


@xfail_when_nonstandard_decimal_separator
def test_nadam():
    opt1 = mx.optimizer.Nadam
    opt2 = mx.optimizer.Nadam
    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}]
    beta2_options = [{}, {'beta2': 0.8}]
    schedule_decay_options = [{}, {'schedule_decay': 0.008}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options,
                                        schedule_decay_options, rg_options, wd_options,
                                        mp_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and
                    ('multi_precision' not in kwarg or not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shapes, dtype)


class PySparseAdaGrad(mx.optimizer.Optimizer):
    """python reference implemenation of sparse Adagrad optimizer.

    This class implements the AdaGrad optimizer described in *Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization*, and available at
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.

    Parameters
    ----------
    learning_rate : float, default 0.01
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    epsilon : float, default 1e-6
        Small value to avoid division by 0.
    """

    def __init__(self, learning_rate=0.01, epsilon=1e-6, **kwargs):
        super(PySparseAdaGrad, self).__init__(learning_rate=learning_rate,
                                              **kwargs)
        self.epsilon = epsilon

    def create_state(self, index, weight):
        return mx.nd.zeros(weight.shape, weight.context, stype=weight.stype)  # history

    def step(self, indices, weights, grads, states):
        """Perform an optimization step using gradients and states.

        Parameters
        ----------
        indices : list of int
            List of unique indices of the parameters into the individual learning rates
            and weight decays. Learning rates and weight decay may be set via `set_lr_mult()`
            and `set_wd_mult()`, respectively.
        weights : list of NDArray
            List of parameters to be updated.
        grads : list of NDArray
            List of gradients of the objective with respect to this parameter.
        states : List of any obj
            List of state returned by `create_state()`.
        """
        for index, weight, grad, state in zip(indices, weights, grads, states):
            self._update_count(index)
            wd = self._get_wd(index)
            lr = self._get_lr(index)
            num_rows = weight.shape[0]

            history = state
            for row in range(num_rows):
                all_zeros = mx.test_utils.almost_equal(grad[row].asnumpy(), np.zeros_like(grad[row].asnumpy()))
                if all_zeros:
                    continue
                grad[row] *= self.rescale_grad
                if self.clip_gradient is not None:
                    mx.nd.clip(grad[row], -self.clip_gradient, self.clip_gradient, out=grad[row])
                grad[row] += wd * weight[row]

                # update history[row]
                history[row] += mx.nd.square(grad[row])
                denom = mx.nd.sqrt(history[row])
                denom += self.epsilon

                # update weight
                weight[row] -= lr * grad[row] / denom


def test_adagrad():
    opt1 = mx.optimizer.AdaGrad
    opt2 = mx.optimizer.AdaGrad
    shapes = [(3, 4, 5), (10, 4), (7,)]
    eps_options = [{}, {'epsilon': 1e-8}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.0}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(eps_options, cg_options,
                                        rg_options, wd_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if dtype is np.float16:
                kwarg.update({'multi_precision': True})
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg), shapes, dtype)


@xfail_when_nonstandard_decimal_separator
def test_sparse_adagrad():
    opt1 = PySparseAdaGrad
    opt2 = mx.optimizer.AdaGrad
    shapes = [(3, 4, 5), (10, 4), (7,)]
    eps_options = [{}, {'epsilon': 1e-8}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.0}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(eps_options, cg_options,
                                        rg_options, wd_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if dtype is np.float16:
                kwarg.update({'multi_precision': True})
            if kwarg.get('wd', 0.0) == 0.0:
                compare_optimizer(opt1(**kwarg), opt2(use_fused_step=True, **kwarg), shapes, dtype,
                                  w_stype='row_sparse', g_stype='row_sparse')
                compare_optimizer(opt1(**kwarg), opt2(use_fused_step=True, **kwarg), shapes, dtype,
                                  g_stype='row_sparse')


def test_adadelta():
    opt1 = mx.optimizer.AdaDelta
    opt2 = mx.optimizer.AdaDelta
    shapes = [(3, 4, 5), (10, 4), (7,)]
    rho_options = [{'rho': 0.9}]
    eps_options = [{}, {'epsilon': 1e-8}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(rho_options, eps_options, cg_options,
                                        rg_options, wd_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if dtype is np.float16:
                kwarg.update({'multi_precision': True})
            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shapes, dtype)


def test_dcasgd():
    opt1 = mx.optimizer.DCASGD
    opt2 = mx.optimizer.DCASGD
    shapes = [(3, 4, 5), (10, 4), (7,)]
    lamda_options = [{}, {'lamda': 0.01}, {'lamda': 0.1}]
    mom_options = [{}, {'momentum': 0.0}, {'momentum': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(lamda_options, mom_options, cg_options,
                                        rg_options, wd_options, agg_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if dtype is np.float16:
                kwarg.update({'multi_precision': True})
            compare_optimizer(opt1(**kwarg), opt2(**kwarg), shapes, dtype)


def test_adamW():
    opt1 = mx.optimizer.AdamW
    opt2 = mx.optimizer.AdamW
    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}, {'beta1': 0.7}]
    beta2_options = [{}, {'beta2': 0.8}, {'beta2': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    correct_bias_options = [{'correct_bias': True}, {'correct_bias': False}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options,
                                        rg_options, wd_options, mp_options,
                                        agg_options, correct_bias_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg), shapes, dtype,
                              rtol=1e-3, atol=2e-3)

def test_adabelief():
    opt1 = mx.optimizer.AdaBelief
    opt2 = mx.optimizer.AdaBelief
    shapes = [(3, 4, 5), (10, 4), (7,)]
    beta1_options = [{}, {'beta1': 0.5}, {'beta1': 0.7}]
    beta2_options = [{}, {'beta2': 0.8}, {'beta2': 0.9}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    mp_options = [{'multi_precision': False}, {'multi_precision': True}]
    agg_options = [{'aggregate_num': 0}, {'aggregate_num': 1},
                   {'aggregate_num': 4}, {'aggregate_num': np.inf}]
    correct_bias_options = [{'correct_bias': True}, {'correct_bias': False}]
    for dtype in [np.float16, np.float32]:
        for params in itertools.product(beta1_options, beta2_options, cg_options,
                                        rg_options, wd_options, mp_options,
                                        agg_options, correct_bias_options):
            kwarg = {k: v for param in params for k, v in param.items()}
            if (dtype == np.float16 and ('multi_precision' not in kwarg or
                                         not kwarg['multi_precision'])):
                continue
            compare_optimizer(opt1(use_fused_step=False, **kwarg),
                              opt2(use_fused_step=True, **kwarg), shapes, dtype,
                              rtol=1e-3, atol=2e-3)

def test_factor_scheduler():
    base_lr = 1
    step = 100
    factor = 0.1
    sched = mx.lr_scheduler.FactorScheduler(step, factor, stop_factor_lr=1e-4, base_lr=base_lr,
                                        warmup_steps=20, warmup_begin_lr=0.1, warmup_mode='constant')

    assert (sched(0) == 0.1)
    np.testing.assert_almost_equal(sched(10), 0.1)
    assert (sched(21) == base_lr), sched(21)
    np.testing.assert_almost_equal(sched(101), base_lr * factor)
    np.testing.assert_almost_equal(sched(201), base_lr * factor * factor)
    np.testing.assert_almost_equal(sched(1000), 1e-4)


def test_multifactor_scheduler():
    base_lr = 0.1
    steps = [15, 25]
    factor = 0.1
    sched = mx.lr_scheduler.MultiFactorScheduler(steps, factor, base_lr=base_lr,
                                        warmup_steps=10, warmup_begin_lr=0.05, warmup_mode='linear')

    assert sched(0) == 0.05
    np.testing.assert_almost_equal(sched(5), 0.05 + (base_lr - 0.05)/2)
    np.testing.assert_almost_equal(sched(15), base_lr)
    np.testing.assert_almost_equal(sched(16), base_lr * factor)
    np.testing.assert_almost_equal(sched(20), base_lr * factor)
    np.testing.assert_almost_equal(sched(26), base_lr * factor * factor)
    np.testing.assert_almost_equal(sched(100), base_lr * factor * factor)


def test_poly_scheduler():
    base_lr = 3
    final_lr = 0
    steps = 1000
    poly_sched = mx.lr_scheduler.PolyScheduler(steps, base_lr=base_lr, pwr=2, final_lr=final_lr,
                                    warmup_steps=100, warmup_begin_lr=0, warmup_mode='linear')

    np.testing.assert_almost_equal(poly_sched(0), 0)
    np.testing.assert_almost_equal(poly_sched(50), float(base_lr)/2)
    np.testing.assert_almost_equal(poly_sched(100), base_lr)
    assert (poly_sched(101) <  poly_sched(100))
    assert (poly_sched(500) < 1.6)
    np.testing.assert_almost_equal(poly_sched(steps), final_lr)


def test_cosine_scheduler():
    # also tests case without warmup
    base_lr = 3
    final_lr = 0.1
    steps = 1000
    cosine_sched = mx.lr_scheduler.CosineScheduler(steps, base_lr=base_lr, final_lr=final_lr)
    np.testing.assert_almost_equal(cosine_sched(0), base_lr)
    np.testing.assert_almost_equal(cosine_sched(steps), final_lr)
    assert (cosine_sched(500) > 1.5)
