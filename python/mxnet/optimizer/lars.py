# coding: utf-8
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
"""LARS optimizer."""
from __future__ import absolute_import
import numpy
from ..ndarray import (zeros, clip, array,
                       multi_sum_sq, multi_lars,
                       norm as NDnorm,
                       where, ones_like)
from ..ndarray import (sgd_update, sgd_mom_update,
                       mp_sgd_update, mp_sgd_mom_update,
                       preloaded_multi_sgd_update, preloaded_multi_sgd_mom_update,
                       preloaded_multi_mp_sgd_update, preloaded_multi_mp_sgd_mom_update)
from .optimizer import Optimizer, register
from .utils import _flatten_list

__all__ = ['LARS']


@register
class LARS(Optimizer):
    """the LARS optimizer from 'Large Batch Training of Convolution Networks' \
    (https://arxiv.org/abs/1708.03888)

    Behave mostly like SGD with momentum and weight decay but is scaling \
    adaptively the learning for each layer:

    .. code-block::

       w_norm = L2norm(weights)
       g_norm = L2norm(gradients)
       if w_norm > 0 and g_norm > 0:
           lr_layer = lr * w_norm / (g_norm + weight_decay * w_norm + epsilon)
       else:
           lr_layer = lr


    Parameters
    ----------
    learning_rate : float, default 0.1
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    momentum : float, default 0.
        The momentum value.
    eta : float, default 0.001
        LARS coefficient used to scale the learning rate.
    epsilon : float, default 1e-8
        Small value to avoid division by 0.
    lazy_update : bool, default False
        Default is False. If True, lazy updates are applied \
        if the storage types of weight and grad are both ``row_sparse``.
    aggregate_num : int, default 1
        Number of weights to be aggregated in a list.
        They are passed to the optimizer for a single optimization step.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.1, momentum=0.0, eta=0.001,
                 epsilon=1e-8, lazy_update=False, use_fused_step=True,
                 aggregate_num=1, **kwargs):
        super(LARS, self).__init__(learning_rate=learning_rate,
                                   use_fused_step=use_fused_step,
                                   aggregate_num=aggregate_num,
                                   **kwargs)
        if not self.use_fused_step:
            assert not lazy_update,\
                'When use_fused_step is set to False, lazy_update has to be turned off.'
        if lazy_update:
            assert not self.multi_precision, \
                'When lazy_update is set to True, multi_precision has be turned off.'
        self.lazy_update = lazy_update
        self.momentum = momentum
        self.eta = eta
        self.epsilon = epsilon
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _l2norm(self, v, rescale=False):
        """L2 Norm implementation"""
        v = v.astype('float32')
        if rescale:
            v *= self.rescale_grad
        norm = NDnorm(v)
        return norm

    def _get_lars(self, index, weight, grad, wd):
        """Returns a scaling factor for the learning rate for this layer"""
        lars = 1.0
        name = self.idx2name[index] if index in self.idx2name else str(index)
        if name.endswith('gamma') or name.endswith('beta') or name.endswith('bias'):
            return lars

        w_norm = self._l2norm(weight)
        g_norm = self._l2norm(grad, rescale=True)

        # calculate lars_trust_ratio
        ratio = w_norm / g_norm
        # becomes NaN if ratio == NaN or 0, otherwise 0
        nan_or_zero = 1 - ratio / ratio
        lars = self.eta * w_norm / (g_norm + wd * w_norm + self.epsilon)
        lars = where(nan_or_zero, ones_like(lars), lars)

        return lars.asscalar()

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

            # compute lars
            # clip grad + wd * weight is performed after computing lars
            lars = self._get_lars(index, weight, grad, wd)
            lr *= lars

            # preprocess grad
            grad *= self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            grad += wd * weight

            # update mom
            mom = state
            if mom is not None:
                mom[:] *= self.momentum
                mom[:] -= grad
            else:
                mom = -grad

            # update weight
            weight[:] += lr * mom

    def fused_step(self, indices, weights, grads, states):
        """Perform a fused optimization step using gradients and states.
        Fused kernel is used for update.

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
        aggregate = self.aggregate_num > 1
        for weight, grad in zip(weights, grads):
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum
        if self.clip_gradient is not None:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            nb_params = len(indices)
            names = [self.idx2name[i] if i in self.idx2name else str(i) for i in indices]
            lars_idx = [i for i in range(nb_params) if
                        not(names[i].endswith('gamma') or names[i].endswith('beta') or
                            names[i].endswith('bias'))]
            nb_lars = len(lars_idx)
            no_lars_idx = [i for i in range(nb_params) if
                           (names[i].endswith('gamma') or names[i].endswith('beta') or
                            names[i].endswith('bias'))]
            cur_ctx = weights[0].context
            full_idx = lars_idx + no_lars_idx
            new_lrs = array([lrs[i] for i in full_idx], ctx=cur_ctx, dtype='float32')
            new_wds = array([wds[i] for i in full_idx], ctx=cur_ctx, dtype='float32')
            new_weights = [weights[i] for i in full_idx]
            new_grads = [grads[i] for i in full_idx]
            new_states = [states[i] for i in full_idx]
            if nb_lars > 0:
                w_sum_sq = multi_sum_sq(*new_weights[:nb_lars], num_arrays=nb_lars)
                g_sum_sq = multi_sum_sq(*new_grads[:nb_lars], num_arrays=nb_lars)
                multi_lars(new_lrs[:nb_lars], w_sum_sq, g_sum_sq, new_wds[:nb_lars],
                           eta=self.eta, eps=self.epsilon, rescale_grad=self.rescale_grad,
                           out=new_lrs[:nb_lars])
            # Same than usual using preloaded sgd functions
            multi_precision = self.multi_precision and weights[0].dtype == numpy.float16
            if not multi_precision:
                if self.momentum > 0:
                    preloaded_multi_sgd_mom_update(
                        *(_flatten_list(zip(new_weights, new_grads, new_states)) +
                          [new_lrs, new_wds]), out=new_weights, num_weights=len(new_weights),
                        **kwargs)
                else:
                    preloaded_multi_sgd_update(
                        *(_flatten_list(zip(new_weights, new_grads)) +
                          [new_lrs, new_wds]), out=new_weights, num_weights=len(new_weights),
                        **kwargs)
            else:
                states = list(zip(*states))
                weights32, moms = states
                if self.momentum > 0:
                    preloaded_multi_mp_sgd_mom_update(
                        *(_flatten_list(zip(new_weights, new_grads, moms, weights32)) +
                          [new_lrs, new_wds]), out=new_weights, num_weights=len(new_weights),
                        **kwargs)
                else:
                    preloaded_multi_mp_sgd_update(
                        *(_flatten_list(zip(new_weights, new_grads, weights32)) +
                          [new_lrs, new_wds]), out=new_weights, num_weights=len(new_weights),
                        **kwargs)
        else:
            for i, (index, weight, grad, state) in enumerate(zip(indices, weights, grads, states)):
                wd = wds[i]
                lr = lrs[i]
                lr *= self._get_lars(index, weight, grad, wd)
                multi_precision = self.multi_precision and weights[0].dtype == numpy.float16
                if not multi_precision:
                    mom = state
                    if state is not None:
                        sgd_mom_update(weight, grad, mom, out=weight,
                                       lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)
                    else:
                        sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
                                   lr=lr, wd=wd, **kwargs)
                else:
                    weight32, mom = state
                    if mom is not None:
                        mp_sgd_mom_update(weight, grad, mom, weight32, out=weight,
                                          lr=lr, wd=wd, **kwargs)
                    else:
                        mp_sgd_update(weight, grad, weight32, out=weight,
                                      lr=lr, wd=wd, **kwargs)

    def update_multi_precision(self, indices, weights, grads, states):
        """Override update_multi_precision.
        """
        if self.use_fused_step:
            self.update(indices, weights, grads, states)
        else:
            super(LARS, self).update_multi_precision(indices, weights, grads, states)
