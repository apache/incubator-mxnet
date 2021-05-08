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
"""NAG optimizer."""
from __future__ import absolute_import
import numpy
from ..ndarray import (zeros, clip)
from ..ndarray import (sgd_update, mp_sgd_update, nag_mom_update, mp_nag_mom_update)
from .optimizer import Optimizer, register

__all__ = ['NAG']


@register
class NAG(Optimizer):
    """Nesterov accelerated gradient.

    This optimizer updates each weight by::

        grad = clip(grad * rescale_grad, clip_gradient) + wd * weight
        state = momentum * state + lr * grad
        weight = weight - (momentum * state + lr * grad)

    Parameters
    ----------
    learning_rate : float, default 0.1
        The initial learning rate. If None, the optimization will use the
        learning rate from ``lr_scheduler``. If not None, it will overwrite
        the learning rate in ``lr_scheduler``. If None and ``lr_scheduler``
        is also None, then it will be set to 0.01 by default.
    momentum : float, default 0.9
       The momentum value.
    multi_precision: bool, default False
        Flag to control the internal precision of the optimizer.
        False: results in using the same precision as the weights (default),
        True: makes internal 32-bit copy of the weights and applies gradients
        in 32-bit precision even if actual weights used in the model have lower precision.
        Turning this on can improve convergence and accuracy when training with float16.
    use_fused_step : bool, default True
        Whether or not to use fused kernels for optimizer.
        When use_fused_step=False, step is called,
        otherwise, fused_step is called.
    """
    def __init__(self, learning_rate=0.1, momentum=0.9, multi_precision=False,
                 use_fused_step=True, **kwargs):
        super(NAG, self).__init__(learning_rate=learning_rate,
                                  multi_precision=multi_precision,
                                  use_fused_step=use_fused_step,
                                  **kwargs)
        self.momentum = momentum

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
        return momentum

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
                d = self.momentum * mom - grad
            else:
                d = -grad

            # update weight
            weight[:] += lr * d

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
        for index, weight, grad, state in zip(indices, weights, grads, states):
            self._update_count(index)
            lr = self._get_lr(index)
            wd = self._get_wd(index)

            kwargs = {'rescale_grad': self.rescale_grad}
            if self.momentum > 0:
                kwargs['momentum'] = self.momentum
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient

            multi_precision = self.multi_precision and weight.dtype == numpy.float16

            if not multi_precision:
                mom = state
                if mom is not None:
                    nag_mom_update(weight, grad, mom, out=weight, lr=lr, wd=wd, **kwargs)
                else:
                    sgd_update(weight, grad, out=weight, lr=lr, wd=wd, **kwargs)
            else:
                weight32, mom = state
                if mom is not None:
                    mp_nag_mom_update(weight, grad, mom, weight32, out=weight,
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
            super(NAG, self).update_multi_precision(indices, weights, grads, states)
