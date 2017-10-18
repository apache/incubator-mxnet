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

"""Scheduling learning rate."""
import logging

class LRScheduler(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    """
    def __init__(self, base_lr=0.01):
        self.base_lr = base_lr

    def __call__(self, num_update):
        """Return a new learning rate.

        The ``num_update`` is the upper bound of the number of updates applied to
        every weight.

        Assume the optimizer has updated *i*-th weight by *k_i* times, namely
        ``optimizer.update(i, weight_i)`` is called by *k_i* times. Then::

            num_update = max([k_i for all i])

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")

class FactorScheduler(LRScheduler):
    """Reduce the learning rate by a factor for every *n* steps.

    It returns a new learning rate by::

        base_lr * pow(factor, floor(num_update/step))

    Parameters
    ----------
    step : int
        Changes the learning rate for every n updates.
    factor : float, optional
        The factor to change the learning rate.
    stop_factor_lr : float, optional
        Stop updating the learning rate if it is less than this value.
    """
    def __init__(self, step, factor=1, stop_factor_lr=1e-8):
        super(FactorScheduler, self).__init__()
        if step < 1:
            raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.count = 0

    def __call__(self, num_update):
        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while num_update > self.count + self.step:
            self.count += self.step
            self.base_lr *= self.factor
            if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr
                logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                             "change in the future", num_update, self.base_lr)
            else:
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
        return self.base_lr

class MultiFactorScheduler(LRScheduler):
    """Reduce the learning rate by given a list of steps.

    Assume there exists *k* such that::

        step[k] <= num_update and num_update < step[k+1]

    Then calculate the new learning rate by::

        base_lr * pow(factor, k+1)

    When warmup_step>1, warmup the learning rate by a const value for first warmup_step steps.

    It returns a new learning rate by::

        begin_lr + (num_update - 1) * const_update

    Parameters
    ----------
    step: list of int
        The list of steps to schedule a change.
    factor: float
        The factor to change the learning rate.
    warmup_step : int
        Changes the learning rate for first 'warmup_step' updates.
    begin_lr : float, optional
        The learning rate at begin.
    stop_lr : float, optional
        Stop updating the learning rate if it is less than this value.
    """
    def __init__(self, step, factor=1, warmup_step=0, begin_lr=0, stop_lr=0):
        super(MultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")

        #multifactor parameter
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

        #warmup parameter
        self.warmup_step = warmup_step
        if warmup_step > 1:
            if step[0] <= warmup_step:
                raise ValueError("Schedule step must be greater than warmup_step")
            if stop_lr <= begin_lr:
                raise ValueError("Stop lr must be greater than begin lr")
            self.begin_lr = begin_lr
            self.stop_lr = stop_lr
            self.const_update = (self.stop_lr - self.begin_lr) / (self.warmup_step - 1)
            self.cur_step = 0

    def __call__(self, num_update):
        """
        Call to schedule current learning rate
        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        if self.warmup_step > 1 and num_update <= self.warmup_step:
            if num_update > self.cur_step:
                self.base_lr = (num_update - 1) * self.const_update + self.begin_lr
                self.cur_step = num_update
                if num_update == self.warmup_step or self.base_lr >= self.stop_lr:
                    self.base_lr = self.stop_lr
                    logging.info("Update[%d]: now learning rate arrived at %0.5e, will not "
                                 "warm up in the future", num_update, self.base_lr)
            else:
                return self.base_lr
        else:
            # NOTE: use while rather than if  (for continuing training via load_epoch)
            while self.cur_step_ind <= len(self.step)-1:
                if num_update > self.step[self.cur_step_ind]:
                    self.count = self.step[self.cur_step_ind]
                    self.cur_step_ind += 1
                    self.base_lr *= self.factor
                    logging.info("Update[%d]: Change learning rate to %0.5e",
                                 num_update, self.base_lr)
                else:
                    return self.base_lr
        return self.base_lr
