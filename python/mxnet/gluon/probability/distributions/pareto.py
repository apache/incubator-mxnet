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
# pylint: disable=wildcard-import
"""Pareto Distribution."""
__all__ = ['Pareto']

from .transformed_distribution import TransformedDistribution
from .exponential import Exponential
from .constraint import Positive
from ..transformation import ExpTransform, AffineTransform
from .utils import getF, sample_n_shape_converter

class Pareto(TransformedDistribution):

    has_grad = True
    support = Positive()
    arg_constraints = {'scale': Positive(),
                       'alpha': Positive()}
    
    def __init__(self, alpha, scale=1.0, F=None, validate_args=None):
        _F = F if F is not None else getF(alpha, scale)
        self.alpha = alpha
        self.scale = scale
        base_dist = Exponential(1 / self.alpha)
        super(Pareto, self).__init__(base_dist, [ExpTransform(), AffineTransform(0, self.scale)])
    
    def sample(self, size=None):
        F = self.F
        return self.scale * (F.np.random.pareto(self.alpha, size) + 1)
    
    def sample_n(self, size=None):
        F = self.F
        return self.scale * (F.np.random.pareto(self.alpha, sample_n_shape_converter(size)) + 1)

    @property
    def mean(self):
        F = self.F
        a = F.np.clip(self.alpha, min=1)
        return a * self.scale / (a - 1)

    def variance(self):
        F = self.F
        a = F.np.clip(self.alpha, min=2)
        return (self.scale ** 2) * a / ((a - 1) ** 2 * (a - 2))

    def entropy(self):
        F = self.F
        return F.np.log(self.scale / self.alpha) + 1 / self.alpha + 1