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
Description : util module
"""
import numpy as np
from mxnet import nd
# pylint: disable=invalid-name, no-member, redefined-builtin
def batch_for_few_shot(num_cls, num_samples, batch_size, x, y):
    """
    Description : generate batch for few shot
    """
    seq_size = num_cls * num_samples + 1
    one_hots = []
    last_targets = []
    for i in range(batch_size):
        one_hot, idxs = labels_to_one_hot(y[i * seq_size: (i + 1) * seq_size])
        one_hots.append(one_hot)
        last_targets.append(idxs[-1])
    one_hots = [nd.array(temp) for temp in one_hots]
    y = nd.stack(*one_hots)
    y = y.reshape(-1, y.shape[-1])
    last_targets = nd.array(last_targets)
    return x, y, last_targets

def labels_to_one_hot(labels):
    """
    Description : process labels into one hot data
    """
    labels = labels.asnumpy()
    unique = np.unique(labels)
    map = {label:idx for idx, label in enumerate(unique)}
    idxs = [map[labels[i]] for i in range(labels.size)]
    one_hot = np.zeros((labels.size, unique.size))
    one_hot[np.arange(labels.size), idxs] = 1
    return one_hot, idxs
