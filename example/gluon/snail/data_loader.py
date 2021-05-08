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
# under the License
"""
Description : Set Dataloader for SNAIL
"""

from mxnet.gluon.data import DataLoader
from omniglot_dataset import OmniglotDataset
from batch_sampler import BatchSampler
# pylint: disable=invalid-name
def loader(config, ctx):
    """
    Description : dataloder for omniglot dataset
    """
    N = config.N
    K = config.K
    iterations = config.iterations
    batch_size = config.batch_size
    download = config.download

    train_dataset = OmniglotDataset(mode='train', download=download)
    test_dataset = OmniglotDataset(mode='test', download=download)

    tr_sampler = BatchSampler(labels=train_dataset.y,\
                                          classes_per_it=N,\
                                          num_samples=K,\
                                          iterations=iterations,\
                                          batch_size=batch_size)

    te_sampler = BatchSampler(labels=test_dataset.y,\
                                          classes_per_it=N,\
                                          num_samples=K,\
                                          iterations=iterations,\
                                          batch_size=int(batch_size / len(ctx)))

    tr_dataloader = DataLoader(train_dataset, batch_sampler=tr_sampler)
    te_dataloader = DataLoader(test_dataset, batch_sampler=te_sampler)

    return tr_dataloader, te_dataloader
