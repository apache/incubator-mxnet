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

from __future__ import print_function
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model
import sys
from common import setup_module, with_seed, teardown_module
import multiprocessing
import pytest


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

mx.reset_np()

@with_seed()
@pytest.mark.parametrize('model_name', [
    'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
    'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    'alexnet', 'inceptionv3',
    'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'squeezenet1.0', 'squeezenet1.1',
    'mobilenet1.0', 'mobilenet0.75', 'mobilenet0.5', 'mobilenet0.25',
    'mobilenetv2_1.0', 'mobilenetv2_0.75', 'mobilenetv2_0.5', 'mobilenetv2_0.25'
])
def test_models(model_name):
    pretrained_to_test = set(['vgg19_bn'])

    test_pretrain = model_name in pretrained_to_test
    model = get_model(model_name, pretrained=test_pretrain, root='model/')
    data_shape = (2, 3, 224, 224) if 'inception' not in model_name else (2, 3, 299, 299)
    eprint('testing forward for %s' % model_name)
    print(model)
    if not test_pretrain:
        model.initialize()
    model(mx.nd.random.uniform(shape=data_shape)).wait_to_read()

def parallel_download(model_name):
    model = get_model(model_name, pretrained=True, root='./parallel_download')
    print(type(model))

@with_seed()
@pytest.mark.skip(reason='MXNet is not yet safe for forking. Tracked in #17782.')
def test_parallel_download():
    processes = []
    name = 'mobilenetv2_0.25'
    for _ in range(10):
        p = multiprocessing.Process(target=parallel_download, args=(name,))
        processes.append(p)
    for p in processes:
        p.start()
    for p in processes:
        p.join()

