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
Description : Set Dataset module for Wavenet
"""
import os
import numpy as np
from mxnet import nd
from scipy.io import wavfile
from utils import encode_mu_law
# pylint: disable=invalid-name, too-many-arguments

def load_wav(file_nm):
    """
    Description : load wav file
    """
    return wavfile.read(os.getcwd()+'/dataset/'+file_nm)

def data_generation(data, seq_size, mu, ctx):
    """
    Description : data generation to loading data
    """
    div = max(data.max(), abs(data.min()))
    data = data/div
    while True:
        start = np.random.randint(0, data.shape[0]-seq_size)
        ys = data[start:start+seq_size]
        ys = encode_mu_law(ys, mu)
        yield nd.array(ys[:seq_size], ctx=ctx)

def data_generation_sample(data, seq_size, mu, ctx):
    """
    Description : sample data generation to loading data
    """
    div = max(data.max(), abs(data.min()))
    if div == 0:
        raise ("check the data values")
    data = data/div
    start = 0
    ys = data[start:start+seq_size]
    ys = encode_mu_law(ys, mu)
    return nd.array(ys[:seq_size], ctx=ctx)
