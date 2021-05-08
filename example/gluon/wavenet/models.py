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
Module: WaveNet network block
"""
from mxnet import nd
from mxnet.gluon import nn
# pylint: disable=invalid-name, too-many-arguments, arguments-differ, attribute-defined-outside-init, too-many-instance-attributes, invalid-sequence-index, no-self-use

class WaveNet(nn.HybridBlock):
    """
    mu: audio quantization size
    n_residue: residue channels
    n_skip: skip channels
    dilation_depth : set the dilation depth for dilation layer
    n_repeat: set number of repeat for dilation layer
    """
    def __init__(self, input_length, mu=256, n_residue=32, n_skip=512, dilation_depth=10, n_repeat=5):
        super(WaveNet, self).__init__()
        self.mu = mu
        self.input_length = input_length
        self.dilation_depth = dilation_depth
        self.dilations = [2**i for i in range(dilation_depth)] * n_repeat
        with self.name_scope():
            self.from_input = nn.Conv1D(in_channels=mu, channels=n_residue, kernel_size=1)
            self.conv_sigmoid = nn.HybridSequential()
            self.conv_tanh = nn.HybridSequential()
            self.skip_scale = nn.HybridSequential()
            self.residue_scale = nn.HybridSequential()
            for d in self.dilations:
                self.conv_sigmoid.add(nn.Conv1D(in_channels=n_residue,\
                 channels=n_residue, kernel_size=2, dilation=d))
                self.conv_tanh.add(nn.Conv1D(in_channels=n_residue,\
                 channels=n_residue, kernel_size=2, dilation=d))
                self.skip_scale.add(nn.Conv1D(in_channels=n_residue,\
                 channels=n_skip, kernel_size=1, dilation=d))
                self.residue_scale.add(nn.Conv1D(in_channels=n_residue,\
                 channels=n_residue, kernel_size=1, dilation=d))
            self.conv_post_1 = nn.Conv1D(in_channels=n_skip, channels=n_skip, kernel_size=1)
            self.conv_post_2 = nn.Conv1D(in_channels=n_skip, channels=mu, kernel_size=1)

    def hybrid_forward(self, F, x):
        output = self.preprocess(x)
        skip_connections = [] # save for generation purposes
        idx = 1
        for s, t, skip_scale, residue_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale, self.residue_scale):
            output, skip = self.residue_forward(F, output, s, t, skip_scale, residue_scale, idx)
            skip_connections.append(skip)
            idx = idx + 1
        # sum up skip connections
        # previous code : output = sum([s[:,:,-output.shape[2]:] for s in skip_connections])
        output_length = self.calc_output_size(idx)
        output = sum([F.slice_axis(s, axis=2, begin=0, end=output_length) for s in skip_connections])
        output = self.postprocess(F, output)
        return output

    def preprocess(self, x):
        """
        Description : module for preprocess
        """
        return self.from_input(x)

    def postprocess(self, F, x):
        """
        Description : module for postprocess
        """
        output = F.relu(x)
        output = self.conv_post_1(output)
        output = F.relu(output)
        output = self.conv_post_2(output)
        output = output.squeeze()
        output = F.transpose(output, axes=(1, 0))
        return output

    def residue_forward(self, F, x, conv_sigmoid, conv_tanh, skip_scale, residue_scale, idx):
        """
        Description : module for residue forward
        """
        output = x
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        #replace code for output = F.sigmoid(output_sigmoid) * F.tanh(output_tanh)
        output = F.sigmoid(output_sigmoid) * F.tanh(output_tanh)

        skip = skip_scale(output)
        
        output = residue_scale(output)

        end_length =  self.calc_output_size(idx-1)
        start_length = end_length - self.calc_output_size(idx)
        # replace code for output = output + x[:, :, -output.shape[2]:]
        output = output + F.slice_axis(x, axis=2, begin=start_length, end=end_length)

        return output, skip
    
    def calc_output_size(self, idx):
        output_length = self.input_length - 1
        
        for di in self.dilations[:idx]:
            output_length = output_length - di
        return output_length
            
        
