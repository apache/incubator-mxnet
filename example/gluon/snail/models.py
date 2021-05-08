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
Module: SNAIL network modulep
"""
import math
import numpy as np
from mxnet import nd
from mxnet.gluon import nn
# pylint: disable=invalid-name, too-many-arguments, arguments-differ, no-member, too-many-instance-attributes
class CasualConv1d(nn.Block):
    """
    Description : Casual convolution 1D class
    """

    def __init__(self, in_channels, out_channels, kernel_size,\
                 stride=1, dilation=1, groups=1, bias=True, **kwargs):
        super(CasualConv1d, self).__init__(**kwargs)
        self.dilation = dilation
        self.padding = dilation * (kernel_size - 1)

        with self.name_scope():
            self.casual_conv = nn.Conv1D(in_channels=in_channels, channels=out_channels,\
                                         kernel_size=kernel_size,\
                                         strides=stride,\
                                         padding=self.padding, dilation=dilation,\
                                         groups=groups, use_bias=bias)

    def forward(self, x):
        out = self.casual_conv(x)
        return out[:, :, :-self.dilation]

class DenseBlock(nn.Block):
    """
    Description : DenseBlock class
    """
    def __init__(self, in_channels, filters, dilation=1, kernel_size=2, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.casual_conv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
            self.casual_conv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, x):
        tanh = nd.tanh(self.casual_conv1(x))
        sigmoid = nd.sigmoid(self.casual_conv1(x))
        out = nd.concat(x, tanh*sigmoid, dim=1)
        return out

class TCBlock(nn.Block):
    """
    Description : TC Block class
    """
    def __init__(self, in_channels, seq_len, filters, **kwargs):
        super(TCBlock, self).__init__(**kwargs)
        layer_count = int(math.ceil(math.log(seq_len)))
        with self.name_scope():
            self.blocks = nn.Sequential()
            for i in range(layer_count):
                self.blocks.add(DenseBlock(in_channels + i * filters, filters, dilation=2 ** (i+1)))

    def forward(self, x):
        x = x.swapaxes(1, 2)
        out = self.blocks(x)
        return out.swapaxes(1, 2)

class AttentionBlock(nn.Block):
    """
    Description : Attention Block class
    """
    def __init__(self, k_size, v_size, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.sqrt_k = math.sqrt(k_size)
        self.show_shape = False
        with self.name_scope():
            self.key_layer = nn.Dense(k_size, flatten=False)
            self.query_layer = nn.Dense(k_size, flatten=False)
            self.value_layer = nn.Dense(v_size, flatten=False)

    def forward(self, x):
        with x.context:
            keys = self.key_layer(x)
            queries = self.query_layer(x)
            values = self.value_layer(x)
            logits = nd.linalg_gemm2(queries, keys.swapaxes(2, 1))
            if self.show_shape:
                print("keys shape:{}".format(keys.shape))
                print("queries shape:{}".format(queries.shape))
                print("logits shape:{}".format(logits.shape))
            #Generate masking part
            mask = np.full(shape=(logits.shape[1], logits.shape[2]), fill_value=1).astype('float')
            mask = np.triu(mask, 1)
            mask = np.expand_dims(mask, 0)
            mask = np.repeat(mask, logits.shape[0], 0)
            np.place(mask, mask == 1, 0.0)
            np.place(mask, mask == 0, 1.0)
            mask = nd.array(mask)
            logits = nd.elemwise_mul(logits, mask)
            probs = nd.softmax(logits / self.sqrt_k, axis=2)
            if self.show_shape:
                print("probs shape:{}".format(probs.shape))
                print("values shape:{}".format(values.shape))
            read = nd.linalg_gemm2(probs, values)
            concat_data = nd.concat(x, read, dim=2)
            return concat_data

class CnnEmbedding(nn.Block):
    """
    Description : CNN embedding class
    """
    def __init__(self, **kwargs):
        super(CnnEmbedding, self).__init__(**kwargs)
        with self.name_scope():
            self.cnn1 = nn.Conv2D(64, 3, padding=1, activation='relu')
            self.bn1 = nn.BatchNorm()
            self.max1 = nn.MaxPool2D(2, 2)
            self.cnn2 = nn.Conv2D(64, 3, padding=1, activation='relu')
            self.bn2 = nn.BatchNorm()
            self.max2 = nn.MaxPool2D(2, 2)
            self.cnn3 = nn.Conv2D(64, 3, padding=1, activation='relu')
            self.bn3 = nn.BatchNorm()
            self.max3 = nn.MaxPool2D(2, 2)
            self.cnn4 = nn.Conv2D(64, 3, padding=1, activation='relu')
            self.bn4 = nn.BatchNorm()
            self.max4 = nn.MaxPool2D(2)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.max1(out)
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.max2(out)
        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.max3(out)
        out = self.cnn4(out)
        out = self.bn4(out)
        out = self.max4(out)
        return out.reshape(out.shape[0], -1)

class SNAIL(nn.Block):
    """
    Description : SNAIL network class
    """
    def __init__(self, N, K, input_dims, ctx, **kwargs):
        super(SNAIL, self).__init__(**kwargs)
        self.N = N
        self.K = K
        self.num_filters = int(math.ceil(math.log(N * K + 1)))
        self.ctx = ctx
        self.num_channels = input_dims + N
        with self.name_scope():
            self.cnn_emb = CnnEmbedding()
            self.attn1 = AttentionBlock(64, 32)
            attn1_out_shape = self.num_channels + 32
            self.tc1 = TCBlock(attn1_out_shape, N*K+1, 128)
            tc1_out_shape = attn1_out_shape + self.num_filters * 128
            self.attn2 = AttentionBlock(256, 128)
            attn2_out_shape = tc1_out_shape + 128
            self.tc2 = TCBlock(attn2_out_shape, N*K+1, 128)
            self.attn3 = AttentionBlock(512, 256)
            self.fc = nn.Dense(N, flatten=False)

    def forward(self, x, labels):
        with x.context:
            batch_size = int(labels.shape[0] / (self.N * self.K + 1))
            last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)]
            labels[last_idxs] = nd.zeros(shape=(batch_size, labels.shape[1]))
            x = self.cnn_emb(x)
            x = nd.concat(x, labels, dim=1)
            x = x.reshape((batch_size, self.N * self.K + 1, -1))
            x = self.attn1(x)
            x = self.tc1(x)
            x = self.attn2(x)
            x = self.tc2(x)
            x = self.attn3(x)
            x = self.fc(x)

        return x
