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

import mxnet as mx
import mxnet.gluon as gluon
import time
import logging
import sys
import argparse
from mxnet.gluon import nn, rnn

parser = argparse.ArgumentParser(description='Gluon RNN Benchmarking.')
parser.add_argument('--num-layer', type=int, default=1,
                    help='The number of layers of the RNN model')
parser.add_argument('--layout', type=str, default='TNC',
                    help='The layout of the input shape, can be either TNC or NTC.')
parser.add_argument('--specify-shape', type=str,
                    help='Specify the input shape, format batchsize, time-step, embed-size, hidden-size.')
parser.add_argument('--cell-type', type=str, default='lstm',
                    help='RNN cell type, can be either lstm, gru or all to cover both.')
parser.add_argument('--unfuse', action='store_true', default=False,
                    help='Unfuse the RNN layer to stacked RNN cell instead.') 
parser.add_argument('--latency', action='store_true', default=False,
                    help='Measursing the latency, batchsize will be set to 1.')
parser.add_argument('--train', action='store_true', default=False,
                    help='Run backward benchmark.')
parser.add_argument('--dropout', type=float, default=0, 
                    help='float, default 0; Fraction of the input that gets dropped out during training time.')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='bool, default Flase; set True to run benchmarks on GPU.')
parser.add_argument('--no-hybridize', action='store_true', default=False,
                    help='bool, default False; hybridize will introduce a better perf.')
parser.add_argument('--bidirection', action='store_true', default=False,
                    help='bool, default False; Whether to use bidirectional unroll.')

opt = parser.parse_args()
logging.basicConfig(level=logging.INFO)

#[bs, sequence length, embedding size, hidden size]
input_shape_list = [[64,15,500,500],
    [64,20,500,500],
    [64,25,500,500],
    [64,30,500,500],
    [64,35,500,500],
    [64,40,500,500],
    [64,45,500,500],
    [64,50,500,500],
    [16,25,512,512],
    [32,25,512,512],
    [64,25,512,512],
    [128,25,512,512],
    [16,25,1024,1024],
    [32,25,1024,1024],
    [64,25,1024,1024],
    [128,25,1024,1024],
    [16,25,2048,2048],
    [32,25,2048,2048],
    [64,25,2048,2048],
    [128,25,2048,2048],
    [16,25,4096,4096],
    [32,25,4096,4096],
    [64,25,4096,4096],
    [128,25,4096,4096]
 ]

rnncell_type = ['lstm', 'gru', 'all']
input_layout = ['TNC', 'NTC']

ctx = mx.gpu(0) if opt.gpu else mx.cpu()

dropout = opt.dropout
bidirection = opt.bidirection
unfuse = opt.unfuse
celltype = opt.cell_type


def get_rnn_model(input_shape, num_layer, cell_type, ctx, dropout, bidirection=False, layout='TNC', unfuse=False, hybridize=True, is_train=False):

    bs, seq_len, embedding_size, hidden_size = input_shape
    if cell_type == 'lstm':
        rnn_layer = rnn.LSTM(hidden_size, num_layer, dropout=dropout,
                             bidirectional=bidirection, input_size=embedding_size,
                             prefix='_lstm_layer')
    elif cell_type == 'gru':
        rnn_layer = rnn.GRU(hidden_size, num_layer, dropout=dropout,
                            bidirectional=bidirection, input_size=embedding_size,
                            prefix='_gru_layer')

    input_data = mx.sym.Variable('data')

    if unfuse:
        rnn_cell = rnn_layer._unfuse()
        if hybridize:
            rnn_cell.hybridize()
        out, _ = rnn_cell.unroll(length=seq_len, inputs = input_data, layout=layout, merge_outputs=True)
    else:
        if hybridize:
            rnn_layer.hybridize()
        out = rnn_layer(input_data)

        dshape = (bs, seq_len, embedding_size) if layout == 'NTC' else (seq_len, bs, embedding_size)

    if is_train:
        # add training layers
        hidden = mx.sym.Reshape(data = out, shape=(-1, hidden_size))
        pred = mx.sym.FullyConnected(data=hidden, num_hidden=embedding_size, name='pred')
        if layout == 'TNC':
            pred = mx.sym.Reshape(data=pred, shape=(seq_len, -1, embedding_size))
        elif layout == 'NTC':
            pred = mx.sym.Reshape(data=pred, shape=(-1, seq_len, embedding_size))
        softmax_output = mx.sym.SoftmaxOutput(data=pred, name='softmax')

        mod = mx.mod.Module(softmax_output, label_names=('softmax_label',), context=ctx)
        mod.bind(for_training = True, data_shapes=[('data', dshape)],
                label_shapes=[('softmax_label', dshape)])

    else:
        mod = mx.mod.Module(out, label_names=None, context=ctx)
        mod.bind(data_shapes=[('data', dshape)], label_shapes=None)

    batch = mx.io.DataBatch(data=[mx.random.uniform(shape=dshape)], label=[])
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(optimizer='sgd')

    return mod, batch


def rnn_bench_score(mod, batch, is_train, unfuse):
    
    if is_train:
        mod.forward(batch, is_train=True)
        if unfuse:
            for o in mod.get_outputs():
                o.wait_to_read()
        else:
            o = mod.get_outputs()[0]
            o.wait_to_read()
        mod.backward()
        mod.update()
    else:
        mod.forward(batch, is_train=False)
        if unfuse:
            for o in mod.get_outputs():
                o.wait_to_read()
        else:
            o = mod.get_outputs()[0]
            o.wait_to_read()


if __name__ == '__main__':

    num_layer = opt.num_layer
    layout = opt.layout
    latency = opt.latency
    hybridize = not(opt.no_hybridize)

    assert layout in input_layout, "Only TNC or NTC are supported!"
    assert celltype in rnncell_type, "Only LSTM and GRU cell are supported!"

    dry_run = 20
    num_iter = 100
            
    cell_lst = ['lstm', 'gru'] if celltype == 'all' else [celltype]

    if opt.specify_shape:
        input_shape_list = [[int(x) for x in opt.specify_shape.split(',')]]

    for cell in cell_lst:
        logging_type = 'training' if opt.train else 'inference'
        logging.info('%s %s benchmark.', cell, logging_type)
        
        for input_shape in input_shape_list:
            mod, batch = get_rnn_model(input_shape, num_layer, cell, ctx, dropout=dropout, bidirection=bidirection,
                    layout=opt.layout, unfuse=unfuse, hybridize=hybridize, is_train=opt.train)
            # batch will set to 1 for latency test
            if latency:
                input_shape[0] = 1
            for i in range(dry_run + num_iter):
                if i == dry_run:
                    tic = time.time()
                rnn_bench_score(mod, batch, is_train=opt.train, unfuse=unfuse)
            toc = time.time() - tic
            
            bs, seq_len, _, _ = input_shape
            sps = (bs * num_iter) / toc
            if latency:
                logging.info('For BS = %d, Layers = %d, Shape=%s, latency=%0.6f s'%(bs, num_layer, input_shape, (1/sps)))
            else:
                logging.info('For BS = %d, Layers = %d, Shape=%s, SPS=%0.3f sps'%(bs, num_layer, input_shape, sps))

