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

import argparse
import logging
import time
import mxnet as mx
import numpy as np
from get_data import get_movielens_iter, get_movielens_data
from matrix_fact_parallel_model import matrix_fact_model_parallel_net


logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Run model parallel version of matrix factorization \
                                 with sparse embedding",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epoch', type=int, default=3,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='number of examples per batch')
parser.add_argument('--print-every', type=int, default=100,
                    help='logging frequency')
parser.add_argument('--factor-size', type=int, default=128,
                    help="the factor size of the embedding operation")
parser.add_argument('--num-gpus', type=int, default=2,
                    help="number of gpus to use")
parser.add_argument('--dummy-iter', action='store_true',
                    help="use the dummy data iterator for speed test")

MOVIELENS = {
    'dataset': 'ml-10m',
    'train': './ml-10M100K/r1.train',
    'val': './ml-10M100K/r1.test',
    'max_user': 71569,
    'max_movie': 65135,
}

if __name__ == '__main__':
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    optimizer = 'sgd'
    factor_size = args.factor_size
    dummy_iter = args.dummy_iter
    print_every = args.print_every
    num_gpus = args.num_gpus    
 
    momentum = 0.9
    learning_rate = 0.1

    # prepare dataset and iterators
    max_user = MOVIELENS['max_user']
    max_movies = MOVIELENS['max_movie']
    get_movielens_data(MOVIELENS['dataset'])
    train_iter = get_movielens_iter(MOVIELENS['train'], batch_size, dummy_iter)
    val_iter = get_movielens_iter(MOVIELENS['val'], batch_size, dummy_iter)

    # construct the model
    net = matrix_fact_model_parallel_net(factor_size, factor_size, max_user, max_movies)
    a = time.time()

    # create kvstore
    kv = mx.kvstore.create('local') if num_gpus > 1 else None

    # initialize the module
    mod = mx.module.Module(symbol=net, context=[mx.cpu()]*num_gpus, data_names=['user', 'item'],
        label_names=['score'], group2ctxs={'dev1':mx.cpu(), 'dev2':[mx.cpu(i) for i in range(num_gpus)]})
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
    optim = mx.optimizer.create(optimizer, learning_rate=learning_rate, momentum=momentum,
                                wd=1e-4, rescale_grad=1.0/batch_size)
    mod.init_optimizer(optimizer=optim, kvstore=kv)
    # use MSE as the metric
    metric = mx.metric.create(['MSE'])
    # get the row_sparse user_weight parameter
    user_weight_index = mod._exec_group.param_names.index('user_weight')
    user_weight_params = mod._exec_group.param_arrays[user_weight_index]
    # get the row_sparse item_weight parameter
    item_weight_index = mod._exec_group.param_names.index('item_weight')
    item_weight_params = mod._exec_group.param_arrays[item_weight_index]

    speedometer = mx.callback.Speedometer(batch_size, print_every)
    
    logging.info('Training started ...')

    for epoch in range(num_epoch):
        nbatch = 0
        metric.reset()
        for batch in train_iter:
            nbatch += 1
            if kv:
                # if kvstore is used, we need manually pull sparse weights from kvstore
                user_row_ids = batch.data[0]
                kv.row_sparse_pull('user_weight', user_weight_params, row_ids=[user_row_ids] * num_gpus,
                                   priority=-user_weight_index)
                item_row_ids = batch.data[1]
                kv.row_sparse_pull('item_weight', item_weight_params, row_ids=[item_row_ids] * num_gpus,
                                   priority=-item_weight_index)
            mod.forward_backward(batch)
            # update all parameters
            mod.update()
            # update training metric
            mod.update_metric(metric, batch.label)
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=metric, locals=locals())
            speedometer(speedometer_param)
        # evaluate metric on validation dataset
        score = mod.score(val_iter, ['MSE'])
        logging.info('epoch %d, eval MSE = %s ' % (epoch, score[0][1]))
        # reset the iterator for next pass of data
        train_iter.reset()
        val_iter.reset()
    
    logging.info('Training completed.')
