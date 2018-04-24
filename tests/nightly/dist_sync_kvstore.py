#!/usr/bin/env python

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

# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
import argparse
import mxnet as mx
import numpy as np
import numpy.random as rnd
from mxnet.test_utils import assert_almost_equal
from test_kvstore import compute_expected_2bit_quantization

def check_diff(A, x, rank=None):
    """ assert A == x
        x can be scalar as well as numpy array
    """
    assert (np.sum(np.abs((A - x).asnumpy())) == 0), (rank, A.asnumpy(), x.asnumpy())

# setup
shape = (2, 3)
irregular_shape = (1211,1211)
big_shape = (1200, 1200)        # bigger than MXNET_KVSTORE_BIGARRAY_BOUND

keys_shape = ['3', '5', '7']
keys_big_shape = ['99']
fp16_keys_shape = ['4', '6', '8']
fp16_keys_big_shape = ['100']

rsp_keys_shape = ['9', '11', '13']
rsp_keys_big_shape = ['97']
fp16_rsp_keys_shape = ['10', '12', '14']
fp16_rsp_keys_big_shape = ['98']

keys_shapes = [(k, shape) for k in keys_shape] + [(k, big_shape) for k in keys_big_shape]
fp16_keys_shapes = [(k, shape) for k in fp16_keys_shape] + [(k, big_shape) for k in fp16_keys_big_shape]

init_test_keys = [str(i) for i in range(200, 300)]
init_test_keys_big = [str(i) for i in range(300, 400)]
init_test_keys_device = [str(i) for i in range(400, 500)]
init_test_keys_device_big = [str(i) for i in range(500, 600)]

compr_keys_shapes = [('1000', shape), ('1200', irregular_shape),('1300', big_shape)]
compr_init_keys_shapes = [('1001', shape), ('1201', irregular_shape),('1301', big_shape)]
compr_random_keys_shapes = [('1002', shape),('1202', irregular_shape),('1302', big_shape)]

rate = 2

kv = mx.kv.create('dist_sync')

my_rank = kv.rank
nworker = kv.num_workers

def init_kv():
    # # init kv dns keys
    kv.init(keys_shape, [mx.nd.ones(shape)] * len(keys_shape))
    kv.init(keys_big_shape, [mx.nd.ones(big_shape)] * len(keys_big_shape))
    # # init kv row_sparse keys
    kv.init(rsp_keys_shape, [mx.nd.ones(shape).tostype('row_sparse')] * len(rsp_keys_shape))
    kv.init(rsp_keys_big_shape, [mx.nd.ones(big_shape).tostype('row_sparse')] * len(rsp_keys_big_shape))
    # init fp16 dns keys
    kv.init(fp16_keys_shape, [mx.nd.ones(shape, dtype='float16')] * len(keys_shape))
    kv.init(fp16_keys_big_shape, [mx.nd.ones(big_shape, dtype='float16')] * len(keys_big_shape))
    # init fp16 row_sparse keys
    kv.init(fp16_rsp_keys_shape, [mx.nd.ones(shape, dtype='float16').tostype('row_sparse')] * len(fp16_rsp_keys_shape))
    kv.init(fp16_rsp_keys_big_shape, [mx.nd.ones(big_shape, dtype='float16').tostype('row_sparse')] * len(fp16_rsp_keys_big_shape))
    return kv

def set_optimizer(use_multiprecision):
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate, multi_precision=use_multiprecision))
    return kv

def init_kv_compressed(kv):
    threshold = 0.5
    kv.set_gradient_compression({'type': '2bit', 'threshold': threshold})
    # init kv compression keys
    for k, s in compr_keys_shapes:
        kv.init(k, mx.nd.zeros(s))
    # to test inactive mode
    for k, s in compr_init_keys_shapes:
        kv.init(k, mx.nd.ones(s))
    return kv, threshold

def test_sync_push_pull(nrepeat):
    def check_default_keys(dtype, nrepeat):
        # checks pull after push in loop, because behavior during
        # consecutive pushes doesn't offer any guarantees
        ks = keys_shapes if dtype == 'float32' else fp16_keys_shapes
        for k, s in ks:
            for i in range(nrepeat):
                kv.push(k, mx.nd.ones(s, dtype=dtype)*(my_rank+1))
                num = (nworker + 1) * nworker * rate / 2 * (i + 1) + 1
                val = mx.nd.zeros(s, dtype=dtype)
                kv.pull(k, out=val)
                check_diff(val, num)

    def check_row_sparse_keys(dtype, nrepeat):
        # prepare gradient
        v = mx.nd.zeros(shape, dtype=dtype)
        my_row = my_rank % shape[0]
        v[my_row] = my_rank + 1
        # push
        if dtype == 'float32':
            k = rsp_keys_shape[0]
        else:
            k = fp16_rsp_keys_shape[0]
        s = shape
        for i in range(nrepeat):
            kv.push(k, v.tostype('row_sparse'))
            # select a random subset of rows this worker is interested in
            num_rows = s[0]
            row_ids_np = np.random.randint(num_rows, size=num_rows)
            row_ids = mx.nd.array(row_ids_np).reshape((num_rows/2, 2)).astype(dtype)
            # perform pull
            val = mx.nd.zeros(s, stype='row_sparse', dtype=dtype)
            kv.row_sparse_pull(k, out=val, row_ids=row_ids)
            # prepare updated values
            updated_val = mx.nd.ones(s, dtype=dtype)
            for rank in range(nworker):
                row = rank % s[0]
                updated_val[row] += (rank + 1) * rate * (i+1)
            # verify subset of updated values
            expected = mx.nd.zeros(s, dtype=dtype)
            for row in row_ids_np:
                expected[row] = updated_val[row]
            check_diff(val, expected, kv.rank)

    def check_row_sparse_keys_with_zeros(dtype, nrepeat):
        if dtype == 'float32':
            k1 = rsp_keys_shape[1]
            k2 = rsp_keys_big_shape[0]
        else:
            k1 = fp16_rsp_keys_shape[1]
            k2 = fp16_rsp_keys_big_shape[0]
        # prepare gradient
        v = mx.nd.sparse.zeros('row_sparse', shape, dtype=dtype)
        big_v = mx.nd.sparse.zeros('row_sparse', big_shape, dtype=dtype)
        # push
        for i in range(nrepeat):
            kv.push(k1, v)
            kv.push(k2, big_v)
            # pull a subset of rows this worker is interested in
            all_row_ids = np.arange(shape[0])
            val = mx.nd.sparse.zeros('row_sparse', shape)
            big_val = mx.nd.sparse.zeros('row_sparse', big_shape)
            kv.row_sparse_pull(k1, out=val, row_ids=mx.nd.array(all_row_ids))
            big_all_row_ids = np.arange(big_shape[0])
            kv.row_sparse_pull(k2, out=big_val, row_ids=mx.nd.array(big_all_row_ids))
            # verify results
            check_diff(val, 1)
            check_diff(big_val, 1)
            # pull empty weights
            kv.row_sparse_pull(k1, out=val, row_ids=mx.nd.array([]))
            kv.row_sparse_pull(k2, out=big_val, row_ids=mx.nd.array([]))
            check_diff(val, 0)
            check_diff(big_val, 0)

    def check_big_row_sparse_keys(dtype, nrepeat):
        if dtype == 'float32':
            k = rsp_keys_big_shape[0]
        else:
            k = fp16_rsp_keys_big_shape[0]

        mx.random.seed(123)
        rnd.seed(123)
        density = 0.3
        # prepare gradient
        v = mx.nd.zeros(big_shape, dtype=dtype)
        idx_sample = rnd.rand(big_shape[0])
        indices = np.argwhere(idx_sample < density).flatten()
        # each worker chooses a subset of the indices to update
        update_rows = []
        for rank in range(nworker):
            rows = []
            i = 0
            step = (rank + 1) * 2
            while i < len(indices):
                rows.append(indices[i])
                i += step
            update_rows.append(np.array(rows))
        # rows to update for this worker
        for row in update_rows[my_rank]:
            v[row] = my_rank + 1
        # push
        for i in range(nrepeat):
            kv.push(k, v.tostype('row_sparse'))
            # select a random subset of rows this worker is interested in
            mx.random.seed(my_rank)
            rnd.seed(my_rank)
            num_rows = big_shape[0]
            row_ids_np = np.random.randint(num_rows, size=num_rows)
            row_ids = mx.nd.array(row_ids_np).reshape((num_rows/2, 2)).astype(dtype)
            # perform pull
            val = mx.nd.zeros(big_shape, stype='row_sparse', dtype=dtype)
            kv.row_sparse_pull(k, out=val, row_ids=row_ids)
            # prepare expected result
            updated_val = mx.nd.ones(big_shape, dtype=dtype)
            # apply updates from each worker
            for rank in range(nworker):
                for row in update_rows[rank]:
                    updated_val[row] += (rank + 1) * rate * (i+1)

            expected = mx.nd.zeros(big_shape, dtype=dtype)
            for row in row_ids_np:
                expected[row] = updated_val[row]
            check_diff(val, expected.astype(dtype), rank=my_rank)

    for dtype in ['float16', 'float32']:
        check_default_keys(dtype, nrepeat)
        check_row_sparse_keys(dtype, nrepeat)
        check_row_sparse_keys_with_zeros(dtype, nrepeat)
        check_big_row_sparse_keys(dtype, nrepeat)
    print('worker ' + str(my_rank) + ' is done with non compression tests')

def test_sync_2bit_compression(threshold, nrepeat):
    def check_compr_residual(threshold):
        for k, s in compr_keys_shapes:
            # doesn't meet threshold
            kv.push(k, mx.nd.ones(s) * 0.4)
            val = mx.nd.zeros(s)
            kv.pull(k,val)
            check_diff(val, 0)

            # just meets threshold with residual
            kv.push(k, mx.nd.ones(s) * (threshold - 0.4))
            val2 = mx.nd.zeros(s)
            kv.pull(k,val2)
            curval = threshold * rate * nworker
            check_diff(val2, curval)

            # doesn't meet threshold
            kv.push(k, mx.nd.ones(s) * 0.2)
            val3 = mx.nd.zeros(s)
            kv.pull(k, val3)
            check_diff(val3, curval)

            # exceeds again
            kv.push(k, mx.nd.ones(s) * (threshold-0.2))
            val4 = mx.nd.zeros(s)
            kv.pull(k, val4)
            curval += threshold * rate * nworker
            check_diff(val4, curval)
            # residual is 0 now

    def check_compr_ones(threshold):
        for k, s in compr_keys_shapes:
            val = mx.nd.zeros(s)
            kv.pull(k, val)
            curval = val[0][0].asnumpy()[0]
            kv.push(k,mx.nd.ones(s) * threshold)
            val2 = mx.nd.zeros(s)
            kv.pull(k, val2)
            newval = curval + rate * nworker * threshold
            check_diff(val2, newval)
            # residual = 0  again

    def check_compr_pull_before_push():
        for k,s in compr_keys_shapes:
            val = mx.nd.ones(s)
            kv.pull(k, val)
            check_diff(val, 0)
        for k, s in compr_init_keys_shapes:
            # tests that GC is not used for init of a key
            val = mx.nd.zeros(s)
            kv.pull(k, val)
            check_diff(val, 1)

    def check_compr_zero():
        for k,s in compr_keys_shapes:
            kv.push(k, mx.nd.zeros(s))
            # to check that all are set to 0s
            val = mx.nd.ones(s)
            kv.pull(k, val)
            check_diff(val, 0)

    def check_compr_random(threshold, nrepeat):
        # set a seed so all workers generate same data. knowing this helps
        # calculate expected value after pull
        mx.random.seed(123)
        rnd.seed(123)

        # use new keys so residual is 0 for calculation of expected
        for k,s in compr_random_keys_shapes:
            kv.init(k, mx.nd.zeros(s))
        for k,s in compr_random_keys_shapes:
            curr_residual = np.zeros(s)
            for l in range(nrepeat):
                orig_val = mx.nd.zeros(s)
                kv.pull(k, orig_val)

                grad = mx.nd.array(rnd.rand(s[0], s[1]))
                # creates a copy because push changes grad because of assignment
                grad_cpy = mx.nd.array(grad)
                kv.push(k, grad)
                val = mx.nd.zeros(s)
                kv.pull(k, val)

                diff = val - orig_val

                # compute expected by using simulation of operator
                compr, curr_residual, decompr = compute_expected_2bit_quantization(grad_cpy, curr_residual, threshold)
                decompr *= nworker * rate
                assert_almost_equal(diff.asnumpy(), decompr)

    print ('worker ' + str(my_rank) + ' started with compression tests')
    check_compr_pull_before_push()
    check_compr_zero()
    check_compr_residual(threshold)
    check_compr_ones(threshold)
    check_compr_random(threshold, nrepeat)
    print('worker ' + str(my_rank) + ' is done with compression tests')

def test_sync_init(gpu_tests=False):
    def get_dtype(idx, cur_keys):
        if idx < len(cur_keys)/2:
            dtype = 'float32'
        else:
            dtype = 'float16'
        return dtype

    def check_init(kv, cur_keys, cur_shape, device=False):
        ctx = mx.gpu(0) if device else mx.cpu()
        val = [mx.nd.zeros(cur_shape, ctx=ctx, dtype=get_dtype(i, cur_keys)) for i in range(len(cur_keys))]
        for i in range(len(cur_keys)):
            expected = i
            kv.init(cur_keys[i], [mx.nd.ones(cur_shape, ctx=ctx, dtype=get_dtype(i, cur_keys)) * i])
            kv.pull(cur_keys[i], out=val[i])
            check_diff(val[i], expected)
    check_init(kv, init_test_keys, shape)
    check_init(kv, init_test_keys_big, big_shape)
    if gpu_tests:
        check_init(kv, init_test_keys_device, shape, device=True)
        check_init(kv, init_test_keys_device_big, big_shape, device=True)
    print('worker ' + str(kv.rank) + ' is initialized')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test distributed kvstore in dist_sync mode')
    parser.add_argument('--nrepeat', type=int, default=7)
    parser.add_argument('--type', type=str, default='all')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.add_argument('--no-multiprecision', dest='multiprecision', action='store_false')
    opt = parser.parse_args()
    if opt.type == 'all' or  opt.type == 'init':
        test_sync_init(opt.gpu)
    kv = init_kv()
    if opt.type == 'all' or  opt.type == 'default':
        kv = set_optimizer(use_multiprecision=opt.multiprecision)
        test_sync_push_pull(opt.nrepeat)
    # dont run non compressed tests after this as kvstore compression will be set here
    if opt.type == 'all' or  opt.type == 'compressed':
        kv, threshold = init_kv_compressed(kv)
        test_sync_2bit_compression(threshold, opt.nrepeat)
