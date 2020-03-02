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


import tvm
from .. import defop, AllTypes, RealTypes
from ..utils import reduce_axes, assign_by_req


def _compute_with_initial(itype, otype, ndim, reducer, reduce1st_dim, req):
    axes = ([reduce1st_dim, 1 - reduce1st_dim] * ndim)[:ndim]
    a = tvm.placeholder([tvm.size_var() for _ in range(ndim)], name='a', dtype=itype)
    init = tvm.var('init', dtype='float64')
    reduce_output = reduce_axes(a, axes, reducer, otype)
    output_placeholder, final_output = assign_by_req(reduce_output, req, init, tvm.sum, itype=itype)
    s = tvm.create_schedule(final_output.op)
    return s, a, init, output_placeholder, final_output, [reduce_output, final_output]


def _compute(itype, otype, ndim, reducer, reduce1st_dim, req):
    axes = ([reduce1st_dim, 1 - reduce1st_dim] * ndim)[:ndim]
    a = tvm.placeholder([tvm.size_var() for _ in range(ndim)], name='a', dtype=itype)
    reduce_output = reduce_axes(a, axes, tvm.sum, otype)
    output_placeholder, final_output = assign_by_req(reduce_output, req)
    s = tvm.create_schedule(final_output.op)
    return s, a, output_placeholder, final_output, [reduce_output, final_output]


@defop(name='sum_cpu', target='cpu', itype=['float16', 'float32', 'float64', 'int8', 'int32', 'int64', 'bool'],
       otype=['float16', 'float32', 'float64', 'int8', 'int32', 'int64'],
       ndim=[5], req=['kWriteTo', 'kAddTo'], reduce1st_dim=[0, 1],
       initial=[True, False], attrs=["reduce1st_dim", "req", "initial"])
def _sum_cpu(itype, otype, ndim, reduce1st_dim, req, initial):
    if initial:
        s, a, init, output_placeholder, final_output, tensor_list = \
                _compute_with_initial(itype, otype, ndim, tvm.sum, reduce1st_dim, req)
        for t in tensor_list:
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            s[t].parallel(fused)
        return s, [a, init, output_placeholder, final_output]
    else:
        s, a, output_placeholder, final_output, tensor_list = _compute(
            itype, otype, ndim, tvm.sum, reduce1st_dim, req)
        for t in tensor_list:
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            s[t].parallel(fused)
        return s, [a, output_placeholder, final_output]


@defop(name='sum_gpu', target='gpu', itype=['float16', 'float32', 'float64', 'int8', 'int32', 'int64', 'bool'],
       otype=['float16', 'float32', 'float64', 'int8', 'int32', 'int64'],
       ndim=[5], req=['kWriteTo', 'kAddTo'], reduce1st_dim=[0, 1],
       initial=[True, False], attrs=["reduce1st_dim", "req", "initial"])
def _sum_gpu(itype, otype, ndim, reduce1st_dim, req, initial):
    if initial:
        s, a, init, output_placeholder, final_output, tensor_list = \
                _compute_with_initial(itype, otype, ndim, tvm.sum, reduce1st_dim, req)
        num_threads = 64
        for t in tensor_list:
            block_x = tvm.thread_axis("blockIdx.x")
            thread_x = tvm.thread_axis("threadIdx.x")
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            bx, tx = s[t].split(fused, factor=num_threads)
            s[t].bind(bx, block_x)
            s[t].bind(tx, thread_x)
        return s, [a, init, output_placeholder, final_output]
    else:
        s, a, output_placeholder, final_output, tensor_list = _compute(
            itype, otype, ndim, tvm.sum, reduce1st_dim, req)
        num_threads = 64
        for t in tensor_list:
            block_x = tvm.thread_axis("blockIdx.x")
            thread_x = tvm.thread_axis("threadIdx.x")
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            bx, tx = s[t].split(fused, factor=num_threads)
            s[t].bind(bx, block_x)
            s[t].bind(tx, thread_x)
        return s, [a, output_placeholder, final_output]


@defop(name='min_cpu', target='cpu', itype=['float16', 'float32', 'float64', 'int8', 'int32', 'int64', 'bool'],
       otype=['float32', 'float64', 'int8', 'int32', 'int64'],
       ndim=[5], req=['kWriteTo', 'kAddTo'], reduce1st_dim=[0, 1],
       initial=[True, False], attrs=["reduce1st_dim", "req", "initial"])
def _min_cpu(itype, otype, ndim, reduce1st_dim, req, initial):
    if initial:
        s, a, init, output_placeholder, final_output, tensor_list = \
                _compute_with_initial(itype, otype, ndim, tvm.min, reduce1st_dim, req)
        for t in tensor_list:
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            s[t].parallel(fused)
        return s, [a, init, output_placeholder, final_output]
    else:
        s, a, output_placeholder, final_output, tensor_list = _compute(
            itype, otype, ndim, tvm.min, reduce1st_dim, req)
        for t in tensor_list:
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            s[t].parallel(fused)
        return s, [a, output_placeholder, final_output]


@defop(name='min_gpu', target='gpu', itype=['float16', 'float32', 'float64', 'int8', 'int32', 'int64', 'bool'],
       otype=['float32', 'float64', 'int8', 'int32', 'int64'],
       ndim=[5], req=['kWriteTo', 'kAddTo'], reduce1st_dim=[0, 1],
       initial=[True, False], attrs=["reduce1st_dim", "req", "initial"])
def _min_gpu(itype, otype, ndim, reduce1st_dim, req, initial):
    if initial:
        s, a, init, output_placeholder, final_output, tensor_list = \
                _compute_with_initial(itype, otype, ndim, tvm.min, reduce1st_dim, req)
        num_threads = 64
        for t in tensor_list:
            block_x = tvm.thread_axis("blockIdx.x")
            thread_x = tvm.thread_axis("threadIdx.x")
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            bx, tx = s[t].split(fused, factor=num_threads)
            s[t].bind(bx, block_x)
            s[t].bind(tx, thread_x)
        return s, [a, init, output_placeholder, final_output]
    else:
        s, a, output_placeholder, final_output, tensor_list = _compute(
            itype, otype, ndim, tvm.min, reduce1st_dim, req)
        num_threads = 64
        for t in tensor_list:
            block_x = tvm.thread_axis("blockIdx.x")
            thread_x = tvm.thread_axis("threadIdx.x")
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            bx, tx = s[t].split(fused, factor=num_threads)
            s[t].bind(bx, block_x)
            s[t].bind(tx, thread_x)
        return s, [a, output_placeholder, final_output]


@defop(name='max_cpu', target='cpu', itype=['float16', 'float32', 'float64', 'int8', 'int32', 'int64', 'bool'],
       otype=['float32', 'float64', 'int8', 'int32', 'int64'],
       ndim=[5], req=['kWriteTo', 'kAddTo'], reduce1st_dim=[0, 1],
       initial=[True, False], attrs=["reduce1st_dim", "req", "initial"])
def _max_cpu(itype, otype, ndim, reduce1st_dim, req, initial):
    if initial:
        s, a, init, output_placeholder, final_output, tensor_list = \
                _compute_with_initial(itype, otype, ndim, tvm.max, reduce1st_dim, req)
        for t in tensor_list:
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            s[t].parallel(fused)
        return s, [a, init, output_placeholder, final_output]
    else:
        s, a, output_placeholder, final_output, tensor_list = _compute(
            itype, otype, ndim, tvm.max, reduce1st_dim, req)
        for t in tensor_list:
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            s[t].parallel(fused)
        return s, [a, output_placeholder, final_output]


@defop(name='max_gpu', target='gpu', itype=['float16', 'float32', 'float64', 'int8', 'int32', 'int64', 'bool'],
       otype=['float32', 'float64', 'int8', 'int32', 'int64'],
       ndim=[5], req=['kWriteTo', 'kAddTo'], reduce1st_dim=[0, 1],
       initial=[True, False], attrs=["reduce1st_dim", "req", "initial"])
def _max_gpu(itype, otype, ndim, reduce1st_dim, req, initial):
    if initial:
        s, a, init, output_placeholder, final_output, tensor_list = \
                _compute_with_initial(itype, otype, ndim, tvm.max, reduce1st_dim, req)
        num_threads = 64
        for t in tensor_list:
            block_x = tvm.thread_axis("blockIdx.x")
            thread_x = tvm.thread_axis("threadIdx.x")
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            bx, tx = s[t].split(fused, factor=num_threads)
            s[t].bind(bx, block_x)
            s[t].bind(tx, thread_x)
        return s, [a, init, output_placeholder, final_output]
    else:
        s, a, output_placeholder, final_output, tensor_list = _compute(
            itype, otype, ndim, tvm.max, reduce1st_dim, req)
        num_threads = 64
        for t in tensor_list:
            block_x = tvm.thread_axis("blockIdx.x")
            thread_x = tvm.thread_axis("threadIdx.x")
            axes = [axis for axis in t.op.axis]
            fused = s[t].fuse(*axes)
            bx, tx = s[t].split(fused, factor=num_threads)
            s[t].bind(bx, block_x)
            s[t].bind(tx, thread_x)
        return s, [a, output_placeholder, final_output]
