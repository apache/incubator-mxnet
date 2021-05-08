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
import timeit
import itertools
import argparse
import os

class OpArgMngr(object):
    """Operator argument manager for storing operator workloads."""
    args = {}

    @staticmethod
    def add_workload(funcname, *args, **kwargs):
        if "_specifier" not in kwargs:
            _specifier = funcname
        else:
            _specifier = kwargs["_specififer"]
            del kwargs["_specififer"]
        if _specifier in OpArgMngr.args:
            raise ValueError("duplicate {}".format(_specifier))
        OpArgMngr.args[_specifier] = {'args': args, 'kwargs': kwargs, 'funcname': funcname}


def generate_workloads():
    array_pool = {}
    shapes = []
    for ndim in range(4):
        shapes.extend(list(itertools.product(range(4), repeat=ndim)))
    for shape in shapes:
        name = 'x'.join(str(i) for i in shape)
        if name in array_pool:
            raise ValueError("duplicate array {}".format(name))
        array_pool[name] = dnp.ones(shape)
    return array_pool


def prepare_workloads():
    pool = generate_workloads()
    OpArgMngr.add_workload("zeros", (2, 2))
    OpArgMngr.add_workload("full", (2, 2), 10)
    OpArgMngr.add_workload("identity", 3)
    OpArgMngr.add_workload("ones", (2, 2))
    OpArgMngr.add_workload("einsum", "ii", pool['2x2'], optimize=False)
    OpArgMngr.add_workload("unique", pool['1'], return_index=True, return_inverse=True, return_counts=True, axis=-1)
    OpArgMngr.add_workload("dstack", (pool['2x1'], pool['2x1'], pool['2x1'], pool['2x1']))
    OpArgMngr.add_workload("polyval", dnp.arange(10), pool['2x2'])
    OpArgMngr.add_workload("ediff1d", pool['2x2'], pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("nan_to_num", pool['2x2'])
    OpArgMngr.add_workload("tri", 2, 3, 4)
    OpArgMngr.add_workload("tensordot", pool['2x2'], pool['2x2'], ((1, 0), (0, 1)))
    OpArgMngr.add_workload("cumsum", pool['3x2'], axis=0, out=pool['3x2'])
    OpArgMngr.add_workload("random.shuffle", pool['3'])
    OpArgMngr.add_workload("equal", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("not_equal", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("less", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("greater_equal", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("less_equal", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("maximum", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("minimum", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("sum", pool['2x2'], axis=0, keepdims=True, out=pool['1x2'])
    OpArgMngr.add_workload("std", pool['2x2'], axis=0, ddof=0, keepdims=True, out=pool['1x2'])
    OpArgMngr.add_workload("var", pool['2x2'], axis=0, ddof=1, keepdims=True, out=pool['1x2'])
    OpArgMngr.add_workload("average", pool['2x2'], weights=pool['2'], axis=1, returned=True)
    OpArgMngr.add_workload("histogram", pool['2x2'], bins=10, range=(0.0, 10.0))
    OpArgMngr.add_workload("add", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("cross", pool['2'], pool['2'])
    OpArgMngr.add_workload("linalg.eig", pool['3x3'])
    OpArgMngr.add_workload("linalg.eigh", pool['3x3'])
    OpArgMngr.add_workload("linalg.det", pool['3x3'])
    OpArgMngr.add_workload("linalg.slogdet", pool['3x3'])
    OpArgMngr.add_workload("linalg.matrix_rank", pool['3x3'], pool['1'], hermitian=False)
    OpArgMngr.add_workload("linalg.svd", pool['3x3'])
    OpArgMngr.add_workload("linalg.cholesky", pool['1x1'])
    OpArgMngr.add_workload("linalg.qr", pool['3x3'])
    OpArgMngr.add_workload("linalg.lstsq", pool['2x1'], pool['2'], rcond=None)
    OpArgMngr.add_workload("linalg.eigvals", pool['1x1'])
    OpArgMngr.add_workload("linalg.eigvalsh", pool['1x1'], UPLO='L')
    OpArgMngr.add_workload("linalg.inv", pool['1x1'])
    OpArgMngr.add_workload("linalg.pinv", pool['2x3x3'], pool['1'], hermitian=False)
    OpArgMngr.add_workload("linalg.solve", pool['1x1'], pool['1'])
    OpArgMngr.add_workload("linalg.tensorinv", pool['1x1'], ind=2)
    OpArgMngr.add_workload("linalg.norm", pool['3x3'])
    OpArgMngr.add_workload("linalg.tensorsolve", pool['1x1x1'], pool['1x1x1'], (2, 0, 1))
    OpArgMngr.add_workload("tile", pool['2x2'], 1)
    OpArgMngr.add_workload("trace", pool['2x2'])
    OpArgMngr.add_workload("transpose", pool['2x2'])
    OpArgMngr.add_workload("split", pool['3x3'], (0, 1, 2), axis=1)
    OpArgMngr.add_workload("vstack", (pool['3x3'], pool['3x3'], pool['3x3']))
    OpArgMngr.add_workload("argmax", pool['3x2'], axis=-1)
    OpArgMngr.add_workload("argmin", pool['3x2'], axis=-1)
    OpArgMngr.add_workload("atleast_1d", pool['2'], pool['2x2'])
    OpArgMngr.add_workload("atleast_2d", pool['2'], pool['2x2'])
    OpArgMngr.add_workload("atleast_3d", pool['2'], pool['2x2'])
    OpArgMngr.add_workload("argsort", pool['3x2'], axis=-1)
    OpArgMngr.add_workload("sort", pool['3x2'], axis=-1)
    OpArgMngr.add_workload("indices", dimensions=(1, 2, 3))
    OpArgMngr.add_workload("subtract", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("multiply", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("mod", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("remainder", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("divide", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("true_divide", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("power", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("lcm", pool['2x2'].astype('int32'), pool['2x2'].astype('int32'))
    OpArgMngr.add_workload("diff", pool['2x2'], n=1, axis=-1)
    OpArgMngr.add_workload("inner", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("random.multinomial", n=2, pvals=[1/6.]*6, size=(2,2))
    OpArgMngr.add_workload("random.rand", 3, 2)
    OpArgMngr.add_workload("random.randn", 2, 2)
    OpArgMngr.add_workload("nonzero", pool['2x2'])
    OpArgMngr.add_workload("tril", pool['2x2'], k=0)
    OpArgMngr.add_workload("random.choice", pool['2'], size=(2, 2))
    OpArgMngr.add_workload("take", pool['2'], dnp.array([1,0], dtype='int64'))
    OpArgMngr.add_workload("clip", pool['2x2'], 0, 1)
    OpArgMngr.add_workload("expand_dims", pool['2x2'], axis=0)
    OpArgMngr.add_workload("broadcast_to", pool['2x2'], (2, 2, 2))
    OpArgMngr.add_workload("full_like", pool['2x2'], 2)
    OpArgMngr.add_workload("zeros_like", pool['2x2'])
    OpArgMngr.add_workload("ones_like", pool['2x2'])
    OpArgMngr.add_workload("bitwise_and", pool['2x2'].astype(int), pool['2x2'].astype(int))
    OpArgMngr.add_workload("bitwise_xor", pool['2x2'].astype(int), pool['2x2'].astype(int))
    OpArgMngr.add_workload("bitwise_or", pool['2x2'].astype(int), pool['2x2'].astype(int))
    OpArgMngr.add_workload("copysign", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("arctan2", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("hypot", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("ldexp", pool['2x2'].astype(int), pool['2x2'].astype(int))
    OpArgMngr.add_workload("logical_and", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("logical_or", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("logical_xor", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("random.uniform", low=0, high=1, size=1)
    OpArgMngr.add_workload("random.exponential", scale=2, size=(2,2))
    OpArgMngr.add_workload("random.rayleigh", scale=2, size=(2,2))
    OpArgMngr.add_workload("random.weibull", a=2, size=(2,2))
    OpArgMngr.add_workload("random.pareto", a=2, size=(2,2))
    OpArgMngr.add_workload("random.power", a=2, size=(2,2))
    OpArgMngr.add_workload("random.logistic", loc=2, scale=2, size=(2,2))
    OpArgMngr.add_workload("random.gumbel", loc=2, scale=2, size=(2,2))
    OpArgMngr.add_workload("where", pool['2x3'], pool['2x3'], pool['2x1'])
    OpArgMngr.add_workload("may_share_memory", pool['2x3'][:0], pool['2x3'][:1])
    OpArgMngr.add_workload('squeeze', pool['2x2'], axis=None)
    OpArgMngr.add_workload("pad", pool['2x2'], pad_width=((1,2),(1,2)), mode="constant")
    OpArgMngr.add_workload("prod", pool['2x2'], axis=1, dtype="float64", keepdims=False)
    OpArgMngr.add_workload("around", pool['2x2'], decimals=0)
    OpArgMngr.add_workload("round", pool['2x2'], decimals=1)
    OpArgMngr.add_workload("repeat", pool['2x2'], repeats=1, axis=None)
    OpArgMngr.add_workload("diagflat", pool['2x2'], k=1)
    OpArgMngr.add_workload("diag", pool['2x2'], k=1)
    OpArgMngr.add_workload("diagonal", pool['2x2x2'], offset=-1, axis1=0, axis2=1)
    OpArgMngr.add_workload("diag_indices_from", pool['2x2'])
    OpArgMngr.add_workload("bincount", dnp.arange(3, dtype=int), pool['3'], minlength=4)
    OpArgMngr.add_workload("percentile", pool['2x2x2'], 80, axis=0, out=pool['2x2'],\
                           interpolation='midpoint')
    OpArgMngr.add_workload("quantile", pool['2x2x2'], 0.8, axis=0, out=pool['2x2'],\
                           interpolation='midpoint')
    OpArgMngr.add_workload("all", pool['2x2x2'], axis=(0, 1),\
                           out=dnp.array([False, False], dtype=bool), keepdims=False)
    OpArgMngr.add_workload("any", pool['2x2x2'], axis=(0, 1),\
                           out=dnp.array([False, False], dtype=bool), keepdims=False)
    OpArgMngr.add_workload("roll", pool["2x2"], 1, axis=0)
    OpArgMngr.add_workload("rot90", pool["2x2"], 2)
    OpArgMngr.add_workload("array_split", pool['2X2'], 2, axis=1)
    OpArgMngr.add_workload("vsplit", pool['2X2'], 2)
    OpArgMngr.add_workload("hsplit", pool['2X2'], 2)
    OpArgMngr.add_workload("dsplit", pool['2X2x2'], 2)
    OpArgMngr.add_workload("column_stack", (pool['3x3'], pool['3x3'], pool['3x3']))
    OpArgMngr.add_workload("hstack", (pool['3x3'], pool['3x3'], pool['3x3']))
    OpArgMngr.add_workload("triu", pool['3x3'])
    OpArgMngr.add_workload("array_split", pool['2x2'], 2, axis=1)
    OpArgMngr.add_workload("vsplit", pool['2x2'], 2)
    OpArgMngr.add_workload("hsplit", pool['2x2'], 2)
    OpArgMngr.add_workload("dsplit", pool['2x2x2'], 2)
    OpArgMngr.add_workload("arange", 10)
    OpArgMngr.add_workload("concatenate", (pool['1x2'], pool['1x2'], pool['1x2']), axis=0)
    OpArgMngr.add_workload("append", pool['2x2'], pool['1x2'], axis=0)
    OpArgMngr.add_workload("insert", pool['3x2'], 1, pool['1x1'], axis=0)
    OpArgMngr.add_workload("delete", pool['3x2'], 1, axis=0)
    OpArgMngr.add_workload("blackman", 12)
    OpArgMngr.add_workload("eye", 5)
    OpArgMngr.add_workload("hamming", 12)
    OpArgMngr.add_workload("hanning", 12)
    OpArgMngr.add_workload("linspace", 0, 10, 8, endpoint=False)
    OpArgMngr.add_workload("logspace", 2.0, 3.0, num=4, base=2.0, dtype=onp.float32)
    OpArgMngr.add_workload("matmul", pool['2x2'], pool['2x2'])
    OpArgMngr.add_workload("mean", pool['2x2'], axis=0, keepdims=True)
    OpArgMngr.add_workload("random.gamma", 1, size=(2, 3))
    OpArgMngr.add_workload("random.normal", 1, size=(2, 3))
    OpArgMngr.add_workload("max", pool["2x2"], axis=0, out=pool['2'], keepdims=False)
    OpArgMngr.add_workload("min", pool["2x2"], axis=0, out=pool['2'], keepdims=False)
    OpArgMngr.add_workload("amax", pool["2x2"], axis=1, out=pool['2'], keepdims=False)
    OpArgMngr.add_workload("amin", pool["2x2"], axis=1, out=pool['2'], keepdims=False)

    unary_ops = ['negative', 'reciprocal', 'abs', 'sign', 'rint', 'ceil', 'floor',
                 'bitwise_not', 'trunc', 'fix', 'square', 'sqrt', 'cbrt', 'exp',
                 'log', 'log10', 'log2', 'log1p', 'expm1', 'logical_not', 'isnan',
                 'isinf', 'isposinf', 'isneginf', 'isfinite', 'sin', 'cos', 'tan',
                 'arcsin', 'arccos', 'arctan', 'degrees', 'radians', 'sinh', 'cosh',
                 'tanh', 'arcsinh', 'arccosh', 'arctanh']  # 'rad2deg', 'deg2rad' cannot run without tvm
    for unary_op in unary_ops:
        if unary_op == "bitwise_not":
            OpArgMngr.add_workload(unary_op, dnp.ones((2, 2), dtype=int))
        else:
            OpArgMngr.add_workload(unary_op, pool['2x2'])


def benchmark_helper(f, *args, **kwargs):
    number = 10000
    return timeit.timeit(lambda: f(*args, **kwargs), number=number) / number


def get_op(module, funcname):
    funcname = funcname.split(".")
    for fname in funcname:
        module = getattr(module, fname)
    return module


def run_benchmark(packages):
    results = {}
    for (k, v) in OpArgMngr.args.items():
        result = {}
        for (name, package) in packages.items():
            print('{}.{} running...'.format(name, k))
            op = get_op(package["module"], v["funcname"])
            args = [package["data"](arg) for arg in v["args"]]
            kwargs = {k: package["data"](v) for (k, v) in v["kwargs"].items()}
            benchmark = benchmark_helper(op, *args, **kwargs)
            result[name] = benchmark
        results[k] = result
    return results


def show_results(results):
    print("{:>24}{:>24}{:>24}".format("name", "package", "time(us)"))
    for (specifier, d) in results.items():
        for (k, v) in d.items():
            print("{:>24}{:>24}{:>24}".format(specifier, k, v * 10 ** 6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ffi_type')
    parsed = parser.parse_args()
    if parsed.ffi_type == "cython":
        os.environ['MXNET_ENABLE_CYTHON'] = '1'
        os.environ['MXNET_ENFORCE_CYTHON'] = '1'
    elif parsed.ffi_type == "ctypes":
        os.environ['MXNET_ENABLE_CYTHON'] = '0'
    else:
        raise ValueError("unknown ffi_type {}",format(parsed.ffi_type))
    os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
    import mxnet as mx
    import numpy as onp
    from mxnet import np as dnp

    mx.npx.set_np(dtype=False)
    packages = {
        "onp": {
            "module": onp,
            "data": lambda arr: arr.asnumpy() if isinstance(arr, dnp.ndarray) else arr
        },
        "dnp": {
            "module": dnp,
            "data": lambda arr: arr
        }
    }
    prepare_workloads()
    results = run_benchmark(packages)
    show_results(results)
