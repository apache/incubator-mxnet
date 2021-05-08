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


import math
import random
from functools import reduce
from operator import mul
import random

from common import xfail_when_nonstandard_decimal_separator
import mxnet
from mxnet import nd, autograd, gluon
from mxnet.test_utils import (
    assert_almost_equal, random_arrays, random_uniform_arrays, rand_shape_nd, same)


def test_sin():
    def sin(x):
        return nd.sin(x)

    def grad_grad_op(x):
        return -nd.sin(x)

    def grad_grad_grad_op(x):
        return -nd.cos(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sin, grad_grad_op)
        # TODO(kshitij12345): Remove
        check_nth_order_unary(array, sin,
                              [grad_grad_op, grad_grad_grad_op], [2, 3])


def test_cos():
    def cos(x):
        return nd.cos(x)

    def grad_grad_op(x):
        return -nd.cos(x)

    def grad_grad_grad_op(x):
        return nd.sin(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, cos, grad_grad_op)
        # TODO(kshitij12345): Remove
        check_nth_order_unary(array, cos,
                              [grad_grad_op, grad_grad_grad_op], [2, 3])


def test_tan():
    def tan(x):
        return nd.tan(x)

    def grad_op(x):
        return 1 / nd.cos(x)**2

    def grad_grad_op(x):
        return 2 * tan(x) * grad_op(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, tan, grad_grad_op)


def test_sinh():
    def sinh(x):
        return nd.sinh(x)

    def grad_grad_op(x):
        return sinh(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sinh, grad_grad_op)


def test_cosh():
    def cosh(x):
        return nd.cosh(x)

    def grad_grad_op(x):
        return cosh(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, cosh, grad_grad_op)


def test_tanh():
    def tanh(x):
        return nd.tanh(x)

    def grad_op(x):
        return 1 - tanh(x)**2

    def grad_grad_op(x):
        return -2 * tanh(x) * grad_op(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_nth_order_unary(array, tanh, grad_op, 1, rtol=1e-6, atol=1e-6)
        check_second_order_unary(
            array, tanh, grad_grad_op, rtol=1e-6, atol=1e-5)


def test_arcsin():
    def arcsin(x):
        return nd.arcsin(x)

    def grad_grad_op(x):
        return x / nd.sqrt((1-x**2)**3)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        # Domain of arcsin is [-1, 1]
        array = random_uniform_arrays(shape, low=-0.99, high=0.99)[0]
        check_second_order_unary(array, arcsin, grad_grad_op)


def test_arccos():
    def arccos(x):
        return nd.arccos(x)

    def grad_grad_op(x):
        return -x / nd.sqrt((1-x**2)**3)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        # Domain of arccos is [-1, 1]
        array = random_uniform_arrays(shape, low=-0.99, high=0.99)[0]
        check_second_order_unary(array, arccos, grad_grad_op)


@xfail_when_nonstandard_decimal_separator
def test_arctan():
    def arctan(x):
        return nd.arctan(x)

    def grad_grad_op(x):
        return (-2 * x)/((1 + x**2)**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        # Domain of arctan is all real numbers.
        # Scale std_dev
        array *= random.randint(500, 10000)
        check_second_order_unary(array, arctan, grad_grad_op)


def test_arcsinh():
    def arcsinh(x):
        return nd.arcsinh(x)

    def grad_grad_op(x):
        return x/nd.sqrt((nd.square(x)+1)**3)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, arcsinh, grad_grad_op)


def test_arccosh():
    def arccosh(x):
        return nd.arccosh(x)

    def grad_grad_op(x):
        return x/(nd.sqrt(x-1) * nd.sqrt(x+1) * (x+1) * (x-1))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = array * sigma + mu
        # Domain of arccosh 1 to infinity.
        assert((array > 1).all())
        check_second_order_unary(array, arccosh, grad_grad_op)


@xfail_when_nonstandard_decimal_separator
def test_arctanh():
    def arctanh(x):
        return nd.arctanh(x)

    def grad_grad_op(x):
        return (2 * x)/((1 - x**2)**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        # Domain of arctanh is (-1, 1)
        array = random_uniform_arrays(shape, low=-0.99, high=0.99)[0]
        check_second_order_unary(array, arctanh, grad_grad_op)


def test_radians():
    def radians(x):
        return nd.radians(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, radians, grad_grad_op)


def test_relu():
    def relu(x):
        return nd.relu(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, relu, grad_grad_op)


def test_log():
    def log(x):
        return nd.log(x)

    def grad_op(x):
        return 1/x

    def grad_grad_op(x):
        return -1/(x**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log, grad_grad_op)
        # TODO(kshitij12345): Remove
        check_nth_order_unary(array, log, [grad_op, grad_grad_op], [1, 2])


@xfail_when_nonstandard_decimal_separator
def test_log2():
    def log2(x):
        return nd.log2(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(2))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log2, grad_grad_op)


@xfail_when_nonstandard_decimal_separator
def test_log10():
    def log10(x):
        return nd.log10(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(10))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log10, grad_grad_op)


@xfail_when_nonstandard_decimal_separator
def test_square():
    def grad_grad_op(x):
        return nd.ones_like(x) * 2

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, nd.square, grad_grad_op)


def test_expm1():
    def grad_grad_op(x):
        return nd.exp(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, nd.expm1, grad_grad_op)


def test_log1p():
    def grad_grad_op(x):
        return -1/((1+x)**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, nd.log1p, grad_grad_op)


def test_reciprocal():
    def reciprocal(x):
        return nd.reciprocal(x)

    def grad_grad_op(x):
        return 2 / x**3

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, reciprocal, grad_grad_op)


def test_abs():
    def abs(x):
        return nd.abs(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, abs, grad_grad_op)


def test_clip():
    def clip(x):
        a_min, a_max = sorted([random.random(), random.random()])

        return nd.clip(x, a_min, a_max)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, clip, grad_grad_op)


def test_dropout():
    def dropout(x):
        return nd.Dropout(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, dropout, grad_grad_op)


def test_sigmoid():
    def sigmoid(x):
        return nd.sigmoid(x)

    def grad_op(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def grad_grad_op(x):
        return grad_op(x) * (1 - 2 * sigmoid(x))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sigmoid, grad_grad_op)
        # TODO(kshitij12345): Remove
        check_nth_order_unary(array, sigmoid, [grad_op, grad_grad_op], [1, 2])
        check_nth_order_unary(array, sigmoid, grad_grad_op, 2)


@xfail_when_nonstandard_decimal_separator
def test_sqrt():
    def sqrt(x):
        return nd.sqrt(x)

    def grad_grad_op(x):
        return -1/(4 * sqrt(x**3))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = sigma * array + mu
        # Only positive numbers
        assert((array > 0).all())
        check_second_order_unary(array, sqrt, grad_grad_op)


def test_cbrt():
    def cbrt(x):
        return nd.cbrt(x)

    def grad_grad_op(x):
        return -2/(9 * cbrt(x**5))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = sigma * array + mu
        # Only positive numbers
        assert((array > 0).all())
        check_second_order_unary(array, cbrt, grad_grad_op)


@xfail_when_nonstandard_decimal_separator
def test_rsqrt():
    def rsqrt(x):
        return nd.rsqrt(x)

    def grad_grad_op(x):
        return 3/(4 * nd.sqrt(x**5))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = sigma * array + mu
        # Only positive numbers
        assert((array > 0).all())
        check_second_order_unary(array, rsqrt, grad_grad_op)


@xfail_when_nonstandard_decimal_separator
def test_rcbrt():
    def rcbrt(x):
        return nd.rcbrt(x)

    def grad_grad_op(x):
        return 4/(9 * nd.cbrt(x**7))

    sigma = random.randint(25, 100)
    mu = random.randint(500, 1000)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        array = sigma * array + mu
        # Only positive numbers
        assert((array > 0).all())
        check_second_order_unary(array, rcbrt, grad_grad_op)


def check_second_order_unary(x, op, grad_grad_op, rtol=None, atol=None):
    check_nth_order_unary(x, op, grad_grad_op, 2, rtol, atol)


def check_nth_order_unary(x, op, grad_ops, orders, rtol=None, atol=None):
    """Assert n-th order autograd gradient against expected gradient.

    Multiple order of gradients can be checked by passing list of
    function computing the particular order gradient and passing the
    corresponding list of order.

    Note
    ----
    1. Orders should always be monotonically increasing.
    2. Elements of grads_ops should correspond to elements of orders
    i.e. grads_op = [grad_op, grad_grad_grad_op] should be passed with
         orders = [1, 3]

    Parameters
    ----------
    x : mxnet.NDArray
        Input Array.
    op : Callable
        Operation to perform on Input Array.
    grad_ops : Callable or List of Callable
        Function to compute and assert gradient of given order.
    orders : int or List of int
        Order/s to assert expected and computed gradients.

    Returns
    -------
    None

    """
    if isinstance(orders, int):
        orders = [orders]
        grad_ops = [grad_ops]

    assert all(i < j for i, j in zip(orders[0:-1], orders[1:])), \
        "orders should be monotonically increasing"
    assert len(set(orders)) == len(orders), \
        "orders should have unique elements"
    highest_order = max(orders)

    x = nd.array(x)
    x.attach_grad()

    expected_grads = [grad_op(x) for grad_op in grad_ops]
    computed_grads = []
    head_grads = []


    # Perform compute.
    with autograd.record():
        y = op(x)
        for current_order in range(1, highest_order+1):
            head_grad = nd.random.normal(shape=x.shape)
            y = autograd.grad(heads=y, variables=x, head_grads=head_grad,
                              create_graph=True, retain_graph=True)[0]
            if current_order in orders:
                computed_grads.append(y)
            head_grads.append(head_grad)

    # Validate all the gradients.
    for order, grad, computed_grad in \
            zip(orders, expected_grads, computed_grads):
        # Compute expected values.
        expected_grad = grad.asnumpy()
        for head_grad in head_grads[:order]:
            expected_grad *= head_grad.asnumpy()

        assert_almost_equal(
            expected_grad, computed_grad.asnumpy(), rtol=rtol, atol=atol)

@with_seed()
def test_elemwise_sub():
    def sub(inputs):
        return nd.elemwise_sub(inputs[0], inputs[1])
    def grad_op(inputs):
        return  [nd.ones_like(inputs[0]), nd.negative(nd.ones_like(inputs[1]))]
    def grad_grad_op(inputs):
        return  [nd.zeros_like(inputs[0]),  nd.zeros_like(inputs[1])]

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        x, y = random_arrays(shape, shape)
        check_nth_order_binary([x, y], sub, [grad_op, grad_grad_op], [1,  2])

@with_seed()
def test_elemwise_mul():
    def mul(inputs):
        return nd.elemwise_mul(inputs[0], inputs[1])
    def grad_op(inputs):
        return  [inputs[1], inputs[0]]
    def grad_grad_op(inputs):
        return [nd.zeros_like(inputs[0]) ,nd.zeros_like(inputs[1])]

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        x, y = random_arrays(shape, shape)
        check_nth_order_binary([x, y], mul, [grad_op, grad_grad_op], [1,  2])

@with_seed()
def test_power():
    def power(inputs):
        return nd.power(inputs[0], inputs[1])

    def grad_op(inputs):
        x, y = inputs
        return  [y * nd.power(x, y - 1), nd.power(x, y) * nd.log(x)]

    def grad_grad_op(inputs):
        x, y = inputs
        return   [y * (y - 1) * nd.power(x, y - 2), nd.power(x, y) * (nd.log(x) ** 2)]

    def grad_grad_grad_op(inputs):
        x, y = inputs
        return   [y * (y - 1) * (y - 2) * nd.power(x, y - 3), nd.power(x, y) * (nd.log(x) ** 3)]

    low = 1.0
    high = 3.0
    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        x = nd.random.uniform(low, high, shape)
        y = nd.random.uniform(low, high, shape)
        check_nth_order_binary([x, y], power, [grad_op, grad_grad_op, grad_grad_grad_op], [1, 2, 3])

#  based on gen_broadcast_data in test_operation.py
def gen_broadcast_shape(idx):
    # Manually set test cases
    binary_op_data_shape = nd.array(
        [[[2, 5, 1, 30, 7], [1, 5, 448, 30, 1]],
         [[10, 49, 1, 77, 17], [10, 1, 2, 1, 17]],
         [[13, 2, 65, 2,  1], [13, 1, 65, 1, 225]],
         [[9, 434, 4, 2, 37], [9, 1, 4, 1, 37]],
         [[2, 52, 1, 4, 1], [1, 52, 60, 1, 37]],
         [[1, 23, 7, 122, 50], [2, 1, 7, 1, 50]],
         [[1, 17, 1, 5, 1], [22, 1, 2, 1, 28]],
         [[29, 1, 2, 1, 8], [29, 22, 1, 130, 1]],
         [[2, 36, 1, 427, 3], [1, 36, 11, 427, 1]],
         [[1, 2, 1, 100, 7], [1, 2, 448, 100, 1]],
         [[1, 2, 495, 77, 7], [1, 2, 1, 1, 7]],
         [[1, 43, 65, 2, 1], [1, 43, 65, 1, 225]],
         [[1, 92, 434, 2, 2], [1, 92, 1, 2, 2]],
         [[1, 92, 1, 4, 1], [1, 92, 134, 1, 17]],
         [[1, 53, 2, 122, 143], [1, 1, 2, 1, 143]],
         [[1, 179, 1, 87, 17], [1, 179, 1, 1, 17]],
         [[1, 1, 17, 5, 1], [1, 22, 1, 1, 28]],
         [[1, 2, 1, 1, 8], [1, 2, 52, 430, 1]],
         [[1, 163, 1, 22, 3], [1, 163, 116, 22, 1]],
         [[1, 1, 44, 30, 7], [1, 1, 44, 30, 1]],
         [[1, 1, 1, 1, 28], [1, 127, 1, 5, 28]],
         [[1, 2, 394, 38, 1], [1, 2, 394, 38, 16]],
         [[1, 10, 49, 77, 17], [1, 1, 1, 1, 17]],
         [[1, 431, 6, 2, 225], [1, 1, 6, 2, 225]],
         [[1, 15, 1, 28, 1], [1, 15, 1, 28, 463]], [[1, 129, 2, 48, 96], [1, 129, 2, 1, 1]],
         [[1, 1, 403, 17, 2], [1, 44, 403, 17, 2]],
         [[1, 1, 65, 2, 22], [1, 1, 65, 1, 1]],
         [[1, 24, 103, 17, 18], [1, 24, 1, 1, 1]],
         [[1, 1, 1, 1, 2], [1, 24, 194, 50, 1]],
         [[1, 1, 107, 84, 9], [1, 1, 1, 1, 1]]])
    if idx < binary_op_data_shape.shape[0]:
        l_shape = binary_op_data_shape[idx][0]
        r_shape = binary_op_data_shape[idx][1]
    else:
        # Generate random data that has ndim between 1-7 and all the shape dims between 1-5
        ndim = nd.random.randint(1, 6)
        shape = nd.random.randint(1, 6, size=(ndim,))
        l_same_dim = nd.random.randint(0, 5)
        r_same_dim = nd.random.randint(0, 5)
        l_axis_flags = nd.random.randint(0, 2, size=ndim)
        r_axis_flags = nd.random.randint(0, 2, size=ndim)
        if l_same_dim == 4:
            l_axis_flags = nd.ones(ndim)
        if r_same_dim == 4:
            r_axis_flags = nd.ones(ndim)
        l_shape = shape.copy()
        r_shape = shape.copy()
        l_shape[nd.where(l_axis_flags == 0)] = 1
        r_shape[nd.where(r_axis_flags == 0)] = 1
    return tuple(l_shape.asnumpy().astype(int)), tuple(r_shape.asnumpy().astype(int))

# from test_operation.py
def reduce_op(shape, x):
    if shape == x.shape:
        return x
    keepdims_shape = list(x.shape)
    for i in range(len(shape)):
        if x.shape[i] != shape[i]:
            keepdims_shape[i] = 1
            x = nd.sum(x, axis=i).reshape(keepdims_shape)
    return x

@with_seed()
def test_broadcast_power():
    def broadcast_power(inputs):
        return nd.broadcast_power(inputs[0], inputs[1])

    def unreduced_grad_op(inputs):
        x, y = inputs
        return [y * nd.broadcast_power(x, y - 1), nd.broadcast_power(x, y) * nd.log(x)]

    def unreduced_grad_grad_op(inputs):
        x, y = inputs
        return   [y * (y - 1) * nd.broadcast_power(x, y - 2), nd.broadcast_power(x, y) * (nd.log(x) ** 2)]

    def unreduced_grad_grad_grad_op(inputs):
        x, y = inputs
        return   [y * (y - 1) * (y - 2) * nd.broadcast_power(x, y - 3), nd.broadcast_power(x, y) * (nd.log(x) ** 3)]

    low = 1.0
    high = 3.0
    for dim in range(1, 5):
        x_shape, y_shape = gen_broadcast_shape(dim)
        x = nd.random.uniform(low, high, x_shape)
        y = nd.random.uniform(low, high, y_shape)

        check_nth_order_binary([x, y], broadcast_power, [unreduced_grad_op, unreduced_grad_grad_op,
            unreduced_grad_grad_grad_op], [1, 2, 3], True, rtol=1e-3, atol=1e-5)

def autograd_grad_ex(heads, variables, head_grads=None, retain_graph=None, create_graph=False,
            train_mode=True):
    """ If some variables don't in the path of computing heads, we set the heads grad of them to zero
    instead of throwing exceptions.

    The autograd.grad requires user knows which variables involved to compute the heads grad of them.
    That's fine for first order grad, but for higher order grad, the variables used to compute the heads,
    may not used to compute their higher order grad. It's impossible to ask user to know
    the formulas of every order grad.

    E.g. we use such code to compute 2-nd order gradient:
      with autograd.record():
          z = op(x, y)
          head_grad = nd.ones_like(z)
          dz_dx, _  = autograd.grad(heads=z, variables=[x, y], head_grads=nd.ones_like(z),
                              create_graph=True, retain_graph=True)
          d2z_d2x, _  = autograd.grad(heads=dz_dx, variables=[x, y], head_grads=nd.ones_like(dz_dx),
                              create_graph=True, retain_graph=True)
    If z = x * y, because d2z_d2x = 0, MXNET will report the input is unreachable from the output.
    But it seems in that case MXNET returns zeros is more reasonable.
    """
    # xxx: only consider one head currently
    argument_names =  autograd.get_symbol(heads).list_arguments()

    # XXX: in some cases, a variable may has more than one outputs, we need a other way ot get  the name of various.
    # But in the unittest, it is fine
    variable_names = [autograd.get_symbol(variable).list_outputs()[0] for variable in variables]
    involved_variable_indexes = []
    involved_variables = []
    for i in range(0, len(variables)):
        if variable_names[i] in argument_names:
            involved_variables.append(variables[i])
            involved_variable_indexes.append(i)

    if involved_variables:
        partial_grads = autograd.grad(heads, involved_variables, head_grads, retain_graph, create_graph, train_mode)
    else:
        partial_grads = []

    grads = []
    partial_grads_index = 0
    for i in range(0, len(variables)):
       if i in involved_variable_indexes:
           grads.append(partial_grads[partial_grads_index])
           partial_grads_index += 1
       else:
           grads.append(nd.zeros_like(variables[i]))
    return grads


def check_nth_order_binary(inputs, op, grad_ops, orders, broadcast_op = False, rtol=None, atol=None):
    """Assert n-th order autograd gradient against expected gradient.

    Multiple order of gradients can be checked by passing list of
    function computing the particular order gradient and passing the corresponding list of order.
    Note
    ----
    1. Orders should always be monotonically increasing.
    2. Elements of grads_ops should correspond to elements of orders
    i.e. grads_op = [grad_op, grad_grad_grad_op] should be passed with
         orders = [1, 3]

    Parameters
    ----------
    inputs : tuple of mxnet.NDArray (x, y)
        Input Array.
    op : Callable (x,y) -> z
        Operation to perform on Input Array.
    grad_ops : Callable or List of Callable
        Function (x,y) -> (n_grad_x, n_grad_y) to compute and assert gradient of given order.
    orders : int or List of int
        Order/s to assert expected and computed gradients.

    Returns
    -------
    None

    """
    if isinstance(orders, int):
        orders = [orders]
        grad_ops = [grad_ops]

    assert all(i < j for i, j in zip(orders[0:-1], orders[1:])), \
        "orders should be monotonically increasing"
    assert len(set(orders)) == len(orders), \
        "orders should have unique elements"
    highest_order = max(orders)

    inputs = [nd.array(input) for input in inputs]
    for input in inputs:
        input.attach_grad()

    expected_grads = [grad_op(inputs) for grad_op in grad_ops]
    computed_grads = []
    head_grads = [[]]

    # Perform compute.
    with autograd.record():
        z = op(inputs)
        heads = [z for _ in inputs]
        for current_order in range(1, highest_order+1):
            grads = []
            new_head_grads = []
            new_heads = []
            for i in range(0, len(heads)):
                head = heads[i]
                head_grad = nd.random.normal(shape=head.shape)
                new_head_grads.append(head_grad)
                grads.append(autograd_grad_ex(heads=head, variables=inputs, head_grads=head_grad,
                                              create_graph=True, retain_graph=True)[i])
                # If we only use once auto grad with head_grads = head_grad in every iteration,
                # in the i-th iteration, we use head = derivative_(i-1) * head_grad_(i-1)
                # but in the expected computed, we use head = derivative_(i-1)
                new_heads.append(autograd_grad_ex(heads=head, variables=inputs, head_grads=nd.ones_like(head),
                                              create_graph=True, retain_graph=True)[i])
            heads = new_heads
            if current_order in orders:
                computed_grads.append(grads)
            head_grads.append(new_head_grads)

    # Validate all the gradients.
    for order, grad_list, computed_grad_list in \
            zip(orders, expected_grads, computed_grads):
        # Compute expected values.
        # keep as numpy value and use dot mul
        expected_grad_list = [grad for grad in grad_list]
        for expected_grad, head_grad, computed_grad, input in zip(expected_grad_list, head_grads[order], computed_grad_list, inputs):
            if broadcast_op:
                expected_grad = reduce_op(input.shape, expected_grad * head_grad)
            else:
                expected_grad *= head_grad
            assert_almost_equal(expected_grad.asnumpy(), computed_grad.asnumpy(), rtol=rtol, atol=atol)

def arange_shape_like(y):
    shape = y.shape
    nelems = reduce(mul, shape)
    x = nd.arange(nelems).reshape(shape)
    return x


class NDArrayGenerator(object):
    def __init__(self, dim, startdim=1):
        self.dim = dim
        self.curdim = startdim

    def __iter__(self):
        return self

    @staticmethod
    def gen(dimensions):
        shape = rand_shape_nd(dimensions, 4)
        nelems = reduce(mul, shape)
        x = nd.arange(nelems).reshape(shape)
        return x

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.curdim > self.dim:
            raise StopIteration
        x = NDArrayGenerator.gen(self.curdim)
        self.curdim += 1
        return x


def flatten2d_right(x):
    s_0 = x.shape[0]
    s_1 = reduce(mul, x.shape[1:])
    return x.reshape((s_0, s_1))


def flatten2d_left(x):
    s_0 = reduce(mul, x.shape[:-1])
    s_1 = x.shape[-1]
    return x.reshape((s_0, s_1))


def test_dense_backward_flatten():
    print("2nd order gradient for Fully Connected, flatten=True")
    for x in NDArrayGenerator(4,2):
        hidden = random.randrange(1, 4)
        net = gluon.nn.Sequential()
        net.add(gluon.nn.Dense(hidden, flatten=True))
        net.initialize(mxnet.initializer.Constant(.5))
        x.attach_grad()
        with autograd.record():
            y = net.forward(x)
            o_y = arange_shape_like(y)  # head gradient of y
            params = [p.data() for p in net.collect_params().values()]
            w = params[0]
            b = params[1]
            print("Checking y ({}) = x({}) * w^T({}) + b({})".format(y.shape, x.shape, w.shape, b.shape))
            x_grad = autograd.grad(heads=y, variables=x, head_grads=o_y,
                                   create_graph=True, retain_graph=True)[0]
            o_x_grad = arange_shape_like(x_grad)
            w_grad_grad = autograd.grad(heads=x_grad, variables=w,
                                        head_grads=o_x_grad, create_graph=False)[0]
            w_grad = autograd.grad(heads=y, variables=w, head_grads=o_y,
                                   create_graph=True, retain_graph=True)[0]
            o_w_grad = arange_shape_like(w_grad)
            x_grad_grad = autograd.grad(heads=w_grad, variables=x,
                                        head_grads=o_w_grad, create_graph=False)[0]
        # Expected results
        w_grad_e = nd.dot(o_y, x, transpose_a=True)
        w_grad_grad_e = nd.dot(o_y, o_x_grad, transpose_a=True)
        x_grad_e = nd.dot(o_y, w)
        x_grad_grad_e = nd.dot(o_y, o_w_grad)
        assert w_grad.shape == w.shape
        assert w_grad_grad.shape == w.shape
        assert x_grad.shape == x.shape
        assert x_grad_grad.shape == x.shape
        w_grad_check = same(flatten2d_right(w_grad), flatten2d_right(w_grad_e))
        w_grad_grad_check = same(flatten2d_right(w_grad_grad), flatten2d_right(w_grad_grad_e))
        x_grad_check = same(flatten2d_right(x_grad), flatten2d_right(x_grad_e))
        x_grad_grad_check = same(flatten2d_right(x_grad_grad), flatten2d_right(x_grad_grad_e))
        assert x_grad_check
        assert w_grad_check
        assert x_grad_grad_check
        assert w_grad_grad_check

def test_dense_backward_no_flatten():
    print("2nd order gradient for Fully Connected, flatten=False")
    for x in NDArrayGenerator(5,3):
        hidden = random.randrange(1, 4)
        net = gluon.nn.Sequential()
        net.add(gluon.nn.Dense(hidden, flatten=False))
        net.initialize(mxnet.initializer.Constant(.5))
        x.attach_grad()
        with autograd.record():
            y = net.forward(x)
            o_y = arange_shape_like(y)  # head gradient of y
            params = [p.data() for p in net.collect_params().values()]
            w = params[0]
            b = params[1]
            print("Checking y ({}) = x({}) * w^T({}) + b({})".format(y.shape, x.shape, w.shape, b.shape))
            x_grad = autograd.grad(heads=y, variables=x, head_grads=o_y,
                                   create_graph=True, retain_graph=True)[0]
            o_x_grad = arange_shape_like(x_grad)
            w_grad_grad = autograd.grad(heads=x_grad, variables=w,
                                        head_grads=o_x_grad, create_graph=False)[0]
            w_grad = autograd.grad(heads=y, variables=w, head_grads=o_y,
                                   create_graph=True, retain_graph=True)[0]
            o_w_grad = arange_shape_like(w_grad)
            x_grad_grad = autograd.grad(heads=w_grad, variables=x,
                                        head_grads=o_w_grad, create_graph=False)[0]
        # Expected results
        o_y = flatten2d_left(o_y)
        x = flatten2d_left(x)
        o_x_grad = flatten2d_left(o_x_grad)
        o_w_grad = flatten2d_left(o_w_grad)
        w_grad_e = nd.dot(o_y, x, transpose_a=True)
        w_grad_grad_e = nd.dot(o_y, o_x_grad, transpose_a=True)
        x_grad_e = nd.dot(o_y, w)
        x_grad_grad_e = nd.dot(o_y, o_w_grad)
        w_grad_check = same(flatten2d_left(w_grad), flatten2d_left(w_grad_e))
        w_grad_grad_check = same(flatten2d_left(w_grad_grad), flatten2d_left(w_grad_grad_e))
        x_grad_check = same(flatten2d_left(x_grad), flatten2d_left(x_grad_e))
        x_grad_grad_check = same(flatten2d_left(x_grad_grad), flatten2d_left(x_grad_grad_e))
        assert x_grad_check
        assert w_grad_check
        assert x_grad_grad_check
        assert w_grad_grad_check

