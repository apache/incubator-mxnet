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
from mxnet import nd, autograd
from mxnet.test_utils import assert_almost_equal, random_arrays, rand_shape_nd
from common import with_seed


@with_seed()
def test_sin():
    def sin(x):
        return nd.sin(x)

    def grad_grad_op(x):
        return -nd.sin(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sin, grad_grad_op)


@with_seed()
def test_cos():
    def cos(x):
        return nd.cos(x)

    def grad_grad_op(x):
        return -nd.cos(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, cos, grad_grad_op)


@with_seed()
def test_relu():
    def relu(x):
        return nd.relu(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, relu, grad_grad_op)


@with_seed()
def test_log():
    def log(x):
        return nd.log(x)

    def grad_grad_op(x):
        return -1/(x**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log, grad_grad_op)


@with_seed()
def test_log2():
    def log2(x):
        return nd.log2(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(2))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log2, grad_grad_op)


@with_seed()
def test_log10():
    def log10(x):
        return nd.log10(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(10))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log10, grad_grad_op)


@with_seed()
def test_reciprocal():
    def reciprocal(x):
        return nd.reciprocal(x)

    def grad_grad_op(x):
        return 2 / x**3

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, reciprocal, grad_grad_op)


@with_seed()
def test_abs():
    def abs(x):
        return nd.abs(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, abs, grad_grad_op)


@with_seed()
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


@with_seed()
def test_elemwise_mul():
    def elemwise_mul(x, y):
        return nd.elemwise_mul(x, y)

    def grad_grad_op(x, y):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array_x, array_y = random_arrays(shape, shape)
        inputs = (array_x, array_y)
        grad_grad_ops = (grad_grad_op, grad_grad_op)
        check_second_order_binary(inputs, elemwise_mul, grad_grad_ops)


@with_seed()
def test_elemwise_mul_same_var():
    def fourth_power(x):
        y = nd.elemwise_mul(x, x)
        return nd.elemwise_mul(y, y)

    def grad_grad_op(x):
        return 12 * (x**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array_x = random_arrays(shape)
        check_second_order_unary(array_x, fourth_power, grad_grad_op)


@with_seed()
def test_elemwise_div():
    def elemwise_div(x, y):
        return nd.elemwise_div(x, y)

    def grad_grad_op_x(x, y):
        return nd.zeros_like(x)

    def grad_grad_op_y(x, y):
        return (x * 2)/(y**3)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array_x, array_y = random_arrays(shape, shape)
        inputs = (array_x, array_y)
        grad_grad_ops = (grad_grad_op_x, grad_grad_op_y)
        check_second_order_binary(inputs, elemwise_div, grad_grad_ops)


@with_seed()
def test_elemwise_div_same_var():
    def f(x):
        # f = x / (x^2 + 1)
        return nd.elemwise_div(x, nd.elemwise_mul(x, x) + 1)

    def grad_grad_op(x):
        # d^2f    -2*x*(-x^2 + 3)
        # ---- = ----------------
        # dx^2     (x^2 + 1)^3
        return (-2*x*(-x**2 + 3))/((x**2+1)**3)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        x = random_arrays(shape)
        x += 10
        check_second_order_unary(x, f, grad_grad_op)


def check_second_order_unary(x, op, grad_grad_op):
    x = nd.array(x)
    grad_grad_x = grad_grad_op(x)
    x.attach_grad()

    # Manual head_grads.
    y_grad = nd.random.normal(shape=x.shape)
    head_grad_grads = nd.random.normal(shape=x.shape)

    # Perform compute.
    with autograd.record():
        y = op(x)
        x_grad = autograd.grad(heads=y, variables=x, head_grads=y_grad,
                               create_graph=True, retain_graph=True)[0]
    x_grad.backward(head_grad_grads)

    # Compute expected values.
    expected_grad_grad = grad_grad_x.asnumpy() * head_grad_grads.asnumpy() * \
        y_grad.asnumpy()

    # Validate the gradients.
    assert_almost_equal(expected_grad_grad, x.grad.asnumpy())


def check_second_order_binary(inputs, binary_op, grad_grad_ops):
    assert (len(inputs) == len(grad_grad_ops))

    x, y = map(nd.array, inputs)
    grad_grad_x, grad_grad_y = [gg_op(x, y) for gg_op in grad_grad_ops]
    x.attach_grad()
    y.attach_grad()

    # Manual head_grads.
    z_grad = nd.random.normal(shape=x.shape)
    head_grad_grads = nd.random.normal(shape=x.shape)

    # Perform compute.
    with autograd.record():
        z = binary_op(x, y)
        x_grad, y_grad = autograd.grad(heads=z, variables=[x, y],
                                       head_grads=z_grad,
                                       create_graph=True, retain_graph=True)

    # Compute expected values.
    expected_grad_grad_x = grad_grad_x.asnumpy() * head_grad_grads.asnumpy() *\
        z_grad.asnumpy()
    expected_grad_grad_y = grad_grad_y.asnumpy() * head_grad_grads.asnumpy() *\
        z_grad.asnumpy()

    # Validate the gradients.
    x_grad.backward(head_grad_grads, retain_graph=True)
    assert_almost_equal(expected_grad_grad_x, x.grad.asnumpy())

    # TODO Figure what is happening.
    # Somehow gradients of y are affected with this backward.
    # Only in case of `test_elemwise_mul`
    # As y.grad == head_grad_grads * z_grad
    # Temporary Hack: Reset them.
    nd.zeros_like(y.grad, out=y.grad)

    y_grad.backward(head_grad_grads)
    assert_almost_equal(expected_grad_grad_y, y.grad.asnumpy())


if __name__ == '__main__':
    import nose
    nose.runmodule()
