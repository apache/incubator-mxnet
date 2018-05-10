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
MKL-DNN related test cases
"""

import mxnet as mx
import numpy as np
import sys,os,logging
import random
from mxnet import gluon
from mxnet.gluon import nn
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '../unittest/'))
from common import setup_module, with_seed
from nose.tools import raises
from mxnet.test_utils import assert_almost_equal


def test_mkldnn_install():
    """
    This test will verify that MXNet is built/installed correctly when
    compiled with Intel MKL-DNN library. The method will try to import
    the mxnet module and see if the mkldnn library is mapped to this
    process's address space.
    """
    logging.basicConfig(level=logging.INFO)

    if not sys.platform.startswith('linux'):
        logging.info("Bypass mkldnn install test for non-Linux OS")
        return

    try:
        #pylint: disable=unused-variable
        import mxnet as mx
    except (ImportError, OSError) as e:
        assert 0, "Import mxnet error: %s. Please double check your build/" \
            "install steps or environment variable settings" % str(e)

    pid = os.getpid()
    rc = os.system("cat /proc/" + str(pid) +
                   "/maps | grep libmkldnn > /dev/null")

    if rc == 0:
        logging.info("MXNet is built/installed correctly with MKL-DNN")
    else:
        assert 0, "MXNet is built/installed incorrectly with MKL-DNN, please " \
            "double check your build/install steps or environment " \
            "variable settings"


def test_mkldnn_model():
    """
    This test will run a sample model for couple of iterations.
    """

    import mxnet as mx
    model = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data",
                         "test_mkldnn_test_mkldnn_model_model1.json")
    shape = (32, 3, 300, 300)
    ctx = mx.cpu()

    sym = mx.sym.load(model)
    args = sym.list_arguments()
    shapes = sym.infer_shape(data=shape)

    def get_tensors(args, shapes, ctx):
        return {x: mx.nd.ones(y, ctx) for x, y in zip(args, shapes)}

    inputs = get_tensors(args, shapes[0], ctx)
    grads = get_tensors(args, shapes[0], ctx)

    try:
        exe = sym.bind(ctx, inputs, args_grad=grads)
        for _ in range(2):
            exe.forward(is_train=True)
            for y in exe.outputs:
                y.wait_to_read()
            exe.backward()
            for y in exe.grad_arrays:
                y.wait_to_read()
    except:  # pylint: disable=bare-except
        assert 0, "test_mkldnn_model exception in bind and execution"


def test_mkldnn_engine_threading():
    """
    This test will trigger mkldnn engine on different thread of execution.
    The test will first kickoff simple model calculation, and then uses a
    gluon data iterator to trigger different thread context, and executes
    the model on this new thread.
    """

    import mxnet as mx
    from mxnet import gluon, nd

    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation=None))
    net.collect_params().initialize(ctx=mx.cpu())
    class Dummy(gluon.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, key):
            return key, np.ones((3, 224, 224)), np.ones((10, ))

    loader = gluon.data.DataLoader(Dummy(), batch_size=2, num_workers=1)

    X = (32, 3, 32, 32)
    # trigger mkldnn execution thread
    y = net(nd.array(np.ones(X))).asnumpy()

    # Use Gluon dataloader to trigger different thread.
    # below line triggers different execution thread
    for _ in loader:
        y = net(nd.array(np.ones(X))).asnumpy()
        # output should have 0.3376348
        assert_almost_equal(y[0, 0, 0, 0], 0.3376348)
        break


def test_mkldnn_ndarray_slice():
    """
    This test will trigger gluon computation on mkldnn with ndarray slice
    """

    import mxnet as mx
    from mxnet import gluon
    ctx = mx.cpu()
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=32, kernel_size=3, activation=None))
        net.collect_params().initialize(ctx=ctx)
        x = mx.nd.array(np.ones([32, 3, 224, 224]), ctx)
        y = net(x)

        # trigger computation on ndarray slice
        assert_almost_equal(y[0].asnumpy()[0, 0, 0], 0.3376348)


def check_layer_forward(net, x):
    x_hybrid = x.copy()
    x.attach_grad()
    x_hybrid.attach_grad()
    net.collect_params().initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()
    net.hybridize()
    with mx.autograd.record():
        out2 = net(x_hybrid)
    out2.backward()
    mx.test_utils.assert_almost_equal(x.grad.asnumpy(), x_hybrid.grad.asnumpy(), rtol=1e-5, atol=1e-6)
    mx.test_utils.assert_almost_equal(out1.asnumpy(), out2.asnumpy(), rtol=1e-5, atol=1e-6)


@with_seed()
def test_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 448, 112))
            out = self.conv0(x_reshape)
            return out
    x = mx.nd.random.uniform(shape=(32, 3, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_reshape_conv_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 448, 112))
            y = self.conv0(x_reshape)
            y_reshape = y.reshape((0, 0, 223, 220))
            out = self.conv1(y_reshape)
            return out
    x = mx.nd.random.uniform(shape=(32, 3, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 2, 0, 0), end=(32, 5, 224, 224))
            out = self.conv0(x_slice)
            return out
    x = mx.nd.random.uniform(shape=(32, 6, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_conv_slice_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 2, 0, 0), end=(32, 5, 224, 224))
            y = self.conv0(x_slice)
            y_slice = y.slice(begin=(0, 32, 0, 0), end=(32, 64, 222, 222))
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(32, 6, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_conv_reshape_conv():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=(0, 0, 1, 1), end=(32, 3, 225, 225))
            y = self.conv0(x_slice)
            y_reshape = y.reshape((0, 0, 444, 111))
            out = self.conv1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(32, 3, 299, 299))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_reshape_conv_slice_conv():
    """
    This test will test gluon Conv2d computation on mkldnn with ndarray reshape and slice
    """
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(64, (3, 3))
                self.conv1 = nn.Conv2D(256, (3, 3))

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((0, 0, 448, 112))
            y = self.conv0(x_reshape)
            y_slice = y.slice(begin=(0, 32, 0, 0), end=(32, 64, 446, 110))
            out = self.conv1(y_slice)
            return out
    x = mx.nd.random.uniform(shape=(32, 6, 224, 224))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 600, -1))
            out = self.dense0(x_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]),
                              end=tuple(self.slice[1]))
            out = self.dense0(x_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    slice = [[0, 64, 50, 0], [8, 128, 300, 300]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_slice_dense_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = 50
                channel1 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.dense0(x_slice)
            y_slice = y.slice(begin=(4, 0), end=(-1, 10))
            out = self.dense1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    slice = [[0, 64, 50, 0], [8, 128, 300, 300]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_reshape_dense_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = random.randint(1, 1000)
                channel1 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 600, -1))
            y = self.dense0(x_reshape)
            y_reshape = y.reshape((1, -1))
            out = self.dense1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_slice_dense_reshape_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = random.randint(1, 1000)
                channel1 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_slice = x.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.dense0(x_slice)
            y_reshape = y.reshape((1, -1))
            out = self.dense1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    slice = [[0, 64, 50, 0], [8, 128, 300, 300]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_reshape_dense_slice_dense():
    class Net(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                channel0 = 800
                channel1 = random.randint(1, 1000)
                self.dense0 = nn.Dense(channel0)
                self.dense1 = nn.Dense(channel1)

        def hybrid_forward(self, F, x):
            x_reshape = x.reshape((8, 64, 600, -1))
            y = self.dense0(x_reshape)
            y_slice = y.slice(begin=(0, 500), end=(8, 628))
            out = self.dense1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 300, 300))
    net = Net()
    check_layer_forward(net, x)


@with_seed()
def test_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm()
                self.reshape = shape

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape)
            out = self.bn0(x_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    shape = (32, 512, 128, -1)
    net = Net(shape)
    check_layer_forward(net, x)


@with_seed()
def test_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(3)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0]),
                              end=tuple(self.slice[1]))
            out = self.bn0(x_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 64, 50, 0], [8, 128, 256, 256]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_slice_batchnorm_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(3)
                self.bn1 = nn.BatchNorm(1)
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0][0]), end=tuple(self.slice[0][1]))
            y = self.bn0(x_slice)
            y_slice = y.slice(begin=tuple(self.slice[1][0]), end=tuple(self.slice[1][1]))
            out = self.bn1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[[0, 64, 50, 0], [8, 128, 200, 256]], [[4, 50, 0, 128], [7, -1, -1, -1]]]
    net = Net(slice)
    check_layer_forward(net, x)


@with_seed()
def test_reshape_batchnorm_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(0)
                self.bn1 = nn.BatchNorm(2)
                self.reshape = shape

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape[0])
            y = self.bn0(x_reshape)
            y_reshape = y.reshape(self.reshape[1])
            out = self.bn1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 512))
    shape = [(8, 256, 128, -1), (32, 128, 512, -1)]
    net = Net(shape)
    check_layer_forward(net, x)


@with_seed()
def test_slice_batchnorm_reshape_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(0)
                self.bn1 = nn.BatchNorm(2)
                self.reshape = shape
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_slice = x_in.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            y = self.bn0(x_slice)
            y_reshape = y.reshape(self.reshape)
            out = self.bn1(y_reshape)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 64, 50, 0], [8, 128, 200, 256]]
    shape = (1, 128, 256, -1)
    net = Net(shape, slice)
    check_layer_forward(net, x)


@with_seed()
def test_reshape_batchnorm_slice_batchnorm():
    class Net(gluon.HybridBlock):
        def __init__(self, shape, slice, **kwargs):
            super(Net, self).__init__(**kwargs)
            with self.name_scope():
                self.conv0 = nn.Conv2D(128, (1, 1))
                self.bn0 = nn.BatchNorm(2)
                self.bn1 = nn.BatchNorm(0)
                self.reshape = shape
                self.slice = slice

        def hybrid_forward(self, F, x):
            x_in = self.conv0(x)
            x_reshape = x_in.reshape(self.reshape)
            y = self.bn0(x_reshape)
            y_slice = y.slice(begin=tuple(self.slice[0]), end=tuple(self.slice[1]))
            out = self.bn1(y_slice)
            return out

    x = mx.nd.random.uniform(shape=(16, 128, 256, 256))
    slice = [[0, 0, 50, 0], [8, 1, -1, 100]]
    shape = (128, 1, 256, -1)
    net = Net(shape, slice)
    check_layer_forward(net, x)


if __name__ == '__main__':
    test_mkldnn_install()
