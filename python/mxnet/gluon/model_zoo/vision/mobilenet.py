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

# coding: utf-8
# pylint: disable= arguments-differ
"""MobileNet and MobileNetV2, implemented in Gluon."""
__all__ = ['MobileNet', 'MobileNetV2', 'mobilenet1_0', 'mobilenet0_75',
           'mobilenet0_5', 'mobilenet0_25', 'get_mobilenet']

__modify__ = 'dwSun'
__modified_date__ = '18/01/31'

import os

from ... import nn
from ....context import cpu
from ...block import HybridBlock


# Helpers
# pylint: disable= too-many-arguments
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0, num_group=1):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    out.add(nn.Activation('relu'))


def _add_conv_dw(out, dw_channels, channels, stride):
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride, pad=1, num_group=dw_channels)
    _add_conv(out, channels=channels)


class BottleNeck(nn.HybridBlock):
    r"""BottleNeck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    s : int
        strides
    """

    def __init__(self, in_channels, channels, t, s, **kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self.use_shortcut = s == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()

            _add_conv(self.out, in_channels * t)
            _add_conv(self.out, in_channels * t, kernel=3, stride=s, pad=1, num_group=in_channels * t)

            self.out.add(
                nn.Conv2D(channels, 1, use_bias=False),
                nn.BatchNorm(scale=True),
            )

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


# Net
class MobileNet(HybridBlock):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                _add_conv(self.features, channels=int(32 * multiplier), kernel=3, pad=1, stride=2)
                dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2
                               + [256] * 2 + [512] * 6 + [1024]]
                channels = [int(x * multiplier) for x in [64] + [128] * 2 + [256] * 2
                            + [512] * 6 + [1024] * 2]
                strides = [1, 2] * 3 + [1] * 5 + [2, 1]
                for dwc, c, s in zip(dw_channels, channels, strides):
                    _add_conv_dw(self.features, dw_channels=dwc, channels=c, stride=s)
                self.features.add(nn.GlobalAvgPool2D())
                self.features.add(nn.Flatten())

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class MobileNetV2(nn.HybridBlock):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    """

    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                _add_conv(self.features, int(32 * multiplier), kernel=3, stride=2, pad=1)

                in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                                     + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
                channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                                  + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
                ts = [1] + [6] * 16
                strides = [1, 2] * 2 + [1] * 2 + [2] + [1] * 3 + [1] * 3 + [2] + [1] * 3

                for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
                    self.features.add(BottleNeck(in_channels=in_c, channels=c, t=t, s=s))

                last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
                _add_conv(self.features, last_channels)

                self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.HybridSequential(prefix='output_')
            self.output.add(
                nn.Conv2D(classes, 1, use_bias=False, prefix='pred_'),
                nn.Flatten(prefix='flat_')
            )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Constructor
def get_mobilenet(multiplier, pretrained=False, version=1, ctx=cpu(),
                  root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    multiplier : float
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    version : int, default 1
        Model version, 1 for MobileNet, 2 for MobileNetV2
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert version in [1, 2], '1 for MobileNet, 2 for MobileNetV2, others are not supported.'
    if version == 1:
        version_tag = ''
        net = MobileNet(multiplier, **kwargs)
    else:
        version_tag = 'v2_'
        net = MobileNetV2(multiplier, **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        version_suffix = '{0:.2f}'.format(multiplier)
        if version_suffix in ('1.00', '0.50'):
            version_suffix = version_suffix[:-1]
        net.load_params(
            get_model_file('mobilenet%s%s' % (version_tag, version_suffix), root=root), ctx=ctx)
    return net


def mobilenet1_0(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 1.0.

    MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    version : int, default 1
        Model version, 1 for MobileNet, 2 for MobileNetV2
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_mobilenet(1.0, **kwargs)


def mobilenet0_75(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.75.

    MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    version : int, default 1
        Model version, 1 for MobileNet, 2 for MobileNetV2
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_mobilenet(0.75, **kwargs)


def mobilenet0_5(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.5.

    MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    version : int, default 1
        Model version, 1 for MobileNet, 2 for MobileNetV2
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_mobilenet(0.5, **kwargs)


def mobilenet0_25(**kwargs):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper, with width multiplier 0.25.

    MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    version : int, default 1
        Model version, 1 for MobileNet, 2 for MobileNetV2
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    return get_mobilenet(0.25, **kwargs)
