import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn

class ReluOp(mx.operator.CustomOp):

    guided_backprop = True

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = nd.maximum(x, nd.zeros_like(x))
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if ReluOp.guided_backprop:
            y = out_data[0]
            dy = out_grad[0]
            dy_positives = nd.maximum(dy, nd.zeros_like(dy))
            y_ones = y.__gt__(0)
            dx = dy_positives * y_ones
            self.assign(in_grad[0], req[0], dx)
        else:
            x = in_data[0]
            x_gt_zero = x.__gt__(0)
            dx = out_grad[0] * x_gt_zero
            self.assign(in_grad[0], req[0], dx)

@mx.operator.register("relu")
class ReluProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ReluProp, self).__init__(True)

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        output_shape = data_shape
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return ReluOp()  

class Activation(mx.gluon.HybridBlock):
    @staticmethod
    def set_guided_backprop(mode=False):
        ReluOp.guided_backprop = mode

    def __init__(self, act_type, **kwargs):
        assert act_type == 'relu'
        super(Activation, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.Custom(x, op_type='relu')

class Conv2D(mx.gluon.HybridBlock):

    conv_output = None
    capture_layer_name = None

    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.conv = nn.Conv2D(channels, kernel_size, strides=strides, padding=padding,
                             dilation=dilation, groups=groups, layout=layout,
                             activation=activation, use_bias=use_bias, weight_initializer=weight_initializer,
                             bias_initializer=bias_initializer, in_channels=in_channels)

    def hybrid_forward(self, F, x):
        out = self.conv(x)
        name = self._prefix[:-1]
        if name == Conv2D.capture_layer_name:
            out.attach_grad()
            Conv2D.conv_output = out
        return out

def set_capture_layer_name(name):
    Conv2D.capture_layer_name = name

import mxnet as mx
from mxnet import autograd

import numpy as np
import cv2

from cnnviz.layers import Conv2D, Activation

def _get_grad(net, image, class_id=None, conv_layer_name=None, image_grad=False):

    if image_grad:
        image.attach_grad()
        Conv2D.capture_layer_name = None
        Activation.set_guided_backprop(True)
    else:
        # Tell convviz.Conv2D which layer's output and gradient needs to be recorded
        Conv2D.capture_layer_name = conv_layer_name
        Activation.set_guided_backprop(False)
    
    # Run the network
    with autograd.record(train_mode=False):
        out = net(image)
    
    # If user didn't provide a class id, we'll use the class that the network predicted
    if class_id == None:
        model_output = out.asnumpy()
        target_class = np.argmax(model_output)

    # Create a one-hot target with class_id and backprop with the created target
    one_hot_target = mx.nd.one_hot(mx.nd.array([target_class]), 1000)
    out.backward(one_hot_target, train_mode=False)

    if image_grad:
        return image.grad[0].asnumpy()
    else:
        # Return the recorded convolution output and gradient
        conv_out = Conv2D.conv_output
        return conv_out[0].asnumpy(), conv_out.grad[0].asnumpy()

def get_conv_out_grad(net, image, class_id=None, conv_layer_name=None):
    return _get_grad(net, image, class_id, conv_layer_name, image_grad=False)

def get_image_grad(net, image, class_id=None):
    return _get_grad(net, image, class_id, image_grad=True)

def grad_to_image(gradient):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    gradient = gradient[..., ::-1]
    return gradient

def get_cam(imggrad, conv_out):
    weights = np.mean(imggrad, axis=(1, 2))

    cam = np.ones(conv_out.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_out[i, :, :]

    cam = cv2.resize(cam, (imggrad.shape[1], imggrad.shape[2]))
    cam = np.maximum(cam, 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam)) 
    cam = np.uint8(cam * 255)
    return cam

def get_guided_grad_cam(cam, imggrad):
    return np.multiply(cam, imggrad)

def get_img_heatmap(orig_img, activation_map):
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (orig_img.shape[0], orig_img.shape[1]))

    img_heatmap = np.float32(heatmap) + np.float32(orig_img)
    img_heatmap = img_heatmap / np.max(img_heatmap)

    return img_heatmap

def to_grayscale(cv2im):
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def visualize_class_activation(net, preprocessed_img, orig_img, conv_layer_name):
    # Returns grad-cam heatmap, guided grad-cam, guided grad-cam saliency
    imggrad = get_image_grad(net, preprocessed_img)
    conv_out, conv_out_grad = get_conv_out_grad(net, preprocessed_img, conv_layer_name=conv_layer_name)

    cam = get_cam(imggrad, conv_out)
    
    ggcam = get_guided_grad_cam(cam, imggrad)
    img_ggcam = grad_to_image(ggcam)
    
    img_heatmap = get_img_heatmap(orig_img, cam)
    
    ggcam_gray = to_grayscale(ggcam)
    img_ggcam_gray = np.squeeze(grad_to_image(ggcam_gray))
    
    return img_heatmap, img_ggcam, img_ggcam_gray

