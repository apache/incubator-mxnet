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
# pylint: disable=
"""Dataset container."""
__all__ = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100',
           'ImageRecordDataset', 'ImageFolderDataset', 'ImageListDataset']

import os
import gzip
import tarfile
import struct
import warnings
import numpy as np

from .. import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from .... import ndarray as nd, image, recordio, base
from .... import numpy as _mx_np  # pylint: disable=reimported
from ....util import is_np_array, default_array
from ....base import numeric_types


class MNIST(dataset._DownloadedDataset):
    """MNIST handwritten digits dataset from http://yann.lecun.com/exdb/mnist

    Each sample is an image (in 3D NDArray) with shape (28, 28, 1).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/mnist
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        DEPRECATED FUNCTION ARGUMENTS.
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'mnist'),
                 train=True, transform=None):
        self._train = train
        self._train_data = ('train-images-idx3-ubyte.gz',
                            '6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d')
        self._train_label = ('train-labels-idx1-ubyte.gz',
                             '2a80914081dc54586dbdf242f9805a6b8d2a15fc')
        self._test_data = ('t10k-images-idx3-ubyte.gz',
                           'c3a25af1f52dad7f726cce8cacb138654b760d48')
        self._test_label = ('t10k-labels-idx1-ubyte.gz',
                            '763e7fa3757d93b0cdec073cef058b2004252c17')
        self._namespace = 'mnist'
        super(MNIST, self).__init__(root, transform)

    def _get_data(self):
        if self._train:
            data, label = self._train_data, self._train_label
        else:
            data, label = self._test_data, self._test_label

        namespace = 'gluon/dataset/'+self._namespace
        data_file = download(_get_repo_file_url(namespace, data[0]),
                             path=self._root,
                             sha1_hash=data[1])
        label_file = download(_get_repo_file_url(namespace, label[0]),
                              path=self._root,
                              sha1_hash=label[1])

        with gzip.open(label_file, 'rb') as fin:
            struct.unpack(">II", fin.read(8))
            label = np.frombuffer(fin.read(), dtype=np.uint8).astype(np.int32)
            if is_np_array():
                label = _mx_np.array(label, dtype=label.dtype)

        with gzip.open(data_file, 'rb') as fin:
            struct.unpack(">IIII", fin.read(16))
            data = np.frombuffer(fin.read(), dtype=np.uint8)
            data = data.reshape(len(label), 28, 28, 1)

        array_fn = _mx_np.array if is_np_array() else nd.array
        self._data = array_fn(data, dtype=data.dtype)
        self._label = label


class FashionMNIST(MNIST):
    """A dataset of Zalando's article images consisting of fashion products,
    a drop-in replacement of the original MNIST dataset from
    https://github.com/zalandoresearch/fashion-mnist

    Each sample is an image (in 3D NDArray) with shape (28, 28, 1).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/fashion-mnist'
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        DEPRECATED FUNCTION ARGUMENTS.
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'fashion-mnist'),
                 train=True, transform=None):
        self._train = train
        self._train_data = ('train-images-idx3-ubyte.gz',
                            '0cf37b0d40ed5169c6b3aba31069a9770ac9043d')
        self._train_label = ('train-labels-idx1-ubyte.gz',
                             '236021d52f1e40852b06a4c3008d8de8aef1e40b')
        self._test_data = ('t10k-images-idx3-ubyte.gz',
                           '626ed6a7c06dd17c0eec72fa3be1740f146a2863')
        self._test_label = ('t10k-labels-idx1-ubyte.gz',
                            '17f9ab60e7257a1620f4ad76bbbaf857c3920701')
        self._namespace = 'fashion-mnist'
        super(MNIST, self).__init__(root, transform) # pylint: disable=bad-super-call


class CIFAR10(dataset._DownloadedDataset):
    """CIFAR10 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html

    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/cifar10
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        DEPRECATED FUNCTION ARGUMENTS.
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'cifar10'),
                 train=True, transform=None):
        self._train = train
        self._archive_file = ('cifar-10-binary.tar.gz', 'fab780a1e191a7eda0f345501ccd62d20f7ed891')
        self._train_data = [('data_batch_1.bin', 'aadd24acce27caa71bf4b10992e9e7b2d74c2540'),
                            ('data_batch_2.bin', 'c0ba65cce70568cd57b4e03e9ac8d2a5367c1795'),
                            ('data_batch_3.bin', '1dd00a74ab1d17a6e7d73e185b69dbf31242f295'),
                            ('data_batch_4.bin', 'aab85764eb3584312d3c7f65fd2fd016e36a258e'),
                            ('data_batch_5.bin', '26e2849e66a845b7f1e4614ae70f4889ae604628')]
        self._test_data = [('test_batch.bin', '67eb016db431130d61cd03c7ad570b013799c88c')]
        self._namespace = 'cifar10'
        super(CIFAR10, self).__init__(root, transform)

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072+1)

        return data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
               data[:, 0].astype(np.int32)

    def _get_data(self):
        if any(not os.path.exists(path) or not check_sha1(path, sha1)
               for path, sha1 in ((os.path.join(self._root, name), sha1)
                                  for name, sha1 in self._train_data + self._test_data)):
            namespace = 'gluon/dataset/'+self._namespace
            filename = download(_get_repo_file_url(namespace, self._archive_file[0]),
                                path=self._root,
                                sha1_hash=self._archive_file[1])

            with tarfile.open(filename) as tar:
                tar.extractall(self._root)

        if self._train:
            data_files = self._train_data
        else:
            data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name, _ in data_files))
        data = np.concatenate(data)
        label = np.concatenate(label)

        array_fn = _mx_np.array if is_np_array() else nd.array
        self._data = array_fn(data, dtype=data.dtype)
        self._label = array_fn(label, dtype=label.dtype) if is_np_array() else label


class CIFAR100(CIFAR10):
    """CIFAR100 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html

    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/cifar100
        Path to temp folder for storing data.
    fine_label : bool, default False
        Whether to load the fine-grained (100 classes) or coarse-grained (20 super-classes) labels.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        DEPRECATED FUNCTION ARGUMENTS.
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'cifar100'),
                 fine_label=False, train=True, transform=None):
        self._train = train
        self._archive_file = ('cifar-100-binary.tar.gz', 'a0bb982c76b83111308126cc779a992fa506b90b')
        self._train_data = [('train.bin', 'e207cd2e05b73b1393c74c7f5e7bea451d63e08e')]
        self._test_data = [('test.bin', '8fb6623e830365ff53cf14adec797474f5478006')]
        self._fine_label = fine_label
        self._namespace = 'cifar100'
        super(CIFAR10, self).__init__(root, transform) # pylint: disable=bad-super-call

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072+2)

        return data[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
               data[:, 0+self._fine_label].astype(np.int32)


class ImageRecordDataset(dataset.RecordFileDataset):
    """A dataset wrapping over a RecordIO file containing images.

    Each sample is an image and its corresponding label.

    Parameters
    ----------
    filename : str
        Path to rec file.
    flag : {0, 1}, default 1
        If 0, always convert images to greyscale. \
        If 1, always convert images to colored (RGB).
    transform : function, default None
        DEPRECATED FUNCTION ARGUMENTS.
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """
    def __init__(self, filename, flag=1, transform=None):
        super(ImageRecordDataset, self).__init__(filename)
        if transform is not None:
            raise DeprecationWarning(
                'Directly apply transform to dataset is deprecated. '
                'Please use dataset.transform() or dataset.transform_first() instead...')
        self._flag = flag
        self._transform = transform

    def __getitem__(self, idx):
        record = super(ImageRecordDataset, self).__getitem__(idx)
        header, img = recordio.unpack(record)
        if self._transform is not None:
            return self._transform(image.imdecode(img, self._flag), header.label)
        return image.imdecode(img, self._flag), header.label

    def __mx_handle__(self):
        from .._internal import ImageRecordFileDataset as _ImageRecordFileDataset
        return _ImageRecordFileDataset(rec_file=self.filename, idx_file=self.idx_file,
                                       flag=self._flag)


class ImageFolderDataset(dataset.Dataset):
    """A dataset for loading image files stored in a folder structure.

    like::

        root/car/0001.jpg
        root/car/xxxa.jpg
        root/car/yyyb.jpg
        root/bus/123.jpg
        root/bus/023.jpg
        root/bus/wwww.jpg

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).
    transform : callable, default None
        DEPRECATED FUNCTION ARGUMENTS.
        A function that takes data and label and transforms them::

            transform = lambda data, label: (data.astype(np.float32)/255, label)
    synsets : list, default None
        User defined synsets.

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all images in (filename, label) pairs.
    """

    def __init__(self, root, flag=1, transform=None, synsets=None):
        self._root = os.path.expanduser(root)
        self._flag = flag
        if transform is not None:
            raise DeprecationWarning(
                'Directly apply transform to dataset is deprecated. '
                'Please use dataset.transform() or dataset.transform_first() instead...')
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self.synsets = synsets
        self._list_images(self._root)
        self._handle = None

    def _list_images(self, root):
        if self.synsets is None:
            self.synsets = sorted(os.listdir(root))

        self.items = []

        for label, folder in enumerate(self.synsets):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s'%(
                        filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)

    def __mx_handle__(self):
        if self._handle is None:
            from .._internal import ImageSequenceDataset, NDArrayDataset, GroupDataset
            path_sep = '|'
            im_names = path_sep.join([x[0] for x in self.items])
            label = default_array([x[1] for x in self.items])
            self._handle = GroupDataset(datasets=(
                ImageSequenceDataset(img_list=im_names, path_sep=path_sep, flag=self._flag),
                NDArrayDataset(arr=label)))
        return self._handle


class ImageListDataset(dataset.Dataset):
    """A dataset for loading image files specified by a list of entries.

    like::

        # if written to text file *.lst
        0\t0\troot/car/0001.jpg
        1\t0\troot/car/xxxa.jpg
        2\t0\troot/car/yyyb.jpg
        3\t1\troot/bus/123.jpg
        4\t1\troot/bus/023.jpg
        5\t1\troot/bus/wwww.jpg

        # if as a pure list, each item is a list [imagelabel: float or list of float, imgpath]
        [[0, root/car/0001.jpg]
         [0, root/car/xxxa.jpg]
         [0, root/car/yyyb.jpg]
         [1, root/bus/123.jpg]
         [1, root/bus/023.jpg]
         [1, root/bus/wwww.jpg]]

    Parameters
    ----------
    root : str
        Path to root directory.
    imglist : str or list
        Specify the path of imglist file or a list directly
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).

    Attributes
    ----------
    items : list of tuples
        List of all images in (filename, label) pairs.
    """
    def __init__(self, root='.', imglist=None, flag=1):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._imglist = {}
        self._imgkeys = []
        self._handle = None
        array_fn = _mx_np.array if is_np_array() else nd.array
        if isinstance(imglist, str):
            # read from file
            fname = os.path.join(self._root, imglist)
            with open(fname, 'rt') as fin:
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = array_fn(line[1:-1])
                    key = int(line[0])
                    self._imglist[key] = (label, os.path.join(self._root, line[-1]))
                    self._imgkeys.append(key)
        elif isinstance(imglist, list):
            index = 1
            for img in imglist:
                key = str(index)
                index += 1
                if len(img) > 2:
                    label = array_fn(img[:-1])
                elif isinstance(img[0], numeric_types):
                    label = array_fn([img[0]])
                else:
                    label = array_fn(img[0])
                assert isinstance(img[-1], str)
                self._imglist[key] = (label, os.path.join(self._root, img[-1]))
                self._imgkeys.append(key)
        else:
            raise ValueError(
                "imglist must be filename or list of valid entries, given {}".format(
                    type(imglist)))

    def __getitem__(self, idx):
        key = self._imgkeys[idx]
        img = image.imread(self._imglist[key][1], self._flag)
        label = self._imglist[key][0]
        return img, label

    def __len__(self):
        return len(self._imgkeys)

    def __mx_handle__(self):
        if self._handle is None:
            from .._internal import ImageSequenceDataset, NDArrayDataset, GroupDataset
            path_sep = '|'
            im_names = path_sep.join([self._imglist[x][1] for x in self._imgkeys])
            label = default_array(np.array([self._imglist[x][0].asnumpy() for x in self._imgkeys]))
            self._handle = GroupDataset(datasets=(
                ImageSequenceDataset(img_list=im_names, path_sep=path_sep, flag=self._flag),
                NDArrayDataset(arr=label)))
        return self._handle
