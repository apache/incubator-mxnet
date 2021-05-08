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
Description:
Inspired by https://github.com/brendenlake/omniglot
"""
from __future__ import print_function
import os
import shutil
import errno
import numpy as np
from mxnet import nd
import mxnet.gluon.data as data
from PIL import Image
# pylint: disable=invalid-name, too-many-instance-attributes, redefined-outer-name, line-too-long, too-many-arguments, unused-argument


IMG_CACHE = {}
class OmniglotDataset(data.Dataset):
    """
    Description : set the omniglot dataset
    """
    vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='../dataset/omniglot',\
                 transform=None, target_transform=None,\
                 download=False):
        '''
        The items are (filename,category).
        The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the dataset will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the dataset
        '''
        super(OmniglotDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes)

        self.idx_classes = index_classes(self.all_items)

        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))])

        self.x = map(load_img, paths, range(len(paths)))
        self.x = list(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        """
        Description : get path for label
        """
        filename = self.all_items[index][0]
        rot = self.all_items[index][-1]
        img = str.join('/', [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        print(os.path.join(self.root, self.processed_folder))
        return os.path.exists(os.path.join(self.root, self.processed_folder))

    def download(self):
        """
        Description : download omniglot dataset
        """
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.splits_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for _, url in self.vinyals_split_sizes.items():
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[-1]
            file_path = os.path.join(self.root, self.splits_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            orig_root = os.path.join(self.root, self.raw_folder)
            print("== Unzip from " + file_path + " to " + orig_root)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(orig_root)
            zip_ref.close()
        file_processed = os.path.join(self.root, self.processed_folder)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(orig_root, p)):
                shutil.move(os.path.join(orig_root, p, f), file_processed)
            os.rmdir(os.path.join(orig_root, p))
        print("Download finished.")


def find_items(root_dir, classes):
    """
    Description : Find items
    """
    retour = []
    rots = ['/rot000', '/rot090', '/rot180', '/rot270']
    for (root, _, files) in os.walk(root_dir):
        for f in files:
            r = root.split('/')
            lr = len(r)
            label = r[lr - 2] + "/" + r[lr - 1]
            for rot in rots:
                if label + rot in classes and (f.endswith("png")):
                    retour.extend([(f, label, root, rot)])
    print("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    """
    Description : check number of classes
    """
    idx = {}
    for i in items:
        if not i[1] + i[-1] in idx:
            idx[i[1] + i[-1]] = len(idx)
    print("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    """
    Description : get the current classes
    """
    with open(fname) as f:
        classes = f.read().splitlines()
    return classes


def load_img(path, idx):
    """
    Description : load image files
    """
    path, rot = path.split('/rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - nd.array(x)
    x = x.transpose(axes=(0, 1))
    x = nd.expand_dims(x, 0)
    return x
