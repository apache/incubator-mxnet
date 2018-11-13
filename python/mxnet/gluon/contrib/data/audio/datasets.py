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
""" Audio Dataset container."""
__all__ = ['AudioFolderDataset']

import os
import warnings
from sklearn.preprocessing import LabelEncoder
from mxnet import gluon, nd


class AudioFolderDataset(gluon.data.dataset.Dataset):
    """A dataset for loading Audio files stored in a folder structure like::

        root/children_playing/0.wav
        root/siren/23.wav
        root/drilling/26.wav
        root/dog_barking/42.wav
            OR
        Files(wav) and a csv file that has filename and associated label

    Parameters
    ----------
    root : str
        Path to root directory.

    transform : callable, default None
        A function that takes data and label and transforms them

    has_csv: default True
        If True, it means that a csv file has filename and its corresponding label
        If False, we have folder like structure

    train_csv: str, default None
        If has_csv is True, train_csv should be populated by the training csv filename

    file_format: str, default '.wav'
        The format of the audio files(.wav, .mp3)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all audio in (filename, label) pairs.
    """
    def __init__(self, root, transform=None, has_csv=False, train_csv=None, file_format='.wav'):
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._exts = ['.wav']
        self._format = file_format
        self._has_csv = has_csv
        self._train_csv = train_csv
        self._list_audio_files(self._root)


    def _list_audio_files(self, root):
        """
            Populates synsets - a map of index to label for the data items.
            Populates the data in the dataset, making tuples of (data, label)
        """
        if not self._has_csv:
            self.synsets = []
            self.items = []

            for folder in sorted(os.listdir(root)):
                path = os.path.join(root, folder)
                if not os.path.isdir(path):
                    warnings.warn('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                    continue
                label = len(self.synsets)
                self.synsets.append(folder)
                for filename in sorted(os.listdir(path)):
                    file_name = os.path.join(path, filename)
                    ext = os.path.splitext(file_name)[1]
                    if ext.lower() not in self._exts:
                        warnings.warn('Ignoring %s of type %s. Only support %s'%(filename, ext, ', '.join(self._exts)))
                        continue
                    self.items.append((file_name, label))
        else:
            self.synsets = []
            self.items = []
            data_tmp = []
            label_tmp = []
            with open(self._train_csv, "r") as traincsv:
                for line in traincsv:
                    filename = os.path.join(root, line.split(",")[0])
                    label = line.split(",")[1].strip()
                    data_tmp.append(os.path.join(self._root, line.split(",")[0]))
                    label_tmp.append(line.split(",")[1].strip())
            data_tmp = data_tmp[1:]
            label_tmp = label_tmp[1:]
            le = LabelEncoder()
            self.raw_data = []
            self._label = nd.array(le.fit_transform(label_tmp))
            for i, class_name in enumerate(le.classes_):
                self.synsets.append(class_name)
            for i, _ in enumerate(data_tmp):
                if self._format not in data_tmp[i]:
                    self.items.append((data_tmp[i]+self._format, self._label[i]))

    def __getitem__(self, idx):
        """
            Retrieve the item (data, label) stored at idx in items
        """

        return self.items[idx][0], self.items[idx][1]

    def __len__(self):
        """
            Retrieves the number of items in the dataset
        """
        return len(self.items)


    def transform_first(self, fn, lazy=True):
        """Returns a new dataset with the first element of each sample
        transformed by the transformer function `fn`.

        This is useful, for example, when you only want to transform data
        while keeping label as is.

        Parameters
        ----------
        fn : callable
            A transformer function that takes the first elemtn of a sample
            as input and returns the transformed element.
        lazy : bool, default True
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        return super(AudioFolderDataset, self).transform_first(fn, lazy=False)
