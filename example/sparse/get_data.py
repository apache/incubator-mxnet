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

# pylint: skip-file
from csv import DictReader
import os, gzip
import sys
import mxnet as mx

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch


def get_libsvm_data(data_dir, data_name, url):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)
    if (not os.path.exists(data_name)):
        print("Dataset " + data_name + " not present. Downloading now ...")
        import urllib
        zippath = os.path.join(data_dir, data_name + ".bz2")
        urllib.urlretrieve(url + data_name + ".bz2", zippath)
        os.system("bzip2 -d %r" % data_name + ".bz2")
        print("Dataset " + data_name + " is now present.")
    os.chdir("..")


def get_uci_data(data_dir, data_name, url):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    os.chdir(data_dir)
    if (not os.path.exists(data_name)):
        print("Dataset " + data_name + " not present. Downloading now ...")
        os.system("wget %r" % url + data_name)
        if "test" in data_name:
            os.system("sed -i '1d' %r" % data_name)
        preprocess(data_name, data_name + ".libsvm")
        print("Dataset " + data_name + ".libsvm" + " is now present.")
    os.chdir("..")

def preprocess(data_name, out_name):
    # Some tricks of feature engineering are adapted from tensorflow's wide and deep tutorial
    csv_columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "gender",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income_bracket"
    ]

    vocabulary_dict = {
        "gender": [
            "Female", "Male"
        ],
        "education": [
            "Bachelors", "HS-grad", "11th", "Masters", "9th",
            "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
            "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
            "Preschool", "12th"
        ],
        "marital_status": [
            "Married-civ-spouse", "Divorced", "Married-spouse-absent",
            "Never-married", "Separated", "Married-AF-spouse", "Widowed"
        ],
        "relationship": [
            "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
            "Other-relative"
        ],
        "workclass": [
            "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
            "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
        ]
    }

    crossed_columns = [
        ["education", "occupation"],
        ["native_country", "occupation"]
    ]

    indicator_columns = ['workclass', 'education', 'gender', 'relationship']

    embedding_columns = ['native_country', 'occupation']

    continuous_columns = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

    labels = ["<", ">"]

    hash_bucket_size = 1000

    all_rows = []

    with open(data_name) as f:
        for t, row in enumerate(DictReader(f, fieldnames=csv_columns)):
            feats = [labels.index(row['income_bracket'].strip()[0])]

            for cols in crossed_columns:
                s = '_'.join([row[col].strip() for col in cols])
                feats.append([hash_bucket_size, hash(s) % hash_bucket_size])

            for col in embedding_columns:
                feats.append([1, hash(row[col].strip()) % hash_bucket_size])

            for col in indicator_columns:
                feats.append([len(vocabulary_dict[col]), vocabulary_dict[col].index(row[col].strip())])

            for col in continuous_columns:
                feats.append([1, float(row[col].strip())])

            all_rows.append(feats)

    with open(out_name, "w") as f:
        for row in all_rows:
            s = ''
            feat_dim = 0
            s += str(row[0])
            for feat in row[1:]:
                s += ' '
                if feat[0] != 1:
                   s += str(feat_dim + feat[1]) + ':1.0'
                else:
                   s += str(feat_dim) + ':' + str(feat[1])
                feat_dim += feat[0]
            s += '\n'

            f.write(s)

def get_movielens_data(prefix):
    if not os.path.exists("%s.zip" % prefix):
        print("Dataset MovieLens 10M not present. Downloading now ...")
        os.system("wget http://files.grouplens.org/datasets/movielens/%s.zip" % prefix)
        os.system("unzip %s.zip" % prefix)
        os.system("cd ml-10M100K; sh split_ratings.sh; cd -;")

def get_movielens_iter(filename, batch_size, dummy_iter):
    """Not particularly fast code to parse the text file and load into NDArrays.
    return two data iters, one for train, the other for validation.
    """
    print("Preparing data iterators for " + filename + " ... ")
    user = []
    item = []
    score = []
    with file(filename) as f:
        num_samples = 0
        for line in f:
            tks = line.strip().split('::')
            if len(tks) != 4:
                continue
            num_samples += 1
            user.append((tks[0]))
            item.append((tks[1]))
            score.append((tks[2]))
            if dummy_iter and num_samples > batch_size * 10:
                break
    # convert to ndarrays
    user = mx.nd.array(user, dtype='int32')
    item = mx.nd.array(item)
    score = mx.nd.array(score)
    # prepare data iters
    data_train = {'user':user, 'item':item}
    label_train = {'score':score}
    iter_train = mx.io.NDArrayIter(data=data_train,label=label_train,
                                   batch_size=batch_size, shuffle=True)
    iter_train = DummyIter(iter_train) if dummy_iter else iter_train
    return mx.io.PrefetchingIter(iter_train)
