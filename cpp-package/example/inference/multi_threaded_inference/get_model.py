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

import logging
import argparse
import requests
import errno
import os

models = ["imagenet1k-inception-bn", "imagenet1k-resnet-50",
          "imagenet1k-resnet-152", "imagenet1k-resnet-18"]

def download(url, fname=None, dirname=None, overwrite=False, retries=5):
    """Download an given URL

    Parameters
    ----------

    url : str
        URL to download
    fname : str, optional
        filename of the downloaded file. If None, then will guess a filename
        from url.
    dirname : str, optional
        output directory name. If None, then guess from fname or use the current
        directory
    overwrite : bool, optional
        Default is false, which means skipping download if the local file
        exists. If true, then download the url to overwrite the local file if
        exists.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes

    Returns
    -------
    str
        The filename of the downloaded file
    """

    assert retries >= 0, "Number of retries should be at least 0"

    if fname is None:
        fname = url.split('/')[-1]

    if dirname is None:
        dirname = os.path.dirname(fname)
    else:
        fname = os.path.join(dirname, fname)
    if dirname != "":
        if not os.path.exists(dirname):
            try:
                logging.info('create directory %s', dirname)
                os.makedirs(dirname)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise OSError('failed to create ' + dirname)

    if not overwrite and os.path.exists(fname):
        logging.info("%s exists, skipping download", fname)
        return fname

    while retries+1 > 0:
        # Disable pyling too broad Exception
        # pylint: disable=W0703
        try:
            r = requests.get(url, stream=True)
            assert r.status_code == 200, "failed to open %s" % url
            with open(fname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                break
        except Exception as e:
            retries -= 1
            if retries <= 0:
                raise e

            print("download failed, retrying, {} attempt{} left"
                  .format(retries, 's' if retries > 1 else ''))
    logging.info("downloaded %s into %s successfully", url, fname)
    return fname

def download_model(model_name, dst_dir='./', meta_info=None):
    """Download a model from data.mxnet.io

    Parameters
    ----------
    model_name : str
        Model name to download
    dst_dir : str
        Destination Directory to download the model
    meta_info : dict of dict
        Mapping from model_name to dict of the following structure:
        {'symbol': url, 'params': url}

    Returns
    -------
    Two element tuple containing model_name and epoch for the params saved
    """
    _base_model_url = 'http://data.mxnet.io/models/'
    _default_model_info = {
        'imagenet1k-inception-bn': {'symbol':_base_model_url+'imagenet/inception-bn/Inception-BN-symbol.json',
                                    'params':_base_model_url+'imagenet/inception-bn/Inception-BN-0126.params'},
        'imagenet1k-resnet-18': {'symbol':_base_model_url+'imagenet/resnet/18-layers/resnet-18-symbol.json',
                                 'params':_base_model_url+'imagenet/resnet/18-layers/resnet-18-0000.params'},
        'imagenet1k-resnet-34': {'symbol':_base_model_url+'imagenet/resnet/34-layers/resnet-34-symbol.json',
                                 'params':_base_model_url+'imagenet/resnet/34-layers/resnet-34-0000.params'},
        'imagenet1k-resnet-50': {'symbol':_base_model_url+'imagenet/resnet/50-layers/resnet-50-symbol.json',
                                 'params':_base_model_url+'imagenet/resnet/50-layers/resnet-50-0000.params'},
        'imagenet1k-resnet-101': {'symbol':_base_model_url+'imagenet/resnet/101-layers/resnet-101-symbol.json',
                                  'params':_base_model_url+'imagenet/resnet/101-layers/resnet-101-0000.params'},
        'imagenet1k-resnet-152': {'symbol':_base_model_url+'imagenet/resnet/152-layers/resnet-152-symbol.json',
                                  'params':_base_model_url+'imagenet/resnet/152-layers/resnet-152-0000.params'},
        'imagenet1k-resnext-50': {'symbol':_base_model_url+'imagenet/resnext/50-layers/resnext-50-symbol.json',
                                  'params':_base_model_url+'imagenet/resnext/50-layers/resnext-50-0000.params'},
        'imagenet1k-resnext-101': {'symbol':_base_model_url+'imagenet/resnext/101-layers/resnext-101-symbol.json',
                                   'params':_base_model_url+'imagenet/resnext/101-layers/resnext-101-0000.params'},
        'imagenet1k-resnext-101-64x4d':
            {'symbol':_base_model_url+'imagenet/resnext/101-layers/resnext-101-64x4d-symbol.json',
             'params':_base_model_url+'imagenet/resnext/101-layers/resnext-101-64x4d-0000.params'},
        'imagenet11k-resnet-152':
            {'symbol':_base_model_url+'imagenet-11k/resnet-152/resnet-152-symbol.json',
             'params':_base_model_url+'imagenet-11k/resnet-152/resnet-152-0000.params'},
        'imagenet11k-place365ch-resnet-152':
            {'symbol':_base_model_url+'imagenet-11k-place365-ch/resnet-152-symbol.json',
             'params':_base_model_url+'imagenet-11k-place365-ch/resnet-152-0000.params'},
        'imagenet11k-place365ch-resnet-50':
            {'symbol':_base_model_url+'imagenet-11k-place365-ch/resnet-50-symbol.json',
             'params':_base_model_url+'imagenet-11k-place365-ch/resnet-50-0000.params'},
    }


    if meta_info is None:
        meta_info = _default_model_info
    meta_info = dict(meta_info)
    if model_name not in meta_info:
        return (None, 0)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    meta = dict(meta_info[model_name])
    assert 'symbol' in meta, "missing symbol url"
    model_name = os.path.join(dst_dir, model_name)
    download(meta['symbol'], model_name+'-symbol.json')
    assert 'params' in meta, "mssing parameter file url"
    download(meta['params'], model_name+'-0000.params')
    download(_base_model_url + 'imagenet/synset.txt')
    return (model_name, 0)

def main():
    logging.basicConfig()
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Download model hybridize and save as symbolic model for multithreaded inference')
    parser.add_argument("--model", type=str, choices=models, required=True)
    args = parser.parse_args()

    download_model(args.model)

if __name__ == "__main__":
    main()
