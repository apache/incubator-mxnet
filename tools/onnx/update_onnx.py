#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import mxnet
import os
import logging
import argparse

def update_onnx(branch='v1.x'):
    # Detect MXNet
    print('Detected MXNet version: %s' % mxnet.__version__)
    mx_path = os.path.abspath(mxnet.__file__)
    mx_path = mx_path[:mx_path.rfind('/')]
    onnx_path = mx_path + '/contrib/onnx/'
    if os.path.isdir(onnx_path):
        print('Found ONNX path: %s' % onnx_path)
    else:
        logging.error('ONNX path not found. %s does not exist' % onnx_path)

    # Backup the current onnx dir
    backup_path = onnx_path + 'backup'
    os.system('mkdir %s' % backup_path)
    os.system('mv -v %s/* %s' % (onnx_path, backup_path))

    # Clone the latest repo and copy the onnx dir
    clone_path = './mxnet_repo_tmp'
    os.system('mkdir %s' % clone_path)
    cwd = os.getcwd()
    os.chdir(clone_path)
    os.system('git clone https://github.com/apache/incubator-mxnet mxnet')
    os.chdir('./mxnet')
    os.system('git checkout %s' % branch)
    os.system('cp -r python/mxnet/contrib/onnx/* %s/' % onnx_path)
    os.chdir(cwd)
    os.system('rm -rf %s' %clone_path)
    print('Done')


def restore_onnx():
    # Detect MXNet
    print('Detected MXNet version: %s' % mxnet.__version__)
    mx_path = os.path.abspath(mxnet.__file__)
    mx_path = mx_path[:mx_path.rfind('/')]
    onnx_path = mx_path + '/contrib/onnx'
    backup_path = onnx_path + '/backup'
    if os.path.isdir(backup_path):
        print('Found ONNX path: %s' % onnx_path)
        print('Found ONNX backup path: %s' % backup_path)
    else:
        logging.error('ONNX backup path not found. %s does not exist' % backup_path)

    # Restore backup
    os.chdir(onnx_path)
    os.system('find . -mindepth 1 -maxdepth 1 ! -name backup -exec rm -r "{}" \;')
    os.system('cp -r ./backup/* .')
    os.system('rm -rf backup')
    print('Done')


parser = argparse.ArgumentParser(description='Update/Restore ONNX dir with the latest changes '
                                 'on GitHub')
parser.add_argument('--branch', default='v1.x',
                    help='which branch to checkout')
parser.add_argument('--restore', action='store_true', help='restore the backup files')
args = parser.parse_args()

if args.restore:
    print('Restoring')
    restore_onnx()
else:
    print('Updating to changes in branch %s' % args.branch)
    update_onnx(args.branch)
