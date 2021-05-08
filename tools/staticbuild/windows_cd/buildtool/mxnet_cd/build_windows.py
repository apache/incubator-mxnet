#!/usr/bin/env python
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

"""User friendly / multi platform builder script"""

import argparse
import datetime
import glob
import logging
import os
import platform
import shutil
import sys
import time
from distutils.dir_util import copy_tree
from enum import Enum
from subprocess import check_call, CalledProcessError

from util import *


class VCVARS(Enum):
    VS_2017 = 'VS_2017'
    VS_2019 = 'VS_2019'


KNOWN_VCVARS = {
    'VS_2017': r'C:\BuildTools\Common7\Tools\VsDevCmd.bat -host_arch=amd64 -arch=amd64 -vcvars_ver=14.16',
    'VS_2019': r'C:\BuildTools\Common7\Tools\VsDevCmd.bat -host_arch=amd64 -arch=amd64 '
}


class BuildFlavour(Enum):
    WIN_CPU = 'WIN_CPU'
    WIN_CPU_MKLDNN = 'WIN_CPU_MKLDNN'
    WIN_GPU = 'WIN_GPU'
    WIN_GPU_MKLDNN = 'WIN_GPU_MKLDNN'


CMAKE_FLAGS = {
    'WIN_CPU': ('-DUSE_CUDA=0 '
                '-DUSE_CUDNN=0 '
                '-DUSE_NVRTC=0 '
                '-DUSE_OPENCV=1 '
                '-DUSE_OPENMP=1 '
                '-DUSE_PROFILER=0 '
                '-DUSE_BLAS=open '
                '-DUSE_LAPACK=1 '
                '-DUSE_DIST_KVSTORE=0 '
                '-DBUILD_CPP_EXAMPLES=0 '
                '-DUSE_MKL_IF_AVAILABLE=0 '
                '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_CPU_MKLDNN': ('-DUSE_CUDA=0 '
                         '-DUSE_CUDNN=0 '
                         '-DUSE_NVRTC=0 '
                         '-DUSE_OPENCV=1 '
                         '-DUSE_OPENMP=1 '
                         '-DUSE_PROFILER=0 '
                         '-DUSE_BLAS=open '
                         '-DUSE_LAPACK=1 '
                         '-DUSE_DIST_KVSTORE=0 '
                         '-DUSE_MKL_IF_AVAILABLE=1 '
                         '-DUSE_MKLDNN=1 '
                         '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_GPU': ('-DUSE_CUDA=1 '
                  '-DUSE_CUDNN=1 '
                  '-DUSE_NVRTC=1 '
                  '-DUSE_OPENCV=1  '
                  '-DUSE_OPENMP=1 '
                  '-DUSE_PROFILER=0 '
                  '-DUSE_BLAS=open '
                  '-DUSE_LAPACK=1 '
                  '-DUSE_DIST_KVSTORE=0 '
                  '-DCUDA_ARCH_LIST=Common '
                  # '-DCMAKE_CXX_FLAGS="/FS /MT /O2 /Ob2" '
                  '-DUSE_MKL_IF_AVAILABLE=0 '
                  '-DCMAKE_BUILD_TYPE=Release')

    , 'WIN_GPU_MKLDNN': ('-DUSE_CUDA=1 '
                         '-DUSE_CUDNN=1 '
                         '-DUSE_NVRTC=1 '
                         '-DUSE_OPENCV=1 '
                         '-DUSE_OPENMP=1 '
                         '-DUSE_PROFILER=0 '
                         '-DUSE_BLAS=open '
                         '-DUSE_LAPACK=1 '
                         '-DUSE_DIST_KVSTORE=0 '
                         '-DCUDA_ARCH_LIST=Common '
                         '-DUSE_MKL_IF_AVAILABLE=1 '
                         '-DUSE_MKLDNN=1 '
                         #  '-DCMAKE_CXX_FLAGS="/FS /MT /O2 /Ob2" '
                         '-DCMAKE_BUILD_TYPE=Release')

}


def modify(mxnet_src):
    base_p = os.path.join(mxnet_src, "python/mxnet/base.py")
    lines = None
    with open(base_p, 'r', encoding='utf-8') as base_obj:
        lines = base_obj.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace("py_str = lambda x: x.decode('utf-8')",
                                    "py_str = lambda x: x.decode('utf-8','replace')")
        pass
    with open(base_p, 'w', encoding='utf-8') as base_obj:
        for line in lines:
            base_obj.write(line)
    pass


def windows_build(args):
    os.chdir(args.root)
    logging.info("Using vcvars environment:\n{}".format(args.vcvars))

    j = os.path.join
    path = j(args.root, "build_" + args.name)
    os.makedirs(path, exist_ok=True)

    mxnet_src = j(args.root, "mxnet")
    logging.info("Found MXNet root: {}".format(mxnet_src))

    modify(mxnet_src)

    with remember_cwd():
        os.chdir(path)
        cmd = "{} && cmake -G \"Ninja\" {} -DMXNET_CUDA_ARCH=\"3.0\" {}".format(KNOWN_VCVARS[args.vcvars],
                                                                                CMAKE_FLAGS[args.flavour],
                                                                                mxnet_src)
        logging.info("Generating project with CMake:\n{}".format(cmd))

        check_call(cmd, shell=True)

        # from fixtime import build_config
        # build_config(args.root)
        # os.system(cmd)
        # cmd = "{} && BuildConsole /command=\"jom -j 128\""
        cmd = "{} && ninja" \
            .format(KNOWN_VCVARS[args.vcvars])
        logging.info("Building with jom:\n{}".format(cmd))

        t0 = int(time.time())
        is_success = False
        for i in range(5):
            try:
                check_call(cmd, shell=True)
                is_success = True
                break
            except CalledProcessError as e:
                pass

        if not is_success:
            raise CalledProcessError(1, cmd)

        logging.info("Build flavour: {} complete in directory: \"{}\"".format(args.flavour, os.path.abspath(path)))
        logging.info("Build took {}".format(datetime.timedelta(seconds=int(time.time() - t0))))


def windows_package(args):
    pkgfile = args.build_timestamp + "_mxnet_x64_" + args.name
    pkgdir = os.path.abspath("pkg_" + args.name)
    logging.info("Packaging libraries and headers in package: %s", pkgfile)
    j = os.path.join
    pkgdir_lib = os.path.abspath(j(pkgdir, 'lib'))
    pkgdir_build = os.path.abspath(j(pkgdir, 'build'))
    if os.path.exists(pkgdir):
        shutil.rmtree(pkgdir)
    if os.path.exists(pkgfile):
        os.remove(pkgfile)
    with remember_cwd():
        os.chdir(j(args.root, "build_" + args.name))
        logging.info("Looking for static libraries and dlls in: \"%s", os.getcwd())
        libs = list(glob.iglob('**/*.lib', recursive=True))
        dlls = list(glob.iglob('**/*.dll', recursive=True))
        os.makedirs(pkgdir_lib, exist_ok=True)
        os.makedirs(pkgdir_build, exist_ok=True)
        for lib in libs:
            logging.info("packing lib: %s", lib)
            shutil.copy(lib, pkgdir_lib)
        for dll in dlls:
            logging.info("packing dll: %s", dll)
            shutil.copy(dll, pkgdir_build)
        os.chdir(j(args.root, "mxnet"))
        logging.info('packing python bindings')
        copy_tree('python', j(pkgdir, 'python'))
        logging.info('packing headers')
        copy_tree('include', j(pkgdir, 'include'))
        check_and_remove(j(pkgdir, 'include', 'dlpack'))
        check_and_remove(j(pkgdir, 'include', 'dmlc'))
        check_and_remove(j(pkgdir, 'include', 'mshadow'))
        check_and_remove(j(pkgdir, 'include', 'nnvm'))
        copy_tree(j('3rdparty', 'dmlc-core', 'include'), j(pkgdir, 'include'))
        copy_tree(j('3rdparty', 'mshadow', 'mshadow'), j(pkgdir, 'include', 'mshadow'))
        copy_tree(j('3rdparty', 'tvm', 'nnvm', 'include'), j(pkgdir, 'include', 'nnvm', 'include'))
        logging.info("Compressing package: %s", pkgfile)
    check_call(['7z', 'a', pkgfile, pkgdir])


def check_and_remove(path):
    if os.path.exists(path):
        os.remove(path)


def nix_build(args):
    path = args.output
    os.makedirs(path, exist_ok=True)
    with remember_cwd():
        os.chdir(path)
        logging.info("Generating project with CMake")
        check_call("cmake \
            -DUSE_CUDA=OFF \
            -DUSE_OPENCV=OFF \
            -DUSE_OPENMP=OFF \
            -DCMAKE_BUILD_TYPE=Debug \
            -GNinja ..", shell=True)
        check_call("ninja", shell=True)


def get_cuda_version(name: str):
    name_sp = name.split("_")
    if name_sp[-1] == "mkl":
        return name_sp[-2]
    else:
        return name_sp[-1]


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)-15s %(message)s')
    logging.info("MXNet Windows build helper")

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root",
                        help="root directory",
                        default='build',
                        type=str)

    parser.add_argument("--vcvars",
                        help="vcvars batch file location, typically inside vs studio install dir",
                        default="VS_2015",
                        choices=[x.name for x in VCVARS],
                        type=str)

    parser.add_argument("-f", "--flavour",
                        help="build flavour",
                        default='WIN_CPU',
                        choices=[x.name for x in BuildFlavour],
                        type=str)

    parser.add_argument("--mxnet_src",
                        help="mxnet src",
                        type=str)

    parser.add_argument("--name",
                        help="name",
                        type=str)

    parser.add_argument("--build_timestamp",
                        help="build timestamp",
                        type=str)

    args = parser.parse_args()
    logging.info("Build flavour: %s", args.flavour)

    system = platform.system()
    if system == 'Windows':
        logging.info("Detected Windows platform")
        if 'OpenBLAS_HOME' not in os.environ:
            os.environ["OpenBLAS_HOME"] = "C:\\dep\\openblas"
        if 'OpenCV_DIR' not in os.environ:
            os.environ["OpenCV_DIR"] = "C:\\dep\\opencv_vc141"
        if 'CUDA_PATH' not in os.environ:
            raise RuntimeError("not found cuda")
        try:
            windows_build(args)
        except CalledProcessError as e:
            return e.returncode
            pass

    elif system == 'Linux' or system == 'Darwin':
        nix_build(args)
    else:
        logging.error("Don't know how to build for {} yet".format(platform.system()))

    return 0


if __name__ == '__main__':
    sys.exit(main())
