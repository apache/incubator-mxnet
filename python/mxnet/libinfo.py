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
"""Information about mxnet."""
from __future__ import absolute_import
import os
import platform
import logging


def find_lib_path():
    """Find MXNet dynamic library files.

    Returns
    -------
    lib_path : list(string)
        List of all found path to the libraries.
    """
    lib_from_env = os.environ.get('MXNET_LIBRARY_PATH')
    if lib_from_env:
        if os.path.isfile(lib_from_env):
            if not os.path.isabs(lib_from_env):
                logging.warning("MXNET_LIBRARY_PATH should be an absolute path, instead of: %s",
                                lib_from_env)
            else:
                if os.name == 'nt':
                    os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.dirname(lib_from_env)
                return [lib_from_env]
        else:
            logging.warning("MXNET_LIBRARY_PATH '%s' doesn't exist", lib_from_env)

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    api_path = os.path.join(curr_path, '../../lib/')
    cmake_build_path = os.path.join(curr_path, '../../build/')
    dll_path = [curr_path, api_path, cmake_build_path]
    if os.name == 'nt':
        dll_path.append(os.path.join(curr_path, '../../build'))
        vs_configuration = 'Release'
        if platform.architecture()[0] == '64bit':
            dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
            dll_path.append(os.path.join(curr_path, '../../windows/x64', vs_configuration))
        else:
            dll_path.append(os.path.join(curr_path, '../../build', vs_configuration))
            dll_path.append(os.path.join(curr_path, '../../windows', vs_configuration))
    elif os.name == "posix" and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path[0:0] = [p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")]
    if os.name == 'nt':
        os.environ['PATH'] = os.path.dirname(__file__) + ';' + os.environ['PATH']
        dll_path = [os.path.join(p, 'libmxnet.dll') for p in dll_path]
    elif platform.system() == 'Darwin':
        dll_path = [os.path.join(p, 'libmxnet.dylib') for p in dll_path] + \
                   [os.path.join(p, 'libmxnet.so') for p in dll_path]
    else:
        dll_path.append('../../../')
        dll_path = [os.path.join(p, 'libmxnet.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) == 0:
        raise RuntimeError('Cannot find the MXNet library.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))
    if os.name == 'nt':
        os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.dirname(lib_path[0])
    return lib_path

def find_include_path():
    """Find MXNet included header files.

    Returns
    -------
    incl_path : string
        Path to the header files.
    """
    incl_from_env = os.environ.get('MXNET_INCLUDE_PATH')
    if incl_from_env:
        if os.path.isfile(incl_from_env):
            if not os.path.isabs(incl_from_env):
                logging.warning("MXNET_INCLUDE_PATH should be an absolute path, instead of: %s",
                                incl_from_env)
            else:
                if os.name == 'nt':
                    os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.dirname(incl_from_env)
                return incl_from_env
        else:
            logging.warning("MXNET_INCLUDE_PATH '%s' doesn't exist", incl_from_env)
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    incl_path = os.path.join(curr_path, '../../include/')
    if len(incl_path) == 0:
        raise RuntimeError('Cannot find the MXNet include path.\n')
    if os.name == 'nt':
        os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.dirname(incl_path)
    return incl_path


# current version
__version__ = "1.4.0"
