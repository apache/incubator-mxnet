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
"""TVM Operator compile entry point"""
import tvm
from tvm import autotvm

import os
import argparse
import re
import json
import logging
import sys
import subprocess
from tvmop.opdef import __OP_DEF__
from tvmop.space import ConfigSpaces, ConfigSpace
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

logging.basicConfig(level=logging.INFO)


def create_shared(output,
                  objects,
                  options=None,
                  cc="g++"):
    """Create shared library.
    Parameters
    ----------
    output : str
        The target shared library.
    objects : List[str]
        List of object files.
    options : List[str]
        The list of additional options string.
    cc : Optional[str]
        The compiler command.
    """
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        _linux_compile(output, objects, options, cc)
    # TODO(yzhliu): elif sys.platform == "win32":
    else:
        raise ValueError("Unsupported platform")


def _linux_compile(output, objects, options, compile_cmd="g++"):
    cmd = [compile_cmd]
    if output.endswith(".so") or output.endswith(".dylib"):
        cmd += ["-shared", "-fPIC"]
        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]
    elif output.endswith(".obj"):
        cmd += ["-c"]
    cmd += ["-o", output]
    if isinstance(objects, str):
        cmd += [objects]
    else:
        cmd += objects
    if options:
        cmd += options
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += str(out)
        raise RuntimeError(msg)


def get_target(device):
    if device == "cpu":
        return "llvm"
    elif device == "cuda" or device == "gpu":
        return "cuda"
    assert False, "Unknown device " + device


def get_cuda_arch(arch):
    if arch is None:
        return None

    if not isinstance(arch, str):
        raise TypeError('Expecting parameter arch as a str, while got a {}'.format(str(type(arch))))

    if len(arch) == 0:
        return None

    # an example of arch string,
    # -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
    # -gencode;arch=compute_75,code=[sm_75,compute_75] --fatbin-options -compress-all
    # -gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35
    archs = []
    flags = arch.replace(";", " ").split()
    for flag in flags:
        if flag.startswith('-gencode') or flag.startswith('arch='):
            archs.append(flag)

    return archs


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))
    parser = argparse.ArgumentParser(description="Generate tvm operators")
    parser.add_argument("-o", action="store", required=True, dest="target_path",
                        help="Target path which stores compiled library")
    parser.add_argument("-L", action="store", default=None, dest="ld_path",
                        help="library link path")
    parser.add_argument('--cuda-arch', type=str, default=None, dest='cuda_arch',
                        help='The cuda arch for compiling kernels for')
    parser.add_argument("--config", action="store", required=True, dest="config_path",
                        help="Path which stores the config file")
    arguments = parser.parse_args()

    mod_llvm = tvm.IRModule({})
    mod_cuda = tvm.IRModule({})
    has_cuda = False

    # TODO: attach instruction features to the library, e.g., avx-512, etc.
    for operator_def in __OP_DEF__:
        for sch, args, name in operator_def.invoke_all():
            name = operator_def.get_op_name(name, args)
            if tvm.runtime.module.enabled(get_target(operator_def.target)):
                func_lower = tvm.lower(sch, args,
                                       name=name,
                                       binds=operator_def.get_binds(args))
                if operator_def.target == "cpu":
                    mod = mod_llvm.update(func_lower)
                else:
                    has_cuda = True
                    mod_cuda.update(func_lower)

    lowered_funcs = {get_target("cpu"): mod_llvm}
    if has_cuda > 0:
        lowered_funcs[get_target("cuda")] = mod_cuda
        cuda_arch = get_cuda_arch(arguments.cuda_arch)
        if cuda_arch is None:
            logging.info('No cuda arch specified. TVM will try to detect it from the build platform.')
        else:
            logging.info('Cuda arch {} set for compiling TVM operator kernels.'.format(cuda_arch))
            set_cuda_target_arch(cuda_arch)
    func_binary = tvm.build(lowered_funcs, name="tvmop")
    # we create libtvmop.o first, which gives us chance to link tvm_runtime together with the libtvmop
    # to allow mxnet find external helper functions in libtvm_runtime
    func_binary.save(arguments.target_path + "/libtvmop.o")
    if len(func_binary.imported_modules):
        func_binary.imported_modules[0].save(arguments.target_path + "/libtvmop.cubin")
    ld_path = arguments.target_path if arguments.ld_path is None else arguments.ld_path
    create_shared(arguments.target_path + "/libtvmop.so",
                  arguments.target_path + "/libtvmop.o",
                  options=["-L", ld_path, "-ltvm_runtime"])

    config_spaces = ConfigSpaces()
    for operator_def in __OP_DEF__:
        for config_space, name in operator_def.get_config_spaces():
            config_spaces[name] = ConfigSpace.from_tvm(config_space)
    with open(arguments.config_path, "w") as f:
        json.dump(config_spaces.to_json_dict(), f)
