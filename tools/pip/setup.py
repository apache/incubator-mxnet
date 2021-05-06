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
# pylint: disable=invalid-name, exec-used
"""Setup mxnet package for pip."""
from datetime import datetime
import os
import sys
import shutil
import platform
from setuptools import setup, find_packages

if platform.system() == 'Linux':
    sys.argv.append('--python-tag')
    sys.argv.append('py3')
    sys.argv.append('--plat-name=manylinux2014_'+platform.machine())
elif platform.system() == 'Darwin':
    sys.argv.append('--python-tag')
    sys.argv.append('py3')
    sys.argv.append('--plat-name=macosx_10_13_x86_64')

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'mxnet-build/python/mxnet/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']

# set by the CD pipeline
is_release = os.environ.get("IS_RELEASE", "").strip()

# set by the travis build pipeline
travis_tag = os.environ.get("TRAVIS_TAG", "").strip()

# nightly build tag
if not travis_tag and not is_release:
    __version__ += 'b{0}'.format(datetime.today().strftime('%Y%m%d'))

# patch build tag
elif travis_tag.startswith('patch-'):
    __version__ = os.environ['TRAVIS_TAG'].split('-')[1]


DEPENDENCIES = [
    'numpy<2.0.0,>1.16.0',
    'requests>=2.20.0,<3',
    'graphviz<0.9.0,>=0.8.1'
]

shutil.rmtree(os.path.join(CURRENT_DIR, 'mxnet'), ignore_errors=True)
shutil.rmtree(os.path.join(CURRENT_DIR, 'dmlc_tracker'), ignore_errors=True)
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/python/mxnet'),
                os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/dmlc-core/tracker/dmlc_tracker'),
                os.path.join(CURRENT_DIR, 'dmlc_tracker'))
shutil.copy(LIB_PATH[0], os.path.join(CURRENT_DIR, 'mxnet'))

package_data = {'mxnet': [os.path.join('mxnet', os.path.basename(LIB_PATH[0]))],
                'dmlc_tracker': []}
if platform.system() == 'Linux':
    libdir, mxdir = os.path.dirname(LIB_PATH[0]), os.path.join(CURRENT_DIR, 'mxnet')
    if os.path.exists(os.path.join(libdir, 'libgfortran.so.3')):
        shutil.copy(os.path.join(libdir, 'libgfortran.so.3'), mxdir)
        package_data['mxnet'].append('mxnet/libgfortran.so.3')
    elif os.path.exists(os.path.join(libdir, 'libgfortran.so.4')):
        shutil.copy(os.path.join(libdir, 'libgfortran.so.4'), mxdir)
        package_data['mxnet'].append('mxnet/libgfortran.so.4')
    else:
        shutil.copy(os.path.join(libdir, 'libgfortran.so.5'), mxdir)
        package_data['mxnet'].append('mxnet/libgfortran.so.5')
    if os.path.exists(os.path.join(libdir, 'libopenblas.so.0')):
        shutil.copy(os.path.join(libdir, 'libopenblas.so.0'), mxdir)
        package_data['mxnet'].append('mxnet/libopenblas.so.0')

# copy license and notice
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/licenses'),
            os.path.join(CURRENT_DIR, 'mxnet/licenses'))

# copy tools to mxnet package
shutil.rmtree(os.path.join(CURRENT_DIR, 'mxnet/tools'), ignore_errors=True)
os.mkdir(os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/launch.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/im2rec.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/kill-mxnet.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/parse_log.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/tools/diagnose.py'), os.path.join(CURRENT_DIR, 'mxnet/tools'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/tools/caffe_converter'), os.path.join(CURRENT_DIR, 'mxnet/tools/caffe_converter'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/tools/bandwidth'), os.path.join(CURRENT_DIR, 'mxnet/tools/bandwidth'))

# copy headers to mxnet package
shutil.rmtree(os.path.join(CURRENT_DIR, 'mxnet/include'), ignore_errors=True)
os.mkdir(os.path.join(CURRENT_DIR, 'mxnet/include'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/include/mxnet'),
                os.path.join(CURRENT_DIR, 'mxnet/include/mxnet'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/dlpack/include/dlpack'),
                os.path.join(CURRENT_DIR, 'mxnet/include/dlpack'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/dmlc-core/include/dmlc'),
                os.path.join(CURRENT_DIR, 'mxnet/include/dmlc'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/mshadow/mshadow'),
                os.path.join(CURRENT_DIR, 'mxnet/include/mshadow'))
shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/tvm/nnvm/include/nnvm'),
                os.path.join(CURRENT_DIR, 'mxnet/include/nnvm'))

# copy cc file for mxnet extensions
os.mkdir(os.path.join(CURRENT_DIR, 'mxnet/src'))
shutil.copy(os.path.join(CURRENT_DIR, 'mxnet-build/src/lib_api.cc'),
            os.path.join(CURRENT_DIR, 'mxnet/src'))

package_name = 'mxnet'

variant = os.environ['mxnet_variant'].upper()
if variant != 'CPU':
    package_name = 'mxnet_{0}'.format(variant.lower())

def skip_markdown_comments(md):
    lines = md.splitlines()
    for i in range(len(lines)):
        if lines[i].strip():
            if not lines[i].startswith('<!--') or not lines[i].endswith('-->'):
                return '\n'.join(lines[i:])

with open('doc/PYPI_README.md') as readme_file:
    long_description = skip_markdown_comments(readme_file.read())

with open('doc/{0}_ADDITIONAL.md'.format(variant)) as variant_doc:
    long_description = long_description + skip_markdown_comments(variant_doc.read())

short_description = 'Apache MXNet is an ultra-scalable deep learning framework.'
libraries = []
if variant == 'CPU':
    libraries.append('openblas')
else:
    if variant.startswith('CU112'):
        libraries.append('CUDA-11.2')
    elif variant.startswith('CU110'):
        libraries.append('CUDA-11.0')
    elif variant.startswith('CU102'):
        libraries.append('CUDA-10.2')
    elif variant.startswith('CU101'):
        libraries.append('CUDA-10.1')
    elif variant.startswith('CU100'):
        libraries.append('CUDA-10.0')

from mxnet.runtime import Features
if Features().is_enabled("MKLDNN"):
    libraries.append('MKLDNN')

short_description += ' This version uses {0}.'.format(' and '.join(libraries))

if Features().is_enabled("MKLDNN"):
    shutil.copytree(os.path.join(CURRENT_DIR, 'mxnet-build/3rdparty/mkldnn/include'),
                    os.path.join(CURRENT_DIR, 'mxnet/include/mkldnn'))

# Copy licenses and notice
for f in os.listdir('mxnet/licenses'):
  package_data['mxnet'].append('mxnet/licenses/{}'.format(f))

from mxnet.base import _generate_op_module_signature
from mxnet.ndarray.register import _generate_ndarray_function_code
from mxnet.symbol.register import _generate_symbol_function_code
_generate_op_module_signature('mxnet', 'symbol', _generate_symbol_function_code)
_generate_op_module_signature('mxnet', 'ndarray', _generate_ndarray_function_code)

setup(name=package_name,
      version=__version__,
      long_description=long_description,
      long_description_content_type='text/markdown',
      description=short_description,
      zip_safe=False,
      packages=find_packages(),
      package_data=package_data,
      include_package_data=True,
      install_requires=DEPENDENCIES,
      license='Apache 2.0',
      classifiers=[ # https://pypi.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: C++',
          'Programming Language :: Cython',
          'Programming Language :: Other',  # R, Scala
          'Programming Language :: Perl',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      url='https://github.com/apache/incubator-mxnet')

