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

7z x -y windows_package.7z

$env:MXNET_LIBRARY_PATH=join-path $pwd.Path windows_package\lib\libmxnet.dll
$env:PYTHONPATH=join-path $pwd.Path windows_package\python
$env:MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
$env:MXNET_SUBGRAPH_VERBOSE=0
$env:MXNET_HOME=[io.path]::combine($PSScriptRoot, 'mxnet_home')

C:\Python37\Scripts\pip install -r ci\docker\install\requirements
C:\Python37\python.exe -m pytest -v --durations=50 --cov-report xml:tests_unittest.xml tests\python\unittest
if ($LastExitCode -ne 0) { Throw ("Error running unittest, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
C:\Python37\python.exe -m pytest -v --durations=50 --cov-report xml:tests_operator.xml tests\python\gpu\test_operator_gpu.py
if ($LastExitCode -ne 0) { Throw ("Error running tests, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
C:\Python37\python.exe -m pytest -v --durations=50 --cov-report xml:tests_forward.xml tests\python\gpu\test_forward.py
if ($LastExitCode -ne 0) { Throw ("Error running tests, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
C:\Python37\python.exe -m pytest -v --durations=50 --cov-report xml:tests_train.xml tests\python\train
if ($LastExitCode -ne 0) { Throw ("Error running tests, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
# Adding this extra test since it's not possible to set env var on the fly in Windows.
$env:MXNET_SAFE_ACCUMULATION=1
C:\Python37\python.exe -m pytest -v --durations=50 --cov-report xml:tests_operator.xml --cov-append tests\python\gpu\test_operator_gpu.py::test_norm
if ($LastExitCode -ne 0) { Throw ("Error running tests, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
C:\Python37\python.exe -m pytest -v --durations=50 --cov-report xml:tests_tvm_op.xml tests\python\gpu\test_tvm_op_gpu.py
if ($LastExitCode -ne 0) { Throw ("Error running TVM op tests, python exited with status code " + ('{0:X}' -f $LastExitCode)) }
