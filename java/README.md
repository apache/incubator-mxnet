<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

MXNet Java Package
====================
This directory and nested files contain MXNet Java package and language binding.

## Installation
To install requirements for MXNet Java package, visit MXNet [Install Instruction](https://mxnet.apache.org/get_started)

## Running the unit tests
For running unit tests, you will need the [Gradle build system](https://gradle.org/). To install:
 * https://gradle.org/install/

Once ```gradle``` is installed, run the following from MXNet java subdirectory (please make sure the installation path of ```gradle``` is included in your ```$PATH``` environment variable):
```
gradle clean build test -Dorg.gradle.jvmargs=-Xmx2048m
```
