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

# README

Preview at https://mxnet-beta.staged.apache.org/

## [Building the Docs and Website](https://cwiki.apache.org/confluence/display/MXNET/Building+the+New+Website)

## [Technical details for the building the Python microsite](python/README.md)

# Python binding docs

The following guide will help you build a local version of the Python API website,
so that you may work on and test any contributions.

It is recommended that you read the MXNet developer wiki's info on [building the website & docs](https://cwiki.apache.org/confluence/display/MXNET/Building+the+New+Website) as that includes info on how to test and build the site using Docker. The following information should only be used if you can't use Docker or if you're trying to run the site locally.

## Setup

The default configuration requires a GPU and CUDA and expects Ubuntu.
However, you may setup the website on macOS or Windows with or without a GPU.

### Prerequisites

To run the full build, including tests of all tutorials, **you will need at
least two GPUs**. Distributed training is a key feature of MXNet, so multiple
GPUs are required for running through some of the tutorials.

You need to install MXNet, for example, by following the build from source
guide. Further, you need to install the Python requirements listed in the
`requirements` file: 

```bash
python3 -m pip install -r requirements
```

## Build the docs

* Change directories to `python-docs/python`.

To build without GPUs and without testing the notebooks (faster):

```bash
make EVAL=0
```

To build with testing the notebooks (requires GPU):

```bash
make
```

The build docs will be available at `build/_build/html`.

Each build may take a few minutes even without evaluation. To accelerate it, we can use one of the following ways:

1. open `build/conf.py`, add the folders you want to skip into `exclude_patterns`, such as `exclude_patterns = ['templates', 'api', 'develop', 'blog']`.
2. move the files into a different folder, such as `mv api /tmp/`, and then `make clean`.

## Check results

To run a server to see the website:

1. Start a http server: `cd build/_build/html; python -m http.server`
2. For viewing a remote machine, ssh to your machine with port forwarding: `ssh -L8000:localhost:8000 your_machine`
3. Open http://localhost:8000 in your local machine

## Run tutorials

In addition to view the built html pages, you can run the Jupyter notebook from a remote machine.
1. Install `notedown` plugin: `pip install https://github.com/mli/notedown/tarball/master` in remote server
2. Start Jupyter notebook `jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'` in remote server
3. ssh to your machine with port forwarding: `ssh -L8888:localhost:8888 your_machine`
4. Open http://localhost:8888 in your local machine and run the md files directly

Optionally, one can run the following to launch the notedown plugin automatically when starting jupyter notebook.
1. Generate the jupyter configure file `~/.jupyter/jupyter_notebook_config.py` if it
is not existing by run `jupyter notebook --generate-config`
2. Add `c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'` to `~/.jupyter/jupyter_notebook_config.py`
3. Simply run `jupyter notebook`


## Troubleshooting
The `-W` Sphinx option enforces "warnings as errors". This will help you debug your builds and get them through CI.
**CI will not let a PR through if it breaks the website.** Refer to the [MXNet Developer wiki's documentation guide](https://cwiki.apache.org/confluence/display/MXNET/Documentation+Guide) for troubleshooting tips.
