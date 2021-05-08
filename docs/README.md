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

# Building and Updating MXNet Documentation

The website is hosted at https://mxnet.apache.org/.
https://mxnet.io redirects to this site and advised to use links with https://mxnet.apache.org/ instead of https://mxnet.io/.

## Website & Documentation Contributions

Detailed information on website development, continuous integration, and proposals for future projects can be found on the [MXNet Wiki](https://cwiki.apache.org/confluence/display/MXNET/Website).

The website is built using Jekyll. You may run your own version of the static website by following the instructions on the wiki.

Each language documentation is built in a modular way, so that if you are a contributor to Julia, for example, you only need Julia-related tools to build it. Each language API has a section on installation and building along with how to build the docs locally.

You can also use the project's CI tools to emulate any changes with Docker. You can use these tools to install dependencies and run the parts of the build you want to test.

Refer to the [MXNet Developer Wiki](https://cwiki.apache.org/confluence/pages/viewpage.action?pageId=125309983) for instructions on building the docs locally.

If you plan to contribute changes to the documentation or website, please submit a pull request. Contributions are welcome!

## Python Docs

MXNet's Python documentation is built with [Sphinx](https://www.sphinx-doc.org) and a variety of plugins including [pandoc](https://pandoc.org/), and [recommonmark](https://github.com/rtfd/recommonmark).

You can run just the Python docs by following the instructions in the `python_docs/README.md`.

## Other API Docs

The docs are hosted on the website in each language API's section. You can find installation and build instructions there.

## How to Build the MXNet Website for Development and QA

If you only need to make changes to tutorials or other pages that are not generated from one of the API source code folders, then you can [install a pre-build MXNet binary](https://mxnet.apache.org/versions/master/get_started). But if you want edit the API source and have the reference API docs update, you also need to build MXNet from source. Refer to the [build from source instructions for this requirement](https://mxnet.apache.org/versions/master/get_started/build_from_source.html).

### Caveat for Rendering Outputs

Note that without a GPU you will not be able to generate the docs with the outputs in the tutorials.


## Production Website Deployment Process

[Apache Jenkins MXNet website building job](https://builds.apache.org/job/incubator-mxnet-build-site/) is used to build MXNet website.

The Jenkins docs build job will fetch MXNet repository, build MXNet website and push all static files to [host repository](https://github.com/apache/incubator-mxnet-site.git).

The host repo is hooked with [Apache gitbox](https://gitbox.apache.org/repos/asf?p=incubator-mxnet-site.git;a=summary) to host website.

### Processes for Running the Docs Build Jobs

This information is maintained on the [MXNet Wiki](https://cwiki.apache.org/confluence/display/MXNET/Website).


## Other Docs Build Processes

* Perl API docs are maintained separately at [metacpan](https://metacpan.org/release/AI-MXNet).


## Troubleshooting

- If C++ code has been changed, remove the previous results to trigger the rebuild for all pages. To do this, run `make clean_docs`.
- If C++ code fails to build, run `make clean`.
- If CSS or javascript are changed, clear the cache in the browser with a *forced refresh*.
- If search doesn't work, run `make clean` and then `make docs`.
