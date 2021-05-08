---
layout: page
title: Contribute
subtitle: Contribute to the Apache MXNet project
action: Get Started
action_url: /get_started
permalink: /community/index
---
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

# Contributing to MXNet

Apache MXNet (incubating) is a community-led open-source deep learning project. We welcome new members and look forward to your contributions. Here you will find how to stay connected with the MXNet community, get started to contribute, and best practices and processes in MXNet.

## Stay Connected

In MXNet, we have the following communication channels.

| Channel | Purpose |
|---|---|
| [Follow MXNet Development on GitHub](#github-issues) | See what's going on in the MXNet project. |
| [Check out the MXNet Confluence Wiki](https://cwiki.apache.org/confluence/display/MXNET/Apache+MXNet+Home) <i class="fas fa-external-link-alt"> | MXNet developer wiki for information related to project development, maintained by contributors and developers. To request write access, send an email to [send request to the dev list](mailto:dev@mxnet.apache.org?subject=Requesting%20CWiki%20write%20access) <i class="far fa-envelope"></i>. |
| [dev@mxnet.apache.org mailing list](https://lists.apache.org/list.html?dev@mxnet.apache.org) | The "dev list". Discussions about the development of MXNet. To subscribe, send an email to [dev-subscribe@mxnet.apache.org](mailto:dev-subscribe@mxnet.apache.org) <i class="far fa-envelope"></i>. |
| [discuss.mxnet.io](https://discuss.mxnet.io) <i class="fas fa-external-link-alt"></i> | Asking & answering MXNet usage questions. |
| [Apache Slack #mxnet Channel](https://the-asf.slack.com/archives/C7FN4FCP9) <i class="fas fa-external-link-alt"> | Connect with MXNet and other Apache developers. To join the MXNet slack channel [send request to the dev list](mailto:dev@mxnet.apache.org?subject=Requesting%20slack%20access) <i class="far fa-envelope"></i>. |
| [Follow MXNet on Social Media](#social-media) | Get updates about new features and events. |

### Social Media

Keep connected with the latest MXNet news and updates.

<p>
<a href="https://twitter.com/apachemxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/twitter.svg?sanitize=true" height="30px"/> Apache MXNet on Twitter</a>
</p>
<p>
<a href="https://medium.com/apache-mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/medium_black.svg?sanitize=true" height="30px"/> Contributor and user blogs about MXNet</a>
</p>
<p>
<a href="https://reddit.com/r/mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/reddit_blue.svg?sanitize=true" height="30px" alt="reddit"/> Discuss MXNet on r/mxnet</a>
</p>
<p>
<a href="https://www.youtube.com/apachemxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/youtube_red.svg?sanitize=true" height="30px"/> Apache MXNet YouTube channel</a>
</p>
<p>
<a href="https://www.linkedin.com/company/apache-mxnet"><img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/social/linkedin.svg?sanitize=true" height="30px"/> Apache MXNet on LinkedIn</a>
</p>

## Start Contributing

We value all forms of contributions, including, but not limited to:

- Code reviewing of the existing patches.
- Documentation and usage examples
- Community participation in forums and issues.
- Code readability and developer guide
  - We welcome contributions that add code comments
    to improve readability
  - We also welcome contributions to docs to explain the
    design choices of the internal.

- Test cases to make the codebase more robust.
- [Tutorials]({% link pages/community/document.md %}), [blog posts](https://medium.com/apache-mxnet), talks that promote the project.
- [Examples](http://github.com/apache/incubator-mxnet-examples) <i class="fab fa-github"></i> for deep learning applications.


{% include note.html content="Looking for ideas to start contributing? Check out the [good first issues](https://github.com/apache/incubator-mxnet/labels/good%20first%20issue), and [PRs that need review](https://github.com/apache/incubator-mxnet/labels/pr-awaiting-review)" %}
<br/>

### Contribution Guides

- [MXNet Community Guideline]({% link pages/community/community.md %})
- [Write Document and Tutorials]({% link pages/community/document.md %})
- [Committer Guide]({% link pages/community/committer_guide.md %})
- [Submit a Pull Request]({% link pages/community/pull_request.md %})
- [Perform Code Reviews]({% link pages/community/code_review.md %})
- [Code Guide and Tips]({% link pages/community/code_guide.md %})
- [Error Handling Guide]({% link pages/community/error_handling.md %})
- [Git Usage Tips]({% link pages/community/git_howto.md %})


#### RFC Process

Any new features of improvements that are non-trivial should follow the [RFC](https://github.com/apache/incubator-mxnet/issues?q=label%3ARFC+) <i class="fab fa-github"></i> process:

1. [Create an RFC issue on GitHub](https://github.com/apache/incubator-mxnet/issues/new/choose): RFC issues will notify MXNet developer community through all channels, including dev@ list and Slack.
1. [Create the PR on GitHub](https://github.com/apache/incubator-mxnet/pulls) and mention the RFC issue in description.

#### GitHub Issues

Apache MXNet uses GitHub issues to track feature requests and bug reports. [Open a GitHub issue](https://github.com/apache/incubator-mxnet/issues/new/choose) <i class="fas fa-external-link-alt"></i>.

We also use GitHub projects for tracking larger projects, and GitHub milestones for tracking releases.

* [GitHub Projects](https://github.com/apache/incubator-mxnet/projects) <i class="fab fa-github"></i>
* [GitHub Milestones](https://github.com/apache/incubator-mxnet/milestones) <i class="fab fa-github"></i>
* [Roadmaps](https://github.com/apache/incubator-mxnet/labels/Roadmap) <i class="fab fa-github"></i>


The process for setting up MXNet for development depends on several factors, and is constantly being improved and expanded for more development languages. Setup information is on the MXNet Confluence Wiki.

* [MXNet Confluence Wiki: Development](https://cwiki.apache.org/confluence/display/MXNET/Development) <i class="fas fa-external-link-alt"></i>

<br/>
## Contributors

MXNet has been developed by and is used by a group of active community members. Contribute to improving it!

[Contributors and Committers](https://github.com/apache/incubator-mxnet/blob/master/CONTRIBUTORS.md) <i class="fab fa-github"></i>

<br/>

<script defer src="https://use.fontawesome.com/releases/v5.0.12/js/all.js" integrity="sha384-Voup2lBiiyZYkRto2XWqbzxHXwzcm4A5RfdfG6466bu5LqjwwrjXCMBQBLMWh7qR" crossorigin="anonymous"></script>
<script async defer src="https://buttons.github.io/buttons.js"></script>
<script src="https://apis.google.com/js/platform.js"></script>
