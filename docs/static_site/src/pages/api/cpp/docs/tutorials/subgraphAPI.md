---
layout: page_api
title: Subgraph API
action: Get Started
action_url: /get_started
permalink: /api/cpp/docs/tutorials/subgraph_api.html
is_tutorial: true
tag: cpp
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

## Subgraph API

The subgraph API has been proposed and implemented as the default mechanism for integrating backend libraries to MXNet. The subgraph API is a very flexible interface. Although it was proposed as an integration mechanism, it has been used as a tool for manipulating NNVM graphs for graph-level optimizations, such as operator fusion.

The subgraph API works as the following steps:

* Search for particular patterns in a graph.
* Group the operators/nodes with particular patterns into a subgraph and shrink the subgraph into a single node.
* Replace the subgraph in the original graph with the subgraph node.

The figure below illustrates the subgraph mechanism.

![](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tutorials/subgraph/subgraph.png)

The subgraph API allows the backend developers to customize the subgraph mechanism in two places:

* Subgraph searching: define a subgraph selector to search for particular patterns in a computation graph.
* Subgraph node creation: attach an operator to run the computation in the subgraph. We can potentially manipulate the subgraph here.


The following is a demonstration of how the subgraph API can be applied to a simple task. Refer to the previous figure for an overview of the process. That is, replacing `Convolution` and `BatchNorm` with the conv_bn.

The first step is to define a subgraph selector to find the required pattern. To find a pattern that has `Convolution` and `BatchNorm`, we can start the search on the node with `Convolution`. Then from the `Convolution` node, we search for `BatchNorm` along the outgoing edge.

```c++
class SgSelector : public SubgraphSelector {
 public:
  SgSelector() {
    find_bn = false;
  }
  bool Select(const nnvm::Node &n) override {
    // Here we start on the Convolution node to search for a subgraph.
    return n.op() && n.op()->name == "Convolution";
  }
  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    // We don't need to search on the incoming edge.
    return false;
  }
  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    // We search on the outgoing edge. Once we find a BatchNorm node, we won't
    // accept any more BatchNorm nodes.
    if (new_node.op() && new_node.op()->name == "BatchNorm" && !find_bn) {
      find_bn = true;
      return true;
    } else {
      return false;
    }
  }
  std::vector<nnvm::Node *> Filter(const std::vector<nnvm::Node *> &candidates) override {
    // We might have found a Convolution node, but we might have failed to find a BatchNorm
    // node that uses the output of the Convolution node. If we failed, we should skip
    // the Convolution node as well.
    if (find_bn)
      return candidates;
    else
      return std::vector<nnvm::Node *>();
  }
 private:
  bool find_bn;
};
```

The second step is to define a subgraph property to use the subgraph selector above to customize the subgraph searching. By defining this class, we can also customize subgraph node creation. When customizing node creation, we can specify what operator to run the subgraph on the node. In this example, we use `CachedOp`, which itself is a graph executor, to run the subgraph with `Convolution` and `BatchNorm`. In practice, it's most likely that we use a single operator from a backend library to replace the two operators for execution.

{% raw %}
```c++
class SgProperty : public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() {
    return std::make_shared<SgProperty>();
  }
  nnvm::ObjectPtr CreateSubgraphNode(
      const nnvm::Symbol &sym, const int subgraph_id = 0) const override {
    // We can use CachedOp to execute the subgraph.
    nnvm::ObjectPtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_CachedOp");
    n->attrs.name = "ConvBN" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));
    std::vector<std::pair<std::string, std::string> > flags{{"static_alloc", "true"}};
    n->attrs.parsed = CachedOpPtr(new CachedOp(sym, flags));
    return n;
  }
  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto property = std::make_shared<CreateSubgraphSelector>();
    property->SetAttr<std::string>("property_name", "subgraph example pass"); // Optional, better to have it.
    property->SetAttr<bool>("inference_only", true); // Optional, only for inference_only pass.
    return property;
  }
};
```
{% endraw %}
`SetAttr` is optional and developer can define their own attributes to control property behavior.
There're some built-in attributes that used by MXNet executor.

`property_name`  : std::string, name of this property, used for diagnose.

`disable` : bool, whther to disable this property.

`inference_only` : bool, apply this property only for inference. Property will be skiped when need_grad=True. Default `false` if this attribute isn't defined.

After defining the subgraph property, we need to register it under a backend in .cc file.

Firstly, we need to register the backend

```C++
MXNET_REGISTER_SUBGRAPH_BACKEND(SgTest);
```

Then register the property under it.

```c++
MXNET_REGISTER_SUBGRAPH_PROPERTY(SgTest, SgProperty);
```

It's possible to register multiple properties for same backend. In practice, we recommend to put each property definition into .h file, and register backend in single .cc file. Property will be executed according to the register order.

```c++
#include "SgProperty.h" // Define SgProperty class
#include "SgProperty2.h" // Define SgProperty2 class
#include "SgProperty3.h" // Define SgProperty3 class

MXNET_REGISTER_SUBGRAPH_BACKEND(SgTest);
MXNET_REGISTER_SUBGRAPH_PROPERTY(SgTest, SgProperty);  // Execution order 1.
MXNET_REGISTER_SUBGRAPH_PROPERTY(SgTest, SgProperty2); // Execution order 2.
MXNET_REGISTER_SUBGRAPH_PROPERTY(SgTest, SgProperty3); // Execution order 3.
```

After compiling this subgraph mechanism into MXNet, we can use the environment variable `MXNET_SUBGRAPH_BACKEND` to activate it during symbol bind.

```bash
export MXNET_SUBGRAPH_BACKEND=SgTest
```

Or you can use python symbol API `get_backend_symbol` to run all properties registered for this backend and get returned symbol.

```python
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
sym = sym.get_backend_symbol('SgTest')
```

When `SgProperty` is activated, a message will be shown in terminal as

```bash
start to execute subgraph example pass.
```

This tutorial shows a simple example of how to use the subgraph API to search for patterns in an NNVM graph.
Intested users can try different pattern matching rules (i.e., define their own `SubgraphSelector`) and
attach different operators to execute the subgraphs.

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
