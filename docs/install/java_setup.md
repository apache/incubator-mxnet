# Setup the MXNet Package for Java

The following instructions are provided for macOS and Ubuntu. Windows is not yet available.

**Note:** If you use IntelliJ or a similar IDE, you may want to follow the [MXNet-Java on IntelliJ tutorial](../tutorials/java/mxnet_java_on_intellij.html) instead of these instructions.

<hr>

## Maven

### Setup Instructions

**Step 1.** Install dependencies:

**macOS Steps**

```bash
brew update
brew tap caskroom/versions
brew cask install java8
brew install opencv
brew install maven
```

**Ubuntu Steps**

These scripts will install Maven and its dependencies. You will be running the Scala scripts because the MXNet-Java project has a dependency on the MXNet-Scala project.

```bash
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_core.sh
wget https://raw.githubusercontent.com/apache/incubator-mxnet/master/ci/docker/install/ubuntu_scala.sh
chmod +x ubuntu_core.sh
chmod +x ubuntu_scala.sh
sudo ./ubuntu_core.sh
sudo ./ubuntu_scala.sh
```

**Step 2.** Run the demo MXNet-Java project.

Go to the [MXNet-Java demo project's README](https://github.com/apache/incubator-mxnet/tree/master/scala-package/mxnet-demo/java-demo) and follow the directions to test the MXNet-Java package installation.

#### Maven Repository

Package information can be found in this [Maven Repository](https://repository.apache.org/#nexus-search;gav~org.apache.mxnet~~1.3.1-SNAPSHOT~~)

**Linux CPU**
```html
<!-- https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-cpu -->
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-cpu</artifactId>
    <scope>system</scope>
    <systemPath>/system/path/to/jar/mxnet-full_2.11-linux-x86_64-cpu-1.3.1-SNAPSHOT.jar</systemPath>
</dependency>
```

**Linux GPU**
```html
<!-- https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-linux-x86_64-gpu -->
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-linux-x86_64-gpu</artifactId>
    <scope>system</scope>
    <systemPath>/system/path/to/jar/mxnet-full_2.11-linux-x86_64-gpu-1.3.1-SNAPSHOT.jar</systemPath>
</dependency>
```

**macOS CPU**
```html
<!-- https://mvnrepository.com/artifact/org.apache.mxnet/mxnet-full_2.11-osx-x86_64-cpu -->
<dependency>
    <groupId>org.apache.mxnet</groupId>
    <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
    <scope>system</scope>
    <systemPath>/system/path/to/jar/mxnet-full_2.11-osx-x86_64-cpu-1.3.1-SNAPSHOT.jar</systemPath>
</dependency>
```

<hr>

## Source

The previously mentioned setup with Maven is recommended. Otherwise, the following instructions for macOS and Ubuntu are provided for reference only:

| OS | Step 1 | Step 2 |
|---|---|---|
|macOS | [Shared Library for macOS](../install/osx_setup.html#build-the-shared-library) | [Scala Package for macOS](http://mxnet.incubator.apache.org/install/osx_setup.html#install-the-mxnet-package-for-scala) |
| Ubuntu | [Shared Library for Ubuntu](../install/ubuntu_setup.html#installing-mxnet-on-ubuntu) | [Scala Package for Ubuntu](http://mxnet.incubator.apache.org/install/ubuntu_setup.html#install-the-mxnet-package-for-scala) |
| Windows | [Shared Library for Windows](../install/windows_setup.html#build-the-shared-library) | <a class="github-button" href="https://github.com/apache/incubator-mxnet/issues/10549" data-size="large" data-show-count="true" aria-label="Issue apache/incubator-mxnet on GitHub">Call for Contribution</a> |


#### Build Java from an Existing MXNet Installation
If you have already built MXNet **from source** and are looking to setup Java from that point, you may simply run the following from the MXNet source root:

```
make scalapkg
make scalainstall
```
This will install both the Java Inference API and the required MXNet-Scala package. 
<hr>

## Documentation

Javadocs are generated as part of the docs build pipeline. You can find them published in the [Java API](../api/java/index.html) section of the website or by going to the [scaladocs output](https://mxnet.incubator.apache.org/api/scala/docs/index.html#org.apache.mxnet.package) directly.

To build the docs yourself, follow the [developer build docs instructions](https://github.com/apache/incubator-mxnet/tree/master/docs/build_version_doc#developer-instructions).

<hr>

## Resources

* [Java API](../api/java/index.html)
* [javadocs](../api/java/docs/index.html#org.apache.mxnet.package)
* [MXNet-Java Tutorials](../../tutorials/index.html#java-tutorials)
