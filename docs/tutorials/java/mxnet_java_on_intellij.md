# Run MXNet Java Examples Using the IntelliJ IDE (macOS)

This tutorial guides you through setting up a simple Java project in IntelliJ IDE on macOS and demonstrates usage of the MXNet Java APIs. 

## Prerequisites:
To use this tutorial you need the following pre-requisites:

- [Java 8 JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html)
- [Maven](https://maven.apache.org/install.html)
- [OpenCV](https://opencv.org/)
- [IntelliJ IDEA](https://www.jetbrains.com/idea/) (One can download the community edition from [here](https://www.jetbrains.com/idea/download))

### MacOS Prerequisites

**Step 1.** Install brew:
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Or, if you already have brew, update it:
```
brew update
```

**Step 2.** Install Java 8:
```
brew tap caskroom/versions
brew cask install java8
```

**Step 3.** Install maven:
```
brew install maven
```

**Step 4.** Install OpenCV:
```
brew install opencv
```

You can also run this tutorial on an Ubuntu machine after installing the following prerequisites.
### Ubuntu Prerequisites

**Step 1.** Download the MXNet source.

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
cd mxnet
```

**Step 2.** Run the dependency installation scripts.

```bash
sudo ./ci/docker/install/ubuntu_core.sh
sudo ./ci/docker/install/ubuntu_scala.sh
```

The `ubuntu_scala.sh` installs the common dependencies required for both MXNet Scala and MXNet Java packages.

## Set Up Your Project

**Step 1.** Install and setup [IntelliJ IDEA](https://www.jetbrains.com/idea/)

**Step 2.** Create a new Project:

![intellij welcome](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/scala/intellij-welcome.png)

From the IntelliJ welcome screen, select "Create New Project".

Choose the Maven project type. 

Select the checkbox for `Create from archetype`, then choose `org.apache.maven.archetypes:maven-archetype-quickstart` from the list below. More on this can be found on a Maven tutorial : [Maven in 5 Minutes](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). 

![maven project type - archetype](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/java/project-archetype.png)

click `Next`.

![project metadata](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/java/intellij-project-metadata.png)

Set the project's metadata. For this tutorial, use the following:

**GroupId**
```
mxnet
```
**ArtifactId**
```
ArtifactId: javaMXNet
```
**Version**
```
1.0-SNAPSHOT
```

TODO
![project properties](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/java/intellij-project-properties.png)

Review the project's properties. The settings can be left as their default.

TODO
![project location](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/java/intellij-project-location.png)

Set the project's location. The rest of the settings can be left as their default.

TODO
![project 1](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/java/intellij-project-pom.png)

After clicking Finish, you will be presented with the project's first view.
The project's `pom.xml` will be open for editing.

**Step 3.** Add the following Maven dependency to your `pom.xml` file under the `dependencies` tag: 
 
```html
<dependency>
  <groupId>org.apache.mxnet</groupId>
  <artifactId>mxnet-full_2.11-osx-x86_64-cpu</artifactId>
  <version>1.4.0</version>
</dependency>
```

To view the latest MXNet Maven packages, you can check [MXNet Maven package repository](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22org.apache.mxnet%22)


**Step 4.** Import dependencies with Maven:

  - Note the prompt in the lower right corner that states "Maven projects need to be imported". If this is not visible, click on the little greed balloon that appears in the lower right corner.

![import_dependencies](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/java/project-import-changes.png)

Click "Import Changes" in this prompt.

**Step 5.** Build the project:
- To build the project, from the menu choose Build, and then choose Build Project.

**Step 6.** Navigate to the App.java class in the project and paste the following code, overwriting the original hello world code.
```java
package mxnet;

import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;

public class App 
{
    public static void main( String[] args )
    {
        NDArray nd = NDArray.ones(Context.cpu(), new int[] {10, 20});
        System.out.println( "Testing MXNet by generating a 10x20 NDArray" );
        System.out.println("Shape of NDArray is : " + nd.shape());
    }
}
``` 
 
**Step 7.** Now run the App.java by clicking the green arrow as highlighted in the image below.

![run hello mxnet](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/java/intellij-run-projects.png)


The result should be this output:

```
Testing MXNet by generating a 10x20 NDArray
Shape of NDArray is : (10,20)

Process finished with exit code 0
```


### Troubleshooting

If you get an error, check the dependencies at the beginning of this tutorial. For example, you might see the following in the middle of the error messages, where `x.x` would the version it's looking for.

```
...
Library not loaded: /usr/local/opt/opencv/lib/libopencv_calib3d.x.x.dylib
...
```

This can be resolved be installing OpenCV.


### Command Line Build Option

- You can also compile the project by using the following command at the command line. Change directories to this project's root folder then run the following:

```bash
mvn clean install dependency:copy-dependencies
```
If the command succeeds, you should see a lot of info and some warning messages, followed by:

```bash
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time: 3.475 s
[INFO] Finished at: 2018-11-08T05:06:31-08:00
[INFO] ------------------------------------------------------------------------
```
The build generates a new jar file in the `target` folder called `javaMXNet-1.0-SNAPSHOT.jar`.

To run the App.java use the following command from the project's root folder and you should see the same output as we got when the project was run from IntelliJ.
```bash
java -cp target/javaMXNet-1.0-SNAPSHOT.jar:target/dependency/* mxnet.App
```

## Next Steps
For more information about MXNet Java resources, see the following:

* [Java Inference API](https://mxnet.incubator.apache.org/api/java/infer.html)
* [Java Inference Examples](https://github.com/apache/incubator-mxnet/tree/java-api/scala-package/examples/src/main/java/org/apache/mxnetexamples/infer/)
* [MXNet Tutorials Index](http://mxnet.io/tutorials/index.html)
