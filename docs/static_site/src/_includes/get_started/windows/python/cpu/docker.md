WARNING: the following links and names of binary distributions are provided for
your convenience but they point to packages that are *not* provided nor endorsed
by the Apache Software Foundation. As such, they might contain software
components with more restrictive licenses than the Apache License and you'll
need to decide whether they are appropriate for your usage. Like all Apache
Releases, the official Apache MXNet (incubating) releases consist of source code
only and are found at
the [Download page](https://mxnet.apache.org/get_started/download).
    

Docker images with *MXNet* are available at [DockerHub](https://hub.docker.com/r/mxnet/).
After you installed Docker on your machine, you can use them via:

{% highlight bash %}
$ docker pull mxnet/python
{% endhighlight %}

You can list docker images to see if mxnet/python docker image pull was successful.

{% highlight bash %}
$ docker images # Use sudo if you skip Step 2

REPOSITORY TAG IMAGE ID CREATED SIZE
mxnet/python latest 00d026968b3c 3 weeks ago 1.41 GB
{% endhighlight %}

You can then <a href="/get_started/validate_mxnet.html">validate the installation</a>.
