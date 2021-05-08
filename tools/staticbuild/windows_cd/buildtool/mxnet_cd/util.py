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

import os
import contextlib
import logging
import requests


def get_mxnet_root() -> str:
    curpath = os.path.abspath(os.path.dirname(__file__))

    def is_mxnet_root(path: str) -> bool:
        return os.path.exists(os.path.join(path, ".mxnet_root"))

    while not is_mxnet_root(curpath):
        parent = os.path.abspath(os.path.join(curpath, os.pardir))
        if parent == curpath:
            raise RuntimeError("Got to the root and couldn't find a parent folder with .mxnet_root")
        curpath = parent
    return curpath


@contextlib.contextmanager
def remember_cwd():
    """
    Restore current directory when exiting context
    """
    curdir = os.getcwd()
    try:
        yield
    finally:
        os.chdir(curdir)


def retry(target_exception, tries=4, delay_s=1, backoff=2):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param target_exception: the exception to check. may be a tuple of
        exceptions to check
    :type target_exception: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay_s: initial delay between retries in seconds
    :type delay_s: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    """
    import time
    from functools import wraps

    def decorated_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay_s
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except target_exception as e:
                    logging.warning("Exception: %s, Retrying in %d seconds...", str(e), mdelay)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return decorated_retry


# noinspection SyntaxError
def under_ci() -> bool:
    """:return: True if we run in Jenkins."""
    return 'JOB_NAME' in os.environ


def ec2_instance_id_hostname() -> str:
    if under_ci():
        result = []
        try:
            r = requests.get("http://instance-data/latest/meta-data/instance-id")
            if r.status_code == 200:
                result.append(r.content.decode())
            r = requests.get("http://instance-data/latest/meta-data/public-hostname")
            if r.status_code == 200:
                result.append(r.content.decode())
            return ' '.join(result)
        except ConnectionError:
            pass
        return '?'
    else:
        return ''


def chdir_to_script_directory():
    # We need to be in the same directory than the script so the commands in the dockerfiles work as
    # expected. But the script can be invoked from a different path
    base = os.path.split(os.path.realpath(__file__))[0]
    os.chdir(base)
