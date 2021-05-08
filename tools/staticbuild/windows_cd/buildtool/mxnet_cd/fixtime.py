import hashlib
import os

import argparse
from shutil import copyfile

from win32file import CreateFile, SetFileTime, GetFileTime, CloseHandle
from win32file import GENERIC_READ, GENERIC_WRITE, OPEN_EXISTING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root",
                        help="root directory",
                        default='build',
                        type=str)
    args = parser.parse_args()


def build_config(root):
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    save_path = os.path.join(curr_path, "save")

    hasher0 = hashlib.sha1()
    hasher0.update(root.encode("utf-8"))

    save_path = os.path.join(save_path, str(hasher0.hexdigest()))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_build_config = os.path.join(save_path, "build_config.h")

    build_config = os.path.join(root, "mxnet/3rdparty/dmlc-core/include/dmlc/build_config.h")
    if not os.path.exists(build_config):
        return
    if not os.path.exists(save_build_config):
        copyfile(build_config, save_build_config)

    hasher1 = hashlib.sha512()
    with open(build_config, 'rb') as afile:
        buf = afile.read()
        hasher1.update(buf)

    hasher2 = hashlib.sha512()
    with open(save_build_config, 'rb') as afile:
        buf = afile.read()
        hasher2.update(buf)

    if hasher1.hexdigest() == hasher2.hexdigest():
        fh = CreateFile(save_build_config, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, 0)
        create_times, access_times, modify_times = GetFileTime(fh)
        CloseHandle(fh)

        fh = CreateFile(build_config, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, 0)
        SetFileTime(fh, create_times, access_times, modify_times)
        CloseHandle(fh)
    else:
        copyfile(build_config, save_build_config)


if __name__ == '__main__':
    main()
