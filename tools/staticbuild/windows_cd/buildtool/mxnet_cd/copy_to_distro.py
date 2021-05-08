import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser(description='MXNet Build Pack 7z file')
parser.add_argument('--work-dir', type=str, help='work dir')
parser.add_argument('--distro-dir', type=str, help='distro dir')
parser.add_argument('--name', type=str, help='pack name')
opt = parser.parse_args()


# rmdir /s /q %BASE_DIR%\mxnet-distro\build
# rmdir /s /q %BASE_DIR%\mxnet-distro\mxnet-build
# mkdir %BASE_DIR%\mxnet-distro\mxnet-build
# xcopy mxnet\python %BASE_DIR%\mxnet-distro\mxnet-build\python /E /I /Y
# xcopy mxnet\tools %BASE_DIR%\mxnet-distro\mxnet-build\tools /E /I /Y
# copy build_vc14_gpu_cu80\Release\libmxnet.dll %BASE_DIR%\mxnet-distro\mxnet-build\python\mxnet\libmxnet.dll
# copy build_vc14_gpu_cu80\Release\libmxnet.lib %BASE_DIR%\mxnet-distro\mxnet-build\python\mxnet\libmxnet.lib
# rem %BASE_DIR%\mxnet-compress.exe  %BASE_DIR%\mxnet-distro\mxnet-build\python\mxnet\libmxnet.dll
# cd %BASE_DIR%\mxnet-distro\
# c:
# call .\scripts\win_gpu_cu80.bat#

def main():
    name: str = opt.name
    work_dir = opt.work_dir
    distro_dir = opt.distro_dir
    build_name = "build_{0}".format(name)
    shutil.rmtree(os.path.join(distro_dir, "build"), ignore_errors=True)
    shutil.rmtree(os.path.join(distro_dir, "mxnet-build"), ignore_errors=True)
    shutil.copytree(os.path.join(work_dir, "mxnet", "python", ), os.path.join(distro_dir, "mxnet-build", "python"))
    shutil.copytree(os.path.join(work_dir, "mxnet", "include", ), os.path.join(distro_dir, "mxnet-build", "include"))
    shutil.copytree(os.path.join(work_dir, "mxnet", "tools", ), os.path.join(distro_dir, "mxnet-build", "tools"))
    shutil.copytree(os.path.join(work_dir, "mxnet", "3rdparty/dmlc-core/tracker/dmlc_tracker", ),
                    os.path.join(distro_dir, "mxnet-build", "3rdparty/dmlc-core/tracker/dmlc_tracker"))

    mxnet_path = os.path.join(distro_dir, "mxnet-build", "python", "mxnet")
    shutil.copy(os.path.join(work_dir, build_name, "libmxnet.lib"), mxnet_path)
    shutil.copy(os.path.join(work_dir, build_name, "libmxnet.dll"), mxnet_path)
    dlls = list(glob.iglob(os.path.join(work_dir, build_name, "mxnet_*.dll"), recursive=True))
    for dll in dlls:
        shutil.copy(dll, mxnet_path)

    if name.lower().endswith("mkl"):
        # shutil.copy(os.path.join(work_dir, build_name, "3rdparty", "mkldnn", "src", "mkldnn.dll"), mxnet_path)
        shutil.copy("MKLML_LICENSE ", os.path.join(mxnet_path, "../"))


if __name__ == '__main__':
    main()
