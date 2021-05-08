cd c:\t\mxnet_cd\
python build_windows.py -r C:\work -f WIN_GPU --name vc141_gpu_cu100 --build_timestamp 20200730 --vcvars VS_2017
python copy_to_distro.py --work-dir=C:\work  --distro-dir=c:\t\mxnet-distro --name=vc141_gpu_cu100 
cd c:\t\mxnet-distro
c:\t\mxnet-distro\scripts\win_gpu_cu100.bat
xcopy /S c:\t\mxnet-distro\dist c:\mount\dist\