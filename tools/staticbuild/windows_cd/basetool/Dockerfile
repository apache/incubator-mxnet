# escape=`

# Use the latest Windows Server Core image with .NET Framework 4.8.
FROM yajiedesign/msvcbuildtools2019:20200729

# Restore the default Windows shell for correct batch processing.
SHELL ["cmd", "/S", "/C"]

# Download the Build Tools bootstrapper.
COPY openblas C:\dep\openblas
COPY opencv_vc141 C:\dep\opencv_vc141 
COPY cmake-3.16.4-win64-x64 C:\t\cmake
COPY ninja C:\t\ninja
COPY pandoc-2.10.1 C:\t\pandoc
COPY nvcuda.dll C:\Windows\System32\
COPY python-3.8.5-amd64.exe C:\TMP\
COPY Git-2.28.0-64-bit.exe C:\TMP\

RUN START /WAIT C:\TMP\python-3.8.5-amd64.exe /quiet InstallAllUsers=1 DefaultAllUsersTargetDir=C:\Python && `
    START /WAIT C:\TMP\Git-2.28.0-64-bit.exe /SILENT /NORESTART  && `
    set PATH=%PATH%;C:\Python;C:\Python\Scripts;C:\Python\Library\bin && `
    setx /M PATH "%PATH%;C:\t\cmake\bin;C:\t\ninja;C:\Python;C:\Python\Scripts;C:\Python\Library\bin;C:\Program Files\Git\bin;C:\t\pandoc" && `
    del /Q C:\TMP\python-3.8.5-amd64.exe && `
    del /Q C:\TMP\Git-2.28.0-64-bit.exe && `
    python -m pip install --upgrade pip && `
    pip.exe install requests -i https://mirrors.aliyun.com/pypi/simple/ && `
    pip.exe install pypandoc -i https://mirrors.aliyun.com/pypi/simple/ && `
    pip.exe install numpy -i https://mirrors.aliyun.com/pypi/simple/



# Define the entry point for the docker container.
# This entry point starts the developer command prompt and launches the PowerShell shell.
# ENTRYPOINT ["C:\\BuildTools\\Common7\\Tools\\VsDevCmd.bat", "&&", "powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]
ENTRYPOINT ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]