# escape=`

# Use the latest Windows Server Core image with .NET Framework 4.8.
FROM yajiedesign/cuda_100:1.0

# Restore the default Windows shell for correct batch processing.
SHELL ["cmd", "/S", "/C"]

# Download the Build Tools bootstrapper.
COPY work C:\work
COPY mxnet_cd C:\t\mxnet_cd
COPY mxnet-distro C:\t\mxnet-distro
COPY run.bat C:\t\run_demo.bat

VOLUME C:\mount

# Define the entry point for the docker container.
# This entry point starts the developer command prompt and launches the PowerShell shell.
ENTRYPOINT ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]