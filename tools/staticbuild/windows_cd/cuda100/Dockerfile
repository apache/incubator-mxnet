# escape=`

# Use the latest Windows Server Core image with .NET Framework 4.8.
FROM yajiedesign/basetool:20200730

# Restore the default Windows shell for correct batch processing.
SHELL ["cmd", "/S", "/C"]

# Download the Build Tools bootstrapper.
COPY v10.0 C:\CUDA_V100

RUN setx /M PATH "%PATH%;C:\CUDA_V100\bin" && `
    setx /M CUDA_PATH "C:\CUDA_V100"

VOLUME C:\mount
# Define the entry point for the docker container.
# This entry point starts the developer command prompt and launches the PowerShell shell.
ENTRYPOINT ["powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]