if "%VC_YEAR%" == "2017" set VSDEVCMD_ARGS=-vcvars_ver=14.13
if "%VC_YEAR%" == "2017" powershell packaging/windows/internal/vs2017_install.ps1
if errorlevel 1 exit /b 1

call packaging/windows/internal/cuda_install.bat
if errorlevel 1 exit /b 1

call packaging/windows/internal/nightly_defaults.bat Conda
if errorlevel 1 exit /b 1

set PYTORCH_FINAL_PACKAGE_DIR=%CD%\packaging\windows\output
if not exist "%PYTORCH_FINAL_PACKAGE_DIR%" mkdir %PYTORCH_FINAL_PACKAGE_DIR%

bash ./packaging/conda/build_csprng.sh %CUDA_VERSION% %TORCHCSPRNG_BUILD_VERSION% %TORCHCSPRNG_BUILD_NUMBER%
if errorlevel 1 exit /b 1
