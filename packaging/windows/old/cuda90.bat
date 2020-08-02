@echo off

IF NOT "%BUILD_CSPRNG%" == "" (
    set MODULE_NAME=csprng
) ELSE (
    set MODULE_NAME=pytorch
)

IF NOT EXIST "setup.py" IF NOT EXIST "%MODULE_NAME%" (
    call internal\clone.bat
    cd ..
    IF ERRORLEVEL 1 goto eof
) ELSE (
    call internal\clean.bat
)

call internal\check_deps.bat
IF ERRORLEVEL 1 goto eof

REM Check for optional components

set NO_CUDA=
set CMAKE_GENERATOR=Visual Studio 15 2017 Win64

IF "%NVTOOLSEXT_PATH%"=="" (
    echo NVTX ^(Visual Studio Extension ^for CUDA^) ^not installed, failing
    exit /b 1
    goto optcheck
)

IF "%CUDA_PATH_V9_0%"=="" (
    echo CUDA 9 not found, failing
    exit /b 1
) ELSE (
    IF "%BUILD_CSPRNG%" == "" (
        set TORCH_CUDA_ARCH_LIST=3.5;5.0+PTX;6.0;7.0
        set TORCH_NVCC_FLAGS=-Xfatbin -compress-all
    ) ELSE (
        set NVCC_FLAGS=-D__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_50,code=compute_50
    )

    set "CUDA_PATH=%CUDA_PATH_V9_0%"
    set "PATH=%CUDA_PATH_V9_0%\bin;%PATH%"
)

:optcheck

IF "%BUILD_CSPRNG%" == "" (
    call internal\check_opts.bat
    IF ERRORLEVEL 1 goto eof

    call internal\copy.bat
    IF ERRORLEVEL 1 goto eof
)

call internal\setup.bat
IF ERRORLEVEL 1 goto eof

:eof
