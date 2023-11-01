# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 16:19:40 on Sun, May 29, 2022
#
# Description: compile script

#!/bin/bash

set -euo pipefail

echo "========== build enter =========="

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

CUDA_ARCHITECTURE=86 # a: (Tesla P100: 60, GTX1080Ti: 61, Tesla V100: 70, RTX2080Ti: 75, NVIDIA A100: 80, RTX3080Ti / RTX3090 / RTX A6000: 86, RTX4090: 89, NVIDIA H100: 90)
BUILD_TYPE=Debug # t: (Debug, Release)
WITH_SAMPLE=ON # s: (ON, OFF)
VERBOSE_MAKEFILE=OFF # b: (ON, OFF)

while getopts ":a:t:s:b:" opt
do
    case $opt in
        a)
        CUDA_ARCHITECTURE=$OPTARG
        echo "CUDA_ARCHITECTURE: $CUDA_ARCHITECTURE"
        ;;
        t)
        BUILD_TYPE=$OPTARG
        echo "BUILD_TYPE: $BUILD_TYPE"
        ;;
        s)
        WITH_SAMPLE=$OPTARG
        echo "WITH_SAMPLE: $WITH_SAMPLE"
        ;;
        b)
        VERBOSE_MAKEFILE=$OPTARG
        echo "VERBOSE_MAKEFILE: $VERBOSE_MAKEFILE"
        ;;
        ?)
        echo "invalid param: $OPTARG"
        exit 1
        ;;
    esac
done

echo_cmd() {
    echo $1
    $1
}

echo "========== build cuda_hook =========="

echo_cmd "rm -rf build output"
echo_cmd "mkdir build"

echo_cmd "cd build"
echo_cmd "cmake -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURE -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DHOOK_WITH_SAMPLE=$WITH_SAMPLE -DHOOK_VERBOSE_MAKEFILE=$VERBOSE_MAKEFILE -DCMAKE_INSTALL_PREFIX=$WORK_PATH/output -DCMAKE_SKIP_RPATH=ON .."
echo_cmd "make -j$(nproc --ignore=2)"
echo_cmd "make install"

echo "========== create soft link =========="

# cuda
echo_cmd "ln -s libcuda_hook.so libcuda.so.1"
echo_cmd "ln -s libcuda.so.1 libcuda.so"

# nvml
echo_cmd "ln -s libcuda_hook.so libnvidia-ml.so.1"
echo_cmd "ln -s libnvidia-ml.so.1 libnvidia-ml.so"

# cudart
echo_cmd "ln -s libcuda_hook.so libcudart.so.11.0"
echo_cmd "ln -s libcudart.so.11.0 libcudart.so"

# cudnn
echo_cmd "ln -s libcuda_hook.so libcudnn.so.7"
echo_cmd "ln -s libcudnn.so.7 libcudnn.so"

# cublas
echo_cmd "ln -s libcuda_hook.so libcublas.so.11"
echo_cmd "ln -s libcublas.so.11 libcublas.so"

# cublasLt
echo_cmd "ln -s libcuda_hook.so libcublasLt.so.11"
echo_cmd "ln -s libcublasLt.so.11 libcublasLt.so"

# cufft
echo_cmd "ln -s libcuda_hook.so libcufft.so.10"
echo_cmd "ln -s libcufft.so.10 libcufft.so"

# nvtx
echo_cmd "ln -s libcuda_hook.so libnvToolsExt.so.1"
echo_cmd "ln -s libnvToolsExt.so.1 libnvToolsExt.so"

# nvrtc
echo_cmd "ln -s libcuda_hook.so libnvrtc.so.11.2"
echo_cmd "ln -s libnvrtc.so.11.2 libnvrtc.so"

# curand
echo_cmd "ln -s libcuda_hook.so libcurand.so.10"
echo_cmd "ln -s libcurand.so.10 libcurand.so"

# cusparse
echo_cmd "ln -s libcuda_hook.so libcusparse.so.11"
echo_cmd "ln -s libcusparse.so.11 libcusparse.so"

# cusolver
echo_cmd "ln -s libcuda_hook.so libcusolver.so.11"
echo_cmd "ln -s libcusolver.so.11 libcusolver.so"

# nvjpeg
echo_cmd "ln -s libcuda_hook.so libnvjpeg.so.11"
echo_cmd "ln -s libnvjpeg.so.11 libnvjpeg.so"

# nvblas
echo_cmd "ln -s libcuda_hook.so libnvblas.so.11"
echo_cmd "ln -s libnvblas.so.11 libnvblas.so"

echo_cmd "cp -d *.so *.so.* $WORK_PATH/output/lib64"

echo "========== build info =========="

BRANCH=`git rev-parse --abbrev-ref HEAD`
COMMIT=`git rev-parse HEAD`
GCC_VERSION=`gcc -dumpversion`
COMPILE_TIME=$(date "+%H:%M:%S %Y-%m-%d")

echo "branch: $BRANCH" >> $WORK_PATH/output/hook_version
echo "commit: $COMMIT" >> $WORK_PATH/output/hook_version
echo "gcc_version: $GCC_VERSION" >> $WORK_PATH/output/hook_version
echo "compile_time: $COMPILE_TIME" >> $WORK_PATH/output/hook_version

echo "========== build exit =========="
