# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:56:07 on Sat, May 28, 2022
#
# Description: code generate script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

# python3 -m CppHeaderParser.tojson include/cuda.h > output/cuda.json
python3 code_generate.py -t cuda -f include/cuda.h -o output
# mkdir -p ../../src/cuda
# cp output/cuda_hook.cpp ../../src/cuda

# python3 -m CppHeaderParser.tojson include/nvml.h > output/nvml.json
python3 code_generate.py -t nvml -f include/nvml.h -o output
# mkdir -p ../../src/nvml
# cp output/nvml_hook.cpp ../../src/nvml

# python3 -m CppHeaderParser.tojson include/cuda_runtime_api.h > output/cuda_runtime_api.json
python3 code_generate.py -t cudart -f include/cuda_runtime_api.h -o output
# mkdir -p ../../src/cudart
# cp output/cudart_hook.cpp ../../src/cudart

# python3 -m CppHeaderParser.tojson include/cudnn.h > output/cudnn.json
python3 code_generate.py -t cudnn -f include/cudnn.h -o output
# mkdir -p ../../src/cudnn
# cp output/cudnn_hook.cpp ../../src/cudnn

# python3 -m CppHeaderParser.tojson include/cublas.h > output/cublas.json
python3 code_generate.py -t cublas -f include/cublas.h -o output
# mkdir -p ../../src/cublas
# cp output/cublas_hook.cpp ../../src/cublas

# python3 -m CppHeaderParser.tojson include/cublasLt.h > output/cublasLt.json
python3 code_generate.py -t cublasLt -f include/cublasLt.h -o output
# mkdir -p ../../src/cublasLt
# cp output/cublasLt_hook.cpp ../../src/cublasLt

# python3 -m CppHeaderParser.tojson include/cufft.h > output/cufft.json
python3 code_generate.py -t cufft -f include/cufft.h -o output
# mkdir -p ../../src/cufft
# cp output/cufft_hook.cpp ../../src/cufft

# python3 -m CppHeaderParser.tojson include/nvToolsExt.h > output/nvToolsExt.json
python3 code_generate.py -t nvtx -f include/nvToolsExt.h -o output
# mkdir -p ../../src/nvtx
# cp output/nvtx_hook.cpp ../../src/nvtx

# python3 -m CppHeaderParser.tojson include/nvrtc.h > output/nvrtc.json
python3 code_generate.py -t nvrtc -f include/nvrtc.h -o output
# mkdir -p ../../src/nvrtc
# cp output/nvrtc_hook.cpp ../../src/nvrtc

# python3 -m CppHeaderParser.tojson include/curand.h > output/curand.json
python3 code_generate.py -t curand -f include/curand.h -o output
# mkdir -p ../../src/curand
# cp output/curand_hook.cpp ../../src/curand

# python3 -m CppHeaderParser.tojson include/cusparse.h > output/cusparse.json
python3 code_generate.py -t cusparse -f include/cusparse.h -o output
# mkdir -p ../../src/cusparse
# cp output/cusparse_hook.cpp ../../src/cusparse

# python3 -m CppHeaderParser.tojson include/cusolver_common.h > output/cusolver_common.json
python3 code_generate.py -t cusolver -f include/cusolver_common.h -o output
# mkdir -p ../../src/cusolver
# cp output/cusolver_hook.cpp ../../src/cusolver

# python3 -m CppHeaderParser.tojson include/nvjpeg.h > output/nvjpeg.json
python3 code_generate.py -t nvjpeg -f include/nvjpeg.h -o output
# mkdir -p ../../src/nvjpeg
# cp output/nvjpeg_hook.cpp ../../src/nvjpeg

# python3 -m CppHeaderParser.tojson include/nvblas.h > output/nvblas.json
python3 code_generate.py -t nvblas -f include/nvblas.h -o output
# mkdir -p ../../src/nvblas
# cp output/nvblas_hook.cpp ../../src/nvblas
