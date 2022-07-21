# CUDA Hook
Hooked CUDA-related dynamic libraries by using automated code generation tools. Based on this, you can easily obtain the CUDA API called by the CUDA program, and you can also hijack the CUDA API to insert custom logic.

It implements an ingenious tool to automatically generate code that hooks the CUDA api with CUDA native header files, and is extremely practical and extensible.

At present, the hooking of dynamic libraries such as cuda driver, nvml, cuda runtime, cudnn, cublas, cublasLt, cufft, nvtx, nvrtc, curand, cusparse, cusolver, nvjpeg and nvblas has been completed, and it can also be easily extended to the hooking of other cuda dynamic libraries.

# Support Dynamic Libraries
- CUDA Driver: libcuda.so
- NVML: libnvidia-ml.so
- CUDA Runtime: libcudart.so
- CUDNN: libcudnn.so
- CUBLAS: libcublas.so
- CUBLASLT: libcublasLt.so
- CUFFT: libcufft.so
- NVTX: libnvToolsExt.so
- NVRTC: libnvrtc.so
- CURAND: libcurand.so
- CUSPARSE: libcusparse.so
- CUSOLVER: libcusolver.so
- NVJPEG: libnvjpeg.so
- NVBLAS: libnvblas.so

# Compile
## Environment
- OS: Linux
- Cmake Version: >= 3.12
- GCC Version: >= 4.8
- CUDA Version: 11.4 (best)
- CUDA Driver Version: 470.129.06 (best)
- CUDNN Version: 7.6.5 (best)

## Clone
```
git clone https://github.com/Bruce-Lee-LY/cuda_hook.git
```

## Build
### GTX1080Ti
```
cd cuda_hook
./build.sh -a 61 -t Release -s ON -b OFF
./build.sh -a 61 -t Debug -s OFF -b ON
```

### Tesla V100
```
cd cuda_hook
./build.sh -a 70 -t Release -s ON -b OFF
./build.sh -a 70 -t Debug -s OFF -b ON
```

### RTX2080Ti
```
cd cuda_hook
./build.sh -a 75 -t Release -s ON -b OFF
./build.sh -a 75 -t Debug -s OFF -b ON
```

### NVIDIA A100
```
cd cuda_hook
./build.sh -a 80 -t Release -s ON -b OFF
./build.sh -a 80 -t Debug -s OFF -b ON
```

### RTX3080Ti / RTX3090 / RTX A6000
```
cd cuda_hook
./build.sh -a 86 -t Release -s ON -b OFF
./build.sh -a 86 -t Debug -s OFF -b ON
```

# Run Sample
```
./run_sample.sh
```

# Tools
## Code Generate
Use CUDA native header files to automatically generate code that hooks CUDA API.
```
cd tools/code_generate
./code_generate.sh
```
