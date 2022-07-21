// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 15:04:22 on Sun, May 29, 2022
//
// Description: hook

#ifndef __CUDA_HOOK_HOOK_H__
#define __CUDA_HOOK_HOOK_H__

#include <dlfcn.h>

#include <string>

#include "macro_common.h"

#define HOOK_GET_SYMBOL(type, symbol_name)                                          \
    do {                                                                            \
        static void *type##_handle = dlopen(s_##type##_dso, RTLD_NOW | RTLD_LOCAL); \
        HOOK_CHECK(type##_handle);                                                  \
        return dlsym(type##_handle, symbol_name.c_str());                           \
    } while (0)

class Hook {
public:
    Hook() = default;
    ~Hook() = default;

    static void *GetCUDASymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(cuda, symbol_name);
    }

    static void *GetNVMLSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(nvml, symbol_name);
    }

    static void *GetCUDARTSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(cudart, symbol_name);
    }

    static void *GetCUDNNSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(cudnn, symbol_name);
    }

    static void *GetCUBLASSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(cublas, symbol_name);
    }

    static void *GetCUBLASLTSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(cublasLt, symbol_name);
    }

    static void *GetCUFFTSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(cufft, symbol_name);
    }

    static void *GetNVTXSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(nvtx, symbol_name);
    }

    static void *GetNVRTCSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(nvrtc, symbol_name);
    }

    static void *GetCURANDSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(curand, symbol_name);
    }

    static void *GetCUSPARSESymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(cusparse, symbol_name);
    }

    static void *GetCUSOLVERSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(cusolver, symbol_name);
    }

    static void *GetNVJPEGSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(nvjpeg, symbol_name);
    }

    static void *GetNVBLASSymbol(const std::string &symbol_name) {
        HOOK_GET_SYMBOL(nvblas, symbol_name);
    }

private:
    // nvidia native cuda dynamic library can be modified to any other unambiguous name, or moved to any path
    static constexpr const char *s_cuda_dso = "/usr/lib/x86_64-linux-gnu/libcuda.so";
    static constexpr const char *s_nvml_dso = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so";
    static constexpr const char *s_cudart_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so";
    static constexpr const char *s_cudnn_dso = "/usr/local/cudnn/lib64/libcudnn.so";
    static constexpr const char *s_cublas_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libcublas.so";
    static constexpr const char *s_cublasLt_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so";
    static constexpr const char *s_cufft_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libcufft.so";
    static constexpr const char *s_nvtx_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libnvToolsExt.so";
    static constexpr const char *s_nvrtc_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so";
    static constexpr const char *s_curand_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libcurand.so";
    static constexpr const char *s_cusparse_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libcusparse.so";
    static constexpr const char *s_cusolver_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libcusolver.so";
    static constexpr const char *s_nvjpeg_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libnvjpeg.so";
    static constexpr const char *s_nvblas_dso = "/usr/local/cuda/targets/x86_64-linux/lib/libnvblas.so";

    HOOK_DISALLOW_COPY_AND_ASSIGN(Hook);
};

#define HOOK_CUDA_SYMBOL(symbol_name) Hook::GetCUDASymbol(symbol_name)
#define HOOK_NVML_SYMBOL(symbol_name) Hook::GetNVMLSymbol(symbol_name)
#define HOOK_CUDART_SYMBOL(symbol_name) Hook::GetCUDARTSymbol(symbol_name)
#define HOOK_CUDNN_SYMBOL(symbol_name) Hook::GetCUDNNSymbol(symbol_name)
#define HOOK_CUBLAS_SYMBOL(symbol_name) Hook::GetCUBLASSymbol(symbol_name)
#define HOOK_CUBLASLT_SYMBOL(symbol_name) Hook::GetCUBLASLTSymbol(symbol_name)
#define HOOK_CUFFT_SYMBOL(symbol_name) Hook::GetCUFFTSymbol(symbol_name)
#define HOOK_NVTX_SYMBOL(symbol_name) Hook::GetNVTXSymbol(symbol_name)
#define HOOK_NVRTC_SYMBOL(symbol_name) Hook::GetNVRTCSymbol(symbol_name)
#define HOOK_CURAND_SYMBOL(symbol_name) Hook::GetCURANDSymbol(symbol_name)
#define HOOK_CUSPARSE_SYMBOL(symbol_name) Hook::GetCUSPARSESymbol(symbol_name)
#define HOOK_CUSOLVER_SYMBOL(symbol_name) Hook::GetCUSOLVERSymbol(symbol_name)
#define HOOK_NVJPEG_SYMBOL(symbol_name) Hook::GetNVJPEGSymbol(symbol_name)
#define HOOK_NVBLAS_SYMBOL(symbol_name) Hook::GetNVBLASSymbol(symbol_name)

#endif  // __CUDA_HOOK_HOOK_H__
