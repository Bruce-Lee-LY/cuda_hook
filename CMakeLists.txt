# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 16:19:40 on Sun, May 29, 2022
#
# Description: cmake for cuda hook

cmake_minimum_required (VERSION 3.12)

project (cuda_hook LANGUAGES C CXX)

set (CMAKE_VERBOSE_MAKEFILE ${HOOK_VERBOSE_MAKEFILE})

set (CMAKE_C_VISIBILITY_PRESET hidden)
set (CMAKE_CXX_VISIBILITY_PRESET hidden)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set (CMAKE_C_FLAGS "-std=c11")
set (CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb -DHOOK_BUILD_DEBUG")
set (CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

set (CMAKE_CXX_FLAGS "-std=c++11")
set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -DHOOK_BUILD_DEBUG")
set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

set (CMAKE_SHARED_LINKER_FLAGS "-s -Wl,--exclude-libs,ALL")
set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

add_compile_options (
    -Wall
    -Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

include_directories (
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/src/cuda
    ${PROJECT_SOURCE_DIR}/src/nvml
    ${PROJECT_SOURCE_DIR}/src/cudart
    ${PROJECT_SOURCE_DIR}/src/cudnn
    ${PROJECT_SOURCE_DIR}/src/cublas
    ${PROJECT_SOURCE_DIR}/src/cublasLt
    ${PROJECT_SOURCE_DIR}/src/cufft
    ${PROJECT_SOURCE_DIR}/src/nvtx
    ${PROJECT_SOURCE_DIR}/src/nvrtc
    ${PROJECT_SOURCE_DIR}/src/curand
    ${PROJECT_SOURCE_DIR}/src/cusparse
    ${PROJECT_SOURCE_DIR}/src/cusolver
    ${PROJECT_SOURCE_DIR}/src/nvjpeg
    ${PROJECT_SOURCE_DIR}/src/nvblas
)

file (GLOB HOOK_SRCS 
    ${PROJECT_SOURCE_DIR}/src/*/*.cpp
)

# libcuda_hook.so
add_library (cuda_hook SHARED ${HOOK_SRCS})
target_link_libraries (cuda_hook -ldl)

install (TARGETS cuda_hook LIBRARY DESTINATION lib64)

# libcuda_hook.a
add_library (cuda_hook_static STATIC ${HOOK_SRCS})
target_link_libraries (cuda_hook_static -ldl)
set_target_properties (cuda_hook_static PROPERTIES OUTPUT_NAME cuda_hook)

install (TARGETS cuda_hook_static ARCHIVE DESTINATION lib64)

if (HOOK_WITH_SAMPLE)
    find_package (CUDA REQUIRED)
    unset (CUDA_USE_STATIC_CUDA_RUNTIME CACHE)
    option (CUDA_USE_STATIC_CUDA_RUNTIME OFF)

    set (CUDA_VERBOSE_BUILD ${HOOK_VERBOSE_MAKEFILE})
    set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES} -g -lineinfo -O0")
    else ()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} --use_fast_math -O3")
    endif ()

    set (SYS_CUDART_PATH "/usr/local/cuda")
    set (SYS_CUDA_DRIVER_PATH "/usr/lib/x86_64-linux-gnu")

    include_directories (
        ${SYS_CUDART_PATH}/include
        ${PROJECT_SOURCE_DIR}/sample/cuda/include
    )

    link_directories (
        ${SYS_CUDART_PATH}/lib64
        ${SYS_CUDA_DRIVER_PATH}
    )

    # cuda
    cuda_add_executable (bandwidth_test ${PROJECT_SOURCE_DIR}/sample/cuda/bandwidth_test.cu)
    cuda_add_executable (matrix_mul ${PROJECT_SOURCE_DIR}/sample/cuda/matrix_mul.cu)
    cuda_add_executable (vector_add ${PROJECT_SOURCE_DIR}/sample/cuda/vector_add.cu)

    install (TARGETS bandwidth_test matrix_mul vector_add RUNTIME DESTINATION sample/cuda)

    # nvml
    add_executable (nvml_example ${PROJECT_SOURCE_DIR}/sample/nvml/nvml_example.c)
    target_link_libraries (nvml_example -lnvidia-ml)
    add_executable (supported_vgpus ${PROJECT_SOURCE_DIR}/sample/nvml/supported_vgpus.c)
    target_link_libraries (supported_vgpus -lnvidia-ml)

    install (TARGETS nvml_example supported_vgpus RUNTIME DESTINATION sample/nvml)
endif ()
