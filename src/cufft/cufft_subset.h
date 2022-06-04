// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: cufft subset

#ifndef __CUDA_HOOK_CUFFT_SUBSET_H__
#define __CUDA_HOOK_CUFFT_SUBSET_H__

#include "cublas_subset.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CUFFT_VER_MAJOR 10
#define CUFFT_VER_MINOR 5
#define CUFFT_VER_PATCH 2
#define CUFFT_VER_BUILD 100

// cuFFT library version
//
// CUFFT_VERSION / 1000 - major version
// CUFFT_VERSION / 100 % 100 - minor version
// CUFFT_VERSION % 100 - patch level
#define CUFFT_VERSION 10502

// CUFFT API function return values
typedef enum cufftResult_t {
    CUFFT_SUCCESS = 0x0,
    CUFFT_INVALID_PLAN = 0x1,
    CUFFT_ALLOC_FAILED = 0x2,
    CUFFT_INVALID_TYPE = 0x3,
    CUFFT_INVALID_VALUE = 0x4,
    CUFFT_INTERNAL_ERROR = 0x5,
    CUFFT_EXEC_FAILED = 0x6,
    CUFFT_SETUP_FAILED = 0x7,
    CUFFT_INVALID_SIZE = 0x8,
    CUFFT_UNALIGNED_DATA = 0x9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
    CUFFT_INVALID_DEVICE = 0xB,
    CUFFT_PARSE_ERROR = 0xC,
    CUFFT_NO_WORKSPACE = 0xD,
    CUFFT_NOT_IMPLEMENTED = 0xE,
    CUFFT_LICENSE_ERROR = 0x0F,
    CUFFT_NOT_SUPPORTED = 0x10

} cufftResult;

#define MAX_CUFFT_ERROR 0x11

// CUFFT defines and supports the following data types

// cufftReal is a single-precision, floating-point real data type.
// cufftDoubleReal is a double-precision, real data type.
typedef float cufftReal;
typedef double cufftDoubleReal;

// cufftComplex is a single-precision, floating-point complex data type that
// consists of interleaved real and imaginary components.
// cufftDoubleComplex is the double-precision equivalent.
typedef cuComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;

// CUFFT transform directions
#define CUFFT_FORWARD -1  // Forward FFT
#define CUFFT_INVERSE 1   // Inverse FFT

// CUFFT supports the following transform types
typedef enum cufftType_t {
    CUFFT_R2C = 0x2a,  // Real to Complex (interleaved)
    CUFFT_C2R = 0x2c,  // Complex (interleaved) to Real
    CUFFT_C2C = 0x29,  // Complex to Complex, interleaved
    CUFFT_D2Z = 0x6a,  // Double to Double-Complex
    CUFFT_Z2D = 0x6c,  // Double-Complex to Double
    CUFFT_Z2Z = 0x69   // Double-Complex to Double-Complex
} cufftType;

// CUFFT supports the following data layouts
typedef enum cufftCompatibility_t {
    CUFFT_COMPATIBILITY_FFTW_PADDING = 0x01  // The default value
} cufftCompatibility;

#define CUFFT_COMPATIBILITY_DEFAULT CUFFT_COMPATIBILITY_FFTW_PADDING

//
// structure definition used by the shim between old and new APIs
//
#define MAX_SHIM_RANK 3

// cufftHandle is a handle type used to store and access CUFFT plans.
typedef int cufftHandle;

//
// cufftXtSubFormat identifies the data layout of
// a memory descriptor owned by cufft.
// note that multi GPU cufft does not yet support out-of-place transforms
//

typedef enum cufftXtSubFormat_t {
    CUFFT_XT_FORMAT_INPUT = 0x00,              // by default input is in linear order across GPUs
    CUFFT_XT_FORMAT_OUTPUT = 0x01,             // by default output is in scrambled order depending on transform
    CUFFT_XT_FORMAT_INPLACE = 0x02,            // by default inplace is input order, which is linear across GPUs
    CUFFT_XT_FORMAT_INPLACE_SHUFFLED = 0x03,   // shuffled output order after execution of the transform
    CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04,  // shuffled input order prior to execution of 1D transforms
    CUFFT_FORMAT_UNDEFINED = 0x05
} cufftXtSubFormat;

//
// cufftXtCopyType specifies the type of copy for cufftXtMemcpy
//
typedef enum cufftXtCopyType_t {
    CUFFT_COPY_HOST_TO_DEVICE = 0x00,
    CUFFT_COPY_DEVICE_TO_HOST = 0x01,
    CUFFT_COPY_DEVICE_TO_DEVICE = 0x02,
    CUFFT_COPY_UNDEFINED = 0x03
} cufftXtCopyType;

//
// cufftXtQueryType specifies the type of query for cufftXtQueryPlan
//
typedef enum cufftXtQueryType_t { CUFFT_QUERY_1D_FACTORS = 0x00, CUFFT_QUERY_UNDEFINED = 0x01 } cufftXtQueryType;

typedef struct cufftXt1dFactors_t {
    long long int size;
    long long int stringCount;
    long long int stringLength;
    long long int substringLength;
    long long int factor1;
    long long int factor2;
    long long int stringMask;
    long long int substringMask;
    long long int factor1Mask;
    long long int factor2Mask;
    int stringShift;
    int substringShift;
    int factor1Shift;
    int factor2Shift;
} cufftXt1dFactors;

//
// cufftXtWorkAreaPolicy specifies policy for cufftXtSetWorkAreaPolicy
//
typedef enum cufftXtWorkAreaPolicy_t {
    CUFFT_WORKAREA_MINIMAL = 0,     /* maximum reduction */
    CUFFT_WORKAREA_USER = 1,        /* use workSize parameter as limit */
    CUFFT_WORKAREA_PERFORMANCE = 2, /* default - 1x overhead or more, maximum performance */
} cufftXtWorkAreaPolicy;

// callbacks

typedef enum cufftXtCallbackType_t {
    CUFFT_CB_LD_COMPLEX = 0x0,
    CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    CUFFT_CB_LD_REAL = 0x2,
    CUFFT_CB_LD_REAL_DOUBLE = 0x3,
    CUFFT_CB_ST_COMPLEX = 0x4,
    CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    CUFFT_CB_ST_REAL = 0x6,
    CUFFT_CB_ST_REAL_DOUBLE = 0x7,
    CUFFT_CB_UNDEFINED = 0x8

} cufftXtCallbackType;

typedef cufftComplex (*cufftCallbackLoadC)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftDoubleComplex (*cufftCallbackLoadZ)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftReal (*cufftCallbackLoadR)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);
typedef cufftDoubleReal (*cufftCallbackLoadD)(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer);

typedef void (*cufftCallbackStoreC)(void *dataOut, size_t offset, cufftComplex element, void *callerInfo,
                                    void *sharedPointer);
typedef void (*cufftCallbackStoreZ)(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo,
                                    void *sharedPointer);
typedef void (*cufftCallbackStoreR)(void *dataOut, size_t offset, cufftReal element, void *callerInfo,
                                    void *sharedPointer);
typedef void (*cufftCallbackStoreD)(void *dataOut, size_t offset, cufftDoubleReal element, void *callerInfo,
                                    void *sharedPointer);

#define CUDA_XT_DESCRIPTOR_VERSION 0x01000000  // This is added to CUDART_VERSION

enum cudaXtCopyType_t { LIB_XT_COPY_HOST_TO_DEVICE, LIB_XT_COPY_DEVICE_TO_HOST, LIB_XT_COPY_DEVICE_TO_DEVICE };
typedef enum cudaXtCopyType_t cudaLibXtCopyType;

enum libFormat_t { LIB_FORMAT_CUFFT = 0x0, LIB_FORMAT_UNDEFINED = 0x1 };

typedef enum libFormat_t libFormat;

#define MAX_CUDA_DESCRIPTOR_GPUS 64

struct cudaXtDesc_t {
    int version;                            // descriptor version
    int nGPUs;                              // number of GPUs
    int GPUs[MAX_CUDA_DESCRIPTOR_GPUS];     // array of device IDs
    void *data[MAX_CUDA_DESCRIPTOR_GPUS];   // array of pointers to data, one per GPU
    size_t size[MAX_CUDA_DESCRIPTOR_GPUS];  // array of data sizes, one per GPU
    void *cudaXtState;                      // opaque CUDA utility structure
};
typedef struct cudaXtDesc_t cudaXtDesc;

struct cudaLibXtDesc_t {
    int version;             // descriptor version
    cudaXtDesc *descriptor;  // multi-GPU memory descriptor
    libFormat library;       // which library recognizes the format
    int subFormat;           // library specific enumerator of sub formats
    void *libDescriptor;     // library specific descriptor e.g. FFT transform plan object
};
typedef struct cudaLibXtDesc_t cudaLibXtDesc;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_CUFFT_SUBSET_H__
