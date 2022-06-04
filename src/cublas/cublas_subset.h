// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: cublas subset

#ifndef __CUDA_HOOK_CUBLAS_SUBSET_H__
#define __CUDA_HOOK_CUBLAS_SUBSET_H__

#include "cudart_subset.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __half __half;

typedef enum cudaDataType_t {
    CUDA_R_16F = 2,  /* real as a half */
    CUDA_C_16F = 6,  /* complex as a pair of half numbers */
    CUDA_R_32F = 0,  /* real as a float */
    CUDA_C_32F = 4,  /* complex as a pair of float numbers */
    CUDA_R_64F = 1,  /* real as a double */
    CUDA_C_64F = 5,  /* complex as a pair of double numbers */
    CUDA_R_8I = 3,   /* real as a signed char */
    CUDA_C_8I = 7,   /* complex as a pair of signed char numbers */
    CUDA_R_8U = 8,   /* real as a unsigned char */
    CUDA_C_8U = 9,   /* complex as a pair of unsigned char numbers */
    CUDA_R_32I = 10, /* real as a signed int */
    CUDA_C_32I = 11, /* complex as a pair of signed int numbers */
    CUDA_R_32U = 12, /* real as a unsigned int */
    CUDA_C_32U = 13  /* complex as a pair of unsigned int numbers */
} cudaDataType;

typedef enum libraryPropertyType_t { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL } libraryPropertyType;

struct float2 {
    float x, y;
};

typedef float2 cuFloatComplex;
/* aliases */
typedef cuFloatComplex cuComplex;

struct double2 {
    double x, y;
};

/* Double precision */
typedef double2 cuDoubleComplex;

/* CUBLAS data types */
#define cublasStatus cublasStatus_t

#define CUBLAS_VER_MAJOR 11
#define CUBLAS_VER_MINOR 6
#define CUBLAS_VER_PATCH 5
#define CUBLAS_VER_BUILD 2
#define CUBLAS_VERSION (CUBLAS_VER_MAJOR * 1000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH)

/* CUBLAS status type returns */
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum { CUBLAS_FILL_MODE_LOWER = 0, CUBLAS_FILL_MODE_UPPER = 1, CUBLAS_FILL_MODE_FULL = 2 } cublasFillMode_t;

typedef enum { CUBLAS_DIAG_NON_UNIT = 0, CUBLAS_DIAG_UNIT = 1 } cublasDiagType_t;

typedef enum { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 } cublasSideMode_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_HERMITAN = 2, /* synonym if CUBLAS_OP_C */
    CUBLAS_OP_CONJG = 3     /* conjugate, placeholder - not supported in the current release */
} cublasOperation_t;

typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;

typedef enum { CUBLAS_ATOMICS_NOT_ALLOWED = 0, CUBLAS_ATOMICS_ALLOWED = 1 } cublasAtomicsMode_t;

/*For different GEMM algorithm */
typedef enum {
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_DEFAULT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7,
    CUBLAS_GEMM_ALGO8 = 8,
    CUBLAS_GEMM_ALGO9 = 9,
    CUBLAS_GEMM_ALGO10 = 10,
    CUBLAS_GEMM_ALGO11 = 11,
    CUBLAS_GEMM_ALGO12 = 12,
    CUBLAS_GEMM_ALGO13 = 13,
    CUBLAS_GEMM_ALGO14 = 14,
    CUBLAS_GEMM_ALGO15 = 15,
    CUBLAS_GEMM_ALGO16 = 16,
    CUBLAS_GEMM_ALGO17 = 17,
    CUBLAS_GEMM_ALGO18 = 18,  // sliced 32x32
    CUBLAS_GEMM_ALGO19 = 19,  // sliced 64x32
    CUBLAS_GEMM_ALGO20 = 20,  // sliced 128x32
    CUBLAS_GEMM_ALGO21 = 21,  // sliced 32x32  -splitK
    CUBLAS_GEMM_ALGO22 = 22,  // sliced 64x32  -splitK
    CUBLAS_GEMM_ALGO23 = 23,  // sliced 128x32 -splitK
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
    CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
    CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
    CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
    CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
    CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
    CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
    CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
    CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
    CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
    CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
    CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
    CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
    CUBLAS_GEMM_ALGO15_TENSOR_OP = 115
} cublasGemmAlgo_t;

/*Enum for default math mode/tensor operation*/
typedef enum {
    CUBLAS_DEFAULT_MATH = 0,

    /* deprecated, same effect as using CUBLAS_COMPUTE_32F_FAST_16F, will be removed in a future release */
    CUBLAS_TENSOR_OP_MATH = 1,

    /* same as using matching _PEDANTIC compute type when using cublas<T>routine calls or cublasEx() calls with
       cudaDataType as compute type */
    CUBLAS_PEDANTIC_MATH = 2,

    /* allow accelerating single precision routines using TF32 tensor cores */
    CUBLAS_TF32_TENSOR_OP_MATH = 3,

    /* flag to force any reductons to use the accumulator type and not output type in case of mixed precision routines
       with lower size output type */
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
} cublasMath_t;

/* For backward compatibility purposes */
typedef cudaDataType cublasDataType_t;

/* Enum for compute type
 *
 * - default types provide best available performance using all available hardware features
 *   and guarantee internal storage precision with at least the same precision and range;
 * - _PEDANTIC types ensure standard arithmetic and exact specified internal storage format;
 * - _FAST types allow for some loss of precision to enable higher throughput arithmetic.
 */
typedef enum {
    CUBLAS_COMPUTE_16F = 64,           /* half - default */
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,  /* half - pedantic */
    CUBLAS_COMPUTE_32F = 68,           /* float - default */
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,  /* float - pedantic */
    CUBLAS_COMPUTE_32F_FAST_16F = 74,  /* float - fast, allows down-converting inputs to half or TF32 */
    CUBLAS_COMPUTE_32F_FAST_16BF = 75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77, /* float - fast, allows down-converting inputs to TF32 */
    CUBLAS_COMPUTE_64F = 70,           /* double - default */
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,  /* double - pedantic */
    CUBLAS_COMPUTE_32I = 72,           /* signed 32-bit int - default */
    CUBLAS_COMPUTE_32I_PEDANTIC = 73,  /* signed 32-bit int - pedantic */
} cublasComputeType_t;

/* Opaque structure holding CUBLAS library context */
struct cublasContext;
typedef struct cublasContext *cublasHandle_t;

/* Cublas logging */
typedef void (*cublasLogCallback)(const char *msg);

struct cublasXtContext;
typedef struct cublasXtContext *cublasXtHandle_t;

typedef enum { CUBLASXT_PINNING_DISABLED = 0, CUBLASXT_PINNING_ENABLED = 1 } cublasXtPinnedMemMode_t;

/* This routines is to provide a CPU Blas routines, used for too small sizes or hybrid computation */
typedef enum {
    CUBLASXT_FLOAT = 0,
    CUBLASXT_DOUBLE = 1,
    CUBLASXT_COMPLEX = 2,
    CUBLASXT_DOUBLECOMPLEX = 3,
} cublasXtOpType_t;

typedef enum {
    CUBLASXT_GEMM = 0,
    CUBLASXT_SYRK = 1,
    CUBLASXT_HERK = 2,
    CUBLASXT_SYMM = 3,
    CUBLASXT_HEMM = 4,
    CUBLASXT_TRSM = 5,
    CUBLASXT_SYR2K = 6,
    CUBLASXT_HER2K = 7,

    CUBLASXT_SPMM = 8,
    CUBLASXT_SYRKX = 9,
    CUBLASXT_HERKX = 10,
    CUBLASXT_TRMM = 11,
    CUBLASXT_ROUTINE_MAX = 12,
} cublasXtBlasOp_t;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_CUBLAS_SUBSET_H__
