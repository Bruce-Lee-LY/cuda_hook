/*
 * Copyright 2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// the following one modifications have been made
// (1) add content in the bottom from cusolverDn.h, cusolverMg.h, cusolverRf.h, cusolverSp.h and
// cusolverSp_LOWLEVEL_PREVIEW.h
// (2) delete CUSOLVER_DEPRECATED

#if !defined(CUSOLVER_COMMON_H_)
#define CUSOLVER_COMMON_H_

#include "library_types.h"

#ifndef CUSOLVERAPI
#ifdef _WIN32
#define CUSOLVERAPI __stdcall
#else
#define CUSOLVERAPI 
#endif
#endif


#if defined(_MSC_VER)
typedef __int64 int64_t;
#else
#include <inttypes.h>
#endif

typedef int cusolver_int_t;


#define CUSOLVER_VER_MAJOR 11
#define CUSOLVER_VER_MINOR 2
#define CUSOLVER_VER_PATCH 0
#define CUSOLVER_VER_BUILD 120
#define CUSOLVER_VERSION (CUSOLVER_VER_MAJOR * 1000 + \
                        CUSOLVER_VER_MINOR *  100 + \
                        CUSOLVER_VER_PATCH)

/*
 * disable this macro to proceed old API
 */
#define DISABLE_CUSOLVER_DEPRECATED

//------------------------------------------------------------------------------

#if !defined(_MSC_VER)
#   define CUSOLVER_CPP_VERSION __cplusplus
#elif _MSC_FULL_VER >= 190024210 // Visual Studio 2015 Update 3
#   define CUSOLVER_CPP_VERSION _MSVC_LANG
#else
#   define CUSOLVER_CPP_VERSION 0
#endif

//------------------------------------------------------------------------------

#if !defined(DISABLE_CUSOLVER_DEPRECATED)

#   if CUSOLVER_CPP_VERSION >= 201402L

#       define CUSOLVER_DEPRECATED(new_func)                                   \
            [[deprecated("please use " #new_func " instead")]]

#   elif defined(_MSC_VER)

#       define CUSOLVER_DEPRECATED(new_func)                                   \
            __declspec(deprecated("please use " #new_func " instead"))

#   elif defined(__INTEL_COMPILER) || defined(__clang__) ||                    \
         (defined(__GNUC__) &&                                                 \
          (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)))

#       define CUSOLVER_DEPRECATED(new_func)                                   \
            __attribute__((deprecated("please use " #new_func " instead")))

#   elif defined(__GNUC__) || defined(__xlc__)

#       define CUSOLVER_DEPRECATED(new_func)                                   \
            __attribute__((deprecated))

#   else

#       define CUSOLVER_DEPRECATED(new_func)

#   endif // defined(__cplusplus) && __cplusplus >= 201402L
//------------------------------------------------------------------------------

#   if CUSOLVER_CPP_VERSION >= 201703L

#       define CUSOLVER_DEPRECATED_ENUM(new_enum)                              \
            [[deprecated("please use " #new_enum " instead")]]

#   elif defined(__clang__) ||                                                 \
         (defined(__GNUC__) && __GNUC__ >= 6 && !defined(__PGI))

#       define CUSOLVER_DEPRECATED_ENUM(new_enum)                              \
            __attribute__((deprecated("please use " #new_enum " instead")))

#   else

#       define CUSOLVER_DEPRECATED_ENUM(new_enum)

#   endif // defined(__cplusplus) && __cplusplus >= 201402L

#else // defined(DISABLE_CUSOLVER_DEPRECATED)

#   define CUSOLVER_DEPRECATED(new_func)
#   define CUSOLVER_DEPRECATED_ENUM(new_enum)

#endif // !defined(DISABLE_CUSOLVER_DEPRECATED)

#undef CUSOLVER_CPP_VERSION






#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

typedef enum{
    CUSOLVER_STATUS_SUCCESS=0,
    CUSOLVER_STATUS_NOT_INITIALIZED=1,
    CUSOLVER_STATUS_ALLOC_FAILED=2,
    CUSOLVER_STATUS_INVALID_VALUE=3,
    CUSOLVER_STATUS_ARCH_MISMATCH=4,
    CUSOLVER_STATUS_MAPPING_ERROR=5,
    CUSOLVER_STATUS_EXECUTION_FAILED=6,
    CUSOLVER_STATUS_INTERNAL_ERROR=7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT=10,
    CUSOLVER_STATUS_INVALID_LICENSE=11,
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED=12,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID=13,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC=14,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE=15,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER=16,
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR=20,
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED=21,
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE=22,
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES=23,
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED=25,
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED=26,
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR=30,
    CUSOLVER_STATUS_INVALID_WORKSPACE=31
} cusolverStatus_t;

typedef enum {
    CUSOLVER_EIG_TYPE_1=1,
    CUSOLVER_EIG_TYPE_2=2,
    CUSOLVER_EIG_TYPE_3=3
} cusolverEigType_t ;

typedef enum {
    CUSOLVER_EIG_MODE_NOVECTOR=0,
    CUSOLVER_EIG_MODE_VECTOR=1
} cusolverEigMode_t ;


typedef enum {
    CUSOLVER_EIG_RANGE_ALL=1001,
    CUSOLVER_EIG_RANGE_I=1002,
    CUSOLVER_EIG_RANGE_V=1003,
} cusolverEigRange_t ;



typedef enum {
    CUSOLVER_INF_NORM=104,
    CUSOLVER_MAX_NORM=105,
    CUSOLVER_ONE_NORM=106,
    CUSOLVER_FRO_NORM=107,
} cusolverNorm_t ;

typedef enum {
    CUSOLVER_IRS_REFINE_NOT_SET          = 1100,
    CUSOLVER_IRS_REFINE_NONE             = 1101,
    CUSOLVER_IRS_REFINE_CLASSICAL        = 1102,
    CUSOLVER_IRS_REFINE_CLASSICAL_GMRES  = 1103,
    CUSOLVER_IRS_REFINE_GMRES            = 1104,
    CUSOLVER_IRS_REFINE_GMRES_GMRES      = 1105,
    CUSOLVER_IRS_REFINE_GMRES_NOPCOND    = 1106,

    CUSOLVER_PREC_DD           = 1150,
    CUSOLVER_PREC_SS           = 1151,
    CUSOLVER_PREC_SHT          = 1152,

} cusolverIRSRefinement_t;


typedef enum {
    CUSOLVER_R_8I  = 1201,
    CUSOLVER_R_8U  = 1202,
    CUSOLVER_R_64F = 1203,
    CUSOLVER_R_32F = 1204,
    CUSOLVER_R_16F = 1205,
    CUSOLVER_R_16BF  = 1206,
    CUSOLVER_R_TF32  = 1207,
    CUSOLVER_R_AP  = 1208,
    CUSOLVER_C_8I  = 1211,
    CUSOLVER_C_8U  = 1212,
    CUSOLVER_C_64F = 1213,
    CUSOLVER_C_32F = 1214,
    CUSOLVER_C_16F = 1215,
    CUSOLVER_C_16BF  = 1216,
    CUSOLVER_C_TF32  = 1217,
    CUSOLVER_C_AP  = 1218,
} cusolverPrecType_t ;

typedef enum {
   CUSOLVER_ALG_0 = 0,  /* default algorithm */
   CUSOLVER_ALG_1 = 1
} cusolverAlgMode_t;


typedef enum {
    CUBLAS_STOREV_COLUMNWISE=0, 
    CUBLAS_STOREV_ROWWISE=1
} cusolverStorevMode_t; 

typedef enum {
    CUBLAS_DIRECT_FORWARD=0, 
    CUBLAS_DIRECT_BACKWARD=1
} cusolverDirectMode_t;

cusolverStatus_t CUSOLVERAPI cusolverGetProperty(
    libraryPropertyType type, 
    int *value);

cusolverStatus_t CUSOLVERAPI cusolverGetVersion(
    int *version);

// cuda/targets/x86_64-linux/include/cusolverDn.h
cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t CUSOLVERAPI cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t CUSOLVERAPI cusolverDnSetStream (cusolverDnHandle_t handle, cudaStream_t streamId);
cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId);

//============================================================
// IRS headers 
//============================================================

// =============================================================================
// IRS helper function API
// =============================================================================
cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsCreate(
            cusolverDnIRSParams_t* params_ptr );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsDestroy(
            cusolverDnIRSParams_t params );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetRefinementSolver(
            cusolverDnIRSParams_t params,
            cusolverIRSRefinement_t refinement_solver );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetSolverMainPrecision(
            cusolverDnIRSParams_t params,
            cusolverPrecType_t solver_main_precision ); 

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetSolverLowestPrecision(
            cusolverDnIRSParams_t params,
            cusolverPrecType_t solver_lowest_precision );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetSolverPrecisions(
            cusolverDnIRSParams_t params,
            cusolverPrecType_t solver_main_precision,
            cusolverPrecType_t solver_lowest_precision );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetTol(
            cusolverDnIRSParams_t params,
            double val );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetTolInner(
            cusolverDnIRSParams_t params,
            double val );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetMaxIters(
            cusolverDnIRSParams_t params,
            cusolver_int_t maxiters );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsSetMaxItersInner(
            cusolverDnIRSParams_t params,
            cusolver_int_t maxiters_inner );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSParamsGetMaxIters(
            cusolverDnIRSParams_t params,
            cusolver_int_t *maxiters );

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsEnableFallback(
    cusolverDnIRSParams_t params );

cusolverStatus_t CUSOLVERAPI
cusolverDnIRSParamsDisableFallback(
    cusolverDnIRSParams_t params );


// =============================================================================
// cusolverDnIRSInfos prototypes
// =============================================================================
cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosDestroy(
        cusolverDnIRSInfos_t infos );

cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosCreate(
        cusolverDnIRSInfos_t* infos_ptr );

cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosGetNiters(
            cusolverDnIRSInfos_t infos,
            cusolver_int_t *niters );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSInfosGetOuterNiters(
            cusolverDnIRSInfos_t infos,
            cusolver_int_t *outer_niters );

cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosRequestResidual(
        cusolverDnIRSInfos_t infos );

cusolverStatus_t CUSOLVERAPI 
    cusolverDnIRSInfosGetResidualHistory(
            cusolverDnIRSInfos_t infos,
            void **residual_history );

cusolverStatus_t CUSOLVERAPI
    cusolverDnIRSInfosGetMaxIters(
            cusolverDnIRSInfos_t infos,
            cusolver_int_t *maxiters );

//============================================================
//  IRS functions API
//============================================================

/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gesv 
 * users API Prototypes */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZCgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZKgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZEgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZYgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCEgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCKgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCYgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDSgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDHgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDBgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDXgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSHgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSBgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSXgesv(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

/*******************************************************************************/


/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gesv_bufferSize 
 * users API Prototypes */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZCgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZKgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZEgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZYgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCKgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCEgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCYgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDSgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDHgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDBgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDXgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSHgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSBgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSXgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t n, cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        cusolver_int_t *dipiv,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);
/*******************************************************************************/


/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels 
 * users API Prototypes */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnZZgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZCgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZKgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZEgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnZYgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCCgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCKgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCEgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnCYgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDDgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDSgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDHgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDBgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnDXgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSSgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSHgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSBgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnSXgels(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *iter,
        cusolver_int_t *d_info);
/*******************************************************************************/

/*******************************************************************************//*
 * [ZZ, ZC, ZK, ZE, ZY, CC, CK, CE, CY, DD, DS, DH, DB, DX, SS, SH, SB, SX]gels_bufferSize 
 * API prototypes */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnZZgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZCgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZKgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZEgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnZYgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuDoubleComplex *dA, cusolver_int_t ldda,
        cuDoubleComplex *dB, cusolver_int_t lddb,
        cuDoubleComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCCgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCKgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCEgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnCYgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        cuComplex *dA, cusolver_int_t ldda,
        cuComplex *dB, cusolver_int_t lddb,
        cuComplex *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDDgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDSgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDHgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDBgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnDXgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        double *dA, cusolver_int_t ldda,
        double *dB, cusolver_int_t lddb,
        double *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSSgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSHgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSBgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSXgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        float *dA, cusolver_int_t ldda,
        float *dB, cusolver_int_t lddb,
        float *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t *lwork_bytes);
/*******************************************************************************/



/*******************************************************************************//*
 * expert users API for IRS Prototypes
 * */
/*******************************************************************************/
cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgesv(
        cusolverDnHandle_t handle,
        cusolverDnIRSParams_t gesv_irs_params,
        cusolverDnIRSInfos_t  gesv_irs_infos,
        cusolver_int_t n, cusolver_int_t nrhs,
        void *dA, cusolver_int_t ldda,
        void *dB, cusolver_int_t lddb,
        void *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *niters,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgesv_bufferSize(
        cusolverDnHandle_t handle,
        cusolverDnIRSParams_t params,
        cusolver_int_t n, cusolver_int_t nrhs,
        size_t *lwork_bytes);


cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgels(
        cusolverDnHandle_t handle,
        cusolverDnIRSParams_t gels_irs_params,
        cusolverDnIRSInfos_t  gels_irs_infos,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs,
        void *dA, cusolver_int_t ldda,
        void *dB, cusolver_int_t lddb,
        void *dX, cusolver_int_t lddx,
        void *dWorkspace, size_t lwork_bytes,
        cusolver_int_t *niters,
        cusolver_int_t *d_info);

cusolverStatus_t CUSOLVERAPI cusolverDnIRSXgels_bufferSize(
        cusolverDnHandle_t handle,
        cusolverDnIRSParams_t params,
        cusolver_int_t m, 
        cusolver_int_t n, 
        cusolver_int_t nrhs, 
        size_t *lwork_bytes);
/*******************************************************************************/


/* Cholesky factorization and its solver */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuComplex *A, 
    int lda, 
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf_bufferSize( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int *Lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    float *A, 
    int lda,  
    float *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    double *A, 
    int lda, 
    double *Workspace, 
    int Lwork, 
    int *devInfo );



cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf( 
    cusolverDnHandle_t handle, 
    cublasFillMode_t uplo, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *Workspace, 
    int Lwork, 
    int *devInfo );


cusolverStatus_t CUSOLVERAPI cusolverDnSpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const float *A,
    int lda,
    float *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const double *A,
    int lda,
    double *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const cuComplex *A,
    int lda,
    cuComplex *B,
    int ldb,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrs(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs,
    const cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *B,
    int ldb,
    int *devInfo);

/* batched Cholesky factorization and its solver */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrfBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *Aarray[],
    int lda,
    int *infoArray,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnSpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    float *A[],
    int lda,
    float *B[],
    int ldb,
    int *d_info,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    double *A[],
    int lda,
    double *B[],
    int ldb,
    int *d_info,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    cuComplex *A[],
    int lda,
    cuComplex *B[],
    int ldb,
    int *d_info,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotrsBatched(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    int nrhs, /* only support rhs = 1*/
    cuDoubleComplex *A[],
    int lda,
    cuDoubleComplex *B[],
    int ldb,
    int *d_info,
    int batchSize);

/* s.p.d. matrix inversion (POTRI) and auxiliary routines (TRTRI and LAUUM)  */
cusolverStatus_t CUSOLVERAPI cusolverDnSpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnCpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZpotri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnXtrtri_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXtrtri(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *devInfo);

/* lauum, auxiliar routine for s.p.d matrix inversion */
cusolverStatus_t CUSOLVERAPI cusolverDnSlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnClauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZlauum_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnClauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZlauum(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo);



/* LU Factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *Lwork );


cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *Workspace, 
    int *devIpiv, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *Workspace, 
    int *devIpiv, 
    int *devInfo );

/* Row pivoting */
cusolverStatus_t CUSOLVERAPI cusolverDnSlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    float *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUSOLVERAPI cusolverDnDlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    double *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUSOLVERAPI cusolverDnClaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    cuComplex *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

cusolverStatus_t CUSOLVERAPI cusolverDnZlaswp( 
    cusolverDnHandle_t handle, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    int k1, 
    int k2, 
    const int *devIpiv, 
    int incx);

/* LU solve */
cusolverStatus_t CUSOLVERAPI cusolverDnSgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const float *A, 
    int lda, 
    const int *devIpiv, 
    float *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const double *A, 
    int lda, 
    const int *devIpiv, 
    double *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnCgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const cuComplex *A, 
    int lda, 
    const int *devIpiv, 
    cuComplex *B, 
    int ldb, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZgetrs( 
    cusolverDnHandle_t handle, 
    cublasOperation_t trans, 
    int n, 
    int nrhs, 
    const cuDoubleComplex *A, 
    int lda, 
    const int *devIpiv, 
    cuDoubleComplex *B, 
    int ldb, 
    int *devInfo );


/* QR factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    float *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    double *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuComplex *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A,  
    int lda, 
    float *TAU,  
    float *Workspace,  
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *TAU, 
    double *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    cuComplex *TAU, 
    cuComplex *Workspace, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    cuDoubleComplex *TAU, 
    cuDoubleComplex *Workspace, 
    int Lwork, 
    int *devInfo );


/* generate unitary matrix Q from QR factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCungqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZungqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCungqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZungqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info);



/* compute Q**T*b in solve min||A*x = b|| */
cusolverStatus_t CUSOLVERAPI cusolverDnSormqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    const float *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDormqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    const double *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    const cuComplex *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    const cuDoubleComplex *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSormqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnDormqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *C,
    int ldc,
    cuComplex *work,
    int lwork,
    int *devInfo);

cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasOperation_t trans,
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *C,
    int ldc,
    cuDoubleComplex *work,
    int lwork,
    int *devInfo);


/* L*D*L**T,U*D*U**T factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    float *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    double *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    cuComplex *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf_bufferSize(
    cusolverDnHandle_t handle,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    int *ipiv,
    float *work,
    int lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    int *ipiv,
    double *work,
    int lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    int *ipiv,
    cuComplex *work,
    int lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    int *ipiv,
    cuDoubleComplex *work,
    int lwork,
    int *info );

/* Symmetric indefinite solve (SYTRS) */
cusolverStatus_t CUSOLVERAPI cusolverDnXsytrs_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int64_t n,
        int64_t nrhs,
        cudaDataType dataTypeA,
        const void *A,
        int64_t lda,
        const int64_t *ipiv,
        cudaDataType dataTypeB,
        void *B,
        int64_t ldb,
        size_t *workspaceInBytesOnDevice,
        size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXsytrs(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int64_t n,
        int64_t nrhs,
        cudaDataType dataTypeA,
        const void *A,
        int64_t lda,
        const int64_t *ipiv,
        cudaDataType dataTypeB,
        void *B,
        int64_t ldb,
        void *bufferOnDevice,
        size_t workspaceInBytesOnDevice,
        void *bufferOnHost,
        size_t workspaceInBytesOnHost,
        int *info);

/* Symmetric indefinite inversion (sytri) */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        float *A,
        int lda,
        const int *ipiv,
        int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        double *A,
        int lda,
        const int *ipiv,
        int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuComplex *A,
        int lda,
        const int *ipiv,
        int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZsytri_bufferSize(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuDoubleComplex *A,
        int lda,
        const int *ipiv,
        int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        float *A,
        int lda,
        const int *ipiv,
        float *work,
        int lwork,
        int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        double *A,
        int lda,
        const int *ipiv,
        double *work,
        int lwork,
        int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuComplex *A,
        int lda,
        const int *ipiv,
        cuComplex *work,
        int lwork,
        int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZsytri(
        cusolverDnHandle_t handle,
        cublasFillMode_t uplo,
        int n,
        cuDoubleComplex *A,
        int lda,
        const int *ipiv,
        cuDoubleComplex *work,
        int lwork,
        int *info);


/* bidiagonal factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *Lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    float *A,  
    int lda,
    float *D, 
    float *E, 
    float *TAUQ,  
    float *TAUP, 
    float *Work,
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    double *A, 
    int lda,
    double *D, 
    double *E, 
    double *TAUQ, 
    double *TAUP, 
    double *Work,
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuComplex *A, 
    int lda, 
    float *D, 
    float *E, 
    cuComplex *TAUQ, 
    cuComplex *TAUP,
    cuComplex *Work, 
    int Lwork, 
    int *devInfo );

cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd( 
    cusolverDnHandle_t handle, 
    int m, 
    int n, 
    cuDoubleComplex *A,
    int lda, 
    double *D, 
    double *E, 
    cuDoubleComplex *TAUQ,
    cuDoubleComplex *TAUP, 
    cuDoubleComplex *Work, 
    int Lwork, 
    int *devInfo );

/* generates one of the unitary matrices Q or P**T determined by GEBRD*/
cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCungbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZungbr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCungbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZungbr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side, 
    int m,
    int n,
    int k,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info);


/* tridiagonal factorization */
cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *d,
    const float *e,
    const float *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *d,
    const double *e,
    const double *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnChetrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *d,
    const float *e,
    const cuComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *d,
    const double *e,
    const cuDoubleComplex *tau,
    int *lwork);


cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *d,
    float *e,
    float *tau,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *d,
    double *e,
    double *tau,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnChetrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *d,
    float *e,
    cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *d,
    double *e,
    cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info);



/* generate unitary Q comes from sytrd */
cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCungtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZungtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    const float *tau,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    const double *tau,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCungtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    cuComplex *A,
    int lda,
    const cuComplex *tau,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZungtr(
    cusolverDnHandle_t handle,
    cublasFillMode_t uplo, 
    int n,
    cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    cuDoubleComplex *work,
    int lwork,
    int *info);



/* compute op(Q)*C or C*op(Q) where Q comes from sytrd */
cusolverStatus_t CUSOLVERAPI cusolverDnSormtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const float *A,
    int lda,
    const float *tau,
    const float *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDormtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const double *A,
    int lda,
    const double *tau,
    const double *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const cuComplex *tau,
    const cuComplex *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr_bufferSize(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *tau,
    const cuDoubleComplex *C,
    int ldc,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSormtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    float *A,
    int lda,
    float *tau,
    float *C,
    int ldc,
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDormtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    double *A,
    int lda,
    double *tau,
    double *C,
    int ldc,
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    cuComplex *A,
    int lda,
    cuComplex *tau,
    cuComplex *C,
    int ldc,
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr(
    cusolverDnHandle_t handle,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *tau,
    cuDoubleComplex *C,
    int ldc,
    cuDoubleComplex *work,
    int lwork,
    int *info);



/* singular value decomposition, A = U * Sigma * V^H */
cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int *lwork );

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd (
    cusolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *S, 
    float *U, 
    int ldu, 
    float *VT, 
    int ldvt, 
    float *work, 
    int lwork, 
    float *rwork, 
    int  *info );

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd (
    cusolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    double *A, 
    int lda, 
    double *S, 
    double *U, 
    int ldu, 
    double *VT, 
    int ldvt, 
    double *work,
    int lwork, 
    double *rwork, 
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd (
    cusolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    cuComplex *A,
    int lda, 
    float *S, 
    cuComplex *U, 
    int ldu, 
    cuComplex *VT, 
    int ldvt,
    cuComplex *work, 
    int lwork, 
    float *rwork, 
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd (
    cusolverDnHandle_t handle, 
    signed char jobu, 
    signed char jobvt, 
    int m, 
    int n, 
    cuDoubleComplex *A, 
    int lda, 
    double *S, 
    cuDoubleComplex *U, 
    int ldu, 
    cuDoubleComplex *VT, 
    int ldvt, 
    cuDoubleComplex *work, 
    int lwork, 
    double *rwork, 
    int *info );


/* standard symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const cuComplex *A,
    int lda,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float *W, 
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double *W, 
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    cuComplex *A,
    int lda,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info);

/* standard selective symmetric eigenvalue solver, A*x = lambda*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const double *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const cuComplex *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const cuDoubleComplex *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    cuComplex *A,
    int lda,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    cuDoubleComplex *A,
    int lda,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info);

/* selective generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo, 
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    const double *A, 
    int lda,
    const double *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    const cuComplex *A, 
    int lda,
    const cuComplex *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz, 
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    const double *W,
    int *lwork);


cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,  
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    double *A, 
    int lda,
    double *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B, 
    int ldb,
    float vl,
    float vu,
    int il,
    int iu,
    int *meig,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cusolverEigRange_t range,
    cublasFillMode_t uplo,  
    int n,
    cuDoubleComplex *A, 
    int lda,
    cuDoubleComplex *B, 
    int ldb,
    double vl,
    double vu,
    int il,
    int iu,
    int *meig,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info);


/* generalized symmetric eigenvalue solver, A*x = lambda*B*x, by divide-and-conquer  */
cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    const double *A, 
    int lda,
    const double *B, 
    int ldb,
    const double *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    const cuComplex *A, 
    int lda,
    const cuComplex *B, 
    int ldb,
    const float *W,
    int *lwork);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,  
    int n,
    const cuDoubleComplex *A,
    int lda,
    const cuDoubleComplex *B, 
    int ldb,
    const double *W,
    int *lwork);


cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float *W, 
    float *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,  
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    double *A, 
    int lda,
    double *B, 
    int ldb,
    double *W, 
    double *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    cuComplex *A,
    int lda,
    cuComplex *B, 
    int ldb,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,   
    cusolverEigMode_t jobz,  
    cublasFillMode_t uplo,  
    int n,
    cuDoubleComplex *A, 
    int lda,
    cuDoubleComplex *B, 
    int ldb,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info);


cusolverStatus_t CUSOLVERAPI cusolverDnCreateSyevjInfo(
    syevjInfo_t *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDestroySyevjInfo(
    syevjInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetTolerance(
    syevjInfo_t info,
    double tolerance);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetMaxSweeps(
    syevjInfo_t info,
    int max_sweeps);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetSortEig(
    syevjInfo_t info,
    int sort_eig);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetResidual(
    cusolverDnHandle_t handle,
    syevjInfo_t info,
    double *residual);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetSweeps(
    cusolverDnHandle_t handle,
    syevjInfo_t info,
    int *executed_sweeps);


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const double *A, 
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const cuComplex *A, 
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    const cuDoubleComplex *A, 
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,   
    float *A,
    int lda,
    float *W, 
    float *work,
    int lwork,
    int *info, 
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    double *A,
    int lda,
    double *W,
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    cuComplex *A,
    int lda,
    float *W,
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const float *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const double *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnCheevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A,
    int lda,
    const float *W,
    int *lwork,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnZheevj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *W,
    int *lwork,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int n,
    float *A,
    int lda,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W, 
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnCheevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    cuComplex *A,
    int lda,
    float *W, 
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnZheevj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    const float *A, 
    int lda,
    const float *B, 
    int ldb,
    const float *W,
    int *lwork,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    const double *A, 
    int lda,
    const double *B,
    int ldb,
    const double *W,
    int *lwork,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuComplex *A, 
    int lda,
    const cuComplex *B, 
    int ldb,
    const float *W,
    int *lwork,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *A, 
    int lda,
    const cuDoubleComplex *B, 
    int ldb,
    const double *W,
    int *lwork,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    float *A, 
    int lda,
    float *B, 
    int ldb,
    float *W,
    float *work,
    int lwork,
    int *info,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo, 
    int n,
    double *A, 
    int lda,
    double *B,
    int ldb,
    double *W, 
    double *work,
    int lwork,
    int *info,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnChegvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo,
    int n,
    cuComplex *A, 
    int lda,
    cuComplex *B, 
    int ldb,
    float *W,
    cuComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj(
    cusolverDnHandle_t handle,
    cusolverEigType_t itype, 
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,  
    int n,
    cuDoubleComplex *A, 
    int lda,
    cuDoubleComplex *B, 
    int ldb,
    double *W, 
    cuDoubleComplex *work,
    int lwork,
    int *info,
    syevjInfo_t params);


cusolverStatus_t CUSOLVERAPI cusolverDnCreateGesvdjInfo(
    gesvdjInfo_t *info);

cusolverStatus_t CUSOLVERAPI cusolverDnDestroyGesvdjInfo(
    gesvdjInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetTolerance(
    gesvdjInfo_t info,
    double tolerance);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetMaxSweeps(
    gesvdjInfo_t info,
    int max_sweeps);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetSortEig(
    gesvdjInfo_t info,
    int sort_svd);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetResidual(
    cusolverDnHandle_t handle,
    gesvdjInfo_t info,
    double *residual);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetSweeps(
    cusolverDnHandle_t handle,
    gesvdjInfo_t info,
    int *executed_sweeps);

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,                
    int n,                
    const float *A,    
    int lda,           
    const float *S, 
    const float *U,   
    int ldu, 
    const float *V,
    int ldv,  
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    const double *A, 
    int lda,
    const double *S,
    const double *U,
    int ldu,
    const double *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const float *S,
    const cuComplex *U,
    int ldu,
    const cuComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int m, 
    int n, 
    const cuDoubleComplex *A,
    int lda,
    const double *S,
    const cuDoubleComplex *U,
    int ldu, 
    const cuDoubleComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int m, 
    int n, 
    float *A, 
    int lda, 
    float *S, 
    float *U,
    int ldu,
    float *V,
    int ldv, 
    float *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    double *A,
    int lda,
    double *S,
    double *U,
    int ldu,
    double *V,
    int ldv, 
    double *work,
    int lwork,
    int *info, 
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m, 
    int n,
    cuComplex *A,
    int lda,
    float *S,
    cuComplex *U,
    int ldu,
    cuComplex *V,
    int ldv,
    cuComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda, 
    double *S, 
    cuDoubleComplex *U,
    int ldu,
    cuDoubleComplex *V,
    int ldv,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int econ,
    int m,
    int n, 
    const float *A,
    int lda,
    const float *S,
    const float *U,
    int ldu, 
    const float *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int econ,
    int m,
    int n,
    const double *A, 
    int lda,
    const double *S,
    const double *U,
    int ldu,
    const double *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int econ,
    int m,
    int n,
    const cuComplex *A,
    int lda,
    const float *S,
    const cuComplex *U,
    int ldu,
    const cuComplex *V,
    int ldv,
    int *lwork,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    const cuDoubleComplex *A,
    int lda,
    const double *S,
    const cuDoubleComplex *U,
    int ldu,
    const cuDoubleComplex *V,
    int ldv, 
    int *lwork,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    float *A, 
    int lda,
    float *S,
    float *U,
    int ldu,
    float *V,
    int ldv,
    float *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ, 
    int m, 
    int n, 
    double *A, 
    int lda,
    double *S,
    double *U,
    int ldu,
    double *V,
    int ldv,
    double *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    cuComplex *A,
    int lda,
    float *S,
    cuComplex *U,
    int ldu,
    cuComplex *V,
    int ldv,
    cuComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int econ,
    int m,
    int n,
    cuDoubleComplex *A,
    int lda,
    double *S,
    cuDoubleComplex *U, 
    int ldu, 
    cuDoubleComplex *V,
    int ldv,
    cuDoubleComplex *work,
    int lwork,
    int *info,
    gesvdjInfo_t params);


/* batched approximate SVD */

cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const float *d_A, 
    int lda,
    long long int strideA, 
    const float *d_S, 
    long long int strideS, 
    const float *d_U, 
    int ldu,
    long long int strideU, 
    const float *d_V, 
    int ldv,
    long long int strideV,
    int *lwork,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const double *d_A, 
    int lda,
    long long int strideA, 
    const double *d_S,   
    long long int strideS, 
    const double *d_U,  
    int ldu,
    long long int strideU, 
    const double *d_V,
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const cuComplex *d_A, 
    int lda,
    long long int strideA, 
    const float *d_S, 
    long long int strideS, 
    const cuComplex *d_U,
    int ldu,
    long long int strideU, 
    const cuComplex *d_V, 
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );

cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    int rank,
    int m,
    int n,
    const cuDoubleComplex *d_A,
    int lda,
    long long int strideA,
    const double *d_S, 
    long long int strideS, 
    const cuDoubleComplex *d_U, 
    int ldu,
    long long int strideU,
    const cuDoubleComplex *d_V,
    int ldv,
    long long int strideV, 
    int *lwork,
    int batchSize
    );


cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int rank, 
    int m,   
    int n,  
    const float *d_A, 
    int lda, 
    long long int strideA,
    float *d_S, 
    long long int strideS, 
    float *d_U, 
    int ldu, 
    long long int strideU,
    float *d_V, 
    int ldv,    
    long long int strideV, 
    float *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize);


cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int rank,
    int m, 
    int n, 
    const double *d_A,
    int lda,  
    long long int strideA, 
    double *d_S, 
    long long int strideS,
    double *d_U, 
    int ldu, 
    long long int strideU, 
    double *d_V, 
    int ldv, 
    long long int strideV,
    double *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF, 
    int batchSize);


cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int rank,  
    int m, 
    int n, 
    const cuComplex *d_A, 
    int lda,
    long long int strideA,
    float *d_S,
    long long int strideS,
    cuComplex *d_U, 
    int ldu,   
    long long int strideU,  
    cuComplex *d_V, 
    int ldv, 
    long long int strideV,
    cuComplex *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF, 
    int batchSize);


cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz, 
    int rank, 
    int m,   
    int n,  
    const cuDoubleComplex *d_A, 
    int lda,    
    long long int strideA,
    double *d_S,
    long long int strideS,
    cuDoubleComplex *d_U, 
    int ldu,   
    long long int strideU, 
    cuDoubleComplex *d_V,
    int ldv, 
    long long int strideV, 
    cuDoubleComplex *d_work,
    int lwork,
    int *d_info,
    double *h_R_nrmF,
    int batchSize);

cusolverStatus_t CUSOLVERAPI cusolverDnCreateParams(
    cusolverDnParams_t *params);

cusolverStatus_t CUSOLVERAPI cusolverDnDestroyParams(
    cusolverDnParams_t params);

cusolverStatus_t CUSOLVERAPI cusolverDnSetAdvOptions (
    cusolverDnParams_t params,
    cusolverDnFunction_t function,
    cusolverAlgMode_t algo   );

/* 64-bit API for POTRF */
cusolverStatus_t CUSOLVERAPI cusolverDnPotrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType computeType,
    size_t *workspaceInBytes );

cusolverStatus_t CUSOLVERAPI cusolverDnPotrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for POTRS */
cusolverStatus_t CUSOLVERAPI cusolverDnPotrs(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    int64_t nrhs,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info);


/* 64-bit API for GEQRF */
cusolverStatus_t CUSOLVERAPI cusolverDnGeqrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeTau,
    const void *tau,
    cudaDataType computeType,
    size_t *workspaceInBytes );

cusolverStatus_t CUSOLVERAPI cusolverDnGeqrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeTau,
    void *tau,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for GETRF */
cusolverStatus_t CUSOLVERAPI cusolverDnGetrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType computeType,
    size_t *workspaceInBytes );

cusolverStatus_t CUSOLVERAPI cusolverDnGetrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    int64_t *ipiv,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info );

/* 64-bit API for GETRS */
cusolverStatus_t CUSOLVERAPI cusolverDnGetrs(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasOperation_t trans,
    int64_t n,
    int64_t nrhs,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    const int64_t *ipiv,
    cudaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info );

/* 64-bit API for SYEVD */
cusolverStatus_t CUSOLVERAPI cusolverDnSyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeW,
    const void *W,
    cudaDataType computeType,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSyevd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeW,
    void *W,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/* 64-bit API for SYEVDX */
cusolverStatus_t CUSOLVERAPI cusolverDnSyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    void *vl,
    void *vu,
    int64_t il,
    int64_t iu,
    int64_t *h_meig,
    cudaDataType dataTypeW,
    const void *W,
    cudaDataType computeType,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverDnSyevdx(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    void * vl,
    void * vu,
    int64_t il,
    int64_t iu,
    int64_t *meig64,
    cudaDataType dataTypeW,
    void *W,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/* 64-bit API for GESVD */
cusolverStatus_t CUSOLVERAPI cusolverDnGesvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    const void *S,
    cudaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    cudaDataType dataTypeVT,
    const void *VT,
    int64_t ldvt,
    cudaDataType computeType,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverDnGesvd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    void *S,
    cudaDataType dataTypeU,
    void *U,
    int64_t ldu,
    cudaDataType dataTypeVT,
    void *VT,
    int64_t ldvt,
    cudaDataType computeType,
    void *pBuffer,
    size_t workspaceInBytes,
    int *info);

/*
 * new 64-bit API
 */
/* 64-bit API for POTRF */
cusolverStatus_t CUSOLVERAPI cusolverDnXpotrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXpotrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for POTRS */
cusolverStatus_t CUSOLVERAPI cusolverDnXpotrs(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasFillMode_t uplo,
    int64_t n,
    int64_t nrhs,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info);

/* 64-bit API for GEQRF */
cusolverStatus_t CUSOLVERAPI cusolverDnXgeqrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeTau,
    const void *tau,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXgeqrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeTau,
    void *tau,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for GETRF */
cusolverStatus_t CUSOLVERAPI cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXgetrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    int64_t *ipiv,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info );

/* 64-bit API for GETRS */
cusolverStatus_t CUSOLVERAPI cusolverDnXgetrs(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cublasOperation_t trans,
    int64_t n,
    int64_t nrhs,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    const int64_t *ipiv,
    cudaDataType dataTypeB,
    void *B,
    int64_t ldb,
    int *info );

/* 64-bit API for SYEVD */
cusolverStatus_t CUSOLVERAPI cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeW,
    const void *W,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeW,
    void *W,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for SYEVDX */
cusolverStatus_t CUSOLVERAPI cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    void *vl,
    void *vu,
    int64_t il,
    int64_t iu,
    int64_t *h_meig,
    cudaDataType dataTypeW,
    const void *W,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXsyevdx(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    cusolverEigRange_t range,
    cublasFillMode_t uplo,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    void * vl,
    void * vu,
    int64_t il,
    int64_t iu,
    int64_t *meig64,
    cudaDataType dataTypeW,
    void *W,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for GESVD */
cusolverStatus_t CUSOLVERAPI cusolverDnXgesvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    const void *S,
    cudaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    cudaDataType dataTypeVT,
    const void *VT,
    int64_t ldvt,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    signed char jobu,
    signed char jobvt,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    void *S,
    cudaDataType dataTypeU,
    void *U,
    int64_t ldu,
    cudaDataType dataTypeVT,
    void *VT,
    int64_t ldvt,
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *info);

/* 64-bit API for GESVDP */
cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdp_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz,
    int econ,
    int64_t m,
    int64_t n,
    cudaDataType dataTypeA,
    const void *A,
    int64_t lda,
    cudaDataType dataTypeS,
    const void *S,
    cudaDataType dataTypeU,
    const void *U,
    int64_t ldu,
    cudaDataType dataTypeV,
    const void *V,
    int64_t ldv,
    cudaDataType computeType,
    size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdp(
    cusolverDnHandle_t handle,
    cusolverDnParams_t params,
    cusolverEigMode_t jobz, 
    int econ,   
    int64_t m,   
    int64_t n,   
    cudaDataType dataTypeA,
    void *A,            
    int64_t lda,     
    cudaDataType dataTypeS,
    void *S,  
    cudaDataType dataTypeU,
    void *U,    
    int64_t ldu,   
    cudaDataType dataTypeV,
    void *V,  
    int64_t ldv, 
    cudaDataType computeType,
    void *bufferOnDevice,
    size_t workspaceInBytesOnDevice,
    void *bufferOnHost,
    size_t workspaceInBytesOnHost,
    int *d_info,
    double *h_err_sigma);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdr_bufferSize (
		cusolverDnHandle_t handle,
		cusolverDnParams_t params,
		signed char        jobu,
		signed char        jobv,
		int64_t            m,
		int64_t            n,
		int64_t            k,
		int64_t            p,
		int64_t            niters,
		cudaDataType       dataTypeA,
		const void         *A,
		int64_t            lda,
		cudaDataType       dataTypeSrand,
		const void         *Srand,
		cudaDataType       dataTypeUrand,
		const void         *Urand,
		int64_t            ldUrand,
		cudaDataType       dataTypeVrand,
		const void         *Vrand,
		int64_t            ldVrand,
		cudaDataType       computeType,
		size_t             *workspaceInBytesOnDevice,
		size_t             *workspaceInBytesOnHost
		);

cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdr(
		cusolverDnHandle_t handle,
		cusolverDnParams_t params,
		signed char        jobu,
		signed char        jobv,
		int64_t            m,
		int64_t            n,
		int64_t            k,
		int64_t            p,
		int64_t            niters,
		cudaDataType       dataTypeA,
		void               *A,
		int64_t            lda,
		cudaDataType       dataTypeSrand,
		void               *Srand,
		cudaDataType       dataTypeUrand,
		void               *Urand,
		int64_t            ldUrand,
		cudaDataType       dataTypeVrand,
		void               *Vrand,
		int64_t            ldVrand,
		cudaDataType       computeType,
		void               *bufferOnDevice,
		size_t             workspaceInBytesOnDevice,
		void               *bufferOnHost,
		size_t             workspaceInBytesOnHost,
		int                *d_info
		);

// cuda/targets/x86_64-linux/include/cusolverMg.h
struct cusolverMgContext;
typedef struct cusolverMgContext *cusolverMgHandle_t;


/**
 * \beief This enum decides how 1D device Ids (or process ranks) get mapped to a 2D grid.
 */
typedef enum {

  CUDALIBMG_GRID_MAPPING_ROW_MAJOR = 1,
  CUDALIBMG_GRID_MAPPING_COL_MAJOR = 0

} cusolverMgGridMapping_t;

/** \brief Opaque structure of the distributed grid */
typedef void * cudaLibMgGrid_t;
/** \brief Opaque structure of the distributed matrix descriptor */
typedef void * cudaLibMgMatrixDesc_t;


cusolverStatus_t CUSOLVERAPI cusolverMgCreate(
    cusolverMgHandle_t *handle);

cusolverStatus_t CUSOLVERAPI cusolverMgDestroy(
    cusolverMgHandle_t handle);

cusolverStatus_t CUSOLVERAPI cusolverMgDeviceSelect(
    cusolverMgHandle_t handle,
    int nbDevices,
    int deviceId[]);


/**
 * \brief Allocates resources related to the shared memory device grid.
 * \param[out] grid the opaque data strcuture that holds the grid
 * \param[in] numRowDevices number of devices in the row
 * \param[in] numColDevices number of devices in the column
 * \param[in] deviceId This array of size height * width stores the
 *            device-ids of the 2D grid; each entry must correspond to a valid gpu or to -1 (denoting CPU).
 * \param[in] mapping whether the 2D grid is in row/column major
 * \returns the status code
 */
cusolverStatus_t CUSOLVERAPI cusolverMgCreateDeviceGrid(
    cudaLibMgGrid_t* grid, 
    int32_t numRowDevices, 
    int32_t numColDevices,
    const int32_t deviceId[], 
    cusolverMgGridMapping_t mapping);

/**
 * \brief Releases the allocated resources related to the distributed grid.
 * \param[in] grid the opaque data strcuture that holds the distributed grid
 * \returns the status code
 */
cusolverStatus_t CUSOLVERAPI cusolverMgDestroyGrid(
    cudaLibMgGrid_t grid);

/**
 * \brief Allocates resources related to the distributed matrix descriptor.
 * \param[out] desc the opaque data strcuture that holds the descriptor
 * \param[in] numRows number of total rows
 * \param[in] numCols number of total columns
 * \param[in] rowBlockSize row block size
 * \param[in] colBlockSize column block size
 * \param[in] dataType the data type of each element in cudaDataType
 * \param[in] grid the opaque data structure of the distributed grid
 * \returns the status code
 */
cusolverStatus_t CUSOLVERAPI cusolverMgCreateMatrixDesc(
    cudaLibMgMatrixDesc_t * desc,
    int64_t numRows, 
    int64_t numCols, 
    int64_t rowBlockSize, 
    int64_t colBlockSize,
    cudaDataType dataType, 
    const cudaLibMgGrid_t grid);

/**
 * \brief Releases the allocated resources related to the distributed matrix descriptor.
 * \param[in] desc the opaque data strcuture that holds the descriptor
 * \returns the status code
 */
cusolverStatus_t CUSOLVERAPI cusolverMgDestroyMatrixDesc(
    cudaLibMgMatrixDesc_t desc);



cusolverStatus_t CUSOLVERAPI cusolverMgSyevd_bufferSize(
    cusolverMgHandle_t handle,
    cusolverEigMode_t jobz, 
    cublasFillMode_t uplo, 
    int N,
    void *array_d_A[], 
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    void *W,
    cudaDataType dataTypeW,
    cudaDataType computeType,
    int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgSyevd(
    cusolverMgHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    cudaLibMgMatrixDesc_t descrA,
    void *W,
    cudaDataType dataTypeW,
    cudaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverMgGetrf_bufferSize(
    cusolverMgHandle_t handle,
    int M,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    cudaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],
    cudaDataType computeType,
    int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgGetrf(
    cusolverMgHandle_t handle,
    int M,
    int N,
    void *array_d_A[],
    int IA,
    int JA,
    cudaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],
    cudaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverMgGetrs_bufferSize(
    cusolverMgHandle_t handle,
    cublasOperation_t TRANS,
    int N,
    int NRHS,
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[],  
    void *array_d_B[],
    int IB, 
    int JB, 
    cudaLibMgMatrixDesc_t descrB,
    cudaDataType computeType,
    int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgGetrs(
    cusolverMgHandle_t handle,
    cublasOperation_t TRANS,
    int N,
    int NRHS,
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    int *array_d_IPIV[], 
    void *array_d_B[],
    int IB, 
    int JB, 
    cudaLibMgMatrixDesc_t descrB,
    cudaDataType computeType,
    void *array_d_work[],
    int64_t lwork,
    int *info );

cusolverStatus_t CUSOLVERAPI cusolverMgPotrf_bufferSize( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA,
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    cudaDataType computeType, 
	int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgPotrf( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    cudaDataType computeType, 
    void *array_d_work[],
    int64_t lwork,
    int *h_info);

cusolverStatus_t CUSOLVERAPI cusolverMgPotrs_bufferSize( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int n, 
	int nrhs,
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    void *array_d_B[],
    int IB, 
    int JB, 
    cudaLibMgMatrixDesc_t descrB,
    cudaDataType computeType, 
	int64_t *lwork );

cusolverStatus_t CUSOLVERAPI cusolverMgPotrs( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int n, 
	int nrhs,
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    void *array_d_B[],
    int IB, 
    int JB, 
    cudaLibMgMatrixDesc_t descrB,
    cudaDataType computeType, 
    void *array_d_work[],
	int64_t lwork,
	int *h_info);

cusolverStatus_t CUSOLVERAPI cusolverMgPotri_bufferSize( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    cudaDataType computeType, 
	int64_t *lwork);

cusolverStatus_t CUSOLVERAPI cusolverMgPotri( 
    cusolverMgHandle_t handle,
	cublasFillMode_t uplo,
    int N, 
    void *array_d_A[],
    int IA, 
    int JA, 
    cudaLibMgMatrixDesc_t descrA,
    cudaDataType computeType, 
    void *array_d_work[],
	int64_t lwork,
    int *h_info);

// cuda/targets/x86_64-linux/include/cusolverRf.h
/* CUSOLVERRF mode */
typedef enum { 
    CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0, //default   
    CUSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1        
} cusolverRfResetValuesFastMode_t;

/* CUSOLVERRF matrix format */
typedef enum { 
    CUSOLVERRF_MATRIX_FORMAT_CSR = 0, //default   
    CUSOLVERRF_MATRIX_FORMAT_CSC = 1        
} cusolverRfMatrixFormat_t;

/* CUSOLVERRF unit diagonal */
typedef enum { 
    CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0, //default   
    CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1, 
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,        
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3        
} cusolverRfUnitDiagonal_t;

/* CUSOLVERRF factorization algorithm */
typedef enum {
    CUSOLVERRF_FACTORIZATION_ALG0 = 0, // default
    CUSOLVERRF_FACTORIZATION_ALG1 = 1,
    CUSOLVERRF_FACTORIZATION_ALG2 = 2,
} cusolverRfFactorization_t;

/* CUSOLVERRF triangular solve algorithm */
typedef enum {
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1, // default
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
} cusolverRfTriangularSolve_t;

/* CUSOLVERRF numeric boost report */
typedef enum {
    CUSOLVERRF_NUMERIC_BOOST_NOT_USED = 0, //default
    CUSOLVERRF_NUMERIC_BOOST_USED = 1
} cusolverRfNumericBoostReport_t;

/* Opaque structure holding CUSOLVERRF library common */
struct cusolverRfCommon;
typedef struct cusolverRfCommon *cusolverRfHandle_t;

/* CUSOLVERRF create (allocate memory) and destroy (free memory) in the handle */
cusolverStatus_t CUSOLVERAPI cusolverRfCreate(cusolverRfHandle_t *handle);
cusolverStatus_t CUSOLVERAPI cusolverRfDestroy(cusolverRfHandle_t handle);

/* CUSOLVERRF set and get input format */
cusolverStatus_t CUSOLVERAPI cusolverRfGetMatrixFormat(cusolverRfHandle_t handle, 
                                                       cusolverRfMatrixFormat_t *format, 
                                                       cusolverRfUnitDiagonal_t *diag);

cusolverStatus_t CUSOLVERAPI cusolverRfSetMatrixFormat(cusolverRfHandle_t handle, 
                                                       cusolverRfMatrixFormat_t format, 
                                                       cusolverRfUnitDiagonal_t diag);
    
/* CUSOLVERRF set and get numeric properties */
cusolverStatus_t CUSOLVERAPI cusolverRfSetNumericProperties(cusolverRfHandle_t handle, 
                                                            double zero,
                                                            double boost);
											 
cusolverStatus_t CUSOLVERAPI cusolverRfGetNumericProperties(cusolverRfHandle_t handle, 
                                                            double* zero,
                                                            double* boost);
											 
cusolverStatus_t CUSOLVERAPI cusolverRfGetNumericBoostReport(cusolverRfHandle_t handle, 
                                                             cusolverRfNumericBoostReport_t *report);

/* CUSOLVERRF choose the triangular solve algorithm */
cusolverStatus_t CUSOLVERAPI cusolverRfSetAlgs(cusolverRfHandle_t handle,
                                               cusolverRfFactorization_t factAlg,
                                               cusolverRfTriangularSolve_t solveAlg);

cusolverStatus_t CUSOLVERAPI cusolverRfGetAlgs(cusolverRfHandle_t handle, 
                                               cusolverRfFactorization_t* factAlg,
                                               cusolverRfTriangularSolve_t* solveAlg);

/* CUSOLVERRF set and get fast mode */
cusolverStatus_t CUSOLVERAPI cusolverRfGetResetValuesFastMode(cusolverRfHandle_t handle, 
                                                              cusolverRfResetValuesFastMode_t *fastMode);

cusolverStatus_t CUSOLVERAPI cusolverRfSetResetValuesFastMode(cusolverRfHandle_t handle, 
                                                              cusolverRfResetValuesFastMode_t fastMode);

/*** Non-Batched Routines ***/
/* CUSOLVERRF setup of internal structures from host or device memory */
cusolverStatus_t CUSOLVERAPI cusolverRfSetupHost(/* Input (in the host memory) */
                                                 int n,
                                                 int nnzA,
                                                 int* h_csrRowPtrA,
                                                 int* h_csrColIndA,
                                                 double* h_csrValA,
                                                 int nnzL,
                                                 int* h_csrRowPtrL,
                                                 int* h_csrColIndL,
                                                 double* h_csrValL,
                                                 int nnzU,
                                                 int* h_csrRowPtrU,
                                                 int* h_csrColIndU,
                                                 double* h_csrValU,
                                                 int* h_P,
                                                 int* h_Q,
                                                 /* Output */
                                                 cusolverRfHandle_t handle);
    
cusolverStatus_t CUSOLVERAPI cusolverRfSetupDevice(/* Input (in the device memory) */
                                                   int n,
                                                   int nnzA,
                                                   int* csrRowPtrA,
                                                   int* csrColIndA,
                                                   double* csrValA,
                                                   int nnzL,
                                                   int* csrRowPtrL,
                                                   int* csrColIndL,
                                                   double* csrValL,
                                                   int nnzU,
                                                   int* csrRowPtrU,
                                                   int* csrColIndU,
                                                   double* csrValU,
                                                   int* P,
                                                   int* Q,
                                                   /* Output */
                                                   cusolverRfHandle_t handle);

/* CUSOLVERRF update the matrix values (assuming the reordering, pivoting 
   and consequently the sparsity pattern of L and U did not change),
   and zero out the remaining values. */
cusolverStatus_t CUSOLVERAPI cusolverRfResetValues(/* Input (in the device memory) */
                                                   int n,
                                                   int nnzA,
                                                   int* csrRowPtrA, 
                                                   int* csrColIndA, 
                                                   double* csrValA,
                                                   int* P,
                                                   int* Q,
                                                   /* Output */
                                                   cusolverRfHandle_t handle);

/* CUSOLVERRF analysis (for parallelism) */
cusolverStatus_t CUSOLVERAPI cusolverRfAnalyze(cusolverRfHandle_t handle);

/* CUSOLVERRF re-factorization (for parallelism) */
cusolverStatus_t CUSOLVERAPI cusolverRfRefactor(cusolverRfHandle_t handle);

/* CUSOLVERRF extraction: Get L & U packed into a single matrix M */
cusolverStatus_t CUSOLVERAPI cusolverRfAccessBundledFactorsDevice(/* Input */
                                                                  cusolverRfHandle_t handle,
                                                                  /* Output (in the host memory) */
                                                                  int* nnzM, 
                                                                  /* Output (in the device memory) */
                                                                  int** Mp, 
                                                                  int** Mi, 
                                                                  double** Mx);

cusolverStatus_t CUSOLVERAPI cusolverRfExtractBundledFactorsHost(/* Input */
                                                                 cusolverRfHandle_t handle, 
                                                                 /* Output (in the host memory) */
                                                                 int* h_nnzM,
                                                                 int** h_Mp, 
                                                                 int** h_Mi, 
                                                                 double** h_Mx);

/* CUSOLVERRF extraction: Get L & U individually */
cusolverStatus_t CUSOLVERAPI cusolverRfExtractSplitFactorsHost(/* Input */
                                                               cusolverRfHandle_t handle, 
                                                               /* Output (in the host memory) */
                                                               int* h_nnzL, 
                                                               int** h_csrRowPtrL, 
                                                               int** h_csrColIndL, 
                                                               double** h_csrValL, 
                                                               int* h_nnzU, 
                                                               int** h_csrRowPtrU, 
                                                               int** h_csrColIndU, 
                                                               double** h_csrValU);

/* CUSOLVERRF (forward and backward triangular) solves */
cusolverStatus_t CUSOLVERAPI cusolverRfSolve(/* Input (in the device memory) */
                                             cusolverRfHandle_t handle,
                                             int *P,
                                             int *Q,
                                             int nrhs,     //only nrhs=1 is supported
                                             double *Temp, //of size ldt*nrhs (ldt>=n)
                                             int ldt,      
                                             /* Input/Output (in the device memory) */
                                             double *XF,
                                             /* Input */
                                             int ldxf);

/*** Batched Routines ***/
/* CUSOLVERRF-batch setup of internal structures from host */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchSetupHost(/* Input (in the host memory)*/
                                                      int batchSize,
                                                      int n,
                                                      int nnzA,
                                                      int* h_csrRowPtrA,
                                                      int* h_csrColIndA,
                                                      double* h_csrValA_array[],
                                                      int nnzL,
                                                      int* h_csrRowPtrL,
                                                      int* h_csrColIndL,
                                                      double *h_csrValL,
                                                      int nnzU,
                                                      int* h_csrRowPtrU,
                                                      int* h_csrColIndU,
                                                      double *h_csrValU,
                                                      int* h_P,
                                                      int* h_Q,
                                                      /* Output (in the device memory) */
                                                      cusolverRfHandle_t handle);

/* CUSOLVERRF-batch update the matrix values (assuming the reordering, pivoting 
   and consequently the sparsity pattern of L and U did not change),
   and zero out the remaining values. */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchResetValues(/* Input (in the device memory) */
                                                        int batchSize,
                                                        int n,
                                                        int nnzA,
                                                        int* csrRowPtrA,
                                                        int* csrColIndA,
                                                        double* csrValA_array[],
                                                        int* P,
                                                        int* Q,
                                                        /* Output */
                                                        cusolverRfHandle_t handle);
 
/* CUSOLVERRF-batch analysis (for parallelism) */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchAnalyze(cusolverRfHandle_t handle);

/* CUSOLVERRF-batch re-factorization (for parallelism) */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchRefactor(cusolverRfHandle_t handle);

/* CUSOLVERRF-batch (forward and backward triangular) solves */
cusolverStatus_t CUSOLVERAPI cusolverRfBatchSolve(/* Input (in the device memory) */
                                                  cusolverRfHandle_t handle,
                                                  int *P,
                                                  int *Q,
                                                  int nrhs,     //only nrhs=1 is supported
                                                  double *Temp, //of size 2*batchSize*(n*nrhs)
                                                  int ldt,      //only ldt=n is supported
                                                  /* Input/Output (in the device memory) */
                                                  double *XF_array[],
                                                  /* Input */
                                                  int ldxf);

/* CUSOLVERRF-batch obtain the position of zero pivot */    
cusolverStatus_t CUSOLVERAPI cusolverRfBatchZeroPivot(/* Input */
                                                      cusolverRfHandle_t handle,
                                                      /* Output (in the host memory) */
                                                      int *position);

// cuda/targets/x86_64-linux/include/cusolverSp.h
struct cusolverSpContext;
typedef struct cusolverSpContext *cusolverSpHandle_t;

struct csrqrInfo;
typedef struct csrqrInfo *csrqrInfo_t;

cusolverStatus_t CUSOLVERAPI cusolverSpCreate(cusolverSpHandle_t *handle);
cusolverStatus_t CUSOLVERAPI cusolverSpDestroy(cusolverSpHandle_t handle);
cusolverStatus_t CUSOLVERAPI cusolverSpSetStream (cusolverSpHandle_t handle, cudaStream_t streamId);
cusolverStatus_t CUSOLVERAPI cusolverSpGetStream(cusolverSpHandle_t handle, cudaStream_t *streamId);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrissymHost(
    cusolverSpHandle_t handle,
    int m,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,
    int *issym);

/* -------- GPU linear solver by LU factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [lu] stands for LU factorization
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvluHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol, 
    int reorder,
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvluHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvluHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *b,
    float tol,
    int reorder,
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvluHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    cuDoubleComplex *x,
    int *singularity);


/* -------- GPU linear solver by QR factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [qr] stands for QR factorization
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuComplex *b,
    float tol,
    int reorder,
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    cuDoubleComplex *x,
    int *singularity);



/* -------- CPU linear solver by QR factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [qr] stands for QR factorization
 */ 
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *b,
    float tol,
    int reorder,
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    cuDoubleComplex *x,
    int *singularity);


/* -------- CPU linear solver by Cholesky factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [chol] stands for Cholesky factorization
 *
 * Only works for symmetric positive definite matrix.
 * The upper part of A is ignored.
 */ 
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvcholHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvcholHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvcholHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuComplex *b,
    float tol,
    int reorder,
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvcholHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    cuDoubleComplex *x,
    int *singularity);

/* -------- GPU linear solver by Cholesky factorization
 *       solve A*x = b, A can be singular 
 * [ls] stands for linear solve
 * [v] stands for vector
 * [chol] stands for Cholesky factorization
 *
 * Only works for symmetric positive definite matrix.
 * The upper part of A is ignored.
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *b,
    float tol,
    int reorder,
    // output
    float *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *b,
    double tol,
    int reorder,
    // output
    double *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuComplex *b,
    float tol,
    int reorder,
    // output
    cuComplex *x,
    int *singularity);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsvchol(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuDoubleComplex *b,
    double tol,
    int reorder,
    // output
    cuDoubleComplex *x,
    int *singularity);



/* ----------- CPU least square solver by QR factorization
 *       solve min|b - A*x| 
 * [lsq] stands for least square
 * [v] stands for vector
 * [qr] stands for QR factorization
 */ 
cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsqvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol,
    int *rankA,
    float *x,
    int *p,
    float *min_norm);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsqvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int *rankA,
    double *x,
    int *p,
    double *min_norm);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrlsqvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *b,
    float tol,
    int *rankA,
    cuComplex *x,
    int *p,
    float *min_norm);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrlsqvqrHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *b,
    double tol,
    int *rankA,
    cuDoubleComplex *x,
    int *p,
    double *min_norm);

/* --------- CPU eigenvalue solver by shift inverse
 *      solve A*x = lambda * x 
 *   where lambda is the eigenvalue nearest mu0.
 * [eig] stands for eigenvalue solver
 * [si] stands for shift-inverse
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsreigvsiHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu0,
    const float *x0,
    int maxite,
    float tol,
    float *mu,
    float *x);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsreigvsiHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu0,
    const double *x0,
    int maxite,
    double tol,
    double *mu,
    double *x);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsreigvsiHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex mu0,
    const cuComplex *x0,
    int maxite,
    float tol,
    cuComplex *mu,
    cuComplex *x);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsreigvsiHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex mu0,
    const cuDoubleComplex *x0,
    int maxite,
    double tol,
    cuDoubleComplex *mu,
    cuDoubleComplex *x);


/* --------- GPU eigenvalue solver by shift inverse
 *      solve A*x = lambda * x 
 *   where lambda is the eigenvalue nearest mu0.
 * [eig] stands for eigenvalue solver
 * [si] stands for shift-inverse
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsreigvsi(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu0,
    const float *x0,
    int maxite,
    float eps,
    float *mu,
    float *x);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsreigvsi(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu0,
    const double *x0,
    int maxite,
    double eps,
    double *mu, 
    double *x);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsreigvsi(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex mu0,
    const cuComplex *x0,
    int maxite,
    float eps,
    cuComplex *mu, 
    cuComplex *x);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsreigvsi(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex mu0,
    const cuDoubleComplex *x0,
    int maxite,
    double eps,
    cuDoubleComplex *mu, 
    cuDoubleComplex *x);


// ----------- enclosed eigenvalues

cusolverStatus_t CUSOLVERAPI cusolverSpScsreigsHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex left_bottom_corner,
    cuComplex right_upper_corner,
    int *num_eigs);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsreigsHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex left_bottom_corner,
    cuDoubleComplex right_upper_corner,
    int *num_eigs);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsreigsHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex left_bottom_corner,
    cuComplex right_upper_corner,
    int *num_eigs);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsreigsHost(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex left_bottom_corner,
    cuDoubleComplex right_upper_corner,
    int *num_eigs);



/* --------- CPU symrcm
 *   Symmetric reverse Cuthill McKee permutation         
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymrcmHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU symmdq 
 *   Symmetric minimum degree algorithm by quotient graph
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymmdqHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU symmdq 
 *   Symmetric Approximate minimum degree algorithm by quotient graph
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrsymamdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *p);

/* --------- CPU metis 
 *   symmetric reordering 
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrmetisndHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const int64_t *options,
    int *p);


/* --------- CPU zfd
 *  Zero free diagonal reordering
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrzfdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrzfdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrzfdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrzfdHost(
    cusolverSpHandle_t handle,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *P,
    int *numnz);


/* --------- CPU permuation
 *   P*A*Q^T        
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrperm_bufferSizeHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const int *p,
    const int *q,
    size_t *bufferSizeInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrpermHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    int *csrRowPtrA,
    int *csrColIndA,
    const int *p,
    const int *q,
    int *map,
    void *pBuffer);



/*
 *  Low-level API: Batched QR
 *
 */

cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrqrInfo(
    csrqrInfo_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrqrInfo(
    csrqrInfo_t info);


cusolverStatus_t CUSOLVERAPI cusolverSpXcsrqrAnalysisBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrBufferInfoBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrsvBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,   
    float *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrsvBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,   
    double *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrsvBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *b, 
    cuComplex *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrsvBatched(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *b,  
    cuDoubleComplex *x,  
    int batchSize,
    csrqrInfo_t info,
    void *pBuffer);

// cuda/targets/x86_64-linux/include/cusolverSp_LOWLEVEL_PREVIEW.h
struct csrluInfoHost;
typedef struct csrluInfoHost *csrluInfoHost_t;


struct csrqrInfoHost;
typedef struct csrqrInfoHost *csrqrInfoHost_t;


struct csrcholInfoHost;
typedef struct csrcholInfoHost *csrcholInfoHost_t;


struct csrcholInfo;
typedef struct csrcholInfo *csrcholInfo_t;



/*
 * Low level API for CPU LU
 * 
 */
cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrluInfoHost(
    csrluInfoHost_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrluInfoHost(
    csrluInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrluAnalysisHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrluBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrluFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    float pivot_threshold,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    double pivot_threshold,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    float pivot_threshold,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrluInfoHost_t info,
    double pivot_threshold,
    void *pBuffer);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrluZeroPivotHost(
    cusolverSpHandle_t handle,
    csrluInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluZeroPivotHost(
    cusolverSpHandle_t handle,
    csrluInfoHost_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluZeroPivotHost(
    cusolverSpHandle_t handle,
    csrluInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluZeroPivotHost(
    cusolverSpHandle_t handle,
    csrluInfoHost_t info,
    double tol,
    int *position);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrluSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const cuComplex *b,
    cuComplex *x,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrluNnzHost(
    cusolverSpHandle_t handle,
    int *nnzLRef,
    int *nnzURef,
    csrluInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrluExtractHost(
    cusolverSpHandle_t handle,
    int *P,
    int *Q,
    const cusparseMatDescr_t descrL,
    float *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const cusparseMatDescr_t descrU,
    float *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrluExtractHost(
    cusolverSpHandle_t handle,
    int *P,
    int *Q,
    const cusparseMatDescr_t descrL,
    double *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const cusparseMatDescr_t descrU,
    double *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrluExtractHost(
    cusolverSpHandle_t handle,
    int *P,
    int *Q,
    const cusparseMatDescr_t descrL,
    cuComplex *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const cusparseMatDescr_t descrU,
    cuComplex *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrluExtractHost(
    cusolverSpHandle_t handle,
    int *P,
    int *Q,
    const cusparseMatDescr_t descrL,
    cuDoubleComplex *csrValL,
    int *csrRowPtrL,
    int *csrColIndL,
    const cusparseMatDescr_t descrU,
    cuDoubleComplex *csrValU,
    int *csrRowPtrU,
    int *csrColIndU,
    csrluInfoHost_t info,
    void *pBuffer);


/*
 * Low level API for CPU QR
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrqrInfoHost(
    csrqrInfoHost_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrqrInfoHost(
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrqrAnalysisHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrBufferInfoHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrBufferInfoHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrBufferInfoHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrBufferInfoHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSetupHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrSetupHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrSetupHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex mu,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrSetupHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex mu,
    csrqrInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrFactorHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    float *b,
    float *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrFactorHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    double *b,
    double *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrFactorHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    cuComplex *b,
    cuComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrFactorHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrZeroPivotHost(
    cusolverSpHandle_t handle,
    csrqrInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrZeroPivotHost(
    cusolverSpHandle_t handle,
    csrqrInfoHost_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrZeroPivotHost(
    cusolverSpHandle_t handle,
    csrqrInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrZeroPivotHost(
    cusolverSpHandle_t handle,
    csrqrInfoHost_t info,
    double tol,
    int *position);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSolveHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    float *b,
    float *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrSolveHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    double *b,
    double *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrSolveHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    cuComplex *b,
    cuComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrSolveHost(
    cusolverSpHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrqrInfoHost_t info,
    void *pBuffer);


/*
 * Low level API for GPU QR
 *
 */
cusolverStatus_t CUSOLVERAPI cusolverSpXcsrqrAnalysis(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrBufferInfo(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrBufferInfo(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrBufferInfo(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrBufferInfo(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrqrInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSetup(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    float mu,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrSetup(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    double mu,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrSetup(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuComplex mu,
    csrqrInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrSetup(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    cuDoubleComplex mu,
    csrqrInfo_t info);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrFactor(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    float *b,
    float *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrFactor(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    double *b,
    double *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrFactor(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    cuComplex *b,
    cuComplex *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrFactor(
    cusolverSpHandle_t handle,
    int m,
    int n,
    int nnzA,
    cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrqrInfo_t info,
    void *pBuffer);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrZeroPivot(
    cusolverSpHandle_t handle,
    csrqrInfo_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrZeroPivot(
    cusolverSpHandle_t handle,
    csrqrInfo_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrZeroPivot(
    cusolverSpHandle_t handle,
    csrqrInfo_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrZeroPivot(
    cusolverSpHandle_t handle,
    csrqrInfo_t info,
    double tol,
    int *position);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSolve(
    cusolverSpHandle_t handle,
    int m,
    int n,
    float *b,
    float *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrqrSolve(
    cusolverSpHandle_t handle,
    int m,
    int n,
    double *b,
    double *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrqrSolve(
    cusolverSpHandle_t handle,
    int m,
    int n,
    cuComplex *b,
    cuComplex *x,
    csrqrInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrqrSolve(
    cusolverSpHandle_t handle,
    int m,
    int n,
    cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrqrInfo_t info,
    void *pBuffer);


/*
 * Low level API for CPU Cholesky
 * 
 */
cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrcholInfoHost(
    csrcholInfoHost_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrcholInfoHost(
    csrcholInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholAnalysisHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholBufferInfoHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);


cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholFactorHost(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholZeroPivotHost(
    cusolverSpHandle_t handle,
    csrcholInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholZeroPivotHost(
    cusolverSpHandle_t handle,
    csrcholInfoHost_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholZeroPivotHost(
    cusolverSpHandle_t handle,
    csrcholInfoHost_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholZeroPivotHost(
    cusolverSpHandle_t handle,
    csrcholInfoHost_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const cuComplex *b,
    cuComplex *x,
    csrcholInfoHost_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholSolveHost(
    cusolverSpHandle_t handle,
    int n,
    const cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrcholInfoHost_t info,
    void *pBuffer);

/*
 * Low level API for GPU Cholesky
 * 
 */
cusolverStatus_t CUSOLVERAPI cusolverSpCreateCsrcholInfo(
    csrcholInfo_t *info);

cusolverStatus_t CUSOLVERAPI cusolverSpDestroyCsrcholInfo(
    csrcholInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholAnalysis(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholBufferInfo(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholBufferInfo(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholBufferInfo(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholBufferInfo(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    size_t *internalDataInBytes,
    size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholFactor(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholFactor(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholFactor(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholFactor(
    cusolverSpHandle_t handle,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholZeroPivot(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholZeroPivot(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholZeroPivot(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    float tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholZeroPivot(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    double tol,
    int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholSolve(
    cusolverSpHandle_t handle,
    int n,
    const float *b,
    float *x,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholSolve(
    cusolverSpHandle_t handle,
    int n,
    const double *b,
    double *x,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholSolve(
    cusolverSpHandle_t handle,
    int n,
    const cuComplex *b,
    cuComplex *x,
    csrcholInfo_t info,
    void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholSolve(
    cusolverSpHandle_t handle,
    int n,
    const cuDoubleComplex *b,
    cuDoubleComplex *x,
    csrcholInfo_t info,
    void *pBuffer);

/*
 * "diag" is a device array of size N.
 * cusolverSp<t>csrcholDiag returns diag(L) to "diag" where A(P,P) = L*L**T
 * "diag" can estimate det(A) because det(A(P,P)) = det(A) = det(L)^2 if A = L*L**T.
 * 
 * cusolverSp<t>csrcholDiag must be called after cusolverSp<t>csrcholFactor.
 * otherwise "diag" is wrong.
 */
cusolverStatus_t CUSOLVERAPI cusolverSpScsrcholDiag(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    float *diag);

cusolverStatus_t CUSOLVERAPI cusolverSpDcsrcholDiag(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    double *diag);

cusolverStatus_t CUSOLVERAPI cusolverSpCcsrcholDiag(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    float *diag);

cusolverStatus_t CUSOLVERAPI cusolverSpZcsrcholDiag(
    cusolverSpHandle_t handle,
    csrcholInfo_t info,
    double *diag);

#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif // CUSOLVER_COMMON_H_



