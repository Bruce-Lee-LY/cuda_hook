// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: cusolver subset

#ifndef __CUDA_HOOK_CUSOLVER_SUBSET_H__
#define __CUDA_HOOK_CUSOLVER_SUBSET_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef int cusolver_int_t;

#define CUSOLVER_VER_MAJOR 11
#define CUSOLVER_VER_MINOR 2
#define CUSOLVER_VER_PATCH 0
#define CUSOLVER_VER_BUILD 120
#define CUSOLVER_VERSION (CUSOLVER_VER_MAJOR * 1000 + CUSOLVER_VER_MINOR * 100 + CUSOLVER_VER_PATCH)

typedef enum {
    CUSOLVER_STATUS_SUCCESS = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED = 1,
    CUSOLVER_STATUS_ALLOC_FAILED = 2,
    CUSOLVER_STATUS_INVALID_VALUE = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH = 4,
    CUSOLVER_STATUS_MAPPING_ERROR = 5,
    CUSOLVER_STATUS_EXECUTION_FAILED = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT = 10,
    CUSOLVER_STATUS_INVALID_LICENSE = 11,
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID = 13,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC = 14,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE = 15,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER = 16,
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR = 20,
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED = 21,
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE = 22,
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23,
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 25,
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 26,
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR = 30,
    CUSOLVER_STATUS_INVALID_WORKSPACE = 31
} cusolverStatus_t;

typedef enum { CUSOLVER_EIG_TYPE_1 = 1, CUSOLVER_EIG_TYPE_2 = 2, CUSOLVER_EIG_TYPE_3 = 3 } cusolverEigType_t;

typedef enum { CUSOLVER_EIG_MODE_NOVECTOR = 0, CUSOLVER_EIG_MODE_VECTOR = 1 } cusolverEigMode_t;

typedef enum {
    CUSOLVER_EIG_RANGE_ALL = 1001,
    CUSOLVER_EIG_RANGE_I = 1002,
    CUSOLVER_EIG_RANGE_V = 1003,
} cusolverEigRange_t;

typedef enum {
    CUSOLVER_INF_NORM = 104,
    CUSOLVER_MAX_NORM = 105,
    CUSOLVER_ONE_NORM = 106,
    CUSOLVER_FRO_NORM = 107,
} cusolverNorm_t;

typedef enum {
    CUSOLVER_IRS_REFINE_NOT_SET = 1100,
    CUSOLVER_IRS_REFINE_NONE = 1101,
    CUSOLVER_IRS_REFINE_CLASSICAL = 1102,
    CUSOLVER_IRS_REFINE_CLASSICAL_GMRES = 1103,
    CUSOLVER_IRS_REFINE_GMRES = 1104,
    CUSOLVER_IRS_REFINE_GMRES_GMRES = 1105,
    CUSOLVER_IRS_REFINE_GMRES_NOPCOND = 1106,

    CUSOLVER_PREC_DD = 1150,
    CUSOLVER_PREC_SS = 1151,
    CUSOLVER_PREC_SHT = 1152,

} cusolverIRSRefinement_t;

typedef enum {
    CUSOLVER_R_8I = 1201,
    CUSOLVER_R_8U = 1202,
    CUSOLVER_R_64F = 1203,
    CUSOLVER_R_32F = 1204,
    CUSOLVER_R_16F = 1205,
    CUSOLVER_R_16BF = 1206,
    CUSOLVER_R_TF32 = 1207,
    CUSOLVER_R_AP = 1208,
    CUSOLVER_C_8I = 1211,
    CUSOLVER_C_8U = 1212,
    CUSOLVER_C_64F = 1213,
    CUSOLVER_C_32F = 1214,
    CUSOLVER_C_16F = 1215,
    CUSOLVER_C_16BF = 1216,
    CUSOLVER_C_TF32 = 1217,
    CUSOLVER_C_AP = 1218,
} cusolverPrecType_t;

typedef enum {
    CUSOLVER_ALG_0 = 0, /* default algorithm */
    CUSOLVER_ALG_1 = 1
} cusolverAlgMode_t;

typedef enum { CUBLAS_STOREV_COLUMNWISE = 0, CUBLAS_STOREV_ROWWISE = 1 } cusolverStorevMode_t;

typedef enum { CUBLAS_DIRECT_FORWARD = 0, CUBLAS_DIRECT_BACKWARD = 1 } cusolverDirectMode_t;

struct cusolverDnContext;
typedef struct cusolverDnContext *cusolverDnHandle_t;

struct syevjInfo;
typedef struct syevjInfo *syevjInfo_t;

struct gesvdjInfo;
typedef struct gesvdjInfo *gesvdjInfo_t;

//------------------------------------------------------
// opaque cusolverDnIRS structure for IRS solver
struct cusolverDnIRSParams;
typedef struct cusolverDnIRSParams *cusolverDnIRSParams_t;

struct cusolverDnIRSInfos;
typedef struct cusolverDnIRSInfos *cusolverDnIRSInfos_t;
//------------------------------------------------------

struct cusolverDnParams;
typedef struct cusolverDnParams *cusolverDnParams_t;

typedef enum { CUSOLVERDN_GETRF = 0 } cusolverDnFunction_t;

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
typedef void *cudaLibMgGrid_t;
/** \brief Opaque structure of the distributed matrix descriptor */
typedef void *cudaLibMgMatrixDesc_t;

/* CUSOLVERRF mode */
typedef enum {
    CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0,  // default
    CUSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1
} cusolverRfResetValuesFastMode_t;

/* CUSOLVERRF matrix format */
typedef enum {
    CUSOLVERRF_MATRIX_FORMAT_CSR = 0,  // default
    CUSOLVERRF_MATRIX_FORMAT_CSC = 1
} cusolverRfMatrixFormat_t;

/* CUSOLVERRF unit diagonal */
typedef enum {
    CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0,  // default
    CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1,
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3
} cusolverRfUnitDiagonal_t;

/* CUSOLVERRF factorization algorithm */
typedef enum {
    CUSOLVERRF_FACTORIZATION_ALG0 = 0,  // default
    CUSOLVERRF_FACTORIZATION_ALG1 = 1,
    CUSOLVERRF_FACTORIZATION_ALG2 = 2,
} cusolverRfFactorization_t;

/* CUSOLVERRF triangular solve algorithm */
typedef enum {
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1,  // default
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
} cusolverRfTriangularSolve_t;

/* CUSOLVERRF numeric boost report */
typedef enum {
    CUSOLVERRF_NUMERIC_BOOST_NOT_USED = 0,  // default
    CUSOLVERRF_NUMERIC_BOOST_USED = 1
} cusolverRfNumericBoostReport_t;

/* Opaque structure holding CUSOLVERRF library common */
struct cusolverRfCommon;
typedef struct cusolverRfCommon *cusolverRfHandle_t;

struct cusolverSpContext;
typedef struct cusolverSpContext *cusolverSpHandle_t;

struct csrqrInfo;
typedef struct csrqrInfo *csrqrInfo_t;

struct csrluInfoHost;
typedef struct csrluInfoHost *csrluInfoHost_t;

struct csrqrInfoHost;
typedef struct csrqrInfoHost *csrqrInfoHost_t;

struct csrcholInfoHost;
typedef struct csrcholInfoHost *csrcholInfoHost_t;

struct csrcholInfo;
typedef struct csrcholInfo *csrcholInfo_t;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_CUSOLVER_SUBSET_H__
