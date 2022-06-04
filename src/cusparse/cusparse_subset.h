// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: cusparse subset

#ifndef __CUDA_HOOK_CUSPARSE_SUBSET_H__
#define __CUDA_HOOK_CUSPARSE_SUBSET_H__

#ifdef __cplusplus
extern "C" {
#endif

//##############################################################################
//# CUSPARSE VERSION INFORMATION
//##############################################################################

#define CUSPARSE_VER_MAJOR 11
#define CUSPARSE_VER_MINOR 6
#define CUSPARSE_VER_PATCH 0
#define CUSPARSE_VER_BUILD 120
#define CUSPARSE_VERSION (CUSPARSE_VER_MAJOR * 1000 + CUSPARSE_VER_MINOR * 100 + CUSPARSE_VER_PATCH)

//------------------------------------------------------------------------------

struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;

struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

struct csrsv2Info;
typedef struct csrsv2Info *csrsv2Info_t;

struct csrsm2Info;
typedef struct csrsm2Info *csrsm2Info_t;

struct bsrsv2Info;
typedef struct bsrsv2Info *bsrsv2Info_t;

struct bsrsm2Info;
typedef struct bsrsm2Info *bsrsm2Info_t;

struct csric02Info;
typedef struct csric02Info *csric02Info_t;

struct bsric02Info;
typedef struct bsric02Info *bsric02Info_t;

struct csrilu02Info;
typedef struct csrilu02Info *csrilu02Info_t;

struct bsrilu02Info;
typedef struct bsrilu02Info *bsrilu02Info_t;

struct csrgemm2Info;
typedef struct csrgemm2Info *csrgemm2Info_t;

struct csru2csrInfo;
typedef struct csru2csrInfo *csru2csrInfo_t;

struct cusparseColorInfo;
typedef struct cusparseColorInfo *cusparseColorInfo_t;

struct pruneInfo;
typedef struct pruneInfo *pruneInfo_t;

//##############################################################################
//# ENUMERATORS
//##############################################################################

typedef enum {
    CUSPARSE_STATUS_SUCCESS = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED = 1,
    CUSPARSE_STATUS_ALLOC_FAILED = 2,
    CUSPARSE_STATUS_INVALID_VALUE = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH = 4,
    CUSPARSE_STATUS_MAPPING_ERROR = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED = 10,
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11
} cusparseStatus_t;

typedef enum { CUSPARSE_POINTER_MODE_HOST = 0, CUSPARSE_POINTER_MODE_DEVICE = 1 } cusparsePointerMode_t;

typedef enum { CUSPARSE_ACTION_SYMBOLIC = 0, CUSPARSE_ACTION_NUMERIC = 1 } cusparseAction_t;

typedef enum {
    CUSPARSE_MATRIX_TYPE_GENERAL = 0,
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} cusparseMatrixType_t;

typedef enum { CUSPARSE_FILL_MODE_LOWER = 0, CUSPARSE_FILL_MODE_UPPER = 1 } cusparseFillMode_t;

typedef enum { CUSPARSE_DIAG_TYPE_NON_UNIT = 0, CUSPARSE_DIAG_TYPE_UNIT = 1 } cusparseDiagType_t;

typedef enum { CUSPARSE_INDEX_BASE_ZERO = 0, CUSPARSE_INDEX_BASE_ONE = 1 } cusparseIndexBase_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

typedef enum { CUSPARSE_DIRECTION_ROW = 0, CUSPARSE_DIRECTION_COLUMN = 1 } cusparseDirection_t;

typedef enum { CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0, CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1 } cusparseSolvePolicy_t;

typedef enum { CUSPARSE_SIDE_LEFT = 0, CUSPARSE_SIDE_RIGHT = 1 } cusparseSideMode_t;

typedef enum {
    CUSPARSE_COLOR_ALG0 = 0,  // default
    CUSPARSE_COLOR_ALG1 = 1
} cusparseColorAlg_t;

typedef enum {
    CUSPARSE_ALG_MERGE_PATH  // merge path alias
} cusparseAlgMode_t;

typedef enum {
    CUSPARSE_CSR2CSC_ALG1 = 1,  // faster than V2 (in general), deterministc
    CUSPARSE_CSR2CSC_ALG2 = 2   // low memory requirement, non-deterministc
} cusparseCsr2CscAlg_t;

typedef enum {
    CUSPARSE_FORMAT_CSR = 1,          ///< Compressed Sparse Row (CSR)
    CUSPARSE_FORMAT_CSC = 2,          ///< Compressed Sparse Column (CSC)
    CUSPARSE_FORMAT_COO = 3,          ///< Coordinate (COO) - Structure of Arrays
    CUSPARSE_FORMAT_COO_AOS = 4,      ///< Coordinate (COO) - Array of Structures
    CUSPARSE_FORMAT_BLOCKED_ELL = 5,  ///< Blocked ELL
} cusparseFormat_t;

typedef enum {
    CUSPARSE_ORDER_COL = 1,  ///< Column-Major Order - Matrix memory layout
    CUSPARSE_ORDER_ROW = 2   ///< Row-Major Order - Matrix memory layout
} cusparseOrder_t;

typedef enum {
    CUSPARSE_INDEX_16U = 1,  ///< 16-bit unsigned integer for matrix/vector
                             ///< indices
    CUSPARSE_INDEX_32I = 2,  ///< 32-bit signed integer for matrix/vector indices
    CUSPARSE_INDEX_64I = 3   ///< 64-bit signed integer for matrix/vector indices
} cusparseIndexType_t;

//------------------------------------------------------------------------------

struct cusparseSpVecDescr;
struct cusparseDnVecDescr;
struct cusparseSpMatDescr;
struct cusparseDnMatDescr;
typedef struct cusparseSpVecDescr *cusparseSpVecDescr_t;
typedef struct cusparseDnVecDescr *cusparseDnVecDescr_t;
typedef struct cusparseSpMatDescr *cusparseSpMatDescr_t;
typedef struct cusparseDnMatDescr *cusparseDnMatDescr_t;

typedef enum { CUSPARSE_SPMAT_FILL_MODE, CUSPARSE_SPMAT_DIAG_TYPE } cusparseSpMatAttribute_t;

typedef enum { CUSPARSE_SPARSETODENSE_ALG_DEFAULT = 0 } cusparseSparseToDenseAlg_t;

typedef enum { CUSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0 } cusparseDenseToSparseAlg_t;

typedef enum {
    CUSPARSE_MV_ALG_DEFAULT
    /*CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMV_ALG_DEFAULT)*/
    = 0,
    // CUSPARSE_COOMV_ALG CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMV_COO_ALG1) = 1,
    // CUSPARSE_CSRMV_ALG1 CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMV_CSR_ALG1) = 2,
    // CUSPARSE_CSRMV_ALG2 CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMV_CSR_ALG2) = 3,
    CUSPARSE_SPMV_ALG_DEFAULT = 0,
    CUSPARSE_SPMV_CSR_ALG1 = 2,
    CUSPARSE_SPMV_CSR_ALG2 = 3,
    CUSPARSE_SPMV_COO_ALG1 = 1,
    CUSPARSE_SPMV_COO_ALG2 = 4
} cusparseSpMVAlg_t;

typedef enum {
    CUSPARSE_SPSV_ALG_DEFAULT = 0,
} cusparseSpSVAlg_t;

struct cusparseSpSVDescr;
typedef struct cusparseSpSVDescr *cusparseSpSVDescr_t;

typedef enum {
    CUSPARSE_SPSM_ALG_DEFAULT = 0,
} cusparseSpSMAlg_t;

struct cusparseSpSMDescr;
typedef struct cusparseSpSMDescr *cusparseSpSMDescr_t;

typedef enum {
    // CUSPARSE_MM_ALG_DEFAULT CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_ALG_DEFAULT) = 0,
    // CUSPARSE_COOMM_ALG1 CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_COO_ALG1) = 1,
    // CUSPARSE_COOMM_ALG2 CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_COO_ALG2) = 2,
    // CUSPARSE_COOMM_ALG3 CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_COO_ALG3) = 3,
    // CUSPARSE_CSRMM_ALG1 CUSPARSE_DEPRECATED_ENUM(CUSPARSE_SPMM_CSR_ALG1) = 4,
    CUSPARSE_SPMM_ALG_DEFAULT = 0,
    CUSPARSE_SPMM_COO_ALG1 = 1,
    CUSPARSE_SPMM_COO_ALG2 = 2,
    CUSPARSE_SPMM_COO_ALG3 = 3,
    CUSPARSE_SPMM_COO_ALG4 = 5,
    CUSPARSE_SPMM_CSR_ALG1 = 4,
    CUSPARSE_SPMM_CSR_ALG2 = 6,
    CUSPARSE_SPMM_CSR_ALG3 = 12,
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
} cusparseSpMMAlg_t;

typedef enum {
    CUSPARSE_SPGEMM_DEFAULT = 0,
    CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC = 1,
    CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2
} cusparseSpGEMMAlg_t;

struct cusparseSpGEMMDescr;
typedef struct cusparseSpGEMMDescr *cusparseSpGEMMDescr_t;

typedef enum { CUSPARSE_SDDMM_ALG_DEFAULT = 0 } cusparseSDDMMAlg_t;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_CUSPARSE_SUBSET_H__
