// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 515 apis

#include "cublas_subset.h"
#include "cudart_subset.h"
#include "cusparse_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
    HOOK_TRACE_PROFILE("cusparseCreate");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
    HOOK_TRACE_PROFILE("cusparseDestroy");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int *version) {
    HOOK_TRACE_PROFILE("cusparseGetVersion");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, version);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseGetProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("cusparseGetProperty");
    using func_ptr = cusparseStatus_t (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT const char *cusparseGetErrorName(cusparseStatus_t status) {
    HOOK_TRACE_PROFILE("cusparseGetErrorName");
    using func_ptr = const char *(*)(cusparseStatus_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetErrorName"));
    HOOK_CHECK(func_entry);
    return func_entry(status);
}

HOOK_C_API HOOK_DECL_EXPORT const char *cusparseGetErrorString(cusparseStatus_t status) {
    HOOK_TRACE_PROFILE("cusparseGetErrorString");
    using func_ptr = const char *(*)(cusparseStatus_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetErrorString"));
    HOOK_CHECK(func_entry);
    return func_entry(status);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
    HOOK_TRACE_PROFILE("cusparseSetStream");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t *streamId) {
    HOOK_TRACE_PROFILE("cusparseGetStream");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cudaStream_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle,
                                                                    cusparsePointerMode_t *mode) {
    HOOK_TRACE_PROFILE("cusparseGetPointerMode");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparsePointerMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetPointerMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle,
                                                                    cusparsePointerMode_t mode) {
    HOOK_TRACE_PROFILE("cusparseSetPointerMode");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparsePointerMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSetPointerMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descrA) {
    HOOK_TRACE_PROFILE("cusparseCreateMatDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseMatDescr_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateMatDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
    HOOK_TRACE_PROFILE("cusparseDestroyMatDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseMatDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyMatDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCopyMatDescr(cusparseMatDescr_t dest,
                                                                  const cusparseMatDescr_t src) {
    HOOK_TRACE_PROFILE("cusparseCopyMatDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseMatDescr_t, const cusparseMatDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCopyMatDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(dest, src);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) {
    HOOK_TRACE_PROFILE("cusparseSetMatType");
    using func_ptr = cusparseStatus_t (*)(cusparseMatDescr_t, cusparseMatrixType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSetMatType"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA, type);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA) {
    HOOK_TRACE_PROFILE("cusparseGetMatType");
    using func_ptr = cusparseMatrixType_t (*)(const cusparseMatDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetMatType"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA,
                                                                    cusparseFillMode_t fillMode) {
    HOOK_TRACE_PROFILE("cusparseSetMatFillMode");
    using func_ptr = cusparseStatus_t (*)(cusparseMatDescr_t, cusparseFillMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSetMatFillMode"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA, fillMode);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseFillMode_t cusparseGetMatFillMode(const cusparseMatDescr_t descrA) {
    HOOK_TRACE_PROFILE("cusparseGetMatFillMode");
    using func_ptr = cusparseFillMode_t (*)(const cusparseMatDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetMatFillMode"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA,
                                                                    cusparseDiagType_t diagType) {
    HOOK_TRACE_PROFILE("cusparseSetMatDiagType");
    using func_ptr = cusparseStatus_t (*)(cusparseMatDescr_t, cusparseDiagType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSetMatDiagType"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA, diagType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseDiagType_t cusparseGetMatDiagType(const cusparseMatDescr_t descrA) {
    HOOK_TRACE_PROFILE("cusparseGetMatDiagType");
    using func_ptr = cusparseDiagType_t (*)(const cusparseMatDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetMatDiagType"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA,
                                                                     cusparseIndexBase_t base) {
    HOOK_TRACE_PROFILE("cusparseSetMatIndexBase");
    using func_ptr = cusparseStatus_t (*)(cusparseMatDescr_t, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSetMatIndexBase"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA, base);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) {
    HOOK_TRACE_PROFILE("cusparseGetMatIndexBase");
    using func_ptr = cusparseIndexBase_t (*)(const cusparseMatDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetMatIndexBase"));
    HOOK_CHECK(func_entry);
    return func_entry(descrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCsrsv2Info(csrsv2Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateCsrsv2Info");
    using func_ptr = cusparseStatus_t (*)(csrsv2Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCsrsv2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyCsrsv2Info(csrsv2Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyCsrsv2Info");
    using func_ptr = cusparseStatus_t (*)(csrsv2Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyCsrsv2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateCsric02Info");
    using func_ptr = cusparseStatus_t (*)(csric02Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCsric02Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyCsric02Info");
    using func_ptr = cusparseStatus_t (*)(csric02Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyCsric02Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateBsric02Info(bsric02Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateBsric02Info");
    using func_ptr = cusparseStatus_t (*)(bsric02Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateBsric02Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyBsric02Info(bsric02Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyBsric02Info");
    using func_ptr = cusparseStatus_t (*)(bsric02Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyBsric02Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateCsrilu02Info");
    using func_ptr = cusparseStatus_t (*)(csrilu02Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCsrilu02Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyCsrilu02Info");
    using func_ptr = cusparseStatus_t (*)(csrilu02Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyCsrilu02Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateBsrilu02Info(bsrilu02Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateBsrilu02Info");
    using func_ptr = cusparseStatus_t (*)(bsrilu02Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateBsrilu02Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyBsrilu02Info(bsrilu02Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyBsrilu02Info");
    using func_ptr = cusparseStatus_t (*)(bsrilu02Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyBsrilu02Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateBsrsv2Info(bsrsv2Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateBsrsv2Info");
    using func_ptr = cusparseStatus_t (*)(bsrsv2Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateBsrsv2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyBsrsv2Info(bsrsv2Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyBsrsv2Info");
    using func_ptr = cusparseStatus_t (*)(bsrsv2Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyBsrsv2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateBsrsm2Info(bsrsm2Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateBsrsm2Info");
    using func_ptr = cusparseStatus_t (*)(bsrsm2Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateBsrsm2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyBsrsm2Info(bsrsm2Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyBsrsm2Info");
    using func_ptr = cusparseStatus_t (*)(bsrsm2Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyBsrsm2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCsru2csrInfo(csru2csrInfo_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateCsru2csrInfo");
    using func_ptr = cusparseStatus_t (*)(csru2csrInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCsru2csrInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyCsru2csrInfo(csru2csrInfo_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyCsru2csrInfo");
    using func_ptr = cusparseStatus_t (*)(csru2csrInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyCsru2csrInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateColorInfo(cusparseColorInfo_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateColorInfo");
    using func_ptr = cusparseStatus_t (*)(cusparseColorInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateColorInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyColorInfo(cusparseColorInfo_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyColorInfo");
    using func_ptr = cusparseStatus_t (*)(cusparseColorInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyColorInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSetColorAlgs(cusparseColorInfo_t info, cusparseColorAlg_t alg) {
    HOOK_TRACE_PROFILE("cusparseSetColorAlgs");
    using func_ptr = cusparseStatus_t (*)(cusparseColorInfo_t, cusparseColorAlg_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSetColorAlgs"));
    HOOK_CHECK(func_entry);
    return func_entry(info, alg);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseGetColorAlgs(cusparseColorInfo_t info, cusparseColorAlg_t *alg) {
    HOOK_TRACE_PROFILE("cusparseGetColorAlgs");
    using func_ptr = cusparseStatus_t (*)(cusparseColorInfo_t, cusparseColorAlg_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGetColorAlgs"));
    HOOK_CHECK(func_entry);
    return func_entry(info, alg);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreatePruneInfo(pruneInfo_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreatePruneInfo");
    using func_ptr = cusparseStatus_t (*)(pruneInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreatePruneInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyPruneInfo(pruneInfo_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyPruneInfo");
    using func_ptr = cusparseStatus_t (*)(pruneInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyPruneInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float *alpha,
                                                            const float *xVal, const int *xInd, float *y,
                                                            cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseSaxpyi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const float *, const float *, const int *, float *,
                                          cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSaxpyi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDaxpyi(cusparseHandle_t handle, int nnz, const double *alpha,
                                                            const double *xVal, const int *xInd, double *y,
                                                            cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseDaxpyi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const double *, const double *, const int *, double *,
                                          cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDaxpyi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCaxpyi(cusparseHandle_t handle, int nnz, const cuComplex *alpha,
                                                            const cuComplex *xVal, const int *xInd, cuComplex *y,
                                                            cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseCaxpyi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cuComplex *, const cuComplex *, const int *,
                                          cuComplex *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCaxpyi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZaxpyi(cusparseHandle_t handle, int nnz,
                                                            const cuDoubleComplex *alpha, const cuDoubleComplex *xVal,
                                                            const int *xInd, cuDoubleComplex *y,
                                                            cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseZaxpyi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const int *, cuDoubleComplex *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZaxpyi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, alpha, xVal, xInd, y, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgthr(cusparseHandle_t handle, int nnz, const float *y,
                                                           float *xVal, const int *xInd, cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseSgthr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, const float *, float *, const int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgthr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, y, xVal, xInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgthr(cusparseHandle_t handle, int nnz, const double *y,
                                                           double *xVal, const int *xInd, cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseDgthr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, const double *, double *, const int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgthr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, y, xVal, xInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgthr(cusparseHandle_t handle, int nnz, const cuComplex *y,
                                                           cuComplex *xVal, const int *xInd,
                                                           cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseCgthr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, const cuComplex *, cuComplex *, const int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgthr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, y, xVal, xInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgthr(cusparseHandle_t handle, int nnz, const cuDoubleComplex *y,
                                                           cuDoubleComplex *xVal, const int *xInd,
                                                           cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseZgthr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cuDoubleComplex *, cuDoubleComplex *,
                                          const int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgthr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, y, xVal, xInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgthrz(cusparseHandle_t handle, int nnz, float *y, float *xVal,
                                                            const int *xInd, cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseSgthrz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, float *, float *, const int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgthrz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, y, xVal, xInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgthrz(cusparseHandle_t handle, int nnz, double *y, double *xVal,
                                                            const int *xInd, cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseDgthrz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, double *, double *, const int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgthrz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, y, xVal, xInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgthrz(cusparseHandle_t handle, int nnz, cuComplex *y,
                                                            cuComplex *xVal, const int *xInd,
                                                            cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseCgthrz");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cuComplex *, cuComplex *, const int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgthrz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, y, xVal, xInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgthrz(cusparseHandle_t handle, int nnz, cuDoubleComplex *y,
                                                            cuDoubleComplex *xVal, const int *xInd,
                                                            cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseZgthrz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, cuDoubleComplex *, cuDoubleComplex *, const int *,
                                          cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgthrz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, y, xVal, xInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSsctr(cusparseHandle_t handle, int nnz, const float *xVal,
                                                           const int *xInd, float *y, cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseSsctr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, const float *, const int *, float *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSsctr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, xVal, xInd, y, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDsctr(cusparseHandle_t handle, int nnz, const double *xVal,
                                                           const int *xInd, double *y, cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseDsctr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, const double *, const int *, double *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDsctr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, xVal, xInd, y, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCsctr(cusparseHandle_t handle, int nnz, const cuComplex *xVal,
                                                           const int *xInd, cuComplex *y, cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseCsctr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, const cuComplex *, const int *, cuComplex *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCsctr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, xVal, xInd, y, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZsctr(cusparseHandle_t handle, int nnz,
                                                           const cuDoubleComplex *xVal, const int *xInd,
                                                           cuDoubleComplex *y, cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseZsctr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cuDoubleComplex *, const int *,
                                          cuDoubleComplex *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZsctr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, xVal, xInd, y, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSroti(cusparseHandle_t handle, int nnz, float *xVal,
                                                           const int *xInd, float *y, const float *c, const float *s,
                                                           cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseSroti");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, float *, const int *, float *, const float *,
                                          const float *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSroti"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, xVal, xInd, y, c, s, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDroti(cusparseHandle_t handle, int nnz, double *xVal,
                                                           const int *xInd, double *y, const double *c, const double *s,
                                                           cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseDroti");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, double *, const int *, double *, const double *,
                                          const double *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDroti"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnz, xVal, xInd, y, c, s, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                            int n, const float *alpha, const float *A, int lda, int nnz,
                                                            const float *xVal, const int *xInd, const float *beta,
                                                            float *y, cusparseIndexBase_t idxBase, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSgemvi");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const float *, const float *, int, int,
                             const float *, const int *, const float *, float *, cusparseIndexBase_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgemvi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgemvi_bufferSize(cusparseHandle_t handle,
                                                                       cusparseOperation_t transA, int m, int n,
                                                                       int nnz, int *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseSgemvi_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgemvi_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, n, nnz, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                            int n, const double *alpha, const double *A, int lda,
                                                            int nnz, const double *xVal, const int *xInd,
                                                            const double *beta, double *y, cusparseIndexBase_t idxBase,
                                                            void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDgemvi");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const double *, const double *, int, int,
                             const double *, const int *, const double *, double *, cusparseIndexBase_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgemvi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgemvi_bufferSize(cusparseHandle_t handle,
                                                                       cusparseOperation_t transA, int m, int n,
                                                                       int nnz, int *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDgemvi_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgemvi_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, n, nnz, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                            int n, const cuComplex *alpha, const cuComplex *A, int lda,
                                                            int nnz, const cuComplex *xVal, const int *xInd,
                                                            const cuComplex *beta, cuComplex *y,
                                                            cusparseIndexBase_t idxBase, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCgemvi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cuComplex *,
                                          const cuComplex *, int, int, const cuComplex *, const int *,
                                          const cuComplex *, cuComplex *, cusparseIndexBase_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgemvi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgemvi_bufferSize(cusparseHandle_t handle,
                                                                       cusparseOperation_t transA, int m, int n,
                                                                       int nnz, int *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCgemvi_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgemvi_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, n, nnz, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m,
                                                            int n, const cuDoubleComplex *alpha,
                                                            const cuDoubleComplex *A, int lda, int nnz,
                                                            const cuDoubleComplex *xVal, const int *xInd,
                                                            const cuDoubleComplex *beta, cuDoubleComplex *y,
                                                            cusparseIndexBase_t idxBase, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZgemvi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cuDoubleComplex *,
                                          const cuDoubleComplex *, int, int, const cuDoubleComplex *, const int *,
                                          const cuDoubleComplex *, cuDoubleComplex *, cusparseIndexBase_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgemvi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgemvi_bufferSize(cusparseHandle_t handle,
                                                                       cusparseOperation_t transA, int m, int n,
                                                                       int nnz, int *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZgemvi_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgemvi_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, n, nnz, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCsrmvEx_bufferSize(
    cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA, int m, int n, int nnz,
    const void *alpha, cudaDataType alphatype, const cusparseMatDescr_t descrA, const void *csrValA,
    cudaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA, const void *x, cudaDataType xtype,
    const void *beta, cudaDataType betatype, void *y, cudaDataType ytype, cudaDataType executiontype,
    size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCsrmvEx_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseAlgMode_t, cusparseOperation_t, int, int, int,
                                          const void *, cudaDataType, const cusparseMatDescr_t, const void *,
                                          cudaDataType, const int *, const int *, const void *, cudaDataType,
                                          const void *, cudaDataType, void *, cudaDataType, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCsrmvEx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA,
                      csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCsrmvEx(
    cusparseHandle_t handle, cusparseAlgMode_t alg, cusparseOperation_t transA, int m, int n, int nnz,
    const void *alpha, cudaDataType alphatype, const cusparseMatDescr_t descrA, const void *csrValA,
    cudaDataType csrValAtype, const int *csrRowPtrA, const int *csrColIndA, const void *x, cudaDataType xtype,
    const void *beta, cudaDataType betatype, void *y, cudaDataType ytype, cudaDataType executiontype, void *buffer) {
    HOOK_TRACE_PROFILE("cusparseCsrmvEx");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseAlgMode_t, cusparseOperation_t, int, int, int,
                                          const void *, cudaDataType, const cusparseMatDescr_t, const void *,
                                          cudaDataType, const int *, const int *, const void *, cudaDataType,
                                          const void *, cudaDataType, void *, cudaDataType, cudaDataType, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCsrmvEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA,
                      csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, buffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                            cusparseOperation_t transA, int mb, int nb, int nnzb,
                                                            const float *alpha, const cusparseMatDescr_t descrA,
                                                            const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                            const int *bsrSortedColIndA, int blockDim, const float *x,
                                                            const float *beta, float *y) {
    HOOK_TRACE_PROFILE("cusparseSbsrmv");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
                                          const float *, const cusparseMatDescr_t, const float *, const int *,
                                          const int *, int, const float *, const float *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrmv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                      bsrSortedColIndA, blockDim, x, beta, y);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                            cusparseOperation_t transA, int mb, int nb, int nnzb,
                                                            const double *alpha, const cusparseMatDescr_t descrA,
                                                            const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                            const int *bsrSortedColIndA, int blockDim, const double *x,
                                                            const double *beta, double *y) {
    HOOK_TRACE_PROFILE("cusparseDbsrmv");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
                                          const double *, const cusparseMatDescr_t, const double *, const int *,
                                          const int *, int, const double *, const double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrmv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                      bsrSortedColIndA, blockDim, x, beta, y);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                            cusparseOperation_t transA, int mb, int nb, int nnzb,
                                                            const cuComplex *alpha, const cusparseMatDescr_t descrA,
                                                            const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                            const int *bsrSortedColIndA, int blockDim,
                                                            const cuComplex *x, const cuComplex *beta, cuComplex *y) {
    HOOK_TRACE_PROFILE("cusparseCbsrmv");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
                                          const cuComplex *, const cusparseMatDescr_t, const cuComplex *, const int *,
                                          const int *, int, const cuComplex *, const cuComplex *, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrmv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                      bsrSortedColIndA, blockDim, x, beta, y);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseZbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb,
                   int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
                   const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                   int blockDim, const cuDoubleComplex *x, const cuDoubleComplex *beta, cuDoubleComplex *y) {
    HOOK_TRACE_PROFILE("cusparseZbsrmv");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
                             const cuDoubleComplex *, const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
                             const int *, int, const cuDoubleComplex *, const cuDoubleComplex *, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrmv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                      bsrSortedColIndA, blockDim, x, beta, y);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrxmv(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
    int nnzb, const float *alpha, const cusparseMatDescr_t descrA, const float *bsrSortedValA,
    const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA,
    int blockDim, const float *x, const float *beta, float *y) {
    HOOK_TRACE_PROFILE("cusparseSbsrxmv");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int, int,
                             const float *, const cusparseMatDescr_t, const float *, const int *, const int *,
                             const int *, const int *, int, const float *, const float *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrxmv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA,
                      bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrxmv(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
    int nnzb, const double *alpha, const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA,
    int blockDim, const double *x, const double *beta, double *y) {
    HOOK_TRACE_PROFILE("cusparseDbsrxmv");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int, int,
                             const double *, const cusparseMatDescr_t, const double *, const int *, const int *,
                             const int *, const int *, int, const double *, const double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrxmv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA,
                      bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrxmv(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
    int nnzb, const cuComplex *alpha, const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA,
    int blockDim, const cuComplex *x, const cuComplex *beta, cuComplex *y) {
    HOOK_TRACE_PROFILE("cusparseCbsrxmv");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int, int,
                             const cuComplex *, const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
                             const int *, const int *, int, const cuComplex *, const cuComplex *, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrxmv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA,
                      bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrxmv(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb,
    int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA,
    int blockDim, const cuDoubleComplex *x, const cuDoubleComplex *beta, cuDoubleComplex *y) {
    HOOK_TRACE_PROFILE("cusparseZbsrxmv");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, int,
                                          int, const cuDoubleComplex *, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, const int *, const int *,
                                          int, const cuDoubleComplex *, const cuDoubleComplex *, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrxmv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA,
                      bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle, csrsv2Info_t info,
                                                                       int *position) {
    HOOK_TRACE_PROFILE("cusparseXcsrsv2_zeroPivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, csrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsrsv2_zeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseScsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                               const cusparseMatDescr_t descrA, float *csrSortedValA, const int *csrSortedRowPtrA,
                               const int *csrSortedColIndA, csrsv2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseScsrsv2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          float *, const int *, const int *, csrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrsv2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                               const cusparseMatDescr_t descrA, double *csrSortedValA, const int *csrSortedRowPtrA,
                               const int *csrSortedColIndA, csrsv2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDcsrsv2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          double *, const int *, const int *, csrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrsv2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                               const cusparseMatDescr_t descrA, cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                               const int *csrSortedColIndA, csrsv2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCcsrsv2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          cuComplex *, const int *, const int *, csrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrsv2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZcsrsv2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, csrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrsv2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseScsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                                  const cusparseMatDescr_t descrA, float *csrSortedValA, const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA, csrsv2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseScsrsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          float *, const int *, const int *, csrsv2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz,
                                  const cusparseMatDescr_t descrA, double *csrSortedValA, const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA, csrsv2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDcsrsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          double *, const int *, const int *, csrsv2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA,
    cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCcsrsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          cuComplex *, const int *, const int *, csrsv2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZcsrsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, csrsv2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsrsv2_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t, const float *,
                             const int *, const int *, csrsv2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrsv2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsrsv2_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t, const double *,
                             const int *, const int *, csrsv2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrsv2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsrsv2_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                             const cuComplex *, const int *, const int *, csrsv2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrsv2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrsv2_analysis(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrsv2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsrsv2_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, csrsv2Info_t,
                                          cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrsv2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseScsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const float *alpha,
                          const cusparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA,
                          const int *csrSortedColIndA, csrsv2Info_t info, const float *f, float *x,
                          cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsrsv2_solve");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const float *,
                                          const cusparseMatDescr_t, const float *, const int *, const int *,
                                          csrsv2Info_t, const float *, float *, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrsv2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f,
                      x, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const double *alpha,
                          const cusparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA,
                          const int *csrSortedColIndA, csrsv2Info_t info, const double *f, double *x,
                          cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsrsv2_solve");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const double *,
                                          const cusparseMatDescr_t, const double *, const int *, const int *,
                                          csrsv2Info_t, const double *, double *, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrsv2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f,
                      x, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cuComplex *alpha,
                          const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA,
                          const int *csrSortedColIndA, csrsv2Info_t info, const cuComplex *f, cuComplex *x,
                          cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsrsv2_solve");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cuComplex *,
                                          const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
                                          csrsv2Info_t, const cuComplex *, cuComplex *, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrsv2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f,
                      x, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrsv2_solve(
    cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, csrsv2Info_t info, const cuDoubleComplex *f, cuDoubleComplex *x,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsrsv2_solve");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, int, int, const cuDoubleComplex *,
                             const cusparseMatDescr_t, const cuDoubleComplex *, const int *, const int *, csrsv2Info_t,
                             const cuDoubleComplex *, cuDoubleComplex *, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrsv2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f,
                      x, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle, bsrsv2Info_t info,
                                                                       int *position) {
    HOOK_TRACE_PROFILE("cusparseXbsrsv2_zeroPivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, bsrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXbsrsv2_zeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
    int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSbsrsv2_bufferSize");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                             const cusparseMatDescr_t, float *, const int *, const int *, int, bsrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrsv2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
    int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDbsrsv2_bufferSize");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                             const cusparseMatDescr_t, double *, const int *, const int *, int, bsrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrsv2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
    int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCbsrsv2_bufferSize");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                             const cusparseMatDescr_t, cuComplex *, const int *, const int *, int, bsrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrsv2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrsv2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZbsrsv2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                                          const cusparseMatDescr_t, cuDoubleComplex *, const int *, const int *, int,
                                          bsrsv2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrsv2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
    int blockSize, bsrsv2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseSbsrsv2_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                             const cusparseMatDescr_t, float *, const int *, const int *, int, bsrsv2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockSize, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
    int blockSize, bsrsv2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDbsrsv2_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                             const cusparseMatDescr_t, double *, const int *, const int *, int, bsrsv2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockSize, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
    int blockSize, bsrsv2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCbsrsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                                          const cusparseMatDescr_t, cuComplex *, const int *, const int *, int,
                                          bsrsv2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockSize, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrsv2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZbsrsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                                          const cusparseMatDescr_t, cuDoubleComplex *, const int *, const int *, int,
                                          bsrsv2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockSize, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSbsrsv2_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                                          const cusparseMatDescr_t, const float *, const int *, const int *, int,
                                          bsrsv2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrsv2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDbsrsv2_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                                          const cusparseMatDescr_t, const double *, const int *, const int *, int,
                                          bsrsv2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrsv2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCbsrsv2_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                                          const cusparseMatDescr_t, const cuComplex *, const int *, const int *, int,
                                          bsrsv2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrsv2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrsv2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
    const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZbsrsv2_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                                          const cusparseMatDescr_t, const cuDoubleComplex *, const int *, const int *,
                                          int, bsrsv2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrsv2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseSbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb,
                          int nnzb, const float *alpha, const cusparseMatDescr_t descrA, const float *bsrSortedValA,
                          const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info,
                          const float *f, float *x, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSbsrsv2_solve");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, const float *,
                             const cusparseMatDescr_t, const float *, const int *, const int *, int, bsrsv2Info_t,
                             const float *, float *, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrsv2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, f, x, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb,
                          int nnzb, const double *alpha, const cusparseMatDescr_t descrA, const double *bsrSortedValA,
                          const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info,
                          const double *f, double *x, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDbsrsv2_solve");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, const double *,
                             const cusparseMatDescr_t, const double *, const int *, const int *, int, bsrsv2Info_t,
                             const double *, double *, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrsv2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, f, x, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cuComplex *alpha, const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const cuComplex *f,
    cuComplex *x, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCbsrsv2_solve");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int, const cuComplex *,
                             const cusparseMatDescr_t, const cuComplex *, const int *, const int *, int, bsrsv2Info_t,
                             const cuComplex *, cuComplex *, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrsv2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, f, x, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrsv2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const cuDoubleComplex *f,
    cuDoubleComplex *x, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZbsrsv2_solve");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, int, int,
                                          const cuDoubleComplex *, const cusparseMatDescr_t, const cuDoubleComplex *,
                                          const int *, const int *, int, bsrsv2Info_t, const cuDoubleComplex *,
                                          cuDoubleComplex *, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrsv2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      blockDim, info, f, x, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                            cusparseOperation_t transA, cusparseOperation_t transB,
                                                            int mb, int n, int kb, int nnzb, const float *alpha,
                                                            const cusparseMatDescr_t descrA, const float *bsrSortedValA,
                                                            const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                                                            const int blockSize, const float *B, const int ldb,
                                                            const float *beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cusparseSbsrmm");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int,
                             int, int, const float *, const cusparseMatDescr_t, const float *, const int *, const int *,
                             const int, const float *, const int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                      bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrmm(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb,
    int n, int kb, int nnzb, const double *alpha, const cusparseMatDescr_t descrA, const double *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize, const double *B, const int ldb,
    const double *beta, double *C, int ldc) {
    HOOK_TRACE_PROFILE("cusparseDbsrmm");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int,
                             int, int, const double *, const cusparseMatDescr_t, const double *, const int *,
                             const int *, const int, const double *, const int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                      bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrmm(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb,
    int n, int kb, int nnzb, const cuComplex *alpha, const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA,
    const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize, const cuComplex *B, const int ldb,
    const cuComplex *beta, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cusparseCbsrmm");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int,
                             int, int, const cuComplex *, const cusparseMatDescr_t, const cuComplex *, const int *,
                             const int *, const int, const cuComplex *, const int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                      bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrmm(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb,
    int n, int kb, int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize,
    const cuDoubleComplex *B, const int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cusparseZbsrmm");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int, int, int,
        const cuDoubleComplex *, const cusparseMatDescr_t, const cuDoubleComplex *, const int *, const int *, const int,
        const cuDoubleComplex *, const int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA,
                      bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
                                                            const float *alpha, const float *A, int lda,
                                                            const float *cscValB, const int *cscColPtrB,
                                                            const int *cscRowIndB, const float *beta, float *C,
                                                            int ldc) {
    HOOK_TRACE_PROFILE("cusparseSgemmi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, int, const float *, const float *, int,
                                          const float *, const int *, const int *, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgemmi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
                                                            const double *alpha, const double *A, int lda,
                                                            const double *cscValB, const int *cscColPtrB,
                                                            const int *cscRowIndB, const double *beta, double *C,
                                                            int ldc) {
    HOOK_TRACE_PROFILE("cusparseDgemmi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, int, const double *, const double *, int,
                                          const double *, const int *, const int *, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgemmi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
                                                            const cuComplex *alpha, const cuComplex *A, int lda,
                                                            const cuComplex *cscValB, const int *cscColPtrB,
                                                            const int *cscRowIndB, const cuComplex *beta, cuComplex *C,
                                                            int ldc) {
    HOOK_TRACE_PROFILE("cusparseCgemmi");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, int, const cuComplex *, const cuComplex *, int,
                             const cuComplex *, const int *, const int *, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgemmi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz,
                                                            const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                            int lda, const cuDoubleComplex *cscValB,
                                                            const int *cscColPtrB, const int *cscRowIndB,
                                                            const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cusparseZgemmi");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, int, const cuDoubleComplex *,
                                          const cuDoubleComplex *, int, const cuDoubleComplex *, const int *,
                                          const int *, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgemmi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCsrsm2Info(csrsm2Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateCsrsm2Info");
    using func_ptr = cusparseStatus_t (*)(csrsm2Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCsrsm2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyCsrsm2Info(csrsm2Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyCsrsm2Info");
    using func_ptr = cusparseStatus_t (*)(csrsm2Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyCsrsm2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle, csrsm2Info_t info,
                                                                       int *position) {
    HOOK_TRACE_PROFILE("cusparseXcsrsm2_zeroPivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, csrsm2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsrsm2_zeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const float *alpha, const cusparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const float *B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
    size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseScsrsm2_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const float *, const cusparseMatDescr_t, const float *, const int *, const int *,
                             const float *, int, csrsm2Info_t, cusparseSolvePolicy_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrsm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const double *alpha, const cusparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const double *B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy,
    size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDcsrsm2_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const double *, const cusparseMatDescr_t, const double *, const int *, const int *,
                             const double *, int, csrsm2Info_t, cusparseSolvePolicy_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrsm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCcsrsm2_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const cuComplex *, const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
                             const cuComplex *, int, csrsm2Info_t, cusparseSolvePolicy_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrsm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrsm2_bufferSizeExt(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuDoubleComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZcsrsm2_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const cuDoubleComplex *, const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
                             const int *, const cuDoubleComplex *, int, csrsm2Info_t, cusparseSolvePolicy_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrsm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseScsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB,
                             int m, int nrhs, int nnz, const float *alpha, const cusparseMatDescr_t descrA,
                             const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                             const float *B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsrsm2_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
                                          int, const float *, const cusparseMatDescr_t, const float *, const int *,
                                          const int *, const float *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrsm2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB,
                             int m, int nrhs, int nnz, const double *alpha, const cusparseMatDescr_t descrA,
                             const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                             const double *B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsrsm2_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const double *, const cusparseMatDescr_t, const double *, const int *, const int *,
                             const double *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrsm2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuComplex *alpha, const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsrsm2_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const cuComplex *, const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
                             const cuComplex *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrsm2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrsm2_analysis(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuDoubleComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsrsm2_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const cuDoubleComplex *, const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
                             const int *, const cuDoubleComplex *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrsm2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const float *alpha, const cusparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, float *B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsrsm2_solve");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
                                          int, const float *, const cusparseMatDescr_t, const float *, const int *,
                                          const int *, float *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrsm2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const double *alpha, const cusparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, double *B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsrsm2_solve");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int,
                                          int, const double *, const cusparseMatDescr_t, const double *, const int *,
                                          const int *, double *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrsm2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB,
                          int m, int nrhs, int nnz, const cuComplex *alpha, const cusparseMatDescr_t descrA,
                          const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                          cuComplex *B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsrsm2_solve");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const cuComplex *, const cusparseMatDescr_t, const cuComplex *, const int *, const int *,
                             cuComplex *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrsm2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrsm2_solve(
    cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz,
    const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, cuDoubleComplex *B, int ldb, csrsm2Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsrsm2_solve");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, cusparseOperation_t, cusparseOperation_t, int, int, int,
                             const cuDoubleComplex *, const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
                             const int *, cuDoubleComplex *, int, csrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrsm2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                      csrSortedColIndA, B, ldb, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle, bsrsm2Info_t info,
                                                                       int *position) {
    HOOK_TRACE_PROFILE("cusparseXbsrsm2_zeroPivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, bsrsm2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXbsrsm2_zeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSbsrsm2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t, float *,
                                          const int *, const int *, int, bsrsm2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrsm2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDbsrsm2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t, double *,
                                          const int *, const int *, int, bsrsm2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrsm2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCbsrsm2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, const int *, int, bsrsm2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrsm2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrsm2_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZbsrsm2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsrsm2Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrsm2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrsm2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseSbsrsm2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t, float *,
                                          const int *, const int *, int, bsrsm2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrsm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                      blockSize, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrsm2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDbsrsm2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t, double *,
                                          const int *, const int *, int, bsrsm2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrsm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                      blockSize, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrsm2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCbsrsm2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, const int *, int, bsrsm2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrsm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                      blockSize, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrsm2_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZbsrsm2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsrsm2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrsm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                      blockSize, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, const float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSbsrsm2_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, int, bsrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrsm2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, const double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDbsrsm2_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                                          cusparseOperation_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, int, bsrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrsm2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCbsrsm2_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int,
                             int, const cusparseMatDescr_t, const cuComplex *, const int *, const int *, int,
                             bsrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrsm2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrsm2_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZbsrsm2_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int,
                             int, const cusparseMatDescr_t, const cuDoubleComplex *, const int *, const int *, int,
                             bsrsm2Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrsm2_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const float *alpha, const cusparseMatDescr_t descrA, const float *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const float *B, int ldb,
    float *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSbsrsm2_solve");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int,
                             int, const float *, const cusparseMatDescr_t, const float *, const int *, const int *, int,
                             bsrsm2Info_t, const float *, int, float *, int, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrsm2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const double *alpha, const cusparseMatDescr_t descrA, const double *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const double *B, int ldb,
    double *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDbsrsm2_solve");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int,
                             int, const double *, const cusparseMatDescr_t, const double *, const int *, const int *,
                             int, bsrsm2Info_t, const double *, int, double *, int, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrsm2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cuComplex *alpha, const cusparseMatDescr_t descrA, const cuComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuComplex *B,
    int ldb, cuComplex *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCbsrsm2_solve");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int, int,
        const cuComplex *, const cusparseMatDescr_t, const cuComplex *, const int *, const int *, int, bsrsm2Info_t,
        const cuComplex *, int, cuComplex *, int, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrsm2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrsm2_solve(
    cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb,
    int n, int nnzb, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal,
    const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuDoubleComplex *B,
    int ldb, cuDoubleComplex *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZbsrsm2_solve");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, int, int, int,
        const cuDoubleComplex *, const cusparseMatDescr_t, const cuDoubleComplex *, const int *, const int *, int,
        bsrsm2Info_t, const cuDoubleComplex *, int, cuDoubleComplex *, int, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrsm2_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr,
                      bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                                                                            csrilu02Info_t info, int enable_boost,
                                                                            double *tol, float *boost_val) {
    HOOK_TRACE_PROFILE("cusparseScsrilu02_numericBoost");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, csrilu02Info_t, int, double *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrilu02_numericBoost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, enable_boost, tol, boost_val);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                                                                            csrilu02Info_t info, int enable_boost,
                                                                            double *tol, double *boost_val) {
    HOOK_TRACE_PROFILE("cusparseDcsrilu02_numericBoost");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, csrilu02Info_t, int, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrilu02_numericBoost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, enable_boost, tol, boost_val);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                                                                            csrilu02Info_t info, int enable_boost,
                                                                            double *tol, cuComplex *boost_val) {
    HOOK_TRACE_PROFILE("cusparseCcsrilu02_numericBoost");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, csrilu02Info_t, int, double *, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrilu02_numericBoost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, enable_boost, tol, boost_val);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                                                                            csrilu02Info_t info, int enable_boost,
                                                                            double *tol, cuDoubleComplex *boost_val) {
    HOOK_TRACE_PROFILE("cusparseZcsrilu02_numericBoost");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, csrilu02Info_t, int, double *, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrilu02_numericBoost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, enable_boost, tol, boost_val);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle, csrilu02Info_t info,
                                                                         int *position) {
    HOOK_TRACE_PROFILE("cusparseXcsrilu02_zeroPivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, csrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsrilu02_zeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseScsrilu02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, float *, const int *,
                                          const int *, csrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrilu02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDcsrilu02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, double *, const int *,
                                          const int *, csrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrilu02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCcsrilu02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, const int *, csrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrilu02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrilu02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZcsrilu02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
                                          const int *, const int *, csrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrilu02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrilu02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float *csrSortedVal,
    const int *csrSortedRowPtr, const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseScsrilu02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, float *, const int *,
                                          const int *, csrilu02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrilu02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrilu02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double *csrSortedVal,
    const int *csrSortedRowPtr, const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDcsrilu02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, double *, const int *,
                                          const int *, csrilu02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrilu02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrilu02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex *csrSortedVal,
    const int *csrSortedRowPtr, const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCcsrilu02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, const int *, csrilu02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrilu02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrilu02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedVal,
    const int *csrSortedRowPtr, const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZcsrilu02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
                                          const int *, const int *, csrilu02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrilu02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseScsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                               const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                               csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsrilu02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrilu02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                               const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                               csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsrilu02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrilu02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                               const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                               csrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsrilu02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrilu02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrilu02_analysis(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy,
    void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsrilu02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                                          const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrilu02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrilu02(cusparseHandle_t handle, int m, int nnz,
                                                               const cusparseMatDescr_t descrA,
                                                               float *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                                               const int *csrSortedColIndA, csrilu02Info_t info,
                                                               cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsrilu02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, float *, const int *,
                                          const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrilu02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrilu02(cusparseHandle_t handle, int m, int nnz,
                                                               const cusparseMatDescr_t descrA,
                                                               double *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                                               const int *csrSortedColIndA, csrilu02Info_t info,
                                                               cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsrilu02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, double *, const int *,
                                          const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrilu02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrilu02(cusparseHandle_t handle, int m, int nnz,
                                                               const cusparseMatDescr_t descrA,
                                                               cuComplex *csrSortedValA_valM,
                                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                               csrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                               void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsrilu02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrilu02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrilu02(cusparseHandle_t handle, int m, int nnz,
                                                               const cusparseMatDescr_t descrA,
                                                               cuDoubleComplex *csrSortedValA_valM,
                                                               const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                               csrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                               void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsrilu02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
                                          const int *, const int *, csrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrilu02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrilu02_numericBoost(cusparseHandle_t handle,
                                                                            bsrilu02Info_t info, int enable_boost,
                                                                            double *tol, float *boost_val) {
    HOOK_TRACE_PROFILE("cusparseSbsrilu02_numericBoost");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, bsrilu02Info_t, int, double *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrilu02_numericBoost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, enable_boost, tol, boost_val);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrilu02_numericBoost(cusparseHandle_t handle,
                                                                            bsrilu02Info_t info, int enable_boost,
                                                                            double *tol, double *boost_val) {
    HOOK_TRACE_PROFILE("cusparseDbsrilu02_numericBoost");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, bsrilu02Info_t, int, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrilu02_numericBoost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, enable_boost, tol, boost_val);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrilu02_numericBoost(cusparseHandle_t handle,
                                                                            bsrilu02Info_t info, int enable_boost,
                                                                            double *tol, cuComplex *boost_val) {
    HOOK_TRACE_PROFILE("cusparseCbsrilu02_numericBoost");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, bsrilu02Info_t, int, double *, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrilu02_numericBoost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, enable_boost, tol, boost_val);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrilu02_numericBoost(cusparseHandle_t handle,
                                                                            bsrilu02Info_t info, int enable_boost,
                                                                            double *tol, cuDoubleComplex *boost_val) {
    HOOK_TRACE_PROFILE("cusparseZbsrilu02_numericBoost");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, bsrilu02Info_t, int, double *, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrilu02_numericBoost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, enable_boost, tol, boost_val);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle, bsrilu02Info_t info,
                                                                         int *position) {
    HOOK_TRACE_PROFILE("cusparseXbsrilu02_zeroPivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, bsrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXbsrilu02_zeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
    int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSbsrilu02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          float *, const int *, const int *, int, bsrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrilu02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
    int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDbsrilu02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          double *, const int *, const int *, int, bsrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrilu02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
    int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCbsrilu02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuComplex *, const int *, const int *, int, bsrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrilu02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrilu02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZbsrilu02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsrilu02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrilu02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrilu02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrilu02Info_t info,
    size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseSbsrilu02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          float *, const int *, const int *, int, bsrilu02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrilu02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrilu02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrilu02Info_t info,
    size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDbsrilu02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          double *, const int *, const int *, int, bsrilu02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrilu02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrilu02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrilu02Info_t info,
    size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCbsrilu02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuComplex *, const int *, const int *, int, bsrilu02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrilu02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrilu02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsrilu02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZbsrilu02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsrilu02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrilu02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSbsrilu02_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, float *,
                             const int *, const int *, int, bsrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrilu02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDbsrilu02_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, double *,
                             const int *, const int *, int, bsrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrilu02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info,
    cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCbsrilu02_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, cuComplex *,
                             const int *, const int *, int, bsrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrilu02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrilu02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsrilu02Info_t info, cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZbsrilu02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsrilu02Info_t,
                                          cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrilu02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                               int mb, int nnzb, const cusparseMatDescr_t descrA,
                                                               float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                               const int *bsrSortedColInd, int blockDim,
                                                               bsrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                               void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSbsrilu02");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, float *,
                             const int *, const int *, int, bsrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsrilu02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                               int mb, int nnzb, const cusparseMatDescr_t descrA,
                                                               double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                               const int *bsrSortedColInd, int blockDim,
                                                               bsrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                               void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDbsrilu02");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, double *,
                             const int *, const int *, int, bsrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsrilu02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                               int mb, int nnzb, const cusparseMatDescr_t descrA,
                                                               cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                               const int *bsrSortedColInd, int blockDim,
                                                               bsrilu02Info_t info, cusparseSolvePolicy_t policy,
                                                               void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCbsrilu02");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, cuComplex *,
                             const int *, const int *, int, bsrilu02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsrilu02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                               int mb, int nnzb, const cusparseMatDescr_t descrA,
                                                               cuDoubleComplex *bsrSortedVal,
                                                               const int *bsrSortedRowPtr, const int *bsrSortedColInd,
                                                               int blockDim, bsrilu02Info_t info,
                                                               cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZbsrilu02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsrilu02Info_t,
                                          cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsrilu02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsric02_zeroPivot(cusparseHandle_t handle, csric02Info_t info,
                                                                        int *position) {
    HOOK_TRACE_PROFILE("cusparseXcsric02_zeroPivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, csric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsric02_zeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseScsric02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, float *, const int *,
                                          const int *, csric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsric02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDcsric02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, double *, const int *,
                                          const int *, csric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsric02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCcsric02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, const int *, csric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsric02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsric02_bufferSize(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZcsric02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
                                          const int *, const int *, csric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsric02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsric02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float *csrSortedVal,
    const int *csrSortedRowPtr, const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseScsric02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, float *, const int *,
                                          const int *, csric02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsric02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsric02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double *csrSortedVal,
    const int *csrSortedRowPtr, const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDcsric02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, double *, const int *,
                                          const int *, csric02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsric02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsric02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex *csrSortedVal,
    const int *csrSortedRowPtr, const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCcsric02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, const int *, csric02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsric02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsric02_bufferSizeExt(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex *csrSortedVal,
    const int *csrSortedRowPtr, const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZcsric02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
                                          const int *, const int *, csric02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsric02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, info, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const float *csrSortedValA,
                                                                       const int *csrSortedRowPtrA,
                                                                       const int *csrSortedColIndA, csric02Info_t info,
                                                                       cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsric02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsric02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const double *csrSortedValA,
                                                                       const int *csrSortedRowPtrA,
                                                                       const int *csrSortedColIndA, csric02Info_t info,
                                                                       cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsric02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsric02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const cuComplex *csrSortedValA,
                                                                       const int *csrSortedRowPtrA,
                                                                       const int *csrSortedColIndA, csric02Info_t info,
                                                                       cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsric02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsric02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const cuDoubleComplex *csrSortedValA,
                                                                       const int *csrSortedRowPtrA,
                                                                       const int *csrSortedColIndA, csric02Info_t info,
                                                                       cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsric02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                                          const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsric02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsric02(cusparseHandle_t handle, int m, int nnz,
                                                              const cusparseMatDescr_t descrA,
                                                              float *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA, csric02Info_t info,
                                                              cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsric02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, float *, const int *,
                                          const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsric02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsric02(cusparseHandle_t handle, int m, int nnz,
                                                              const cusparseMatDescr_t descrA,
                                                              double *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA, csric02Info_t info,
                                                              cusparseSolvePolicy_t policy, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsric02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, double *, const int *,
                                          const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsric02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsric02(cusparseHandle_t handle, int m, int nnz,
                                                              const cusparseMatDescr_t descrA,
                                                              cuComplex *csrSortedValA_valM,
                                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                              csric02Info_t info, cusparseSolvePolicy_t policy,
                                                              void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsric02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsric02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsric02(cusparseHandle_t handle, int m, int nnz,
                                                              const cusparseMatDescr_t descrA,
                                                              cuDoubleComplex *csrSortedValA_valM,
                                                              const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                                                              csric02Info_t info, cusparseSolvePolicy_t policy,
                                                              void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsric02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
                                          const int *, const int *, csric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsric02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXbsric02_zeroPivot(cusparseHandle_t handle, bsric02Info_t info,
                                                                        int *position) {
    HOOK_TRACE_PROFILE("cusparseXbsric02_zeroPivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, bsric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXbsric02_zeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseSbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                const cusparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd, int blockDim, bsric02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSbsric02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          float *, const int *, const int *, int, bsric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsric02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                const cusparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd, int blockDim, bsric02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDbsric02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          double *, const int *, const int *, int, bsric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsric02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd, int blockDim, bsric02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCbsric02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuComplex *, const int *, const int *, int, bsric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsric02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsric02_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZbsric02_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsric02Info_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsric02_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseSbsric02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                   const cusparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd, int blockSize, bsric02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseSbsric02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          float *, const int *, const int *, int, bsric02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsric02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDbsric02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                   const cusparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd, int blockSize, bsric02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDbsric02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          double *, const int *, const int *, int, bsric02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsric02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCbsric02_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,
                                   const cusparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd, int blockSize, bsric02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCbsric02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuComplex *, const int *, const int *, int, bsric02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsric02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsric02_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize,
    bsric02Info_t info, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZbsric02_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsric02Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsric02_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    const float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info,
    cusparseSolvePolicy_t policy, void *pInputBuffer) {
    HOOK_TRACE_PROFILE("cusparseSbsric02_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float *,
                             const int *, const int *, int, bsric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsric02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pInputBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    const double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer) {
    HOOK_TRACE_PROFILE("cusparseDbsric02_analysis");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double *,
                             const int *, const int *, int, bsric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsric02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pInputBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer) {
    HOOK_TRACE_PROFILE("cusparseCbsric02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, int, bsric02Info_t,
                                          cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsric02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pInputBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsric02_analysis(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim,
    bsric02Info_t info, cusparseSolvePolicy_t policy, void *pInputBuffer) {
    HOOK_TRACE_PROFILE("cusparseZbsric02_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, int, bsric02Info_t,
                                          cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsric02_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pInputBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                              int nnzb, const cusparseMatDescr_t descrA,
                                                              float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                              const int *bsrSortedColInd, int blockDim,
                                                              bsric02Info_t info, cusparseSolvePolicy_t policy,
                                                              void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSbsric02");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, float *,
                             const int *, const int *, int, bsric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsric02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                              int nnzb, const cusparseMatDescr_t descrA,
                                                              double *bsrSortedVal, const int *bsrSortedRowPtr,
                                                              const int *bsrSortedColInd, int blockDim,
                                                              bsric02Info_t info, cusparseSolvePolicy_t policy,
                                                              void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDbsric02");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, double *,
                             const int *, const int *, int, bsric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsric02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                              int nnzb, const cusparseMatDescr_t descrA,
                                                              cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                              const int *bsrSortedColInd, int blockDim,
                                                              bsric02Info_t info, cusparseSolvePolicy_t policy,
                                                              void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCbsric02");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, cuComplex *,
                             const int *, const int *, int, bsric02Info_t, cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsric02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                              int nnzb, const cusparseMatDescr_t descrA,
                                                              cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
                                                              const int *bsrSortedColInd, int blockDim,
                                                              bsric02Info_t info, cusparseSolvePolicy_t policy,
                                                              void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZbsric02");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          cuDoubleComplex *, const int *, const int *, int, bsric02Info_t,
                                          cusparseSolvePolicy_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsric02"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                      policy, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                          const float *dl, const float *d,
                                                                          const float *du, const float *B, int ldb,
                                                                          size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSgtsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, const float *, const float *,
                                          const float *, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgtsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                          const double *dl, const double *d,
                                                                          const double *du, const double *B, int ldb,
                                                                          size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDgtsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, const double *, const double *,
                                          const double *, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgtsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                          const cuComplex *dl, const cuComplex *d,
                                                                          const cuComplex *du, const cuComplex *B,
                                                                          int ldb, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCgtsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
                                          const cuComplex *, const cuComplex *, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgtsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgtsv2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl, const cuDoubleComplex *d,
    const cuDoubleComplex *du, const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZgtsv2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, const cuDoubleComplex *, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgtsv2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgtsv2(cusparseHandle_t handle, int m, int n, const float *dl,
                                                            const float *d, const float *du, float *B, int ldb,
                                                            void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSgtsv2");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, const float *, const float *,
                                          float *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgtsv2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgtsv2(cusparseHandle_t handle, int m, int n, const double *dl,
                                                            const double *d, const double *du, double *B, int ldb,
                                                            void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDgtsv2");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, const double *, const double *,
                                          double *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgtsv2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgtsv2(cusparseHandle_t handle, int m, int n, const cuComplex *dl,
                                                            const cuComplex *d, const cuComplex *du, cuComplex *B,
                                                            int ldb, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCgtsv2");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
                                          const cuComplex *, cuComplex *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgtsv2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgtsv2(cusparseHandle_t handle, int m, int n,
                                                            const cuDoubleComplex *dl, const cuDoubleComplex *d,
                                                            const cuDoubleComplex *du, cuDoubleComplex *B, int ldb,
                                                            void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZgtsv2");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, cuDoubleComplex *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgtsv2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                                  const float *dl, const float *d,
                                                                                  const float *du, const float *B,
                                                                                  int ldb, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSgtsv2_nopivot_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, const float *, const float *,
                                          const float *, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgtsv2_nopivot_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                                  const double *dl, const double *d,
                                                                                  const double *du, const double *B,
                                                                                  int ldb, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDgtsv2_nopivot_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, const double *, const double *,
                                          const double *, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgtsv2_nopivot_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex *dl, const cuComplex *d,
                                         const cuComplex *du, const cuComplex *B, int ldb, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCgtsv2_nopivot_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
                                          const cuComplex *, const cuComplex *, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgtsv2_nopivot_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgtsv2_nopivot_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *dl, const cuDoubleComplex *d,
    const cuDoubleComplex *du, const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZgtsv2_nopivot_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, const cuDoubleComplex *, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgtsv2_nopivot_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgtsv2_nopivot(cusparseHandle_t handle, int m, int n,
                                                                    const float *dl, const float *d, const float *du,
                                                                    float *B, int ldb, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSgtsv2_nopivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, const float *, const float *,
                                          float *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgtsv2_nopivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgtsv2_nopivot(cusparseHandle_t handle, int m, int n,
                                                                    const double *dl, const double *d, const double *du,
                                                                    double *B, int ldb, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDgtsv2_nopivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, const double *, const double *,
                                          double *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgtsv2_nopivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgtsv2_nopivot(cusparseHandle_t handle, int m, int n,
                                                                    const cuComplex *dl, const cuComplex *d,
                                                                    const cuComplex *du, cuComplex *B, int ldb,
                                                                    void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCgtsv2_nopivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
                                          const cuComplex *, cuComplex *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgtsv2_nopivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgtsv2_nopivot(cusparseHandle_t handle, int m, int n,
                                                                    const cuDoubleComplex *dl, const cuDoubleComplex *d,
                                                                    const cuDoubleComplex *du, cuDoubleComplex *B,
                                                                    int ldb, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZgtsv2_nopivot");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, cuDoubleComplex *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgtsv2_nopivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, dl, d, du, B, ldb, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m,
                                                                                      const float *dl, const float *d,
                                                                                      const float *du, const float *x,
                                                                                      int batchCount, int batchStride,
                                                                                      size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSgtsv2StridedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const float *, const float *, const float *,
                                          const float *, int, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgtsv2StridedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m,
                                                                                      const double *dl, const double *d,
                                                                                      const double *du, const double *x,
                                                                                      int batchCount, int batchStride,
                                                                                      size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDgtsv2StridedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const double *, const double *, const double *,
                                          const double *, int, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgtsv2StridedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const cuComplex *dl, const cuComplex *d, const cuComplex *du, const cuComplex *x,
    int batchCount, int batchStride, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCgtsv2StridedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cuComplex *, const cuComplex *,
                                          const cuComplex *, const cuComplex *, int, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgtsv2StridedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgtsv2StridedBatch_bufferSizeExt(
    cusparseHandle_t handle, int m, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du,
    const cuDoubleComplex *x, int batchCount, int batchStride, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZgtsv2StridedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, const cuDoubleComplex *, int, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgtsv2StridedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgtsv2StridedBatch(cusparseHandle_t handle, int m, const float *dl,
                                                                        const float *d, const float *du, float *x,
                                                                        int batchCount, int batchStride,
                                                                        void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSgtsv2StridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const float *, const float *, const float *, float *,
                                          int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgtsv2StridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m,
                                                                        const double *dl, const double *d,
                                                                        const double *du, double *x, int batchCount,
                                                                        int batchStride, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDgtsv2StridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const double *, const double *, const double *,
                                          double *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgtsv2StridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgtsv2StridedBatch(cusparseHandle_t handle, int m,
                                                                        const cuComplex *dl, const cuComplex *d,
                                                                        const cuComplex *du, cuComplex *x,
                                                                        int batchCount, int batchStride,
                                                                        void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCgtsv2StridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cuComplex *, const cuComplex *,
                                          const cuComplex *, cuComplex *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgtsv2StridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgtsv2StridedBatch(
    cusparseHandle_t handle, int m, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du,
    cuDoubleComplex *x, int batchCount, int batchStride, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZgtsv2StridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, cuDoubleComplex *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgtsv2StridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const float *dl, const float *d, const float *du, const float *x,
    int batchCount, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSgtsvInterleavedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, const float *, const float *,
                                          const float *, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgtsvInterleavedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const double *dl, const double *d, const double *du, const double *x,
    int batchCount, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDgtsvInterleavedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, const double *, const double *,
                                          const double *, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgtsvInterleavedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuComplex *dl, const cuComplex *d, const cuComplex *du,
    const cuComplex *x, int batchCount, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCgtsvInterleavedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex *, const cuComplex *,
                                          const cuComplex *, const cuComplex *, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgtsvInterleavedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgtsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *dl, const cuDoubleComplex *d,
    const cuDoubleComplex *du, const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZgtsvInterleavedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, const cuDoubleComplex *, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgtsvInterleavedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                                           float *dl, float *d, float *du, float *x,
                                                                           int batchCount, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSgtsvInterleavedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, float *, float *, float *, float *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgtsvInterleavedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                                           double *dl, double *d, double *du, double *x,
                                                                           int batchCount, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDgtsvInterleavedBatch");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, double *, double *, double *, double *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgtsvInterleavedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                                           cuComplex *dl, cuComplex *d, cuComplex *du,
                                                                           cuComplex *x, int batchCount,
                                                                           void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCgtsvInterleavedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, cuComplex *, cuComplex *, cuComplex *,
                                          cuComplex *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgtsvInterleavedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                                           cuDoubleComplex *dl, cuDoubleComplex *d,
                                                                           cuDoubleComplex *du, cuDoubleComplex *x,
                                                                           int batchCount, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZgtsvInterleavedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, cuDoubleComplex *, cuDoubleComplex *,
                                          cuDoubleComplex *, cuDoubleComplex *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgtsvInterleavedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, dl, d, du, x, batchCount, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const float *ds, const float *dl, const float *d, const float *du,
    const float *dw, const float *x, int batchCount, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSgpsvInterleavedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, const float *, const float *,
                                          const float *, const float *, const float *, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgpsvInterleavedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const double *ds, const double *dl, const double *d, const double *du,
    const double *dw, const double *x, int batchCount, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDgpsvInterleavedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, const double *, const double *,
                                          const double *, const double *, const double *, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgpsvInterleavedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuComplex *ds, const cuComplex *dl, const cuComplex *d,
    const cuComplex *du, const cuComplex *dw, const cuComplex *x, int batchCount, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCgpsvInterleavedBatch_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex *, const cuComplex *, const cuComplex *,
                             const cuComplex *, const cuComplex *, const cuComplex *, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgpsvInterleavedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgpsvInterleavedBatch_bufferSizeExt(
    cusparseHandle_t handle, int algo, int m, const cuDoubleComplex *ds, const cuDoubleComplex *dl,
    const cuDoubleComplex *d, const cuDoubleComplex *du, const cuDoubleComplex *dw, const cuDoubleComplex *x,
    int batchCount, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZgpsvInterleavedBatch_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, const cuDoubleComplex *, const cuDoubleComplex *,
                                          const cuDoubleComplex *, int, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgpsvInterleavedBatch_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                                           float *ds, float *dl, float *d, float *du,
                                                                           float *dw, float *x, int batchCount,
                                                                           void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSgpsvInterleavedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, float *, float *, float *, float *, float *,
                                          float *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgpsvInterleavedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                                           double *ds, double *dl, double *d,
                                                                           double *du, double *dw, double *x,
                                                                           int batchCount, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDgpsvInterleavedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, double *, double *, double *, double *, double *,
                                          double *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgpsvInterleavedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                                           cuComplex *ds, cuComplex *dl, cuComplex *d,
                                                                           cuComplex *du, cuComplex *dw, cuComplex *x,
                                                                           int batchCount, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCgpsvInterleavedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, cuComplex *, cuComplex *, cuComplex *,
                                          cuComplex *, cuComplex *, cuComplex *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgpsvInterleavedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m,
                                                                           cuDoubleComplex *ds, cuDoubleComplex *dl,
                                                                           cuDoubleComplex *d, cuDoubleComplex *du,
                                                                           cuDoubleComplex *dw, cuDoubleComplex *x,
                                                                           int batchCount, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZgpsvInterleavedBatch");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *,
                             cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgpsvInterleavedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCsrgemm2Info(csrgemm2Info_t *info) {
    HOOK_TRACE_PROFILE("cusparseCreateCsrgemm2Info");
    using func_ptr = cusparseStatus_t (*)(csrgemm2Info_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCsrgemm2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyCsrgemm2Info(csrgemm2Info_t info) {
    HOOK_TRACE_PROFILE("cusparseDestroyCsrgemm2Info");
    using func_ptr = cusparseStatus_t (*)(csrgemm2Info_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyCsrgemm2Info"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const float *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const float *beta, const cusparseMatDescr_t descrD,
    int nnzD, const int *csrSortedRowPtrD, const int *csrSortedColIndD, csrgemm2Info_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseScsrgemm2_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const float *, const cusparseMatDescr_t, int, const int *,
                             const int *, const cusparseMatDescr_t, int, const int *, const int *, const float *,
                             const cusparseMatDescr_t, int, const int *, const int *, csrgemm2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrgemm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                      csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const double *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const double *beta, const cusparseMatDescr_t descrD,
    int nnzD, const int *csrSortedRowPtrD, const int *csrSortedColIndD, csrgemm2Info_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDcsrgemm2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const double *, const cusparseMatDescr_t,
                                          int, const int *, const int *, const cusparseMatDescr_t, int, const int *,
                                          const int *, const double *, const cusparseMatDescr_t, int, const int *,
                                          const int *, csrgemm2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrgemm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                      csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const cuComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cuComplex *beta, const cusparseMatDescr_t descrD,
    int nnzD, const int *csrSortedRowPtrD, const int *csrSortedColIndD, csrgemm2Info_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCcsrgemm2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuComplex *, const cusparseMatDescr_t,
                                          int, const int *, const int *, const cusparseMatDescr_t, int, const int *,
                                          const int *, const cuComplex *, const cusparseMatDescr_t, int, const int *,
                                          const int *, csrgemm2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrgemm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                      csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrgemm2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cuDoubleComplex *beta,
    const cusparseMatDescr_t descrD, int nnzD, const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    csrgemm2Info_t info, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZcsrgemm2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, int, int, int, const cuDoubleComplex *, const cusparseMatDescr_t, int, const int *,
        const int *, const cusparseMatDescr_t, int, const int *, const int *, const cuDoubleComplex *,
        const cusparseMatDescr_t, int, const int *, const int *, csrgemm2Info_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrgemm2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                      csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsrgemm2Nnz(
    cusparseHandle_t handle, int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrD, int nnzD,
    const int *csrSortedRowPtrD, const int *csrSortedColIndD, const cusparseMatDescr_t descrC, int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr, const csrgemm2Info_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseXcsrgemm2Nnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, int, const int *,
                                          const int *, const cusparseMatDescr_t, int, const int *, const int *,
                                          const cusparseMatDescr_t, int, const int *, const int *,
                                          const cusparseMatDescr_t, int *, int *, const csrgemm2Info_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsrgemm2Nnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                      csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedRowPtrC,
                      nnzTotalDevHostPtr, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const float *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const float *beta, const cusparseMatDescr_t descrD, int nnzD,
    const float *csrSortedValD, const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    const cusparseMatDescr_t descrC, float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsrgemm2");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, int, int, int, const float *, const cusparseMatDescr_t, int, const float *, const int *,
        const int *, const cusparseMatDescr_t, int, const float *, const int *, const int *, const float *,
        const cusparseMatDescr_t, int, const float *, const int *, const int *, const cusparseMatDescr_t, float *,
        const int *, int *, const csrgemm2Info_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrgemm2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB,
                      nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD,
                      csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const double *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const double *beta, const cusparseMatDescr_t descrD, int nnzD,
    const double *csrSortedValD, const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    const cusparseMatDescr_t descrC, double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsrgemm2");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, int, int, int, const double *, const cusparseMatDescr_t, int, const double *, const int *,
        const int *, const cusparseMatDescr_t, int, const double *, const int *, const int *, const double *,
        const cusparseMatDescr_t, int, const double *, const int *, const int *, const cusparseMatDescr_t, double *,
        const int *, int *, const csrgemm2Info_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrgemm2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB,
                      nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD,
                      csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const cuComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cuComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const cuComplex *csrSortedValD, const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    const cusparseMatDescr_t descrC, cuComplex *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsrgemm2");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, int, int, int, const cuComplex *, const cusparseMatDescr_t, int, const cuComplex *,
        const int *, const int *, const cusparseMatDescr_t, int, const cuComplex *, const int *, const int *,
        const cuComplex *, const cusparseMatDescr_t, int, const cuComplex *, const int *, const int *,
        const cusparseMatDescr_t, cuComplex *, const int *, int *, const csrgemm2Info_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrgemm2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB,
                      nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD,
                      csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrgemm2(
    cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
    int nnzA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cuDoubleComplex *beta, const cusparseMatDescr_t descrD, int nnzD,
    const cuDoubleComplex *csrSortedValD, const int *csrSortedRowPtrD, const int *csrSortedColIndD,
    const cusparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC,
    const csrgemm2Info_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsrgemm2");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, int, int, int, const cuDoubleComplex *, const cusparseMatDescr_t, int,
        const cuDoubleComplex *, const int *, const int *, const cusparseMatDescr_t, int, const cuDoubleComplex *,
        const int *, const int *, const cuDoubleComplex *, const cusparseMatDescr_t, int, const cuDoubleComplex *,
        const int *, const int *, const cusparseMatDescr_t, cuDoubleComplex *, const int *, int *, const csrgemm2Info_t,
        void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrgemm2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB,
                      nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD,
                      csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta,
    const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC, const float *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseScsrgeam2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, const cusparseMatDescr_t, int,
                                          const float *, const int *, const int *, const float *,
                                          const cusparseMatDescr_t, int, const float *, const int *, const int *,
                                          const cusparseMatDescr_t, const float *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrgeam2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                      descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                      csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *beta,
    const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC, const double *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDcsrgeam2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, const cusparseMatDescr_t, int,
                                          const double *, const int *, const int *, const double *,
                                          const cusparseMatDescr_t, int, const double *, const int *, const int *,
                                          const cusparseMatDescr_t, const double *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrgeam2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                      descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                      csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuComplex *beta,
    const cusparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC, const cuComplex *csrSortedValC,
    const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCcsrgeam2_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, int, int, const cuComplex *, const cusparseMatDescr_t, int, const cuComplex *, const int *,
        const int *, const cuComplex *, const cusparseMatDescr_t, int, const cuComplex *, const int *, const int *,
        const cusparseMatDescr_t, const cuComplex *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrgeam2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                      descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                      csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrgeam2_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cuDoubleComplex *beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    const cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZcsrgeam2_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuDoubleComplex *, const cusparseMatDescr_t, int,
                             const cuDoubleComplex *, const int *, const int *, const cuDoubleComplex *,
                             const cusparseMatDescr_t, int, const cuDoubleComplex *, const int *, const int *,
                             const cusparseMatDescr_t, const cuDoubleComplex *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrgeam2_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                      descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                      csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsrgeam2Nnz(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA,
    const int *csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr,
    void *workspace) {
    HOOK_TRACE_PROFILE("cusparseXcsrgeam2Nnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, int, const int *,
                                          const int *, const cusparseMatDescr_t, int, const int *, const int *,
                                          const cusparseMatDescr_t, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsrgeam2Nnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                      csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseScsrgeam2(cusparseHandle_t handle, int m, int n, const float *alpha, const cusparseMatDescr_t descrA,
                      int nnzA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                      const float *beta, const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB,
                      const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
                      float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsrgeam2");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, const cusparseMatDescr_t, int, const float *,
                             const int *, const int *, const float *, const cusparseMatDescr_t, int, const float *,
                             const int *, const int *, const cusparseMatDescr_t, float *, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrgeam2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                      descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                      csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double *alpha, const cusparseMatDescr_t descrA,
                      int nnzA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                      const double *beta, const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB,
                      const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
                      double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsrgeam2");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, const cusparseMatDescr_t, int, const double *,
                             const int *, const int *, const double *, const cusparseMatDescr_t, int, const double *,
                             const int *, const int *, const cusparseMatDescr_t, double *, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrgeam2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                      descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                      csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrgeam2(
    cusparseHandle_t handle, int m, int n, const cuComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuComplex *beta,
    const cusparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB, const int *csrSortedRowPtrB,
    const int *csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex *csrSortedValC, int *csrSortedRowPtrC,
    int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsrgeam2");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cuComplex *, const cusparseMatDescr_t, int,
                                          const cuComplex *, const int *, const int *, const cuComplex *,
                                          const cusparseMatDescr_t, int, const cuComplex *, const int *, const int *,
                                          const cusparseMatDescr_t, cuComplex *, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrgeam2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                      descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                      csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrgeam2(
    cusparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA, int nnzA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
    const cuDoubleComplex *beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex *csrSortedValB,
    const int *csrSortedRowPtrB, const int *csrSortedColIndB, const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsrgeam2");
    using func_ptr = cusparseStatus_t (*)(
        cusparseHandle_t, int, int, const cuDoubleComplex *, const cusparseMatDescr_t, int, const cuDoubleComplex *,
        const int *, const int *, const cuDoubleComplex *, const cusparseMatDescr_t, int, const cuDoubleComplex *,
        const int *, const int *, const cusparseMatDescr_t, cuDoubleComplex *, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrgeam2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta,
                      descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                      csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *fractionToColor, int *ncolors, int *coloring,
    int *reordering, const cusparseColorInfo_t info) {
    HOOK_TRACE_PROFILE("cusparseScsrcolor");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *, const int *,
                             const int *, const float *, int *, int *, int *, const cusparseColorInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsrcolor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor,
                      ncolors, coloring, reordering, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const cusparseColorInfo_t info) {
    HOOK_TRACE_PROFILE("cusparseDcsrcolor");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *, const int *,
                             const int *, const double *, int *, int *, int *, const cusparseColorInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsrcolor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor,
                      ncolors, coloring, reordering, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *fractionToColor, int *ncolors, int *coloring,
    int *reordering, const cusparseColorInfo_t info) {
    HOOK_TRACE_PROFILE("cusparseCcsrcolor");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *, const int *,
                             const int *, const float *, int *, int *, int *, const cusparseColorInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsrcolor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor,
                      ncolors, coloring, reordering, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsrcolor(
    cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *fractionToColor, int *ncolors,
    int *coloring, int *reordering, const cusparseColorInfo_t info) {
    HOOK_TRACE_PROFILE("cusparseZcsrcolor");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
                             const int *, const double *, int *, int *, int *, const cusparseColorInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsrcolor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor,
                      ncolors, coloring, reordering, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                          int n, const cusparseMatDescr_t descrA, const float *A,
                                                          int lda, int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
    HOOK_TRACE_PROFILE("cusparseSnnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const float *, int, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSnnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                          int n, const cusparseMatDescr_t descrA, const double *A,
                                                          int lda, int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
    HOOK_TRACE_PROFILE("cusparseDnnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const double *, int, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                          int n, const cusparseMatDescr_t descrA, const cuComplex *A,
                                                          int lda, int *nnzPerRowCol, int *nnzTotalDevHostPtr) {
    HOOK_TRACE_PROFILE("cusparseCnnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, int, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCnnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                          int n, const cusparseMatDescr_t descrA,
                                                          const cuDoubleComplex *A, int lda, int *nnzPerRowCol,
                                                          int *nnzTotalDevHostPtr) {
    HOOK_TRACE_PROFILE("cusparseZnnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, int, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZnnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSnnz_compress(cusparseHandle_t handle, int m,
                                                                   const cusparseMatDescr_t descr,
                                                                   const float *csrSortedValA,
                                                                   const int *csrSortedRowPtrA, int *nnzPerRow,
                                                                   int *nnzC, float tol) {
    HOOK_TRACE_PROFILE("cusparseSnnz_compress");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cusparseMatDescr_t, const float *, const int *,
                                          int *, int *, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSnnz_compress"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnnz_compress(cusparseHandle_t handle, int m,
                                                                   const cusparseMatDescr_t descr,
                                                                   const double *csrSortedValA,
                                                                   const int *csrSortedRowPtrA, int *nnzPerRow,
                                                                   int *nnzC, double tol) {
    HOOK_TRACE_PROFILE("cusparseDnnz_compress");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cusparseMatDescr_t, const double *, const int *,
                                          int *, int *, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnnz_compress"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCnnz_compress(cusparseHandle_t handle, int m,
                                                                   const cusparseMatDescr_t descr,
                                                                   const cuComplex *csrSortedValA,
                                                                   const int *csrSortedRowPtrA, int *nnzPerRow,
                                                                   int *nnzC, cuComplex tol) {
    HOOK_TRACE_PROFILE("cusparseCnnz_compress");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, int *, int *, cuComplex);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCnnz_compress"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZnnz_compress(cusparseHandle_t handle, int m,
                                                                   const cusparseMatDescr_t descr,
                                                                   const cuDoubleComplex *csrSortedValA,
                                                                   const int *csrSortedRowPtrA, int *nnzPerRow,
                                                                   int *nnzC, cuDoubleComplex tol) {
    HOOK_TRACE_PROFILE("cusparseZnnz_compress");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                                          const int *, int *, int *, cuDoubleComplex);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZnnz_compress"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow, float *csrSortedValC,
    int *csrSortedColIndC, int *csrSortedRowPtrC, float tol) {
    HOOK_TRACE_PROFILE("cusparseScsr2csr_compress");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, int, const int *, float *, int *, int *, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsr2csr_compress"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow,
                      csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow, double *csrSortedValC,
    int *csrSortedColIndC, int *csrSortedRowPtrC, double tol) {
    HOOK_TRACE_PROFILE("cusparseDcsr2csr_compress");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, int, const int *, double *, int *, int *, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsr2csr_compress"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow,
                      csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex *csrSortedValA,
    const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow, cuComplex *csrSortedValC,
    int *csrSortedColIndC, int *csrSortedRowPtrC, cuComplex tol) {
    HOOK_TRACE_PROFILE("cusparseCcsr2csr_compress");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *, const int *,
                             const int *, int, const int *, cuComplex *, int *, int *, cuComplex);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsr2csr_compress"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow,
                      csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsr2csr_compress(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow,
    cuDoubleComplex *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC, cuDoubleComplex tol) {
    HOOK_TRACE_PROFILE("cusparseZcsr2csr_compress");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *, const int *,
                             const int *, int, const int *, cuDoubleComplex *, int *, int *, cuDoubleComplex);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsr2csr_compress"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow,
                      csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSdense2csr(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA, const float *A,
                                                                int lda, const int *nnzPerRow, float *csrSortedValA,
                                                                int *csrSortedRowPtrA, int *csrSortedColIndA) {
    HOOK_TRACE_PROFILE("cusparseSdense2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *, int,
                                          const int *, float *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSdense2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDdense2csr(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA, const double *A,
                                                                int lda, const int *nnzPerRow, double *csrSortedValA,
                                                                int *csrSortedRowPtrA, int *csrSortedColIndA) {
    HOOK_TRACE_PROFILE("cusparseDdense2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *, int,
                                          const int *, double *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDdense2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCdense2csr(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA, const cuComplex *A,
                                                                int lda, const int *nnzPerRow, cuComplex *csrSortedValA,
                                                                int *csrSortedRowPtrA, int *csrSortedColIndA) {
    HOOK_TRACE_PROFILE("cusparseCdense2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *, int,
                                          const int *, cuComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCdense2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZdense2csr(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA,
                                                                const cuDoubleComplex *A, int lda, const int *nnzPerRow,
                                                                cuDoubleComplex *csrSortedValA, int *csrSortedRowPtrA,
                                                                int *csrSortedColIndA) {
    HOOK_TRACE_PROFILE("cusparseZdense2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                                          int, const int *, cuDoubleComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZdense2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsr2dense(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA,
                                                                const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                                const int *csrSortedColIndA, float *A, int lda) {
    HOOK_TRACE_PROFILE("cusparseScsr2dense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsr2dense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsr2dense(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA,
                                                                const double *csrSortedValA,
                                                                const int *csrSortedRowPtrA,
                                                                const int *csrSortedColIndA, double *A, int lda) {
    HOOK_TRACE_PROFILE("cusparseDcsr2dense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsr2dense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsr2dense(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA,
                                                                const cuComplex *csrSortedValA,
                                                                const int *csrSortedRowPtrA,
                                                                const int *csrSortedColIndA, cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cusparseCcsr2dense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsr2dense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsr2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cusparseZcsr2dense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                                          const int *, const int *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsr2dense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSdense2csc(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA, const float *A,
                                                                int lda, const int *nnzPerCol, float *cscSortedValA,
                                                                int *cscSortedRowIndA, int *cscSortedColPtrA) {
    HOOK_TRACE_PROFILE("cusparseSdense2csc");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *, int,
                                          const int *, float *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSdense2csc"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDdense2csc(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA, const double *A,
                                                                int lda, const int *nnzPerCol, double *cscSortedValA,
                                                                int *cscSortedRowIndA, int *cscSortedColPtrA) {
    HOOK_TRACE_PROFILE("cusparseDdense2csc");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *, int,
                                          const int *, double *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDdense2csc"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCdense2csc(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA, const cuComplex *A,
                                                                int lda, const int *nnzPerCol, cuComplex *cscSortedValA,
                                                                int *cscSortedRowIndA, int *cscSortedColPtrA) {
    HOOK_TRACE_PROFILE("cusparseCdense2csc");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *, int,
                                          const int *, cuComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCdense2csc"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZdense2csc(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA,
                                                                const cuDoubleComplex *A, int lda, const int *nnzPerCol,
                                                                cuDoubleComplex *cscSortedValA, int *cscSortedRowIndA,
                                                                int *cscSortedColPtrA) {
    HOOK_TRACE_PROFILE("cusparseZdense2csc");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                                          int, const int *, cuDoubleComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZdense2csc"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsc2dense(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA,
                                                                const float *cscSortedValA, const int *cscSortedRowIndA,
                                                                const int *cscSortedColPtrA, float *A, int lda) {
    HOOK_TRACE_PROFILE("cusparseScsc2dense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsc2dense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsc2dense(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA,
                                                                const double *cscSortedValA,
                                                                const int *cscSortedRowIndA,
                                                                const int *cscSortedColPtrA, double *A, int lda) {
    HOOK_TRACE_PROFILE("cusparseDcsc2dense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsc2dense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsc2dense(cusparseHandle_t handle, int m, int n,
                                                                const cusparseMatDescr_t descrA,
                                                                const cuComplex *cscSortedValA,
                                                                const int *cscSortedRowIndA,
                                                                const int *cscSortedColPtrA, cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cusparseCcsc2dense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsc2dense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsc2dense(
    cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex *cscSortedValA,
    const int *cscSortedRowIndA, const int *cscSortedColPtrA, cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cusparseZcsc2dense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                                          const int *, const int *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsc2dense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int *cooRowInd, int nnz,
                                                              int m, int *csrSortedRowPtr,
                                                              cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseXcoo2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, const int *, int, int, int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcoo2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t handle, const int *csrSortedRowPtr,
                                                              int nnz, int m, int *cooRowInd,
                                                              cusparseIndexBase_t idxBase) {
    HOOK_TRACE_PROFILE("cusparseXcsr2coo");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, const int *, int, int, int *, cusparseIndexBase_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsr2coo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsr2bsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                                 int m, int n, const cusparseMatDescr_t descrA,
                                                                 const int *csrSortedRowPtrA,
                                                                 const int *csrSortedColIndA, int blockDim,
                                                                 const cusparseMatDescr_t descrC, int *bsrSortedRowPtrC,
                                                                 int *nnzTotalDevHostPtr) {
    HOOK_TRACE_PROFILE("cusparseXcsr2bsrNnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const int *, const int *, int, const cusparseMatDescr_t, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsr2bsrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC,
                      bsrSortedRowPtrC, nnzTotalDevHostPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                              int n, const cusparseMatDescr_t descrA,
                                                              const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA, int blockDim,
                                                              const cusparseMatDescr_t descrC, float *bsrSortedValC,
                                                              int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseScsr2bsr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float *,
                             const int *, const int *, int, const cusparseMatDescr_t, float *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsr2bsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC,
                      bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                                              int n, const cusparseMatDescr_t descrA,
                                                              const double *csrSortedValA, const int *csrSortedRowPtrA,
                                                              const int *csrSortedColIndA, int blockDim,
                                                              const cusparseMatDescr_t descrC, double *bsrSortedValC,
                                                              int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseDcsr2bsr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double *,
                             const int *, const int *, int, const cusparseMatDescr_t, double *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsr2bsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC,
                      bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsr2bsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseCcsr2bsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, int, const cusparseMatDescr_t,
                                          cuComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsr2bsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC,
                      bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsr2bsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseZcsr2bsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, int,
                                          const cusparseMatDescr_t, cuDoubleComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsr2bsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC,
                      bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                              int nb, const cusparseMatDescr_t descrA,
                                                              const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                              const int *bsrSortedColIndA, int blockDim,
                                                              const cusparseMatDescr_t descrC, float *csrSortedValC,
                                                              int *csrSortedRowPtrC, int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseSbsr2csr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float *,
                             const int *, const int *, int, const cusparseMatDescr_t, float *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSbsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb,
                                                              int nb, const cusparseMatDescr_t descrA,
                                                              const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                              const int *bsrSortedColIndA, int blockDim,
                                                              const cusparseMatDescr_t descrC, double *csrSortedValC,
                                                              int *csrSortedRowPtrC, int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseDbsr2csr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double *,
                             const int *, const int *, int, const cusparseMatDescr_t, double *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDbsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCbsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseCbsr2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, int, const cusparseMatDescr_t,
                                          cuComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCbsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZbsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim,
    const cusparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseZbsr2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, int,
                                          const cusparseMatDescr_t, cuDoubleComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZbsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgebsr2gebsc_bufferSize(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSgebsr2gebsc_bufferSize");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const float *, const int *, const int *, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgebsr2gebsc_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgebsr2gebsc_bufferSize(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDgebsr2gebsc_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const double *, const int *, const int *,
                                          int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgebsr2gebsc_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgebsr2gebsc_bufferSize(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCgebsr2gebsc_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuComplex *, const int *, const int *,
                                          int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgebsr2gebsc_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgebsr2gebsc_bufferSize(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZgebsr2gebsc_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuDoubleComplex *, const int *,
                                          const int *, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgebsr2gebsc_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgebsr2gebsc_bufferSizeExt(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseSgebsr2gebsc_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const float *, const int *, const int *, int,
                                          int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgebsr2gebsc_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgebsr2gebsc_bufferSizeExt(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDgebsr2gebsc_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const double *, const int *, const int *,
                                          int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgebsr2gebsc_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgebsr2gebsc_bufferSizeExt(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCgebsr2gebsc_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuComplex *, const int *, const int *,
                                          int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgebsr2gebsc_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgebsr2gebsc_bufferSizeExt(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZgebsr2gebsc_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuDoubleComplex *, const int *,
                                          const int *, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgebsr2gebsc_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb,
                                                                  const float *bsrSortedVal, const int *bsrSortedRowPtr,
                                                                  const int *bsrSortedColInd, int rowBlockDim,
                                                                  int colBlockDim, float *bscVal, int *bscRowInd,
                                                                  int *bscColPtr, cusparseAction_t copyValues,
                                                                  cusparseIndexBase_t idxBase, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSgebsr2gebsc");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const float *, const int *, const int *, int,
                                          int, float *, int *, int *, cusparseAction_t, cusparseIndexBase_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgebsr2gebsc"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgebsr2gebsc(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, double *bscVal, int *bscRowInd, int *bscColPtr,
    cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDgebsr2gebsc");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const double *, const int *, const int *, int, int,
                             double *, int *, int *, cusparseAction_t, cusparseIndexBase_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgebsr2gebsc"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgebsr2gebsc(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, cuComplex *bscVal, int *bscRowInd, int *bscColPtr,
    cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCgebsr2gebsc");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuComplex *, const int *, const int *, int, int,
                             cuComplex *, int *, int *, cusparseAction_t, cusparseIndexBase_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgebsr2gebsc"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgebsr2gebsc(
    cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr,
    const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, cuDoubleComplex *bscVal, int *bscRowInd,
    int *bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZgebsr2gebsc");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cuDoubleComplex *, const int *, const int *, int,
                             int, cuDoubleComplex *, int *, int *, cusparseAction_t, cusparseIndexBase_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgebsr2gebsc"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim,
                      bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                                int mb, int nb, const cusparseMatDescr_t descrA,
                                                                const int *bsrSortedRowPtrA,
                                                                const int *bsrSortedColIndA, int rowBlockDim,
                                                                int colBlockDim, const cusparseMatDescr_t descrC,
                                                                int *csrSortedRowPtrC, int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseXgebsr2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const int *, const int *, int, int, const cusparseMatDescr_t, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXgebsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim,
                      descrC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                                int mb, int nb, const cusparseMatDescr_t descrA,
                                                                const float *bsrSortedValA, const int *bsrSortedRowPtrA,
                                                                const int *bsrSortedColIndA, int rowBlockDim,
                                                                int colBlockDim, const cusparseMatDescr_t descrC,
                                                                float *csrSortedValC, int *csrSortedRowPtrC,
                                                                int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseSgebsr2csr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const float *,
                             const int *, const int *, int, int, const cusparseMatDescr_t, float *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgebsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim,
                      colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                       const cusparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA,
                       const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC,
                       double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseDgebsr2csr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const double *,
                             const int *, const int *, int, int, const cusparseMatDescr_t, double *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgebsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim,
                      colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb,
                       const cusparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA,
                       const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC,
                       cuComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseCgebsr2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, int, int,
                                          const cusparseMatDescr_t, cuComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgebsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim,
                      colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgebsr2csr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim,
    int colBlockDim, const cusparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC,
    int *csrSortedColIndC) {
    HOOK_TRACE_PROFILE("cusparseZgebsr2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, int, int,
                                          const cusparseMatDescr_t, cuDoubleComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgebsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim,
                      colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseScsr2gebsr_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const float *, const int *, const int *, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsr2gebsr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                      colBlockDim, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDcsr2gebsr_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const double *, const int *, const int *, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsr2gebsr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                      colBlockDim, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCcsr2gebsr_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsr2gebsr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                      colBlockDim, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZcsr2gebsr_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsr2gebsr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                      colBlockDim, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseScsr2gebsr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const float *, const int *, const int *, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsr2gebsr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                      colBlockDim, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDcsr2gebsr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const double *, const int *, const int *, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsr2gebsr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                      colBlockDim, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCcsr2gebsr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsr2gebsr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                      colBlockDim, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim,
    int colBlockDim, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZcsr2gebsr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsr2gebsr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                      colBlockDim, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsr2gebsrNnz(
    cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cusparseMatDescr_t descrC, int *bsrSortedRowPtrC,
    int rowBlockDim, int colBlockDim, int *nnzTotalDevHostPtr, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseXcsr2gebsrNnz");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t, const int *,
                             const int *, const cusparseMatDescr_t, int *, int, int, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsr2gebsrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedRowPtrC,
                      rowBlockDim, colBlockDim, nnzTotalDevHostPtr, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA,
                                                                int m, int n, const cusparseMatDescr_t descrA,
                                                                const float *csrSortedValA, const int *csrSortedRowPtrA,
                                                                const int *csrSortedColIndA,
                                                                const cusparseMatDescr_t descrC, float *bsrSortedValC,
                                                                int *bsrSortedRowPtrC, int *bsrSortedColIndC,
                                                                int rowBlockDim, int colBlockDim, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsr2gebsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const float *, const int *, const int *, const cusparseMatDescr_t, float *,
                                          int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsr2gebsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC,
                      bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseDcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
                       const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                       const cusparseMatDescr_t descrC, double *bsrSortedValC, int *bsrSortedRowPtrC,
                       int *bsrSortedColIndC, int rowBlockDim, int colBlockDim, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsr2gebsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const double *, const int *, const int *, const cusparseMatDescr_t, double *,
                                          int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsr2gebsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC,
                      bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseCcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
                       const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                       const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC, int *bsrSortedRowPtrC,
                       int *bsrSortedColIndC, int rowBlockDim, int colBlockDim, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsr2gebsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, const cusparseMatDescr_t,
                                          cuComplex *, int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsr2gebsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC,
                      bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseZcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA,
                       const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
                       const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC,
                       int *bsrSortedColIndC, int rowBlockDim, int colBlockDim, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsr2gebsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, const cusparseMatDescr_t,
                                          cuDoubleComplex *, int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsr2gebsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC,
                      bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSgebsr2gebsr_bufferSize");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const float *, const int *, const int *, int, int, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgebsr2gebsr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDgebsr2gebsr_bufferSize");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const double *, const int *, const int *, int, int, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgebsr2gebsr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCgebsr2gebsr_bufferSize");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const cuComplex *, const int *, const int *, int, int, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgebsr2gebsr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgebsr2gebsr_bufferSize(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, int *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZgebsr2gebsr_bufferSize");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const cuDoubleComplex *, const int *, const int *, int, int, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgebsr2gebsr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseSgebsr2gebsr_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const float *, const int *, const int *, int, int, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgebsr2gebsr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseDgebsr2gebsr_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const double *, const int *, const int *, int, int, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgebsr2gebsr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseCgebsr2gebsr_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const cuComplex *, const int *, const int *, int, int, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgebsr2gebsr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgebsr2gebsr_bufferSizeExt(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize) {
    HOOK_TRACE_PROFILE("cusparseZgebsr2gebsr_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const cuDoubleComplex *, const int *, const int *, int, int, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgebsr2gebsr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseXgebsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb,
                            const cusparseMatDescr_t descrA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA,
                            int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, int *bsrSortedRowPtrC,
                            int rowBlockDimC, int colBlockDimC, int *nnzTotalDevHostPtr, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseXgebsr2gebsrNnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int,
                                          const cusparseMatDescr_t, const int *, const int *, int, int,
                                          const cusparseMatDescr_t, int *, int, int, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXgebsr2gebsrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                      colBlockDimA, descrC, bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, nnzTotalDevHostPtr, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, float *bsrSortedValC, int *bsrSortedRowPtrC,
    int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSgebsr2gebsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int,
                                          const cusparseMatDescr_t, const float *, const int *, const int *, int, int,
                                          const cusparseMatDescr_t, float *, int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSgebsr2gebsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC,
                      rowBlockDimC, colBlockDimC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, double *bsrSortedValC, int *bsrSortedRowPtrC,
    int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDgebsr2gebsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int,
                                          const cusparseMatDescr_t, const double *, const int *, const int *, int, int,
                                          const cusparseMatDescr_t, double *, int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDgebsr2gebsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC,
                      rowBlockDimC, colBlockDimC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, cuComplex *bsrSortedValC, int *bsrSortedRowPtrC,
    int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCgebsr2gebsr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int,
                                          const cusparseMatDescr_t, const cuComplex *, const int *, const int *, int,
                                          int, const cusparseMatDescr_t, cuComplex *, int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCgebsr2gebsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC,
                      rowBlockDimC, colBlockDimC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZgebsr2gebsr(
    cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA,
    const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA,
    int colBlockDimA, const cusparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC,
    int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZgebsr2gebsr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseDirection_t, int, int, int, const cusparseMatDescr_t,
                             const cuDoubleComplex *, const int *, const int *, int, int, const cusparseMatDescr_t,
                             cuDoubleComplex *, int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZgebsr2gebsr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                      rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC,
                      rowBlockDimC, colBlockDimC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int *p) {
    HOOK_TRACE_PROFILE("cusparseCreateIdentityPermutation");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateIdentityPermutation"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, p);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                            int nnz, const int *cooRowsA,
                                                                            const int *cooColsA,
                                                                            size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseXcoosort_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcoosort_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz,
                                                                   int *cooRowsA, int *cooColsA, int *P,
                                                                   void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseXcoosortByRow");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, int *, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcoosortByRow"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz,
                                                                      int *cooRowsA, int *cooColsA, int *P,
                                                                      void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseXcoosortByColumn");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, int *, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcoosortByColumn"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                            int nnz, const int *csrRowPtrA,
                                                                            const int *csrColIndA,
                                                                            size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseXcsrsort_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsrsort_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz,
                                                              const cusparseMatDescr_t descrA, const int *csrRowPtrA,
                                                              int *csrColIndA, int *P, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseXcsrsort");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const int *, int *,
                                          int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcsrsort"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                            int nnz, const int *cscColPtrA,
                                                                            const int *cscRowIndA,
                                                                            size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseXcscsort_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcscsort_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseXcscsort(cusparseHandle_t handle, int m, int n, int nnz,
                                                              const cusparseMatDescr_t descrA, const int *cscColPtrA,
                                                              int *cscRowIndA, int *P, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseXcscsort");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const int *, int *,
                                          int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseXcscsort"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                             int nnz, float *csrVal,
                                                                             const int *csrRowPtr, int *csrColInd,
                                                                             csru2csrInfo_t info,
                                                                             size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseScsru2csr_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, float *, const int *, int *, csru2csrInfo_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsru2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                             int nnz, double *csrVal,
                                                                             const int *csrRowPtr, int *csrColInd,
                                                                             csru2csrInfo_t info,
                                                                             size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDcsru2csr_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, double *, const int *, int *, csru2csrInfo_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsru2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                             int nnz, cuComplex *csrVal,
                                                                             const int *csrRowPtr, int *csrColInd,
                                                                             csru2csrInfo_t info,
                                                                             size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseCcsru2csr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, cuComplex *, const int *, int *,
                                          csru2csrInfo_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsru2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n,
                                                                             int nnz, cuDoubleComplex *csrVal,
                                                                             const int *csrRowPtr, int *csrColInd,
                                                                             csru2csrInfo_t info,
                                                                             size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseZcsru2csr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, cuDoubleComplex *, const int *, int *,
                                          csru2csrInfo_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsru2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsru2csr(cusparseHandle_t handle, int m, int n, int nnz,
                                                               const cusparseMatDescr_t descrA, float *csrVal,
                                                               const int *csrRowPtr, int *csrColInd,
                                                               csru2csrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsru2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, float *,
                                          const int *, int *, csru2csrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsru2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsru2csr(cusparseHandle_t handle, int m, int n, int nnz,
                                                               const cusparseMatDescr_t descrA, double *csrVal,
                                                               const int *csrRowPtr, int *csrColInd,
                                                               csru2csrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsru2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, double *,
                                          const int *, int *, csru2csrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsru2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsru2csr(cusparseHandle_t handle, int m, int n, int nnz,
                                                               const cusparseMatDescr_t descrA, cuComplex *csrVal,
                                                               const int *csrRowPtr, int *csrColInd,
                                                               csru2csrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsru2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, int *, csru2csrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsru2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsru2csr(cusparseHandle_t handle, int m, int n, int nnz,
                                                               const cusparseMatDescr_t descrA, cuDoubleComplex *csrVal,
                                                               const int *csrRowPtr, int *csrColInd,
                                                               csru2csrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsru2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
                                          const int *, int *, csru2csrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsru2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScsr2csru(cusparseHandle_t handle, int m, int n, int nnz,
                                                               const cusparseMatDescr_t descrA, float *csrVal,
                                                               const int *csrRowPtr, int *csrColInd,
                                                               csru2csrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseScsr2csru");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, float *,
                                          const int *, int *, csru2csrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScsr2csru"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDcsr2csru(cusparseHandle_t handle, int m, int n, int nnz,
                                                               const cusparseMatDescr_t descrA, double *csrVal,
                                                               const int *csrRowPtr, int *csrColInd,
                                                               csru2csrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDcsr2csru");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, double *,
                                          const int *, int *, csru2csrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDcsr2csru"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCcsr2csru(cusparseHandle_t handle, int m, int n, int nnz,
                                                               const cusparseMatDescr_t descrA, cuComplex *csrVal,
                                                               const int *csrRowPtr, int *csrColInd,
                                                               csru2csrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseCcsr2csru");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, cuComplex *,
                                          const int *, int *, csru2csrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCcsr2csru"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseZcsr2csru(cusparseHandle_t handle, int m, int n, int nnz,
                                                               const cusparseMatDescr_t descrA, cuDoubleComplex *csrVal,
                                                               const int *csrRowPtr, int *csrColInd,
                                                               csru2csrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseZcsr2csru");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, cuDoubleComplex *,
                                          const int *, int *, csru2csrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseZcsr2csru"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda, const __half *threshold,
    const cusparseMatDescr_t descrC, const __half *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseHpruneDense2csr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const __half *, int, const __half *,
                                          const cusparseMatDescr_t, const __half *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneDense2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *A, int lda, const float *threshold,
    const cusparseMatDescr_t descrC, const float *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSpruneDense2csr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, int, const float *,
                                          const cusparseMatDescr_t, const float *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneDense2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneDense2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *A, int lda, const double *threshold,
    const cusparseMatDescr_t descrC, const double *csrSortedValC, const int *csrSortedRowPtrC,
    const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDpruneDense2csr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, int, const double *,
                                          const cusparseMatDescr_t, const double *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneDense2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneDense2csrNnz(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda, const __half *threshold,
    const cusparseMatDescr_t descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseHpruneDense2csrNnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const __half *, int, const __half *,
                                          const cusparseMatDescr_t, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneDense2csrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrRowPtrC, nnzTotalDevHostPtr, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneDense2csrNnz(cusparseHandle_t handle, int m, int n,
                                                                        const float *A, int lda, const float *threshold,
                                                                        const cusparseMatDescr_t descrC,
                                                                        int *csrRowPtrC, int *nnzTotalDevHostPtr,
                                                                        void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpruneDense2csrNnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, int, const float *,
                                          const cusparseMatDescr_t, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneDense2csrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrRowPtrC, nnzTotalDevHostPtr, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneDense2csrNnz(
    cusparseHandle_t handle, int m, int n, const double *A, int lda, const double *threshold,
    const cusparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDpruneDense2csrNnz");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, int, const double *,
                                          const cusparseMatDescr_t, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneDense2csrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneDense2csr(cusparseHandle_t handle, int m, int n,
                                                                     const __half *A, int lda, const __half *threshold,
                                                                     const cusparseMatDescr_t descrC,
                                                                     __half *csrSortedValC, const int *csrSortedRowPtrC,
                                                                     int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseHpruneDense2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const __half *, int, const __half *,
                                          const cusparseMatDescr_t, __half *, const int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneDense2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneDense2csr(cusparseHandle_t handle, int m, int n,
                                                                     const float *A, int lda, const float *threshold,
                                                                     const cusparseMatDescr_t descrC,
                                                                     float *csrSortedValC, const int *csrSortedRowPtrC,
                                                                     int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpruneDense2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, int, const float *,
                                          const cusparseMatDescr_t, float *, const int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneDense2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneDense2csr(cusparseHandle_t handle, int m, int n,
                                                                     const double *A, int lda, const double *threshold,
                                                                     const cusparseMatDescr_t descrC,
                                                                     double *csrSortedValC, const int *csrSortedRowPtrC,
                                                                     int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDpruneDense2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, int, const double *,
                                          const cusparseMatDescr_t, double *, const int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneDense2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const __half *threshold, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseHpruneCsr2csr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const __half *,
                                          const int *, const int *, const __half *, const cusparseMatDescr_t,
                                          const __half *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneCsr2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *threshold, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSpruneCsr2csr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, const float *, const cusparseMatDescr_t,
                                          const float *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneCsr2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneCsr2csr_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *threshold, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDpruneCsr2csr_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, const double *, const cusparseMatDescr_t,
                                          const double *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneCsr2csr_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneCsr2csrNnz(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const __half *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseHpruneCsr2csrNnz");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const __half *, const int *,
                             const int *, const __half *, const cusparseMatDescr_t, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneCsr2csrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneCsr2csrNnz(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpruneCsr2csrNnz");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *, const int *,
                             const int *, const float *, const cusparseMatDescr_t, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneCsr2csrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneCsr2csrNnz(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *threshold, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDpruneCsr2csrNnz");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *, const int *,
                             const int *, const double *, const cusparseMatDescr_t, int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneCsr2csrNnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneCsr2csr(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const __half *threshold, const cusparseMatDescr_t descrC,
    __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseHpruneCsr2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const __half *,
                                          const int *, const int *, const __half *, const cusparseMatDescr_t, __half *,
                                          const int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneCsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneCsr2csr(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *threshold, const cusparseMatDescr_t descrC,
    float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpruneCsr2csr");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *, const int *,
                             const int *, const float *, const cusparseMatDescr_t, float *, const int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneCsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneCsr2csr(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *threshold, const cusparseMatDescr_t descrC,
    double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDpruneCsr2csr");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, const double *, const cusparseMatDescr_t, double *,
                                          const int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneCsr2csr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseHpruneDense2csrByPercentage_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const __half *, int, float, const cusparseMatDescr_t,
                             const __half *, const int *, const int *, pruneInfo_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneDense2csrByPercentage_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const float *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSpruneDense2csrByPercentage_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, int, float, const cusparseMatDescr_t,
                             const float *, const int *, const int *, pruneInfo_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneDense2csrByPercentage_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, const double *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDpruneDense2csrByPercentage_bufferSizeExt");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, int, float, const cusparseMatDescr_t,
                             const double *, const int *, const int *, pruneInfo_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneDense2csrByPercentage_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info,
                      pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    int *csrRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseHpruneDense2csrNnzByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const __half *, int, float,
                                          const cusparseMatDescr_t, int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneDense2csrNnzByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, const float *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    int *csrRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpruneDense2csrNnzByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, int, float,
                                          const cusparseMatDescr_t, int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneDense2csrNnzByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, const double *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    int *csrRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDpruneDense2csrNnzByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, int, float,
                                          const cusparseMatDescr_t, int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneDense2csrNnzByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneDense2csrByPercentage(
    cusparseHandle_t handle, int m, int n, const __half *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseHpruneDense2csrByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const __half *, int, float,
                                          const cusparseMatDescr_t, __half *, const int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneDense2csrByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneDense2csrByPercentage(
    cusparseHandle_t handle, int m, int n, const float *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpruneDense2csrByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const float *, int, float,
                                          const cusparseMatDescr_t, float *, const int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneDense2csrByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneDense2csrByPercentage(
    cusparseHandle_t handle, int m, int n, const double *A, int lda, float percentage, const cusparseMatDescr_t descrC,
    double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDpruneDense2csrByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, const double *, int, float,
                                          const cusparseMatDescr_t, double *, const int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneDense2csrByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info,
                      pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseHpruneCsr2csrByPercentage_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const __half *,
                                          const int *, const int *, float, const cusparseMatDescr_t, const __half *,
                                          const int *, const int *, pruneInfo_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneCsr2csrByPercentage_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseSpruneCsr2csrByPercentage_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, float, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, pruneInfo_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneCsr2csrByPercentage_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info,
    size_t *pBufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusparseDpruneCsr2csrByPercentage_bufferSizeExt");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, float, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, pruneInfo_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneCsr2csrByPercentage_bufferSizeExt"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseHpruneCsr2csrNnzByPercentage");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const __half *, const int *,
                             const int *, float, const cusparseMatDescr_t, int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneCsr2csrNnzByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpruneCsr2csrNnzByPercentage");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *, const int *,
                             const int *, float, const cusparseMatDescr_t, int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneCsr2csrNnzByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDpruneCsr2csrNnzByPercentage");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *, const int *,
                             const int *, float, const cusparseMatDescr_t, int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneCsr2csrNnzByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseHpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const __half *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseHpruneCsr2csrByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const __half *,
                                          const int *, const int *, float, const cusparseMatDescr_t, __half *,
                                          const int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseHpruneCsr2csrByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpruneCsr2csrByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, float, const cusparseMatDescr_t, float *,
                                          const int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpruneCsr2csrByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDpruneCsr2csrByPercentage(
    cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double *csrSortedValA,
    const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC,
    double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusparseDpruneCsr2csrByPercentage");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, float, const cusparseMatDescr_t, double *,
                                          const int *, int *, pruneInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDpruneCsr2csrByPercentage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC,
                      csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCsr2cscEx2(
    cusparseHandle_t handle, int m, int n, int nnz, const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd, cudaDataType valType, cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void *buffer) {
    HOOK_TRACE_PROFILE("cusparseCsr2cscEx2");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const void *, const int *, const int *, void *, int *,
                             int *, cudaDataType, cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCsr2cscEx2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType,
                      copyValues, idxBase, alg, buffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCsr2cscEx2_bufferSize(
    cusparseHandle_t handle, int m, int n, int nnz, const void *csrVal, const int *csrRowPtr, const int *csrColInd,
    void *cscVal, int *cscColPtr, int *cscRowInd, cudaDataType valType, cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseCsr2cscEx2_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, int, int, int, const void *, const int *, const int *,
                                          void *, int *, int *, cudaDataType, cusparseAction_t, cusparseIndexBase_t,
                                          cusparseCsr2CscAlg_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCsr2cscEx2_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType,
                      copyValues, idxBase, alg, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t *spVecDescr, int64_t size,
                                                                 int64_t nnz, void *indices, void *values,
                                                                 cusparseIndexType_t idxType,
                                                                 cusparseIndexBase_t idxBase, cudaDataType valueType) {
    HOOK_TRACE_PROFILE("cusparseCreateSpVec");
    using func_ptr = cusparseStatus_t (*)(cusparseSpVecDescr_t *, int64_t, int64_t, void *, void *, cusparseIndexType_t,
                                          cusparseIndexBase_t, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateSpVec"));
    HOOK_CHECK(func_entry);
    return func_entry(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr) {
    HOOK_TRACE_PROFILE("cusparseDestroySpVec");
    using func_ptr = cusparseStatus_t (*)(cusparseSpVecDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroySpVec"));
    HOOK_CHECK(func_entry);
    return func_entry(spVecDescr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t *size,
                                                              int64_t *nnz, void **indices, void **values,
                                                              cusparseIndexType_t *idxType,
                                                              cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
    HOOK_TRACE_PROFILE("cusparseSpVecGet");
    using func_ptr = cusparseStatus_t (*)(cusparseSpVecDescr_t, int64_t *, int64_t *, void **, void **,
                                          cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpVecGet"));
    HOOK_CHECK(func_entry);
    return func_entry(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr,
                                                                       cusparseIndexBase_t *idxBase) {
    HOOK_TRACE_PROFILE("cusparseSpVecGetIndexBase");
    using func_ptr = cusparseStatus_t (*)(cusparseSpVecDescr_t, cusparseIndexBase_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpVecGetIndexBase"));
    HOOK_CHECK(func_entry);
    return func_entry(spVecDescr, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void **values) {
    HOOK_TRACE_PROFILE("cusparseSpVecGetValues");
    using func_ptr = cusparseStatus_t (*)(cusparseSpVecDescr_t, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpVecGetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(spVecDescr, values);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void *values) {
    HOOK_TRACE_PROFILE("cusparseSpVecSetValues");
    using func_ptr = cusparseStatus_t (*)(cusparseSpVecDescr_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpVecSetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(spVecDescr, values);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t *dnVecDescr, int64_t size,
                                                                 void *values, cudaDataType valueType) {
    HOOK_TRACE_PROFILE("cusparseCreateDnVec");
    using func_ptr = cusparseStatus_t (*)(cusparseDnVecDescr_t *, int64_t, void *, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateDnVec"));
    HOOK_CHECK(func_entry);
    return func_entry(dnVecDescr, size, values, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
    HOOK_TRACE_PROFILE("cusparseDestroyDnVec");
    using func_ptr = cusparseStatus_t (*)(cusparseDnVecDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyDnVec"));
    HOOK_CHECK(func_entry);
    return func_entry(dnVecDescr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr, int64_t *size,
                                                              void **values, cudaDataType *valueType) {
    HOOK_TRACE_PROFILE("cusparseDnVecGet");
    using func_ptr = cusparseStatus_t (*)(cusparseDnVecDescr_t, int64_t *, void **, cudaDataType *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnVecGet"));
    HOOK_CHECK(func_entry);
    return func_entry(dnVecDescr, size, values, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void **values) {
    HOOK_TRACE_PROFILE("cusparseDnVecGetValues");
    using func_ptr = cusparseStatus_t (*)(cusparseDnVecDescr_t, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnVecGetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(dnVecDescr, values);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void *values) {
    HOOK_TRACE_PROFILE("cusparseDnVecSetValues");
    using func_ptr = cusparseStatus_t (*)(cusparseDnVecDescr_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnVecSetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(dnVecDescr, values);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
    HOOK_TRACE_PROFILE("cusparseDestroySpMat");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroySpMat"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatGetFormat(cusparseSpMatDescr_t spMatDescr,
                                                                    cusparseFormat_t *format) {
    HOOK_TRACE_PROFILE("cusparseSpMatGetFormat");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, cusparseFormat_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatGetFormat"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, format);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatGetIndexBase(cusparseSpMatDescr_t spMatDescr,
                                                                       cusparseIndexBase_t *idxBase) {
    HOOK_TRACE_PROFILE("cusparseSpMatGetIndexBase");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, cusparseIndexBase_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatGetIndexBase"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, idxBase);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void **values) {
    HOOK_TRACE_PROFILE("cusparseSpMatGetValues");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatGetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, values);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void *values) {
    HOOK_TRACE_PROFILE("cusparseSpMatSetValues");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatSetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, values);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr, int64_t *rows,
                                                                  int64_t *cols, int64_t *nnz) {
    HOOK_TRACE_PROFILE("cusparseSpMatGetSize");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatGetSize"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, nnz);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                                                                          int batchCount) {
    HOOK_TRACE_PROFILE("cusparseSpMatSetStridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatSetStridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatGetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                                                                          int *batchCount) {
    HOOK_TRACE_PROFILE("cusparseSpMatGetStridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatGetStridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount,
                                                                        int64_t batchStride) {
    HOOK_TRACE_PROFILE("cusparseCooSetStridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, int, int64_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCooSetStridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, batchCount, batchStride);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount,
                                                                        int64_t offsetsBatchStride,
                                                                        int64_t columnsValuesBatchStride) {
    HOOK_TRACE_PROFILE("cusparseCsrSetStridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, int, int64_t, int64_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCsrSetStridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatGetAttribute(cusparseSpMatDescr_t spMatDescr,
                                                                       cusparseSpMatAttribute_t attribute, void *data,
                                                                       size_t dataSize) {
    HOOK_TRACE_PROFILE("cusparseSpMatGetAttribute");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, cusparseSpMatAttribute_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, attribute, data, dataSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr,
                                                                       cusparseSpMatAttribute_t attribute, void *data,
                                                                       size_t dataSize) {
    HOOK_TRACE_PROFILE("cusparseSpMatSetAttribute");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, cusparseSpMatAttribute_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMatSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, attribute, data, dataSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t *spMatDescr, int64_t rows,
                                                               int64_t cols, int64_t nnz, void *csrRowOffsets,
                                                               void *csrColInd, void *csrValues,
                                                               cusparseIndexType_t csrRowOffsetsType,
                                                               cusparseIndexType_t csrColIndType,
                                                               cusparseIndexBase_t idxBase, cudaDataType valueType) {
    HOOK_TRACE_PROFILE("cusparseCreateCsr");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, void *, void *, void *,
                                          cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCsr"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType,
                      csrColIndType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t *spMatDescr, int64_t rows,
                                                               int64_t cols, int64_t nnz, void *cscColOffsets,
                                                               void *cscRowInd, void *cscValues,
                                                               cusparseIndexType_t cscColOffsetsType,
                                                               cusparseIndexType_t cscRowIndType,
                                                               cusparseIndexBase_t idxBase, cudaDataType valueType) {
    HOOK_TRACE_PROFILE("cusparseCreateCsc");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, void *, void *, void *,
                                          cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCsc"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType,
                      cscRowIndType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows,
                                                            int64_t *cols, int64_t *nnz, void **csrRowOffsets,
                                                            void **csrColInd, void **csrValues,
                                                            cusparseIndexType_t *csrRowOffsetsType,
                                                            cusparseIndexType_t *csrColIndType,
                                                            cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
    HOOK_TRACE_PROFILE("cusparseCsrGet");
    using func_ptr =
        cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *, void **, void **, void **,
                             cusparseIndexType_t *, cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCsrGet"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType,
                      csrColIndType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr,
                                                                    void *csrRowOffsets, void *csrColInd,
                                                                    void *csrValues) {
    HOOK_TRACE_PROFILE("cusparseCsrSetPointers");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, void *, void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCsrSetPointers"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, csrRowOffsets, csrColInd, csrValues);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr,
                                                                    void *cscColOffsets, void *cscRowInd,
                                                                    void *cscValues) {
    HOOK_TRACE_PROFILE("cusparseCscSetPointers");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, void *, void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCscSetPointers"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, cscColOffsets, cscRowInd, cscValues);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t *spMatDescr, int64_t rows,
                                                               int64_t cols, int64_t nnz, void *cooRowInd,
                                                               void *cooColInd, void *cooValues,
                                                               cusparseIndexType_t cooIdxType,
                                                               cusparseIndexBase_t idxBase, cudaDataType valueType) {
    HOOK_TRACE_PROFILE("cusparseCreateCoo");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, void *, void *, void *,
                                          cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCoo"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateCooAoS(cusparseSpMatDescr_t *spMatDescr, int64_t rows,
                                                                  int64_t cols, int64_t nnz, void *cooInd,
                                                                  void *cooValues, cusparseIndexType_t cooIdxType,
                                                                  cusparseIndexBase_t idxBase, cudaDataType valueType) {
    HOOK_TRACE_PROFILE("cusparseCreateCooAoS");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, void *, void *,
                                          cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateCooAoS"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows,
                                                            int64_t *cols, int64_t *nnz, void **cooRowInd,
                                                            void **cooColInd, void **cooValues,
                                                            cusparseIndexType_t *idxType, cusparseIndexBase_t *idxBase,
                                                            cudaDataType *valueType) {
    HOOK_TRACE_PROFILE("cusparseCooGet");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *, void **, void **,
                                          void **, cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCooGet"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows,
                                                               int64_t *cols, int64_t *nnz, void **cooInd,
                                                               void **cooValues, cusparseIndexType_t *idxType,
                                                               cusparseIndexBase_t *idxBase, cudaDataType *valueType) {
    HOOK_TRACE_PROFILE("cusparseCooAoSGet");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *, void **, void **,
                                          cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCooAoSGet"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType, idxBase, valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr, void *cooRows,
                                                                    void *cooColumns, void *cooValues) {
    HOOK_TRACE_PROFILE("cusparseCooSetPointers");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, void *, void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCooSetPointers"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, cooRows, cooColumns, cooValues);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateBlockedEll(cusparseSpMatDescr_t *spMatDescr, int64_t rows,
                                                                      int64_t cols, int64_t ellBlockSize,
                                                                      int64_t ellCols, void *ellColInd, void *ellValue,
                                                                      cusparseIndexType_t ellIdxType,
                                                                      cusparseIndexBase_t idxBase,
                                                                      cudaDataType valueType) {
    HOOK_TRACE_PROFILE("cusparseCreateBlockedEll");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t *, int64_t, int64_t, int64_t, int64_t, void *, void *,
                                          cusparseIndexType_t, cusparseIndexBase_t, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateBlockedEll"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase,
                      valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseBlockedEllGet(cusparseSpMatDescr_t spMatDescr, int64_t *rows,
                                                                   int64_t *cols, int64_t *ellBlockSize,
                                                                   int64_t *ellCols, void **ellColInd, void **ellValue,
                                                                   cusparseIndexType_t *ellIdxType,
                                                                   cusparseIndexBase_t *idxBase,
                                                                   cudaDataType *valueType) {
    HOOK_TRACE_PROFILE("cusparseBlockedEllGet");
    using func_ptr = cusparseStatus_t (*)(cusparseSpMatDescr_t, int64_t *, int64_t *, int64_t *, int64_t *, void **,
                                          void **, cusparseIndexType_t *, cusparseIndexBase_t *, cudaDataType *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseBlockedEllGet"));
    HOOK_CHECK(func_entry);
    return func_entry(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase,
                      valueType);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t *dnMatDescr, int64_t rows,
                                                                 int64_t cols, int64_t ld, void *values,
                                                                 cudaDataType valueType, cusparseOrder_t order) {
    HOOK_TRACE_PROFILE("cusparseCreateDnMat");
    using func_ptr =
        cusparseStatus_t (*)(cusparseDnMatDescr_t *, int64_t, int64_t, int64_t, void *, cudaDataType, cusparseOrder_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseCreateDnMat"));
    HOOK_CHECK(func_entry);
    return func_entry(dnMatDescr, rows, cols, ld, values, valueType, order);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) {
    HOOK_TRACE_PROFILE("cusparseDestroyDnMat");
    using func_ptr = cusparseStatus_t (*)(cusparseDnMatDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDestroyDnMat"));
    HOOK_CHECK(func_entry);
    return func_entry(dnMatDescr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t *rows,
                                                              int64_t *cols, int64_t *ld, void **values,
                                                              cudaDataType *type, cusparseOrder_t *order) {
    HOOK_TRACE_PROFILE("cusparseDnMatGet");
    using func_ptr = cusparseStatus_t (*)(cusparseDnMatDescr_t, int64_t *, int64_t *, int64_t *, void **,
                                          cudaDataType *, cusparseOrder_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnMatGet"));
    HOOK_CHECK(func_entry);
    return func_entry(dnMatDescr, rows, cols, ld, values, type, order);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void **values) {
    HOOK_TRACE_PROFILE("cusparseDnMatGetValues");
    using func_ptr = cusparseStatus_t (*)(cusparseDnMatDescr_t, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnMatGetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(dnMatDescr, values);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void *values) {
    HOOK_TRACE_PROFILE("cusparseDnMatSetValues");
    using func_ptr = cusparseStatus_t (*)(cusparseDnMatDescr_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnMatSetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(dnMatDescr, values);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr,
                                                                          int batchCount, int64_t batchStride) {
    HOOK_TRACE_PROFILE("cusparseDnMatSetStridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseDnMatDescr_t, int, int64_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnMatSetStridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(dnMatDescr, batchCount, batchStride);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDnMatGetStridedBatch(cusparseDnMatDescr_t dnMatDescr,
                                                                          int *batchCount, int64_t *batchStride) {
    HOOK_TRACE_PROFILE("cusparseDnMatGetStridedBatch");
    using func_ptr = cusparseStatus_t (*)(cusparseDnMatDescr_t, int *, int64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDnMatGetStridedBatch"));
    HOOK_CHECK(func_entry);
    return func_entry(dnMatDescr, batchCount, batchStride);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseAxpby(cusparseHandle_t handle, const void *alpha,
                                                           cusparseSpVecDescr_t vecX, const void *beta,
                                                           cusparseDnVecDescr_t vecY) {
    HOOK_TRACE_PROFILE("cusparseAxpby");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, const void *, cusparseSpVecDescr_t, const void *, cusparseDnVecDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseAxpby"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alpha, vecX, beta, vecY);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseGather(cusparseHandle_t handle, cusparseDnVecDescr_t vecY,
                                                            cusparseSpVecDescr_t vecX) {
    HOOK_TRACE_PROFILE("cusparseGather");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDnVecDescr_t, cusparseSpVecDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseGather"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, vecY, vecX);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseScatter(cusparseHandle_t handle, cusparseSpVecDescr_t vecX,
                                                             cusparseDnVecDescr_t vecY) {
    HOOK_TRACE_PROFILE("cusparseScatter");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseScatter"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, vecX, vecY);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseRot(cusparseHandle_t handle, const void *c_coeff,
                                                         const void *s_coeff, cusparseSpVecDescr_t vecX,
                                                         cusparseDnVecDescr_t vecY) {
    HOOK_TRACE_PROFILE("cusparseRot");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, const void *, const void *, cusparseSpVecDescr_t, cusparseDnVecDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseRot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, c_coeff, s_coeff, vecX, vecY);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX,
                                                                     cusparseSpVecDescr_t vecX,
                                                                     cusparseDnVecDescr_t vecY, const void *result,
                                                                     cudaDataType computeType, size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseSpVV_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t,
                                          cusparseDnVecDescr_t, const void *, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpVV_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opX, vecX, vecY, result, computeType, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX,
                                                          cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                                                          void *result, cudaDataType computeType,
                                                          void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpVV");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t,
                                          cusparseDnVecDescr_t, void *, cudaDataType, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpVV"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opX, vecX, vecY, result, computeType, externalBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t handle,
                                                                              cusparseSpMatDescr_t matA,
                                                                              cusparseDnMatDescr_t matB,
                                                                              cusparseSparseToDenseAlg_t alg,
                                                                              size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseSparseToDense_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseSpMatDescr_t, cusparseDnMatDescr_t,
                                          cusparseSparseToDenseAlg_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSparseToDense_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, matA, matB, alg, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSparseToDense(cusparseHandle_t handle, cusparseSpMatDescr_t matA,
                                                                   cusparseDnMatDescr_t matB,
                                                                   cusparseSparseToDenseAlg_t alg, void *buffer) {
    HOOK_TRACE_PROFILE("cusparseSparseToDense");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseSpMatDescr_t, cusparseDnMatDescr_t,
                                          cusparseSparseToDenseAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSparseToDense"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, matA, matB, alg, buffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t handle,
                                                                              cusparseDnMatDescr_t matA,
                                                                              cusparseSpMatDescr_t matB,
                                                                              cusparseDenseToSparseAlg_t alg,
                                                                              size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseDenseToSparse_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t,
                                          cusparseDenseToSparseAlg_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDenseToSparse_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, matA, matB, alg, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t handle,
                                                                            cusparseDnMatDescr_t matA,
                                                                            cusparseSpMatDescr_t matB,
                                                                            cusparseDenseToSparseAlg_t alg,
                                                                            void *buffer) {
    HOOK_TRACE_PROFILE("cusparseDenseToSparse_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t,
                                          cusparseDenseToSparseAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDenseToSparse_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, matA, matB, alg, buffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t handle,
                                                                           cusparseDnMatDescr_t matA,
                                                                           cusparseSpMatDescr_t matB,
                                                                           cusparseDenseToSparseAlg_t alg,
                                                                           void *buffer) {
    HOOK_TRACE_PROFILE("cusparseDenseToSparse_convert");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t,
                                          cusparseDenseToSparseAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseDenseToSparse_convert"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, matA, matB, alg, buffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA,
                                                          const void *alpha, cusparseSpMatDescr_t matA,
                                                          cusparseDnVecDescr_t vecX, const void *beta,
                                                          cusparseDnVecDescr_t vecY, cudaDataType computeType,
                                                          cusparseSpMVAlg_t alg, void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpMV");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
                                          cusparseDnVecDescr_t, const void *, cusparseDnVecDescr_t, cudaDataType,
                                          cusparseSpMVAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMV"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                                                     const void *alpha, cusparseSpMatDescr_t matA,
                                                                     cusparseDnVecDescr_t vecX, const void *beta,
                                                                     cusparseDnVecDescr_t vecY,
                                                                     cudaDataType computeType, cusparseSpMVAlg_t alg,
                                                                     size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseSpMV_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
                                          cusparseDnVecDescr_t, const void *, cusparseDnVecDescr_t, cudaDataType,
                                          cusparseSpMVAlg_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMV_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t *descr) {
    HOOK_TRACE_PROFILE("cusparseSpSV_createDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseSpSVDescr_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSV_createDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(descr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) {
    HOOK_TRACE_PROFILE("cusparseSpSV_destroyDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseSpSVDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSV_destroyDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(descr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSV_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, const void *alpha, cusparseSpMatDescr_t matA,
    cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg,
    cusparseSpSVDescr_t spsvDescr, size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseSpSV_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
                                          cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t,
                                          cusparseSpSVDescr_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSV_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA,
                                                                   const void *alpha, cusparseSpMatDescr_t matA,
                                                                   cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                                                                   cudaDataType computeType, cusparseSpSVAlg_t alg,
                                                                   cusparseSpSVDescr_t spsvDescr,
                                                                   void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpSV_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
                                          cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t,
                                          cusparseSpSVDescr_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSV_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA,
                                                                const void *alpha, cusparseSpMatDescr_t matA,
                                                                cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                                                                cudaDataType computeType, cusparseSpSVAlg_t alg,
                                                                cusparseSpSVDescr_t spsvDescr) {
    HOOK_TRACE_PROFILE("cusparseSpSV_solve");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, const void *, cusparseSpMatDescr_t,
                                          cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t,
                                          cusparseSpSVDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSV_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t *descr) {
    HOOK_TRACE_PROFILE("cusparseSpSM_createDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseSpSMDescr_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSM_createDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(descr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) {
    HOOK_TRACE_PROFILE("cusparseSpSM_destroyDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseSpSMDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSM_destroyDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(descr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType,
    cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseSpSM_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t,
                                          cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSM_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA,
                                                                   cusparseOperation_t opB, const void *alpha,
                                                                   cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                                                   cusparseDnMatDescr_t matC, cudaDataType computeType,
                                                                   cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr,
                                                                   void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpSM_analysis");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t,
                                          cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSM_analysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA,
                                                                cusparseOperation_t opB, const void *alpha,
                                                                cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                                                cusparseDnMatDescr_t matC, cudaDataType computeType,
                                                                cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr) {
    HOOK_TRACE_PROFILE("cusparseSpSM_solve");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t,
                                          cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpSM_solve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const void *beta, cusparseDnMatDescr_t matC,
    cudaDataType computeType, cusparseSpMMAlg_t alg, size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseSpMM_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseSpMatDescr_t, cusparseDnMatDescr_t, const void *,
                                          cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMM_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMM_preprocess(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const void *beta, cusparseDnMatDescr_t matC,
    cudaDataType computeType, cusparseSpMMAlg_t alg, void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpMM_preprocess");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseSpMatDescr_t, cusparseDnMatDescr_t, const void *,
                                          cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMM_preprocess"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA,
                                                          cusparseOperation_t opB, const void *alpha,
                                                          cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                                          const void *beta, cusparseDnMatDescr_t matC,
                                                          cudaDataType computeType, cusparseSpMMAlg_t alg,
                                                          void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseSpMM");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseSpMatDescr_t, cusparseDnMatDescr_t, const void *,
                                          cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpMM"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t *descr) {
    HOOK_TRACE_PROFILE("cusparseSpGEMM_createDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseSpGEMMDescr_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMM_createDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(descr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) {
    HOOK_TRACE_PROFILE("cusparseSpGEMM_destroyDescr");
    using func_ptr = cusparseStatus_t (*)(cusparseSpGEMMDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMM_destroyDescr"));
    HOOK_CHECK(func_entry);
    return func_entry(descr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpGEMM_workEstimation(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize1,
    void *externalBuffer1) {
    HOOK_TRACE_PROFILE("cusparseSpGEMM_workEstimation");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                             cusparseSpMatDescr_t, cusparseSpMatDescr_t, const void *, cusparseSpMatDescr_t,
                             cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMM_workEstimation"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize1,
                      externalBuffer1);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t
    cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
                           cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void *beta,
                           cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
                           cusparseSpGEMMDescr_t spgemmDescr, size_t *bufferSize2, void *externalBuffer2) {
    HOOK_TRACE_PROFILE("cusparseSpGEMM_compute");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                             cusparseSpMatDescr_t, cusparseSpMatDescr_t, const void *, cusparseSpMatDescr_t,
                             cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMM_compute"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize2,
                      externalBuffer2);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t opA,
                                                                 cusparseOperation_t opB, const void *alpha,
                                                                 cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB,
                                                                 const void *beta, cusparseSpMatDescr_t matC,
                                                                 cudaDataType computeType, cusparseSpGEMMAlg_t alg,
                                                                 cusparseSpGEMMDescr_t spgemmDescr) {
    HOOK_TRACE_PROFILE("cusparseSpGEMM_copy");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                             cusparseSpMatDescr_t, cusparseSpMatDescr_t, const void *, cusparseSpMatDescr_t,
                             cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMM_copy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpGEMMreuse_workEstimation(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseSpMatDescr_t matA,
    cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize1, void *externalBuffer1) {
    HOOK_TRACE_PROFILE("cusparseSpGEMMreuse_workEstimation");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t,
                                          cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpMatDescr_t,
                                          cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMMreuse_workEstimation"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize1, externalBuffer1);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpGEMMreuse_nnz(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseSpMatDescr_t matA,
    cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize2, void *externalBuffer2, size_t *bufferSize3, void *externalBuffer3, size_t *bufferSize4,
    void *externalBuffer4) {
    HOOK_TRACE_PROFILE("cusparseSpGEMMreuse_nnz");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseSpMatDescr_t,
                             cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t,
                             size_t *, void *, size_t *, void *, size_t *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMMreuse_nnz"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize2, externalBuffer2, bufferSize3,
                      externalBuffer3, bufferSize4, externalBuffer4);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpGEMMreuse_copy(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseSpMatDescr_t matA,
    cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr,
    size_t *bufferSize5, void *externalBuffer5) {
    HOOK_TRACE_PROFILE("cusparseSpGEMMreuse_copy");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t,
                                          cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpMatDescr_t,
                                          cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, size_t *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMMreuse_copy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize5, externalBuffer5);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSpGEMMreuse_compute(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr) {
    HOOK_TRACE_PROFILE("cusparseSpGEMMreuse_compute");
    using func_ptr =
        cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                             cusparseSpMatDescr_t, cusparseSpMatDescr_t, const void *, cusparseSpMatDescr_t,
                             cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSpGEMMreuse_compute"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseConstrainedGeMM(cusparseHandle_t handle, cusparseOperation_t opA,
                                                                     cusparseOperation_t opB, const void *alpha,
                                                                     cusparseDnMatDescr_t matA,
                                                                     cusparseDnMatDescr_t matB, const void *beta,
                                                                     cusparseSpMatDescr_t matC,
                                                                     cudaDataType computeType, void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseConstrainedGeMM");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
                                          cusparseSpMatDescr_t, cudaDataType, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseConstrainedGeMM"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, externalBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseConstrainedGeMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
    cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseConstrainedGeMM_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
                                          cusparseSpMatDescr_t, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseConstrainedGeMM_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSDDMM_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
    cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t *bufferSize) {
    HOOK_TRACE_PROFILE("cusparseSDDMM_bufferSize");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
                                          cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSDDMM_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSDDMM_preprocess(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void *alpha,
    cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB, const void *beta, cusparseSpMatDescr_t matC,
    cudaDataType computeType, cusparseSDDMMAlg_t alg, void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseSDDMM_preprocess");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
                                          cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSDDMM_preprocess"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusparseStatus_t cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA,
                                                           cusparseOperation_t opB, const void *alpha,
                                                           cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB,
                                                           const void *beta, cusparseSpMatDescr_t matC,
                                                           cudaDataType computeType, cusparseSDDMMAlg_t alg,
                                                           void *externalBuffer) {
    HOOK_TRACE_PROFILE("cusparseSDDMM");
    using func_ptr = cusparseStatus_t (*)(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, const void *,
                                          cusparseDnMatDescr_t, cusparseDnMatDescr_t, const void *,
                                          cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSPARSE_SYMBOL("cusparseSDDMM"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
}
