// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 46 apis

#include "cublasLt_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle) {
    HOOK_TRACE_PROFILE("cublasLtCreate");
    using func_ptr = cublasStatus_t (*)(cublasLtHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(lightHandle);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
    HOOK_TRACE_PROFILE("cublasLtDestroy");
    using func_ptr = cublasStatus_t (*)(cublasLtHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(lightHandle);
}

HOOK_C_API HOOK_DECL_EXPORT const char *cublasLtGetStatusName(cublasStatus_t status) {
    HOOK_TRACE_PROFILE("cublasLtGetStatusName");
    using func_ptr = const char *(*)(cublasStatus_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtGetStatusName"));
    HOOK_CHECK(func_entry);
    return func_entry(status);
}

HOOK_C_API HOOK_DECL_EXPORT const char *cublasLtGetStatusString(cublasStatus_t status) {
    HOOK_TRACE_PROFILE("cublasLtGetStatusString");
    using func_ptr = const char *(*)(cublasStatus_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtGetStatusString"));
    HOOK_CHECK(func_entry);
    return func_entry(status);
}

HOOK_C_API HOOK_DECL_EXPORT size_t cublasLtGetVersion() {
    HOOK_TRACE_PROFILE("cublasLtGetVersion");
    using func_ptr = size_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT size_t cublasLtGetCudartVersion() {
    HOOK_TRACE_PROFILE("cublasLtGetCudartVersion");
    using func_ptr = size_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtGetCudartVersion"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtGetProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("cublasLtGetProperty");
    using func_ptr = cublasStatus_t (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtGetProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void *alpha, const void *A,
    cublasLtMatrixLayout_t Adesc, const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta, const void *C,
    cublasLtMatrixLayout_t Cdesc, void *D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo,
    void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cublasLtMatmul");
    using func_ptr =
        cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *, cublasLtMatrixLayout_t,
                           const void *, cublasLtMatrixLayout_t, const void *, const void *, cublasLtMatrixLayout_t,
                           void *, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *, size_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmul"));
    HOOK_CHECK(func_entry);
    return func_entry(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, algo, workspace,
                      workspaceSizeInBytes, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixTransform(cublasLtHandle_t lightHandle,
                                                                   cublasLtMatrixTransformDesc_t transformDesc,
                                                                   const void *alpha, const void *A,
                                                                   cublasLtMatrixLayout_t Adesc, const void *beta,
                                                                   const void *B, cublasLtMatrixLayout_t Bdesc, void *C,
                                                                   cublasLtMatrixLayout_t Cdesc, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cublasLtMatrixTransform");
    using func_ptr = cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatrixTransformDesc_t, const void *, const void *,
                                        cublasLtMatrixLayout_t, const void *, const void *, cublasLtMatrixLayout_t,
                                        void *, cublasLtMatrixLayout_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixTransform"));
    HOOK_CHECK(func_entry);
    return func_entry(lightHandle, transformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayout_t matLayout,
                                                                             size_t size, cudaDataType type,
                                                                             uint64_t rows, uint64_t cols, int64_t ld) {
    HOOK_TRACE_PROFILE("cublasLtMatrixLayoutInit_internal");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixLayout_t, size_t, cudaDataType, uint64_t, uint64_t, int64_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixLayoutInit_internal"));
    HOOK_CHECK(func_entry);
    return func_entry(matLayout, size, type, rows, cols, ld);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixLayoutInit(cublasLtMatrixLayout_t matLayout, cudaDataType type,
                                                                    uint64_t rows, uint64_t cols, int64_t ld) {
    HOOK_TRACE_PROFILE("cublasLtMatrixLayoutInit");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixLayout_t, cudaDataType, uint64_t, uint64_t, int64_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixLayoutInit"));
    HOOK_CHECK(func_entry);
    return func_entry(matLayout, type, rows, cols, ld);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                                                      cudaDataType type, uint64_t rows, uint64_t cols,
                                                                      int64_t ld) {
    HOOK_TRACE_PROFILE("cublasLtMatrixLayoutCreate");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixLayout_t *, cudaDataType, uint64_t, uint64_t, int64_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixLayoutCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(matLayout, type, rows, cols, ld);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    HOOK_TRACE_PROFILE("cublasLtMatrixLayoutDestroy");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixLayout_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixLayoutDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(matLayout);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                                            cublasLtMatrixLayoutAttribute_t attr,
                                                                            const void *buf, size_t sizeInBytes) {
    HOOK_TRACE_PROFILE("cublasLtMatrixLayoutSetAttribute");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixLayoutSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(matLayout, attr, buf, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout,
                                                                            cublasLtMatrixLayoutAttribute_t attr,
                                                                            void *buf, size_t sizeInBytes,
                                                                            size_t *sizeWritten) {
    HOOK_TRACE_PROFILE("cublasLtMatrixLayoutGetAttribute");
    using func_ptr =
        cublasStatus_t (*)(cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, void *, size_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixLayoutGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(matLayout, attr, buf, sizeInBytes, sizeWritten);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t matmulDesc, size_t size,
                                                                           cublasComputeType_t computeType,
                                                                           cudaDataType_t scaleType) {
    HOOK_TRACE_PROFILE("cublasLtMatmulDescInit_internal");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulDesc_t, size_t, cublasComputeType_t, cudaDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulDescInit_internal"));
    HOOK_CHECK(func_entry);
    return func_entry(matmulDesc, size, computeType, scaleType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulDescInit(cublasLtMatmulDesc_t matmulDesc,
                                                                  cublasComputeType_t computeType,
                                                                  cudaDataType_t scaleType) {
    HOOK_TRACE_PROFILE("cublasLtMatmulDescInit");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulDesc_t, cublasComputeType_t, cudaDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulDescInit"));
    HOOK_CHECK(func_entry);
    return func_entry(matmulDesc, computeType, scaleType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                                                    cublasComputeType_t computeType,
                                                                    cudaDataType_t scaleType) {
    HOOK_TRACE_PROFILE("cublasLtMatmulDescCreate");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulDesc_t *, cublasComputeType_t, cudaDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulDescCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(matmulDesc, computeType, scaleType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    HOOK_TRACE_PROFILE("cublasLtMatmulDescDestroy");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulDesc_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulDescDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(matmulDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                                          cublasLtMatmulDescAttributes_t attr,
                                                                          const void *buf, size_t sizeInBytes) {
    HOOK_TRACE_PROFILE("cublasLtMatmulDescSetAttribute");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulDescSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(matmulDesc, attr, buf, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                                          cublasLtMatmulDescAttributes_t attr,
                                                                          void *buf, size_t sizeInBytes,
                                                                          size_t *sizeWritten) {
    HOOK_TRACE_PROFILE("cublasLtMatmulDescGetAttribute");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, void *, size_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulDescGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(matmulDesc, attr, buf, sizeInBytes, sizeWritten);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixTransformDescInit_internal(
    cublasLtMatrixTransformDesc_t transformDesc, size_t size, cudaDataType scaleType) {
    HOOK_TRACE_PROFILE("cublasLtMatrixTransformDescInit_internal");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixTransformDesc_t, size_t, cudaDataType);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixTransformDescInit_internal"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc, size, scaleType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixTransformDescInit(cublasLtMatrixTransformDesc_t transformDesc,
                                                                           cudaDataType scaleType) {
    HOOK_TRACE_PROFILE("cublasLtMatrixTransformDescInit");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixTransformDesc_t, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixTransformDescInit"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc, scaleType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t
    cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t *transformDesc, cudaDataType scaleType) {
    HOOK_TRACE_PROFILE("cublasLtMatrixTransformDescCreate");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixTransformDesc_t *, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixTransformDescCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc, scaleType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t
    cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc) {
    HOOK_TRACE_PROFILE("cublasLtMatrixTransformDescDestroy");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixTransformDesc_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixTransformDescDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixTransformDescSetAttribute(
    cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, const void *buf,
    size_t sizeInBytes) {
    HOOK_TRACE_PROFILE("cublasLtMatrixTransformDescSetAttribute");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescAttributes_t,
                                        const void *, size_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixTransformDescSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc, attr, buf, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatrixTransformDescGetAttribute(
    cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, void *buf,
    size_t sizeInBytes, size_t *sizeWritten) {
    HOOK_TRACE_PROFILE("cublasLtMatrixTransformDescGetAttribute");
    using func_ptr = cublasStatus_t (*)(cublasLtMatrixTransformDesc_t, cublasLtMatrixTransformDescAttributes_t, void *,
                                        size_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatrixTransformDescGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc, attr, buf, sizeInBytes, sizeWritten);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t pref,
                                                                                 size_t size) {
    HOOK_TRACE_PROFILE("cublasLtMatmulPreferenceInit_internal");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulPreference_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulPreferenceInit_internal"));
    HOOK_CHECK(func_entry);
    return func_entry(pref, size);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulPreferenceInit(cublasLtMatmulPreference_t pref) {
    HOOK_TRACE_PROFILE("cublasLtMatmulPreferenceInit");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulPreference_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulPreferenceInit"));
    HOOK_CHECK(func_entry);
    return func_entry(pref);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    HOOK_TRACE_PROFILE("cublasLtMatmulPreferenceCreate");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulPreference_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulPreferenceCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pref);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    HOOK_TRACE_PROFILE("cublasLtMatmulPreferenceDestroy");
    using func_ptr = cublasStatus_t (*)(cublasLtMatmulPreference_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulPreferenceDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(pref);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulPreferenceSetAttribute(
    cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void *buf, size_t sizeInBytes) {
    HOOK_TRACE_PROFILE("cublasLtMatmulPreferenceSetAttribute");
    using func_ptr =
        cublasStatus_t (*)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulPreferenceSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(pref, attr, buf, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t
    cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr,
                                         void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    HOOK_TRACE_PROFILE("cublasLtMatmulPreferenceGetAttribute");
    using func_ptr =
        cublasStatus_t (*)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, void *, size_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulPreferenceGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(pref, attr, buf, sizeInBytes, sizeWritten);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference, int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray, int *returnAlgoCount) {
    HOOK_TRACE_PROFILE("cublasLtMatmulAlgoGetHeuristic");
    using func_ptr = cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
                                        cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                        cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulAlgoGetHeuristic"));
    HOOK_CHECK(func_entry);
    return func_entry(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, requestedAlgoCount,
                      heuristicResultsArray, returnAlgoCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulAlgoGetIds(cublasLtHandle_t lightHandle,
                                                                    cublasComputeType_t computeType,
                                                                    cudaDataType_t scaleType, cudaDataType_t Atype,
                                                                    cudaDataType_t Btype, cudaDataType_t Ctype,
                                                                    cudaDataType_t Dtype, int requestedAlgoCount,
                                                                    int algoIdsArray, int *returnAlgoCount) {
    HOOK_TRACE_PROFILE("cublasLtMatmulAlgoGetIds");
    using func_ptr = cublasStatus_t (*)(cublasLtHandle_t, cublasComputeType_t, cudaDataType_t, cudaDataType_t,
                                        cudaDataType_t, cudaDataType_t, cudaDataType_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulAlgoGetIds"));
    HOOK_CHECK(func_entry);
    return func_entry(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, requestedAlgoCount, algoIdsArray,
                      returnAlgoCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulAlgoInit(
    cublasLtHandle_t lightHandle, cublasComputeType_t computeType, cudaDataType_t scaleType, cudaDataType_t Atype,
    cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t Dtype, int algoId, cublasLtMatmulAlgo_t *algo) {
    HOOK_TRACE_PROFILE("cublasLtMatmulAlgoInit");
    using func_ptr = cublasStatus_t (*)(cublasLtHandle_t, cublasComputeType_t, cudaDataType_t, cudaDataType_t,
                                        cudaDataType_t, cudaDataType_t, cudaDataType_t, int, cublasLtMatmulAlgo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulAlgoInit"));
    HOOK_CHECK(func_entry);
    return func_entry(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId, algo);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulAlgoCheck(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo, cublasLtMatmulHeuristicResult_t *result) {
    HOOK_TRACE_PROFILE("cublasLtMatmulAlgoCheck");
    using func_ptr = cublasStatus_t (*)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
                                        cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                        const cublasLtMatmulAlgo_t *, cublasLtMatmulHeuristicResult_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulAlgoCheck"));
    HOOK_CHECK(func_entry);
    return func_entry(lightHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, algo, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t *algo,
                                                                             cublasLtMatmulAlgoCapAttributes_t attr,
                                                                             void *buf, size_t sizeInBytes,
                                                                             size_t *sizeWritten) {
    HOOK_TRACE_PROFILE("cublasLtMatmulAlgoCapGetAttribute");
    using func_ptr =
        cublasStatus_t (*)(const cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoCapAttributes_t, void *, size_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulAlgoCapGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(algo, attr, buf, sizeInBytes, sizeWritten);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(
    cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoConfigAttributes_t attr, const void *buf, size_t sizeInBytes) {
    HOOK_TRACE_PROFILE("cublasLtMatmulAlgoConfigSetAttribute");
    using func_ptr =
        cublasStatus_t (*)(cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoConfigAttributes_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulAlgoConfigSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(algo, attr, buf, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t
    cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t *algo, cublasLtMatmulAlgoConfigAttributes_t attr,
                                         void *buf, size_t sizeInBytes, size_t *sizeWritten) {
    HOOK_TRACE_PROFILE("cublasLtMatmulAlgoConfigGetAttribute");
    using func_ptr = cublasStatus_t (*)(const cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoConfigAttributes_t, void *,
                                        size_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtMatmulAlgoConfigGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(algo, attr, buf, sizeInBytes, sizeWritten);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) {
    HOOK_TRACE_PROFILE("cublasLtLoggerSetCallback");
    using func_ptr = cublasStatus_t (*)(cublasLtLoggerCallback_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtLoggerSetCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(callback);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtLoggerSetFile(FILE *file) {
    HOOK_TRACE_PROFILE("cublasLtLoggerSetFile");
    using func_ptr = cublasStatus_t (*)(FILE *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtLoggerSetFile"));
    HOOK_CHECK(func_entry);
    return func_entry(file);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtLoggerOpenFile(const char *logFile) {
    HOOK_TRACE_PROFILE("cublasLtLoggerOpenFile");
    using func_ptr = cublasStatus_t (*)(const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtLoggerOpenFile"));
    HOOK_CHECK(func_entry);
    return func_entry(logFile);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtLoggerSetLevel(int level) {
    HOOK_TRACE_PROFILE("cublasLtLoggerSetLevel");
    using func_ptr = cublasStatus_t (*)(int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtLoggerSetLevel"));
    HOOK_CHECK(func_entry);
    return func_entry(level);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtLoggerSetMask(int mask) {
    HOOK_TRACE_PROFILE("cublasLtLoggerSetMask");
    using func_ptr = cublasStatus_t (*)(int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtLoggerSetMask"));
    HOOK_CHECK(func_entry);
    return func_entry(mask);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLtLoggerForceDisable() {
    HOOK_TRACE_PROFILE("cublasLtLoggerForceDisable");
    using func_ptr = cublasStatus_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLASLT_SYMBOL("cublasLtLoggerForceDisable"));
    HOOK_CHECK(func_entry);
    return func_entry();
}
