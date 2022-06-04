// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 592 apis

#include "cublas_subset.h"
#include "cusolver_subset.h"
#include "cusparse_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("cusolverGetProperty");
    using func_ptr = cusolverStatus_t (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverGetProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverGetVersion(int *version) {
    HOOK_TRACE_PROFILE("cusolverGetVersion");
    using func_ptr = cusolverStatus_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(version);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) {
    HOOK_TRACE_PROFILE("cusolverDnCreate");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverDnDestroy");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) {
    HOOK_TRACE_PROFILE("cusolverDnSetStream");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId) {
    HOOK_TRACE_PROFILE("cusolverDnGetStream");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cudaStream_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnGetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsCreate(cusolverDnIRSParams_t *params_ptr) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsCreate");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(params_ptr);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsDestroy(cusolverDnIRSParams_t params) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsDestroy");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverDnIRSParamsSetRefinementSolver(cusolverDnIRSParams_t params, cusolverIRSRefinement_t refinement_solver) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsSetRefinementSolver");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverIRSRefinement_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsSetRefinementSolver"));
    HOOK_CHECK(func_entry);
    return func_entry(params, refinement_solver);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverDnIRSParamsSetSolverMainPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsSetSolverMainPrecision");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsSetSolverMainPrecision"));
    HOOK_CHECK(func_entry);
    return func_entry(params, solver_main_precision);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsSetSolverLowestPrecision(
    cusolverDnIRSParams_t params, cusolverPrecType_t solver_lowest_precision) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsSetSolverLowestPrecision");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsSetSolverLowestPrecision"));
    HOOK_CHECK(func_entry);
    return func_entry(params, solver_lowest_precision);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverDnIRSParamsSetSolverPrecisions(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision,
                                           cusolverPrecType_t solver_lowest_precision) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsSetSolverPrecisions");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolverPrecType_t, cusolverPrecType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsSetSolverPrecisions"));
    HOOK_CHECK(func_entry);
    return func_entry(params, solver_main_precision, solver_lowest_precision);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsSetTol(cusolverDnIRSParams_t params, double val) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsSetTol");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsSetTol"));
    HOOK_CHECK(func_entry);
    return func_entry(params, val);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams_t params, double val) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsSetTolInner");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsSetTolInner"));
    HOOK_CHECK(func_entry);
    return func_entry(params, val);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsSetMaxIters(cusolverDnIRSParams_t params,
                                                                            cusolver_int_t maxiters) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsSetMaxIters");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsSetMaxIters"));
    HOOK_CHECK(func_entry);
    return func_entry(params, maxiters);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsSetMaxItersInner(cusolverDnIRSParams_t params,
                                                                                 cusolver_int_t maxiters_inner) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsSetMaxItersInner");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsSetMaxItersInner"));
    HOOK_CHECK(func_entry);
    return func_entry(params, maxiters_inner);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsGetMaxIters(cusolverDnIRSParams_t params,
                                                                            cusolver_int_t *maxiters) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsGetMaxIters");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsGetMaxIters"));
    HOOK_CHECK(func_entry);
    return func_entry(params, maxiters);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams_t params) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsEnableFallback");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsEnableFallback"));
    HOOK_CHECK(func_entry);
    return func_entry(params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams_t params) {
    HOOK_TRACE_PROFILE("cusolverDnIRSParamsDisableFallback");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSParams_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSParamsDisableFallback"));
    HOOK_CHECK(func_entry);
    return func_entry(params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSInfosDestroy(cusolverDnIRSInfos_t infos) {
    HOOK_TRACE_PROFILE("cusolverDnIRSInfosDestroy");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSInfos_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSInfosDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(infos);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSInfosCreate(cusolverDnIRSInfos_t *infos_ptr) {
    HOOK_TRACE_PROFILE("cusolverDnIRSInfosCreate");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSInfos_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSInfosCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(infos_ptr);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSInfosGetNiters(cusolverDnIRSInfos_t infos,
                                                                         cusolver_int_t *niters) {
    HOOK_TRACE_PROFILE("cusolverDnIRSInfosGetNiters");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSInfosGetNiters"));
    HOOK_CHECK(func_entry);
    return func_entry(infos, niters);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSInfosGetOuterNiters(cusolverDnIRSInfos_t infos,
                                                                              cusolver_int_t *outer_niters) {
    HOOK_TRACE_PROFILE("cusolverDnIRSInfosGetOuterNiters");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSInfosGetOuterNiters"));
    HOOK_CHECK(func_entry);
    return func_entry(infos, outer_niters);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSInfosRequestResidual(cusolverDnIRSInfos_t infos) {
    HOOK_TRACE_PROFILE("cusolverDnIRSInfosRequestResidual");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSInfos_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSInfosRequestResidual"));
    HOOK_CHECK(func_entry);
    return func_entry(infos);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSInfosGetResidualHistory(cusolverDnIRSInfos_t infos,
                                                                                  void **residual_history) {
    HOOK_TRACE_PROFILE("cusolverDnIRSInfosGetResidualHistory");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSInfos_t, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSInfosGetResidualHistory"));
    HOOK_CHECK(func_entry);
    return func_entry(infos, residual_history);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSInfosGetMaxIters(cusolverDnIRSInfos_t infos,
                                                                           cusolver_int_t *maxiters) {
    HOOK_TRACE_PROFILE("cusolverDnIRSInfosGetMaxIters");
    using func_ptr = cusolverStatus_t (*)(cusolverDnIRSInfos_t, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSInfosGetMaxIters"));
    HOOK_CHECK(func_entry);
    return func_entry(infos, maxiters);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZZgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cusolver_int_t *dipiv, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZZgesv");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             cusolver_int_t *, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZZgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZCgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cusolver_int_t *dipiv, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZCgesv");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             cusolver_int_t *, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZCgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZKgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cusolver_int_t *dipiv, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZKgesv");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             cusolver_int_t *, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZKgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZEgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cusolver_int_t *dipiv, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZEgesv");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             cusolver_int_t *, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZEgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZYgesv(
    cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA, cusolver_int_t ldda,
    cusolver_int_t *dipiv, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZYgesv");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             cusolver_int_t *, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZYgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCCgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, cuComplex *dB, cusolver_int_t lddb,
                                                              cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnCCgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCCgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCEgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, cuComplex *dB, cusolver_int_t lddb,
                                                              cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnCEgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCEgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCKgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, cuComplex *dB, cusolver_int_t lddb,
                                                              cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnCKgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCKgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCYgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, cuComplex *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, cuComplex *dB, cusolver_int_t lddb,
                                                              cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnCYgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCYgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDDgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDDgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                                          cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDDgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDSgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDSgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                                          cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDSgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDHgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDHgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                                          cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDHgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDBgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDBgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                                          cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDBgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDXgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDXgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                                          cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDXgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSSgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, float *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
                                                              float *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnSSgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *, cusolver_int_t,
                                          cusolver_int_t *, float *, cusolver_int_t, float *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSSgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSHgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, float *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
                                                              float *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnSHgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *, cusolver_int_t,
                                          cusolver_int_t *, float *, cusolver_int_t, float *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSHgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSBgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, float *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
                                                              float *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnSBgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *, cusolver_int_t,
                                          cusolver_int_t *, float *, cusolver_int_t, float *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSBgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSXgesv(cusolverDnHandle_t handle, cusolver_int_t n,
                                                              cusolver_int_t nrhs, float *dA, cusolver_int_t ldda,
                                                              cusolver_int_t *dipiv, float *dB, cusolver_int_t lddb,
                                                              float *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnSXgesv");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *, cusolver_int_t,
                                          cusolver_int_t *, float *, cusolver_int_t, float *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSXgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZZgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuDoubleComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZZgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZZgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuDoubleComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZCgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZCgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZKgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuDoubleComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZKgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZKgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZEgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuDoubleComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZEgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZEgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZYgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuDoubleComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZYgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZYgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuComplex *dB, cusolver_int_t lddb,
                                                                         cuComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnCCgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCCgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCKgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuComplex *dB, cusolver_int_t lddb,
                                                                         cuComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnCKgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCKgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCEgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuComplex *dB, cusolver_int_t lddb,
                                                                         cuComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnCEgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCEgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCYgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, cuComplex *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         cuComplex *dB, cusolver_int_t lddb,
                                                                         cuComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnCYgesv_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, cusolver_int_t *, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCYgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDDgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, double *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         double *dB, cusolver_int_t lddb, double *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDDgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                             cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDDgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, double *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         double *dB, cusolver_int_t lddb, double *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDSgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                             cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDSgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDHgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, double *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         double *dB, cusolver_int_t lddb, double *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDHgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                             cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDHgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDBgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, double *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         double *dB, cusolver_int_t lddb, double *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDBgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                             cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDBgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDXgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, double *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         double *dB, cusolver_int_t lddb, double *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDXgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, double *, cusolver_int_t,
                             cusolver_int_t *, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDXgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, float *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         float *dB, cusolver_int_t lddb, float *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnSSgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *, cusolver_int_t,
                             cusolver_int_t *, float *, cusolver_int_t, float *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSSgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSHgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, float *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         float *dB, cusolver_int_t lddb, float *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnSHgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *, cusolver_int_t,
                             cusolver_int_t *, float *, cusolver_int_t, float *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSHgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSBgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, float *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         float *dB, cusolver_int_t lddb, float *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnSBgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *, cusolver_int_t,
                             cusolver_int_t *, float *, cusolver_int_t, float *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSBgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSXgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n,
                                                                         cusolver_int_t nrhs, float *dA,
                                                                         cusolver_int_t ldda, cusolver_int_t *dipiv,
                                                                         float *dB, cusolver_int_t lddb, float *dX,
                                                                         cusolver_int_t lddx, void *dWorkspace,
                                                                         size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnSXgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, float *, cusolver_int_t,
                             cusolver_int_t *, float *, cusolver_int_t, float *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSXgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZZgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA,
    cusolver_int_t ldda, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZZgels");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                             cusolver_int_t, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZZgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZCgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA,
    cusolver_int_t ldda, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZCgels");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                             cusolver_int_t, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZCgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZKgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA,
    cusolver_int_t ldda, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZKgels");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                             cusolver_int_t, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZKgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZEgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA,
    cusolver_int_t ldda, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZEgels");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                             cusolver_int_t, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZEgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZYgels(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex *dA,
    cusolver_int_t ldda, cuDoubleComplex *dB, cusolver_int_t lddb, cuDoubleComplex *dX, cusolver_int_t lddx,
    void *dWorkspace, size_t lwork_bytes, cusolver_int_t *iter, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnZYgels");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, cuDoubleComplex *,
                             cusolver_int_t, cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZYgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCCgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, cuComplex *dA,
                                                              cusolver_int_t ldda, cuComplex *dB, cusolver_int_t lddb,
                                                              cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnCCgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCCgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCKgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, cuComplex *dA,
                                                              cusolver_int_t ldda, cuComplex *dB, cusolver_int_t lddb,
                                                              cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnCKgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCKgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCEgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, cuComplex *dA,
                                                              cusolver_int_t ldda, cuComplex *dB, cusolver_int_t lddb,
                                                              cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnCEgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCEgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCYgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, cuComplex *dA,
                                                              cusolver_int_t ldda, cuComplex *dB, cusolver_int_t lddb,
                                                              cuComplex *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnCYgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCYgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDDgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, double *dA,
                                                              cusolver_int_t ldda, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDDgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                                          cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDDgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDSgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, double *dA,
                                                              cusolver_int_t ldda, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDSgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                                          cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDSgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDHgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, double *dA,
                                                              cusolver_int_t ldda, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDHgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                                          cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDHgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDBgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, double *dA,
                                                              cusolver_int_t ldda, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDBgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                                          cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDBgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDXgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, double *dA,
                                                              cusolver_int_t ldda, double *dB, cusolver_int_t lddb,
                                                              double *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnDXgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                                          cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDXgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSSgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, float *dA,
                                                              cusolver_int_t ldda, float *dB, cusolver_int_t lddb,
                                                              float *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnSSgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float *,
                                          cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSSgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSHgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, float *dA,
                                                              cusolver_int_t ldda, float *dB, cusolver_int_t lddb,
                                                              float *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnSHgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float *,
                                          cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSHgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSBgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, float *dA,
                                                              cusolver_int_t ldda, float *dB, cusolver_int_t lddb,
                                                              float *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnSBgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float *,
                                          cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSBgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSXgels(cusolverDnHandle_t handle, cusolver_int_t m,
                                                              cusolver_int_t n, cusolver_int_t nrhs, float *dA,
                                                              cusolver_int_t ldda, float *dB, cusolver_int_t lddb,
                                                              float *dX, cusolver_int_t lddx, void *dWorkspace,
                                                              size_t lwork_bytes, cusolver_int_t *iter,
                                                              cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnSXgels");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float *,
                                          cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t, void *,
                                          size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSXgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZZgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuDoubleComplex *dA, cusolver_int_t ldda,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZZgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZZgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuDoubleComplex *dA, cusolver_int_t ldda,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZCgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZCgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZKgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuDoubleComplex *dA, cusolver_int_t ldda,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZKgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZKgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZEgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuDoubleComplex *dA, cusolver_int_t ldda,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZEgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZEgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZYgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuDoubleComplex *dA, cusolver_int_t ldda,
                                                                         cuDoubleComplex *dB, cusolver_int_t lddb,
                                                                         cuDoubleComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnZYgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, cuDoubleComplex *, cusolver_int_t,
                                          cuDoubleComplex *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZYgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuComplex *dA, cusolver_int_t ldda,
                                                                         cuComplex *dB, cusolver_int_t lddb,
                                                                         cuComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnCCgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCCgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCKgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuComplex *dA, cusolver_int_t ldda,
                                                                         cuComplex *dB, cusolver_int_t lddb,
                                                                         cuComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnCKgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCKgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCEgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuComplex *dA, cusolver_int_t ldda,
                                                                         cuComplex *dB, cusolver_int_t lddb,
                                                                         cuComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnCEgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCEgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCYgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m,
                                                                         cusolver_int_t n, cusolver_int_t nrhs,
                                                                         cuComplex *dA, cusolver_int_t ldda,
                                                                         cuComplex *dB, cusolver_int_t lddb,
                                                                         cuComplex *dX, cusolver_int_t lddx,
                                                                         void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnCYgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t,
                                          cuComplex *, cusolver_int_t, cuComplex *, cusolver_int_t, cuComplex *,
                                          cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCYgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDDgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
    double *dB, cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDDgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                             cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDDgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDSgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
    double *dB, cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDSgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                             cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDSgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDHgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
    double *dB, cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDHgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                             cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDHgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDBgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
    double *dB, cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDBgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                             cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDBgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDXgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double *dA, cusolver_int_t ldda,
    double *dB, cusolver_int_t lddb, double *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnDXgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, double *,
                             cusolver_int_t, double *, cusolver_int_t, double *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDXgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSSgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float *dA, cusolver_int_t ldda,
    float *dB, cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnSSgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float *,
                             cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSSgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSHgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float *dA, cusolver_int_t ldda,
    float *dB, cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnSHgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float *,
                             cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSHgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSBgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float *dA, cusolver_int_t ldda,
    float *dB, cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnSBgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float *,
                             cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSBgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSXgels_bufferSize(
    cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float *dA, cusolver_int_t ldda,
    float *dB, cusolver_int_t lddb, float *dX, cusolver_int_t lddx, void *dWorkspace, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnSXgels_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, float *,
                             cusolver_int_t, float *, cusolver_int_t, float *, cusolver_int_t, void *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSXgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSXgesv(
    cusolverDnHandle_t handle, cusolverDnIRSParams_t gesv_irs_params, cusolverDnIRSInfos_t gesv_irs_infos,
    cusolver_int_t n, cusolver_int_t nrhs, void *dA, cusolver_int_t ldda, void *dB, cusolver_int_t lddb, void *dX,
    cusolver_int_t lddx, void *dWorkspace, size_t lwork_bytes, cusolver_int_t *niters, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnIRSXgesv");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t, cusolver_int_t,
                             cusolver_int_t, void *, cusolver_int_t, void *, cusolver_int_t, void *, cusolver_int_t,
                             void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSXgesv"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                      lwork_bytes, niters, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSXgesv_bufferSize(cusolverDnHandle_t handle,
                                                                           cusolverDnIRSParams_t params,
                                                                           cusolver_int_t n, cusolver_int_t nrhs,
                                                                           size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnIRSXgesv_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSXgesv_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, n, nrhs, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverDnIRSXgels(cusolverDnHandle_t handle, cusolverDnIRSParams_t gels_irs_params,
                       cusolverDnIRSInfos_t gels_irs_infos, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs,
                       void *dA, cusolver_int_t ldda, void *dB, cusolver_int_t lddb, void *dX, cusolver_int_t lddx,
                       void *dWorkspace, size_t lwork_bytes, cusolver_int_t *niters, cusolver_int_t *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnIRSXgels");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t, cusolver_int_t,
                             cusolver_int_t, cusolver_int_t, void *, cusolver_int_t, void *, cusolver_int_t, void *,
                             cusolver_int_t, void *, size_t, cusolver_int_t *, cusolver_int_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSXgels"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, gels_irs_params, gels_irs_infos, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,
                      lwork_bytes, niters, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnIRSXgels_bufferSize(cusolverDnHandle_t handle,
                                                                           cusolverDnIRSParams_t params,
                                                                           cusolver_int_t m, cusolver_int_t n,
                                                                           cusolver_int_t nrhs, size_t *lwork_bytes) {
    HOOK_TRACE_PROFILE("cusolverDnIRSXgels_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t,
                                          cusolver_int_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnIRSXgels_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, nrhs, lwork_bytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, float *A,
                                                                         int lda, int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSpotrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSpotrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, double *A,
                                                                         int lda, int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDpotrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDpotrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, cuComplex *A,
                                                                         int lda, int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCpotrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCpotrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n,
                                                                         cuDoubleComplex *A, int lda, int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZpotrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZpotrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, float *Workspace, int Lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSpotrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSpotrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, double *Workspace, int Lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDpotrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDpotrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, cuComplex *Workspace, int Lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnCpotrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCpotrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, cuDoubleComplex *Workspace,
                                                              int Lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZpotrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZpotrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              int nrhs, const float *A, int lda, float *B, int ldb,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSpotrs");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const float *, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSpotrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              int nrhs, const double *A, int lda, double *B, int ldb,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDpotrs");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const double *, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDpotrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              int nrhs, const cuComplex *A, int lda, cuComplex *B,
                                                              int ldb, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnCpotrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const cuComplex *, int,
                                          cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCpotrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              int nrhs, const cuDoubleComplex *A, int lda,
                                                              cuDoubleComplex *B, int ldb, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZpotrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex *, int,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZpotrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                                     int n, float *Aarray, int lda, int *infoArray,
                                                                     int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnSpotrfBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSpotrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                                     int n, double *Aarray, int lda, int *infoArray,
                                                                     int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnDpotrfBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDpotrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                                     int n, cuComplex *Aarray, int lda, int *infoArray,
                                                                     int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnCpotrfBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCpotrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                                     int n, cuDoubleComplex *Aarray, int lda,
                                                                     int *infoArray, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnZpotrfBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZpotrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                                     int n, int nrhs, float *A, int lda, float *B,
                                                                     int ldb, int *d_info, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnSpotrsBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, float *, int, float *, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSpotrsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                                     int n, int nrhs, double *A, int lda, double *B,
                                                                     int ldb, int *d_info, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnDpotrsBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, double *, int, double *, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDpotrsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                                     int n, int nrhs, cuComplex *A, int lda,
                                                                     cuComplex *B, int ldb, int *d_info,
                                                                     int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnCpotrsBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, cuComplex *, int, cuComplex *,
                                          int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCpotrsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                                     int n, int nrhs, cuDoubleComplex *A, int lda,
                                                                     cuDoubleComplex *B, int ldb, int *d_info,
                                                                     int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnZpotrsBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, int, cuDoubleComplex *, int,
                                          cuDoubleComplex *, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZpotrsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSpotri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, float *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSpotri_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSpotri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDpotri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, double *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDpotri_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDpotri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCpotri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, cuComplex *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCpotri_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCpotri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZpotri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n,
                                                                         cuDoubleComplex *A, int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZpotri_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZpotri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, float *work, int lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSpotri");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSpotri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, double *work, int lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDpotri");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDpotri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, cuComplex *work, int lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnCpotri");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCpotri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, cuDoubleComplex *work,
                                                              int lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZpotri");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZpotri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXtrtri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, cublasDiagType_t diag,
                                                                         int64_t n, cudaDataType dataTypeA, void *A,
                                                                         int64_t lda, size_t *workspaceInBytesOnDevice,
                                                                         size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXtrtri_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int64_t, cudaDataType,
                                          void *, int64_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXtrtri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, diag, n, dataTypeA, A, lda, workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXtrtri(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                                              cublasDiagType_t diag, int64_t n, cudaDataType dataTypeA,
                                                              void *A, int64_t lda, void *bufferOnDevice,
                                                              size_t workspaceInBytesOnDevice, void *bufferOnHost,
                                                              size_t workspaceInBytesOnHost, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnXtrtri");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, int64_t, cudaDataType,
                                          void *, int64_t, void *, size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXtrtri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, diag, n, dataTypeA, A, lda, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                      workspaceInBytesOnHost, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSlauum_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, float *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSlauum_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSlauum_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDlauum_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, double *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDlauum_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDlauum_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnClauum_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, cuComplex *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnClauum_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnClauum_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZlauum_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n,
                                                                         cuDoubleComplex *A, int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZlauum_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZlauum_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, float *work, int lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSlauum");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSlauum"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, double *work, int lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDlauum");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDlauum"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnClauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, cuComplex *work, int lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnClauum");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnClauum"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZlauum(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, cuDoubleComplex *work,
                                                              int lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZlauum");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZlauum"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         float *A, int lda, int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSgetrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgetrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         double *A, int lda, int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDgetrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgetrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         cuComplex *A, int lda, int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCgetrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgetrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         cuDoubleComplex *A, int lda, int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZgetrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgetrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float *A,
                                                              int lda, float *Workspace, int *devIpiv, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSgetrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float *, int, float *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgetrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double *A,
                                                              int lda, double *Workspace, int *devIpiv, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDgetrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double *, int, double *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgetrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgetrf(cusolverDnHandle_t handle, int m, int n, cuComplex *A,
                                                              int lda, cuComplex *Workspace, int *devIpiv,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnCgetrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex *, int, cuComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgetrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t handle, int m, int n,
                                                              cuDoubleComplex *A, int lda, cuDoubleComplex *Workspace,
                                                              int *devIpiv, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZgetrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex *, int, cuDoubleComplex *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgetrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSlaswp(cusolverDnHandle_t handle, int n, float *A, int lda,
                                                              int k1, int k2, const int *devIpiv, int incx) {
    HOOK_TRACE_PROFILE("cusolverDnSlaswp");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, float *, int, int, int, const int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSlaswp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, k1, k2, devIpiv, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDlaswp(cusolverDnHandle_t handle, int n, double *A, int lda,
                                                              int k1, int k2, const int *devIpiv, int incx) {
    HOOK_TRACE_PROFILE("cusolverDnDlaswp");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, double *, int, int, int, const int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDlaswp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, k1, k2, devIpiv, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnClaswp(cusolverDnHandle_t handle, int n, cuComplex *A, int lda,
                                                              int k1, int k2, const int *devIpiv, int incx) {
    HOOK_TRACE_PROFILE("cusolverDnClaswp");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, cuComplex *, int, int, int, const int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnClaswp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, k1, k2, devIpiv, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZlaswp(cusolverDnHandle_t handle, int n, cuDoubleComplex *A,
                                                              int lda, int k1, int k2, const int *devIpiv, int incx) {
    HOOK_TRACE_PROFILE("cusolverDnZlaswp");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, cuDoubleComplex *, int, int, int, const int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZlaswp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, k1, k2, devIpiv, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n,
                                                              int nrhs, const float *A, int lda, const int *devIpiv,
                                                              float *B, int ldb, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSgetrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const float *, int,
                                          const int *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgetrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n,
                                                              int nrhs, const double *A, int lda, const int *devIpiv,
                                                              double *B, int ldb, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDgetrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const double *, int,
                                          const int *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgetrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n,
                                                              int nrhs, const cuComplex *A, int lda, const int *devIpiv,
                                                              cuComplex *B, int ldb, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnCgetrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const cuComplex *, int,
                                          const int *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgetrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n,
                                                              int nrhs, const cuDoubleComplex *A, int lda,
                                                              const int *devIpiv, cuDoubleComplex *B, int ldb,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZgetrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasOperation_t, int, int, const cuDoubleComplex *, int,
                                          const int *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgetrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         float *A, int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSgeqrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgeqrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         double *A, int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDgeqrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgeqrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         cuComplex *A, int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCgeqrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgeqrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         cuDoubleComplex *A, int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZgeqrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgeqrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float *A,
                                                              int lda, float *TAU, float *Workspace, int Lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSgeqrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float *, int, float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgeqrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double *A,
                                                              int lda, double *TAU, double *Workspace, int Lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDgeqrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double *, int, double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgeqrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgeqrf(cusolverDnHandle_t handle, int m, int n, cuComplex *A,
                                                              int lda, cuComplex *TAU, cuComplex *Workspace, int Lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnCgeqrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex *, int, cuComplex *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgeqrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgeqrf(cusolverDnHandle_t handle, int m, int n,
                                                              cuDoubleComplex *A, int lda, cuDoubleComplex *TAU,
                                                              cuDoubleComplex *Workspace, int Lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZgeqrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex *, int, cuDoubleComplex *,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgeqrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k,
                                                                         const float *A, int lda, const float *tau,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSorgqr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const float *, int, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSorgqr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k,
                                                                         const double *A, int lda, const double *tau,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDorgqr_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const double *, int, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDorgqr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k,
                                                                         const cuComplex *A, int lda,
                                                                         const cuComplex *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCungqr_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const cuComplex *, int, const cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCungqr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k,
                                                                         const cuDoubleComplex *A, int lda,
                                                                         const cuDoubleComplex *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZungqr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, const cuDoubleComplex *, int,
                                          const cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZungqr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n, int k, float *A,
                                                              int lda, const float *tau, float *work, int lwork,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSorgqr");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, float *, int, const float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSorgqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n, int k, double *A,
                                                              int lda, const double *tau, double *work, int lwork,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDorgqr");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, double *, int, const double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDorgqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCungqr(cusolverDnHandle_t handle, int m, int n, int k,
                                                              cuComplex *A, int lda, const cuComplex *tau,
                                                              cuComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCungqr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, cuComplex *, int, const cuComplex *,
                                          cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCungqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZungqr(cusolverDnHandle_t handle, int m, int n, int k,
                                                              cuDoubleComplex *A, int lda, const cuDoubleComplex *tau,
                                                              cuDoubleComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZungqr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int, cuDoubleComplex *, int,
                                          const cuDoubleComplex *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZungqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, k, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, cublasOperation_t trans,
                                                                         int m, int n, int k, const float *A, int lda,
                                                                         const float *tau, const float *C, int ldc,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSormqr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
                                          const float *, int, const float *, const float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSormqr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, cublasOperation_t trans,
                                                                         int m, int n, int k, const double *A, int lda,
                                                                         const double *tau, const double *C, int ldc,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDormqr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
                                          const double *, int, const double *, const double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDormqr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, cublasOperation_t trans,
                                                                         int m, int n, int k, const cuComplex *A,
                                                                         int lda, const cuComplex *tau,
                                                                         const cuComplex *C, int ldc, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCunmqr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
                                          const cuComplex *, int, const cuComplex *, const cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCunmqr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZunmqr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZunmqr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
                                          const cuDoubleComplex *, int, const cuDoubleComplex *,
                                          const cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZunmqr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                              cublasOperation_t trans, int m, int n, int k,
                                                              const float *A, int lda, const float *tau, float *C,
                                                              int ldc, float *work, int lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSormqr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
                                          const float *, int, const float *, float *, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSormqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                              cublasOperation_t trans, int m, int n, int k,
                                                              const double *A, int lda, const double *tau, double *C,
                                                              int ldc, double *work, int lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDormqr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
                                          const double *, int, const double *, double *, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDormqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCunmqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                              cublasOperation_t trans, int m, int n, int k,
                                                              const cuComplex *A, int lda, const cuComplex *tau,
                                                              cuComplex *C, int ldc, cuComplex *work, int lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnCunmqr");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int, const cuComplex *,
                             int, const cuComplex *, cuComplex *, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCunmqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZunmqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                              cublasOperation_t trans, int m, int n, int k,
                                                              const cuDoubleComplex *A, int lda,
                                                              const cuDoubleComplex *tau, cuDoubleComplex *C, int ldc,
                                                              cuDoubleComplex *work, int lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZunmqr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, int, int, int,
                                          const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZunmqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsytrf_bufferSize(cusolverDnHandle_t handle, int n, float *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSsytrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsytrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsytrf_bufferSize(cusolverDnHandle_t handle, int n, double *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDsytrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsytrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuComplex *A,
                                                                         int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCsytrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCsytrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZsytrf_bufferSize(cusolverDnHandle_t handle, int n,
                                                                         cuDoubleComplex *A, int lda, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZsytrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZsytrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, int *ipiv, float *work, int lwork,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSsytrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, int *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsytrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, int *ipiv, double *work, int lwork,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDsytrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, int *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsytrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, int *ipiv, cuComplex *work,
                                                              int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCsytrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, int *,
                                          cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCsytrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, int *ipiv,
                                                              cuDoubleComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZsytrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, int *,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZsytrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsytrs_bufferSize(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A,
    int64_t lda, const int64_t *ipiv, cudaDataType dataTypeB, void *B, int64_t ldb, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXsytrs_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int64_t, int64_t, cudaDataType, const void *,
                             int64_t, const int64_t *, cudaDataType, void *, int64_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsytrs_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, workspaceInBytesOnDevice,
                      workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsytrs(
    cusolverDnHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t nrhs, cudaDataType dataTypeA, const void *A,
    int64_t lda, const int64_t *ipiv, cudaDataType dataTypeB, void *B, int64_t ldb, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXsytrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int64_t, int64_t, cudaDataType,
                                          const void *, int64_t, const int64_t *, cudaDataType, void *, int64_t, void *,
                                          size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsytrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, bufferOnDevice,
                      workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsytri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, float *A,
                                                                         int lda, const int *ipiv, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSsytri_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, const int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsytri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsytri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, double *A,
                                                                         int lda, const int *ipiv, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDsytri_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, const int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsytri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCsytri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, cuComplex *A,
                                                                         int lda, const int *ipiv, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCsytri_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, const int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCsytri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZsytri_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n,
                                                                         cuDoubleComplex *A, int lda, const int *ipiv,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZsytri_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, const int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZsytri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, const int *ipiv, float *work,
                                                              int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSsytri");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, const int *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsytri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, const int *ipiv, double *work,
                                                              int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDsytri");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, const int *,
                                          double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsytri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, const int *ipiv, cuComplex *work,
                                                              int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCsytri");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, const int *,
                                          cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCsytri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZsytri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, const int *ipiv,
                                                              cuDoubleComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZsytri");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
                                          const int *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZsytri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSgebrd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgebrd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDgebrd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgebrd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCgebrd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgebrd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         int *Lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZgebrd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgebrd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, Lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgebrd(cusolverDnHandle_t handle, int m, int n, float *A,
                                                              int lda, float *D, float *E, float *TAUQ, float *TAUP,
                                                              float *Work, int Lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnSgebrd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, float *, int, float *, float *, float *,
                                          float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgebrd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgebrd(cusolverDnHandle_t handle, int m, int n, double *A,
                                                              int lda, double *D, double *E, double *TAUQ, double *TAUP,
                                                              double *Work, int Lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnDgebrd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double *, int, double *, double *, double *,
                                          double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgebrd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgebrd(cusolverDnHandle_t handle, int m, int n, cuComplex *A,
                                                              int lda, float *D, float *E, cuComplex *TAUQ,
                                                              cuComplex *TAUP, cuComplex *Work, int Lwork,
                                                              int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnCgebrd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuComplex *, int, float *, float *, cuComplex *,
                                          cuComplex *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgebrd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgebrd(cusolverDnHandle_t handle, int m, int n,
                                                              cuDoubleComplex *A, int lda, double *D, double *E,
                                                              cuDoubleComplex *TAUQ, cuDoubleComplex *TAUP,
                                                              cuDoubleComplex *Work, int Lwork, int *devInfo) {
    HOOK_TRACE_PROFILE("cusolverDnZgebrd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, cuDoubleComplex *, int, double *, double *,
                                          cuDoubleComplex *, cuDoubleComplex *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgebrd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSorgbr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, int m, int n, int k,
                                                                         const float *A, int lda, const float *tau,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSorgbr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const float *, int,
                                          const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSorgbr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, m, n, k, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDorgbr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, int m, int n, int k,
                                                                         const double *A, int lda, const double *tau,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDorgbr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const double *, int,
                                          const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDorgbr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, m, n, k, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCungbr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, int m, int n, int k,
                                                                         const cuComplex *A, int lda,
                                                                         const cuComplex *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCungbr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const cuComplex *, int,
                                          const cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCungbr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, m, n, k, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZungbr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, int m, int n, int k,
                                                                         const cuDoubleComplex *A, int lda,
                                                                         const cuDoubleComplex *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZungbr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, const cuDoubleComplex *,
                                          int, const cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZungbr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, m, n, k, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m,
                                                              int n, int k, float *A, int lda, const float *tau,
                                                              float *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSorgbr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, float *, int,
                                          const float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSorgbr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m,
                                                              int n, int k, double *A, int lda, const double *tau,
                                                              double *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDorgbr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, double *, int,
                                          const double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDorgbr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m,
                                                              int n, int k, cuComplex *A, int lda, const cuComplex *tau,
                                                              cuComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCungbr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, cuComplex *, int,
                                          const cuComplex *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCungbr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m,
                                                              int n, int k, cuDoubleComplex *A, int lda,
                                                              const cuDoubleComplex *tau, cuDoubleComplex *work,
                                                              int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZungbr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, int, int, int, cuDoubleComplex *, int,
                                          const cuDoubleComplex *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZungbr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsytrd_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, const float *A,
                                                                         int lda, const float *d, const float *e,
                                                                         const float *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSsytrd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const float *, int, const float *,
                                          const float *, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsytrd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, d, e, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, const double *A,
                                                                         int lda, const double *d, const double *e,
                                                                         const double *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDsytrd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const double *, int,
                                          const double *, const double *, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsytrd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, d, e, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnChetrd_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n,
                                                                         const cuComplex *A, int lda, const float *d,
                                                                         const float *e, const cuComplex *tau,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnChetrd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuComplex *, int,
                                          const float *, const float *, const cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnChetrd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, d, e, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZhetrd_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n,
                                                                         const cuDoubleComplex *A, int lda,
                                                                         const double *d, const double *e,
                                                                         const cuDoubleComplex *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZhetrd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, int,
                                          const double *, const double *, const cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZhetrd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, d, e, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, float *d, float *e, float *tau,
                                                              float *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSsytrd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, float *, float *,
                                          float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsytrd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, d, e, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, double *d, double *e, double *tau,
                                                              double *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDsytrd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, double *, double *,
                                          double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsytrd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, d, e, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnChetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, float *d, float *e, cuComplex *tau,
                                                              cuComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnChetrd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int, float *, float *,
                                          cuComplex *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnChetrd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, d, e, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZhetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, double *d, double *e,
                                                              cuDoubleComplex *tau, cuDoubleComplex *work, int lwork,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZhetrd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int, double *,
                                          double *, cuDoubleComplex *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZhetrd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, d, e, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSorgtr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, const float *A,
                                                                         int lda, const float *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSorgtr_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const float *, int, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSorgtr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDorgtr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n, const double *A,
                                                                         int lda, const double *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDorgtr_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const double *, int, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDorgtr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCungtr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n,
                                                                         const cuComplex *A, int lda,
                                                                         const cuComplex *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCungtr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuComplex *, int,
                                          const cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCungtr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZungtr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasFillMode_t uplo, int n,
                                                                         const cuDoubleComplex *A, int lda,
                                                                         const cuDoubleComplex *tau, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZungtr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, int,
                                          const cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZungtr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, tau, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, const float *tau, float *work,
                                                              int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSorgtr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, float *, int, const float *,
                                          float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSorgtr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, const double *tau, double *work,
                                                              int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDorgtr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, double *, int, const double *,
                                          double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDorgtr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, const cuComplex *tau,
                                                              cuComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCungtr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuComplex *, int,
                                          const cuComplex *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCungtr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, const cuDoubleComplex *tau,
                                                              cuDoubleComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZungtr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasFillMode_t, int, cuDoubleComplex *, int,
                                          const cuDoubleComplex *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZungtr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, tau, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSormtr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, cublasFillMode_t uplo,
                                                                         cublasOperation_t trans, int m, int n,
                                                                         const float *A, int lda, const float *tau,
                                                                         const float *C, int ldc, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSormtr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                          int, int, const float *, int, const float *, const float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSormtr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDormtr_bufferSize(cusolverDnHandle_t handle,
                                                                         cublasSideMode_t side, cublasFillMode_t uplo,
                                                                         cublasOperation_t trans, int m, int n,
                                                                         const double *A, int lda, const double *tau,
                                                                         const double *C, int ldc, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDormtr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                          int, int, const double *, int, const double *, const double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDormtr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCunmtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
    const cuComplex *A, int lda, const cuComplex *tau, const cuComplex *C, int ldc, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCunmtr_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int,
                             const cuComplex *, int, const cuComplex *, const cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCunmtr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZunmtr_bufferSize(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZunmtr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                          int, int, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                          const cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZunmtr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSormtr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                              cublasFillMode_t uplo, cublasOperation_t trans, int m,
                                                              int n, float *A, int lda, float *tau, float *C, int ldc,
                                                              float *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSormtr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                          int, int, float *, int, float *, float *, int, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSormtr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDormtr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                              cublasFillMode_t uplo, cublasOperation_t trans, int m,
                                                              int n, double *A, int lda, double *tau, double *C,
                                                              int ldc, double *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDormtr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                          int, int, double *, int, double *, double *, int, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDormtr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCunmtr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                              cublasFillMode_t uplo, cublasOperation_t trans, int m,
                                                              int n, cuComplex *A, int lda, cuComplex *tau,
                                                              cuComplex *C, int ldc, cuComplex *work, int lwork,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCunmtr");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int, int,
                             cuComplex *, int, cuComplex *, cuComplex *, int, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCunmtr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZunmtr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                              cublasFillMode_t uplo, cublasOperation_t trans, int m,
                                                              int n, cuDoubleComplex *A, int lda, cuDoubleComplex *tau,
                                                              cuDoubleComplex *C, int ldc, cuDoubleComplex *work,
                                                              int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZunmtr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                          int, int, cuDoubleComplex *, int, cuDoubleComplex *, cuDoubleComplex *, int,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZunmtr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSgesvd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgesvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDgesvd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgesvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCgesvd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgesvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZgesvd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgesvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu,
                                                              signed char jobvt, int m, int n, float *A, int lda,
                                                              float *S, float *U, int ldu, float *VT, int ldvt,
                                                              float *work, int lwork, float *rwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSgesvd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, float *, int, float *,
                                          float *, int, float *, int, float *, int, float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgesvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu,
                                                              signed char jobvt, int m, int n, double *A, int lda,
                                                              double *S, double *U, int ldu, double *VT, int ldvt,
                                                              double *work, int lwork, double *rwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDgesvd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, double *, int,
                                          double *, double *, int, double *, int, double *, int, double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgesvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu,
                                                              signed char jobvt, int m, int n, cuComplex *A, int lda,
                                                              float *S, cuComplex *U, int ldu, cuComplex *VT, int ldvt,
                                                              cuComplex *work, int lwork, float *rwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCgesvd");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, cuComplex *, int, float *,
                             cuComplex *, int, cuComplex *, int, cuComplex *, int, float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgesvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu,
                                                              signed char jobvt, int m, int n, cuDoubleComplex *A,
                                                              int lda, double *S, cuDoubleComplex *U, int ldu,
                                                              cuDoubleComplex *VT, int ldvt, cuDoubleComplex *work,
                                                              int lwork, double *rwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZgesvd");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, signed char, signed char, int, int, cuDoubleComplex *, int, double *,
                             cuDoubleComplex *, int, cuDoubleComplex *, int, cuDoubleComplex *, int, double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgesvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const float *A, int lda, const float *W,
                                                                         int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSsyevd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float *,
                                          int, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsyevd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const double *A, int lda,
                                                                         const double *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDsyevd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double *,
                                          int, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsyevd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const cuComplex *A, int lda,
                                                                         const float *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCheevd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
                                          const cuComplex *, int, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCheevd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const cuDoubleComplex *A, int lda,
                                                                         const double *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZheevd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
                                          const cuDoubleComplex *, int, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZheevd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                              cublasFillMode_t uplo, int n, float *A, int lda, float *W,
                                                              float *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSsyevd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float *, int,
                                          float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsyevd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                              cublasFillMode_t uplo, int n, double *A, int lda,
                                                              double *W, double *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDsyevd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double *, int,
                                          double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsyevd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                              cublasFillMode_t uplo, int n, cuComplex *A, int lda,
                                                              float *W, cuComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCheevd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex *,
                                          int, float *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCheevd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                              cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda,
                                                              double *W, cuDoubleComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZheevd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
                                          cuDoubleComplex *, int, double *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZheevd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n,
    const float *A, int lda, float vl, float vu, int il, int iu, int *meig, const float *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSsyevdx_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t,
                                          int, const float *, int, float, float, int, int, int *, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsyevdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n,
    const double *A, int lda, double vl, double vu, int il, int iu, int *meig, const double *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDsyevdx_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int,
                             const double *, int, double, double, int, int, int *, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsyevdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCheevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n,
    const cuComplex *A, int lda, float vl, float vu, int il, int iu, int *meig, const float *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnCheevdx_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int,
                             const cuComplex *, int, float, float, int, int, int *, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCheevdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZheevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n,
    const cuDoubleComplex *A, int lda, double vl, double vu, int il, int iu, int *meig, const double *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZheevdx_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int,
                             const cuDoubleComplex *, int, double, double, int, int, int *, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZheevdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                               cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                                                               float *A, int lda, float vl, float vu, int il, int iu,
                                                               int *meig, float *W, float *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSsyevdx");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, float *,
                             int, float, float, int, int, int *, float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsyevdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                               cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                                                               double *A, int lda, double vl, double vu, int il, int iu,
                                                               int *meig, double *W, double *work, int lwork,
                                                               int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDsyevdx");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int, double *,
                             int, double, double, int, int, int *, double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsyevdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                               cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                                                               cuComplex *A, int lda, float vl, float vu, int il,
                                                               int iu, int *meig, float *W, cuComplex *work, int lwork,
                                                               int *info) {
    HOOK_TRACE_PROFILE("cusolverDnCheevdx");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int,
                             cuComplex *, int, float, float, int, int, int *, float *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCheevdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                               cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                                                               cuDoubleComplex *A, int lda, double vl, double vu,
                                                               int il, int iu, int *meig, double *W,
                                                               cuDoubleComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZheevdx");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t,
                                          int, cuDoubleComplex *, int, double, double, int, int, int *, double *,
                                          cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZheevdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsygvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const float *A, int lda, const float *B, int ldb, float vl, float vu, int il, int iu,
    int *meig, const float *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSsygvdx_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t,
                                          cublasFillMode_t, int, const float *, int, const float *, int, float, float,
                                          int, int, int *, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsygvdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsygvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const double *A, int lda, const double *B, int ldb, double vl, double vu, int il,
    int iu, int *meig, const double *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDsygvdx_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t,
                                          cublasFillMode_t, int, const double *, int, const double *, int, double,
                                          double, int, int, int *, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsygvdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnChegvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const cuComplex *A, int lda, const cuComplex *B, int ldb, float vl, float vu, int il,
    int iu, int *meig, const float *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnChegvdx_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t,
                                          cublasFillMode_t, int, const cuComplex *, int, const cuComplex *, int, float,
                                          float, int, int, int *, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnChegvdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZhegvdx_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, double vl,
    double vu, int il, int iu, int *meig, const double *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZhegvdx_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t,
                                          cublasFillMode_t, int, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                          int, double, double, int, int, int *, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZhegvdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                               cusolverEigMode_t jobz, cusolverEigRange_t range,
                                                               cublasFillMode_t uplo, int n, float *A, int lda,
                                                               float *B, int ldb, float vl, float vu, int il, int iu,
                                                               int *meig, float *W, float *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSsygvdx");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t,
                                          cublasFillMode_t, int, float *, int, float *, int, float, float, int, int,
                                          int *, float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsygvdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                               cusolverEigMode_t jobz, cusolverEigRange_t range,
                                                               cublasFillMode_t uplo, int n, double *A, int lda,
                                                               double *B, int ldb, double vl, double vu, int il, int iu,
                                                               int *meig, double *W, double *work, int lwork,
                                                               int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDsygvdx");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t,
                                          cublasFillMode_t, int, double *, int, double *, int, double, double, int, int,
                                          int *, double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsygvdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnChegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                               cusolverEigMode_t jobz, cusolverEigRange_t range,
                                                               cublasFillMode_t uplo, int n, cuComplex *A, int lda,
                                                               cuComplex *B, int ldb, float vl, float vu, int il,
                                                               int iu, int *meig, float *W, cuComplex *work, int lwork,
                                                               int *info) {
    HOOK_TRACE_PROFILE("cusolverDnChegvdx");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t,
                                          cublasFillMode_t, int, cuComplex *, int, cuComplex *, int, float, float, int,
                                          int, int *, float *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnChegvdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZhegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                               cusolverEigMode_t jobz, cusolverEigRange_t range,
                                                               cublasFillMode_t uplo, int n, cuDoubleComplex *A,
                                                               int lda, cuDoubleComplex *B, int ldb, double vl,
                                                               double vu, int il, int iu, int *meig, double *W,
                                                               cuDoubleComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZhegvdx");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t,
                                          cublasFillMode_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int, double,
                                          double, int, int, int *, double *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZhegvdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsygvd_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigType_t itype,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const float *A, int lda, const float *B,
                                                                         int ldb, const float *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnSsygvd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, const float *, int, const float *, int, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsygvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsygvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
    const double *A, int lda, const double *B, int ldb, const double *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnDsygvd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, const double *, int, const double *, int, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsygvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnChegvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
    const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnChegvd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, const cuComplex *, int, const cuComplex *, int, const float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnChegvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZhegvd_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
    const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double *W, int *lwork) {
    HOOK_TRACE_PROFILE("cusolverDnZhegvd_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int,
                             const cuDoubleComplex *, int, const cuDoubleComplex *, int, const double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZhegvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, float *B, int ldb, float *W,
                                                              float *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSsygvd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, float *, int, float *, int, float *, float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsygvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, double *B, int ldb, double *W,
                                                              double *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnDsygvd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, double *, int, double *, int, double *, double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsygvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnChegvd(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, cuComplex *B, int ldb, float *W,
                                                              cuComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnChegvd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, cuComplex *, int, cuComplex *, int, float *, cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnChegvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZhegvd(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb,
                                                              double *W, cuDoubleComplex *work, int lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnZhegvd");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int,
                             cuDoubleComplex *, int, cuDoubleComplex *, int, double *, cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZhegvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t *info) {
    HOOK_TRACE_PROFILE("cusolverDnCreateSyevjInfo");
    using func_ptr = cusolverStatus_t (*)(syevjInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCreateSyevjInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverDnDestroySyevjInfo");
    using func_ptr = cusolverStatus_t (*)(syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDestroySyevjInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevjSetTolerance");
    using func_ptr = cusolverStatus_t (*)(syevjInfo_t, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevjSetTolerance"));
    HOOK_CHECK(func_entry);
    return func_entry(info, tolerance);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevjSetMaxSweeps");
    using func_ptr = cusolverStatus_t (*)(syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevjSetMaxSweeps"));
    HOOK_CHECK(func_entry);
    return func_entry(info, max_sweeps);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevjSetSortEig");
    using func_ptr = cusolverStatus_t (*)(syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevjSetSortEig"));
    HOOK_CHECK(func_entry);
    return func_entry(info, sort_eig);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle, syevjInfo_t info,
                                                                         double *residual) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevjGetResidual");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, syevjInfo_t, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevjGetResidual"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, residual);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle, syevjInfo_t info,
                                                                       int *executed_sweeps) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevjGetSweeps");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, syevjInfo_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevjGetSweeps"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, executed_sweeps);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *A, int lda,
    const float *W, int *lwork, syevjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnSsyevjBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float *,
                                          int, const float *, int *, syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsyevjBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A, int lda,
    const double *W, int *lwork, syevjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnDsyevjBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double *,
                                          int, const double *, int *, syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsyevjBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCheevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex *A, int lda,
    const float *W, int *lwork, syevjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnCheevjBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
                                          const cuComplex *, int, const float *, int *, syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCheevjBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZheevjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda,
    const double *W, int *lwork, syevjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnZheevjBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
                                          const cuDoubleComplex *, int, const double *, int *, syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZheevjBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                                     cublasFillMode_t uplo, int n, float *A, int lda,
                                                                     float *W, float *work, int lwork, int *info,
                                                                     syevjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnSsyevjBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float *, int,
                                          float *, float *, int, int *, syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsyevjBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                                     cublasFillMode_t uplo, int n, double *A, int lda,
                                                                     double *W, double *work, int lwork, int *info,
                                                                     syevjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnDsyevjBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double *, int,
                                          double *, double *, int, int *, syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsyevjBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                                     cublasFillMode_t uplo, int n, cuComplex *A,
                                                                     int lda, float *W, cuComplex *work, int lwork,
                                                                     int *info, syevjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnCheevjBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex *,
                                          int, float *, cuComplex *, int, int *, syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCheevjBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                                     cublasFillMode_t uplo, int n, cuDoubleComplex *A,
                                                                     int lda, double *W, cuDoubleComplex *work,
                                                                     int lwork, int *info, syevjInfo_t params,
                                                                     int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnZheevjBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuDoubleComplex *, int,
                             double *, cuDoubleComplex *, int, int *, syevjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZheevjBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const float *A, int lda, const float *W,
                                                                         int *lwork, syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnSsyevj_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const float *,
                                          int, const float *, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsyevj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const double *A, int lda,
                                                                         const double *W, int *lwork,
                                                                         syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnDsyevj_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const double *,
                                          int, const double *, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsyevj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCheevj_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const cuComplex *A, int lda,
                                                                         const float *W, int *lwork,
                                                                         syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnCheevj_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
                                          const cuComplex *, int, const float *, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCheevj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZheevj_bufferSize(cusolverDnHandle_t handle,
                                                                         cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                                         int n, const cuDoubleComplex *A, int lda,
                                                                         const double *W, int *lwork,
                                                                         syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnZheevj_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
                                          const cuDoubleComplex *, int, const double *, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZheevj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                              cublasFillMode_t uplo, int n, float *A, int lda, float *W,
                                                              float *work, int lwork, int *info, syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnSsyevj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, float *, int,
                                          float *, float *, int, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsyevj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                              cublasFillMode_t uplo, int n, double *A, int lda,
                                                              double *W, double *work, int lwork, int *info,
                                                              syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnDsyevj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, double *, int,
                                          double *, double *, int, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsyevj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                              cublasFillMode_t uplo, int n, cuComplex *A, int lda,
                                                              float *W, cuComplex *work, int lwork, int *info,
                                                              syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnCheevj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, cuComplex *,
                                          int, float *, cuComplex *, int, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCheevj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                              cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda,
                                                              double *W, cuDoubleComplex *work, int lwork, int *info,
                                                              syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnZheevj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int,
                                          cuDoubleComplex *, int, double *, cuDoubleComplex *, int, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZheevj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsygvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
    const float *A, int lda, const float *B, int ldb, const float *W, int *lwork, syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnSsygvj_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int,
                             const float *, int, const float *, int, const float *, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsygvj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsygvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
    const double *A, int lda, const double *B, int ldb, const double *W, int *lwork, syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnDsygvj_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int,
                             const double *, int, const double *, int, const double *, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsygvj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnChegvj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
    const cuComplex *A, int lda, const cuComplex *B, int ldb, const float *W, int *lwork, syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnChegvj_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int,
                             const cuComplex *, int, const cuComplex *, int, const float *, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnChegvj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverDnZhegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz,
                                cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda,
                                const cuDoubleComplex *B, int ldb, const double *W, int *lwork, syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnZhegvj_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                                          const double *, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZhegvj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                                              float *A, int lda, float *B, int ldb, float *W,
                                                              float *work, int lwork, int *info, syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnSsygvj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, float *, int, float *, int, float *, float *, int, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSsygvj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                                              double *A, int lda, double *B, int ldb, double *W,
                                                              double *work, int lwork, int *info, syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnDsygvj");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int, double *,
                             int, double *, int, double *, double *, int, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDsygvj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnChegvj(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                                              cuComplex *A, int lda, cuComplex *B, int ldb, float *W,
                                                              cuComplex *work, int lwork, int *info,
                                                              syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnChegvj");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int,
                             cuComplex *, int, cuComplex *, int, float *, cuComplex *, int, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnChegvj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZhegvj(cusolverDnHandle_t handle, cusolverEigType_t itype,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                                              cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb,
                                                              double *W, cuDoubleComplex *work, int lwork, int *info,
                                                              syevjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnZhegvj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t,
                                          int, cuDoubleComplex *, int, cuDoubleComplex *, int, double *,
                                          cuDoubleComplex *, int, int *, syevjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZhegvj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t *info) {
    HOOK_TRACE_PROFILE("cusolverDnCreateGesvdjInfo");
    using func_ptr = cusolverStatus_t (*)(gesvdjInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCreateGesvdjInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverDnDestroyGesvdjInfo");
    using func_ptr = cusolverStatus_t (*)(gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDestroyGesvdjInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdjSetTolerance");
    using func_ptr = cusolverStatus_t (*)(gesvdjInfo_t, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdjSetTolerance"));
    HOOK_CHECK(func_entry);
    return func_entry(info, tolerance);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdjSetMaxSweeps");
    using func_ptr = cusolverStatus_t (*)(gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdjSetMaxSweeps"));
    HOOK_CHECK(func_entry);
    return func_entry(info, max_sweeps);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdjSetSortEig");
    using func_ptr = cusolverStatus_t (*)(gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdjSetSortEig"));
    HOOK_CHECK(func_entry);
    return func_entry(info, sort_svd);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle, gesvdjInfo_t info,
                                                                          double *residual) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdjGetResidual");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, gesvdjInfo_t, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdjGetResidual"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, residual);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle, gesvdjInfo_t info,
                                                                        int *executed_sweeps) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdjGetSweeps");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, gesvdjInfo_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdjGetSweeps"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, executed_sweeps);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const float *A, int lda, const float *S,
    const float *U, int ldu, const float *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnSgesvdjBatched_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const float *, int, const float *,
                             const float *, int, const float *, int, int *, gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgesvdjBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const double *A, int lda, const double *S,
    const double *U, int ldu, const double *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnDgesvdjBatched_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const double *, int, const double *,
                             const double *, int, const double *, int, int *, gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgesvdjBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuComplex *A, int lda, const float *S,
    const cuComplex *U, int ldu, const cuComplex *V, int ldv, int *lwork, gesvdjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnCgesvdjBatched_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const cuComplex *, int, const float *,
                             const cuComplex *, int, const cuComplex *, int, int *, gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgesvdjBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgesvdjBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuDoubleComplex *A, int lda, const double *S,
    const cuDoubleComplex *U, int ldu, const cuDoubleComplex *V, int ldv, int *lwork, gesvdjInfo_t params,
    int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnZgesvdjBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, const cuDoubleComplex *, int,
                                          const double *, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                                          int *, gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgesvdjBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                                      int m, int n, float *A, int lda, float *S,
                                                                      float *U, int ldu, float *V, int ldv, float *work,
                                                                      int lwork, int *info, gesvdjInfo_t params,
                                                                      int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnSgesvdjBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, float *, int, float *,
                                          float *, int, float *, int, float *, int, int *, gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgesvdjBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                                      int m, int n, double *A, int lda, double *S,
                                                                      double *U, int ldu, double *V, int ldv,
                                                                      double *work, int lwork, int *info,
                                                                      gesvdjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnDgesvdjBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, double *, int, double *,
                                          double *, int, double *, int, double *, int, int *, gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgesvdjBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                                      int m, int n, cuComplex *A, int lda, float *S,
                                                                      cuComplex *U, int ldu, cuComplex *V, int ldv,
                                                                      cuComplex *work, int lwork, int *info,
                                                                      gesvdjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnCgesvdjBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, cuComplex *, int, float *, cuComplex *,
                             int, cuComplex *, int, cuComplex *, int, int *, gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgesvdjBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                                      int m, int n, cuDoubleComplex *A, int lda,
                                                                      double *S, cuDoubleComplex *U, int ldu,
                                                                      cuDoubleComplex *V, int ldv,
                                                                      cuDoubleComplex *work, int lwork, int *info,
                                                                      gesvdjInfo_t params, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnZgesvdjBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, cuDoubleComplex *, int,
                                          double *, cuDoubleComplex *, int, cuDoubleComplex *, int, cuDoubleComplex *,
                                          int, int *, gesvdjInfo_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgesvdjBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const float *A, int lda, const float *S,
    const float *U, int ldu, const float *V, int ldv, int *lwork, gesvdjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnSgesvdj_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float *, int,
                                          const float *, const float *, int, const float *, int, int *, gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgesvdj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const double *A, int lda,
    const double *S, const double *U, int ldu, const double *V, int ldv, int *lwork, gesvdjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnDgesvdj_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double *, int, const double *,
                             const double *, int, const double *, int, int *, gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgesvdj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgesvdj_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuComplex *A, int lda,
    const float *S, const cuComplex *U, int ldu, const cuComplex *V, int ldv, int *lwork, gesvdjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnCgesvdj_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex *, int,
                             const float *, const cuComplex *, int, const cuComplex *, int, int *, gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgesvdj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t handle,
                                                                          cusolverEigMode_t jobz, int econ, int m,
                                                                          int n, const cuDoubleComplex *A, int lda,
                                                                          const double *S, const cuDoubleComplex *U,
                                                                          int ldu, const cuDoubleComplex *V, int ldv,
                                                                          int *lwork, gesvdjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnZgesvdj_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex *,
                                          int, const double *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                          int, int *, gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgesvdj_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                               int econ, int m, int n, float *A, int lda, float *S,
                                                               float *U, int ldu, float *V, int ldv, float *work,
                                                               int lwork, int *info, gesvdjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnSgesvdj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, float *, int, float *,
                                          float *, int, float *, int, float *, int, int *, gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgesvdj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                               int econ, int m, int n, double *A, int lda, double *S,
                                                               double *U, int ldu, double *V, int ldv, double *work,
                                                               int lwork, int *info, gesvdjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnDgesvdj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, double *, int, double *,
                                          double *, int, double *, int, double *, int, int *, gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgesvdj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                               int econ, int m, int n, cuComplex *A, int lda, float *S,
                                                               cuComplex *U, int ldu, cuComplex *V, int ldv,
                                                               cuComplex *work, int lwork, int *info,
                                                               gesvdjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnCgesvdj");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, cuComplex *, int, float *,
                             cuComplex *, int, cuComplex *, int, cuComplex *, int, int *, gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgesvdj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                                               int econ, int m, int n, cuDoubleComplex *A, int lda,
                                                               double *S, cuDoubleComplex *U, int ldu,
                                                               cuDoubleComplex *V, int ldv, cuDoubleComplex *work,
                                                               int lwork, int *info, gesvdjInfo_t params) {
    HOOK_TRACE_PROFILE("cusolverDnZgesvdj");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, cuDoubleComplex *, int,
                                          double *, cuDoubleComplex *, int, cuDoubleComplex *, int, cuDoubleComplex *,
                                          int, int *, gesvdjInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgesvdj"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float *d_A, int lda,
    long long int strideA, const float *d_S, long long int strideS, const float *d_U, int ldu, long long int strideU,
    const float *d_V, int ldv, long long int strideV, int *lwork, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnSgesvdaStridedBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float *, int,
                                          long long int, const float *, long long int, const float *, int,
                                          long long int, const float *, int, long long int, int *, int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgesvdaStridedBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                      lwork, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double *d_A, int lda,
    long long int strideA, const double *d_S, long long int strideS, const double *d_U, int ldu, long long int strideU,
    const double *d_V, int ldv, long long int strideV, int *lwork, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnDgesvdaStridedBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double *, int,
                                          long long int, const double *, long long int, const double *, int,
                                          long long int, const double *, int, long long int, int *, int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgesvdaStridedBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                      lwork, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex *d_A, int lda,
    long long int strideA, const float *d_S, long long int strideS, const cuComplex *d_U, int ldu,
    long long int strideU, const cuComplex *d_V, int ldv, long long int strideV, int *lwork, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnCgesvdaStridedBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex *, int,
                                          long long int, const float *, long long int, const cuComplex *, int,
                                          long long int, const cuComplex *, int, long long int, int *, int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgesvdaStridedBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                      lwork, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgesvdaStridedBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex *d_A, int lda,
    long long int strideA, const double *d_S, long long int strideS, const cuDoubleComplex *d_U, int ldu,
    long long int strideU, const cuDoubleComplex *d_V, int ldv, long long int strideV, int *lwork, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnZgesvdaStridedBatched_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex *,
                                          int, long long int, const double *, long long int, const cuDoubleComplex *,
                                          int, long long int, const cuDoubleComplex *, int, long long int, int *, int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgesvdaStridedBatched_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                      lwork, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float *d_A, int lda,
    long long int strideA, float *d_S, long long int strideS, float *d_U, int ldu, long long int strideU, float *d_V,
    int ldv, long long int strideV, float *d_work, int lwork, int *d_info, double *h_R_nrmF, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnSgesvdaStridedBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const float *, int,
                                          long long int, float *, long long int, float *, int, long long int, float *,
                                          int, long long int, float *, int, int *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSgesvdaStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                      d_work, lwork, d_info, h_R_nrmF, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double *d_A, int lda,
    long long int strideA, double *d_S, long long int strideS, double *d_U, int ldu, long long int strideU, double *d_V,
    int ldv, long long int strideV, double *d_work, int lwork, int *d_info, double *h_R_nrmF, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnDgesvdaStridedBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const double *, int,
                                          long long int, double *, long long int, double *, int, long long int,
                                          double *, int, long long int, double *, int, int *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDgesvdaStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                      d_work, lwork, d_info, h_R_nrmF, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex *d_A, int lda,
    long long int strideA, float *d_S, long long int strideS, cuComplex *d_U, int ldu, long long int strideU,
    cuComplex *d_V, int ldv, long long int strideV, cuComplex *d_work, int lwork, int *d_info, double *h_R_nrmF,
    int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnCgesvdaStridedBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuComplex *, int,
                                          long long int, float *, long long int, cuComplex *, int, long long int,
                                          cuComplex *, int, long long int, cuComplex *, int, int *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCgesvdaStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                      d_work, lwork, d_info, h_R_nrmF, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnZgesvdaStridedBatched(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex *d_A, int lda,
    long long int strideA, double *d_S, long long int strideS, cuDoubleComplex *d_U, int ldu, long long int strideU,
    cuDoubleComplex *d_V, int ldv, long long int strideV, cuDoubleComplex *d_work, int lwork, int *d_info,
    double *h_R_nrmF, int batchSize) {
    HOOK_TRACE_PROFILE("cusolverDnZgesvdaStridedBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverEigMode_t, int, int, int, const cuDoubleComplex *, int,
                             long long int, double *, long long int, cuDoubleComplex *, int, long long int,
                             cuDoubleComplex *, int, long long int, cuDoubleComplex *, int, int *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnZgesvdaStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV,
                      d_work, lwork, d_info, h_R_nrmF, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t *params) {
    HOOK_TRACE_PROFILE("cusolverDnCreateParams");
    using func_ptr = cusolverStatus_t (*)(cusolverDnParams_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnCreateParams"));
    HOOK_CHECK(func_entry);
    return func_entry(params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params) {
    HOOK_TRACE_PROFILE("cusolverDnDestroyParams");
    using func_ptr = cusolverStatus_t (*)(cusolverDnParams_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnDestroyParams"));
    HOOK_CHECK(func_entry);
    return func_entry(params);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSetAdvOptions(cusolverDnParams_t params,
                                                                     cusolverDnFunction_t function,
                                                                     cusolverAlgMode_t algo) {
    HOOK_TRACE_PROFILE("cusolverDnSetAdvOptions");
    using func_ptr = cusolverStatus_t (*)(cusolverDnParams_t, cusolverDnFunction_t, cusolverAlgMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSetAdvOptions"));
    HOOK_CHECK(func_entry);
    return func_entry(params, function, algo);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnPotrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
    const void *A, int64_t lda, cudaDataType computeType, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverDnPotrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
                                          cudaDataType, const void *, int64_t, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnPotrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnPotrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                             cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
                                                             void *A, int64_t lda, cudaDataType computeType,
                                                             void *pBuffer, size_t workspaceInBytes, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnPotrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
                                          cudaDataType, void *, int64_t, cudaDataType, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnPotrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, uplo, n, dataTypeA, A, lda, computeType, pBuffer, workspaceInBytes, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnPotrs(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                             cublasFillMode_t uplo, int64_t n, int64_t nrhs,
                                                             cudaDataType dataTypeA, const void *A, int64_t lda,
                                                             cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnPotrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, int64_t,
                                          cudaDataType, const void *, int64_t, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnPotrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnGeqrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType dataTypeTau, const void *tau, cudaDataType computeType, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverDnGeqrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
                                          const void *, int64_t, cudaDataType, const void *, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnGeqrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnGeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                             int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
                                                             int64_t lda, cudaDataType dataTypeTau, void *tau,
                                                             cudaDataType computeType, void *pBuffer,
                                                             size_t workspaceInBytes, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnGeqrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
                                          void *, int64_t, cudaDataType, void *, cudaDataType, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnGeqrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, pBuffer, workspaceInBytes,
                      info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnGetrf_bufferSize(cusolverDnHandle_t handle,
                                                                        cusolverDnParams_t params, int64_t m, int64_t n,
                                                                        cudaDataType dataTypeA, const void *A,
                                                                        int64_t lda, cudaDataType computeType,
                                                                        size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverDnGetrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
                                          const void *, int64_t, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnGetrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnGetrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                             int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
                                                             int64_t lda, int64_t *ipiv, cudaDataType computeType,
                                                             void *pBuffer, size_t workspaceInBytes, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnGetrf");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
                                          void *, int64_t, int64_t *, cudaDataType, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnGetrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, pBuffer, workspaceInBytes, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnGetrs(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                             cublasOperation_t trans, int64_t n, int64_t nrhs,
                                                             cudaDataType dataTypeA, const void *A, int64_t lda,
                                                             const int64_t *ipiv, cudaDataType dataTypeB, void *B,
                                                             int64_t ldb, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnGetrs");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, int64_t, int64_t, cudaDataType,
                             const void *, int64_t, const int64_t *, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnGetrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW, const void *W, cudaDataType computeType,
    size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverDnSyevd_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t,
                             cudaDataType, const void *, int64_t, cudaDataType, const void *, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSyevd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSyevd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                             cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                                                             cudaDataType dataTypeA, void *A, int64_t lda,
                                                             cudaDataType dataTypeW, void *W, cudaDataType computeType,
                                                             void *pBuffer, size_t workspaceInBytes, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSyevd");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, int64_t,
                             cudaDataType, void *, int64_t, cudaDataType, void *, cudaDataType, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSyevd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, pBuffer,
                      workspaceInBytes, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, void *vl, void *vu,
    int64_t il, int64_t iu, int64_t *h_meig, cudaDataType dataTypeW, const void *W, cudaDataType computeType,
    size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverDnSyevdx_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t,
                             cublasFillMode_t, int64_t, cudaDataType, const void *, int64_t, void *, void *, int64_t,
                             int64_t, int64_t *, cudaDataType, const void *, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSyevdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W,
                      computeType, workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnSyevdx(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                              cusolverEigMode_t jobz, cusolverEigRange_t range,
                                                              cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
                                                              void *A, int64_t lda, void *vl, void *vu, int64_t il,
                                                              int64_t iu, int64_t *meig64, cudaDataType dataTypeW,
                                                              void *W, cudaDataType computeType, void *pBuffer,
                                                              size_t workspaceInBytes, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnSyevdx");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t,
                             cublasFillMode_t, int64_t, cudaDataType, void *, int64_t, void *, void *, int64_t, int64_t,
                             int64_t *, cudaDataType, void *, cudaDataType, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnSyevdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W,
                      computeType, pBuffer, workspaceInBytes, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnGesvd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeS, const void *S, cudaDataType dataTypeU,
    const void *U, int64_t ldu, cudaDataType dataTypeVT, const void *VT, int64_t ldvt, cudaDataType computeType,
    size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverDnGesvd_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t,
                             cudaDataType, const void *, int64_t, cudaDataType, const void *, cudaDataType,
                             const void *, int64_t, cudaDataType, const void *, int64_t, cudaDataType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnGesvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT,
                      VT, ldvt, computeType, workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnGesvd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                             signed char jobu, signed char jobvt, int64_t m, int64_t n,
                                                             cudaDataType dataTypeA, void *A, int64_t lda,
                                                             cudaDataType dataTypeS, void *S, cudaDataType dataTypeU,
                                                             void *U, int64_t ldu, cudaDataType dataTypeVT, void *VT,
                                                             int64_t ldvt, cudaDataType computeType, void *pBuffer,
                                                             size_t workspaceInBytes, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnGesvd");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t,
                             cudaDataType, void *, int64_t, cudaDataType, void *, cudaDataType, void *, int64_t,
                             cudaDataType, void *, int64_t, cudaDataType, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnGesvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT,
                      VT, ldvt, computeType, pBuffer, workspaceInBytes, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverDnXpotrf_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, cublasFillMode_t uplo, int64_t n,
                                cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType computeType,
                                size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXpotrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t,
                                          cudaDataType, const void *, int64_t, cudaDataType, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXpotrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice,
                      workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXpotrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                              cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA,
                                                              void *A, int64_t lda, cudaDataType computeType,
                                                              void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                                                              void *bufferOnHost, size_t workspaceInBytesOnHost,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXpotrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, cudaDataType, void *,
                             int64_t, cudaDataType, void *, size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXpotrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice,
                      bufferOnHost, workspaceInBytesOnHost, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXpotrs(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                              cublasFillMode_t uplo, int64_t n, int64_t nrhs,
                                                              cudaDataType dataTypeA, const void *A, int64_t lda,
                                                              cudaDataType dataTypeB, void *B, int64_t ldb, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXpotrs");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, int64_t, int64_t,
                                          cudaDataType, const void *, int64_t, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXpotrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgeqrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType dataTypeTau, const void *tau, cudaDataType computeType, size_t *workspaceInBytesOnDevice,
    size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXgeqrf_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, const void *,
                             int64_t, cudaDataType, const void *, cudaDataType, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgeqrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytesOnDevice,
                      workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgeqrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                              int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
                                                              int64_t lda, cudaDataType dataTypeTau, void *tau,
                                                              cudaDataType computeType, void *bufferOnDevice,
                                                              size_t workspaceInBytesOnDevice, void *bufferOnHost,
                                                              size_t workspaceInBytesOnHost, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXgeqrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, void *, int64_t,
                             cudaDataType, void *, cudaDataType, void *, size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgeqrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, bufferOnDevice,
                      workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, int64_t m, int64_t n, cudaDataType dataTypeA, const void *A,
    int64_t lda, cudaDataType computeType, size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXgetrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType,
                                          const void *, int64_t, cudaDataType, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgetrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice,
                      workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgetrf(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                              int64_t m, int64_t n, cudaDataType dataTypeA, void *A,
                                                              int64_t lda, int64_t *ipiv, cudaDataType computeType,
                                                              void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                                                              void *bufferOnHost, size_t workspaceInBytesOnHost,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXgetrf");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int64_t, int64_t, cudaDataType, void *, int64_t,
                             int64_t *, cudaDataType, void *, size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgetrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, bufferOnDevice,
                      workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                              cublasOperation_t trans, int64_t n, int64_t nrhs,
                                                              cudaDataType dataTypeA, const void *A, int64_t lda,
                                                              const int64_t *ipiv, cudaDataType dataTypeB, void *B,
                                                              int64_t ldb, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXgetrs");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, int64_t, int64_t, cudaDataType,
                             const void *, int64_t, const int64_t *, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgetrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeW, const void *W, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t,
                                          int64_t, cudaDataType, const void *, int64_t, cudaDataType, const void *,
                                          cudaDataType, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType,
                      workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                                              cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                                                              cudaDataType dataTypeA, void *A, int64_t lda,
                                                              cudaDataType dataTypeW, void *W, cudaDataType computeType,
                                                              void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                                                              void *bufferOnHost, size_t workspaceInBytesOnHost,
                                                              int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevd");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t,
                                          int64_t, cudaDataType, void *, int64_t, cudaDataType, void *, cudaDataType,
                                          void *, size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice,
                      workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevdx_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void *A, int64_t lda, void *vl, void *vu,
    int64_t il, int64_t iu, int64_t *h_meig, cudaDataType dataTypeW, const void *W, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevdx_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t,
                             cublasFillMode_t, int64_t, cudaDataType, const void *, int64_t, void *, void *, int64_t,
                             int64_t, int64_t *, cudaDataType, const void *, cudaDataType, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevdx_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W,
                      computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXsyevdx(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, cusolverEigRange_t range,
    cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, void *vl, void *vu, int64_t il,
    int64_t iu, int64_t *meig64, cudaDataType dataTypeW, void *W, cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXsyevdx");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t,
                             cublasFillMode_t, int64_t, cudaDataType, void *, int64_t, void *, void *, int64_t, int64_t,
                             int64_t *, cudaDataType, void *, cudaDataType, void *, size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXsyevdx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W,
                      computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost,
                      info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvd_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeS, const void *S, cudaDataType dataTypeU,
    const void *U, int64_t ldu, cudaDataType dataTypeVT, const void *VT, int64_t ldvt, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
                                          int64_t, cudaDataType, const void *, int64_t, cudaDataType, const void *,
                                          cudaDataType, const void *, int64_t, cudaDataType, const void *, int64_t,
                                          cudaDataType, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT,
                      VT, ldvt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvd(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobvt, int64_t m, int64_t n,
    cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeS, void *S, cudaDataType dataTypeU, void *U,
    int64_t ldu, cudaDataType dataTypeVT, void *VT, int64_t ldvt, cudaDataType computeType, void *bufferOnDevice,
    size_t workspaceInBytesOnDevice, void *bufferOnHost, size_t workspaceInBytesOnHost, int *info) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvd");
    using func_ptr =
        cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t, int64_t,
                             cudaDataType, void *, int64_t, cudaDataType, void *, cudaDataType, void *, int64_t,
                             cudaDataType, void *, int64_t, cudaDataType, void *, size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT,
                      VT, ldvt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                      workspaceInBytesOnHost, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvdp_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, int econ, int64_t m, int64_t n,
    cudaDataType dataTypeA, const void *A, int64_t lda, cudaDataType dataTypeS, const void *S, cudaDataType dataTypeU,
    const void *U, int64_t ldu, cudaDataType dataTypeV, const void *V, int64_t ldv, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdp_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, int, int64_t,
                                          int64_t, cudaDataType, const void *, int64_t, cudaDataType, const void *,
                                          cudaDataType, const void *, int64_t, cudaDataType, const void *, int64_t,
                                          cudaDataType, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdp_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV,
                      V, ldv, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverDnXgesvdp(cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz, int econ, int64_t m,
                      int64_t n, cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeS, void *S,
                      cudaDataType dataTypeU, void *U, int64_t ldu, cudaDataType dataTypeV, void *V, int64_t ldv,
                      cudaDataType computeType, void *bufferOnDevice, size_t workspaceInBytesOnDevice,
                      void *bufferOnHost, size_t workspaceInBytesOnHost, int *d_info, double *h_err_sigma) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdp");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, int, int64_t,
                                          int64_t, cudaDataType, void *, int64_t, cudaDataType, void *, cudaDataType,
                                          void *, int64_t, cudaDataType, void *, int64_t, cudaDataType, void *, size_t,
                                          void *, size_t, int *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV,
                      V, ldv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost,
                      workspaceInBytesOnHost, d_info, h_err_sigma);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvdr_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobv, int64_t m, int64_t n,
    int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA, const void *A, int64_t lda,
    cudaDataType dataTypeSrand, const void *Srand, cudaDataType dataTypeUrand, const void *Urand, int64_t ldUrand,
    cudaDataType dataTypeVrand, const void *Vrand, int64_t ldVrand, cudaDataType computeType,
    size_t *workspaceInBytesOnDevice, size_t *workspaceInBytesOnHost) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdr_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
                                          int64_t, int64_t, int64_t, int64_t, cudaDataType, const void *, int64_t,
                                          cudaDataType, const void *, cudaDataType, const void *, int64_t, cudaDataType,
                                          const void *, int64_t, cudaDataType, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdr_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand,
                      dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType,
                      workspaceInBytesOnDevice, workspaceInBytesOnHost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverDnXgesvdr(
    cusolverDnHandle_t handle, cusolverDnParams_t params, signed char jobu, signed char jobv, int64_t m, int64_t n,
    int64_t k, int64_t p, int64_t niters, cudaDataType dataTypeA, void *A, int64_t lda, cudaDataType dataTypeSrand,
    void *Srand, cudaDataType dataTypeUrand, void *Urand, int64_t ldUrand, cudaDataType dataTypeVrand, void *Vrand,
    int64_t ldVrand, cudaDataType computeType, void *bufferOnDevice, size_t workspaceInBytesOnDevice,
    void *bufferOnHost, size_t workspaceInBytesOnHost, int *d_info) {
    HOOK_TRACE_PROFILE("cusolverDnXgesvdr");
    using func_ptr = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, signed char, signed char, int64_t,
                                          int64_t, int64_t, int64_t, int64_t, cudaDataType, void *, int64_t,
                                          cudaDataType, void *, cudaDataType, void *, int64_t, cudaDataType, void *,
                                          int64_t, cudaDataType, void *, size_t, void *, size_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverDnXgesvdr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand,
                      dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice,
                      workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgCreate(cusolverMgHandle_t *handle) {
    HOOK_TRACE_PROFILE("cusolverMgCreate");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgDestroy(cusolverMgHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverMgDestroy");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgDeviceSelect(cusolverMgHandle_t handle, int nbDevices,
                                                                    int deviceId) {
    HOOK_TRACE_PROFILE("cusolverMgDeviceSelect");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgDeviceSelect"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nbDevices, deviceId);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgCreateDeviceGrid(cudaLibMgGrid_t *grid, int32_t numRowDevices,
                                                                        int32_t numColDevices, const int32_t deviceId,
                                                                        cusolverMgGridMapping_t mapping) {
    HOOK_TRACE_PROFILE("cusolverMgCreateDeviceGrid");
    using func_ptr = cusolverStatus_t (*)(cudaLibMgGrid_t *, int32_t, int32_t, const int32_t, cusolverMgGridMapping_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgCreateDeviceGrid"));
    HOOK_CHECK(func_entry);
    return func_entry(grid, numRowDevices, numColDevices, deviceId, mapping);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgDestroyGrid(cudaLibMgGrid_t grid) {
    HOOK_TRACE_PROFILE("cusolverMgDestroyGrid");
    using func_ptr = cusolverStatus_t (*)(cudaLibMgGrid_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgDestroyGrid"));
    HOOK_CHECK(func_entry);
    return func_entry(grid);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgCreateMatrixDesc(cudaLibMgMatrixDesc_t *desc, int64_t numRows,
                                                                        int64_t numCols, int64_t rowBlockSize,
                                                                        int64_t colBlockSize, cudaDataType dataType,
                                                                        const cudaLibMgGrid_t grid) {
    HOOK_TRACE_PROFILE("cusolverMgCreateMatrixDesc");
    using func_ptr = cusolverStatus_t (*)(cudaLibMgMatrixDesc_t *, int64_t, int64_t, int64_t, int64_t, cudaDataType,
                                          const cudaLibMgGrid_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgCreateMatrixDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(desc, numRows, numCols, rowBlockSize, colBlockSize, dataType, grid);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgDestroyMatrixDesc(cudaLibMgMatrixDesc_t desc) {
    HOOK_TRACE_PROFILE("cusolverMgDestroyMatrixDesc");
    using func_ptr = cusolverStatus_t (*)(cudaLibMgMatrixDesc_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgDestroyMatrixDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(desc);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgSyevd_bufferSize(
    cusolverMgHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int N, void *array_d_A, int IA, int JA,
    cudaLibMgMatrixDesc_t descrA, void *W, cudaDataType dataTypeW, cudaDataType computeType, int64_t *lwork) {
    HOOK_TRACE_PROFILE("cusolverMgSyevd_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t, cusolverEigMode_t, cublasFillMode_t, int, void *, int,
                                          int, cudaLibMgMatrixDesc_t, void *, cudaDataType, cudaDataType, int64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgSyevd_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgSyevd(cusolverMgHandle_t handle, cusolverEigMode_t jobz,
                                                             cublasFillMode_t uplo, int N, void *array_d_A, int IA,
                                                             int JA, cudaLibMgMatrixDesc_t descrA, void *W,
                                                             cudaDataType dataTypeW, cudaDataType computeType,
                                                             void *array_d_work, int64_t lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverMgSyevd");
    using func_ptr =
        cusolverStatus_t (*)(cusolverMgHandle_t, cusolverEigMode_t, cublasFillMode_t, int, void *, int, int,
                             cudaLibMgMatrixDesc_t, void *, cudaDataType, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgSyevd"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, array_d_work, lwork,
                      info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgGetrf_bufferSize(cusolverMgHandle_t handle, int M, int N,
                                                                        void *array_d_A, int IA, int JA,
                                                                        cudaLibMgMatrixDesc_t descrA, int *array_d_IPIV,
                                                                        cudaDataType computeType, int64_t *lwork) {
    HOOK_TRACE_PROFILE("cusolverMgGetrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t, int, int, void *, int, int, cudaLibMgMatrixDesc_t, int *,
                                          cudaDataType, int64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgGetrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgGetrf(cusolverMgHandle_t handle, int M, int N, void *array_d_A,
                                                             int IA, int JA, cudaLibMgMatrixDesc_t descrA,
                                                             int *array_d_IPIV, cudaDataType computeType,
                                                             void *array_d_work, int64_t lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverMgGetrf");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t, int, int, void *, int, int, cudaLibMgMatrixDesc_t, int *,
                                          cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgGetrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, array_d_work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverMgGetrs_bufferSize(cusolverMgHandle_t handle, cublasOperation_t TRANS, int N, int NRHS, void *array_d_A,
                               int IA, int JA, cudaLibMgMatrixDesc_t descrA, int *array_d_IPIV, void *array_d_B, int IB,
                               int JB, cudaLibMgMatrixDesc_t descrB, cudaDataType computeType, int64_t *lwork) {
    HOOK_TRACE_PROFILE("cusolverMgGetrs_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverMgHandle_t, cublasOperation_t, int, int, void *, int, int, cudaLibMgMatrixDesc_t,
                             int *, void *, int, int, cudaLibMgMatrixDesc_t, cudaDataType, int64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgGetrs_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB,
                      computeType, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgGetrs(cusolverMgHandle_t handle, cublasOperation_t TRANS, int N,
                                                             int NRHS, void *array_d_A, int IA, int JA,
                                                             cudaLibMgMatrixDesc_t descrA, int *array_d_IPIV,
                                                             void *array_d_B, int IB, int JB,
                                                             cudaLibMgMatrixDesc_t descrB, cudaDataType computeType,
                                                             void *array_d_work, int64_t lwork, int *info) {
    HOOK_TRACE_PROFILE("cusolverMgGetrs");
    using func_ptr =
        cusolverStatus_t (*)(cusolverMgHandle_t, cublasOperation_t, int, int, void *, int, int, cudaLibMgMatrixDesc_t,
                             int *, void *, int, int, cudaLibMgMatrixDesc_t, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgGetrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB,
                      computeType, array_d_work, lwork, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgPotrf_bufferSize(cusolverMgHandle_t handle,
                                                                        cublasFillMode_t uplo, int N, void *array_d_A,
                                                                        int IA, int JA, cudaLibMgMatrixDesc_t descrA,
                                                                        cudaDataType computeType, int64_t *lwork) {
    HOOK_TRACE_PROFILE("cusolverMgPotrf_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t, cublasFillMode_t, int, void *, int, int,
                                          cudaLibMgMatrixDesc_t, cudaDataType, int64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgPotrf_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgPotrf(cusolverMgHandle_t handle, cublasFillMode_t uplo, int N,
                                                             void *array_d_A, int IA, int JA,
                                                             cudaLibMgMatrixDesc_t descrA, cudaDataType computeType,
                                                             void *array_d_work, int64_t lwork, int *h_info) {
    HOOK_TRACE_PROFILE("cusolverMgPotrf");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t, cublasFillMode_t, int, void *, int, int,
                                          cudaLibMgMatrixDesc_t, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgPotrf"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgPotrs_bufferSize(cusolverMgHandle_t handle,
                                                                        cublasFillMode_t uplo, int n, int nrhs,
                                                                        void *array_d_A, int IA, int JA,
                                                                        cudaLibMgMatrixDesc_t descrA, void *array_d_B,
                                                                        int IB, int JB, cudaLibMgMatrixDesc_t descrB,
                                                                        cudaDataType computeType, int64_t *lwork) {
    HOOK_TRACE_PROFILE("cusolverMgPotrs_bufferSize");
    using func_ptr =
        cusolverStatus_t (*)(cusolverMgHandle_t, cublasFillMode_t, int, int, void *, int, int, cudaLibMgMatrixDesc_t,
                             void *, int, int, cudaLibMgMatrixDesc_t, cudaDataType, int64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgPotrs_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgPotrs(cusolverMgHandle_t handle, cublasFillMode_t uplo, int n,
                                                             int nrhs, void *array_d_A, int IA, int JA,
                                                             cudaLibMgMatrixDesc_t descrA, void *array_d_B, int IB,
                                                             int JB, cudaLibMgMatrixDesc_t descrB,
                                                             cudaDataType computeType, void *array_d_work,
                                                             int64_t lwork, int *h_info) {
    HOOK_TRACE_PROFILE("cusolverMgPotrs");
    using func_ptr =
        cusolverStatus_t (*)(cusolverMgHandle_t, cublasFillMode_t, int, int, void *, int, int, cudaLibMgMatrixDesc_t,
                             void *, int, int, cudaLibMgMatrixDesc_t, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgPotrs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType,
                      array_d_work, lwork, h_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgPotri_bufferSize(cusolverMgHandle_t handle,
                                                                        cublasFillMode_t uplo, int N, void *array_d_A,
                                                                        int IA, int JA, cudaLibMgMatrixDesc_t descrA,
                                                                        cudaDataType computeType, int64_t *lwork) {
    HOOK_TRACE_PROFILE("cusolverMgPotri_bufferSize");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t, cublasFillMode_t, int, void *, int, int,
                                          cudaLibMgMatrixDesc_t, cudaDataType, int64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgPotri_bufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverMgPotri(cusolverMgHandle_t handle, cublasFillMode_t uplo, int N,
                                                             void *array_d_A, int IA, int JA,
                                                             cudaLibMgMatrixDesc_t descrA, cudaDataType computeType,
                                                             void *array_d_work, int64_t lwork, int *h_info) {
    HOOK_TRACE_PROFILE("cusolverMgPotri");
    using func_ptr = cusolverStatus_t (*)(cusolverMgHandle_t, cublasFillMode_t, int, void *, int, int,
                                          cudaLibMgMatrixDesc_t, cudaDataType, void *, int64_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverMgPotri"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfCreate(cusolverRfHandle_t *handle) {
    HOOK_TRACE_PROFILE("cusolverRfCreate");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfDestroy(cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfDestroy");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfGetMatrixFormat(cusolverRfHandle_t handle,
                                                                       cusolverRfMatrixFormat_t *format,
                                                                       cusolverRfUnitDiagonal_t *diag) {
    HOOK_TRACE_PROFILE("cusolverRfGetMatrixFormat");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, cusolverRfMatrixFormat_t *, cusolverRfUnitDiagonal_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfGetMatrixFormat"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, format, diag);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfSetMatrixFormat(cusolverRfHandle_t handle,
                                                                       cusolverRfMatrixFormat_t format,
                                                                       cusolverRfUnitDiagonal_t diag) {
    HOOK_TRACE_PROFILE("cusolverRfSetMatrixFormat");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, cusolverRfMatrixFormat_t, cusolverRfUnitDiagonal_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfSetMatrixFormat"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, format, diag);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfSetNumericProperties(cusolverRfHandle_t handle, double zero,
                                                                            double boost) {
    HOOK_TRACE_PROFILE("cusolverRfSetNumericProperties");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, double, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfSetNumericProperties"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, zero, boost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfGetNumericProperties(cusolverRfHandle_t handle, double *zero,
                                                                            double *boost) {
    HOOK_TRACE_PROFILE("cusolverRfGetNumericProperties");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfGetNumericProperties"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, zero, boost);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfGetNumericBoostReport(cusolverRfHandle_t handle,
                                                                             cusolverRfNumericBoostReport_t *report) {
    HOOK_TRACE_PROFILE("cusolverRfGetNumericBoostReport");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, cusolverRfNumericBoostReport_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfGetNumericBoostReport"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, report);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfSetAlgs(cusolverRfHandle_t handle,
                                                               cusolverRfFactorization_t factAlg,
                                                               cusolverRfTriangularSolve_t solveAlg) {
    HOOK_TRACE_PROFILE("cusolverRfSetAlgs");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, cusolverRfFactorization_t, cusolverRfTriangularSolve_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfSetAlgs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, factAlg, solveAlg);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfGetAlgs(cusolverRfHandle_t handle,
                                                               cusolverRfFactorization_t *factAlg,
                                                               cusolverRfTriangularSolve_t *solveAlg) {
    HOOK_TRACE_PROFILE("cusolverRfGetAlgs");
    using func_ptr =
        cusolverStatus_t (*)(cusolverRfHandle_t, cusolverRfFactorization_t *, cusolverRfTriangularSolve_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfGetAlgs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, factAlg, solveAlg);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverRfGetResetValuesFastMode(cusolverRfHandle_t handle, cusolverRfResetValuesFastMode_t *fastMode) {
    HOOK_TRACE_PROFILE("cusolverRfGetResetValuesFastMode");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, cusolverRfResetValuesFastMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfGetResetValuesFastMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, fastMode);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverRfSetResetValuesFastMode(cusolverRfHandle_t handle, cusolverRfResetValuesFastMode_t fastMode) {
    HOOK_TRACE_PROFILE("cusolverRfSetResetValuesFastMode");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, cusolverRfResetValuesFastMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfSetResetValuesFastMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, fastMode);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfSetupHost(int n, int nnzA, int *h_csrRowPtrA, int *h_csrColIndA,
                                                                 double *h_csrValA, int nnzL, int *h_csrRowPtrL,
                                                                 int *h_csrColIndL, double *h_csrValL, int nnzU,
                                                                 int *h_csrRowPtrU, int *h_csrColIndU,
                                                                 double *h_csrValU, int *h_P, int *h_Q,
                                                                 cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfSetupHost");
    using func_ptr = cusolverStatus_t (*)(int, int, int *, int *, double *, int, int *, int *, double *, int, int *,
                                          int *, double *, int *, int *, cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfSetupHost"));
    HOOK_CHECK(func_entry);
    return func_entry(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU,
                      h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfSetupDevice(int n, int nnzA, int *csrRowPtrA, int *csrColIndA,
                                                                   double *csrValA, int nnzL, int *csrRowPtrL,
                                                                   int *csrColIndL, double *csrValL, int nnzU,
                                                                   int *csrRowPtrU, int *csrColIndU, double *csrValU,
                                                                   int *P, int *Q, cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfSetupDevice");
    using func_ptr = cusolverStatus_t (*)(int, int, int *, int *, double *, int, int *, int *, double *, int, int *,
                                          int *, double *, int *, int *, cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfSetupDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU,
                      csrColIndU, csrValU, P, Q, handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfResetValues(int n, int nnzA, int *csrRowPtrA, int *csrColIndA,
                                                                   double *csrValA, int *P, int *Q,
                                                                   cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfResetValues");
    using func_ptr = cusolverStatus_t (*)(int, int, int *, int *, double *, int *, int *, cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfResetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfAnalyze(cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfAnalyze");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfAnalyze"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfRefactor(cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfRefactor");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfRefactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfAccessBundledFactorsDevice(cusolverRfHandle_t handle, int *nnzM,
                                                                                  int **Mp, int **Mi, double **Mx) {
    HOOK_TRACE_PROFILE("cusolverRfAccessBundledFactorsDevice");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, int *, int **, int **, double **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfAccessBundledFactorsDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnzM, Mp, Mi, Mx);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfExtractBundledFactorsHost(cusolverRfHandle_t handle, int *h_nnzM,
                                                                                 int **h_Mp, int **h_Mi,
                                                                                 double **h_Mx) {
    HOOK_TRACE_PROFILE("cusolverRfExtractBundledFactorsHost");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, int *, int **, int **, double **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfExtractBundledFactorsHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, h_nnzM, h_Mp, h_Mi, h_Mx);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfExtractSplitFactorsHost(cusolverRfHandle_t handle, int *h_nnzL,
                                                                               int **h_csrRowPtrL, int **h_csrColIndL,
                                                                               double **h_csrValL, int *h_nnzU,
                                                                               int **h_csrRowPtrU, int **h_csrColIndU,
                                                                               double **h_csrValU) {
    HOOK_TRACE_PROFILE("cusolverRfExtractSplitFactorsHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverRfHandle_t, int *, int **, int **, double **, int *, int **, int **, double **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfExtractSplitFactorsHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, h_nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU,
                      h_csrValU);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfSolve(cusolverRfHandle_t handle, int *P, int *Q, int nrhs,
                                                             double *Temp, int ldt, double *XF, int ldxf) {
    HOOK_TRACE_PROFILE("cusolverRfSolve");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, int *, int *, int, double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, P, Q, nrhs, Temp, ldt, XF, ldxf);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfBatchSetupHost(int batchSize, int n, int nnzA, int *h_csrRowPtrA,
                                                                      int *h_csrColIndA, double *h_csrValA_array,
                                                                      int nnzL, int *h_csrRowPtrL, int *h_csrColIndL,
                                                                      double *h_csrValL, int nnzU, int *h_csrRowPtrU,
                                                                      int *h_csrColIndU, double *h_csrValU, int *h_P,
                                                                      int *h_Q, cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfBatchSetupHost");
    using func_ptr = cusolverStatus_t (*)(int, int, int, int *, int *, double *, int, int *, int *, double *, int,
                                          int *, int *, double *, int *, int *, cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfBatchSetupHost"));
    HOOK_CHECK(func_entry);
    return func_entry(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL,
                      h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfBatchResetValues(int batchSize, int n, int nnzA, int *csrRowPtrA,
                                                                        int *csrColIndA, double *csrValA_array, int *P,
                                                                        int *Q, cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfBatchResetValues");
    using func_ptr = cusolverStatus_t (*)(int, int, int, int *, int *, double *, int *, int *, cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfBatchResetValues"));
    HOOK_CHECK(func_entry);
    return func_entry(batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfBatchAnalyze(cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfBatchAnalyze");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfBatchAnalyze"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfBatchRefactor(cusolverRfHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverRfBatchRefactor");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfBatchRefactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfBatchSolve(cusolverRfHandle_t handle, int *P, int *Q, int nrhs,
                                                                  double *Temp, int ldt, double *XF_array, int ldxf) {
    HOOK_TRACE_PROFILE("cusolverRfBatchSolve");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, int *, int *, int, double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfBatchSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverRfBatchZeroPivot(cusolverRfHandle_t handle, int *position) {
    HOOK_TRACE_PROFILE("cusolverRfBatchZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverRfHandle_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverRfBatchZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t *handle) {
    HOOK_TRACE_PROFILE("cusolverSpCreate");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t handle) {
    HOOK_TRACE_PROFILE("cusolverSpDestroy");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t handle, cudaStream_t streamId) {
    HOOK_TRACE_PROFILE("cusolverSpSetStream");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpSetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpGetStream(cusolverSpHandle_t handle, cudaStream_t *streamId) {
    HOOK_TRACE_PROFILE("cusolverSpGetStream");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, cudaStream_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpGetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrissymHost(cusolverSpHandle_t handle, int m, int nnzA,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const int *csrRowPtrA, const int *csrEndPtrA,
                                                                     const int *csrColIndA, int *issym) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrissymHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, const int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrissymHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, issym);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrlsvluHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const float *csrValA, const int *csrRowPtrA,
                                                                     const int *csrColIndA, const float *b, float tol,
                                                                     int reorder, float *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpScsrlsvluHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, const float *, float, int, float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrlsvluHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrlsvluHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const double *csrValA, const int *csrRowPtrA,
                                                                     const int *csrColIndA, const double *b, double tol,
                                                                     int reorder, double *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrlsvluHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, const double *, double, int, double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrlsvluHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrlsvluHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const cuComplex *csrValA, const int *csrRowPtrA,
                                                                     const int *csrColIndA, const cuComplex *b,
                                                                     float tol, int reorder, cuComplex *x,
                                                                     int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrlsvluHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, const cuComplex *, float, int, cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrlsvluHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrlsvluHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const cuDoubleComplex *csrValA,
                                                                     const int *csrRowPtrA, const int *csrColIndA,
                                                                     const cuDoubleComplex *b, double tol, int reorder,
                                                                     cuDoubleComplex *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrlsvluHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, const cuDoubleComplex *, double, int, cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrlsvluHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                                                 const cusparseMatDescr_t descrA, const float *csrVal,
                                                                 const int *csrRowPtr, const int *csrColInd,
                                                                 const float *b, float tol, int reorder, float *x,
                                                                 int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpScsrlsvqr");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, const float *, float, int, float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrlsvqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                                                 const cusparseMatDescr_t descrA, const double *csrVal,
                                                                 const int *csrRowPtr, const int *csrColInd,
                                                                 const double *b, double tol, int reorder, double *x,
                                                                 int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrlsvqr");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, const double *, double, int, double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrlsvqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                                                 const cusparseMatDescr_t descrA,
                                                                 const cuComplex *csrVal, const int *csrRowPtr,
                                                                 const int *csrColInd, const cuComplex *b, float tol,
                                                                 int reorder, cuComplex *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrlsvqr");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, const cuComplex *, float, int, cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrlsvqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                                                 const cusparseMatDescr_t descrA,
                                                                 const cuDoubleComplex *csrVal, const int *csrRowPtr,
                                                                 const int *csrColInd, const cuDoubleComplex *b,
                                                                 double tol, int reorder, cuDoubleComplex *x,
                                                                 int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrlsvqr");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, const cuDoubleComplex *, double, int, cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrlsvqr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrlsvqrHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const float *csrValA, const int *csrRowPtrA,
                                                                     const int *csrColIndA, const float *b, float tol,
                                                                     int reorder, float *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpScsrlsvqrHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, const float *, float, int, float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrlsvqrHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrlsvqrHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const double *csrValA, const int *csrRowPtrA,
                                                                     const int *csrColIndA, const double *b, double tol,
                                                                     int reorder, double *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrlsvqrHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, const double *, double, int, double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrlsvqrHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrlsvqrHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const cuComplex *csrValA, const int *csrRowPtrA,
                                                                     const int *csrColIndA, const cuComplex *b,
                                                                     float tol, int reorder, cuComplex *x,
                                                                     int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrlsvqrHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, const cuComplex *, float, int, cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrlsvqrHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrlsvqrHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                     const cusparseMatDescr_t descrA,
                                                                     const cuDoubleComplex *csrValA,
                                                                     const int *csrRowPtrA, const int *csrColIndA,
                                                                     const cuDoubleComplex *b, double tol, int reorder,
                                                                     cuDoubleComplex *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrlsvqrHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, const cuDoubleComplex *, double, int, cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrlsvqrHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrlsvcholHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const float *csrVal, const int *csrRowPtr,
                                                                       const int *csrColInd, const float *b, float tol,
                                                                       int reorder, float *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpScsrlsvcholHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, const float *, float, int, float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrlsvcholHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrlsvcholHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double *csrVal,
    const int *csrRowPtr, const int *csrColInd, const double *b, double tol, int reorder, double *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrlsvcholHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, const double *, double, int, double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrlsvcholHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrlsvcholHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const cuComplex *csrVal, const int *csrRowPtr,
                                                                       const int *csrColInd, const cuComplex *b,
                                                                       float tol, int reorder, cuComplex *x,
                                                                       int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrlsvcholHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, const cuComplex *, float, int, cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrlsvcholHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpZcsrlsvcholHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,
                              const cuDoubleComplex *csrVal, const int *csrRowPtr, const int *csrColInd,
                              const cuDoubleComplex *b, double tol, int reorder, cuDoubleComplex *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrlsvcholHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, const cuDoubleComplex *, double, int, cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrlsvcholHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrlsvchol(cusolverSpHandle_t handle, int m, int nnz,
                                                                   const cusparseMatDescr_t descrA, const float *csrVal,
                                                                   const int *csrRowPtr, const int *csrColInd,
                                                                   const float *b, float tol, int reorder, float *x,
                                                                   int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpScsrlsvchol");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, const float *, float, int, float *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrlsvchol"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrlsvchol(cusolverSpHandle_t handle, int m, int nnz,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const double *csrVal, const int *csrRowPtr,
                                                                   const int *csrColInd, const double *b, double tol,
                                                                   int reorder, double *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrlsvchol");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, const double *, double, int, double *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrlsvchol"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrlsvchol(cusolverSpHandle_t handle, int m, int nnz,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const cuComplex *csrVal, const int *csrRowPtr,
                                                                   const int *csrColInd, const cuComplex *b, float tol,
                                                                   int reorder, cuComplex *x, int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrlsvchol");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, const cuComplex *, float, int, cuComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrlsvchol"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrlsvchol(cusolverSpHandle_t handle, int m, int nnz,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const cuDoubleComplex *csrVal, const int *csrRowPtr,
                                                                   const int *csrColInd, const cuDoubleComplex *b,
                                                                   double tol, int reorder, cuDoubleComplex *x,
                                                                   int *singularity) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrlsvchol");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, const cuDoubleComplex *, double, int, cuDoubleComplex *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrlsvchol"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const float *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, const float *b, float tol,
                                                                      int *rankA, float *x, int *p, float *min_norm) {
    HOOK_TRACE_PROFILE("cusolverSpScsrlsqvqrHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const float *, const int *,
                             const int *, const float *, float, int *, float *, int *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrlsqvqrHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const double *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, const double *b,
                                                                      double tol, int *rankA, double *x, int *p,
                                                                      double *min_norm) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrlsqvqrHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const double *, const int *,
                             const int *, const double *, double, int *, double *, int *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrlsqvqrHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const cuComplex *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, const cuComplex *b,
                                                                      float tol, int *rankA, cuComplex *x, int *p,
                                                                      float *min_norm) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrlsqvqrHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuComplex *,
                             const int *, const int *, const cuComplex *, float, int *, cuComplex *, int *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrlsqvqrHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrlsqvqrHost(cusolverSpHandle_t handle, int m, int n, int nnz,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const cuDoubleComplex *csrValA,
                                                                      const int *csrRowPtrA, const int *csrColIndA,
                                                                      const cuDoubleComplex *b, double tol, int *rankA,
                                                                      cuDoubleComplex *x, int *p, double *min_norm) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrlsqvqrHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, const cuDoubleComplex *,
                                          double, int *, cuDoubleComplex *, int *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrlsqvqrHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsreigvsiHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const float *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, float mu0, const float *x0,
                                                                      int maxite, float tol, float *mu, float *x) {
    HOOK_TRACE_PROFILE("cusolverSpScsreigvsiHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, float, const float *, int, float, float *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsreigvsiHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsreigvsiHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const double *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, double mu0,
                                                                      const double *x0, int maxite, double tol,
                                                                      double *mu, double *x) {
    HOOK_TRACE_PROFILE("cusolverSpDcsreigvsiHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *, const int *,
                             const int *, double, const double *, int, double, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsreigvsiHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsreigvsiHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const cuComplex *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, cuComplex mu0,
                                                                      const cuComplex *x0, int maxite, float tol,
                                                                      cuComplex *mu, cuComplex *x) {
    HOOK_TRACE_PROFILE("cusolverSpCcsreigvsiHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *, const int *,
                             const int *, cuComplex, const cuComplex *, int, float, cuComplex *, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsreigvsiHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsreigvsiHost(
    cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, cuDoubleComplex mu0, const cuDoubleComplex *x0, int maxite,
    double tol, cuDoubleComplex *mu, cuDoubleComplex *x) {
    HOOK_TRACE_PROFILE("cusolverSpZcsreigvsiHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, cuDoubleComplex,
                                          const cuDoubleComplex *, int, double, cuDoubleComplex *, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsreigvsiHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                                                  const cusparseMatDescr_t descrA, const float *csrValA,
                                                                  const int *csrRowPtrA, const int *csrColIndA,
                                                                  float mu0, const float *x0, int maxite, float eps,
                                                                  float *mu, float *x) {
    HOOK_TRACE_PROFILE("cusolverSpScsreigvsi");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, float, const float *, int, float, float *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsreigvsi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                                                  const cusparseMatDescr_t descrA,
                                                                  const double *csrValA, const int *csrRowPtrA,
                                                                  const int *csrColIndA, double mu0, const double *x0,
                                                                  int maxite, double eps, double *mu, double *x) {
    HOOK_TRACE_PROFILE("cusolverSpDcsreigvsi");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *, const int *,
                             const int *, double, const double *, int, double, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsreigvsi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                                                  const cusparseMatDescr_t descrA,
                                                                  const cuComplex *csrValA, const int *csrRowPtrA,
                                                                  const int *csrColIndA, cuComplex mu0,
                                                                  const cuComplex *x0, int maxite, float eps,
                                                                  cuComplex *mu, cuComplex *x) {
    HOOK_TRACE_PROFILE("cusolverSpCcsreigvsi");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *, const int *,
                             const int *, cuComplex, const cuComplex *, int, float, cuComplex *, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsreigvsi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                                                  const cusparseMatDescr_t descrA,
                                                                  const cuDoubleComplex *csrValA, const int *csrRowPtrA,
                                                                  const int *csrColIndA, cuDoubleComplex mu0,
                                                                  const cuDoubleComplex *x0, int maxite, double eps,
                                                                  cuDoubleComplex *mu, cuDoubleComplex *x) {
    HOOK_TRACE_PROFILE("cusolverSpZcsreigvsi");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, cuDoubleComplex,
                                          const cuDoubleComplex *, int, double, cuDoubleComplex *, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsreigvsi"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsreigsHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                    const cusparseMatDescr_t descrA,
                                                                    const float *csrValA, const int *csrRowPtrA,
                                                                    const int *csrColIndA, cuComplex left_bottom_corner,
                                                                    cuComplex right_upper_corner, int *num_eigs) {
    HOOK_TRACE_PROFILE("cusolverSpScsreigsHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, cuComplex, cuComplex, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsreigsHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner,
                      num_eigs);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsreigsHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                    const cusparseMatDescr_t descrA,
                                                                    const double *csrValA, const int *csrRowPtrA,
                                                                    const int *csrColIndA,
                                                                    cuDoubleComplex left_bottom_corner,
                                                                    cuDoubleComplex right_upper_corner, int *num_eigs) {
    HOOK_TRACE_PROFILE("cusolverSpDcsreigsHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, cuDoubleComplex, cuDoubleComplex, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsreigsHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner,
                      num_eigs);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsreigsHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                    const cusparseMatDescr_t descrA,
                                                                    const cuComplex *csrValA, const int *csrRowPtrA,
                                                                    const int *csrColIndA, cuComplex left_bottom_corner,
                                                                    cuComplex right_upper_corner, int *num_eigs) {
    HOOK_TRACE_PROFILE("cusolverSpCcsreigsHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, cuComplex, cuComplex, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsreigsHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner,
                      num_eigs);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsreigsHost(cusolverSpHandle_t handle, int m, int nnz,
                                                                    const cusparseMatDescr_t descrA,
                                                                    const cuDoubleComplex *csrValA,
                                                                    const int *csrRowPtrA, const int *csrColIndA,
                                                                    cuDoubleComplex left_bottom_corner,
                                                                    cuDoubleComplex right_upper_corner, int *num_eigs) {
    HOOK_TRACE_PROFILE("cusolverSpZcsreigsHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, cuDoubleComplex, cuDoubleComplex, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsreigsHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner,
                      num_eigs);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrsymrcmHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const int *csrRowPtrA, const int *csrColIndA,
                                                                      int *p) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrsymrcmHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const int *, const int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrsymrcmHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrsymmdqHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const int *csrRowPtrA, const int *csrColIndA,
                                                                      int *p) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrsymmdqHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const int *, const int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrsymmdqHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrsymamdHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const int *csrRowPtrA, const int *csrColIndA,
                                                                      int *p) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrsymamdHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const int *, const int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrsymamdHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrmetisndHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const int *csrRowPtrA, const int *csrColIndA,
                                                                       const int64_t *options, int *p) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrmetisndHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, const int64_t *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrmetisndHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, options, p);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrzfdHost(cusolverSpHandle_t handle, int n, int nnz,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const float *csrValA, const int *csrRowPtrA,
                                                                   const int *csrColIndA, int *P, int *numnz) {
    HOOK_TRACE_PROFILE("cusolverSpScsrzfdHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrzfdHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrzfdHost(cusolverSpHandle_t handle, int n, int nnz,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const double *csrValA, const int *csrRowPtrA,
                                                                   const int *csrColIndA, int *P, int *numnz) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrzfdHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrzfdHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrzfdHost(cusolverSpHandle_t handle, int n, int nnz,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const cuComplex *csrValA, const int *csrRowPtrA,
                                                                   const int *csrColIndA, int *P, int *numnz) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrzfdHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrzfdHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrzfdHost(cusolverSpHandle_t handle, int n, int nnz,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const cuDoubleComplex *csrValA,
                                                                   const int *csrRowPtrA, const int *csrColIndA, int *P,
                                                                   int *numnz) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrzfdHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrzfdHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrperm_bufferSizeHost(
    cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const int *csrRowPtrA,
    const int *csrColIndA, const int *p, const int *q, size_t *bufferSizeInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrperm_bufferSizeHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, const int *, const int *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrperm_bufferSizeHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, bufferSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrpermHost(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                    const cusparseMatDescr_t descrA, int *csrRowPtrA,
                                                                    int *csrColIndA, const int *p, const int *q,
                                                                    int *map, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrpermHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, int *, int *,
                                          const int *, const int *, int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrpermHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCreateCsrqrInfo(csrqrInfo_t *info) {
    HOOK_TRACE_PROFILE("cusolverSpCreateCsrqrInfo");
    using func_ptr = cusolverStatus_t (*)(csrqrInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCreateCsrqrInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDestroyCsrqrInfo(csrqrInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpDestroyCsrqrInfo");
    using func_ptr = cusolverStatus_t (*)(csrqrInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDestroyCsrqrInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrqrAnalysisBatched(cusolverSpHandle_t handle, int m, int n,
                                                                             int nnzA, const cusparseMatDescr_t descrA,
                                                                             const int *csrRowPtrA,
                                                                             const int *csrColIndA, csrqrInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrqrAnalysisBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, csrqrInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrqrAnalysisBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpScsrqrBufferInfoBatched(cusolverSpHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA,
                                      const float *csrVal, const int *csrRowPtr, const int *csrColInd, int batchSize,
                                      csrqrInfo_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrBufferInfoBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, int, csrqrInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrBufferInfoBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpDcsrqrBufferInfoBatched(cusolverSpHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA,
                                      const double *csrVal, const int *csrRowPtr, const int *csrColInd, int batchSize,
                                      csrqrInfo_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrBufferInfoBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, int, csrqrInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrBufferInfoBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const cuComplex *csrVal,
    const int *csrRowPtr, const int *csrColInd, int batchSize, csrqrInfo_t info, size_t *internalDataInBytes,
    size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrBufferInfoBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuComplex *,
                             const int *, const int *, int, csrqrInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrBufferInfoBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrBufferInfoBatched(
    cusolverSpHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrVal,
    const int *csrRowPtr, const int *csrColInd, int batchSize, csrqrInfo_t info, size_t *internalDataInBytes,
    size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrBufferInfoBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, int, csrqrInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrBufferInfoBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrsvBatched(cusolverSpHandle_t handle, int m, int n, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const float *csrValA, const int *csrRowPtrA,
                                                                       const int *csrColIndA, const float *b, float *x,
                                                                       int batchSize, csrqrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrsvBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, const float *, float *, int, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrsvBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrsvBatched(cusolverSpHandle_t handle, int m, int n, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const double *csrValA, const int *csrRowPtrA,
                                                                       const int *csrColIndA, const double *b,
                                                                       double *x, int batchSize, csrqrInfo_t info,
                                                                       void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrsvBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, const double *, double *, int, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrsvBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrsvBatched(cusolverSpHandle_t handle, int m, int n, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const cuComplex *csrValA, const int *csrRowPtrA,
                                                                       const int *csrColIndA, const cuComplex *b,
                                                                       cuComplex *x, int batchSize, csrqrInfo_t info,
                                                                       void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrsvBatched");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuComplex *,
                             const int *, const int *, const cuComplex *, cuComplex *, int, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrsvBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrsvBatched(cusolverSpHandle_t handle, int m, int n, int nnz,
                                                                       const cusparseMatDescr_t descrA,
                                                                       const cuDoubleComplex *csrValA,
                                                                       const int *csrRowPtrA, const int *csrColIndA,
                                                                       const cuDoubleComplex *b, cuDoubleComplex *x,
                                                                       int batchSize, csrqrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrsvBatched");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, const cuDoubleComplex *,
                                          cuDoubleComplex *, int, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrsvBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCreateCsrluInfoHost(csrluInfoHost_t *info) {
    HOOK_TRACE_PROFILE("cusolverSpCreateCsrluInfoHost");
    using func_ptr = cusolverStatus_t (*)(csrluInfoHost_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCreateCsrluInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDestroyCsrluInfoHost(csrluInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpDestroyCsrluInfoHost");
    using func_ptr = cusolverStatus_t (*)(csrluInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDestroyCsrluInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrluAnalysisHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                          const cusparseMatDescr_t descrA,
                                                                          const int *csrRowPtrA, const int *csrColIndA,
                                                                          csrluInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrluAnalysisHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, csrluInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrluAnalysisHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrluBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                            const cusparseMatDescr_t descrA,
                                                                            const float *csrValA, const int *csrRowPtrA,
                                                                            const int *csrColIndA, csrluInfoHost_t info,
                                                                            size_t *internalDataInBytes,
                                                                            size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpScsrluBufferInfoHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrluInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrluBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpDcsrluBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                   const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                   csrluInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrluBufferInfoHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrluInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrluBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpCcsrluBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                   const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                   csrluInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrluBufferInfoHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, csrluInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrluBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpZcsrluBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                   const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                   csrluInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrluBufferInfoHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, csrluInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrluBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrluFactorHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                        const cusparseMatDescr_t descrA,
                                                                        const float *csrValA, const int *csrRowPtrA,
                                                                        const int *csrColIndA, csrluInfoHost_t info,
                                                                        float pivot_threshold, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrluFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrluInfoHost_t, float, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrluFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrluFactorHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                        const cusparseMatDescr_t descrA,
                                                                        const double *csrValA, const int *csrRowPtrA,
                                                                        const int *csrColIndA, csrluInfoHost_t info,
                                                                        double pivot_threshold, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrluFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrluInfoHost_t, double, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrluFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrluFactorHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                        const cusparseMatDescr_t descrA,
                                                                        const cuComplex *csrValA, const int *csrRowPtrA,
                                                                        const int *csrColIndA, csrluInfoHost_t info,
                                                                        float pivot_threshold, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrluFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, csrluInfoHost_t, float, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrluFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrluFactorHost(
    cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA,
    const int *csrRowPtrA, const int *csrColIndA, csrluInfoHost_t info, double pivot_threshold, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrluFactorHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, csrluInfoHost_t, double, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrluFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrluZeroPivotHost(cusolverSpHandle_t handle,
                                                                           csrluInfoHost_t info, float tol,
                                                                           int *position) {
    HOOK_TRACE_PROFILE("cusolverSpScsrluZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrluInfoHost_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrluZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrluZeroPivotHost(cusolverSpHandle_t handle,
                                                                           csrluInfoHost_t info, double tol,
                                                                           int *position) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrluZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrluInfoHost_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrluZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrluZeroPivotHost(cusolverSpHandle_t handle,
                                                                           csrluInfoHost_t info, float tol,
                                                                           int *position) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrluZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrluInfoHost_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrluZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrluZeroPivotHost(cusolverSpHandle_t handle,
                                                                           csrluInfoHost_t info, double tol,
                                                                           int *position) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrluZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrluInfoHost_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrluZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrluSolveHost(cusolverSpHandle_t handle, int n, const float *b,
                                                                       float *x, csrluInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrluSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const float *, float *, csrluInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrluSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrluSolveHost(cusolverSpHandle_t handle, int n,
                                                                       const double *b, double *x, csrluInfoHost_t info,
                                                                       void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrluSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const double *, double *, csrluInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrluSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrluSolveHost(cusolverSpHandle_t handle, int n,
                                                                       const cuComplex *b, cuComplex *x,
                                                                       csrluInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrluSolveHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, const cuComplex *, cuComplex *, csrluInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrluSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrluSolveHost(cusolverSpHandle_t handle, int n,
                                                                       const cuDoubleComplex *b, cuDoubleComplex *x,
                                                                       csrluInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrluSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const cuDoubleComplex *, cuDoubleComplex *,
                                          csrluInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrluSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrluNnzHost(cusolverSpHandle_t handle, int *nnzLRef,
                                                                     int *nnzURef, csrluInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrluNnzHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int *, int *, csrluInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrluNnzHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nnzLRef, nnzURef, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpScsrluExtractHost(cusolverSpHandle_t handle, int *P, int *Q, const cusparseMatDescr_t descrL,
                                float *csrValL, int *csrRowPtrL, int *csrColIndL, const cusparseMatDescr_t descrU,
                                float *csrValU, int *csrRowPtrU, int *csrColIndU, csrluInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrluExtractHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int *, int *, const cusparseMatDescr_t, float *, int *, int *,
                             const cusparseMatDescr_t, float *, int *, int *, csrluInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrluExtractHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU,
                      info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrluExtractHost(
    cusolverSpHandle_t handle, int *P, int *Q, const cusparseMatDescr_t descrL, double *csrValL, int *csrRowPtrL,
    int *csrColIndL, const cusparseMatDescr_t descrU, double *csrValU, int *csrRowPtrU, int *csrColIndU,
    csrluInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrluExtractHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int *, int *, const cusparseMatDescr_t, double *, int *, int *,
                             const cusparseMatDescr_t, double *, int *, int *, csrluInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrluExtractHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU,
                      info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrluExtractHost(
    cusolverSpHandle_t handle, int *P, int *Q, const cusparseMatDescr_t descrL, cuComplex *csrValL, int *csrRowPtrL,
    int *csrColIndL, const cusparseMatDescr_t descrU, cuComplex *csrValU, int *csrRowPtrU, int *csrColIndU,
    csrluInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrluExtractHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int *, int *, const cusparseMatDescr_t, cuComplex *, int *, int *,
                             const cusparseMatDescr_t, cuComplex *, int *, int *, csrluInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrluExtractHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU,
                      info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrluExtractHost(
    cusolverSpHandle_t handle, int *P, int *Q, const cusparseMatDescr_t descrL, cuDoubleComplex *csrValL,
    int *csrRowPtrL, int *csrColIndL, const cusparseMatDescr_t descrU, cuDoubleComplex *csrValU, int *csrRowPtrU,
    int *csrColIndU, csrluInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrluExtractHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int *, int *, const cusparseMatDescr_t, cuDoubleComplex *, int *,
                             int *, const cusparseMatDescr_t, cuDoubleComplex *, int *, int *, csrluInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrluExtractHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU,
                      info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCreateCsrqrInfoHost(csrqrInfoHost_t *info) {
    HOOK_TRACE_PROFILE("cusolverSpCreateCsrqrInfoHost");
    using func_ptr = cusolverStatus_t (*)(csrqrInfoHost_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCreateCsrqrInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDestroyCsrqrInfoHost(csrqrInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpDestroyCsrqrInfoHost");
    using func_ptr = cusolverStatus_t (*)(csrqrInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDestroyCsrqrInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrqrAnalysisHost(cusolverSpHandle_t handle, int m, int n,
                                                                          int nnzA, const cusparseMatDescr_t descrA,
                                                                          const int *csrRowPtrA, const int *csrColIndA,
                                                                          csrqrInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrqrAnalysisHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, csrqrInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrqrAnalysisHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrBufferInfoHost(cusolverSpHandle_t handle, int m, int n,
                                                                            int nnzA, const cusparseMatDescr_t descrA,
                                                                            const float *csrValA, const int *csrRowPtrA,
                                                                            const int *csrColIndA, csrqrInfoHost_t info,
                                                                            size_t *internalDataInBytes,
                                                                            size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrBufferInfoHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrqrInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpDcsrqrBufferInfoHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA,
                                   const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                   csrqrInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrBufferInfoHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrqrInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpCcsrqrBufferInfoHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA,
                                   const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                   csrqrInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrBufferInfoHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuComplex *,
                             const int *, const int *, csrqrInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpZcsrqrBufferInfoHost(cusolverSpHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA,
                                   const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                   csrqrInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrBufferInfoHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, csrqrInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrSetupHost(cusolverSpHandle_t handle, int m, int n,
                                                                       int nnzA, const cusparseMatDescr_t descrA,
                                                                       const float *csrValA, const int *csrRowPtrA,
                                                                       const int *csrColIndA, float mu,
                                                                       csrqrInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrSetupHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, float, csrqrInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrSetupHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrSetupHost(cusolverSpHandle_t handle, int m, int n,
                                                                       int nnzA, const cusparseMatDescr_t descrA,
                                                                       const double *csrValA, const int *csrRowPtrA,
                                                                       const int *csrColIndA, double mu,
                                                                       csrqrInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrSetupHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, double, csrqrInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrSetupHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrSetupHost(cusolverSpHandle_t handle, int m, int n,
                                                                       int nnzA, const cusparseMatDescr_t descrA,
                                                                       const cuComplex *csrValA, const int *csrRowPtrA,
                                                                       const int *csrColIndA, cuComplex mu,
                                                                       csrqrInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrSetupHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, cuComplex, csrqrInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrSetupHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrSetupHost(cusolverSpHandle_t handle, int m, int n,
                                                                       int nnzA, const cusparseMatDescr_t descrA,
                                                                       const cuDoubleComplex *csrValA,
                                                                       const int *csrRowPtrA, const int *csrColIndA,
                                                                       cuDoubleComplex mu, csrqrInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrSetupHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, cuDoubleComplex, csrqrInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrSetupHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrFactorHost(cusolverSpHandle_t handle, int m, int n,
                                                                        int nnzA, float *b, float *x,
                                                                        csrqrInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, float *, float *, csrqrInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrFactorHost(cusolverSpHandle_t handle, int m, int n,
                                                                        int nnzA, double *b, double *x,
                                                                        csrqrInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrFactorHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, double *, double *, csrqrInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrFactorHost(cusolverSpHandle_t handle, int m, int n,
                                                                        int nnzA, cuComplex *b, cuComplex *x,
                                                                        csrqrInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrFactorHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, cuComplex *, cuComplex *, csrqrInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrFactorHost(cusolverSpHandle_t handle, int m, int n,
                                                                        int nnzA, cuDoubleComplex *b,
                                                                        cuDoubleComplex *x, csrqrInfoHost_t info,
                                                                        void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, cuDoubleComplex *, cuDoubleComplex *,
                                          csrqrInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrZeroPivotHost(cusolverSpHandle_t handle,
                                                                           csrqrInfoHost_t info, float tol,
                                                                           int *position) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrqrInfoHost_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrZeroPivotHost(cusolverSpHandle_t handle,
                                                                           csrqrInfoHost_t info, double tol,
                                                                           int *position) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrqrInfoHost_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrZeroPivotHost(cusolverSpHandle_t handle,
                                                                           csrqrInfoHost_t info, float tol,
                                                                           int *position) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrqrInfoHost_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrZeroPivotHost(cusolverSpHandle_t handle,
                                                                           csrqrInfoHost_t info, double tol,
                                                                           int *position) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrqrInfoHost_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrSolveHost(cusolverSpHandle_t handle, int m, int n,
                                                                       float *b, float *x, csrqrInfoHost_t info,
                                                                       void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, float *, float *, csrqrInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrSolveHost(cusolverSpHandle_t handle, int m, int n,
                                                                       double *b, double *x, csrqrInfoHost_t info,
                                                                       void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, double *, double *, csrqrInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrSolveHost(cusolverSpHandle_t handle, int m, int n,
                                                                       cuComplex *b, cuComplex *x, csrqrInfoHost_t info,
                                                                       void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrSolveHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, cuComplex *, cuComplex *, csrqrInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrSolveHost(cusolverSpHandle_t handle, int m, int n,
                                                                       cuDoubleComplex *b, cuDoubleComplex *x,
                                                                       csrqrInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, cuDoubleComplex *, cuDoubleComplex *,
                                          csrqrInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrqrAnalysis(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const int *csrRowPtrA, const int *csrColIndA,
                                                                      csrqrInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrqrAnalysis");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, csrqrInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrqrAnalysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrBufferInfo(cusolverSpHandle_t handle, int m, int n,
                                                                        int nnzA, const cusparseMatDescr_t descrA,
                                                                        const float *csrValA, const int *csrRowPtrA,
                                                                        const int *csrColIndA, csrqrInfo_t info,
                                                                        size_t *internalDataInBytes,
                                                                        size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrBufferInfo");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrqrInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrBufferInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrBufferInfo(cusolverSpHandle_t handle, int m, int n,
                                                                        int nnzA, const cusparseMatDescr_t descrA,
                                                                        const double *csrValA, const int *csrRowPtrA,
                                                                        const int *csrColIndA, csrqrInfo_t info,
                                                                        size_t *internalDataInBytes,
                                                                        size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrBufferInfo");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrqrInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrBufferInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrBufferInfo(cusolverSpHandle_t handle, int m, int n,
                                                                        int nnzA, const cusparseMatDescr_t descrA,
                                                                        const cuComplex *csrValA, const int *csrRowPtrA,
                                                                        const int *csrColIndA, csrqrInfo_t info,
                                                                        size_t *internalDataInBytes,
                                                                        size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrBufferInfo");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, csrqrInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrBufferInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrBufferInfo(cusolverSpHandle_t handle, int m, int n,
                                                                        int nnzA, const cusparseMatDescr_t descrA,
                                                                        const cuDoubleComplex *csrValA,
                                                                        const int *csrRowPtrA, const int *csrColIndA,
                                                                        csrqrInfo_t info, size_t *internalDataInBytes,
                                                                        size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrBufferInfo");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, csrqrInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrBufferInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrSetup(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const float *csrValA, const int *csrRowPtrA,
                                                                   const int *csrColIndA, float mu, csrqrInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrSetup");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, float, csrqrInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrSetup"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrSetup(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const double *csrValA, const int *csrRowPtrA,
                                                                   const int *csrColIndA, double mu, csrqrInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrSetup");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, double, csrqrInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrSetup"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrSetup(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const cuComplex *csrValA, const int *csrRowPtrA,
                                                                   const int *csrColIndA, cuComplex mu,
                                                                   csrqrInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrSetup");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t,
                                          const cuComplex *, const int *, const int *, cuComplex, csrqrInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrSetup"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrSetup(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                   const cusparseMatDescr_t descrA,
                                                                   const cuDoubleComplex *csrValA,
                                                                   const int *csrRowPtrA, const int *csrColIndA,
                                                                   cuDoubleComplex mu, csrqrInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrSetup");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, cuDoubleComplex, csrqrInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrSetup"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, mu, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrFactor(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                    float *b, float *x, csrqrInfo_t info,
                                                                    void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrFactor");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, float *, float *, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrFactor(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                    double *b, double *x, csrqrInfo_t info,
                                                                    void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrFactor");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, double *, double *, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrFactor(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                    cuComplex *b, cuComplex *x, csrqrInfo_t info,
                                                                    void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrFactor");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, cuComplex *, cuComplex *, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrFactor(cusolverSpHandle_t handle, int m, int n, int nnzA,
                                                                    cuDoubleComplex *b, cuDoubleComplex *x,
                                                                    csrqrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrFactor");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, int, cuDoubleComplex *, cuDoubleComplex *,
                                          csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, nnzA, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrZeroPivot(cusolverSpHandle_t handle, csrqrInfo_t info,
                                                                       float tol, int *position) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrqrInfo_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrZeroPivot(cusolverSpHandle_t handle, csrqrInfo_t info,
                                                                       double tol, int *position) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrqrInfo_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrZeroPivot(cusolverSpHandle_t handle, csrqrInfo_t info,
                                                                       float tol, int *position) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrqrInfo_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrZeroPivot(cusolverSpHandle_t handle, csrqrInfo_t info,
                                                                       double tol, int *position) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrqrInfo_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrqrSolve(cusolverSpHandle_t handle, int m, int n, float *b,
                                                                   float *x, csrqrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrqrSolve");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, float *, float *, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrqrSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrqrSolve(cusolverSpHandle_t handle, int m, int n, double *b,
                                                                   double *x, csrqrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrqrSolve");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, double *, double *, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrqrSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrqrSolve(cusolverSpHandle_t handle, int m, int n,
                                                                   cuComplex *b, cuComplex *x, csrqrInfo_t info,
                                                                   void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrqrSolve");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, cuComplex *, cuComplex *, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrqrSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrqrSolve(cusolverSpHandle_t handle, int m, int n,
                                                                   cuDoubleComplex *b, cuDoubleComplex *x,
                                                                   csrqrInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrqrSolve");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, cuDoubleComplex *, cuDoubleComplex *, csrqrInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrqrSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCreateCsrcholInfoHost(csrcholInfoHost_t *info) {
    HOOK_TRACE_PROFILE("cusolverSpCreateCsrcholInfoHost");
    using func_ptr = cusolverStatus_t (*)(csrcholInfoHost_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCreateCsrcholInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDestroyCsrcholInfoHost(csrcholInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpDestroyCsrcholInfoHost");
    using func_ptr = cusolverStatus_t (*)(csrcholInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDestroyCsrcholInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrcholAnalysisHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                            const cusparseMatDescr_t descrA,
                                                                            const int *csrRowPtrA,
                                                                            const int *csrColIndA,
                                                                            csrcholInfoHost_t info) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrcholAnalysisHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, csrcholInfoHost_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrcholAnalysisHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpScsrcholBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                     const float *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                     csrcholInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholBufferInfoHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrcholInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpDcsrcholBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                     const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                     csrcholInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholBufferInfoHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrcholInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpCcsrcholBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                     const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                     csrcholInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholBufferInfoHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, csrcholInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpZcsrcholBufferInfoHost(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                     const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                     csrcholInfoHost_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholBufferInfoHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, csrcholInfoHost_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholBufferInfoHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrcholFactorHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                          const cusparseMatDescr_t descrA,
                                                                          const float *csrValA, const int *csrRowPtrA,
                                                                          const int *csrColIndA, csrcholInfoHost_t info,
                                                                          void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrcholInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrcholFactorHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                          const cusparseMatDescr_t descrA,
                                                                          const double *csrValA, const int *csrRowPtrA,
                                                                          const int *csrColIndA, csrcholInfoHost_t info,
                                                                          void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrcholInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrcholFactorHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                          const cusparseMatDescr_t descrA,
                                                                          const cuComplex *csrValA,
                                                                          const int *csrRowPtrA, const int *csrColIndA,
                                                                          csrcholInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, csrcholInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrcholFactorHost(cusolverSpHandle_t handle, int n, int nnzA,
                                                                          const cusparseMatDescr_t descrA,
                                                                          const cuDoubleComplex *csrValA,
                                                                          const int *csrRowPtrA, const int *csrColIndA,
                                                                          csrcholInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholFactorHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, csrcholInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholFactorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrcholZeroPivotHost(cusolverSpHandle_t handle,
                                                                             csrcholInfoHost_t info, float tol,
                                                                             int *position) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfoHost_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrcholZeroPivotHost(cusolverSpHandle_t handle,
                                                                             csrcholInfoHost_t info, double tol,
                                                                             int *position) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfoHost_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrcholZeroPivotHost(cusolverSpHandle_t handle,
                                                                             csrcholInfoHost_t info, float tol,
                                                                             int *position) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfoHost_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrcholZeroPivotHost(cusolverSpHandle_t handle,
                                                                             csrcholInfoHost_t info, double tol,
                                                                             int *position) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholZeroPivotHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfoHost_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholZeroPivotHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrcholSolveHost(cusolverSpHandle_t handle, int n,
                                                                         const float *b, float *x,
                                                                         csrcholInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const float *, float *, csrcholInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrcholSolveHost(cusolverSpHandle_t handle, int n,
                                                                         const double *b, double *x,
                                                                         csrcholInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const double *, double *, csrcholInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrcholSolveHost(cusolverSpHandle_t handle, int n,
                                                                         const cuComplex *b, cuComplex *x,
                                                                         csrcholInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholSolveHost");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, const cuComplex *, cuComplex *, csrcholInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrcholSolveHost(cusolverSpHandle_t handle, int n,
                                                                         const cuDoubleComplex *b, cuDoubleComplex *x,
                                                                         csrcholInfoHost_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholSolveHost");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const cuDoubleComplex *, cuDoubleComplex *,
                                          csrcholInfoHost_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholSolveHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCreateCsrcholInfo(csrcholInfo_t *info) {
    HOOK_TRACE_PROFILE("cusolverSpCreateCsrcholInfo");
    using func_ptr = cusolverStatus_t (*)(csrcholInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCreateCsrcholInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDestroyCsrcholInfo(csrcholInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpDestroyCsrcholInfo");
    using func_ptr = cusolverStatus_t (*)(csrcholInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDestroyCsrcholInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpXcsrcholAnalysis(cusolverSpHandle_t handle, int n, int nnzA,
                                                                        const cusparseMatDescr_t descrA,
                                                                        const int *csrRowPtrA, const int *csrColIndA,
                                                                        csrcholInfo_t info) {
    HOOK_TRACE_PROFILE("cusolverSpXcsrcholAnalysis");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const int *,
                                          const int *, csrcholInfo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpXcsrcholAnalysis"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, info);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrcholBufferInfo(cusolverSpHandle_t handle, int n, int nnzA,
                                                                          const cusparseMatDescr_t descrA,
                                                                          const float *csrValA, const int *csrRowPtrA,
                                                                          const int *csrColIndA, csrcholInfo_t info,
                                                                          size_t *internalDataInBytes,
                                                                          size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholBufferInfo");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrcholInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholBufferInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrcholBufferInfo(cusolverSpHandle_t handle, int n, int nnzA,
                                                                          const cusparseMatDescr_t descrA,
                                                                          const double *csrValA, const int *csrRowPtrA,
                                                                          const int *csrColIndA, csrcholInfo_t info,
                                                                          size_t *internalDataInBytes,
                                                                          size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholBufferInfo");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrcholInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholBufferInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpCcsrcholBufferInfo(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                 const cuComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                 csrcholInfo_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholBufferInfo");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, csrcholInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholBufferInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t
    cusolverSpZcsrcholBufferInfo(cusolverSpHandle_t handle, int n, int nnzA, const cusparseMatDescr_t descrA,
                                 const cuDoubleComplex *csrValA, const int *csrRowPtrA, const int *csrColIndA,
                                 csrcholInfo_t info, size_t *internalDataInBytes, size_t *workspaceInBytes) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholBufferInfo");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuDoubleComplex *,
                             const int *, const int *, csrcholInfo_t, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholBufferInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes,
                      workspaceInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrcholFactor(cusolverSpHandle_t handle, int n, int nnzA,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const float *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, csrcholInfo_t info,
                                                                      void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholFactor");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const float *,
                                          const int *, const int *, csrcholInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrcholFactor(cusolverSpHandle_t handle, int n, int nnzA,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const double *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, csrcholInfo_t info,
                                                                      void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholFactor");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const double *,
                                          const int *, const int *, csrcholInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrcholFactor(cusolverSpHandle_t handle, int n, int nnzA,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const cuComplex *csrValA, const int *csrRowPtrA,
                                                                      const int *csrColIndA, csrcholInfo_t info,
                                                                      void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholFactor");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t, const cuComplex *,
                                          const int *, const int *, csrcholInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrcholFactor(cusolverSpHandle_t handle, int n, int nnzA,
                                                                      const cusparseMatDescr_t descrA,
                                                                      const cuDoubleComplex *csrValA,
                                                                      const int *csrRowPtrA, const int *csrColIndA,
                                                                      csrcholInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholFactor");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, int, const cusparseMatDescr_t,
                                          const cuDoubleComplex *, const int *, const int *, csrcholInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrcholZeroPivot(cusolverSpHandle_t handle, csrcholInfo_t info,
                                                                         float tol, int *position) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfo_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrcholZeroPivot(cusolverSpHandle_t handle, csrcholInfo_t info,
                                                                         double tol, int *position) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfo_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrcholZeroPivot(cusolverSpHandle_t handle, csrcholInfo_t info,
                                                                         float tol, int *position) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfo_t, float, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrcholZeroPivot(cusolverSpHandle_t handle, csrcholInfo_t info,
                                                                         double tol, int *position) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholZeroPivot");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfo_t, double, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholZeroPivot"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, tol, position);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrcholSolve(cusolverSpHandle_t handle, int n, const float *b,
                                                                     float *x, csrcholInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholSolve");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const float *, float *, csrcholInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrcholSolve(cusolverSpHandle_t handle, int n, const double *b,
                                                                     double *x, csrcholInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholSolve");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const double *, double *, csrcholInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrcholSolve(cusolverSpHandle_t handle, int n,
                                                                     const cuComplex *b, cuComplex *x,
                                                                     csrcholInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholSolve");
    using func_ptr =
        cusolverStatus_t (*)(cusolverSpHandle_t, int, const cuComplex *, cuComplex *, csrcholInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrcholSolve(cusolverSpHandle_t handle, int n,
                                                                     const cuDoubleComplex *b, cuDoubleComplex *x,
                                                                     csrcholInfo_t info, void *pBuffer) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholSolve");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, int, const cuDoubleComplex *, cuDoubleComplex *,
                                          csrcholInfo_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholSolve"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, b, x, info, pBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpScsrcholDiag(cusolverSpHandle_t handle, csrcholInfo_t info,
                                                                    float *diag) {
    HOOK_TRACE_PROFILE("cusolverSpScsrcholDiag");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfo_t, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpScsrcholDiag"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, diag);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpDcsrcholDiag(cusolverSpHandle_t handle, csrcholInfo_t info,
                                                                    double *diag) {
    HOOK_TRACE_PROFILE("cusolverSpDcsrcholDiag");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfo_t, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpDcsrcholDiag"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, diag);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpCcsrcholDiag(cusolverSpHandle_t handle, csrcholInfo_t info,
                                                                    float *diag) {
    HOOK_TRACE_PROFILE("cusolverSpCcsrcholDiag");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfo_t, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpCcsrcholDiag"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, diag);
}

HOOK_C_API HOOK_DECL_EXPORT cusolverStatus_t cusolverSpZcsrcholDiag(cusolverSpHandle_t handle, csrcholInfo_t info,
                                                                    double *diag) {
    HOOK_TRACE_PROFILE("cusolverSpZcsrcholDiag");
    using func_ptr = cusolverStatus_t (*)(cusolverSpHandle_t, csrcholInfo_t, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUSOLVER_SYMBOL("cusolverSpZcsrcholDiag"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, info, diag);
}
