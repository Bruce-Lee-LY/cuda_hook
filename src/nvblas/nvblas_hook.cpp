// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:53:42 on Thu, Jul 21, 2022
//
// Description: auto generate 60 apis

#include "cublas_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT void sgemm_(const char *transa, const char *transb, const int *m, const int *n,
                                        const int *k, const float *alpha, const float *a, const int *lda,
                                        const float *b, const int *ldb, const float *beta, float *c, const int *ldc) {
    HOOK_TRACE_PROFILE("sgemm_");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const int *, const float *, const float *,
                 const int *, const float *, const int *, const float *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("sgemm_"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
                                        const int *k, const double *alpha, const double *a, const int *lda,
                                        const double *b, const int *ldb, const double *beta, double *c,
                                        const int *ldc) {
    HOOK_TRACE_PROFILE("dgemm_");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const int *, const double *, const double *,
                 const int *, const double *, const int *, const double *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dgemm_"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cgemm_(const char *transa, const char *transb, const int *m, const int *n,
                                        const int *k, const cuComplex *alpha, const cuComplex *a, const int *lda,
                                        const cuComplex *b, const int *ldb, const cuComplex *beta, cuComplex *c,
                                        const int *ldc) {
    HOOK_TRACE_PROFILE("cgemm_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const int *, const cuComplex *,
                              const cuComplex *, const int *, const cuComplex *, const int *, const cuComplex *,
                              cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("cgemm_"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zgemm_(const char *transa, const char *transb, const int *m, const int *n,
                                        const int *k, const cuDoubleComplex *alpha, const cuDoubleComplex *a,
                                        const int *lda, const cuDoubleComplex *b, const int *ldb,
                                        const cuDoubleComplex *beta, cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zgemm_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const int *,
                              const cuDoubleComplex *, const cuDoubleComplex *, const int *, const cuDoubleComplex *,
                              const int *, const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zgemm_"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void sgemm(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                                       const float *alpha, const float *a, const int *lda, const float *b,
                                       const int *ldb, const float *beta, float *c, const int *ldc) {
    HOOK_TRACE_PROFILE("sgemm");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const int *, const float *, const float *,
                 const int *, const float *, const int *, const float *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("sgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void dgemm(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                                       const double *alpha, const double *a, const int *lda, const double *b,
                                       const int *ldb, const double *beta, double *c, const int *ldc) {
    HOOK_TRACE_PROFILE("dgemm");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const int *, const double *, const double *,
                 const int *, const double *, const int *, const double *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cgemm(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                                       const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                       const int *ldb, const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("cgemm");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const int *, const cuComplex *,
                              const cuComplex *, const int *, const cuComplex *, const int *, const cuComplex *,
                              cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("cgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zgemm(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                                       const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                       const cuDoubleComplex *b, const int *ldb, const cuDoubleComplex *beta,
                                       cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zgemm");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const int *,
                              const cuDoubleComplex *, const cuDoubleComplex *, const int *, const cuDoubleComplex *,
                              const int *, const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void ssyrk_(const char *uplo, const char *trans, const int *n, const int *k,
                                        const float *alpha, const float *a, const int *lda, const float *beta, float *c,
                                        const int *ldc) {
    HOOK_TRACE_PROFILE("ssyrk_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const float *, const float *,
                              const int *, const float *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ssyrk_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
                                        const double *alpha, const double *a, const int *lda, const double *beta,
                                        double *c, const int *ldc) {
    HOOK_TRACE_PROFILE("dsyrk_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const double *, const double *,
                              const int *, const double *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dsyrk_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void csyrk_(const char *uplo, const char *trans, const int *n, const int *k,
                                        const cuComplex *alpha, const cuComplex *a, const int *lda,
                                        const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("csyrk_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const cuComplex *,
                              const cuComplex *, const int *, const cuComplex *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("csyrk_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zsyrk_(const char *uplo, const char *trans, const int *n, const int *k,
                                        const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                        const cuDoubleComplex *beta, cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zsyrk_");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *, const cuDoubleComplex *,
                 const int *, const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zsyrk_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void ssyrk(const char *uplo, const char *trans, const int *n, const int *k,
                                       const float *alpha, const float *a, const int *lda, const float *beta, float *c,
                                       const int *ldc) {
    HOOK_TRACE_PROFILE("ssyrk");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const float *, const float *,
                              const int *, const float *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ssyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void dsyrk(const char *uplo, const char *trans, const int *n, const int *k,
                                       const double *alpha, const double *a, const int *lda, const double *beta,
                                       double *c, const int *ldc) {
    HOOK_TRACE_PROFILE("dsyrk");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const double *, const double *,
                              const int *, const double *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void csyrk(const char *uplo, const char *trans, const int *n, const int *k,
                                       const cuComplex *alpha, const cuComplex *a, const int *lda,
                                       const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("csyrk");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const cuComplex *,
                              const cuComplex *, const int *, const cuComplex *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("csyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zsyrk(const char *uplo, const char *trans, const int *n, const int *k,
                                       const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                       const cuDoubleComplex *beta, cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zsyrk");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *, const cuDoubleComplex *,
                 const int *, const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cherk_(const char *uplo, const char *trans, const int *n, const int *k,
                                        const float *alpha, const cuComplex *a, const int *lda, const float *beta,
                                        cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("cherk_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const float *, const cuComplex *,
                              const int *, const float *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("cherk_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zherk_(const char *uplo, const char *trans, const int *n, const int *k,
                                        const double *alpha, const cuDoubleComplex *a, const int *lda,
                                        const double *beta, cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zherk_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const double *,
                              const cuDoubleComplex *, const int *, const double *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zherk_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cherk(const char *uplo, const char *trans, const int *n, const int *k,
                                       const float *alpha, const cuComplex *a, const int *lda, const float *beta,
                                       cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("cherk");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const float *, const cuComplex *,
                              const int *, const float *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("cherk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zherk(const char *uplo, const char *trans, const int *n, const int *k,
                                       const double *alpha, const cuDoubleComplex *a, const int *lda,
                                       const double *beta, cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zherk");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const double *,
                              const cuDoubleComplex *, const int *, const double *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zherk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void strsm_(const char *side, const char *uplo, const char *transa, const char *diag,
                                        const int *m, const int *n, const float *alpha, const float *a, const int *lda,
                                        float *b, const int *ldb) {
    HOOK_TRACE_PROFILE("strsm_");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const float *, const float *, const int *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("strsm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
                                        const int *m, const int *n, const double *alpha, const double *a,
                                        const int *lda, double *b, const int *ldb) {
    HOOK_TRACE_PROFILE("dtrsm_");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const double *, const double *, const int *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dtrsm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ctrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
                                        const int *m, const int *n, const cuComplex *alpha, const cuComplex *a,
                                        const int *lda, cuComplex *b, const int *ldb) {
    HOOK_TRACE_PROFILE("ctrsm_");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const cuComplex *, const cuComplex *, const int *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ctrsm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ztrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
                                        const int *m, const int *n, const cuDoubleComplex *alpha,
                                        const cuDoubleComplex *a, const int *lda, cuDoubleComplex *b, const int *ldb) {
    HOOK_TRACE_PROFILE("ztrsm_");
    using func_ptr =
        void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                 const cuDoubleComplex *, const cuDoubleComplex *, const int *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ztrsm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void strsm(const char *side, const char *uplo, const char *transa, const char *diag,
                                       const int *m, const int *n, const float *alpha, const float *a, const int *lda,
                                       float *b, const int *ldb) {
    HOOK_TRACE_PROFILE("strsm");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const float *, const float *, const int *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("strsm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
                                       const int *m, const int *n, const double *alpha, const double *a, const int *lda,
                                       double *b, const int *ldb) {
    HOOK_TRACE_PROFILE("dtrsm");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const double *, const double *, const int *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dtrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ctrsm(const char *side, const char *uplo, const char *transa, const char *diag,
                                       const int *m, const int *n, const cuComplex *alpha, const cuComplex *a,
                                       const int *lda, cuComplex *b, const int *ldb) {
    HOOK_TRACE_PROFILE("ctrsm");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const cuComplex *, const cuComplex *, const int *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ctrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ztrsm(const char *side, const char *uplo, const char *transa, const char *diag,
                                       const int *m, const int *n, const cuDoubleComplex *alpha,
                                       const cuDoubleComplex *a, const int *lda, cuDoubleComplex *b, const int *ldb) {
    HOOK_TRACE_PROFILE("ztrsm");
    using func_ptr =
        void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                 const cuDoubleComplex *, const cuDoubleComplex *, const int *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ztrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ssymm_(const char *side, const char *uplo, const int *m, const int *n,
                                        const float *alpha, const float *a, const int *lda, const float *b,
                                        const int *ldb, const float *beta, float *c, const int *ldc) {
    HOOK_TRACE_PROFILE("ssymm_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const float *, const float *,
                              const int *, const float *, const int *, const float *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ssymm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void dsymm_(const char *side, const char *uplo, const int *m, const int *n,
                                        const double *alpha, const double *a, const int *lda, const double *b,
                                        const int *ldb, const double *beta, double *c, const int *ldc) {
    HOOK_TRACE_PROFILE("dsymm_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const double *, const double *,
                              const int *, const double *, const int *, const double *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dsymm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void csymm_(const char *side, const char *uplo, const int *m, const int *n,
                                        const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                        const int *ldb, const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("csymm_");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuComplex *, const cuComplex *,
                 const int *, const cuComplex *, const int *, const cuComplex *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("csymm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zsymm_(const char *side, const char *uplo, const int *m, const int *n,
                                        const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                        const cuDoubleComplex *b, const int *ldb, const cuDoubleComplex *beta,
                                        cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zsymm_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *,
                              const cuDoubleComplex *, const int *, const cuDoubleComplex *, const int *,
                              const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zsymm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void ssymm(const char *side, const char *uplo, const int *m, const int *n,
                                       const float *alpha, const float *a, const int *lda, const float *b,
                                       const int *ldb, const float *beta, float *c, const int *ldc) {
    HOOK_TRACE_PROFILE("ssymm");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const float *, const float *,
                              const int *, const float *, const int *, const float *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ssymm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void dsymm(const char *side, const char *uplo, const int *m, const int *n,
                                       const double *alpha, const double *a, const int *lda, const double *b,
                                       const int *ldb, const double *beta, double *c, const int *ldc) {
    HOOK_TRACE_PROFILE("dsymm");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const double *, const double *,
                              const int *, const double *, const int *, const double *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void csymm(const char *side, const char *uplo, const int *m, const int *n,
                                       const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                       const int *ldb, const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("csymm");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuComplex *, const cuComplex *,
                 const int *, const cuComplex *, const int *, const cuComplex *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("csymm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zsymm(const char *side, const char *uplo, const int *m, const int *n,
                                       const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                       const cuDoubleComplex *b, const int *ldb, const cuDoubleComplex *beta,
                                       cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zsymm");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *,
                              const cuDoubleComplex *, const int *, const cuDoubleComplex *, const int *,
                              const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void chemm_(const char *side, const char *uplo, const int *m, const int *n,
                                        const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                        const int *ldb, const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("chemm_");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuComplex *, const cuComplex *,
                 const int *, const cuComplex *, const int *, const cuComplex *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("chemm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zhemm_(const char *side, const char *uplo, const int *m, const int *n,
                                        const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                        const cuDoubleComplex *b, const int *ldb, const cuDoubleComplex *beta,
                                        cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zhemm_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *,
                              const cuDoubleComplex *, const int *, const cuDoubleComplex *, const int *,
                              const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zhemm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void chemm(const char *side, const char *uplo, const int *m, const int *n,
                                       const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                       const int *ldb, const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("chemm");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuComplex *, const cuComplex *,
                 const int *, const cuComplex *, const int *, const cuComplex *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("chemm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zhemm(const char *side, const char *uplo, const int *m, const int *n,
                                       const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                       const cuDoubleComplex *b, const int *ldb, const cuDoubleComplex *beta,
                                       cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zhemm");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *,
                              const cuDoubleComplex *, const int *, const cuDoubleComplex *, const int *,
                              const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zhemm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void ssyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
                                         const float *alpha, const float *a, const int *lda, const float *b,
                                         const int *ldb, const float *beta, float *c, const int *ldc) {
    HOOK_TRACE_PROFILE("ssyr2k_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const float *, const float *,
                              const int *, const float *, const int *, const float *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ssyr2k_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void dsyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
                                         const double *alpha, const double *a, const int *lda, const double *b,
                                         const int *ldb, const double *beta, double *c, const int *ldc) {
    HOOK_TRACE_PROFILE("dsyr2k_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const double *, const double *,
                              const int *, const double *, const int *, const double *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dsyr2k_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void csyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
                                         const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                         const int *ldb, const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("csyr2k_");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuComplex *, const cuComplex *,
                 const int *, const cuComplex *, const int *, const cuComplex *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("csyr2k_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zsyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
                                         const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                         const cuDoubleComplex *b, const int *ldb, const cuDoubleComplex *beta,
                                         cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zsyr2k_");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *,
                              const cuDoubleComplex *, const int *, const cuDoubleComplex *, const int *,
                              const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zsyr2k_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void ssyr2k(const char *uplo, const char *trans, const int *n, const int *k,
                                        const float *alpha, const float *a, const int *lda, const float *b,
                                        const int *ldb, const float *beta, float *c, const int *ldc) {
    HOOK_TRACE_PROFILE("ssyr2k");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const float *, const float *,
                              const int *, const float *, const int *, const float *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ssyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void dsyr2k(const char *uplo, const char *trans, const int *n, const int *k,
                                        const double *alpha, const double *a, const int *lda, const double *b,
                                        const int *ldb, const double *beta, double *c, const int *ldc) {
    HOOK_TRACE_PROFILE("dsyr2k");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const double *, const double *,
                              const int *, const double *, const int *, const double *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void csyr2k(const char *uplo, const char *trans, const int *n, const int *k,
                                        const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                        const int *ldb, const cuComplex *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("csyr2k");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuComplex *, const cuComplex *,
                 const int *, const cuComplex *, const int *, const cuComplex *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("csyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zsyr2k(const char *uplo, const char *trans, const int *n, const int *k,
                                        const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                        const cuDoubleComplex *b, const int *ldb, const cuDoubleComplex *beta,
                                        cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zsyr2k");
    using func_ptr = void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *,
                              const cuDoubleComplex *, const int *, const cuDoubleComplex *, const int *,
                              const cuDoubleComplex *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cher2k_(const char *uplo, const char *trans, const int *n, const int *k,
                                         const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                         const int *ldb, const float *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("cher2k_");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuComplex *, const cuComplex *,
                 const int *, const cuComplex *, const int *, const float *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("cher2k_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zher2k_(const char *uplo, const char *trans, const int *n, const int *k,
                                         const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                         const cuDoubleComplex *b, const int *ldb, const double *beta,
                                         cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zher2k_");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *, const cuDoubleComplex *,
                 const int *, const cuDoubleComplex *, const int *, const double *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zher2k_"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cher2k(const char *uplo, const char *trans, const int *n, const int *k,
                                        const cuComplex *alpha, const cuComplex *a, const int *lda, const cuComplex *b,
                                        const int *ldb, const float *beta, cuComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("cher2k");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuComplex *, const cuComplex *,
                 const int *, const cuComplex *, const int *, const float *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("cher2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void zher2k(const char *uplo, const char *trans, const int *n, const int *k,
                                        const cuDoubleComplex *alpha, const cuDoubleComplex *a, const int *lda,
                                        const cuDoubleComplex *b, const int *ldb, const double *beta,
                                        cuDoubleComplex *c, const int *ldc) {
    HOOK_TRACE_PROFILE("zher2k");
    using func_ptr =
        void (*)(const char *, const char *, const int *, const int *, const cuDoubleComplex *, const cuDoubleComplex *,
                 const int *, const cuDoubleComplex *, const int *, const double *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("zher2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void strmm_(const char *side, const char *uplo, const char *transa, const char *diag,
                                        const int *m, const int *n, const float *alpha, const float *a, const int *lda,
                                        float *b, const int *ldb) {
    HOOK_TRACE_PROFILE("strmm_");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const float *, const float *, const int *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("strmm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
                                        const int *m, const int *n, const double *alpha, const double *a,
                                        const int *lda, double *b, const int *ldb) {
    HOOK_TRACE_PROFILE("dtrmm_");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const double *, const double *, const int *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dtrmm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ctrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
                                        const int *m, const int *n, const cuComplex *alpha, const cuComplex *a,
                                        const int *lda, cuComplex *b, const int *ldb) {
    HOOK_TRACE_PROFILE("ctrmm_");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const cuComplex *, const cuComplex *, const int *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ctrmm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ztrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
                                        const int *m, const int *n, const cuDoubleComplex *alpha,
                                        const cuDoubleComplex *a, const int *lda, cuDoubleComplex *b, const int *ldb) {
    HOOK_TRACE_PROFILE("ztrmm_");
    using func_ptr =
        void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                 const cuDoubleComplex *, const cuDoubleComplex *, const int *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ztrmm_"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void strmm(const char *side, const char *uplo, const char *transa, const char *diag,
                                       const int *m, const int *n, const float *alpha, const float *a, const int *lda,
                                       float *b, const int *ldb) {
    HOOK_TRACE_PROFILE("strmm");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const float *, const float *, const int *, float *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("strmm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void dtrmm(const char *side, const char *uplo, const char *transa, const char *diag,
                                       const int *m, const int *n, const double *alpha, const double *a, const int *lda,
                                       double *b, const int *ldb) {
    HOOK_TRACE_PROFILE("dtrmm");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const double *, const double *, const int *, double *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("dtrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ctrmm(const char *side, const char *uplo, const char *transa, const char *diag,
                                       const int *m, const int *n, const cuComplex *alpha, const cuComplex *a,
                                       const int *lda, cuComplex *b, const int *ldb) {
    HOOK_TRACE_PROFILE("ctrmm");
    using func_ptr = void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                              const cuComplex *, const cuComplex *, const int *, cuComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ctrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void ztrmm(const char *side, const char *uplo, const char *transa, const char *diag,
                                       const int *m, const int *n, const cuDoubleComplex *alpha,
                                       const cuDoubleComplex *a, const int *lda, cuDoubleComplex *b, const int *ldb) {
    HOOK_TRACE_PROFILE("ztrmm");
    using func_ptr =
        void (*)(const char *, const char *, const char *, const char *, const int *, const int *,
                 const cuDoubleComplex *, const cuDoubleComplex *, const int *, cuDoubleComplex *, const int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVBLAS_SYMBOL("ztrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}
