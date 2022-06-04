// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 485 apis

#include "cublas_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT cublasStatus cublasInit() {
    HOOK_TRACE_PROFILE("cublasInit");
    using func_ptr = cublasStatus (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasInit"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus cublasShutdown() {
    HOOK_TRACE_PROFILE("cublasShutdown");
    using func_ptr = cublasStatus (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasShutdown"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus cublasGetError() {
    HOOK_TRACE_PROFILE("cublasGetError");
    using func_ptr = cublasStatus (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetError"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus cublasGetVersion(int *version) {
    HOOK_TRACE_PROFILE("cublasGetVersion");
    using func_ptr = cublasStatus (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(version);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus cublasAlloc(int n, int elemSize, void **devicePtr) {
    HOOK_TRACE_PROFILE("cublasAlloc");
    using func_ptr = cublasStatus (*)(int, int, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasAlloc"));
    HOOK_CHECK(func_entry);
    return func_entry(n, elemSize, devicePtr);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus cublasFree(void *devicePtr) {
    HOOK_TRACE_PROFILE("cublasFree");
    using func_ptr = cublasStatus (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasFree"));
    HOOK_CHECK(func_entry);
    return func_entry(devicePtr);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus cublasSetKernelStream(cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cublasSetKernelStream");
    using func_ptr = cublasStatus (*)(cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetKernelStream"));
    HOOK_CHECK(func_entry);
    return func_entry(stream);
}

HOOK_C_API HOOK_DECL_EXPORT float cublasSnrm2(int n, const float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasSnrm2");
    using func_ptr = float (*)(int, const float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSnrm2"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT double cublasDnrm2(int n, const double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDnrm2");
    using func_ptr = double (*)(int, const double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDnrm2"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT float cublasScnrm2(int n, const cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasScnrm2");
    using func_ptr = float (*)(int, const cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasScnrm2"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT double cublasDznrm2(int n, const cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDznrm2");
    using func_ptr = double (*)(int, const cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDznrm2"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT float cublasSdot(int n, const float *x, int incx, const float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSdot");
    using func_ptr = float (*)(int, const float *, int, const float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSdot"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT double cublasDdot(int n, const double *x, int incx, const double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDdot");
    using func_ptr = double (*)(int, const double *, int, const double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDdot"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cuComplex cublasCdotu(int n, const cuComplex *x, int incx, const cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCdotu");
    using func_ptr = cuComplex (*)(int, const cuComplex *, int, const cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCdotu"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cuComplex cublasCdotc(int n, const cuComplex *x, int incx, const cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCdotc");
    using func_ptr = cuComplex (*)(int, const cuComplex *, int, const cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCdotc"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cuDoubleComplex cublasZdotu(int n, const cuDoubleComplex *x, int incx,
                                                        const cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZdotu");
    using func_ptr = cuDoubleComplex (*)(int, const cuDoubleComplex *, int, const cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdotu"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cuDoubleComplex cublasZdotc(int n, const cuDoubleComplex *x, int incx,
                                                        const cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZdotc");
    using func_ptr = cuDoubleComplex (*)(int, const cuDoubleComplex *, int, const cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdotc"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSscal(int n, float alpha, float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasSscal");
    using func_ptr = void (*)(int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSscal"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDscal(int n, double alpha, double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDscal");
    using func_ptr = void (*)(int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDscal"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCscal(int n, cuComplex alpha, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCscal");
    using func_ptr = void (*)(int, cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCscal"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZscal(int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZscal");
    using func_ptr = void (*)(int, cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZscal"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCsscal(int n, float alpha, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCsscal");
    using func_ptr = void (*)(int, float, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsscal"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZdscal(int n, double alpha, cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZdscal");
    using func_ptr = void (*)(int, double, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdscal"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSaxpy(int n, float alpha, const float *x, int incx, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSaxpy");
    using func_ptr = void (*)(int, float, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSaxpy"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDaxpy(int n, double alpha, const double *x, int incx, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDaxpy");
    using func_ptr = void (*)(int, double, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDaxpy"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCaxpy(int n, cuComplex alpha, const cuComplex *x, int incx, cuComplex *y,
                                             int incy) {
    HOOK_TRACE_PROFILE("cublasCaxpy");
    using func_ptr = void (*)(int, cuComplex, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCaxpy"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZaxpy(int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
                                             cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZaxpy");
    using func_ptr = void (*)(int, cuDoubleComplex, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZaxpy"));
    HOOK_CHECK(func_entry);
    return func_entry(n, alpha, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasScopy(int n, const float *x, int incx, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasScopy");
    using func_ptr = void (*)(int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasScopy"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDcopy(int n, const double *x, int incx, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDcopy");
    using func_ptr = void (*)(int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDcopy"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCcopy(int n, const cuComplex *x, int incx, cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCcopy");
    using func_ptr = void (*)(int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCcopy"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZcopy(int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZcopy");
    using func_ptr = void (*)(int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZcopy"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSswap(int n, float *x, int incx, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSswap");
    using func_ptr = void (*)(int, float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSswap"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDswap(int n, double *x, int incx, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDswap");
    using func_ptr = void (*)(int, double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDswap"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCswap(int n, cuComplex *x, int incx, cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCswap");
    using func_ptr = void (*)(int, cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCswap"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZswap(int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZswap");
    using func_ptr = void (*)(int, cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZswap"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT int cublasIsamax(int n, const float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasIsamax");
    using func_ptr = int (*)(int, const float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIsamax"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT int cublasIdamax(int n, const double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasIdamax");
    using func_ptr = int (*)(int, const double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIdamax"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT int cublasIcamax(int n, const cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasIcamax");
    using func_ptr = int (*)(int, const cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIcamax"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT int cublasIzamax(int n, const cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasIzamax");
    using func_ptr = int (*)(int, const cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIzamax"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT int cublasIsamin(int n, const float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasIsamin");
    using func_ptr = int (*)(int, const float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIsamin"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT int cublasIdamin(int n, const double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasIdamin");
    using func_ptr = int (*)(int, const double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIdamin"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT int cublasIcamin(int n, const cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasIcamin");
    using func_ptr = int (*)(int, const cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIcamin"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT int cublasIzamin(int n, const cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasIzamin");
    using func_ptr = int (*)(int, const cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIzamin"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT float cublasSasum(int n, const float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasSasum");
    using func_ptr = float (*)(int, const float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSasum"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT double cublasDasum(int n, const double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDasum");
    using func_ptr = double (*)(int, const double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDasum"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT float cublasScasum(int n, const cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasScasum");
    using func_ptr = float (*)(int, const cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasScasum"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT double cublasDzasum(int n, const cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDzasum");
    using func_ptr = double (*)(int, const cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDzasum"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSrot(int n, float *x, int incx, float *y, int incy, float sc, float ss) {
    HOOK_TRACE_PROFILE("cublasSrot");
    using func_ptr = void (*)(int, float *, int, float *, int, float, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSrot"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy, sc, ss);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDrot(int n, double *x, int incx, double *y, int incy, double sc, double ss) {
    HOOK_TRACE_PROFILE("cublasDrot");
    using func_ptr = void (*)(int, double *, int, double *, int, double, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDrot"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy, sc, ss);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCrot(int n, cuComplex *x, int incx, cuComplex *y, int incy, float c,
                                            cuComplex s) {
    HOOK_TRACE_PROFILE("cublasCrot");
    using func_ptr = void (*)(int, cuComplex *, int, cuComplex *, int, float, cuComplex);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCrot"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZrot(int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy,
                                            double sc, cuDoubleComplex cs) {
    HOOK_TRACE_PROFILE("cublasZrot");
    using func_ptr = void (*)(int, cuDoubleComplex *, int, cuDoubleComplex *, int, double, cuDoubleComplex);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZrot"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy, sc, cs);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCsrot(int n, cuComplex *x, int incx, cuComplex *y, int incy, float c, float s) {
    HOOK_TRACE_PROFILE("cublasCsrot");
    using func_ptr = void (*)(int, cuComplex *, int, cuComplex *, int, float, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsrot"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZdrot(int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy,
                                             double c, double s) {
    HOOK_TRACE_PROFILE("cublasZdrot");
    using func_ptr = void (*)(int, cuDoubleComplex *, int, cuDoubleComplex *, int, double, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdrot"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSrotg(float *sa, float *sb, float *sc, float *ss) {
    HOOK_TRACE_PROFILE("cublasSrotg");
    using func_ptr = void (*)(float *, float *, float *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSrotg"));
    HOOK_CHECK(func_entry);
    return func_entry(sa, sb, sc, ss);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDrotg(double *sa, double *sb, double *sc, double *ss) {
    HOOK_TRACE_PROFILE("cublasDrotg");
    using func_ptr = void (*)(double *, double *, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDrotg"));
    HOOK_CHECK(func_entry);
    return func_entry(sa, sb, sc, ss);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCrotg(cuComplex *ca, cuComplex cb, float *sc, cuComplex *cs) {
    HOOK_TRACE_PROFILE("cublasCrotg");
    using func_ptr = void (*)(cuComplex *, cuComplex, float *, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCrotg"));
    HOOK_CHECK(func_entry);
    return func_entry(ca, cb, sc, cs);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZrotg(cuDoubleComplex *ca, cuDoubleComplex cb, double *sc, cuDoubleComplex *cs) {
    HOOK_TRACE_PROFILE("cublasZrotg");
    using func_ptr = void (*)(cuDoubleComplex *, cuDoubleComplex, double *, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZrotg"));
    HOOK_CHECK(func_entry);
    return func_entry(ca, cb, sc, cs);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSrotm(int n, float *x, int incx, float *y, int incy, const float *sparam) {
    HOOK_TRACE_PROFILE("cublasSrotm");
    using func_ptr = void (*)(int, float *, int, float *, int, const float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSrotm"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy, sparam);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDrotm(int n, double *x, int incx, double *y, int incy, const double *sparam) {
    HOOK_TRACE_PROFILE("cublasDrotm");
    using func_ptr = void (*)(int, double *, int, double *, int, const double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDrotm"));
    HOOK_CHECK(func_entry);
    return func_entry(n, x, incx, y, incy, sparam);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSrotmg(float *sd1, float *sd2, float *sx1, const float *sy1, float *sparam) {
    HOOK_TRACE_PROFILE("cublasSrotmg");
    using func_ptr = void (*)(float *, float *, float *, const float *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSrotmg"));
    HOOK_CHECK(func_entry);
    return func_entry(sd1, sd2, sx1, sy1, sparam);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDrotmg(double *sd1, double *sd2, double *sx1, const double *sy1,
                                              double *sparam) {
    HOOK_TRACE_PROFILE("cublasDrotmg");
    using func_ptr = void (*)(double *, double *, double *, const double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDrotmg"));
    HOOK_CHECK(func_entry);
    return func_entry(sd1, sd2, sx1, sy1, sparam);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSgemv(char trans, int m, int n, float alpha, const float *A, int lda,
                                             const float *x, int incx, float beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSgemv");
    using func_ptr = void (*)(char, int, int, float, const float *, int, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgemv"));
    HOOK_CHECK(func_entry);
    return func_entry(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDgemv(char trans, int m, int n, double alpha, const double *A, int lda,
                                             const double *x, int incx, double beta, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDgemv");
    using func_ptr = void (*)(char, int, int, double, const double *, int, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgemv"));
    HOOK_CHECK(func_entry);
    return func_entry(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCgemv(char trans, int m, int n, cuComplex alpha, const cuComplex *A, int lda,
                                             const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCgemv");
    using func_ptr = void (*)(char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex,
                              cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemv"));
    HOOK_CHECK(func_entry);
    return func_entry(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZgemv(char trans, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *A,
                                             int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta,
                                             cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZgemv");
    using func_ptr = void (*)(char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *,
                              int, cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgemv"));
    HOOK_CHECK(func_entry);
    return func_entry(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float *A,
                                             int lda, const float *x, int incx, float beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSgbmv");
    using func_ptr =
        void (*)(char, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double *A,
                                             int lda, const double *x, int incx, double beta, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDgbmv");
    using func_ptr =
        void (*)(char, int, int, int, int, double, const double *, int, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCgbmv(char trans, int m, int n, int kl, int ku, cuComplex alpha,
                                             const cuComplex *A, int lda, const cuComplex *x, int incx, cuComplex beta,
                                             cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCgbmv");
    using func_ptr = void (*)(char, int, int, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int,
                              cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZgbmv(char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha,
                                             const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx,
                                             cuDoubleComplex beta, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZgbmv");
    using func_ptr = void (*)(char, int, int, int, int, cuDoubleComplex, const cuDoubleComplex *, int,
                              const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasStrmv(char uplo, char trans, char diag, int n, const float *A, int lda, float *x,
                                             int incx) {
    HOOK_TRACE_PROFILE("cublasStrmv");
    using func_ptr = void (*)(char, char, char, int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDtrmv(char uplo, char trans, char diag, int n, const double *A, int lda,
                                             double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtrmv");
    using func_ptr = void (*)(char, char, char, int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCtrmv(char uplo, char trans, char diag, int n, const cuComplex *A, int lda,
                                             cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtrmv");
    using func_ptr = void (*)(char, char, char, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZtrmv(char uplo, char trans, char diag, int n, const cuDoubleComplex *A, int lda,
                                             cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZtrmv");
    using func_ptr = void (*)(char, char, char, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasStbmv(char uplo, char trans, char diag, int n, int k, const float *A, int lda,
                                             float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasStbmv");
    using func_ptr = void (*)(char, char, char, int, int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDtbmv(char uplo, char trans, char diag, int n, int k, const double *A, int lda,
                                             double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtbmv");
    using func_ptr = void (*)(char, char, char, int, int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCtbmv(char uplo, char trans, char diag, int n, int k, const cuComplex *A,
                                             int lda, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtbmv");
    using func_ptr = void (*)(char, char, char, int, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZtbmv(char uplo, char trans, char diag, int n, int k, const cuDoubleComplex *A,
                                             int lda, cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZtbmv");
    using func_ptr = void (*)(char, char, char, int, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasStpmv(char uplo, char trans, char diag, int n, const float *AP, float *x,
                                             int incx) {
    HOOK_TRACE_PROFILE("cublasStpmv");
    using func_ptr = void (*)(char, char, char, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStpmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDtpmv(char uplo, char trans, char diag, int n, const double *AP, double *x,
                                             int incx) {
    HOOK_TRACE_PROFILE("cublasDtpmv");
    using func_ptr = void (*)(char, char, char, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtpmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCtpmv(char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x,
                                             int incx) {
    HOOK_TRACE_PROFILE("cublasCtpmv");
    using func_ptr = void (*)(char, char, char, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtpmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZtpmv(char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,
                                             cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZtpmv");
    using func_ptr = void (*)(char, char, char, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtpmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasStrsv(char uplo, char trans, char diag, int n, const float *A, int lda, float *x,
                                             int incx) {
    HOOK_TRACE_PROFILE("cublasStrsv");
    using func_ptr = void (*)(char, char, char, int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDtrsv(char uplo, char trans, char diag, int n, const double *A, int lda,
                                             double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtrsv");
    using func_ptr = void (*)(char, char, char, int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCtrsv(char uplo, char trans, char diag, int n, const cuComplex *A, int lda,
                                             cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtrsv");
    using func_ptr = void (*)(char, char, char, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZtrsv(char uplo, char trans, char diag, int n, const cuDoubleComplex *A, int lda,
                                             cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZtrsv");
    using func_ptr = void (*)(char, char, char, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasStpsv(char uplo, char trans, char diag, int n, const float *AP, float *x,
                                             int incx) {
    HOOK_TRACE_PROFILE("cublasStpsv");
    using func_ptr = void (*)(char, char, char, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStpsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDtpsv(char uplo, char trans, char diag, int n, const double *AP, double *x,
                                             int incx) {
    HOOK_TRACE_PROFILE("cublasDtpsv");
    using func_ptr = void (*)(char, char, char, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtpsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCtpsv(char uplo, char trans, char diag, int n, const cuComplex *AP, cuComplex *x,
                                             int incx) {
    HOOK_TRACE_PROFILE("cublasCtpsv");
    using func_ptr = void (*)(char, char, char, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtpsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZtpsv(char uplo, char trans, char diag, int n, const cuDoubleComplex *AP,
                                             cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZtpsv");
    using func_ptr = void (*)(char, char, char, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtpsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasStbsv(char uplo, char trans, char diag, int n, int k, const float *A, int lda,
                                             float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasStbsv");
    using func_ptr = void (*)(char, char, char, int, int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStbsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDtbsv(char uplo, char trans, char diag, int n, int k, const double *A, int lda,
                                             double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtbsv");
    using func_ptr = void (*)(char, char, char, int, int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtbsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCtbsv(char uplo, char trans, char diag, int n, int k, const cuComplex *A,
                                             int lda, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtbsv");
    using func_ptr = void (*)(char, char, char, int, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtbsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZtbsv(char uplo, char trans, char diag, int n, int k, const cuDoubleComplex *A,
                                             int lda, cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZtbsv");
    using func_ptr = void (*)(char, char, char, int, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtbsv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSsymv(char uplo, int n, float alpha, const float *A, int lda, const float *x,
                                             int incx, float beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSsymv");
    using func_ptr = void (*)(char, int, float, const float *, int, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsymv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDsymv(char uplo, int n, double alpha, const double *A, int lda, const double *x,
                                             int incx, double beta, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDsymv");
    using func_ptr = void (*)(char, int, double, const double *, int, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsymv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasChemv(char uplo, int n, cuComplex alpha, const cuComplex *A, int lda,
                                             const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasChemv");
    using func_ptr =
        void (*)(char, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChemv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZhemv(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                             const cuDoubleComplex *x, int incx, cuDoubleComplex beta,
                                             cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZhemv");
    using func_ptr = void (*)(char, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                              cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhemv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSsbmv(char uplo, int n, int k, float alpha, const float *A, int lda,
                                             const float *x, int incx, float beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSsbmv");
    using func_ptr = void (*)(char, int, int, float, const float *, int, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDsbmv(char uplo, int n, int k, double alpha, const double *A, int lda,
                                             const double *x, int incx, double beta, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDsbmv");
    using func_ptr = void (*)(char, int, int, double, const double *, int, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasChbmv(char uplo, int n, int k, cuComplex alpha, const cuComplex *A, int lda,
                                             const cuComplex *x, int incx, cuComplex beta, cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasChbmv");
    using func_ptr = void (*)(char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex,
                              cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZhbmv(char uplo, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex *A,
                                             int lda, const cuDoubleComplex *x, int incx, cuDoubleComplex beta,
                                             cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZhbmv");
    using func_ptr = void (*)(char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *,
                              int, cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhbmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSspmv(char uplo, int n, float alpha, const float *AP, const float *x, int incx,
                                             float beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSspmv");
    using func_ptr = void (*)(char, int, float, const float *, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSspmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDspmv(char uplo, int n, double alpha, const double *AP, const double *x,
                                             int incx, double beta, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDspmv");
    using func_ptr = void (*)(char, int, double, const double *, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDspmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasChpmv(char uplo, int n, cuComplex alpha, const cuComplex *AP, const cuComplex *x,
                                             int incx, cuComplex beta, cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasChpmv");
    using func_ptr =
        void (*)(char, int, cuComplex, const cuComplex *, const cuComplex *, int, cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChpmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZhpmv(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *AP,
                                             const cuDoubleComplex *x, int incx, cuDoubleComplex beta,
                                             cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZhpmv");
    using func_ptr = void (*)(char, int, cuDoubleComplex, const cuDoubleComplex *, const cuDoubleComplex *, int,
                              cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhpmv"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, AP, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSger(int m, int n, float alpha, const float *x, int incx, const float *y,
                                            int incy, float *A, int lda) {
    HOOK_TRACE_PROFILE("cublasSger");
    using func_ptr = void (*)(int, int, float, const float *, int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSger"));
    HOOK_CHECK(func_entry);
    return func_entry(m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDger(int m, int n, double alpha, const double *x, int incx, const double *y,
                                            int incy, double *A, int lda) {
    HOOK_TRACE_PROFILE("cublasDger");
    using func_ptr = void (*)(int, int, double, const double *, int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDger"));
    HOOK_CHECK(func_entry);
    return func_entry(m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCgeru(int m, int n, cuComplex alpha, const cuComplex *x, int incx,
                                             const cuComplex *y, int incy, cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCgeru");
    using func_ptr = void (*)(int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgeru"));
    HOOK_CHECK(func_entry);
    return func_entry(m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCgerc(int m, int n, cuComplex alpha, const cuComplex *x, int incx,
                                             const cuComplex *y, int incy, cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCgerc");
    using func_ptr = void (*)(int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgerc"));
    HOOK_CHECK(func_entry);
    return func_entry(m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZgeru(int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
                                             const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZgeru");
    using func_ptr = void (*)(int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                              cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgeru"));
    HOOK_CHECK(func_entry);
    return func_entry(m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZgerc(int m, int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx,
                                             const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZgerc");
    using func_ptr = void (*)(int, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                              cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgerc"));
    HOOK_CHECK(func_entry);
    return func_entry(m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSsyr(char uplo, int n, float alpha, const float *x, int incx, float *A,
                                            int lda) {
    HOOK_TRACE_PROFILE("cublasSsyr");
    using func_ptr = void (*)(char, int, float, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyr"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDsyr(char uplo, int n, double alpha, const double *x, int incx, double *A,
                                            int lda) {
    HOOK_TRACE_PROFILE("cublasDsyr");
    using func_ptr = void (*)(char, int, double, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyr"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCher(char uplo, int n, float alpha, const cuComplex *x, int incx, cuComplex *A,
                                            int lda) {
    HOOK_TRACE_PROFILE("cublasCher");
    using func_ptr = void (*)(char, int, float, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCher"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZher(char uplo, int n, double alpha, const cuDoubleComplex *x, int incx,
                                            cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZher");
    using func_ptr = void (*)(char, int, double, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZher"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSspr(char uplo, int n, float alpha, const float *x, int incx, float *AP) {
    HOOK_TRACE_PROFILE("cublasSspr");
    using func_ptr = void (*)(char, int, float, const float *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSspr"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, AP);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDspr(char uplo, int n, double alpha, const double *x, int incx, double *AP) {
    HOOK_TRACE_PROFILE("cublasDspr");
    using func_ptr = void (*)(char, int, double, const double *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDspr"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, AP);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasChpr(char uplo, int n, float alpha, const cuComplex *x, int incx,
                                            cuComplex *AP) {
    HOOK_TRACE_PROFILE("cublasChpr");
    using func_ptr = void (*)(char, int, float, const cuComplex *, int, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChpr"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, AP);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZhpr(char uplo, int n, double alpha, const cuDoubleComplex *x, int incx,
                                            cuDoubleComplex *AP) {
    HOOK_TRACE_PROFILE("cublasZhpr");
    using func_ptr = void (*)(char, int, double, const cuDoubleComplex *, int, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhpr"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, AP);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSsyr2(char uplo, int n, float alpha, const float *x, int incx, const float *y,
                                             int incy, float *A, int lda) {
    HOOK_TRACE_PROFILE("cublasSsyr2");
    using func_ptr = void (*)(char, int, float, const float *, int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyr2"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDsyr2(char uplo, int n, double alpha, const double *x, int incx, const double *y,
                                             int incy, double *A, int lda) {
    HOOK_TRACE_PROFILE("cublasDsyr2");
    using func_ptr = void (*)(char, int, double, const double *, int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyr2"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCher2(char uplo, int n, cuComplex alpha, const cuComplex *x, int incx,
                                             const cuComplex *y, int incy, cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCher2");
    using func_ptr = void (*)(char, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCher2"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZher2(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,
                                             int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A,
                                             int lda) {
    HOOK_TRACE_PROFILE("cublasZher2");
    using func_ptr = void (*)(char, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                              cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZher2"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSspr2(char uplo, int n, float alpha, const float *x, int incx, const float *y,
                                             int incy, float *AP) {
    HOOK_TRACE_PROFILE("cublasSspr2");
    using func_ptr = void (*)(char, int, float, const float *, int, const float *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSspr2"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, y, incy, AP);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDspr2(char uplo, int n, double alpha, const double *x, int incx, const double *y,
                                             int incy, double *AP) {
    HOOK_TRACE_PROFILE("cublasDspr2");
    using func_ptr = void (*)(char, int, double, const double *, int, const double *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDspr2"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, y, incy, AP);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasChpr2(char uplo, int n, cuComplex alpha, const cuComplex *x, int incx,
                                             const cuComplex *y, int incy, cuComplex *AP) {
    HOOK_TRACE_PROFILE("cublasChpr2");
    using func_ptr = void (*)(char, int, cuComplex, const cuComplex *, int, const cuComplex *, int, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChpr2"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, y, incy, AP);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZhpr2(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex *x,
                                             int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP) {
    HOOK_TRACE_PROFILE("cublasZhpr2");
    using func_ptr = void (*)(char, int, cuDoubleComplex, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                              cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhpr2"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, n, alpha, x, incx, y, incy, AP);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A,
                                             int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSgemm");
    using func_ptr =
        void (*)(char, char, int, int, int, float, const float *, int, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDgemm(char transa, char transb, int m, int n, int k, double alpha,
                                             const double *A, int lda, const double *B, int ldb, double beta, double *C,
                                             int ldc) {
    HOOK_TRACE_PROFILE("cublasDgemm");
    using func_ptr =
        void (*)(char, char, int, int, int, double, const double *, int, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCgemm(char transa, char transb, int m, int n, int k, cuComplex alpha,
                                             const cuComplex *A, int lda, const cuComplex *B, int ldb, cuComplex beta,
                                             cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCgemm");
    using func_ptr = void (*)(char, char, int, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int,
                              cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZgemm(char transa, char transb, int m, int n, int k, cuDoubleComplex alpha,
                                             const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                                             cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZgemm");
    using func_ptr = void (*)(char, char, int, int, int, cuDoubleComplex, const cuDoubleComplex *, int,
                              const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSsyrk(char uplo, char trans, int n, int k, float alpha, const float *A, int lda,
                                             float beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSsyrk");
    using func_ptr = void (*)(char, char, int, int, float, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDsyrk(char uplo, char trans, int n, int k, double alpha, const double *A,
                                             int lda, double beta, double *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasDsyrk");
    using func_ptr = void (*)(char, char, int, int, double, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCsyrk(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A,
                                             int lda, cuComplex beta, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCsyrk");
    using func_ptr = void (*)(char, char, int, int, cuComplex, const cuComplex *, int, cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZsyrk(char uplo, char trans, int n, int k, cuDoubleComplex alpha,
                                             const cuDoubleComplex *A, int lda, cuDoubleComplex beta,
                                             cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZsyrk");
    using func_ptr = void (*)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int, cuDoubleComplex,
                              cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCherk(char uplo, char trans, int n, int k, float alpha, const cuComplex *A,
                                             int lda, float beta, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCherk");
    using func_ptr = void (*)(char, char, int, int, float, const cuComplex *, int, float, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCherk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZherk(char uplo, char trans, int n, int k, double alpha,
                                             const cuDoubleComplex *A, int lda, double beta, cuDoubleComplex *C,
                                             int ldc) {
    HOOK_TRACE_PROFILE("cublasZherk");
    using func_ptr =
        void (*)(char, char, int, int, double, const cuDoubleComplex *, int, double, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZherk"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSsyr2k(char uplo, char trans, int n, int k, float alpha, const float *A, int lda,
                                              const float *B, int ldb, float beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSsyr2k");
    using func_ptr = void (*)(char, char, int, int, float, const float *, int, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDsyr2k(char uplo, char trans, int n, int k, double alpha, const double *A,
                                              int lda, const double *B, int ldb, double beta, double *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasDsyr2k");
    using func_ptr =
        void (*)(char, char, int, int, double, const double *, int, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCsyr2k(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A,
                                              int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C,
                                              int ldc) {
    HOOK_TRACE_PROFILE("cublasCsyr2k");
    using func_ptr = void (*)(char, char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int,
                              cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZsyr2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha,
                                              const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                                              cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZsyr2k");
    using func_ptr = void (*)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int,
                              const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCher2k(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex *A,
                                              int lda, const cuComplex *B, int ldb, float beta, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCher2k");
    using func_ptr = void (*)(char, char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int, float,
                              cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCher2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZher2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha,
                                              const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                                              double beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZher2k");
    using func_ptr = void (*)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int,
                              const cuDoubleComplex *, int, double, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZher2k"));
    HOOK_CHECK(func_entry);
    return func_entry(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasSsymm(char side, char uplo, int m, int n, float alpha, const float *A, int lda,
                                             const float *B, int ldb, float beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSsymm");
    using func_ptr = void (*)(char, char, int, int, float, const float *, int, const float *, int, float, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDsymm(char side, char uplo, int m, int n, double alpha, const double *A, int lda,
                                             const double *B, int ldb, double beta, double *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasDsymm");
    using func_ptr =
        void (*)(char, char, int, int, double, const double *, int, const double *, int, double, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCsymm(char side, char uplo, int m, int n, cuComplex alpha, const cuComplex *A,
                                             int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C,
                                             int ldc) {
    HOOK_TRACE_PROFILE("cublasCsymm");
    using func_ptr = void (*)(char, char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int,
                              cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZsymm(char side, char uplo, int m, int n, cuDoubleComplex alpha,
                                             const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                                             cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZsymm");
    using func_ptr = void (*)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int,
                              const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasChemm(char side, char uplo, int m, int n, cuComplex alpha, const cuComplex *A,
                                             int lda, const cuComplex *B, int ldb, cuComplex beta, cuComplex *C,
                                             int ldc) {
    HOOK_TRACE_PROFILE("cublasChemm");
    using func_ptr = void (*)(char, char, int, int, cuComplex, const cuComplex *, int, const cuComplex *, int,
                              cuComplex, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChemm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZhemm(char side, char uplo, int m, int n, cuDoubleComplex alpha,
                                             const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb,
                                             cuDoubleComplex beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZhemm");
    using func_ptr = void (*)(char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int,
                              const cuDoubleComplex *, int, cuDoubleComplex, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhemm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasStrsm(char side, char uplo, char transa, char diag, int m, int n, float alpha,
                                             const float *A, int lda, float *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasStrsm");
    using func_ptr = void (*)(char, char, char, char, int, int, float, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha,
                                             const double *A, int lda, double *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasDtrsm");
    using func_ptr = void (*)(char, char, char, char, int, int, double, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCtrsm(char side, char uplo, char transa, char diag, int m, int n,
                                             cuComplex alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasCtrsm");
    using func_ptr = void (*)(char, char, char, char, int, int, cuComplex, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZtrsm(char side, char uplo, char transa, char diag, int m, int n,
                                             cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                             cuDoubleComplex *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasZtrsm");
    using func_ptr = void (*)(char, char, char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int,
                              cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasStrmm(char side, char uplo, char transa, char diag, int m, int n, float alpha,
                                             const float *A, int lda, float *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasStrmm");
    using func_ptr = void (*)(char, char, char, char, int, int, float, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasDtrmm(char side, char uplo, char transa, char diag, int m, int n, double alpha,
                                             const double *A, int lda, double *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasDtrmm");
    using func_ptr = void (*)(char, char, char, char, int, int, double, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasCtrmm(char side, char uplo, char transa, char diag, int m, int n,
                                             cuComplex alpha, const cuComplex *A, int lda, cuComplex *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasCtrmm");
    using func_ptr = void (*)(char, char, char, char, int, int, cuComplex, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasZtrmm(char side, char uplo, char transa, char diag, int m, int n,
                                             cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                             cuDoubleComplex *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasZtrmm");
    using func_ptr = void (*)(char, char, char, char, int, int, cuDoubleComplex, const cuDoubleComplex *, int,
                              cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    HOOK_TRACE_PROFILE("cublasCreate_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCreate_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    HOOK_TRACE_PROFILE("cublasDestroy_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDestroy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version) {
    HOOK_TRACE_PROFILE("cublasGetVersion_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetVersion_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, version);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("cublasGetProperty");
    using func_ptr = cublasStatus_t (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT size_t cublasGetCudartVersion() {
    HOOK_TRACE_PROFILE("cublasGetCudartVersion");
    using func_ptr = size_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetCudartVersion"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace,
                                                                 size_t workspaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cublasSetWorkspace_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetWorkspace_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, workspace, workspaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
    HOOK_TRACE_PROFILE("cublasSetStream_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetStream_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId) {
    HOOK_TRACE_PROFILE("cublasGetStream_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cudaStream_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetStream_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t *mode) {
    HOOK_TRACE_PROFILE("cublasGetPointerMode_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetPointerMode_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode) {
    HOOK_TRACE_PROFILE("cublasSetPointerMode_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasPointerMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetPointerMode_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode) {
    HOOK_TRACE_PROFILE("cublasGetAtomicsMode");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasAtomicsMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetAtomicsMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode) {
    HOOK_TRACE_PROFILE("cublasSetAtomicsMode");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasAtomicsMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetAtomicsMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    HOOK_TRACE_PROFILE("cublasGetMathMode");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasMath_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetMathMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    HOOK_TRACE_PROFILE("cublasSetMathMode");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasMath_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetMathMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetSmCountTarget(cublasHandle_t handle, int *smCountTarget) {
    HOOK_TRACE_PROFILE("cublasGetSmCountTarget");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetSmCountTarget"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, smCountTarget);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetSmCountTarget(cublasHandle_t handle, int smCountTarget) {
    HOOK_TRACE_PROFILE("cublasSetSmCountTarget");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetSmCountTarget"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, smCountTarget);
}

HOOK_C_API HOOK_DECL_EXPORT const char *cublasGetStatusName(cublasStatus_t status) {
    HOOK_TRACE_PROFILE("cublasGetStatusName");
    using func_ptr = const char *(*)(cublasStatus_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetStatusName"));
    HOOK_CHECK(func_entry);
    return func_entry(status);
}

HOOK_C_API HOOK_DECL_EXPORT const char *cublasGetStatusString(cublasStatus_t status) {
    HOOK_TRACE_PROFILE("cublasGetStatusString");
    using func_ptr = const char *(*)(cublasStatus_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetStatusString"));
    HOOK_CHECK(func_entry);
    return func_entry(status);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut, int logToStdErr,
                                                                 const char *logFileName) {
    HOOK_TRACE_PROFILE("cublasLoggerConfigure");
    using func_ptr = cublasStatus_t (*)(int, int, int, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasLoggerConfigure"));
    HOOK_CHECK(func_entry);
    return func_entry(logIsOn, logToStdOut, logToStdErr, logFileName);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback) {
    HOOK_TRACE_PROFILE("cublasSetLoggerCallback");
    using func_ptr = cublasStatus_t (*)(cublasLogCallback);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetLoggerCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(userCallback);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetLoggerCallback(cublasLogCallback *userCallback) {
    HOOK_TRACE_PROFILE("cublasGetLoggerCallback");
    using func_ptr = cublasStatus_t (*)(cublasLogCallback *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetLoggerCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(userCallback);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx,
                                                           void *devicePtr, int incy) {
    HOOK_TRACE_PROFILE("cublasSetVector");
    using func_ptr = cublasStatus_t (*)(int, int, const void *, int, void *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetVector"));
    HOOK_CHECK(func_entry);
    return func_entry(n, elemSize, x, incx, devicePtr, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx, void *y,
                                                           int incy) {
    HOOK_TRACE_PROFILE("cublasGetVector");
    using func_ptr = cublasStatus_t (*)(int, int, const void *, int, void *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetVector"));
    HOOK_CHECK(func_entry);
    return func_entry(n, elemSize, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda,
                                                           void *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasSetMatrix");
    using func_ptr = cublasStatus_t (*)(int, int, int, const void *, int, void *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetMatrix"));
    HOOK_CHECK(func_entry);
    return func_entry(rows, cols, elemSize, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A, int lda,
                                                           void *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasGetMatrix");
    using func_ptr = cublasStatus_t (*)(int, int, int, const void *, int, void *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetMatrix"));
    HOOK_CHECK(func_entry);
    return func_entry(rows, cols, elemSize, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr, int incx,
                                                                void *devicePtr, int incy, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cublasSetVectorAsync");
    using func_ptr = cublasStatus_t (*)(int, int, const void *, int, void *, int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetVectorAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr, int incx,
                                                                void *hostPtr, int incy, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cublasGetVectorAsync");
    using func_ptr = cublasStatus_t (*)(int, int, const void *, int, void *, int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetVectorAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void *A,
                                                                int lda, void *B, int ldb, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cublasSetMatrixAsync");
    using func_ptr = cublasStatus_t (*)(int, int, int, const void *, int, void *, int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSetMatrixAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(rows, cols, elemSize, A, lda, B, ldb, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void *A,
                                                                int lda, void *B, int ldb, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cublasGetMatrixAsync");
    using func_ptr = cublasStatus_t (*)(int, int, int, const void *, int, void *, int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGetMatrixAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(rows, cols, elemSize, A, lda, B, ldb, stream);
}

HOOK_C_API HOOK_DECL_EXPORT void cublasXerbla(const char *srName, int info) {
    HOOK_TRACE_PROFILE("cublasXerbla");
    using func_ptr = void (*)(const char *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXerbla"));
    HOOK_CHECK(func_entry);
    return func_entry(srName, info);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                        int incx, void *result, cudaDataType resultType,
                                                        cudaDataType executionType) {
    HOOK_TRACE_PROFILE("cublasNrm2Ex");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, int, void *, cudaDataType, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasNrm2Ex"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, result, resultType, executionType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                          float *result) {
    HOOK_TRACE_PROFILE("cublasSnrm2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSnrm2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                          double *result) {
    HOOK_TRACE_PROFILE("cublasDnrm2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDnrm2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                           float *result) {
    HOOK_TRACE_PROFILE("cublasScnrm2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasScnrm2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x,
                                                           int incx, double *result) {
    HOOK_TRACE_PROFILE("cublasDznrm2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDznrm2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                       int incx, const void *y, cudaDataType yType, int incy,
                                                       void *result, cudaDataType resultType,
                                                       cudaDataType executionType) {
    HOOK_TRACE_PROFILE("cublasDotEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, int, const void *,
                                        cudaDataType, int, void *, cudaDataType, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDotEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                        int incx, const void *y, cudaDataType yType, int incy,
                                                        void *result, cudaDataType resultType,
                                                        cudaDataType executionType) {
    HOOK_TRACE_PROFILE("cublasDotcEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, int, const void *,
                                        cudaDataType, int, void *, cudaDataType, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDotcEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                         const float *y, int incy, float *result) {
    HOOK_TRACE_PROFILE("cublasSdot_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, int, const float *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSdot_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                         const double *y, int incy, double *result) {
    HOOK_TRACE_PROFILE("cublasDdot_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, int, const double *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDdot_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                          const cuComplex *y, int incy, cuComplex *result) {
    HOOK_TRACE_PROFILE("cublasCdotu_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, int, const cuComplex *, int, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCdotu_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                          const cuComplex *y, int incy, cuComplex *result) {
    HOOK_TRACE_PROFILE("cublasCdotc_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, int, const cuComplex *, int, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCdotc_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x,
                                                          int incx, const cuDoubleComplex *y, int incy,
                                                          cuDoubleComplex *result) {
    HOOK_TRACE_PROFILE("cublasZdotu_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                                        cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdotu_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x,
                                                          int incx, const cuDoubleComplex *y, int incy,
                                                          cuDoubleComplex *result) {
    HOOK_TRACE_PROFILE("cublasZdotc_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                                        cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdotc_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void *alpha,
                                                        cudaDataType alphaType, void *x, cudaDataType xType, int incx,
                                                        cudaDataType executionType) {
    HOOK_TRACE_PROFILE("cublasScalEx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, void *, cudaDataType, int, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasScalEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, alphaType, x, xType, incx, executionType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n, const float *alpha, float *x,
                                                          int incx) {
    HOOK_TRACE_PROFILE("cublasSscal_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSscal_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n, const double *alpha, double *x,
                                                          int incx) {
    HOOK_TRACE_PROFILE("cublasDscal_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDscal_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex *alpha,
                                                          cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCscal_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCscal_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsscal_v2(cublasHandle_t handle, int n, const float *alpha,
                                                           cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCsscal_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsscal_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha,
                                                          cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZscal_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZscal_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZdscal_v2(cublasHandle_t handle, int n, const double *alpha,
                                                           cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZdscal_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdscal_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void *alpha,
                                                        cudaDataType alphaType, const void *x, cudaDataType xType,
                                                        int incx, void *y, cudaDataType yType, int incy,
                                                        cudaDataType executiontype) {
    HOOK_TRACE_PROFILE("cublasAxpyEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, const void *, cudaDataType,
                                        int, void *, cudaDataType, int, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasAxpyEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha,
                                                          const float *x, int incx, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSaxpy_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSaxpy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha,
                                                          const double *x, int incx, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDaxpy_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDaxpy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex *alpha,
                                                          const cuComplex *x, int incx, cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCaxpy_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCaxpy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha,
                                                          const cuDoubleComplex *x, int incx, cuDoubleComplex *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasZaxpy_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int,
                                        cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZaxpy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, alpha, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCopyEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                        int incx, void *y, cudaDataType yType, int incy) {
    HOOK_TRACE_PROFILE("cublasCopyEx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, int, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCopyEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, y, yType, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                          float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasScopy_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasScopy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                          double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDcopy_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDcopy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                          cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCcopy_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCcopy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x,
                                                          int incx, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZcopy_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZcopy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n, float *x, int incx, float *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasSswap_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSswap_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n, double *x, int incx, double *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasDswap_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDswap_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCswap_v2(cublasHandle_t handle, int n, cuComplex *x, int incx,
                                                          cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCswap_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCswap_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx,
                                                          cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZswap_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZswap_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSwapEx(cublasHandle_t handle, int n, void *x, cudaDataType xType,
                                                        int incx, void *y, cudaDataType yType, int incy) {
    HOOK_TRACE_PROFILE("cublasSwapEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, void *, cudaDataType, int, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSwapEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, y, yType, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                           int *result) {
    HOOK_TRACE_PROFILE("cublasIsamax_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIsamax_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                           int *result) {
    HOOK_TRACE_PROFILE("cublasIdamax_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIdamax_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                           int *result) {
    HOOK_TRACE_PROFILE("cublasIcamax_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIcamax_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x,
                                                           int incx, int *result) {
    HOOK_TRACE_PROFILE("cublasIzamax_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIzamax_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIamaxEx(cublasHandle_t handle, int n, const void *x,
                                                         cudaDataType xType, int incx, int *result) {
    HOOK_TRACE_PROFILE("cublasIamaxEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIamaxEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                           int *result) {
    HOOK_TRACE_PROFILE("cublasIsamin_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIsamin_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                           int *result) {
    HOOK_TRACE_PROFILE("cublasIdamin_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIdamin_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                           int *result) {
    HOOK_TRACE_PROFILE("cublasIcamin_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIcamin_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x,
                                                           int incx, int *result) {
    HOOK_TRACE_PROFILE("cublasIzamin_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIzamin_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasIaminEx(cublasHandle_t handle, int n, const void *x,
                                                         cudaDataType xType, int incx, int *result) {
    HOOK_TRACE_PROFILE("cublasIaminEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasIaminEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasAsumEx(cublasHandle_t handle, int n, const void *x, cudaDataType xType,
                                                        int incx, void *result, cudaDataType resultType,
                                                        cudaDataType executiontype) {
    HOOK_TRACE_PROFILE("cublasAsumEx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const void *, cudaDataType, int, void *, cudaDataType, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasAsumEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, result, resultType, executiontype);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx,
                                                          float *result) {
    HOOK_TRACE_PROFILE("cublasSasum_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSasum_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx,
                                                          double *result) {
    HOOK_TRACE_PROFILE("cublasDasum_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDasum_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx,
                                                           float *result) {
    HOOK_TRACE_PROFILE("cublasScasum_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasScasum_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x,
                                                           int incx, double *result) {
    HOOK_TRACE_PROFILE("cublasDzasum_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDzasum_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, result);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSrot_v2(cublasHandle_t handle, int n, float *x, int incx, float *y,
                                                         int incy, const float *c, const float *s) {
    HOOK_TRACE_PROFILE("cublasSrot_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, float *, int, float *, int, const float *, const float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSrot_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDrot_v2(cublasHandle_t handle, int n, double *x, int incx, double *y,
                                                         int incy, const double *c, const double *s) {
    HOOK_TRACE_PROFILE("cublasDrot_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, double *, int, double *, int, const double *, const double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDrot_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx,
                                                         cuComplex *y, int incy, const float *c, const cuComplex *s) {
    HOOK_TRACE_PROFILE("cublasCrot_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, cuComplex *, int, cuComplex *, int, const float *, const cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCrot_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx,
                                                          cuComplex *y, int incy, const float *c, const float *s) {
    HOOK_TRACE_PROFILE("cublasCsrot_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, cuComplex *, int, cuComplex *, int, const float *, const float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsrot_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx,
                                                         cuDoubleComplex *y, int incy, const double *c,
                                                         const cuDoubleComplex *s) {
    HOOK_TRACE_PROFILE("cublasZrot_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int,
                                        const double *, const cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZrot_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx,
                                                          cuDoubleComplex *y, int incy, const double *c,
                                                          const double *s) {
    HOOK_TRACE_PROFILE("cublasZdrot_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex *, int, cuDoubleComplex *, int,
                                        const double *, const double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdrot_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void *x, cudaDataType xType,
                                                       int incx, void *y, cudaDataType yType, int incy, const void *c,
                                                       const void *s, cudaDataType csType, cudaDataType executiontype) {
    HOOK_TRACE_PROFILE("cublasRotEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, void *, cudaDataType, int, void *, cudaDataType, int,
                                        const void *, const void *, cudaDataType, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasRotEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSrotg_v2(cublasHandle_t handle, float *a, float *b, float *c,
                                                          float *s) {
    HOOK_TRACE_PROFILE("cublasSrotg_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, float *, float *, float *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSrotg_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, a, b, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDrotg_v2(cublasHandle_t handle, double *a, double *b, double *c,
                                                          double *s) {
    HOOK_TRACE_PROFILE("cublasDrotg_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, double *, double *, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDrotg_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, a, b, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCrotg_v2(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c,
                                                          cuComplex *s) {
    HOOK_TRACE_PROFILE("cublasCrotg_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cuComplex *, cuComplex *, float *, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCrotg_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, a, b, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b,
                                                          double *c, cuDoubleComplex *s) {
    HOOK_TRACE_PROFILE("cublasZrotg_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cuDoubleComplex *, cuDoubleComplex *, double *, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZrotg_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, a, b, c, s);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasRotgEx(cublasHandle_t handle, void *a, void *b, cudaDataType abType,
                                                        void *c, void *s, cudaDataType csType,
                                                        cudaDataType executiontype) {
    HOOK_TRACE_PROFILE("cublasRotgEx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, void *, void *, cudaDataType, void *, void *, cudaDataType, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasRotgEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, a, b, abType, c, s, csType, executiontype);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSrotm_v2(cublasHandle_t handle, int n, float *x, int incx, float *y,
                                                          int incy, const float *param) {
    HOOK_TRACE_PROFILE("cublasSrotm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, float *, int, float *, int, const float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSrotm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, param);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDrotm_v2(cublasHandle_t handle, int n, double *x, int incx, double *y,
                                                          int incy, const double *param) {
    HOOK_TRACE_PROFILE("cublasDrotm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, double *, int, double *, int, const double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDrotm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, incx, y, incy, param);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasRotmEx(cublasHandle_t handle, int n, void *x, cudaDataType xType,
                                                        int incx, void *y, cudaDataType yType, int incy,
                                                        const void *param, cudaDataType paramType,
                                                        cudaDataType executiontype) {
    HOOK_TRACE_PROFILE("cublasRotmEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, void *, cudaDataType, int, void *, cudaDataType, int,
                                        const void *, cudaDataType, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasRotmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSrotmg_v2(cublasHandle_t handle, float *d1, float *d2, float *x1,
                                                           const float *y1, float *param) {
    HOOK_TRACE_PROFILE("cublasSrotmg_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, float *, float *, float *, const float *, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSrotmg_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, d1, d2, x1, y1, param);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDrotmg_v2(cublasHandle_t handle, double *d1, double *d2, double *x1,
                                                           const double *y1, double *param) {
    HOOK_TRACE_PROFILE("cublasDrotmg_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, double *, double *, double *, const double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDrotmg_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, d1, d2, x1, y1, param);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasRotmgEx(cublasHandle_t handle, void *d1, cudaDataType d1Type, void *d2,
                                                         cudaDataType d2Type, void *x1, cudaDataType x1Type,
                                                         const void *y1, cudaDataType y1Type, void *param,
                                                         cudaDataType paramType, cudaDataType executiontype) {
    HOOK_TRACE_PROFILE("cublasRotmgEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, void *, cudaDataType, void *, cudaDataType, void *,
                                        cudaDataType, const void *, cudaDataType, void *, cudaDataType, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasRotmgEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                          const float *alpha, const float *A, int lda, const float *x,
                                                          int incx, const float *beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSgemv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float *, const float *, int,
                                        const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgemv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                          const double *alpha, const double *A, int lda,
                                                          const double *x, int incx, const double *beta, double *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasDgemv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double *, const double *,
                                        int, const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgemv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                          const cuComplex *alpha, const cuComplex *A, int lda,
                                                          const cuComplex *x, int incx, const cuComplex *beta,
                                                          cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCgemv_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex *, const cuComplex *, int,
                           const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *x, int incx,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZgemv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex *,
                                        const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                                        const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgemv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                          int kl, int ku, const float *alpha, const float *A, int lda,
                                                          const float *x, int incx, const float *beta, float *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasSgbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const float *,
                                        const float *, int, const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                          int kl, int ku, const double *alpha, const double *A, int lda,
                                                          const double *x, int incx, const double *beta, double *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasDgbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const double *,
                                        const double *, int, const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                          int kl, int ku, const cuComplex *alpha, const cuComplex *A,
                                                          int lda, const cuComplex *x, int incx, const cuComplex *beta,
                                                          cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCgbmv_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const cuComplex *, const cuComplex *,
                           int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                                                          int kl, int ku, const cuDoubleComplex *alpha,
                                                          const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                                                          int incx, const cuDoubleComplex *beta, cuDoubleComplex *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasZgbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, int, const cuDoubleComplex *,
                                        const cuDoubleComplex *, int, const cuDoubleComplex *, int,
                                        const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const float *A, int lda, float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasStrmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const double *A, int lda, double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtrmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const cuComplex *A, int lda, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtrmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const cuDoubleComplex *A, int lda, cuDoubleComplex *x,
                                                          int incx) {
    HOOK_TRACE_PROFILE("cublasZtrmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                          const float *A, int lda, float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasStbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int,
                                        const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                          const double *A, int lda, double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int,
                                        const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                          const cuComplex *A, int lda, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int,
                                        const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                          const cuDoubleComplex *A, int lda, cuDoubleComplex *x,
                                                          int incx) {
    HOOK_TRACE_PROFILE("cublasZtbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const float *AP, float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasStpmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStpmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const double *AP, double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtpmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtpmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const cuComplex *AP, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtpmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtpmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZtpmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtpmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const float *A, int lda, float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasStrsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const double *A, int lda, double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtrsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const cuComplex *A, int lda, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtrsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const cuDoubleComplex *A, int lda, cuDoubleComplex *x,
                                                          int incx) {
    HOOK_TRACE_PROFILE("cublasZtrsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const float *AP, float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasStpsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStpsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const double *AP, double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtpsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtpsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const cuComplex *AP, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtpsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtpsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n,
                                                          const cuDoubleComplex *AP, cuDoubleComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasZtpsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                                        const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtpsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, AP, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                          const float *A, int lda, float *x, int incx) {
    HOOK_TRACE_PROFILE("cublasStbsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int,
                                        const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStbsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                          const double *A, int lda, double *x, int incx) {
    HOOK_TRACE_PROFILE("cublasDtbsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int,
                                        const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtbsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                          const cuComplex *A, int lda, cuComplex *x, int incx) {
    HOOK_TRACE_PROFILE("cublasCtbsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int,
                                        const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtbsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
                                                          const cuDoubleComplex *A, int lda, cuDoubleComplex *x,
                                                          int incx) {
    HOOK_TRACE_PROFILE("cublasZtbsv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtbsv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const float *alpha, const float *A, int lda, const float *x,
                                                          int incx, const float *beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSsymv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int,
                                        const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsymv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const double *alpha, const double *A, int lda,
                                                          const double *x, int incx, const double *beta, double *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasDsymv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int,
                                        const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsymv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuComplex *alpha, const cuComplex *A, int lda,
                                                          const cuComplex *x, int incx, const cuComplex *beta,
                                                          cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasCsymv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *,
                                        int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsymv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *x, int incx,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZsymv_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int,
                           const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsymv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuComplex *alpha, const cuComplex *A, int lda,
                                                          const cuComplex *x, int incx, const cuComplex *beta,
                                                          cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasChemv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *,
                                        int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChemv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *x, int incx,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZhemv_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int,
                           const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhemv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                                                          const float *alpha, const float *A, int lda, const float *x,
                                                          int incx, const float *beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSsbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const float *, const float *, int,
                                        const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                                                          const double *alpha, const double *A, int lda,
                                                          const double *x, int incx, const double *beta, double *y,
                                                          int incy) {
    HOOK_TRACE_PROFILE("cublasDsbmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const double *, const double *, int,
                                        const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                                                          const cuComplex *alpha, const cuComplex *A, int lda,
                                                          const cuComplex *x, int incx, const cuComplex *beta,
                                                          cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasChbmv_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const cuComplex *, const cuComplex *, int,
                           const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *x, int incx,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZhbmv_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *,
                           int, const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhbmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const float *alpha, const float *AP, const float *x, int incx,
                                                          const float *beta, float *y, int incy) {
    HOOK_TRACE_PROFILE("cublasSspmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *,
                                        const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSspmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const double *alpha, const double *AP, const double *x,
                                                          int incx, const double *beta, double *y, int incy) {
    HOOK_TRACE_PROFILE("cublasDspmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *,
                                        const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDspmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuComplex *alpha, const cuComplex *AP,
                                                          const cuComplex *x, int incx, const cuComplex *beta,
                                                          cuComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasChpmv_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *,
                                        const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChpmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *AP,
                                                          const cuDoubleComplex *x, int incx,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *y, int incy) {
    HOOK_TRACE_PROFILE("cublasZhpmv_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *,
                           const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhpmv_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n, const float *alpha,
                                                         const float *x, int incx, const float *y, int incy, float *A,
                                                         int lda) {
    HOOK_TRACE_PROFILE("cublasSger_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, const float *, const float *, int, const float *, int,
                                        float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSger_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n, const double *alpha,
                                                         const double *x, int incx, const double *y, int incy,
                                                         double *A, int lda) {
    HOOK_TRACE_PROFILE("cublasDger_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, const double *, const double *, int, const double *,
                                        int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDger_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha,
                                                          const cuComplex *x, int incx, const cuComplex *y, int incy,
                                                          cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCgeru_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, const cuComplex *, const cuComplex *, int,
                                        const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgeru_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha,
                                                          const cuComplex *x, int incx, const cuComplex *y, int incy,
                                                          cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCgerc_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, const cuComplex *, const cuComplex *, int,
                                        const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgerc_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgeru_v2(cublasHandle_t handle, int m, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *x,
                                                          int incx, const cuDoubleComplex *y, int incy,
                                                          cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZgeru_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgeru_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgerc_v2(cublasHandle_t handle, int m, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *x,
                                                          int incx, const cuDoubleComplex *y, int incy,
                                                          cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZgerc_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *, int,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgerc_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const float *alpha, const float *x, int incx, float *A,
                                                         int lda) {
    HOOK_TRACE_PROFILE("cublasSsyr_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyr_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const double *alpha, const double *x, int incx, double *A,
                                                         int lda) {
    HOOK_TRACE_PROFILE("cublasDsyr_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyr_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const cuComplex *alpha, const cuComplex *x, int incx,
                                                         cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCsyr_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *,
                                        int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyr_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *x,
                                                         int incx, cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZsyr_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsyr_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const float *alpha, const cuComplex *x, int incx, cuComplex *A,
                                                         int lda) {
    HOOK_TRACE_PROFILE("cublasCher_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, const cuComplex *, int,
                                        cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCher_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const double *alpha, const cuDoubleComplex *x, int incx,
                                                         cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZher_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, const cuDoubleComplex *,
                                        int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZher_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const float *alpha, const float *x, int incx, float *AP) {
    HOOK_TRACE_PROFILE("cublasSspr_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSspr_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const double *alpha, const double *x, int incx, double *AP) {
    HOOK_TRACE_PROFILE("cublasDspr_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDspr_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const float *alpha, const cuComplex *x, int incx,
                                                         cuComplex *AP) {
    HOOK_TRACE_PROFILE("cublasChpr_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, const cuComplex *, int, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChpr_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                         const double *alpha, const cuDoubleComplex *x, int incx,
                                                         cuDoubleComplex *AP) {
    HOOK_TRACE_PROFILE("cublasZhpr_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, const cuDoubleComplex *,
                                        int, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhpr_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const float *alpha, const float *x, int incx, const float *y,
                                                          int incy, float *A, int lda) {
    HOOK_TRACE_PROFILE("cublasSsyr2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int,
                                        const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyr2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const double *alpha, const double *x, int incx,
                                                          const double *y, int incy, double *A, int lda) {
    HOOK_TRACE_PROFILE("cublasDsyr2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int,
                                        const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyr2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuComplex *alpha, const cuComplex *x, int incx,
                                                          const cuComplex *y, int incy, cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCsyr2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *,
                                        int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyr2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *x,
                                                          int incx, const cuDoubleComplex *y, int incy,
                                                          cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZsyr2_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int,
                           const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsyr2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuComplex *alpha, const cuComplex *x, int incx,
                                                          const cuComplex *y, int incy, cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCher2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *,
                                        int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCher2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *x,
                                                          int incx, const cuDoubleComplex *y, int incy,
                                                          cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZher2_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, const cuDoubleComplex *, int,
                           const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZher2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const float *alpha, const float *x, int incx, const float *y,
                                                          int incy, float *AP) {
    HOOK_TRACE_PROFILE("cublasSspr2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, const float *, int,
                                        const float *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSspr2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const double *alpha, const double *x, int incx,
                                                          const double *y, int incy, double *AP) {
    HOOK_TRACE_PROFILE("cublasDspr2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, const double *, int,
                                        const double *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDspr2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuComplex *alpha, const cuComplex *x, int incx,
                                                          const cuComplex *y, int incy, cuComplex *AP) {
    HOOK_TRACE_PROFILE("cublasChpr2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, const cuComplex *,
                                        int, const cuComplex *, int, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChpr2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *x,
                                                          int incx, const cuDoubleComplex *y, int incy,
                                                          cuDoubleComplex *AP) {
    HOOK_TRACE_PROFILE("cublasZhpr2_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *,
                                        const cuDoubleComplex *, int, const cuDoubleComplex *, int, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhpr2_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, alpha, x, incx, y, incy, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                                          cublasOperation_t transb, int m, int n, int k,
                                                          const float *alpha, const float *A, int lda, const float *B,
                                                          int ldb, const float *beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSgemm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *,
                           const float *, int, const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgemm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                                          cublasOperation_t transb, int m, int n, int k,
                                                          const double *alpha, const double *A, int lda,
                                                          const double *B, int ldb, const double *beta, double *C,
                                                          int ldc) {
    HOOK_TRACE_PROFILE("cublasDgemm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *,
                           const double *, int, const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgemm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                                          cublasOperation_t transb, int m, int n, int k,
                                                          const cuComplex *alpha, const cuComplex *A, int lda,
                                                          const cuComplex *B, int ldb, const cuComplex *beta,
                                                          cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCgemm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const cuComplex *alpha, const cuComplex *A, int lda,
                                                         const cuComplex *B, int ldb, const cuComplex *beta,
                                                         cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCgemm3m");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemm3m"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa,
                                                           cublasOperation_t transb, int m, int n, int k,
                                                           const cuComplex *alpha, const void *A, cudaDataType Atype,
                                                           int lda, const void *B, cudaDataType Btype, int ldb,
                                                           const cuComplex *beta, void *C, cudaDataType Ctype,
                                                           int ldc) {
    HOOK_TRACE_PROFILE("cublasCgemm3mEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const cuComplex *, const void *, cudaDataType, int, const void *, cudaDataType,
                                        int, const cuComplex *, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemm3mEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                                                          cublasOperation_t transb, int m, int n, int k,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *B, int ldb,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZgemm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgemm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                         int lda, const cuDoubleComplex *B, int ldb,
                                                         const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZgemm3m");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgemm3m"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                                                       cublasOperation_t transb, int m, int n, int k,
                                                       const __half *alpha, const __half *A, int lda, const __half *B,
                                                       int ldb, const __half *beta, __half *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasHgemm");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const __half *,
                           const __half *, int, const __half *, int, const __half *, __half *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasHgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const float *alpha, const void *A, cudaDataType Atype, int lda,
                                                         const void *B, cudaDataType Btype, int ldb, const float *beta,
                                                         void *C, cudaDataType Ctype, int ldc) {
    HOOK_TRACE_PROFILE("cublasSgemmEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const float *, const void *, cudaDataType, int, const void *, cudaDataType, int,
                                        const float *, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgemmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                                                        cublasOperation_t transb, int m, int n, int k,
                                                        const void *alpha, const void *A, cudaDataType Atype, int lda,
                                                        const void *B, cudaDataType Btype, int ldb, const void *beta,
                                                        void *C, cudaDataType Ctype, int ldc,
                                                        cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    HOOK_TRACE_PROFILE("cublasGemmEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const void *, const void *, cudaDataType, int, const void *, cudaDataType, int,
                                        const void *, void *, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGemmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
                      computeType, algo);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, int m, int n, int k,
                                                         const cuComplex *alpha, const void *A, cudaDataType Atype,
                                                         int lda, const void *B, cudaDataType Btype, int ldb,
                                                         const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
    HOOK_TRACE_PROFILE("cublasCgemmEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const cuComplex *, const void *, cudaDataType, int, const void *, cudaDataType,
                                        int, const cuComplex *, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasUint8gemmBias(cublasHandle_t handle, cublasOperation_t transa,
                                                               cublasOperation_t transb, cublasOperation_t transc,
                                                               int m, int n, int k, const unsigned char *A, int A_bias,
                                                               int lda, const unsigned char *B, int B_bias, int ldb,
                                                               unsigned char *C, int C_bias, int ldc, int C_mult,
                                                               int C_shift) {
    HOOK_TRACE_PROFILE("cublasUint8gemmBias");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t, int,
                                        int, int, const unsigned char *, int, int, const unsigned char *, int, int,
                                        unsigned char *, int, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasUint8gemmBias"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult,
                      C_shift);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, int n, int k, const float *alpha,
                                                          const float *A, int lda, const float *beta, float *C,
                                                          int ldc) {
    HOOK_TRACE_PROFILE("cublasSsyrk_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *,
                                        const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyrk_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, int n, int k, const double *alpha,
                                                          const double *A, int lda, const double *beta, double *C,
                                                          int ldc) {
    HOOK_TRACE_PROFILE("cublasDsyrk_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double *,
                                        const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyrk_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, int n, int k, const cuComplex *alpha,
                                                          const cuComplex *A, int lda, const cuComplex *beta,
                                                          cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCsyrk_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int,
                                        const cuComplex *, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyrk_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, int n, int k,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *beta, cuDoubleComplex *C,
                                                          int ldc) {
    HOOK_TRACE_PROFILE("cublasZsyrk_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuDoubleComplex *,
                           const cuDoubleComplex *, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsyrk_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                         cublasOperation_t trans, int n, int k, const cuComplex *alpha,
                                                         const void *A, cudaDataType Atype, int lda,
                                                         const cuComplex *beta, void *C, cudaDataType Ctype, int ldc) {
    HOOK_TRACE_PROFILE("cublasCsyrkEx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *,
                           const void *, cudaDataType, int, const cuComplex *, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyrkEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                           cublasOperation_t trans, int n, int k,
                                                           const cuComplex *alpha, const void *A, cudaDataType Atype,
                                                           int lda, const cuComplex *beta, void *C, cudaDataType Ctype,
                                                           int ldc) {
    HOOK_TRACE_PROFILE("cublasCsyrk3mEx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *,
                           const void *, cudaDataType, int, const cuComplex *, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyrk3mEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, int n, int k, const float *alpha,
                                                          const cuComplex *A, int lda, const float *beta, cuComplex *C,
                                                          int ldc) {
    HOOK_TRACE_PROFILE("cublasCherk_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *,
                                        const cuComplex *, int, const float *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCherk_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, int n, int k, const double *alpha,
                                                          const cuDoubleComplex *A, int lda, const double *beta,
                                                          cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZherk_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double *,
                                        const cuDoubleComplex *, int, const double *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZherk_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                         cublasOperation_t trans, int n, int k, const float *alpha,
                                                         const void *A, cudaDataType Atype, int lda, const float *beta,
                                                         void *C, cudaDataType Ctype, int ldc) {
    HOOK_TRACE_PROFILE("cublasCherkEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *,
                                        const void *, cudaDataType, int, const float *, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCherkEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                           cublasOperation_t trans, int n, int k, const float *alpha,
                                                           const void *A, cudaDataType Atype, int lda,
                                                           const float *beta, void *C, cudaDataType Ctype, int ldc) {
    HOOK_TRACE_PROFILE("cublasCherk3mEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *,
                                        const void *, cudaDataType, int, const float *, void *, cudaDataType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCherk3mEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                           cublasOperation_t trans, int n, int k, const float *alpha,
                                                           const float *A, int lda, const float *B, int ldb,
                                                           const float *beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSsyr2k_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *,
                                        const float *, int, const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyr2k_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                           cublasOperation_t trans, int n, int k, const double *alpha,
                                                           const double *A, int lda, const double *B, int ldb,
                                                           const double *beta, double *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasDsyr2k_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double *,
                                        const double *, int, const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyr2k_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                           cublasOperation_t trans, int n, int k,
                                                           const cuComplex *alpha, const cuComplex *A, int lda,
                                                           const cuComplex *B, int ldb, const cuComplex *beta,
                                                           cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCsyr2k_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyr2k_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                           cublasOperation_t trans, int n, int k,
                                                           const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                           int lda, const cuDoubleComplex *B, int ldb,
                                                           const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZsyr2k_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsyr2k_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                           cublasOperation_t trans, int n, int k,
                                                           const cuComplex *alpha, const cuComplex *A, int lda,
                                                           const cuComplex *B, int ldb, const float *beta, cuComplex *C,
                                                           int ldc) {
    HOOK_TRACE_PROFILE("cublasCher2k_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, int, const float *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCher2k_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                                           cublasOperation_t trans, int n, int k,
                                                           const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                           int lda, const cuDoubleComplex *B, int ldb,
                                                           const double *beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZher2k_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        int, const double *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZher2k_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                        cublasOperation_t trans, int n, int k, const float *alpha,
                                                        const float *A, int lda, const float *B, int ldb,
                                                        const float *beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSsyrkx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const float *,
                                        const float *, int, const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsyrkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                        cublasOperation_t trans, int n, int k, const double *alpha,
                                                        const double *A, int lda, const double *B, int ldb,
                                                        const double *beta, double *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasDsyrkx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const double *,
                                        const double *, int, const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsyrkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                        cublasOperation_t trans, int n, int k, const cuComplex *alpha,
                                                        const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                        const cuComplex *beta, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCsyrkx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsyrkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                        cublasOperation_t trans, int n, int k,
                                                        const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                                        const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
                                                        cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZsyrkx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsyrkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                        cublasOperation_t trans, int n, int k, const cuComplex *alpha,
                                                        const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                        const float *beta, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCherkx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, int, const float *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCherkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo,
                                                        cublasOperation_t trans, int n, int k,
                                                        const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                                        const cuDoubleComplex *B, int ldb, const double *beta,
                                                        cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZherkx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        int, const double *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZherkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, int m, int n, const float *alpha,
                                                          const float *A, int lda, const float *B, int ldb,
                                                          const float *beta, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSsymm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const float *,
                                        const float *, int, const float *, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSsymm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, int m, int n, const double *alpha,
                                                          const double *A, int lda, const double *B, int ldb,
                                                          const double *beta, double *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasDsymm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const double *,
                                        const double *, int, const double *, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDsymm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, int m, int n, const cuComplex *alpha,
                                                          const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                          const cuComplex *beta, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCsymm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCsymm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, int m, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *B, int ldb,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZsymm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZsymm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, int m, int n, const cuComplex *alpha,
                                                          const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                          const cuComplex *beta, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasChemm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasChemm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, int m, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *B, int ldb,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZhemm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZhemm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, cublasOperation_t trans,
                                                          cublasDiagType_t diag, int m, int n, const float *alpha,
                                                          const float *A, int lda, float *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasStrsm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                        cublasDiagType_t, int, int, const float *, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrsm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, cublasOperation_t trans,
                                                          cublasDiagType_t diag, int m, int n, const double *alpha,
                                                          const double *A, int lda, double *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasDtrsm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                        cublasDiagType_t, int, int, const double *, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrsm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, cublasOperation_t trans,
                                                          cublasDiagType_t diag, int m, int n, const cuComplex *alpha,
                                                          const cuComplex *A, int lda, cuComplex *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasCtrsm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                           int, const cuComplex *, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrsm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, cublasOperation_t trans,
                                                          cublasDiagType_t diag, int m, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, cuDoubleComplex *B, int ldb) {
    HOOK_TRACE_PROFILE("cublasZtrsm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                           int, const cuDoubleComplex *, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrsm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, cublasOperation_t trans,
                                                          cublasDiagType_t diag, int m, int n, const float *alpha,
                                                          const float *A, int lda, const float *B, int ldb, float *C,
                                                          int ldc) {
    HOOK_TRACE_PROFILE("cublasStrmm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                           int, const float *, const float *, int, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrmm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, cublasOperation_t trans,
                                                          cublasDiagType_t diag, int m, int n, const double *alpha,
                                                          const double *A, int lda, const double *B, int ldb, double *C,
                                                          int ldc) {
    HOOK_TRACE_PROFILE("cublasDtrmm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                           int, const double *, const double *, int, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrmm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, cublasOperation_t trans,
                                                          cublasDiagType_t diag, int m, int n, const cuComplex *alpha,
                                                          const cuComplex *A, int lda, const cuComplex *B, int ldb,
                                                          cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCtrmm_v2");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                           int, const cuComplex *, const cuComplex *, int, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrmm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side,
                                                          cublasFillMode_t uplo, cublasOperation_t trans,
                                                          cublasDiagType_t diag, int m, int n,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          int lda, const cuDoubleComplex *B, int ldb,
                                                          cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZtrmm_v2");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                        cublasDiagType_t, int, int, const cuDoubleComplex *, const cuDoubleComplex *,
                                        int, const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrmm_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                              cublasOperation_t transb, int m, int n, int k,
                                                              const __half *alpha, const __half *const Aarray, int lda,
                                                              const __half *const Barray, int ldb, const __half *beta,
                                                              __half *const Carray, int ldc, int batchCount) {
    HOOK_TRACE_PROFILE("cublasHgemmBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const __half *,
                           const __half *const, int, const __half *const, int, const __half *, __half *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasHgemmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                              cublasOperation_t transb, int m, int n, int k,
                                                              const float *alpha, const float *const Aarray, int lda,
                                                              const float *const Barray, int ldb, const float *beta,
                                                              float *const Carray, int ldc, int batchCount) {
    HOOK_TRACE_PROFILE("cublasSgemmBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float *,
                           const float *const, int, const float *const, int, const float *, float *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgemmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                              cublasOperation_t transb, int m, int n, int k,
                                                              const double *alpha, const double *const Aarray, int lda,
                                                              const double *const Barray, int ldb, const double *beta,
                                                              double *const Carray, int ldc, int batchCount) {
    HOOK_TRACE_PROFILE("cublasDgemmBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *,
                           const double *const, int, const double *const, int, const double *, double *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgemmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                              cublasOperation_t transb, int m, int n, int k,
                                                              const cuComplex *alpha, const cuComplex *const Aarray,
                                                              int lda, const cuComplex *const Barray, int ldb,
                                                              const cuComplex *beta, cuComplex *const Carray, int ldc,
                                                              int batchCount) {
    HOOK_TRACE_PROFILE("cublasCgemmBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const cuComplex *, const cuComplex *const, int, const cuComplex *const, int,
                                        const cuComplex *, cuComplex *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                                cublasOperation_t transb, int m, int n, int k,
                                                                const cuComplex *alpha, const cuComplex *const Aarray,
                                                                int lda, const cuComplex *const Barray, int ldb,
                                                                const cuComplex *beta, cuComplex *const Carray, int ldc,
                                                                int batchCount) {
    HOOK_TRACE_PROFILE("cublasCgemm3mBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const cuComplex *, const cuComplex *const, int, const cuComplex *const, int,
                                        const cuComplex *, cuComplex *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemm3mBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuDoubleComplex *alpha, const cuDoubleComplex *const Aarray, int lda, const cuDoubleComplex *const Barray,
    int ldb, const cuDoubleComplex *beta, cuDoubleComplex *const Carray, int ldc, int batchCount) {
    HOOK_TRACE_PROFILE("cublasZgemmBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex *,
                           const cuDoubleComplex *const, int, const cuDoubleComplex *const, int,
                           const cuDoubleComplex *, cuDoubleComplex *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgemmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGemmBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha,
    const void *const Aarray, cudaDataType Atype, int lda, const void *const Barray, cudaDataType Btype, int ldb,
    const void *beta, void *const Carray, cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType,
    cublasGemmAlgo_t algo) {
    HOOK_TRACE_PROFILE("cublasGemmBatchedEx");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *,
                           const void *const, cudaDataType, int, const void *const, cudaDataType, int, const void *,
                           void *const, cudaDataType, int, int, cublasComputeType_t, cublasGemmAlgo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGemmBatchedEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray,
                      Ctype, ldc, batchCount, computeType, algo);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha,
    const void *A, cudaDataType Atype, int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb,
    long long int strideB, const void *beta, void *C, cudaDataType Ctype, int ldc, long long int strideC,
    int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    HOOK_TRACE_PROFILE("cublasGemmStridedBatchedEx");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const void *, const void *, cudaDataType, int, long long int, const void *,
                                        cudaDataType, int, long long int, const void *, void *, cudaDataType, int,
                                        long long int, int, cublasComputeType_t, cublasGemmAlgo_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasGemmStridedBatchedEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C,
                      Ctype, ldc, strideC, batchCount, computeType, algo);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                                                                     cublasOperation_t transb, int m, int n, int k,
                                                                     const float *alpha, const float *A, int lda,
                                                                     long long int strideA, const float *B, int ldb,
                                                                     long long int strideB, const float *beta, float *C,
                                                                     int ldc, long long int strideC, int batchCount) {
    HOOK_TRACE_PROFILE("cublasSgemmStridedBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const float *, const float *, int, long long int, const float *, int,
                                        long long int, const float *, float *, int, long long int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgemmStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
                      batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha,
    const double *A, int lda, long long int strideA, const double *B, int ldb, long long int strideB,
    const double *beta, double *C, int ldc, long long int strideC, int batchCount) {
    HOOK_TRACE_PROFILE("cublasDgemmStridedBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const double *, const double *, int, long long int, const double *, int,
                                        long long int, const double *, double *, int, long long int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgemmStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
                      batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb,
    long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount) {
    HOOK_TRACE_PROFILE("cublasCgemmStridedBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const cuComplex *, const cuComplex *, int, long long int, const cuComplex *,
                                        int, long long int, const cuComplex *, cuComplex *, int, long long int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemmStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
                      batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgemm3mStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuComplex *alpha, const cuComplex *A, int lda, long long int strideA, const cuComplex *B, int ldb,
    long long int strideB, const cuComplex *beta, cuComplex *C, int ldc, long long int strideC, int batchCount) {
    HOOK_TRACE_PROFILE("cublasCgemm3mStridedBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const cuComplex *, const cuComplex *, int, long long int, const cuComplex *,
                                        int, long long int, const cuComplex *, cuComplex *, int, long long int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgemm3mStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
                      batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, long long int strideA, const cuDoubleComplex *B,
    int ldb, long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc, long long int strideC,
    int batchCount) {
    HOOK_TRACE_PROFILE("cublasZgemmStridedBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const cuDoubleComplex *,
                           const cuDoubleComplex *, int, long long int, const cuDoubleComplex *, int, long long int,
                           const cuDoubleComplex *, cuDoubleComplex *, int, long long int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgemmStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
                      batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasHgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha,
    const __half *A, int lda, long long int strideA, const __half *B, int ldb, long long int strideB,
    const __half *beta, __half *C, int ldc, long long int strideC, int batchCount) {
    HOOK_TRACE_PROFILE("cublasHgemmStridedBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                        const __half *, const __half *, int, long long int, const __half *, int,
                                        long long int, const __half *, __half *, int, long long int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasHgemmStridedBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
                      batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa,
                                                       cublasOperation_t transb, int m, int n, const float *alpha,
                                                       const float *A, int lda, const float *beta, const float *B,
                                                       int ldb, float *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasSgeam");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const float *,
                                        const float *, int, const float *, const float *, int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgeam"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa,
                                                       cublasOperation_t transb, int m, int n, const double *alpha,
                                                       const double *A, int lda, const double *beta, const double *B,
                                                       int ldb, double *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasDgeam");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const double *,
                                        const double *, int, const double *, const double *, int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgeam"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa,
                                                       cublasOperation_t transb, int m, int n, const cuComplex *alpha,
                                                       const cuComplex *A, int lda, const cuComplex *beta,
                                                       const cuComplex *B, int ldb, cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCgeam");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const cuComplex *,
                           const cuComplex *, int, const cuComplex *, const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgeam"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa,
                                                       cublasOperation_t transb, int m, int n,
                                                       const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                                       const cuDoubleComplex *beta, const cuDoubleComplex *B, int ldb,
                                                       cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZgeam");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int,
                                        const cuDoubleComplex *, const cuDoubleComplex *, int, const cuDoubleComplex *,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgeam"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n, float *const A, int lda,
                                                               int *P, int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasSgetrfBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, float *const, int, int *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgetrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, P, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n, double *const A, int lda,
                                                               int *P, int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasDgetrfBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, double *const, int, int *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgetrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, P, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex *const A,
                                                               int lda, int *P, int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasCgetrfBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, cuComplex *const, int, int *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgetrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, P, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex *const A,
                                                               int lda, int *P, int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasZgetrfBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, cuDoubleComplex *const, int, int *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgetrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, P, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n, const float *const A,
                                                               int lda, const int *P, float *const C, int ldc,
                                                               int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasSgetriBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const float *const, int, const int *, float *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgetriBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, P, C, ldc, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n, const double *const A,
                                                               int lda, const int *P, double *const C, int ldc,
                                                               int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasDgetriBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const double *const, int, const int *, double *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgetriBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, P, C, ldc, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex *const A,
                                                               int lda, const int *P, cuComplex *const C, int ldc,
                                                               int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasCgetriBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *const, int, const int *, cuComplex *const,
                                        int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgetriBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, P, C, ldc, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n,
                                                               const cuDoubleComplex *const A, int lda, const int *P,
                                                               cuDoubleComplex *const C, int ldc, int *info,
                                                               int batchSize) {
    HOOK_TRACE_PROFILE("cublasZgetriBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *const, int, const int *,
                                        cuDoubleComplex *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgetriBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, P, C, ldc, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n,
                                                               int nrhs, const float *const Aarray, int lda,
                                                               const int *devIpiv, float *const Barray, int ldb,
                                                               int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasSgetrsBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const float *const, int,
                                        const int *, float *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgetrsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n,
                                                               int nrhs, const double *const Aarray, int lda,
                                                               const int *devIpiv, double *const Barray, int ldb,
                                                               int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasDgetrsBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const double *const, int,
                                        const int *, double *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgetrsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n,
                                                               int nrhs, const cuComplex *const Aarray, int lda,
                                                               const int *devIpiv, cuComplex *const Barray, int ldb,
                                                               int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasCgetrsBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuComplex *const, int,
                                        const int *, cuComplex *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgetrsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n,
                                                               int nrhs, const cuDoubleComplex *const Aarray, int lda,
                                                               const int *devIpiv, cuDoubleComplex *const Barray,
                                                               int ldb, int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasZgetrsBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, const cuDoubleComplex *const, int,
                                        const int *, cuDoubleComplex *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgetrsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                                              cublasFillMode_t uplo, cublasOperation_t trans,
                                                              cublasDiagType_t diag, int m, int n, const float *alpha,
                                                              const float *const A, int lda, float *const B, int ldb,
                                                              int batchCount) {
    HOOK_TRACE_PROFILE("cublasStrsmBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                           int, const float *, const float *const, int, float *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrsmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                                              cublasFillMode_t uplo, cublasOperation_t trans,
                                                              cublasDiagType_t diag, int m, int n, const double *alpha,
                                                              const double *const A, int lda, double *const B, int ldb,
                                                              int batchCount) {
    HOOK_TRACE_PROFILE("cublasDtrsmBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                           int, const double *, const double *const, int, double *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrsmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                                              cublasFillMode_t uplo, cublasOperation_t trans,
                                                              cublasDiagType_t diag, int m, int n,
                                                              const cuComplex *alpha, const cuComplex *const A, int lda,
                                                              cuComplex *const B, int ldb, int batchCount) {
    HOOK_TRACE_PROFILE("cublasCtrsmBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int,
                           int, const cuComplex *, const cuComplex *const, int, cuComplex *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrsmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                                              cublasFillMode_t uplo, cublasOperation_t trans,
                                                              cublasDiagType_t diag, int m, int n,
                                                              const cuDoubleComplex *alpha,
                                                              const cuDoubleComplex *const A, int lda,
                                                              cuDoubleComplex *const B, int ldb, int batchCount) {
    HOOK_TRACE_PROFILE("cublasZtrsmBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                        cublasDiagType_t, int, int, const cuDoubleComplex *,
                                        const cuDoubleComplex *const, int, cuDoubleComplex *const, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrsmBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n, const float *const A,
                                                                int lda, float *const Ainv, int lda_inv, int *info,
                                                                int batchSize) {
    HOOK_TRACE_PROFILE("cublasSmatinvBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const float *const, int, float *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSmatinvBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n, const double *const A,
                                                                int lda, double *const Ainv, int lda_inv, int *info,
                                                                int batchSize) {
    HOOK_TRACE_PROFILE("cublasDmatinvBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const double *const, int, double *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDmatinvBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n, const cuComplex *const A,
                                                                int lda, cuComplex *const Ainv, int lda_inv, int *info,
                                                                int batchSize) {
    HOOK_TRACE_PROFILE("cublasCmatinvBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, const cuComplex *const, int, cuComplex *const, int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCmatinvBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n,
                                                                const cuDoubleComplex *const A, int lda,
                                                                cuDoubleComplex *const Ainv, int lda_inv, int *info,
                                                                int batchSize) {
    HOOK_TRACE_PROFILE("cublasZmatinvBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, const cuDoubleComplex *const, int, cuDoubleComplex *const,
                                        int, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZmatinvBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float *const Aarray,
                                                               int lda, float *const TauArray, int *info,
                                                               int batchSize) {
    HOOK_TRACE_PROFILE("cublasSgeqrfBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, float *const, int, float *const, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgeqrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n,
                                                               double *const Aarray, int lda, double *const TauArray,
                                                               int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasDgeqrfBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, double *const, int, double *const, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgeqrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n,
                                                               cuComplex *const Aarray, int lda,
                                                               cuComplex *const TauArray, int *info, int batchSize) {
    HOOK_TRACE_PROFILE("cublasCgeqrfBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, int, int, cuComplex *const, int, cuComplex *const, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgeqrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n,
                                                               cuDoubleComplex *const Aarray, int lda,
                                                               cuDoubleComplex *const TauArray, int *info,
                                                               int batchSize) {
    HOOK_TRACE_PROFILE("cublasZgeqrfBatched");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, int, int, cuDoubleComplex *const, int, cuDoubleComplex *const, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgeqrfBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, m, n, Aarray, lda, TauArray, info, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m,
                                                              int n, int nrhs, float *const Aarray, int lda,
                                                              float *const Carray, int ldc, int *info,
                                                              int *devInfoArray, int batchSize) {
    HOOK_TRACE_PROFILE("cublasSgelsBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, float *const, int,
                                        float *const, int, int *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSgelsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m,
                                                              int n, int nrhs, double *const Aarray, int lda,
                                                              double *const Carray, int ldc, int *info,
                                                              int *devInfoArray, int batchSize) {
    HOOK_TRACE_PROFILE("cublasDgelsBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, double *const, int,
                                        double *const, int, int *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDgelsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m,
                                                              int n, int nrhs, cuComplex *const Aarray, int lda,
                                                              cuComplex *const Carray, int ldc, int *info,
                                                              int *devInfoArray, int batchSize) {
    HOOK_TRACE_PROFILE("cublasCgelsBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, cuComplex *const, int,
                                        cuComplex *const, int, int *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCgelsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m,
                                                              int n, int nrhs, cuDoubleComplex *const Aarray, int lda,
                                                              cuDoubleComplex *const Carray, int ldc, int *info,
                                                              int *devInfoArray, int batchSize) {
    HOOK_TRACE_PROFILE("cublasZgelsBatched");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, int, int, int, cuDoubleComplex *const, int,
                                        cuDoubleComplex *const, int, int *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZgelsBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                                                       const float *A, int lda, const float *x, int incx, float *C,
                                                       int ldc) {
    HOOK_TRACE_PROFILE("cublasSdgmm");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const float *, int, const float *,
                                        int, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasSdgmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                                                       const double *A, int lda, const double *x, int incx, double *C,
                                                       int ldc) {
    HOOK_TRACE_PROFILE("cublasDdgmm");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const double *, int, const double *,
                                        int, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDdgmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                                                       const cuComplex *A, int lda, const cuComplex *x, int incx,
                                                       cuComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasCdgmm");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const cuComplex *, int,
                                        const cuComplex *, int, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCdgmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n,
                                                       const cuDoubleComplex *A, int lda, const cuDoubleComplex *x,
                                                       int incx, cuDoubleComplex *C, int ldc) {
    HOOK_TRACE_PROFILE("cublasZdgmm");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, int, int, const cuDoubleComplex *, int,
                                        const cuDoubleComplex *, int, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZdgmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, m, n, A, lda, x, incx, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                        const float *AP, float *A, int lda) {
    HOOK_TRACE_PROFILE("cublasStpttr");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, float *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStpttr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, AP, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                        const double *AP, double *A, int lda) {
    HOOK_TRACE_PROFILE("cublasDtpttr");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, double *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtpttr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, AP, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                        const cuComplex *AP, cuComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasCtpttr");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, cuComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtpttr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, AP, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                        const cuDoubleComplex *AP, cuDoubleComplex *A, int lda) {
    HOOK_TRACE_PROFILE("cublasZtpttr");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, cuDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtpttr"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, AP, A, lda);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                        const float *A, int lda, float *AP) {
    HOOK_TRACE_PROFILE("cublasStrttp");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const float *, int, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasStrttp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                        const double *A, int lda, double *AP) {
    HOOK_TRACE_PROFILE("cublasDtrttp");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const double *, int, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasDtrttp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                        const cuComplex *A, int lda, cuComplex *AP) {
    HOOK_TRACE_PROFILE("cublasCtrttp");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuComplex *, int, cuComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasCtrttp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                                                        const cuDoubleComplex *A, int lda, cuDoubleComplex *AP) {
    HOOK_TRACE_PROFILE("cublasZtrttp");
    using func_ptr =
        cublasStatus_t (*)(cublasHandle_t, cublasFillMode_t, int, const cuDoubleComplex *, int, cuDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasZtrttp"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, n, A, lda, AP);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasMigrateComputeType(cublasHandle_t handle, cudaDataType_t dataType,
                                                                    cublasComputeType_t *computeType) {
    HOOK_TRACE_PROFILE("cublasMigrateComputeType");
    using func_ptr = cublasStatus_t (*)(cublasHandle_t, cudaDataType_t, cublasComputeType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasMigrateComputeType"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dataType, computeType);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCreate(cublasXtHandle_t *handle) {
    HOOK_TRACE_PROFILE("cublasXtCreate");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDestroy(cublasXtHandle_t handle) {
    HOOK_TRACE_PROFILE("cublasXtDestroy");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtGetNumBoards(int nbDevices, int deviceId, int *nbBoards) {
    HOOK_TRACE_PROFILE("cublasXtGetNumBoards");
    using func_ptr = cublasStatus_t (*)(int, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtGetNumBoards"));
    HOOK_CHECK(func_entry);
    return func_entry(nbDevices, deviceId, nbBoards);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtMaxBoards(int *nbGpuBoards) {
    HOOK_TRACE_PROFILE("cublasXtMaxBoards");
    using func_ptr = cublasStatus_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtMaxBoards"));
    HOOK_CHECK(func_entry);
    return func_entry(nbGpuBoards);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDeviceSelect(cublasXtHandle_t handle, int nbDevices, int deviceId) {
    HOOK_TRACE_PROFILE("cublasXtDeviceSelect");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDeviceSelect"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nbDevices, deviceId);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSetBlockDim(cublasXtHandle_t handle, int blockDim) {
    HOOK_TRACE_PROFILE("cublasXtSetBlockDim");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSetBlockDim"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, blockDim);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtGetBlockDim(cublasXtHandle_t handle, int *blockDim) {
    HOOK_TRACE_PROFILE("cublasXtGetBlockDim");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtGetBlockDim"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, blockDim);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtGetPinningMemMode(cublasXtHandle_t handle,
                                                                     cublasXtPinnedMemMode_t *mode) {
    HOOK_TRACE_PROFILE("cublasXtGetPinningMemMode");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasXtPinnedMemMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtGetPinningMemMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSetPinningMemMode(cublasXtHandle_t handle,
                                                                     cublasXtPinnedMemMode_t mode) {
    HOOK_TRACE_PROFILE("cublasXtSetPinningMemMode");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasXtPinnedMemMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSetPinningMemMode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSetCpuRoutine(cublasXtHandle_t handle, cublasXtBlasOp_t blasOp,
                                                                 cublasXtOpType_t type, void *blasFunctor) {
    HOOK_TRACE_PROFILE("cublasXtSetCpuRoutine");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasXtBlasOp_t, cublasXtOpType_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSetCpuRoutine"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, blasOp, type, blasFunctor);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSetCpuRatio(cublasXtHandle_t handle, cublasXtBlasOp_t blasOp,
                                                               cublasXtOpType_t type, float ratio) {
    HOOK_TRACE_PROFILE("cublasXtSetCpuRatio");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasXtBlasOp_t, cublasXtOpType_t, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSetCpuRatio"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, blasOp, type, ratio);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSgemm(cublasXtHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, size_t m, size_t n, size_t k,
                                                         const float *alpha, const float *A, size_t lda, const float *B,
                                                         size_t ldb, const float *beta, float *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtSgemm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasOperation_t, cublasOperation_t, size_t, size_t, size_t,
                           const float *, const float *, size_t, const float *, size_t, const float *, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDgemm(cublasXtHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, size_t m, size_t n, size_t k,
                                                         const double *alpha, const double *A, size_t lda,
                                                         const double *B, size_t ldb, const double *beta, double *C,
                                                         size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtDgemm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasOperation_t, cublasOperation_t, size_t, size_t, size_t,
                                        const double *, const double *, size_t, const double *, size_t, const double *,
                                        double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCgemm(cublasXtHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, size_t m, size_t n, size_t k,
                                                         const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                         const cuComplex *B, size_t ldb, const cuComplex *beta,
                                                         cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCgemm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasOperation_t, cublasOperation_t, size_t, size_t, size_t,
                                        const cuComplex *, const cuComplex *, size_t, const cuComplex *, size_t,
                                        const cuComplex *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZgemm(cublasXtHandle_t handle, cublasOperation_t transa,
                                                         cublasOperation_t transb, size_t m, size_t n, size_t k,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                         size_t lda, const cuDoubleComplex *B, size_t ldb,
                                                         const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZgemm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasOperation_t, cublasOperation_t, size_t, size_t, size_t,
                           const cuDoubleComplex *, const cuDoubleComplex *, size_t, const cuDoubleComplex *, size_t,
                           const cuDoubleComplex *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZgemm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSsyrk(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                         cublasOperation_t trans, size_t n, size_t k,
                                                         const float *alpha, const float *A, size_t lda,
                                                         const float *beta, float *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtSsyrk");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                        const float *, const float *, size_t, const float *, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDsyrk(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                         cublasOperation_t trans, size_t n, size_t k,
                                                         const double *alpha, const double *A, size_t lda,
                                                         const double *beta, double *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtDsyrk");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                        const double *, const double *, size_t, const double *, double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCsyrk(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                         cublasOperation_t trans, size_t n, size_t k,
                                                         const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                         const cuComplex *beta, cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCsyrk");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t, const cuComplex *,
                           const cuComplex *, size_t, const cuComplex *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZsyrk(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                         cublasOperation_t trans, size_t n, size_t k,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                         size_t lda, const cuDoubleComplex *beta, cuDoubleComplex *C,
                                                         size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZsyrk");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                        const cuDoubleComplex *, const cuDoubleComplex *, size_t,
                                        const cuDoubleComplex *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZsyrk"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCherk(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                         cublasOperation_t trans, size_t n, size_t k,
                                                         const float *alpha, const cuComplex *A, size_t lda,
                                                         const float *beta, cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCherk");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                        const float *, const cuComplex *, size_t, const float *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCherk"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZherk(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                         cublasOperation_t trans, size_t n, size_t k,
                                                         const double *alpha, const cuDoubleComplex *A, size_t lda,
                                                         const double *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZherk");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t, const double *,
                           const cuDoubleComplex *, size_t, const double *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZherk"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const float *alpha, const float *A, size_t lda,
                                                          const float *B, size_t ldb, const float *beta, float *C,
                                                          size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtSsyr2k");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t, const float *,
                           const float *, size_t, const float *, size_t, const float *, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const double *alpha, const double *A, size_t lda,
                                                          const double *B, size_t ldb, const double *beta, double *C,
                                                          size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtDsyr2k");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t, const double *,
                           const double *, size_t, const double *, size_t, const double *, double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                          const cuComplex *B, size_t ldb, const cuComplex *beta,
                                                          cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCsyr2k");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                        const cuComplex *, const cuComplex *, size_t, const cuComplex *, size_t,
                                        const cuComplex *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          size_t lda, const cuDoubleComplex *B, size_t ldb,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZsyr2k");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                           const cuDoubleComplex *, const cuDoubleComplex *, size_t, const cuDoubleComplex *, size_t,
                           const cuDoubleComplex *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZsyr2k"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCherkx(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                          const cuComplex *B, size_t ldb, const float *beta,
                                                          cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCherkx");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t, const cuComplex *,
                           const cuComplex *, size_t, const cuComplex *, size_t, const float *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCherkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZherkx(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          size_t lda, const cuDoubleComplex *B, size_t ldb,
                                                          const double *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZherkx");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                        const cuDoubleComplex *, const cuDoubleComplex *, size_t,
                                        const cuDoubleComplex *, size_t, const double *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZherkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtStrsm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, size_t m, size_t n, const float *alpha,
                                                         const float *A, size_t lda, float *B, size_t ldb) {
    HOOK_TRACE_PROFILE("cublasXtStrsm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                           size_t, size_t, const float *, const float *, size_t, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtStrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDtrsm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, size_t m, size_t n, const double *alpha,
                                                         const double *A, size_t lda, double *B, size_t ldb) {
    HOOK_TRACE_PROFILE("cublasXtDtrsm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                           size_t, size_t, const double *, const double *, size_t, double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDtrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCtrsm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, size_t m, size_t n,
                                                         const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                         cuComplex *B, size_t ldb) {
    HOOK_TRACE_PROFILE("cublasXtCtrsm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                           size_t, size_t, const cuComplex *, const cuComplex *, size_t, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCtrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZtrsm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, size_t m, size_t n,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                         size_t lda, cuDoubleComplex *B, size_t ldb) {
    HOOK_TRACE_PROFILE("cublasXtZtrsm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                        cublasDiagType_t, size_t, size_t, const cuDoubleComplex *,
                                        const cuDoubleComplex *, size_t, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZtrsm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSsymm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n, const float *alpha,
                                                         const float *A, size_t lda, const float *B, size_t ldb,
                                                         const float *beta, float *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtSsymm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t, const float *,
                           const float *, size_t, const float *, size_t, const float *, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDsymm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n, const double *alpha,
                                                         const double *A, size_t lda, const double *B, size_t ldb,
                                                         const double *beta, double *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtDsymm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t, const double *,
                           const double *, size_t, const double *, size_t, const double *, double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCsymm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n,
                                                         const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                         const cuComplex *B, size_t ldb, const cuComplex *beta,
                                                         cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCsymm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t,
                                        const cuComplex *, const cuComplex *, size_t, const cuComplex *, size_t,
                                        const cuComplex *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZsymm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                         size_t lda, const cuDoubleComplex *B, size_t ldb,
                                                         const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZsymm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t,
                           const cuDoubleComplex *, const cuDoubleComplex *, size_t, const cuDoubleComplex *, size_t,
                           const cuDoubleComplex *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZsymm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtChemm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n,
                                                         const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                         const cuComplex *B, size_t ldb, const cuComplex *beta,
                                                         cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtChemm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t,
                                        const cuComplex *, const cuComplex *, size_t, const cuComplex *, size_t,
                                        const cuComplex *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtChemm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZhemm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                         size_t lda, const cuDoubleComplex *B, size_t ldb,
                                                         const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZhemm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t,
                           const cuDoubleComplex *, const cuDoubleComplex *, size_t, const cuDoubleComplex *, size_t,
                           const cuDoubleComplex *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZhemm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSsyrkx(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const float *alpha, const float *A, size_t lda,
                                                          const float *B, size_t ldb, const float *beta, float *C,
                                                          size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtSsyrkx");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t, const float *,
                           const float *, size_t, const float *, size_t, const float *, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSsyrkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDsyrkx(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const double *alpha, const double *A, size_t lda,
                                                          const double *B, size_t ldb, const double *beta, double *C,
                                                          size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtDsyrkx");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t, const double *,
                           const double *, size_t, const double *, size_t, const double *, double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDsyrkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCsyrkx(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                          const cuComplex *B, size_t ldb, const cuComplex *beta,
                                                          cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCsyrkx");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                        const cuComplex *, const cuComplex *, size_t, const cuComplex *, size_t,
                                        const cuComplex *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCsyrkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZsyrkx(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          size_t lda, const cuDoubleComplex *B, size_t ldb,
                                                          const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZsyrkx");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                           const cuDoubleComplex *, const cuDoubleComplex *, size_t, const cuDoubleComplex *, size_t,
                           const cuDoubleComplex *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZsyrkx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCher2k(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                          const cuComplex *B, size_t ldb, const float *beta,
                                                          cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCher2k");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t, const cuComplex *,
                           const cuComplex *, size_t, const cuComplex *, size_t, const float *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCher2k"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZher2k(cublasXtHandle_t handle, cublasFillMode_t uplo,
                                                          cublasOperation_t trans, size_t n, size_t k,
                                                          const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                          size_t lda, const cuDoubleComplex *B, size_t ldb,
                                                          const double *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZher2k");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                        const cuDoubleComplex *, const cuDoubleComplex *, size_t,
                                        const cuDoubleComplex *, size_t, const double *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZher2k"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtSspmm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n, const float *alpha,
                                                         const float *AP, const float *B, size_t ldb, const float *beta,
                                                         float *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtSspmm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t, const float *,
                           const float *, const float *, size_t, const float *, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtSspmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDspmm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n, const double *alpha,
                                                         const double *AP, const double *B, size_t ldb,
                                                         const double *beta, double *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtDspmm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t, const double *,
                           const double *, const double *, size_t, const double *, double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDspmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCspmm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n,
                                                         const cuComplex *alpha, const cuComplex *AP,
                                                         const cuComplex *B, size_t ldb, const cuComplex *beta,
                                                         cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCspmm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t, const cuComplex *,
                           const cuComplex *, const cuComplex *, size_t, const cuComplex *, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCspmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZspmm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, size_t m, size_t n,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *AP,
                                                         const cuDoubleComplex *B, size_t ldb,
                                                         const cuDoubleComplex *beta, cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZspmm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t,
                                        const cuDoubleComplex *, const cuDoubleComplex *, const cuDoubleComplex *,
                                        size_t, const cuDoubleComplex *, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZspmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, m, n, alpha, AP, B, ldb, beta, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtStrmm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, size_t m, size_t n, const float *alpha,
                                                         const float *A, size_t lda, const float *B, size_t ldb,
                                                         float *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtStrmm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                        cublasDiagType_t, size_t, size_t, const float *, const float *, size_t,
                                        const float *, size_t, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtStrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtDtrmm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, size_t m, size_t n, const double *alpha,
                                                         const double *A, size_t lda, const double *B, size_t ldb,
                                                         double *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtDtrmm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                        cublasDiagType_t, size_t, size_t, const double *, const double *, size_t,
                                        const double *, size_t, double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtDtrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtCtrmm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, size_t m, size_t n,
                                                         const cuComplex *alpha, const cuComplex *A, size_t lda,
                                                         const cuComplex *B, size_t ldb, cuComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtCtrmm");
    using func_ptr = cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                        cublasDiagType_t, size_t, size_t, const cuComplex *, const cuComplex *, size_t,
                                        const cuComplex *, size_t, cuComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtCtrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}

HOOK_C_API HOOK_DECL_EXPORT cublasStatus_t cublasXtZtrmm(cublasXtHandle_t handle, cublasSideMode_t side,
                                                         cublasFillMode_t uplo, cublasOperation_t trans,
                                                         cublasDiagType_t diag, size_t m, size_t n,
                                                         const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                                         size_t lda, const cuDoubleComplex *B, size_t ldb,
                                                         cuDoubleComplex *C, size_t ldc) {
    HOOK_TRACE_PROFILE("cublasXtZtrmm");
    using func_ptr =
        cublasStatus_t (*)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
                           size_t, size_t, const cuDoubleComplex *, const cuDoubleComplex *, size_t,
                           const cuDoubleComplex *, size_t, cuDoubleComplex *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUBLAS_SYMBOL("cublasXtZtrmm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}
