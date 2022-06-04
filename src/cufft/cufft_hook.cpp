// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 52 apis

#include "cufft_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch) {
    HOOK_TRACE_PROFILE("cufftPlan1d");
    using func_ptr = cufftResult (*)(cufftHandle *, int, cufftType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftPlan1d"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, nx, type, batch);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type) {
    HOOK_TRACE_PROFILE("cufftPlan2d");
    using func_ptr = cufftResult (*)(cufftHandle *, int, int, cufftType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftPlan2d"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, nx, ny, type);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type) {
    HOOK_TRACE_PROFILE("cufftPlan3d");
    using func_ptr = cufftResult (*)(cufftHandle *, int, int, int, cufftType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftPlan3d"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, nx, ny, nz, type);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride,
                                                      int idist, int *onembed, int ostride, int odist, cufftType type,
                                                      int batch) {
    HOOK_TRACE_PROFILE("cufftPlanMany");
    using func_ptr = cufftResult (*)(cufftHandle *, int, int *, int *, int, int, int *, int, int, cufftType, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftPlanMany"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
                                                        size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftMakePlan1d");
    using func_ptr = cufftResult (*)(cufftHandle, int, cufftType, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftMakePlan1d"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, nx, type, batch, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type,
                                                        size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftMakePlan2d");
    using func_ptr = cufftResult (*)(cufftHandle, int, int, cufftType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftMakePlan2d"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, nx, ny, type, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type,
                                                        size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftMakePlan3d");
    using func_ptr = cufftResult (*)(cufftHandle, int, int, int, cufftType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftMakePlan3d"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, nx, ny, nz, type, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed, int istride,
                                                          int idist, int *onembed, int ostride, int odist,
                                                          cufftType type, int batch, size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftMakePlanMany");
    using func_ptr =
        cufftResult (*)(cufftHandle, int, int *, int *, int, int, int *, int, int, cufftType, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftMakePlanMany"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n,
                                                            long long int *inembed, long long int istride,
                                                            long long int idist, long long int *onembed,
                                                            long long int ostride, long long int odist, cufftType type,
                                                            long long int batch, size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftMakePlanMany64");
    using func_ptr = cufftResult (*)(cufftHandle, int, long long int *, long long int *, long long int, long long int,
                                     long long int *, long long int, long long int, cufftType, long long int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftMakePlanMany64"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int *n,
                                                           long long int *inembed, long long int istride,
                                                           long long int idist, long long int *onembed,
                                                           long long int ostride, long long int odist, cufftType type,
                                                           long long int batch, size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftGetSizeMany64");
    using func_ptr = cufftResult (*)(cufftHandle, int, long long int *, long long int *, long long int, long long int,
                                     long long int *, long long int, long long int, cufftType, long long int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftGetSizeMany64"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftEstimate1d");
    using func_ptr = cufftResult (*)(int, cufftType, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftEstimate1d"));
    HOOK_CHECK(func_entry);
    return func_entry(nx, type, batch, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftEstimate2d");
    using func_ptr = cufftResult (*)(int, int, cufftType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftEstimate2d"));
    HOOK_CHECK(func_entry);
    return func_entry(nx, ny, type, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftEstimate3d");
    using func_ptr = cufftResult (*)(int, int, int, cufftType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftEstimate3d"));
    HOOK_CHECK(func_entry);
    return func_entry(nx, ny, nz, type, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftEstimateMany(int rank, int *n, int *inembed, int istride, int idist,
                                                          int *onembed, int ostride, int odist, cufftType type,
                                                          int batch, size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftEstimateMany");
    using func_ptr = cufftResult (*)(int, int *, int *, int, int, int *, int, int, cufftType, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftEstimateMany"));
    HOOK_CHECK(func_entry);
    return func_entry(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftCreate(cufftHandle *handle) {
    HOOK_TRACE_PROFILE("cufftCreate");
    using func_ptr = cufftResult (*)(cufftHandle *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type, int batch,
                                                       size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftGetSize1d");
    using func_ptr = cufftResult (*)(cufftHandle, int, cufftType, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftGetSize1d"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nx, type, batch, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type,
                                                       size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftGetSize2d");
    using func_ptr = cufftResult (*)(cufftHandle, int, int, cufftType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftGetSize2d"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nx, ny, type, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type,
                                                       size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftGetSize3d");
    using func_ptr = cufftResult (*)(cufftHandle, int, int, int, cufftType, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftGetSize3d"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nx, ny, nz, type, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int *n, int *inembed,
                                                         int istride, int idist, int *onembed, int ostride, int odist,
                                                         cufftType type, int batch, size_t *workArea) {
    HOOK_TRACE_PROFILE("cufftGetSizeMany");
    using func_ptr =
        cufftResult (*)(cufftHandle, int, int *, int *, int, int, int *, int, int, cufftType, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftGetSizeMany"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workArea);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftGetSize(cufftHandle handle, size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftGetSize");
    using func_ptr = cufftResult (*)(cufftHandle, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftGetSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, workSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea) {
    HOOK_TRACE_PROFILE("cufftSetWorkArea");
    using func_ptr = cufftResult (*)(cufftHandle, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftSetWorkArea"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, workArea);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) {
    HOOK_TRACE_PROFILE("cufftSetAutoAllocation");
    using func_ptr = cufftResult (*)(cufftHandle, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftSetAutoAllocation"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, autoAllocate);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata,
                                                     int direction) {
    HOOK_TRACE_PROFILE("cufftExecC2C");
    using func_ptr = cufftResult (*)(cufftHandle, cufftComplex *, cufftComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftExecC2C"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, idata, odata, direction);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata) {
    HOOK_TRACE_PROFILE("cufftExecR2C");
    using func_ptr = cufftResult (*)(cufftHandle, cufftReal *, cufftComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftExecR2C"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, idata, odata);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata) {
    HOOK_TRACE_PROFILE("cufftExecC2R");
    using func_ptr = cufftResult (*)(cufftHandle, cufftComplex *, cufftReal *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftExecC2R"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, idata, odata);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata,
                                                     cufftDoubleComplex *odata, int direction) {
    HOOK_TRACE_PROFILE("cufftExecZ2Z");
    using func_ptr = cufftResult (*)(cufftHandle, cufftDoubleComplex *, cufftDoubleComplex *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftExecZ2Z"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, idata, odata, direction);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata,
                                                     cufftDoubleComplex *odata) {
    HOOK_TRACE_PROFILE("cufftExecD2Z");
    using func_ptr = cufftResult (*)(cufftHandle, cufftDoubleReal *, cufftDoubleComplex *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftExecD2Z"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, idata, odata);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata,
                                                     cufftDoubleReal *odata) {
    HOOK_TRACE_PROFILE("cufftExecZ2D");
    using func_ptr = cufftResult (*)(cufftHandle, cufftDoubleComplex *, cufftDoubleReal *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftExecZ2D"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, idata, odata);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cufftSetStream");
    using func_ptr = cufftResult (*)(cufftHandle, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftSetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftDestroy(cufftHandle plan) {
    HOOK_TRACE_PROFILE("cufftDestroy");
    using func_ptr = cufftResult (*)(cufftHandle);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(plan);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftGetVersion(int *version) {
    HOOK_TRACE_PROFILE("cufftGetVersion");
    using func_ptr = cufftResult (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(version);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftGetProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("cufftGetProperty");
    using func_ptr = cufftResult (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftGetProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtSetGPUs(cufftHandle handle, int nGPUs, int *whichGPUs) {
    HOOK_TRACE_PROFILE("cufftXtSetGPUs");
    using func_ptr = cufftResult (*)(cufftHandle, int, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtSetGPUs"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, nGPUs, whichGPUs);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc **descriptor,
                                                      cufftXtSubFormat format) {
    HOOK_TRACE_PROFILE("cufftXtMalloc");
    using func_ptr = cufftResult (*)(cufftHandle, cudaLibXtDesc **, cufftXtSubFormat);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtMalloc"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, descriptor, format);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtMemcpy(cufftHandle plan, void *dstPointer, void *srcPointer,
                                                      cufftXtCopyType type) {
    HOOK_TRACE_PROFILE("cufftXtMemcpy");
    using func_ptr = cufftResult (*)(cufftHandle, void *, void *, cufftXtCopyType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtMemcpy"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, dstPointer, srcPointer, type);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtFree(cudaLibXtDesc *descriptor) {
    HOOK_TRACE_PROFILE("cufftXtFree");
    using func_ptr = cufftResult (*)(cudaLibXtDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtFree"));
    HOOK_CHECK(func_entry);
    return func_entry(descriptor);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtSetWorkArea(cufftHandle plan, void **workArea) {
    HOOK_TRACE_PROFILE("cufftXtSetWorkArea");
    using func_ptr = cufftResult (*)(cufftHandle, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtSetWorkArea"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, workArea);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc *input,
                                                                 cudaLibXtDesc *output, int direction) {
    HOOK_TRACE_PROFILE("cufftXtExecDescriptorC2C");
    using func_ptr = cufftResult (*)(cufftHandle, cudaLibXtDesc *, cudaLibXtDesc *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtExecDescriptorC2C"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, input, output, direction);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtExecDescriptorR2C(cufftHandle plan, cudaLibXtDesc *input,
                                                                 cudaLibXtDesc *output) {
    HOOK_TRACE_PROFILE("cufftXtExecDescriptorR2C");
    using func_ptr = cufftResult (*)(cufftHandle, cudaLibXtDesc *, cudaLibXtDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtExecDescriptorR2C"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, input, output);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtExecDescriptorC2R(cufftHandle plan, cudaLibXtDesc *input,
                                                                 cudaLibXtDesc *output) {
    HOOK_TRACE_PROFILE("cufftXtExecDescriptorC2R");
    using func_ptr = cufftResult (*)(cufftHandle, cudaLibXtDesc *, cudaLibXtDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtExecDescriptorC2R"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, input, output);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc *input,
                                                                 cudaLibXtDesc *output, int direction) {
    HOOK_TRACE_PROFILE("cufftXtExecDescriptorZ2Z");
    using func_ptr = cufftResult (*)(cufftHandle, cudaLibXtDesc *, cudaLibXtDesc *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtExecDescriptorZ2Z"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, input, output, direction);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtExecDescriptorD2Z(cufftHandle plan, cudaLibXtDesc *input,
                                                                 cudaLibXtDesc *output) {
    HOOK_TRACE_PROFILE("cufftXtExecDescriptorD2Z");
    using func_ptr = cufftResult (*)(cufftHandle, cudaLibXtDesc *, cudaLibXtDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtExecDescriptorD2Z"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, input, output);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtExecDescriptorZ2D(cufftHandle plan, cudaLibXtDesc *input,
                                                                 cudaLibXtDesc *output) {
    HOOK_TRACE_PROFILE("cufftXtExecDescriptorZ2D");
    using func_ptr = cufftResult (*)(cufftHandle, cudaLibXtDesc *, cudaLibXtDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtExecDescriptorZ2D"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, input, output);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtQueryPlan(cufftHandle plan, void *queryStruct,
                                                         cufftXtQueryType queryType) {
    HOOK_TRACE_PROFILE("cufftXtQueryPlan");
    using func_ptr = cufftResult (*)(cufftHandle, void *, cufftXtQueryType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtQueryPlan"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, queryStruct, queryType);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtSetCallback(cufftHandle plan, void **callback_routine,
                                                           cufftXtCallbackType cbType, void **caller_info) {
    HOOK_TRACE_PROFILE("cufftXtSetCallback");
    using func_ptr = cufftResult (*)(cufftHandle, void **, cufftXtCallbackType, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtSetCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, callback_routine, cbType, caller_info);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType cbType) {
    HOOK_TRACE_PROFILE("cufftXtClearCallback");
    using func_ptr = cufftResult (*)(cufftHandle, cufftXtCallbackType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtClearCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, cbType);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtSetCallbackSharedSize(cufftHandle plan, cufftXtCallbackType cbType,
                                                                     size_t sharedSize) {
    HOOK_TRACE_PROFILE("cufftXtSetCallbackSharedSize");
    using func_ptr = cufftResult (*)(cufftHandle, cufftXtCallbackType, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtSetCallbackSharedSize"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, cbType, sharedSize);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtMakePlanMany(
    cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist,
    cudaDataType inputtype, long long int *onembed, long long int ostride, long long int odist, cudaDataType outputtype,
    long long int batch, size_t *workSize, cudaDataType executiontype) {
    HOOK_TRACE_PROFILE("cufftXtMakePlanMany");
    using func_ptr = cufftResult (*)(cufftHandle, int, long long int *, long long int *, long long int, long long int,
                                     cudaDataType, long long int *, long long int, long long int, cudaDataType,
                                     long long int, size_t *, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtMakePlanMany"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch,
                      workSize, executiontype);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtGetSizeMany(
    cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist,
    cudaDataType inputtype, long long int *onembed, long long int ostride, long long int odist, cudaDataType outputtype,
    long long int batch, size_t *workSize, cudaDataType executiontype) {
    HOOK_TRACE_PROFILE("cufftXtGetSizeMany");
    using func_ptr = cufftResult (*)(cufftHandle, int, long long int *, long long int *, long long int, long long int,
                                     cudaDataType, long long int *, long long int, long long int, cudaDataType,
                                     long long int, size_t *, cudaDataType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtGetSizeMany"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, rank, n, inembed, istride, idist, inputtype, onembed, ostride, odist, outputtype, batch,
                      workSize, executiontype);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtExec(cufftHandle plan, void *input, void *output, int direction) {
    HOOK_TRACE_PROFILE("cufftXtExec");
    using func_ptr = cufftResult (*)(cufftHandle, void *, void *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtExec"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, input, output, direction);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc *input,
                                                              cudaLibXtDesc *output, int direction) {
    HOOK_TRACE_PROFILE("cufftXtExecDescriptor");
    using func_ptr = cufftResult (*)(cufftHandle, cudaLibXtDesc *, cudaLibXtDesc *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtExecDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, input, output, direction);
}

HOOK_C_API HOOK_DECL_EXPORT cufftResult cufftXtSetWorkAreaPolicy(cufftHandle plan, cufftXtWorkAreaPolicy policy,
                                                                 size_t *workSize) {
    HOOK_TRACE_PROFILE("cufftXtSetWorkAreaPolicy");
    using func_ptr = cufftResult (*)(cufftHandle, cufftXtWorkAreaPolicy, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUFFT_SYMBOL("cufftXtSetWorkAreaPolicy"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, policy, workSize);
}
