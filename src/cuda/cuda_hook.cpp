// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 378 apis, manually add 33 apis and delete 1 api

#include "cuda_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetErrorString(CUresult error, const char **pStr) {
    HOOK_TRACE_PROFILE("cuGetErrorString");
    using func_ptr = CUresult (*)(CUresult, const char **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGetErrorString"));
    HOOK_CHECK(func_entry);
    return func_entry(error, pStr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetErrorName(CUresult error, const char **pStr) {
    HOOK_TRACE_PROFILE("cuGetErrorName");
    using func_ptr = CUresult (*)(CUresult, const char **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGetErrorName"));
    HOOK_CHECK(func_entry);
    return func_entry(error, pStr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuInit(unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuInit");
    using func_ptr = CUresult (*)(unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuInit"));
    HOOK_CHECK(func_entry);
    return func_entry(Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDriverGetVersion(int *driverVersion) {
    HOOK_TRACE_PROFILE("cuDriverGetVersion");
    using func_ptr = CUresult (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDriverGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(driverVersion);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    HOOK_TRACE_PROFILE("cuDeviceGet");
    using func_ptr = CUresult (*)(CUdevice *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGet"));
    HOOK_CHECK(func_entry);
    return func_entry(device, ordinal);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetCount(int *count) {
    HOOK_TRACE_PROFILE("cuDeviceGetCount");
    using func_ptr = CUresult (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetCount"));
    HOOK_CHECK(func_entry);
    return func_entry(count);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetName");
    using func_ptr = CUresult (*)(char *, int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetName"));
    HOOK_CHECK(func_entry);
    return func_entry(name, len, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetUuid");
    using func_ptr = CUresult (*)(CUuuid *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetUuid"));
    HOOK_CHECK(func_entry);
    return func_entry(uuid, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetUuid_v2");
    using func_ptr = CUresult (*)(CUuuid *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetUuid_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(uuid, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetLuid");
    using func_ptr = CUresult (*)(char *, unsigned int *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetLuid"));
    HOOK_CHECK(func_entry);
    return func_entry(luid, deviceNodeMask, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceTotalMem");
    using func_ptr = CUresult (*)(size_t *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceTotalMem"));
    HOOK_CHECK(func_entry);
    return func_entry(bytes, dev);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceTotalMem_v2");
    using func_ptr = CUresult (*)(size_t *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceTotalMem_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(bytes, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
                                                                        CUarray_format format, unsigned numChannels,
                                                                        CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetTexture1DLinearMaxWidth");
    using func_ptr = CUresult (*)(size_t *, CUarray_format, unsigned, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetTexture1DLinearMaxWidth"));
    HOOK_CHECK(func_entry);
    return func_entry(maxWidthInElements, format, numChannels, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetAttribute");
    using func_ptr = CUresult (*)(int *, CUdevice_attribute, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(pi, attrib, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, CUdevice dev, int flags) {
    HOOK_TRACE_PROFILE("cuDeviceGetNvSciSyncAttributes");
    using func_ptr = CUresult (*)(void *, CUdevice, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetNvSciSyncAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(nvSciSyncAttrList, dev, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    HOOK_TRACE_PROFILE("cuDeviceSetMemPool");
    using func_ptr = CUresult (*)(CUdevice, CUmemoryPool);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceSetMemPool"));
    HOOK_CHECK(func_entry);
    return func_entry(dev, pool);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetMemPool");
    using func_ptr = CUresult (*)(CUmemoryPool *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetMemPool"));
    HOOK_CHECK(func_entry);
    return func_entry(pool, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetDefaultMemPool");
    using func_ptr = CUresult (*)(CUmemoryPool *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetDefaultMemPool"));
    HOOK_CHECK(func_entry);
    return func_entry(pool_out, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target,
                                                                CUflushGPUDirectRDMAWritesScope scope) {
    HOOK_TRACE_PROFILE("cuFlushGPUDirectRDMAWrites");
    using func_ptr = CUresult (*)(CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFlushGPUDirectRDMAWrites"));
    HOOK_CHECK(func_entry);
    return func_entry(target, scope);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetProperties");
    using func_ptr = CUresult (*)(CUdevprop *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetProperties"));
    HOOK_CHECK(func_entry);
    return func_entry(prop, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceComputeCapability");
    using func_ptr = CUresult (*)(int *, int *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceComputeCapability"));
    HOOK_CHECK(func_entry);
    return func_entry(major, minor, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDevicePrimaryCtxRetain");
    using func_ptr = CUresult (*)(CUcontext *, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxRetain"));
    HOOK_CHECK(func_entry);
    return func_entry(pctx, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDevicePrimaryCtxRelease");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxRelease"));
    HOOK_CHECK(func_entry);
    return func_entry(dev);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDevicePrimaryCtxRelease_v2");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxRelease_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuDevicePrimaryCtxSetFlags");
    using func_ptr = CUresult (*)(CUdevice, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxSetFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(dev, flags);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuDevicePrimaryCtxSetFlags_v2");
    using func_ptr = CUresult (*)(CUdevice, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxSetFlags_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dev, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    HOOK_TRACE_PROFILE("cuDevicePrimaryCtxGetState");
    using func_ptr = CUresult (*)(CUdevice, unsigned int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxGetState"));
    HOOK_CHECK(func_entry);
    return func_entry(dev, flags, active);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDevicePrimaryCtxReset");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxReset"));
    HOOK_CHECK(func_entry);
    return func_entry(dev);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDevicePrimaryCtxReset_v2");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxReset_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetExecAffinitySupport");
    using func_ptr = CUresult (*)(int *, CUexecAffinityType, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetExecAffinitySupport"));
    HOOK_CHECK(func_entry);
    return func_entry(pi, type, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuCtxCreate");
    using func_ptr = CUresult (*)(CUcontext *, unsigned int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pctx, flags, dev);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuCtxCreate_v2");
    using func_ptr = CUresult (*)(CUcontext *, unsigned int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxCreate_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pctx, flags, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *paramsArray, int numParams,
                                                    unsigned int flags, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuCtxCreate_v3");
    using func_ptr = CUresult (*)(CUcontext *, CUexecAffinityParam *, int, unsigned int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxCreate_v3"));
    HOOK_CHECK(func_entry);
    return func_entry(pctx, paramsArray, numParams, flags, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxDestroy(CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuCtxDestroy");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(ctx);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxDestroy_v2(CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuCtxDestroy_v2");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxDestroy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(ctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxPushCurrent(CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuCtxPushCurrent");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxPushCurrent"));
    HOOK_CHECK(func_entry);
    return func_entry(ctx);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuCtxPushCurrent_v2");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxPushCurrent_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(ctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxPopCurrent(CUcontext *pctx) {
    HOOK_TRACE_PROFILE("cuCtxPopCurrent");
    using func_ptr = CUresult (*)(CUcontext *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxPopCurrent"));
    HOOK_CHECK(func_entry);
    return func_entry(pctx);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    HOOK_TRACE_PROFILE("cuCtxPopCurrent_v2");
    using func_ptr = CUresult (*)(CUcontext *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxPopCurrent_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxSetCurrent(CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuCtxSetCurrent");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxSetCurrent"));
    HOOK_CHECK(func_entry);
    return func_entry(ctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetCurrent(CUcontext *pctx) {
    HOOK_TRACE_PROFILE("cuCtxGetCurrent");
    using func_ptr = CUresult (*)(CUcontext *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetCurrent"));
    HOOK_CHECK(func_entry);
    return func_entry(pctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetDevice(CUdevice *device) {
    HOOK_TRACE_PROFILE("cuCtxGetDevice");
    using func_ptr = CUresult (*)(CUdevice *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetFlags(unsigned int *flags) {
    HOOK_TRACE_PROFILE("cuCtxGetFlags");
    using func_ptr = CUresult (*)(unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxSynchronize() {
    HOOK_TRACE_PROFILE("cuCtxSynchronize");
    using func_ptr = CUresult (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    HOOK_TRACE_PROFILE("cuCtxSetLimit");
    using func_ptr = CUresult (*)(CUlimit, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxSetLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(limit, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    HOOK_TRACE_PROFILE("cuCtxGetLimit");
    using func_ptr = CUresult (*)(size_t *, CUlimit);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(pvalue, limit);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
    HOOK_TRACE_PROFILE("cuCtxGetCacheConfig");
    using func_ptr = CUresult (*)(CUfunc_cache *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetCacheConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(pconfig);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
    HOOK_TRACE_PROFILE("cuCtxSetCacheConfig");
    using func_ptr = CUresult (*)(CUfunc_cache);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxSetCacheConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(config);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
    HOOK_TRACE_PROFILE("cuCtxGetSharedMemConfig");
    using func_ptr = CUresult (*)(CUsharedconfig *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetSharedMemConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(pConfig);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
    HOOK_TRACE_PROFILE("cuCtxSetSharedMemConfig");
    using func_ptr = CUresult (*)(CUsharedconfig);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxSetSharedMemConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(config);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    HOOK_TRACE_PROFILE("cuCtxGetApiVersion");
    using func_ptr = CUresult (*)(CUcontext, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetApiVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(ctx, version);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    HOOK_TRACE_PROFILE("cuCtxGetStreamPriorityRange");
    using func_ptr = CUresult (*)(int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetStreamPriorityRange"));
    HOOK_CHECK(func_entry);
    return func_entry(leastPriority, greatestPriority);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxResetPersistingL2Cache() {
    HOOK_TRACE_PROFILE("cuCtxResetPersistingL2Cache");
    using func_ptr = CUresult (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxResetPersistingL2Cache"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity, CUexecAffinityType type) {
    HOOK_TRACE_PROFILE("cuCtxGetExecAffinity");
    using func_ptr = CUresult (*)(CUexecAffinityParam *, CUexecAffinityType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetExecAffinity"));
    HOOK_CHECK(func_entry);
    return func_entry(pExecAffinity, type);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuCtxAttach");
    using func_ptr = CUresult (*)(CUcontext *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxAttach"));
    HOOK_CHECK(func_entry);
    return func_entry(pctx, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxDetach(CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuCtxDetach");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxDetach"));
    HOOK_CHECK(func_entry);
    return func_entry(ctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    HOOK_TRACE_PROFILE("cuModuleLoad");
    using func_ptr = CUresult (*)(CUmodule *, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleLoad"));
    HOOK_CHECK(func_entry);
    return func_entry(module, fname);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    HOOK_TRACE_PROFILE("cuModuleLoadData");
    using func_ptr = CUresult (*)(CUmodule *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleLoadData"));
    HOOK_CHECK(func_entry);
    return func_entry(module, image);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                                                        CUjit_option *options, void **optionValues) {
    HOOK_TRACE_PROFILE("cuModuleLoadDataEx");
    using func_ptr = CUresult (*)(CUmodule *, const void *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleLoadDataEx"));
    HOOK_CHECK(func_entry);
    return func_entry(module, image, numOptions, options, optionValues);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    HOOK_TRACE_PROFILE("cuModuleLoadFatBinary");
    using func_ptr = CUresult (*)(CUmodule *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleLoadFatBinary"));
    HOOK_CHECK(func_entry);
    return func_entry(module, fatCubin);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleUnload(CUmodule hmod) {
    HOOK_TRACE_PROFILE("cuModuleUnload");
    using func_ptr = CUresult (*)(CUmodule);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleUnload"));
    HOOK_CHECK(func_entry);
    return func_entry(hmod);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    HOOK_TRACE_PROFILE("cuModuleGetFunction");
    using func_ptr = CUresult (*)(CUfunction *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleGetFunction"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, hmod, name);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                                       const char *name) {
    HOOK_TRACE_PROFILE("cuModuleGetGlobal");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleGetGlobal"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, bytes, hmod, name);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                                                          const char *name) {
    HOOK_TRACE_PROFILE("cuModuleGetGlobal_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleGetGlobal_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, bytes, hmod, name);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
    HOOK_TRACE_PROFILE("cuModuleGetTexRef");
    using func_ptr = CUresult (*)(CUtexref *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleGetTexRef"));
    HOOK_CHECK(func_entry);
    return func_entry(pTexRef, hmod, name);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name) {
    HOOK_TRACE_PROFILE("cuModuleGetSurfRef");
    using func_ptr = CUresult (*)(CUsurfref *, CUmodule, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleGetSurfRef"));
    HOOK_CHECK(func_entry);
    return func_entry(pSurfRef, hmod, name);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options, void **optionValues,
                                                  CUlinkState *stateOut) {
    HOOK_TRACE_PROFILE("cuLinkCreate");
    using func_ptr = CUresult (*)(unsigned int, CUjit_option *, void **, CUlinkState *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLinkCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(numOptions, options, optionValues, stateOut);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options,
                                                     void **optionValues, CUlinkState *stateOut) {
    HOOK_TRACE_PROFILE("cuLinkCreate_v2");
    using func_ptr = CUresult (*)(unsigned int, CUjit_option *, void **, CUlinkState *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLinkCreate_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(numOptions, options, optionValues, stateOut);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size,
                                                   const char *name, unsigned int numOptions, CUjit_option *options,
                                                   void **optionValues) {
    HOOK_TRACE_PROFILE("cuLinkAddData");
    using func_ptr =
        CUresult (*)(CUlinkState, CUjitInputType, void *, size_t, const char *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLinkAddData"));
    HOOK_CHECK(func_entry);
    return func_entry(state, type, data, size, name, numOptions, options, optionValues);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size,
                                                      const char *name, unsigned int numOptions, CUjit_option *options,
                                                      void **optionValues) {
    HOOK_TRACE_PROFILE("cuLinkAddData_v2");
    using func_ptr =
        CUresult (*)(CUlinkState, CUjitInputType, void *, size_t, const char *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLinkAddData_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(state, type, data, size, name, numOptions, options, optionValues);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char *path,
                                                   unsigned int numOptions, CUjit_option *options,
                                                   void **optionValues) {
    HOOK_TRACE_PROFILE("cuLinkAddFile");
    using func_ptr = CUresult (*)(CUlinkState, CUjitInputType, const char *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLinkAddFile"));
    HOOK_CHECK(func_entry);
    return func_entry(state, type, path, numOptions, options, optionValues);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path,
                                                      unsigned int numOptions, CUjit_option *options,
                                                      void **optionValues) {
    HOOK_TRACE_PROFILE("cuLinkAddFile_v2");
    using func_ptr = CUresult (*)(CUlinkState, CUjitInputType, const char *, unsigned int, CUjit_option *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLinkAddFile_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(state, type, path, numOptions, options, optionValues);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    HOOK_TRACE_PROFILE("cuLinkComplete");
    using func_ptr = CUresult (*)(CUlinkState, void **, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLinkComplete"));
    HOOK_CHECK(func_entry);
    return func_entry(state, cubinOut, sizeOut);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLinkDestroy(CUlinkState state) {
    HOOK_TRACE_PROFILE("cuLinkDestroy");
    using func_ptr = CUresult (*)(CUlinkState);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLinkDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(state);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemGetInfo(size_t *free, size_t *total) {
    HOOK_TRACE_PROFILE("cuMemGetInfo");
    using func_ptr = CUresult (*)(size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemGetInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(free, total);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    HOOK_TRACE_PROFILE("cuMemGetInfo_v2");
    using func_ptr = CUresult (*)(size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemGetInfo_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(free, total);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    HOOK_TRACE_PROFILE("cuMemAlloc");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAlloc"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, bytesize);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    HOOK_TRACE_PROFILE("cuMemAlloc_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAlloc_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, bytesize);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                                                     size_t Height, unsigned int ElementSizeBytes) {
    HOOK_TRACE_PROFILE("cuMemAllocPitch");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAllocPitch"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                                                        size_t Height, unsigned int ElementSizeBytes) {
    HOOK_TRACE_PROFILE("cuMemAllocPitch_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAllocPitch_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemFree(CUdeviceptr dptr) {
    HOOK_TRACE_PROFILE("cuMemFree");
    using func_ptr = CUresult (*)(CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemFree"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemFree_v2(CUdeviceptr dptr) {
    HOOK_TRACE_PROFILE("cuMemFree_v2");
    using func_ptr = CUresult (*)(CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemFree_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    HOOK_TRACE_PROFILE("cuMemGetAddressRange");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemGetAddressRange"));
    HOOK_CHECK(func_entry);
    return func_entry(pbase, psize, dptr);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    HOOK_TRACE_PROFILE("cuMemGetAddressRange_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemGetAddressRange_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pbase, psize, dptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    HOOK_TRACE_PROFILE("cuMemAllocHost");
    using func_ptr = CUresult (*)(void **, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAllocHost"));
    HOOK_CHECK(func_entry);
    return func_entry(pp, bytesize);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
    HOOK_TRACE_PROFILE("cuMemAllocHost_v2");
    using func_ptr = CUresult (*)(void **, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAllocHost_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pp, bytesize);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemFreeHost(void *p) {
    HOOK_TRACE_PROFILE("cuMemFreeHost");
    using func_ptr = CUresult (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemFreeHost"));
    HOOK_CHECK(func_entry);
    return func_entry(p);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuMemHostAlloc");
    using func_ptr = CUresult (*)(void **, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemHostAlloc"));
    HOOK_CHECK(func_entry);
    return func_entry(pp, bytesize, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuMemHostGetDevicePointer");
    using func_ptr = CUresult (*)(CUdeviceptr *, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemHostGetDevicePointer"));
    HOOK_CHECK(func_entry);
    return func_entry(pdptr, p, Flags);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuMemHostGetDevicePointer_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemHostGetDevicePointer_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pdptr, p, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    HOOK_TRACE_PROFILE("cuMemHostGetFlags");
    using func_ptr = CUresult (*)(unsigned int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemHostGetFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(pFlags, p);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuMemAllocManaged");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAllocManaged"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, bytesize, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    HOOK_TRACE_PROFILE("cuDeviceGetByPCIBusId");
    using func_ptr = CUresult (*)(CUdevice *, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetByPCIBusId"));
    HOOK_CHECK(func_entry);
    return func_entry(dev, pciBusId);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    HOOK_TRACE_PROFILE("cuDeviceGetPCIBusId");
    using func_ptr = CUresult (*)(char *, int, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetPCIBusId"));
    HOOK_CHECK(func_entry);
    return func_entry(pciBusId, len, dev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
    HOOK_TRACE_PROFILE("cuIpcGetEventHandle");
    using func_ptr = CUresult (*)(CUipcEventHandle *, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuIpcGetEventHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(pHandle, event);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
    HOOK_TRACE_PROFILE("cuIpcOpenEventHandle");
    using func_ptr = CUresult (*)(CUevent *, CUipcEventHandle);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuIpcOpenEventHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(phEvent, handle);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
    HOOK_TRACE_PROFILE("cuIpcGetMemHandle");
    using func_ptr = CUresult (*)(CUipcMemHandle *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuIpcGetMemHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(pHandle, dptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuIpcOpenMemHandle");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUipcMemHandle, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuIpcOpenMemHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(pdptr, handle, Flags);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                                                           unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuIpcOpenMemHandle_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUipcMemHandle, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuIpcOpenMemHandle_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pdptr, handle, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    HOOK_TRACE_PROFILE("cuIpcCloseMemHandle");
    using func_ptr = CUresult (*)(CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuIpcCloseMemHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuMemHostRegister");
    using func_ptr = CUresult (*)(void *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemHostRegister"));
    HOOK_CHECK(func_entry);
    return func_entry(p, bytesize, Flags);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuMemHostRegister_v2");
    using func_ptr = CUresult (*)(void *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemHostRegister_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(p, bytesize, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemHostUnregister(void *p) {
    HOOK_TRACE_PROFILE("cuMemHostUnregister");
    using func_ptr = CUresult (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemHostUnregister"));
    HOOK_CHECK(func_entry);
    return func_entry(p);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpy");
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice,
                                                  CUcontext srcContext, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyPeer");
    using func_ptr = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyPeer"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyHtoD");
    using func_ptr = CUresult (*)(CUdeviceptr, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoD"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcHost, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoH");
    using func_ptr = CUresult (*)(void *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoH"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcDevice, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoD");
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoD"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcDevice, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice,
                                                  size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoA");
    using func_ptr = CUresult (*)(CUarray, size_t, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoA"));
    HOOK_CHECK(func_entry);
    return func_entry(dstArray, dstOffset, srcDevice, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset,
                                                  size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyAtoD");
    using func_ptr = CUresult (*)(CUdeviceptr, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAtoD"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcArray, srcOffset, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                                  size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyHtoA");
    using func_ptr = CUresult (*)(CUarray, size_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoA"));
    HOOK_CHECK(func_entry);
    return func_entry(dstArray, dstOffset, srcHost, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyAtoH");
    using func_ptr = CUresult (*)(void *, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAtoH"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcArray, srcOffset, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray,
                                                  size_t srcOffset, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyAtoA");
    using func_ptr = CUresult (*)(CUarray, size_t, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAtoA"));
    HOOK_CHECK(func_entry);
    return func_entry(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
    HOOK_TRACE_PROFILE("cuMemcpy2D");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy2D"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
    HOOK_TRACE_PROFILE("cuMemcpy2DUnaligned");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy2DUnaligned"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
    HOOK_TRACE_PROFILE("cuMemcpy3D");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy3D"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) {
    HOOK_TRACE_PROFILE("cuMemcpy3DPeer");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D_PEER *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy3DPeer"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                                                   CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyAsync");
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                                       CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount,
                                                       CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyPeerAsync");
    using func_ptr = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyPeerAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount,
                                                       CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyHtoDAsync");
    using func_ptr = CUresult (*)(CUdeviceptr, const void *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoDAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcHost, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                                       CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoHAsync");
    using func_ptr = CUresult (*)(void *, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoHAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcDevice, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount,
                                                       CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoDAsync");
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoDAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcDevice, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                                       size_t ByteCount, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyHtoAAsync");
    using func_ptr = CUresult (*)(CUarray, size_t, const void *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoAAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, size_t srcOffset,
                                                       size_t ByteCount, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyAtoHAsync");
    using func_ptr = CUresult (*)(void *, CUarray, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAtoHAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpy2DAsync");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy2DAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpy3DAsync");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy3DAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpy3DPeerAsync");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D_PEER *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy3DPeerAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    HOOK_TRACE_PROFILE("cuMemsetD8");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned char, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD8"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, uc, N);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    HOOK_TRACE_PROFILE("cuMemsetD16");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned short, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD16"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, us, N);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    HOOK_TRACE_PROFILE("cuMemsetD32");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned int, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD32"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, ui, N);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                                  size_t Width, size_t Height) {
    HOOK_TRACE_PROFILE("cuMemsetD2D8");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D8"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, uc, Width, Height);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us,
                                                   size_t Width, size_t Height) {
    HOOK_TRACE_PROFILE("cuMemsetD2D16");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D16"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, us, Width, Height);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                                                   size_t Width, size_t Height) {
    HOOK_TRACE_PROFILE("cuMemsetD2D32");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D32"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, ui, Width, Height);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N,
                                                     CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemsetD8Async");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD8Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, uc, N, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                                                      CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemsetD16Async");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD16Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, us, N, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                                                      CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemsetD32Async");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD32Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, ui, N, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                                       size_t Width, size_t Height, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemsetD2D8Async");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D8Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, uc, Width, Height, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us,
                                                        size_t Width, size_t Height, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemsetD2D16Async");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D16Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, us, Width, Height, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                                                        size_t Width, size_t Height, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemsetD2D32Async");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D32Async"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, ui, Width, Height, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    HOOK_TRACE_PROFILE("cuArrayCreate");
    using func_ptr = CUresult (*)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArrayCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pHandle, pAllocateArray);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    HOOK_TRACE_PROFILE("cuArrayCreate_v2");
    using func_ptr = CUresult (*)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArrayCreate_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pHandle, pAllocateArray);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    HOOK_TRACE_PROFILE("cuArrayGetDescriptor");
    using func_ptr = CUresult (*)(CUDA_ARRAY_DESCRIPTOR *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArrayGetDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(pArrayDescriptor, hArray);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    HOOK_TRACE_PROFILE("cuArrayGetDescriptor_v2");
    using func_ptr = CUresult (*)(CUDA_ARRAY_DESCRIPTOR *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArrayGetDescriptor_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pArrayDescriptor, hArray);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties,
                                                                CUarray array) {
    HOOK_TRACE_PROFILE("cuArrayGetSparseProperties");
    using func_ptr = CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArrayGetSparseProperties"));
    HOOK_CHECK(func_entry);
    return func_entry(sparseProperties, array);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMipmappedArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties,
                                                                         CUmipmappedArray mipmap) {
    HOOK_TRACE_PROFILE("cuMipmappedArrayGetSparseProperties");
    using func_ptr = CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES *, CUmipmappedArray);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMipmappedArrayGetSparseProperties"));
    HOOK_CHECK(func_entry);
    return func_entry(sparseProperties, mipmap);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx) {
    HOOK_TRACE_PROFILE("cuArrayGetPlane");
    using func_ptr = CUresult (*)(CUarray *, CUarray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArrayGetPlane"));
    HOOK_CHECK(func_entry);
    return func_entry(pPlaneArray, hArray, planeIdx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuArrayDestroy(CUarray hArray) {
    HOOK_TRACE_PROFILE("cuArrayDestroy");
    using func_ptr = CUresult (*)(CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArrayDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hArray);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    HOOK_TRACE_PROFILE("cuArray3DCreate");
    using func_ptr = CUresult (*)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArray3DCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pHandle, pAllocateArray);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuArray3DCreate_v2(CUarray *pHandle,
                                                        const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    HOOK_TRACE_PROFILE("cuArray3DCreate_v2");
    using func_ptr = CUresult (*)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArray3DCreate_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pHandle, pAllocateArray);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    HOOK_TRACE_PROFILE("cuArray3DGetDescriptor");
    using func_ptr = CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArray3DGetDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(pArrayDescriptor, hArray);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                                               CUarray hArray) {
    HOOK_TRACE_PROFILE("cuArray3DGetDescriptor_v2");
    using func_ptr = CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuArray3DGetDescriptor_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pArrayDescriptor, hArray);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                                                            const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                                                            unsigned int numMipmapLevels) {
    HOOK_TRACE_PROFILE("cuMipmappedArrayCreate");
    using func_ptr = CUresult (*)(CUmipmappedArray *, const CUDA_ARRAY3D_DESCRIPTOR *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMipmappedArrayCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pHandle, pMipmappedArrayDesc, numMipmapLevels);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray,
                                                              unsigned int level) {
    HOOK_TRACE_PROFILE("cuMipmappedArrayGetLevel");
    using func_ptr = CUresult (*)(CUarray *, CUmipmappedArray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMipmappedArrayGetLevel"));
    HOOK_CHECK(func_entry);
    return func_entry(pLevelArray, hMipmappedArray, level);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
    HOOK_TRACE_PROFILE("cuMipmappedArrayDestroy");
    using func_ptr = CUresult (*)(CUmipmappedArray);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMipmappedArrayDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hMipmappedArray);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                                                         CUdeviceptr addr, unsigned long long flags) {
    HOOK_TRACE_PROFILE("cuMemAddressReserve");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAddressReserve"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, size, alignment, addr, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    HOOK_TRACE_PROFILE("cuMemAddressFree");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAddressFree"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, size);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                                                 const CUmemAllocationProp *prop, unsigned long long flags) {
    HOOK_TRACE_PROFILE("cuMemCreate");
    using func_ptr =
        CUresult (*)(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, size, prop, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    HOOK_TRACE_PROFILE("cuMemRelease");
    using func_ptr = CUresult (*)(CUmemGenericAllocationHandle);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemRelease"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                                              CUmemGenericAllocationHandle handle, unsigned long long flags) {
    HOOK_TRACE_PROFILE("cuMemMap");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemMap"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, size, offset, handle, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count,
                                                        CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemMapArrayAsync");
    using func_ptr = CUresult (*)(CUarrayMapInfo *, unsigned int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemMapArrayAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(mapInfoList, count, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    HOOK_TRACE_PROFILE("cuMemUnmap");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemUnmap"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, size);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc,
                                                    size_t count) {
    HOOK_TRACE_PROFILE("cuMemSetAccess");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemSetAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, size, desc, count);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location,
                                                    CUdeviceptr ptr) {
    HOOK_TRACE_PROFILE("cuMemGetAccess");
    using func_ptr = CUresult (*)(unsigned long long *, const CUmemLocation *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemGetAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(flags, location, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemExportToShareableHandle(void *shareableHandle,
                                                                  CUmemGenericAllocationHandle handle,
                                                                  CUmemAllocationHandleType handleType,
                                                                  unsigned long long flags) {
    HOOK_TRACE_PROFILE("cuMemExportToShareableHandle");
    using func_ptr = CUresult (*)(void *, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemExportToShareableHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(shareableHandle, handle, handleType, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
                                                                    void *osHandle,
                                                                    CUmemAllocationHandleType shHandleType) {
    HOOK_TRACE_PROFILE("cuMemImportFromShareableHandle");
    using func_ptr = CUresult (*)(CUmemGenericAllocationHandle *, void *, CUmemAllocationHandleType);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemImportFromShareableHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, osHandle, shHandleType);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop,
                                                                   CUmemAllocationGranularity_flags option) {
    HOOK_TRACE_PROFILE("cuMemGetAllocationGranularity");
    using func_ptr = CUresult (*)(size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemGetAllocationGranularity"));
    HOOK_CHECK(func_entry);
    return func_entry(granularity, prop, option);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop,
                                                                            CUmemGenericAllocationHandle handle) {
    HOOK_TRACE_PROFILE("cuMemGetAllocationPropertiesFromHandle");
    using func_ptr = CUresult (*)(CUmemAllocationProp *, CUmemGenericAllocationHandle);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemGetAllocationPropertiesFromHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(prop, handle);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr) {
    HOOK_TRACE_PROFILE("cuMemRetainAllocationHandle");
    using func_ptr = CUresult (*)(CUmemGenericAllocationHandle *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemRetainAllocationHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, addr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemFreeAsync");
    using func_ptr = CUresult (*)(CUdeviceptr, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemFreeAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemAllocAsync");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAllocAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, bytesize, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
    HOOK_TRACE_PROFILE("cuMemPoolTrimTo");
    using func_ptr = CUresult (*)(CUmemoryPool, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolTrimTo"));
    HOOK_CHECK(func_entry);
    return func_entry(pool, minBytesToKeep);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    HOOK_TRACE_PROFILE("cuMemPoolSetAttribute");
    using func_ptr = CUresult (*)(CUmemoryPool, CUmemPool_attribute, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(pool, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
    HOOK_TRACE_PROFILE("cuMemPoolGetAttribute");
    using func_ptr = CUresult (*)(CUmemoryPool, CUmemPool_attribute, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(pool, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count) {
    HOOK_TRACE_PROFILE("cuMemPoolSetAccess");
    using func_ptr = CUresult (*)(CUmemoryPool, const CUmemAccessDesc *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolSetAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(pool, map, count);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool,
                                                        CUmemLocation *location) {
    HOOK_TRACE_PROFILE("cuMemPoolGetAccess");
    using func_ptr = CUresult (*)(CUmemAccess_flags *, CUmemoryPool, CUmemLocation *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolGetAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(flags, memPool, location);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
    HOOK_TRACE_PROFILE("cuMemPoolCreate");
    using func_ptr = CUresult (*)(CUmemoryPool *, const CUmemPoolProps *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pool, poolProps);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolDestroy(CUmemoryPool pool) {
    HOOK_TRACE_PROFILE("cuMemPoolDestroy");
    using func_ptr = CUresult (*)(CUmemoryPool);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(pool);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool,
                                                             CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemAllocFromPoolAsync");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t, CUmemoryPool, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAllocFromPoolAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dptr, bytesize, pool, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool,
                                                                      CUmemAllocationHandleType handleType,
                                                                      unsigned long long flags) {
    HOOK_TRACE_PROFILE("cuMemPoolExportToShareableHandle");
    using func_ptr = CUresult (*)(void *, CUmemoryPool, CUmemAllocationHandleType, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolExportToShareableHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(handle_out, pool, handleType, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolImportFromShareableHandle(CUmemoryPool *pool_out, void *handle,
                                                                        CUmemAllocationHandleType handleType,
                                                                        unsigned long long flags) {
    HOOK_TRACE_PROFILE("cuMemPoolImportFromShareableHandle");
    using func_ptr = CUresult (*)(CUmemoryPool *, void *, CUmemAllocationHandleType, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolImportFromShareableHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(pool_out, handle, handleType, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr) {
    HOOK_TRACE_PROFILE("cuMemPoolExportPointer");
    using func_ptr = CUresult (*)(CUmemPoolPtrExportData *, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolExportPointer"));
    HOOK_CHECK(func_entry);
    return func_entry(shareData_out, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool,
                                                            CUmemPoolPtrExportData *shareData) {
    HOOK_TRACE_PROFILE("cuMemPoolImportPointer");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUmemoryPool, CUmemPoolPtrExportData *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPoolImportPointer"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr_out, pool, shareData);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    HOOK_TRACE_PROFILE("cuPointerGetAttribute");
    using func_ptr = CUresult (*)(void *, CUpointer_attribute, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuPointerGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(data, attribute, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice,
                                                        CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemPrefetchAsync");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, CUdevice, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemPrefetchAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, count, dstDevice, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice,
                                                 CUdevice device) {
    HOOK_TRACE_PROFILE("cuMemAdvise");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, CUmem_advise, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAdvise"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, count, advice, device);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemRangeGetAttribute(void *data, size_t dataSize,
                                                            CUmem_range_attribute attribute, CUdeviceptr devPtr,
                                                            size_t count) {
    HOOK_TRACE_PROFILE("cuMemRangeGetAttribute");
    using func_ptr = CUresult (*)(void *, size_t, CUmem_range_attribute, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemRangeGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(data, dataSize, attribute, devPtr, count);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                                             CUmem_range_attribute *attributes, size_t numAttributes,
                                                             CUdeviceptr devPtr, size_t count) {
    HOOK_TRACE_PROFILE("cuMemRangeGetAttributes");
    using func_ptr = CUresult (*)(void **, size_t *, CUmem_range_attribute *, size_t, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemRangeGetAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(data, dataSizes, attributes, numAttributes, devPtr, count);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute,
                                                           CUdeviceptr ptr) {
    HOOK_TRACE_PROFILE("cuPointerSetAttribute");
    using func_ptr = CUresult (*)(const void *, CUpointer_attribute, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuPointerSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(value, attribute, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes,
                                                            void **data, CUdeviceptr ptr) {
    HOOK_TRACE_PROFILE("cuPointerGetAttributes");
    using func_ptr = CUresult (*)(unsigned int, CUpointer_attribute *, void **, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuPointerGetAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(numAttributes, attributes, data, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuStreamCreate");
    using func_ptr = CUresult (*)(CUstream *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(phStream, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    HOOK_TRACE_PROFILE("cuStreamCreateWithPriority");
    using func_ptr = CUresult (*)(CUstream *, unsigned int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamCreateWithPriority"));
    HOOK_CHECK(func_entry);
    return func_entry(phStream, flags, priority);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    HOOK_TRACE_PROFILE("cuStreamGetPriority");
    using func_ptr = CUresult (*)(CUstream, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamGetPriority"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, priority);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    HOOK_TRACE_PROFILE("cuStreamGetFlags");
    using func_ptr = CUresult (*)(CUstream, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamGetFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    HOOK_TRACE_PROFILE("cuStreamGetCtx");
    using func_ptr = CUresult (*)(CUstream, CUcontext *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamGetCtx"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, pctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuStreamWaitEvent");
    using func_ptr = CUresult (*)(CUstream, CUevent, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamWaitEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, hEvent, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData,
                                                         unsigned int flags) {
    HOOK_TRACE_PROFILE("cuStreamAddCallback");
    using func_ptr = CUresult (*)(CUstream, CUstreamCallback, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamAddCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, callback, userData, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode) {
    HOOK_TRACE_PROFILE("cuStreamBeginCapture");
    using func_ptr = CUresult (*)(CUstream, CUstreamCaptureMode);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamBeginCapture"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, mode);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
    HOOK_TRACE_PROFILE("cuThreadExchangeStreamCaptureMode");
    using func_ptr = CUresult (*)(CUstreamCaptureMode *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuThreadExchangeStreamCaptureMode"));
    HOOK_CHECK(func_entry);
    return func_entry(mode);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
    HOOK_TRACE_PROFILE("cuStreamEndCapture");
    using func_ptr = CUresult (*)(CUstream, CUgraph *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamEndCapture"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, phGraph);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus) {
    HOOK_TRACE_PROFILE("cuStreamIsCapturing");
    using func_ptr = CUresult (*)(CUstream, CUstreamCaptureStatus *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamIsCapturing"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, captureStatus);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out,
                                                            cuuint64_t *id_out) {
    HOOK_TRACE_PROFILE("cuStreamGetCaptureInfo");
    using func_ptr = CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamGetCaptureInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, captureStatus_out, id_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamGetCaptureInfo_v2(CUstream hStream,
                                                               CUstreamCaptureStatus *captureStatus_out,
                                                               cuuint64_t *id_out, CUgraph *graph_out,
                                                               const CUgraphNode **dependencies_out,
                                                               size_t *numDependencies_out) {
    HOOK_TRACE_PROFILE("cuStreamGetCaptureInfo_v2");
    using func_ptr =
        CUresult (*)(CUstream, CUstreamCaptureStatus *, cuuint64_t *, CUgraph *, const CUgraphNode **, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamGetCaptureInfo_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies,
                                                                       size_t numDependencies, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuStreamUpdateCaptureDependencies");
    using func_ptr = CUresult (*)(CUstream, CUgraphNode *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamUpdateCaptureDependencies"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, dependencies, numDependencies, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length,
                                                            unsigned int flags) {
    HOOK_TRACE_PROFILE("cuStreamAttachMemAsync");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamAttachMemAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, dptr, length, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamQuery(CUstream hStream) {
    HOOK_TRACE_PROFILE("cuStreamQuery");
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamQuery"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamSynchronize(CUstream hStream) {
    HOOK_TRACE_PROFILE("cuStreamSynchronize");
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamDestroy(CUstream hStream) {
    HOOK_TRACE_PROFILE("cuStreamDestroy");
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamDestroy_v2(CUstream hStream) {
    HOOK_TRACE_PROFILE("cuStreamDestroy_v2");
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamDestroy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
    HOOK_TRACE_PROFILE("cuStreamCopyAttributes");
    using func_ptr = CUresult (*)(CUstream, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamCopyAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr,
                                                          CUstreamAttrValue *value_out) {
    HOOK_TRACE_PROFILE("cuStreamGetAttribute");
    using func_ptr = CUresult (*)(CUstream, CUstreamAttrID, CUstreamAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, attr, value_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr,
                                                          const CUstreamAttrValue *value) {
    HOOK_TRACE_PROFILE("cuStreamSetAttribute");
    using func_ptr = CUresult (*)(CUstream, CUstreamAttrID, const CUstreamAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuEventCreate");
    using func_ptr = CUresult (*)(CUevent *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(phEvent, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuEventRecord");
    using func_ptr = CUresult (*)(CUevent, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventRecord"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuEventRecordWithFlags");
    using func_ptr = CUresult (*)(CUevent, CUstream, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventRecordWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent, hStream, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventQuery(CUevent hEvent) {
    HOOK_TRACE_PROFILE("cuEventQuery");
    using func_ptr = CUresult (*)(CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventQuery"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventSynchronize(CUevent hEvent) {
    HOOK_TRACE_PROFILE("cuEventSynchronize");
    using func_ptr = CUresult (*)(CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventDestroy(CUevent hEvent) {
    HOOK_TRACE_PROFILE("cuEventDestroy");
    using func_ptr = CUresult (*)(CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventDestroy_v2(CUevent hEvent) {
    HOOK_TRACE_PROFILE("cuEventDestroy_v2");
    using func_ptr = CUresult (*)(CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventDestroy_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(hEvent);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    HOOK_TRACE_PROFILE("cuEventElapsedTime");
    using func_ptr = CUresult (*)(float *, CUevent, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuEventElapsedTime"));
    HOOK_CHECK(func_entry);
    return func_entry(pMilliseconds, hStart, hEnd);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuImportExternalMemory(CUexternalMemory *extMem_out,
                                                            const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
    HOOK_TRACE_PROFILE("cuImportExternalMemory");
    using func_ptr = CUresult (*)(CUexternalMemory *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuImportExternalMemory"));
    HOOK_CHECK(func_entry);
    return func_entry(extMem_out, memHandleDesc);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuExternalMemoryGetMappedBuffer(
    CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
    HOOK_TRACE_PROFILE("cuExternalMemoryGetMappedBuffer");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuExternalMemoryGetMappedBuffer"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, extMem, bufferDesc);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray *mipmap, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
    HOOK_TRACE_PROFILE("cuExternalMemoryGetMappedMipmappedArray");
    using func_ptr =
        CUresult (*)(CUmipmappedArray *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuExternalMemoryGetMappedMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(mipmap, extMem, mipmapDesc);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
    HOOK_TRACE_PROFILE("cuDestroyExternalMemory");
    using func_ptr = CUresult (*)(CUexternalMemory);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDestroyExternalMemory"));
    HOOK_CHECK(func_entry);
    return func_entry(extMem);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuImportExternalSemaphore(
    CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
    HOOK_TRACE_PROFILE("cuImportExternalSemaphore");
    using func_ptr = CUresult (*)(CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuImportExternalSemaphore"));
    HOOK_CHECK(func_entry);
    return func_entry(extSem_out, semHandleDesc);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuSignalExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
    HOOK_TRACE_PROFILE("cuSignalExternalSemaphoresAsync");
    using func_ptr = CUresult (*)(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *,
                                  unsigned int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuSignalExternalSemaphoresAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
    HOOK_TRACE_PROFILE("cuWaitExternalSemaphoresAsync");
    using func_ptr =
        CUresult (*)(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *, unsigned int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuWaitExternalSemaphoresAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
    HOOK_TRACE_PROFILE("cuDestroyExternalSemaphore");
    using func_ptr = CUresult (*)(CUexternalSemaphore);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDestroyExternalSemaphore"));
    HOOK_CHECK(func_entry);
    return func_entry(extSem);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value,
                                                         unsigned int flags) {
    HOOK_TRACE_PROFILE("cuStreamWaitValue32");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamWaitValue32"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, addr, value, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value,
                                                         unsigned int flags) {
    HOOK_TRACE_PROFILE("cuStreamWaitValue64");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamWaitValue64"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, addr, value, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value,
                                                          unsigned int flags) {
    HOOK_TRACE_PROFILE("cuStreamWriteValue32");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamWriteValue32"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, addr, value, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value,
                                                          unsigned int flags) {
    HOOK_TRACE_PROFILE("cuStreamWriteValue64");
    using func_ptr = CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamWriteValue64"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, addr, value, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                                        CUstreamBatchMemOpParams *paramArray, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuStreamBatchMemOp");
    using func_ptr = CUresult (*)(CUstream, unsigned int, CUstreamBatchMemOpParams *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamBatchMemOp"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, count, paramArray, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) {
    HOOK_TRACE_PROFILE("cuFuncGetAttribute");
    using func_ptr = CUresult (*)(int *, CUfunction_attribute, CUfunction);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(pi, attrib, hfunc);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) {
    HOOK_TRACE_PROFILE("cuFuncSetAttribute");
    using func_ptr = CUresult (*)(CUfunction, CUfunction_attribute, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, attrib, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
    HOOK_TRACE_PROFILE("cuFuncSetCacheConfig");
    using func_ptr = CUresult (*)(CUfunction, CUfunc_cache);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncSetCacheConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, config);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
    HOOK_TRACE_PROFILE("cuFuncSetSharedMemConfig");
    using func_ptr = CUresult (*)(CUfunction, CUsharedconfig);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncSetSharedMemConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, config);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
    HOOK_TRACE_PROFILE("cuFuncGetModule");
    using func_ptr = CUresult (*)(CUmodule *, CUfunction);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncGetModule"));
    HOOK_CHECK(func_entry);
    return func_entry(hmod, hfunc);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                                                    unsigned int gridDimZ, unsigned int blockDimX,
                                                    unsigned int blockDimY, unsigned int blockDimZ,
                                                    unsigned int sharedMemBytes, CUstream hStream, void **kernelParams,
                                                    void **extra) {
    HOOK_TRACE_PROFILE("cuLaunchKernel");
    using func_ptr = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, CUstream, void **, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunchKernel"));
    HOOK_CHECK(func_entry);
    return func_entry(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
                      kernelParams, extra);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX,
                                                               unsigned int gridDimY, unsigned int gridDimZ,
                                                               unsigned int blockDimX, unsigned int blockDimY,
                                                               unsigned int blockDimZ, unsigned int sharedMemBytes,
                                                               CUstream hStream, void **kernelParams) {
    HOOK_TRACE_PROFILE("cuLaunchCooperativeKernel");
    using func_ptr = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, CUstream, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunchCooperativeKernel"));
    HOOK_CHECK(func_entry);
    return func_entry(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream,
                      kernelParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList,
                                                                          unsigned int numDevices, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuLaunchCooperativeKernelMultiDevice");
    using func_ptr = CUresult (*)(CUDA_LAUNCH_PARAMS *, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunchCooperativeKernelMultiDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(launchParamsList, numDevices, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) {
    HOOK_TRACE_PROFILE("cuLaunchHostFunc");
    using func_ptr = CUresult (*)(CUstream, CUhostFn, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunchHostFunc"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, fn, userData);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
    HOOK_TRACE_PROFILE("cuFuncSetBlockShape");
    using func_ptr = CUresult (*)(CUfunction, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncSetBlockShape"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, x, y, z);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
    HOOK_TRACE_PROFILE("cuFuncSetSharedSize");
    using func_ptr = CUresult (*)(CUfunction, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncSetSharedSize"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, bytes);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
    HOOK_TRACE_PROFILE("cuParamSetSize");
    using func_ptr = CUresult (*)(CUfunction, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuParamSetSize"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, numbytes);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
    HOOK_TRACE_PROFILE("cuParamSeti");
    using func_ptr = CUresult (*)(CUfunction, int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuParamSeti"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, offset, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
    HOOK_TRACE_PROFILE("cuParamSetf");
    using func_ptr = CUresult (*)(CUfunction, int, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuParamSetf"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, offset, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
    HOOK_TRACE_PROFILE("cuParamSetv");
    using func_ptr = CUresult (*)(CUfunction, int, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuParamSetv"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, offset, ptr, numbytes);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLaunch(CUfunction f) {
    HOOK_TRACE_PROFILE("cuLaunch");
    using func_ptr = CUresult (*)(CUfunction);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunch"));
    HOOK_CHECK(func_entry);
    return func_entry(f);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
    HOOK_TRACE_PROFILE("cuLaunchGrid");
    using func_ptr = CUresult (*)(CUfunction, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunchGrid"));
    HOOK_CHECK(func_entry);
    return func_entry(f, grid_width, grid_height);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                                                       CUstream hStream) {
    HOOK_TRACE_PROFILE("cuLaunchGridAsync");
    using func_ptr = CUresult (*)(CUfunction, int, int, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLaunchGridAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(f, grid_width, grid_height, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuParamSetTexRef");
    using func_ptr = CUresult (*)(CUfunction, int, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuParamSetTexRef"));
    HOOK_CHECK(func_entry);
    return func_entry(hfunc, texunit, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuGraphCreate");
    using func_ptr = CUresult (*)(CUgraph *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraph, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                          const CUgraphNode *dependencies, size_t numDependencies,
                                                          const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphAddKernelNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_KERNEL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddKernelNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphKernelNodeGetParams(CUgraphNode hNode,
                                                                CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphKernelNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_KERNEL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphKernelNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphKernelNodeSetParams(CUgraphNode hNode,
                                                                const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphKernelNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphKernelNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                          const CUgraphNode *dependencies, size_t numDependencies,
                                                          const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuGraphAddMemcpyNode");
    using func_ptr =
        CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_MEMCPY3D *, CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddMemcpyNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphMemcpyNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_MEMCPY3D *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphMemcpyNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphMemcpyNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_MEMCPY3D *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphMemcpyNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                          const CUgraphNode *dependencies, size_t numDependencies,
                                                          const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuGraphAddMemsetNode");
    using func_ptr =
        CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddMemsetNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode,
                                                                CUDA_MEMSET_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphMemsetNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_MEMSET_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphMemsetNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode,
                                                                const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphMemsetNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_MEMSET_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphMemsetNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                        const CUgraphNode *dependencies, size_t numDependencies,
                                                        const CUDA_HOST_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphAddHostNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_HOST_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddHostNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphHostNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_HOST_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphHostNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphHostNodeSetParams(CUgraphNode hNode,
                                                              const CUDA_HOST_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphHostNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphHostNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                              const CUgraphNode *dependencies, size_t numDependencies,
                                                              CUgraph childGraph) {
    HOOK_TRACE_PROFILE("cuGraphAddChildGraphNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddChildGraphNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, childGraph);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
    HOOK_TRACE_PROFILE("cuGraphChildGraphNodeGetGraph");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraph *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphChildGraphNodeGetGraph"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, phGraph);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                         const CUgraphNode *dependencies, size_t numDependencies) {
    HOOK_TRACE_PROFILE("cuGraphAddEmptyNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddEmptyNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                               const CUgraphNode *dependencies, size_t numDependencies,
                                                               CUevent event) {
    HOOK_TRACE_PROFILE("cuGraphAddEventRecordNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddEventRecordNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, event);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
    HOOK_TRACE_PROFILE("cuGraphEventRecordNodeGetEvent");
    using func_ptr = CUresult (*)(CUgraphNode, CUevent *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphEventRecordNodeGetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, event_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
    HOOK_TRACE_PROFILE("cuGraphEventRecordNodeSetEvent");
    using func_ptr = CUresult (*)(CUgraphNode, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphEventRecordNodeSetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, event);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                             const CUgraphNode *dependencies, size_t numDependencies,
                                                             CUevent event) {
    HOOK_TRACE_PROFILE("cuGraphAddEventWaitNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddEventWaitNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, event);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
    HOOK_TRACE_PROFILE("cuGraphEventWaitNodeGetEvent");
    using func_ptr = CUresult (*)(CUgraphNode, CUevent *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphEventWaitNodeGetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, event_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
    HOOK_TRACE_PROFILE("cuGraphEventWaitNodeSetEvent");
    using func_ptr = CUresult (*)(CUgraphNode, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphEventWaitNodeSetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, event);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult
    cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
                                           size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphAddExternalSemaphoresSignalNode");
    using func_ptr =
        CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddExternalSemaphoresSignalNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult
    cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
    HOOK_TRACE_PROFILE("cuGraphExternalSemaphoresSignalNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExternalSemaphoresSignalNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, params_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult
    cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphExternalSemaphoresSignalNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExternalSemaphoresSignalNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult
    cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
                                         size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphAddExternalSemaphoresWaitNode");
    using func_ptr =
        CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddExternalSemaphoresWaitNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult
    cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
    HOOK_TRACE_PROFILE("cuGraphExternalSemaphoresWaitNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExternalSemaphoresWaitNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, params_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult
    cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphExternalSemaphoresWaitNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExternalSemaphoresWaitNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                            const CUgraphNode *dependencies, size_t numDependencies,
                                                            CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphAddMemAllocNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUDA_MEM_ALLOC_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddMemAllocNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode,
                                                                  CUDA_MEM_ALLOC_NODE_PARAMS *params_out) {
    HOOK_TRACE_PROFILE("cuGraphMemAllocNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUDA_MEM_ALLOC_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphMemAllocNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, params_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                                           const CUgraphNode *dependencies, size_t numDependencies,
                                                           CUdeviceptr dptr) {
    HOOK_TRACE_PROFILE("cuGraphAddMemFreeNode");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, CUdeviceptr);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddMemFreeNode"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphNode, hGraph, dependencies, numDependencies, dptr);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out) {
    HOOK_TRACE_PROFILE("cuGraphMemFreeNodeGetParams");
    using func_ptr = CUresult (*)(CUgraphNode, CUdeviceptr *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphMemFreeNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, dptr_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGraphMemTrim(CUdevice device) {
    HOOK_TRACE_PROFILE("cuDeviceGraphMemTrim");
    using func_ptr = CUresult (*)(CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGraphMemTrim"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr,
                                                                  void *value) {
    HOOK_TRACE_PROFILE("cuDeviceGetGraphMemAttribute");
    using func_ptr = CUresult (*)(CUdevice, CUgraphMem_attribute, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetGraphMemAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(device, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr,
                                                                  void *value) {
    HOOK_TRACE_PROFILE("cuDeviceSetGraphMemAttribute");
    using func_ptr = CUresult (*)(CUdevice, CUgraphMem_attribute, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceSetGraphMemAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(device, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
    HOOK_TRACE_PROFILE("cuGraphClone");
    using func_ptr = CUresult (*)(CUgraph *, CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphClone"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphClone, originalGraph);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode,
                                                            CUgraph hClonedGraph) {
    HOOK_TRACE_PROFILE("cuGraphNodeFindInClone");
    using func_ptr = CUresult (*)(CUgraphNode *, CUgraphNode, CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphNodeFindInClone"));
    HOOK_CHECK(func_entry);
    return func_entry(phNode, hOriginalNode, hClonedGraph);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
    HOOK_TRACE_PROFILE("cuGraphNodeGetType");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNodeType *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphNodeGetType"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, type);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
    HOOK_TRACE_PROFILE("cuGraphGetNodes");
    using func_ptr = CUresult (*)(CUgraph, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphGetNodes"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraph, nodes, numNodes);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes) {
    HOOK_TRACE_PROFILE("cuGraphGetRootNodes");
    using func_ptr = CUresult (*)(CUgraph, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphGetRootNodes"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraph, rootNodes, numRootNodes);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to,
                                                     size_t *numEdges) {
    HOOK_TRACE_PROFILE("cuGraphGetEdges");
    using func_ptr = CUresult (*)(CUgraph, CUgraphNode *, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphGetEdges"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraph, from, to, numEdges);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies,
                                                                size_t *numDependencies) {
    HOOK_TRACE_PROFILE("cuGraphNodeGetDependencies");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphNodeGetDependencies"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, dependencies, numDependencies);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes,
                                                                  size_t *numDependentNodes) {
    HOOK_TRACE_PROFILE("cuGraphNodeGetDependentNodes");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNode *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphNodeGetDependentNodes"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, dependentNodes, numDependentNodes);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from,
                                                            const CUgraphNode *to, size_t numDependencies) {
    HOOK_TRACE_PROFILE("cuGraphAddDependencies");
    using func_ptr = CUresult (*)(CUgraph, const CUgraphNode *, const CUgraphNode *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphAddDependencies"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraph, from, to, numDependencies);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from,
                                                               const CUgraphNode *to, size_t numDependencies) {
    HOOK_TRACE_PROFILE("cuGraphRemoveDependencies");
    using func_ptr = CUresult (*)(CUgraph, const CUgraphNode *, const CUgraphNode *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphRemoveDependencies"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraph, from, to, numDependencies);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphDestroyNode(CUgraphNode hNode) {
    HOOK_TRACE_PROFILE("cuGraphDestroyNode");
    using func_ptr = CUresult (*)(CUgraphNode);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphDestroyNode"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph,
                                                        CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize) {
    HOOK_TRACE_PROFILE("cuGraphInstantiate");
    using func_ptr = CUresult (*)(CUgraphExec *, CUgraph, CUgraphNode *, char *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphInstantiate"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph,
                                                           CUgraphNode *phErrorNode, char *logBuffer,
                                                           size_t bufferSize) {
    HOOK_TRACE_PROFILE("cuGraphInstantiate_v2");
    using func_ptr = CUresult (*)(CUgraphExec *, CUgraph, CUgraphNode *, char *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphInstantiate_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph,
                                                                 unsigned long long flags) {
    HOOK_TRACE_PROFILE("cuGraphInstantiateWithFlags");
    using func_ptr = CUresult (*)(CUgraphExec *, CUgraph, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphInstantiateWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(phGraphExec, hGraph, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                                    const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphExecKernelNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecKernelNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                                    const CUDA_MEMCPY3D *copyParams, CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuGraphExecMemcpyNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMCPY3D *, CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecMemcpyNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, copyParams, ctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                                    const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                                                                    CUcontext ctx) {
    HOOK_TRACE_PROFILE("cuGraphExecMemsetNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecMemsetNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, memsetParams, ctx);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                                  const CUDA_HOST_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphExecHostNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_HOST_NODE_PARAMS *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecHostNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                                        CUgraph childGraph) {
    HOOK_TRACE_PROFILE("cuGraphExecChildGraphNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecChildGraphNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, childGraph);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                                        CUevent event) {
    HOOK_TRACE_PROFILE("cuGraphExecEventRecordNodeSetEvent");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecEventRecordNodeSetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, event);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode,
                                                                      CUevent event) {
    HOOK_TRACE_PROFILE("cuGraphExecEventWaitNodeSetEvent");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, CUevent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecEventWaitNodeSetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, event);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphExecExternalSemaphoresSignalNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecExternalSemaphoresSignalNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
    HOOK_TRACE_PROFILE("cuGraphExecExternalSemaphoresWaitNodeSetParams");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecExternalSemaphoresWaitNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuGraphUpload");
    using func_ptr = CUresult (*)(CUgraphExec, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphUpload"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuGraphLaunch");
    using func_ptr = CUresult (*)(CUgraphExec, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphLaunch"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
    HOOK_TRACE_PROFILE("cuGraphExecDestroy");
    using func_ptr = CUresult (*)(CUgraphExec);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphDestroy(CUgraph hGraph) {
    HOOK_TRACE_PROFILE("cuGraphDestroy");
    using func_ptr = CUresult (*)(CUgraph);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraph);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                                                       CUgraphNode *hErrorNode_out,
                                                       CUgraphExecUpdateResult *updateResult_out) {
    HOOK_TRACE_PROFILE("cuGraphExecUpdate");
    using func_ptr = CUresult (*)(CUgraphExec, CUgraph, CUgraphNode *, CUgraphExecUpdateResult *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphExecUpdate"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
    HOOK_TRACE_PROFILE("cuGraphKernelNodeCopyAttributes");
    using func_ptr = CUresult (*)(CUgraphNode, CUgraphNode);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphKernelNodeCopyAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                                                                   CUkernelNodeAttrValue *value_out) {
    HOOK_TRACE_PROFILE("cuGraphKernelNodeGetAttribute");
    using func_ptr = CUresult (*)(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphKernelNodeGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, attr, value_out);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr,
                                                                   const CUkernelNodeAttrValue *value) {
    HOOK_TRACE_PROFILE("cuGraphKernelNodeSetAttribute");
    using func_ptr = CUresult (*)(CUgraphNode, CUkernelNodeAttrID, const CUkernelNodeAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphKernelNodeSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuGraphDebugDotPrint");
    using func_ptr = CUresult (*)(CUgraph, const char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphDebugDotPrint"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraph, path, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy,
                                                        unsigned int initialRefcount, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuUserObjectCreate");
    using func_ptr = CUresult (*)(CUuserObject *, void *, CUhostFn, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuUserObjectCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(object_out, ptr, destroy, initialRefcount, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) {
    HOOK_TRACE_PROFILE("cuUserObjectRetain");
    using func_ptr = CUresult (*)(CUuserObject, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuUserObjectRetain"));
    HOOK_CHECK(func_entry);
    return func_entry(object, count);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
    HOOK_TRACE_PROFILE("cuUserObjectRelease");
    using func_ptr = CUresult (*)(CUuserObject, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuUserObjectRelease"));
    HOOK_CHECK(func_entry);
    return func_entry(object, count);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count,
                                                             unsigned int flags) {
    HOOK_TRACE_PROFILE("cuGraphRetainUserObject");
    using func_ptr = CUresult (*)(CUgraph, CUuserObject, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphRetainUserObject"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, object, count, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count) {
    HOOK_TRACE_PROFILE("cuGraphReleaseUserObject");
    using func_ptr = CUresult (*)(CUgraph, CUuserObject, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphReleaseUserObject"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, object, count);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func,
                                                                                 int blockSize,
                                                                                 size_t dynamicSMemSize) {
    HOOK_TRACE_PROFILE("cuOccupancyMaxActiveBlocksPerMultiprocessor");
    using func_ptr = CUresult (*)(int *, CUfunction, int, size_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuOccupancyMaxActiveBlocksPerMultiprocessor"));
    HOOK_CHECK(func_entry);
    return func_entry(numBlocks, func, blockSize, dynamicSMemSize);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    using func_ptr = CUresult (*)(int *, CUfunction, int, size_t, unsigned int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(numBlocks, func, blockSize, dynamicSMemSize, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func,
                                                                      CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                                                      size_t dynamicSMemSize, int blockSizeLimit) {
    HOOK_TRACE_PROFILE("cuOccupancyMaxPotentialBlockSize");
    using func_ptr = CUresult (*)(int *, int *, CUfunction, CUoccupancyB2DSize, size_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuOccupancyMaxPotentialBlockSize"));
    HOOK_CHECK(func_entry);
    return func_entry(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize,
    size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuOccupancyMaxPotentialBlockSizeWithFlags");
    using func_ptr = CUresult (*)(int *, int *, CUfunction, CUoccupancyB2DSize, size_t, int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuOccupancyMaxPotentialBlockSizeWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize, CUfunction func,
                                                                             int numBlocks, int blockSize) {
    HOOK_TRACE_PROFILE("cuOccupancyAvailableDynamicSMemPerBlock");
    using func_ptr = CUresult (*)(size_t *, CUfunction, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuOccupancyAvailableDynamicSMemPerBlock"));
    HOOK_CHECK(func_entry);
    return func_entry(dynamicSmemSize, func, numBlocks, blockSize);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuTexRefSetArray");
    using func_ptr = CUresult (*)(CUtexref, CUarray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetArray"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, hArray, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray,
                                                               unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuTexRefSetMipmappedArray");
    using func_ptr = CUresult (*)(CUtexref, CUmipmappedArray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, hMipmappedArray, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr,
                                                        size_t bytes) {
    HOOK_TRACE_PROFILE("cuTexRefSetAddress");
    using func_ptr = CUresult (*)(size_t *, CUtexref, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetAddress"));
    HOOK_CHECK(func_entry);
    return func_entry(ByteOffset, hTexRef, dptr, bytes);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr,
                                                           size_t bytes) {
    HOOK_TRACE_PROFILE("cuTexRefSetAddress_v2");
    using func_ptr = CUresult (*)(size_t *, CUtexref, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetAddress_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(ByteOffset, hTexRef, dptr, bytes);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                                          CUdeviceptr dptr, size_t Pitch) {
    HOOK_TRACE_PROFILE("cuTexRefSetAddress2D");
    using func_ptr = CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetAddress2D"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, desc, dptr, Pitch);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                                             CUdeviceptr dptr, size_t Pitch) {
    HOOK_TRACE_PROFILE("cuTexRefSetAddress2D_v3");
    using func_ptr = CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetAddress2D_v3"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, desc, dptr, Pitch);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
    HOOK_TRACE_PROFILE("cuTexRefSetFormat");
    using func_ptr = CUresult (*)(CUtexref, CUarray_format, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetFormat"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, fmt, NumPackedComponents);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
    HOOK_TRACE_PROFILE("cuTexRefSetAddressMode");
    using func_ptr = CUresult (*)(CUtexref, int, CUaddress_mode);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetAddressMode"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, dim, am);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    HOOK_TRACE_PROFILE("cuTexRefSetFilterMode");
    using func_ptr = CUresult (*)(CUtexref, CUfilter_mode);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetFilterMode"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, fm);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    HOOK_TRACE_PROFILE("cuTexRefSetMipmapFilterMode");
    using func_ptr = CUresult (*)(CUtexref, CUfilter_mode);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetMipmapFilterMode"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, fm);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
    HOOK_TRACE_PROFILE("cuTexRefSetMipmapLevelBias");
    using func_ptr = CUresult (*)(CUtexref, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetMipmapLevelBias"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, bias);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp,
                                                                 float maxMipmapLevelClamp) {
    HOOK_TRACE_PROFILE("cuTexRefSetMipmapLevelClamp");
    using func_ptr = CUresult (*)(CUtexref, float, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetMipmapLevelClamp"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
    HOOK_TRACE_PROFILE("cuTexRefSetMaxAnisotropy");
    using func_ptr = CUresult (*)(CUtexref, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetMaxAnisotropy"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, maxAniso);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
    HOOK_TRACE_PROFILE("cuTexRefSetBorderColor");
    using func_ptr = CUresult (*)(CUtexref, float *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetBorderColor"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, pBorderColor);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuTexRefSetFlags");
    using func_ptr = CUresult (*)(CUtexref, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetAddress");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetAddress"));
    HOOK_CHECK(func_entry);
    return func_entry(pdptr, hTexRef);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetAddress_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetAddress_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pdptr, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetArray");
    using func_ptr = CUresult (*)(CUarray *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetArray"));
    HOOK_CHECK(func_entry);
    return func_entry(phArray, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetMipmappedArray");
    using func_ptr = CUresult (*)(CUmipmappedArray *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(phMipmappedArray, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) {
    HOOK_TRACE_PROFILE("cuTexRefGetAddressMode");
    using func_ptr = CUresult (*)(CUaddress_mode *, CUtexref, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetAddressMode"));
    HOOK_CHECK(func_entry);
    return func_entry(pam, hTexRef, dim);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetFilterMode");
    using func_ptr = CUresult (*)(CUfilter_mode *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetFilterMode"));
    HOOK_CHECK(func_entry);
    return func_entry(pfm, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetFormat");
    using func_ptr = CUresult (*)(CUarray_format *, int *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetFormat"));
    HOOK_CHECK(func_entry);
    return func_entry(pFormat, pNumChannels, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetMipmapFilterMode");
    using func_ptr = CUresult (*)(CUfilter_mode *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetMipmapFilterMode"));
    HOOK_CHECK(func_entry);
    return func_entry(pfm, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetMipmapLevelBias");
    using func_ptr = CUresult (*)(float *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetMipmapLevelBias"));
    HOOK_CHECK(func_entry);
    return func_entry(pbias, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                                                 float *pmaxMipmapLevelClamp, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetMipmapLevelClamp");
    using func_ptr = CUresult (*)(float *, float *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetMipmapLevelClamp"));
    HOOK_CHECK(func_entry);
    return func_entry(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetMaxAnisotropy");
    using func_ptr = CUresult (*)(int *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetMaxAnisotropy"));
    HOOK_CHECK(func_entry);
    return func_entry(pmaxAniso, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetBorderColor");
    using func_ptr = CUresult (*)(float *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetBorderColor"));
    HOOK_CHECK(func_entry);
    return func_entry(pBorderColor, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefGetFlags");
    using func_ptr = CUresult (*)(unsigned int *, CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefGetFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(pFlags, hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefCreate(CUtexref *pTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefCreate");
    using func_ptr = CUresult (*)(CUtexref *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefDestroy(CUtexref hTexRef) {
    HOOK_TRACE_PROFILE("cuTexRefDestroy");
    using func_ptr = CUresult (*)(CUtexref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuSurfRefSetArray");
    using func_ptr = CUresult (*)(CUsurfref, CUarray, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuSurfRefSetArray"));
    HOOK_CHECK(func_entry);
    return func_entry(hSurfRef, hArray, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
    HOOK_TRACE_PROFILE("cuSurfRefGetArray");
    using func_ptr = CUresult (*)(CUarray *, CUsurfref);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuSurfRefGetArray"));
    HOOK_CHECK(func_entry);
    return func_entry(phArray, hSurfRef);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                                                       const CUDA_TEXTURE_DESC *pTexDesc,
                                                       const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
    HOOK_TRACE_PROFILE("cuTexObjectCreate");
    using func_ptr = CUresult (*)(CUtexObject *, const CUDA_RESOURCE_DESC *, const CUDA_TEXTURE_DESC *,
                                  const CUDA_RESOURCE_VIEW_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexObjectCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexObjectDestroy(CUtexObject texObject) {
    HOOK_TRACE_PROFILE("cuTexObjectDestroy");
    using func_ptr = CUresult (*)(CUtexObject);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexObjectDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(texObject);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject) {
    HOOK_TRACE_PROFILE("cuTexObjectGetResourceDesc");
    using func_ptr = CUresult (*)(CUDA_RESOURCE_DESC *, CUtexObject);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexObjectGetResourceDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(pResDesc, texObject);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject) {
    HOOK_TRACE_PROFILE("cuTexObjectGetTextureDesc");
    using func_ptr = CUresult (*)(CUDA_TEXTURE_DESC *, CUtexObject);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexObjectGetTextureDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(pTexDesc, texObject);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
                                                                    CUtexObject texObject) {
    HOOK_TRACE_PROFILE("cuTexObjectGetResourceViewDesc");
    using func_ptr = CUresult (*)(CUDA_RESOURCE_VIEW_DESC *, CUtexObject);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexObjectGetResourceViewDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(pResViewDesc, texObject);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc) {
    HOOK_TRACE_PROFILE("cuSurfObjectCreate");
    using func_ptr = CUresult (*)(CUsurfObject *, const CUDA_RESOURCE_DESC *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuSurfObjectCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pSurfObject, pResDesc);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
    HOOK_TRACE_PROFILE("cuSurfObjectDestroy");
    using func_ptr = CUresult (*)(CUsurfObject);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuSurfObjectDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(surfObject);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                                                 CUsurfObject surfObject) {
    HOOK_TRACE_PROFILE("cuSurfObjectGetResourceDesc");
    using func_ptr = CUresult (*)(CUDA_RESOURCE_DESC *, CUsurfObject);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuSurfObjectGetResourceDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(pResDesc, surfObject);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    HOOK_TRACE_PROFILE("cuDeviceCanAccessPeer");
    using func_ptr = CUresult (*)(int *, CUdevice, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceCanAccessPeer"));
    HOOK_CHECK(func_entry);
    return func_entry(canAccessPeer, dev, peerDev);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    HOOK_TRACE_PROFILE("cuCtxEnablePeerAccess");
    using func_ptr = CUresult (*)(CUcontext, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxEnablePeerAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(peerContext, Flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    HOOK_TRACE_PROFILE("cuCtxDisablePeerAccess");
    using func_ptr = CUresult (*)(CUcontext);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxDisablePeerAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(peerContext);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib,
                                                             CUdevice srcDevice, CUdevice dstDevice) {
    HOOK_TRACE_PROFILE("cuDeviceGetP2PAttribute");
    using func_ptr = CUresult (*)(int *, CUdevice_P2PAttribute, CUdevice, CUdevice);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetP2PAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(value, attrib, srcDevice, dstDevice);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
    HOOK_TRACE_PROFILE("cuGraphicsUnregisterResource");
    using func_ptr = CUresult (*)(CUgraphicsResource);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsUnregisterResource"));
    HOOK_CHECK(func_entry);
    return func_entry(resource);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource,
                                                                         unsigned int arrayIndex,
                                                                         unsigned int mipLevel) {
    HOOK_TRACE_PROFILE("cuGraphicsSubResourceGetMappedArray");
    using func_ptr = CUresult (*)(CUarray *, CUgraphicsResource, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsSubResourceGetMappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(pArray, resource, arrayIndex, mipLevel);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray,
                                                                               CUgraphicsResource resource) {
    HOOK_TRACE_PROFILE("cuGraphicsResourceGetMappedMipmappedArray");
    using func_ptr = CUresult (*)(CUmipmappedArray *, CUgraphicsResource);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsResourceGetMappedMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(pMipmappedArray, resource);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsResourceGetMappedPointer(CUdeviceptr *pDevPtr, size_t *pSize,
                                                                        CUgraphicsResource resource) {
    HOOK_TRACE_PROFILE("cuGraphicsResourceGetMappedPointer");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUgraphicsResource);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsResourceGetMappedPointer"));
    HOOK_CHECK(func_entry);
    return func_entry(pDevPtr, pSize, resource);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize,
                                                                           CUgraphicsResource resource) {
    HOOK_TRACE_PROFILE("cuGraphicsResourceGetMappedPointer_v2");
    using func_ptr = CUresult (*)(CUdeviceptr *, size_t *, CUgraphicsResource);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsResourceGetMappedPointer_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pDevPtr, pSize, resource);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsResourceSetMapFlags(CUgraphicsResource resource, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuGraphicsResourceSetMapFlags");
    using func_ptr = CUresult (*)(CUgraphicsResource, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsResourceSetMapFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(resource, flags);
}

// manually add
HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) {
    HOOK_TRACE_PROFILE("cuGraphicsResourceSetMapFlags_v2");
    using func_ptr = CUresult (*)(CUgraphicsResource, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsResourceSetMapFlags_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(resource, flags);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources,
                                                            CUstream hStream) {
    HOOK_TRACE_PROFILE("cuGraphicsMapResources");
    using func_ptr = CUresult (*)(unsigned int, CUgraphicsResource *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsMapResources"));
    HOOK_CHECK(func_entry);
    return func_entry(count, resources, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources,
                                                              CUstream hStream) {
    HOOK_TRACE_PROFILE("cuGraphicsUnmapResources");
    using func_ptr = CUresult (*)(unsigned int, CUgraphicsResource *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGraphicsUnmapResources"));
    HOOK_CHECK(func_entry);
    return func_entry(count, resources, hStream);
}

// manually delete
// HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
//                                                       cuuint64_t flags) {
//     HOOK_TRACE_PROFILE("cuGetProcAddress");
//     using func_ptr = CUresult (*)(const char *, void **, int, cuuint64_t);
//     static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGetProcAddress"));
//     HOOK_CHECK(func_entry);
//     return func_entry(symbol, pfn, cudaVersion, flags);
// }

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
    HOOK_TRACE_PROFILE("cuGetExportTable");
    using func_ptr = CUresult (*)(const void **, const CUuuid *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGetExportTable"));
    HOOK_CHECK(func_entry);
    return func_entry(ppExportTable, pExportTableId);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuTexRefSetAddress2D_v2(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc,
                                                             CUdeviceptr dptr, size_t Pitch) {
    HOOK_TRACE_PROFILE("cuTexRefSetAddress2D_v2");
    using func_ptr = CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuTexRefSetAddress2D_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(hTexRef, desc, dptr, Pitch);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyHtoD_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoD_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcHost, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoH_v2");
    using func_ptr = CUresult (*)(void *, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoH_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcDevice, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoD_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoD_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcDevice, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice,
                                                     size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoA_v2");
    using func_ptr = CUresult (*)(CUarray, size_t, CUdeviceptr, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoA_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstArray, dstOffset, srcDevice, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset,
                                                     size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyAtoD_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAtoD_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcArray, srcOffset, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                                     size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyHtoA_v2");
    using func_ptr = CUresult (*)(CUarray, size_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoA_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstArray, dstOffset, srcHost, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset,
                                                     size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyAtoH_v2");
    using func_ptr = CUresult (*)(void *, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAtoH_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcArray, srcOffset, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray,
                                                     size_t srcOffset, size_t ByteCount) {
    HOOK_TRACE_PROFILE("cuMemcpyAtoA_v2");
    using func_ptr = CUresult (*)(CUarray, size_t, CUarray, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAtoA_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost,
                                                          size_t ByteCount, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyHtoAAsync_v2");
    using func_ptr = CUresult (*)(CUarray, size_t, const void *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoAAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset,
                                                          size_t ByteCount, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyAtoHAsync_v2");
    using func_ptr = CUresult (*)(void *, CUarray, size_t, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyAtoHAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) {
    HOOK_TRACE_PROFILE("cuMemcpy2D_v2");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy2D_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) {
    HOOK_TRACE_PROFILE("cuMemcpy2DUnaligned_v2");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy2DUnaligned_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) {
    HOOK_TRACE_PROFILE("cuMemcpy3D_v2");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy3D_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount,
                                                          CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyHtoDAsync_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, const void *, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyHtoDAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcHost, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount,
                                                          CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoHAsync_v2");
    using func_ptr = CUresult (*)(void *, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoHAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstHost, srcDevice, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                                          size_t ByteCount, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpyDtoDAsync_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpyDtoDAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, srcDevice, ByteCount, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpy2DAsync_v2");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY2D *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy2DAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    HOOK_TRACE_PROFILE("cuMemcpy3DAsync_v2");
    using func_ptr = CUresult (*)(const CUDA_MEMCPY3D *, CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemcpy3DAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pCopy, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    HOOK_TRACE_PROFILE("cuMemsetD8_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned char, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD8_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, uc, N);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    HOOK_TRACE_PROFILE("cuMemsetD16_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned short, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD16_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, us, N);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    HOOK_TRACE_PROFILE("cuMemsetD32_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, unsigned int, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD32_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, ui, N);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc,
                                                     size_t Width, size_t Height) {
    HOOK_TRACE_PROFILE("cuMemsetD2D8_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D8_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, uc, Width, Height);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us,
                                                      size_t Width, size_t Height) {
    HOOK_TRACE_PROFILE("cuMemsetD2D16_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D16_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, us, Width, Height);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                                                      size_t Width, size_t Height) {
    HOOK_TRACE_PROFILE("cuMemsetD2D32_v2");
    using func_ptr = CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemsetD2D32_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(dstDevice, dstPitch, ui, Width, Height);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamBeginCapture_ptsz(CUstream hStream) {
    HOOK_TRACE_PROFILE("cuStreamBeginCapture_ptsz");
    using func_ptr = CUresult (*)(CUstream);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamBeginCapture_ptsz"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
    HOOK_TRACE_PROFILE("cuStreamBeginCapture_v2");
    using func_ptr = CUresult (*)(CUstream, CUstreamCaptureMode);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuStreamBeginCapture_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, mode);
}

HOOK_C_API HOOK_DECL_EXPORT CUresult cuGetProcAddress_ptsz(const char *symbol, void **funcPtr, int driverVersion,
                                                           cuuint64_t flags) {
    HOOK_TRACE_PROFILE("cuGetProcAddress_ptsz");
    using func_ptr = CUresult (*)(const char *, void **, int, cuuint64_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGetProcAddress_ptsz"));
    HOOK_CHECK(func_entry);
    return func_entry(symbol, funcPtr, driverVersion, flags);
}
