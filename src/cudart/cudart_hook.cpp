// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 297 apis

#include "cudart_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceReset() {
    HOOK_TRACE_PROFILE("cudaDeviceReset");
    using func_ptr = cudaError_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceReset"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceSynchronize() {
    HOOK_TRACE_PROFILE("cudaDeviceSynchronize");
    using func_ptr = cudaError_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
    HOOK_TRACE_PROFILE("cudaDeviceSetLimit");
    using func_ptr = cudaError_t (*)(enum cudaLimit, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceSetLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(limit, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
    HOOK_TRACE_PROFILE("cudaDeviceGetLimit");
    using func_ptr = cudaError_t (*)(size_t *, enum cudaLimit);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(pValue, limit);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetTexture1DLinearMaxWidth(
    size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device) {
    HOOK_TRACE_PROFILE("cudaDeviceGetTexture1DLinearMaxWidth");
    using func_ptr = cudaError_t (*)(size_t *, const struct cudaChannelFormatDesc *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetTexture1DLinearMaxWidth"));
    HOOK_CHECK(func_entry);
    return func_entry(maxWidthInElements, fmtDesc, device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
    HOOK_TRACE_PROFILE("cudaDeviceGetCacheConfig");
    using func_ptr = cudaError_t (*)(enum cudaFuncCache *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetCacheConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(pCacheConfig);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
    HOOK_TRACE_PROFILE("cudaDeviceGetStreamPriorityRange");
    using func_ptr = cudaError_t (*)(int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetStreamPriorityRange"));
    HOOK_CHECK(func_entry);
    return func_entry(leastPriority, greatestPriority);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig) {
    HOOK_TRACE_PROFILE("cudaDeviceSetCacheConfig");
    using func_ptr = cudaError_t (*)(enum cudaFuncCache);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceSetCacheConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(cacheConfig);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig) {
    HOOK_TRACE_PROFILE("cudaDeviceGetSharedMemConfig");
    using func_ptr = cudaError_t (*)(enum cudaSharedMemConfig *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetSharedMemConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(pConfig);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config) {
    HOOK_TRACE_PROFILE("cudaDeviceSetSharedMemConfig");
    using func_ptr = cudaError_t (*)(enum cudaSharedMemConfig);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceSetSharedMemConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(config);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
    HOOK_TRACE_PROFILE("cudaDeviceGetByPCIBusId");
    using func_ptr = cudaError_t (*)(int *, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetByPCIBusId"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pciBusId);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    HOOK_TRACE_PROFILE("cudaDeviceGetPCIBusId");
    using func_ptr = cudaError_t (*)(char *, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetPCIBusId"));
    HOOK_CHECK(func_entry);
    return func_entry(pciBusId, len, device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaIpcGetEventHandle");
    using func_ptr = cudaError_t (*)(cudaIpcEventHandle_t *, cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaIpcGetEventHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) {
    HOOK_TRACE_PROFILE("cudaIpcOpenEventHandle");
    using func_ptr = cudaError_t (*)(cudaEvent_t *, cudaIpcEventHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaIpcOpenEventHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(event, handle);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
    HOOK_TRACE_PROFILE("cudaIpcGetMemHandle");
    using func_ptr = cudaError_t (*)(cudaIpcMemHandle_t *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaIpcGetMemHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, devPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle,
                                                             unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaIpcOpenMemHandle");
    using func_ptr = cudaError_t (*)(void **, cudaIpcMemHandle_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaIpcOpenMemHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, handle, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
    HOOK_TRACE_PROFILE("cudaIpcCloseMemHandle");
    using func_ptr = cudaError_t (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaIpcCloseMemHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(
    enum cudaFlushGPUDirectRDMAWritesTarget target, enum cudaFlushGPUDirectRDMAWritesScope scope) {
    HOOK_TRACE_PROFILE("cudaDeviceFlushGPUDirectRDMAWrites");
    using func_ptr = cudaError_t (*)(enum cudaFlushGPUDirectRDMAWritesTarget, enum cudaFlushGPUDirectRDMAWritesScope);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceFlushGPUDirectRDMAWrites"));
    HOOK_CHECK(func_entry);
    return func_entry(target, scope);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaThreadExit() {
    HOOK_TRACE_PROFILE("cudaThreadExit");
    using func_ptr = cudaError_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaThreadExit"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaThreadSynchronize() {
    HOOK_TRACE_PROFILE("cudaThreadSynchronize");
    using func_ptr = cudaError_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaThreadSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
    HOOK_TRACE_PROFILE("cudaThreadSetLimit");
    using func_ptr = cudaError_t (*)(enum cudaLimit, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaThreadSetLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(limit, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit) {
    HOOK_TRACE_PROFILE("cudaThreadGetLimit");
    using func_ptr = cudaError_t (*)(size_t *, enum cudaLimit);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaThreadGetLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(pValue, limit);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
    HOOK_TRACE_PROFILE("cudaThreadGetCacheConfig");
    using func_ptr = cudaError_t (*)(enum cudaFuncCache *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaThreadGetCacheConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(pCacheConfig);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
    HOOK_TRACE_PROFILE("cudaThreadSetCacheConfig");
    using func_ptr = cudaError_t (*)(enum cudaFuncCache);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaThreadSetCacheConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(cacheConfig);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetLastError() {
    HOOK_TRACE_PROFILE("cudaGetLastError");
    using func_ptr = cudaError_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetLastError"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaPeekAtLastError() {
    HOOK_TRACE_PROFILE("cudaPeekAtLastError");
    using func_ptr = cudaError_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaPeekAtLastError"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT const char *cudaGetErrorName(cudaError_t error) {
    HOOK_TRACE_PROFILE("cudaGetErrorName");
    using func_ptr = const char *(*)(cudaError_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetErrorName"));
    HOOK_CHECK(func_entry);
    return func_entry(error);
}

HOOK_C_API HOOK_DECL_EXPORT const char *cudaGetErrorString(cudaError_t error) {
    HOOK_TRACE_PROFILE("cudaGetErrorString");
    using func_ptr = const char *(*)(cudaError_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetErrorString"));
    HOOK_CHECK(func_entry);
    return func_entry(error);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetDeviceCount(int *count) {
    HOOK_TRACE_PROFILE("cudaGetDeviceCount");
    using func_ptr = cudaError_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetDeviceCount"));
    HOOK_CHECK(func_entry);
    return func_entry(count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    HOOK_TRACE_PROFILE("cudaGetDeviceProperties");
    using func_ptr = cudaError_t (*)(struct cudaDeviceProp *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetDeviceProperties"));
    HOOK_CHECK(func_entry);
    return func_entry(prop, device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
    HOOK_TRACE_PROFILE("cudaDeviceGetAttribute");
    using func_ptr = cudaError_t (*)(int *, enum cudaDeviceAttr, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(value, attr, device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device) {
    HOOK_TRACE_PROFILE("cudaDeviceGetDefaultMemPool");
    using func_ptr = cudaError_t (*)(cudaMemPool_t *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetDefaultMemPool"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool, device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
    HOOK_TRACE_PROFILE("cudaDeviceSetMemPool");
    using func_ptr = cudaError_t (*)(int, cudaMemPool_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceSetMemPool"));
    HOOK_CHECK(func_entry);
    return func_entry(device, memPool);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device) {
    HOOK_TRACE_PROFILE("cudaDeviceGetMemPool");
    using func_ptr = cudaError_t (*)(cudaMemPool_t *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetMemPool"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool, device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetNvSciSyncAttributes(void *nvSciSyncAttrList, int device,
                                                                         int flags) {
    HOOK_TRACE_PROFILE("cudaDeviceGetNvSciSyncAttributes");
    using func_ptr = cudaError_t (*)(void *, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetNvSciSyncAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(nvSciSyncAttrList, device, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr,
                                                                  int srcDevice, int dstDevice) {
    HOOK_TRACE_PROFILE("cudaDeviceGetP2PAttribute");
    using func_ptr = cudaError_t (*)(int *, enum cudaDeviceP2PAttr, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetP2PAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(value, attr, srcDevice, dstDevice);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
    HOOK_TRACE_PROFILE("cudaChooseDevice");
    using func_ptr = cudaError_t (*)(int *, const struct cudaDeviceProp *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaChooseDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(device, prop);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaSetDevice(int device) {
    HOOK_TRACE_PROFILE("cudaSetDevice");
    using func_ptr = cudaError_t (*)(int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSetDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetDevice(int *device) {
    HOOK_TRACE_PROFILE("cudaGetDevice");
    using func_ptr = cudaError_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaSetValidDevices(int *device_arr, int len) {
    HOOK_TRACE_PROFILE("cudaSetValidDevices");
    using func_ptr = cudaError_t (*)(int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSetValidDevices"));
    HOOK_CHECK(func_entry);
    return func_entry(device_arr, len);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaSetDeviceFlags");
    using func_ptr = cudaError_t (*)(unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSetDeviceFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetDeviceFlags(unsigned int *flags) {
    HOOK_TRACE_PROFILE("cudaGetDeviceFlags");
    using func_ptr = cudaError_t (*)(unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetDeviceFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    HOOK_TRACE_PROFILE("cudaStreamCreate");
    using func_ptr = cudaError_t (*)(cudaStream_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pStream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaStreamCreateWithFlags");
    using func_ptr = cudaError_t (*)(cudaStream_t *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamCreateWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(pStream, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags,
                                                                     int priority) {
    HOOK_TRACE_PROFILE("cudaStreamCreateWithPriority");
    using func_ptr = cudaError_t (*)(cudaStream_t *, unsigned int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamCreateWithPriority"));
    HOOK_CHECK(func_entry);
    return func_entry(pStream, flags, priority);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int *priority) {
    HOOK_TRACE_PROFILE("cudaStreamGetPriority");
    using func_ptr = cudaError_t (*)(cudaStream_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamGetPriority"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, priority);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags) {
    HOOK_TRACE_PROFILE("cudaStreamGetFlags");
    using func_ptr = cudaError_t (*)(cudaStream_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamGetFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaCtxResetPersistingL2Cache() {
    HOOK_TRACE_PROFILE("cudaCtxResetPersistingL2Cache");
    using func_ptr = cudaError_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaCtxResetPersistingL2Cache"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src) {
    HOOK_TRACE_PROFILE("cudaStreamCopyAttributes");
    using func_ptr = cudaError_t (*)(cudaStream_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamCopyAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamGetAttribute(cudaStream_t hStream, enum cudaStreamAttrID attr,
                                                               union cudaStreamAttrValue *value_out) {
    HOOK_TRACE_PROFILE("cudaStreamGetAttribute");
    using func_ptr = cudaError_t (*)(cudaStream_t, enum cudaStreamAttrID, union cudaStreamAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, attr, value_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, enum cudaStreamAttrID attr,
                                                               const union cudaStreamAttrValue *value) {
    HOOK_TRACE_PROFILE("cudaStreamSetAttribute");
    using func_ptr = cudaError_t (*)(cudaStream_t, enum cudaStreamAttrID, const union cudaStreamAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hStream, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaStreamDestroy");
    using func_ptr = cudaError_t (*)(cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                            unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaStreamWaitEvent");
    using func_ptr = cudaError_t (*)(cudaStream_t, cudaEvent_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamWaitEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, event, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback,
                                                              void *userData, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaStreamAddCallback");
    using func_ptr = cudaError_t (*)(cudaStream_t, cudaStreamCallback_t, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamAddCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, callback, userData, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaStreamSynchronize");
    using func_ptr = cudaError_t (*)(cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry(stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamQuery(cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaStreamQuery");
    using func_ptr = cudaError_t (*)(cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamQuery"));
    HOOK_CHECK(func_entry);
    return func_entry(stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length,
                                                                 unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaStreamAttachMemAsync");
    using func_ptr = cudaError_t (*)(cudaStream_t, void *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamAttachMemAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, devPtr, length, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode) {
    HOOK_TRACE_PROFILE("cudaStreamBeginCapture");
    using func_ptr = cudaError_t (*)(cudaStream_t, enum cudaStreamCaptureMode);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamBeginCapture"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
    HOOK_TRACE_PROFILE("cudaThreadExchangeStreamCaptureMode");
    using func_ptr = cudaError_t (*)(enum cudaStreamCaptureMode *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaThreadExchangeStreamCaptureMode"));
    HOOK_CHECK(func_entry);
    return func_entry(mode);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph) {
    HOOK_TRACE_PROFILE("cudaStreamEndCapture");
    using func_ptr = cudaError_t (*)(cudaStream_t, cudaGraph_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamEndCapture"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, pGraph);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                                              enum cudaStreamCaptureStatus *pCaptureStatus) {
    HOOK_TRACE_PROFILE("cudaStreamIsCapturing");
    using func_ptr = cudaError_t (*)(cudaStream_t, enum cudaStreamCaptureStatus *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamIsCapturing"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, pCaptureStatus);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream,
                                                                 enum cudaStreamCaptureStatus *pCaptureStatus,
                                                                 unsigned long long *pId) {
    HOOK_TRACE_PROFILE("cudaStreamGetCaptureInfo");
    using func_ptr = cudaError_t (*)(cudaStream_t, enum cudaStreamCaptureStatus *, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamGetCaptureInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, pCaptureStatus, pId);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamGetCaptureInfo_v2(cudaStream_t stream,
                                                                    enum cudaStreamCaptureStatus *captureStatus_out,
                                                                    unsigned long long *id_out, cudaGraph_t *graph_out,
                                                                    const cudaGraphNode_t **dependencies_out,
                                                                    size_t *numDependencies_out) {
    HOOK_TRACE_PROFILE("cudaStreamGetCaptureInfo_v2");
    using func_ptr = cudaError_t (*)(cudaStream_t, enum cudaStreamCaptureStatus *, unsigned long long *, cudaGraph_t *,
                                     const cudaGraphNode_t **, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamGetCaptureInfo_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamUpdateCaptureDependencies(cudaStream_t stream,
                                                                            cudaGraphNode_t *dependencies,
                                                                            size_t numDependencies,
                                                                            unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaStreamUpdateCaptureDependencies");
    using func_ptr = cudaError_t (*)(cudaStream_t, cudaGraphNode_t *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamUpdateCaptureDependencies"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, dependencies, numDependencies, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaEventCreate(cudaEvent_t *event) {
    HOOK_TRACE_PROFILE("cudaEventCreate");
    using func_ptr = cudaError_t (*)(cudaEvent_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaEventCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaEventCreateWithFlags");
    using func_ptr = cudaError_t (*)(cudaEvent_t *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaEventCreateWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(event, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaEventRecord");
    using func_ptr = cudaError_t (*)(cudaEvent_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaEventRecord"));
    HOOK_CHECK(func_entry);
    return func_entry(event, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                                 unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaEventRecordWithFlags");
    using func_ptr = cudaError_t (*)(cudaEvent_t, cudaStream_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaEventRecordWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(event, stream, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaEventQuery(cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaEventQuery");
    using func_ptr = cudaError_t (*)(cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaEventQuery"));
    HOOK_CHECK(func_entry);
    return func_entry(event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaEventSynchronize");
    using func_ptr = cudaError_t (*)(cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaEventSynchronize"));
    HOOK_CHECK(func_entry);
    return func_entry(event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaEventDestroy(cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaEventDestroy");
    using func_ptr = cudaError_t (*)(cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaEventDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    HOOK_TRACE_PROFILE("cudaEventElapsedTime");
    using func_ptr = cudaError_t (*)(float *, cudaEvent_t, cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaEventElapsedTime"));
    HOOK_CHECK(func_entry);
    return func_entry(ms, start, end);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaImportExternalMemory(
    cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc) {
    HOOK_TRACE_PROFILE("cudaImportExternalMemory");
    using func_ptr = cudaError_t (*)(cudaExternalMemory_t *, const struct cudaExternalMemoryHandleDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaImportExternalMemory"));
    HOOK_CHECK(func_entry);
    return func_entry(extMem_out, memHandleDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaExternalMemoryGetMappedBuffer(
    void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc) {
    HOOK_TRACE_PROFILE("cudaExternalMemoryGetMappedBuffer");
    using func_ptr = cudaError_t (*)(void **, cudaExternalMemory_t, const struct cudaExternalMemoryBufferDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaExternalMemoryGetMappedBuffer"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, extMem, bufferDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t
    cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray_t *mipmap, cudaExternalMemory_t extMem,
                                              const struct cudaExternalMemoryMipmappedArrayDesc *mipmapDesc) {
    HOOK_TRACE_PROFILE("cudaExternalMemoryGetMappedMipmappedArray");
    using func_ptr = cudaError_t (*)(cudaMipmappedArray_t *, cudaExternalMemory_t,
                                     const struct cudaExternalMemoryMipmappedArrayDesc *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaExternalMemoryGetMappedMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(mipmap, extMem, mipmapDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t extMem) {
    HOOK_TRACE_PROFILE("cudaDestroyExternalMemory");
    using func_ptr = cudaError_t (*)(cudaExternalMemory_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDestroyExternalMemory"));
    HOOK_CHECK(func_entry);
    return func_entry(extMem);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaImportExternalSemaphore(
    cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc) {
    HOOK_TRACE_PROFILE("cudaImportExternalSemaphore");
    using func_ptr = cudaError_t (*)(cudaExternalSemaphore_t *, const struct cudaExternalSemaphoreHandleDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaImportExternalSemaphore"));
    HOOK_CHECK(func_entry);
    return func_entry(extSem_out, semHandleDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaSignalExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaSignalExternalSemaphoresAsync");
    using func_ptr = cudaError_t (*)(const cudaExternalSemaphore_t *, const struct cudaExternalSemaphoreSignalParams *,
                                     unsigned int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSignalExternalSemaphoresAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaWaitExternalSemaphoresAsync(
    const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaWaitExternalSemaphoresAsync");
    using func_ptr = cudaError_t (*)(const cudaExternalSemaphore_t *, const struct cudaExternalSemaphoreWaitParams *,
                                     unsigned int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaWaitExternalSemaphoresAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) {
    HOOK_TRACE_PROFILE("cudaDestroyExternalSemaphore");
    using func_ptr = cudaError_t (*)(cudaExternalSemaphore_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDestroyExternalSemaphore"));
    HOOK_CHECK(func_entry);
    return func_entry(extSem);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                                                         size_t sharedMem, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaLaunchKernel");
    using func_ptr = cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaLaunchKernel"));
    HOOK_CHECK(func_entry);
    return func_entry(func, gridDim, blockDim, args, sharedMem, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                                                    void **args, size_t sharedMem,
                                                                    cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaLaunchCooperativeKernel");
    using func_ptr = cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaLaunchCooperativeKernel"));
    HOOK_CHECK(func_entry);
    return func_entry(func, gridDim, blockDim, args, sharedMem, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaLaunchCooperativeKernelMultiDevice(
    struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaLaunchCooperativeKernelMultiDevice");
    using func_ptr = cudaError_t (*)(struct cudaLaunchParams *, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaLaunchCooperativeKernelMultiDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(launchParamsList, numDevices, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig) {
    HOOK_TRACE_PROFILE("cudaFuncSetCacheConfig");
    using func_ptr = cudaError_t (*)(const void *, enum cudaFuncCache);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFuncSetCacheConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(func, cacheConfig);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config) {
    HOOK_TRACE_PROFILE("cudaFuncSetSharedMemConfig");
    using func_ptr = cudaError_t (*)(const void *, enum cudaSharedMemConfig);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFuncSetSharedMemConfig"));
    HOOK_CHECK(func_entry);
    return func_entry(func, config);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func) {
    HOOK_TRACE_PROFILE("cudaFuncGetAttributes");
    using func_ptr = cudaError_t (*)(struct cudaFuncAttributes *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFuncGetAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(attr, func);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
    HOOK_TRACE_PROFILE("cudaFuncSetAttribute");
    using func_ptr = cudaError_t (*)(const void *, enum cudaFuncAttribute, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFuncSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(func, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaSetDoubleForDevice(double *d) {
    HOOK_TRACE_PROFILE("cudaSetDoubleForDevice");
    using func_ptr = cudaError_t (*)(double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSetDoubleForDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(d);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaSetDoubleForHost(double *d) {
    HOOK_TRACE_PROFILE("cudaSetDoubleForHost");
    using func_ptr = cudaError_t (*)(double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSetDoubleForHost"));
    HOOK_CHECK(func_entry);
    return func_entry(d);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData) {
    HOOK_TRACE_PROFILE("cudaLaunchHostFunc");
    using func_ptr = cudaError_t (*)(cudaStream_t, cudaHostFn_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaLaunchHostFunc"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, fn, userData);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func,
                                                                                      int blockSize,
                                                                                      size_t dynamicSMemSize) {
    HOOK_TRACE_PROFILE("cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    using func_ptr = cudaError_t (*)(int *, const void *, int, size_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaOccupancyMaxActiveBlocksPerMultiprocessor"));
    HOOK_CHECK(func_entry);
    return func_entry(numBlocks, func, blockSize, dynamicSMemSize);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize,
                                                                                  const void *func, int numBlocks,
                                                                                  int blockSize) {
    HOOK_TRACE_PROFILE("cudaOccupancyAvailableDynamicSMemPerBlock");
    using func_ptr = cudaError_t (*)(size_t *, const void *, int, int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaOccupancyAvailableDynamicSMemPerBlock"));
    HOOK_CHECK(func_entry);
    return func_entry(dynamicSmemSize, func, numBlocks, blockSize);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    using func_ptr = cudaError_t (*)(int *, const void *, int, size_t, unsigned int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(numBlocks, func, blockSize, dynamicSMemSize, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaMallocManaged");
    using func_ptr = cudaError_t (*)(void **, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMallocManaged"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, size, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMalloc(void **devPtr, size_t size) {
    HOOK_TRACE_PROFILE("cudaMalloc");
    using func_ptr = cudaError_t (*)(void **, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMalloc"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, size);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMallocHost(void **ptr, size_t size) {
    HOOK_TRACE_PROFILE("cudaMallocHost");
    using func_ptr = cudaError_t (*)(void **, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMallocHost"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, size);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
    HOOK_TRACE_PROFILE("cudaMallocPitch");
    using func_ptr = cudaError_t (*)(void **, size_t *, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMallocPitch"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, pitch, width, height);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc,
                                                        size_t width, size_t height, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaMallocArray");
    using func_ptr = cudaError_t (*)(cudaArray_t *, const struct cudaChannelFormatDesc *, size_t, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMallocArray"));
    HOOK_CHECK(func_entry);
    return func_entry(array, desc, width, height, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFree(void *devPtr) {
    HOOK_TRACE_PROFILE("cudaFree");
    using func_ptr = cudaError_t (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFree"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFreeHost(void *ptr) {
    HOOK_TRACE_PROFILE("cudaFreeHost");
    using func_ptr = cudaError_t (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFreeHost"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFreeArray(cudaArray_t array) {
    HOOK_TRACE_PROFILE("cudaFreeArray");
    using func_ptr = cudaError_t (*)(cudaArray_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFreeArray"));
    HOOK_CHECK(func_entry);
    return func_entry(array);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
    HOOK_TRACE_PROFILE("cudaFreeMipmappedArray");
    using func_ptr = cudaError_t (*)(cudaMipmappedArray_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFreeMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(mipmappedArray);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaHostAlloc");
    using func_ptr = cudaError_t (*)(void **, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaHostAlloc"));
    HOOK_CHECK(func_entry);
    return func_entry(pHost, size, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaHostRegister");
    using func_ptr = cudaError_t (*)(void *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaHostRegister"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, size, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaHostUnregister(void *ptr) {
    HOOK_TRACE_PROFILE("cudaHostUnregister");
    using func_ptr = cudaError_t (*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaHostUnregister"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaHostGetDevicePointer");
    using func_ptr = cudaError_t (*)(void **, void *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaHostGetDevicePointer"));
    HOOK_CHECK(func_entry);
    return func_entry(pDevice, pHost, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    HOOK_TRACE_PROFILE("cudaHostGetFlags");
    using func_ptr = cudaError_t (*)(unsigned int *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaHostGetFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(pFlags, pHost);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent) {
    HOOK_TRACE_PROFILE("cudaMalloc3D");
    using func_ptr = cudaError_t (*)(struct cudaPitchedPtr *, struct cudaExtent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMalloc3D"));
    HOOK_CHECK(func_entry);
    return func_entry(pitchedDevPtr, extent);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc,
                                                          struct cudaExtent extent, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaMalloc3DArray");
    using func_ptr =
        cudaError_t (*)(cudaArray_t *, const struct cudaChannelFormatDesc *, struct cudaExtent, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMalloc3DArray"));
    HOOK_CHECK(func_entry);
    return func_entry(array, desc, extent, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray,
                                                                 const struct cudaChannelFormatDesc *desc,
                                                                 struct cudaExtent extent, unsigned int numLevels,
                                                                 unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaMallocMipmappedArray");
    using func_ptr = cudaError_t (*)(cudaMipmappedArray_t *, const struct cudaChannelFormatDesc *, struct cudaExtent,
                                     unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMallocMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(mipmappedArray, desc, extent, numLevels, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t *levelArray,
                                                                   cudaMipmappedArray_const_t mipmappedArray,
                                                                   unsigned int level) {
    HOOK_TRACE_PROFILE("cudaGetMipmappedArrayLevel");
    using func_ptr = cudaError_t (*)(cudaArray_t *, cudaMipmappedArray_const_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetMipmappedArrayLevel"));
    HOOK_CHECK(func_entry);
    return func_entry(levelArray, mipmappedArray, level);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
    HOOK_TRACE_PROFILE("cudaMemcpy3D");
    using func_ptr = cudaError_t (*)(const struct cudaMemcpy3DParms *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy3D"));
    HOOK_CHECK(func_entry);
    return func_entry(p);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) {
    HOOK_TRACE_PROFILE("cudaMemcpy3DPeer");
    using func_ptr = cudaError_t (*)(const struct cudaMemcpy3DPeerParms *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy3DPeer"));
    HOOK_CHECK(func_entry);
    return func_entry(p);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpy3DAsync");
    using func_ptr = cudaError_t (*)(const struct cudaMemcpy3DParms *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy3DAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(p, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p,
                                                              cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpy3DPeerAsync");
    using func_ptr = cudaError_t (*)(const struct cudaMemcpy3DPeerParms *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy3DPeerAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(p, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
    HOOK_TRACE_PROFILE("cudaMemGetInfo");
    using func_ptr = cudaError_t (*)(size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemGetInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(free, total);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent,
                                                         unsigned int *flags, cudaArray_t array) {
    HOOK_TRACE_PROFILE("cudaArrayGetInfo");
    using func_ptr = cudaError_t (*)(struct cudaChannelFormatDesc *, struct cudaExtent *, unsigned int *, cudaArray_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaArrayGetInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(desc, extent, flags, array);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaArrayGetPlane(cudaArray_t *pPlaneArray, cudaArray_t hArray,
                                                          unsigned int planeIdx) {
    HOOK_TRACE_PROFILE("cudaArrayGetPlane");
    using func_ptr = cudaError_t (*)(cudaArray_t *, cudaArray_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaArrayGetPlane"));
    HOOK_CHECK(func_entry);
    return func_entry(pPlaneArray, hArray, planeIdx);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties,
                                                                     cudaArray_t array) {
    HOOK_TRACE_PROFILE("cudaArrayGetSparseProperties");
    using func_ptr = cudaError_t (*)(struct cudaArraySparseProperties *, cudaArray_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaArrayGetSparseProperties"));
    HOOK_CHECK(func_entry);
    return func_entry(sparseProperties, array);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMipmappedArrayGetSparseProperties(
    struct cudaArraySparseProperties *sparseProperties, cudaMipmappedArray_t mipmap) {
    HOOK_TRACE_PROFILE("cudaMipmappedArrayGetSparseProperties");
    using func_ptr = cudaError_t (*)(struct cudaArraySparseProperties *, cudaMipmappedArray_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMipmappedArrayGetSparseProperties"));
    HOOK_CHECK(func_entry);
    return func_entry(sparseProperties, mipmap);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpy");
    using func_ptr = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src, count, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice,
                                                       size_t count) {
    HOOK_TRACE_PROFILE("cudaMemcpyPeer");
    using func_ptr = cudaError_t (*)(void *, int, const void *, int, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyPeer"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, dstDevice, src, srcDevice, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                                                     size_t width, size_t height, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpy2D");
    using func_ptr = cudaError_t (*)(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy2D"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, dpitch, src, spitch, width, height, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                                            const void *src, size_t spitch, size_t width, size_t height,
                                                            enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpy2DToArray");
    using func_ptr =
        cudaError_t (*)(cudaArray_t, size_t, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy2DToArray"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, wOffset, hOffset, src, spitch, width, height, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src,
                                                              size_t wOffset, size_t hOffset, size_t width,
                                                              size_t height, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpy2DFromArray");
    using func_ptr =
        cudaError_t (*)(void *, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy2DFromArray"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, dpitch, src, wOffset, hOffset, width, height, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                                                 cudaArray_const_t src, size_t wOffsetSrc,
                                                                 size_t hOffsetSrc, size_t width, size_t height,
                                                                 enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpy2DArrayToArray");
    using func_ptr = cudaError_t (*)(cudaArray_t, size_t, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t,
                                     enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy2DArrayToArray"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count,
                                                           size_t offset, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpyToSymbol");
    using func_ptr = cudaError_t (*)(const void *, const void *, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyToSymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(symbol, src, count, offset, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset,
                                                             enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpyFromSymbol");
    using func_ptr = cudaError_t (*)(void *, const void *, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyFromSymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, symbol, count, offset, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                                        enum cudaMemcpyKind kind, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpyAsync");
    using func_ptr = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src, count, kind, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice,
                                                            size_t count, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpyPeerAsync");
    using func_ptr = cudaError_t (*)(void *, int, const void *, int, size_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyPeerAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, dstDevice, src, srcDevice, count, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                                                          size_t width, size_t height, enum cudaMemcpyKind kind,
                                                          cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpy2DAsync");
    using func_ptr =
        cudaError_t (*)(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy2DAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, dpitch, src, spitch, width, height, kind, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                                                 const void *src, size_t spitch, size_t width,
                                                                 size_t height, enum cudaMemcpyKind kind,
                                                                 cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpy2DToArrayAsync");
    using func_ptr = cudaError_t (*)(cudaArray_t, size_t, size_t, const void *, size_t, size_t, size_t,
                                     enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy2DToArrayAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src,
                                                                   size_t wOffset, size_t hOffset, size_t width,
                                                                   size_t height, enum cudaMemcpyKind kind,
                                                                   cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpy2DFromArrayAsync");
    using func_ptr = cudaError_t (*)(void *, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t,
                                     enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpy2DFromArrayAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count,
                                                                size_t offset, enum cudaMemcpyKind kind,
                                                                cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpyToSymbolAsync");
    using func_ptr = cudaError_t (*)(const void *, const void *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyToSymbolAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(symbol, src, count, offset, kind, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count,
                                                                  size_t offset, enum cudaMemcpyKind kind,
                                                                  cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpyFromSymbolAsync");
    using func_ptr = cudaError_t (*)(void *, const void *, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyFromSymbolAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, symbol, count, offset, kind, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    HOOK_TRACE_PROFILE("cudaMemset");
    using func_ptr = cudaError_t (*)(void *, int, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemset"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, value, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width,
                                                     size_t height) {
    HOOK_TRACE_PROFILE("cudaMemset2D");
    using func_ptr = cudaError_t (*)(void *, size_t, int, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemset2D"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, pitch, value, width, height);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value,
                                                     struct cudaExtent extent) {
    HOOK_TRACE_PROFILE("cudaMemset3D");
    using func_ptr = cudaError_t (*)(struct cudaPitchedPtr, int, struct cudaExtent);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemset3D"));
    HOOK_CHECK(func_entry);
    return func_entry(pitchedDevPtr, value, extent);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemsetAsync");
    using func_ptr = cudaError_t (*)(void *, int, size_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemsetAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, value, count, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width,
                                                          size_t height, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemset2DAsync");
    using func_ptr = cudaError_t (*)(void *, size_t, int, size_t, size_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemset2DAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, pitch, value, width, height, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value,
                                                          struct cudaExtent extent, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemset3DAsync");
    using func_ptr = cudaError_t (*)(struct cudaPitchedPtr, int, struct cudaExtent, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemset3DAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(pitchedDevPtr, value, extent, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    HOOK_TRACE_PROFILE("cudaGetSymbolAddress");
    using func_ptr = cudaError_t (*)(void **, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetSymbolAddress"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, symbol);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
    HOOK_TRACE_PROFILE("cudaGetSymbolSize");
    using func_ptr = cudaError_t (*)(size_t *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetSymbolSize"));
    HOOK_CHECK(func_entry);
    return func_entry(size, symbol);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice,
                                                             cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemPrefetchAsync");
    using func_ptr = cudaError_t (*)(const void *, size_t, int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPrefetchAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, count, dstDevice, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice,
                                                      int device) {
    HOOK_TRACE_PROFILE("cudaMemAdvise");
    using func_ptr = cudaError_t (*)(const void *, size_t, enum cudaMemoryAdvise, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemAdvise"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, count, advice, device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize,
                                                                 enum cudaMemRangeAttribute attribute,
                                                                 const void *devPtr, size_t count) {
    HOOK_TRACE_PROFILE("cudaMemRangeGetAttribute");
    using func_ptr = cudaError_t (*)(void *, size_t, enum cudaMemRangeAttribute, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemRangeGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(data, dataSize, attribute, devPtr, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes,
                                                                  enum cudaMemRangeAttribute *attributes,
                                                                  size_t numAttributes, const void *devPtr,
                                                                  size_t count) {
    HOOK_TRACE_PROFILE("cudaMemRangeGetAttributes");
    using func_ptr = cudaError_t (*)(void **, size_t *, enum cudaMemRangeAttribute *, size_t, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemRangeGetAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(data, dataSizes, attributes, numAttributes, devPtr, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                                          const void *src, size_t count, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpyToArray");
    using func_ptr = cudaError_t (*)(cudaArray_t, size_t, size_t, const void *, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyToArray"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, wOffset, hOffset, src, count, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset,
                                                            size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpyFromArray");
    using func_ptr = cudaError_t (*)(void *, cudaArray_const_t, size_t, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyFromArray"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src, wOffset, hOffset, count, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst,
                                                               cudaArray_const_t src, size_t wOffsetSrc,
                                                               size_t hOffsetSrc, size_t count,
                                                               enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaMemcpyArrayToArray");
    using func_ptr =
        cudaError_t (*)(cudaArray_t, size_t, size_t, cudaArray_const_t, size_t, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyArrayToArray"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                                               const void *src, size_t count, enum cudaMemcpyKind kind,
                                                               cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpyToArrayAsync");
    using func_ptr =
        cudaError_t (*)(cudaArray_t, size_t, size_t, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyToArrayAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, wOffset, hOffset, src, count, kind, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src, size_t wOffset,
                                                                 size_t hOffset, size_t count, enum cudaMemcpyKind kind,
                                                                 cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMemcpyFromArrayAsync");
    using func_ptr =
        cudaError_t (*)(void *, cudaArray_const_t, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemcpyFromArrayAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(dst, src, wOffset, hOffset, count, kind, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream) {
    HOOK_TRACE_PROFILE("cudaMallocAsync");
    using func_ptr = cudaError_t (*)(void **, size_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMallocAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, size, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream) {
    HOOK_TRACE_PROFILE("cudaFreeAsync");
    using func_ptr = cudaError_t (*)(void *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFreeAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, hStream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep) {
    HOOK_TRACE_PROFILE("cudaMemPoolTrimTo");
    using func_ptr = cudaError_t (*)(cudaMemPool_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolTrimTo"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool, minBytesToKeep);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr,
                                                                void *value) {
    HOOK_TRACE_PROFILE("cudaMemPoolSetAttribute");
    using func_ptr = cudaError_t (*)(cudaMemPool_t, enum cudaMemPoolAttr, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr,
                                                                void *value) {
    HOOK_TRACE_PROFILE("cudaMemPoolGetAttribute");
    using func_ptr = cudaError_t (*)(cudaMemPool_t, enum cudaMemPoolAttr, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolSetAccess(cudaMemPool_t memPool,
                                                             const struct cudaMemAccessDesc *descList, size_t count) {
    HOOK_TRACE_PROFILE("cudaMemPoolSetAccess");
    using func_ptr = cudaError_t (*)(cudaMemPool_t, const struct cudaMemAccessDesc *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolSetAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool, descList, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool,
                                                             struct cudaMemLocation *location) {
    HOOK_TRACE_PROFILE("cudaMemPoolGetAccess");
    using func_ptr = cudaError_t (*)(enum cudaMemAccessFlags *, cudaMemPool_t, struct cudaMemLocation *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolGetAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(flags, memPool, location);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool,
                                                          const struct cudaMemPoolProps *poolProps) {
    HOOK_TRACE_PROFILE("cudaMemPoolCreate");
    using func_ptr = cudaError_t (*)(cudaMemPool_t *, const struct cudaMemPoolProps *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool, poolProps);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolDestroy(cudaMemPool_t memPool) {
    HOOK_TRACE_PROFILE("cudaMemPoolDestroy");
    using func_ptr = cudaError_t (*)(cudaMemPool_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool,
                                                                cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaMallocFromPoolAsync");
    using func_ptr = cudaError_t (*)(void **, size_t, cudaMemPool_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMallocFromPoolAsync"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, size, memPool, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolExportToShareableHandle(void *shareableHandle, cudaMemPool_t memPool,
                                                                           enum cudaMemAllocationHandleType handleType,
                                                                           unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaMemPoolExportToShareableHandle");
    using func_ptr = cudaError_t (*)(void *, cudaMemPool_t, enum cudaMemAllocationHandleType, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolExportToShareableHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(shareableHandle, memPool, handleType, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t *memPool, void *shareableHandle, enum cudaMemAllocationHandleType handleType, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaMemPoolImportFromShareableHandle");
    using func_ptr = cudaError_t (*)(cudaMemPool_t *, void *, enum cudaMemAllocationHandleType, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolImportFromShareableHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(memPool, shareableHandle, handleType, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData,
                                                                 void *ptr) {
    HOOK_TRACE_PROFILE("cudaMemPoolExportPointer");
    using func_ptr = cudaError_t (*)(struct cudaMemPoolPtrExportData *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolExportPointer"));
    HOOK_CHECK(func_entry);
    return func_entry(exportData, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool,
                                                                 struct cudaMemPoolPtrExportData *exportData) {
    HOOK_TRACE_PROFILE("cudaMemPoolImportPointer");
    using func_ptr = cudaError_t (*)(void **, cudaMemPool_t, struct cudaMemPoolPtrExportData *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaMemPoolImportPointer"));
    HOOK_CHECK(func_entry);
    return func_entry(ptr, memPool, exportData);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                                                                 const void *ptr) {
    HOOK_TRACE_PROFILE("cudaPointerGetAttributes");
    using func_ptr = cudaError_t (*)(struct cudaPointerAttributes *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaPointerGetAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(attributes, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
    HOOK_TRACE_PROFILE("cudaDeviceCanAccessPeer");
    using func_ptr = cudaError_t (*)(int *, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceCanAccessPeer"));
    HOOK_CHECK(func_entry);
    return func_entry(canAccessPeer, device, peerDevice);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaDeviceEnablePeerAccess");
    using func_ptr = cudaError_t (*)(int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceEnablePeerAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(peerDevice, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
    HOOK_TRACE_PROFILE("cudaDeviceDisablePeerAccess");
    using func_ptr = cudaError_t (*)(int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceDisablePeerAccess"));
    HOOK_CHECK(func_entry);
    return func_entry(peerDevice);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    HOOK_TRACE_PROFILE("cudaGraphicsUnregisterResource");
    using func_ptr = cudaError_t (*)(cudaGraphicsResource_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphicsUnregisterResource"));
    HOOK_CHECK(func_entry);
    return func_entry(resource);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource,
                                                                        unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaGraphicsResourceSetMapFlags");
    using func_ptr = cudaError_t (*)(cudaGraphicsResource_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphicsResourceSetMapFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(resource, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources,
                                                                 cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaGraphicsMapResources");
    using func_ptr = cudaError_t (*)(int, cudaGraphicsResource_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphicsMapResources"));
    HOOK_CHECK(func_entry);
    return func_entry(count, resources, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources,
                                                                   cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaGraphicsUnmapResources");
    using func_ptr = cudaError_t (*)(int, cudaGraphicsResource_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphicsUnmapResources"));
    HOOK_CHECK(func_entry);
    return func_entry(count, resources, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size,
                                                                             cudaGraphicsResource_t resource) {
    HOOK_TRACE_PROFILE("cudaGraphicsResourceGetMappedPointer");
    using func_ptr = cudaError_t (*)(void **, size_t *, cudaGraphicsResource_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphicsResourceGetMappedPointer"));
    HOOK_CHECK(func_entry);
    return func_entry(devPtr, size, resource);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t *array,
                                                                              cudaGraphicsResource_t resource,
                                                                              unsigned int arrayIndex,
                                                                              unsigned int mipLevel) {
    HOOK_TRACE_PROFILE("cudaGraphicsSubResourceGetMappedArray");
    using func_ptr = cudaError_t (*)(cudaArray_t *, cudaGraphicsResource_t, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphicsSubResourceGetMappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(array, resource, arrayIndex, mipLevel);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t
    cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t *mipmappedArray, cudaGraphicsResource_t resource) {
    HOOK_TRACE_PROFILE("cudaGraphicsResourceGetMappedMipmappedArray");
    using func_ptr = cudaError_t (*)(cudaMipmappedArray_t *, cudaGraphicsResource_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphicsResourceGetMappedMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(mipmappedArray, resource);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaBindTexture(size_t *offset, const struct textureReference *texref,
                                                        const void *devPtr, const struct cudaChannelFormatDesc *desc,
                                                        size_t size) {
    HOOK_TRACE_PROFILE("cudaBindTexture");
    using func_ptr = cudaError_t (*)(size_t *, const struct textureReference *, const void *,
                                     const struct cudaChannelFormatDesc *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaBindTexture"));
    HOOK_CHECK(func_entry);
    return func_entry(offset, texref, devPtr, desc, size);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaBindTexture2D(size_t *offset, const struct textureReference *texref,
                                                          const void *devPtr, const struct cudaChannelFormatDesc *desc,
                                                          size_t width, size_t height, size_t pitch) {
    HOOK_TRACE_PROFILE("cudaBindTexture2D");
    using func_ptr = cudaError_t (*)(size_t *, const struct textureReference *, const void *,
                                     const struct cudaChannelFormatDesc *, size_t, size_t, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaBindTexture2D"));
    HOOK_CHECK(func_entry);
    return func_entry(offset, texref, devPtr, desc, width, height, pitch);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaBindTextureToArray(const struct textureReference *texref,
                                                               cudaArray_const_t array,
                                                               const struct cudaChannelFormatDesc *desc) {
    HOOK_TRACE_PROFILE("cudaBindTextureToArray");
    using func_ptr =
        cudaError_t (*)(const struct textureReference *, cudaArray_const_t, const struct cudaChannelFormatDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaBindTextureToArray"));
    HOOK_CHECK(func_entry);
    return func_entry(texref, array, desc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaBindTextureToMipmappedArray(const struct textureReference *texref,
                                                                        cudaMipmappedArray_const_t mipmappedArray,
                                                                        const struct cudaChannelFormatDesc *desc) {
    HOOK_TRACE_PROFILE("cudaBindTextureToMipmappedArray");
    using func_ptr = cudaError_t (*)(const struct textureReference *, cudaMipmappedArray_const_t,
                                     const struct cudaChannelFormatDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaBindTextureToMipmappedArray"));
    HOOK_CHECK(func_entry);
    return func_entry(texref, mipmappedArray, desc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaUnbindTexture(const struct textureReference *texref) {
    HOOK_TRACE_PROFILE("cudaUnbindTexture");
    using func_ptr = cudaError_t (*)(const struct textureReference *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaUnbindTexture"));
    HOOK_CHECK(func_entry);
    return func_entry(texref);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
                                                                      const struct textureReference *texref) {
    HOOK_TRACE_PROFILE("cudaGetTextureAlignmentOffset");
    using func_ptr = cudaError_t (*)(size_t *, const struct textureReference *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetTextureAlignmentOffset"));
    HOOK_CHECK(func_entry);
    return func_entry(offset, texref);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetTextureReference(const struct textureReference **texref,
                                                                const void *symbol) {
    HOOK_TRACE_PROFILE("cudaGetTextureReference");
    using func_ptr = cudaError_t (*)(const struct textureReference **, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetTextureReference"));
    HOOK_CHECK(func_entry);
    return func_entry(texref, symbol);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaBindSurfaceToArray(const struct surfaceReference *surfref,
                                                               cudaArray_const_t array,
                                                               const struct cudaChannelFormatDesc *desc) {
    HOOK_TRACE_PROFILE("cudaBindSurfaceToArray");
    using func_ptr =
        cudaError_t (*)(const struct surfaceReference *, cudaArray_const_t, const struct cudaChannelFormatDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaBindSurfaceToArray"));
    HOOK_CHECK(func_entry);
    return func_entry(surfref, array, desc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetSurfaceReference(const struct surfaceReference **surfref,
                                                                const void *symbol) {
    HOOK_TRACE_PROFILE("cudaGetSurfaceReference");
    using func_ptr = cudaError_t (*)(const struct surfaceReference **, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetSurfaceReference"));
    HOOK_CHECK(func_entry);
    return func_entry(surfref, symbol);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
                                                           cudaArray_const_t array) {
    HOOK_TRACE_PROFILE("cudaGetChannelDesc");
    using func_ptr = cudaError_t (*)(struct cudaChannelFormatDesc *, cudaArray_const_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetChannelDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(desc, array);
}

HOOK_C_API HOOK_DECL_EXPORT struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                                                               enum cudaChannelFormatKind f) {
    HOOK_TRACE_PROFILE("cudaCreateChannelDesc");
    using func_ptr = struct cudaChannelFormatDesc (*)(int, int, int, int, enum cudaChannelFormatKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaCreateChannelDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(x, y, z, w, f);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                                                                const struct cudaResourceDesc *pResDesc,
                                                                const struct cudaTextureDesc *pTexDesc,
                                                                const struct cudaResourceViewDesc *pResViewDesc) {
    HOOK_TRACE_PROFILE("cudaCreateTextureObject");
    using func_ptr = cudaError_t (*)(cudaTextureObject_t *, const struct cudaResourceDesc *,
                                     const struct cudaTextureDesc *, const struct cudaResourceViewDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaCreateTextureObject"));
    HOOK_CHECK(func_entry);
    return func_entry(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    HOOK_TRACE_PROFILE("cudaDestroyTextureObject");
    using func_ptr = cudaError_t (*)(cudaTextureObject_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDestroyTextureObject"));
    HOOK_CHECK(func_entry);
    return func_entry(texObject);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                                                         cudaTextureObject_t texObject) {
    HOOK_TRACE_PROFILE("cudaGetTextureObjectResourceDesc");
    using func_ptr = cudaError_t (*)(struct cudaResourceDesc *, cudaTextureObject_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetTextureObjectResourceDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(pResDesc, texObject);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc,
                                                                        cudaTextureObject_t texObject) {
    HOOK_TRACE_PROFILE("cudaGetTextureObjectTextureDesc");
    using func_ptr = cudaError_t (*)(struct cudaTextureDesc *, cudaTextureObject_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetTextureObjectTextureDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(pTexDesc, texObject);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc,
                                                                             cudaTextureObject_t texObject) {
    HOOK_TRACE_PROFILE("cudaGetTextureObjectResourceViewDesc");
    using func_ptr = cudaError_t (*)(struct cudaResourceViewDesc *, cudaTextureObject_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetTextureObjectResourceViewDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(pResViewDesc, texObject);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject,
                                                                const struct cudaResourceDesc *pResDesc) {
    HOOK_TRACE_PROFILE("cudaCreateSurfaceObject");
    using func_ptr = cudaError_t (*)(cudaSurfaceObject_t *, const struct cudaResourceDesc *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaCreateSurfaceObject"));
    HOOK_CHECK(func_entry);
    return func_entry(pSurfObject, pResDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    HOOK_TRACE_PROFILE("cudaDestroySurfaceObject");
    using func_ptr = cudaError_t (*)(cudaSurfaceObject_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDestroySurfaceObject"));
    HOOK_CHECK(func_entry);
    return func_entry(surfObject);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                                                         cudaSurfaceObject_t surfObject) {
    HOOK_TRACE_PROFILE("cudaGetSurfaceObjectResourceDesc");
    using func_ptr = cudaError_t (*)(struct cudaResourceDesc *, cudaSurfaceObject_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetSurfaceObjectResourceDesc"));
    HOOK_CHECK(func_entry);
    return func_entry(pResDesc, surfObject);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDriverGetVersion(int *driverVersion) {
    HOOK_TRACE_PROFILE("cudaDriverGetVersion");
    using func_ptr = cudaError_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDriverGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(driverVersion);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    HOOK_TRACE_PROFILE("cudaRuntimeGetVersion");
    using func_ptr = cudaError_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaRuntimeGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(runtimeVersion);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaGraphCreate");
    using func_ptr = cudaError_t (*)(cudaGraph_t *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraph, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                               const cudaGraphNode_t *pDependencies,
                                                               size_t numDependencies,
                                                               const struct cudaKernelNodeParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphAddKernelNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t,
                                     const struct cudaKernelNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddKernelNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node,
                                                                     struct cudaKernelNodeParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphKernelNodeGetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, struct cudaKernelNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphKernelNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node,
                                                                     const struct cudaKernelNodeParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphKernelNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, const struct cudaKernelNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphKernelNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst) {
    HOOK_TRACE_PROFILE("cudaGraphKernelNodeCopyAttributes");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, cudaGraphNode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphKernelNodeCopyAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(hSrc, hDst);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode,
                                                                        enum cudaKernelNodeAttrID attr,
                                                                        union cudaKernelNodeAttrValue *value_out) {
    HOOK_TRACE_PROFILE("cudaGraphKernelNodeGetAttribute");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, enum cudaKernelNodeAttrID, union cudaKernelNodeAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphKernelNodeGetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, attr, value_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode,
                                                                        enum cudaKernelNodeAttrID attr,
                                                                        const union cudaKernelNodeAttrValue *value) {
    HOOK_TRACE_PROFILE("cudaGraphKernelNodeSetAttribute");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, enum cudaKernelNodeAttrID, const union cudaKernelNodeAttrValue *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphKernelNodeSetAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                               const cudaGraphNode_t *pDependencies,
                                                               size_t numDependencies,
                                                               const struct cudaMemcpy3DParms *pCopyParams) {
    HOOK_TRACE_PROFILE("cudaGraphAddMemcpyNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t,
                                     const struct cudaMemcpy3DParms *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddMemcpyNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, pCopyParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                                       const cudaGraphNode_t *pDependencies,
                                                                       size_t numDependencies, const void *symbol,
                                                                       const void *src, size_t count, size_t offset,
                                                                       enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphAddMemcpyNodeToSymbol");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, const void *,
                                     const void *, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddMemcpyNodeToSymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                                         const cudaGraphNode_t *pDependencies,
                                                                         size_t numDependencies, void *dst,
                                                                         const void *symbol, size_t count,
                                                                         size_t offset, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphAddMemcpyNodeFromSymbol");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, void *,
                                     const void *, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddMemcpyNodeFromSymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                                 const cudaGraphNode_t *pDependencies,
                                                                 size_t numDependencies, void *dst, const void *src,
                                                                 size_t count, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphAddMemcpyNode1D");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, void *,
                                     const void *, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddMemcpyNode1D"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,
                                                                     struct cudaMemcpy3DParms *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphMemcpyNodeGetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, struct cudaMemcpy3DParms *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemcpyNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node,
                                                                     const struct cudaMemcpy3DParms *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphMemcpyNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, const struct cudaMemcpy3DParms *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemcpyNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void *symbol,
                                                                             const void *src, size_t count,
                                                                             size_t offset, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphMemcpyNodeSetParamsToSymbol");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, const void *, const void *, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemcpyNodeSetParamsToSymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(node, symbol, src, count, offset, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void *dst,
                                                                               const void *symbol, size_t count,
                                                                               size_t offset,
                                                                               enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphMemcpyNodeSetParamsFromSymbol");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, void *, const void *, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemcpyNodeSetParamsFromSymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(node, dst, symbol, count, offset, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void *dst, const void *src,
                                                                       size_t count, enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphMemcpyNodeSetParams1D");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, void *, const void *, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemcpyNodeSetParams1D"));
    HOOK_CHECK(func_entry);
    return func_entry(node, dst, src, count, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                               const cudaGraphNode_t *pDependencies,
                                                               size_t numDependencies,
                                                               const struct cudaMemsetParams *pMemsetParams) {
    HOOK_TRACE_PROFILE("cudaGraphAddMemsetNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t,
                                     const struct cudaMemsetParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddMemsetNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node,
                                                                     struct cudaMemsetParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphMemsetNodeGetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, struct cudaMemsetParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemsetNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemsetNodeSetParams(cudaGraphNode_t node,
                                                                     const struct cudaMemsetParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphMemsetNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, const struct cudaMemsetParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemsetNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                             const cudaGraphNode_t *pDependencies,
                                                             size_t numDependencies,
                                                             const struct cudaHostNodeParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphAddHostNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t,
                                     const struct cudaHostNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddHostNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node,
                                                                   struct cudaHostNodeParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphHostNodeGetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, struct cudaHostNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphHostNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphHostNodeSetParams(cudaGraphNode_t node,
                                                                   const struct cudaHostNodeParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphHostNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, const struct cudaHostNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphHostNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                                   const cudaGraphNode_t *pDependencies,
                                                                   size_t numDependencies, cudaGraph_t childGraph) {
    HOOK_TRACE_PROFILE("cudaGraphAddChildGraphNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, cudaGraph_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddChildGraphNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, childGraph);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph) {
    HOOK_TRACE_PROFILE("cudaGraphChildGraphNodeGetGraph");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, cudaGraph_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphChildGraphNodeGetGraph"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pGraph);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                              const cudaGraphNode_t *pDependencies,
                                                              size_t numDependencies) {
    HOOK_TRACE_PROFILE("cudaGraphAddEmptyNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddEmptyNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                                    const cudaGraphNode_t *pDependencies,
                                                                    size_t numDependencies, cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaGraphAddEventRecordNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddEventRecordNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out) {
    HOOK_TRACE_PROFILE("cudaGraphEventRecordNodeGetEvent");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, cudaEvent_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphEventRecordNodeGetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(node, event_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaGraphEventRecordNodeSetEvent");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphEventRecordNodeSetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(node, event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                                  const cudaGraphNode_t *pDependencies,
                                                                  size_t numDependencies, cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaGraphAddEventWaitNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddEventWaitNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out) {
    HOOK_TRACE_PROFILE("cudaGraphEventWaitNodeGetEvent");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, cudaEvent_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphEventWaitNodeGetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(node, event_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaGraphEventWaitNodeSetEvent");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphEventWaitNodeSetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(node, event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddExternalSemaphoresSignalNode(
    cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies,
    const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphAddExternalSemaphoresSignalNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t,
                                     const struct cudaExternalSemaphoreSignalNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddExternalSemaphoresSignalNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams(
    cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams *params_out) {
    HOOK_TRACE_PROFILE("cudaGraphExternalSemaphoresSignalNodeGetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, struct cudaExternalSemaphoreSignalNodeParams *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExternalSemaphoresSignalNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, params_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams(
    cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphExternalSemaphoresSignalNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, const struct cudaExternalSemaphoreSignalNodeParams *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExternalSemaphoresSignalNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddExternalSemaphoresWaitNode(
    cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies,
    const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphAddExternalSemaphoresWaitNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t,
                                     const struct cudaExternalSemaphoreWaitNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddExternalSemaphoresWaitNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams(
    cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams *params_out) {
    HOOK_TRACE_PROFILE("cudaGraphExternalSemaphoresWaitNodeGetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, struct cudaExternalSemaphoreWaitNodeParams *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExternalSemaphoresWaitNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, params_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams(
    cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphExternalSemaphoresWaitNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, const struct cudaExternalSemaphoreWaitNodeParams *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExternalSemaphoresWaitNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                                 const cudaGraphNode_t *pDependencies,
                                                                 size_t numDependencies,
                                                                 struct cudaMemAllocNodeParams *nodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphAddMemAllocNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t,
                                     struct cudaMemAllocNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddMemAllocNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node,
                                                                       struct cudaMemAllocNodeParams *params_out) {
    HOOK_TRACE_PROFILE("cudaGraphMemAllocNodeGetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, struct cudaMemAllocNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemAllocNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, params_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                                const cudaGraphNode_t *pDependencies,
                                                                size_t numDependencies, void *dptr) {
    HOOK_TRACE_PROFILE("cudaGraphAddMemFreeNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraph_t, const cudaGraphNode_t *, size_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddMemFreeNode"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphNode, graph, pDependencies, numDependencies, dptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out) {
    HOOK_TRACE_PROFILE("cudaGraphMemFreeNodeGetParams");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphMemFreeNodeGetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(node, dptr_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGraphMemTrim(int device) {
    HOOK_TRACE_PROFILE("cudaDeviceGraphMemTrim");
    using func_ptr = cudaError_t (*)(int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGraphMemTrim"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr,
                                                                       void *value) {
    HOOK_TRACE_PROFILE("cudaDeviceGetGraphMemAttribute");
    using func_ptr = cudaError_t (*)(int, enum cudaGraphMemAttributeType, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetGraphMemAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(device, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr,
                                                                       void *value) {
    HOOK_TRACE_PROFILE("cudaDeviceSetGraphMemAttribute");
    using func_ptr = cudaError_t (*)(int, enum cudaGraphMemAttributeType, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceSetGraphMemAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(device, attr, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph) {
    HOOK_TRACE_PROFILE("cudaGraphClone");
    using func_ptr = cudaError_t (*)(cudaGraph_t *, cudaGraph_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphClone"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphClone, originalGraph);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode,
                                                                 cudaGraph_t clonedGraph) {
    HOOK_TRACE_PROFILE("cudaGraphNodeFindInClone");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t *, cudaGraphNode_t, cudaGraph_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphNodeFindInClone"));
    HOOK_CHECK(func_entry);
    return func_entry(pNode, originalNode, clonedGraph);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType) {
    HOOK_TRACE_PROFILE("cudaGraphNodeGetType");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, enum cudaGraphNodeType *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphNodeGetType"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pType);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes) {
    HOOK_TRACE_PROFILE("cudaGraphGetNodes");
    using func_ptr = cudaError_t (*)(cudaGraph_t, cudaGraphNode_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphGetNodes"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, nodes, numNodes);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes,
                                                              size_t *pNumRootNodes) {
    HOOK_TRACE_PROFILE("cudaGraphGetRootNodes");
    using func_ptr = cudaError_t (*)(cudaGraph_t, cudaGraphNode_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphGetRootNodes"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, pRootNodes, pNumRootNodes);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to,
                                                          size_t *numEdges) {
    HOOK_TRACE_PROFILE("cudaGraphGetEdges");
    using func_ptr = cudaError_t (*)(cudaGraph_t, cudaGraphNode_t *, cudaGraphNode_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphGetEdges"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, from, to, numEdges);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphNodeGetDependencies(cudaGraphNode_t node,
                                                                     cudaGraphNode_t *pDependencies,
                                                                     size_t *pNumDependencies) {
    HOOK_TRACE_PROFILE("cudaGraphNodeGetDependencies");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, cudaGraphNode_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphNodeGetDependencies"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pDependencies, pNumDependencies);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
                                                                       cudaGraphNode_t *pDependentNodes,
                                                                       size_t *pNumDependentNodes) {
    HOOK_TRACE_PROFILE("cudaGraphNodeGetDependentNodes");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t, cudaGraphNode_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphNodeGetDependentNodes"));
    HOOK_CHECK(func_entry);
    return func_entry(node, pDependentNodes, pNumDependentNodes);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from,
                                                                 const cudaGraphNode_t *to, size_t numDependencies) {
    HOOK_TRACE_PROFILE("cudaGraphAddDependencies");
    using func_ptr = cudaError_t (*)(cudaGraph_t, const cudaGraphNode_t *, const cudaGraphNode_t *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphAddDependencies"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, from, to, numDependencies);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from,
                                                                    const cudaGraphNode_t *to, size_t numDependencies) {
    HOOK_TRACE_PROFILE("cudaGraphRemoveDependencies");
    using func_ptr = cudaError_t (*)(cudaGraph_t, const cudaGraphNode_t *, const cudaGraphNode_t *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphRemoveDependencies"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, from, to, numDependencies);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphDestroyNode(cudaGraphNode_t node) {
    HOOK_TRACE_PROFILE("cudaGraphDestroyNode");
    using func_ptr = cudaError_t (*)(cudaGraphNode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphDestroyNode"));
    HOOK_CHECK(func_entry);
    return func_entry(node);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                                             cudaGraphNode_t *pErrorNode, char *pLogBuffer,
                                                             size_t bufferSize) {
    HOOK_TRACE_PROFILE("cudaGraphInstantiate");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t *, cudaGraph_t, cudaGraphNode_t *, char *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphInstantiate"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
                                                                      unsigned long long flags) {
    HOOK_TRACE_PROFILE("cudaGraphInstantiateWithFlags");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t *, cudaGraph_t, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphInstantiateWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(pGraphExec, graph, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecKernelNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphExecKernelNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, const struct cudaKernelNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecKernelNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec,
                                                                         cudaGraphNode_t node,
                                                                         const struct cudaMemcpy3DParms *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphExecMemcpyNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, const struct cudaMemcpy3DParms *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecMemcpyNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec,
                                                                                 cudaGraphNode_t node,
                                                                                 const void *symbol, const void *src,
                                                                                 size_t count, size_t offset,
                                                                                 enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphExecMemcpyNodeSetParamsToSymbol");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, const void *, const void *, size_t, size_t,
                                     enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecMemcpyNodeSetParamsToSymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, node, symbol, src, count, offset, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec,
                                                                                   cudaGraphNode_t node, void *dst,
                                                                                   const void *symbol, size_t count,
                                                                                   size_t offset,
                                                                                   enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphExecMemcpyNodeSetParamsFromSymbol");
    using func_ptr =
        cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, void *, const void *, size_t, size_t, enum cudaMemcpyKind);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecMemcpyNodeSetParamsFromSymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, node, dst, symbol, count, offset, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec,
                                                                           cudaGraphNode_t node, void *dst,
                                                                           const void *src, size_t count,
                                                                           enum cudaMemcpyKind kind) {
    HOOK_TRACE_PROFILE("cudaGraphExecMemcpyNodeSetParams1D");
    using func_ptr =
        cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, void *, const void *, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecMemcpyNodeSetParams1D"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, node, dst, src, count, kind);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec,
                                                                         cudaGraphNode_t node,
                                                                         const struct cudaMemsetParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphExecMemsetNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, const struct cudaMemsetParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecMemsetNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
                                                                       const struct cudaHostNodeParams *pNodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphExecHostNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, const struct cudaHostNodeParams *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecHostNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, node, pNodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec,
                                                                             cudaGraphNode_t node,
                                                                             cudaGraph_t childGraph) {
    HOOK_TRACE_PROFILE("cudaGraphExecChildGraphNodeSetParams");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, cudaGraph_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecChildGraphNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, node, childGraph);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec,
                                                                             cudaGraphNode_t hNode, cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaGraphExecEventRecordNodeSetEvent");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecEventRecordNodeSetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec,
                                                                           cudaGraphNode_t hNode, cudaEvent_t event) {
    HOOK_TRACE_PROFILE("cudaGraphExecEventWaitNodeSetEvent");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, cudaEvent_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecEventWaitNodeSetEvent"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, event);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecExternalSemaphoresSignalNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphExecExternalSemaphoresSignalNodeSetParams");
    using func_ptr =
        cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, const struct cudaExternalSemaphoreSignalNodeParams *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecExternalSemaphoresSignalNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecExternalSemaphoresWaitNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams) {
    HOOK_TRACE_PROFILE("cudaGraphExecExternalSemaphoresWaitNodeSetParams");
    using func_ptr =
        cudaError_t (*)(cudaGraphExec_t, cudaGraphNode_t, const struct cudaExternalSemaphoreWaitNodeParams *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecExternalSemaphoresWaitNodeSetParams"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hNode, nodeParams);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                                            cudaGraphNode_t *hErrorNode_out,
                                                            enum cudaGraphExecUpdateResult *updateResult_out) {
    HOOK_TRACE_PROFILE("cudaGraphExecUpdate");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaGraph_t, cudaGraphNode_t *, enum cudaGraphExecUpdateResult *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecUpdate"));
    HOOK_CHECK(func_entry);
    return func_entry(hGraphExec, hGraph, hErrorNode_out, updateResult_out);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaGraphUpload");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphUpload"));
    HOOK_CHECK(func_entry);
    return func_entry(graphExec, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaGraphLaunch");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphLaunch"));
    HOOK_CHECK(func_entry);
    return func_entry(graphExec, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    HOOK_TRACE_PROFILE("cudaGraphExecDestroy");
    using func_ptr = cudaError_t (*)(cudaGraphExec_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphExecDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(graphExec);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    HOOK_TRACE_PROFILE("cudaGraphDestroy");
    using func_ptr = cudaError_t (*)(cudaGraph_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(graph);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path,
                                                               unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaGraphDebugDotPrint");
    using func_ptr = cudaError_t (*)(cudaGraph_t, const char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphDebugDotPrint"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, path, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr,
                                                             cudaHostFn_t destroy, unsigned int initialRefcount,
                                                             unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaUserObjectCreate");
    using func_ptr = cudaError_t (*)(cudaUserObject_t *, void *, cudaHostFn_t, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaUserObjectCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(object_out, ptr, destroy, initialRefcount, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaUserObjectRetain(cudaUserObject_t object, unsigned int count) {
    HOOK_TRACE_PROFILE("cudaUserObjectRetain");
    using func_ptr = cudaError_t (*)(cudaUserObject_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaUserObjectRetain"));
    HOOK_CHECK(func_entry);
    return func_entry(object, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaUserObjectRelease(cudaUserObject_t object, unsigned int count) {
    HOOK_TRACE_PROFILE("cudaUserObjectRelease");
    using func_ptr = cudaError_t (*)(cudaUserObject_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaUserObjectRelease"));
    HOOK_CHECK(func_entry);
    return func_entry(object, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object,
                                                                  unsigned int count, unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaGraphRetainUserObject");
    using func_ptr = cudaError_t (*)(cudaGraph_t, cudaUserObject_t, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphRetainUserObject"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, object, count, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object,
                                                                   unsigned int count) {
    HOOK_TRACE_PROFILE("cudaGraphReleaseUserObject");
    using func_ptr = cudaError_t (*)(cudaGraph_t, cudaUserObject_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGraphReleaseUserObject"));
    HOOK_CHECK(func_entry);
    return func_entry(graph, object, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                                                                unsigned long long flags) {
    HOOK_TRACE_PROFILE("cudaGetDriverEntryPoint");
    using func_ptr = cudaError_t (*)(const char *, void **, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetDriverEntryPoint"));
    HOOK_CHECK(func_entry);
    return func_entry(symbol, funcPtr, flags);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetExportTable(const void **ppExportTable,
                                                           const cudaUUID_t *pExportTableId) {
    HOOK_TRACE_PROFILE("cudaGetExportTable");
    using func_ptr = cudaError_t (*)(const void **, const cudaUUID_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetExportTable"));
    HOOK_CHECK(func_entry);
    return func_entry(ppExportTable, pExportTableId);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr, const void *symbolPtr) {
    HOOK_TRACE_PROFILE("cudaGetFuncBySymbol");
    using func_ptr = cudaError_t (*)(cudaFunction_t *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetFuncBySymbol"));
    HOOK_CHECK(func_entry);
    return func_entry(functionPtr, symbolPtr);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaSignalExternalSemaphoresAsync_ptsz(
    const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams_v1 *paramsArray,
    unsigned int numExtSems, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaSignalExternalSemaphoresAsync_ptsz");
    using func_ptr = cudaError_t (*)(const cudaExternalSemaphore_t *,
                                     const struct cudaExternalSemaphoreSignalParams_v1 *, unsigned int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSignalExternalSemaphoresAsync_ptsz"));
    HOOK_CHECK(func_entry);
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaSignalExternalSemaphoresAsync_v2(
    const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaSignalExternalSemaphoresAsync_v2");
    using func_ptr = cudaError_t (*)(const cudaExternalSemaphore_t *, const struct cudaExternalSemaphoreSignalParams *,
                                     unsigned int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSignalExternalSemaphoresAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaWaitExternalSemaphoresAsync_ptsz(
    const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams_v1 *paramsArray,
    unsigned int numExtSems, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaWaitExternalSemaphoresAsync_ptsz");
    using func_ptr = cudaError_t (*)(const cudaExternalSemaphore_t *, const struct cudaExternalSemaphoreWaitParams_v1 *,
                                     unsigned int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaWaitExternalSemaphoresAsync_ptsz"));
    HOOK_CHECK(func_entry);
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaWaitExternalSemaphoresAsync_v2(
    const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray,
    unsigned int numExtSems, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("cudaWaitExternalSemaphoresAsync_v2");
    using func_ptr = cudaError_t (*)(const cudaExternalSemaphore_t *, const struct cudaExternalSemaphoreWaitParams *,
                                     unsigned int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaWaitExternalSemaphoresAsync_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(extSemArray, paramsArray, numExtSems, stream);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t cudaStreamUpdateCaptureDependencies_ptsz(cudaStream_t stream,
                                                                                 cudaGraphNode_t *dependencies,
                                                                                 size_t numDependencies,
                                                                                 unsigned int flags) {
    HOOK_TRACE_PROFILE("cudaStreamUpdateCaptureDependencies_ptsz");
    using func_ptr = cudaError_t (*)(cudaStream_t, cudaGraphNode_t *, size_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaStreamUpdateCaptureDependencies_ptsz"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, dependencies, numDependencies, flags);
}

HOOK_C_API HOOK_DECL_EXPORT void **__cudaRegisterFatBinary(void *fatCubin) {
    HOOK_TRACE_PROFILE("__cudaRegisterFatBinary");
    using func_ptr = void **(*)(void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterFatBinary"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubin);
}

HOOK_C_API HOOK_DECL_EXPORT void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    HOOK_TRACE_PROFILE("__cudaRegisterFatBinaryEnd");
    using func_ptr = void (*)(void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterFatBinaryEnd"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubinHandle);
}

HOOK_C_API HOOK_DECL_EXPORT void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    HOOK_TRACE_PROFILE("__cudaUnregisterFatBinary");
    using func_ptr = void (*)(void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaUnregisterFatBinary"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubinHandle);
}

HOOK_C_API HOOK_DECL_EXPORT void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                                                   const char *deviceName, int ext, size_t size, int constant,
                                                   int global) {
    HOOK_TRACE_PROFILE("__cudaRegisterVar");
    using func_ptr = void (*)(void **, char *, char *, const char *, int, size_t, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterVar"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
}

HOOK_C_API HOOK_DECL_EXPORT void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                                                          char *deviceAddress, const char *deviceName, int ext,
                                                          size_t size, int constant, int global) {
    HOOK_TRACE_PROFILE("__cudaRegisterManagedVar");
    using func_ptr = void (*)(void **, void **, char *, const char *, int, size_t, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterManagedVar"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubinHandle, hostVarPtrAddress, deviceAddress, deviceName, ext, size, constant, global);
}

HOOK_C_API HOOK_DECL_EXPORT char __cudaInitModule(void **fatCubinHandle) {
    HOOK_TRACE_PROFILE("__cudaInitModule");
    using func_ptr = char (*)(void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaInitModule"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubinHandle);
}

HOOK_C_API HOOK_DECL_EXPORT void __cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar,
                                                       const void **deviceAddress, const char *deviceName, int dim,
                                                       int norm, int ext) {
    HOOK_TRACE_PROFILE("__cudaRegisterTexture");
    using func_ptr = void (*)(void **, const struct textureReference *, const void **, const char *, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterTexture"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext);
}

HOOK_C_API HOOK_DECL_EXPORT void __cudaRegisterSurface(void **fatCubinHandle, const struct surfaceReference *hostVar,
                                                       const void **deviceAddress, const char *deviceName, int dim,
                                                       int ext) {
    HOOK_TRACE_PROFILE("__cudaRegisterSurface");
    using func_ptr = void (*)(void **, const struct surfaceReference *, const void **, const char *, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterSurface"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, ext);
}

HOOK_C_API HOOK_DECL_EXPORT void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                                                        const char *deviceName, int thread_limit, uint3 *tid,
                                                        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    HOOK_TRACE_PROFILE("__cudaRegisterFunction");
    using func_ptr =
        void (*)(void **, const char *, char *, const char *, int, uint3 *, uint3 *, dim3 *, dim3 *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterFunction"));
    HOOK_CHECK(func_entry);
    return func_entry(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

HOOK_C_API HOOK_DECL_EXPORT cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem,
                                                                   void *stream) {
    HOOK_TRACE_PROFILE("__cudaPopCallConfiguration");
    using func_ptr = cudaError_t (*)(dim3 *, dim3 *, size_t *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaPopCallConfiguration"));
    HOOK_CHECK(func_entry);
    return func_entry(gridDim, blockDim, sharedMem, stream);
}

HOOK_C_API HOOK_DECL_EXPORT unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                                                 struct CUstream_st *stream) {
    HOOK_TRACE_PROFILE("__cudaPushCallConfiguration");
    using func_ptr = unsigned (*)(dim3, dim3, size_t, struct CUstream_st *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaPushCallConfiguration"));
    HOOK_CHECK(func_entry);
    return func_entry(gridDim, blockDim, sharedMem, stream);
}
