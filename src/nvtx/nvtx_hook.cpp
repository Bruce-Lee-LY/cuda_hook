// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 64 apis

#include "cuda_subset.h"
#include "cudart_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "nvtx_subset.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT int nvtxInitialize(const nvtxInitializationAttributes_t *initAttrib) {
    HOOK_TRACE_PROFILE("nvtxInitialize");
    using func_ptr = int (*)(const nvtxInitializationAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxInitialize"));
    HOOK_CHECK(func_entry);
    return func_entry(initAttrib);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainMarkEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t *eventAttrib) {
    HOOK_TRACE_PROFILE("nvtxDomainMarkEx");
    using func_ptr = void (*)(nvtxDomainHandle_t, const nvtxEventAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainMarkEx"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, eventAttrib);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxMarkEx(const nvtxEventAttributes_t *eventAttrib) {
    HOOK_TRACE_PROFILE("nvtxMarkEx");
    using func_ptr = void (*)(const nvtxEventAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxMarkEx"));
    HOOK_CHECK(func_entry);
    return func_entry(eventAttrib);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxMarkA(const char *message) {
    HOOK_TRACE_PROFILE("nvtxMarkA");
    using func_ptr = void (*)(const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxMarkA"));
    HOOK_CHECK(func_entry);
    return func_entry(message);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxMarkW(const wchar_t *message) {
    HOOK_TRACE_PROFILE("nvtxMarkW");
    using func_ptr = void (*)(const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxMarkW"));
    HOOK_CHECK(func_entry);
    return func_entry(message);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxRangeId_t nvtxDomainRangeStartEx(nvtxDomainHandle_t domain,
                                                                 const nvtxEventAttributes_t *eventAttrib) {
    HOOK_TRACE_PROFILE("nvtxDomainRangeStartEx");
    using func_ptr = nvtxRangeId_t (*)(nvtxDomainHandle_t, const nvtxEventAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainRangeStartEx"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, eventAttrib);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxRangeId_t nvtxRangeStartEx(const nvtxEventAttributes_t *eventAttrib) {
    HOOK_TRACE_PROFILE("nvtxRangeStartEx");
    using func_ptr = nvtxRangeId_t (*)(const nvtxEventAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxRangeStartEx"));
    HOOK_CHECK(func_entry);
    return func_entry(eventAttrib);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxRangeId_t nvtxRangeStartA(const char *message) {
    HOOK_TRACE_PROFILE("nvtxRangeStartA");
    using func_ptr = nvtxRangeId_t (*)(const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxRangeStartA"));
    HOOK_CHECK(func_entry);
    return func_entry(message);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxRangeId_t nvtxRangeStartW(const wchar_t *message) {
    HOOK_TRACE_PROFILE("nvtxRangeStartW");
    using func_ptr = nvtxRangeId_t (*)(const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxRangeStartW"));
    HOOK_CHECK(func_entry);
    return func_entry(message);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainRangeEnd(nvtxDomainHandle_t domain, nvtxRangeId_t id) {
    HOOK_TRACE_PROFILE("nvtxDomainRangeEnd");
    using func_ptr = void (*)(nvtxDomainHandle_t, nvtxRangeId_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainRangeEnd"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, id);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxRangeEnd(nvtxRangeId_t id) {
    HOOK_TRACE_PROFILE("nvtxRangeEnd");
    using func_ptr = void (*)(nvtxRangeId_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxRangeEnd"));
    HOOK_CHECK(func_entry);
    return func_entry(id);
}

HOOK_C_API HOOK_DECL_EXPORT int nvtxDomainRangePushEx(nvtxDomainHandle_t domain,
                                                      const nvtxEventAttributes_t *eventAttrib) {
    HOOK_TRACE_PROFILE("nvtxDomainRangePushEx");
    using func_ptr = int (*)(nvtxDomainHandle_t, const nvtxEventAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainRangePushEx"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, eventAttrib);
}

HOOK_C_API HOOK_DECL_EXPORT int nvtxRangePushEx(const nvtxEventAttributes_t *eventAttrib) {
    HOOK_TRACE_PROFILE("nvtxRangePushEx");
    using func_ptr = int (*)(const nvtxEventAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxRangePushEx"));
    HOOK_CHECK(func_entry);
    return func_entry(eventAttrib);
}

HOOK_C_API HOOK_DECL_EXPORT int nvtxRangePushA(const char *message) {
    HOOK_TRACE_PROFILE("nvtxRangePushA");
    using func_ptr = int (*)(const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxRangePushA"));
    HOOK_CHECK(func_entry);
    return func_entry(message);
}

HOOK_C_API HOOK_DECL_EXPORT int nvtxRangePushW(const wchar_t *message) {
    HOOK_TRACE_PROFILE("nvtxRangePushW");
    using func_ptr = int (*)(const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxRangePushW"));
    HOOK_CHECK(func_entry);
    return func_entry(message);
}

HOOK_C_API HOOK_DECL_EXPORT int nvtxDomainRangePop(nvtxDomainHandle_t domain) {
    HOOK_TRACE_PROFILE("nvtxDomainRangePop");
    using func_ptr = int (*)(nvtxDomainHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainRangePop"));
    HOOK_CHECK(func_entry);
    return func_entry(domain);
}

HOOK_C_API HOOK_DECL_EXPORT int nvtxRangePop() {
    HOOK_TRACE_PROFILE("nvtxRangePop");
    using func_ptr = int (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxRangePop"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT nvtxResourceHandle_t nvtxDomainResourceCreate(nvtxDomainHandle_t domain,
                                                                          nvtxResourceAttributes_t *attribs) {
    HOOK_TRACE_PROFILE("nvtxDomainResourceCreate");
    using func_ptr = nvtxResourceHandle_t (*)(nvtxDomainHandle_t, nvtxResourceAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainResourceCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, attribs);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainResourceDestroy(nvtxResourceHandle_t resource) {
    HOOK_TRACE_PROFILE("nvtxDomainResourceDestroy");
    using func_ptr = void (*)(nvtxResourceHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainResourceDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(resource);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainNameCategoryA(nvtxDomainHandle_t domain, uint32_t category,
                                                         const char *name) {
    HOOK_TRACE_PROFILE("nvtxDomainNameCategoryA");
    using func_ptr = void (*)(nvtxDomainHandle_t, uint32_t, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainNameCategoryA"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, category, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainNameCategoryW(nvtxDomainHandle_t domain, uint32_t category,
                                                         const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxDomainNameCategoryW");
    using func_ptr = void (*)(nvtxDomainHandle_t, uint32_t, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainNameCategoryW"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, category, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCategoryA(uint32_t category, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameCategoryA");
    using func_ptr = void (*)(uint32_t, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCategoryA"));
    HOOK_CHECK(func_entry);
    return func_entry(category, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCategoryW(uint32_t category, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameCategoryW");
    using func_ptr = void (*)(uint32_t, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCategoryW"));
    HOOK_CHECK(func_entry);
    return func_entry(category, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameOsThreadA(uint32_t threadId, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameOsThreadA");
    using func_ptr = void (*)(uint32_t, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameOsThreadA"));
    HOOK_CHECK(func_entry);
    return func_entry(threadId, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameOsThreadW(uint32_t threadId, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameOsThreadW");
    using func_ptr = void (*)(uint32_t, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameOsThreadW"));
    HOOK_CHECK(func_entry);
    return func_entry(threadId, name);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t domain,
                                                                         const char *string) {
    HOOK_TRACE_PROFILE("nvtxDomainRegisterStringA");
    using func_ptr = nvtxStringHandle_t (*)(nvtxDomainHandle_t, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainRegisterStringA"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, string);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxStringHandle_t nvtxDomainRegisterStringW(nvtxDomainHandle_t domain,
                                                                         const wchar_t *string) {
    HOOK_TRACE_PROFILE("nvtxDomainRegisterStringW");
    using func_ptr = nvtxStringHandle_t (*)(nvtxDomainHandle_t, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainRegisterStringW"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, string);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxDomainHandle_t nvtxDomainCreateA(const char *name) {
    HOOK_TRACE_PROFILE("nvtxDomainCreateA");
    using func_ptr = nvtxDomainHandle_t (*)(const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainCreateA"));
    HOOK_CHECK(func_entry);
    return func_entry(name);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxDomainHandle_t nvtxDomainCreateW(const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxDomainCreateW");
    using func_ptr = nvtxDomainHandle_t (*)(const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainCreateW"));
    HOOK_CHECK(func_entry);
    return func_entry(name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainDestroy(nvtxDomainHandle_t domain) {
    HOOK_TRACE_PROFILE("nvtxDomainDestroy");
    using func_ptr = void (*)(nvtxDomainHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(domain);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCuDeviceA(CUdevice device, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameCuDeviceA");
    using func_ptr = void (*)(CUdevice, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCuDeviceA"));
    HOOK_CHECK(func_entry);
    return func_entry(device, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCuDeviceW(CUdevice device, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameCuDeviceW");
    using func_ptr = void (*)(CUdevice, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCuDeviceW"));
    HOOK_CHECK(func_entry);
    return func_entry(device, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCuContextA(CUcontext context, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameCuContextA");
    using func_ptr = void (*)(CUcontext, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCuContextA"));
    HOOK_CHECK(func_entry);
    return func_entry(context, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCuContextW(CUcontext context, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameCuContextW");
    using func_ptr = void (*)(CUcontext, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCuContextW"));
    HOOK_CHECK(func_entry);
    return func_entry(context, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCuStreamA(CUstream stream, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameCuStreamA");
    using func_ptr = void (*)(CUstream, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCuStreamA"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCuStreamW(CUstream stream, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameCuStreamW");
    using func_ptr = void (*)(CUstream, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCuStreamW"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCuEventA(CUevent event, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameCuEventA");
    using func_ptr = void (*)(CUevent, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCuEventA"));
    HOOK_CHECK(func_entry);
    return func_entry(event, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCuEventW(CUevent event, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameCuEventW");
    using func_ptr = void (*)(CUevent, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCuEventW"));
    HOOK_CHECK(func_entry);
    return func_entry(event, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCudaDeviceA(int device, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameCudaDeviceA");
    using func_ptr = void (*)(int, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCudaDeviceA"));
    HOOK_CHECK(func_entry);
    return func_entry(device, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCudaDeviceW(int device, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameCudaDeviceW");
    using func_ptr = void (*)(int, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCudaDeviceW"));
    HOOK_CHECK(func_entry);
    return func_entry(device, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCudaStreamA(cudaStream_t stream, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameCudaStreamA");
    using func_ptr = void (*)(cudaStream_t, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCudaStreamA"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCudaStreamW(cudaStream_t stream, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameCudaStreamW");
    using func_ptr = void (*)(cudaStream_t, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCudaStreamW"));
    HOOK_CHECK(func_entry);
    return func_entry(stream, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCudaEventA(cudaEvent_t event, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameCudaEventA");
    using func_ptr = void (*)(cudaEvent_t, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCudaEventA"));
    HOOK_CHECK(func_entry);
    return func_entry(event, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameCudaEventW(cudaEvent_t event, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameCudaEventW");
    using func_ptr = void (*)(cudaEvent_t, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameCudaEventW"));
    HOOK_CHECK(func_entry);
    return func_entry(event, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClDeviceA(cl_device_id device, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameClDeviceA");
    using func_ptr = void (*)(cl_device_id, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClDeviceA"));
    HOOK_CHECK(func_entry);
    return func_entry(device, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClDeviceW(cl_device_id device, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameClDeviceW");
    using func_ptr = void (*)(cl_device_id, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClDeviceW"));
    HOOK_CHECK(func_entry);
    return func_entry(device, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClContextA(cl_context context, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameClContextA");
    using func_ptr = void (*)(cl_context, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClContextA"));
    HOOK_CHECK(func_entry);
    return func_entry(context, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClContextW(cl_context context, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameClContextW");
    using func_ptr = void (*)(cl_context, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClContextW"));
    HOOK_CHECK(func_entry);
    return func_entry(context, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClCommandQueueA(cl_command_queue command_queue, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameClCommandQueueA");
    using func_ptr = void (*)(cl_command_queue, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClCommandQueueA"));
    HOOK_CHECK(func_entry);
    return func_entry(command_queue, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClCommandQueueW(cl_command_queue command_queue, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameClCommandQueueW");
    using func_ptr = void (*)(cl_command_queue, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClCommandQueueW"));
    HOOK_CHECK(func_entry);
    return func_entry(command_queue, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClMemObjectA(cl_mem memobj, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameClMemObjectA");
    using func_ptr = void (*)(cl_mem, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClMemObjectA"));
    HOOK_CHECK(func_entry);
    return func_entry(memobj, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClMemObjectW(cl_mem memobj, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameClMemObjectW");
    using func_ptr = void (*)(cl_mem, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClMemObjectW"));
    HOOK_CHECK(func_entry);
    return func_entry(memobj, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClSamplerA(cl_sampler sampler, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameClSamplerA");
    using func_ptr = void (*)(cl_sampler, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClSamplerA"));
    HOOK_CHECK(func_entry);
    return func_entry(sampler, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClSamplerW(cl_sampler sampler, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameClSamplerW");
    using func_ptr = void (*)(cl_sampler, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClSamplerW"));
    HOOK_CHECK(func_entry);
    return func_entry(sampler, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClProgramA(cl_program program, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameClProgramA");
    using func_ptr = void (*)(cl_program, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClProgramA"));
    HOOK_CHECK(func_entry);
    return func_entry(program, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClProgramW(cl_program program, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameClProgramW");
    using func_ptr = void (*)(cl_program, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClProgramW"));
    HOOK_CHECK(func_entry);
    return func_entry(program, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClEventA(cl_event evnt, const char *name) {
    HOOK_TRACE_PROFILE("nvtxNameClEventA");
    using func_ptr = void (*)(cl_event, const char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClEventA"));
    HOOK_CHECK(func_entry);
    return func_entry(evnt, name);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxNameClEventW(cl_event evnt, const wchar_t *name) {
    HOOK_TRACE_PROFILE("nvtxNameClEventW");
    using func_ptr = void (*)(cl_event, const wchar_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxNameClEventW"));
    HOOK_CHECK(func_entry);
    return func_entry(evnt, name);
}

HOOK_C_API HOOK_DECL_EXPORT nvtxSyncUser_t nvtxDomainSyncUserCreate(nvtxDomainHandle_t domain,
                                                                    const nvtxSyncUserAttributes_t *attribs) {
    HOOK_TRACE_PROFILE("nvtxDomainSyncUserCreate");
    using func_ptr = nvtxSyncUser_t (*)(nvtxDomainHandle_t, const nvtxSyncUserAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainSyncUserCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(domain, attribs);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainSyncUserDestroy(nvtxSyncUser_t handle) {
    HOOK_TRACE_PROFILE("nvtxDomainSyncUserDestroy");
    using func_ptr = void (*)(nvtxSyncUser_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainSyncUserDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainSyncUserAcquireStart(nvtxSyncUser_t handle) {
    HOOK_TRACE_PROFILE("nvtxDomainSyncUserAcquireStart");
    using func_ptr = void (*)(nvtxSyncUser_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainSyncUserAcquireStart"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainSyncUserAcquireFailed(nvtxSyncUser_t handle) {
    HOOK_TRACE_PROFILE("nvtxDomainSyncUserAcquireFailed");
    using func_ptr = void (*)(nvtxSyncUser_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainSyncUserAcquireFailed"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainSyncUserAcquireSuccess(nvtxSyncUser_t handle) {
    HOOK_TRACE_PROFILE("nvtxDomainSyncUserAcquireSuccess");
    using func_ptr = void (*)(nvtxSyncUser_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainSyncUserAcquireSuccess"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT void nvtxDomainSyncUserReleasing(nvtxSyncUser_t handle) {
    HOOK_TRACE_PROFILE("nvtxDomainSyncUserReleasing");
    using func_ptr = void (*)(nvtxSyncUser_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVTX_SYMBOL("nvtxDomainSyncUserReleasing"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}
