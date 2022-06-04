// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 18 apis

#include "hook.h"
#include "macro_common.h"
#include "nvrtc_subset.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT const char *nvrtcGetErrorString(nvrtcResult result) {
    HOOK_TRACE_PROFILE("nvrtcGetErrorString");
    using func_ptr = const char *(*)(nvrtcResult);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetErrorString"));
    HOOK_CHECK(func_entry);
    return func_entry(result);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcVersion(int *major, int *minor) {
    HOOK_TRACE_PROFILE("nvrtcVersion");
    using func_ptr = nvrtcResult (*)(int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(major, minor);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetNumSupportedArchs(int *numArchs) {
    HOOK_TRACE_PROFILE("nvrtcGetNumSupportedArchs");
    using func_ptr = nvrtcResult (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetNumSupportedArchs"));
    HOOK_CHECK(func_entry);
    return func_entry(numArchs);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetSupportedArchs(int *supportedArchs) {
    HOOK_TRACE_PROFILE("nvrtcGetSupportedArchs");
    using func_ptr = nvrtcResult (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetSupportedArchs"));
    HOOK_CHECK(func_entry);
    return func_entry(supportedArchs);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src, const char *name,
                                                           int numHeaders, const char *const *headers,
                                                           const char *const *includeNames) {
    HOOK_TRACE_PROFILE("nvrtcCreateProgram");
    using func_ptr =
        nvrtcResult (*)(nvrtcProgram *, const char *, const char *, int, const char *const *, const char *const *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcCreateProgram"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, src, name, numHeaders, headers, includeNames);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog) {
    HOOK_TRACE_PROFILE("nvrtcDestroyProgram");
    using func_ptr = nvrtcResult (*)(nvrtcProgram *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcDestroyProgram"));
    HOOK_CHECK(func_entry);
    return func_entry(prog);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                                                            const char *const *options) {
    HOOK_TRACE_PROFILE("nvrtcCompileProgram");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, int, const char *const *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcCompileProgram"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, numOptions, options);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet) {
    HOOK_TRACE_PROFILE("nvrtcGetPTXSize");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetPTXSize"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, ptxSizeRet);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx) {
    HOOK_TRACE_PROFILE("nvrtcGetPTX");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetPTX"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, ptx);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t *cubinSizeRet) {
    HOOK_TRACE_PROFILE("nvrtcGetCUBINSize");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetCUBINSize"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, cubinSizeRet);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin) {
    HOOK_TRACE_PROFILE("nvrtcGetCUBIN");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetCUBIN"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, cubin);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, size_t *nvvmSizeRet) {
    HOOK_TRACE_PROFILE("nvrtcGetNVVMSize");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetNVVMSize"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, nvvmSizeRet);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char *nvvm) {
    HOOK_TRACE_PROFILE("nvrtcGetNVVM");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetNVVM"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, nvvm);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet) {
    HOOK_TRACE_PROFILE("nvrtcGetProgramLogSize");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetProgramLogSize"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, logSizeRet);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log) {
    HOOK_TRACE_PROFILE("nvrtcGetProgramLog");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, char *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetProgramLog"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, log);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char *const name_expression) {
    HOOK_TRACE_PROFILE("nvrtcAddNameExpression");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, const char *const);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcAddNameExpression"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, name_expression);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog, const char *const name_expression,
                                                            const char **lowered_name) {
    HOOK_TRACE_PROFILE("nvrtcGetLoweredName");
    using func_ptr = nvrtcResult (*)(nvrtcProgram, const char *const, const char **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetLoweredName"));
    HOOK_CHECK(func_entry);
    return func_entry(prog, name_expression, lowered_name);
}

HOOK_C_API HOOK_DECL_EXPORT nvrtcResult nvrtcGetTypeName(const std::type_info &tinfo, std::string *result) {
    HOOK_TRACE_PROFILE("nvrtcGetTypeName");
    using func_ptr = nvrtcResult (*)(const std::type_info &, std::string *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVRTC_SYMBOL("nvrtcGetTypeName"));
    HOOK_CHECK(func_entry);
    return func_entry(tinfo, result);
}
