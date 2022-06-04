// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 251 apis

#include "hook.h"
#include "macro_common.h"
#include "nvml_subset.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlInit_v2() {
    HOOK_TRACE_PROFILE("nvmlInit_v2");
    using func_ptr = nvmlReturn_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlInit_v2"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
    HOOK_TRACE_PROFILE("nvmlInitWithFlags");
    using func_ptr = nvmlReturn_t (*)(unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlInitWithFlags"));
    HOOK_CHECK(func_entry);
    return func_entry(flags);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlShutdown() {
    HOOK_TRACE_PROFILE("nvmlShutdown");
    using func_ptr = nvmlReturn_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlShutdown"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT const char *nvmlErrorString(nvmlReturn_t result) {
    HOOK_TRACE_PROFILE("nvmlErrorString");
    using func_ptr = const char *(*)(nvmlReturn_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlErrorString"));
    HOOK_CHECK(func_entry);
    return func_entry(result);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlSystemGetDriverVersion");
    using func_ptr = nvmlReturn_t (*)(char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlSystemGetDriverVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(version, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlSystemGetNVMLVersion");
    using func_ptr = nvmlReturn_t (*)(char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlSystemGetNVMLVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(version, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion) {
    HOOK_TRACE_PROFILE("nvmlSystemGetCudaDriverVersion");
    using func_ptr = nvmlReturn_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlSystemGetCudaDriverVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(cudaDriverVersion);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion) {
    HOOK_TRACE_PROFILE("nvmlSystemGetCudaDriverVersion_v2");
    using func_ptr = nvmlReturn_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlSystemGetCudaDriverVersion_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(cudaDriverVersion);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlSystemGetProcessName");
    using func_ptr = nvmlReturn_t (*)(unsigned int, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlSystemGetProcessName"));
    HOOK_CHECK(func_entry);
    return func_entry(pid, name, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount) {
    HOOK_TRACE_PROFILE("nvmlUnitGetCount");
    using func_ptr = nvmlReturn_t (*)(unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitGetCount"));
    HOOK_CHECK(func_entry);
    return func_entry(unitCount);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit) {
    HOOK_TRACE_PROFILE("nvmlUnitGetHandleByIndex");
    using func_ptr = nvmlReturn_t (*)(unsigned int, nvmlUnit_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitGetHandleByIndex"));
    HOOK_CHECK(func_entry);
    return func_entry(index, unit);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info) {
    HOOK_TRACE_PROFILE("nvmlUnitGetUnitInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlUnit_t, nvmlUnitInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitGetUnitInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(unit, info);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state) {
    HOOK_TRACE_PROFILE("nvmlUnitGetLedState");
    using func_ptr = nvmlReturn_t (*)(nvmlUnit_t, nvmlLedState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitGetLedState"));
    HOOK_CHECK(func_entry);
    return func_entry(unit, state);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu) {
    HOOK_TRACE_PROFILE("nvmlUnitGetPsuInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlUnit_t, nvmlPSUInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitGetPsuInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(unit, psu);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type,
                                                                unsigned int *temp) {
    HOOK_TRACE_PROFILE("nvmlUnitGetTemperature");
    using func_ptr = nvmlReturn_t (*)(nvmlUnit_t, unsigned int, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitGetTemperature"));
    HOOK_CHECK(func_entry);
    return func_entry(unit, type, temp);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t *fanSpeeds) {
    HOOK_TRACE_PROFILE("nvmlUnitGetFanSpeedInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlUnit_t, nvmlUnitFanSpeeds_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitGetFanSpeedInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(unit, fanSpeeds);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount,
                                                            nvmlDevice_t *devices) {
    HOOK_TRACE_PROFILE("nvmlUnitGetDevices");
    using func_ptr = nvmlReturn_t (*)(nvmlUnit_t, unsigned int *, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitGetDevices"));
    HOOK_CHECK(func_entry);
    return func_entry(unit, deviceCount, devices);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount,
                                                                 nvmlHwbcEntry_t *hwbcEntries) {
    HOOK_TRACE_PROFILE("nvmlSystemGetHicVersion");
    using func_ptr = nvmlReturn_t (*)(unsigned int *, nvmlHwbcEntry_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlSystemGetHicVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(hwbcCount, hwbcEntries);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCount_v2");
    using func_ptr = nvmlReturn_t (*)(unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCount_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(deviceCount);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device,
                                                                    nvmlDeviceAttributes_t *attributes) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetAttributes_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetAttributes_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, attributes);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetHandleByIndex_v2");
    using func_ptr = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetHandleByIndex_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(index, device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetHandleBySerial(const char *serial, nvmlDevice_t *device) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetHandleBySerial");
    using func_ptr = nvmlReturn_t (*)(const char *, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetHandleBySerial"));
    HOOK_CHECK(func_entry);
    return func_entry(serial, device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetHandleByUUID");
    using func_ptr = nvmlReturn_t (*)(const char *, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetHandleByUUID"));
    HOOK_CHECK(func_entry);
    return func_entry(uuid, device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetHandleByPciBusId_v2");
    using func_ptr = nvmlReturn_t (*)(const char *, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetHandleByPciBusId_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pciBusId, device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetName");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetName"));
    HOOK_CHECK(func_entry);
    return func_entry(device, name, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetBrand");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlBrandType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetBrand"));
    HOOK_CHECK(func_entry);
    return func_entry(device, type);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetIndex");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetIndex"));
    HOOK_CHECK(func_entry);
    return func_entry(device, index);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial, unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetSerial");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetSerial"));
    HOOK_CHECK(func_entry);
    return func_entry(device, serial, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize,
                                                                     unsigned long *nodeSet,
                                                                     nvmlAffinityScope_t scope) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMemoryAffinity");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned long *, nvmlAffinityScope_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMemoryAffinity"));
    HOOK_CHECK(func_entry);
    return func_entry(device, nodeSetSize, nodeSet, scope);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device,
                                                                             unsigned int cpuSetSize,
                                                                             unsigned long *cpuSet,
                                                                             nvmlAffinityScope_t scope) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCpuAffinityWithinScope");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned long *, nvmlAffinityScope_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCpuAffinityWithinScope"));
    HOOK_CHECK(func_entry);
    return func_entry(device, cpuSetSize, cpuSet, scope);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize,
                                                                  unsigned long *cpuSet) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCpuAffinity");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCpuAffinity"));
    HOOK_CHECK(func_entry);
    return func_entry(device, cpuSetSize, cpuSet);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetCpuAffinity");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetCpuAffinity"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
    HOOK_TRACE_PROFILE("nvmlDeviceClearCpuAffinity");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceClearCpuAffinity"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2,
                                                                             nvmlGpuTopologyLevel_t *pathInfo) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetTopologyCommonAncestor");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, nvmlGpuTopologyLevel_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetTopologyCommonAncestor"));
    HOOK_CHECK(func_entry);
    return func_entry(device1, device2, pathInfo);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device,
                                                                          nvmlGpuTopologyLevel_t level,
                                                                          unsigned int *count,
                                                                          nvmlDevice_t *deviceArray) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetTopologyNearestGpus");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuTopologyLevel_t, unsigned int *, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetTopologyNearestGpus"));
    HOOK_CHECK(func_entry);
    return func_entry(device, level, count, deviceArray);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count,
                                                                     nvmlDevice_t *deviceArray) {
    HOOK_TRACE_PROFILE("nvmlSystemGetTopologyGpuSet");
    using func_ptr = nvmlReturn_t (*)(unsigned int, unsigned int *, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlSystemGetTopologyGpuSet"));
    HOOK_CHECK(func_entry);
    return func_entry(cpuNumber, count, deviceArray);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2,
                                                                nvmlGpuP2PCapsIndex_t p2pIndex,
                                                                nvmlGpuP2PStatus_t *p2pStatus) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetP2PStatus");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t, nvmlGpuP2PStatus_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetP2PStatus"));
    HOOK_CHECK(func_entry);
    return func_entry(device1, device2, p2pIndex, p2pStatus);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetUUID");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetUUID"));
    HOOK_CHECK(func_entry);
    return func_entry(device, uuid, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char *mdevUuid,
                                                                     unsigned int size) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetMdevUUID");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetMdevUUID"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, mdevUuid, size);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMinorNumber");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMinorNumber"));
    HOOK_CHECK(func_entry);
    return func_entry(device, minorNumber);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber,
                                                                      unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetBoardPartNumber");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetBoardPartNumber"));
    HOOK_CHECK(func_entry);
    return func_entry(device, partNumber, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object,
                                                                     char *version, unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetInforomVersion");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlInforomObject_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetInforomVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(device, object, version, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char *version,
                                                                          unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetInforomImageVersion");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetInforomImageVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(device, version, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device,
                                                                                   unsigned int *checksum) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetInforomConfigurationChecksum");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetInforomConfigurationChecksum"));
    HOOK_CHECK(func_entry);
    return func_entry(device, checksum);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
    HOOK_TRACE_PROFILE("nvmlDeviceValidateInforom");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceValidateInforom"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t *display) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetDisplayMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetDisplayMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, display);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t *isActive) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetDisplayActive");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetDisplayActive"));
    HOOK_CHECK(func_entry);
    return func_entry(device, isActive);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPersistenceMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPersistenceMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPciInfo_v3");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPciInfo_v3"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pci);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device,
                                                                            unsigned int *maxLinkGen) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMaxPcieLinkGeneration");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMaxPcieLinkGeneration"));
    HOOK_CHECK(func_entry);
    return func_entry(device, maxLinkGen);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device,
                                                                       unsigned int *maxLinkWidth) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMaxPcieLinkWidth");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMaxPcieLinkWidth"));
    HOOK_CHECK(func_entry);
    return func_entry(device, maxLinkWidth);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device,
                                                                             unsigned int *currLinkGen) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCurrPcieLinkGeneration");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCurrPcieLinkGeneration"));
    HOOK_CHECK(func_entry);
    return func_entry(device, currLinkGen);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device,
                                                                        unsigned int *currLinkWidth) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCurrPcieLinkWidth");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCurrPcieLinkWidth"));
    HOOK_CHECK(func_entry);
    return func_entry(device, currLinkWidth);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter,
                                                                     unsigned int *value) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPcieThroughput");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPcieUtilCounter_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPcieThroughput"));
    HOOK_CHECK(func_entry);
    return func_entry(device, counter, value);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int *value) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPcieReplayCounter");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPcieReplayCounter"));
    HOOK_CHECK(func_entry);
    return func_entry(device, value);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                                                unsigned int *clock) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetClockInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetClockInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(device, type, clock);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                                                   unsigned int *clock) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMaxClockInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMaxClockInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(device, type, clock);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType,
                                                                        unsigned int *clockMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetApplicationsClock");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetApplicationsClock"));
    HOOK_CHECK(func_entry);
    return func_entry(device, clockType, clockMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device,
                                                                               nvmlClockType_t clockType,
                                                                               unsigned int *clockMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetDefaultApplicationsClock");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetDefaultApplicationsClock"));
    HOOK_CHECK(func_entry);
    return func_entry(device, clockType, clockMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device) {
    HOOK_TRACE_PROFILE("nvmlDeviceResetApplicationsClocks");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceResetApplicationsClocks"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType,
                                                            nvmlClockId_t clockId, unsigned int *clockMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetClock");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetClock"));
    HOOK_CHECK(func_entry);
    return func_entry(device, clockType, clockId, clockMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device,
                                                                            nvmlClockType_t clockType,
                                                                            unsigned int *clockMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMaxCustomerBoostClock");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMaxCustomerBoostClock"));
    HOOK_CHECK(func_entry);
    return func_entry(device, clockType, clockMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int *count,
                                                                            unsigned int *clocksMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetSupportedMemoryClocks");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetSupportedMemoryClocks"));
    HOOK_CHECK(func_entry);
    return func_entry(device, count, clocksMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device,
                                                                              unsigned int memoryClockMHz,
                                                                              unsigned int *count,
                                                                              unsigned int *clocksMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetSupportedGraphicsClocks");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetSupportedGraphicsClocks"));
    HOOK_CHECK(func_entry);
    return func_entry(device, memoryClockMHz, count, clocksMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device,
                                                                               nvmlEnableState_t *isEnabled,
                                                                               nvmlEnableState_t *defaultIsEnabled) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetAutoBoostedClocksEnabled");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetAutoBoostedClocksEnabled"));
    HOOK_CHECK(func_entry);
    return func_entry(device, isEnabled, defaultIsEnabled);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device,
                                                                               nvmlEnableState_t enabled) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetAutoBoostedClocksEnabled");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetAutoBoostedClocksEnabled"));
    HOOK_CHECK(func_entry);
    return func_entry(device, enabled);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device,
                                                                                      nvmlEnableState_t enabled,
                                                                                      unsigned int flags) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetDefaultAutoBoostedClocksEnabled");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t, unsigned int);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetDefaultAutoBoostedClocksEnabled"));
    HOOK_CHECK(func_entry);
    return func_entry(device, enabled, flags);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetFanSpeed");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetFanSpeed"));
    HOOK_CHECK(func_entry);
    return func_entry(device, speed);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan,
                                                                  unsigned int *speed) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetFanSpeed_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetFanSpeed_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, fan, speed);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                                                  nvmlTemperatureSensors_t sensorType,
                                                                  unsigned int *temp) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetTemperature");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetTemperature"));
    HOOK_CHECK(func_entry);
    return func_entry(device, sensorType, temp);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device,
                                                                           nvmlTemperatureThresholds_t thresholdType,
                                                                           unsigned int *temp) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetTemperatureThreshold");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureThresholds_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetTemperatureThreshold"));
    HOOK_CHECK(func_entry);
    return func_entry(device, thresholdType, temp);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device,
                                                                           nvmlTemperatureThresholds_t thresholdType,
                                                                           int *temp) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetTemperatureThreshold");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureThresholds_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetTemperatureThreshold"));
    HOOK_CHECK(func_entry);
    return func_entry(device, thresholdType, temp);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t *pState) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPerformanceState");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPstates_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPerformanceState"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pState);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t
    nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long *clocksThrottleReasons) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCurrentClocksThrottleReasons");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCurrentClocksThrottleReasons"));
    HOOK_CHECK(func_entry);
    return func_entry(device, clocksThrottleReasons);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(
    nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetSupportedClocksThrottleReasons");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetSupportedClocksThrottleReasons"));
    HOOK_CHECK(func_entry);
    return func_entry(device, supportedClocksThrottleReasons);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t *pState) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPowerState");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPstates_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPowerState"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pState);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device,
                                                                          nvmlEnableState_t *mode) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPowerManagementMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPowerManagementMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int *limit) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPowerManagementLimit");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPowerManagementLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(device, limit);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device,
                                                                                      unsigned int *minLimit,
                                                                                      unsigned int *maxLimit) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPowerManagementLimitConstraints");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, unsigned int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPowerManagementLimitConstraints"));
    HOOK_CHECK(func_entry);
    return func_entry(device, minLimit, maxLimit);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device,
                                                                                  unsigned int *defaultLimit) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPowerManagementDefaultLimit");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPowerManagementDefaultLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(device, defaultLimit);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPowerUsage");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPowerUsage"));
    HOOK_CHECK(func_entry);
    return func_entry(device, power);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device,
                                                                             unsigned long long *energy) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetTotalEnergyConsumption");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetTotalEnergyConsumption"));
    HOOK_CHECK(func_entry);
    return func_entry(device, energy);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int *limit) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetEnforcedPowerLimit");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetEnforcedPowerLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(device, limit);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device,
                                                                       nvmlGpuOperationMode_t *current,
                                                                       nvmlGpuOperationMode_t *pending) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGpuOperationMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuOperationMode_t *, nvmlGpuOperationMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGpuOperationMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, current, pending);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMemoryInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMemoryInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(device, memory);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t *mode) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetComputeMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlComputeMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetComputeMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major,
                                                                            int *minor) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCudaComputeCapability");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCudaComputeCapability"));
    HOOK_CHECK(func_entry);
    return func_entry(device, major, minor);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t *current,
                                                              nvmlEnableState_t *pending) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetEccMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetEccMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, current, pending);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetBoardId");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetBoardId"));
    HOOK_CHECK(func_entry);
    return func_entry(device, boardId);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int *multiGpuBool) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMultiGpuBoard");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMultiGpuBoard"));
    HOOK_CHECK(func_entry);
    return func_entry(device, multiGpuBool);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device,
                                                                     nvmlMemoryErrorType_t errorType,
                                                                     nvmlEccCounterType_t counterType,
                                                                     unsigned long long *eccCounts) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetTotalEccErrors");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetTotalEccErrors"));
    HOOK_CHECK(func_entry);
    return func_entry(device, errorType, counterType, eccCounts);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device,
                                                                        nvmlMemoryErrorType_t errorType,
                                                                        nvmlEccCounterType_t counterType,
                                                                        nvmlEccErrorCounts_t *eccCounts) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetDetailedEccErrors");
    using func_ptr =
        nvmlReturn_t (*)(nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlEccErrorCounts_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetDetailedEccErrors"));
    HOOK_CHECK(func_entry);
    return func_entry(device, errorType, counterType, eccCounts);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device,
                                                                         nvmlMemoryErrorType_t errorType,
                                                                         nvmlEccCounterType_t counterType,
                                                                         nvmlMemoryLocation_t locationType,
                                                                         unsigned long long *count) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMemoryErrorCounter");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t, nvmlMemoryLocation_t,
                                      unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMemoryErrorCounter"));
    HOOK_CHECK(func_entry);
    return func_entry(device, errorType, counterType, locationType, count);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device,
                                                                       nvmlUtilization_t *utilization) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetUtilizationRates");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetUtilizationRates"));
    HOOK_CHECK(func_entry);
    return func_entry(device, utilization);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int *utilization,
                                                                         unsigned int *samplingPeriodUs) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetEncoderUtilization");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetEncoderUtilization"));
    HOOK_CHECK(func_entry);
    return func_entry(device, utilization, samplingPeriodUs);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device,
                                                                      nvmlEncoderType_t encoderQueryType,
                                                                      unsigned int *encoderCapacity) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetEncoderCapacity");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEncoderType_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetEncoderCapacity"));
    HOOK_CHECK(func_entry);
    return func_entry(device, encoderQueryType, encoderCapacity);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int *sessionCount,
                                                                   unsigned int *averageFps,
                                                                   unsigned int *averageLatency) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetEncoderStats");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetEncoderStats"));
    HOOK_CHECK(func_entry);
    return func_entry(device, sessionCount, averageFps, averageLatency);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount,
                                                                      nvmlEncoderSessionInfo_t *sessionInfos) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetEncoderSessions");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlEncoderSessionInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetEncoderSessions"));
    HOOK_CHECK(func_entry);
    return func_entry(device, sessionCount, sessionInfos);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int *utilization,
                                                                         unsigned int *samplingPeriodUs) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetDecoderUtilization");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetDecoderUtilization"));
    HOOK_CHECK(func_entry);
    return func_entry(device, utilization, samplingPeriodUs);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t *fbcStats) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetFBCStats");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlFBCStats_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetFBCStats"));
    HOOK_CHECK(func_entry);
    return func_entry(device, fbcStats);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int *sessionCount,
                                                                  nvmlFBCSessionInfo_t *sessionInfo) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetFBCSessions");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlFBCSessionInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetFBCSessions"));
    HOOK_CHECK(func_entry);
    return func_entry(device, sessionCount, sessionInfo);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t *current,
                                                                  nvmlDriverModel_t *pending) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetDriverModel");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDriverModel_t *, nvmlDriverModel_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetDriverModel"));
    HOOK_CHECK(func_entry);
    return func_entry(device, current, pending);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version,
                                                                   unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetVbiosVersion");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetVbiosVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(device, version, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device,
                                                                     nvmlBridgeChipHierarchy_t *bridgeHierarchy) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetBridgeChipInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlBridgeChipHierarchy_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetBridgeChipInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(device, bridgeHierarchy);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t device,
                                                                                 unsigned int *infoCount,
                                                                                 nvmlProcessInfo_t *infos) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetComputeRunningProcesses_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetComputeRunningProcesses_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, infoCount, infos);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t device,
                                                                                  unsigned int *infoCount,
                                                                                  nvmlProcessInfo_t *infos) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGraphicsRunningProcesses_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGraphicsRunningProcesses_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, infoCount, infos);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v2(nvmlDevice_t device,
                                                                                    unsigned int *infoCount,
                                                                                    nvmlProcessInfo_t *infos) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMPSComputeRunningProcesses_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMPSComputeRunningProcesses_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, infoCount, infos);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2,
                                                               int *onSameBoard) {
    HOOK_TRACE_PROFILE("nvmlDeviceOnSameBoard");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceOnSameBoard"));
    HOOK_CHECK(func_entry);
    return func_entry(device1, device2, onSameBoard);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType,
                                                                     nvmlEnableState_t *isRestricted) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetAPIRestriction");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetAPIRestriction"));
    HOOK_CHECK(func_entry);
    return func_entry(device, apiType, isRestricted);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type,
                                                              unsigned long long lastSeenTimeStamp,
                                                              nvmlValueType_t *sampleValType, unsigned int *sampleCount,
                                                              nvmlSample_t *samples) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetSamples");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlSamplingType_t, unsigned long long, nvmlValueType_t *,
                                      unsigned int *, nvmlSample_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetSamples"));
    HOOK_CHECK(func_entry);
    return func_entry(device, type, lastSeenTimeStamp, sampleValType, sampleCount, samples);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device,
                                                                     nvmlBAR1Memory_t *bar1Memory) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetBAR1MemoryInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlBAR1Memory_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetBAR1MemoryInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(device, bar1Memory);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device,
                                                                      nvmlPerfPolicyType_t perfPolicyType,
                                                                      nvmlViolationTime_t *violTime) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetViolationStatus");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPerfPolicyType_t, nvmlViolationTime_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetViolationStatus"));
    HOOK_CHECK(func_entry);
    return func_entry(device, perfPolicyType, violTime);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t *mode) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetAccountingMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetAccountingMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid,
                                                                      nvmlAccountingStats_t *stats) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetAccountingStats");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlAccountingStats_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetAccountingStats"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pid, stats);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int *count,
                                                                     unsigned int *pids) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetAccountingPids");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetAccountingPids"));
    HOOK_CHECK(func_entry);
    return func_entry(device, count, pids);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device,
                                                                           unsigned int *bufferSize) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetAccountingBufferSize");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetAccountingBufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(device, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause,
                                                                   unsigned int *pageCount,
                                                                   unsigned long long *addresses) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetRetiredPages");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPageRetirementCause_t, unsigned int *, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetRetiredPages"));
    HOOK_CHECK(func_entry);
    return func_entry(device, cause, pageCount, addresses);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device,
                                                                      nvmlPageRetirementCause_t cause,
                                                                      unsigned int *pageCount,
                                                                      unsigned long long *addresses,
                                                                      unsigned long long *timestamps) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetRetiredPages_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPageRetirementCause_t, unsigned int *, unsigned long long *,
                                      unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetRetiredPages_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, cause, pageCount, addresses, timestamps);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device,
                                                                                nvmlEnableState_t *isPending) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetRetiredPagesPendingStatus");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetRetiredPagesPendingStatus"));
    HOOK_CHECK(func_entry);
    return func_entry(device, isPending);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int *corrRows,
                                                                   unsigned int *uncRows, unsigned int *isPending,
                                                                   unsigned int *failureOccurred) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetRemappedRows");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, unsigned int *, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetRemappedRows"));
    HOOK_CHECK(func_entry);
    return func_entry(device, corrRows, uncRows, isPending, failureOccurred);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device,
                                                                           nvmlRowRemapperHistogramValues_t *values) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetRowRemapperHistogram");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlRowRemapperHistogramValues_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetRowRemapperHistogram"));
    HOOK_CHECK(func_entry);
    return func_entry(device, values);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device,
                                                                   nvmlDeviceArchitecture_t *arch) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetArchitecture");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceArchitecture_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetArchitecture"));
    HOOK_CHECK(func_entry);
    return func_entry(device, arch);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) {
    HOOK_TRACE_PROFILE("nvmlUnitSetLedState");
    using func_ptr = nvmlReturn_t (*)(nvmlUnit_t, nvmlLedColor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlUnitSetLedState"));
    HOOK_CHECK(func_entry);
    return func_entry(unit, color);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetPersistenceMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetPersistenceMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetComputeMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlComputeMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetComputeMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetEccMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetEccMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, ecc);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device,
                                                                       nvmlEccCounterType_t counterType) {
    HOOK_TRACE_PROFILE("nvmlDeviceClearEccErrorCounts");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEccCounterType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceClearEccErrorCounts"));
    HOOK_CHECK(func_entry);
    return func_entry(device, counterType);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel,
                                                                  unsigned int flags) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetDriverModel");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDriverModel_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetDriverModel"));
    HOOK_CHECK(func_entry);
    return func_entry(device, driverModel, flags);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz,
                                                                      unsigned int maxGpuClockMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetGpuLockedClocks");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetGpuLockedClocks"));
    HOOK_CHECK(func_entry);
    return func_entry(device, minGpuClockMHz, maxGpuClockMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
    HOOK_TRACE_PROFILE("nvmlDeviceResetGpuLockedClocks");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceResetGpuLockedClocks"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device,
                                                                         unsigned int minMemClockMHz,
                                                                         unsigned int maxMemClockMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetMemoryLockedClocks");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetMemoryLockedClocks"));
    HOOK_CHECK(func_entry);
    return func_entry(device, minMemClockMHz, maxMemClockMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) {
    HOOK_TRACE_PROFILE("nvmlDeviceResetMemoryLockedClocks");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceResetMemoryLockedClocks"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz,
                                                                         unsigned int graphicsClockMHz) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetApplicationsClocks");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetApplicationsClocks"));
    HOOK_CHECK(func_entry);
    return func_entry(device, memClockMHz, graphicsClockMHz);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t *status) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetClkMonStatus");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlClkMonStatus_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetClkMonStatus"));
    HOOK_CHECK(func_entry);
    return func_entry(device, status);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetPowerManagementLimit");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetPowerManagementLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(device, limit);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device,
                                                                       nvmlGpuOperationMode_t mode) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetGpuOperationMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuOperationMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetGpuOperationMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType,
                                                                     nvmlEnableState_t isRestricted) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetAPIRestriction");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetAPIRestriction"));
    HOOK_CHECK(func_entry);
    return func_entry(device, apiType, isRestricted);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetAccountingMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlEnableState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetAccountingMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) {
    HOOK_TRACE_PROFILE("nvmlDeviceClearAccountingPids");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceClearAccountingPids"));
    HOOK_CHECK(func_entry);
    return func_entry(device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link,
                                                                  nvmlEnableState_t *isActive) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkState");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkState"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, isActive);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link,
                                                                    unsigned int *version) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkVersion");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, version);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                                                       nvmlNvLinkCapability_t capability,
                                                                       unsigned int *capResult) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkCapability");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlNvLinkCapability_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkCapability"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, capability, capResult);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link,
                                                                             nvmlPciInfo_t *pci) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkRemotePciInfo_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlPciInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkRemotePciInfo_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, pci);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link,
                                                                         nvmlNvLinkErrorCounter_t counter,
                                                                         unsigned long long *counterValue) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkErrorCounter");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlNvLinkErrorCounter_t, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkErrorCounter"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, counter, counterValue);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) {
    HOOK_TRACE_PROFILE("nvmlDeviceResetNvLinkErrorCounters");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceResetNvLinkErrorCounters"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link,
                                                                               unsigned int counter,
                                                                               nvmlNvLinkUtilizationControl_t *control,
                                                                               unsigned int reset) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetNvLinkUtilizationControl");
    using func_ptr =
        nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int, nvmlNvLinkUtilizationControl_t *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetNvLinkUtilizationControl"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, counter, control, reset);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(
    nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t *control) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkUtilizationControl");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int, nvmlNvLinkUtilizationControl_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkUtilizationControl"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, counter, control);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link,
                                                                               unsigned int counter,
                                                                               unsigned long long *rxcounter,
                                                                               unsigned long long *txcounter) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkUtilizationCounter");
    using func_ptr =
        nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int, unsigned long long *, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkUtilizationCounter"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, counter, rxcounter, txcounter);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device,
                                                                                  unsigned int link,
                                                                                  unsigned int counter,
                                                                                  nvmlEnableState_t freeze) {
    HOOK_TRACE_PROFILE("nvmlDeviceFreezeNvLinkUtilizationCounter");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int, nvmlEnableState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceFreezeNvLinkUtilizationCounter"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, counter, freeze);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link,
                                                                                 unsigned int counter) {
    HOOK_TRACE_PROFILE("nvmlDeviceResetNvLinkUtilizationCounter");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceResetNvLinkUtilizationCounter"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, counter);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(
    nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkRemoteDeviceType");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlIntNvLinkDeviceType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkRemoteDeviceType"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, pNvLinkDeviceType);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set) {
    HOOK_TRACE_PROFILE("nvmlEventSetCreate");
    using func_ptr = nvmlReturn_t (*)(nvmlEventSet_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlEventSetCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(set);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes,
                                                                  nvmlEventSet_t set) {
    HOOK_TRACE_PROFILE("nvmlDeviceRegisterEvents");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long, nvmlEventSet_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceRegisterEvents"));
    HOOK_CHECK(func_entry);
    return func_entry(device, eventTypes, set);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device,
                                                                          unsigned long long *eventTypes) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetSupportedEventTypes");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetSupportedEventTypes"));
    HOOK_CHECK(func_entry);
    return func_entry(device, eventTypes);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data,
                                                             unsigned int timeoutms) {
    HOOK_TRACE_PROFILE("nvmlEventSetWait_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventData_t *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlEventSetWait_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(set, data, timeoutms);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
    HOOK_TRACE_PROFILE("nvmlEventSetFree");
    using func_ptr = nvmlReturn_t (*)(nvmlEventSet_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlEventSetFree"));
    HOOK_CHECK(func_entry);
    return func_entry(set);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t *pciInfo,
                                                                    nvmlEnableState_t newState) {
    HOOK_TRACE_PROFILE("nvmlDeviceModifyDrainState");
    using func_ptr = nvmlReturn_t (*)(nvmlPciInfo_t *, nvmlEnableState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceModifyDrainState"));
    HOOK_CHECK(func_entry);
    return func_entry(pciInfo, newState);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t *pciInfo,
                                                                   nvmlEnableState_t *currentState) {
    HOOK_TRACE_PROFILE("nvmlDeviceQueryDrainState");
    using func_ptr = nvmlReturn_t (*)(nvmlPciInfo_t *, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceQueryDrainState"));
    HOOK_CHECK(func_entry);
    return func_entry(pciInfo, currentState);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo, nvmlDetachGpuState_t gpuState,
                                                                nvmlPcieLinkState_t linkState) {
    HOOK_TRACE_PROFILE("nvmlDeviceRemoveGpu_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlPciInfo_t *, nvmlDetachGpuState_t, nvmlPcieLinkState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceRemoveGpu_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(pciInfo, gpuState, linkState);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t *pciInfo) {
    HOOK_TRACE_PROFILE("nvmlDeviceDiscoverGpus");
    using func_ptr = nvmlReturn_t (*)(nvmlPciInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceDiscoverGpus"));
    HOOK_CHECK(func_entry);
    return func_entry(pciInfo);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount,
                                                                  nvmlFieldValue_t *values) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetFieldValues");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, int, nvmlFieldValue_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetFieldValues"));
    HOOK_CHECK(func_entry);
    return func_entry(device, valuesCount, values);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device,
                                                                         nvmlGpuVirtualizationMode_t *pVirtualMode) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetVirtualizationMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuVirtualizationMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetVirtualizationMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pVirtualMode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device,
                                                                   nvmlHostVgpuMode_t *pHostVgpuMode) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetHostVgpuMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlHostVgpuMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetHostVgpuMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pHostVgpuMode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device,
                                                                         nvmlGpuVirtualizationMode_t virtualMode) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetVirtualizationMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuVirtualizationMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetVirtualizationMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, virtualMode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t
    nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGridLicensableFeatures_v4");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGridLicensableFeatures_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGridLicensableFeatures_v4"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pGridLicensableFeatures);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device,
                                                                         nvmlProcessUtilizationSample_t *utilization,
                                                                         unsigned int *processSamplesCount,
                                                                         unsigned long long lastSeenTimeStamp) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetProcessUtilization");
    using func_ptr =
        nvmlReturn_t (*)(nvmlDevice_t, nvmlProcessUtilizationSample_t *, unsigned int *, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetProcessUtilization"));
    HOOK_CHECK(func_entry);
    return func_entry(device, utilization, processSamplesCount, lastSeenTimeStamp);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int *vgpuCount,
                                                                     nvmlVgpuTypeId_t *vgpuTypeIds) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetSupportedVgpus");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlVgpuTypeId_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetSupportedVgpus"));
    HOOK_CHECK(func_entry);
    return func_entry(device, vgpuCount, vgpuTypeIds);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int *vgpuCount,
                                                                     nvmlVgpuTypeId_t *vgpuTypeIds) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCreatableVgpus");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlVgpuTypeId_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCreatableVgpus"));
    HOOK_CHECK(func_entry);
    return func_entry(device, vgpuCount, vgpuTypeIds);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass,
                                                              unsigned int *size) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetClass");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, char *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetClass"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, vgpuTypeClass, size);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName,
                                                             unsigned int *size) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetName");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, char *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetName"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, vgpuTypeName, size);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId,
                                                                             unsigned int *gpuInstanceProfileId) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetGpuInstanceProfileId");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetGpuInstanceProfileId"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, gpuInstanceProfileId);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId,
                                                                 unsigned long long *deviceID,
                                                                 unsigned long long *subsystemID) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetDeviceID");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned long long *, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetDeviceID"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, deviceID, subsystemID);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId,
                                                                        unsigned long long *fbSize) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetFramebufferSize");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetFramebufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, fbSize);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId,
                                                                        unsigned int *numDisplayHeads) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetNumDisplayHeads");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetNumDisplayHeads"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, numDisplayHeads);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId,
                                                                   unsigned int displayIndex, unsigned int *xdim,
                                                                   unsigned int *ydim) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetResolution");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetResolution"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, displayIndex, xdim, ydim);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId,
                                                                char *vgpuTypeLicenseString, unsigned int size) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetLicense");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetLicense"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, vgpuTypeLicenseString, size);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId,
                                                                       unsigned int *frameRateLimit) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetFrameRateLimit");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetFrameRateLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, frameRateLimit);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId,
                                                                     unsigned int *vgpuInstanceCount) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetMaxInstances");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuTypeId_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetMaxInstances"));
    HOOK_CHECK(func_entry);
    return func_entry(device, vgpuTypeId, vgpuInstanceCount);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId,
                                                                          unsigned int *vgpuInstanceCountPerVm) {
    HOOK_TRACE_PROFILE("nvmlVgpuTypeGetMaxInstancesPerVm");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuTypeId_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuTypeGetMaxInstancesPerVm"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuTypeId, vgpuInstanceCountPerVm);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int *vgpuCount,
                                                                  nvmlVgpuInstance_t *vgpuInstances) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetActiveVgpus");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlVgpuInstance_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetActiveVgpus"));
    HOOK_CHECK(func_entry);
    return func_entry(device, vgpuCount, vgpuInstances);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char *vmId,
                                                                 unsigned int size, nvmlVgpuVmIdType_t *vmIdType) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetVmID");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, char *, unsigned int, nvmlVgpuVmIdType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetVmID"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, vmId, size, vmIdType);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char *uuid,
                                                                 unsigned int size) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetUUID");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetUUID"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, uuid, size);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance,
                                                                            char *version, unsigned int length) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetVmDriverVersion");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, char *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetVmDriverVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, version, length);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance,
                                                                    unsigned long long *fbUsage) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetFbUsage");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetFbUsage"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, fbUsage);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance,
                                                                          unsigned int *licensed) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetLicenseStatus");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetLicenseStatus"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, licensed);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance,
                                                                 nvmlVgpuTypeId_t *vgpuTypeId) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetType");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlVgpuTypeId_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetType"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, vgpuTypeId);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance,
                                                                           unsigned int *frameRateLimit) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetFrameRateLimit");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetFrameRateLimit"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, frameRateLimit);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance,
                                                                    nvmlEnableState_t *eccMode) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetEccMode");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetEccMode"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, eccMode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance,
                                                                            unsigned int *encoderCapacity) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetEncoderCapacity");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetEncoderCapacity"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, encoderCapacity);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance,
                                                                            unsigned int encoderCapacity) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceSetEncoderCapacity");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceSetEncoderCapacity"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, encoderCapacity);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance,
                                                                         unsigned int *sessionCount,
                                                                         unsigned int *averageFps,
                                                                         unsigned int *averageLatency) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetEncoderStats");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int *, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetEncoderStats"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, sessionCount, averageFps, averageLatency);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance,
                                                                            unsigned int *sessionCount,
                                                                            nvmlEncoderSessionInfo_t *sessionInfo) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetEncoderSessions");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int *, nvmlEncoderSessionInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetEncoderSessions"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, sessionCount, sessionInfo);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance,
                                                                     nvmlFBCStats_t *fbcStats) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetFBCStats");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlFBCStats_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetFBCStats"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, fbcStats);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance,
                                                                        unsigned int *sessionCount,
                                                                        nvmlFBCSessionInfo_t *sessionInfo) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetFBCSessions");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int *, nvmlFBCSessionInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetFBCSessions"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, sessionCount, sessionInfo);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance,
                                                                          unsigned int *gpuInstanceId) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetGpuInstanceId");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetGpuInstanceId"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, gpuInstanceId);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance,
                                                                     nvmlVgpuMetadata_t *vgpuMetadata,
                                                                     unsigned int *bufferSize) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetMetadata");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlVgpuMetadata_t *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetMetadata"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, vgpuMetadata, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device,
                                                                   nvmlVgpuPgpuMetadata_t *pgpuMetadata,
                                                                   unsigned int *bufferSize) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetVgpuMetadata");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlVgpuPgpuMetadata_t *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetVgpuMetadata"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pgpuMetadata, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata,
                                                                  nvmlVgpuPgpuMetadata_t *pgpuMetadata,
                                                                  nvmlVgpuPgpuCompatibility_t *compatibilityInfo) {
    HOOK_TRACE_PROFILE("nvmlGetVgpuCompatibility");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuMetadata_t *, nvmlVgpuPgpuMetadata_t *, nvmlVgpuPgpuCompatibility_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGetVgpuCompatibility"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuMetadata, pgpuMetadata, compatibilityInfo);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char *pgpuMetadata,
                                                                         unsigned int *bufferSize) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPgpuMetadataString");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPgpuMetadataString"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pgpuMetadata, bufferSize);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported, nvmlVgpuVersion_t *current) {
    HOOK_TRACE_PROFILE("nvmlGetVgpuVersion");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuVersion_t *, nvmlVgpuVersion_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGetVgpuVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(supported, current);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion) {
    HOOK_TRACE_PROFILE("nvmlSetVgpuVersion");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuVersion_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlSetVgpuVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuVersion);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetVgpuUtilization(
    nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t *sampleValType,
    unsigned int *vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t *utilizationSamples) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetVgpuUtilization");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned long long, nvmlValueType_t *, unsigned int *,
                                      nvmlVgpuInstanceUtilizationSample_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetVgpuUtilization"));
    HOOK_CHECK(func_entry);
    return func_entry(device, lastSeenTimeStamp, sampleValType, vgpuInstanceSamplesCount, utilizationSamples);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(
    nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int *vgpuProcessSamplesCount,
    nvmlVgpuProcessUtilizationSample_t *utilizationSamples) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetVgpuProcessUtilization");
    using func_ptr =
        nvmlReturn_t (*)(nvmlDevice_t, unsigned long long, unsigned int *, nvmlVgpuProcessUtilizationSample_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetVgpuProcessUtilization"));
    HOOK_CHECK(func_entry);
    return func_entry(device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance,
                                                                           nvmlEnableState_t *mode) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetAccountingMode");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, nvmlEnableState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetAccountingMode"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, mode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance,
                                                                           unsigned int *count, unsigned int *pids) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetAccountingPids");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetAccountingPids"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, count, pids);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance,
                                                                            unsigned int pid,
                                                                            nvmlAccountingStats_t *stats) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceGetAccountingStats");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t, unsigned int, nvmlAccountingStats_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceGetAccountingStats"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance, pid, stats);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) {
    HOOK_TRACE_PROFILE("nvmlVgpuInstanceClearAccountingPids");
    using func_ptr = nvmlReturn_t (*)(nvmlVgpuInstance_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlVgpuInstanceClearAccountingPids"));
    HOOK_CHECK(func_entry);
    return func_entry(vgpuInstance);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int *deviceCount) {
    HOOK_TRACE_PROFILE("nvmlGetExcludedDeviceCount");
    using func_ptr = nvmlReturn_t (*)(unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGetExcludedDeviceCount"));
    HOOK_CHECK(func_entry);
    return func_entry(deviceCount);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index,
                                                                          nvmlExcludedDeviceInfo_t *info) {
    HOOK_TRACE_PROFILE("nvmlGetExcludedDeviceInfoByIndex");
    using func_ptr = nvmlReturn_t (*)(unsigned int, nvmlExcludedDeviceInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGetExcludedDeviceInfoByIndex"));
    HOOK_CHECK(func_entry);
    return func_entry(index, info);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode,
                                                              nvmlReturn_t *activationStatus) {
    HOOK_TRACE_PROFILE("nvmlDeviceSetMigMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlReturn_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceSetMigMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, mode, activationStatus);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int *currentMode,
                                                              unsigned int *pendingMode) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMigMode");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMigMode"));
    HOOK_CHECK(func_entry);
    return func_entry(device, currentMode, pendingMode);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile,
                                                                             nvmlGpuInstanceProfileInfo_t *info) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGpuInstanceProfileInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstanceProfileInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGpuInstanceProfileInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(device, profile, info);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(
    nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGpuInstancePossiblePlacements_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstancePlacement_t *, unsigned int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGpuInstancePossiblePlacements_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, profileId, placements, count);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device,
                                                                                   unsigned int profileId,
                                                                                   unsigned int *count) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGpuInstanceRemainingCapacity");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGpuInstanceRemainingCapacity"));
    HOOK_CHECK(func_entry);
    return func_entry(device, profileId, count);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId,
                                                                     nvmlGpuInstance_t *gpuInstance) {
    HOOK_TRACE_PROFILE("nvmlDeviceCreateGpuInstance");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstance_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceCreateGpuInstance"));
    HOOK_CHECK(func_entry);
    return func_entry(device, profileId, gpuInstance);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(
    nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t *placement,
    nvmlGpuInstance_t *gpuInstance) {
    HOOK_TRACE_PROFILE("nvmlDeviceCreateGpuInstanceWithPlacement");
    using func_ptr =
        nvmlReturn_t (*)(nvmlDevice_t, unsigned int, const nvmlGpuInstancePlacement_t *, nvmlGpuInstance_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceCreateGpuInstanceWithPlacement"));
    HOOK_CHECK(func_entry);
    return func_entry(device, profileId, placement, gpuInstance);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) {
    HOOK_TRACE_PROFILE("nvmlGpuInstanceDestroy");
    using func_ptr = nvmlReturn_t (*)(nvmlGpuInstance_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGpuInstanceDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(gpuInstance);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId,
                                                                   nvmlGpuInstance_t *gpuInstances,
                                                                   unsigned int *count) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGpuInstances");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstance_t *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGpuInstances"));
    HOOK_CHECK(func_entry);
    return func_entry(device, profileId, gpuInstances, count);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id,
                                                                      nvmlGpuInstance_t *gpuInstance) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGpuInstanceById");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstance_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGpuInstanceById"));
    HOOK_CHECK(func_entry);
    return func_entry(device, id, gpuInstance);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance,
                                                                nvmlGpuInstanceInfo_t *info) {
    HOOK_TRACE_PROFILE("nvmlGpuInstanceGetInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlGpuInstance_t, nvmlGpuInstanceInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGpuInstanceGetInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(gpuInstance, info);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t
    nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile,
                                                 unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info) {
    HOOK_TRACE_PROFILE("nvmlGpuInstanceGetComputeInstanceProfileInfo");
    using func_ptr =
        nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, unsigned int, nvmlComputeInstanceProfileInfo_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGpuInstanceGetComputeInstanceProfileInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(gpuInstance, profile, engProfile, info);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(
    nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count) {
    HOOK_TRACE_PROFILE("nvmlGpuInstanceGetComputeInstanceRemainingCapacity");
    using func_ptr = nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, unsigned int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGpuInstanceGetComputeInstanceRemainingCapacity"));
    HOOK_CHECK(func_entry);
    return func_entry(gpuInstance, profileId, count);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance,
                                                                              unsigned int profileId,
                                                                              nvmlComputeInstance_t *computeInstance) {
    HOOK_TRACE_PROFILE("nvmlGpuInstanceCreateComputeInstance");
    using func_ptr = nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, nvmlComputeInstance_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGpuInstanceCreateComputeInstance"));
    HOOK_CHECK(func_entry);
    return func_entry(gpuInstance, profileId, computeInstance);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) {
    HOOK_TRACE_PROFILE("nvmlComputeInstanceDestroy");
    using func_ptr = nvmlReturn_t (*)(nvmlComputeInstance_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlComputeInstanceDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(computeInstance);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance,
                                                                            unsigned int profileId,
                                                                            nvmlComputeInstance_t *computeInstances,
                                                                            unsigned int *count) {
    HOOK_TRACE_PROFILE("nvmlGpuInstanceGetComputeInstances");
    using func_ptr = nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, nvmlComputeInstance_t *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGpuInstanceGetComputeInstances"));
    HOOK_CHECK(func_entry);
    return func_entry(gpuInstance, profileId, computeInstances, count);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance,
                                                                               unsigned int id,
                                                                               nvmlComputeInstance_t *computeInstance) {
    HOOK_TRACE_PROFILE("nvmlGpuInstanceGetComputeInstanceById");
    using func_ptr = nvmlReturn_t (*)(nvmlGpuInstance_t, unsigned int, nvmlComputeInstance_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlGpuInstanceGetComputeInstanceById"));
    HOOK_CHECK(func_entry);
    return func_entry(gpuInstance, id, computeInstance);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance,
                                                                       nvmlComputeInstanceInfo_t *info) {
    HOOK_TRACE_PROFILE("nvmlComputeInstanceGetInfo_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlComputeInstance_t, nvmlComputeInstanceInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlComputeInstanceGetInfo_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(computeInstance, info);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int *isMigDevice) {
    HOOK_TRACE_PROFILE("nvmlDeviceIsMigDeviceHandle");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceIsMigDeviceHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(device, isMigDevice);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGpuInstanceId");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGpuInstanceId"));
    HOOK_CHECK(func_entry);
    return func_entry(device, id);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int *id) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetComputeInstanceId");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetComputeInstanceId"));
    HOOK_CHECK(func_entry);
    return func_entry(device, id);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int *count) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMaxMigDeviceCount");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMaxMigDeviceCount"));
    HOOK_CHECK(func_entry);
    return func_entry(device, count);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index,
                                                                             nvmlDevice_t *migDevice) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMigDeviceHandleByIndex");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMigDeviceHandleByIndex"));
    HOOK_CHECK(func_entry);
    return func_entry(device, index, migDevice);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice,
                                                                                      nvmlDevice_t *device) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetDeviceHandleFromMigDeviceHandle");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetDeviceHandleFromMigDeviceHandle"));
    HOOK_CHECK(func_entry);
    return func_entry(migDevice, device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlInit() {
    HOOK_TRACE_PROFILE("nvmlInit");
    using func_ptr = nvmlReturn_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlInit"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetCount");
    using func_ptr = nvmlReturn_t (*)(unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetCount"));
    HOOK_CHECK(func_entry);
    return func_entry(deviceCount);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetHandleByIndex");
    using func_ptr = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetHandleByIndex"));
    HOOK_CHECK(func_entry);
    return func_entry(index, device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetHandleByPciBusId");
    using func_ptr = nvmlReturn_t (*)(const char *, nvmlDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetHandleByPciBusId"));
    HOOK_CHECK(func_entry);
    return func_entry(pciBusId, device);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPciInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPciInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pci);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetPciInfo_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetPciInfo_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pci);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link,
                                                                          nvmlPciInfo_t *pci) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetNvLinkRemotePciInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlPciInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetNvLinkRemotePciInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(device, link, pci);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t
    nvmlDeviceGetGridLicensableFeatures(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGridLicensableFeatures");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGridLicensableFeatures_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGridLicensableFeatures"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pGridLicensableFeatures);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t
    nvmlDeviceGetGridLicensableFeatures_v2(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGridLicensableFeatures_v2");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGridLicensableFeatures_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGridLicensableFeatures_v2"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pGridLicensableFeatures);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t
    nvmlDeviceGetGridLicensableFeatures_v3(nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGridLicensableFeatures_v3");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlGridLicensableFeatures_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGridLicensableFeatures_v3"));
    HOOK_CHECK(func_entry);
    return func_entry(device, pGridLicensableFeatures);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceRemoveGpu(nvmlPciInfo_t *pciInfo) {
    HOOK_TRACE_PROFILE("nvmlDeviceRemoveGpu");
    using func_ptr = nvmlReturn_t (*)(nvmlPciInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceRemoveGpu"));
    HOOK_CHECK(func_entry);
    return func_entry(pciInfo);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlEventSetWait(nvmlEventSet_t set, nvmlEventData_t *data,
                                                          unsigned int timeoutms) {
    HOOK_TRACE_PROFILE("nvmlEventSetWait");
    using func_ptr = nvmlReturn_t (*)(nvmlEventSet_t, nvmlEventData_t *, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlEventSetWait"));
    HOOK_CHECK(func_entry);
    return func_entry(set, data, timeoutms);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetAttributes(nvmlDevice_t device,
                                                                 nvmlDeviceAttributes_t *attributes) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetAttributes");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, nvmlDeviceAttributes_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetAttributes"));
    HOOK_CHECK(func_entry);
    return func_entry(device, attributes);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlComputeInstanceGetInfo(nvmlComputeInstance_t computeInstance,
                                                                    nvmlComputeInstanceInfo_t *info) {
    HOOK_TRACE_PROFILE("nvmlComputeInstanceGetInfo");
    using func_ptr = nvmlReturn_t (*)(nvmlComputeInstance_t, nvmlComputeInstanceInfo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlComputeInstanceGetInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(computeInstance, info);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device,
                                                                              unsigned int *infoCount,
                                                                              nvmlProcessInfo_v1_t *infos) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetComputeRunningProcesses");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_v1_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetComputeRunningProcesses"));
    HOOK_CHECK(func_entry);
    return func_entry(device, infoCount, infos);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t device,
                                                                               unsigned int *infoCount,
                                                                               nvmlProcessInfo_v1_t *infos) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGraphicsRunningProcesses");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_v1_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGraphicsRunningProcesses"));
    HOOK_CHECK(func_entry);
    return func_entry(device, infoCount, infos);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses(nvmlDevice_t device,
                                                                                 unsigned int *infoCount,
                                                                                 nvmlProcessInfo_v1_t *infos) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetMPSComputeRunningProcesses");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_v1_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetMPSComputeRunningProcesses"));
    HOOK_CHECK(func_entry);
    return func_entry(device, infoCount, infos);
}

HOOK_C_API HOOK_DECL_EXPORT nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements(
    nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t *placements, unsigned int *count) {
    HOOK_TRACE_PROFILE("nvmlDeviceGetGpuInstancePossiblePlacements");
    using func_ptr = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, nvmlGpuInstancePlacement_t *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVML_SYMBOL("nvmlDeviceGetGpuInstancePossiblePlacements"));
    HOOK_CHECK(func_entry);
    return func_entry(device, profileId, placements, count);
}
