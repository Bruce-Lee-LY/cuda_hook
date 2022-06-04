// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:35:09 on Sun, May 29, 2022
//
// Description: source file in cuda/nvml/example/supportedVgpus.c

/***************************************************************************\
|*                                                                           *|
|*      Copyright 2010-2016 NVIDIA Corporation.  All rights reserved.        *|
|*                                                                           *|
|*   NOTICE TO USER:                                                         *|
|*                                                                           *|
|*   This source code is subject to NVIDIA ownership rights under U.S.       *|
|*   and international Copyright laws.  Users and possessors of this         *|
|*   source code are hereby granted a nonexclusive, royalty-free             *|
|*   license to use this code in individual and commercial software.         *|
|*                                                                           *|
|*   NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE     *|
|*   CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR         *|
|*   IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH      *|
|*   REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF         *|
|*   MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR          *|
|*   PURPOSE. IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL,            *|
|*   INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES          *|
|*   WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN      *|
|*   AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING     *|
|*   OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE      *|
|*   CODE.                                                                   *|
|*                                                                           *|
|*   U.S. Government End Users. This source code is a "commercial item"      *|
|*   as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting       *|
|*   of "commercial computer  software" and "commercial computer software    *|
|*   documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)   *|
|*   and is provided to the U.S. Government only as a commercial end item.   *|
|*   Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through        *|
|*   227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the       *|
|*   source code with only those rights set forth herein.                    *|
|*                                                                           *|
|*   Any use of this source code in individual and commercial software must  *|
|*   include, in the user documentation and internal comments to the code,   *|
|*   the above Disclaimer and U.S. Government End Users Notice.              *|
|*                                                                           *|
|*                                                                           *|
\***************************************************************************/

#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    nvmlReturn_t result;
    unsigned int device_count, i;

    // First initialize NVML library
    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    result = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != result) {
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        goto Error;
    }

    printf("Found %u device%s\n", device_count, device_count != 1 ? "s" : "");
    printf("Listing devices:\n");

    for (i = 0; i < device_count; i++) {
        nvmlDevice_t device;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlPciInfo_t pci;

        // Query for device handle to perform operations on a device
        // You can also query device handle by other features like:
        // nvmlDeviceGetHandleBySerial
        // nvmlDeviceGetHandleByPciBusId
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result) {
            printf("Failed to get handle for device %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        }

        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result) {
            printf("Failed to get name of device %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        }

        // pci.busId is very useful to know which device physically you're talking to
        // Using PCI identifier you can also match nvmlDevice handle to CUDA device.
        result = nvmlDeviceGetPciInfo(device, &pci);
        if (NVML_SUCCESS != result) {
            printf("Failed to get pci info for device %u: %s\n", i, nvmlErrorString(result));
            goto Error;
        }

        printf("%u. %s [%s]\n", i, name, pci.busId);

        // This is an example to get the supported vGPUs type names
        unsigned int vgpuCount = 0;
        nvmlVgpuTypeId_t *vgpuTypeIds = NULL;
        unsigned int j;

        result = nvmlDeviceGetSupportedVgpus(device, &vgpuCount, NULL);
        if (NVML_ERROR_INSUFFICIENT_SIZE != result)
            goto Error;

        if (vgpuCount != 0) {
            vgpuTypeIds = malloc(sizeof(nvmlVgpuTypeId_t) * vgpuCount);
            if (!vgpuTypeIds) {
                printf("Memory allocation of %d bytes failed \n", (int)(sizeof(*vgpuTypeIds) * vgpuCount));
                goto Error;
            }

            result = nvmlDeviceGetSupportedVgpus(device, &vgpuCount, vgpuTypeIds);
            if (NVML_SUCCESS != result) {
                printf("Failed to get the supported vGPUs with status %d \n", (int)result);
                goto Error;
            }

            printf("  Displaying vGPU type names: \n");
            for (j = 0; j < vgpuCount; j++) {
                char vgpuTypeName[NVML_DEVICE_NAME_BUFFER_SIZE];
                unsigned int bufferSize = NVML_DEVICE_NAME_BUFFER_SIZE;

                if (NVML_SUCCESS == (result = nvmlVgpuTypeGetName(vgpuTypeIds[j], vgpuTypeName, &bufferSize))) {
                    printf("  %s\n", vgpuTypeName);
                } else {
                    printf("Failed to query the vGPU type name with status %d \n", (int)result);
                }
            }
        }
        if (vgpuTypeIds)
            free(vgpuTypeIds);
    }

    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

    printf("All done.\n");
    return 0;

Error:
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

    return 1;
}
