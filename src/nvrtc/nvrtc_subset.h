// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: nvrtc subset

#ifndef __CUDA_HOOK_NVRTC_SUBSET_H__
#define __CUDA_HOOK_NVRTC_SUBSET_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \ingroup error
 * \brief   The enumerated type nvrtcResult defines API call result codes.
 *          NVRTC API functions return nvrtcResult to indicate the call
 *          result.
 */
typedef enum {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11
} nvrtcResult;

/**
 * \ingroup compilation
 * \brief   nvrtcProgram is the unit of compilation, and an opaque handle for
 *          a program.
 *
 * To compile a CUDA program string, an instance of nvrtcProgram must be
 * created first with ::nvrtcCreateProgram, then compiled with
 * ::nvrtcCompileProgram.
 */
typedef struct _nvrtcProgram *nvrtcProgram;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_NVRTC_SUBSET_H__
