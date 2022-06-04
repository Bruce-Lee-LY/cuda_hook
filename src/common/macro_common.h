// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 15:40:15 on Sun, May 29, 2022
//
// Description: common macro

#ifndef __CUDA_HOOK_MACRO_COMMON_H__
#define __CUDA_HOOK_MACRO_COMMON_H__

#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define HOOK_C_API extern "C"
#define HOOK_DECL_EXPORT __attribute__((visibility("default")))

#define HOOK_LIKELY(x) __builtin_expect(!!(x), 1)
#define HOOK_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define HOOK_LOG_TAG "CUDA-HOOK"
#define HOOK_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define HLOG(format, ...)                                                                             \
    fprintf(stderr, "[%s %d:%ld %s:%d %s] " format "\n", HOOK_LOG_TAG, getpid(), syscall(SYS_gettid), \
            HOOK_LOG_FILE(__FILE__), __LINE__, __FUNCTION__, ##__VA_ARGS__)

#define HOOK_CHECK(x)                     \
    do {                                  \
        if (HOOK_UNLIKELY(!(x))) {        \
            HLOG("check failed: %s", #x); \
            exit(EXIT_FAILURE);           \
        }                                 \
    } while (0)

#define HOOK_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;        \
    void operator=(const TypeName &) = delete;

#endif  // __CUDA_HOOK_MACRO_COMMON_H__
