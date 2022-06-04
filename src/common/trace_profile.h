// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 14:54:28 on Sun, May 29, 2022
//
// Description: trace and profile

#ifndef __CUDA_HOOK_TRACE_PROFILE_H__
#define __CUDA_HOOK_TRACE_PROFILE_H__

#include <chrono>
#include <string>

#include "macro_common.h"

class TraceProfile {
public:
    TraceProfile(const std::string &name) : m_name(name), m_start(std::chrono::steady_clock::now()) {
        HLOG("%s enter", m_name.c_str());
    }

    ~TraceProfile() {
        m_end = std::chrono::steady_clock::now();
        m_duration = std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start);
        HLOG("%s exit, taken %.3lf ms", m_name.c_str(), m_duration.count());
    }

private:
    const std::string m_name;
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_end;
    std::chrono::duration<double, std::milli> m_duration;

    HOOK_DISALLOW_COPY_AND_ASSIGN(TraceProfile);
};

#ifdef HOOK_BUILD_DEBUG
#define HOOK_TRACE_PROFILE(name) TraceProfile _tp_##name_(name)
#else
#define HOOK_TRACE_PROFILE(name)
#endif

#endif  // __CUDA_HOOK_TRACE_PROFILE_H__
