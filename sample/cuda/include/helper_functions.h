// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:17:08 on Sun, May 29, 2022
//
// Description: source file in /usr/local/cuda/samples/common/inc

// These are helper functions for the SDK samples (string parsing,
// timers, image helpers, etc)
#ifndef COMMON_HELPER_FUNCTIONS_H_
#define COMMON_HELPER_FUNCTIONS_H_

#ifdef WIN32
#pragma warning(disable : 4996)
#endif

// includes, project
#include <assert.h>
#include <exception.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// includes, timer, string parsing, image helpers
#include <helper_image.h>   // helper functions for image compare, dump, data comparisons
#include <helper_string.h>  // helper functions for string parsing
#include <helper_timer.h>   // helper functions for timers

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#endif  // COMMON_HELPER_FUNCTIONS_H_
