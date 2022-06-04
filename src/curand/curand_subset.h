// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: curand subset

#ifndef __CUDA_HOOK_CURAND_SUBSET_H__
#define __CUDA_HOOK_CURAND_SUBSET_H__

#ifdef __cplusplus
extern "C" {
#endif

#define CURAND_VER_MAJOR 10
#define CURAND_VER_MINOR 2
#define CURAND_VER_PATCH 5
#define CURAND_VER_BUILD 120
#define CURAND_VERSION (CURAND_VER_MAJOR * 1000 + CURAND_VER_MINOR * 100 + CURAND_VER_PATCH)
/* CURAND Host API datatypes */

/**
 * @{
 */

/**
 * CURAND function call status types
 */
enum curandStatus {
    CURAND_STATUS_SUCCESS = 0,                      ///< No errors
    CURAND_STATUS_VERSION_MISMATCH = 100,           ///< Header file and linked library version do not match
    CURAND_STATUS_NOT_INITIALIZED = 101,            ///< Generator not initialized
    CURAND_STATUS_ALLOCATION_FAILED = 102,          ///< Memory allocation failed
    CURAND_STATUS_TYPE_ERROR = 103,                 ///< Generator is wrong type
    CURAND_STATUS_OUT_OF_RANGE = 104,               ///< Argument out of range
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,        ///< Length requested is not a multple of dimension
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,  ///< GPU does not have double precision required by MRG32k3a
    CURAND_STATUS_LAUNCH_FAILURE = 201,             ///< Kernel launch failure
    CURAND_STATUS_PREEXISTING_FAILURE = 202,        ///< Preexisting failure on library entry
    CURAND_STATUS_INITIALIZATION_FAILED = 203,      ///< Initialization of CUDA failed
    CURAND_STATUS_ARCH_MISMATCH = 204,              ///< Architecture mismatch, GPU does not support requested feature
    CURAND_STATUS_INTERNAL_ERROR = 999              ///< Internal library error
};

/*
 * CURAND function call status types
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum curandStatus curandStatus_t;
/** \endcond */

/**
 * CURAND generator types
 */
enum curandRngType {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100,           ///< Default pseudorandom generator
    CURAND_RNG_PSEUDO_XORWOW = 101,            ///< XORWOW pseudorandom generator
    CURAND_RNG_PSEUDO_MRG32K3A = 121,          ///< MRG32k3a pseudorandom generator
    CURAND_RNG_PSEUDO_MTGP32 = 141,            ///< Mersenne Twister MTGP32 pseudorandom generator
    CURAND_RNG_PSEUDO_MT19937 = 142,           ///< Mersenne Twister MT19937 pseudorandom generator
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,     ///< PHILOX-4x32-10 pseudorandom generator
    CURAND_RNG_QUASI_DEFAULT = 200,            ///< Default quasirandom generator
    CURAND_RNG_QUASI_SOBOL32 = 201,            ///< Sobol32 quasirandom generator
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,  ///< Scrambled Sobol32 quasirandom generator
    CURAND_RNG_QUASI_SOBOL64 = 203,            ///< Sobol64 quasirandom generator
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204   ///< Scrambled Sobol64 quasirandom generator
};

/*
 * CURAND generator types
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum curandRngType curandRngType_t;
/** \endcond */

/**
 * CURAND ordering of results in memory
 */
enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100,  ///< Best ordering for pseudorandom results
    CURAND_ORDERING_PSEUDO_DEFAULT =
        101,  ///< Specific default thread sequence for pseudorandom results, same as CURAND_ORDERING_PSEUDO_BEST
    CURAND_ORDERING_PSEUDO_SEEDED = 102,  ///< Specific seeding pattern for fast lower quality pseudorandom results
    CURAND_ORDERING_PSEUDO_LEGACY = 103,  ///< Specific legacy sequence for pseudorandom results, guaranteed to remain
                                          ///< the same for all cuRAND release
    CURAND_ORDERING_QUASI_DEFAULT = 201   ///< Specific n-dimensional ordering for quasirandom results
};

/*
 * CURAND ordering of results in memory
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum curandOrdering curandOrdering_t;
/** \endcond */

/**
 * CURAND choice of direction vector set
 */
enum curandDirectionVectorSet {
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,  ///< Specific set of 32-bit direction vectors generated from polynomials
                                                ///< recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 =
        102,  ///< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y.
              ///< Kuo, for up to 20,000 dimensions, and scrambled
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,  ///< Specific set of 64-bit direction vectors generated from polynomials
                                                ///< recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 =
        104  ///< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y.
             ///< Kuo, for up to 20,000 dimensions, and scrambled
};

/*
 * CURAND choice of direction vector set
 */
/** \cond UNHIDE_TYPEDEFS */
typedef enum curandDirectionVectorSet curandDirectionVectorSet_t;
/** \endcond */

/**
 * CURAND array of 32-bit direction vectors
 */
/** \cond UNHIDE_TYPEDEFS */
typedef unsigned int curandDirectionVectors32_t[32];
/** \endcond */

/**
 * CURAND array of 64-bit direction vectors
 */
/** \cond UNHIDE_TYPEDEFS */
typedef unsigned long long curandDirectionVectors64_t[64];
/** \endcond **/

/**
 * CURAND generator (opaque)
 */
struct curandGenerator_st;

/**
 * CURAND generator
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandGenerator_st *curandGenerator_t;
/** \endcond */

/**
 * CURAND distribution
 */
/** \cond UNHIDE_TYPEDEFS */
typedef double curandDistribution_st;
typedef curandDistribution_st *curandDistribution_t;
typedef struct curandDistributionShift_st *curandDistributionShift_t;
/** \endcond */
/**
 * CURAND distribution M2
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandDistributionM2Shift_st *curandDistributionM2Shift_t;
typedef struct curandHistogramM2_st *curandHistogramM2_t;
typedef unsigned int curandHistogramM2K_st;
typedef curandHistogramM2K_st *curandHistogramM2K_t;
typedef curandDistribution_st curandHistogramM2V_st;
typedef curandHistogramM2V_st *curandHistogramM2V_t;

typedef struct curandDiscreteDistribution_st *curandDiscreteDistribution_t;
/** \endcond */

/*
 * CURAND METHOD
 */
/** \cond UNHIDE_ENUMS */
enum curandMethod {
    CURAND_CHOOSE_BEST = 0,  // choose best depends on args
    CURAND_ITR = 1,
    CURAND_KNUTH = 2,
    CURAND_HITR = 3,
    CURAND_M1 = 4,
    CURAND_M2 = 5,
    CURAND_BINARY_SEARCH = 6,
    CURAND_DISCRETE_GAUSS = 7,
    CURAND_REJECTION = 8,
    CURAND_DEVICE_API = 9,
    CURAND_FAST_REJECTION = 10,
    CURAND_3RD = 11,
    CURAND_DEFINITION = 12,
    CURAND_POISSON = 13
};

typedef enum curandMethod curandMethod_t;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_CURAND_SUBSET_H__
