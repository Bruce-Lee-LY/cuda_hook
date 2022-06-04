// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 29 apis

#include "cublas_subset.h"
#include "curand_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandCreateGenerator(curandGenerator_t *generator,
                                                                 curandRngType_t rng_type) {
    HOOK_TRACE_PROFILE("curandCreateGenerator");
    using func_ptr = curandStatus_t (*)(curandGenerator_t *, curandRngType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandCreateGenerator"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, rng_type);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandCreateGeneratorHost(curandGenerator_t *generator,
                                                                     curandRngType_t rng_type) {
    HOOK_TRACE_PROFILE("curandCreateGeneratorHost");
    using func_ptr = curandStatus_t (*)(curandGenerator_t *, curandRngType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandCreateGeneratorHost"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, rng_type);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    HOOK_TRACE_PROFILE("curandDestroyGenerator");
    using func_ptr = curandStatus_t (*)(curandGenerator_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandDestroyGenerator"));
    HOOK_CHECK(func_entry);
    return func_entry(generator);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGetVersion(int *version) {
    HOOK_TRACE_PROFILE("curandGetVersion");
    using func_ptr = curandStatus_t (*)(int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry(version);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGetProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("curandGetProperty");
    using func_ptr = curandStatus_t (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGetProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("curandSetStream");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandSetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, stream);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                                              unsigned long long seed) {
    HOOK_TRACE_PROFILE("curandSetPseudoRandomGeneratorSeed");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandSetPseudoRandomGeneratorSeed"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, seed);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator,
                                                                    unsigned long long offset) {
    HOOK_TRACE_PROFILE("curandSetGeneratorOffset");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandSetGeneratorOffset"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, offset);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator,
                                                                      curandOrdering_t order) {
    HOOK_TRACE_PROFILE("curandSetGeneratorOrdering");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, curandOrdering_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandSetGeneratorOrdering"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, order);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                                                                   unsigned int num_dimensions) {
    HOOK_TRACE_PROFILE("curandSetQuasiRandomGeneratorDimensions");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandSetQuasiRandomGeneratorDimensions"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, num_dimensions);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int *outputPtr,
                                                          size_t num) {
    HOOK_TRACE_PROFILE("curandGenerate");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, unsigned int *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerate"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, num);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateLongLong(curandGenerator_t generator,
                                                                  unsigned long long *outputPtr, size_t num) {
    HOOK_TRACE_PROFILE("curandGenerateLongLong");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, unsigned long long *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateLongLong"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, num);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *outputPtr,
                                                                 size_t num) {
    HOOK_TRACE_PROFILE("curandGenerateUniform");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, float *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateUniform"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, num);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr,
                                                                       size_t num) {
    HOOK_TRACE_PROFILE("curandGenerateUniformDouble");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, double *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateUniformDouble"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, num);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n,
                                                                float mean, float stddev) {
    HOOK_TRACE_PROFILE("curandGenerateNormal");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, float *, size_t, float, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateNormal"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, n, mean, stddev);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr,
                                                                      size_t n, double mean, double stddev) {
    HOOK_TRACE_PROFILE("curandGenerateNormalDouble");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, double *, size_t, double, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateNormalDouble"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, n, mean, stddev);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr,
                                                                   size_t n, float mean, float stddev) {
    HOOK_TRACE_PROFILE("curandGenerateLogNormal");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, float *, size_t, float, float);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateLogNormal"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, n, mean, stddev);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr,
                                                                         size_t n, double mean, double stddev) {
    HOOK_TRACE_PROFILE("curandGenerateLogNormalDouble");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, double *, size_t, double, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateLogNormalDouble"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, n, mean, stddev);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t
    curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t *discrete_distribution) {
    HOOK_TRACE_PROFILE("curandCreatePoissonDistribution");
    using func_ptr = curandStatus_t (*)(double, curandDiscreteDistribution_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandCreatePoissonDistribution"));
    HOOK_CHECK(func_entry);
    return func_entry(lambda, discrete_distribution);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t
    curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution) {
    HOOK_TRACE_PROFILE("curandDestroyDistribution");
    using func_ptr = curandStatus_t (*)(curandDiscreteDistribution_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandDestroyDistribution"));
    HOOK_CHECK(func_entry);
    return func_entry(discrete_distribution);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr,
                                                                 size_t n, double lambda) {
    HOOK_TRACE_PROFILE("curandGeneratePoisson");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, unsigned int *, size_t, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGeneratePoisson"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, n, lambda);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGeneratePoissonMethod(curandGenerator_t generator,
                                                                       unsigned int *outputPtr, size_t n, double lambda,
                                                                       curandMethod_t method) {
    HOOK_TRACE_PROFILE("curandGeneratePoissonMethod");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, unsigned int *, size_t, double, curandMethod_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGeneratePoissonMethod"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, n, lambda, method);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateBinomial(curandGenerator_t generator, unsigned int *outputPtr,
                                                                  size_t num, unsigned int n, double p) {
    HOOK_TRACE_PROFILE("curandGenerateBinomial");
    using func_ptr = curandStatus_t (*)(curandGenerator_t, unsigned int *, size_t, unsigned int, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateBinomial"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, num, n, p);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateBinomialMethod(curandGenerator_t generator,
                                                                        unsigned int *outputPtr, size_t num,
                                                                        unsigned int n, double p,
                                                                        curandMethod_t method) {
    HOOK_TRACE_PROFILE("curandGenerateBinomialMethod");
    using func_ptr =
        curandStatus_t (*)(curandGenerator_t, unsigned int *, size_t, unsigned int, double, curandMethod_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateBinomialMethod"));
    HOOK_CHECK(func_entry);
    return func_entry(generator, outputPtr, num, n, p, method);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGenerateSeeds(curandGenerator_t generator) {
    HOOK_TRACE_PROFILE("curandGenerateSeeds");
    using func_ptr = curandStatus_t (*)(curandGenerator_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGenerateSeeds"));
    HOOK_CHECK(func_entry);
    return func_entry(generator);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t *vectors,
                                                                       curandDirectionVectorSet_t set) {
    HOOK_TRACE_PROFILE("curandGetDirectionVectors32");
    using func_ptr = curandStatus_t (*)(curandDirectionVectors32_t *, curandDirectionVectorSet_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGetDirectionVectors32"));
    HOOK_CHECK(func_entry);
    return func_entry(vectors, set);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGetScrambleConstants32(unsigned int **constants) {
    HOOK_TRACE_PROFILE("curandGetScrambleConstants32");
    using func_ptr = curandStatus_t (*)(unsigned int **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGetScrambleConstants32"));
    HOOK_CHECK(func_entry);
    return func_entry(constants);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t *vectors,
                                                                       curandDirectionVectorSet_t set) {
    HOOK_TRACE_PROFILE("curandGetDirectionVectors64");
    using func_ptr = curandStatus_t (*)(curandDirectionVectors64_t *, curandDirectionVectorSet_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGetDirectionVectors64"));
    HOOK_CHECK(func_entry);
    return func_entry(vectors, set);
}

HOOK_C_API HOOK_DECL_EXPORT curandStatus_t curandGetScrambleConstants64(unsigned long long **constants) {
    HOOK_TRACE_PROFILE("curandGetScrambleConstants64");
    using func_ptr = curandStatus_t (*)(unsigned long long **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CURAND_SYMBOL("curandGetScrambleConstants64"));
    HOOK_CHECK(func_entry);
    return func_entry(constants);
}
