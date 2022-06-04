// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: auto generate 231 apis

#include "cudnn_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT size_t cudnnGetVersion() {
    HOOK_TRACE_PROFILE("cudnnGetVersion");
    using func_ptr = size_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetVersion"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT size_t cudnnGetCudartVersion() {
    HOOK_TRACE_PROFILE("cudnnGetCudartVersion");
    using func_ptr = size_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetCudartVersion"));
    HOOK_CHECK(func_entry);
    return func_entry();
}

HOOK_C_API HOOK_DECL_EXPORT const char *cudnnGetErrorString(cudnnStatus_t status) {
    HOOK_TRACE_PROFILE("cudnnGetErrorString");
    using func_ptr = const char *(*)(cudnnStatus_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetErrorString"));
    HOOK_CHECK(func_entry);
    return func_entry(status);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnQueryRuntimeError(cudnnHandle_t handle, cudnnStatus_t *rstatus,
                                                                 cudnnErrQueryMode_t mode, cudnnRuntimeTag_t *tag) {
    HOOK_TRACE_PROFILE("cudnnQueryRuntimeError");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnStatus_t *, cudnnErrQueryMode_t, cudnnRuntimeTag_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnQueryRuntimeError"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rstatus, mode, tag);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("cudnnGetProperty");
    using func_ptr = cudnnStatus_t (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
    HOOK_TRACE_PROFILE("cudnnCreate");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
    HOOK_TRACE_PROFILE("cudnnDestroy");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    HOOK_TRACE_PROFILE("cudnnSetStream");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId) {
    HOOK_TRACE_PROFILE("cudnnGetStream");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetStream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, streamId);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                                                     cudnnTensorFormat_t format,
                                                                     cudnnDataType_t dataType, int n, int c, int h,
                                                                     int w) {
    HOOK_TRACE_PROFILE("cudnnSetTensor4dDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetTensor4dDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc, format, dataType, n, c, h, w);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                                                       cudnnDataType_t dataType, int n, int c, int h,
                                                                       int w, int nStride, int cStride, int hStride,
                                                                       int wStride) {
    HOOK_TRACE_PROFILE("cudnnSetTensor4dDescriptorEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnTensorDescriptor_t, cudnnDataType_t, int, int, int, int, int, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetTensor4dDescriptorEx"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                                                                     cudnnDataType_t *dataType, int *n, int *c, int *h,
                                                                     int *w, int *nStride, int *cStride, int *hStride,
                                                                     int *wStride) {
    HOOK_TRACE_PROFILE("cudnnGetTensor4dDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnTensorDescriptor_t, cudnnDataType_t *, int *, int *, int *, int *,
                                       int *, int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetTensor4dDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                                                     cudnnDataType_t dataType, int nbDims,
                                                                     const int dimA, const int strideA) {
    HOOK_TRACE_PROFILE("cudnnSetTensorNdDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorDescriptor_t, cudnnDataType_t, int, const int, const int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetTensorNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc, dataType, nbDims, dimA, strideA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                                                       cudnnTensorFormat_t format,
                                                                       cudnnDataType_t dataType, int nbDims,
                                                                       const int dimA) {
    HOOK_TRACE_PROFILE("cudnnSetTensorNdDescriptorEx");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, const int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetTensorNdDescriptorEx"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc, format, dataType, nbDims, dimA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                                                                     int nbDimsRequested, cudnnDataType_t *dataType,
                                                                     int *nbDims, int dimA, int strideA) {
    HOOK_TRACE_PROFILE("cudnnGetTensorNdDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnTensorDescriptor_t, int, cudnnDataType_t *, int *, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetTensorNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc, nbDimsRequested, dataType, nbDims, dimA, strideA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc,
                                                                    size_t *size) {
    HOOK_TRACE_PROFILE("cudnnGetTensorSizeInBytes");
    using func_ptr = cudnnStatus_t (*)(const cudnnTensorDescriptor_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetTensorSizeInBytes"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc, size);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(tensorDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                                                                 const cudnnTensorDescriptor_t srcDesc,
                                                                 cudnnTensorDescriptor_t destDesc,
                                                                 size_t *destSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnInitTransformDest");
    using func_ptr = cudnnStatus_t (*)(const cudnnTensorTransformDescriptor_t, const cudnnTensorDescriptor_t,
                                       cudnnTensorDescriptor_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnInitTransformDest"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc, srcDesc, destDesc, destSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateTensorTransformDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorTransformDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateTensorTransformDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, const uint32_t nbDims, const cudnnTensorFormat_t destFormat,
    const int32_t padBeforeA, const int32_t padAfterA, const uint32_t foldA, const cudnnFoldingDirection_t direction) {
    HOOK_TRACE_PROFILE("cudnnSetTensorTransformDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorTransformDescriptor_t, const uint32_t, const cudnnTensorFormat_t,
                                       const int32_t, const int32_t, const uint32_t, const cudnnFoldingDirection_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetTensorTransformDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc, nbDims, destFormat, padBeforeA, padAfterA, foldA, direction);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetTensorTransformDescriptor(
    cudnnTensorTransformDescriptor_t transformDesc, uint32_t nbDimsRequested, cudnnTensorFormat_t *destFormat,
    int32_t padBeforeA, int32_t padAfterA, uint32_t foldA, cudnnFoldingDirection_t *direction) {
    HOOK_TRACE_PROFILE("cudnnGetTensorTransformDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorTransformDescriptor_t, uint32_t, cudnnTensorFormat_t *, int32_t,
                                       int32_t, uint32_t, cudnnFoldingDirection_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetTensorTransformDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc, nbDimsRequested, destFormat, padBeforeA, padAfterA, foldA, direction);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyTensorTransformDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorTransformDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyTensorTransformDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(transformDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnTransformTensor(cudnnHandle_t handle, const void *alpha,
                                                               const cudnnTensorDescriptor_t xDesc, const void *x,
                                                               const void *beta, const cudnnTensorDescriptor_t yDesc,
                                                               void *y) {
    HOOK_TRACE_PROFILE("cudnnTransformTensor");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
                                       const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnTransformTensor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alpha, xDesc, x, beta, yDesc, y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnTransformTensorEx(cudnnHandle_t handle, const cudnnTensorTransformDescriptor_t transDesc, const void *alpha,
                           const cudnnTensorDescriptor_t srcDesc, const void *srcData, const void *beta,
                           const cudnnTensorDescriptor_t destDesc, void *destData) {
    HOOK_TRACE_PROFILE("cudnnTransformTensorEx");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorTransformDescriptor_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const void *,
                                       const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnTransformTensorEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(
    const cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc,
    const cudnnTensorFormat_t transformFormat, cudnnFilterDescriptor_t foldedFilterDesc,
    cudnnTensorDescriptor_t paddedDiffDesc, cudnnConvolutionDescriptor_t foldedConvDesc,
    cudnnTensorDescriptor_t foldedGradDesc, cudnnTensorTransformDescriptor_t filterFoldTransDesc,
    cudnnTensorTransformDescriptor_t diffPadTransDesc, cudnnTensorTransformDescriptor_t gradFoldTransDesc,
    cudnnTensorTransformDescriptor_t gradUnfoldTransDesc) {
    HOOK_TRACE_PROFILE("cudnnGetFoldedConvBackwardDataDescriptors");
    using func_ptr =
        cudnnStatus_t (*)(const cudnnHandle_t, const cudnnFilterDescriptor_t, const cudnnTensorDescriptor_t,
                          const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorFormat_t,
                          cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t,
                          cudnnTensorDescriptor_t, cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t,
                          cudnnTensorTransformDescriptor_t, cudnnTensorTransformDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetFoldedConvBackwardDataDescriptors"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, filterDesc, diffDesc, convDesc, gradDesc, transformFormat, foldedFilterDesc,
                      paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc,
                      gradFoldTransDesc, gradUnfoldTransDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle, const void *alpha,
                                                         const cudnnTensorDescriptor_t aDesc, const void *A,
                                                         const void *beta, const cudnnTensorDescriptor_t cDesc,
                                                         void *C) {
    HOOK_TRACE_PROFILE("cudnnAddTensor");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
                                       const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnAddTensor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alpha, aDesc, A, beta, cDesc, C);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateOpTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnOpTensorDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateOpTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(opTensorDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                                                                     cudnnOpTensorOp_t opTensorOp,
                                                                     cudnnDataType_t opTensorCompType,
                                                                     cudnnNanPropagation_t opTensorNanOpt) {
    HOOK_TRACE_PROFILE("cudnnSetOpTensorDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetOpTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                                                                     cudnnOpTensorOp_t *opTensorOp,
                                                                     cudnnDataType_t *opTensorCompType,
                                                                     cudnnNanPropagation_t *opTensorNanOpt) {
    HOOK_TRACE_PROFILE("cudnnGetOpTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t *, cudnnDataType_t *,
                                       cudnnNanPropagation_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetOpTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(opTensorDesc, opTensorOp, opTensorCompType, opTensorNanOpt);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyOpTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnOpTensorDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyOpTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(opTensorDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnOpTensor(
    cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc, const void *alpha1,
    const cudnnTensorDescriptor_t aDesc, const void *A, const void *alpha2, const cudnnTensorDescriptor_t bDesc,
    const void *B, const void *beta, const cudnnTensorDescriptor_t cDesc, void *C) {
    HOOK_TRACE_PROFILE("cudnnOpTensor");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const cudnnOpTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const void *, const cudnnTensorDescriptor_t, const void *, const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnOpTensor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, opTensorDesc, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateReduceTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnReduceTensorDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateReduceTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(reduceTensorDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                         cudnnReduceTensorOp_t reduceTensorOp,
                                                                         cudnnDataType_t reduceTensorCompType,
                                                                         cudnnNanPropagation_t reduceTensorNanOpt,
                                                                         cudnnReduceTensorIndices_t reduceTensorIndices,
                                                                         cudnnIndicesType_t reduceTensorIndicesType) {
    HOOK_TRACE_PROFILE("cudnnSetReduceTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t, cudnnDataType_t,
                                       cudnnNanPropagation_t, cudnnReduceTensorIndices_t, cudnnIndicesType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetReduceTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices,
                      reduceTensorIndicesType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetReduceTensorDescriptor(
    const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t *reduceTensorOp,
    cudnnDataType_t *reduceTensorCompType, cudnnNanPropagation_t *reduceTensorNanOpt,
    cudnnReduceTensorIndices_t *reduceTensorIndices, cudnnIndicesType_t *reduceTensorIndicesType) {
    HOOK_TRACE_PROFILE("cudnnGetReduceTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnReduceTensorDescriptor_t, cudnnReduceTensorOp_t *, cudnnDataType_t *,
                                       cudnnNanPropagation_t *, cudnnReduceTensorIndices_t *, cudnnIndicesType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetReduceTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(reduceTensorDesc, reduceTensorOp, reduceTensorCompType, reduceTensorNanOpt, reduceTensorIndices,
                      reduceTensorIndicesType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyReduceTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnReduceTensorDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyReduceTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(reduceTensorDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetReductionIndicesSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetReductionIndicesSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnReduceTensorDescriptor_t,
                                       const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetReductionIndicesSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetReductionWorkspaceSize(
    cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc,
    const cudnnTensorDescriptor_t cDesc, size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetReductionWorkspaceSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnReduceTensorDescriptor_t,
                                       const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetReductionWorkspaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, reduceTensorDesc, aDesc, cDesc, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnReduceTensor(cudnnHandle_t handle,
                                                            const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                            void *indices, size_t indicesSizeInBytes, void *workspace,
                                                            size_t workspaceSizeInBytes, const void *alpha,
                                                            const cudnnTensorDescriptor_t aDesc, const void *A,
                                                            const void *beta, const cudnnTensorDescriptor_t cDesc,
                                                            void *C) {
    HOOK_TRACE_PROFILE("cudnnReduceTensor");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnReduceTensorDescriptor_t, void *, size_t, void *,
                                       size_t, const void *, const cudnnTensorDescriptor_t, const void *, const void *,
                                       const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnReduceTensor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, reduceTensorDesc, indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, alpha,
                      aDesc, A, beta, cDesc, C);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
                                                         void *y, const void *valuePtr) {
    HOOK_TRACE_PROFILE("cudnnSetTensor");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, void *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetTensor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, yDesc, y, valuePtr);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc,
                                                           void *y, const void *alpha) {
    HOOK_TRACE_PROFILE("cudnnScaleTensor");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, void *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnScaleTensor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, yDesc, y, alpha);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateFilterDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnFilterDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateFilterDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(filterDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                                                     cudnnDataType_t dataType,
                                                                     cudnnTensorFormat_t format, int k, int c, int h,
                                                                     int w) {
    HOOK_TRACE_PROFILE("cudnnSetFilter4dDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, int, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetFilter4dDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(filterDesc, dataType, format, k, c, h, w);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetFilter4dDescriptor(const cudnnFilterDescriptor_t filterDesc,
                                                                     cudnnDataType_t *dataType,
                                                                     cudnnTensorFormat_t *format, int *k, int *c,
                                                                     int *h, int *w) {
    HOOK_TRACE_PROFILE("cudnnGetFilter4dDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnFilterDescriptor_t, cudnnDataType_t *, cudnnTensorFormat_t *, int *,
                                       int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetFilter4dDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(filterDesc, dataType, format, k, c, h, w);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc,
                                                                     cudnnDataType_t dataType,
                                                                     cudnnTensorFormat_t format, int nbDims,
                                                                     const int filterDimA) {
    HOOK_TRACE_PROFILE("cudnnSetFilterNdDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, int, const int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetFilterNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(filterDesc, dataType, format, nbDims, filterDimA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetFilterNdDescriptor(const cudnnFilterDescriptor_t filterDesc,
                                                                     int nbDimsRequested, cudnnDataType_t *dataType,
                                                                     cudnnTensorFormat_t *format, int *nbDims,
                                                                     int filterDimA) {
    HOOK_TRACE_PROFILE("cudnnGetFilterNdDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(const cudnnFilterDescriptor_t, int, cudnnDataType_t *, cudnnTensorFormat_t *, int *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetFilterNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(filterDesc, nbDimsRequested, dataType, format, nbDims, filterDimA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc,
                                                                    size_t *size) {
    HOOK_TRACE_PROFILE("cudnnGetFilterSizeInBytes");
    using func_ptr = cudnnStatus_t (*)(const cudnnFilterDescriptor_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetFilterSizeInBytes"));
    HOOK_CHECK(func_entry);
    return func_entry(filterDesc, size);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnTransformFilter(cudnnHandle_t handle,
                                                               const cudnnTensorTransformDescriptor_t transDesc,
                                                               const void *alpha, const cudnnFilterDescriptor_t srcDesc,
                                                               const void *srcData, const void *beta,
                                                               const cudnnFilterDescriptor_t destDesc, void *destData) {
    HOOK_TRACE_PROFILE("cudnnTransformFilter");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorTransformDescriptor_t, const void *,
                                       const cudnnFilterDescriptor_t, const void *, const void *,
                                       const cudnnFilterDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnTransformFilter"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, transDesc, alpha, srcDesc, srcData, beta, destDesc, destData);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyFilterDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnFilterDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyFilterDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(filterDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnReorderFilterAndBias(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, cudnnReorderType_t reorderType,
    const void *filterData, void *reorderedFilterData, int reorderBias, const void *biasData, void *reorderedBiasData) {
    HOOK_TRACE_PROFILE("cudnnReorderFilterAndBias");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnFilterDescriptor_t, cudnnReorderType_t, const void *,
                                       void *, int, const void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnReorderFilterAndBias"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, filterDesc, reorderType, filterData, reorderedFilterData, reorderBias, biasData,
                      reorderedBiasData);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateConvolutionDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateConvolutionDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                                                      cudnnMathType_t mathType) {
    HOOK_TRACE_PROFILE("cudnnSetConvolutionMathType");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t, cudnnMathType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetConvolutionMathType"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, mathType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                                                      cudnnMathType_t *mathType) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionMathType");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t, cudnnMathType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionMathType"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, mathType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                                                                        int groupCount) {
    HOOK_TRACE_PROFILE("cudnnSetConvolutionGroupCount");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetConvolutionGroupCount"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, groupCount);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                                                                        int *groupCount) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionGroupCount");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionGroupCount"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, groupCount);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                                         cudnnReorderType_t reorderType) {
    HOOK_TRACE_PROFILE("cudnnSetConvolutionReorderType");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t, cudnnReorderType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetConvolutionReorderType"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, reorderType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                                         cudnnReorderType_t *reorderType) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionReorderType");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t, cudnnReorderType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionReorderType"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, reorderType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                                                          int pad_h, int pad_w, int u, int v,
                                                                          int dilation_h, int dilation_w,
                                                                          cudnnConvolutionMode_t mode,
                                                                          cudnnDataType_t computeType) {
    HOOK_TRACE_PROFILE("cudnnSetConvolution2dDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t, int, int, int, int, int, int,
                                       cudnnConvolutionMode_t, cudnnDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetConvolution2dDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolution2dDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                                                          int *pad_h, int *pad_w, int *u, int *v,
                                                                          int *dilation_h, int *dilation_w,
                                                                          cudnnConvolutionMode_t *mode,
                                                                          cudnnDataType_t *computeType) {
    HOOK_TRACE_PROFILE("cudnnGetConvolution2dDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnConvolutionDescriptor_t, int *, int *, int *, int *, int *, int *,
                                       cudnnConvolutionMode_t *, cudnnDataType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolution2dDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w) {
    HOOK_TRACE_PROFILE("cudnnGetConvolution2dForwardOutputDim");
    using func_ptr = cudnnStatus_t (*)(const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnFilterDescriptor_t, int *, int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolution2dForwardOutputDim"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, inputTensorDesc, filterDesc, n, c, h, w);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                                                          int arrayLength, const int padA,
                                                                          const int filterStrideA, const int dilationA,
                                                                          cudnnConvolutionMode_t mode,
                                                                          cudnnDataType_t computeType) {
    HOOK_TRACE_PROFILE("cudnnSetConvolutionNdDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t, int, const int, const int, const int,
                                       cudnnConvolutionMode_t, cudnnDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetConvolutionNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, computeType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                                                          int arrayLengthRequested, int *arrayLength,
                                                                          int padA, int strideA, int dilationA,
                                                                          cudnnConvolutionMode_t *mode,
                                                                          cudnnDataType_t *computeType) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionNdDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnConvolutionDescriptor_t, int, int *, int, int, int,
                                       cudnnConvolutionMode_t *, cudnnDataType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, arrayLengthRequested, arrayLength, padA, strideA, dilationA, mode, computeType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc,
    const cudnnFilterDescriptor_t filterDesc, int nbDims, int tensorOuputDimA) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionNdForwardOutputDim");
    using func_ptr = cudnnStatus_t (*)(const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnFilterDescriptor_t, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionNdForwardOutputDim"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc, inputTensorDesc, filterDesc, nbDims, tensorOuputDimA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyConvolutionDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnConvolutionDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyConvolutionDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(convDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle,
                                                                                      int *count) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionForwardAlgorithmMaxCount");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionForwardAlgorithmMaxCount"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    HOOK_TRACE_PROFILE("cudnnFindConvolutionForwardAlgorithm");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, const int,
                                       int *, cudnnConvolutionFwdAlgoPerf_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindConvolutionForwardAlgorithm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void *y,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnFindConvolutionForwardAlgorithmEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const void *, const cudnnFilterDescriptor_t,
                          const void *, const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, void *,
                          const int, int *, cudnnConvolutionFwdAlgoPerf_t *, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindConvolutionForwardAlgorithmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount, returnedAlgoCount,
                      perfResults, workSpace, workSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionFwdAlgo_t *algo) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionForwardAlgorithm");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
                                       cudnnConvolutionFwdPreference_t, size_t, cudnnConvolutionFwdAlgo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionForwardAlgorithm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, wDesc, convDesc, yDesc, preference, memoryLimitInBytes, algo);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnFilterDescriptor_t filterDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t destDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t *perfResults) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionForwardAlgorithm_v7");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, const int,
                                       int *, cudnnConvolutionFwdAlgoPerf_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionForwardAlgorithm_v7"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, srcDesc, filterDesc, convDesc, destDesc, requestedAlgoCount, returnedAlgoCount,
                      perfResults);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionForwardWorkspaceSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
                                       cudnnConvolutionFwdAlgo_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionForwardWorkspaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                                                                  const cudnnTensorDescriptor_t xDesc, const void *x,
                                                                  const cudnnFilterDescriptor_t wDesc, const void *w,
                                                                  const cudnnConvolutionDescriptor_t convDesc,
                                                                  cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                                                                  size_t workSpaceSizeInBytes, const void *beta,
                                                                  const cudnnTensorDescriptor_t yDesc, void *y) {
    HOOK_TRACE_PROFILE("cudnnConvolutionForward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
                                       const cudnnFilterDescriptor_t, const void *, const cudnnConvolutionDescriptor_t,
                                       cudnnConvolutionFwdAlgo_t, void *, size_t, const void *,
                                       const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnConvolutionForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc,
                      y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle, const void *alpha1, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *alpha2,
    const cudnnTensorDescriptor_t zDesc, const void *z, const cudnnTensorDescriptor_t biasDesc, const void *bias,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void *y) {
    HOOK_TRACE_PROFILE("cudnnConvolutionBiasActivationForward");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *, const cudnnFilterDescriptor_t,
        const void *, const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void *, size_t, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnActivationDescriptor_t, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnConvolutionBiasActivationForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, alpha2,
                      zDesc, z, biasDesc, bias, activationDesc, yDesc, y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void *alpha,
                                                                       const cudnnTensorDescriptor_t dyDesc,
                                                                       const void *dy, const void *beta,
                                                                       const cudnnTensorDescriptor_t dbDesc, void *db) {
    HOOK_TRACE_PROFILE("cudnnConvolutionBackwardBias");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
                                       const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnConvolutionBackwardBias"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alpha, dyDesc, dy, beta, dbDesc, db);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle,
                                                                                             int *count) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
    HOOK_TRACE_PROFILE("cudnnFindConvolutionBackwardFilterAlgorithm");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnFilterDescriptor_t, const int,
                                       int *, cudnnConvolutionBwdFilterAlgoPerf_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindConvolutionBackwardFilterAlgorithm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, dyDesc, convDesc, dwDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t dyDesc,
    const void *y, const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnFindConvolutionBackwardFilterAlgorithmEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const cudnnConvolutionDescriptor_t, const cudnnFilterDescriptor_t, void *,
                          const int, int *, cudnnConvolutionBwdFilterAlgoPerf_t *, void *, size_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindConvolutionBackwardFilterAlgorithmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, x, dyDesc, y, convDesc, dwDesc, dw, requestedAlgoCount, returnedAlgoCount,
                      perfResults, workSpace, workSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t dwDesc,
    cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInBytes,
    cudnnConvolutionBwdFilterAlgo_t *algo) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionBackwardFilterAlgorithm");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
                          const cudnnConvolutionDescriptor_t, const cudnnFilterDescriptor_t,
                          cudnnConvolutionBwdFilterPreference_t, size_t, cudnnConvolutionBwdFilterAlgo_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionBackwardFilterAlgorithm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, dyDesc, convDesc, dwDesc, preference, memoryLimitInBytes, algo);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t srcDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t *perfResults) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionBackwardFilterAlgorithm_v7");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnFilterDescriptor_t, const int,
                                       int *, cudnnConvolutionBwdFilterAlgoPerf_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionBackwardFilterAlgorithm_v7"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, srcDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount,
                      perfResults);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnFilterDescriptor_t gradDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionBackwardFilterWorkspaceSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnFilterDescriptor_t,
                                       cudnnConvolutionBwdFilterAlgo_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionBackwardFilterWorkspaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, dyDesc, convDesc, gradDesc, algo, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {
    HOOK_TRACE_PROFILE("cudnnConvolutionBackwardFilter");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const cudnnConvolutionDescriptor_t,
                                       cudnnConvolutionBwdFilterAlgo_t, void *, size_t, const void *,
                                       const cudnnFilterDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnConvolutionBackwardFilter"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta,
                      dwDesc, dw);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle,
                                                                                           int *count) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionBackwardDataAlgorithmMaxCount"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    HOOK_TRACE_PROFILE("cudnnFindConvolutionBackwardDataAlgorithm");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnFilterDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, const int,
                                       int *, cudnnConvolutionBwdDataAlgoPerf_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindConvolutionBackwardDataAlgorithm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, wDesc, dyDesc, convDesc, dxDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t dyDesc,
    const void *dy, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnFindConvolutionBackwardDataAlgorithmEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnFilterDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, void *,
                          const int, int *, cudnnConvolutionBwdDataAlgoPerf_t *, void *, size_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindConvolutionBackwardDataAlgorithmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx, requestedAlgoCount, returnedAlgoCount,
                      perfResults, workSpace, workSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes, cudnnConvolutionBwdDataAlgo_t *algo) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionBackwardDataAlgorithm");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnFilterDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
                                       cudnnConvolutionBwdDataPreference_t, size_t, cudnnConvolutionBwdDataAlgo_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionBackwardDataAlgorithm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, wDesc, dyDesc, convDesc, dxDesc, preference, memoryLimitInBytes, algo);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t filterDesc, const cudnnTensorDescriptor_t diffDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t gradDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionBackwardDataAlgorithm_v7");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnFilterDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, const int,
                                       int *, cudnnConvolutionBwdDataAlgoPerf_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionBackwardDataAlgorithm_v7"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, filterDesc, diffDesc, convDesc, gradDesc, requestedAlgoCount, returnedAlgoCount,
                      perfResults);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc,
    const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc,
    cudnnConvolutionBwdDataAlgo_t algo, size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetConvolutionBackwardDataWorkspaceSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnFilterDescriptor_t, const cudnnTensorDescriptor_t,
                                       const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t,
                                       cudnnConvolutionBwdDataAlgo_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetConvolutionBackwardDataWorkspaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, wDesc, dyDesc, convDesc, dxDesc, algo, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha, const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
    HOOK_TRACE_PROFILE("cudnnConvolutionBackwardData");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const void *, const cudnnFilterDescriptor_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const cudnnConvolutionDescriptor_t,
                                       cudnnConvolutionBwdDataAlgo_t, void *, size_t, const void *,
                                       const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnConvolutionBackwardData"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace, workSpaceSizeInBytes, beta,
                      dxDesc, dx);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnIm2Col(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
                                                      const void *x, const cudnnFilterDescriptor_t wDesc,
                                                      const cudnnConvolutionDescriptor_t convDesc, void *colBuffer) {
    HOOK_TRACE_PROFILE("cudnnIm2Col");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const void *,
                                       const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnIm2Col"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, xDesc, x, wDesc, convDesc, colBuffer);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                                              cudnnSoftmaxMode_t mode, const void *alpha,
                                                              const cudnnTensorDescriptor_t xDesc, const void *x,
                                                              const void *beta, const cudnnTensorDescriptor_t yDesc,
                                                              void *y) {
    HOOK_TRACE_PROFILE("cudnnSoftmaxForward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const void *,
                                       const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSoftmaxForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, mode, alpha, xDesc, x, beta, yDesc, y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo,
                                                               cudnnSoftmaxMode_t mode, const void *alpha,
                                                               const cudnnTensorDescriptor_t yDesc, const void *y,
                                                               const cudnnTensorDescriptor_t dyDesc, const void *dy,
                                                               const void *beta, const cudnnTensorDescriptor_t dxDesc,
                                                               void *dx) {
    HOOK_TRACE_PROFILE("cudnnSoftmaxBackward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                                       const void *, const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSoftmaxBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algo, mode, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc) {
    HOOK_TRACE_PROFILE("cudnnCreatePoolingDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnPoolingDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreatePoolingDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(poolingDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                                      cudnnPoolingMode_t mode,
                                                                      cudnnNanPropagation_t maxpoolingNanOpt,
                                                                      int windowHeight, int windowWidth,
                                                                      int verticalPadding, int horizontalPadding,
                                                                      int verticalStride, int horizontalStride) {
    HOOK_TRACE_PROFILE("cudnnSetPooling2dDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, int, int,
                                       int, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetPooling2dDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding,
                      horizontalPadding, verticalStride, horizontalStride);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                                                                      cudnnPoolingMode_t *mode,
                                                                      cudnnNanPropagation_t *maxpoolingNanOpt,
                                                                      int *windowHeight, int *windowWidth,
                                                                      int *verticalPadding, int *horizontalPadding,
                                                                      int *verticalStride, int *horizontalStride) {
    HOOK_TRACE_PROFILE("cudnnGetPooling2dDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnPoolingDescriptor_t, cudnnPoolingMode_t *, cudnnNanPropagation_t *,
                                       int *, int *, int *, int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetPooling2dDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(poolingDesc, mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding,
                      horizontalPadding, verticalStride, horizontalStride);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                                      const cudnnPoolingMode_t mode,
                                                                      const cudnnNanPropagation_t maxpoolingNanOpt,
                                                                      int nbDims, const int windowDimA,
                                                                      const int paddingA, const int strideA) {
    HOOK_TRACE_PROFILE("cudnnSetPoolingNdDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnPoolingDescriptor_t, const cudnnPoolingMode_t, const cudnnNanPropagation_t,
                                       int, const int, const int, const int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetPoolingNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(poolingDesc, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
                                                                      int nbDimsRequested, cudnnPoolingMode_t *mode,
                                                                      cudnnNanPropagation_t *maxpoolingNanOpt,
                                                                      int *nbDims, int windowDimA, int paddingA,
                                                                      int strideA) {
    HOOK_TRACE_PROFILE("cudnnGetPoolingNdDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnPoolingDescriptor_t, int, cudnnPoolingMode_t *,
                                       cudnnNanPropagation_t *, int *, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetPoolingNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(poolingDesc, nbDimsRequested, mode, maxpoolingNanOpt, nbDims, windowDimA, paddingA, strideA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc, int nbDims, int outputTensorDimA) {
    HOOK_TRACE_PROFILE("cudnnGetPoolingNdForwardOutputDim");
    using func_ptr = cudnnStatus_t (*)(const cudnnPoolingDescriptor_t, const cudnnTensorDescriptor_t, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetPoolingNdForwardOutputDim"));
    HOOK_CHECK(func_entry);
    return func_entry(poolingDesc, inputTensorDesc, nbDims, outputTensorDimA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                      const cudnnTensorDescriptor_t inputTensorDesc, int *n, int *c, int *h, int *w) {
    HOOK_TRACE_PROFILE("cudnnGetPooling2dForwardOutputDim");
    using func_ptr =
        cudnnStatus_t (*)(const cudnnPoolingDescriptor_t, const cudnnTensorDescriptor_t, int *, int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetPooling2dForwardOutputDim"));
    HOOK_CHECK(func_entry);
    return func_entry(poolingDesc, inputTensorDesc, n, c, h, w);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyPoolingDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnPoolingDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyPoolingDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(poolingDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle,
                                                              const cudnnPoolingDescriptor_t poolingDesc,
                                                              const void *alpha, const cudnnTensorDescriptor_t xDesc,
                                                              const void *x, const void *beta,
                                                              const cudnnTensorDescriptor_t yDesc, void *y) {
    HOOK_TRACE_PROFILE("cudnnPoolingForward");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnPoolingDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnPoolingForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t handle,
                                                               const cudnnPoolingDescriptor_t poolingDesc,
                                                               const void *alpha, const cudnnTensorDescriptor_t yDesc,
                                                               const void *y, const cudnnTensorDescriptor_t dyDesc,
                                                               const void *dy, const cudnnTensorDescriptor_t xDesc,
                                                               const void *x, const void *beta,
                                                               const cudnnTensorDescriptor_t dxDesc, void *dx) {
    HOOK_TRACE_PROFILE("cudnnPoolingBackward");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnPoolingDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnPoolingBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateActivationDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnActivationDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateActivationDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(activationDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                                                       cudnnActivationMode_t mode,
                                                                       cudnnNanPropagation_t reluNanOpt, double coef) {
    HOOK_TRACE_PROFILE("cudnnSetActivationDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnNanPropagation_t, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetActivationDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(activationDesc, mode, reluNanOpt, coef);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                                                                       cudnnActivationMode_t *mode,
                                                                       cudnnNanPropagation_t *reluNanOpt,
                                                                       double *coef) {
    HOOK_TRACE_PROFILE("cudnnGetActivationDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnActivationDescriptor_t, cudnnActivationMode_t *,
                                       cudnnNanPropagation_t *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetActivationDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(activationDesc, mode, reluNanOpt, coef);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyActivationDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnActivationDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyActivationDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(activationDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle,
                                                                 cudnnActivationDescriptor_t activationDesc,
                                                                 const void *alpha, const cudnnTensorDescriptor_t xDesc,
                                                                 const void *x, const void *beta,
                                                                 const cudnnTensorDescriptor_t yDesc, void *y) {
    HOOK_TRACE_PROFILE("cudnnActivationForward");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, cudnnActivationDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnActivationForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnActivationBackward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void *alpha,
                            const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc,
                            const void *dy, const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
                            const cudnnTensorDescriptor_t dxDesc, void *dx) {
    HOOK_TRACE_PROFILE("cudnnActivationBackward");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, cudnnActivationDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnActivationBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateLRNDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnLRNDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateLRNDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(normDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN,
                                                                double lrnAlpha, double lrnBeta, double lrnK) {
    HOOK_TRACE_PROFILE("cudnnSetLRNDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnLRNDescriptor_t, unsigned, double, double, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetLRNDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned *lrnN,
                                                                double *lrnAlpha, double *lrnBeta, double *lrnK) {
    HOOK_TRACE_PROFILE("cudnnGetLRNDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnLRNDescriptor_t, unsigned *, double *, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetLRNDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyLRNDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnLRNDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyLRNDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(lrnDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnLRNCrossChannelForward(cudnnHandle_t handle,
                                                                      cudnnLRNDescriptor_t normDesc,
                                                                      cudnnLRNMode_t lrnMode, const void *alpha,
                                                                      const cudnnTensorDescriptor_t xDesc,
                                                                      const void *x, const void *beta,
                                                                      const cudnnTensorDescriptor_t yDesc, void *y) {
    HOOK_TRACE_PROFILE("cudnnLRNCrossChannelForward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const void *,
                                       const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnLRNCrossChannelForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, normDesc, lrnMode, alpha, xDesc, x, beta, yDesc, y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnLRNCrossChannelBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnLRNMode_t lrnMode, const void *alpha,
    const cudnnTensorDescriptor_t yDesc, const void *y, const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx) {
    HOOK_TRACE_PROFILE("cudnnLRNCrossChannelBackward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnLRNDescriptor_t, cudnnLRNMode_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                                       const void *, const cudnnTensorDescriptor_t, const void *, const void *,
                                       const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnLRNCrossChannelBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDivisiveNormalizationForward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *means, void *temp, void *temp2, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y) {
    HOOK_TRACE_PROFILE("cudnnDivisiveNormalizationForward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const void *, void *, void *,
                                       const void *, const cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDivisiveNormalizationForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, normDesc, mode, alpha, xDesc, x, means, temp, temp2, beta, yDesc, y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDivisiveNormalizationBackward(
    cudnnHandle_t handle, cudnnLRNDescriptor_t normDesc, cudnnDivNormMode_t mode, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *means, const void *dy, void *temp, void *temp2,
    const void *beta, const cudnnTensorDescriptor_t dXdMeansDesc, void *dx, void *dMeans) {
    HOOK_TRACE_PROFILE("cudnnDivisiveNormalizationBackward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnLRNDescriptor_t, cudnnDivNormMode_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const void *, const void *, void *,
                                       void *, const void *, const cudnnTensorDescriptor_t, void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDivisiveNormalizationBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, normDesc, mode, alpha, xDesc, x, means, dy, temp, temp2, beta, dXdMeansDesc, dx, dMeans);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
                                                                        const cudnnTensorDescriptor_t xDesc,
                                                                        cudnnBatchNormMode_t mode) {
    HOOK_TRACE_PROFILE("cudnnDeriveBNTensorDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, cudnnBatchNormMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDeriveBNTensorDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(derivedBnDesc, xDesc, mode);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t zDesc, const cudnnTensorDescriptor_t yDesc,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const cudnnActivationDescriptor_t activationDesc,
    size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const cudnnTensorDescriptor_t,
                          const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
                          const cudnnActivationDescriptor_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, bnOps, xDesc, zDesc, yDesc, bnScaleBiasMeanVarDesc, activationDesc, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const cudnnTensorDescriptor_t xDesc,
    const cudnnTensorDescriptor_t yDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnTensorDescriptor_t dzDesc,
    const cudnnTensorDescriptor_t dxDesc, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const cudnnActivationDescriptor_t activationDesc, size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetBatchNormalizationBackwardExWorkspaceSize");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const cudnnTensorDescriptor_t,
        const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
        const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnActivationDescriptor_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetBatchNormalizationBackwardExWorkspaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, bnOps, xDesc, yDesc, dyDesc, dzDesc, dxDesc, dBnScaleBiasDesc, activationDesc,
                      sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps,
    const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
                                       const cudnnActivationDescriptor_t, const cudnnTensorDescriptor_t, size_t *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetBatchNormalizationTrainingExReserveSpaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, bnOps, activationDesc, xDesc, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
    double exponentialAverageFactor, void *resultRunningMean, void *resultRunningVariance, double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance) {
    HOOK_TRACE_PROFILE("cudnnBatchNormalizationForwardTraining");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, const void *, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                                       void *, const cudnnTensorDescriptor_t, const void *, const void *, double,
                                       void *, void *, double, void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnBatchNormalizationForwardTraining"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias,
                      exponentialAverageFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean,
                      resultSaveInvVariance);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc, const void *zData,
    const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void *bnScale, const void *bnBias, double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon, void *resultSaveMean, void *resultSaveInvVariance,
    cudnnActivationDescriptor_t activationDesc, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnBatchNormalizationForwardTrainingEx");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const void *, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, const void *, const void *, double,
        void *, void *, double, void *, void *, cudnnActivationDescriptor_t, void *, size_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnBatchNormalizationForwardTrainingEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, bnOps, alpha, beta, xDesc, xData, zDesc, zData, yDesc, yData,
                      bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor, resultRunningMean,
                      resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance, activationDesc, workspace,
                      workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias,
    const void *estimatedMean, const void *estimatedVariance, double epsilon) {
    HOOK_TRACE_PROFILE("cudnnBatchNormalizationForwardInference");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, const void *, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                                       void *, const cudnnTensorDescriptor_t, const void *, const void *, const void *,
                                       const void *, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnBatchNormalizationForwardInference"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias,
                      estimatedMean, estimatedVariance, epsilon);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnBatchNormalizationBackward(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff, const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const cudnnTensorDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void *bnScale, void *dBnScaleResult, void *dBnBiasResult,
    double epsilon, const void *savedMean, const void *savedInvVariance) {
    HOOK_TRACE_PROFILE("cudnnBatchNormalizationBackward");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, cudnnBatchNormMode_t, const void *, const void *, const void *, const void *,
                          const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
                          const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, const void *, void *,
                          void *, double, const void *, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnBatchNormalizationBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy,
                      dxDesc, dx, dBnScaleBiasDesc, bnScale, dBnScaleResult, dBnBiasResult, epsilon, savedMean,
                      savedInvVariance);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alphaDataDiff,
    const void *betaDataDiff, const void *alphaParamDiff, const void *betaParamDiff,
    const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t yDesc, const void *yData,
    const cudnnTensorDescriptor_t dyDesc, const void *dyData, const cudnnTensorDescriptor_t dzDesc, void *dzData,
    const cudnnTensorDescriptor_t dxDesc, void *dxData, const cudnnTensorDescriptor_t dBnScaleBiasDesc,
    const void *bnScaleData, const void *bnBiasData, void *dBnScaleData, void *dBnBiasData, double epsilon,
    const void *savedMean, const void *savedInvVariance, cudnnActivationDescriptor_t activationDesc, void *workSpace,
    size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnBatchNormalizationBackwardEx");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const void *, const void *, const void *,
        const void *, const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, void *,
        const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, const void *, const void *, void *,
        void *, double, const void *, const void *, cudnnActivationDescriptor_t, void *, size_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnBatchNormalizationBackwardEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, mode, bnOps, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, xData,
                      yDesc, yData, dyDesc, dyData, dzDesc, dzData, dxDesc, dxData, dBnScaleBiasDesc, bnScaleData,
                      bnBiasData, dBnScaleData, dBnBiasData, epsilon, savedMean, savedInvVariance, activationDesc,
                      workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateSpatialTransformerDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnSpatialTransformerDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateSpatialTransformerDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(stDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc, cudnnSamplerType_t samplerType,
                                           cudnnDataType_t dataType, const int nbDims, const int dimA) {
    HOOK_TRACE_PROFILE("cudnnSetSpatialTransformerNdDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnSpatialTransformerDescriptor_t, cudnnSamplerType_t, cudnnDataType_t,
                                       const int, const int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetSpatialTransformerNdDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(stDesc, samplerType, dataType, nbDims, dimA);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroySpatialTransformerDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnSpatialTransformerDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroySpatialTransformerDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(stDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSpatialTfGridGeneratorForward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void *theta, void *grid) {
    HOOK_TRACE_PROFILE("cudnnSpatialTfGridGeneratorForward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnSpatialTransformerDescriptor_t, const void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSpatialTfGridGeneratorForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, stDesc, theta, grid);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(
    cudnnHandle_t handle, const cudnnSpatialTransformerDescriptor_t stDesc, const void *dgrid, void *dtheta) {
    HOOK_TRACE_PROFILE("cudnnSpatialTfGridGeneratorBackward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnSpatialTransformerDescriptor_t, const void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSpatialTfGridGeneratorBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, stDesc, dgrid, dtheta);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnSpatialTfSamplerForward(cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha,
                                 const cudnnTensorDescriptor_t xDesc, const void *x, const void *grid, const void *beta,
                                 cudnnTensorDescriptor_t yDesc, void *y) {
    HOOK_TRACE_PROFILE("cudnnSpatialTfSamplerForward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const void *, const void *,
                                       cudnnTensorDescriptor_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSpatialTfSamplerForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, stDesc, alpha, xDesc, x, grid, beta, yDesc, y);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSpatialTfSamplerBackward(
    cudnnHandle_t handle, cudnnSpatialTransformerDescriptor_t stDesc, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta, const cudnnTensorDescriptor_t dxDesc,
    void *dx, const void *alphaDgrid, const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *grid,
    const void *betaDgrid, void *dgrid) {
    HOOK_TRACE_PROFILE("cudnnSpatialTfSamplerBackward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnSpatialTransformerDescriptor_t, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const void *,
                                       const cudnnTensorDescriptor_t, void *, const void *,
                                       const cudnnTensorDescriptor_t, const void *, const void *, const void *, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSpatialTfSamplerBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, stDesc, alpha, xDesc, x, beta, dxDesc, dx, alphaDgrid, dyDesc, dy, grid, betaDgrid,
                      dgrid);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateDropoutDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnDropoutDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateDropoutDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(dropoutDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyDropoutDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnDropoutDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyDropoutDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(dropoutDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnDropoutGetStatesSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDropoutGetStatesSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc,
                                                                          size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnDropoutGetReserveSpaceSize");
    using func_ptr = cudnnStatus_t (*)(cudnnTensorDescriptor_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDropoutGetReserveSpaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(xdesc, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                                    cudnnHandle_t handle, float dropout, void *states,
                                                                    size_t stateSizeInBytes, unsigned long long seed) {
    HOOK_TRACE_PROFILE("cudnnSetDropoutDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(cudnnDropoutDescriptor_t, cudnnHandle_t, float, void *, size_t, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetDropoutDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                                        cudnnHandle_t handle, float dropout,
                                                                        void *states, size_t stateSizeInBytes,
                                                                        unsigned long long seed) {
    HOOK_TRACE_PROFILE("cudnnRestoreDropoutDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(cudnnDropoutDescriptor_t, cudnnHandle_t, float, void *, size_t, unsigned long long);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRestoreDropoutDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(dropoutDesc, handle, dropout, states, stateSizeInBytes, seed);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                                                    cudnnHandle_t handle, float *dropout, void **states,
                                                                    unsigned long long *seed) {
    HOOK_TRACE_PROFILE("cudnnGetDropoutDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnDropoutDescriptor_t, cudnnHandle_t, float *, void **, unsigned long long *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetDropoutDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(dropoutDesc, handle, dropout, states, seed);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDropoutForward(cudnnHandle_t handle,
                                                              const cudnnDropoutDescriptor_t dropoutDesc,
                                                              const cudnnTensorDescriptor_t xdesc, const void *x,
                                                              const cudnnTensorDescriptor_t ydesc, void *y,
                                                              void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnDropoutForward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnDropoutDescriptor_t, const cudnnTensorDescriptor_t,
                                       const void *, const cudnnTensorDescriptor_t, void *, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDropoutForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dropoutDesc, xdesc, x, ydesc, y, reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t handle,
                                                               const cudnnDropoutDescriptor_t dropoutDesc,
                                                               const cudnnTensorDescriptor_t dydesc, const void *dy,
                                                               const cudnnTensorDescriptor_t dxdesc, void *dx,
                                                               void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnDropoutBackward");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnDropoutDescriptor_t, const cudnnTensorDescriptor_t,
                                       const void *, const cudnnTensorDescriptor_t, void *, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDropoutBackward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, dropoutDesc, dydesc, dy, dxdesc, dx, reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateRNNDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateRNNDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyRNNDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyRNNDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                                const int hiddenSize, const int numLayers,
                                                                cudnnDropoutDescriptor_t dropoutDesc,
                                                                cudnnRNNInputMode_t inputMode,
                                                                cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                                cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
    HOOK_TRACE_PROFILE("cudnnSetRNNDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, cudnnRNNDescriptor_t, const int, const int, cudnnDropoutDescriptor_t,
                          cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                                int *hiddenSize, int *numLayers,
                                                                cudnnDropoutDescriptor_t *dropoutDesc,
                                                                cudnnRNNInputMode_t *inputMode,
                                                                cudnnDirectionMode_t *direction, cudnnRNNMode_t *mode,
                                                                cudnnRNNAlgo_t *algo, cudnnDataType_t *mathPrec) {
    HOOK_TRACE_PROFILE("cudnnGetRNNDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnRNNDescriptor_t, int *, int *, cudnnDropoutDescriptor_t *,
                                       cudnnRNNInputMode_t *, cudnnDirectionMode_t *, cudnnRNNMode_t *,
                                       cudnnRNNAlgo_t *, cudnnDataType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                                                    cudnnMathType_t mType) {
    HOOK_TRACE_PROFILE("cudnnSetRNNMatrixMathType");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t, cudnnMathType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNMatrixMathType"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, mType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc,
                                                                    cudnnMathType_t *mType) {
    HOOK_TRACE_PROFILE("cudnnGetRNNMatrixMathType");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t, cudnnMathType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNMatrixMathType"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, mType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc,
                                                              cudnnRNNBiasMode_t biasMode) {
    HOOK_TRACE_PROFILE("cudnnSetRNNBiasMode");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t, cudnnRNNBiasMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNBiasMode"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, biasMode);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc,
                                                              cudnnRNNBiasMode_t *biasMode) {
    HOOK_TRACE_PROFILE("cudnnGetRNNBiasMode");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t, cudnnRNNBiasMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNBiasMode"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, biasMode);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNSetClip(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                          cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt,
                                                          double lclip, double rclip) {
    HOOK_TRACE_PROFILE("cudnnRNNSetClip");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNClipMode_t, cudnnNanPropagation_t,
                                       double, double);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNSetClip"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNGetClip(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                          cudnnRNNClipMode_t *clipMode,
                                                          cudnnNanPropagation_t *clipNanOpt, double *lclip,
                                                          double *rclip) {
    HOOK_TRACE_PROFILE("cudnnRNNGetClip");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnRNNDescriptor_t, cudnnRNNClipMode_t *,
                                       cudnnNanPropagation_t *, double *, double *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNGetClip"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, clipMode, clipNanOpt, lclip, rclip);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                                                                      cudnnRNNDescriptor_t rnnDesc,
                                                                      const int recProjSize, const int outProjSize) {
    HOOK_TRACE_PROFILE("cudnnSetRNNProjectionLayers");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnRNNDescriptor_t, const int, const int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNProjectionLayers"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, recProjSize, outProjSize);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                                                                      const cudnnRNNDescriptor_t rnnDesc,
                                                                      int *recProjSize, int *outProjSize) {
    HOOK_TRACE_PROFILE("cudnnGetRNNProjectionLayers");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNProjectionLayers"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, recProjSize, outProjSize);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                                       const int minibatch,
                                                                       const cudnnDataType_t dataType,
                                                                       cudnnPersistentRNNPlan_t *plan) {
    HOOK_TRACE_PROFILE("cudnnCreatePersistentRNNPlan");
    using func_ptr =
        cudnnStatus_t (*)(cudnnRNNDescriptor_t, const int, const cudnnDataType_t, cudnnPersistentRNNPlan_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreatePersistentRNNPlan"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, minibatch, dataType, plan);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan) {
    HOOK_TRACE_PROFILE("cudnnDestroyPersistentRNNPlan");
    using func_ptr = cudnnStatus_t (*)(cudnnPersistentRNNPlan_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyPersistentRNNPlan"));
    HOOK_CHECK(func_entry);
    return func_entry(plan);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                                                                    cudnnPersistentRNNPlan_t plan) {
    HOOK_TRACE_PROFILE("cudnnSetPersistentRNNPlan");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t, cudnnPersistentRNNPlan_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetPersistentRNNPlan"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, plan);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                                                                   const cudnnRNNDescriptor_t rnnDesc,
                                                                   const int seqLength,
                                                                   const cudnnTensorDescriptor_t *xDesc,
                                                                   size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetRNNWorkspaceSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
                                       const cudnnTensorDescriptor_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNWorkspaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
                                                                         const cudnnRNNDescriptor_t rnnDesc,
                                                                         const int seqLength,
                                                                         const cudnnTensorDescriptor_t *xDesc,
                                                                         size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetRNNTrainingReserveSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
                                       const cudnnTensorDescriptor_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNTrainingReserveSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, xDesc, sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t handle,
                                                                const cudnnRNNDescriptor_t rnnDesc,
                                                                const cudnnTensorDescriptor_t xDesc,
                                                                size_t *sizeInBytes, cudnnDataType_t dataType) {
    HOOK_TRACE_PROFILE("cudnnGetRNNParamsSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnTensorDescriptor_t,
                                       size_t *, cudnnDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNParamsSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, xDesc, sizeInBytes, dataType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cudnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat) {
    HOOK_TRACE_PROFILE("cudnnGetRNNLinLayerMatrixParams");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t,
                          const cudnnFilterDescriptor_t, const void *, const int, cudnnFilterDescriptor_t, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNLinLayerMatrixParams"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerMatDesc, linLayerMat);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNLinLayerBiasParams(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int pseudoLayer,
    const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const void *w, const int linLayerID,
    cudnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias) {
    HOOK_TRACE_PROFILE("cudnnGetRNNLinLayerBiasParams");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t,
                          const cudnnFilterDescriptor_t, const void *, const int, cudnnFilterDescriptor_t, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNLinLayerBiasParams"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, pseudoLayer, xDesc, wDesc, w, linLayerID, linLayerBiasDesc, linLayerBias);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNForwardInference(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnRNNForwardInference");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t *, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnFilterDescriptor_t, const void *, const cudnnTensorDescriptor_t *, void *,
        const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, void *, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNForwardInference"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
                      cyDesc, cy, workspace, workSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNForwardTraining(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnRNNForwardTraining");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t *, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnFilterDescriptor_t, const void *, const cudnnTensorDescriptor_t *, void *,
        const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, void *, void *, size_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNForwardTraining"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
                      cyDesc, cy, workspace, workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNBackwardData(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnRNNBackwardData");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t *,
                          const void *, const cudnnTensorDescriptor_t *, const void *, const cudnnTensorDescriptor_t,
                          const void *, const cudnnTensorDescriptor_t, const void *, const cudnnFilterDescriptor_t,
                          const void *, const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const cudnnTensorDescriptor_t *, void *, const cudnnTensorDescriptor_t, void *,
                          const cudnnTensorDescriptor_t, void *, void *, size_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNBackwardData"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc,
                      hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workSpaceSizeInBytes,
                      reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNBackwardWeights(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const void *workspace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw,
    const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnRNNBackwardWeights");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int,
                                       const cudnnTensorDescriptor_t *, const void *, const cudnnTensorDescriptor_t,
                                       const void *, const cudnnTensorDescriptor_t *, const void *, const void *,
                                       size_t, const cudnnFilterDescriptor_t, void *, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNBackwardWeights"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace, workSpaceSizeInBytes,
                      dwDesc, dw, reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc,
                                                                 cudnnRNNPaddingMode_t paddingMode) {
    HOOK_TRACE_PROFILE("cudnnSetRNNPaddingMode");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t, cudnnRNNPaddingMode_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNPaddingMode"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, paddingMode);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc,
                                                                 cudnnRNNPaddingMode_t *paddingMode) {
    HOOK_TRACE_PROFILE("cudnnGetRNNPaddingMode");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t, cudnnRNNPaddingMode_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNPaddingMode"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, paddingMode);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateRNNDataDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDataDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateRNNDataDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDataDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyRNNDataDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDataDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyRNNDataDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDataDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                                                                    cudnnDataType_t dataType,
                                                                    cudnnRNNDataLayout_t layout, int maxSeqLength,
                                                                    int batchSize, int vectorSize,
                                                                    const int seqLengthArray, void *paddingFill) {
    HOOK_TRACE_PROFILE("cudnnSetRNNDataDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDataDescriptor_t, cudnnDataType_t, cudnnRNNDataLayout_t, int, int, int,
                                       const int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNDataDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNDataDescriptor(
    cudnnRNNDataDescriptor_t rnnDataDesc, cudnnDataType_t *dataType, cudnnRNNDataLayout_t *layout, int *maxSeqLength,
    int *batchSize, int *vectorSize, int arrayLengthRequested, int seqLengthArray, void *paddingFill) {
    HOOK_TRACE_PROFILE("cudnnGetRNNDataDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDataDescriptor_t, cudnnDataType_t *, cudnnRNNDataLayout_t *, int *,
                                       int *, int *, int, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNDataDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDataDesc, dataType, layout, maxSeqLength, batchSize, vectorSize, arrayLengthRequested,
                      seqLengthArray, paddingFill);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNForwardTrainingEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys, const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn, const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnRNNForwardTrainingEx");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnRNNDataDescriptor_t, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnFilterDescriptor_t, const void *, const cudnnRNNDataDescriptor_t, void *,
        const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, void *, const cudnnRNNDataDescriptor_t,
        const void *, const cudnnRNNDataDescriptor_t, void *, const cudnnRNNDataDescriptor_t, void *,
        const cudnnRNNDataDescriptor_t, void *, void *, size_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNForwardTrainingEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
                      kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes,
                      reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNForwardInferenceEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc, const void *cx,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnRNNDataDescriptor_t yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const cudnnRNNDataDescriptor_t kDesc, const void *keys, const cudnnRNNDataDescriptor_t cDesc, void *cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void *iAttn, const cudnnRNNDataDescriptor_t qDesc, void *queries,
    void *workSpace, size_t workSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnRNNForwardInferenceEx");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnRNNDataDescriptor_t, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnFilterDescriptor_t, const void *, const cudnnRNNDataDescriptor_t, void *,
        const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, void *, const cudnnRNNDataDescriptor_t,
        const void *, const cudnnRNNDataDescriptor_t, void *, const cudnnRNNDataDescriptor_t, void *,
        const cudnnRNNDataDescriptor_t, void *, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNForwardInferenceEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
                      kDesc, keys, cDesc, cAttn, iDesc, iAttn, qDesc, queries, workSpace, workSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNBackwardDataEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t yDesc, const void *y,
    const cudnnRNNDataDescriptor_t dyDesc, const void *dy, const cudnnRNNDataDescriptor_t dcDesc, const void *dcAttn,
    const cudnnTensorDescriptor_t dhyDesc, const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy,
    const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx,
    const cudnnTensorDescriptor_t cxDesc, const void *cx, const cudnnRNNDataDescriptor_t dxDesc, void *dx,
    const cudnnTensorDescriptor_t dhxDesc, void *dhx, const cudnnTensorDescriptor_t dcxDesc, void *dcx,
    const cudnnRNNDataDescriptor_t dkDesc, void *dkeys, void *workSpace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnRNNBackwardDataEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnRNNDataDescriptor_t, const void *,
                          const cudnnRNNDataDescriptor_t, const void *, const cudnnRNNDataDescriptor_t, const void *,
                          const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
                          const cudnnFilterDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
                          const cudnnTensorDescriptor_t, const void *, const cudnnRNNDataDescriptor_t, void *,
                          const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, void *,
                          const cudnnRNNDataDescriptor_t, void *, void *, size_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNBackwardDataEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, yDesc, y, dyDesc, dy, dcDesc, dcAttn, dhyDesc, dhy, dcyDesc, dcy, wDesc, w,
                      hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, dkDesc, dkeys, workSpace,
                      workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRNNBackwardWeightsEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnRNNDataDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnRNNDataDescriptor_t yDesc, const void *y,
    void *workSpace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void *dw, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnRNNBackwardWeightsEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const cudnnRNNDataDescriptor_t, const void *,
                          const cudnnTensorDescriptor_t, const void *, const cudnnRNNDataDescriptor_t, const void *,
                          void *, size_t, const cudnnFilterDescriptor_t, void *, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRNNBackwardWeightsEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, xDesc, x, hxDesc, hx, yDesc, y, workSpace, workSpaceSizeInBytes, dwDesc, dw,
                      reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle,
                                                                         cudnnRNNDescriptor_t rnnDesc,
                                                                         cudnnAlgorithmDescriptor_t algoDesc) {
    HOOK_TRACE_PROFILE("cudnnSetRNNAlgorithmDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnRNNDescriptor_t, cudnnAlgorithmDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNAlgorithmDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, algoDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    HOOK_TRACE_PROFILE("cudnnGetRNNForwardInferenceAlgorithmMaxCount");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNForwardInferenceAlgorithmMaxCount"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnFindRNNForwardInferenceAlgorithmEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t *,
                          const void *, const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const cudnnFilterDescriptor_t, const void *, const cudnnTensorDescriptor_t *,
                          void *, const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, void *,
                          const float, const int, int *, cudnnAlgorithmPerformance_t *, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindRNNForwardInferenceAlgorithmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
                      cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
                      workSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    HOOK_TRACE_PROFILE("cudnnGetRNNForwardTrainingAlgorithmMaxCount");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNForwardTrainingAlgorithmMaxCount"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnTensorDescriptor_t *yDesc, void *y,
    const cudnnTensorDescriptor_t hyDesc, void *hy, const cudnnTensorDescriptor_t cyDesc, void *cy,
    const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnFindRNNForwardTrainingAlgorithmEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t *,
                          const void *, const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t,
                          const void *, const cudnnFilterDescriptor_t, const void *, const cudnnTensorDescriptor_t *,
                          void *, const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t, void *,
                          const float, const int, int *, cudnnAlgorithmPerformance_t *, void *, size_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindRNNForwardTrainingAlgorithmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
                      cyDesc, cy, findIntensity, requestedAlgoCount, returnedAlgoCount, perfResults, workspace,
                      workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle,
                                                                                   const cudnnRNNDescriptor_t rnnDesc,
                                                                                   int *count) {
    HOOK_TRACE_PROFILE("cudnnGetRNNBackwardDataAlgorithmMaxCount");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNBackwardDataAlgorithmMaxCount"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const cudnnTensorDescriptor_t *dyDesc, const void *dy, const cudnnTensorDescriptor_t dhyDesc,
    const void *dhy, const cudnnTensorDescriptor_t dcyDesc, const void *dcy, const cudnnFilterDescriptor_t wDesc,
    const void *w, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t cxDesc,
    const void *cx, const cudnnTensorDescriptor_t *dxDesc, void *dx, const cudnnTensorDescriptor_t dhxDesc, void *dhx,
    const cudnnTensorDescriptor_t dcxDesc, void *dcx, const float findIntensity, const int requestedAlgoCount,
    int *returnedAlgoCount, cudnnAlgorithmPerformance_t *perfResults, void *workspace, size_t workSpaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnFindRNNBackwardDataAlgorithmEx");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t *, const void *,
        const cudnnTensorDescriptor_t *, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnFilterDescriptor_t, const void *,
        const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t, const void *,
        const cudnnTensorDescriptor_t *, void *, const cudnnTensorDescriptor_t, void *, const cudnnTensorDescriptor_t,
        void *, const float, const int, int *, cudnnAlgorithmPerformance_t *, void *, size_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindRNNBackwardDataAlgorithmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc,
                      hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, findIntensity, requestedAlgoCount,
                      returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, reserveSpace,
                      reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count) {
    HOOK_TRACE_PROFILE("cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, int *);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetRNNBackwardWeightsAlgorithmMaxCount"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, count);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t *xDesc,
    const void *x, const cudnnTensorDescriptor_t hxDesc, const void *hx, const cudnnTensorDescriptor_t *yDesc,
    const void *y, const float findIntensity, const int requestedAlgoCount, int *returnedAlgoCount,
    cudnnAlgorithmPerformance_t *perfResults, const void *workspace, size_t workSpaceSizeInBytes,
    const cudnnFilterDescriptor_t dwDesc, void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnFindRNNBackwardWeightsAlgorithmEx");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnRNNDescriptor_t, const int, const cudnnTensorDescriptor_t *,
                          const void *, const cudnnTensorDescriptor_t, const void *, const cudnnTensorDescriptor_t *,
                          const void *, const float, const int, int *, cudnnAlgorithmPerformance_t *, const void *,
                          size_t, const cudnnFilterDescriptor_t, void *, const void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFindRNNBackwardWeightsAlgorithmEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, findIntensity, requestedAlgoCount,
                      returnedAlgoCount, perfResults, workspace, workSpaceSizeInBytes, dwDesc, dw, reserveSpace,
                      reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateSeqDataDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnSeqDataDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateSeqDataDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(seqDataDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroySeqDataDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnSeqDataDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroySeqDataDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(seqDataDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc,
                                                                    cudnnDataType_t dataType, int nbDims,
                                                                    const int dimA, const cudnnSeqDataAxis_t axes,
                                                                    size_t seqLengthArraySize, const int seqLengthArray,
                                                                    void *paddingFill) {
    HOOK_TRACE_PROFILE("cudnnSetSeqDataDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnSeqDataDescriptor_t, cudnnDataType_t, int, const int,
                                       const cudnnSeqDataAxis_t, size_t, const int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetSeqDataDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(seqDataDesc, dataType, nbDims, dimA, axes, seqLengthArraySize, seqLengthArray, paddingFill);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,
                                                                    cudnnDataType_t *dataType, int *nbDims,
                                                                    int nbDimsRequested, int dimA,
                                                                    cudnnSeqDataAxis_t axes, size_t *seqLengthArraySize,
                                                                    size_t seqLengthSizeRequested, int seqLengthArray,
                                                                    void *paddingFill) {
    HOOK_TRACE_PROFILE("cudnnGetSeqDataDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnSeqDataDescriptor_t, cudnnDataType_t *, int *, int, int,
                                       cudnnSeqDataAxis_t, size_t *, size_t, int, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetSeqDataDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(seqDataDesc, dataType, nbDims, nbDimsRequested, dimA, axes, seqLengthArraySize,
                      seqLengthSizeRequested, seqLengthArray, paddingFill);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateAttnDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnAttnDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateAttnDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(attnDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyAttnDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnAttnDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyAttnDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(attnDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned attnMode, int nHeads, double smScaler, cudnnDataType_t dataType,
    cudnnDataType_t computePrec, cudnnMathType_t mathType, cudnnDropoutDescriptor_t attnDropoutDesc,
    cudnnDropoutDescriptor_t postDropoutDesc, int qSize, int kSize, int vSize, int qProjSize, int kProjSize,
    int vProjSize, int oProjSize, int qoMaxSeqLength, int kvMaxSeqLength, int maxBatchSize, int maxBeamSize) {
    HOOK_TRACE_PROFILE("cudnnSetAttnDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnAttnDescriptor_t, unsigned, int, double, cudnnDataType_t, cudnnDataType_t,
                                       cudnnMathType_t, cudnnDropoutDescriptor_t, cudnnDropoutDescriptor_t, int, int,
                                       int, int, int, int, int, int, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetAttnDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc,
                      postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
                      kvMaxSeqLength, maxBatchSize, maxBeamSize);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetAttnDescriptor(
    cudnnAttnDescriptor_t attnDesc, unsigned *attnMode, int *nHeads, double *smScaler, cudnnDataType_t *dataType,
    cudnnDataType_t *computePrec, cudnnMathType_t *mathType, cudnnDropoutDescriptor_t *attnDropoutDesc,
    cudnnDropoutDescriptor_t *postDropoutDesc, int *qSize, int *kSize, int *vSize, int *qProjSize, int *kProjSize,
    int *vProjSize, int *oProjSize, int *qoMaxSeqLength, int *kvMaxSeqLength, int *maxBatchSize, int *maxBeamSize) {
    HOOK_TRACE_PROFILE("cudnnGetAttnDescriptor");
    using func_ptr =
        cudnnStatus_t (*)(cudnnAttnDescriptor_t, unsigned *, int *, double *, cudnnDataType_t *, cudnnDataType_t *,
                          cudnnMathType_t *, cudnnDropoutDescriptor_t *, cudnnDropoutDescriptor_t *, int *, int *,
                          int *, int *, int *, int *, int *, int *, int *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetAttnDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(attnDesc, attnMode, nHeads, smScaler, dataType, computePrec, mathType, attnDropoutDesc,
                      postDropoutDesc, qSize, kSize, vSize, qProjSize, kProjSize, vProjSize, oProjSize, qoMaxSeqLength,
                      kvMaxSeqLength, maxBatchSize, maxBeamSize);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                                                                       const cudnnAttnDescriptor_t attnDesc,
                                                                       size_t *weightSizeInBytes,
                                                                       size_t *workSpaceSizeInBytes,
                                                                       size_t *reserveSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetMultiHeadAttnBuffers");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnAttnDescriptor_t, size_t *, size_t *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetMultiHeadAttnBuffers"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, attnDesc, weightSizeInBytes, workSpaceSizeInBytes, reserveSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle,
                                                                       const cudnnAttnDescriptor_t attnDesc,
                                                                       cudnnMultiHeadAttnWeightKind_t wKind,
                                                                       size_t weightSizeInBytes, const void *weights,
                                                                       cudnnTensorDescriptor_t wDesc, void **wAddr) {
    HOOK_TRACE_PROFILE("cudnnGetMultiHeadAttnWeights");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnAttnDescriptor_t, cudnnMultiHeadAttnWeightKind_t,
                                       size_t, const void *, cudnnTensorDescriptor_t, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetMultiHeadAttnWeights"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, attnDesc, wKind, weightSizeInBytes, weights, wDesc, wAddr);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnMultiHeadAttnForward(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, int currIdx, const int loWinIdx, const int hiWinIdx,
    const int devSeqLengthsQO, const int devSeqLengthsKV, const cudnnSeqDataDescriptor_t qDesc, const void *queries,
    const void *residuals, const cudnnSeqDataDescriptor_t kDesc, const void *keys, const cudnnSeqDataDescriptor_t vDesc,
    const void *values, const cudnnSeqDataDescriptor_t oDesc, void *out, size_t weightSizeInBytes, const void *weights,
    size_t workSpaceSizeInBytes, void *workSpace, size_t reserveSpaceSizeInBytes, void *reserveSpace) {
    HOOK_TRACE_PROFILE("cudnnMultiHeadAttnForward");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnAttnDescriptor_t, int, const int, const int, const int, const int,
                          const cudnnSeqDataDescriptor_t, const void *, const void *, const cudnnSeqDataDescriptor_t,
                          const void *, const cudnnSeqDataDescriptor_t, const void *, const cudnnSeqDataDescriptor_t,
                          void *, size_t, const void *, size_t, void *, size_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnMultiHeadAttnForward"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, attnDesc, currIdx, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV, qDesc, queries,
                      residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights,
                      workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnMultiHeadAttnBackwardData(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, const int loWinIdx, const int hiWinIdx,
    const int devSeqLengthsDQDO, const int devSeqLengthsDKDV, const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    const cudnnSeqDataDescriptor_t dqDesc, void *dqueries, const void *queries, const cudnnSeqDataDescriptor_t dkDesc,
    void *dkeys, const void *keys, const cudnnSeqDataDescriptor_t dvDesc, void *dvalues, const void *values,
    size_t weightSizeInBytes, const void *weights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
    HOOK_TRACE_PROFILE("cudnnMultiHeadAttnBackwardData");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const cudnnAttnDescriptor_t, const int, const int, const int, const int,
        const cudnnSeqDataDescriptor_t, const void *, const cudnnSeqDataDescriptor_t, void *, const void *,
        const cudnnSeqDataDescriptor_t, void *, const void *, const cudnnSeqDataDescriptor_t, void *, const void *,
        size_t, const void *, size_t, void *, size_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnMultiHeadAttnBackwardData"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, attnDesc, loWinIdx, hiWinIdx, devSeqLengthsDQDO, devSeqLengthsDKDV, doDesc, dout, dqDesc,
                      dqueries, queries, dkDesc, dkeys, keys, dvDesc, dvalues, values, weightSizeInBytes, weights,
                      workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(
    cudnnHandle_t handle, const cudnnAttnDescriptor_t attnDesc, cudnnWgradMode_t addGrad,
    const cudnnSeqDataDescriptor_t qDesc, const void *queries, const cudnnSeqDataDescriptor_t kDesc, const void *keys,
    const cudnnSeqDataDescriptor_t vDesc, const void *values, const cudnnSeqDataDescriptor_t doDesc, const void *dout,
    size_t weightSizeInBytes, const void *weights, void *dweights, size_t workSpaceSizeInBytes, void *workSpace,
    size_t reserveSpaceSizeInBytes, void *reserveSpace) {
    HOOK_TRACE_PROFILE("cudnnMultiHeadAttnBackwardWeights");
    using func_ptr = cudnnStatus_t (*)(
        cudnnHandle_t, const cudnnAttnDescriptor_t, cudnnWgradMode_t, const cudnnSeqDataDescriptor_t, const void *,
        const cudnnSeqDataDescriptor_t, const void *, const cudnnSeqDataDescriptor_t, const void *,
        const cudnnSeqDataDescriptor_t, const void *, size_t, const void *, void *, size_t, void *, size_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnMultiHeadAttnBackwardWeights"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, attnDesc, addGrad, qDesc, queries, kDesc, keys, vDesc, values, doDesc, dout,
                      weightSizeInBytes, weights, dweights, workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes,
                      reserveSpace);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateCTCLossDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnCTCLossDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateCTCLossDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(ctcLossDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                                    cudnnDataType_t compType) {
    HOOK_TRACE_PROFILE("cudnnSetCTCLossDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnCTCLossDescriptor_t, cudnnDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetCTCLossDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(ctcLossDesc, compType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                                      cudnnDataType_t compType,
                                                                      cudnnLossNormalizationMode_t normMode,
                                                                      cudnnNanPropagation_t gradMode) {
    HOOK_TRACE_PROFILE("cudnnSetCTCLossDescriptorEx");
    using func_ptr = cudnnStatus_t (*)(cudnnCTCLossDescriptor_t, cudnnDataType_t, cudnnLossNormalizationMode_t,
                                       cudnnNanPropagation_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetCTCLossDescriptorEx"));
    HOOK_CHECK(func_entry);
    return func_entry(ctcLossDesc, compType, normMode, gradMode);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                                    cudnnDataType_t *compType) {
    HOOK_TRACE_PROFILE("cudnnGetCTCLossDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnCTCLossDescriptor_t, cudnnDataType_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetCTCLossDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(ctcLossDesc, compType);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                                                                      cudnnDataType_t *compType,
                                                                      cudnnLossNormalizationMode_t *normMode,
                                                                      cudnnNanPropagation_t *gradMode) {
    HOOK_TRACE_PROFILE("cudnnGetCTCLossDescriptorEx");
    using func_ptr = cudnnStatus_t (*)(cudnnCTCLossDescriptor_t, cudnnDataType_t *, cudnnLossNormalizationMode_t *,
                                       cudnnNanPropagation_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetCTCLossDescriptorEx"));
    HOOK_CHECK(func_entry);
    return func_entry(ctcLossDesc, compType, normMode, gradMode);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyCTCLossDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnCTCLossDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyCTCLossDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(ctcLossDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCTCLoss(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc,
                                                       const void *probs, const int *labels, const int *labelLengths,
                                                       const int *inputLengths, void *costs,
                                                       const cudnnTensorDescriptor_t gradientsDesc,
                                                       const void *gradients, cudnnCTCLossAlgo_t algo,
                                                       cudnnCTCLossDescriptor_t ctcLossDesc, void *workspace,
                                                       size_t workSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnCTCLoss");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const void *, const int *,
                                       const int *, const int *, void *, const cudnnTensorDescriptor_t, const void *,
                                       cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCTCLoss"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, probsDesc, probs, labels, labelLengths, inputLengths, costs, gradientsDesc, gradients,
                      algo, ctcLossDesc, workspace, workSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetCTCLossWorkspaceSize(
    cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const cudnnTensorDescriptor_t gradientsDesc,
    const int *labels, const int *labelLengths, const int *inputLengths, cudnnCTCLossAlgo_t algo,
    cudnnCTCLossDescriptor_t ctcLossDesc, size_t *sizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetCTCLossWorkspaceSize");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const int *,
                          const int *, const int *, cudnnCTCLossAlgo_t, cudnnCTCLossDescriptor_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetCTCLossWorkspaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, probsDesc, gradientsDesc, labels, labelLengths, inputLengths, algo, ctcLossDesc,
                      sizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc) {
    HOOK_TRACE_PROFILE("cudnnCreateAlgorithmDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnAlgorithmDescriptor_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateAlgorithmDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(algoDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc,
                                                                      cudnnAlgorithm_t algorithm) {
    HOOK_TRACE_PROFILE("cudnnSetAlgorithmDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnAlgorithmDescriptor_t, cudnnAlgorithm_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetAlgorithmDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(algoDesc, algorithm);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc,
                                                                      cudnnAlgorithm_t *algorithm) {
    HOOK_TRACE_PROFILE("cudnnGetAlgorithmDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnAlgorithmDescriptor_t, cudnnAlgorithm_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetAlgorithmDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(algoDesc, algorithm);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src,
                                                                       cudnnAlgorithmDescriptor_t dest) {
    HOOK_TRACE_PROFILE("cudnnCopyAlgorithmDescriptor");
    using func_ptr = cudnnStatus_t (*)(const cudnnAlgorithmDescriptor_t, cudnnAlgorithmDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCopyAlgorithmDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(src, dest);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc) {
    HOOK_TRACE_PROFILE("cudnnDestroyAlgorithmDescriptor");
    using func_ptr = cudnnStatus_t (*)(cudnnAlgorithmDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyAlgorithmDescriptor"));
    HOOK_CHECK(func_entry);
    return func_entry(algoDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf,
                                                                          int numberToCreate) {
    HOOK_TRACE_PROFILE("cudnnCreateAlgorithmPerformance");
    using func_ptr = cudnnStatus_t (*)(cudnnAlgorithmPerformance_t *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateAlgorithmPerformance"));
    HOOK_CHECK(func_entry);
    return func_entry(algoPerf, numberToCreate);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                                                                       cudnnAlgorithmDescriptor_t algoDesc,
                                                                       cudnnStatus_t status, float time,
                                                                       size_t memory) {
    HOOK_TRACE_PROFILE("cudnnSetAlgorithmPerformance");
    using func_ptr =
        cudnnStatus_t (*)(cudnnAlgorithmPerformance_t, cudnnAlgorithmDescriptor_t, cudnnStatus_t, float, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetAlgorithmPerformance"));
    HOOK_CHECK(func_entry);
    return func_entry(algoPerf, algoDesc, status, time, memory);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                                                                       cudnnAlgorithmDescriptor_t *algoDesc,
                                                                       cudnnStatus_t *status, float *time,
                                                                       size_t *memory) {
    HOOK_TRACE_PROFILE("cudnnGetAlgorithmPerformance");
    using func_ptr = cudnnStatus_t (*)(const cudnnAlgorithmPerformance_t, cudnnAlgorithmDescriptor_t *, cudnnStatus_t *,
                                       float *, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetAlgorithmPerformance"));
    HOOK_CHECK(func_entry);
    return func_entry(algoPerf, algoDesc, status, time, memory);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf,
                                                                           int numberToDestroy) {
    HOOK_TRACE_PROFILE("cudnnDestroyAlgorithmPerformance");
    using func_ptr = cudnnStatus_t (*)(cudnnAlgorithmPerformance_t *, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyAlgorithmPerformance"));
    HOOK_CHECK(func_entry);
    return func_entry(algoPerf, numberToDestroy);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle,
                                                                     cudnnAlgorithmDescriptor_t algoDesc,
                                                                     size_t *algoSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnGetAlgorithmSpaceSize");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnAlgorithmDescriptor_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetAlgorithmSpaceSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algoDesc, algoSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSaveAlgorithm(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc,
                                                             void *algoSpace, size_t algoSpaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnSaveAlgorithm");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudnnAlgorithmDescriptor_t, void *, size_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSaveAlgorithm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algoDesc, algoSpace, algoSpaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnRestoreAlgorithm(cudnnHandle_t handle, void *algoSpace,
                                                                size_t algoSpaceSizeInBytes,
                                                                cudnnAlgorithmDescriptor_t algoDesc) {
    HOOK_TRACE_PROFILE("cudnnRestoreAlgorithm");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, void *, size_t, cudnnAlgorithmDescriptor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnRestoreAlgorithm"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, algoSpace, algoSpaceSizeInBytes, algoDesc);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr) {
    HOOK_TRACE_PROFILE("cudnnSetCallback");
    using func_ptr = cudnnStatus_t (*)(unsigned, void *, cudnnCallback_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(mask, udata, fptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr) {
    HOOK_TRACE_PROFILE("cudnnGetCallback");
    using func_ptr = cudnnStatus_t (*)(unsigned *, void **, cudnnCallback_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetCallback"));
    HOOK_CHECK(func_entry);
    return func_entry(mask, udata, fptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack,
                                                                            cudnnFusedOps_t ops) {
    HOOK_TRACE_PROFILE("cudnnCreateFusedOpsConstParamPack");
    using func_ptr = cudnnStatus_t (*)(cudnnFusedOpsConstParamPack_t *, cudnnFusedOps_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateFusedOpsConstParamPack"));
    HOOK_CHECK(func_entry);
    return func_entry(constPack, ops);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack) {
    HOOK_TRACE_PROFILE("cudnnDestroyFusedOpsConstParamPack");
    using func_ptr = cudnnStatus_t (*)(cudnnFusedOpsConstParamPack_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyFusedOpsConstParamPack"));
    HOOK_CHECK(func_entry);
    return func_entry(constPack);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(
    cudnnFusedOpsConstParamPack_t constPack, cudnnFusedOpsConstParamLabel_t paramLabel, const void *param) {
    HOOK_TRACE_PROFILE("cudnnSetFusedOpsConstParamPackAttribute");
    using func_ptr = cudnnStatus_t (*)(cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t, const void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetFusedOpsConstParamPackAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(constPack, paramLabel, param);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t constPack,
                                            cudnnFusedOpsConstParamLabel_t paramLabel, void *param, int *isNULL) {
    HOOK_TRACE_PROFILE("cudnnGetFusedOpsConstParamPackAttribute");
    using func_ptr =
        cudnnStatus_t (*)(const cudnnFusedOpsConstParamPack_t, cudnnFusedOpsConstParamLabel_t, void *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetFusedOpsConstParamPackAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(constPack, paramLabel, param, isNULL);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack,
                                                                              cudnnFusedOps_t ops) {
    HOOK_TRACE_PROFILE("cudnnCreateFusedOpsVariantParamPack");
    using func_ptr = cudnnStatus_t (*)(cudnnFusedOpsVariantParamPack_t *, cudnnFusedOps_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateFusedOpsVariantParamPack"));
    HOOK_CHECK(func_entry);
    return func_entry(varPack, ops);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t
    cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack) {
    HOOK_TRACE_PROFILE("cudnnDestroyFusedOpsVariantParamPack");
    using func_ptr = cudnnStatus_t (*)(cudnnFusedOpsVariantParamPack_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyFusedOpsVariantParamPack"));
    HOOK_CHECK(func_entry);
    return func_entry(varPack);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(
    cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr) {
    HOOK_TRACE_PROFILE("cudnnSetFusedOpsVariantParamPackAttribute");
    using func_ptr = cudnnStatus_t (*)(cudnnFusedOpsVariantParamPack_t, cudnnFusedOpsVariantParamLabel_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetFusedOpsVariantParamPackAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(varPack, paramLabel, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(
    const cudnnFusedOpsVariantParamPack_t varPack, cudnnFusedOpsVariantParamLabel_t paramLabel, void *ptr) {
    HOOK_TRACE_PROFILE("cudnnGetFusedOpsVariantParamPackAttribute");
    using func_ptr = cudnnStatus_t (*)(const cudnnFusedOpsVariantParamPack_t, cudnnFusedOpsVariantParamLabel_t, void *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetFusedOpsVariantParamPackAttribute"));
    HOOK_CHECK(func_entry);
    return func_entry(varPack, paramLabel, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops) {
    HOOK_TRACE_PROFILE("cudnnCreateFusedOpsPlan");
    using func_ptr = cudnnStatus_t (*)(cudnnFusedOpsPlan_t *, cudnnFusedOps_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreateFusedOpsPlan"));
    HOOK_CHECK(func_entry);
    return func_entry(plan, ops);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan) {
    HOOK_TRACE_PROFILE("cudnnDestroyFusedOpsPlan");
    using func_ptr = cudnnStatus_t (*)(cudnnFusedOpsPlan_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroyFusedOpsPlan"));
    HOOK_CHECK(func_entry);
    return func_entry(plan);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnMakeFusedOpsPlan(cudnnHandle_t handle, cudnnFusedOpsPlan_t plan,
                                                                const cudnnFusedOpsConstParamPack_t constPack,
                                                                size_t *workspaceSizeInBytes) {
    HOOK_TRACE_PROFILE("cudnnMakeFusedOpsPlan");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, cudnnFusedOpsPlan_t, const cudnnFusedOpsConstParamPack_t, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnMakeFusedOpsPlan"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, plan, constPack, workspaceSizeInBytes);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan,
                                                               cudnnFusedOpsVariantParamPack_t varPack) {
    HOOK_TRACE_PROFILE("cudnnFusedOpsExecute");
    using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, const cudnnFusedOpsPlan_t, cudnnFusedOpsVariantParamPack_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnFusedOpsExecute"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, plan, varPack);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc,
                                                                   const int hiddenSize, const int numLayers,
                                                                   cudnnDropoutDescriptor_t dropoutDesc,
                                                                   cudnnRNNInputMode_t inputMode,
                                                                   cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                                   cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec) {
    HOOK_TRACE_PROFILE("cudnnSetRNNDescriptor_v6");
    using func_ptr =
        cudnnStatus_t (*)(cudnnHandle_t, cudnnRNNDescriptor_t, const int, const int, cudnnDropoutDescriptor_t,
                          cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNDescriptor_v6"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, algo, mathPrec);
}

HOOK_C_API HOOK_DECL_EXPORT cudnnStatus_t cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc, int hiddenSize,
                                                                   int numLayers, cudnnDropoutDescriptor_t dropoutDesc,
                                                                   cudnnRNNInputMode_t inputMode,
                                                                   cudnnDirectionMode_t direction, cudnnRNNMode_t mode,
                                                                   cudnnDataType_t mathPrec) {
    HOOK_TRACE_PROFILE("cudnnSetRNNDescriptor_v5");
    using func_ptr = cudnnStatus_t (*)(cudnnRNNDescriptor_t, int, int, cudnnDropoutDescriptor_t, cudnnRNNInputMode_t,
                                       cudnnDirectionMode_t, cudnnRNNMode_t, cudnnDataType_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetRNNDescriptor_v5"));
    HOOK_CHECK(func_entry);
    return func_entry(rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, direction, mode, mathPrec);
}
