// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:06:22 on Wed, Jul 20, 2022
//
// Description: auto generate 67 apis

#include "cublas_subset.h"
#include "hook.h"
#include "macro_common.h"
#include "nvjpeg_subset.h"
#include "trace_profile.h"

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegGetProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("nvjpegGetProperty");
    using func_ptr = nvjpegStatus_t (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegGetProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegGetCudartProperty(libraryPropertyType type, int *value) {
    HOOK_TRACE_PROFILE("nvjpegGetCudartProperty");
    using func_ptr = nvjpegStatus_t (*)(libraryPropertyType, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegGetCudartProperty"));
    HOOK_CHECK(func_entry);
    return func_entry(type, value);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegCreate(nvjpegBackend_t backend, nvjpegDevAllocator_t *dev_allocator,
                                                        nvjpegHandle_t *handle) {
    HOOK_TRACE_PROFILE("nvjpegCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegBackend_t, nvjpegDevAllocator_t *, nvjpegHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(backend, dev_allocator, handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t *handle) {
    HOOK_TRACE_PROFILE("nvjpegCreateSimple");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegCreateSimple"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegCreateEx(nvjpegBackend_t backend, nvjpegDevAllocator_t *dev_allocator,
                                                          nvjpegPinnedAllocator_t *pinned_allocator, unsigned int flags,
                                                          nvjpegHandle_t *handle) {
    HOOK_TRACE_PROFILE("nvjpegCreateEx");
    using func_ptr = nvjpegStatus_t (*)(nvjpegBackend_t, nvjpegDevAllocator_t *, nvjpegPinnedAllocator_t *,
                                        unsigned int, nvjpegHandle_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegCreateEx"));
    HOOK_CHECK(func_entry);
    return func_entry(backend, dev_allocator, pinned_allocator, flags, handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t handle) {
    HOOK_TRACE_PROFILE("nvjpegDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegSetDeviceMemoryPadding(size_t padding, nvjpegHandle_t handle) {
    HOOK_TRACE_PROFILE("nvjpegSetDeviceMemoryPadding");
    using func_ptr = nvjpegStatus_t (*)(size_t, nvjpegHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegSetDeviceMemoryPadding"));
    HOOK_CHECK(func_entry);
    return func_entry(padding, handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegGetDeviceMemoryPadding(size_t *padding, nvjpegHandle_t handle) {
    HOOK_TRACE_PROFILE("nvjpegGetDeviceMemoryPadding");
    using func_ptr = nvjpegStatus_t (*)(size_t *, nvjpegHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegGetDeviceMemoryPadding"));
    HOOK_CHECK(func_entry);
    return func_entry(padding, handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegSetPinnedMemoryPadding(size_t padding, nvjpegHandle_t handle) {
    HOOK_TRACE_PROFILE("nvjpegSetPinnedMemoryPadding");
    using func_ptr = nvjpegStatus_t (*)(size_t, nvjpegHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegSetPinnedMemoryPadding"));
    HOOK_CHECK(func_entry);
    return func_entry(padding, handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegGetPinnedMemoryPadding(size_t *padding, nvjpegHandle_t handle) {
    HOOK_TRACE_PROFILE("nvjpegGetPinnedMemoryPadding");
    using func_ptr = nvjpegStatus_t (*)(size_t *, nvjpegHandle_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegGetPinnedMemoryPadding"));
    HOOK_CHECK(func_entry);
    return func_entry(padding, handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStateCreate(nvjpegHandle_t handle,
                                                                 nvjpegJpegState_t *jpeg_handle) {
    HOOK_TRACE_PROFILE("nvjpegJpegStateCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStateCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStateDestroy(nvjpegJpegState_t jpeg_handle) {
    HOOK_TRACE_PROFILE("nvjpegJpegStateDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStateDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(jpeg_handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegGetImageInfo(nvjpegHandle_t handle, const unsigned char *data,
                                                              size_t length, int *nComponents,
                                                              nvjpegChromaSubsampling_t *subsampling, int *widths,
                                                              int *heights) {
    HOOK_TRACE_PROFILE("nvjpegGetImageInfo");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, const unsigned char *, size_t, int *,
                                        nvjpegChromaSubsampling_t *, int *, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegGetImageInfo"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, data, length, nComponents, subsampling, widths, heights);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecode(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle,
                                                        const unsigned char *data, size_t length,
                                                        nvjpegOutputFormat_t output_format, nvjpegImage_t *destination,
                                                        cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegDecode");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *, size_t,
                                        nvjpegOutputFormat_t, nvjpegImage_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecode"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_handle, data, length, output_format, destination, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeBatchedInitialize(nvjpegHandle_t handle,
                                                                         nvjpegJpegState_t jpeg_handle, int batch_size,
                                                                         int max_cpu_threads,
                                                                         nvjpegOutputFormat_t output_format) {
    HOOK_TRACE_PROFILE("nvjpegDecodeBatchedInitialize");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, int, int, nvjpegOutputFormat_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeBatchedInitialize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_handle, batch_size, max_cpu_threads, output_format);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeBatched(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle,
                                                               const unsigned char *const *data, const size_t *lengths,
                                                               nvjpegImage_t *destinations, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegDecodeBatched");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *const *, const size_t *,
                                        nvjpegImage_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeBatched"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_handle, data, lengths, destinations, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeBatchedPreAllocate(nvjpegHandle_t handle,
                                                                          nvjpegJpegState_t jpeg_handle, int batch_size,
                                                                          int width, int height,
                                                                          nvjpegChromaSubsampling_t chroma_subsampling,
                                                                          nvjpegOutputFormat_t output_format) {
    HOOK_TRACE_PROFILE("nvjpegDecodeBatchedPreAllocate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, int, int, int, nvjpegChromaSubsampling_t,
                                        nvjpegOutputFormat_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeBatchedPreAllocate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_handle, batch_size, width, height, chroma_subsampling, output_format);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderStateCreate(nvjpegHandle_t handle,
                                                                    nvjpegEncoderState_t *encoder_state,
                                                                    cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderStateCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderStateCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, encoder_state, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderStateDestroy(nvjpegEncoderState_t encoder_state) {
    HOOK_TRACE_PROFILE("nvjpegEncoderStateDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderState_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderStateDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(encoder_state);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsCreate(nvjpegHandle_t handle,
                                                                     nvjpegEncoderParams_t *encoder_params,
                                                                     cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderParams_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, encoder_params, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsDestroy(nvjpegEncoderParams_t encoder_params) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderParams_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(encoder_params);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetQuality(nvjpegEncoderParams_t encoder_params,
                                                                         const int quality, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsSetQuality");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, const int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsSetQuality"));
    HOOK_CHECK(func_entry);
    return func_entry(encoder_params, quality, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetEncoding(nvjpegEncoderParams_t encoder_params,
                                                                          nvjpegJpegEncoding_t etype,
                                                                          cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsSetEncoding");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, nvjpegJpegEncoding_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsSetEncoding"));
    HOOK_CHECK(func_entry);
    return func_entry(encoder_params, etype, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetOptimizedHuffman(nvjpegEncoderParams_t encoder_params,
                                                                                  const int optimized,
                                                                                  cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsSetOptimizedHuffman");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, const int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsSetOptimizedHuffman"));
    HOOK_CHECK(func_entry);
    return func_entry(encoder_params, optimized, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsSetSamplingFactors(
    nvjpegEncoderParams_t encoder_params, const nvjpegChromaSubsampling_t chroma_subsampling, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsSetSamplingFactors");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, const nvjpegChromaSubsampling_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsSetSamplingFactors"));
    HOOK_CHECK(func_entry);
    return func_entry(encoder_params, chroma_subsampling, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncodeGetBufferSize(nvjpegHandle_t handle,
                                                                     const nvjpegEncoderParams_t encoder_params,
                                                                     int image_width, int image_height,
                                                                     size_t *max_stream_length) {
    HOOK_TRACE_PROFILE("nvjpegEncodeGetBufferSize");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, const nvjpegEncoderParams_t, int, int, size_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncodeGetBufferSize"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, encoder_params, image_width, image_height, max_stream_length);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncodeYUV(nvjpegHandle_t handle, nvjpegEncoderState_t encoder_state,
                                                           const nvjpegEncoderParams_t encoder_params,
                                                           const nvjpegImage_t *source,
                                                           nvjpegChromaSubsampling_t chroma_subsampling,
                                                           int image_width, int image_height, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncodeYUV");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t,
                                        const nvjpegImage_t *, nvjpegChromaSubsampling_t, int, int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncodeYUV"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, encoder_state, encoder_params, source, chroma_subsampling, image_width, image_height,
                      stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncodeImage(nvjpegHandle_t handle, nvjpegEncoderState_t encoder_state,
                                                             const nvjpegEncoderParams_t encoder_params,
                                                             const nvjpegImage_t *source,
                                                             nvjpegInputFormat_t input_format, int image_width,
                                                             int image_height, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncodeImage");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t, const nvjpegEncoderParams_t,
                                        const nvjpegImage_t *, nvjpegInputFormat_t, int, int, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncodeImage"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, encoder_state, encoder_params, source, input_format, image_width, image_height, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncodeRetrieveBitstreamDevice(nvjpegHandle_t handle,
                                                                               nvjpegEncoderState_t encoder_state,
                                                                               unsigned char *data, size_t *length,
                                                                               cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncodeRetrieveBitstreamDevice");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t, unsigned char *, size_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncodeRetrieveBitstreamDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, encoder_state, data, length, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncodeRetrieveBitstream(nvjpegHandle_t handle,
                                                                         nvjpegEncoderState_t encoder_state,
                                                                         unsigned char *data, size_t *length,
                                                                         cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncodeRetrieveBitstream");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegEncoderState_t, unsigned char *, size_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncodeRetrieveBitstream"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, encoder_state, data, length, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegBufferPinnedCreate(nvjpegHandle_t handle,
                                                                    nvjpegPinnedAllocator_t *pinned_allocator,
                                                                    nvjpegBufferPinned_t *buffer) {
    HOOK_TRACE_PROFILE("nvjpegBufferPinnedCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegPinnedAllocator_t *, nvjpegBufferPinned_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegBufferPinnedCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, pinned_allocator, buffer);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegBufferPinnedDestroy(nvjpegBufferPinned_t buffer) {
    HOOK_TRACE_PROFILE("nvjpegBufferPinnedDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegBufferPinned_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegBufferPinnedDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(buffer);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegBufferDeviceCreate(nvjpegHandle_t handle,
                                                                    nvjpegDevAllocator_t *device_allocator,
                                                                    nvjpegBufferDevice_t *buffer) {
    HOOK_TRACE_PROFILE("nvjpegBufferDeviceCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegDevAllocator_t *, nvjpegBufferDevice_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegBufferDeviceCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, device_allocator, buffer);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegBufferDeviceDestroy(nvjpegBufferDevice_t buffer) {
    HOOK_TRACE_PROFILE("nvjpegBufferDeviceDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegBufferDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegBufferDeviceDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(buffer);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegBufferPinnedRetrieve(nvjpegBufferPinned_t buffer, size_t *size,
                                                                      void **ptr) {
    HOOK_TRACE_PROFILE("nvjpegBufferPinnedRetrieve");
    using func_ptr = nvjpegStatus_t (*)(nvjpegBufferPinned_t, size_t *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegBufferPinnedRetrieve"));
    HOOK_CHECK(func_entry);
    return func_entry(buffer, size, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegBufferDeviceRetrieve(nvjpegBufferDevice_t buffer, size_t *size,
                                                                      void **ptr) {
    HOOK_TRACE_PROFILE("nvjpegBufferDeviceRetrieve");
    using func_ptr = nvjpegStatus_t (*)(nvjpegBufferDevice_t, size_t *, void **);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegBufferDeviceRetrieve"));
    HOOK_CHECK(func_entry);
    return func_entry(buffer, size, ptr);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegStateAttachPinnedBuffer(nvjpegJpegState_t decoder_state,
                                                                         nvjpegBufferPinned_t pinned_buffer) {
    HOOK_TRACE_PROFILE("nvjpegStateAttachPinnedBuffer");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegState_t, nvjpegBufferPinned_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegStateAttachPinnedBuffer"));
    HOOK_CHECK(func_entry);
    return func_entry(decoder_state, pinned_buffer);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegStateAttachDeviceBuffer(nvjpegJpegState_t decoder_state,
                                                                         nvjpegBufferDevice_t device_buffer) {
    HOOK_TRACE_PROFILE("nvjpegStateAttachDeviceBuffer");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegState_t, nvjpegBufferDevice_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegStateAttachDeviceBuffer"));
    HOOK_CHECK(func_entry);
    return func_entry(decoder_state, device_buffer);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamCreate(nvjpegHandle_t handle,
                                                                  nvjpegJpegStream_t *jpeg_stream) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegStream_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamDestroy(nvjpegJpegStream_t jpeg_stream) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(jpeg_stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamParse(nvjpegHandle_t handle, const unsigned char *data,
                                                                 size_t length, int save_metadata, int save_stream,
                                                                 nvjpegJpegStream_t jpeg_stream) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamParse");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, const unsigned char *, size_t, int, int, nvjpegJpegStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamParse"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, data, length, save_metadata, save_stream, jpeg_stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamParseHeader(nvjpegHandle_t handle, const unsigned char *data,
                                                                       size_t length, nvjpegJpegStream_t jpeg_stream) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamParseHeader");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, const unsigned char *, size_t, nvjpegJpegStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamParseHeader"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, data, length, jpeg_stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamGetJpegEncoding(nvjpegJpegStream_t jpeg_stream,
                                                                           nvjpegJpegEncoding_t *jpeg_encoding) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamGetJpegEncoding");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegStream_t, nvjpegJpegEncoding_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamGetJpegEncoding"));
    HOOK_CHECK(func_entry);
    return func_entry(jpeg_stream, jpeg_encoding);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamGetFrameDimensions(nvjpegJpegStream_t jpeg_stream,
                                                                              unsigned int *width,
                                                                              unsigned int *height) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamGetFrameDimensions");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegStream_t, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamGetFrameDimensions"));
    HOOK_CHECK(func_entry);
    return func_entry(jpeg_stream, width, height);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamGetComponentsNum(nvjpegJpegStream_t jpeg_stream,
                                                                            unsigned int *components_num) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamGetComponentsNum");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegStream_t, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamGetComponentsNum"));
    HOOK_CHECK(func_entry);
    return func_entry(jpeg_stream, components_num);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamGetComponentDimensions(nvjpegJpegStream_t jpeg_stream,
                                                                                  unsigned int component,
                                                                                  unsigned int *width,
                                                                                  unsigned int *height) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamGetComponentDimensions");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegStream_t, unsigned int, unsigned int *, unsigned int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamGetComponentDimensions"));
    HOOK_CHECK(func_entry);
    return func_entry(jpeg_stream, component, width, height);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegJpegStreamGetChromaSubsampling(
    nvjpegJpegStream_t jpeg_stream, nvjpegChromaSubsampling_t *chroma_subsampling) {
    HOOK_TRACE_PROFILE("nvjpegJpegStreamGetChromaSubsampling");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegStream_t, nvjpegChromaSubsampling_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegJpegStreamGetChromaSubsampling"));
    HOOK_CHECK(func_entry);
    return func_entry(jpeg_stream, chroma_subsampling);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeParamsCreate(nvjpegHandle_t handle,
                                                                    nvjpegDecodeParams_t *decode_params) {
    HOOK_TRACE_PROFILE("nvjpegDecodeParamsCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegDecodeParams_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeParamsCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, decode_params);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeParamsDestroy(nvjpegDecodeParams_t decode_params) {
    HOOK_TRACE_PROFILE("nvjpegDecodeParamsDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegDecodeParams_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeParamsDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(decode_params);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetOutputFormat(nvjpegDecodeParams_t decode_params,
                                                                             nvjpegOutputFormat_t output_format) {
    HOOK_TRACE_PROFILE("nvjpegDecodeParamsSetOutputFormat");
    using func_ptr = nvjpegStatus_t (*)(nvjpegDecodeParams_t, nvjpegOutputFormat_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeParamsSetOutputFormat"));
    HOOK_CHECK(func_entry);
    return func_entry(decode_params, output_format);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetROI(nvjpegDecodeParams_t decode_params, int offset_x,
                                                                    int offset_y, int roi_width, int roi_height) {
    HOOK_TRACE_PROFILE("nvjpegDecodeParamsSetROI");
    using func_ptr = nvjpegStatus_t (*)(nvjpegDecodeParams_t, int, int, int, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeParamsSetROI"));
    HOOK_CHECK(func_entry);
    return func_entry(decode_params, offset_x, offset_y, roi_width, roi_height);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetAllowCMYK(nvjpegDecodeParams_t decode_params,
                                                                          int allow_cmyk) {
    HOOK_TRACE_PROFILE("nvjpegDecodeParamsSetAllowCMYK");
    using func_ptr = nvjpegStatus_t (*)(nvjpegDecodeParams_t, int);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeParamsSetAllowCMYK"));
    HOOK_CHECK(func_entry);
    return func_entry(decode_params, allow_cmyk);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeParamsSetScaleFactor(nvjpegDecodeParams_t decode_params,
                                                                            nvjpegScaleFactor_t scale_factor) {
    HOOK_TRACE_PROFILE("nvjpegDecodeParamsSetScaleFactor");
    using func_ptr = nvjpegStatus_t (*)(nvjpegDecodeParams_t, nvjpegScaleFactor_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeParamsSetScaleFactor"));
    HOOK_CHECK(func_entry);
    return func_entry(decode_params, scale_factor);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecoderCreate(nvjpegHandle_t nvjpeg_handle,
                                                               nvjpegBackend_t implementation,
                                                               nvjpegJpegDecoder_t *decoder_handle) {
    HOOK_TRACE_PROFILE("nvjpegDecoderCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegBackend_t, nvjpegJpegDecoder_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecoderCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(nvjpeg_handle, implementation, decoder_handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecoderDestroy(nvjpegJpegDecoder_t decoder_handle) {
    HOOK_TRACE_PROFILE("nvjpegDecoderDestroy");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegDecoder_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecoderDestroy"));
    HOOK_CHECK(func_entry);
    return func_entry(decoder_handle);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecoderJpegSupported(nvjpegJpegDecoder_t decoder_handle,
                                                                      nvjpegJpegStream_t jpeg_stream,
                                                                      nvjpegDecodeParams_t decode_params,
                                                                      int *is_supported) {
    HOOK_TRACE_PROFILE("nvjpegDecoderJpegSupported");
    using func_ptr = nvjpegStatus_t (*)(nvjpegJpegDecoder_t, nvjpegJpegStream_t, nvjpegDecodeParams_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecoderJpegSupported"));
    HOOK_CHECK(func_entry);
    return func_entry(decoder_handle, jpeg_stream, decode_params, is_supported);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeBatchedSupported(nvjpegHandle_t handle,
                                                                        nvjpegJpegStream_t jpeg_stream,
                                                                        int *is_supported) {
    HOOK_TRACE_PROFILE("nvjpegDecodeBatchedSupported");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegStream_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeBatchedSupported"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_stream, is_supported);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeBatchedSupportedEx(nvjpegHandle_t handle,
                                                                          nvjpegJpegStream_t jpeg_stream,
                                                                          nvjpegDecodeParams_t decode_params,
                                                                          int *is_supported) {
    HOOK_TRACE_PROFILE("nvjpegDecodeBatchedSupportedEx");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegStream_t, nvjpegDecodeParams_t, int *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeBatchedSupportedEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_stream, decode_params, is_supported);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecoderStateCreate(nvjpegHandle_t nvjpeg_handle,
                                                                    nvjpegJpegDecoder_t decoder_handle,
                                                                    nvjpegJpegState_t *decoder_state) {
    HOOK_TRACE_PROFILE("nvjpegDecoderStateCreate");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t *);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecoderStateCreate"));
    HOOK_CHECK(func_entry);
    return func_entry(nvjpeg_handle, decoder_handle, decoder_state);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeJpeg(nvjpegHandle_t handle, nvjpegJpegDecoder_t decoder,
                                                            nvjpegJpegState_t decoder_state,
                                                            nvjpegJpegStream_t jpeg_bitstream,
                                                            nvjpegImage_t *destination,
                                                            nvjpegDecodeParams_t decode_params, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegDecodeJpeg");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegJpegStream_t,
                                        nvjpegImage_t *, nvjpegDecodeParams_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeJpeg"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, decoder, decoder_state, jpeg_bitstream, destination, decode_params, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeJpegHost(nvjpegHandle_t handle, nvjpegJpegDecoder_t decoder,
                                                                nvjpegJpegState_t decoder_state,
                                                                nvjpegDecodeParams_t decode_params,
                                                                nvjpegJpegStream_t jpeg_stream) {
    HOOK_TRACE_PROFILE("nvjpegDecodeJpegHost");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegDecodeParams_t,
                                        nvjpegJpegStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeJpegHost"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, decoder, decoder_state, decode_params, jpeg_stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeJpegTransferToDevice(nvjpegHandle_t handle,
                                                                            nvjpegJpegDecoder_t decoder,
                                                                            nvjpegJpegState_t decoder_state,
                                                                            nvjpegJpegStream_t jpeg_stream,
                                                                            cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegDecodeJpegTransferToDevice");
    using func_ptr =
        nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegJpegStream_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeJpegTransferToDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, decoder, decoder_state, jpeg_stream, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeJpegDevice(nvjpegHandle_t handle, nvjpegJpegDecoder_t decoder,
                                                                  nvjpegJpegState_t decoder_state,
                                                                  nvjpegImage_t *destination, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegDecodeJpegDevice");
    using func_ptr =
        nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegDecoder_t, nvjpegJpegState_t, nvjpegImage_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeJpegDevice"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, decoder, decoder_state, destination, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegDecodeBatchedEx(nvjpegHandle_t handle, nvjpegJpegState_t jpeg_handle,
                                                                 const unsigned char *const *data,
                                                                 const size_t *lengths, nvjpegImage_t *destinations,
                                                                 nvjpegDecodeParams_t *decode_params,
                                                                 cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegDecodeBatchedEx");
    using func_ptr = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *const *, const size_t *,
                                        nvjpegImage_t *, nvjpegDecodeParams_t *, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegDecodeBatchedEx"));
    HOOK_CHECK(func_entry);
    return func_entry(handle, jpeg_handle, data, lengths, destinations, decode_params, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsCopyMetadata(nvjpegEncoderState_t encoder_state,
                                                                           nvjpegEncoderParams_t encode_params,
                                                                           nvjpegJpegStream_t jpeg_stream,
                                                                           cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsCopyMetadata");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderState_t, nvjpegEncoderParams_t, nvjpegJpegStream_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsCopyMetadata"));
    HOOK_CHECK(func_entry);
    return func_entry(encoder_state, encode_params, jpeg_stream, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsCopyQuantizationTables(
    nvjpegEncoderParams_t encode_params, nvjpegJpegStream_t jpeg_stream, cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsCopyQuantizationTables");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderParams_t, nvjpegJpegStream_t, cudaStream_t);
    static auto func_entry =
        reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsCopyQuantizationTables"));
    HOOK_CHECK(func_entry);
    return func_entry(encode_params, jpeg_stream, stream);
}

HOOK_C_API HOOK_DECL_EXPORT nvjpegStatus_t nvjpegEncoderParamsCopyHuffmanTables(nvjpegEncoderState_t encoder_state,
                                                                                nvjpegEncoderParams_t encode_params,
                                                                                nvjpegJpegStream_t jpeg_stream,
                                                                                cudaStream_t stream) {
    HOOK_TRACE_PROFILE("nvjpegEncoderParamsCopyHuffmanTables");
    using func_ptr = nvjpegStatus_t (*)(nvjpegEncoderState_t, nvjpegEncoderParams_t, nvjpegJpegStream_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_NVJPEG_SYMBOL("nvjpegEncoderParamsCopyHuffmanTables"));
    HOOK_CHECK(func_entry);
    return func_entry(encoder_state, encode_params, jpeg_stream, stream);
}
