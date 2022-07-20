// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 22:07:19 on Wed, Jul 20, 2022
//
// Description: nvjpeg subset

#ifndef __CUDA_HOOK_NVJPEG_SUBSET_H__
#define __CUDA_HOOK_NVJPEG_SUBSET_H__

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of channels nvjpeg decoder supports
#define NVJPEG_MAX_COMPONENT 4

// nvjpeg version information
#define NVJPEG_VER_MAJOR 11
#define NVJPEG_VER_MINOR 5
#define NVJPEG_VER_PATCH 2
#define NVJPEG_VER_BUILD 120

/* nvJPEG status enums, returned by nvJPEG API */
typedef enum {
    NVJPEG_STATUS_SUCCESS = 0,
    NVJPEG_STATUS_NOT_INITIALIZED = 1,
    NVJPEG_STATUS_INVALID_PARAMETER = 2,
    NVJPEG_STATUS_BAD_JPEG = 3,
    NVJPEG_STATUS_JPEG_NOT_SUPPORTED = 4,
    NVJPEG_STATUS_ALLOCATOR_FAILURE = 5,
    NVJPEG_STATUS_EXECUTION_FAILED = 6,
    NVJPEG_STATUS_ARCH_MISMATCH = 7,
    NVJPEG_STATUS_INTERNAL_ERROR = 8,
    NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9,
} nvjpegStatus_t;

// Enum identifies image chroma subsampling values stored inside JPEG input stream
// In the case of NVJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
// Otherwise both chroma planes are present
typedef enum {
    NVJPEG_CSS_444 = 0,
    NVJPEG_CSS_422 = 1,
    NVJPEG_CSS_420 = 2,
    NVJPEG_CSS_440 = 3,
    NVJPEG_CSS_411 = 4,
    NVJPEG_CSS_410 = 5,
    NVJPEG_CSS_GRAY = 6,
    NVJPEG_CSS_410V = 7,
    NVJPEG_CSS_UNKNOWN = -1
} nvjpegChromaSubsampling_t;

// Parameter of this type specifies what type of output user wants for image decoding
typedef enum {
    // return decompressed image as it is - write planar output
    NVJPEG_OUTPUT_UNCHANGED = 0,
    // return planar luma and chroma, assuming YCbCr colorspace
    NVJPEG_OUTPUT_YUV = 1,
    // return luma component only, if YCbCr colorspace,
    // or try to convert to grayscale,
    // writes to 1-st channel of nvjpegImage_t
    NVJPEG_OUTPUT_Y = 2,
    // convert to planar RGB
    NVJPEG_OUTPUT_RGB = 3,
    // convert to planar BGR
    NVJPEG_OUTPUT_BGR = 4,
    // convert to interleaved RGB and write to 1-st channel of nvjpegImage_t
    NVJPEG_OUTPUT_RGBI = 5,
    // convert to interleaved BGR and write to 1-st channel of nvjpegImage_t
    NVJPEG_OUTPUT_BGRI = 6,
    // maximum allowed value
    NVJPEG_OUTPUT_FORMAT_MAX = 6
} nvjpegOutputFormat_t;

// Parameter of this type specifies what type of input user provides for encoding
typedef enum {
    NVJPEG_INPUT_RGB = 3,   // Input is RGB - will be converted to YCbCr before encoding
    NVJPEG_INPUT_BGR = 4,   // Input is RGB - will be converted to YCbCr before encoding
    NVJPEG_INPUT_RGBI = 5,  // Input is interleaved RGB - will be converted to YCbCr before encoding
    NVJPEG_INPUT_BGRI = 6   // Input is interleaved RGB - will be converted to YCbCr before encoding
} nvjpegInputFormat_t;

// Implementation
// NVJPEG_BACKEND_DEFAULT    : default value
// NVJPEG_BACKEND_HYBRID     : uses CPU for Huffman decode
// NVJPEG_BACKEND_GPU_HYBRID : uses GPU assisted Huffman decode. nvjpegDecodeBatched will use GPU decoding for baseline
// JPEG bitstreams with
//                             interleaved scan when batch size is bigger than 100
// NVJPEG_BACKEND_HARDWARE   : supports baseline JPEG bitstream with single scan. 410 and 411 sub-samplings are not
// supported
typedef enum {
    NVJPEG_BACKEND_DEFAULT = 0,
    NVJPEG_BACKEND_HYBRID = 1,
    NVJPEG_BACKEND_GPU_HYBRID = 2,
    NVJPEG_BACKEND_HARDWARE = 3
} nvjpegBackend_t;

// Currently parseable JPEG encodings (SOF markers)
typedef enum {
    NVJPEG_ENCODING_UNKNOWN = 0x0,

    NVJPEG_ENCODING_BASELINE_DCT = 0xc0,
    NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN = 0xc1,
    NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN = 0xc2

} nvjpegJpegEncoding_t;

typedef enum {
    NVJPEG_SCALE_NONE = 0,    // decoded output is not scaled
    NVJPEG_SCALE_1_BY_2 = 1,  // decoded output width and height is scaled by a factor of 1/2
    NVJPEG_SCALE_1_BY_4 = 2,  // decoded output width and height is scaled by a factor of 1/4
    NVJPEG_SCALE_1_BY_8 = 3,  // decoded output width and height is scaled by a factor of 1/8
} nvjpegScaleFactor_t;

#define NVJPEG_FLAGS_DEFAULT 0
#define NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE 1
#define NVJPEG_FLAGS_ENABLE_MEMORY_POOLS 1 << 1
#define NVJPEG_FLAGS_BITSTREAM_STRICT 1 << 2

// Output descriptor.
// Data that is written to planes depends on output format
typedef struct {
    unsigned char *channel[NVJPEG_MAX_COMPONENT];
    size_t pitch[NVJPEG_MAX_COMPONENT];
} nvjpegImage_t;

// Prototype for device memory allocation, modelled after cudaMalloc()
typedef int (*tDevMalloc)(void **, size_t);
// Prototype for device memory release
typedef int (*tDevFree)(void *);

// Prototype for pinned memory allocation, modelled after cudaHostAlloc()
typedef int (*tPinnedMalloc)(void **, size_t, unsigned int flags);
// Prototype for device memory release
typedef int (*tPinnedFree)(void *);

// Memory allocator using mentioned prototypes, provided to nvjpegCreateEx
// This allocator will be used for all device memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct {
    tDevMalloc dev_malloc;
    tDevFree dev_free;
} nvjpegDevAllocator_t;

// Pinned memory allocator using mentioned prototypes, provided to nvjpegCreate
// This allocator will be used for all pinned host memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct {
    tPinnedMalloc pinned_malloc;
    tPinnedFree pinned_free;
} nvjpegPinnedAllocator_t;

// Opaque library handle identifier.
struct nvjpegHandle;
typedef struct nvjpegHandle *nvjpegHandle_t;

// Opaque jpeg decoding state handle identifier - used to store intermediate information between deccding phases
struct nvjpegJpegState;
typedef struct nvjpegJpegState *nvjpegJpegState_t;

struct nvjpegEncoderState;
typedef struct nvjpegEncoderState *nvjpegEncoderState_t;

struct nvjpegEncoderParams;
typedef struct nvjpegEncoderParams *nvjpegEncoderParams_t;

struct nvjpegBufferPinned;
typedef struct nvjpegBufferPinned *nvjpegBufferPinned_t;

struct nvjpegBufferDevice;
typedef struct nvjpegBufferDevice *nvjpegBufferDevice_t;

struct nvjpegJpegStream;
typedef struct nvjpegJpegStream *nvjpegJpegStream_t;

struct nvjpegDecodeParams;
typedef struct nvjpegDecodeParams *nvjpegDecodeParams_t;

struct nvjpegJpegDecoder;
typedef struct nvjpegJpegDecoder *nvjpegJpegDecoder_t;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_NVJPEG_SUBSET_H__
