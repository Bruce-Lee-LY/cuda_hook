// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: cublasLt subset

#ifndef __CUDA_HOOK_CUBLASLT_SUBSET_H__
#define __CUDA_HOOK_CUBLASLT_SUBSET_H__

#include <stdint.h>

#include "cublas_subset.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque structure holding CUBLASLT context
 */
typedef struct cublasLtContext *cublasLtHandle_t;

/** Semi-opaque descriptor for matrix memory layout
 */
typedef struct {
    uint64_t data[8];
} cublasLtMatrixLayoutOpaque_t;

/** Opaque descriptor for matrix memory layout
 */
typedef cublasLtMatrixLayoutOpaque_t *cublasLtMatrixLayout_t;

/** Semi-opaque algorithm descriptor (to avoid complicated alloc/free schemes)
 *
 * This structure can be trivially serialized and later restored for use with the same version of cuBLAS library to save
 * on selecting the right configuration again.
 */
typedef struct {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;

/** Semi-opaque descriptor for cublasLtMatmul() operation details
 */
typedef struct {
    uint64_t data[11];
} cublasLtMatmulDescOpaque_t;

/** Opaque descriptor for cublasLtMatmul() operation details
 */
typedef cublasLtMatmulDescOpaque_t *cublasLtMatmulDesc_t;

/** Semi-opaque descriptor for cublasLtMatrixTransform() operation details
 */
typedef struct {
    uint64_t data[8];
} cublasLtMatrixTransformDescOpaque_t;

/** Opaque descriptor for cublasLtMatrixTransform() operation details
 */
typedef cublasLtMatrixTransformDescOpaque_t *cublasLtMatrixTransformDesc_t;

/** Semi-opaque descriptor for cublasLtMatmulPreference() operation details
 */
typedef struct {
    uint64_t data[10];
} cublasLtMatmulPreferenceOpaque_t;

/** Opaque descriptor for cublasLtMatmulAlgoGetHeuristic() configuration
 */
typedef cublasLtMatmulPreferenceOpaque_t *cublasLtMatmulPreference_t;

/** Tile size (in C/D matrix Rows x Cols)
 *
 * General order of tile IDs is sorted by size first and by first dimension second.
 */
typedef enum {
    CUBLASLT_MATMUL_TILE_UNDEFINED = 0,
    CUBLASLT_MATMUL_TILE_8x8 = 1,
    CUBLASLT_MATMUL_TILE_8x16 = 2,
    CUBLASLT_MATMUL_TILE_16x8 = 3,
    CUBLASLT_MATMUL_TILE_8x32 = 4,
    CUBLASLT_MATMUL_TILE_16x16 = 5,
    CUBLASLT_MATMUL_TILE_32x8 = 6,
    CUBLASLT_MATMUL_TILE_8x64 = 7,
    CUBLASLT_MATMUL_TILE_16x32 = 8,
    CUBLASLT_MATMUL_TILE_32x16 = 9,
    CUBLASLT_MATMUL_TILE_64x8 = 10,
    CUBLASLT_MATMUL_TILE_32x32 = 11,
    CUBLASLT_MATMUL_TILE_32x64 = 12,
    CUBLASLT_MATMUL_TILE_64x32 = 13,
    CUBLASLT_MATMUL_TILE_32x128 = 14,
    CUBLASLT_MATMUL_TILE_64x64 = 15,
    CUBLASLT_MATMUL_TILE_128x32 = 16,
    CUBLASLT_MATMUL_TILE_64x128 = 17,
    CUBLASLT_MATMUL_TILE_128x64 = 18,
    CUBLASLT_MATMUL_TILE_64x256 = 19,
    CUBLASLT_MATMUL_TILE_128x128 = 20,
    CUBLASLT_MATMUL_TILE_256x64 = 21,
    CUBLASLT_MATMUL_TILE_64x512 = 22,
    CUBLASLT_MATMUL_TILE_128x256 = 23,
    CUBLASLT_MATMUL_TILE_256x128 = 24,
    CUBLASLT_MATMUL_TILE_512x64 = 25,
    CUBLASLT_MATMUL_TILE_64x96 = 26,
    CUBLASLT_MATMUL_TILE_96x64 = 27,
    CUBLASLT_MATMUL_TILE_96x128 = 28,
    CUBLASLT_MATMUL_TILE_128x160 = 29,
    CUBLASLT_MATMUL_TILE_160x128 = 30,
    CUBLASLT_MATMUL_TILE_192x128 = 31,
    CUBLASLT_MATMUL_TILE_END
} cublasLtMatmulTile_t;

/** Size and number of stages in which elements are read into shared memory
 *
 * General order of stages IDs is sorted by stage size first and by number of stages second.
 */
typedef enum {
    CUBLASLT_MATMUL_STAGES_UNDEFINED = 0,
    CUBLASLT_MATMUL_STAGES_16x1 = 1,
    CUBLASLT_MATMUL_STAGES_16x2 = 2,
    CUBLASLT_MATMUL_STAGES_16x3 = 3,
    CUBLASLT_MATMUL_STAGES_16x4 = 4,
    CUBLASLT_MATMUL_STAGES_16x5 = 5,
    CUBLASLT_MATMUL_STAGES_16x6 = 6,
    CUBLASLT_MATMUL_STAGES_32x1 = 7,
    CUBLASLT_MATMUL_STAGES_32x2 = 8,
    CUBLASLT_MATMUL_STAGES_32x3 = 9,
    CUBLASLT_MATMUL_STAGES_32x4 = 10,
    CUBLASLT_MATMUL_STAGES_32x5 = 11,
    CUBLASLT_MATMUL_STAGES_32x6 = 12,
    CUBLASLT_MATMUL_STAGES_64x1 = 13,
    CUBLASLT_MATMUL_STAGES_64x2 = 14,
    CUBLASLT_MATMUL_STAGES_64x3 = 15,
    CUBLASLT_MATMUL_STAGES_64x4 = 16,
    CUBLASLT_MATMUL_STAGES_64x5 = 17,
    CUBLASLT_MATMUL_STAGES_64x6 = 18,
    CUBLASLT_MATMUL_STAGES_128x1 = 19,
    CUBLASLT_MATMUL_STAGES_128x2 = 20,
    CUBLASLT_MATMUL_STAGES_128x3 = 21,
    CUBLASLT_MATMUL_STAGES_128x4 = 22,
    CUBLASLT_MATMUL_STAGES_128x5 = 23,
    CUBLASLT_MATMUL_STAGES_128x6 = 24,
    CUBLASLT_MATMUL_STAGES_32x10 = 25,
    CUBLASLT_MATMUL_STAGES_8x4 = 26,
    CUBLASLT_MATMUL_STAGES_16x10 = 27,
    CUBLASLT_MATMUL_STAGES_8x5 = 28,
    CUBLASLT_MATMUL_STAGES_16x80 = 29,
    CUBLASLT_MATMUL_STAGES_64x80 = 30,
    CUBLASLT_MATMUL_STAGES_END
} cublasLtMatmulStages_t;

/** Pointer mode to use for alpha/beta */
typedef enum {
    /** matches CUBLAS_POINTER_MODE_HOST, pointer targets a single value host memory */
    CUBLASLT_POINTER_MODE_HOST = CUBLAS_POINTER_MODE_HOST,
    /** matches CUBLAS_POINTER_MODE_DEVICE, pointer targets a single value device memory */
    CUBLASLT_POINTER_MODE_DEVICE = CUBLAS_POINTER_MODE_DEVICE,
    /** pointer targets an array in device memory */
    CUBLASLT_POINTER_MODE_DEVICE_VECTOR = 2,
    /** alpha pointer targets an array in device memory, beta is zero. Note:
       CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE is not supported, must be 0. */
    CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO = 3,
    /** alpha pointer targets an array in device memory, beta is a single value in host memory. */
    CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = 4,
} cublasLtPointerMode_t;

/** Mask to define and query pointer mode capability */
typedef enum {
    /** no initial filtering is performed when querying pointer mode capabilities, will use gemm pointer mode defined in
       operation description **/
    CUBLASLT_POINTER_MODE_MASK_NO_FILTERING = 0,
    /** see CUBLASLT_POINTER_MODE_HOST */
    CUBLASLT_POINTER_MODE_MASK_HOST = 1,
    /** see CUBLASLT_POINTER_MODE_DEVICE */
    CUBLASLT_POINTER_MODE_MASK_DEVICE = 2,
    /** see CUBLASLT_POINTER_MODE_DEVICE_VECTOR */
    CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR = 4,
    /** see CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO */
    CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO = 8,
    /** see CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST */
    CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST = 16,
} cublasLtPointerModeMask_t;

/** Implementation details that may affect numerical behavior of algorithms. */
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA (0x01ull << 0)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA (0x02ull << 0)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA (0x04ull << 0)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA (0x08ull << 0)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK (0xfeull << 0)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK (0xffull << 0)

#define CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F (0x01ull << 8)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F (0x02ull << 8)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F (0x04ull << 8)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I (0x08ull << 8)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK (0xffull << 8)

#define CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F (0x01ull << 16)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF (0x02ull << 16)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32 (0x04ull << 16)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F (0x08ull << 16)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F (0x10ull << 16)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I (0x20ull << 16)
#define CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK (0xffull << 16)

#define CUBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN (0x01ull << 32)
typedef uint64_t cublasLtNumericalImplFlags_t;

/** Enum for data ordering */
typedef enum {
    /** Column-major
     *
     * Leading dimension is the stride (in elements) to the beginning of next column in memory.
     */
    CUBLASLT_ORDER_COL = 0,
    /** Row major
     *
     * Leading dimension is the stride (in elements) to the beginning of next row in memory.
     */
    CUBLASLT_ORDER_ROW = 1,
    /** Column-major ordered tiles of 32 columns.
     *
     * Leading dimension is the stride (in elements) to the beginning of next group of 32-columns. E.g. if matrix has 33
     * columns and 2 rows, ld must be at least (32) * 2 = 64.
     */
    CUBLASLT_ORDER_COL32 = 2,
    /** Column-major ordered tiles of composite tiles with total 32 columns and 8 rows, tile composed of interleaved
     * inner tiles of 4 columns within 4 even or odd rows in an alternating pattern.
     *
     * Leading dimension is the stride (in elements) to the beginning of the first 32 column x 8 row tile for the next
     * 32-wide group of columns. E.g. if matrix has 33 columns and 1 row, ld must be at least (32 * 8) * 1 = 256.
     */
    CUBLASLT_ORDER_COL4_4R2_8C = 3,
    /** Column-major ordered tiles of composite tiles with total 32 columns ands 32 rows.
     * Element offset within the tile is calculated as (((row%8)/2*4+row/8)*2+row%2)*32+col.
     *
     * Leading dimension is the stride (in elements) to the beginning of the first 32 column x 32 row tile for the next
     * 32-wide group of columns. E.g. if matrix has 33 columns and 1 row, ld must be at least (32*32)*1 = 1024.
     */
    CUBLASLT_ORDER_COL32_2R_4R4 = 4,

} cublasLtOrder_t;

/** Attributes of memory layout */
typedef enum {
    /** Data type, see cudaDataType.
     *
     * uint32_t
     */
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,

    /** Memory order of the data, see cublasLtOrder_t.
     *
     * int32_t, default: CUBLASLT_ORDER_COL
     */
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,

    /** Number of rows.
     *
     * Usually only values that can be expressed as int32_t are supported.
     *
     * uint64_t
     */
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,

    /** Number of columns.
     *
     * Usually only values that can be expressed as int32_t are supported.
     *
     * uint64_t
     */
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,

    /** Matrix leading dimension.
     *
     * For CUBLASLT_ORDER_COL this is stride (in elements) of matrix column, for more details and documentation for
     * other memory orders see documentation for cublasLtOrder_t values.
     *
     * Currently only non-negative values are supported, must be large enough so that matrix memory locations are not
     * overlapping (e.g. greater or equal to CUBLASLT_MATRIX_LAYOUT_ROWS in case of CUBLASLT_ORDER_COL).
     *
     * int64_t;
     */
    CUBLASLT_MATRIX_LAYOUT_LD = 4,

    /** Number of matmul operations to perform in the batch.
     *
     * See also CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT
     *
     * int32_t, default: 1
     */
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,

    /** Stride (in elements) to the next matrix for strided batch operation.
     *
     * When matrix type is planar-complex (CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET != 0), batch stride
     * is interpreted by cublasLtMatmul() in number of real valued sub-elements. E.g. for data of type CUDA_C_16F,
     * offset of 1024B is encoded as a stride of value 512 (since each element of the real and imaginary matrices
     * is a 2B (16bit) floating point type).
     *
     * NOTE: A bug in cublasLtMatrixTransform() causes it to interpret the batch stride for a planar-complex matrix
     * as if it was specified in number of complex elements. Therefore an offset of 1024B must be encoded as stride
     * value 256 when calling cublasLtMatrixTransform() (each complex element is 4B with real and imaginary values 2B
     * each). This behavior is expected to be corrected in the next major cuBLAS version.
     *
     * int64_t, default: 0
     */
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,

    /** Stride (in bytes) to the imaginary plane for planar complex layout.
     *
     * int64_t, default: 0 - 0 means that layout is regular (real and imaginary parts of complex numbers are interleaved
     * in memory in each element)
     */
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7,
} cublasLtMatrixLayoutAttribute_t;

/** Matmul descriptor attributes to define details of the operation. */
typedef enum {
    /** Compute type, see cudaDataType. Defines data type used for multiply and accumulate operations and the
     * accumulator during matrix multiplication.
     *
     * int32_t
     */
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,

    /** Scale type, see cudaDataType. Defines data type of alpha and beta. Accumulator and value from matrix C are
     * typically converted to scale type before final scaling. Value is then converted from scale type to type of matrix
     * D before being stored in memory.
     *
     * int32_t, default: same as CUBLASLT_MATMUL_DESC_COMPUTE_TYPE
     */
    CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,

    /** Pointer mode of alpha and beta, see cublasLtPointerMode_t. When CUBLASLT_POINTER_MODE_DEVICE_VECTOR is in use,
     * alpha/beta vector lenghts must match number of output matrix rows.
     *
     * int32_t, default: CUBLASLT_POINTER_MODE_HOST
     */
    CUBLASLT_MATMUL_DESC_POINTER_MODE = 2,

    /** Transform of matrix A, see cublasOperation_t.
     *
     * int32_t, default: CUBLAS_OP_N
     */
    CUBLASLT_MATMUL_DESC_TRANSA = 3,

    /** Transform of matrix B, see cublasOperation_t.
     *
     * int32_t, default: CUBLAS_OP_N
     */
    CUBLASLT_MATMUL_DESC_TRANSB = 4,

    /** Transform of matrix C, see cublasOperation_t.
     *
     * Currently only CUBLAS_OP_N is supported.
     *
     * int32_t, default: CUBLAS_OP_N
     */
    CUBLASLT_MATMUL_DESC_TRANSC = 5,

    /** Matrix fill mode, see cublasFillMode_t.
     *
     * int32_t, default: CUBLAS_FILL_MODE_FULL
     */
    CUBLASLT_MATMUL_DESC_FILL_MODE = 6,

    /** Epilogue function, see cublasLtEpilogue_t.
     *
     * uint32_t, default: CUBLASLT_EPILOGUE_DEFAULT
     */
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7,

    /** Bias or bias gradient vector pointer in the device memory.
     *
     * Bias case. See CUBLASLT_EPILOGUE_BIAS.
     * Bias vector elements are the same type as
     * the output elements (Ctype) with the exception of IMMA kernels with computeType=CUDA_R_32I and Ctype=CUDA_R_8I
     * where the bias vector elements are the same type as alpha, beta (CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F).
     * Bias vector length must match matrix D rows count.
     *
     * Bias gradient case. See CUBLASLT_EPILOGUE_DRELU_BGRAD and CUBLASLT_EPILOGUE_DGELU_BGRAD.
     * Bias gradient vector elements are the same type as the output elements
     * (Ctype) with the exception of IMMA kernels (see above).
     *
     * Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
     * depend on its value to determine expected pointer alignment.
     *
     * Bias case: const void *, default: NULL
     * Bias gradient case: void *, default: NULL
     */
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,

    /** Batch stride for bias or bias gradient vector.
     *
     * Used together with CUBLASLT_MATMUL_DESC_BIAS_POINTER when matrix D's CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT > 1.
     *
     * int64_t, default: 0
     */
    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 10,

    /** Pointer for epilogue auxiliary buffer.
     *
     * - Output vector for ReLu bit-mask in forward pass when CUBLASLT_EPILOGUE_RELU_AUX
     *   or CUBLASLT_EPILOGUE_RELU_AUX_BIAS epilogue is used.
     * - Input vector for ReLu bit-mask in backward pass when
     *   CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is used.
     *
     * - Output of GELU input matrix in forward pass when
     *   CUBLASLT_EPILOGUE_GELU_AUX_BIAS epilogue is used.
     * - Input of GELU input matrix for backward pass when
     *   CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue is used.
     *
     * GELU input matrix elements type is the same as the type of elements of
     * the output matrix.
     *
     * Routines that don't dereference this pointer, like cublasLtMatmulAlgoGetHeuristic()
     * depend on its value to determine expected pointer alignment.
     *
     * Requires setting CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD attribute.
     *
     * Forward pass: void *, default: NULL
     * Backward pass: const void *, default: NULL
     */
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,

    /** Leading dimension for epilogue auxiliary buffer.
     *
     * - ReLu bit-mask matrix leading dimension in elements (i.e. bits)
     *   when CUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX_BIAS or CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is
     * used. Must be divisible by 128 and be no less than the number of rows in the output matrix.
     *
     * - GELU input matrix leading dimension in elements
     *   when CUBLASLT_EPILOGUE_GELU_AUX_BIAS or CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue used.
     *   Must be divisible by 8 and be no less than the number of rows in the output matrix.
     *
     * int64_t, default: 0
     */
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 12,

    /** Batch stride for epilogue auxiliary buffer.
     *
     * - ReLu bit-mask matrix batch stride in elements (i.e. bits)
     *   when CUBLASLT_EPILOGUE_RELU_AUX, CUBLASLT_EPILOGUE_RELU_AUX_BIAS or CUBLASLT_EPILOGUE_DRELU_BGRAD epilogue is
     * used. Must be divisible by 128.
     *
     * - GELU input matrix batch stride in elements
     *   when CUBLASLT_EPILOGUE_GELU_AUX_BIAS or CUBLASLT_EPILOGUE_DGELU_BGRAD epilogue used.
     *   Must be divisible by 8.
     *
     * int64_t, default: 0
     */
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 13,

    /** Batch stride for alpha vector.
     *
     * Used together with CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST when matrix D's
     * CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT > 1. If CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO is set then
     * CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE must be set to 0 as this mode doesnt supported batched alpha
     * vector.
     *
     * int64_t, default: 0
     */
    CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE = 14,

} cublasLtMatmulDescAttributes_t;

/** Matrix transform descriptor attributes to define details of the operation.
 */
typedef enum {
    /** Scale type, see cudaDataType. Inputs are converted to scale type for scaling and summation and results are then
     * converted to output type to store in memory.
     *
     * int32_t
     */
    CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,

    /** Pointer mode of alpha and beta, see cublasLtPointerMode_t.
     *
     * int32_t, default: CUBLASLT_POINTER_MODE_HOST
     */
    CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,

    /** Transform of matrix A, see cublasOperation_t.
     *
     * int32_t, default: CUBLAS_OP_N
     */
    CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,

    /** Transform of matrix B, see cublasOperation_t.
     *
     * int32_t, default: CUBLAS_OP_N
     */
    CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
} cublasLtMatrixTransformDescAttributes_t;

/** For computation with complex numbers, this enum allows to apply the Gauss Complexity reduction algorithm
 */
typedef enum {
    CUBLASLT_3M_MODE_DISALLOWED = 0,
    CUBLASLT_3M_MODE_ALLOWED = 1,
} cublasLt3mMode_t;

/** Reduction scheme for portions of the dot-product calculated in parallel (a. k. a. "split - K").
 */
typedef enum {
    /** No reduction scheme, dot-product shall be performed in one sequence.
     */
    CUBLASLT_REDUCTION_SCHEME_NONE = 0,

    /** Reduction is performed "in place" - using the output buffer (and output data type) and counters (in workspace)
     * to guarantee the sequentiality.
     */
    CUBLASLT_REDUCTION_SCHEME_INPLACE = 1,

    /** Intermediate results are stored in compute type in the workspace and reduced in a separate step.
     */
    CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE = 2,

    /** Intermediate results are stored in output type in the workspace and reduced in a separate step.
     */
    CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE = 4,

    CUBLASLT_REDUCTION_SCHEME_MASK = 0x7,
} cublasLtReductionScheme_t;

/** Postprocessing options for the epilogue
 */
typedef enum {
    /** No special postprocessing, just scale and quantize results if necessary.
     */
    CUBLASLT_EPILOGUE_DEFAULT = 1,

    /** ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).
     */
    CUBLASLT_EPILOGUE_RELU = 2,

    /** ReLu, apply ReLu point-wise transform to the results (x:=max(x, 0)).
     *
     * This epilogue mode produces an extra output, a ReLu bit-mask matrix,
     * see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
     */
    CUBLASLT_EPILOGUE_RELU_AUX = (CUBLASLT_EPILOGUE_RELU | 128),

    /** Bias, apply (broadcasted) Bias from bias vector. Bias vector length must match matrix D rows, it must be packed
     * (stride between vector elements is 1). Bias vector is broadcasted to all columns and added before applying final
     * postprocessing.
     */
    CUBLASLT_EPILOGUE_BIAS = 4,

    /** ReLu and Bias, apply Bias and then ReLu transform
     */
    CUBLASLT_EPILOGUE_RELU_BIAS = (CUBLASLT_EPILOGUE_RELU | CUBLASLT_EPILOGUE_BIAS),

    /** ReLu and Bias, apply Bias and then ReLu transform
     *
     * This epilogue mode produces an extra output, a ReLu bit-mask matrix,
     * see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
     */
    CUBLASLT_EPILOGUE_RELU_AUX_BIAS = (CUBLASLT_EPILOGUE_RELU_AUX | CUBLASLT_EPILOGUE_BIAS),

    /* ReLu and Bias gradients. Apply independently ReLu and Bias gradient to
     * matmul output. Store ReLu gradient in the output matrix, and Bias gradient
     * in the auxiliary output (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
     *
     * This epilogue mode requires an extra input,
     * see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
     */
    CUBLASLT_EPILOGUE_DRELU_BGRAD = 8 | 16 | 128,

    /** GELU, apply GELU point-wise transform to the results (x:=GELU(x)).
     */
    CUBLASLT_EPILOGUE_GELU = 32,

    /** GELU, apply GELU point-wise transform to the results (x:=GELU(x)).
     *
     * This epilogue mode outputs GELU input as a separate matrix (useful for training).
     * See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
     */
    CUBLASLT_EPILOGUE_GELU_AUX = (CUBLASLT_EPILOGUE_GELU | 128),

    /** GELU and Bias, apply Bias and then GELU transform
     */
    CUBLASLT_EPILOGUE_GELU_BIAS = (CUBLASLT_EPILOGUE_GELU | CUBLASLT_EPILOGUE_BIAS),

    /** GELU and Bias, apply Bias and then GELU transform
     *
     * This epilogue mode outputs GELU input as a separate matrix (useful for training).
     * See CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
     */
    CUBLASLT_EPILOGUE_GELU_AUX_BIAS = (CUBLASLT_EPILOGUE_GELU_AUX | CUBLASLT_EPILOGUE_BIAS),

    /* GELU and Bias gradients. Apply independently GELU and Bias gradient to
     * matmul output. Store GELU gradient in the output matrix, and Bias gradient
     * in the auxiliary output (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
     *
     * This epilogue mode requires an extra input,
     * see CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER.
     */
    CUBLASLT_EPILOGUE_DGELU_BGRAD = 16 | 64 | 128,

    /** Bias gradient based on the input matrix A.
     *
     * The bias size corresponds to the number of rows of the matrix D.
     * The reduction happens over the GEMM's "k" dimension.
     *
     * Stores Bias gradient in the auxiliary output
     * (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
     */
    CUBLASLT_EPILOGUE_BGRADA = 256,

    /** Bias gradient based on the input matrix B.
     *
     * The bias size corresponds to the number of columns of the matrix D.
     * The reduction happens over the GEMM's "k" dimension.
     *
     * Stores Bias gradient in the auxiliary output
     * (see CUBLASLT_MATMUL_DESC_BIAS_POINTER).
     */
    CUBLASLT_EPILOGUE_BGRADB = 512,
} cublasLtEpilogue_t;

/** Matmul heuristic search mode
 */
typedef enum {
    /** ask heuristics for best algo for given usecase
     */
    CUBLASLT_SEARCH_BEST_FIT = 0,
    /** only try to find best config for preconfigured algo id
     */
    CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID = 1,
    /** reserved for future use
     */
    CUBLASLT_SEARCH_RESERVED_02 = 2,
    /** reserved for future use
     */
    CUBLASLT_SEARCH_RESERVED_03 = 3,
    /** reserved for future use
     */
    CUBLASLT_SEARCH_RESERVED_04 = 4,
    /** reserved for future use
     */
    CUBLASLT_SEARCH_RESERVED_05 = 5,
} cublasLtMatmulSearch_t;

/** Algo search preference to fine tune the heuristic function. */
typedef enum {
    /** Search mode, see cublasLtMatmulSearch_t.
     *
     * uint32_t, default: CUBLASLT_SEARCH_BEST_FIT
     */
    CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,

    /** Maximum allowed workspace size in bytes.
     *
     * uint64_t, default: 0 - no workspace allowed
     */
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,

    /** Math mode mask, see cublasMath_t.
     *
     * Only algorithms with CUBLASLT_ALGO_CAP_MATHMODE_IMPL that is not masked out by this attribute are allowed.
     *
     * uint32_t, default: 1 (allows both default and tensor op math)
     * DEPRECATED, will be removed in a future release, see cublasLtNumericalImplFlags_t for replacement
     */
    CUBLASLT_MATMUL_PREF_MATH_MODE_MASK = 2,

    /** Reduction scheme mask, see cublasLtReductionScheme_t. Filters heuristic result to only include algo configs that
     * use one of the required modes.
     *
     * E.g. mask value of 0x03 will allow only INPLACE and COMPUTE_TYPE reduction schemes.
     *
     * uint32_t, default: CUBLASLT_REDUCTION_SCHEME_MASK (allows all reduction schemes)
     */
    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,

    /** Gaussian mode mask, see cublasLt3mMode_t.
     *
     * Only algorithms with CUBLASLT_ALGO_CAP_GAUSSIAN_IMPL that is not masked out by this attribute are allowed.
     *
     * uint32_t, default: CUBLASLT_3M_MODE_ALLOWED (allows both gaussian and non-gaussian algorithms)
     * DEPRECATED, will be removed in a future release, see cublasLtNumericalImplFlags_t for replacement
     */
    CUBLASLT_MATMUL_PREF_GAUSSIAN_MODE_MASK = 4,

    /** Minimum buffer alignment for matrix A (in bytes).
     *
     * Selecting a smaller value will exclude algorithms that can not work with matrix A that is not as strictly aligned
     * as they need.
     *
     * uint32_t, default: 256
     */
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,

    /** Minimum buffer alignment for matrix B (in bytes).
     *
     * Selecting a smaller value will exclude algorithms that can not work with matrix B that is not as strictly aligned
     * as they need.
     *
     * uint32_t, default: 256
     */
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,

    /** Minimum buffer alignment for matrix C (in bytes).
     *
     * Selecting a smaller value will exclude algorithms that can not work with matrix C that is not as strictly aligned
     * as they need.
     *
     * uint32_t, default: 256
     */
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,

    /** Minimum buffer alignment for matrix D (in bytes).
     *
     * Selecting a smaller value will exclude algorithms that can not work with matrix D that is not as strictly aligned
     * as they need.
     *
     * uint32_t, default: 256
     */
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,

    /** Maximum wave count.
     *
     * See cublasLtMatmulHeuristicResult_t::wavesCount.
     *
     * Selecting a non-zero value will exclude algorithms that report device utilization higher than specified.
     *
     * float, default: 0.0f
     */
    CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 9,

    /** Pointer mode mask, see cublasLtPointerModeMask_t. Filters heuristic result to only include algorithms that
     * support all required modes.
     *
     * uint32_t, default: (CUBLASLT_POINTER_MODE_MASK_HOST | CUBLASLT_POINTER_MODE_MASK_DEVICE) (only allows algorithms
     * that support both regular host and device pointers)
     */
    CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK = 10,

    /** Epilogue selector mask, see cublasLtEpilogue_t. Filters heuristic result to only include algorithms that support
     * all required operations.
     *
     * uint32_t, default: CUBLASLT_EPILOGUE_DEFAULT (only allows algorithms that support default epilogue)
     */
    CUBLASLT_MATMUL_PREF_EPILOGUE_MASK = 11,

    /** Numerical implementation details mask, see cublasLtNumericalImplFlags_t. Filters heuristic result to only
     * include algorithms that use the allowed implementations.
     *
     * uint64_t, default: uint64_t(-1) (allow everything)
     */
    CUBLASLT_MATMUL_PREF_IMPL_MASK = 12,

    /** Number of SMs to target for parallel execution. Optimizes heuristics for execution in smaller number of SM when
     * user expects a concurrent stream to be using some of the device resources.
     *
     * int32_t, default: 0 - use the number reported by the device.
     */
    CUBLASLT_MATMUL_PREF_SM_COUNT_TARGET = 13,
} cublasLtMatmulPreferenceAttributes_t;

/** Results structure used by cublasLtMatmulGetAlgo.
 *
 * Holds returned configured algo descriptor and its runtime properties.
 */
typedef struct {
    /** Matmul algorithm descriptor.
     *
     * Must be initialized with cublasLtMatmulAlgoInit() if preferences' CUBLASLT_MATMUL_PERF_SEARCH_MODE is set to
     * CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID
     */
    cublasLtMatmulAlgo_t algo;

    /** Actual size of workspace memory required.
     */
    size_t workspaceSize;

    /** Result status, other fields are only valid if after call to cublasLtMatmulAlgoGetHeuristic() this member is set
     * to CUBLAS_STATUS_SUCCESS.
     */
    cublasStatus_t state;

    /** Waves count - a device utilization metric.
     *
     * wavesCount value of 1.0f suggests that when kernel is launched it will fully occupy the GPU.
     */
    float wavesCount;

    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

/** Capabilities Attributes that can be retrieved from an initialized Algo structure
 */
typedef enum {
    /** support for split K, see CUBLASLT_ALGO_CONFIG_SPLITK_NUM
     *
     * int32_t, 0 means no support, supported otherwise
     */
    CUBLASLT_ALGO_CAP_SPLITK_SUPPORT = 0,
    /** reduction scheme mask, see cublasLtReductionScheme_t; shows supported reduction schemes, if reduction scheme is
     * not masked out it is supported.
     *
     * e.g. int isReductionSchemeComputeTypeSupported ? (reductionSchemeMask & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
     * == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE ? 1 : 0;
     *
     * uint32_t
     */
    CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK = 1,
    /** support for cta swizzling, see CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
     *
     * uint32_t, 0 means no support, 1 means supported value of 1, other values are reserved
     */
    CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT = 2,
    /** support strided batch
     *
     * int32_t, 0 means no support, supported otherwise
     */
    CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT = 3,
    /** support results out of place (D != C in D = alpha.A.B + beta.C)
     *
     * int32_t, 0 means no support, supported otherwise
     */
    CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT = 4,
    /** syrk/herk support (on top of regular gemm)
     *
     * int32_t, 0 means no support, supported otherwise
     */
    CUBLASLT_ALGO_CAP_UPLO_SUPPORT = 5,
    /** tile ids possible to use, see cublasLtMatmulTile_t; if no tile ids are supported use
     * CUBLASLT_MATMUL_TILE_UNDEFINED
     *
     * use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count
     *
     * array of uint32_t
     */
    CUBLASLT_ALGO_CAP_TILE_IDS = 6,
    /** custom option range is from 0 to CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX (inclusive), see
     * CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION
     *
     * int32_t
     */
    CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX = 7,
    /** whether algorithm is using regular compute or tensor operations
     *
     * int32_t 0 means regular compute, 1 means tensor operations;
     * DEPRECATED
     */
    CUBLASLT_ALGO_CAP_MATHMODE_IMPL = 8,
    /** whether algorithm implements gaussian optimization of complex matrix multiplication, see cublasMath_t
     *
     * int32_t 0 means regular compute, 1 means gaussian;
     * DEPRECATED
     */
    CUBLASLT_ALGO_CAP_GAUSSIAN_IMPL = 9,
    /** whether algorithm supports custom (not COL or ROW memory order), see cublasLtOrder_t
     *
     * int32_t 0 means only COL and ROW memory order is allowed, non-zero means that algo might have different
     * requirements;
     */
    CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER = 10,

    /** bitmask enumerating pointer modes algorithm supports
     *
     * uint32_t, see cublasLtPointerModeMask_t
     */
    CUBLASLT_ALGO_CAP_POINTER_MODE_MASK = 11,

    /** bitmask enumerating kinds of postprocessing algorithm supports in the epilogue
     *
     * uint32_t, see cublasLtEpilogue_t
     */
    CUBLASLT_ALGO_CAP_EPILOGUE_MASK = 12,
    /** stages ids possible to use, see cublasLtMatmulStages_t; if no stages ids are supported use
     * CUBLASLT_MATMUL_STAGES_UNDEFINED
     *
     * use cublasLtMatmulAlgoCapGetAttribute() with sizeInBytes=0 to query actual count
     *
     * array of uint32_t
     */
    CUBLASLT_ALGO_CAP_STAGES_IDS = 13,
    /** support for nagative ld for all of the matrices
     *
     * int32_t 0 means no support, supported otherwise
     */
    CUBLASLT_ALGO_CAP_LD_NEGATIVE = 14,
    /** details about algorithm's implementation that affect it's numerical behavior
     *
     * uint64_t, see cublasLtNumericalImplFlags_t
     */
    CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS = 15,
    /** minimum alignment required for A matrix in bytes
     *  (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)
     *
     * uint32_t
     */
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES = 16,
    /** minimum alignment required for B matrix in bytes
     *  (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)
     *
     * uint32_t
     */
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES = 17,
    /** minimum alignment required for C matrix in bytes
     *  (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)
     *
     * uint32_t
     */
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES = 18,
    /** minimum alignment required for D matrix in bytes
     *  (required for buffer pointer, leading dimension, and possibly other strides defined for matrix memory order)
     *
     * uint32_t
     */
    CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES = 19,
} cublasLtMatmulAlgoCapAttributes_t;

/** Algo Configuration Attributes that can be set according to the Algo capabilities
 */
typedef enum {
    /** algorithm index, see cublasLtMatmulAlgoGetIds()
     *
     * readonly, set by cublasLtMatmulAlgoInit()
     * int32_t
     */
    CUBLASLT_ALGO_CONFIG_ID = 0,
    /** tile id, see cublasLtMatmulTile_t
     *
     * uint32_t, default: CUBLASLT_MATMUL_TILE_UNDEFINED
     */
    CUBLASLT_ALGO_CONFIG_TILE_ID = 1,
    /** number of K splits, if != 1, SPLITK_NUM parts of matrix multiplication will be computed in parallel,
     * and then results accumulated according to REDUCTION_SCHEME
     *
     * uint32_t, default: 1
     */
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM = 2,
    /** reduction scheme, see cublasLtReductionScheme_t
     *
     * uint32_t, default: CUBLASLT_REDUCTION_SCHEME_NONE
     */
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME = 3,
    /** cta swizzling, change mapping from CUDA grid coordinates to parts of the matrices
     *
     * possible values: 0, 1, other values reserved
     *
     * uint32_t, default: 0
     */
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING = 4,
    /** custom option, each algorithm can support some custom options that don't fit description of the other config
     * attributes, see CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX to get accepted range for any specific case
     *
     * uint32_t, default: 0
     */
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION = 5,
    /** stages id, see cublasLtMatmulStages_t
     *
     * uint32_t, default: CUBLASLT_MATMUL_STAGES_UNDEFINED
     */
    CUBLASLT_ALGO_CONFIG_STAGES_ID = 6,
} cublasLtMatmulAlgoConfigAttributes_t;

/** Experimental: Logger callback type.
 */
typedef void (*cublasLtLoggerCallback_t)(int logLevel, const char *functionName, const char *message);

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_CUBLASLT_SUBSET_H__
