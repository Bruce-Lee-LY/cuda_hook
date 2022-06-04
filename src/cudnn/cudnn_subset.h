// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: cudnn subset

#ifndef __CUDA_HOOK_CUDNN_SUBSET_H__
#define __CUDA_HOOK_CUDNN_SUBSET_H__

#include <stdint.h>

#include "cudart_subset.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

/*
 * CUDNN return codes
 */
typedef enum {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
} cudnnStatus_t;

/* Forward definition in this version only */
typedef struct cudnnRuntimeTag_t cudnnRuntimeTag_t;

typedef enum {
    CUDNN_ERRQUERY_RAWCODE = 0,
    CUDNN_ERRQUERY_NONBLOCKING = 1,
    CUDNN_ERRQUERY_BLOCKING = 2,
} cudnnErrQueryMode_t;

typedef enum libraryPropertyType_t { MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL } libraryPropertyType;

/* Data structures to represent Image/Filter and the Neural Network Layer */
typedef struct cudnnTensorStruct *cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct *cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct *cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct *cudnnFilterDescriptor_t;
typedef struct cudnnLRNStruct *cudnnLRNDescriptor_t;
typedef struct cudnnActivationStruct *cudnnActivationDescriptor_t;
typedef struct cudnnSpatialTransformerStruct *cudnnSpatialTransformerDescriptor_t;
typedef struct cudnnOpTensorStruct *cudnnOpTensorDescriptor_t;
typedef struct cudnnReduceTensorStruct *cudnnReduceTensorDescriptor_t;
typedef struct cudnnCTCLossStruct *cudnnCTCLossDescriptor_t;
typedef struct cudnnTensorTransformStruct *cudnnTensorTransformDescriptor_t;
/*
 * CUDNN data type
 */
typedef enum {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
} cudnnDataType_t;

/*
 * CUDNN math type
 */
typedef enum {
    CUDNN_DEFAULT_MATH = 0,
    CUDNN_TENSOR_OP_MATH = 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
} cudnnMathType_t;

/*
 * CUDNN propagate Nan
 */
typedef enum {
    CUDNN_NOT_PROPAGATE_NAN = 0,
    CUDNN_PROPAGATE_NAN = 1,
} cudnnNanPropagation_t;

/*
 * CUDNN Determinism
 */
typedef enum {
    CUDNN_NON_DETERMINISTIC = 0,
    CUDNN_DETERMINISTIC = 1,
} cudnnDeterminism_t;

/*
 * CUDNN Reorder
 */
typedef enum {
    CUDNN_DEFAULT_REORDER = 0,
    CUDNN_NO_REORDER = 1,
} cudnnReorderType_t;

/* Maximum supported number of tensor dimensions */
#define CUDNN_DIM_MAX 8

typedef enum {
    CUDNN_TENSOR_NCHW = 0,        /* row major (wStride = 1, hStride = w) */
    CUDNN_TENSOR_NHWC = 1,        /* feature maps interleaved ( cStride = 1 )*/
    CUDNN_TENSOR_NCHW_VECT_C = 2, /* each image point is vector of element of C, vector length in data type */
} cudnnTensorFormat_t;

/* Fold/unfold transforms */
typedef enum {
    CUDNN_TRANSFORM_FOLD = 0U,
    CUDNN_TRANSFORM_UNFOLD = 1U,
} cudnnFoldingDirection_t;

/*
 * CUDNN OpTensor op type
 */
typedef enum {
    CUDNN_OP_TENSOR_ADD = 0,
    CUDNN_OP_TENSOR_MUL = 1,
    CUDNN_OP_TENSOR_MIN = 2,
    CUDNN_OP_TENSOR_MAX = 3,
    CUDNN_OP_TENSOR_SQRT = 4,
    CUDNN_OP_TENSOR_NOT = 5,
} cudnnOpTensorOp_t;

/*
 * CUDNN ReduceTensor op type
 */
typedef enum {
    CUDNN_REDUCE_TENSOR_ADD = 0,
    CUDNN_REDUCE_TENSOR_MUL = 1,
    CUDNN_REDUCE_TENSOR_MIN = 2,
    CUDNN_REDUCE_TENSOR_MAX = 3,
    CUDNN_REDUCE_TENSOR_AMAX = 4,
    CUDNN_REDUCE_TENSOR_AVG = 5,
    CUDNN_REDUCE_TENSOR_NORM1 = 6,
    CUDNN_REDUCE_TENSOR_NORM2 = 7,
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} cudnnReduceTensorOp_t;

/*
 * CUDNN ReduceTensor indices type
 */
typedef enum {
    CUDNN_REDUCE_TENSOR_NO_INDICES = 0,
    CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1,
} cudnnReduceTensorIndices_t;

/*
 * CUDNN tensor indices type size (all unsigned)
 * Currently not supported, default is 32 bit unsigned.
 */
typedef enum {
    CUDNN_32BIT_INDICES = 0,
    CUDNN_64BIT_INDICES = 1,
    CUDNN_16BIT_INDICES = 2,
    CUDNN_8BIT_INDICES = 3,
} cudnnIndicesType_t;

/*
 *  convolution mode
 */
typedef enum { CUDNN_CONVOLUTION = 0, CUDNN_CROSS_CORRELATION = 1 } cudnnConvolutionMode_t;

/* helper function to provide the convolution algo that fit best the requirement */
typedef enum {
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionFwdPreference_t;

typedef enum {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8
} cudnnConvolutionFwdAlgo_t;

typedef struct {
    cudnnConvolutionFwdAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnDeterminism_t determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionFwdAlgoPerf_t;

/* helper function to provide the convolution algo that fit best the requirement */
typedef enum {
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdFilterPreference_t;

typedef enum {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0, /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,        /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4, /* not implemented */
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7
} cudnnConvolutionBwdFilterAlgo_t;

typedef struct {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnDeterminism_t determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionBwdFilterAlgoPerf_t;

/*********************************************************/
/* helper function to provide the convolution algo that fit best the requirement */
typedef enum {
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionBwdDataPreference_t;

typedef enum {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0, /* non-deterministic */
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
} cudnnConvolutionBwdDataAlgo_t;

typedef struct {
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnDeterminism_t determinism;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionBwdDataAlgoPerf_t;

/*
 *  softmax algorithm
 */
typedef enum {
    CUDNN_SOFTMAX_FAST = 0,     /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1, /* subtract max from every point to avoid overflow */
    CUDNN_SOFTMAX_LOG = 2
} cudnnSoftmaxAlgorithm_t;

typedef enum {
    CUDNN_SOFTMAX_MODE_INSTANCE = 0, /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL = 1   /* compute the softmax over all C for each H, W, N */
} cudnnSoftmaxMode_t;

/*
 *  pooling mode
 */
typedef enum {
    CUDNN_POOLING_MAX = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, /* count for average includes padded values */
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2, /* count for average does not include padded values */
    CUDNN_POOLING_MAX_DETERMINISTIC = 3
} cudnnPoolingMode_t;

/*
 * activation mode
 */
typedef enum {
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU = 1,
    CUDNN_ACTIVATION_TANH = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU = 4,
    CUDNN_ACTIVATION_IDENTITY = 5
} cudnnActivationMode_t;

#define CUDNN_LRN_MIN_N 1       /* minimum allowed lrnN */
#define CUDNN_LRN_MAX_N 16      /* maximum allowed lrnN */
#define CUDNN_LRN_MIN_K 1e-5    /* minimum allowed lrnK */
#define CUDNN_LRN_MIN_BETA 0.01 /* minimum allowed lrnBeta */

/* LRN layer mode */
typedef enum {
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0, /* Normalize across tensor's dimA[1] dimension */
} cudnnLRNMode_t;

typedef enum {
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0,
} cudnnDivNormMode_t;

typedef enum {
    /* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice) */
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,

    /* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors) */
    CUDNN_BATCHNORM_SPATIAL = 1,

    /*
     * bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors).
     * May be faster than CUDNN_BATCHNORM_SPATIAL but imposes some limits on the range of values
     */
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
} cudnnBatchNormMode_t;

#define CUDNN_BN_MIN_EPSILON 0.0 /* Minimum epsilon allowed to be used in the Batch Normalization formula */

typedef enum {
    CUDNN_BATCHNORM_OPS_BN = 0,                /* do batch normalization only */
    CUDNN_BATCHNORM_OPS_BN_ACTIVATION = 1,     /* do batchNorm, then activation */
    CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2, /* do batchNorm, then elemWiseAdd, then activation */
} cudnnBatchNormOps_t;

/* APIs for spatial transformer network*/
typedef enum {
    CUDNN_SAMPLER_BILINEAR = 0,
} cudnnSamplerType_t;

typedef struct cudnnDropoutStruct *cudnnDropoutDescriptor_t;

/* BASIC RNN API */

typedef enum {
    CUDNN_RNN_ALGO_STANDARD = 0,
    CUDNN_RNN_ALGO_PERSIST_STATIC = 1,
    CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2,
    CUDNN_RNN_ALGO_COUNT = 3,
} cudnnRNNAlgo_t;

typedef enum {
    CUDNN_RNN_RELU = 0, /* basic RNN cell type with ReLu activation */
    CUDNN_RNN_TANH = 1, /* basic RNN cell type with tanh activation */
    CUDNN_LSTM = 2,     /* LSTM with no peephole connections */
    CUDNN_GRU = 3,      /* Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1); */
} cudnnRNNMode_t;

typedef enum {
    CUDNN_RNN_NO_BIAS = 0,         /* rnn cell formulas do not use biases */
    CUDNN_RNN_SINGLE_INP_BIAS = 1, /* rnn cell formulas use one input bias in input GEMM */
    CUDNN_RNN_DOUBLE_BIAS = 2,     /* default, rnn cell formulas use two bias vectors */
    CUDNN_RNN_SINGLE_REC_BIAS = 3  /* rnn cell formulas use one recurrent bias in recurrent GEMM */
} cudnnRNNBiasMode_t;

typedef enum {
    CUDNN_UNIDIRECTIONAL = 0, /* single direction network */
    CUDNN_BIDIRECTIONAL = 1,  /* output concatination at each layer */
} cudnnDirectionMode_t;

typedef enum {
    CUDNN_LINEAR_INPUT = 0, /* adjustable weight matrix in first layer input GEMM */
    CUDNN_SKIP_INPUT = 1,   /* fixed identity matrix in the first layer input GEMM */
} cudnnRNNInputMode_t;

typedef enum {
    CUDNN_RNN_CLIP_NONE = 0,   /* disables LSTM cell clipping */
    CUDNN_RNN_CLIP_MINMAX = 1, /* enables LSTM cell clipping */
} cudnnRNNClipMode_t;

typedef enum {
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 0,   /* padded, outer stride from one time-step to the next */
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 1,     /* sequence length sorted and packed as in basic RNN api */
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2, /* padded, outer stride from one batch to the next */
} cudnnRNNDataLayout_t;

typedef enum {
    CUDNN_RNN_PADDED_IO_DISABLED = 0,
    CUDNN_RNN_PADDED_IO_ENABLED = 1,
} cudnnRNNPaddingMode_t;

struct cudnnRNNStruct;
typedef struct cudnnRNNStruct *cudnnRNNDescriptor_t;

struct cudnnPersistentRNNPlan;
typedef struct cudnnPersistentRNNPlan *cudnnPersistentRNNPlan_t;

struct cudnnRNNDataStruct;
typedef struct cudnnRNNDataStruct *cudnnRNNDataDescriptor_t;

/* RNN FIND API */

typedef struct cudnnAlgorithmStruct *cudnnAlgorithmDescriptor_t;

typedef struct cudnnAlgorithmPerformanceStruct *cudnnAlgorithmPerformance_t;

/* Sequence data descriptor */

typedef enum {
    CUDNN_SEQDATA_TIME_DIM = 0,  /* index in time */
    CUDNN_SEQDATA_BATCH_DIM = 1, /* index in batch */
    CUDNN_SEQDATA_BEAM_DIM = 2,  /* index in beam */
    CUDNN_SEQDATA_VECT_DIM = 3   /* index in vector */
} cudnnSeqDataAxis_t;

#define CUDNN_SEQDATA_DIM_COUNT 4 /* dimension count */

struct cudnnSeqDataStruct;
typedef struct cudnnSeqDataStruct *cudnnSeqDataDescriptor_t;

/* Multihead Attention */

/* Legacy type for backward compatibility */
typedef unsigned cudnnAttnQueryMap_t;

/* Multi-head attention modes set in attention descriptor */
#define CUDNN_ATTN_QUERYMAP_ALL_TO_ONE 0         /* multiple Q-s map to a single (K,V) set when beam size > 1 */
#define CUDNN_ATTN_QUERYMAP_ONE_TO_ONE (1U << 0) /* multiple Q-s map to multiple (K,V) sets when beam size > 1 */
#define CUDNN_ATTN_DISABLE_PROJ_BIASES 0         /* no biases in attention input and output projections */
#define CUDNN_ATTN_ENABLE_PROJ_BIASES (1U << 1)  /* use biases in attention input and output projections */

struct cudnnAttnStruct;
typedef struct cudnnAttnStruct *cudnnAttnDescriptor_t;

typedef enum {
    CUDNN_MH_ATTN_Q_WEIGHTS = 0, /* input projection weights for 'queries' */
    CUDNN_MH_ATTN_K_WEIGHTS = 1, /* input projection weights for 'keys' */
    CUDNN_MH_ATTN_V_WEIGHTS = 2, /* input projection weights for 'values' */
    CUDNN_MH_ATTN_O_WEIGHTS = 3, /* output projection weights */
    CUDNN_MH_ATTN_Q_BIASES = 4,  /* input projection bias tensor for 'queries' */
    CUDNN_MH_ATTN_K_BIASES = 5,  /* input projection bias for 'keys' */
    CUDNN_MH_ATTN_V_BIASES = 6,  /* input projection bias for 'values' */
    CUDNN_MH_ATTN_O_BIASES = 7,  /* output projection biases */
} cudnnMultiHeadAttnWeightKind_t;

#define CUDNN_ATTN_WKIND_COUNT 8 /* Number of attention weight/bias tensors */

typedef enum {
    CUDNN_WGRAD_MODE_ADD = 0, /* add partial gradients to wgrad output buffers */
    CUDNN_WGRAD_MODE_SET = 1, /* write partial gradients to wgrad output buffers */
} cudnnWgradMode_t;

/* CTC LOSS */
typedef enum { CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0, CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1 } cudnnCTCLossAlgo_t;

/* Input normalization mode for loss function */
typedef enum { CUDNN_LOSS_NORMALIZATION_NONE = 0, CUDNN_LOSS_NORMALIZATION_SOFTMAX = 1 } cudnnLossNormalizationMode_t;

typedef struct {
    union Algorithm {
        cudnnConvolutionFwdAlgo_t convFwdAlgo;
        cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
        cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
        cudnnRNNAlgo_t RNNAlgo;
        cudnnCTCLossAlgo_t CTCLossAlgo;
    } algo;
} cudnnAlgorithm_t;

typedef enum {
    CUDNN_SEV_FATAL = 0,
    CUDNN_SEV_ERROR = 1,
    CUDNN_SEV_WARNING = 2,
    CUDNN_SEV_INFO = 3,
} cudnnSeverity_t;

/* Message masks to be used with cudnnSetCallback() */
#define CUDNN_SEV_ERROR_EN (1U << CUDNN_SEV_ERROR)
#define CUDNN_SEV_WARNING_EN (1U << CUDNN_SEV_WARNING)
#define CUDNN_SEV_INFO_EN (1U << CUDNN_SEV_INFO)

/* struct containing useful informaiton for each API call */
typedef struct {
    unsigned cudnn_version;
    cudnnStatus_t cudnnStatus;
    unsigned time_sec;      /* epoch time in seconds */
    unsigned time_usec;     /* microseconds part of epoch time */
    unsigned time_delta;    /* time since start in seconds */
    cudnnHandle_t handle;   /* cudnn handle */
    cudaStream_t stream;    /* cuda stream ID */
    unsigned long long pid; /* process ID */
    unsigned long long tid; /* thread ID */
    int cudaDeviceId;       /* CUDA device ID */
    int reserved[15];       /* reserved for future use */
} cudnnDebug_t;

typedef void (*cudnnCallback_t)(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg, const char *msg);

struct cudnnFusedOpsConstParamStruct;
typedef struct cudnnFusedOpsConstParamStruct *cudnnFusedOpsConstParamPack_t;

struct cudnnFusedOpsVariantParamStruct;
typedef struct cudnnFusedOpsVariantParamStruct *cudnnFusedOpsVariantParamPack_t;

struct cudnnFusedOpsPlanStruct;
typedef struct cudnnFusedOpsPlanStruct *cudnnFusedOpsPlan_t;

typedef enum {
    /* each op in [ ] can be disabled by passing NULL ptr */
    /* [per channel scale], [per channel bias], [activation], convolution, [generate BN stats] */
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = 0,
    /* [per channel scale], [per channel bias], [activation], convolutionBackwardWeights */
    CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = 1,
    /* utility for BN training in BN-conv fusion */
    /* computes the equivalent scale and bias from ySum ySqSum and learned scale, bias */
    /* optionally update running stats and generate saved stats */
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = 2,
    /* utility for BN inference in BN-conv fusion */
    /* computes the equivalent scale and bias from learned running stats and learned scale, bias */
    CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = 3,
    /* reserved for future use: convolution, [per channel scale], [per channel bias], [residual add], [activation] */
    CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = 4,
    /* reserved for future use: [per channel scale], [per channel bias], [residual add],  activation, bitmask */
    CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = 5,
    /* reserved for future use */
    CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = 6,
} cudnnFusedOps_t;

typedef enum {
    /* set XDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get XDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_XDESC = 0,
    /* set/get XDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_XDATA_PLACEHOLDER = 1,
    /* set/get BN_MODE: pass cudnnBatchNormMode_t* */
    CUDNN_PARAM_BN_MODE = 2,
    /* set CUDNN_PARAM_BN_EQSCALEBIAS_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get CUDNN_PARAM_BN_EQSCALEBIAS_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_BN_EQSCALEBIAS_DESC = 3,
    /* set/get BN_EQSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = 4,
    /* set/get BN_EQBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = 5,
    /* set ACTIVATION_DESC: pass previously initialized cudnnActivationDescriptor_t */
    /* get ACTIVATION_DESC: pass previously created cudnnActivationDescriptor_t */
    CUDNN_PARAM_ACTIVATION_DESC = 6,
    /* set CONV_DESC: pass previously initialized cudnnConvolutionDescriptor_t */
    /* get CONV_DESC: pass previously created cudnnConvolutionDescriptor_t */
    CUDNN_PARAM_CONV_DESC = 7,
    /* set WDESC: pass previously initialized cudnnFilterDescriptor_t */
    /* get WDESC: pass previously created cudnnFilterDescriptor_t */
    CUDNN_PARAM_WDESC = 8,
    /* set/get WDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_WDATA_PLACEHOLDER = 9,
    /* set DWDESC: pass previously initialized cudnnFilterDescriptor_t */
    /* get DWDESC: pass previously created cudnnFilterDescriptor_t */
    CUDNN_PARAM_DWDESC = 10,
    /* set/get DWDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_DWDATA_PLACEHOLDER = 11,
    /* set YDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get YDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_YDESC = 12,
    /* set/get YDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_YDATA_PLACEHOLDER = 13,
    /* set DYDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get DYDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_DYDESC = 14,
    /* set/get DYDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_DYDATA_PLACEHOLDER = 15,
    /* set YSTATS_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get YSTATS_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_YSTATS_DESC = 16,
    /* set/get YSUM_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_YSUM_PLACEHOLDER = 17,
    /* set/get YSQSUM_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_YSQSUM_PLACEHOLDER = 18,
    /* set CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = 19,
    /* set/get CUDNN_PARAM_BN_SCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_SCALE_PLACEHOLDER = 20,
    /* set/get CUDNN_PARAM_BN_BIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_BIAS_PLACEHOLDER = 21,
    /* set/get CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = 22,
    /* set/get CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = 23,
    /* set/get CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = 24,
    /* set/get CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = 25,

    /* set ZDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get ZDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_ZDESC = 26,
    /* set/get ZDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_ZDATA_PLACEHOLDER = 27,
    /* set BN_Z_EQSCALEBIAS_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get BN_Z_EQSCALEBIAS_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = 28,
    /* set/get BN_Z_EQSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = 29,
    /* set/get BN_Z_EQBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = 30,

    /* set ACTIVATION_BITMASK_DESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get ACTIVATION_BITMASK_DESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_ACTIVATION_BITMASK_DESC = 31,
    /* set/get ACTIVATION_BITMASK_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = 32,

    /* set DXDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get DXDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_DXDESC = 33,
    /* set/get DXDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_DXDATA_PLACEHOLDER = 34,
    /* set DZDESC: pass previously initialized cudnnTensorDescriptor_t */
    /* get DZDESC: pass previously created cudnnTensorDescriptor_t */
    CUDNN_PARAM_DZDESC = 35,
    /* set/get DZDATA_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_DZDATA_PLACEHOLDER = 36,
    /* set/get CUDNN_PARAM_BN_DSCALE_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = 37,
    /* set/get CUDNN_PARAM_BN_DBIAS_PLACEHOLDER: pass cudnnFusedOpsPointerPlaceHolder_t* */
    CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = 38,
} cudnnFusedOpsConstParamLabel_t;

typedef enum {
    CUDNN_PTR_NULL = 0,
    CUDNN_PTR_ELEM_ALIGNED = 1,
    CUDNN_PTR_16B_ALIGNED = 2,
} cudnnFusedOpsPointerPlaceHolder_t;

typedef enum {
    /* set: pass void* pointing to dev memory */
    /* get: pass void** pointing to host memory */
    CUDNN_PTR_XDATA = 0,
    CUDNN_PTR_BN_EQSCALE = 1,
    CUDNN_PTR_BN_EQBIAS = 2,
    CUDNN_PTR_WDATA = 3,
    CUDNN_PTR_DWDATA = 4,
    CUDNN_PTR_YDATA = 5,
    CUDNN_PTR_DYDATA = 6,
    CUDNN_PTR_YSUM = 7,
    CUDNN_PTR_YSQSUM = 8,
    CUDNN_PTR_WORKSPACE = 9,
    CUDNN_PTR_BN_SCALE = 10,
    CUDNN_PTR_BN_BIAS = 11,
    CUDNN_PTR_BN_SAVED_MEAN = 12,
    CUDNN_PTR_BN_SAVED_INVSTD = 13,
    CUDNN_PTR_BN_RUNNING_MEAN = 14,
    CUDNN_PTR_BN_RUNNING_VAR = 15,
    CUDNN_PTR_ZDATA = 16,
    CUDNN_PTR_BN_Z_EQSCALE = 17,
    CUDNN_PTR_BN_Z_EQBIAS = 18,
    CUDNN_PTR_ACTIVATION_BITMASK = 19,
    CUDNN_PTR_DXDATA = 20,
    CUDNN_PTR_DZDATA = 21,
    CUDNN_PTR_BN_DSCALE = 22,
    CUDNN_PTR_BN_DBIAS = 23,

    /* set/get: pass size_t* pointing to host memory */
    CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = 100,
    /* set/get: pass int64_t* pointing to host memory */
    CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = 101,
    /* set/get: pass double* pointing to host memory */
    CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = 102,
    /* set/get: pass double* pointing to host memory */
    CUDNN_SCALAR_DOUBLE_BN_EPSILON = 103,
} cudnnFusedOpsVariantParamLabel_t;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_CUDNN_SUBSET_H__
