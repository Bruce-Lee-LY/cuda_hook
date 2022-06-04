// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 17:19:12 on Sun, May 29, 2022
//
// Description: nvtx subset

#ifndef __CUDA_HOOK_NVTX_SUBSET_H__
#define __CUDA_HOOK_NVTX_SUBSET_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Tools Extension API version
 */
#define NVTX_VERSION 2

/**
 * Size of the nvtxEventAttributes_t structure.
 */
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE ((uint16_t)(sizeof(nvtxEventAttributes_t)))

/**
 * Size of the nvtxInitializationAttributes_t structure.
 */
#define NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE ((uint16_t)(sizeof(nvtxInitializationAttributes_t)))

#define NVTX_NO_PUSH_POP_TRACKING ((int)-2)

typedef uint64_t nvtxRangeId_t;

/* \brief String Handle Structure.
 * \anchor STRING_HANDLE_STRUCTURE
 *
 * This structure is opaque to the user and is used as a handle to reference
 * a string.  The tools will return a pointer through the API for the application
 * to hold on it's behalf to reference the string in the future.
 *
 */
typedef struct nvtxStringHandle *nvtxStringHandle_t;

/* \brief Domain Handle Structure.
 * \anchor DOMAIN_HANDLE_STRUCTURE
 *
 * This structure is opaque to the user and is used as a handle to reference
 * a domain.  The tools will return a pointer through the API for the application
 * to hold on its behalf to reference the domain in the future.
 *
 */
typedef struct nvtxDomainHandle *nvtxDomainHandle_t;

/* ========================================================================= */
/** \defgroup GENERAL General
 * @{
 */

/** ---------------------------------------------------------------------------
 * Color Types
 * ------------------------------------------------------------------------- */
typedef enum nvtxColorType_t {
    NVTX_COLOR_UNKNOWN = 0, /**< Color attribute is unused. */
    NVTX_COLOR_ARGB = 1     /**< An ARGB color is provided. */
} nvtxColorType_t;

/** ---------------------------------------------------------------------------
 * Message Types
 * ------------------------------------------------------------------------- */
typedef enum nvtxMessageType_t {
    NVTX_MESSAGE_UNKNOWN = 0,      /**< Message payload is unused. */
    NVTX_MESSAGE_TYPE_ASCII = 1,   /**< A character sequence is used as payload. */
    NVTX_MESSAGE_TYPE_UNICODE = 2, /**< A wide character sequence is used as payload. */
    /* NVTX_VERSION_2 */
    NVTX_MESSAGE_TYPE_REGISTERED = 3 /**< A unique string handle that was registered
                                           with \ref nvtxDomainRegisterStringA() or
                                           \ref nvtxDomainRegisterStringW(). */
} nvtxMessageType_t;

typedef union nvtxMessageValue_t {
    const char *ascii;
    const wchar_t *unicode;
    /* NVTX_VERSION_2 */
    nvtxStringHandle_t registered;
} nvtxMessageValue_t;

/** @} */ /*END defgroup*/

/* ========================================================================= */
/** \defgroup INITIALIZATION Initialization
 * @{
 * Typically the tool's library that plugs into NVTX is indirectly
 * loaded via enviromental properties that are platform specific.
 * For some platform or special cases, the user may be required
 * to instead explicity initialize instead though.  This can also
 * be helpful to control when the API loads a tool's library instead
 * of what would typically be the first function call to emit info.
 */

/** ---------------------------------------------------------------------------
 * Initialization Modes
 * ------------------------------------------------------------------------- */
typedef enum nvtxInitializationMode_t {
    NVTX_INITIALIZATION_MODE_UNKNOWN =
        0, /**< A platform that supports indirect initialization will attempt this style, otherwise expect failure. */
    NVTX_INITIALIZATION_MODE_CALLBACK_V1 = 1, /**< A function pointer conforming to NVTX_VERSION=1 will be used. */
    NVTX_INITIALIZATION_MODE_CALLBACK_V2 = 2, /**< A function pointer conforming to NVTX_VERSION=2 will be used. */
    NVTX_INITIALIZATION_MODE_SIZE
} nvtxInitializationMode_t;

/** \brief Initialization Attribute Structure.
* \anchor INITIALIZATION_ATTRIBUTE_STRUCTURE
*
* This structure is used to describe the attributes used for initialization
* of the NVTX API.
*
* \par Initializing the Attributes
*
* The caller should always perform the following three tasks when using
* attributes:
* <ul>
*    <li>Zero the structure
*    <li>Set the version field
*    <li>Set the size field
* </ul>
*
* Zeroing the structure sets all the event attributes types and values
* to the default value.
*
* The version and size field are used by the Tools Extension
* implementation to handle multiple versions of the attributes structure.
* NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE may be used for the size.
*
* It is recommended that the caller use one of the following to methods
* to initialize the event attributes structure:
*
* \par Method 1: Initializing nvtxInitializationAttributes_t for future compatibility
* \code
* nvtxInitializationAttributes_t initAttribs = {0};
* initAttribs.version = NVTX_VERSION;
* initAttribs.size = NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE;
* \endcode
*
* \par Method 2: Initializing nvtxInitializationAttributes_t for a specific version
* \code
* nvtxInitializationAttributes_t initAttribs = {0};
* initAttribs.version =2;
* initAttribs.size = (uint16_t)(sizeof(nvtxInitializationAttributes_v2));
* \endcode
*
* If the caller uses Method 1 it is critical that the entire binary
* layout of the structure be configured to 0 so that all fields
* are initialized to the default value.
*
* The caller should either use both NVTX_VERSION and
* NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
* and a versioned type (Method 2).  Using a mix of the two methods
* will likely cause either source level incompatibility or binary
* incompatibility in the future.
*
* \par Settings Attribute Types and Values
*
*
* \par Example:
* \code
* // Initialize
* nvtxInitializationAttributes_t initAttribs = {0};
* initAttribs.version = NVTX_VERSION;
* initAttribs.size = NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE;
*
* // Configure the Attributes
* initAttribs.mode = NVTX_INITIALIZATION_MODE_CALLBACK_V2;
* initAttribs.fnptr = InitializeInjectionNvtx2;
* \endcode

* \sa
* ::nvtxInitializationMode_t
* ::nvtxInitialize
*/
typedef struct nvtxInitializationAttributes_v2 {
    /**
     * \brief Version flag of the structure.
     *
     * Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
     * supported in this header file. This can optionally be overridden to
     * another version of the tools extension library.
     */
    uint16_t version;

    /**
     * \brief Size of the structure.
     *
     * Needs to be set to the size in bytes of the event attribute
     * structure used to specify the event.
     */
    uint16_t size;

    /**
     * \brief Mode of initialization.
     *
     * The mode of initialization dictates the overall behavior and which
     * attributes in this struct will be used.
     *
     * Default Value is NVTX_INITIALIZATION_MODE_UNKNOWN = 0
     * \sa
     * ::nvtxInitializationMode_t
     */
    uint32_t mode;

    /**
    * \brief Function pointer used for initialization if the mode requires
    *
    * The user has retrieved this function pointer from the tool library
    * and would like to use it to initialize.  The mode must be set to a
    * NVTX_INITIALIZATION_MODE_CALLBACK_V# for this to be used.  The mode
    * will dictate the expectations for this member.  The function signature
    * will be cast from void(*)() to the appropriate signature for the mode.
    * the expected behavior of the function will also depend on the mode
    * beyond the simple function signature.
    *
    * Default Value is NVTX_INITIALIZATION_MODE_UNKNOWN which will either
    * initialize based on external properties or fail if not supported on
    * the given platform.

    * \sa
    * ::nvtxInitializationMode_t
    */
    void (*fnptr)(void);

} nvtxInitializationAttributes_v2;

typedef struct nvtxInitializationAttributes_v2 nvtxInitializationAttributes_t;

/** ---------------------------------------------------------------------------
 * Payload Types
 * ------------------------------------------------------------------------- */
typedef enum nvtxPayloadType_t {
    NVTX_PAYLOAD_UNKNOWN = 0,             /**< Color payload is unused. */
    NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 = 1, /**< A 64 bit unsigned integer value is used as payload. */
    NVTX_PAYLOAD_TYPE_INT64 = 2,          /**< A 64 bit signed integer value is used as payload. */
    NVTX_PAYLOAD_TYPE_DOUBLE = 3,         /**< A 64 bit floating point value is used as payload. */
    /* NVTX_VERSION_2 */
    NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 = 4, /**< A 32 bit floating point value is used as payload. */
    NVTX_PAYLOAD_TYPE_INT32 = 5,          /**< A 32 bit floating point value is used as payload. */
    NVTX_PAYLOAD_TYPE_FLOAT = 6           /**< A 32 bit floating point value is used as payload. */
} nvtxPayloadType_t;

/** \brief Event Attribute Structure.
 * \anchor EVENT_ATTRIBUTE_STRUCTURE
 *
 * This structure is used to describe the attributes of an event. The layout of
 * the structure is defined by a specific version of the tools extension
 * library and can change between different versions of the Tools Extension
 * library.
 *
 * \par Initializing the Attributes
 *
 * The caller should always perform the following three tasks when using
 * attributes:
 * <ul>
 *    <li>Zero the structure
 *    <li>Set the version field
 *    <li>Set the size field
 * </ul>
 *
 * Zeroing the structure sets all the event attributes types and values
 * to the default value.
 *
 * The version and size field are used by the Tools Extension
 * implementation to handle multiple versions of the attributes structure.
 *
 * It is recommended that the caller use one of the following to methods
 * to initialize the event attributes structure:
 *
 * \par Method 1: Initializing nvtxEventAttributes for future compatibility
 * \code
 * nvtxEventAttributes_t eventAttrib = {0};
 * eventAttrib.version = NVTX_VERSION;
 * eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
 * \endcode
 *
 * \par Method 2: Initializing nvtxEventAttributes for a specific version
 * \code
 * nvtxEventAttributes_t eventAttrib = {0};
 * eventAttrib.version = 1;
 * eventAttrib.size = (uint16_t)(sizeof(nvtxEventAttributes_v1));
 * \endcode
 *
 * If the caller uses Method 1 it is critical that the entire binary
 * layout of the structure be configured to 0 so that all fields
 * are initialized to the default value.
 *
 * The caller should either use both NVTX_VERSION and
 * NVTX_EVENT_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
 * and a versioned type (Method 2).  Using a mix of the two methods
 * will likely cause either source level incompatibility or binary
 * incompatibility in the future.
 *
 * \par Settings Attribute Types and Values
 *
 *
 * \par Example:
 * \code
 * // Initialize
 * nvtxEventAttributes_t eventAttrib = {0};
 * eventAttrib.version = NVTX_VERSION;
 * eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
 *
 * // Configure the Attributes
 * eventAttrib.colorType = NVTX_COLOR_ARGB;
 * eventAttrib.color = 0xFF880000;
 * eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
 * eventAttrib.message.ascii = "Example";
 * \endcode
 *
 * In the example the caller does not have to set the value of
 * \ref ::nvtxEventAttributes_v2::category or
 * \ref ::nvtxEventAttributes_v2::payload as these fields were set to
 * the default value by {0}.
 * \sa
 * ::nvtxDomainMarkEx
 * ::nvtxDomainRangeStartEx
 * ::nvtxDomainRangePushEx
 */
typedef struct nvtxEventAttributes_v2 {
    /**
     * \brief Version flag of the structure.
     *
     * Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
     * supported in this header file. This can optionally be overridden to
     * another version of the tools extension library.
     */
    uint16_t version;

    /**
     * \brief Size of the structure.
     *
     * Needs to be set to the size in bytes of the event attribute
     * structure used to specify the event.
     */
    uint16_t size;

    /**
     * \brief ID of the category the event is assigned to.
     *
     * A category is a user-controlled ID that can be used to group
     * events.  The tool may use category IDs to improve filtering or
     * enable grouping of events in the same category. The functions
     * \ref ::nvtxNameCategoryA or \ref ::nvtxNameCategoryW can be used
     * to name a category.
     *
     * Default Value is 0
     */
    uint32_t category;

    /** \brief Color type specified in this attribute structure.
     *
     * Defines the color format of the attribute structure's \ref COLOR_FIELD
     * "color" field.
     *
     * Default Value is NVTX_COLOR_UNKNOWN
     */
    int32_t colorType; /* nvtxColorType_t */

    /** \brief Color assigned to this event. \anchor COLOR_FIELD
     *
     * The color that the tool should use to visualize the event.
     */
    uint32_t color;

    /**
     * \brief Payload type specified in this attribute structure.
     *
     * Defines the payload format of the attribute structure's \ref PAYLOAD_FIELD
     * "payload" field.
     *
     * Default Value is NVTX_PAYLOAD_UNKNOWN
     */
    int32_t payloadType; /* nvtxPayloadType_t */

    int32_t reserved0;

    /**
     * \brief Payload assigned to this event. \anchor PAYLOAD_FIELD
     *
     * A numerical value that can be used to annotate an event. The tool could
     * use the payload data to reconstruct graphs and diagrams.
     */
    union payload_t {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
        /* NVTX_VERSION_2 */
        uint32_t uiValue;
        int32_t iValue;
        float fValue;
    } payload;

    /** \brief Message type specified in this attribute structure.
     *
     * Defines the message format of the attribute structure's \ref MESSAGE_FIELD
     * "message" field.
     *
     * Default Value is NVTX_MESSAGE_UNKNOWN
     */
    int32_t messageType; /* nvtxMessageType_t */

    /** \brief Message assigned to this attribute structure. \anchor MESSAGE_FIELD
     *
     * The text message that is attached to an event.
     */
    nvtxMessageValue_t message;

} nvtxEventAttributes_v2;

typedef struct nvtxEventAttributes_v2 nvtxEventAttributes_t;

/*  ------------------------------------------------------------------------- */
/** \cond SHOW_HIDDEN
 * \brief Resource typing helpers.
 *
 * Classes are used to make it easy to create a series of resource types
 * per API without collisions
 */
#define NVTX_RESOURCE_MAKE_TYPE(CLASS, INDEX) ((((uint32_t)(NVTX_RESOURCE_CLASS_##CLASS)) << 16) | ((uint32_t)(INDEX)))
#define NVTX_RESOURCE_CLASS_GENERIC 1
/** \endcond */

/* ------------------------------------------------------------------------- */
/** \brief Generic resource type for when a resource class is not available.
 *
 * \sa
 * ::nvtxDomainResourceCreate
 *
 * \version \NVTX_VERSION_2
 */
typedef enum nvtxResourceGenericType_t {
    NVTX_RESOURCE_TYPE_UNKNOWN = 0,
    NVTX_RESOURCE_TYPE_GENERIC_POINTER =
        NVTX_RESOURCE_MAKE_TYPE(GENERIC, 1), /**< Generic pointer assumed to have no collisions with other pointers. */
    NVTX_RESOURCE_TYPE_GENERIC_HANDLE =
        NVTX_RESOURCE_MAKE_TYPE(GENERIC, 2), /**< Generic handle assumed to have no collisions with other handles. */
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE = NVTX_RESOURCE_MAKE_TYPE(GENERIC, 3), /**< OS native thread identifier. */
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX = NVTX_RESOURCE_MAKE_TYPE(GENERIC, 4)   /**< POSIX pthread identifier. */
} nvtxResourceGenericType_t;

/** \brief Resource Attribute Structure.
 * \anchor RESOURCE_ATTRIBUTE_STRUCTURE
 *
 * This structure is used to describe the attributes of a resource. The layout of
 * the structure is defined by a specific version of the tools extension
 * library and can change between different versions of the Tools Extension
 * library.
 *
 * \par Initializing the Attributes
 *
 * The caller should always perform the following three tasks when using
 * attributes:
 * <ul>
 *    <li>Zero the structure
 *    <li>Set the version field
 *    <li>Set the size field
 * </ul>
 *
 * Zeroing the structure sets all the resource attributes types and values
 * to the default value.
 *
 * The version and size field are used by the Tools Extension
 * implementation to handle multiple versions of the attributes structure.
 *
 * It is recommended that the caller use one of the following to methods
 * to initialize the event attributes structure:
 *
 * \par Method 1: Initializing nvtxEventAttributes for future compatibility
 * \code
 * nvtxResourceAttributes_t attribs = {0};
 * attribs.version = NVTX_VERSION;
 * attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
 * \endcode
 *
 * \par Method 2: Initializing nvtxEventAttributes for a specific version
 * \code
 * nvtxResourceAttributes_v0 attribs = {0};
 * attribs.version = 2;
 * attribs.size = (uint16_t)(sizeof(nvtxResourceAttributes_v0));
 * \endcode
 *
 * If the caller uses Method 1 it is critical that the entire binary
 * layout of the structure be configured to 0 so that all fields
 * are initialized to the default value.
 *
 * The caller should either use both NVTX_VERSION and
 * NVTX_RESOURCE_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
 * and a versioned type (Method 2).  Using a mix of the two methods
 * will likely cause either source level incompatibility or binary
 * incompatibility in the future.
 *
 * \par Settings Attribute Types and Values
 *
 *
 * \par Example:
 * \code
 * nvtxDomainHandle_t domain = nvtxDomainCreateA("example domain");
 *
 * // Initialize
 * nvtxResourceAttributes_t attribs = {0};
 * attribs.version = NVTX_VERSION;
 * attribs.size = NVTX_RESOURCE_ATTRIB_STRUCT_SIZE;
 *
 * // Configure the Attributes
 * attribs.identifierType = NVTX_RESOURCE_TYPE_GENERIC_POINTER;
 * attribs.identifier.pValue = (const void*)pMutex;
 * attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
 * attribs.message.ascii = "Single thread access to database.";
 *
 * nvtxResourceHandle_t handle = nvtxDomainResourceCreate(domain, attribs);
 * \endcode
 *
 * \sa
 * ::nvtxDomainResourceCreate
 */
typedef struct nvtxResourceAttributes_v0 {
    /**
     * \brief Version flag of the structure.
     *
     * Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
     * supported in this header file. This can optionally be overridden to
     * another version of the tools extension library.
     */
    uint16_t version;

    /**
     * \brief Size of the structure.
     *
     * Needs to be set to the size in bytes of this attribute
     * structure.
     */
    uint16_t size;

    /**
     * \brief Identifier type specifies how to interpret the identifier field
     *
     * Defines the identifier format of the attribute structure's \ref RESOURCE_IDENTIFIER_FIELD
     * "identifier" field.
     *
     * Default Value is NVTX_RESOURCE_TYPE_UNKNOWN
     */
    int32_t identifierType; /* values from enums following the pattern nvtxResource[name]Type_t */

    /**
     * \brief Identifier for the resource.
     * \anchor RESOURCE_IDENTIFIER_FIELD
     *
     * An identifier may be a pointer or a handle to an OS or middleware API object.
     * The resource type will assist in avoiding collisions where handles values may collide.
     */
    union identifier_t {
        const void *pValue;
        uint64_t ullValue;
    } identifier;

    /** \brief Message type specified in this attribute structure.
     *
     * Defines the message format of the attribute structure's \ref RESOURCE_MESSAGE_FIELD
     * "message" field.
     *
     * Default Value is NVTX_MESSAGE_UNKNOWN
     */
    int32_t messageType; /* nvtxMessageType_t */

    /** \brief Message assigned to this attribute structure. \anchor RESOURCE_MESSAGE_FIELD
     *
     * The text message that is attached to a resource.
     */
    nvtxMessageValue_t message;

} nvtxResourceAttributes_v0;

typedef struct nvtxResourceAttributes_v0 nvtxResourceAttributes_t;

/* \cond SHOW_HIDDEN
 * \version \NVTX_VERSION_2
 */
#define NVTX_RESOURCE_ATTRIB_STRUCT_SIZE ((uint16_t)(sizeof(nvtxResourceAttributes_v0)))
typedef struct nvtxResourceHandle *nvtxResourceHandle_t;

#ifdef UNICODE
#define nvtxMark nvtxMarkW
#define nvtxRangeStart nvtxRangeStartW
#define nvtxRangePush nvtxRangePushW
#define nvtxNameCategory nvtxNameCategoryW
#define nvtxNameOsThread nvtxNameOsThreadW
/* NVTX_VERSION_2 */
#define nvtxDomainCreate nvtxDomainCreateW
#define nvtxDomainRegisterString nvtxDomainRegisterStringW
#define nvtxDomainNameCategory nvtxDomainNameCategoryW
#else
#define nvtxMark nvtxMarkA
#define nvtxRangeStart nvtxRangeStartA
#define nvtxRangePush nvtxRangePushA
#define nvtxNameCategory nvtxNameCategoryA
#define nvtxNameOsThread nvtxNameOsThreadA
/* NVTX_VERSION_2 */
#define nvtxDomainCreate nvtxDomainCreateA
#define nvtxDomainRegisterString nvtxDomainRegisterStringA
#define nvtxDomainNameCategory nvtxDomainNameCategoryA
#endif

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN
 * \brief Used to build a non-colliding value for resource types separated class
 * \version \NVTX_VERSION_2
 */
#define NVTX_RESOURCE_CLASS_CUDA 4
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \brief Resource types for CUDA
 */
typedef enum nvtxResourceCUDAType_t {
    NVTX_RESOURCE_TYPE_CUDA_DEVICE = NVTX_RESOURCE_MAKE_TYPE(CUDA, 1),  /* CUdevice */
    NVTX_RESOURCE_TYPE_CUDA_CONTEXT = NVTX_RESOURCE_MAKE_TYPE(CUDA, 2), /* CUcontext */
    NVTX_RESOURCE_TYPE_CUDA_STREAM = NVTX_RESOURCE_MAKE_TYPE(CUDA, 3),  /* CUstream */
    NVTX_RESOURCE_TYPE_CUDA_EVENT = NVTX_RESOURCE_MAKE_TYPE(CUDA, 4)    /* CUevent */
} nvtxResourceCUDAType_t;

#ifdef UNICODE
#define nvtxNameCuDevice nvtxNameCuDeviceW
#define nvtxNameCuContext nvtxNameCuContextW
#define nvtxNameCuStream nvtxNameCuStreamW
#define nvtxNameCuEvent nvtxNameCuEventW
#else
#define nvtxNameCuDevice nvtxNameCuDeviceA
#define nvtxNameCuContext nvtxNameCuContextA
#define nvtxNameCuStream nvtxNameCuStreamA
#define nvtxNameCuEvent nvtxNameCuEventA
#endif

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN
 * \brief Used to build a non-colliding value for resource types separated class
 * \version \NVTX_VERSION_2
 */
#define NVTX_RESOURCE_CLASS_CUDART 5
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \brief Resource types for CUDART
 */
typedef enum nvtxResourceCUDARTType_t {
    NVTX_RESOURCE_TYPE_CUDART_DEVICE = NVTX_RESOURCE_MAKE_TYPE(CUDART, 0), /* int device */
    NVTX_RESOURCE_TYPE_CUDART_STREAM = NVTX_RESOURCE_MAKE_TYPE(CUDART, 1), /* cudaStream_t */
    NVTX_RESOURCE_TYPE_CUDART_EVENT = NVTX_RESOURCE_MAKE_TYPE(CUDART, 2)   /* cudaEvent_t */
} nvtxResourceCUDARTType_t;

#ifdef UNICODE
#define nvtxNameCudaDevice nvtxNameCudaDeviceW
#define nvtxNameCudaStream nvtxNameCudaStreamW
#define nvtxNameCudaEvent nvtxNameCudaEventW
#else
#define nvtxNameCudaDevice nvtxNameCudaDeviceA
#define nvtxNameCudaStream nvtxNameCudaStreamA
#define nvtxNameCudaEvent nvtxNameCudaEventA
#endif

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN
 * \brief Used to build a non-colliding value for resource types separated class
 * \version \NVTX_VERSION_2
 */
#define NVTX_RESOURCE_CLASS_OPENCL 6
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \brief Resource types for OpenCL
 */
typedef enum nvtxResourceOpenCLType_t {
    NVTX_RESOURCE_TYPE_OPENCL_DEVICE = NVTX_RESOURCE_MAKE_TYPE(OPENCL, 1),
    NVTX_RESOURCE_TYPE_OPENCL_CONTEXT = NVTX_RESOURCE_MAKE_TYPE(OPENCL, 2),
    NVTX_RESOURCE_TYPE_OPENCL_COMMANDQUEUE = NVTX_RESOURCE_MAKE_TYPE(OPENCL, 3),
    NVTX_RESOURCE_TYPE_OPENCL_MEMOBJECT = NVTX_RESOURCE_MAKE_TYPE(OPENCL, 4),
    NVTX_RESOURCE_TYPE_OPENCL_SAMPLER = NVTX_RESOURCE_MAKE_TYPE(OPENCL, 5),
    NVTX_RESOURCE_TYPE_OPENCL_PROGRAM = NVTX_RESOURCE_MAKE_TYPE(OPENCL, 6),
    NVTX_RESOURCE_TYPE_OPENCL_EVENT = NVTX_RESOURCE_MAKE_TYPE(OPENCL, 7)
} nvtxResourceOpenCLType_t;

#ifdef UNICODE
#define nvtxNameClDevice nvtxNameClDeviceW
#define nvtxNameClContext nvtxNameClContextW
#define nvtxNameClCommandQueue nvtxNameClCommandQueueW
#define nvtxNameClMemObject nvtxNameClMemObjectW
#define nvtxNameClSampler nvtxNameClSamplerW
#define nvtxNameClProgram nvtxNameClProgramW
#define nvtxNameClEvent nvtxNameClEventW
#else
#define nvtxNameClDevice nvtxNameClDeviceA
#define nvtxNameClContext nvtxNameClContextA
#define nvtxNameClCommandQueue nvtxNameClCommandQueueA
#define nvtxNameClMemObject nvtxNameClMemObjectA
#define nvtxNameClSampler nvtxNameClSamplerA
#define nvtxNameClProgram nvtxNameClProgramA
#define nvtxNameClEvent nvtxNameClEventA
#endif

/* \cond SHOW_HIDDEN
 * \version \NVTX_VERSION_2
 */
#define NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE ((uint16_t)(sizeof(nvtxSyncUserAttributes_v0)))
/** \endcond */

/**
* \page PAGE_SYNCHRONIZATION Synchronization
*
* This section covers a subset of the API that allow users to track additional
* synchronization details of their application.   Naming OS synchronization primitives
* may allow users to better understand the data collected by traced synchronization
* APIs.  Additionally, a user defined synchronization object can allow the users to
* to tell the tools when the user is building their own synchronization system
* that do not rely on the OS to provide behaviors and instead use techniques like
* atomic operations and spinlocks.
*
* See module \ref SYNCHRONIZATION for details.
*
* \par Example:
* \code
* class MyMutex
* {
*     volatile long bLocked;
*     nvtxSyncUser_t hSync;
* public:
*     MyMutex(const char* name, nvtxDomainHandle_t d){
*          bLocked = 0;
*
*          nvtxSyncUserAttributes_t attribs = { 0 };
*          attribs.version = NVTX_VERSION;
*          attribs.size = NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
*          attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
*          attribs.message.ascii = name;
*          hSync = nvtxDomainSyncUserCreate(d, &attribs);
*     }
*
*     ~MyMutex() {
*          nvtxDomainSyncUserDestroy(hSync);
*     }
*
*     bool Lock() {
*          nvtxDomainSyncUserAcquireStart(hSync);
*          bool acquired = __sync_bool_compare_and_swap(&bLocked, 0, 1);//atomic compiler intrinsic

*          if (acquired) {
*              nvtxDomainSyncUserAcquireSuccess(hSync);
*          }
*          else {
*              nvtxDomainSyncUserAcquireFailed(hSync);
*          }
*          return acquired;
*     }

*     void Unlock() {
*          nvtxDomainSyncUserReleasing(hSync);
*          bLocked = false;
*     }
* };
* \endcode
*
* \version \NVTX_VERSION_2
*/

/*  ------------------------------------------------------------------------- */
/* \cond SHOW_HIDDEN
 * \brief Used to build a non-colliding value for resource types separated class
 * \version \NVTX_VERSION_2
 */
#define NVTX_RESOURCE_CLASS_SYNC_OS 2 /**< Synchronization objects that are OS specific. */
#define NVTX_RESOURCE_CLASS_SYNC_PTHREAD                               \
    3 /**< Synchronization objects that are from the POSIX Threads API \
         (pthread)*/
/** \endcond */

/*  ------------------------------------------------------------------------- */
/** \defgroup SYNCHRONIZATION Synchronization
 * See page \ref PAGE_SYNCHRONIZATION.
 * @{
 */

/** \brief Resource type values for OSs with POSIX Thread API support
 */
typedef enum nvtxResourceSyncPosixThreadType_t {
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_MUTEX = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 1),     /* pthread_mutex_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_CONDITION = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 2), /* pthread_cond_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_RWLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 3),    /* pthread_rwlock_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_BARRIER = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 4),   /* pthread_barrier_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_SPINLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 5),  /* pthread_spinlock_t  */
    NVTX_RESOURCE_TYPE_SYNC_PTHREAD_ONCE = NVTX_RESOURCE_MAKE_TYPE(SYNC_PTHREAD, 6)       /* pthread_once_t  */
} nvtxResourceSyncPosixThreadType_t;

/** \brief Resource type values for Windows OSs
 */
typedef enum nvtxResourceSyncWindowsType_t {
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_MUTEX = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 1),
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_SEMAPHORE = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 2),
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_EVENT = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 3),
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_CRITICAL_SECTION = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 4),
    NVTX_RESOURCE_TYPE_SYNC_WINDOWS_SRWLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 5)
} nvtxResourceSyncWindowsType_t;

/** \brief Resource type values for Linux and Linux derived OSs such as Android
 * \sa
 * ::nvtxResourceSyncPosixThreadType_t
 */
typedef enum nvtxResourceSyncLinuxType_t {
    NVTX_RESOURCE_TYPE_SYNC_LINUX_MUTEX = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 1),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_FUTEX = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 2),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_SEMAPHORE = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 3),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_COMPLETION = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 4),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_SPINLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 5),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_SEQLOCK = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 6),
    NVTX_RESOURCE_TYPE_SYNC_LINUX_RCU = NVTX_RESOURCE_MAKE_TYPE(SYNC_OS, 7)
} nvtxResourceSyncLinuxType_t;

/** \brief Resource type values for Android come from Linux.
 * \sa
 * ::nvtxResourceSyncLinuxType_t
 * ::nvtxResourceSyncPosixThreadType_t
 */
typedef enum nvtxResourceSyncLinuxType_t nvtxResourceSyncAndroidType_t;

/** \brief User Defined Synchronization Object Handle .
 * \anchor SYNCUSER_HANDLE_STRUCTURE
 *
 * This structure is opaque to the user and is used as a handle to reference
 * a user defined syncrhonization object.  The tools will return a pointer through the API for the application
 * to hold on it's behalf to reference the string in the future.
 *
 */
typedef struct nvtxSyncUser *nvtxSyncUser_t;

/** \brief User Defined Synchronization Object Attributes Structure.
 * \anchor USERDEF_SYNC_ATTRIBUTES_STRUCTURE
 *
 * This structure is used to describe the attributes of a user defined synchronization
 * object.  The layout of the structure is defined by a specific version of the tools
 * extension library and can change between different versions of the Tools Extension
 * library.
 *
 * \par Initializing the Attributes
 *
 * The caller should always perform the following three tasks when using
 * attributes:
 * <ul>
 *    <li>Zero the structure
 *    <li>Set the version field
 *    <li>Set the size field
 * </ul>
 *
 * Zeroing the structure sets all the event attributes types and values
 * to the default value.
 *
 * The version and size field are used by the Tools Extension
 * implementation to handle multiple versions of the attributes structure.
 *
 * It is recommended that the caller use one of the following to methods
 * to initialize the event attributes structure:
 *
 * \par Method 1: Initializing nvtxEventAttributes for future compatibility
 * \code
 * nvtxSyncUserAttributes_t attribs = {0};
 * attribs.version = NVTX_VERSION;
 * attribs.size = NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
 * \endcode
 *
 * \par Method 2: Initializing nvtxSyncUserAttributes_t for a specific version
 * \code
 * nvtxSyncUserAttributes_t attribs = {0};
 * attribs.version = 1;
 * attribs.size = (uint16_t)(sizeof(nvtxSyncUserAttributes_t));
 * \endcode
 *
 * If the caller uses Method 1 it is critical that the entire binary
 * layout of the structure be configured to 0 so that all fields
 * are initialized to the default value.
 *
 * The caller should either use both NVTX_VERSION and
 * NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE (Method 1) or use explicit values
 * and a versioned type (Method 2).  Using a mix of the two methods
 * will likely cause either source level incompatibility or binary
 * incompatibility in the future.
 *
 * \par Settings Attribute Types and Values
 *
 *
 * \par Example:
 * \code
 * // Initialize
 * nvtxSyncUserAttributes_t attribs = {0};
 * attribs.version = NVTX_VERSION;
 * attribs.size = NVTX_SYNCUSER_ATTRIB_STRUCT_SIZE;
 *
 * // Configure the Attributes
 * attribs.messageType = NVTX_MESSAGE_TYPE_ASCII;
 * attribs.message.ascii = "Example";
 * \endcode
 *
 * \sa
 * ::nvtxDomainSyncUserCreate
 */
typedef struct nvtxSyncUserAttributes_v0 {
    /**
     * \brief Version flag of the structure.
     *
     * Needs to be set to NVTX_VERSION to indicate the version of NVTX APIs
     * supported in this header file. This can optionally be overridden to
     * another version of the tools extension library.
     */
    uint16_t version;

    /**
     * \brief Size of the structure.
     *
     * Needs to be set to the size in bytes of the event attribute
     * structure used to specify the event.
     */
    uint16_t size;

    /** \brief Message type specified in this attribute structure.
     *
     * Defines the message format of the attribute structure's \ref nvtxSyncUserAttributes_v0::message
     * "message" field.
     *
     * Default Value is NVTX_MESSAGE_UNKNOWN
     */
    int32_t messageType; /* nvtxMessageType_t */

    /** \brief Message assigned to this attribute structure.
     *
     * The text message that is attached to an event.
     */
    nvtxMessageValue_t message;

} nvtxSyncUserAttributes_v0;

typedef struct nvtxSyncUserAttributes_v0 nvtxSyncUserAttributes_t;

typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;
typedef struct _cl_sampler *cl_sampler;

#ifdef __cplusplus
}
#endif

#endif  // __CUDA_HOOK_NVTX_SUBSET_H__
