if("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0)
endif()

set(HIF8_DECODE_PROBE_STAGE "0" CACHE STRING
    "SoftHif8Gemm decode probe: 0=off, 1=DMA, 2=U8-to-half, 3=scale, 4=offset cast, 5=gather, 6=GM staging")
if(NOT HIF8_DECODE_PROBE_STAGE MATCHES "^[0-6]$")
    message(FATAL_ERROR
        "HIF8_DECODE_PROBE_STAGE must be an integer from 0 through 6")
endif()

set(HIF8_ROWS_PER_CHUNK "4" CACHE STRING
    "SoftHif8Gemm online decode rows per vector chunk")
if(NOT HIF8_ROWS_PER_CHUNK MATCHES "^(4|8|16)$")
    message(FATAL_ERROR
        "HIF8_ROWS_PER_CHUNK must be one of 4, 8, or 16")
endif()

set(HIF8_USE_INT_OFFSET_SHIFT "0" CACHE STRING
    "Build soft HiFP8 decode offsets with int32 shift instead of half multiply")
if(NOT HIF8_USE_INT_OFFSET_SHIFT MATCHES "^(0|1)$")
    message(FATAL_ERROR
        "HIF8_USE_INT_OFFSET_SHIFT must be 0 or 1")
endif()

set(HIF8_DECODE_IMPL "0" CACHE STRING
    "SoftHif8Gemm decode implementation: 0=BF16 LUT, 1=FP16 LUT plus BF16 cast")
if(NOT HIF8_DECODE_IMPL MATCHES "^(0|1)$")
    message(FATAL_ERROR
        "HIF8_DECODE_IMPL must be 0 or 1")
endif()

set(HIF8_DECODE_QUEUE_DEPTH "2" CACHE STRING
    "SoftHif8Gemm raw/decode queue depth and initial DMA lookahead")
if(NOT HIF8_DECODE_QUEUE_DEPTH MATCHES "^(2|3|4)$")
    message(FATAL_ERROR
        "HIF8_DECODE_QUEUE_DEPTH must be one of 2, 3, or 4")
endif()

set(HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET "0" CACHE STRING
    "Reuse the BF16 decode output slot as FP16 offset scratch")
if(NOT HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET MATCHES "^(0|1)$")
    message(FATAL_ERROR
        "HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET must be 0 or 1")
endif()

set(HIF8_USE_TSCM_PIPELINE "0" CACHE STRING
    "Decode HiFP8 operands directly into shared TSCM before Cube matmul")
if(NOT HIF8_USE_TSCM_PIPELINE MATCHES "^(0|1)$")
    message(FATAL_ERROR "HIF8_USE_TSCM_PIPELINE must be 0 or 1")
endif()

# With tileK=2048, an eight-row chunk fits C220 UB only when the BF16 output
# queue doubles as the temporary FP16 offset storage.  Sixteen rows exceed
# the UB budget even with that reuse (two raw/decode queue slots plus the
# uint32 Gather offsets alone need 320 KiB), so fail at configure time instead
# of producing a binary that faults on device.
if(HIF8_ROWS_PER_CHUNK STREQUAL "8" AND
   (NOT HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET STREQUAL "1" OR
    NOT HIF8_DECODE_IMPL STREQUAL "0" OR
    NOT HIF8_USE_TSCM_PIPELINE STREQUAL "0"))
    message(FATAL_ERROR
        "HIF8_ROWS_PER_CHUNK=8 requires BF16 LUT/GM staging and HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET=1")
endif()
if(HIF8_ROWS_PER_CHUNK STREQUAL "16")
    message(FATAL_ERROR
        "HIF8_ROWS_PER_CHUNK=16 exceeds the C220 UB budget at tileK=2048")
endif()

set(HIF8_LOCAL_TILE_SYNC "0" CACHE STRING
    "Use per-core staging events instead of a global barrier between M tiles")
if(NOT HIF8_LOCAL_TILE_SYNC MATCHES "^(0|1)$")
    message(FATAL_ERROR "HIF8_LOCAL_TILE_SYNC must be 0 or 1")
endif()

set(HIF8_SKIP_FINAL_TILE_BARRIER "1" CACHE STRING
    "Skip the redundant global barrier after the final output tile")
if(NOT HIF8_SKIP_FINAL_TILE_BARRIER MATCHES "^(0|1)$")
    message(FATAL_ERROR "HIF8_SKIP_FINAL_TILE_BARRIER must be 0 or 1")
endif()

set(HIF8_PAIRWISE_EPILOGUE_SYNC "1" CACHE STRING
    "Use per-AIC/AIV output-ready events before the vector epilogue")
if(NOT HIF8_PAIRWISE_EPILOGUE_SYNC MATCHES "^(0|1)$")
    message(FATAL_ERROR "HIF8_PAIRWISE_EPILOGUE_SYNC must be 0 or 1")
endif()

set(HIF8_REUSE_WEIGHT_TILES "1" CACHE STRING
    "Decode each N/K weight tile once and reuse it across M tiles")
if(NOT HIF8_REUSE_WEIGHT_TILES MATCHES "^(0|1)$")
    message(FATAL_ERROR "HIF8_REUSE_WEIGHT_TILES must be 0 or 1")
endif()

set(HIF8_WIDE_N_TILE "1" CACHE STRING
    "Stage 256 output columns and run two explicit 128-column Cube slices")
if(NOT HIF8_WIDE_N_TILE MATCHES "^(0|1)$")
    message(FATAL_ERROR "HIF8_WIDE_N_TILE must be 0 or 1")
endif()

set(HIF8_FUSED_VECTOR_ENCODE "0" CACHE STRING
    "Use the vector FP32-index activation encoder in fused soft HiFloat8 GEMM")
if(NOT HIF8_FUSED_VECTOR_ENCODE MATCHES "^(0|1)$")
    message(FATAL_ERROR "HIF8_FUSED_VECTOR_ENCODE must be 0 or 1")
endif()

set(SOFT_HIF8_GEMM_OPTIONS "")

if(NOT HIF8_ROWS_PER_CHUNK STREQUAL "4")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_ROWS_PER_CHUNK=${HIF8_ROWS_PER_CHUNK})
endif()

if(NOT HIF8_DECODE_PROBE_STAGE STREQUAL "0")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_DECODE_PROBE_STAGE=${HIF8_DECODE_PROBE_STAGE})
endif()

if(HIF8_USE_INT_OFFSET_SHIFT STREQUAL "1")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_USE_INT_OFFSET_SHIFT=1)
endif()

if(NOT HIF8_DECODE_IMPL STREQUAL "0")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_DECODE_IMPL=${HIF8_DECODE_IMPL})
endif()

if(NOT HIF8_DECODE_QUEUE_DEPTH STREQUAL "2")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_DECODE_QUEUE_DEPTH=${HIF8_DECODE_QUEUE_DEPTH})
endif()

if(HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET STREQUAL "1")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_REUSE_DECODE_OUTPUT_AS_OFFSET=1)
endif()

if(HIF8_USE_TSCM_PIPELINE STREQUAL "1")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_USE_TSCM_PIPELINE=1)
endif()

if(HIF8_LOCAL_TILE_SYNC STREQUAL "1")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_LOCAL_TILE_SYNC=1)
endif()

if(HIF8_SKIP_FINAL_TILE_BARRIER STREQUAL "1")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_SKIP_FINAL_TILE_BARRIER=1)
endif()

if(HIF8_PAIRWISE_EPILOGUE_SYNC STREQUAL "1")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_PAIRWISE_EPILOGUE_SYNC=1)
endif()

if(HIF8_REUSE_WEIGHT_TILES STREQUAL "0")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_REUSE_WEIGHT_TILES=0)
endif()

if(HIF8_WIDE_N_TILE STREQUAL "1")
    list(APPEND SOFT_HIF8_GEMM_OPTIONS
        -DHIF8_WIDE_N_TILE=1)
endif()

# CANN 9.0 replaces options on repeated calls for the same op.
# Submit all defines together to preserve combined settings.
if(NOT "${SOFT_HIF8_GEMM_OPTIONS}" STREQUAL "")
    foreach(SOFT_HIF8_GEMM_OP SoftHif8Gemm SoftHif8PrefillGemm)
        npu_op_kernel_options(
            ascendc_kernels ${SOFT_HIF8_GEMM_OP}
            OPTIONS ${SOFT_HIF8_GEMM_OPTIONS}
        )
    endforeach()
endif()

if(HIF8_FUSED_VECTOR_ENCODE STREQUAL "1")
    npu_op_kernel_options(
        ascendc_kernels SoftHif8FusedGemm
        OPTIONS -DHIF8_FUSED_VECTOR_ENCODE=1
    )
endif()

npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY vllm_ascend_hif8_tiling
)

npu_op_package_add(${package_name} LIBRARY ascendc_kernels)
