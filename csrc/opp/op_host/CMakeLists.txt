set(ops_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/hif8_dynamic_quant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hif8_static_quant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hif8_scaled_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/soft_hif8_gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/soft_hif8_fused_gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/soft_hif8_static_fused_gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/soft_hif8_gemv.cpp
)

npu_op_code_gen(
    SRC ${ops_srcs}
    PACKAGE ${package_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    JOIN_OP_DEF True
    COMPILE_OPTIONS
        -I$ENV{ASCEND_HOME_PATH}/ascendc/include/highlevel_api
        -I$ENV{ASCEND_HOME_PATH}/asc/include/tiling/matrix
        -I$ENV{ASCEND_HOME_PATH}/asc/include/adv_api
        -I$ENV{ASCEND_HOME_PATH}/asc/include
)

file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(vllm_ascend_hif8_opapi ACLNN ${autogen_aclnn_src})
target_compile_options(vllm_ascend_hif8_opapi PRIVATE -fvisibility=hidden)

file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc)
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${group_proto_src} ${proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(vllm_ascend_hif8_proto GRAPH
    ${ops_srcs}
    ${group_proto_src}
    ${proto_src}
)
target_compile_options(vllm_ascend_hif8_proto PRIVATE -fvisibility=hidden)
target_include_directories(vllm_ascend_hif8_proto PRIVATE
    $ENV{ASCEND_HOME_PATH}/ascendc/include/highlevel_api
    $ENV{ASCEND_HOME_PATH}/asc/include
    $ENV{ASCEND_HOME_PATH}/asc/include/adv_api
)
if(HIF8_USE_TSCM_PIPELINE STREQUAL "1")
    target_compile_definitions(vllm_ascend_hif8_proto PRIVATE
        HIF8_USE_TSCM_PIPELINE=1)
endif()
if(HIF8_WIDE_N_TILE STREQUAL "1")
    target_compile_definitions(vllm_ascend_hif8_proto PRIVATE
        HIF8_WIDE_N_TILE=1)
endif()
file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
set_source_files_properties(${fallback_src} PROPERTIES GENERATED TRUE)
npu_op_library(vllm_ascend_hif8_tiling TILING
    ${ops_srcs}
    ${fallback_src}
)
target_compile_options(vllm_ascend_hif8_tiling PRIVATE -fvisibility=hidden)
target_include_directories(vllm_ascend_hif8_tiling PRIVATE
    $ENV{ASCEND_HOME_PATH}/ascendc/include/highlevel_api
    $ENV{ASCEND_HOME_PATH}/asc/include
    $ENV{ASCEND_HOME_PATH}/asc/include/adv_api
)
if(HIF8_USE_TSCM_PIPELINE STREQUAL "1")
    target_compile_definitions(vllm_ascend_hif8_tiling PRIVATE
        HIF8_USE_TSCM_PIPELINE=1)
endif()
if(HIF8_WIDE_N_TILE STREQUAL "1")
    target_compile_definitions(vllm_ascend_hif8_tiling PRIVATE
        HIF8_WIDE_N_TILE=1)
endif()
npu_op_package_add(${package_name}
    LIBRARY
        vllm_ascend_hif8_tiling
        vllm_ascend_hif8_opapi
        vllm_ascend_hif8_proto
)

# CANN defaults every custom ACLNN package to libcust_opapi.so.  Give this
# package a unique SONAME so it can coexist with vllm-ascend custom operators
# in the same process.
set_target_properties(${package_name}_ascendc_cust_opapi PROPERTIES
    OUTPUT_NAME vllm_ascend_hif8_opapi
)
