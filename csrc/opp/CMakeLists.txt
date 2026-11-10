cmake_minimum_required(VERSION 3.16.0)
project(vllm_ascend_hif8_opp)

set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "Ascend 910B build target")
if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    message(FATAL_ERROR "This branch only supports ASCEND_COMPUTE_UNIT=ascend910b")
endif()

find_package(ASC REQUIRED)

set(HIF8_ONLINE_BF16_ARITH "0" CACHE STRING
    "Reserved C220 BF16 arithmetic unpack experiment (must remain disabled)")
if(NOT HIF8_ONLINE_BF16_ARITH STREQUAL "0")
    message(FATAL_ERROR
        "HIF8_ONLINE_BF16_ARITH is not implemented; use the BF16 LUT/Gather path")
endif()

set(package_name ${vendor_name})
npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}/package
)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
