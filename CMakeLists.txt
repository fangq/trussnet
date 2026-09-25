# SPDX-License-Identifier: GPL-3.0-or-later
# trussnet -- GPU particle (truss) multi-label tetrahedral mesher
# Copyright (C) 2026 Qianqian Fang <q.fang at neu.edu>

cmake_minimum_required(VERSION 3.12)
project(trussnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(TN_USE_OPENCL "Build the OpenCL (GPU) path" ON)
option(TN_USE_CDT "Vendor the Diazzi et al. exact Delaunay/CDT for the final tessellation" ON)
option(TN_BUILD_TESTS "Build the unit tests" OFF)

set(TN_CORE_SOURCES
    src/tn_log.cpp
    src/tn_volume.cpp
    src/tn_jmesh.cpp
    src/tn_shapes.cpp
    src/tn_grid.cpp
    src/tn_particles.cpp
    src/tn_tetra.cpp
    src/tn_opt.cpp
    # volume I/O reused from siamize via gpu_brain2mesh (C++11)
    src/io/nifti_io.cpp
    src/io/jnifti_io.cpp
    src/io/orient.cpp
)

add_library(tn_core STATIC ${TN_CORE_SOURCES})
set_target_properties(tn_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/zmat)

if(MSVC)
    target_compile_options(tn_core PUBLIC /O2 /W3)
    target_compile_definitions(tn_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(tn_core PUBLIC $<$<CONFIG:Release>:-O3> $<$<CONFIG:Debug>:-g -Wall -Wextra>)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tn_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(TN_USE_OPENCL)
    find_package(OpenCL REQUIRED)
    target_sources(tn_core PRIVATE src/opencl/tn_cl_host.cpp src/tn_gpu.cpp src/tn_gdel.cpp)
    target_include_directories(tn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl)
    target_link_libraries(tn_core PUBLIC OpenCL::OpenCL)
    target_compile_definitions(tn_core PUBLIC TN_HAS_OPENCL TN_SRC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

if(TN_USE_CDT)
    set(TN_CDT_SOURCES third_party/cdt/src/delaunay.cpp third_party/cdt/src/PLC.cpp third_party/cdt/src/giftWrap.cpp)
    target_sources(tn_core PRIVATE ${TN_CDT_SOURCES})
    target_include_directories(tn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cdt/include
                                              ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cdt/src)
    target_compile_definitions(tn_core PUBLIC TN_HAS_CDT)
    if(NOT MSVC)
        set_source_files_properties(${TN_CDT_SOURCES} PROPERTIES COMPILE_OPTIONS "-frounding-math")
    endif()
endif()

add_executable(trussnet src/trussnet.cpp)
target_link_libraries(trussnet tn_core)
