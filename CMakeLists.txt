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
option(TN_BUILD_PYTHON "Build the pybind11 module (pytrussnet/trussnet/_trussnet*.so)" OFF)
option(TN_BUILD_MATLAB_MEX "Build the MATLAB MEX (matlab/trussnet_mex.mex*)" OFF)
option(TN_BUILD_OCTAVE_MEX "Build the GNU Octave MEX (matlab/trussnet_mex.mex, via mkoctfile)" OFF)

set(TN_CORE_SOURCES
    src/tn_log.cpp
    src/tn_volume.cpp
    src/tn_jmesh.cpp
    src/tn_shapes.cpp
    src/tn_grid.cpp
    src/tn_particles.cpp
    src/tn_tetra.cpp
    src/tn_opt.cpp
    src/tn_pipeline.cpp
    src/tn_tpm.cpp
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
    # the kernels embedded in the binary (tn_cl_sources.h)
    set(_cl_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl)
    set(_cl_files ${_cl_dir}/tn_grid_body.cl ${_cl_dir}/tn_seed_body.cl ${_cl_dir}/tn_particle_body.cl
                  ${_cl_dir}/tn_kernels.cl ${_cl_dir}/tn_del_body.cl ${_cl_dir}/tn_del_kernels.cl)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen/tn_cl_sources.h
        COMMAND ${CMAKE_COMMAND} -DOUT=${CMAKE_CURRENT_BINARY_DIR}/gen/tn_cl_sources.h -DSRC=${_cl_dir}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cl.cmake
        DEPENDS ${_cl_files} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cl.cmake
        COMMENT "Embedding the OpenCL sources")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gen)
    add_custom_target(tn_cl_embed DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen/tn_cl_sources.h)
    add_dependencies(tn_core tn_cl_embed)
    target_include_directories(tn_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
    target_include_directories(tn_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl)
    target_link_libraries(tn_core PUBLIC OpenCL::OpenCL)
    target_compile_definitions(tn_core PUBLIC TN_HAS_OPENCL)
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

# ---- language bindings ------------------------------------------------------
# (ctest runs their unit tests: pytrussnet/tests, matlab/tests/run_trussnet_tests.m)
if(TN_BUILD_PYTHON OR TN_BUILD_MATLAB_MEX OR TN_BUILD_OCTAVE_MEX)
    enable_testing()
endif()
# Python module via pybind11 (links the static, PIC tn_core into the .so)
if(TN_BUILD_PYTHON)
    if(NOT pybind11_DIR)   # pybind11 is usually pip-installed: ask it for its CMake dir
        find_program(_TN_PYEXE NAMES python3 python)
        if(_TN_PYEXE)
            execute_process(COMMAND ${_TN_PYEXE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE _tn_pb_dir OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _tn_pb_rc)
            if(_tn_pb_rc EQUAL 0)
                set(pybind11_DIR "${_tn_pb_dir}")
            endif()
        endif()
    endif()
    # the interpreter pybind11 came with (the modern FindPython, not the legacy
    # FindPythonInterp, which may pick another installed version)
    if(NOT DEFINED PYBIND11_FINDPYTHON)
        set(PYBIND11_FINDPYTHON ON)
    endif()
    if(_TN_PYEXE AND NOT Python_EXECUTABLE)
        set(Python_EXECUTABLE ${_TN_PYEXE})
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_trussnet src/pytrussnet.cpp)
    target_link_libraries(_trussnet PRIVATE tn_core)
    # into the package, so `import trussnet` works in-tree (PYTHONPATH=pytrussnet)
    set_target_properties(_trussnet PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pytrussnet/trussnet)
    message(STATUS "Python module: ON -> pytrussnet/trussnet/_trussnet*.so")
    add_test(NAME python_binding
             COMMAND ${Python_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/pytrussnet/tests
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# MATLAB MEX via FindMatlab (set Matlab_ROOT_DIR if MATLAB is not on the PATH)
if(TN_BUILD_MATLAB_MEX)
    find_package(Matlab COMPONENTS MX_LIBRARY MAIN_PROGRAM)
    if(Matlab_FOUND)
        matlab_add_mex(NAME tn_matlab_mex SRC src/tn_mex.cpp OUTPUT_NAME trussnet_mex)
        target_link_libraries(tn_matlab_mex tn_core)
        if(NOT MSVC)   # MATLAB ships an old libstdc++: embed the toolchain's
            target_link_options(tn_matlab_mex PRIVATE -static-libstdc++ -static-libgcc)
        endif()
        set_target_properties(tn_matlab_mex PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/matlab
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/matlab)
        message(STATUS "MATLAB MEX: ON -> matlab/trussnet_mex.<mexext>")
        if(Matlab_MAIN_PROGRAM)
            add_test(NAME matlab_binding
                     COMMAND ${Matlab_MAIN_PROGRAM} -nodisplay -nosplash -batch
                             "addpath('${CMAKE_CURRENT_SOURCE_DIR}/matlab/tests'); exit(run_trussnet_tests() > 0)")
        endif()
    else()
        message(WARNING "TN_BUILD_MATLAB_MEX=ON but MATLAB was not found (set Matlab_ROOT_DIR); skipping.")
    endif()
endif()

# Octave MEX via mkoctfile, linking the static libtn_core.a
if(TN_BUILD_OCTAVE_MEX)
    find_program(MKOCTFILE NAMES mkoctfile)
    if(MKOCTFILE)
        set(_oct_out ${CMAKE_CURRENT_SOURCE_DIR}/matlab/trussnet_mex.mex)
        set(_oct_inc "-I$<JOIN:$<TARGET_PROPERTY:tn_core,INTERFACE_INCLUDE_DIRECTORIES>, -I>")
        set(_oct_def "-DMATLAB_MEX_FILE")
        set(_oct_ldlibs "")
        if(TN_USE_OPENCL)
            set(_oct_def "${_oct_def} -DTN_HAS_OPENCL")
            set(_oct_ldlibs "${_oct_ldlibs} -lOpenCL")
        endif()
        if(OpenMP_CXX_FOUND)
            set(_oct_ldlibs "${_oct_ldlibs} -fopenmp")
        endif()
        # libstdc++ stays DYNAMIC here (Octave has one loaded already)
        add_custom_command(
            OUTPUT ${_oct_out}
            COMMAND ${CMAKE_COMMAND} -E env
                "CXXFLAGS=-std=c++17 -O2 -fPIC ${_oct_def} ${_oct_inc}"
                "LDFLAGS=$<TARGET_FILE:tn_core> ${_oct_ldlibs}"
                ${MKOCTFILE} --mex -o ${_oct_out} ${CMAKE_CURRENT_SOURCE_DIR}/src/tn_mex.cpp
            DEPENDS tn_core ${CMAKE_CURRENT_SOURCE_DIR}/src/tn_mex.cpp
            COMMENT "Building Octave MEX -> matlab/trussnet_mex.mex"
            VERBATIM)
        add_custom_target(tn_octave_mex ALL DEPENDS ${_oct_out})
        message(STATUS "Octave MEX: ON -> matlab/trussnet_mex.mex (${MKOCTFILE})")
        find_program(OCTAVE_EXE NAMES octave-cli octave)
        if(OCTAVE_EXE)
            add_test(NAME octave_binding
                     COMMAND ${OCTAVE_EXE} --no-gui --quiet --eval
                             "addpath('${CMAKE_CURRENT_SOURCE_DIR}/matlab/tests'); exit(run_trussnet_tests() > 0)")
        endif()
    else()
        message(WARNING "TN_BUILD_OCTAVE_MEX=ON but mkoctfile not found; skipping.")
    endif()
endif()
