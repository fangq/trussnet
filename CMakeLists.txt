# SPDX-License-Identifier: GPL-3.0-or-later
# v2mesh -- GPU particle (truss) multi-label tetrahedral mesher
# Copyright (C) 2026 Qianqian Fang <q.fang at neu.edu>

cmake_minimum_required(VERSION 3.12)
project(v2mesh VERSION 0.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(V2M_USE_OPENCL "Build the OpenCL (GPU) path" ON)
option(V2M_USE_CDT "Vendor the Diazzi et al. exact Delaunay/CDT for the final tessellation" ON)
option(V2M_BUILD_TESTS "Register the command-line tests (tests/run_cli_tests.sh) with ctest" OFF)
option(V2M_STATIC_LINK "Static-link the v2mesh CLI's runtime libraries (libstdc++ / libgcc; all on MinGW)" OFF)
option(V2M_BUILD_PYTHON "Build the pybind11 module (pyv2mesh/v2mesh/_v2mesh*.so)" OFF)
option(V2M_BUILD_MATLAB_MEX "Build the MATLAB MEX (matlab/v2mesh_mex.mex*)" OFF)
option(V2M_BUILD_OCTAVE_MEX "Build the GNU Octave MEX (matlab/v2mesh_mex.mex, via mkoctfile)" OFF)

set(V2M_CORE_SOURCES
    src/v2m_log.cpp
    src/v2m_volume.cpp
    src/v2m_jmesh.cpp
    src/v2m_meshio.cpp
    src/v2m_modes.cpp
    src/v2m_surflabel.cpp
    src/v2m_manifold.cpp
    src/v2m_sdfshape.cpp
    src/v2m_remesh.cpp
    src/v2m_cdt.cpp
    src/v2m_shapes.cpp
    src/v2m_grid.cpp
    src/v2m_particles.cpp
    src/v2m_tetra.cpp
    src/v2m_opt.cpp
    src/v2m_pipeline.cpp
    src/v2m_tpm.cpp
    src/v2m_2d.cpp
    src/v2m_plc.cpp
    src/v2m_step.cpp
    src/v2m_csgsurf.cpp
    # volume I/O reused from siamize via gpu_brain2mesh (C++11)
    src/io/nifti_io.cpp
    src/io/jnifti_io.cpp
    src/io/orient.cpp
)

add_library(v2m_core STATIC ${V2M_CORE_SOURCES})
target_compile_definitions(v2m_core PUBLIC V2M_VERSION="${PROJECT_VERSION}")
set_target_properties(v2m_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(v2m_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mimamo)   # mmm_zlib.hpp + mmm_base64.hpp (zlibmt; src/io/zlibmt.h)
# zlibmt: worker threads, and the system libz opened at run time (dlopen)
find_package(Threads REQUIRED)
target_link_libraries(v2m_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

if(MSVC)
    target_compile_options(v2m_core PUBLIC /O2 /W3)
    target_compile_definitions(v2m_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    # -ffp-contract=off: no fused multiply-adds -- they break the error-free
    # transformations of the exact predicates (third_party/cdt numerics.h,
    # inlined into our sources too); clang contracts by default on arm64
    target_compile_options(v2m_core PUBLIC -ffp-contract=off $<$<CONFIG:Release>:-O3> $<$<CONFIG:Debug>:-g -Wall -Wextra>)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(v2m_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# OpenCL: only the headers are needed; the library (the ICD loader) is loaded at
# run time (src/opencl/v2m_cl_api.h), so a binary built with the GPU path runs on
# machines without one, on the CPU. No headers: a CPU-only build.
if(V2M_USE_OPENCL)
    find_path(V2M_OPENCL_INCLUDE_DIR NAMES CL/cl.h OpenCL/cl.h
              HINTS ENV OpenCL_ROOT ENV CUDA_PATH ENV AMDAPPSDKROOT ENV INTELOCLSDKROOT
              PATH_SUFFIXES include)
    if(NOT V2M_OPENCL_INCLUDE_DIR)
        message(WARNING "OpenCL headers (CL/cl.h) not found: building without the GPU path (CPU only). "
                        "Install them (e.g. opencl-headers / ocl-icd-opencl-dev) or set V2M_OPENCL_INCLUDE_DIR.")
        set(V2M_USE_OPENCL OFF)
    endif()
endif()
message(STATUS "OpenCL (GPU path): ${V2M_USE_OPENCL}")

if(V2M_USE_OPENCL)
    target_sources(v2m_core PRIVATE src/opencl/v2m_cl_host.cpp src/opencl/v2m_cl_loader.cpp src/v2m_gpu.cpp src/v2m_gdel.cpp)
    target_include_directories(v2m_core PUBLIC ${V2M_OPENCL_INCLUDE_DIR})
    # the kernels embedded in the binary (v2m_cl_sources.h)
    set(_cl_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl)
    set(_cl_files ${_cl_dir}/v2m_grid_body.cl ${_cl_dir}/v2m_sdf_body.cl ${_cl_dir}/v2m_seed_body.cl ${_cl_dir}/v2m_particle_body.cl
                  ${_cl_dir}/v2m_kernels.cl ${_cl_dir}/v2m_del_body.cl ${_cl_dir}/v2m_del_kernels.cl)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen/v2m_cl_sources.h
        COMMAND ${CMAKE_COMMAND} -DOUT=${CMAKE_CURRENT_BINARY_DIR}/gen/v2m_cl_sources.h -DSRC=${_cl_dir}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cl.cmake
        DEPENDS ${_cl_files} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_cl.cmake
        COMMENT "Embedding the OpenCL sources")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gen)
    add_custom_target(v2m_cl_embed DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen/v2m_cl_sources.h)
    add_dependencies(v2m_core v2m_cl_embed)
    target_include_directories(v2m_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
    target_include_directories(v2m_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/opencl)
    target_link_libraries(v2m_core PUBLIC ${CMAKE_DL_LIBS})   # dlopen (libdl before glibc 2.34)
    target_compile_definitions(v2m_core PUBLIC V2M_HAS_OPENCL)
endif()

if(V2M_USE_CDT)
    set(V2M_CDT_SOURCES third_party/cdt/src/delaunay.cpp third_party/cdt/src/PLC.cpp third_party/cdt/src/giftWrap.cpp)
    target_sources(v2m_core PRIVATE ${V2M_CDT_SOURCES})
    target_include_directories(v2m_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cdt/include
                                              ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cdt/src)
    target_compile_definitions(v2m_core PUBLIC V2M_HAS_CDT)
    if(NOT MSVC)
        set_source_files_properties(${V2M_CDT_SOURCES} PROPERTIES COMPILE_OPTIONS "-frounding-math")
    endif()
endif()

add_executable(v2mesh src/v2mesh.cpp)
target_link_libraries(v2mesh v2m_core)

if(V2M_STATIC_LINK)
    if(MSVC)
        set_property(TARGET v2mesh PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)
    elseif(MINGW)
        # everything static (libstdc++, libgcc, libgomp, libwinpthread); only the
        # core Windows DLLs stay dynamic (OpenCL.dll is loaded at run time, if present)
        target_link_options(v2mesh PRIVATE -static)
    elseif(NOT APPLE)
        # GNU/Linux: embed libstdc++ / libgcc (libgomp stays a system one; libOpenCL is
        # loaded at run time, if present)
        target_link_options(v2mesh PRIVATE -static-libstdc++ -static-libgcc)
    endif()
endif()

if(V2M_BUILD_TESTS)
    enable_testing()
    find_program(V2M_BASH NAMES bash)
    if(V2M_BASH)
        add_test(NAME cli COMMAND ${V2M_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cli_tests.sh
                 $<TARGET_FILE:v2mesh> ${CMAKE_CURRENT_BINARY_DIR}/cli_tests)
    else()
        message(WARNING "V2M_BUILD_TESTS: bash not found, the CLI tests are not registered")
    endif()
endif()

# ---- language bindings ------------------------------------------------------
# (ctest runs their unit tests: pyv2mesh/tests, matlab/tests/run_v2mesh_tests.m)
if(V2M_BUILD_PYTHON OR V2M_BUILD_MATLAB_MEX OR V2M_BUILD_OCTAVE_MEX)
    enable_testing()
endif()
# Python module via pybind11 (links the static, PIC v2m_core into the .so)
if(V2M_BUILD_PYTHON)
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
    pybind11_add_module(_v2mesh src/pyv2mesh.cpp)
    target_link_libraries(_v2mesh PRIVATE v2m_core)
    # into the package, so `import v2mesh` works in-tree (PYTHONPATH=pyv2mesh)
    set_target_properties(_v2mesh PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pyv2mesh/v2mesh)
    message(STATUS "Python module: ON -> pyv2mesh/v2mesh/_v2mesh*.so")
    add_test(NAME python_binding
             COMMAND ${Python_EXECUTABLE} -m unittest discover -s ${CMAKE_CURRENT_SOURCE_DIR}/pyv2mesh/tests
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# MATLAB MEX via FindMatlab (set Matlab_ROOT_DIR if MATLAB is not on the PATH)
if(V2M_BUILD_MATLAB_MEX)
    find_package(Matlab COMPONENTS MX_LIBRARY MAIN_PROGRAM)
    if(Matlab_FOUND)
        matlab_add_mex(NAME v2m_matlab_mex SRC src/v2m_mex.cpp OUTPUT_NAME v2mesh_mex)
        target_link_libraries(v2m_matlab_mex v2m_core)
        if(NOT MSVC)   # MATLAB ships an old libstdc++: embed the toolchain's
            target_link_options(v2m_matlab_mex PRIVATE -static-libstdc++ -static-libgcc)
        endif()
        set_target_properties(v2m_matlab_mex PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/matlab
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/matlab)
        message(STATUS "MATLAB MEX: ON -> matlab/v2mesh_mex.<mexext>")
        if(Matlab_MAIN_PROGRAM)
            add_test(NAME matlab_binding
                     COMMAND ${Matlab_MAIN_PROGRAM} -nodisplay -nosplash -batch
                             "addpath('${CMAKE_CURRENT_SOURCE_DIR}/matlab/tests'); exit(run_v2mesh_tests() > 0)")
        endif()
    else()
        message(WARNING "V2M_BUILD_MATLAB_MEX=ON but MATLAB was not found (set Matlab_ROOT_DIR); skipping.")
    endif()
endif()

# Octave MEX via mkoctfile, linking the static libv2m_core.a
if(V2M_BUILD_OCTAVE_MEX)
    find_program(MKOCTFILE NAMES mkoctfile)
    if(MKOCTFILE)
        set(_oct_out ${CMAKE_CURRENT_SOURCE_DIR}/matlab/v2mesh_mex.mex)
        set(_oct_inc "-I$<JOIN:$<TARGET_PROPERTY:v2m_core,INTERFACE_INCLUDE_DIRECTORIES>, -I>")
        set(_oct_def "-DMATLAB_MEX_FILE -DV2M_VERSION=\\\"${PROJECT_VERSION}\\\"")
        set(_oct_ldlibs "")
        if(V2M_USE_OPENCL)
            set(_oct_def "${_oct_def} -DV2M_HAS_OPENCL")
            if(CMAKE_DL_LIBS)
                set(_oct_ldlibs "${_oct_ldlibs} -l${CMAKE_DL_LIBS}")
            endif()
        endif()
        if(OpenMP_CXX_FOUND)
            set(_oct_ldlibs "${_oct_ldlibs} -fopenmp")
        endif()
        if(NOT WIN32)   # (zlibmt: threads, and dlopen of the system libz)
            set(_oct_ldlibs "${_oct_ldlibs} -lpthread")
            if(CMAKE_DL_LIBS)
                set(_oct_ldlibs "${_oct_ldlibs} -l${CMAKE_DL_LIBS}")
            endif()
        endif()
        # libstdc++ stays DYNAMIC here (Octave has one loaded already)
        add_custom_command(
            OUTPUT ${_oct_out}
            COMMAND ${CMAKE_COMMAND} -E env
                "CXXFLAGS=-std=c++11 -O2 -fPIC -ffp-contract=off ${_oct_def} ${_oct_inc}"
                "LDFLAGS=$<TARGET_FILE:v2m_core> ${_oct_ldlibs}"
                ${MKOCTFILE} --mex -o ${_oct_out} ${CMAKE_CURRENT_SOURCE_DIR}/src/v2m_mex.cpp
            DEPENDS v2m_core ${CMAKE_CURRENT_SOURCE_DIR}/src/v2m_mex.cpp
            COMMENT "Building Octave MEX -> matlab/v2mesh_mex.mex"
            VERBATIM)
        add_custom_target(v2m_octave_mex ALL DEPENDS ${_oct_out})
        message(STATUS "Octave MEX: ON -> matlab/v2mesh_mex.mex (${MKOCTFILE})")
        find_program(OCTAVE_EXE NAMES octave-cli octave)
        if(OCTAVE_EXE)
            # --norc, no DISPLAY: a user's ~/.octaverc that sets graphics defaults
            # connects to the X display and blocks if that server does not answer
            add_test(NAME octave_binding
                     COMMAND ${OCTAVE_EXE} --no-gui --norc --quiet --eval
                             "addpath('${CMAKE_CURRENT_SOURCE_DIR}/matlab/tests'); exit(run_v2mesh_tests() > 0)")
            set_tests_properties(octave_binding PROPERTIES ENVIRONMENT "DISPLAY=")
        endif()
    else()
        message(WARNING "V2M_BUILD_OCTAVE_MEX=ON but mkoctfile not found; skipping.")
    endif()
endif()
