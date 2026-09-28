// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_cl_api.h -- the OpenCL library loaded at run time (dlopen / LoadLibrary),
// not linked: a machine without an OpenCL loader (libOpenCL.so.1, OpenCL.dll)
// still runs trussnet, on the CPU. Only the headers are needed to build.
//
// Included by tn_cl_host.h after CL/cl.h: each clXxx(...) the GPU code calls
// becomes a call through the loaded entry point. Before the library is found,
// or without one, clGetPlatformIDs (the first call of every GPU path) returns
// CL_PLATFORM_NOT_FOUND_KHR (-1001), as an ICD loader with no driver does; any
// other entry point missing throws, which the GPU paths catch (CPU fallback).
//
// The library: $TN_OPENCL_LIB if set, else libOpenCL.so.1 / libOpenCL.so
// (Linux), the OpenCL framework (macOS), OpenCL.dll (Windows).

#ifndef TRUSSNET_CL_API_H
#define TRUSSNET_CL_API_H

#include <stdexcept>
#include <string>
#include <type_traits>

#ifndef CL_TARGET_OPENCL_VERSION
    #define CL_TARGET_OPENCL_VERSION 120
#endif
#if defined(__APPLE__)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

// the entry points trussnet uses
#define TN_CL_FUNCS(X)                                                                                                  \
    X(clGetPlatformIDs) X(clGetPlatformInfo) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext)                  \
    X(clCreateCommandQueue) X(clCreateProgramWithSource) X(clBuildProgram) X(clGetProgramBuildInfo) X(clGetProgramInfo)\
    X(clCreateKernel) X(clSetKernelArg) X(clCreateBuffer) X(clEnqueueWriteBuffer) X(clEnqueueReadBuffer)              \
    X(clEnqueueCopyBuffer) X(clEnqueueFillBuffer) X(clEnqueueNDRangeKernel) X(clFinish) X(clReleaseMemObject)         \
    X(clReleaseKernel) X(clReleaseProgram) X(clReleaseCommandQueue) X(clReleaseContext)

namespace tn {
namespace cl_dyn {

struct Api {
#define TN_CL_MEMBER(n) decltype(&::n) n = nullptr;
    TN_CL_FUNCS(TN_CL_MEMBER)
#undef TN_CL_MEMBER
    bool loaded = false;
    std::string library;   // the file loaded, or why none was
};

// the library, looked for on the first call (thread-safe)
const Api& api();

template <typename R>
inline R missing(const char*, std::true_type) {
    return -1001;   // CL_PLATFORM_NOT_FOUND_KHR
}

template <typename R>
inline R missing(const char* name, std::false_type) {
    throw std::runtime_error(std::string(name) + ": no OpenCL library (" + api().library + ")");
}

template <typename F, typename... A>
inline auto call(F f, const char* name, A... a) -> decltype(f(a...)) {
    typedef decltype(f(a...)) R;

    if (f) {
        return f(a...);
    }

    return missing<R>(name, typename std::is_same<R, cl_int>::type());
}

}  // namespace cl_dyn
}  // namespace tn

#define TN_CL_CALL(n, ...) ::tn::cl_dyn::call(::tn::cl_dyn::api().n, #n, __VA_ARGS__)
#define clGetPlatformIDs(...) TN_CL_CALL(clGetPlatformIDs, __VA_ARGS__)
#define clGetPlatformInfo(...) TN_CL_CALL(clGetPlatformInfo, __VA_ARGS__)
#define clGetDeviceIDs(...) TN_CL_CALL(clGetDeviceIDs, __VA_ARGS__)
#define clGetDeviceInfo(...) TN_CL_CALL(clGetDeviceInfo, __VA_ARGS__)
#define clCreateContext(...) TN_CL_CALL(clCreateContext, __VA_ARGS__)
#define clCreateCommandQueue(...) TN_CL_CALL(clCreateCommandQueue, __VA_ARGS__)
#define clCreateProgramWithSource(...) TN_CL_CALL(clCreateProgramWithSource, __VA_ARGS__)
#define clBuildProgram(...) TN_CL_CALL(clBuildProgram, __VA_ARGS__)
#define clGetProgramBuildInfo(...) TN_CL_CALL(clGetProgramBuildInfo, __VA_ARGS__)
#define clGetProgramInfo(...) TN_CL_CALL(clGetProgramInfo, __VA_ARGS__)
#define clCreateKernel(...) TN_CL_CALL(clCreateKernel, __VA_ARGS__)
#define clSetKernelArg(...) TN_CL_CALL(clSetKernelArg, __VA_ARGS__)
#define clCreateBuffer(...) TN_CL_CALL(clCreateBuffer, __VA_ARGS__)
#define clEnqueueWriteBuffer(...) TN_CL_CALL(clEnqueueWriteBuffer, __VA_ARGS__)
#define clEnqueueReadBuffer(...) TN_CL_CALL(clEnqueueReadBuffer, __VA_ARGS__)
#define clEnqueueCopyBuffer(...) TN_CL_CALL(clEnqueueCopyBuffer, __VA_ARGS__)
#define clEnqueueFillBuffer(...) TN_CL_CALL(clEnqueueFillBuffer, __VA_ARGS__)
#define clEnqueueNDRangeKernel(...) TN_CL_CALL(clEnqueueNDRangeKernel, __VA_ARGS__)
#define clFinish(...) TN_CL_CALL(clFinish, __VA_ARGS__)
#define clReleaseMemObject(...) TN_CL_CALL(clReleaseMemObject, __VA_ARGS__)
#define clReleaseKernel(...) TN_CL_CALL(clReleaseKernel, __VA_ARGS__)
#define clReleaseProgram(...) TN_CL_CALL(clReleaseProgram, __VA_ARGS__)
#define clReleaseCommandQueue(...) TN_CL_CALL(clReleaseCommandQueue, __VA_ARGS__)
#define clReleaseContext(...) TN_CL_CALL(clReleaseContext, __VA_ARGS__)

#endif  // TRUSSNET_CL_API_H
