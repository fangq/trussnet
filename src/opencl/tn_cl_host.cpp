// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (adapted from gpu_brain2mesh, same author, GPL-3.0-or-later)
//
// clctx.cpp -- see clctx.h. Original code (OpenCL C API).

#include "tn_cl_host.h"
#include "tn_log.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>

namespace tn {

void cl_check(cl_int err, const char* where) {
    if (err == CL_SUCCESS) {
        return;
    }

    throw std::runtime_error(std::string(where) + ": OpenCL error " + std::to_string(err));
}

std::vector<ClDeviceInfo> cl_list_devices() {
    std::vector<ClDeviceInfo> out;
    cl_uint nplat = 0;

    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) {
        return out;
    }

    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    int flat = 0;

    for (cl_uint p = 0; p < nplat; ++p) {
        char pname[256] = { 0 };
        clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(pname) - 1, pname, nullptr);

        cl_uint ndev = 0;

        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev) != CL_SUCCESS || ndev == 0) {
            continue;
        }

        std::vector<cl_device_id> devs(ndev);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr);

        for (cl_uint d = 0; d < ndev; ++d) {
            ClDeviceInfo info;
            info.index = flat++;
            info.platform = pname;

            char dname[256] = { 0 };
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(dname) - 1, dname, nullptr);
            info.name = dname;

            cl_device_type dtype = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
            info.is_gpu = (dtype & CL_DEVICE_TYPE_GPU) != 0;

            info.global_mem = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info.global_mem),
                            &info.global_mem, nullptr);

            out.push_back(info);
        }
    }

    return out;
}

void ClCtx::init(int device_index) {
    cl_uint nplat = 0;
    cl_check(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs");

    if (nplat == 0) {
        throw std::runtime_error("no OpenCL platforms found");
    }

    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    // Flatten platforms x devices. An explicit flat index is honoured as-is; for -1
    // (auto) prefer the first fp64-capable GPU -- the refiner's kernels use double, so a
    // GPU that only advertises fp64 via CL_DEVICE_DOUBLE_FP_CONFIG but NOT the cl_khr_fp64
    // extension (e.g. some Intel iGPUs) would make clBuildProgram fail. Check the extension
    // string directly. Fall back to first GPU, then first device of any kind.
    auto has_fp64 = [](cl_device_id d) -> bool {
        size_t n = 0;
        clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, 0, nullptr, &n);
        std::string ext(n, '\0');

        if (n) {
            clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, n, &ext[0], nullptr);
        }

        return ext.find("cl_khr_fp64") != std::string::npos;
    };

    int flat = 0;
    cl_platform_id chosenPlat = nullptr, firstPlat = nullptr, gpuPlat = nullptr, fp64Plat = nullptr;
    cl_device_id   chosenDev = nullptr, firstDev = nullptr, gpuDev = nullptr, fp64Dev = nullptr;

    for (cl_uint p = 0; p < nplat; ++p) {
        cl_uint ndev = 0;

        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev) != CL_SUCCESS || ndev == 0) {
            continue;
        }

        std::vector<cl_device_id> devs(ndev);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, ndev, devs.data(), nullptr);

        for (cl_uint d = 0; d < ndev; ++d, ++flat) {
            if (firstDev == nullptr) {
                firstDev = devs[d];
                firstPlat = plats[p];
            }

            cl_device_type dtype = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
            bool gpu = (dtype & CL_DEVICE_TYPE_GPU) != 0;

            if (gpu && gpuDev == nullptr) {
                gpuDev = devs[d];
                gpuPlat = plats[p];
            }

            if (gpu && fp64Dev == nullptr && has_fp64(devs[d])) {
                fp64Dev = devs[d];
                fp64Plat = plats[p];
            }

            if (device_index >= 0 && flat == device_index && chosenDev == nullptr) {
                chosenPlat = plats[p];
                chosenDev = devs[d];
            }
        }
    }

    if (device_index < 0) {                     // auto: prefer fp64 GPU, then any GPU
        if (fp64Dev) {
            chosenDev = fp64Dev;
            chosenPlat = fp64Plat;
        } else if (gpuDev) {
            chosenDev = gpuDev;
            chosenPlat = gpuPlat;
        }
    }

    if (chosenDev == nullptr) {                 // fallback: first device of any kind
        chosenDev = firstDev;
        chosenPlat = firstPlat;
    }

    if (chosenDev == nullptr) {
        throw std::runtime_error("no OpenCL device found");
    }

    m_platform = chosenPlat;
    m_device = chosenDev;

    char dname[256] = { 0 };
    clGetDeviceInfo(m_device, CL_DEVICE_NAME, sizeof(dname) - 1, dname, nullptr);
    m_deviceName = dname;

    cl_int err = CL_SUCCESS;
    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &err);
    cl_check(err, "clCreateContext");

    m_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
    cl_check(err, "clCreateCommandQueue");
}

cl_program ClCtx::build(const std::string& source, const std::string& options) {
    const char* src = source.c_str();
    size_t len = source.size();
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(m_context, 1, &src, &len, &err);
    cl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(prog, 1, &m_device, options.c_str(), nullptr, nullptr);

    if (err != CL_SUCCESS) {
        size_t logn = 0;
        clGetProgramBuildInfo(prog, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logn);
        std::string log(logn, '\0');
        clGetProgramBuildInfo(prog, m_device, CL_PROGRAM_BUILD_LOG, logn, &log[0], nullptr);
        clReleaseProgram(prog);
        throw std::runtime_error("OpenCL build failed:\n" + log);
    }

    if (std::getenv("TN_CL_BUILDLOG")) {   // e.g. with -cl-nv-verbose: registers per kernel
        size_t logn = 0;
        clGetProgramBuildInfo(prog, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logn);
        std::string log(logn, '\0');
        clGetProgramBuildInfo(prog, m_device, CL_PROGRAM_BUILD_LOG, logn, &log[0], nullptr);
        std::fprintf(stderr, "%s\n", log.c_str());
    }

    return prog;
}

cl_mem ClCtx::alloc(std::size_t bytes, cl_mem_flags flags) {
    cl_int err = CL_SUCCESS;
    cl_mem buf = clCreateBuffer(m_context, flags, bytes ? bytes : 1, nullptr, &err);
    cl_check(err, "clCreateBuffer");
    return buf;
}

void ClCtx::write(cl_mem buf, const void* host, std::size_t bytes) {
    cl_check(clEnqueueWriteBuffer(m_queue, buf, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr),
             "clEnqueueWriteBuffer");
}

void ClCtx::read(cl_mem buf, void* host, std::size_t bytes) {
    cl_check(clEnqueueReadBuffer(m_queue, buf, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr),
             "clEnqueueReadBuffer");
}

void ClCtx::finish() {
    cl_check(clFinish(m_queue), "clFinish");
}

ClCtx& cl_shared_ctx(int device_index) {
    static std::unique_ptr<ClCtx> c;

    if (!c) {
        std::unique_ptr<ClCtx> n(new ClCtx());
        n->init(device_index);   // throws: c stays empty, the next call retries
        c = std::move(n);
    }

    return *c;
}

ClCtx::~ClCtx() {
    if (m_queue) {
        clReleaseCommandQueue(m_queue);
    }

    if (m_context) {
        clReleaseContext(m_context);
    }
}

void cl_print_devices() {
    std::vector<ClDeviceInfo> devs = cl_list_devices();

    if (devs.empty()) {
        TN_FPRINTF(stderr, "No OpenCL devices found.\n");
        return;
    }

    TN_FPRINTF(stderr, "OpenCL devices (flat index for -G):\n");

    for (const ClDeviceInfo& d : devs) {
        TN_FPRINTF(stderr, "  -G %d  [%s]  %-32s  %s  %llu GB\n",
                    d.index, d.platform.c_str(), d.name.c_str(),
                    d.is_gpu ? "GPU" : "CPU/other",
                    static_cast<unsigned long long>((d.global_mem + (1ULL << 29)) >> 30));
    }
}

int cl_selftest(int device_index) {
    const char* kSrc =
        "__kernel void tn_sq(__global const float* a, __global float* b) {\n"
        "    size_t i = get_global_id(0);\n"
        "    b[i] = a[i] * a[i];\n"
        "}\n";

    const int N = 4096;

    try {
        ClCtx ctx;
        ctx.init(device_index);
        TN_FPRINTF(stderr, "[cl-selftest] device: %s\n", ctx.deviceName().c_str());

        cl_program prog = ctx.build(kSrc);
        cl_int err = CL_SUCCESS;
        cl_kernel k = clCreateKernel(prog, "tn_sq", &err);
        cl_check(err, "clCreateKernel");

        std::vector<float> a(N), b(N, -1.0f);

        for (int i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i);
        }

        cl_mem da = ctx.alloc(N * sizeof(float), CL_MEM_READ_ONLY);
        cl_mem db = ctx.alloc(N * sizeof(float), CL_MEM_WRITE_ONLY);
        ctx.write(da, a.data(), N * sizeof(float));

        cl_check(clSetKernelArg(k, 0, sizeof(cl_mem), &da), "setarg0");
        cl_check(clSetKernelArg(k, 1, sizeof(cl_mem), &db), "setarg1");

        size_t gws = N;
        cl_check(clEnqueueNDRangeKernel(ctx.queue(), k, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel");
        ctx.finish();
        ctx.read(db, b.data(), N * sizeof(float));

        clReleaseMemObject(da);
        clReleaseMemObject(db);
        clReleaseKernel(k);
        clReleaseProgram(prog);

        for (int i = 0; i < N; ++i) {
            float expect = static_cast<float>(i) * static_cast<float>(i);

            if (b[i] != expect) {
                TN_FPRINTF(stderr, "[cl-selftest] FAIL at i=%d: got %g expected %g\n", i, b[i], expect);
                return 1;
            }
        }

        TN_FPRINTF(stderr, "[cl-selftest] PASS (%d elements squared on device)\n", N);
        return 0;
    } catch (const std::exception& e) {
        TN_FPRINTF(stderr, "[cl-selftest] error: %s\n", e.what());
        return 2;
    }
}

}  // namespace tn
