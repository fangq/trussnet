// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (adapted from gpu_brain2mesh, same author, GPL-3.0-or-later)
//
// clctx.h -- minimal OpenCL host layer (context, queue, program build, buffer
// helpers) for the GPU SurfaceNets backend. Original code (OpenCL C API); the
// flat platform x device enumeration follows the same scheme as siamize's
// --list-gpu / mcxcl's mcx_list_gpu. Links libOpenCL via find_package(OpenCL);
// a runtime ICD dlopen loader can replace this later for a link-free binary.

#ifndef TRUSSNET_CLCTX_H
#define TRUSSNET_CLCTX_H

#include <cstddef>
#include <string>
#include <vector>

#ifndef CL_TARGET_OPENCL_VERSION
    #define CL_TARGET_OPENCL_VERSION 120
#endif
#if defined(__APPLE__)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

namespace tn {

struct ClDeviceInfo {
    int          index;        // flat index across all platforms (0-based)
    std::string  platform;
    std::string  name;
    bool         is_gpu;
    cl_ulong     global_mem;   // bytes
};

// Enumerate all OpenCL devices on all platforms (clinfo order).
std::vector<ClDeviceInfo> cl_list_devices();

// Print the device list to stderr (for --cl-info).
void cl_print_devices();

// Build + run a trivial kernel on `device_index` and verify the result.
// Returns 0 on PASS, nonzero on failure. Used by --cl-selftest to confirm the
// OpenCL stack (build, launch, readback) works before the SurfaceNets kernels.
int cl_selftest(int device_index);

// A built OpenCL context bound to one device, with a command queue.
class ClCtx {
  public:
    ClCtx() = default;
    ~ClCtx();
    ClCtx(const ClCtx&) = delete;
    ClCtx& operator=(const ClCtx&) = delete;

    // Bind to the flat device index (see cl_list_devices); -1 = first GPU,
    // else first device. Throws std::runtime_error on failure.
    void init(int device_index);

    cl_program build(const std::string& source, const std::string& options = "");
    cl_mem     alloc(std::size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE);
    void       write(cl_mem buf, const void* host, std::size_t bytes);
    void       read(cl_mem buf, void* host, std::size_t bytes);
    void       finish();

    cl_context        context() const {
        return m_context;
    }
    cl_command_queue  queue()   const {
        return m_queue;
    }
    cl_device_id      device()  const {
        return m_device;
    }
    const std::string& deviceName() const {
        return m_deviceName;
    }

  private:
    cl_platform_id   m_platform = nullptr;
    cl_device_id     m_device   = nullptr;
    cl_context       m_context  = nullptr;
    cl_command_queue m_queue    = nullptr;
    std::string      m_deviceName;
};

// Throw std::runtime_error("<where>: <clErrorName>") if err != CL_SUCCESS.
void cl_check(cl_int err, const char* where);

}  // namespace tn

#endif  // TRUSSNET_CLCTX_H
