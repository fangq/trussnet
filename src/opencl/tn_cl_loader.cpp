// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_cl_loader.cpp -- the OpenCL library, found and loaded on first use (see
// tn_cl_api.h).

#include "tn_cl_host.h"

#include <cstdlib>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace tn {
namespace cl_dyn {

namespace {

void* open_lib(const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

Api load() {
    Api a;
    const char* env = std::getenv("TN_OPENCL_LIB");
    static const char* const names[] = {
#if defined(_WIN32)
        "OpenCL.dll",
#elif defined(__APPLE__)
        "/System/Library/Frameworks/OpenCL.framework/OpenCL", "libOpenCL.dylib",
#else
        "libOpenCL.so.1", "libOpenCL.so",
#endif
    };
    void* lib = nullptr;

    if (env && *env) {
        lib = open_lib(env);
        a.library = env;
    } else {
        for (const char* n : names) {
            if ((lib = open_lib(n)) != nullptr) {
                a.library = n;
                break;
            }
        }
    }

    if (!lib) {
        a.library = env && *env ? std::string(env) + " not found" : "no OpenCL loader found";
        return a;   // (clGetPlatformIDs: no platform)
    }

#define TN_CL_LOAD(n) a.n = reinterpret_cast<decltype(a.n)>(symbol(lib, #n));
    TN_CL_FUNCS(TN_CL_LOAD)
#undef TN_CL_LOAD
    a.loaded = true;   // (never unloaded: the process keeps it)
    return a;
}

}  // namespace

const Api& api() {
    static const Api a = load();
    return a;
}

}  // namespace cl_dyn
}  // namespace tn
