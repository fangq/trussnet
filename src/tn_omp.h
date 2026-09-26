// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (adapted from gpu_brain2mesh, same author, GPL-3.0-or-later)
//
// tn_omp.h -- scoped OpenMP thread cap for the mesher's parallel stages.
//
// By default all logical threads, up to 64: the CPU stages keep scaling on
// many-core machines (Colin27 on a 64-core / 128-thread Threadripper: 22.1 s at 8
// threads, 15.5 s at 32, 15.0 s at 64, but 17.3 s at 128 -- the memory-bound
// loops lose on the second hyperthread of every core). On a busy machine, where
// another process occupies some cores, all-core loops wait for the threads that
// share them (once measured 2.2x slower); TN_OMP_MAX_THREADS=N sets the cap
// (min(#procs, N); 0 = none). The cap holds for the scope of the guard and is
// restored afterwards, leaving a host application's OpenMP setting (e.g. MATLAB)
// intact. OMP_NUM_THREADS is always respected.

#ifndef TRUSSNET_TN_OMP_H
#define TRUSSNET_TN_OMP_H

#include <cstdlib>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace tn {

// Dynamic loops are written schedule(monotonic: dynamic, N): GCC >= 9 compiles a
// plain schedule(dynamic) to GOMP_loop_nonmonotonic_*, which the Intel OpenMP
// runtime MATLAB loads (libiomp5, R2019b) does not provide -- inside MATLAB the MEX
// then ran iomp's parallel regions with libgomp's loop scheduler and crashed.
// (libgomp has no work stealing for nonmonotonic loops: no speed difference.)
struct OmpThreadCap {
#ifdef _OPENMP
    int saved = -1;

    OmpThreadCap() {
        if (std::getenv("OMP_NUM_THREADS")) {
            return;    // the user chose
        }

        const char* e = std::getenv("TN_OMP_MAX_THREADS");
        const int cap = e ? std::atoi(e) : 64;   // 0: no cap (the OpenMP default)

        if (cap <= 0) {
            return;
        }

        const int np = omp_get_num_procs();
        const int want = np < cap ? np : cap;
        saved = omp_get_max_threads();

        if (want < saved) {
            omp_set_num_threads(want);
        } else {
            saved = -1;
        }
    }
    ~OmpThreadCap() {
        if (saved > 0) {
            omp_set_num_threads(saved);
        }
    }
#endif
};

}  // namespace tn

#endif  // TRUSSNET_TN_OMP_H
