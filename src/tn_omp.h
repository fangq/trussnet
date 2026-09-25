// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (adapted from gpu_brain2mesh, same author, GPL-3.0-or-later)
//
// tn_omp.h -- scoped OpenMP thread cap for the mesher's parallel stages.
//
// The refiner / optimiser / complex-remesh loops are memory-bound: 8 threads are
// as fast as 16 on an idle 16-core machine, and when another process occupies
// even one core, using ALL cores makes every parallel loop wait for the thread
// that shares it (measured 2.2x slower). So, unless the user sets
// OMP_NUM_THREADS (always respected), the thread count is capped at
// min(#procs, TN_OMP_MAX_THREADS or 8) for the scope of the guard and restored
// afterwards, leaving a host application's OpenMP setting (e.g. MATLAB) intact.

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
        int cap = e ? std::atoi(e) : 8;

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
