// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_opt.h -- sliver repair of the final labelled mesh (ported from the
// gpu_brain2mesh optimisation phase): 3-2 / 2-3 flips, interior edge collapse,
// sliver Steiner points, guarded interior smoothing; region labels, interfaces
// and trussnet's non-interior nodes are preserved.

#ifndef TRUSSNET_OPT_H
#define TRUSSNET_OPT_H

#include <cstddef>

#include "tn_particles.h"
#include "tn_tetra.h"

namespace tn {

struct OptParams {
    int max_rounds = 6;
    bool flip32 = true, flip23 = true, collapse = true, steiner = true, smooth = true;
    bool verbose = false;
};

struct OptStats {
    int rounds = 0, flips32 = 0, flips23 = 0, collapses = 0, steiner = 0, moves = 0;
    double ms = 0;
};

// Optimise `out` in place (tets may change, nodes may be removed / added); `nd`
// is remapped to the new node list. Returns the number of operations applied.
size_t optimize_mesh(TetOut& out, Nodes& nd, const OptParams& prm, OptStats& os);

}  // namespace tn

#endif  // TRUSSNET_OPT_H
