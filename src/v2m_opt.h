// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_opt.h -- sliver repair of the final labelled mesh (ported from the
// gpu_brain2mesh optimisation phase): 3-2 / 2-3 flips, interior edge collapse,
// sliver Steiner points, guarded interior smoothing; region labels, interfaces
// and v2mesh's non-interior nodes are preserved.

#ifndef V2MESH_OPT_H
#define V2MESH_OPT_H

#include <cstddef>

#include "v2m_particles.h"
#include "v2m_tetra.h"

namespace tn {

struct OptParams {
    int max_rounds = 3;   // rounds 4-6 changed nothing measurable on Colin27 (+0.9 s)
    bool flip32 = true, flip23 = true, collapse = true, steiner = true, smooth = true;
    bool kites = true;        // relabel / delete flat tets lying on an interface
    double kite_deg = 10.0;   // ... whose minimum dihedral is below this
    double q = 0.0;           // radius-edge bound a collapse may not exceed (0 = none)
    // mesh-only refinement first (--mode optimize / cdt): the circumcentre of each
    // tet with radius-edge > refine inserted into its region's cavity -- only inside
    // a region, never encroaching a constrained (interface / boundary) face, so the
    // surfaces are kept exactly; 0 = off
    double refine = 0.0;
    int refine_rounds = 30;
    bool verbose = false;
};

struct OptStats {
    int rounds = 0, flips32 = 0, flips23 = 0, collapses = 0, steiner = 0, moves = 0, kites = 0;
    int refined = 0, refine_rounds = 0;   // circumcentres inserted, rounds of it
    double ms = 0;
    double ms_pass[6] = { 0, 0, 0, 0, 0, 0 };   // 3-2, kites, 2-3, collapse, Steiner, smooth
};

// Optimise `out` in place (tets may change, nodes may be removed / added); `nd`
// is remapped to the new node list. Returns the number of operations applied.
size_t optimize_mesh(TetOut& out, Nodes& nd, const OptParams& prm, OptStats& os);

}  // namespace tn

#endif  // V2MESH_OPT_H
