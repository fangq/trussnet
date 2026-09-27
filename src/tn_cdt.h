// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_cdt.h -- labelled surfaces -> labelled tets, the input surfaces kept exactly
// (--mode cdt): the vendored constrained Delaunay tetrahedrization (Diazzi,
// Panozzo, Vaxman, Attene, ACM TOG 42(6) 2023; third_party/cdt), adapted from
// gpu_brain2mesh's driver (same author).
//
//   1. a closed PLC (junction edges, where 3+ sheets meet, are fine; open edges
//      are not): the triangles, cleaned of duplicates and degeneracies
//   2. Delaunay, then segment and face recovery (exact; the recovery may add
//      Steiner points on the constraints)
//   3. the compartments: floods across the non-constraint faces
//   (fill > 0: interior Steiner points first -- a body-centred cubic lattice of
//   that spacing inside the regions, none within 0.4 x spacing of the surface --
//   so the tets are not all surface to surface; the surface is still kept exactly)
//   4. a compartment's label: where a point inside it lies among the input's
//      regions (tn_remesh.h RegionLocator: inner / outer face labels, or nested
//      shells); outside every region (the exterior, a cavity) it is dropped
//
// The input must not self-intersect (see --mode check); --mode remesh / repair
// handles surfaces that do.

#ifndef TRUSSNET_CDT_H
#define TRUSSNET_CDT_H

#include <cstddef>
#include <string>
#include <vector>

#include "tn_mesh.h"

namespace tn {

struct CdtStats {
    size_t plc_vertices = 0, plc_triangles = 0;
    size_t open_edges = 0, junction_edges = 0;
    size_t steiner = 0;          // vertices the recovery added
    size_t interior = 0;         // interior points added (fill)
    size_t compartments = 0, kept_compartments = 0;
    size_t welded = 0, degenerate = 0;
    std::string labels;          // how the regions were found (tn_surflabel.h describe)
    double ms = 0;
};

// The cells (the enclosed spaces) of a closed surface that does not cross itself:
// its CDT flooded across the non-constraint faces. Cell 0 is the exterior
// (everything reaching the convex hull); side[2 f] / side[2 f + 1]: the cell on
// the negative / positive side of triangle f (normal (b - a) x (c - a)), -1 if
// none was found; vol: per cell. Labels play no part.
struct SurfCells {
    int ncells = 0;
    std::vector<int> side;
    std::vector<double> vol;
    size_t unmatched = 0, conflicts = 0;   // constraint faces on no triangle / two sides disagreeing
};
void cdt_cells(const Mesh& m, SurfCells& sc);

// Tets of the regions of surface `m` (world coordinates), into out.tets /
// out.tet_labels / out.nodes. Throws on an open surface or a CDT failure.
void cdt_mesh(const Mesh& m, Mesh& out, CdtStats& st, double fill = 0);

}  // namespace tn

#endif  // TRUSSNET_CDT_H
