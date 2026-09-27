// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_surflabel.h -- which region is where, for surfaces without trussnet's inner
// / outer labels (--mode cdt / remesh / repair).
//
// Surfaces come with a label per face (M x 4: the region the face bounds), or
// none (M x 3). Where they do not cross, the labels are exact: the surfaces
// split space into cells (the CDT's compartments, tn_cdt.h), and each face says
// one of its two cells is its label:
//
//   * the cell reaching the convex hull is the exterior (0);
//   * across a face labelled l from a known cell: the other cell is l, unless
//     the known one is l itself (then it is not l: a cavity, 0, if nothing else
//     says what it is); a face that appears twice, labelled a and b, lies
//     between a and b;
//   * unlabelled: each cell is a region of its own, numbered outermost first,
//     then largest first (auto-labels "cell"), or by its depth -- the number of
//     surfaces crossed from the exterior (auto-labels "depth"). Where the faces'
//     orientation is consistent (outward shells, nested or touching themselves:
//     the winding number, counted across the cells, agrees everywhere), a cell
//     wound 0 times is exterior -- a pocket or a cavity -- and the depth is the
//     winding number.
//
// The result is trussnet's inner / outer labels (M x 5), exact for the CDT and
// the rasterizer. Surfaces that cross are left to the rasterizer's rules
// (tn_remesh.h), where `overlap` says who owns a volume two regions claim.

#ifndef TRUSSNET_SURFLABEL_H
#define TRUSSNET_SURFLABEL_H

#include <cstddef>
#include <string>
#include <vector>

#include "tn_mesh.h"

namespace tn {

struct SurfLabelOptions {
    // a volume two labels' surfaces both enclose (crossing surfaces only):
    //   nest   the smaller region wins (an inclusion keeps all of itself)   [default]
    //   split  shared halfway between the two surfaces
    //   max / min   the higher / lower label wins
    //   order:L1,L2,..   the first listed wins (unlisted: after, by volume)
    //   union  overlapping regions are one (unlabelled shells)
    //   cells  each overlap is a region of its own (A only, B only, A and B)
    std::string overlap = "nest";
    std::string auto_labels = "cell";   // unlabelled cells: "cell" or "depth"
};

struct SurfLabelStats {
    std::string kind;             // "unlabelled", "one label per face", "inner/outer labels"
    size_t faces = 0, unique_faces = 0, duplicates = 0, degenerate = 0, welded = 0;
    size_t crossings = 0, open_edges = 0;
    int cells = 0;                // enclosed cells (+ the exterior)
    size_t unresolved = 0;        // cells no label reached (cavities: 0)
    size_t conflicts = 0;         // cells two faces gave different labels
    bool oriented = false;        // unlabelled: the faces' orientation gave a winding number
    bool exact = false;           // rewritten to inner / outer labels from the cells
    std::string note;             // why not, or what was done
};

// One label per cell, from the faces' labels and the cells on their sides
// (side[2 f], side[2 f + 1]: -1 unknown; cell 0 the exterior). labels[f]: the
// labels of face f (empty: unlabelled). vol: per cell (for the numbering and the
// nest rule). Unlabelled cells: o.auto_labels.
std::vector<int> solve_cell_labels(int ncells, const std::vector<int>& side, const std::vector<std::vector<int>>& labels,
                                   const std::vector<double>& vol, const SurfLabelOptions& o, SurfLabelStats& st);

// Inner / outer labels for a surface without them (M x 3 / M x 4), when it does
// not cross itself and is closed: nodes welded, repeated faces merged, the cells
// labelled, faces between cells of one label dropped. Returns true when `m` was
// rewritten; false (m unchanged) for inner / outer input, a surface that
// crosses itself or is open (st.note says which) -- the rasterizer's rules then
// apply.
bool normalize_surface_labels(Mesh& m, const SurfLabelOptions& o, SurfLabelStats& st);

// One line: what the input is and what was done with it.
std::string describe(const SurfLabelStats& st);

}  // namespace tn

#endif  // TRUSSNET_SURFLABEL_H
