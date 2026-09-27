// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_remesh.h -- surfaces -> per-label soft fields (--mode remesh / repair).
//
// Closed surfaces, possibly self-intersecting, overlapping or inconsistently
// oriented, are turned into a probability map the mesher already meshes
// (tn_tpm.h, --tpm-fields): channel 0 the exterior, channel l region l, each
// p_l = clamp(0.5 + s_l / w) of a signed distance s_l (> 0 inside). The whole
// mesher then runs on it -- curvature sizing, hex seeding, relaxation,
// tessellation -- and its region surfaces are the repaired surfaces: clean by
// construction.
//
//   inside     an exact winding number: rays along x, signed crossings (a union
//              where parts overlap; indifferent to faces that cross)
//   boundary   a face counts only where the inside differs on its two sides, so
//              sheets buried in the union (self-intersections, overlaps) make no
//              interface
//   distance   exact point-triangle distances in a narrow band
//   regions    faces with inner / outer labels (trussnet's MeshTri M x 5): region
//              l's boundary is its faces, each oriented outward from l; else
//              shells (the connected components, orientation repaired): nested,
//              the innermost containing shell's label wins; a shell's label is
//              its faces' (a label column) or its nesting depth + 1; shells of
//              one label are one region (their union)

#ifndef TRUSSNET_REMESH_H
#define TRUSSNET_REMESH_H

#include <cstddef>
#include <memory>

#include "tn_mesh.h"
#include "tn_tpm.h"

namespace tn {

struct RasterStats {
    int nx = 0, ny = 0, nz = 0;
    double voxel = 0;
    int regions = 0, shells = 0;
    size_t faces = 0;
    size_t boundary_faces = 0;              // the exposed faces / parts of faces (the rest buried)
    size_t flipped = 0;                     // faces reoriented
    double ms = 0;
};

// The probability map of the regions of surface `m` on a grid of spacing
// `voxel` (world units) over its bounding box (+ margin). World coordinates are
// kept (the map's affine).
Tpm rasterize_surfaces(const Mesh& m, double voxel, RasterStats& st);

// Which region of surface `m` a point is in (the rules above; 0 = outside every
// region): the exact winding tests the raster uses, for --mode cdt's compartments.
class RegionLocator {
public:
    explicit RegionLocator(const Mesh& m);
    ~RegionLocator();
    int label_at(const double* p) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tn

#endif  // TRUSSNET_REMESH_H
