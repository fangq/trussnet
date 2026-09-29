// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_remesh.h -- surfaces -> per-label soft fields (--mode remesh / repair).
//
// Closed surfaces, possibly self-intersecting, overlapping or inconsistently
// oriented, are turned into a probability map the mesher already meshes
// (v2m_tpm.h, --tpm-fields): channel 0 the exterior, channel l region l, each
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
//   regions    faces with inner / outer labels (v2mesh's MeshTri M x 5): region
//              l's boundary is its faces, each oriented outward from l (surfaces
//              that do not cross come here from v2m_surflabel.h, exact); else
//              shells (connected components -- within one label's faces -- with
//              the orientation repaired): nested, the innermost containing
//              shell's label wins; a shell's label is its faces' or, unlabelled,
//              its own (outermost then largest first, or the nesting depth + 1);
//              a shell labelled as the region around it is a hole in it; shells
//              of one label are one region (their union)
//   overlap    a volume two regions claim (SurfLabelOptions::overlap): the
//              smaller wins (nest), halfway (split), by label, by a list, one
//              region (union) or each overlap its own (cells)
//   flood      non-manifold surfaces (sheets meeting at junctions), or a label's
//              faces not closed on their own, that also cross: no winding
//              number; the cells flooded on the raster instead, labelled as the
//              exact cells are -- voxel-exact only (the fields then smoothed)

#ifndef V2MESH_REMESH_H
#define V2MESH_REMESH_H

#include <cstddef>
#include <memory>
#include <string>

#include "v2m_mesh.h"
#include "v2m_surflabel.h"
#include "v2m_tpm.h"

namespace tn {

struct RasterStats {
    int nx = 0, ny = 0, nz = 0;
    double voxel = 0;
    int regions = 0, shells = 0;
    size_t faces = 0;
    size_t boundary_faces = 0;              // the exposed faces / parts of faces (the rest buried)
    size_t flipped = 0;                     // faces reoriented
    std::string labels;                     // how the regions were found (v2m_surflabel.h describe)
    bool flood = false;                     // the cells flooded on the raster (voxel-exact only)
    double ms = 0;
};

// The probability map of the regions of surface `m` on a grid of spacing
// `voxel` (world units) over its bounding box (+ margin). World coordinates are
// kept (the map's affine).
Tpm rasterize_surfaces(const Mesh& m, double voxel, RasterStats& st, const SurfLabelOptions& o = SurfLabelOptions());

// Which region of surface `m` a point is in (the rules above; 0 = outside every
// region): the exact winding tests the raster uses, for --mode cdt's compartments.
class RegionLocator {
public:
    explicit RegionLocator(const Mesh& m, const SurfLabelOptions& o = SurfLabelOptions());
    ~RegionLocator();
    int label_at(const double* p) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tn

#endif  // V2MESH_REMESH_H
