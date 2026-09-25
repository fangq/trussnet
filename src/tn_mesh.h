// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- GPU particle (truss) multi-label tetrahedral mesher
// Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// mesh.h -- in-memory mesh container produced by the SurfaceNets stage.
//
// SurfaceNets emits one vertex per interface sheet in each boundary-straddling
// cell and a quad per crossing grid edge, each quad separating exactly two
// materials (src/opencl/b2m_surfnet_body.cl). We keep the native QUAD representation; node coordinates are
// double (world/RAS mm), quad vertex indices are stored 0-based internally
// and emitted 1-based per the JMesh spec.

#ifndef TRUSSNET_MESH_H
#define TRUSSNET_MESH_H

#include <cstdint>
#include <vector>

namespace tn {

struct Mesh {
    // Node coordinates, row-major N x 3 (x,y,z), in physical/world units (mm).
    std::vector<double> nodes;

    // Quad connectivity, row-major M x 4, 0-based vertex indices.
    std::vector<int32_t> quads;

    // Optional per-quad material pair (lower,upper), row-major M x 2.
    // Empty if labels are not tracked. Each SurfaceNets quad separates two
    // materials; this records both so callers can keep or collapse them.
    std::vector<int32_t> quad_labels;

    // Optional triangle connectivity, row-major M x 3, 0-based. Populated by
    // the CGAL post-processing stage (repair/remesh/boolean), which emits
    // triangles. When non-empty the mesh is a triangle surface (`quads` unused)
    // and is written as JMesh MeshTri rather than MeshSurf.
    std::vector<int32_t> tris;

    // Optional per-triangle material pair (lower,upper), row-major M x 2, for
    // the combined multi-compartment surface (each face separates two tissues).
    std::vector<int32_t> tri_labels;

    // Optional tetrahedral connectivity, row-major M x 4, 0-based (Phase 7,
    // CGAL Mesh_3 output), with a per-tet tissue label.
    std::vector<int32_t> tets;
    std::vector<int32_t> tet_labels;

    int64_t numNodes() const {
        return static_cast<int64_t>(nodes.size() / 3);
    }
    int64_t numQuads() const {
        return static_cast<int64_t>(quads.size() / 4);
    }
    int64_t numTris()  const {
        return static_cast<int64_t>(tris.size() / 3);
    }
    int64_t numTets()  const {
        return static_cast<int64_t>(tets.size() / 4);
    }
    bool    hasLabels() const {
        return !quad_labels.empty();
    }
    bool    hasTriLabels() const {
        return !tri_labels.empty();
    }
    bool    isTri()     const {
        return !tris.empty();
    }
    bool    isTet()     const {
        return !tets.empty();
    }
};

}  // namespace tn

#endif  // TRUSSNET_MESH_H
