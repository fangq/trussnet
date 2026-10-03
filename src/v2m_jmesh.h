// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (adapted from gpu_brain2mesh, same author, GPL-3.0-or-later)
//
// jmesh.h -- write a Mesh to a JMesh container (.jmsh text JSON / .bmsh binary
// BJData) per the JMesh specification. Numeric arrays are emitted as
// JData-annotated arrays (zlib-compressed; base64 for text, raw bytes for
// binary), reusing siamize's encoding convention.
//
//   MeshNode  : N x 3 float64 vertex coordinates (mm)
//   MeshSurf  : M x 4 int32 quad node indices (1-based);
//               when the mesh carries labels, two material-pair columns are
//               appended -> M x 6 (v0 v1 v2 v3 matLower matUpper).

#ifndef V2MESH_JMESH_H
#define V2MESH_JMESH_H

#include <string>
#include <utility>
#include <vector>

#include "v2m_mesh.h"

namespace tn {

// The flags that made the meshes (empty: none), written as the string array
// _DataInfo_.CommandFlags in every JMesh file after they are set (v2mesh: its
// arguments but -i / -o and their files, paths as file names unless
// --full-paths).
void set_jmesh_flags(const std::vector<std::string>& flags);

// Write `mesh` to `path`. `binary` selects .bmsh (BJData) vs .jmsh (text JSON).
// Throws std::runtime_error on I/O or codec failure.
void write_jmesh(const std::string& path, const Mesh& mesh, bool binary);

// Convenience: pick text/binary from the file extension (.bmsh -> binary,
// anything else -> text). Returns the chosen `binary` flag.
bool write_jmesh_auto(const std::string& path, const Mesh& mesh);

// Write several named shell meshes into one JMesh container, each stored under
// a top-level JSON key equal to its name (the tissue), e.g.
//   { "_DataInfo_": {...}, "scalp": {MeshNode, MeshSurf}, "skull": {...}, ... }
void write_jmesh_shells(const std::string& path,
                        const std::vector<std::pair<std::string, Mesh>>& shells,
                        bool binary);

// Extension-driven variant of write_jmesh_shells; returns the chosen `binary`.
bool write_jmesh_shells_auto(const std::string& path,
                             const std::vector<std::pair<std::string, Mesh>>& shells);

}  // namespace tn

#endif  // V2MESH_JMESH_H
