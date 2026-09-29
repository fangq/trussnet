// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
// (adapted from gpu_brain2mesh, same author, GPL-3.0-or-later)
//
// jmesh.cpp -- JMesh (.jmsh/.bmsh) writer. See jmesh.h.

#include "v2m_jmesh.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

// zmat (zlib + base64). Declarations only here -- the single implementation
// (ZMAT_IMPLEMENTATION) is owned by src/io/nifti_io.cpp.
#include "zmat.h"

#include "nlohmann/json.hpp"

namespace tn {

namespace {

using json = nlohmann::ordered_json;  // preserve key order in the emitted JSON

// Thin RAII-ish wrapper around zmat_run; returns the transformed bytes or
// throws. iscompress=1 to encode (compress / base64), 0 to decode.
std::vector<uint8_t> zmat_apply(const uint8_t* in, size_t n, int zipid, int iscompress) {
    unsigned char* out = nullptr;
    size_t outlen = 0;
    int zret = 0;
    int rc = zmat_run(n, const_cast<unsigned char*>(in), &outlen, &out, zipid, &zret, iscompress);

    if (rc != 0 || out == nullptr) {
        if (out) {
            zmat_free(&out);
        }

        throw std::runtime_error("zmat_run failed (zipid=" + std::to_string(zipid) + ")");
    }

    std::vector<uint8_t> result(out, out + outlen);
    zmat_free(&out);
    return result;
}

// Build a JData-annotated numeric array object from raw, already-row-major
// bytes. `shape` is in element units (e.g. {N, 3}); `total` element count is
// prod(shape). Pipeline: raw -> zlib -> (base64 for text | BJData bytes for
// binary). Mirrors siamize's jdata_annotated().
json jdata_array(const uint8_t* bytes, size_t nbytes, const char* dtype,
                 const std::vector<int64_t>& shape, bool binary) {
    // zlib level 1 (zmat: iscompress = -level): ~3x faster than the default level
    // for a slightly larger file; still standard zlib for every JData reader
    std::vector<uint8_t> zlib = zmat_apply(bytes, nbytes, zmZlib, -1);

    int64_t total = 1;

    for (int64_t s : shape) {
        total *= s;
    }

    json a = json::object();
    a["_ArrayType_"]    = dtype;
    a["_ArraySize_"]    = shape;
    a["_ArrayZipType_"] = "zlib";
    a["_ArrayZipSize_"] = json::array({ static_cast<int64_t>(1), total });

    if (binary) {
        a["_ArrayZipData_"] = json::binary(zlib);
    } else {
        std::vector<uint8_t> b64 = zmat_apply(zlib.data(), zlib.size(), zmBase64, 1);
        a["_ArrayZipData_"] = std::string(b64.begin(), b64.end());
    }

    return a;
}

void write_bytes(const std::string& path, const uint8_t* data, size_t n) {
    std::ofstream f(path, std::ios::binary);

    if (!f) {
        throw std::runtime_error("cannot open for writing: " + path);
    }

    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));

    if (!f) {
        throw std::runtime_error("write failed: " + path);
    }
}

// Encode one mesh as a JSON object: MeshNode, then whichever of MeshElem
// (tets), MeshTri (triangles) and MeshSurf (quads) it has -- tets and
// triangles together when both are present (a volume mesh and its region
// surfaces) -- and the node attributes (NodeLabel / NodeType / NodePartner)
// when tracked. Used both at the document root (single mesh) and nested under
// a per-tissue key (shells).
json mesh_to_json(const Mesh& mesh, bool binary) {
    const int64_t nnode = mesh.numNodes();
    json obj = json::object();

    // MeshNode: N x 3 float64, already row-major (x,y,z per node).
    obj["MeshNode"] = jdata_array(
                          reinterpret_cast<const uint8_t*>(mesh.nodes.data()),
                          mesh.nodes.size() * sizeof(double),
                          "double", { nnode, 3 }, binary);

    auto int_array = [&](const std::vector<int32_t>& v, int64_t rows, int64_t cols) {
        std::vector<int64_t> shape = cols == 1 ? std::vector<int64_t> { rows } : std::vector<int64_t> { rows, cols };
        return jdata_array(reinterpret_cast<const uint8_t*>(v.data()), v.size() * sizeof(int32_t), "int32", shape, binary);
    };

    // Tetrahedra: MeshElem, M x 4 (1-based) + a per-tet tissue-label column -> M x 5.
    if (mesh.isTet()) {
        const int64_t nt = mesh.numTets();
        std::vector<int32_t> el(static_cast<size_t>(nt) * 5);

        for (int64_t t = 0; t < nt; ++t) {
            for (int c = 0; c < 4; ++c) {
                el[static_cast<size_t>(t) * 5 + c] = mesh.tets[static_cast<size_t>(t) * 4 + c] + 1;
            }

            el[static_cast<size_t>(t) * 5 + 4] =
                mesh.tet_labels.empty() ? 1 : mesh.tet_labels[static_cast<size_t>(t)];
        }

        obj["MeshElem"] = int_array(el, nt, 5);
    }

    // Triangles: MeshTri, M x 3 (1-based), plus two material-pair columns -> M x 5
    // when tri labels are present.
    if (mesh.isTri()) {
        const int64_t ntri = mesh.numTris();
        const int tcols = mesh.hasTriLabels() ? 5 : 3;
        std::vector<int32_t> tri(static_cast<size_t>(ntri) * tcols);

        for (int64_t t = 0; t < ntri; ++t) {
            for (int c = 0; c < 3; ++c) {
                tri[static_cast<size_t>(t) * tcols + c] = mesh.tris[static_cast<size_t>(t) * 3 + c] + 1;
            }

            if (mesh.hasTriLabels()) {
                tri[static_cast<size_t>(t) * tcols + 3] = mesh.tri_labels[static_cast<size_t>(t) * 2 + 0];
                tri[static_cast<size_t>(t) * tcols + 4] = mesh.tri_labels[static_cast<size_t>(t) * 2 + 1];
            }
        }

        obj["MeshTri"] = int_array(tri, ntri, tcols);
    }

    // Quads: MeshSurf, M x (4 [+2]) int32, 1-based indices, optional material pair.
    if (!mesh.isTet() && !mesh.isTri() && mesh.numQuads() > 0) {
        const int64_t nquad = mesh.numQuads();
        const int cols = mesh.hasLabels() ? 6 : 4;
        std::vector<int32_t> surf(static_cast<size_t>(nquad) * cols);

        for (int64_t q = 0; q < nquad; ++q) {
            for (int c = 0; c < 4; ++c) {
                surf[static_cast<size_t>(q) * cols + c] = mesh.quads[static_cast<size_t>(q) * 4 + c] + 1;  // 1-based
            }

            if (mesh.hasLabels()) {
                surf[static_cast<size_t>(q) * cols + 4] = mesh.quad_labels[static_cast<size_t>(q) * 2 + 0];
                surf[static_cast<size_t>(q) * cols + 5] = mesh.quad_labels[static_cast<size_t>(q) * 2 + 1];
            }
        }

        obj["MeshSurf"] = int_array(surf, nquad, cols);
    }

    if (static_cast<int64_t>(mesh.node_labels.size()) == nnode) {
        obj["NodeLabel"] = int_array(mesh.node_labels, nnode, 1);
    }

    if (static_cast<int64_t>(mesh.node_types.size()) == nnode) {
        obj["NodeType"] = int_array(mesh.node_types, nnode, 1);
    }

    if (static_cast<int64_t>(mesh.node_partners.size()) == 3 * nnode) {
        obj["NodePartner"] = int_array(mesh.node_partners, nnode, 3);
    }

    return obj;
}

json data_info() {
    return json{
        { "JMeshVersion", "0.5" },
        { "Comment", "generated by v2mesh" }
    };
}

}  // namespace

void write_jmesh(const std::string& path, const Mesh& mesh, bool binary) {
    json root = json::object();
    root["_DataInfo_"] = data_info();
    json m = mesh_to_json(mesh, binary);

    for (auto& el : m.items()) {   // MeshNode + MeshElem / MeshTri / MeshSurf + node attributes
        root[el.key()] = el.value();
    }

    if (binary) {
        std::vector<uint8_t> bj;
        json::to_bjdata(root, bj, /*use_size=*/true, /*use_type=*/true);
        write_bytes(path, bj.data(), bj.size());
    } else {
        std::string txt = root.dump();
        write_bytes(path, reinterpret_cast<const uint8_t*>(txt.data()), txt.size());
    }
}

bool write_jmesh_auto(const std::string& path, const Mesh& mesh) {
    bool binary = path.size() >= 5 && path.compare(path.size() - 5, 5, ".bmsh") == 0;
    write_jmesh(path, mesh, binary);
    return binary;
}

void write_jmesh_shells(const std::string& path,
                        const std::vector<std::pair<std::string, Mesh>>& shells,
                        bool binary) {
    json root = json::object();
    root["_DataInfo_"] = data_info();

    // Each shell mesh is stored under a JSON key named by its tissue, e.g.
    //   { "scalp": {MeshNode, MeshSurf}, "skull": {MeshNode, MeshSurf}, ... }
    for (const auto& kv : shells) {
        root[kv.first] = mesh_to_json(kv.second, binary);
    }

    if (binary) {
        std::vector<uint8_t> bj;
        json::to_bjdata(root, bj, /*use_size=*/true, /*use_type=*/true);
        write_bytes(path, bj.data(), bj.size());
    } else {
        std::string txt = root.dump();
        write_bytes(path, reinterpret_cast<const uint8_t*>(txt.data()), txt.size());
    }
}

bool write_jmesh_shells_auto(const std::string& path,
                             const std::vector<std::pair<std::string, Mesh>>& shells) {
    bool binary = path.size() >= 5 && path.compare(path.size() - 5, 5, ".bmsh") == 0;
    write_jmesh_shells(path, shells, binary);
    return binary;
}

}  // namespace tn
