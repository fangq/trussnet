// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_meshio.cpp -- see tn_meshio.h.

#include "tn_meshio.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "zmat.h"

#include "nlohmann/json.hpp"

namespace tn {

namespace {

using json = nlohmann::json;

std::string lower_ext(const std::string& path) {
    const size_t d = path.find_last_of('.');
    std::string e = d == std::string::npos ? std::string() : path.substr(d);

    for (char& c : e) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    return e;
}

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);

    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }

    const std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> b(static_cast<size_t>(std::max<std::streamsize>(n, 0)));

    if (n > 0 && !f.read(reinterpret_cast<char*>(b.data()), n)) {
        throw std::runtime_error("cannot read " + path);
    }

    return b;
}

std::vector<uint8_t> unzmat(const uint8_t* in, size_t n, int zipid) {
    unsigned char* out = nullptr;
    size_t outlen = 0;
    int zret = 0;
    const int rc = zmat_run(n, const_cast<unsigned char*>(in), &outlen, &out, zipid, &zret, 0);

    if (rc != 0 || out == nullptr) {
        if (out) {
            zmat_free(&out);
        }

        throw std::runtime_error("JData: cannot decode a compressed array (zmat " + std::to_string(rc) + ")");
    }

    std::vector<uint8_t> r(out, out + outlen);
    zmat_free(&out);
    return r;
}

// A numeric array: values row-major, and its shape.
struct Arr {
    std::vector<double> v;
    std::vector<int64_t> shape;
    int64_t rows() const {
        return shape.empty() ? 0 : shape[0];
    }
    int64_t cols() const {
        return shape.size() < 2 ? 1 : shape[1];
    }
};

void flatten(const json& j, Arr& a, size_t depth) {
    if (j.is_array()) {
        if (a.shape.size() <= depth) {
            a.shape.push_back(static_cast<int64_t>(j.size()));
        }

        for (const json& e : j) {
            flatten(e, a, depth + 1);
        }
    } else if (j.is_number() || j.is_boolean()) {
        a.v.push_back(j.get<double>());
    } else if (j.is_null()) {
        a.v.push_back(NAN);
    } else {
        throw std::runtime_error("JData: not a numeric array");
    }
}

template <typename T>
void cast_into(const uint8_t* p, size_t n, std::vector<double>& v) {
    v.resize(n);

    for (size_t i = 0; i < n; ++i) {
        T x;
        std::memcpy(&x, p + i * sizeof(T), sizeof(T));
        v[i] = static_cast<double>(x);
    }
}

size_t type_size(const std::string& ty) {
    if (ty == "double" || ty == "int64" || ty == "uint64") {
        return 8;
    }

    if (ty == "single" || ty == "int32" || ty == "uint32") {
        return 4;
    }

    if (ty == "int16" || ty == "uint16") {
        return 2;
    }

    if (ty == "int8" || ty == "uint8" || ty == "logical") {
        return 1;
    }

    throw std::runtime_error("JData: unsupported _ArrayType_ " + ty);
}

// n values of type `ty` from raw little-endian bytes
void typed(const std::vector<uint8_t>& raw, size_t n, const std::string& ty, std::vector<double>& v) {
    if (raw.size() < n * type_size(ty)) {
        throw std::runtime_error("JData: array data shorter than its _ArraySize_");
    }

    const uint8_t* p = raw.data();

    if (ty == "double") {
        cast_into<double>(p, n, v);
    } else if (ty == "single") {
        cast_into<float>(p, n, v);
    } else if (ty == "int32") {
        cast_into<int32_t>(p, n, v);
    } else if (ty == "uint32") {
        cast_into<uint32_t>(p, n, v);
    } else if (ty == "int16") {
        cast_into<int16_t>(p, n, v);
    } else if (ty == "uint16") {
        cast_into<uint16_t>(p, n, v);
    } else if (ty == "int8") {
        cast_into<int8_t>(p, n, v);
    } else if (ty == "int64") {
        cast_into<int64_t>(p, n, v);
    } else if (ty == "uint64") {
        cast_into<uint64_t>(p, n, v);
    } else {
        cast_into<uint8_t>(p, n, v);
    }
}

// JData annotated array (_ArrayType_ / _ArraySize_ / _ArrayData_ or
// _ArrayZipData_), or a plain (nested) JSON array. Row-major (JData's default).
Arr decode(const json& j) {
    Arr a;

    if (j.is_array()) {
        flatten(j, a, 0);

        if (a.shape.size() == 1 && !a.v.empty() && a.v.size() != static_cast<size_t>(a.shape[0])) {
            throw std::runtime_error("JData: ragged array");
        }

        return a;
    }

    if (!j.is_object() || !j.contains("_ArrayType_") || !j.contains("_ArraySize_")) {
        throw std::runtime_error("JData: not an array");
    }

    const std::string ty = j["_ArrayType_"].get<std::string>();

    for (const json& s : j["_ArraySize_"]) {
        a.shape.push_back(s.get<int64_t>());
    }

    size_t n = 1;

    for (int64_t s : a.shape) {
        n *= static_cast<size_t>(s);
    }

    if (j.contains("_ArrayData_")) {
        const json& d = j["_ArrayData_"];

        if (d.is_binary()) {
            typed(std::vector<uint8_t>(d.get_binary().begin(), d.get_binary().end()), n, ty, a.v);
        } else {
            Arr f;
            flatten(d, f, 0);
            a.v = f.v;
        }
    } else if (j.contains("_ArrayZipData_")) {
        const std::string zt = j.value("_ArrayZipType_", std::string("zlib"));
        const int zid = zt == "zlib" ? zmZlib : zt == "gzip" ? zmGzip : zt == "lzma" ? zmLzma : -1;

        if (zid < 0) {
            throw std::runtime_error("JData: unsupported _ArrayZipType_ " + zt);
        }

        const json& z = j["_ArrayZipData_"];
        std::vector<uint8_t> comp;

        if (z.is_binary()) {
            comp.assign(z.get_binary().begin(), z.get_binary().end());
        } else if (z.is_string()) {
            const std::string& s = z.get_ref<const std::string&>();
            comp = unzmat(reinterpret_cast<const uint8_t*>(s.data()), s.size(), zmBase64);
        } else if (z.is_array()) {   // BJData's optimized uint8 container comes back as numbers
            comp.reserve(z.size());

            for (const json& e : z) {
                comp.push_back(static_cast<uint8_t>(e.get<int>()));
            }
        } else {
            throw std::runtime_error("JData: _ArrayZipData_ of an unexpected type");
        }

        typed(unzmat(comp.data(), comp.size(), zid), n, ty, a.v);
    } else {
        throw std::runtime_error("JData: an array with no _ArrayData_ / _ArrayZipData_");
    }

    if (a.v.size() != n) {
        throw std::runtime_error("JData: data does not match _ArraySize_");
    }

    return a;
}

// a 1-D array is a column
Arr as_matrix(Arr a) {
    if (a.shape.size() == 1) {
        a.shape.push_back(1);
    }

    return a;
}

void triangles_from(const Arr& a, Mesh& m, bool quads_ok) {
    const int64_t r = a.rows(), c = a.cols();

    if (c < 3) {
        throw std::runtime_error("mesh: a face array with fewer than 3 columns");
    }

    // 3: tri; 4: tri + label (iso2mesh); 5: tri + inner/outer; 6: quad + pair
    const bool quad = quads_ok && c == 6;
    const bool one = !quad && c == 4, two = !quad && c >= 5;

    for (int64_t i = 0; i < r; ++i) {
        const double* f = &a.v[static_cast<size_t>(i * c)];

        auto push = [&](double x, double y, double z) {
            m.tris.push_back(static_cast<int32_t>(x) - 1);
            m.tris.push_back(static_cast<int32_t>(y) - 1);
            m.tris.push_back(static_cast<int32_t>(z) - 1);

            if (quad || two) {
                m.tri_labels.push_back(static_cast<int32_t>(f[quad ? 4 : 3]));
                m.tri_labels.push_back(static_cast<int32_t>(f[quad ? 5 : 4]));
            } else if (one) {
                m.tri_labels.push_back(static_cast<int32_t>(f[3]));
                m.tri_labels.push_back(0);
            }
        };

        push(f[0], f[1], f[2]);

        if (quad) {
            push(f[0], f[2], f[3]);
        }
    }
}

Mesh read_jmesh(const std::string& path, bool binary) {
    const std::vector<uint8_t> b = read_bytes(path);
    json root = binary ? json::from_bjdata(b) : json::parse(b.begin(), b.end());

    // a mesh may sit at the root or under one key (write_jmesh_shells): the
    // first object that has a MeshNode
    const json* doc = &root;

    if (!root.contains("MeshNode")) {
        doc = nullptr;

        for (auto it = root.begin(); it != root.end(); ++it)
            if (it.value().is_object() && it.value().contains("MeshNode")) {
                doc = &it.value();
                break;
            }

        if (!doc) {
            throw std::runtime_error(path + ": no MeshNode");
        }
    }

    Mesh m;
    const Arr node = as_matrix(decode((*doc)["MeshNode"]));

    if (node.cols() < 3) {
        throw std::runtime_error(path + ": MeshNode needs 3 columns");
    }

    const int64_t nn = node.rows();
    m.nodes.resize(static_cast<size_t>(nn) * 3);

    for (int64_t i = 0; i < nn; ++i)
        for (int k = 0; k < 3; ++k) {
            m.nodes[static_cast<size_t>(3 * i + k)] = node.v[static_cast<size_t>(i * node.cols() + k)];
        }

    if (node.cols() >= 4) {
        m.node_labels.resize(static_cast<size_t>(nn));

        for (int64_t i = 0; i < nn; ++i) {
            m.node_labels[static_cast<size_t>(i)] = static_cast<int32_t>(node.v[static_cast<size_t>(i * node.cols() + 3)]);
        }
    }

    if (doc->contains("MeshElem")) {
        const Arr el = as_matrix(decode((*doc)["MeshElem"]));

        if (el.cols() < 4) {
            throw std::runtime_error(path + ": MeshElem needs 4 columns");
        }

        for (int64_t t = 0; t < el.rows(); ++t) {
            for (int k = 0; k < 4; ++k) {
                m.tets.push_back(static_cast<int32_t>(el.v[static_cast<size_t>(t * el.cols() + k)]) - 1);
            }

            m.tet_labels.push_back(el.cols() >= 5 ? static_cast<int32_t>(el.v[static_cast<size_t>(t * el.cols() + 4)]) : 1);
        }
    }

    if (doc->contains("MeshTri")) {
        triangles_from(as_matrix(decode((*doc)["MeshTri"])), m, false);
    } else if (doc->contains("MeshSurf")) {
        triangles_from(as_matrix(decode((*doc)["MeshSurf"])), m, true);
    } else if (doc->contains("MeshFace")) {
        triangles_from(as_matrix(decode((*doc)["MeshFace"])), m, false);
    }

    auto ints = [&](const char* key, size_t per, std::vector<int32_t>& out) {
        if (!doc->contains(key)) {
            return;
        }

        const Arr a = decode((*doc)[key]);

        if (a.v.size() != per * static_cast<size_t>(nn)) {
            throw std::runtime_error(path + ": " + key + " does not match MeshNode");
        }

        out.resize(a.v.size());

        for (size_t i = 0; i < a.v.size(); ++i) {
            out[i] = static_cast<int32_t>(a.v[i]);
        }
    };
    ints("NodeLabel", 1, m.node_labels);
    ints("NodeType", 1, m.node_types);
    ints("NodePartner", 3, m.node_partners);
    return m;
}

Mesh read_off(const std::string& path) {
    std::ifstream f(path);

    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }

    // tokens, comments (#) stripped
    std::vector<std::string> tok;
    std::string line;

    while (std::getline(f, line)) {
        const size_t h = line.find('#');

        if (h != std::string::npos) {
            line.resize(h);
        }

        std::istringstream ss(line);
        std::string t;

        while (ss >> t) {
            tok.push_back(t);
        }
    }

    size_t k = 0;

    if (k < tok.size() && tok[k].find("OFF") != std::string::npos) {
        ++k;
    }

    if (tok.size() < k + 3) {
        throw std::runtime_error(path + ": not an OFF file");
    }

    const long nv = std::stol(tok[k]), nf = std::stol(tok[k + 1]);
    k += 3;
    Mesh m;
    m.nodes.resize(static_cast<size_t>(nv) * 3);

    for (long i = 0; i < 3 * nv; ++i) {
        if (k >= tok.size()) {
            throw std::runtime_error(path + ": truncated vertex list");
        }

        m.nodes[static_cast<size_t>(i)] = std::stod(tok[k++]);
    }

    for (long i = 0; i < nf; ++i) {
        if (k >= tok.size()) {
            throw std::runtime_error(path + ": truncated face list");
        }

        const long c = std::stol(tok[k++]);
        std::vector<int32_t> p(static_cast<size_t>(c));

        for (long j = 0; j < c; ++j) {
            if (k >= tok.size()) {
                throw std::runtime_error(path + ": truncated face");
            }

            p[static_cast<size_t>(j)] = static_cast<int32_t>(std::stol(tok[k++]));
        }

        for (long j = 1; j + 1 < c; ++j) {   // a fan
            m.tris.push_back(p[0]);
            m.tris.push_back(p[static_cast<size_t>(j)]);
            m.tris.push_back(p[static_cast<size_t>(j + 1)]);
        }

        // (face colours after the indices, if any, were read as the next face's
        // count only if present on the same line; OFF files with face colours
        // are rare for meshes and not supported)
    }

    return m;
}

Mesh read_stl(const std::string& path) {
    const std::vector<uint8_t> b = read_bytes(path);
    std::vector<double> corners;   // 9 per triangle
    bool ascii = b.size() >= 5 && std::memcmp(b.data(), "solid", 5) == 0;

    if (ascii && b.size() >= 84) {   // some binary files start with "solid" too
        uint32_t nt;
        std::memcpy(&nt, b.data() + 80, 4);

        if (84 + 50 * static_cast<size_t>(nt) == b.size()) {
            ascii = false;
        }
    }

    if (ascii) {
        std::istringstream ss(std::string(b.begin(), b.end()));
        std::string t;

        while (ss >> t)
            if (t == "vertex") {
                double x, y, z;
                ss >> x >> y >> z;
                corners.push_back(x);
                corners.push_back(y);
                corners.push_back(z);
            }
    } else {
        if (b.size() < 84) {
            throw std::runtime_error(path + ": not an STL file");
        }

        uint32_t nt;
        std::memcpy(&nt, b.data() + 80, 4);

        if (b.size() < 84 + 50 * static_cast<size_t>(nt)) {
            throw std::runtime_error(path + ": truncated binary STL");
        }

        for (uint32_t t = 0; t < nt; ++t)
            for (int v = 0; v < 3; ++v)
                for (int c = 0; c < 3; ++c) {
                    float x;
                    std::memcpy(&x, b.data() + 84 + 50 * static_cast<size_t>(t) + 12 + 12 * v + 4 * c, 4);
                    corners.push_back(x);
                }
    }

    // weld identical corners
    Mesh m;
    std::map<std::tuple<double, double, double>, int32_t> id;

    for (size_t i = 0; i + 2 < corners.size(); i += 3) {
        const auto key = std::make_tuple(corners[i], corners[i + 1], corners[i + 2]);
        auto it = id.find(key);

        if (it == id.end()) {
            it = id.emplace(key, static_cast<int32_t>(m.nodes.size() / 3)).first;
            m.nodes.push_back(corners[i]);
            m.nodes.push_back(corners[i + 1]);
            m.nodes.push_back(corners[i + 2]);
        }

        m.tris.push_back(it->second);
    }

    if (m.tris.size() % 3 != 0) {
        throw std::runtime_error(path + ": a triangle with fewer than 3 vertices");
    }

    return m;
}

Mesh read_xyz(const std::string& path) {
    std::ifstream f(path);

    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }

    Mesh m;
    std::string line;
    bool labels = true;
    std::vector<int32_t> lab;

    while (std::getline(f, line)) {
        for (char& c : line)
            if (c == ',' || c == '\t' || c == ';') {
                c = ' ';
            }

        std::istringstream ss(line);
        double x, y, z, l;

        if (!(ss >> x >> y >> z)) {
            continue;   // a header or a blank line
        }

        m.nodes.push_back(x);
        m.nodes.push_back(y);
        m.nodes.push_back(z);

        if (ss >> l) {
            lab.push_back(static_cast<int32_t>(l));
        } else {
            labels = false;
        }
    }

    if (labels && lab.size() * 3 == m.nodes.size()) {
        m.node_labels = lab;
    }

    return m;
}

}  // namespace

Mesh read_mesh(const std::string& path) {
    const std::string e = lower_ext(path);
    Mesh m;

    if (e == ".jmsh" || e == ".json") {
        m = read_jmesh(path, false);
    } else if (e == ".bmsh" || e == ".bjd") {
        m = read_jmesh(path, true);
    } else if (e == ".off") {
        m = read_off(path);
    } else if (e == ".stl") {
        m = read_stl(path);
    } else if (e == ".xyz" || e == ".txt" || e == ".csv" || e == ".pts") {
        m = read_xyz(path);
    } else {
        throw std::runtime_error(path + ": unknown mesh format (.jmsh .bmsh .off .stl .xyz)");
    }

    const int64_t nn = m.numNodes();

    for (int32_t v : m.tets)
        if (v < 0 || v >= nn) {
            throw std::runtime_error(path + ": a tetrahedron refers to a node out of range");
        }

    for (int32_t v : m.tris)
        if (v < 0 || v >= nn) {
            throw std::runtime_error(path + ": a triangle refers to a node out of range");
        }

    return m;
}

}  // namespace tn
