// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// pytrussnet.cpp -- pybind11 binding (the compiled `_trussnet` of the `trussnet`
// package in pytrussnet/):
//
//   out = trussnet._trussnet.tetmesh(vol, faces=True, affine=None, voxelsize=None, **opts)
//
//   vol   3-D ndarray indexed vol[x, y, z] (x the first axis): integer
//         labels (0 = exterior), or a gray-scale intensity with thresholds=[...];
//         or 4-D vol[x, y, z, class] tissue probabilities (labels = the argmax;
//         tpm_exterior=[0-based channels], default none -> exterior = 1 - sum)
//   sizing  a sizing field in mm with vol's spatial shape (0 = automatic there),
//         or one size per label (N, or N+1 from label 0) / threshold level / TPM
//         channel (0 = default)
//   opts  tn::set_option names (size, hmin, hmax, lsize, thresholds, gray_sigma,
//         gpu, gpuid, reratio, iters, verbose, ...); lsize: {label: size} or a
//         sequence (index i -> label i + 1)
//         isize: interface sizes, a number (every interface), a string
//         "h,L:h,A:B:h", or a dict {label: h, (a, b): h}
//   affine  4x4 voxel(0-based i,j,k) -> world; default diag(voxelsize)
//   returns dict: node (N x 3 float64), elem (M x 5 int32: v1..v4 [1-based], label),
//                 face (P x 5 int32: v1 v2 v3 [1-based], inner, outer label), info

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tn_2d.h"
#include "tn_log.h"
#include "tn_pipeline.h"

namespace py = pybind11;

namespace {

constexpr int kCOrder = py::array::c_style | py::array::forcecast;   // row-major copy
constexpr int kFOrder = py::array::f_style | py::array::forcecast;   // column-major copy

std::vector<double> numbers(const py::handle& h) {
    if (py::isinstance<py::bool_>(h)) {
        return { h.cast<bool>() ? 1.0 : 0.0 };
    }

    if (py::isinstance<py::int_>(h) || py::isinstance<py::float_>(h)) {
        return { h.cast<double>() };
    }

    py::array_t<double, kCOrder> a = py::array_t<double>::ensure(h);

    if (!a) {
        throw std::runtime_error("expected a number or a sequence of numbers");
    }

    return std::vector<double>(a.data(), a.data() + a.size());
}

// isize as (a, b, h) triples (-1 = any): a number, or {label: h, (a, b): h}
std::vector<double> isize_triples(const py::handle& v) {
    std::vector<double> t;

    if (py::isinstance<py::dict>(v)) {
        for (auto kv : v.cast<py::dict>()) {
            const double h = kv.second.cast<double>();

            if (py::isinstance<py::tuple>(kv.first)) {
                const py::tuple ab = kv.first.cast<py::tuple>();

                if (ab.size() != 2) {
                    throw py::value_error("trussnet: isize keys are labels or (a, b) pairs");
                }

                t.insert(t.end(), { ab[0].cast<double>(), ab[1].cast<double>(), h });
            } else {
                t.insert(t.end(), { kv.first.cast<double>(), -1.0, h });
            }
        }
    } else if (py::isinstance<py::int_>(v) || py::isinstance<py::float_>(v)) {
        t.insert(t.end(), { -1.0, -1.0, v.cast<double>() });
    } else {
        throw py::value_error("trussnet: isize must be a number, a string or a dict");
    }

    return t;
}

// `sizing` (a sizing field, x fastest, or one per label / channel) is returned
// apart: it needs the loaded volume
std::vector<double> parse_options(tn::PipelineOptions& o, const py::kwargs& kw) {
    std::vector<double> sizing;

    for (auto item : kw) {
        const std::string name = item.first.cast<std::string>();
        py::handle v = item.second;

        if (v.is_none()) {
            continue;
        }

        if (name == "sizing") {
            py::array_t<double, kFOrder> a = py::array_t<double, kFOrder>::ensure(v);

            if (!a) {
                throw py::value_error("trussnet: sizing must be an array or a sequence of numbers");
            }

            sizing.assign(a.data(), a.data() + a.size());
            continue;
        }

        if (name == "lsize") {
            std::vector<double> pairs;

            if (py::isinstance<py::dict>(v)) {
                for (auto kv : v.cast<py::dict>()) {
                    pairs.push_back(kv.first.cast<double>());
                    pairs.push_back(kv.second.cast<double>());
                }
            } else {
                const std::vector<double> s = numbers(v);

                for (size_t l = 0; l < s.size(); ++l)
                    if (s[l] > 0) {
                        pairs.push_back(static_cast<double>(l + 1));
                        pairs.push_back(s[l]);
                    }
            }

            tn::set_option(o, "lsize", pairs);
        } else if (name == "isize" && !py::isinstance<py::str>(v)) {
            tn::set_option(o, "isize", isize_triples(v));
        } else if (py::isinstance<py::str>(v)) {
            if (!tn::set_option(o, name, {}, v.cast<std::string>())) {
                throw py::value_error("trussnet: unknown option '" + name + "'");
            }
        } else if (!tn::set_option(o, name, numbers(v))) {
            throw py::value_error("trussnet: unknown option '" + name + "'");
        }
    }


    return sizing;
}

py::dict run_and_pack(tn::LabelVolume& lv, tn::PipelineOptions& o, bool want_faces, size_t tpm_filled,
                      const std::vector<double>& sizing, const std::vector<int>& tpm_map);

py::dict tetmesh(py::array vol, bool want_faces, py::object affine, py::object voxelsize, py::kwargs kw) {
    tn::PipelineOptions o;
    const std::vector<double> sizing = parse_options(o, kw);
    std::vector<int> tpm_map;

    if (vol.ndim() != 3 && vol.ndim() != 4) {
        throw py::value_error("trussnet: vol must be a 3-D array, or 4-D (x, y, z, class) tissue probabilities");
    }

    // x fastest (Fortran order), as tn::LabelVolume; a 4-D array is then
    // channel-major, as tn::Tpm
    py::array_t<double, kFOrder> V(vol);
    tn::LabelVolume lv;
    lv.nx = static_cast<int>(V.shape(0));
    lv.ny = static_cast<int>(V.shape(1));
    lv.nz = static_cast<int>(V.shape(2));
    const double* d = V.data();
    const size_t n = static_cast<size_t>(V.size());
    size_t tpm_filled = 0;
    lv.data.assign(vol.ndim() == 4 ? n / V.shape(3) : n, 0);

    if (vol.ndim() == 4) {   // tissue probabilities
        tn::Tpm t;
        t.nx = lv.nx;
        t.ny = lv.ny;
        t.nz = lv.nz;
        t.C = static_cast<int>(V.shape(3));
        t.p.assign(d, d + n);
        tpm_map = tn::apply_tpm(t, o.tpm, lv, &tpm_filled);
    } else if (!o.thresholds.empty()) {
        lv.gray.assign(d, d + n);
    } else {
        for (size_t i = 0; i < n; ++i) {
            const double x = std::round(d[i]);

            if (!(x >= 0 && x <= 65535)) {
                throw py::value_error("trussnet: labels must be integers in 0..65535 (for a gray-scale volume "
                                      "pass thresholds=[...])");
            }

            lv.data[i] = static_cast<uint16_t>(x);
        }
    }

    std::vector<double> vs = { 1, 1, 1 };

    if (!voxelsize.is_none()) {
        vs = numbers(voxelsize);

        if (vs.size() == 1) {
            vs = std::vector<double>(3, vs[0]);   // not assign(n, v[0]): v[0] aliases v
        }

        if (vs.size() != 3) {
            throw py::value_error("trussnet: voxelsize must have 1 or 3 values");
        }
    }

    if (!affine.is_none()) {
        py::array_t<double, kCOrder> A(affine);

        if (A.ndim() != 2 || A.shape(0) != 4 || A.shape(1) != 4) {
            throw py::value_error("trussnet: affine must be 4x4");
        }

        for (int k = 0; k < 16; ++k) {
            lv.affine[k] = A.data()[k];
        }

        if (voxelsize.is_none()) {   // the voxel size from the affine's columns
            for (int c = 0; c < 3; ++c) {
                vs[c] = std::sqrt(lv.affine[c] * lv.affine[c] + lv.affine[4 + c] * lv.affine[4 + c] +
                                  lv.affine[8 + c] * lv.affine[8 + c]);
            }
        }
    } else {
        lv.affine = { { vs[0], 0, 0, 0, 0, vs[1], 0, 0, 0, 0, vs[2], 0, 0, 0, 0, 1 } };
    }

    lv.voxelsize = { { vs[0], vs[1], vs[2] } };
    return run_and_pack(lv, o, want_faces, tpm_filled, sizing, tpm_map);
}

py::dict run_and_pack(tn::LabelVolume& lv, tn::PipelineOptions& o, bool want_faces, size_t tpm_filled,
                      const std::vector<double>& sizing, const std::vector<int>& tpm_map) {
    tn::apply_user_sizing(lv, sizing, tpm_map, o);
    tn::PipelineResult r;
    std::vector<double> nodes;
    std::vector<int32_t> faces;
    {
        py::gil_scoped_release nogil;
        tn::run_pipeline(lv, o, r);
        tn::nodes_to_world(lv, r.mesh, nodes);

        if (want_faces) {
            tn::extract_faces(r.mesh.tets, r.mesh.label, nodes, faces);
        }
    }

    const py::ssize_t nn = static_cast<py::ssize_t>(nodes.size() / 3), ne = static_cast<py::ssize_t>(r.mesh.tets.size() / 4);
    py::array_t<double> node({ nn, py::ssize_t(3) });
    std::copy(nodes.begin(), nodes.end(), node.mutable_data());
    py::array_t<int32_t> elem({ ne, py::ssize_t(5) });
    int32_t* pe = elem.mutable_data();

    for (py::ssize_t i = 0; i < ne; ++i) {
        for (int k = 0; k < 4; ++k) {
            pe[5 * i + k] = r.mesh.tets[4 * i + k] + 1;
        }

        pe[5 * i + 4] = r.mesh.label[i];
    }

    py::dict out;
    out["node"] = node;
    out["elem"] = elem;

    if (want_faces) {
        const py::ssize_t nf = static_cast<py::ssize_t>(faces.size() / 5);
        py::array_t<int32_t> face({ nf, py::ssize_t(5) });
        int32_t* pf = face.mutable_data();

        for (py::ssize_t i = 0; i < nf; ++i)
            for (int k = 0; k < 5; ++k) {
                pf[5 * i + k] = faces[5 * i + k] + (k < 3 ? 1 : 0);
            }

        out["face"] = face;
    }

    const tn::TetStats& t = r.tess;
    py::dict info;
    info["seeds"] = r.seeds;
    info["thinned"] = r.thinned;
    info["iterations"] = r.relax.iters;
    info["repair_rounds"] = t.repair_rounds;
    info["bad_faces"] = t.bad_faces;
    info["bad_edges"] = t.bad_edges;
    info["spanning"] = t.bad_span;
    info["min_dihedral"] = t.min_dihedral;
    info["slivers10"] = t.slivers10;
    info["joe_liu_min"] = t.joe_liu_min;
    info["joe_liu_p5"] = t.joe_liu_p5;
    info["joe_liu_median"] = t.joe_liu_med;
    info["volume"] = t.volume;
    info["used_gpu"] = r.used_gpu;
    info["tpm_filled"] = tpm_filled;
    info["ms"] = py::dict(py::arg("grid") = r.ms_grid, py::arg("seed") = r.ms_seed, py::arg("relax") = r.ms_relax,
                          py::arg("tess") = r.ms_tess, py::arg("total") = r.ms_total);
    out["info"] = info;
    return out;
}

py::dict tetmesh_file(const std::string& path, bool want_faces, py::kwargs kw) {
    tn::PipelineOptions o;
    const std::vector<double> sizing = parse_options(o, kw);
    size_t filled = 0;
    std::vector<int> tpm_map;
    tn::LabelVolume lv = tn::load_volume_file(path, o, &filled, &tpm_map);
    return run_and_pack(lv, o, want_faces, filled, sizing, tpm_map);
}

// 2-D: img[x, y] labels (or a gray-scale intensity with thresholds=[...]) ->
// {node (N x 2), elem (M x 4: v1 v2 v3 [1-based], label), face (P x 4: v1 v2 [1-based],
// inner, outer label), info}
py::dict trimesh(py::array img, bool want_faces, py::object affine, py::object pixelsize, py::kwargs kw) {
    if (img.ndim() != 2) {
        throw py::value_error("trussnet: trimesh wants a 2-D image");
    }

    tn::Mesh2DOptions o;
    tn::Image2D im;
    std::vector<double> sizing, lsize_pairs;

    for (auto item : kw) {
        const std::string name = item.first.cast<std::string>();
        py::handle v = item.second;

        if (v.is_none()) {
            continue;
        }

        if (name == "sizing") {
            py::array_t<double, kFOrder> a = py::array_t<double, kFOrder>::ensure(v);

            if (!a) {
                throw py::value_error("trussnet: sizing must be an array or a sequence of numbers");
            }

            sizing.assign(a.data(), a.data() + a.size());
        } else if (name == "lsize") {
            if (py::isinstance<py::dict>(v)) {
                for (auto kv : v.cast<py::dict>()) {
                    lsize_pairs.push_back(kv.first.cast<double>());
                    lsize_pairs.push_back(kv.second.cast<double>());
                }
            } else {
                const std::vector<double> q = numbers(v);

                for (size_t l = 0; l < q.size(); ++l)
                    if (q[l] > 0) {
                        lsize_pairs.push_back(static_cast<double>(l + 1));
                        lsize_pairs.push_back(q[l]);
                    }
            }

            tn::set_option2d(o, im.thresholds, "lsize", lsize_pairs);
        } else if (name == "isize") {
            if (py::isinstance<py::str>(v)) {
                tn::set_option2d(o, im.thresholds, "isize", {}, v.cast<std::string>());
            } else {
                tn::set_option2d(o, im.thresholds, "isize", isize_triples(v));
            }
        } else if (!tn::set_option2d(o, im.thresholds, name, numbers(v))) {
            throw py::value_error("trussnet: unknown 2-D option '" + name + "'");
        }
    }

    py::array_t<double, kFOrder> V(img);
    im.nx = static_cast<int>(V.shape(0));
    im.ny = static_cast<int>(V.shape(1));
    const size_t n = static_cast<size_t>(V.size());
    const double* d = V.data();

    if (!im.thresholds.empty()) {
        im.gray.assign(d, d + n);
    } else {
        im.lab.resize(n);

        for (size_t i = 0; i < n; ++i) {
            const double x = std::round(d[i]);

            if (!(x >= 0 && x <= 65535)) {
                throw py::value_error("trussnet: labels must be integers in 0..65535 (for a gray-scale image pass "
                                      "thresholds=[...])");
            }

            im.lab[i] = static_cast<uint16_t>(x);
        }
    }

    std::vector<double> ps = { 1, 1 };

    if (!pixelsize.is_none()) {
        ps = numbers(pixelsize);

        if (ps.size() == 1) {
            ps = std::vector<double>(2, ps[0]);   // not assign(n, v[0]): v[0] aliases v
        }

        if (ps.size() != 2) {
            throw py::value_error("trussnet: pixelsize must have 1 or 2 values");
        }
    }

    if (!affine.is_none()) {   // 3 x 3 (or 2 x 3): pixel (0-based i, j) -> world
        py::array_t<double, kCOrder> A(affine);

        if (A.ndim() != 2 || A.shape(1) != 3 || (A.shape(0) != 2 && A.shape(0) != 3)) {
            throw py::value_error("trussnet: a 2-D affine must be 2 x 3 or 3 x 3");
        }

        for (int k = 0; k < 6; ++k) {
            im.affine[k] = A.data()[k];
        }

        if (pixelsize.is_none()) {
            for (int c = 0; c < 2; ++c) {
                ps[c] = std::hypot(im.affine[c], im.affine[3 + c]);
            }
        }
    } else {
        im.affine = { { ps[0], 0, 0, 0, ps[1], 0 } };
    }

    im.vs = { { ps[0], ps[1] } };

    if (!sizing.empty()) {
        if (sizing.size() == n) {
            o.hvox.assign(sizing.begin(), sizing.end());
        } else {   // one per label (N, or N+1 from 0) / threshold level
            int nl = 0;

            if (!im.thresholds.empty()) {
                nl = static_cast<int>(im.thresholds.size());
            } else {
                for (uint16_t l : im.lab) {
                    nl = std::max<int>(nl, l);
                }
            }

            if (sizing.size() != static_cast<size_t>(nl) && sizing.size() != static_cast<size_t>(nl) + 1) {
                throw std::runtime_error("trussnet: sizing: " + std::to_string(sizing.size()) +
                                         " values; want one per pixel or per label (" + std::to_string(nl) + " or " +
                                         std::to_string(nl + 1) + ")");
            }

            const int off = sizing.size() == static_cast<size_t>(nl) ? 1 : 0;

            for (size_t k = 0; k < sizing.size(); ++k)
                if (sizing[k] > 0 && static_cast<int>(k) + off > 0) {
                    const int l = static_cast<int>(k) + off;

                    if (static_cast<int>(o.hlab.size()) <= l) {
                        o.hlab.resize(l + 1, 0.0f);
                    }

                    o.hlab[l] = static_cast<float>(sizing[k]);
                }
        }
    }

    tn::Mesh2D M;
    tn::Mesh2DStats st;
    {
        py::gil_scoped_release nogil;
        tn::mesh2d(im, o, M, st);
    }

    const py::ssize_t nn = static_cast<py::ssize_t>(M.node.size() / 2), nt = static_cast<py::ssize_t>(M.label.size());
    py::array_t<double> node({ nn, py::ssize_t(2) });
    std::copy(M.node.begin(), M.node.end(), node.mutable_data());
    py::array_t<int32_t> elem({ nt, py::ssize_t(4) });
    int32_t* pe = elem.mutable_data();

    for (py::ssize_t t = 0; t < nt; ++t) {
        for (int k = 0; k < 3; ++k) {
            pe[4 * t + k] = M.tri[3 * t + k] + 1;
        }

        pe[4 * t + 3] = M.label[t];
    }

    py::dict out;
    out["node"] = node;
    out["elem"] = elem;

    if (want_faces) {
        const py::ssize_t ne = static_cast<py::ssize_t>(M.edge.size() / 4);
        py::array_t<int32_t> face({ ne, py::ssize_t(4) });
        int32_t* pf = face.mutable_data();

        for (py::ssize_t e = 0; e < ne; ++e)
            for (int k = 0; k < 4; ++k) {
                pf[4 * e + k] = M.edge[4 * e + k] + (k < 2 ? 1 : 0);
            }

        out["face"] = face;
    }

    py::dict info;
    info["seeds"] = st.seeds;
    info["junctions"] = st.junctions;
    info["iterations"] = st.iterations;
    info["repair_rounds"] = st.repair_rounds;
    info["repairs"] = st.repairs;
    info["bad_edges"] = st.bad_edges;
    info["spanning"] = st.spanning;
    info["min_angle"] = st.min_angle;
    info["q_min"] = st.q_min;
    info["q_p5"] = st.q_p5;
    info["q_median"] = st.q_median;
    info["label_area"] = st.label_area;
    info["label_pixels"] = st.label_pixels;
    info["ms"] = py::dict(py::arg("fields") = st.ms_fields, py::arg("relax") = st.ms_relax,
                          py::arg("mesh") = st.ms_mesh, py::arg("total") = st.ms_total);
    out["info"] = info;
    return out;
}

}  // namespace

PYBIND11_MODULE(_trussnet, m) {
    m.doc() = "trussnet: GPU particle (truss) multi-label / gray-scale tetrahedral mesher";
    m.attr("__version__") = TN_VERSION;
    m.def("tetmesh", &tetmesh, py::arg("vol"), py::kw_only(), py::arg("faces") = true, py::arg("affine") = py::none(),
          py::arg("voxelsize") = py::none(),
          "Mesh a 3-D label (or, with thresholds=[...], gray-scale) volume, or 4-D tissue probabilities; see the "
          "trussnet package docs.");
    m.def("trimesh", &trimesh, py::arg("img"), py::kw_only(), py::arg("faces") = true, py::arg("affine") = py::none(),
          py::arg("pixelsize") = py::none(),
          "Triangle mesh of a 2-D label (or, with thresholds=[...], gray-scale) image; see the trussnet package docs.");
    m.def("tetmesh_file", &tetmesh_file, py::arg("path"), py::kw_only(), py::arg("faces") = true,
          "Mesh a volume file (.nii/.nii.gz/.jnii/.bnii: labels, gray-scale with thresholds=, or a 4-D TPM) in its "
          "world coordinates.");
}
