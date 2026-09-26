// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_pipeline.cpp -- see tn_pipeline.h.

#include "tn_pipeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#ifdef TN_HAS_OPENCL
    #include "tn_gpu.h"
#endif
#include "tn_log.h"

namespace tn {

namespace {

typedef std::chrono::steady_clock clk;
double ms_since(clk::time_point a) {
    return std::chrono::duration<double, std::milli>(clk::now() - a).count();
}

// a minimal JNIfTI (BJData) volume writer for debugging: labels, h, grade
nlohmann::json grid_array(const Grid& g, const char* type, const nlohmann::json& data) {
    return nlohmann::json{ { "_ArrayType_", type }, { "_ArraySize_", { g.nz, g.ny, g.nx } }, { "_ArrayData_", data } };
}

void dump_grid(const std::string& path, const LabelVolume& lv, const Grid& g) {
    using nlohmann::json;
    json j;
    j["NIFTIHeader"] = { { "Dim", { g.nx, g.ny, g.nz } }, { "VoxelSize", { g.vs[0], g.vs[1], g.vs[2] } },
        { "hmin", g.hmin }, { "hmax", g.hmax }
    };
    j["Labels"] = grid_array(g, "uint16", lv.data);
    j["Size"] = grid_array(g, "single", g.h);
    j["Grade"] = grid_array(g, "uint8", g.grade);
    std::vector<std::uint8_t> out = json::to_bjdata(j, true, true);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
}

// debug dump: one JSON header line {name: [dtype, shape, byte offset]}, then the
// raw little-endian arrays (read with tools/tnview.py)
void dump_nodes(const std::string& path, const Nodes& nd) {
    using nlohmann::json;
    const int n = static_cast<int>(nd.size());
    const size_t oP = 0, oL = oP + nd.P.size() * 4, oT = oL + nd.lab.size() * 2, oQ = oT + nd.typ.size();
    json j = { { "P", { "float32", { n, 3 }, oP } }, { "Label", { "uint16", { n }, oL } },
        { "Type", { "uint8", { n }, oT } }, { "Partner", { "uint16", { n, 2 }, oQ } }
    };
    std::ofstream f(path, std::ios::binary);
    f << j.dump() << "\n";
    f.write(reinterpret_cast<const char*>(nd.P.data()), static_cast<std::streamsize>(nd.P.size() * 4));
    f.write(reinterpret_cast<const char*>(nd.lab.data()), static_cast<std::streamsize>(nd.lab.size() * 2));
    f.write(reinterpret_cast<const char*>(nd.typ.data()), static_cast<std::streamsize>(nd.typ.size()));
    f.write(reinterpret_cast<const char*>(nd.part.data()), static_cast<std::streamsize>(nd.part.size() * 2));
}

struct FaceSlot {
    std::array<int32_t, 3> k;   // sorted vertices
    int64_t slot;               // 4 t + i: the face of tet t opposite its corner i
};

bool face_slot_less(const FaceSlot& x, const FaceSlot& y) {
    return x.k != y.k ? x.k < y.k : x.slot < y.slot;
}

double label_volume(const LabelVolume& lv) {
    size_t c = 0;

    for (uint16_t l : lv.data) {
        c += l != 0;
    }

    return c * lv.voxelsize[0] * lv.voxelsize[1] * lv.voxelsize[2];
}

void report_tess(const PipelineOptions& o, const LabelVolume& lv, const PipelineResult& r) {
    const TetStats& ts = r.tess;
    TN_FPRINTF(stderr, "[tess]  %zu Delaunay tets -> %zu kept (%zu peeled); conformity: %zu bad faces, %zu edges through label 0, "
               "%zu spanning; %d repair rounds, %zu repairs  (%.0f ms: delaunay %.0f, label %.0f, check %.0f)\n",
               ts.delaunay_tets, ts.kept, ts.peeled, ts.bad_faces, ts.bad_edges, ts.bad_span, ts.repair_rounds,
               ts.repaired, r.ms_tess, ts.ms_delaunay, ts.ms_label, ts.ms_check);
    TN_FPRINTF(stderr, "[conf]  offending-node distance to its voxel interface (voxels, p50/p95/p99/max, 5 = none within 4): faces "
               "%.2f/%.2f/%.2f/%.2f, spanning %.2f/%.2f/%.2f/%.2f\n", ts.dev_face[0], ts.dev_face[1], ts.dev_face[2],
               ts.dev_face[3], ts.dev_span[0], ts.dev_span[1], ts.dev_span[2], ts.dev_span[3]);
    std::string lv_s;
    double worst = 0.0;

    for (size_t l = 1; l < ts.label_vox.size(); ++l)
        if (ts.label_vox[l] > 0) {
            const double e = 100.0 * (ts.label_vol[l] - ts.label_vox[l]) / ts.label_vox[l];
            char b2[48];
            std::snprintf(b2, sizeof(b2), " %zu:%+.1f%%", l, e);
            lv_s += b2;
            worst = std::max(worst, std::fabs(e));
        }

    TN_FPRINTF(stderr, "[conf]  per-label volume error (max |%.2f%%|):%s\n", worst, lv_s.c_str());

    if (!lv.soft_volume.empty()) {   // TPM: against each label's soft volume, sum(p_l) x voxel volume
        std::string sv;
        double w2 = 0.0;

        for (size_t l = 1; l < lv.soft_volume.size() && l < ts.label_vol.size(); ++l) {
            const double s = lv.soft_volume[l];

            if (s > 0) {
                const double e = 100.0 * (ts.label_vol[l] - s) / s;
                char b2[48];
                std::snprintf(b2, sizeof(b2), " %zu:%+.1f%%", l, e);
                sv += b2;
                w2 = std::max(w2, std::fabs(e));
            }
        }

        TN_FPRINTF(stderr, "[conf]  per-label volume vs the TPM soft volume (max |%.2f%%|):%s\n", w2, sv.c_str());
    }

    TN_FPRINTF(stderr, "[snap]  %zu interior nodes pre-snapped onto an interface; %zu coincident nodes dropped\n",
               ts.presnapped, ts.coincident);
    TN_FPRINTF(stderr, "[quality] -q %.3g: %zu nodes added%s\n", o.q, ts.q_added,
               ts.q_rolled_back ? " (a round that cost conformity was rolled back)" : "");
    TN_FPRINTF(stderr, "[smooth] %zu interior-node moves (%.0f ms)\n", ts.smoothed, ts.ms_smooth);
    TN_FPRINTF(stderr, "[opt]   %d 3-2 + %d 2-3 flips, %d kites flattened, %d collapses, %d Steiner points, %d moves "
               "(%.0f ms)\n", ts.opt_flips32, ts.opt_flips23, ts.opt_kites, ts.opt_collapses, ts.opt_steiner,
               ts.opt_moves, ts.ms_opt);
    TN_FPRINTF(stderr, "[qual]  slivers by interior nodes 0/1/2/3/4: %zu/%zu/%zu/%zu/%zu\n", ts.sliver_by_interior[0],
               ts.sliver_by_interior[1], ts.sliver_by_interior[2], ts.sliver_by_interior[3], ts.sliver_by_interior[4]);
    TN_FPRINTF(stderr, "[qual]  min dihedral %.2f deg, slivers <10: %zu (%.2f%%) <5: %zu; Joe-Liu min %.3f p5 %.3f "
               "median %.3f; volume %.1f mm^3 (label volume %.1f)\n", ts.min_dihedral, ts.slivers10,
               100.0 * ts.slivers10 / std::max<size_t>(1, ts.kept), ts.slivers5, ts.joe_liu_min, ts.joe_liu_p5,
               ts.joe_liu_med, ts.volume, label_volume(lv));
}

}  // namespace

bool set_option(PipelineOptions& o, const std::string& name, const std::vector<double>& v, const std::string& str) {
    std::string k;

    for (char c : name) {
        if (c != '_') {
            k += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }

    auto need = [&](size_t n) {
        if (v.size() < n) {
            throw std::runtime_error("trussnet: option '" + name + "' needs a numeric value");
        }
    };
    auto f = [&]() {
        need(1);
        return static_cast<float>(v[0]);
    };
    auto i = [&]() {
        need(1);
        return static_cast<int>(std::lround(v[0]));
    };

    if (k == "size") {
        o.grid.hbase = f();
    } else if (k == "hmin") {
        o.grid.hmin = f();
    } else if (k == "hmax") {
        o.grid.hmax = f();
    } else if (k == "k") {
        o.grid.K = f();
    } else if (k == "grad") {
        o.grid.g = f();
    } else if (k == "sigma") {
        o.grid.sigma = f();
    } else if (k == "sigmathin") {
        o.grid.sigma_thin = f();
    } else if (k == "thick") {
        o.grid.thick = f();
    } else if (k == "thinfloor") {
        o.grid.thin_floor = f();
    } else if (k == "preserve") {
        o.grid.preserve = f();
    } else if (k == "nseed") {
        o.relax.nseed = i();
    } else if (k == "iters" || k == "maxiters") {
        o.relax.max_iters = i();
    } else if (k == "fscale") {
        o.relax.fscale = f();
    } else if (k == "fsurf") {
        o.relax.fsurf = f();
    } else if (k == "dt") {
        o.relax.dt = f();
    } else if (k == "snap") {
        o.relax.snap = f();
    } else if (k == "jseed") {
        o.relax.jseed = f();
    } else if (k == "corners") {
        o.relax.corners = i() != 0;
    } else if (k == "trap") {
        if (str != "smooth" && str != "voxel") {
            throw std::runtime_error("trussnet: trap must be 'smooth' or 'voxel'");
        }

        o.relax.voxel_trap = str == "voxel";
    } else if (k == "q" || k == "reratio" || k == "quality") {
        need(1);
        o.q = v[0];
    } else if (k == "opt") {
        o.opt = i() != 0;
    } else if (k == "smooth") {
        o.smooth = i();
    } else if (k == "repair") {
        o.max_repair = i();
    } else if (k == "thresholds") {
        o.thresholds.assign(v.begin(), v.end());
    } else if (k == "graysigma") {
        o.gray_sigma = f();
    } else if (k == "gpu") {   // on/off: the first GPU (see gpuid)
        if (i() == 0) {
            o.gpu = -2;
        } else if (o.gpu < -1) {
            o.gpu = -1;
        }
    } else if (k == "gpuid") {   // 1-based flat OpenCL device index (as mcxcl); implies gpu
        o.gpu = std::max(1, i()) - 1;
    } else if (k == "verbose") {
        o.relax.verbose = i() != 0;
        o.report = o.relax.verbose;
    } else if (k == "tpmexterior") {
        o.tpm.exterior.clear();

        for (double x : v) {
            o.tpm.exterior.push_back(static_cast<int>(std::lround(x)));
        }
    } else if (k == "tpmmap") {
        o.tpm.map.clear();

        for (double x : v) {
            o.tpm.map.push_back(static_cast<int>(std::lround(x)));
        }
    } else if (k == "tpmspm6") {
        o.tpm.spm6 = i() != 0;
    } else if (k == "tpmsigma") {
        o.tpm.sigma = f();
    } else if (k == "tpmthresh") {   // "T,L:T" or (label, threshold) pairs / one value
        if (!str.empty()) {
            parse_tpm_thresh(str, o.tpm);
        } else {
            set_tpm_thresh(v, o.tpm);
        }
    } else if (k == "tpmholes") {   // 1: keep the enclosed exterior pockets
        o.tpm.fill_holes = i() == 0;
    } else if (k == "tpmfields") {
        o.tpm.fields = i() != 0;
    } else if (k == "lsize") {
        if (v.size() % 2) {
            throw std::runtime_error("trussnet: lsize wants (label, size) pairs");
        }

        for (size_t j = 0; j + 1 < v.size(); j += 2) {
            const int l = static_cast<int>(std::lround(v[j]));

            if (l < 0 || l > 65535) {
                throw std::runtime_error("trussnet: lsize: bad label");
            }

            if (static_cast<int>(o.grid.hlab.size()) <= l) {
                o.grid.hlab.resize(l + 1, 0.0f);
            }

            o.grid.hlab[l] = static_cast<float>(v[j + 1]);
        }
    } else if (k == "thin") {
        o.relax.thin = f();
    } else if (k == "isize") {   // "h,L:h,A:B:h" or (a, b, h) triples, -1 = any
        if (!str.empty()) {
            o.grid.isize.parse(str);
        } else {
            o.grid.isize.add_triples(v);
        }
    } else {
        return false;
    }

    return true;
}

void run_pipeline(LabelVolume& lv, const PipelineOptions& o, PipelineResult& r) {
    const clk::time_point t0 = clk::now();
    r = PipelineResult();

    if (lv.nx < 2 || lv.ny < 2 || lv.nz < 2 || static_cast<int64_t>(lv.data.size()) != int64_t(lv.nx) * lv.ny * lv.nz) {
        throw std::runtime_error("trussnet: the volume must be 3-D (at least 2 voxels per axis)");
    }

    if (!o.thresholds.empty()) {   // gray-scale: (re)label by the iso-values
        if (lv.gray.size() != lv.data.size()) {
            throw std::runtime_error("trussnet: thresholds need a gray-scale volume");
        }

        apply_thresholds(lv, o.thresholds, o.gray_sigma);
    } else {
        int m = 0;

        for (uint16_t l : lv.data) {
            m = std::max<int>(m, l);
        }

        lv.maxlabel = m;
    }

    if (lv.maxlabel == 0) {
        throw std::runtime_error("trussnet: the volume has no non-zero label (0 = exterior)");
    }

    r.ms_input = ms_since(t0);

    if (o.report) {
        TN_FPRINTF(stderr, "[input] %d x %d x %d voxels (%.3g x %.3g x %.3g mm), labels 0..%d  (%.0f ms)\n", lv.nx,
                   lv.ny, lv.nz, lv.voxelsize[0], lv.voxelsize[1], lv.voxelsize[2], lv.maxlabel, r.ms_input);
    }

    clk::time_point t1 = clk::now();
    Grid g;
    build_grid_cpu(lv, o.grid, g);
    r.ms_grid = ms_since(t1);

    if (o.report) {
        TN_FPRINTF(stderr, "[grid]  %zu slots over %d/%d bricks (%d overflow), h in [%.3g, %.3g] mm, %d limit sweeps "
                   "(%.0f ms)\n", g.slot_brick.size(),
        static_cast<int>(std::count_if(g.bl_slot.begin(), g.bl_slot.end(), [](int s) {
            return s >= 0;
        })), g.nbx * g.nby * g.nbz, g.overflow_bricks, g.hmin, g.hmax, g.limit_sweeps, r.ms_grid);
    }

    if (g.overflow_bricks > 0) {   // (not only in report mode: the mesh is degraded there)
        TN_FPRINTF(stderr, "trussnet: warning: %d bricks see more labels within %d voxels than a brick holds; "
                   "the extra labels' interfaces are lost there (raise TN_BL in tn_grid_body.cl)\n", g.overflow_bricks,
                   g.R);
    }

    if (!o.dump_grid.empty()) {
        dump_grid(o.dump_grid, lv, g);
    }

    clk::time_point t2 = clk::now();
    Nodes nd;
    seed_cpu(g, o.relax, nd);
    r.seeds = nd.size();

    // node thinning (--thin): the seeds closer than thin*h to a kept one are dropped
    // before the relaxation (cheaper than thinning afterwards, which needs a second
    // relaxation, and better: ANTS 12.4 s vs 19.7 s, 259 vs 713 slivers)
    if (o.relax.thin > 0.0f) {
        const clk::time_point tt = clk::now();
        const size_t n0 = nd.size();
        r.thinned = thin_nodes(g, o.relax, nd);

        if (o.report) {
            TN_FPRINTF(stderr, "[thin]  %zu of %zu seeds removed (thin %.2f; %.0f ms)\n", r.thinned, n0, o.relax.thin,
                       ms_since(tt));
        }
    }

    r.ms_seed = ms_since(t2);

    if (o.report) {
        TN_FPRINTF(stderr, "[seed]  %zu nodes (%.0f ms)\n", nd.size(), r.ms_seed);
    }

    if (!o.dump_nodes.empty()) {
        dump_nodes(o.dump_nodes + ".seed.bjd", nd);
    }

    clk::time_point t3 = clk::now();
    set_gpu_delaunay(-2);   // (process-wide: reset on every call)
    bool done = false;
#ifdef TN_HAS_OPENCL

    if (o.gpu > -2) {
        const Nodes seeds = nd;   // restored if the device fails part-way

        try {
            RelaxStats rs;
            relax_cl(g, o.relax, nd, rs, o.gpu);
            r.relax = rs;
            done = true;
            r.used_gpu = true;

            if (!std::getenv("TN_GDEL") || std::atoi(std::getenv("TN_GDEL")) != 0) {
                set_gpu_delaunay(o.gpu);   // TN_GDEL=0: keep the CPU Delaunay
            }
        } catch (const std::exception& e) {
            TN_FPRINTF(stderr, "trussnet: OpenCL unavailable (%s); running on the CPU\n", e.what());
            nd = seeds;
        }
    }

#else

    if (o.gpu > -2) {
        TN_FPRINTF(stderr, "trussnet: built without OpenCL; running on the CPU\n");
    }

#endif

    if (!done) {
        relax_cpu(g, o.relax, nd, r.relax);
    }

    r.ms_relax = ms_since(t3);

    if (o.report) {
        const RelaxStats& rs = r.relax;
        TN_FPRINTF(stderr, "[relax] %d iterations, %d rebuilds, last max move %.3g h (p99 < %.2g h); %zu interior, %zu interface, %zu "
                   "junction, %zu corner  (%.0f ms: hash %.0f, force %.0f, move %.0f)\n", rs.iters, rs.rebuilds,
                   rs.last_move, rs.last_p99, rs.n_interior, rs.n_interface, rs.n_junction, rs.n_corner, r.ms_relax,
                   rs.ms_hash, rs.ms_force, rs.ms_move);
    }

    if (!o.dump_nodes.empty()) {
        dump_nodes(o.dump_nodes, nd);
    }

    clk::time_point t4 = clk::now();
    tessellate(g, nd, o.relax.voxel_trap, o.max_repair, r.mesh, r.tess, o.smooth, o.opt, o.q);
    r.ms_tess = ms_since(t4);

    if (!o.dump_nodes.empty()) {   // the final nodes (after repairs / optimisation): mesh node order
        dump_nodes(o.dump_nodes + ".final", nd);
    }

    if (o.report) {
        report_tess(o, lv, r);
    }

    r.ms_total = ms_since(t0);
}

LabelVolume load_volume_file(const std::string& path, const PipelineOptions& o, size_t* tpm_filled,
                             std::vector<int>* tpm_map) {
    LabelVolume lv;

    if (o.thresholds.empty() && is_tpm_file(path)) {
        const Tpm t = load_tpm(path);
        const std::vector<int> map = apply_tpm(t, o.tpm, lv, tpm_filled);

        if (tpm_map) {
            *tpm_map = map;
        }

        return lv;
    }

    lv = load_label_volume(path, !o.thresholds.empty());

    if (!o.thresholds.empty() && lv.gray.empty()) {
        throw std::runtime_error("trussnet: thresholds need a gray-scale volume");
    }

    return lv;
}

void apply_user_sizing(const LabelVolume& lv, const std::vector<double>& h, const std::vector<int>& tpm_map,
                       PipelineOptions& o) {
    if (h.empty()) {
        return;
    }

    const size_t nv = static_cast<size_t>(lv.nx) * lv.ny * lv.nz;
    auto set_label = [&](int l, double v) {
        if (l <= 0 || v <= 0) {   // (the exterior is never meshed; 0 = default)
            return;
        }

        if (static_cast<int>(o.grid.hlab.size()) <= l) {
            o.grid.hlab.resize(l + 1, 0.0f);
        }

        float& d = o.grid.hlab[l];
        d = d > 0 ? std::min(d, static_cast<float>(v)) : static_cast<float>(v);
    };

    if (h.size() == nv) {   // a sizing field
        o.grid.hvox.assign(h.begin(), h.end());

        for (float& v : o.grid.hvox) {
            if (!(v > 0)) {
                v = 0.0f;
            }
        }

        return;
    }

    if (!tpm_map.empty()) {   // one per TPM channel
        if (h.size() != tpm_map.size()) {
            throw std::runtime_error("trussnet: sizing: " + std::to_string(h.size()) + " values for " +
                                     std::to_string(tpm_map.size()) + " TPM channels (or give one per voxel)");
        }

        for (size_t c = 0; c < h.size(); ++c) {
            set_label(tpm_map[c], h[c]);
        }

        return;
    }

    int n = 0;   // the labels 1..n

    if (!o.thresholds.empty()) {
        n = static_cast<int>(o.thresholds.size());
    } else {
        for (uint16_t l : lv.data) {
            n = std::max<int>(n, l);
        }
    }

    if (h.size() == static_cast<size_t>(n) || h.size() == static_cast<size_t>(n) + 1) {
        const int off = h.size() == static_cast<size_t>(n) ? 1 : 0;   // n values: labels 1..n

        for (size_t k = 0; k < h.size(); ++k) {
            set_label(static_cast<int>(k) + off, h[k]);
        }

        return;
    }

    throw std::runtime_error("trussnet: sizing: " + std::to_string(h.size()) + " values; want one per voxel (" +
                             std::to_string(nv) + "), per label (" + std::to_string(n) + " or " +
                             std::to_string(n + 1) + " with the exterior)");
}

void nodes_to_world(const LabelVolume& lv, const TetOut& m, std::vector<double>& world) {
    const size_t nn = m.P.size() / 3;
    world.resize(nn * 3);

    for (size_t i = 0; i < nn; ++i) {
        const double u = m.P[3 * i] / lv.voxelsize[0], v = m.P[3 * i + 1] / lv.voxelsize[1],
                     w = m.P[3 * i + 2] / lv.voxelsize[2];

        for (int k = 0; k < 3; ++k) {
            world[3 * i + k] = lv.affine[4 * k] * u + lv.affine[4 * k + 1] * v + lv.affine[4 * k + 2] * w +
                               lv.affine[4 * k + 3];
        }
    }
}

void extract_faces(const std::vector<int32_t>& tets, const std::vector<int32_t>& labels,
                   const std::vector<double>& nodes, std::vector<int32_t>& faces) {
    const int64_t nt = static_cast<int64_t>(tets.size() / 4);
    const int64_t nf = 4 * nt;
    int32_t nv = 0;

    for (int32_t v : tets) {
        nv = std::max(nv, v + 1);
    }

    // bucket the face-slots (4t+i, the face opposite corner i) by their smallest
    // vertex, then match within each bucket
    typedef FaceSlot F;
    std::vector<int64_t> start(static_cast<size_t>(nv) + 1, 0);
    auto fkey = [&](int64_t s) {
        const int32_t* t = &tets[4 * (s >> 2)];
        const int i = static_cast<int>(s & 3);
        std::array<int32_t, 3> a;
        int n = 0;

        for (int c = 0; c < 4; ++c)
            if (c != i) {
                a[n++] = t[c];
            }

        std::sort(a.begin(), a.end());
        return a;
    };

    for (int64_t s = 0; s < nf; ++s) {
        ++start[fkey(s)[0] + 1];
    }

    for (size_t i = 1; i < start.size(); ++i) {
        start[i] += start[i - 1];
    }

    std::vector<F> fs(nf);
    {
        std::vector<int64_t> cur(start.begin(), start.end() - 1);

        for (int64_t s = 0; s < nf; ++s) {
            const std::array<int32_t, 3> k = fkey(s);
            fs[cur[k[0]]++] = F{ k, s };
        }
    }

    std::vector<uint8_t> keep(nf, 0);   // 1: emit this slot
    #pragma omp parallel for schedule(monotonic: dynamic, 1024)

    for (int64_t b = 0; b < static_cast<int64_t>(nv); ++b) {
        std::sort(fs.begin() + start[b], fs.begin() + start[b + 1], face_slot_less);

        for (int64_t i = start[b]; i < start[b + 1];) {
            int64_t j = i + 1;

            while (j < start[b + 1] && fs[j].k == fs[i].k) {
                ++j;
            }

            if (j - i == 1) {
                keep[fs[i].slot] = 1;   // exterior
            } else {
                const int32_t la = labels[fs[i].slot >> 2], lb = labels[fs[i + 1].slot >> 2];

                if (la != lb) {   // interface: once, from the larger label's side
                    keep[la > lb ? fs[i].slot : fs[i + 1].slot] = 2;
                }
            }

            i = j;
        }
    }

    // partner label of each interface face: found again below by key
    faces.clear();

    for (int64_t b = 0; b < static_cast<int64_t>(nv); ++b)
        for (int64_t i = start[b]; i < start[b + 1]; ++i) {
            const int64_t s = fs[i].slot;

            if (!keep[s]) {
                continue;
            }

            int32_t outer = 0;

            if (keep[s] == 2) {   // the other tet of this face
                const int64_t o = (i + 1 < start[b + 1] && fs[i + 1].k == fs[i].k) ? fs[i + 1].slot : fs[i - 1].slot;
                outer = labels[o >> 2];
            }

            const int32_t* t = &tets[4 * (s >> 2)];
            const int c = static_cast<int>(s & 3);
            int32_t f[3];
            int n = 0;

            for (int q = 0; q < 4; ++q)
                if (q != c) {
                    f[n++] = t[q];
                }

            // orient outward from this (inner) tet: normal away from its opposite corner
            const double* A = &nodes[3 * f[0]];
            const double* B = &nodes[3 * f[1]];
            const double* C = &nodes[3 * f[2]];
            const double* D = &nodes[3 * t[c]];
            const double u[3] = { B[0] - A[0], B[1] - A[1], B[2] - A[2] }, w[3] = { C[0] - A[0], C[1] - A[1], C[2] - A[2] };
            const double nx = u[1] * w[2] - u[2] * w[1];
            const double ny = u[2] * w[0] - u[0] * w[2];
            const double nz = u[0] * w[1] - u[1] * w[0];

            if (nx * (D[0] - A[0]) + ny * (D[1] - A[1]) + nz * (D[2] - A[2]) > 0) {
                std::swap(f[1], f[2]);
            }

            faces.insert(faces.end(), { f[0], f[1], f[2], labels[s >> 2], outer });
        }
}

}  // namespace tn
