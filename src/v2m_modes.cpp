// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_modes.cpp -- see v2m_modes.h.

#include "v2m_modes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "v2m_log.h"
#include "v2m_omp.h"
#include "v2m_pipeline.h"
#include "v2m_tetra.h"
#include "v2m_tpm.h"

#ifdef V2M_HAS_CDT
    #include "implicit_point.h"   // exact orient3d (Attene's predicates, third_party/cdt)
    #include "delaunay.h"         // the exact Delaunay (Diazzi et al., third_party/cdt)
    #include "v2m_gdel.h"
#endif

namespace tn {

namespace {

// sign of the orientation of (a, b, c, d): exact where the vendored predicates
// are built in, else a plain determinant
inline int orient(const double* a, const double* b, const double* c, const double* d) {
#ifdef V2M_HAS_CDT
    return ::orient3d(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]);
#else
    const double ax = a[0] - d[0], ay = a[1] - d[1], az = a[2] - d[2];
    const double bx = b[0] - d[0], by = b[1] - d[1], bz = b[2] - d[2];
    const double cx = c[0] - d[0], cy = c[1] - d[1], cz = c[2] - d[2];
    const double det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
    return det > 0 ? 1 : det < 0 ? -1 : 0;
#endif
}

// does segment p q meet triangle a b c (closed)? p, q not both in its plane
bool seg_tri(const double* p, const double* q, const double* a, const double* b, const double* c) {
    const int sp = orient(a, b, c, p), sq = orient(a, b, c, q);

    if (sp == sq) {   // both on one side (or both in the plane: the coplanar test handles that)
        return false;
    }

    const int s1 = orient(p, q, a, b), s2 = orient(p, q, b, c), s3 = orient(p, q, c, a);
    return (s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0);
}

// 2-D overlap of two coplanar triangles (dropping the dominant normal axis);
// strict interior overlap only -- touching along an edge or at a corner is not
// a crossing
bool coplanar_overlap(const double* t[3], const double* u[3]) {
    const double n[3] = { (t[1][1] - t[0][1]) * (t[2][2] - t[0][2]) - (t[1][2] - t[0][2]) * (t[2][1] - t[0][1]),
                          (t[1][2] - t[0][2]) * (t[2][0] - t[0][0]) - (t[1][0] - t[0][0]) * (t[2][2] - t[0][2]),
                          (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])
                        };
    int ax = 0;

    if (std::fabs(n[1]) > std::fabs(n[ax])) {
        ax = 1;
    }

    if (std::fabs(n[2]) > std::fabs(n[ax])) {
        ax = 2;
    }

    const int i0 = (ax + 1) % 3, i1 = (ax + 2) % 3;
    auto o2 = [&](const double* a, const double* b, const double* c) {
        const double d = (b[i0] - a[i0]) * (c[i1] - a[i1]) - (b[i1] - a[i1]) * (c[i0] - a[i0]);
        return d > 0 ? 1 : d < 0 ? -1 : 0;
    };
    // separating axis on the six edges: overlap unless an edge line has the
    // other triangle entirely on its outer side (touching counts as separated)
    auto separated = [&](const double* A[3], const double* B[3]) {
        const int s = o2(A[0], A[1], A[2]);

        if (s == 0) {
            return true;   // degenerate
        }

        for (int e = 0; e < 3; ++e) {
            bool out = true;

            for (int k = 0; k < 3 && out; ++k) {
                out = o2(A[e], A[(e + 1) % 3], B[k]) * s <= 0;
            }

            if (out) {
                return true;
            }
        }

        return false;
    };
    return !separated(t, u) && !separated(u, t);
}

// do triangles t and u cross, given the corners they share (0, 1 or 2)?
bool tri_tri(const double* t[3], const double* u[3], int shared, const int* ts, const int* us) {
    // coplanar?
    bool cop = true;

    for (int k = 0; k < 3 && cop; ++k) {
        cop = orient(t[0], t[1], t[2], u[k]) == 0;
    }

    if (cop) {
        return coplanar_overlap(t, u);   // a shared edge: overlap only if folded onto each other
    }

    if (shared == 2) {   // a common edge: they cross only if coplanar and folded (handled above)
        return false;
    }

    // an edge of one through the other; with a shared corner, only the edges
    // away from it (those touching it meet the other triangle at that corner)
    for (int e = 0; e < 3; ++e) {
        const int a = e, b = (e + 1) % 3;

        if (!(shared == 1 && (a == ts[0] || b == ts[0])) && seg_tri(t[a], t[b], u[0], u[1], u[2])) {
            return true;
        }

        if (!(shared == 1 && (a == us[0] || b == us[0])) && seg_tri(u[a], u[b], t[0], t[1], t[2])) {
            return true;
        }
    }

    return false;
}

}  // namespace

size_t self_intersections(const std::vector<double>& nodes, const std::vector<int32_t>& tris,
                          std::vector<std::pair<int32_t, int32_t>>* pairs, size_t keep) {
    const size_t nt = tris.size() / 3;

    if (nt < 2) {
        return 0;
    }

    // the grid: cells about two edge lengths across
    double lo[3] = { 1e300, 1e300, 1e300 }, hi[3] = { -1e300, -1e300, -1e300 }, esum = 0;

    for (size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 3; ++k) {
            const double* a = &nodes[3 * static_cast<size_t>(tris[3 * t + k])];
            const double* b = &nodes[3 * static_cast<size_t>(tris[3 * t + (k + 1) % 3])];
            esum += std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));

            for (int c = 0; c < 3; ++c) {
                lo[c] = std::min(lo[c], a[c]);
                hi[c] = std::max(hi[c], a[c]);
            }
        }

    double cell = std::max(2.0 * esum / (3.0 * nt), 1e-12);
    int dim[3];

    for (int c = 0; c < 3; ++c) {
        dim[c] = std::max(1, std::min(1024, static_cast<int>(std::ceil((hi[c] - lo[c]) / cell)) + 1));
    }

    // triangles into the cells their bounding boxes cover
    std::unordered_map<int64_t, std::vector<int32_t>> grid;

    for (size_t t = 0; t < nt; ++t) {
        int b0[3], b1[3];

        for (int c = 0; c < 3; ++c) {
            double mn = 1e300, mx = -1e300;

            for (int k = 0; k < 3; ++k) {
                const double x = nodes[3 * static_cast<size_t>(tris[3 * t + k]) + c];
                mn = std::min(mn, x);
                mx = std::max(mx, x);
            }

            b0[c] = std::min(dim[c] - 1, std::max(0, static_cast<int>((mn - lo[c]) / cell)));
            b1[c] = std::min(dim[c] - 1, std::max(0, static_cast<int>((mx - lo[c]) / cell)));
        }

        for (int z = b0[2]; z <= b1[2]; ++z)
            for (int y = b0[1]; y <= b1[1]; ++y)
                for (int x = b0[0]; x <= b1[0]; ++x) {
                    grid[(static_cast<int64_t>(z) * dim[1] + y) * dim[0] + x].push_back(static_cast<int32_t>(t));
                }
    }

    // candidate pairs (each once: lower index first, deduplicated)
    std::vector<std::pair<int32_t, int32_t>> cand;

    for (auto& kv : grid) {
        const std::vector<int32_t>& v = kv.second;

        for (size_t i = 0; i < v.size(); ++i)
            for (size_t j = i + 1; j < v.size(); ++j) {
                cand.emplace_back(std::min(v[i], v[j]), std::max(v[i], v[j]));
            }
    }

    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    std::vector<char> hit(cand.size(), 0);
    #pragma omp parallel for schedule(monotonic: dynamic, 4096)

    for (int64_t c = 0; c < static_cast<int64_t>(cand.size()); ++c) {
        const int32_t a = cand[c].first, b = cand[c].second;
        const double* t[3], *u[3];
        int ts[3] = { -1, -1, -1 }, us[3] = { -1, -1, -1 }, shared = 0;

        for (int k = 0; k < 3; ++k) {
            t[k] = &nodes[3 * static_cast<size_t>(tris[3 * a + k])];
            u[k] = &nodes[3 * static_cast<size_t>(tris[3 * b + k])];
        }

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (tris[3 * a + i] == tris[3 * b + j]) {
                    ts[shared] = i;
                    us[shared] = j;
                    ++shared;
                }

        if (shared == 3) {   // the same triangle twice
            hit[c] = 1;
            continue;
        }

        // bounding boxes first
        bool apart = false;

        for (int ax = 0; ax < 3 && !apart; ++ax) {
            const double tmn = std::min(t[0][ax], std::min(t[1][ax], t[2][ax])), tmx = std::max(t[0][ax], std::max(t[1][ax], t[2][ax]));
            const double umn = std::min(u[0][ax], std::min(u[1][ax], u[2][ax])), umx = std::max(u[0][ax], std::max(u[1][ax], u[2][ax]));
            apart = tmx < umn || umx < tmn;
        }

        if (!apart && tri_tri(t, u, shared, ts, us)) {
            hit[c] = 1;
        }
    }

    size_t n = 0;

    for (size_t c = 0; c < cand.size(); ++c)
        if (hit[c]) {
            if (pairs && pairs->size() < keep) {
                pairs->push_back(cand[c]);
            }

            ++n;
        }

    return n;
}

void optimize_tets(Mesh& m, const OptParams& prm, OptStats& os) {
    const size_t nn = m.nodes.size() / 3, nt = m.tets.size() / 4;

    if (nt == 0) {
        throw std::runtime_error("optimize: the mesh has no tetrahedra");
    }

    // float coordinates about the centroid (the optimiser's arrays are float)
    double c[3] = { 0, 0, 0 };

    for (size_t i = 0; i < nn; ++i)
        for (int k = 0; k < 3; ++k) {
            c[k] += m.nodes[3 * i + k] / static_cast<double>(nn);
        }

    TetOut out;
    out.P.resize(3 * nn);

    for (size_t i = 0; i < nn; ++i)
        for (int k = 0; k < 3; ++k) {
            out.P[3 * i + k] = static_cast<float>(m.nodes[3 * i + k] - c[k]);
        }

    out.tets = m.tets;
    out.label = m.tet_labels.empty() ? std::vector<int32_t>(nt, 1) : m.tet_labels;
    // the label sets
    std::vector<std::vector<int32_t>> sets(nn);

    for (size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k) {
            std::vector<int32_t>& s = sets[static_cast<size_t>(out.tets[4 * t + k])];

            if (std::find(s.begin(), s.end(), out.label[t]) == s.end()) {
                s.push_back(out.label[t]);
            }
        }

    std::vector<int32_t> f;
    extract_faces(out.tets, out.label, m.nodes, f);

    for (size_t i = 0; i + 4 < f.size(); i += 5)
        if (f[i + 4] == 0)
            for (int k = 0; k < 3; ++k) {
                std::vector<int32_t>& s = sets[static_cast<size_t>(f[i + k])];

                if (std::find(s.begin(), s.end(), 0) == s.end()) {
                    s.push_back(0);
                }
            }

    const uint16_t nolab = 0xFFFF;
    Nodes nd;
    nd.P = out.P;
    nd.lab.resize(nn);
    nd.typ.resize(nn);
    nd.part.assign(2 * nn, nolab);
    nd.part3.assign(nn, nolab);

    for (size_t i = 0; i < nn; ++i) {
        std::vector<int32_t> s = sets[i];
        std::sort(s.begin(), s.end());
        // the own label: a tissue (never the exterior); the rest are partners
        auto own = std::find_if(s.begin(), s.end(), [](int32_t l) {
            return l != 0;
        });
        const int32_t a = own == s.end() ? 1 : *own;

        if (own != s.end()) {
            s.erase(own);
        }

        nd.lab[i] = static_cast<uint16_t>(a);
        nd.typ[i] = static_cast<uint8_t>(std::min<size_t>(s.size(), 3));

        for (size_t k = 0; k < s.size() && k < 3; ++k) {
            if (k < 2) {
                nd.part[2 * i + k] = static_cast<uint16_t>(s[k]);
            } else {
                nd.part3[i] = static_cast<uint16_t>(s[k]);
            }
        }
    }

    optimize_mesh(out, nd, prm, os);
    const size_t n2 = out.P.size() / 3;
    m.nodes.resize(3 * n2);

    for (size_t i = 0; i < n2; ++i)
        for (int k = 0; k < 3; ++k) {
            m.nodes[3 * i + k] = static_cast<double>(out.P[3 * i + k]) + c[k];
        }

    m.tets = out.tets;
    m.tet_labels = out.label;
    m.tris.clear();
    m.tri_labels.clear();
    m.node_labels.clear();
    m.node_types.clear();
    m.node_partners.clear();
}

void tessellate_points(Mesh& m, int gpu) {
#ifdef V2M_HAS_CDT
    const uint32_t n = static_cast<uint32_t>(m.nodes.size() / 3);

    if (n < 4) {
        throw std::runtime_error("tessellate: fewer than 4 points");
    }

    ::TetMesh tm;
    bool done = false;
#ifdef V2M_HAS_OPENCL

    if (gpu > -2) {
        GdelStats gs;
        done = gdel_tetrahedrize(m.nodes.data(), n, tm, gs, gpu);
    }

#else
    (void)gpu;
#endif

    if (!done) {
        tm.init_vertices(m.nodes.data(), n);
        tm.tetrahedrize();
    }

    canonicalize_tets(tm);   // the same tets from either path, in every build

    const bool lab = m.node_labels.size() == n;
    m.tets.clear();
    m.tet_labels.clear();

    for (uint64_t t = 0; t < tm.numTets(); ++t) {
        if (tm.isGhost(t)) {
            continue;
        }

        const uint32_t* v = tm.getTetNodes(t * 4);
        int32_t best = 1;

        if (lab) {   // the most frequent node label (ties: the smallest)
            int bc = 0;

            for (int i = 0; i < 4; ++i) {
                int c = 0;

                for (int j = 0; j < 4; ++j) {
                    c += m.node_labels[v[j]] == m.node_labels[v[i]];
                }

                if (c > bc || (c == bc && m.node_labels[v[i]] < best)) {
                    bc = c;
                    best = m.node_labels[v[i]];
                }
            }
        }

        for (int k = 0; k < 4; ++k) {
            m.tets.push_back(static_cast<int32_t>(v[k]));
        }

        m.tet_labels.push_back(best);
    }

    m.tris.clear();
    m.tri_labels.clear();
#else
    (void)m;
    (void)gpu;
    throw std::runtime_error("tessellate: built without the Delaunay (V2M_USE_CDT=OFF)");
#endif
}

namespace {

double mean_edge(const Mesh& s) {
    double esum = 0;

    for (size_t t = 0; t < s.tris.size() / 3; ++t)
        for (int k = 0; k < 3; ++k) {
            const double* p = &s.nodes[3 * static_cast<size_t>(s.tris[3 * t + k])];
            const double* q = &s.nodes[3 * static_cast<size_t>(s.tris[3 * t + (k + 1) % 3])];
            esum += std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
        }

    return s.tris.empty() ? 0.0 : esum / static_cast<double>(s.tris.size());
}

double extent(const Mesh& s) {
    double ext = 0;

    for (int k = 0; k < 3; ++k) {
        double lo = 1e300, hi = -1e300;

        for (size_t v = 0; v < s.nodes.size() / 3; ++v) {
            lo = std::min(lo, s.nodes[3 * v + k]);
            hi = std::max(hi, s.nodes[3 * v + k]);
        }

        ext = std::max(ext, hi - lo);
    }

    return ext;
}

}  // namespace

double default_cdt_fill(const Mesh& surf, double hbase) {
    return hbase > 0 ? hbase : 1.5 * mean_edge(surf);
}

double run_cdt(const Mesh& surf_in, const PipelineOptions& o, double fill, int opt_rounds, Mesh& out, CdtStats& cs,
               OptStats& os) {
    // regions without inner / outer labels: exact ones from the surfaces' cells
    Mesh surf = surf_in;
    SurfLabelStats ls;
    normalize_surface_labels(surf, o.surf, ls);
    cs.labels = describe(ls);
    const size_t nx = ls.crossings ? ls.crossings : self_intersections(surf.nodes, surf.tris);

    if (nx > 0) {
        throw std::runtime_error("cdt: " + std::to_string(nx) + " pairs of triangles cross; the CDT needs a clean surface "
                                 "(repair it first: --mode repair, or v2mesh.repair)");
    }

    if (fill < 0) {
        fill = default_cdt_fill(surf, o.grid.hbase);
    }

    cdt_mesh(surf, out, cs, fill);

    if (o.opt && !out.tets.empty()) {
        OptParams op;
        op.q = o.q;
        op.refine = o.q;
        op.max_rounds = opt_rounds;
        op.verbose = o.relax.verbose;
        optimize_tets(out, op, os);
    }

    if (o.manifold && !out.tets.empty()) {
        ManifoldStats ms;
        make_manifold(out.tets, out.tet_labels, out.nodes, o.nest, ms);
        compact_nodes(out);
    }

    return fill;
}

void shapes_volume(const std::string& src, double voxel, bool clip, PipelineOptions& o, LabelVolume& lv,
                   ShapeScene& sc) {
    sc = load_shapes(src, clip);

    if (!(voxel > 0)) {
        const double ext = std::max(sc.hi[0] - sc.lo[0], std::max(sc.hi[1] - sc.lo[1], sc.hi[2] - sc.lo[2]));
        voxel = o.grid.hbase > 0 ? o.grid.hbase / 3.0 : ext / 160.0;
    }

    {   // the gap-closing radius (v2m_sdf_body.cl): facing surfaces closer than about
        // shape_gap / 2 elements merge -- a tangent contact is no sliver of zero thickness
        const double hb = o.grid.hbase > 0 ? o.grid.hbase : 3.0 * voxel;
        sc.prog[6 + static_cast<size_t>(sc.nlab)] = static_cast<float>(std::max(0.0, o.shape_gap) * hb);
        // the cull margin: beyond every range the fields are read over -- the phi
        // band (0.75 voxel), the gap radius (shape_gap elements), the curvature band
        // (2 voxels), the thickness filter (an element: the farthest) -- with room
        // (V2M_SDF_MARGIN: another multiple of the element size)
        const char* em = std::getenv("V2M_SDF_MARGIN");
        const double mf = em ? std::atof(em) : 1.25;
        sc.prog[7 + static_cast<size_t>(sc.nlab)] = static_cast<float>(std::max(mf * hb, std::max(o.shape_gap * hb, 3.0 * voxel)));
    }

    // each brick its own program, with only the objects near it (V2M_SDF_BRICKS=0: not)
    if (!(std::getenv("V2M_SDF_BRICKS") && std::atoi(std::getenv("V2M_SDF_BRICKS")) == 0)) {
        build_brick_programs(sc, voxel);
    }

    const Tpm tpm = rasterize_scene(sc, voxel);
    o.tpm.fields = true;       // interfaces at p_a = p_b: sub-voxel
    o.tpm.fill_holes = false;  // a cavity is real
    o.tpm.exterior.assign(1, 0);
    o.tpm.map.clear();
    o.tpm.sigma = 0.0f;
    o.thresholds.clear();
    apply_tpm(tpm, o.tpm, lv);
    lv.sdf = sc.prog;
    lv.sdf_feat = sc.feat;
}

double default_raster_voxel(const Mesh& surf, double hbase) {
    const double ext = extent(surf), em = mean_edge(surf);
    double v = hbase > 0 ? hbase / 3.0 : ext / 160.0;

    if (em > 0) {
        v = std::min(v, 0.5 * em);
    }

    return std::max(v, ext / 600.0);
}

void remesh_volume(const Mesh& surf_in, double voxel, PipelineOptions& o, LabelVolume& lv, RasterStats& rs) {
    if (!(voxel > 0)) {
        voxel = default_raster_voxel(surf_in, o.grid.hbase);
    }

    // regions without inner / outer labels: exact ones from the cells where the
    // surfaces do not cross; else the rasterizer's rules (o.surf.overlap)
    Mesh surf = surf_in;
    SurfLabelStats ls;
    normalize_surface_labels(surf, o.surf, ls);
    const Tpm tpm = rasterize_surfaces(surf, voxel, rs, o.surf);
    rs.labels = describe(ls);

    if (rs.flood) {   // voxel-exact cells: smoothed a voxel, for sub-voxel interfaces
        o.tpm.sigma = std::max(o.tpm.sigma, 1.0f);
    }

    o.tpm.fields = true;       // interfaces at p_a = p_b: the surfaces, sub-voxel
    o.tpm.fill_holes = false;  // a shell's cavity is real
    o.tpm.exterior.assign(1, 0);
    o.tpm.map.clear();
    o.thresholds.clear();
    apply_tpm(tpm, o.tpm, lv);
}

void add_faces(Mesh& m) {
    std::vector<int32_t> f;
    extract_faces(m.tets, m.tet_labels, m.nodes, f);
    m.tris.clear();
    m.tri_labels.clear();

    for (size_t i = 0; i + 4 < f.size(); i += 5) {
        m.tris.insert(m.tris.end(), f.begin() + static_cast<std::ptrdiff_t>(i), f.begin() + static_cast<std::ptrdiff_t>(i) + 3);
        m.tri_labels.push_back(f[i + 3]);
        m.tri_labels.push_back(f[i + 4]);
    }
}

void compact_nodes(Mesh& m) {
    const size_t nn = m.nodes.size() / 3;
    std::vector<int32_t> map(nn, -1);
    int32_t k = 0;

    for (int32_t v : m.tets)
        if (map[v] < 0) {
            map[v] = 0;
        }

    for (int32_t v : m.tris)
        if (map[v] < 0) {
            map[v] = 0;
        }

    for (size_t i = 0; i < nn; ++i)
        if (map[i] == 0) {
            map[i] = k++;
        }

    std::vector<double> P(static_cast<size_t>(k) * 3);
    std::vector<int32_t> lab, typ, par;

    for (size_t i = 0; i < nn; ++i)
        if (map[i] >= 0) {
            for (int c = 0; c < 3; ++c) {
                P[3 * static_cast<size_t>(map[i]) + c] = m.nodes[3 * i + c];
            }

            if (m.node_labels.size() == nn) {
                lab.push_back(m.node_labels[i]);
            }

            if (m.node_types.size() == nn) {
                typ.push_back(m.node_types[i]);
            }

            if (m.node_partners.size() == 3 * nn) {
                par.insert(par.end(), m.node_partners.begin() + 3 * static_cast<std::ptrdiff_t>(i),
                           m.node_partners.begin() + 3 * static_cast<std::ptrdiff_t>(i) + 3);
            }
        }

    for (int32_t& v : m.tets) {
        v = map[v];
    }

    for (int32_t& v : m.tris) {
        v = map[v];
    }

    m.nodes.swap(P);
    m.node_labels.swap(lab);
    m.node_types.swap(typ);
    m.node_partners.swap(par);
}

MeshReport check_mesh(const Mesh& m) {
    MeshReport r;
    r.nodes = m.nodes.size() / 3;
    r.tets = m.tets.size() / 4;
    r.tris = m.tris.size() / 3;
    std::vector<int32_t> tris = m.tris, tlab = m.tri_labels;

    if (r.tets > 0) {
        // quality (as the mesher reports it)
        std::vector<double> jl(r.tets), md(r.tets);
        size_t flat = 0;
        double vol = 0;
        int32_t maxlab = 0;

        for (int32_t l : m.tet_labels) {
            maxlab = std::max(maxlab, l);
        }

        r.label_vol.assign(static_cast<size_t>(std::max(maxlab, 0)) + 1, 0.0);

        for (size_t t = 0; t < r.tets; ++t) {
            const double* p[4];

            for (int k = 0; k < 4; ++k) {
                p[k] = &m.nodes[3 * static_cast<size_t>(m.tets[4 * t + k])];
            }

            double v;
            tet_quality(p, md[t], jl[t], v);
            v = std::fabs(v);
            vol += v;

            if (!m.tet_labels.empty() && m.tet_labels[t] >= 0) {
                r.label_vol[static_cast<size_t>(m.tet_labels[t])] += v;
            }

            if (md[t] < 10.0) {
                ++r.slivers10;
            }
        }

        // inverted = the minority orientation (either convention is fine as long as
        // it is one)
        size_t pos = 0, neg = 0;

        for (size_t t = 0; t < r.tets; ++t) {
            const double* p[4];

            for (int k = 0; k < 4; ++k) {
                p[k] = &m.nodes[3 * static_cast<size_t>(m.tets[4 * t + k])];
            }

            const int o = orient(p[0], p[1], p[2], p[3]);
            pos += o > 0;
            neg += o < 0;
            flat += o == 0;
        }

        r.inverted = std::min(pos, neg);
        r.flat = flat;
        r.volume = vol;
        std::vector<double> s = jl;
        std::sort(s.begin(), s.end());
        r.joe_liu_min = s.front();
        r.joe_liu_p5 = s[s.size() / 20];
        r.joe_liu_med = s[s.size() / 2];
        r.min_dihedral = *std::min_element(md.begin(), md.end());

        if (tris.empty()) {   // the region surfaces
            std::vector<int32_t> f;
            extract_faces(m.tets, m.tet_labels.empty() ? std::vector<int32_t>(r.tets, 1) : m.tet_labels, m.nodes, f);
            tris.clear();
            tlab.clear();

            for (size_t i = 0; i + 4 < f.size(); i += 5) {
                tris.insert(tris.end(), f.begin() + static_cast<std::ptrdiff_t>(i), f.begin() + static_cast<std::ptrdiff_t>(i) + 3);
                tlab.push_back(f[i + 3]);
                tlab.push_back(f[i + 4]);
            }
        }
    }

    // edges: incidence over all triangles, and per region
    const size_t nf = tris.size() / 3;
    std::vector<std::pair<uint64_t, int32_t>> ed;   // (edge key, region), region -1 = all
    ed.reserve(nf * 3 * (tlab.empty() ? 1 : 3));

    for (size_t t = 0; t < nf; ++t)
        for (int k = 0; k < 3; ++k) {
            uint64_t a = static_cast<uint32_t>(tris[3 * t + k]), b = static_cast<uint32_t>(tris[3 * t + (k + 1) % 3]);

            if (a > b) {
                std::swap(a, b);
            }

            const uint64_t key = (a << 32) | b;
            ed.emplace_back(key, -1);

            if (!tlab.empty()) {
                if (tlab[2 * t] != 0) {
                    ed.emplace_back(key, tlab[2 * t]);
                }

                if (tlab[2 * t + 1] != 0 && tlab[2 * t + 1] != tlab[2 * t]) {
                    ed.emplace_back(key, tlab[2 * t + 1]);
                }
            }
        }

    std::sort(ed.begin(), ed.end(), [](const std::pair<uint64_t, int32_t>& x, const std::pair<uint64_t, int32_t>& y) {
        return x.second != y.second ? x.second < y.second : x.first < y.first;
    });

    for (size_t i = 0; i < ed.size();) {
        size_t j = i;

        while (j < ed.size() && ed[j] == ed[i]) {
            ++j;
        }

        const size_t c = j - i;

        if (ed[i].second < 0) {
            r.open_edges += c == 1;
            r.junction_edges += c > 2;
        } else if (c % 2 == 1) {   // a region's closed boundary uses each edge an even number of times
            ++r.region_open_edges;
        }

        i = j;
    }

    r.self_intersections = self_intersections(m.nodes, tris, &r.crossing);
    return r;
}

void print_report(const MeshReport& r, const std::string& name) {
    V2M_FPRINTF(stderr, "[check] %s: %zu nodes, %zu tets, %zu triangles%s\n", name.c_str(), r.nodes, r.tets, r.tris,
               r.tets > 0 && r.tris == 0 ? " (region surfaces from the tets)" : "");

    if (r.tets > 0) {
        V2M_FPRINTF(stderr, "[check] tets: %zu inverted, %zu flat; min dihedral %.2f deg, slivers <10: %zu; Joe-Liu min %.3f "
                   "p5 %.3f median %.3f; volume %.6g\n", r.inverted, r.flat, r.min_dihedral, r.slivers10, r.joe_liu_min,
                   r.joe_liu_p5, r.joe_liu_med, r.volume);

        if (r.label_vol.size() > 1) {
            std::string s;

            for (size_t l = 0; l < r.label_vol.size(); ++l)
                if (r.label_vol[l] > 0) {
                    char b[64];
                    std::snprintf(b, sizeof(b), " %zu:%.6g", l, r.label_vol[l]);
                    s += b;
                }

            V2M_FPRINTF(stderr, "[check] volume per label:%s\n", s.c_str());
        }
    }

    V2M_FPRINTF(stderr, "[check] surface: %zu open edges, %zu junction edges, %zu region-boundary edges not closed, "
               "%zu self-intersections\n", r.open_edges, r.junction_edges, r.region_open_edges, r.self_intersections);

    for (const auto& p : r.crossing) {
        V2M_FPRINTF(stderr, "[check]   triangles %d and %d cross\n", p.first + 1, p.second + 1);
    }

    V2M_FPRINTF(stderr, "[check] %s\n", r.ok() ? "OK" : "PROBLEMS FOUND");
}

}  // namespace tn
