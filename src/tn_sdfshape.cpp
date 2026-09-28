// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_sdfshape.cpp -- see tn_sdfshape.h.

#include "tn_sdfshape.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace tn {

namespace sdf_host {

using std::fabs;
using std::fmax;
using std::fmin;
using std::sqrt;
#define TN_G
#include "opencl/tn_sdf_body.cl"
#undef TN_G

}  // namespace sdf_host

using namespace sdf_host;

namespace {

using ojson = nlohmann::ordered_json;
const double kInf = std::numeric_limits<double>::infinity();

// a CSG tree node
struct Node {
    int kind = 0;   // 0 primitive, TN_SDF_MAX union, TN_SDF_MIN intersection, -1 subtraction
    int type = 0;
    std::vector<float> par;
    std::vector<int> kids;
    std::array<double, 3> lo{ { -kInf, -kInf, -kInf } }, hi{ { kInf, kInf, kInf } };
};

struct Builder {
    std::vector<Node> nodes;
    std::vector<float> feat;   // (tn_sdfshape.h ShapeScene::feat)
    void corner(double x, double y, double z) {
        feat.insert(feat.end(), { 1.0f, float(x), float(y), float(z) });
    }
    void segment(const double* a, const double* b) {
        feat.insert(feat.end(), { 2.0f, float(a[0]), float(a[1]), float(a[2]), float(b[0]), float(b[1]), float(b[2]) });
    }
    void circle(const double* c, const double* n, double r) {
        if (r > 0) {
            feat.insert(feat.end(), { 3.0f, float(c[0]), float(c[1]), float(c[2]), float(n[0]), float(n[1]), float(n[2]), float(r) });
        }
    }
    std::map<std::string, int> named;           // "(name)" -> node
    std::map<std::string, ojson> named_json;    // not yet built
    std::set<std::string> referenced;

    int prim(int type, std::vector<float> par, std::array<double, 3> lo, std::array<double, 3> hi) {
        Node n;
        n.type = type;
        n.par = std::move(par);
        n.lo = lo;
        n.hi = hi;
        nodes.push_back(n);
        return static_cast<int>(nodes.size()) - 1;
    }
    int op(int kind, int a, int b) {
        Node n;
        n.kind = kind;
        n.kids = { a, b };
        const Node& A = nodes[static_cast<size_t>(a)], &B = nodes[static_cast<size_t>(b)];

        for (int k = 0; k < 3; ++k) {
            if (kind == TN_SDF_MAX) {
                n.lo[k] = std::min(A.lo[k], B.lo[k]);
                n.hi[k] = std::max(A.hi[k], B.hi[k]);
            } else if (kind == TN_SDF_MIN) {
                n.lo[k] = std::max(A.lo[k], B.lo[k]);
                n.hi[k] = std::min(A.hi[k], B.hi[k]);
            } else {   // a - b: within a
                n.lo[k] = A.lo[k];
                n.hi[k] = A.hi[k];
            }
        }

        nodes.push_back(n);
        return static_cast<int>(nodes.size()) - 1;
    }
    void emit(int i, std::vector<float>& code) const {
        const Node& n = nodes[static_cast<size_t>(i)];

        if (n.kind == 0) {
            code.push_back(TN_SDF_PRIM);
            code.push_back(static_cast<float>(n.type));
            code.insert(code.end(), n.par.begin(), n.par.end());
        } else {
            emit(n.kids[0], code);
            emit(n.kids[1], code);

            if (n.kind == -1) {
                code.push_back(TN_SDF_NEG);
                code.push_back(TN_SDF_MIN);
            } else {
                code.push_back(static_cast<float>(n.kind));
            }
        }
    }
};

// the key without its "(name)", and the name
std::string base_key(const std::string& k, std::string* name = nullptr) {
    const size_t p = k.find('(');

    if (p == std::string::npos || k.back() != ')') {
        if (name) {
            name->clear();
        }

        return k;
    }

    if (name) {
        *name = k.substr(p + 1, k.size() - p - 2);
    }

    return k.substr(0, p);
}

std::array<double, 3> vec3(const ojson& v, const char* what) {
    if (!v.is_array() || v.size() < 3) {
        throw std::runtime_error(std::string("shapes: ") + what + " wants [x, y, z]");
    }

    return { { v[0].get<double>(), v[1].get<double>(), v[2].get<double>() } };
}

double num(const ojson& o, const char* key, double def = std::numeric_limits<double>::quiet_NaN()) {
    if (o.is_object() && o.contains(key)) {
        const ojson& v = o[key];
        return v.is_array() ? v[0].get<double>() : v.get<double>();
    }

    if (std::isnan(def)) {
        throw std::runtime_error(std::string("shapes: missing \"") + key + "\"");
    }

    return def;
}

std::array<double, 3> unit(std::array<double, 3> n) {
    const double l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

    if (!(l > 0)) {
        throw std::runtime_error("shapes: a zero direction vector");
    }

    return { { n[0] / l, n[1] / l, n[2] / l } };
}

std::vector<float> fv(std::initializer_list<double> v) {
    std::vector<float> o;

    for (double x : v) {
        o.push_back(static_cast<float>(x));
    }

    return o;
}

struct Object {
    int node;
    int tag;
    std::string desc;
};

class Parser {
  public:
    Builder B;
    std::vector<Object> objs;
    std::array<double, 3> grid_hi{ { kInf, kInf, kInf } };   // MCX Grid (for infinite slabs)
    bool have_grid = false;

    // a shape / CSG construct -> a node (key: the construct's name, v: its body)
    int build(const std::string& key_full, const ojson& v, int* tag) {
        std::string name;
        const std::string k = base_key(key_full, &name);
        int n = -1;

        if (tag && v.is_object() && v.contains("Tag")) {
            *tag = v["Tag"].get<int>();
        }

        auto sphere = [&](std::array<double, 3> c, double r) {
            return B.prim(TN_SDF_SPHERE, fv({ c[0], c[1], c[2], r }), { { c[0] - r, c[1] - r, c[2] - r } },
            { { c[0] + r, c[1] + r, c[2] + r } });
        };
        auto box = [&](std::array<double, 3> a, std::array<double, 3> b) {
            std::array<double, 3> lo, hi;

            for (int i = 0; i < 3; ++i) {
                lo[i] = std::min(a[i], b[i]);
                hi[i] = std::max(a[i], b[i]);
            }

            // the 8 corners, 12 edges
            for (int c = 0; c < 8; ++c) {
                B.corner(c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1], c & 4 ? hi[2] : lo[2]);
            }

            for (int ax = 0; ax < 3; ++ax)
                for (int c = 0; c < 4; ++c) {
                    double p0[3], p1[3];
                    const int u = (ax + 1) % 3, w = (ax + 2) % 3;
                    p0[ax] = lo[ax];
                    p1[ax] = hi[ax];
                    p0[u] = p1[u] = c & 1 ? hi[u] : lo[u];
                    p0[w] = p1[w] = c & 2 ? hi[w] : lo[w];
                    B.segment(p0, p1);
                }

            return B.prim(TN_SDF_BOX, fv({ lo[0], lo[1], lo[2], hi[0], hi[1], hi[2] }), lo, hi);
        };
        auto rims = [&](std::array<double, 3> a, std::array<double, 3> b, double ra, double rb) {   // cylinder / cone
            const double ax[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
            const double L = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);

            if (!(L > 0)) {
                return;
            }

            const double n[3] = { ax[0] / L, ax[1] / L, ax[2] / L };
            B.circle(a.data(), n, ra);

            if (rb > 0) {
                B.circle(b.data(), n, rb);
            } else {
                B.corner(b[0], b[1], b[2]);   // a cone's tip
            }
        };
        auto seg_bounds = [&](std::array<double, 3> a, std::array<double, 3> b, double r, std::array<double, 3>& lo,
        std::array<double, 3>& hi) {
            for (int i = 0; i < 3; ++i) {
                lo[i] = std::min(a[i], b[i]) - r;
                hi[i] = std::max(a[i], b[i]) + r;
            }
        };
        auto cylinder = [&](std::array<double, 3> a, std::array<double, 3> b, double r) {
            std::array<double, 3> lo, hi;
            seg_bounds(a, b, r, lo, hi);
            rims(a, b, r, r);
            return B.prim(TN_SDF_CYL, fv({ a[0], a[1], a[2], b[0], b[1], b[2], r }), lo, hi);
        };
        auto frustum = [&](std::array<double, 3> a, std::array<double, 3> b, double ra, double rb) {
            std::array<double, 3> lo, hi;
            seg_bounds(a, b, std::max(ra, rb), lo, hi);
            rims(a, b, ra, rb);
            return B.prim(TN_SDF_CONE, fv({ a[0], a[1], a[2], b[0], b[1], b[2], ra, rb }), lo, hi);
        };
        auto slab = [&](std::array<double, 3> o, std::array<double, 3> nrm, double lo_, double hi_) {
            nrm = unit(nrm);
            std::array<double, 3> lo{ { -kInf, -kInf, -kInf } }, hi{ { kInf, kInf, kInf } };

            for (int i = 0; i < 3; ++i)   // axis-aligned: bounded along its axis
                if (std::fabs(nrm[i]) == 1.0) {
                    lo[i] = o[i] + std::min(nrm[i] * lo_, nrm[i] * hi_);
                    hi[i] = o[i] + std::max(nrm[i] * lo_, nrm[i] * hi_);
                }

            return B.prim(TN_SDF_SLAB, fv({ o[0], o[1], o[2], nrm[0], nrm[1], nrm[2], lo_, hi_ }), lo, hi);
        };
        auto plane = [&](std::array<double, 3> o, std::array<double, 3> nrm) {
            nrm = unit(nrm);
            return B.prim(TN_SDF_PLANE, fv({ o[0], o[1], o[2], nrm[0], nrm[1], nrm[2] }), { { -kInf, -kInf, -kInf } },
            { { kInf, kInf, kInf } });
        };

        if (k == "CSGUnion" || k == "CSGIntersect" || k == "CSGSubtract") {
            std::vector<int> ops;

            if (!v.is_array()) {
                throw std::runtime_error("shapes: " + k + " wants [obj1, obj2]");
            }

            for (const auto& e : v) {
                if (e.is_object() && e.size() == 1 && e.begin().key() == "_DataInfo_") {
                    continue;
                }

                ops.push_back(operand(e));
            }

            if (ops.size() < 2) {
                throw std::runtime_error("shapes: " + k + " wants two operands");
            }

            const int kind = k == "CSGUnion" ? TN_SDF_MAX : k == "CSGIntersect" ? TN_SDF_MIN : -1;
            n = ops[0];

            for (size_t i = 1; i < ops.size(); ++i) {   // (more than two: left to right)
                n = B.op(kind, n, ops[i]);
            }
        } else if (k == "CSGObject" || k == "MeshObject") {
            // [root] or {Tag: .., <construct>: ..}
            const ojson* root = nullptr;
            std::string rk;

            if (v.is_array()) {
                for (const auto& e : v) {
                    if (e.is_object() && e.contains("Tag") && tag) {
                        *tag = e["Tag"].get<int>();
                    }

                    if (e.is_object() || e.is_string()) {
                        if (!root && !(e.is_object() && e.size() == 1 && (e.contains("Tag") || e.contains("_DataInfo_")))) {
                            root = &e;
                        }
                    }
                }
            } else if (v.is_object()) {
                root = &v;
            }

            if (!root) {
                throw std::runtime_error("shapes: an empty " + k);
            }

            n = operand(*root);
        } else if (k == "Grid" || k == "ShapeGrid3") {   // MCX: the domain box [0, Size]
            const std::array<double, 3> s = vec3(v.contains("Size") ? v["Size"] : v["P"], "Grid Size");
            n = box({ { 0, 0, 0 } }, s);
            grid_hi = s;
            have_grid = true;
        } else if (k == "Box") {                          // MCX: O, Size
            const std::array<double, 3> o = vec3(v["O"], "Box O"), s = vec3(v["Size"], "Box Size");
            n = box(o, { { o[0] + s[0], o[1] + s[1], o[2] + s[2] } });
        } else if (k == "Subgrid") {                      // MCX: 1-based O, Size
            std::array<double, 3> o = vec3(v["O"], "Subgrid O");
            const std::array<double, 3> s = vec3(v["Size"], "Subgrid Size");

            for (auto& x : o) {
                x -= 1;
            }

            n = box(o, { { o[0] + s[0], o[1] + s[1], o[2] + s[2] } });
        } else if (k == "ShapeBox3") {
            n = box(vec3(v["O"], "ShapeBox3 O"), vec3(v["P"], "ShapeBox3 P"));
        } else if (k == "Sphere" || k == "ShapeSphere") {
            n = sphere(vec3(v["O"], "Sphere O"), num(v, "R"));
        } else if (k == "Cylinder") {                     // MCX: C0, C1, R
            n = cylinder(vec3(v["C0"], "Cylinder C0"), vec3(v["C1"], "Cylinder C1"), num(v, "R"));
        } else if (k == "ShapeCylinder") {
            n = cylinder(vec3(v["O"], "ShapeCylinder O"), vec3(v["P"], "ShapeCylinder P"), num(v, "R"));
        } else if (k == "ShapeCone") {                    // base O (radius R), tip P
            n = frustum(vec3(v["O"], "ShapeCone O"), vec3(v["P"], "ShapeCone P"), num(v, "R"), 0.0);
        } else if (k == "ShapeConeFrustum") {
            const ojson& r = v["R"];
            n = frustum(vec3(v["O"], "ShapeConeFrustum O"), vec3(v["P"], "ShapeConeFrustum P"), r[0].get<double>(),
                        r[1].get<double>());
        } else if (k == "ShapeSphereShell") {
            const std::array<double, 3> c = vec3(v["O"], "ShapeSphereShell O");
            const ojson& r = v["R"];
            const double r1 = r[0].get<double>(), r2 = r[1].get<double>();
            n = B.op(-1, sphere(c, std::max(r1, r2)), sphere(c, std::min(r1, r2)));
        } else if (k == "ShapeSphereSegment") {           // the sphere between heights h1, h2 along N
            const std::array<double, 3> c = vec3(v["O"], "ShapeSphereSegment O"), nn = vec3(v["N"], "ShapeSphereSegment N");
            const ojson& h = v["Height"];
            const double R = num(v, "R");
            n = B.op(TN_SDF_MIN, sphere(c, R), slab(c, nn, h[0].get<double>(), h[1].get<double>()));
            const std::array<double, 3> u = unit(nn);

            for (int e = 0; e < 2; ++e) {   // the rims at the two heights
                const double t = h[e].get<double>();

                if (std::fabs(t) < R) {
                    const double cc[3] = { c[0] + t* u[0], c[1] + t* u[1], c[2] + t* u[2] }, un[3] = { u[0], u[1], u[2] };
                    B.circle(cc, un, std::sqrt(R * R - t * t));
                }
            }
        } else if (k == "ShapeTorus") {
            const std::array<double, 3> c = vec3(v["O"], "ShapeTorus O"), nn = unit(vec3(v["N"], "ShapeTorus N"));
            const double R = num(v, "R"), r = num(v, "Rtube");
            n = B.prim(TN_SDF_TORUS, fv({ c[0], c[1], c[2], nn[0], nn[1], nn[2], R, r }),
            { { c[0] - R - r, c[1] - R - r, c[2] - R - r } }, { { c[0] + R + r, c[1] + R + r, c[2] + R + r } });
        } else if (k == "ShapeEllipsoid") {   // R = [rx, ry, rz], Angle = [theta (azimuth), phi (zenith)]
            const std::array<double, 3> c = vec3(v["O"], "ShapeEllipsoid O"), r = vec3(v["R"], "ShapeEllipsoid R");
            double th = 0, ph = 0;

            if (v.contains("Angle")) {
                const ojson& a = v["Angle"];
                th = a.is_array() ? a[0].get<double>() : a.get<double>();
                ph = a.is_array() && a.size() > 1 ? a[1].get<double>() : 0.0;
            }

            // local -> world: Rz(theta) Ry(phi); world -> local: its transpose
            const double ct = std::cos(th), st = std::sin(th), cp = std::cos(ph), sp = std::sin(ph);
            const double M[9] = { ct * cp, -st, ct * sp, st * cp, ct, st * sp, -sp, 0, cp };   // local -> world
            const double rm = std::max(r[0], std::max(r[1], r[2]));
            n = B.prim(TN_SDF_ELLIPSOID, fv({ c[0], c[1], c[2], r[0], r[1], r[2], M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8] }),
            { { c[0] - rm, c[1] - rm, c[2] - rm } }, { { c[0] + rm, c[1] + rm, c[2] + rm } });
        } else if (k == "ShapePlane3") {   // the half-space behind N
            n = plane(vec3(v["O"], "ShapePlane3 O"), vec3(v["N"], "ShapePlane3 N"));
        } else if (k == "Lens") {          // (rtmmc-vulkan): O, Dir, R (aperture), Front / Back {D apex, R signed radius}
            const std::array<double, 3> o = vec3(v["O"], "Lens O"), d = unit(vec3(v["Dir"], "Lens Dir"));
            const double R = num(v, "R"), Df = num(v["Front"], "D"), Rf = num(v["Front"], "R"), Db = num(v["Back"], "D"),
                         Rb = num(v["Back"], "R");
            auto along = [&](double t) {
                return std::array<double, 3> { { o[0] + t* d[0], o[1] + t* d[1], o[2] + t* d[2] } };
            };
            // front: below the cap z_f(r) = (Df - Rf) + sign(Rf) sqrt(Rf^2 - r^2)
            const double af = Df - Rf, ab = Rb - Db;
            const int sf = sphere(along(af), std::fabs(Rf)), sb = sphere(along(ab), std::fabs(Rb));
            const std::array<double, 3> nd = { { -d[0], -d[1], -d[2] } };
            const int below_f = plane(along(af), d), above_b = plane(along(ab), nd);
            const int front = Rf >= 0 ? B.op(TN_SDF_MAX, sf, below_f) : B.op(-1, below_f, sf);
            const int back = Rb >= 0 ? B.op(TN_SDF_MAX, sb, above_b) : B.op(-1, above_b, sb);
            const int ax = B.prim(TN_SDF_INFCYL, fv({ o[0], o[1], o[2], d[0], d[1], d[2], R }), { { -kInf, -kInf, -kInf } },
            { { kInf, kInf, kInf } });
            n = B.op(TN_SDF_MIN, B.op(TN_SDF_MIN, ax, front), back);
            {
                // the rims: the caps at r = R (z_f(R), z_b(R))
                const double dd[3] = { d[0], d[1], d[2] };

                if (std::fabs(Rf) > R) {
                    const std::array<double, 3> cf = along(af + (Rf >= 0 ? 1 : -1) * std::sqrt(Rf * Rf - R * R));
                    B.circle(cf.data(), dd, R);
                }

                if (std::fabs(Rb) > R) {
                    const std::array<double, 3> cb = along(ab - (Rb >= 0 ? 1 : -1) * std::sqrt(Rb * Rb - R * R));
                    B.circle(cb.data(), dd, R);
                }
            }
            Node& nn = B.nodes[static_cast<size_t>(n)];   // bounds: the aperture about O, the apexes

            for (int i = 0; i < 3; ++i) {
                const double e = R + std::max(Df, Db);
                nn.lo[i] = o[i] - e;
                nn.hi[i] = o[i] + e;
            }
        } else {
            throw std::runtime_error("shapes: unknown construct \"" + k + "\"");
        }

        if (!name.empty()) {
            B.named[name] = n;
        }

        return n;
    }

    // a CSG operand: an inline construct {Key: body} or a name
    int operand(const ojson& e) {
        if (e.is_string()) {
            const std::string ref = e.get<std::string>();
            std::string nm;
            const std::string b = base_key(ref, &nm);   // ("MeshObject(x)" or "x")

            if (nm.empty()) {
                nm = b;
            }

            B.referenced.insert(nm);
            auto it = B.named.find(nm);

            if (it != B.named.end()) {
                return it->second;
            }

            auto jt = B.named_json.find(nm);

            if (jt == B.named_json.end()) {
                throw std::runtime_error("shapes: no object named \"" + nm + "\"");
            }

            const ojson def = jt->second;
            B.named_json.erase(jt);
            return build(def.begin().key(), def.begin().value(), nullptr);
        }

        if (!e.is_object() || e.empty()) {
            throw std::runtime_error("shapes: a CSG operand must be a construct or a name");
        }

        for (auto it = e.begin(); it != e.end(); ++it) {
            if (it.key() != "Tag" && it.key() != "_DataInfo_") {
                std::string nm;
                base_key(it.key(), &nm);

                if (!nm.empty()) {
                    B.referenced.insert(nm);
                }

                return build(it.key(), it.value(), nullptr);
            }
        }

        throw std::runtime_error("shapes: an empty CSG operand");
    }

    // one entry of the object sequence
    void object(const std::string& key, const ojson& v, std::vector<std::pair<std::string, ojson>>& seq) {
        const std::string k = base_key(key);

        if (k == "Name" || k == "Origin" || k == "Clip" || k == "_DataInfo_" || k == "Session" || k == "Forward" ||
                k == "Domain" || k == "Optode") {
            return;
        }

        seq.emplace_back(key, v);
    }
};

// Label l's code (without its TN_SDF_END): its objects' regions (sequential
// overwriting, cut to object 1), their max. act: the objects present (per
// brick; the others are the culled constants, -margin for an object's own
// region, +margin for a later one it is cut by -- TN_SDF_MCONST); none: all,
// each guarded by its bounds (TN_SDF_BBOX) when cull.
std::vector<float> label_code(const ShapeScene& sc, int l, const std::vector<char>* act) {
    static const bool cull = !(std::getenv("TN_SDF_CULL") && std::atoi(std::getenv("TN_SDF_CULL")) == 0);
    const size_t no = sc.ocode.size();
    std::vector<float> c;
    auto on = [&](size_t i) {
        return !act || (*act)[i];
    };
    auto obj = [&](size_t i) {   // object i's field, guarded if whole-scene
        const std::array<double, 6>& b = sc.obox[i];
        bool finite = cull && !act;

        for (int k = 0; k < 6; ++k) {
            finite = finite && std::isfinite(b[static_cast<size_t>(k)]);
        }

        if (!finite) {
            c.insert(c.end(), sc.ocode[i].begin(), sc.ocode[i].end());
            return;
        }

        const size_t at = c.size();
        c.insert(c.end(), { TN_SDF_BBOX, float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]), 0.0f, -1.0f });
        c.insert(c.end(), sc.ocode[i].begin(), sc.ocode[i].end());
        c[at + 7] = static_cast<float>(c.size() - at - 9);
    };
    auto region = [&](size_t i) {   // s'_i = min(s_i, s_1, -s_j for later j)
        const size_t at = c.size();
        c.insert(c.end(), sc.ocode[i].begin(), sc.ocode[i].end());   // (the region as a whole is guarded below)

        if (sc.clip && i > 0) {
            c.insert(c.end(), sc.ocode[0].begin(), sc.ocode[0].end());
            c.push_back(TN_SDF_MIN);
        }

        bool culled = false;

        for (size_t j = i + 1; j < no; ++j) {
            if (!on(j)) {
                culled = true;
                continue;
            }

            obj(j);
            c.push_back(TN_SDF_NEG);
            c.push_back(TN_SDF_MIN);
        }

        if (culled) {   // (a later object far off: its -s_j >= +margin)
            c.insert(c.end(), { TN_SDF_MCONST, 1.0f, TN_SDF_MIN });
        }

        if (!act && cull) {   // the whole region guarded by object i's bounds
            const std::array<double, 6>& b = sc.obox[i];
            bool finite = true;

            for (int k = 0; k < 6; ++k) {
                finite = finite && std::isfinite(b[static_cast<size_t>(k)]);
            }

            if (finite) {
                const std::vector<float> body(c.begin() + static_cast<std::ptrdiff_t>(at), c.end());
                c.resize(at);
                c.insert(c.end(), { TN_SDF_BBOX, float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5]),
                                    static_cast<float>(body.size()), -1.0f });
                c.insert(c.end(), body.begin(), body.end());
            }
        }
    };
    // the terms: the regions of l's objects (and, for the exterior, the outside)
    int terms = 0;
    bool culled = false;

    if (l == 0) {   // outside object 1 (cut) or outside every object
        if (sc.clip) {
            c.insert(c.end(), sc.ocode[0].begin(), sc.ocode[0].end());
        } else {
            int n = 0;

            for (size_t i = 0; i < no; ++i) {
                if (!on(i)) {
                    culled = true;
                    continue;
                }

                obj(i);

                if (n++ > 0) {
                    c.push_back(TN_SDF_MAX);
                }
            }

            if (n == 0) {
                c.insert(c.end(), { TN_SDF_MCONST, -1.0f });
            } else if (culled) {
                c.insert(c.end(), { TN_SDF_MCONST, -1.0f, TN_SDF_MAX });
            }

            culled = false;
        }

        c.push_back(TN_SDF_NEG);
        terms = 1;
    }

    bool any = l == 0;

    for (size_t i = 0; i < no; ++i) {
        if (sc.otag[i] != l) {
            continue;
        }

        any = true;

        if (!on(i)) {
            culled = true;
            continue;
        }

        region(i);

        if (terms++ > 0) {
            c.push_back(TN_SDF_MAX);
        }
    }

    if (!any) {
        return { TN_SDF_CONST, -1e30f };
    }

    if (culled) {   // (an object of l far off: its region's field <= -margin)
        c.insert(c.end(), { TN_SDF_MCONST, -1.0f });

        if (terms++ > 0) {
            c.push_back(TN_SDF_MAX);
        }
    }

    return c;
}

}  // namespace

bool is_shape_json_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);

    if (!f) {
        return false;
    }

    std::string head(4096, '\0');
    f.read(&head[0], static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<size_t>(f.gcount()));
    return head.find("\"Shapes\"") != std::string::npos || head.find("\"Shape") != std::string::npos ||
           head.find("\"CSG") != std::string::npos;
}

ShapeScene load_shapes(const std::string& path_or_text, bool clip_default) {
    std::string text = path_or_text;
    const size_t f0 = text.find_first_not_of(" \t\r\n");

    if (f0 == std::string::npos || (text[f0] != '{' && text[f0] != '[')) {
        std::ifstream f(path_or_text, std::ios::binary);

        if (!f) {
            throw std::runtime_error("shapes: cannot open " + path_or_text);
        }

        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
    }

    ojson root;

    try {
        root = ojson::parse(text);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("shapes: not valid JSON (") + e.what() + ")");
    }

    ShapeScene sc;
    sc.clip = clip_default;

    if (root.is_object() && root.contains("Clip")) {
        sc.clip = root["Clip"].get<bool>();
    }

    const ojson* list = &root;

    if (root.is_object() && root.contains("Shapes")) {
        list = &root["Shapes"];
    }

    // the sequence: array entries (each {Key: body}, keys in order), or the keys of an object
    std::vector<std::pair<std::string, ojson>> seq;
    Parser P;

    if (list->is_array()) {
        for (const auto& e : *list)
            if (e.is_object())
                for (auto it = e.begin(); it != e.end(); ++it) {
                    P.object(it.key(), it.value(), seq);
                }
    } else if (list->is_object()) {
        for (auto it = list->begin(); it != list->end(); ++it) {
            P.object(it.key(), it.value(), seq);
        }
    } else {
        throw std::runtime_error("shapes: \"Shapes\" must be an array or an object");
    }

    // named constructs first (CSG may refer to any of them); those referenced
    // are building blocks, not objects of their own
    for (const auto& kv : seq) {
        std::string nm;
        base_key(kv.first, &nm);

        if (!nm.empty()) {
            ojson one = ojson::object();
            one[kv.first] = kv.second;
            P.B.named_json[nm] = one;
        }
    }

    // pass 1: find the references (build CSG bodies into a scratch parser)
    {
        Parser scan;
        scan.B.named_json = P.B.named_json;

        for (const auto& kv : seq) {
            const std::string k = base_key(kv.first);

            if (k.compare(0, 3, "CSG") == 0 || k == "MeshObject") {
                try {
                    int t = 1;
                    scan.build(kv.first, kv.second, &t);
                } catch (...) {
                }
            }
        }

        P.B.referenced = scan.B.referenced;
    }

    int maxtag = 0;

    for (const auto& kv : seq) {
        std::string nm;
        const std::string k = base_key(kv.first, &nm);

        if (!nm.empty() && P.B.referenced.count(nm) && k != "CSGObject") {
            continue;   // a building block
        }

        if (k == "XSlabs" || k == "YSlabs" || k == "ZSlabs" || k == "XLayers" || k == "YLayers" || k == "ZLayers") {
            const int dir = k[0] - 'X';
            std::array<double, 3> nrm{ { 0, 0, 0 } }, o{ { 0, 0, 0 } };
            nrm[static_cast<size_t>(dir)] = 1;
            auto add = [&](double lo, double hi, int tag) {
                Builder& B = P.B;
                Node n;
                n.type = TN_SDF_SLAB;
                n.par = fv({ o[0], o[1], o[2], nrm[0], nrm[1], nrm[2], lo, hi });
                n.lo[static_cast<size_t>(dir)] = lo;
                n.hi[static_cast<size_t>(dir)] = hi;
                B.nodes.push_back(n);
                P.objs.push_back({ static_cast<int>(B.nodes.size()) - 1, tag, k + " [" + std::to_string(lo) + ", " + std::to_string(hi) + "]" });
                maxtag = std::max(maxtag, tag);
            };

            if (k[1] == 'S') {   // Slabs: {Bound: [[lo, hi], ..], Tag}
                const int tag = kv.second.value("Tag", 1);

                for (const auto& b : kv.second["Bound"]) {
                    add(b[0].get<double>(), b[1].get<double>(), tag);
                }
            } else {             // Layers: [[lo, hi, tag], ..] (1-based, inclusive)
                for (const auto& b : kv.second) {
                    add(b[0].get<double>() - 1, b[1].get<double>(), b[2].get<int>());
                }
            }

            continue;
        }

        int tag = 1;
        const int n = P.build(kv.first, kv.second, &tag);
        P.objs.push_back({ n, tag, kv.first });
        maxtag = std::max(maxtag, tag);
    }

    if (P.objs.empty()) {
        throw std::runtime_error("shapes: no shape objects");
    }

    // the domain: object 1's bounds (clipped), else all the finite ones
    std::array<double, 3> lo{ { kInf, kInf, kInf } }, hi{ { -kInf, -kInf, -kInf } };

    for (size_t i = 0; i < P.objs.size(); ++i) {
        if (sc.clip && i > 0) {
            break;
        }

        const Node& n = P.B.nodes[static_cast<size_t>(P.objs[i].node)];

        for (int k = 0; k < 3; ++k)
            if (std::isfinite(n.lo[k]) && std::isfinite(n.hi[k])) {
                lo[k] = std::min(lo[k], n.lo[k]);
                hi[k] = std::max(hi[k], n.hi[k]);
            }
    }

    for (int k = 0; k < 3; ++k)
        if (!(lo[k] <= hi[k])) {
            throw std::runtime_error(sc.clip ? "shapes: the first object is unbounded (it clips all others)"
                                     : "shapes: the objects are unbounded");
        }

    sc.lo = lo;
    sc.hi = hi;
    sc.nlab = maxtag + 1;

    // the objects' code and bounds, kept for the per-brick programs
    sc.ocode.clear();
    sc.obox.clear();
    sc.otag.clear();

    for (const Object& ob : P.objs) {
        std::vector<float> c;
        P.B.emit(ob.node, c);
        const Node& n = P.B.nodes[static_cast<size_t>(ob.node)];
        sc.ocode.push_back(c);
        sc.obox.push_back({ { n.lo[0], n.lo[1], n.lo[2], n.hi[0], n.hi[1], n.hi[2] } });
        sc.otag.push_back(ob.tag);
    }

    // the label programs: the whole scene's, objects guarded by their bounds
    sc.prog.assign(9 + static_cast<size_t>(sc.nlab), 0.0f);   // (+ the blend radii and cull margin: shapes_volume; [8+N] the brick table)
    sc.prog[7 + static_cast<size_t>(sc.nlab)] = 1e30f;          // (no culling until the margin is set)
    sc.prog[0] = static_cast<float>(sc.nlab);
    sc.prog[1] = 1.0f;

    for (int l = 0; l < sc.nlab; ++l) {
        const std::vector<float> c = label_code(sc, l, nullptr);
        sc.prog[5 + static_cast<size_t>(l)] = static_cast<float>(sc.prog.size());
        sc.prog.insert(sc.prog.end(), c.begin(), c.end());
        sc.prog.push_back(TN_SDF_END);
    }

    // the evaluator's fixed stack (TN_SDF_STACK): each label's deepest point
    for (int l = 0; l < sc.nlab; ++l) {
        int sp = 0, deep = 0;

        for (size_t pc = static_cast<size_t>(sc.prog[5 + static_cast<size_t>(l)]); pc < sc.prog.size();) {
            const int op = static_cast<int>(sc.prog[pc]);

            if (op == TN_SDF_END) {
                break;
            }

            if (op == TN_SDF_PRIM) {
                ++sp;
                pc += 2 + static_cast<size_t>(tn_sdf_nparam(static_cast<int>(sc.prog[pc + 1])));
            } else if (op == TN_SDF_CONST || op == TN_SDF_MCONST) {
                ++sp;
                pc += 2;
            } else if (op == TN_SDF_BBOX) {
                pc += 9;   // (the body; skipped, it pushes the same one value)
            } else {
                sp -= op == TN_SDF_MAX || op == TN_SDF_MIN ? 1 : 0;
                pc += 1;
            }

            deep = std::max(deep, sp);
        }

        if (deep > TN_SDF_STACK) {
            throw std::runtime_error("shapes: a CSG tree too deep for the evaluator (" + std::to_string(deep) + " > " +
                                     std::to_string(TN_SDF_STACK) + " operands at once); nest it less");
        }
    }

    if (sc.prog.size() > (1u << 24)) {
        throw std::runtime_error("shapes: the compiled program is too large");
    }

    for (const auto& ob : P.objs) {
        sc.objects.push_back(ob.desc + " -> label " + std::to_string(ob.tag));
    }

    sc.feat = P.B.feat;

    return sc;
}

float sdf_eval(const std::vector<float>& prog, int l, const float* p) {
    return tn_sdf_eval(prog.data(), l, p[0], p[1], p[2], nullptr);
}

double scene_sdf(const ShapeScene& sc, int l, const double* p) {
    // (the program's origin: grid mm = world - origin)
    return tn_sdf_eval(sc.prog.data(), l, static_cast<float>(p[0] - sc.prog[2]), static_cast<float>(p[1] - sc.prog[3]),
                       static_cast<float>(p[2] - sc.prog[4]), nullptr);
}

namespace {

// the raster over the domain (+ 4 voxels): its size and origin (world)
void raster_frame(const ShapeScene& sc, double voxel, int* n, double* o) {
    const int margin = 4;

    for (int a = 0; a < 3; ++a) {
        n[a] = static_cast<int>(std::ceil((sc.hi[static_cast<size_t>(a)] - sc.lo[static_cast<size_t>(a)]) / voxel)) + 1 + 2 * margin;
        o[a] = sc.lo[static_cast<size_t>(a)] - margin * voxel;
    }
}

}  // namespace

size_t build_brick_programs(ShapeScene& sc, double voxel) {
    const int N = sc.nlab;
    int n[3];
    double o[3];
    raster_frame(sc, voxel, n, o);
    const int nb[3] = { (n[0] + 7) / 8, (n[1] + 7) / 8, (n[2] + 7) / 8 };
    const double m = sc.prog[7 + static_cast<size_t>(N)];
    const size_t no = sc.ocode.size();
    std::map<std::string, float> made;   // active set -> its block
    std::vector<float> blockof(static_cast<size_t>(nb[0]) * nb[1] * nb[2]);
    std::vector<float> prog = sc.prog;

    for (int bk = 0; bk < nb[2]; ++bk)
        for (int bj = 0; bj < nb[1]; ++bj)
            for (int bi = 0; bi < nb[0]; ++bi) {
                // the brick (world), grown by the cull margin; the objects reaching it
                const int b3[3] = { bi, bj, bk };
                std::string key(no, '\0');
                std::vector<char> act(no, 0);

                for (size_t i = 0; i < no; ++i) {
                    bool hit = true;

                    for (int a = 0; a < 3 && hit; ++a) {
                        const double lo = o[a] + (8.0 * b3[a] - 0.5) * voxel - m, hi = o[a] + (8.0 * b3[a] + 7.5) * voxel + m;
                        hit = sc.obox[i][static_cast<size_t>(a) + 3] >= lo && sc.obox[i][static_cast<size_t>(a)] <= hi;
                    }

                    act[i] = hit;
                    key[i] = hit ? '1' : '0';
                }

                auto it = made.find(key);

                if (it == made.end()) {   // a new program: N code offsets, then the codes
                    const size_t blk = prog.size();
                    prog.resize(blk + static_cast<size_t>(N), 0.0f);

                    for (int l = 0; l < N; ++l) {
                        const std::vector<float> c = label_code(sc, l, &act);
                        prog[blk + static_cast<size_t>(l)] = static_cast<float>(prog.size());
                        prog.insert(prog.end(), c.begin(), c.end());
                        prog.push_back(TN_SDF_END);
                    }

                    it = made.emplace(key, static_cast<float>(blk)).first;
                }

                blockof[static_cast<size_t>(bi) + static_cast<size_t>(nb[0]) * (bj + static_cast<size_t>(nb[1]) * bk)] = it->second;
            }

    const size_t T = prog.size();
    prog.insert(prog.end(), { float(nb[0]), float(nb[1]), float(nb[2]), float(voxel) });
    prog.insert(prog.end(), blockof.begin(), blockof.end());

    if (prog.size() >= (1u << 24)) {   // (offsets are floats: exact below 2^24) -- keep the scene's program
        return 0;
    }

    prog[8 + static_cast<size_t>(N)] = static_cast<float>(T);

    for (int a = 0; a < 3; ++a) {
        prog[2 + static_cast<size_t>(a)] = static_cast<float>(o[a]);
    }

    sc.prog.swap(prog);
    sc.brick_programs = made.size();
    return made.size();
}

Tpm rasterize_scene(ShapeScene& sc, double voxel) {
    if (!(voxel > 0)) {
        throw std::runtime_error("shapes: the raster voxel must be > 0");
    }

    Tpm t;
    int n3[3];
    double o[3];
    raster_frame(sc, voxel, n3, o);
    t.nx = n3[0];
    t.ny = n3[1];
    t.nz = n3[2];

    if (static_cast<double>(t.nx) * t.ny * t.nz * sc.nlab > 2.0e9) {
        throw std::runtime_error("shapes: the raster would be too large; use a larger --raster-voxel");
    }

    t.voxelsize = { { voxel, voxel, voxel } };
    t.affine = { { voxel, 0, 0, o[0], 0, voxel, 0, o[1], 0, 0, voxel, o[2], 0, 0, 0, 1 } };
    // the program: the mesher's grid mm (voxel i at i * voxel) + origin = world
    sc.prog[1] = static_cast<float>(1.5 * voxel);
    sc.prog[2] = static_cast<float>(o[0]);
    sc.prog[3] = static_cast<float>(o[1]);
    sc.prog[4] = static_cast<float>(o[2]);
    {
        // the min / max blend radius: 0.1 w (0.15 voxel). Exact CSG makes the fields
        // only C0 on its creases -- T-junctions where one surface runs through
        // another -- and the junction repairs could not settle there; blended,
        // they do, the surfaces moving only within 0.15 voxel of a crease
        // (TN_SDF_BLEND: another fraction of w, 0 = exact)
        const char* e = std::getenv("TN_SDF_BLEND");
        sc.prog[5 + static_cast<size_t>(sc.nlab)] = static_cast<float>((e ? std::atof(e) : 0.1) * sc.prog[1]);
    }
    const size_t nv = t.nv();
    t.C = sc.nlab;
    t.p.assign(static_cast<size_t>(t.C) * nv, 0.0f);
    t.names.assign(static_cast<size_t>(t.C), "");
    t.names[0] = "background";
    const float w = sc.prog[1];
    const float* prog = sc.prog.data();
    #pragma omp parallel for schedule(static)

    for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
        const int i = static_cast<int>(v % t.nx), j = static_cast<int>((v / t.nx) % t.ny),
                  k = static_cast<int>(v / (static_cast<int64_t>(t.nx) * t.ny));
        const float px = static_cast<float>(i * voxel), py = static_cast<float>(j * voxel), pz = static_cast<float>(k * voxel);

        for (int l = 0; l < t.C; ++l) {
            const float s = tn_sdf_eval(prog, l, px, py, pz, nullptr);
            t.p[static_cast<size_t>(l) * nv + static_cast<size_t>(v)] = std::min(1.0f, std::max(0.0f, 0.5f + s / w));
        }
    }

    return t;
}

std::vector<float> sdf_feature_points(const std::vector<float>& prog, const std::vector<float>& feat, float eps) {
    const int N = static_cast<int>(prog[0]);
    const float* P = prog.data();
    auto mask = [&](const float* x) {
        static const float dir[7][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
        uint64_t m = 0;

        for (int s = 0; s < 7; ++s) {
            int best = 0;
            float bv = -1e30f;

            for (int l = 0; l < N; ++l) {
                const float v = tn_sdf_eval(P, l, x[0] + eps * dir[s][0], x[1] + eps * dir[s][1], x[2] + eps * dir[s][2], nullptr);

                if (v > bv) {
                    bv = v;
                    best = l;
                }
            }

            m |= best < 64 ? uint64_t(1) << best : 0;
        }

        return m;
    };
    auto pop = [](uint64_t m) {
        int n = 0;

        for (; m; m &= m - 1) {
            ++n;
        }

        return n;
    };
    std::vector<float> out;

    for (size_t k = 0; k < feat.size();) {
        const int type = static_cast<int>(feat[k]);
        const float* q = &feat[k + 1];
        k += type == 1 ? 4 : type == 2 ? 7 : 8;

        if (type == 1) {
            if (pop(mask(q)) >= 3) {
                out.insert(out.end(), q, q + 3);
            }

            continue;
        }

        float u[3] = { 0, 0, 0 }, w[3] = { 0, 0, 0 };

        if (type == 3) {
            const float* n = q + 3;
            const float a0 = std::fabs(n[0]) < 0.9f ? 1.0f : 0.0f, a1 = a0 > 0 ? 0.0f : 1.0f;
            u[0] = -n[2] * a1;
            u[1] = n[2] * a0;
            u[2] = n[0] * a1 - n[1] * a0;
            const float lu = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

            for (float& x : u) {
                x /= lu;
            }

            w[0] = n[1] * u[2] - n[2] * u[1];
            w[1] = n[2] * u[0] - n[0] * u[2];
            w[2] = n[0] * u[1] - n[1] * u[0];
        }

        auto at = [&](float t, float* o) {
            for (int a = 0; a < 3; ++a) {
                if (type == 2) {
                    o[a] = q[a] + t * (q[3 + a] - q[a]);
                } else {
                    const float th = 6.28318531f * t;
                    o[a] = q[a] + q[6] * (std::cos(th) * u[a] + std::sin(th) * w[a]);
                }
            }
        };
        const int ns = 128;
        float x[3];
        at(0.0f, x);
        uint64_t prev = mask(x);

        for (int s = 1; s <= ns; ++s) {
            at(static_cast<float>(s) / ns, x);
            const uint64_t m = mask(x);

            if (m != prev) {
                out.insert(out.end(), x, x + 3);
            }

            prev = m;
        }
    }

    return out;
}

std::vector<float> sdf_thickness(const std::vector<float>& prog, const uint16_t* L, int nx, int ny, int nz, double voxel, int R) {
    const size_t nv = static_cast<size_t>(nx) * ny * nz;
    const int N = static_cast<int>(prog[0]);
    std::vector<float> out(nv, 1e30f), v(nv), w(nv);
    const float* P = prog.data();
    const float ninf = -1e30f;

    for (int l = 1; l < N; ++l) {
        bool any = false;
        #pragma omp parallel for schedule(static) reduction(|| : any)

        for (int64_t q = 0; q < static_cast<int64_t>(nv); ++q) {
            if (L[q] != l) {
                v[static_cast<size_t>(q)] = ninf;
                continue;
            }

            any = true;
            const int i = static_cast<int>(q % nx), j = static_cast<int>((q / nx) % ny), k = static_cast<int>(q / (static_cast<int64_t>(nx) * ny));
            v[static_cast<size_t>(q)] = tn_sdf_eval(P, l, static_cast<float>(i * voxel), static_cast<float>(j * voxel),
                                                    static_cast<float>(k * voxel), nullptr);
        }

        if (!any) {
            continue;
        }

        // the max over the (2R+1)^3 cube, separably
        const int64_t st[3] = { 1, nx, static_cast<int64_t>(nx)* ny };
        const int n3[3] = { nx, ny, nz };

        for (int ax = 0; ax < 3; ++ax) {
            #pragma omp parallel for schedule(static)

            for (int64_t q = 0; q < static_cast<int64_t>(nv); ++q) {
                const int c = static_cast<int>((q / st[ax]) % n3[ax]);
                float m = v[static_cast<size_t>(q)];

                for (int d = -R; d <= R; ++d)
                    if (d && c + d >= 0 && c + d < n3[ax]) {
                        m = std::max(m, v[static_cast<size_t>(q + d * st[ax])]);
                    }

                w[static_cast<size_t>(q)] = m;
            }

            v.swap(w);
        }

        #pragma omp parallel for schedule(static)

        for (int64_t q = 0; q < static_cast<int64_t>(nv); ++q)
            if (L[q] == l && v[static_cast<size_t>(q)] > 0.0f) {
                out[static_cast<size_t>(q)] = static_cast<float>(2.0 * v[static_cast<size_t>(q)] / voxel);
            }
    }

    return out;
}

std::vector<float> sdf_curvature(const std::vector<float>& prog, int nx, int ny, int nz, double voxel, double band) {
    // the distinct primitives of the program
    std::vector<size_t> at;
    std::set<std::vector<float>> seen;
    const int N = static_cast<int>(prog[0]);

    for (int l = 0; l < N; ++l)
        for (size_t pc = static_cast<size_t>(prog[5 + static_cast<size_t>(l)]); pc < prog.size();) {
            const int op = static_cast<int>(prog[pc]);

            if (op == TN_SDF_END) {
                break;
            }

            if (op == TN_SDF_PRIM) {
                const int type = static_cast<int>(prog[pc + 1]);
                const size_t np = static_cast<size_t>(tn_sdf_nparam(type));
                std::vector<float> key(prog.begin() + static_cast<std::ptrdiff_t>(pc + 1), prog.begin() + static_cast<std::ptrdiff_t>(pc + 2 + np));

                if (seen.insert(key).second) {
                    at.push_back(pc + 1);
                }

                pc += 2 + np;
            } else {
                pc += op == TN_SDF_CONST || op == TN_SDF_MCONST ? 2 : op == TN_SDF_BBOX ? 9 : 1;   // (a guard: into its body)
            }
        }

    const size_t nv = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> kv(nv, 0.0f);
    const float* P = prog.data();
    #pragma omp parallel for schedule(static)

    for (int64_t v = 0; v < static_cast<int64_t>(nv); ++v) {
        const int i = static_cast<int>(v % nx), j = static_cast<int>((v / nx) % ny), k = static_cast<int>(v / (static_cast<int64_t>(nx) * ny));
        const float x = static_cast<float>(i * voxel) + P[2], y = static_cast<float>(j * voxel) + P[3], z = static_cast<float>(k * voxel) + P[4];
        float kap = 0.0f;

        for (size_t a : at) {
            const int type = static_cast<int>(P[a]);
            const float* q = P + a + 1;
            const float s = tn_sdf_prim(type, q, x, y, z);

            if (std::fabs(s) > band) {
                continue;
            }

            float c = 0.0f;

            if (type == TN_SDF_SPHERE) {
                c = 1.0f / q[3];
            } else if (type == TN_SDF_INFCYL || type == TN_SDF_TORUS) {
                c = 1.0f / q[type == TN_SDF_TORUS ? 7 : 6];
            } else if (type == TN_SDF_CYL || type == TN_SDF_CONE) {   // the side (not the caps)
                const float bx = q[3] - q[0], by = q[4] - q[1], bz = q[5] - q[2];
                const float L2 = bx * bx + by * by + bz * bz;
                const float t = ((x - q[0]) * bx + (y - q[1]) * by + (z - q[2]) * bz) / L2;
                const float rx = x - q[0] - t * bx, ry = y - q[1] - t * by, rz = z - q[2] - t * bz;
                const float rl = std::sqrt(rx * rx + ry * ry + rz * rz);
                const float tc = std::min(1.0f, std::max(0.0f, t));
                const float r = type == TN_SDF_CYL ? q[6] : q[6] + (q[7] - q[6]) * tc;

                if (std::fabs(rl - r) <= band) {
                    const float sl = type == TN_SDF_CYL ? 0.0f : (q[7] - q[6]) / std::sqrt(L2);
                    c = 1.0f / (std::max(r, static_cast<float>(voxel)) * std::sqrt(1.0f + sl * sl));
                }
            } else if (type == TN_SDF_ELLIPSOID) {   // (the largest: a / b^2)
                const float a3 = std::max(q[3], std::max(q[4], q[5])), b3 = std::min(q[3], std::min(q[4], q[5]));
                c = a3 / (b3 * b3);
            }

            kap = std::max(kap, c);
        }

        kv[static_cast<size_t>(v)] = kap;
    }

    return kv;
}

}  // namespace tn
