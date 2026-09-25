// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_seed_body.cl -- stage 3 (graded hex seeding) and the point queries shared
// with the particle stages: the label at a point, the interface function
// psi_ab = phi_a - phi_b and its gradient, and projection onto one interface or
// onto a junction curve (two interfaces). Needs tn_grid_body.cl first. Host and
// device, like the other bodies.
//
// Node types: TN_INTERIOR (free in its label), TN_INTERFACE (on the a|b
// interface, a = own label, b = partner), TN_JUNCTION (on the a|b and a|c
// interfaces at once), TN_CORNER (where 4+ labels meet; fixed).

#define TN_INTERIOR  0
#define TN_INTERFACE 1
#define TN_JUNCTION  2
#define TN_CORNER    3
#define TN_NOLAB     0xFFFF

// the field arguments every point query needs
#define TN_FIELD_ARGS TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab, \
    TN_G const int* bl_slot, TN_G const float* phi
#define TN_FIELD d, L, bl_cnt, bl_lab, bl_slot, phi

// labels that can appear at point p (those of its brick), up to TN_BL
inline int tn_labels_near(TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab,
                          float px, float py, float pz, int* lab) {
    const int i = (int)floor(px / d.vx + 0.5f), j = (int)floor(py / d.vy + 0.5f), k = (int)floor(pz / d.vz + 0.5f);

    if (i < 0 || j < 0 || k < 0 || i >= d.nx || j >= d.ny || k >= d.nz) {
        lab[0] = 0;
        return 1;
    }

    const int b = tn_brick_of(d, i, j, k);
    const int n = bl_cnt[b] < TN_BL ? bl_cnt[b] : TN_BL;

    for (int s = 0; s < n; ++s) {
        lab[s] = bl_lab[b * TN_BL + s];
    }

    return n;
}

// The label owning point p: argmax of phi_l (smooth mode), or the voxel label
// (voxel mode). Also returns the runner-up label and the margin phi_1 - phi_2.
inline int tn_label_of(TN_FIELD_ARGS, int voxmode, float px, float py, float pz, int* second, float* margin) {
    const int i = (int)floor(px / d.vx + 0.5f), j = (int)floor(py / d.vy + 0.5f), k = (int)floor(pz / d.vz + 0.5f);
    int lab[TN_BL];
    const int n = tn_labels_near(d, L, bl_cnt, bl_lab, px, py, pz, lab);
    int best = lab[0], sec = TN_NOLAB;
    float pb = -1.0f, ps = -1.0f;

    if (n == 1) {
        *second = TN_NOLAB;
        *margin = 1.0f;
        return best;
    }

    for (int s = 0; s < n; ++s) {
        const float v = tn_phi_at(TN_FIELD, lab[s], px, py, pz);

        if (v > pb) {
            ps = pb;
            sec = best;
            pb = v;
            best = lab[s];
        } else if (v > ps) {
            ps = v;
            sec = lab[s];
        }
    }

    *second = sec;
    *margin = pb - ps;

    if (voxmode) {
        return tn_label_at(L, d.nx, d.ny, d.nz, i, j, k);
    }

    return best;
}

// psi_ab = phi_a - phi_b at p, and its gradient by central differences
// (step 0.25 voxel)
inline float tn_psi(TN_FIELD_ARGS, int a, int b, float px, float py, float pz) {
    return tn_phi_at(TN_FIELD, a, px, py, pz) - tn_phi_at(TN_FIELD, b, px, py, pz);
}

// psi and its gradient from one trilinear cell: 8 corner fetches per label
// (the former +-0.25 voxel central differences took 7 psi evaluations, 112
// fetches, and dominated the move stage). The analytic gradient is only C0
// across cell faces; phi is Gaussian-smoothed, so the jump is small.
inline float tn_psi_grad(TN_FIELD_ARGS, int a, int b, float px, float py, float pz, float* g) {
    const float u = px / d.vx, v = py / d.vy, w = pz / d.vz;
    const int i0 = (int)floor(u), j0 = (int)floor(v), k0 = (int)floor(w);
    const float fx = u - i0, fy = v - j0, fz = w - k0;
    float c[8];

    for (int n = 0; n < 8; ++n) {
        const int ii = i0 + (n & 1), jj = j0 + ((n >> 1) & 1), kk = k0 + ((n >> 2) & 1);
        c[n] = tn_phi_vox(d, L, bl_cnt, bl_lab, bl_slot, phi, a, ii, jj, kk) -
               tn_phi_vox(d, L, bl_cnt, bl_lab, bl_slot, phi, b, ii, jj, kk);
    }

    const float x00 = c[0] + fx * (c[1] - c[0]), x10 = c[2] + fx * (c[3] - c[2]);
    const float x01 = c[4] + fx * (c[5] - c[4]), x11 = c[6] + fx * (c[7] - c[6]);
    const float y0 = x00 + fy * (x10 - x00), y1 = x01 + fy * (x11 - x01);
    const float dx0 = (c[1] - c[0]) + fy * ((c[3] - c[2]) - (c[1] - c[0]));
    const float dx1 = (c[5] - c[4]) + fy * ((c[7] - c[6]) - (c[5] - c[4]));
    g[0] = (dx0 + fz * (dx1 - dx0)) / d.vx;
    g[1] = ((x10 - x00) + fz * ((x11 - x01) - (x10 - x00))) / d.vy;
    g[2] = (y1 - y0) / d.vz;
    return y0 + fz * (y1 - y0);
}

// Bracketed bisection (Fang, SPIE 2006) of psi_ab along a direction u: find t in
// [0, rad] with psi(p + t s u) = 0, where s points downhill toward the zero set
// (dpsi/du = du > 0 is the directional derivative sign). Unlike Newton, the
// result never leaves the bracket, so a thin layer with a tiny |grad psi| (1-2
// voxel CSF or skull) cannot throw the node away; with no sign change within rad
// it fails and p is left untouched. Returns 1 on success.
inline int tn_bisect_dir(TN_FIELD_ARGS, int a, int b, float* p, const float* u, float v0, float rad) {
    if (v0 == 0.0f) {
        return 1;
    }

    const float sg = v0 > 0.0f ? -1.0f : 1.0f;   // walk toward the zero
    float t0 = 0.0f, t1 = -1.0f, f0 = v0, f1 = 0.0f;

    for (int k = 1; k <= 4; ++k) {   // bracket in 4 probes
        const float t = rad * 0.25f * k;
        const float v = tn_psi(TN_FIELD, a, b, p[0] + sg * t * u[0], p[1] + sg * t * u[1], p[2] + sg * t * u[2]);

        if ((v > 0.0f) != (v0 > 0.0f) || v == 0.0f) {
            t1 = t;
            f1 = v;
            break;
        }

        t0 = t;
        f0 = v;
    }

    if (t1 < 0.0f) {
        return 0;
    }

    // Illinois regula falsi: stays inside the bracket like bisection, converges
    // superlinearly (the plain 10-step bisection doubled the move-stage cost)
    int side = 0;

    for (int it = 0; it < 6 && f1 != 0.0f; ++it) {
        const float tm = (f0 != f1) ? t1 - f1 * (t1 - t0) / (f1 - f0) : 0.5f * (t0 + t1);
        const float v = tn_psi(TN_FIELD, a, b, p[0] + sg * tm * u[0], p[1] + sg * tm * u[1], p[2] + sg * tm * u[2]);

        if ((v > 0.0f) == (f1 > 0.0f)) {   // same side as t1: replace t1
            t1 = tm;
            f1 = v;

            if (side == -1) {
                f0 *= 0.5f;
            }

            side = -1;
        } else {
            t0 = t1;
            f0 = f1;
            t1 = tm;
            f1 = v;
            side = 1;
        }
    }

    const float t = t1;

    for (int k = 0; k < 3; ++k) {
        p[k] += sg * t * u[k];
    }

    return 1;
}

// Projection of p onto psi_ab = 0 by bisection along the local normal, within
// distance rad. Returns 1 on success (p moved), 0 on failure (p unchanged).
inline int tn_project1(TN_FIELD_ARGS, int a, int b, float* p, float rad) {
    float g[3];
    const float v = tn_psi_grad(TN_FIELD, a, b, p[0], p[1], p[2], g);
    const float gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);

    if (gn < 1e-9f) {
        return 0;
    }

    const float u[3] = { g[0] / gn, g[1] / gn, g[2] / gn };
    return tn_bisect_dir(TN_FIELD, a, b, p, u, v, rad);
}

// Projection onto the junction curve psi_ab = psi_ac = 0 by alternating
// bisections: onto a|b along n_ab, then onto a|c along the part of n_ac
// orthogonal to n_ab (which stays on a|b to first order). Every leg is bracketed;
// on any failure, or a total displacement > rad, p is left untouched. Returns 1
// on success.
inline int tn_project2(TN_FIELD_ARGS, int a, int b, int c, float* p, float rad) {
    float q[3] = { p[0], p[1], p[2] };

    for (int it = 0; it < 3; ++it) {
        float g1[3], g2[3];

        if (!tn_project1(TN_FIELD, a, b, q, rad)) {
            return 0;
        }

        tn_psi_grad(TN_FIELD, a, b, q[0], q[1], q[2], g1);
        const float v2 = tn_psi_grad(TN_FIELD, a, c, q[0], q[1], q[2], g2);
        const float a11 = g1[0] * g1[0] + g1[1] * g1[1] + g1[2] * g1[2];

        if (a11 < 1e-18f) {
            return 0;
        }

        const float pr = (g1[0] * g2[0] + g1[1] * g2[1] + g1[2] * g2[2]) / a11;
        float u[3] = { g2[0] - pr * g1[0], g2[1] - pr * g1[1], g2[2] - pr * g1[2] };
        const float un = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);

        if (un < 1e-9f) {   // tangent surfaces: no curve
            return 0;
        }

        u[0] /= un;
        u[1] /= un;
        u[2] /= un;

        if (!tn_bisect_dir(TN_FIELD, a, c, q, u, v2, rad)) {
            return 0;
        }
    }

    if (!tn_project1(TN_FIELD, a, b, q, rad)) {   // end on a|b exactly
        return 0;
    }

    const float m2 = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]) + (q[2] - p[2]) * (q[2] - p[2]);

    if (m2 > rad * rad) {
        return 0;
    }

    p[0] = q[0];
    p[1] = q[1];
    p[2] = q[2];
    return 1;
}

// ---- stage 3: graded HCP seeding ---------------------------------------------------
// HCP (DistMesh-style) lattice of spacing s: rows in x at s, rows in y at
// s*sqrt(3)/2 (odd rows shifted s/2), layers in z at s*sqrt(2/3) (odd layers
// shifted by (s/2, dy/3)). Point (i,j,k) -> x = i s + (j&1) s/2 + (k&1) s/2,
// y = j dy + (k&1) dy/3, z = k dz. Every grade uses the same origin.
inline void tn_hcp_point(float s, int i, int j, int k, float* p) {
    const float dy = s * 0.8660254f, dz = s * 0.8164966f;
    p[0] = i * s + ((j & 1) ? 0.5f * s : 0.0f) + ((k & 1) ? 0.5f * s : 0.0f);
    p[1] = j * dy + ((k & 1) ? dy / 3.0f : 0.0f);
    p[2] = k * dz;
}

// Seeding uses `nseed` coarse LEVELS, not the 256 sizing grades: each level is
// ONE global HCP lattice (common origin, independent of the label), and a voxel
// only claims the points of its level's lattice that fall inside its box. So a
// connected region of one level -- across voxels and labels -- is a single
// contiguous lattice, with seams only where the level changes; with the 256 fine
// grades the level would change every voxel or two in graded zones. The forces
// still use the continuous h.
inline int tn_seed_level(int grade, int nseed) {
    return nseed <= 1 ? 0 : (int)floor((float)grade * (float)(nseed - 1) / 255.0f + 0.5f);
}

inline float tn_seed_spacing(int level, int nseed, float hmin, float hmax) {
    return nseed <= 1 ? hmin : hmin * pow(hmax / hmin, (float)level / (float)(nseed - 1));
}

// Lattice points of the voxel's seed level inside voxel v's box [i-1/2, i+1/2)
// (mm), kept if the point's label (tn_label_of) is not the exterior. write = 0:
// count only; else write them to P/lab starting at `out`. Returns the count.
inline int tn_seed_voxel(TN_FIELD_ARGS, TN_G const uchar* grade, int nseed, float hmin, float hmax, int voxmode,
                         int i, int j, int k, int write, TN_G float* P, TN_G ushort* lab, int out) {
    const size_t v = i + (size_t)d.nx * (j + (size_t)d.ny * k);

    if (L[v] == 0) {
        // an exterior voxel can still own smooth-mode interior points near the
        // interface, but those are covered by the neighbouring interior voxels'
        // projections; seeding only non-exterior voxels keeps label 0 empty
        return 0;
    }

    const float s = tn_seed_spacing(tn_seed_level(grade[v], nseed), nseed, hmin, hmax);
    const float dy = s * 0.8660254f, dz = s * 0.8164966f;
    const float x0 = (i - 0.5f) * d.vx, x1 = (i + 0.5f) * d.vx;
    const float y0 = (j - 0.5f) * d.vy, y1 = (j + 0.5f) * d.vy;
    const float z0 = (k - 0.5f) * d.vz, z1 = (k + 0.5f) * d.vz;
    int n = 0;
    const int k0 = (int)ceil(z0 / dz), k1 = (int)ceil(z1 / dz);

    for (int kk = k0; kk < k1; ++kk) {
        const float oy = (kk & 1) ? dy / 3.0f : 0.0f;
        const int j0 = (int)ceil((y0 - oy) / dy), j1 = (int)ceil((y1 - oy) / dy);

        for (int jj = j0; jj < j1; ++jj) {
            const float ox = ((jj & 1) ? 0.5f * s : 0.0f) + ((kk & 1) ? 0.5f * s : 0.0f);
            const int i0 = (int)ceil((x0 - ox) / s), i1 = (int)ceil((x1 - ox) / s);

            for (int ii = i0; ii < i1; ++ii) {
                float p[3];
                tn_hcp_point(s, ii, jj, kk, p);
                int sec;
                float mg;
                const int l = tn_label_of(TN_FIELD, voxmode, p[0], p[1], p[2], &sec, &mg);

                if (l == 0 || l == TN_NOLAB) {
                    continue;
                }

                if (write) {
                    P[3 * (out + n)] = p[0];
                    P[3 * (out + n) + 1] = p[1];
                    P[3 * (out + n) + 2] = p[2];
                    lab[out + n] = (ushort)l;
                }

                ++n;
            }
        }
    }

    return n;
}

// Node i after seeding: if it lies within half its spacing of an interface it is
// projected onto that interface (INTERFACE); if a third label competes there, onto
// the junction curve (JUNCTION); four labels -> CORNER. On an internal a|b
// interface only the lower label's nodes are projected (the other side's stay
// interior, so the interface is not sampled twice); exterior interfaces (b = 0)
// always. part[2*i], part[2*i+1] = the partner labels.
inline void tn_seed_classify(TN_FIELD_ARGS, TN_G const float* hvox, int i, TN_G float* P, TN_G ushort* lab,
                             TN_G uchar* typ, TN_G ushort* part) {
    float p[3];
    p[0] = P[3 * i];
    p[1] = P[3 * i + 1];
    p[2] = P[3 * i + 2];
    typ[i] = TN_INTERIOR;
    part[2 * i] = part[2 * i + 1] = TN_NOLAB;
    const int a = lab[i];
    int nl[TN_BL];
    const int n = tn_labels_near(d, L, bl_cnt, bl_lab, p[0], p[1], p[2], nl);

    if (n < 2) {
        return;
    }

    // the strongest competitor b
    int b = TN_NOLAB;
    float pb = -1.0f;

    for (int s = 0; s < n; ++s)
        if (nl[s] != a) {
            const float v = tn_phi_at(TN_FIELD, nl[s], p[0], p[1], p[2]);

            if (v > pb) {
                pb = v;
                b = nl[s];
            }
        }

    if (b == TN_NOLAB || (b != 0 && b < a)) {
        return;
    }

    const int vi = (int)floor(p[0] / d.vx + 0.5f), vj = (int)floor(p[1] / d.vy + 0.5f), vk = (int)floor(p[2] / d.vz + 0.5f);
    const float hs = hvox[vi + (size_t)d.nx * (vj + (size_t)d.ny * vk)];
    float g[3];
    const float v = tn_psi_grad(TN_FIELD, a, b, p[0], p[1], p[2], g);
    const float gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);

    if (gn < 1e-9f || fabs(v) / gn > 0.5f * hs) {
        return;    // farther than half a spacing from the interface
    }

    float q[3];
    q[0] = p[0];
    q[1] = p[1];
    q[2] = p[2];
    if (!tn_project1(TN_FIELD, a, b, q, 0.6f * hs)) {
        return;   // no bracketed zero within reach: stay interior
    }

    typ[i] = TN_INTERFACE;
    part[2 * i] = (ushort)b;

    // a third label competing at the projected point: a junction curve
    int c = TN_NOLAB;
    float pc = -1.0f;
    const float pa = tn_phi_at(TN_FIELD, a, q[0], q[1], q[2]);

    for (int s = 0; s < n; ++s)
        if (nl[s] != a && nl[s] != b) {
            const float w = tn_phi_at(TN_FIELD, nl[s], q[0], q[1], q[2]);

            if (w > pc) {
                pc = w;
                c = nl[s];
            }
        }

    if (c != TN_NOLAB && pc > 0.5f * pa) {
        float r[3];
        r[0] = q[0];
        r[1] = q[1];
        r[2] = q[2];
        if (tn_project2(TN_FIELD, a, b, c, r, 0.5f * hs)) {
            q[0] = r[0];
            q[1] = r[1];
            q[2] = r[2];
            typ[i] = TN_JUNCTION;
            part[2 * i + 1] = (ushort)c;
        }
    }

    P[3 * i] = q[0];
    P[3 * i + 1] = q[1];
    P[3 * i + 2] = q[2];
}

#ifdef __OPENCL_VERSION__
// kernels are added with the device path
#endif

// ---- corner nodes: points where >= 4 labels meet ------------------------------------
// Grid vertex (i,j,k), i in [-1, nx-1], is the corner shared by voxels
// (i..i+1, j..j+1, k..k+1), at ((i+0.5) vx, ...). If those 8 voxels hold >= 4
// labels (0 included), a fixed CORNER node goes there: no tet near such a point
// has a consistent label set without one. The 4 labels with the largest phi at
// the vertex are kept (own = the strongest non-zero one); the position is the
// 3-constraint Newton solution of psi_ab = psi_ac = psi_ad = 0 from the vertex,
// kept only if it stays within half a voxel (else the vertex itself). Clusters of
// adjacent 4-label vertices (staircases) keep only their local minimum index.
inline int tn_vertex_nlab(TN_G const ushort* L, int nx, int ny, int nz, int i, int j, int k, int* labs) {
    int n = 0;

    for (int m = 0; m < 8; ++m) {
        const int l = tn_label_at(L, nx, ny, nz, i + (m & 1), j + ((m >> 1) & 1), k + (m >> 2));
        int seen = 0;

        for (int x = 0; x < n; ++x) {
            seen |= labs[x] == l;
        }

        if (!seen) {
            labs[n++] = l;
        }
    }

    return n;
}

inline int tn_corner_vertex(TN_FIELD_ARGS, int i, int j, int k, int write, TN_G float* P, TN_G ushort* lab,
                            TN_G uchar* typ, TN_G ushort* part, TN_G ushort* part3, int out) {
    int labs[8];
    const int n = tn_vertex_nlab(L, d.nx, d.ny, d.nz, i, j, k, labs);

    if (n < 4) {
        return 0;
    }

    // local-minimum suppression over the 26 neighbouring vertices
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if ((dz < 0 || (dz == 0 && (dy < 0 || (dy == 0 && dx < 0))))) {   // a smaller index
                    int l2[8];

                    if (tn_vertex_nlab(L, d.nx, d.ny, d.nz, i + dx, j + dy, k + dz, l2) >= 4) {
                        return 0;
                    }
                }
            }

    if (!write) {
        return 1;
    }

    float x[3];
    x[0] = (i + 0.5f) * d.vx;
    x[1] = (j + 0.5f) * d.vy;
    x[2] = (k + 0.5f) * d.vz;
    float ph[8];

    for (int s = 0; s < n; ++s) {
        ph[s] = tn_phi_at(TN_FIELD, labs[s], x[0], x[1], x[2]);
    }

    for (int s = 0; s < n; ++s)   // sort labels by phi, descending
        for (int t = s + 1; t < n; ++t)
            if (ph[t] > ph[s]) {
                const float f = ph[s];
                ph[s] = ph[t];
                ph[t] = f;
                const int l = labs[s];
                labs[s] = labs[t];
                labs[t] = l;
            }

    int ia = 0;

    while (ia < 4 && labs[ia] == 0) {
        ++ia;
    }

    const int a = labs[ia];
    int o[3], no = 0;

    for (int s = 0; s < 4; ++s)
        if (s != ia) {
            o[no++] = labs[s];
        }

    // Newton on (psi_a o0, psi_a o1, psi_a o2) = 0
    float q[3] = { x[0], x[1], x[2] };
    int ok = 1;

    for (int it = 0; it < 5 && ok; ++it) {
        float J[3][3], v[3];

        for (int r = 0; r < 3; ++r) {
            v[r] = tn_psi_grad(TN_FIELD, a, o[r], q[0], q[1], q[2], J[r]);
        }

        const float det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                          J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

        if (fabs(det) < 1e-12f) {
            ok = 0;
            break;
        }

        float dlt[3];   // Cramer: J dlt = -v

        for (int c = 0; c < 3; ++c) {
            float M[3][3];

            for (int r = 0; r < 3; ++r)
                for (int cc = 0; cc < 3; ++cc) {
                    M[r][cc] = cc == c ? -v[r] : J[r][cc];
                }

            dlt[c] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                      M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) / det;
        }

        for (int c = 0; c < 3; ++c) {
            q[c] += dlt[c];
        }
    }

    const float ex = (q[0] - x[0]) / d.vx, ey = (q[1] - x[1]) / d.vy, ez = (q[2] - x[2]) / d.vz;

    if (ok && ex * ex + ey * ey + ez * ez <= 0.25f) {
        x[0] = q[0];
        x[1] = q[1];
        x[2] = q[2];
    }

    P[3 * out] = x[0];
    P[3 * out + 1] = x[1];
    P[3 * out + 2] = x[2];
    lab[out] = (ushort)a;
    typ[out] = TN_CORNER;
    part[2 * out] = (ushort)o[0];
    part[2 * out + 1] = (ushort)o[1];
    part3[out] = (ushort)o[2];
    return 1;
}
