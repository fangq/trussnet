// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_particle_body.cl -- stages 4-9, all vertex-parallel: the multi-level spatial
// hash, the Verlet K-nearest truss, the spring forces, the move with interface
// trapping, and the convergence measure. Needs tn_grid_body.cl and
// tn_seed_body.cl first. Host and device.
//
// Hash: level L has cubic bins of side b_L = b0 * 2^L over the grid box (b0 = the
// finest node's search radius); node i is registered at the smallest level whose
// bin is >= its own search radius R_i = (t + skin) h_i. All levels share one
// counting sort: key = level offset + bin index in that level. A pair i, j is a
// truss bar if |x_i - x_j| < (t + skin) h_mid, h_mid = (h_i + h_j)/2 (the list
// keeps the skin so it stays valid until a node has moved skin h / 2), and not if
// both are INTERIOR with different labels.

#define TN_K 24          // max truss neighbours per node
#define TN_MAXLEV 8

typedef struct {
    float b0;                  // level-0 bin side (mm)
    int nlev;
    float ox, oy, oz;          // box origin (mm)
    int dim[TN_MAXLEV][3];     // bins per axis per level
    int off[TN_MAXLEV];        // first key of each level
} TnHash;

// the local size h at p: the sizing field at the nearest voxel
inline float tn_h_at(TnDims d, TN_G const float* hvox, float px, float py, float pz) {
    int i = (int)floor(px / d.vx + 0.5f), j = (int)floor(py / d.vy + 0.5f), k = (int)floor(pz / d.vz + 0.5f);
    i = i < 0 ? 0 : (i >= d.nx ? d.nx - 1 : i);
    j = j < 0 ? 0 : (j >= d.ny ? d.ny - 1 : j);
    k = k < 0 ? 0 : (k >= d.nz ? d.nz - 1 : k);
    return hvox[i + (size_t)d.nx * (j + (size_t)d.ny * k)];
}

inline int tn_level_of(const TnHash* H, float R) {
    int L = 0;

    while (L + 1 < H->nlev && H->b0 * (float)(1 << L) < R) {
        ++L;
    }

    return L;
}

inline int tn_bin_coord(const TnHash* H, int L, float x, float o, int ax) {
    int c = (int)floor((x - o) / (H->b0 * (float)(1 << L)));
    const int n = H->dim[L][ax];
    return c < 0 ? 0 : (c >= n ? n - 1 : c);
}

inline int tn_bin_key(const TnHash* H, int L, float px, float py, float pz) {
    const int cx = tn_bin_coord(H, L, px, H->ox, 0), cy = tn_bin_coord(H, L, py, H->oy, 1),
              cz = tn_bin_coord(H, L, pz, H->oz, 2);
    return H->off[L] + cx + H->dim[L][0] * (cy + H->dim[L][1] * cz);
}

// K nearest valid neighbours of node i (sorted by distance), written to
// nbr[i*TN_K ..], count to nnb[i]. cstart[key..key+1) indexes the sorted ids.
inline void tn_neighbors(const TnHash* H, TnDims d, TN_G const float* hvox, TN_G const float* P,
                         TN_G const ushort* lab, TN_G const uchar* typ, TN_G const int* cstart,
                         TN_G const int* sorted, float t, float skin, int i, TN_G int* nbr, TN_G int* nnb) {
    const float px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2];
    const float hi = tn_h_at(d, hvox, px, py, pz);
    const float Ri = (t + skin) * hi;
    const int Li = tn_level_of(H, Ri);
    int ids[TN_K];
    float ds[TN_K];
    int n = 0;

    for (int L = Li - 1; L <= Li + 1; ++L) {
        if (L < 0 || L >= H->nlev) {
            continue;
        }

        const float bL = H->b0 * (float)(1 << L);
        const float r = Ri > bL ? Ri : bL;   // every node of level L has R_j <= b_L
        const int x0 = tn_bin_coord(H, L, px - r, H->ox, 0), x1 = tn_bin_coord(H, L, px + r, H->ox, 0);
        const int y0 = tn_bin_coord(H, L, py - r, H->oy, 1), y1 = tn_bin_coord(H, L, py + r, H->oy, 1);
        const int z0 = tn_bin_coord(H, L, pz - r, H->oz, 2), z1 = tn_bin_coord(H, L, pz + r, H->oz, 2);

        for (int cz = z0; cz <= z1; ++cz)
            for (int cy = y0; cy <= y1; ++cy)
                for (int cx = x0; cx <= x1; ++cx) {
                    const int key = H->off[L] + cx + H->dim[L][0] * (cy + H->dim[L][1] * cz);

                    for (int s = cstart[key]; s < cstart[key + 1]; ++s) {
                        const int j = sorted[s];

                        if (j == i) {
                            continue;
                        }

                        if (typ[i] == TN_INTERIOR && typ[j] == TN_INTERIOR && lab[i] != lab[j]) {
                            continue;   // no bar through an interface
                        }

                        const float ex = P[3 * j] - px, ey = P[3 * j + 1] - py, ez = P[3 * j + 2] - pz;
                        const float dist = sqrt(ex * ex + ey * ey + ez * ez);
                        const float hj = tn_h_at(d, hvox, P[3 * j], P[3 * j + 1], P[3 * j + 2]);

                        if (dist >= (t + skin) * 0.5f * (hi + hj)) {
                            continue;
                        }

                        // insert into the sorted K-list
                        if (n == TN_K && dist >= ds[TN_K - 1]) {
                            continue;
                        }

                        int k = n < TN_K ? n++ : TN_K - 1;

                        while (k > 0 && (ds[k - 1] > dist || (ds[k - 1] == dist && ids[k - 1] > j))) {
                            ds[k] = ds[k - 1];
                            ids[k] = ids[k - 1];
                            --k;
                        }

                        ds[k] = dist;
                        ids[k] = j;
                    }
                }
    }

    nnb[i] = n;

    for (int k = 0; k < n; ++k) {
        nbr[i * TN_K + k] = ids[k];
    }
}

// Spring force on node i (pull: sums its own bars, no atomics). DistMesh law:
// repulsive only, f = (l0 - l)+ along the bar, l0 = fscale h_mid. Bars between
// two interface nodes use fsurf instead: the restricted-Delaunay surface is made
// only of surface nodes if a surface triangle's circumradius (~ spacing/sqrt 3)
// is below the depth of the next interior layer, so the surface is sampled
// denser than the interior.
// F[3] returns the number of active (compressed) bars: the node's stiffness,
// used by the Jacobi step in tn_move.
inline void tn_force(TnDims d, TN_G const float* hvox, TN_G const float* P, TN_G const uchar* typ,
                     TN_G const int* nbr, TN_G const int* nnb, float fscale, float fsurf, int i, float* F) {
    const float px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2];
    const float hi = tn_h_at(d, hvox, px, py, pz);
    F[0] = F[1] = F[2] = F[3] = 0.0f;

    for (int k = 0; k < nnb[i]; ++k) {
        const int j = nbr[i * TN_K + k];
        const float ex = px - P[3 * j], ey = py - P[3 * j + 1], ez = pz - P[3 * j + 2];
        const float l = sqrt(ex * ex + ey * ey + ez * ez);
        const float hj = tn_h_at(d, hvox, P[3 * j], P[3 * j + 1], P[3 * j + 2]);
        const float l0 = ((typ[i] != TN_INTERIOR && typ[j] != TN_INTERIOR) ? fsurf : fscale) * 0.5f * (hi + hj);

        if (l >= l0 || l <= 0.0f) {
            continue;
        }

        const float f = (l0 - l) / l;
        F[0] += f * ex;
        F[1] += f * ey;
        F[2] += f * ez;
        F[3] += 1.0f;
    }
}

// The crossing of segment p -> q with psi_ab = 0 (psi(p) > 0 >= psi(q)): bisection
// refined by a final secant step. Writes the point to c.
inline void tn_crossing(TN_FIELD_ARGS, int a, int b, const float* p, const float* q, float* c) {
    float t0 = 0.0f, t1 = 1.0f;
    float v0 = tn_psi(TN_FIELD, a, b, p[0], p[1], p[2]);
    float v1 = tn_psi(TN_FIELD, a, b, q[0], q[1], q[2]);

    for (int it = 0; it < 6; ++it) {
        const float tm = 0.5f * (t0 + t1);
        const float vm = tn_psi(TN_FIELD, a, b, p[0] + tm * (q[0] - p[0]), p[1] + tm * (q[1] - p[1]),
                                p[2] + tm * (q[2] - p[2]));

        if (vm > 0.0f) {
            t0 = tm;
            v0 = vm;
        } else {
            t1 = tm;
            v1 = vm;
        }
    }

    const float t = (v0 - v1) != 0.0f ? t0 + (t1 - t0) * v0 / (v0 - v1) : 0.5f * (t0 + t1);

    for (int k = 0; k < 3; ++k) {
        c[k] = p[k] + t * (q[k] - p[k]);
    }
}

// Remove the component of v along the unit normal of psi_ab at p.
inline void tn_tangential(TN_FIELD_ARGS, int a, int b, const float* p, float* v) {
    float g[3];
    tn_psi_grad(TN_FIELD, a, b, p[0], p[1], p[2], g);
    const float gn2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];

    if (gn2 < 1e-12f) {
        return;
    }

    const float vn = (v[0] * g[0] + v[1] * g[1] + v[2] * g[2]) / gn2;

    for (int k = 0; k < 3; ++k) {
        v[k] -= vn * g[k];
    }
}

// If a label other than a and b competes at q (phi_c > 0.5 phi_a), return it.
inline int tn_third_label(TN_FIELD_ARGS, int a, int b, const float* q) {
    int nl[TN_BL];
    const int n = tn_labels_near(d, L, bl_cnt, bl_lab, q[0], q[1], q[2], nl);
    const float pa = tn_phi_at(TN_FIELD, a, q[0], q[1], q[2]);
    int c = TN_NOLAB;
    float pc = 0.5f * pa;

    for (int s = 0; s < n; ++s)
        if (nl[s] != a && nl[s] != b) {
            const float w = tn_phi_at(TN_FIELD, nl[s], q[0], q[1], q[2]);

            if (w > pc) {
                pc = w;
                c = nl[s];
            }
        }

    return c;
}

// Is q a valid point of the a|b interface (or, with c != TN_NOLAB, of the a|b|c
// junction): do the node's labels dominate there? The zero set of phi_a - phi_b
// continues past a junction into a third label's region (where phi_a ~ phi_b ~ 0),
// so a glide can land on that meaningless branch; such a move is refused.
inline int tn_valid_on(TN_FIELD_ARGS, int a, int b, int c, const float* q) {
    int nl[TN_BL];
    const int n = tn_labels_near(d, L, bl_cnt, bl_lab, q[0], q[1], q[2], nl);
    float mine = fmin(tn_phi_at(TN_FIELD, a, q[0], q[1], q[2]), tn_phi_at(TN_FIELD, b, q[0], q[1], q[2]));

    if (c != TN_NOLAB) {
        mine = fmin(mine, tn_phi_at(TN_FIELD, c, q[0], q[1], q[2]));
    }

    for (int s = 0; s < n; ++s)
        if (nl[s] != a && nl[s] != b && nl[s] != c && tn_phi_at(TN_FIELD, nl[s], q[0], q[1], q[2]) > mine + 0.02f) {
            return 0;
        }

    return 1;
}

// Move node i by its step (dt F, capped at maxstep h), trapping it on the smooth
// interfaces (see the header). Writes the new position and state; returns
// |displacement| / h for the convergence test.
// `snap`: an INTERIOR node that ends closer than snap*h to an interface is put ON
// it (projected, then it glides). Without it the surface is sampled only by the
// nodes that happen to cross, which left gaps that interior nodes 0.35-0.7 h deep
// filled in the restricted-Delaunay surface (non-conforming boundary faces).
inline float tn_move(TN_FIELD_ARGS, TN_G const float* hvox, TN_G const float* F, float dt, float maxstep,
                     float snap, int i, TN_G float* P, TN_G const ushort* lab, TN_G uchar* typ, TN_G ushort* part) {
    float p[3], s[3], q[3];
    p[0] = P[3 * i];
    p[1] = P[3 * i + 1];
    p[2] = P[3 * i + 2];
    const float h = tn_h_at(d, hvox, p[0], p[1], p[2]);
    const int a = lab[i];
    int ty = typ[i];

    if (ty == TN_CORNER) {
        return 0.0f;
    }

    // Jacobi step: the force over the node's stiffness (its active bars), times
    // dt (relaxation factor). A fixed dt * F overshoots where a node has many
    // compressed bars (thin, over-dense regions) and oscillates.
    const float stiff = F[4 * i + 3] > 1.0f ? F[4 * i + 3] : 1.0f;

    for (int k = 0; k < 3; ++k) {
        s[k] = dt * F[4 * i + k] / stiff;
    }

    const float sl = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);

    if (sl > maxstep * h) {
        for (int k = 0; k < 3; ++k) {
            s[k] *= maxstep * h / sl;
        }
    }

    int b = part[2 * i], c = part[2 * i + 1];

    if (ty == TN_INTERIOR) {
        for (int k = 0; k < 3; ++k) {
            q[k] = p[k] + s[k];
        }

        int sec;
        float mg;
        const int l = tn_label_of(TN_FIELD, 0, q[0], q[1], q[2], &sec, &mg);

        if (l != a) {   // crossed into label l: trap at the crossing and glide on
            b = l;
            float x[3], r[3];
            tn_crossing(TN_FIELD, a, b, p, q, x);

            for (int k = 0; k < 3; ++k) {
                r[k] = q[k] - x[k];
            }

            tn_tangential(TN_FIELD, a, b, x, r);

            for (int k = 0; k < 3; ++k) {
                q[k] = x[k] + r[k];
            }

            tn_project1(TN_FIELD, a, b, q, 3);
            ty = TN_INTERFACE;
            part[2 * i] = (ushort)b;
        } else if (sec != TN_NOLAB && snap > 0.0f) {
            // near an interface? distance ~ psi / |grad psi|
            float gq[3];
            const float v = tn_psi_grad(TN_FIELD, a, sec, q[0], q[1], q[2], gq);
            const float gn = sqrt(gq[0] * gq[0] + gq[1] * gq[1] + gq[2] * gq[2]);

            if (gn > 1e-9f && v / gn < snap * h) {
                tn_project1(TN_FIELD, a, sec, q, 3);
                ty = TN_INTERFACE;
                b = sec;
                part[2 * i] = (ushort)sec;
            }
        }
    } else if (ty == TN_INTERFACE) {
        tn_tangential(TN_FIELD, a, b, p, s);

        for (int k = 0; k < 3; ++k) {
            q[k] = p[k] + s[k];
        }

        tn_project1(TN_FIELD, a, b, q, 3);
    } else {   // TN_JUNCTION: along the curve direction n_ab x n_ac
        float g1[3], g2[3], tdir[3];
        tn_psi_grad(TN_FIELD, a, b, p[0], p[1], p[2], g1);
        tn_psi_grad(TN_FIELD, a, c, p[0], p[1], p[2], g2);
        tdir[0] = g1[1] * g2[2] - g1[2] * g2[1];
        tdir[1] = g1[2] * g2[0] - g1[0] * g2[2];
        tdir[2] = g1[0] * g2[1] - g1[1] * g2[0];
        const float tl2 = tdir[0] * tdir[0] + tdir[1] * tdir[1] + tdir[2] * tdir[2];
        const float st = tl2 > 1e-20f ? (s[0] * tdir[0] + s[1] * tdir[1] + s[2] * tdir[2]) / tl2 : 0.0f;

        for (int k = 0; k < 3; ++k) {
            q[k] = p[k] + st * tdir[k];
        }

        tn_project2(TN_FIELD, a, b, c, q, 3);
    }

    // an interface node that met a third label joins the junction curve
    if (ty == TN_INTERFACE) {
        const int cc = tn_third_label(TN_FIELD, a, b, q);

        if (cc != TN_NOLAB) {
            float r[3];
            r[0] = q[0];
            r[1] = q[1];
            r[2] = q[2];
            tn_project2(TN_FIELD, a, b, cc, r, 4);
            const float dr = sqrt((r[0] - q[0]) * (r[0] - q[0]) + (r[1] - q[1]) * (r[1] - q[1]) + (r[2] - q[2]) * (r[2] - q[2]));

            if (dr < 0.5f * h) {
                q[0] = r[0];
                q[1] = r[1];
                q[2] = r[2];
                ty = TN_JUNCTION;
                part[2 * i + 1] = (ushort)cc;
            }
        }
    }

    // refuse a move onto an invalid branch of the constraint (see tn_valid_on)
    if (ty != TN_INTERIOR && !tn_valid_on(TN_FIELD, a, part[2 * i], ty >= TN_JUNCTION ? part[2 * i + 1] : TN_NOLAB, q)) {
        if (ty == TN_INTERFACE || typ[i] == TN_INTERIOR) {
            // an interface node that ran into a third label: pin it on the junction
            // if that projection is valid, else stay
            const int cc = tn_third_label(TN_FIELD, a, part[2 * i], q);
            float r[3];
            r[0] = q[0];
            r[1] = q[1];
            r[2] = q[2];

            if (cc != TN_NOLAB) {
                tn_project2(TN_FIELD, a, part[2 * i], cc, r, 4);
            }

            if (cc != TN_NOLAB && tn_valid_on(TN_FIELD, a, part[2 * i], cc, r)) {
                q[0] = r[0];
                q[1] = r[1];
                q[2] = r[2];
                ty = TN_JUNCTION;
                part[2 * i + 1] = (ushort)cc;
            } else {
                typ[i] = (uchar)(typ[i] == TN_INTERIOR ? TN_INTERIOR : ty);
                if (typ[i] == TN_INTERIOR) {
                    part[2 * i] = TN_NOLAB;
                }
                return 0.0f;
            }
        } else {
            return 0.0f;   // a junction node stays
        }
    }

    const float mv = sqrt((q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]) + (q[2] - p[2]) * (q[2] - p[2]));
    P[3 * i] = q[0];
    P[3 * i + 1] = q[1];
    P[3 * i + 2] = q[2];
    typ[i] = (uchar)ty;
    return mv / h;
}

#ifdef __OPENCL_VERSION__
// kernels are added with the device path
#endif
