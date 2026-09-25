// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_grid_body.cl -- stage 2: the smooth interface fields, curvature and the
// sizing field, all voxel-parallel. Compiles as OpenCL and (via #include) as the
// host OpenMP reference; the includer defines TN_G (__global or nothing).
//
// Coordinates: "grid mm" -- voxel (i,j,k)'s centre is at (i*vx, j*vy, k*vz); its
// box is [i-1/2, i+1/2) voxels. Outside the grid is label 0 (the exterior).
//
// Smooth fields: for each label l, phi_l = G_sigma * [L == l] (a Gaussian-smoothed
// indicator). They differ from 0/1 only within R = ceil(3 sigma) voxels of a label
// change, so they are stored sparsely: the grid is cut into 8^3-voxel BRICKS; a
// brick lists the labels present within R of it (at most TN_BL); a brick with a
// single label is UNIFORM (phi = its indicator, nothing stored), otherwise each of
// its labels owns a SLOT of 8^3 values. The interface between labels a and b is
// the zero set of psi_ab = phi_a - phi_b.

#define TN_BS   8          // brick side (voxels)
#define TN_SLOT 512        // 8^3 values per (brick, label) slot
#define TN_BL   16         // max labels per brick (a 17-label TPM needs 16 near deep nuclei)
#define TN_NOSLOT (-1)

// ---- brick bookkeeping -----------------------------------------------------------
// Labels present in brick b (bx,by,bz) dilated by R voxels; writes up to TN_BL
// labels to lab[] (ascending), returns the count (TN_BL+1 if it overflowed).
inline int tn_brick_labels(TN_G const ushort* L, int nx, int ny, int nz, int bx, int by, int bz, int R,
                           ushort* lab) {
    int n = 0;
    int ovf = 0;
    const int x0 = bx * TN_BS - R, y0 = by * TN_BS - R, z0 = bz * TN_BS - R;
    const int x1 = bx * TN_BS + TN_BS + R, y1 = by * TN_BS + TN_BS + R, z1 = bz * TN_BS + TN_BS + R;

    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const int in = x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
                const ushort l = in ? L[x + (size_t)nx * (y + (size_t)ny * z)] : (ushort)0;
                int seen = 0;

                for (int i = 0; i < n; ++i)
                    if (lab[i] == l) {
                        seen = 1;
                        break;
                    }

                if (seen) {
                    continue;
                }

                if (n >= TN_BL) {
                    ovf = 1;
                    continue;
                }

                // insertion keeps the list ascending (deterministic slot order)
                int k = n++;

                while (k > 0 && lab[k - 1] > l) {
                    lab[k] = lab[k - 1];
                    --k;
                }

                lab[k] = l;
            }

    return ovf ? TN_BL + 1 : n;
}

// label of voxel (i,j,k), 0 outside the grid
inline int tn_label_at(TN_G const ushort* L, int nx, int ny, int nz, int i, int j, int k) {
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) {
        return 0;
    }

    return L[i + (size_t)nx * (j + (size_t)ny * k)];
}

// Gaussian weight of integer offset d (unnormalised; the caller normalises)
inline float tn_gauss(int d, float sigma) {
    return exp(-0.5f * (float)(d * d) / (sigma * sigma));
}

// Smoothed indicator of label l at voxel (i,j,k) by a direct separable-weight
// 3D convolution (host reference; the device uses a local-memory tile).
inline float tn_smooth_voxel(TN_G const ushort* L, int nx, int ny, int nz, int i, int j, int k, int l,
                             float sigma, int R) {
    float s = 0.0f, w = 0.0f;

    for (int dz = -R; dz <= R; ++dz) {
        const float gz = tn_gauss(dz, sigma);

        for (int dy = -R; dy <= R; ++dy) {
            const float gyz = gz * tn_gauss(dy, sigma);

            for (int dx = -R; dx <= R; ++dx) {
                const float g = gyz * tn_gauss(dx, sigma);
                w += g;
                s += g * (tn_label_at(L, nx, ny, nz, i + dx, j + dy, k + dz) == l ? 1.0f : 0.0f);
            }
        }
    }

    return s / w;
}

// ---- field lookup ----------------------------------------------------------------
// The grid description every lookup needs, passed by value.
typedef struct {
    int nx, ny, nz;          // voxels
    int nbx, nby, nbz;       // bricks
    float vx, vy, vz;        // voxel size (mm)
} TnDims;

inline int tn_brick_of(TnDims d, int i, int j, int k) {
    return (i >> 3) + d.nbx * ((j >> 3) + d.nby * (k >> 3));
}

// slot of label l in brick b, TN_NOSLOT if the brick is uniform or lacks l;
// *has = 1 if l is among the brick's labels
inline int tn_slot_of(TN_G const int* bl_cnt, TN_G const ushort* bl_lab, TN_G const int* bl_slot, int b, int l,
                      int* has) {
    const int n = bl_cnt[b];
    *has = 0;

    for (int i = 0; i < n && i < TN_BL; ++i)
        if (bl_lab[b * TN_BL + i] == l) {
            *has = 1;
            return bl_slot[b] < 0 ? TN_NOSLOT : bl_slot[b] + i;
        }

    return TN_NOSLOT;
}

// phi_l at voxel (i,j,k) (inside the grid)
inline float tn_phi_vox(TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab,
                        TN_G const int* bl_slot, TN_G const float* phi, int l, int i, int j, int k) {
    if (i < 0 || j < 0 || k < 0 || i >= d.nx || j >= d.ny || k >= d.nz) {
        return l == 0 ? 1.0f : 0.0f;
    }

    const int b = tn_brick_of(d, i, j, k);
    int has;
    const int s = tn_slot_of(bl_cnt, bl_lab, bl_slot, b, l, &has);

    if (s == TN_NOSLOT) {   // uniform brick, or l not within R: exact indicator
        return (has && L[i + (size_t)d.nx * (j + (size_t)d.ny * k)] == l) ? 1.0f : 0.0f;
    }

    const int li = i - ((i >> 3) << 3), lj = j - ((j >> 3) << 3), lk = k - ((k >> 3) << 3);
    return phi[(size_t)s * TN_SLOT + li + TN_BS * (lj + TN_BS * lk)];
}

// The 8 corner values of phi_l for the trilinear cell at voxel (i0,j0,k0) (bit 0
// of the corner index -> +i, bit 1 -> +j, bit 2 -> +k). When the cell lies in one
// brick (7 of 8 cells per axis) the slot is looked up ONCE instead of per corner:
// the move / trap stage does hundreds of these per interface node per step.
inline void tn_phi_cell(TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab,
                        TN_G const int* bl_slot, TN_G const float* phi, int l, int i0, int j0, int k0, float* c) {
    if (i0 >= 0 && j0 >= 0 && k0 >= 0 && i0 + 1 < d.nx && j0 + 1 < d.ny && k0 + 1 < d.nz && (i0 >> 3) == ((i0 + 1) >> 3) &&
            (j0 >> 3) == ((j0 + 1) >> 3) && (k0 >> 3) == ((k0 + 1) >> 3)) {
        const int b = tn_brick_of(d, i0, j0, k0);
        int has;
        const int s = tn_slot_of(bl_cnt, bl_lab, bl_slot, b, l, &has);

        if (s != TN_NOSLOT) {
            const size_t base = (size_t)s * TN_SLOT + (i0 & 7) + TN_BS * ((j0 & 7) + TN_BS * (k0 & 7));

            for (int n = 0; n < 8; ++n) {
                c[n] = phi[base + (n & 1) + TN_BS * ((n >> 1) & 1) + TN_BS * TN_BS * (n >> 2)];
            }
        } else {
            for (int n = 0; n < 8; ++n) {
                c[n] = (has && L[(i0 + (n & 1)) + (size_t)d.nx * ((j0 + ((n >> 1) & 1)) + (size_t)d.ny * (k0 + (n >> 2)))] == l)
                       ? 1.0f : 0.0f;
            }
        }

        return;
    }

    for (int n = 0; n < 8; ++n) {
        c[n] = tn_phi_vox(d, L, bl_cnt, bl_lab, bl_slot, phi, l, i0 + (n & 1), j0 + ((n >> 1) & 1), k0 + ((n >> 2) & 1));
    }
}

// phi_l trilinear at a point p (grid mm)
inline float tn_phi_at(TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab,
                       TN_G const int* bl_slot, TN_G const float* phi, int l, float px, float py, float pz) {
    const float u = px / d.vx, v = py / d.vy, w = pz / d.vz;
    const int i0 = (int)floor(u), j0 = (int)floor(v), k0 = (int)floor(w);
    const float fx = u - i0, fy = v - j0, fz = w - k0;
    float c[8];
    tn_phi_cell(d, L, bl_cnt, bl_lab, bl_slot, phi, l, i0, j0, k0, c);
    const float x00 = c[0] + fx * (c[1] - c[0]), x10 = c[2] + fx * (c[3] - c[2]);
    const float x01 = c[4] + fx * (c[5] - c[4]), x11 = c[6] + fx * (c[7] - c[6]);
    const float y0 = x00 + fy * (x10 - x00), y1 = x01 + fy * (x11 - x01);
    return y0 + fz * (y1 - y0);
}

// ---- thin-layer thickness (per voxel) ----------------------------------------------
// erfinv, single precision (M. Giles, "Approximating the erfinv function", 2010)
inline float tn_erfinv(float x) {
    float w = -log((1.0f - x) * (1.0f + x)), p;

    if (w < 5.0f) {
        w -= 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }

    return p * x;
}

// Local thickness (voxels) of the layer containing voxel (i,j,k), from the
// smoothed indicator of its own label: a slab of thickness t smoothed with sigma
// never reaches 1 but peaks at erf(t / (2 sqrt2 sigma)) on its mid-plane, so the
// largest phi_a over the label-a voxels within +-r inverts to t. (On Colin27 this
// tracks the distance-transform thickness with r = 0.95 for CSF/skull, median
// error 0.5 voxel; the tangent curvature is blind to flat thin layers.) Returns a
// large value in uniform bricks.
inline float tn_thick_voxel(TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab,
                            TN_G const int* bl_slot, TN_G const float* phi, float sigma, int r, int i, int j, int k) {
    const int b = tn_brick_of(d, i, j, k);

    if (bl_slot[b] < 0) {
        return 1e30f;
    }

    const int a = L[i + (size_t)d.nx * (j + (size_t)d.ny * k)];
    float m = 0.0f;

    for (int dz = -r; dz <= r; ++dz)
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx) {
                const int u = i + dx, v = j + dy, w = k + dz;

                if (tn_label_at(L, d.nx, d.ny, d.nz, u, v, w) != a) {
                    continue;
                }

                const float f = tn_phi_vox(d, L, bl_cnt, bl_lab, bl_slot, phi, a, u, v, w);
                m = f > m ? f : m;
            }

    if (m >= 0.9999f) {
        return 1e30f;
    }

    return 2.8284271f * sigma * tn_erfinv(m);
}

// ---- label-preserving correction (per voxel) -------------------------------------
// Gaussian smoothing erases layers thinner than ~2 sigma: a 1-voxel CSF sheet
// peaks at phi = erf(1/(2 sqrt2 sigma)) ~ 0.38 at sigma = 1, below its
// neighbours, so its voxels lose the argmax and the layer vanishes from the
// interface field (19.5% of Colin27's CSF voxels). Here every voxel centre is made
// to keep its own label with margin m: phi_own = max(phi_own, max_other + m).
// Each voxel touches only its own values (no races), and for two face-adjacent
// voxels of labels a and b, psi_ab now changes sign between their centres, so the
// interface stays within half a voxel of the voxel faces while remaining smooth
// wherever the smoothing did not erase anything.
inline void tn_preserve_voxel(TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab,
                              TN_G const int* bl_slot, TN_G float* phi, float m, int i, int j, int k) {
    const int b = tn_brick_of(d, i, j, k);

    if (bl_slot[b] < 0) {   // uniform brick: exact indicators already
        return;
    }

    const int a = L[i + (size_t)d.nx * (j + (size_t)d.ny * k)];
    const size_t t = (size_t)(i - ((i >> 3) << 3)) + TN_BS * ((j - ((j >> 3) << 3)) + TN_BS * (k - ((k >> 3) << 3)));
    const int n = bl_cnt[b] < TN_BL ? bl_cnt[b] : TN_BL;
    int sa = -1;
    float mo = 0.0f;

    for (int s = 0; s < n; ++s) {
        const float v = phi[(size_t)(bl_slot[b] + s) * TN_SLOT + t];

        if (bl_lab[b * TN_BL + s] == a) {
            sa = bl_slot[b] + s;
        } else if (v > mo) {
            mo = v;
        }
    }

    if (sa >= 0 && phi[(size_t)sa * TN_SLOT + t] < mo + m) {
        phi[(size_t)sa * TN_SLOT + t] = mo + m;
    }
}

// ---- curvature and sizing (per voxel) -------------------------------------------
// Largest principal curvature |kappa| of the level set of psi = phi_a - phi_b
// through voxel (i,j,k), from central differences (grid mm): with n = g/|g| and
// P = I - n n^T, the tangent Hessian P H P / |g| has eigenvalues (k1, k2) and
// zero; T = k1 + k2 = tr(PHP)/|g|, F = k1^2 + k2^2 = |PHP|_F^2 / |g|^2, so
// max|k| = (|T| + sqrt(max(0, 2F - T^2))) / 2.
// `st` is the difference stride (voxels). The field must be smooth enough for a
// second derivative: sizing uses a wider sigma (tn_grid.cpp) than the interface
// fields, since a sigma = 1 indicator still carries the voxel staircase (curvature
// hot spots at the stair steps of a radius-34 ball, h down to 1/3 of flat).
inline float tn_kappa_max(TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab,
                          TN_G const int* bl_slot, TN_G const float* phi, int a, int b, int i, int j, int k, int st) {
    float f[27];

    for (int n = 0; n < 27; ++n) {
        const int dx = (n % 3 - 1) * st, dy = ((n / 3) % 3 - 1) * st, dz = (n / 9 - 1) * st;
        f[n] = tn_phi_vox(d, L, bl_cnt, bl_lab, bl_slot, phi, a, i + dx, j + dy, k + dz) -
               tn_phi_vox(d, L, bl_cnt, bl_lab, bl_slot, phi, b, i + dx, j + dy, k + dz);
    }

#define TN_F(dx, dy, dz) f[((dx) + 1) + 3 * ((dy) + 1) + 9 * ((dz) + 1)]
    const float hx = d.vx * st, hy = d.vy * st, hz = d.vz * st;
    float g[3], H[9];
    g[0] = (TN_F(1, 0, 0) - TN_F(-1, 0, 0)) / (2 * hx);
    g[1] = (TN_F(0, 1, 0) - TN_F(0, -1, 0)) / (2 * hy);
    g[2] = (TN_F(0, 0, 1) - TN_F(0, 0, -1)) / (2 * hz);
    H[0] = (TN_F(1, 0, 0) - 2 * TN_F(0, 0, 0) + TN_F(-1, 0, 0)) / (hx * hx);
    H[4] = (TN_F(0, 1, 0) - 2 * TN_F(0, 0, 0) + TN_F(0, -1, 0)) / (hy * hy);
    H[8] = (TN_F(0, 0, 1) - 2 * TN_F(0, 0, 0) + TN_F(0, 0, -1)) / (hz * hz);
    H[1] = H[3] = (TN_F(1, 1, 0) - TN_F(1, -1, 0) - TN_F(-1, 1, 0) + TN_F(-1, -1, 0)) / (4 * hx * hy);
    H[2] = H[6] = (TN_F(1, 0, 1) - TN_F(1, 0, -1) - TN_F(-1, 0, 1) + TN_F(-1, 0, -1)) / (4 * hx * hz);
    H[5] = H[7] = (TN_F(0, 1, 1) - TN_F(0, 1, -1) - TN_F(0, -1, 1) + TN_F(0, -1, -1)) / (4 * hy * hz);
#undef TN_F
    const float gn = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);

    if (gn < 1e-6f) {
        return 0.0f;
    }

    float n[3], PHP[9];
    n[0] = g[0] / gn;
    n[1] = g[1] / gn;
    n[2] = g[2] / gn;
    float Pm[9];

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            Pm[3 * r + c] = (r == c ? 1.0f : 0.0f) - n[r] * n[c];
        }

    // PHP = P * H * P
    float HP[9];

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            HP[3 * r + c] = H[3 * r] * Pm[c] + H[3 * r + 1] * Pm[3 + c] + H[3 * r + 2] * Pm[6 + c];
        }

    float T = 0.0f, F = 0.0f;

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            PHP[3 * r + c] = Pm[3 * r] * HP[c] + Pm[3 * r + 1] * HP[3 + c] + Pm[3 * r + 2] * HP[6 + c];
            F += PHP[3 * r + c] * PHP[3 * r + c];
        }

    T = (PHP[0] + PHP[4] + PHP[8]) / gn;
    F /= gn * gn;
    const float disc = 2.0f * F - T * T;
    return 0.5f * (fabs(T) + sqrt(disc > 0.0f ? disc : 0.0f));
}

// Initial size at voxel (i,j,k): the label's own size (hlab[l], 0 = hbase), and
// where the voxel is near an interface (a competing label b with psi within
// 0.45, i.e. inside the transition band), also min over the curvature bound
// 1/(K |kappa|) and the neighbour label's size. Clamped to [hmin, hmax]. Exterior
// voxels get hmax (never meshed; they only feed the gradient limiter).
inline float tn_size_voxel(TnDims d, TN_G const ushort* L, TN_G const int* bl_cnt, TN_G const ushort* bl_lab,
                           TN_G const int* bl_slot, TN_G const float* phi, TN_G const float* hlab, int nlab,
                           float hbase, float hmin, float hmax, float K, int i, int j, int k) {
    const int a = L[i + (size_t)d.nx * (j + (size_t)d.ny * k)];
    float h = (a < nlab && hlab[a] > 0.0f) ? hlab[a] : hbase;

    if (a == 0) {
        return hmax;
    }

    const int bk = tn_brick_of(d, i, j, k);

    if (bl_slot[bk] >= 0) {   // near some interface: the strongest competitor
        const float pa = tn_phi_vox(d, L, bl_cnt, bl_lab, bl_slot, phi, a, i, j, k);
        int best = -1;
        float pb = -1.0f;

        for (int s = 0; s < bl_cnt[bk] && s < TN_BL; ++s) {
            const int l = bl_lab[bk * TN_BL + s];

            if (l == a) {
                continue;
            }

            const float v = tn_phi_vox(d, L, bl_cnt, bl_lab, bl_slot, phi, l, i, j, k);

            if (v > pb) {
                pb = v;
                best = l;
            }
        }

        if (best >= 0 && pa - pb < 0.45f) {
            if (best < nlab && hlab[best] > 0.0f) {
                h = fmin(h, hlab[best]);
            }

            const float kap = tn_kappa_max(d, L, bl_cnt, bl_lab, bl_slot, phi, a, best, i, j, k, 1);

            if (kap > 0.0f) {
                h = fmin(h, 1.0f / (K * kap));
            }
        }
    }

    return h < hmin ? hmin : (h > hmax ? hmax : h);
}

// One Jacobi gradient-limiting sweep at voxel (i,j,k): h <= h(n) + g |x - x_n|
// over the 26 neighbours (the discrete form of |grad h| <= g). Returns 1 if the
// value dropped.
inline int tn_limit_voxel(TnDims d, TN_G const float* h, TN_G float* hn, float g, int i, int j, int k) {
    const size_t v = i + (size_t)d.nx * (j + (size_t)d.ny * k);
    float m = h[v];

    for (int n = 0; n < 27; ++n) {
        const int dx = n % 3 - 1, dy = (n / 3) % 3 - 1, dz = n / 9 - 1;
        const int x = i + dx, y = j + dy, z = k + dz;

        if (n == 13 || x < 0 || y < 0 || z < 0 || x >= d.nx || y >= d.ny || z >= d.nz) {
            continue;
        }

        const float ex = dx * d.vx, ey = dy * d.vy, ez = dz * d.vz;
        m = fmin(m, h[x + (size_t)d.nx * (y + (size_t)d.ny * z)] + g * sqrt(ex * ex + ey * ey + ez * ez));
    }

    hn[v] = m;
    return m < h[v];
}

// 256 log-spaced grades between hmin and hmax
inline uchar tn_grade(float h, float hmin, float hmax) {
    if (hmax <= hmin * 1.0001f) {
        return 0;
    }

    float t = log(h / hmin) / log(hmax / hmin);
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return (uchar)(t * 255.0f + 0.5f);
}

inline float tn_grade_size(int k, float hmin, float hmax) {
    return hmin * pow(hmax / hmin, (float)k / 255.0f);
}

#ifdef __OPENCL_VERSION__
// ------------------------------------------------------------------ kernels ---
__kernel void g_brick_labels(__global const ushort* L, int nx, int ny, int nz, int nbx, int nby, int nbz, int R,
                             __global int* bl_cnt, __global ushort* bl_lab) {
    const int b = get_global_id(0);

    if (b >= nbx * nby * nbz) {
        return;
    }

    ushort lab[TN_BL];
    const int n = tn_brick_labels(L, nx, ny, nz, b % nbx, (b / nbx) % nby, b / (nbx * nby), R, lab);
    bl_cnt[b] = n;

    for (int i = 0; i < TN_BL; ++i) {
        bl_lab[b * TN_BL + i] = i < n ? lab[i] : (ushort)0xFFFF;
    }
}

// One work-group per slot: the (8+2R)^3 label tile of the brick is staged in
// local memory as the 0/1 indicator of the slot's label, then blurred along x, y
// and z (separable Gaussian), and the 8^3 centre written out. R <= 6.
__kernel void g_smooth(__global const ushort* L, int nx, int ny, int nz, int nbx, int nby,
                       __global const int* slot_brick, __global const ushort* slot_label, float sigma, int R,
                       __global float* phi, __local float* A, __local float* B) {
    const int s = get_group_id(0);
    const int lid = get_local_id(0), ls = get_local_size(0);
    const int b = slot_brick[s];
    const int l = slot_label[s];
    const int bx = b % nbx, by = (b / nbx) % nby, bz = b / (nbx * nby);
    const int T = TN_BS + 2 * R, T3 = T * T * T;
    const int x0 = bx * TN_BS - R, y0 = by * TN_BS - R, z0 = bz * TN_BS - R;
    float w[2 * 6 + 1];
    float ws = 0.0f;

    for (int d = -R; d <= R; ++d) {
        w[d + R] = tn_gauss(d, sigma);
        ws += w[d + R];
    }

    for (int d = 0; d <= 2 * R; ++d) {
        w[d] /= ws;
    }

    for (int t = lid; t < T3; t += ls) {
        const int x = t % T, y = (t / T) % T, z = t / (T * T);
        A[t] = tn_label_at(L, nx, ny, nz, x0 + x, y0 + y, z0 + z) == l ? 1.0f : 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = lid; t < T3; t += ls) {   // x
        const int x = t % T, y = (t / T) % T, z = t / (T * T);
        float v = 0.0f;

        for (int d = -R; d <= R; ++d) {
            const int xx = x + d < 0 ? 0 : (x + d >= T ? T - 1 : x + d);
            v += w[d + R] * A[xx + T * (y + T * z)];
        }

        B[t] = v;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = lid; t < T3; t += ls) {   // y
        const int x = t % T, y = (t / T) % T, z = t / (T * T);
        float v = 0.0f;

        for (int d = -R; d <= R; ++d) {
            const int yy = y + d < 0 ? 0 : (y + d >= T ? T - 1 : y + d);
            v += w[d + R] * B[x + T * (yy + T * z)];
        }

        A[t] = v;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = lid; t < TN_SLOT; t += ls) {   // z, centre only
        const int x = t % TN_BS + R, y = (t / TN_BS) % TN_BS + R, z = t / (TN_BS * TN_BS) + R;
        float v = 0.0f;

        for (int d = -R; d <= R; ++d) {
            v += w[d + R] * A[x + T * (y + T * (z + d))];
        }

        phi[(size_t)s * TN_SLOT + t] = v;
    }
}

__kernel void g_size(__global const ushort* L, int nx, int ny, int nz, int nbx, int nby, int nbz, float vx, float vy,
                     float vz, __global const int* bl_cnt, __global const ushort* bl_lab, __global const int* bl_slot,
                     __global const float* phi, __global const float* hlab, int nlab, float hbase, float hmin,
                     float hmax, float K, __global float* h) {
    const size_t v = get_global_id(0);

    if (v >= (size_t)nx * ny * nz) {
        return;
    }

    TnDims d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.nbx = nbx;
    d.nby = nby;
    d.nbz = nbz;
    d.vx = vx;
    d.vy = vy;
    d.vz = vz;
    const int i = (int)(v % nx), j = (int)((v / nx) % ny), k = (int)(v / ((size_t)nx * ny));
    h[v] = tn_size_voxel(d, L, bl_cnt, bl_lab, bl_slot, phi, hlab, nlab, hbase, hmin, hmax, K, i, j, k);
}

__kernel void g_limit(int nx, int ny, int nz, float vx, float vy, float vz, __global const float* h,
                      __global float* hn, float g, __global int* changed) {
    const size_t v = get_global_id(0);

    if (v >= (size_t)nx * ny * nz) {
        return;
    }

    TnDims d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.nbx = d.nby = d.nbz = 0;
    d.vx = vx;
    d.vy = vy;
    d.vz = vz;
    const int i = (int)(v % nx), j = (int)((v / nx) % ny), k = (int)(v / ((size_t)nx * ny));

    if (tn_limit_voxel(d, h, hn, g, i, j, k)) {
        *changed = 1;
    }
}

__kernel void g_preserve(__global const ushort* L, int nx, int ny, int nz, int nbx, int nby, int nbz,
                         __global const int* bl_cnt, __global const ushort* bl_lab, __global const int* bl_slot,
                         __global float* phi, float m) {
    const size_t v = get_global_id(0);

    if (v >= (size_t)nx * ny * nz) {
        return;
    }

    TnDims d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.nbx = nbx;
    d.nby = nby;
    d.nbz = nbz;
    d.vx = d.vy = d.vz = 1.0f;
    tn_preserve_voxel(d, L, bl_cnt, bl_lab, bl_slot, phi, m, (int)(v % nx), (int)((v / nx) % ny),
                      (int)(v / ((size_t)nx * ny)));
}

__kernel void g_quantize(__global const float* h, int n, float hmin, float hmax, __global uchar* grade) {
    const int v = get_global_id(0);

    if (v < n) {
        grade[v] = tn_grade(h[v], hmin, hmax);
    }
}
#endif
