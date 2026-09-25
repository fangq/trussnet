// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_kernels.cl -- device kernels of the relaxation (stages 4-9) and the grid
// extras; the program is built from
//   #define TN_G __global + tn_grid_body.cl + tn_seed_body.cl + tn_particle_body.cl
//   + this file.
// Every kernel is one work-item per voxel or per node over the shared bodies,
// so the device path computes exactly what the OpenMP reference does.

#define TN_DIMS_ARGS int nx, int ny, int nz, int nbx, int nby, int nbz, float vx, float vy, float vz
#define TN_MKDIMS(d)  \
    TnDims d;         \
    d.nx = nx;        \
    d.ny = ny;        \
    d.nz = nz;        \
    d.nbx = nbx;      \
    d.nby = nby;      \
    d.nbz = nbz;      \
    d.vx = vx;        \
    d.vy = vy;        \
    d.vz = vz
#define TN_KFIELD_ARGS __global const ushort* L, __global const int* bl_cnt, __global const ushort* bl_lab, \
    __global const int* bl_slot, __global const float* phi, __global const float* gI, __global const float* gTW, int gm

// ---- grid extras -----------------------------------------------------------------
__kernel void g_thick(__global const ushort* L, TN_DIMS_ARGS, __global const int* bl_cnt,
                      __global const ushort* bl_lab, __global const int* bl_slot, __global const float* phi,
                      float sigma, float thick, float floor_mm, float vmin, __global float* h) {
    const size_t v = get_global_id(0);

    if (v >= (size_t)nx * ny * nz) {
        return;
    }

    TN_MKDIMS(d);
    const int i = (int)(v % nx), j = (int)((v / nx) % ny), k = (int)(v / ((size_t)nx * ny));
    const float t = tn_thick_voxel(d, L, bl_cnt, bl_lab, bl_slot, phi, sigma, 2, i, j, k);

    if (t < 1e29f) {
        h[v] = fmin(h[v], fmax(floor_mm, t * vmin / thick));
    }
}

// ---- exclusive scan (Blelloch, 2 elements per work-item, local size TN_SCAN_LS) ----
#define TN_SCAN_LS 256
#define TN_SCAN_B (2 * TN_SCAN_LS)

__kernel void k_scan_block(__global const int* in, __global int* out, int n, __global int* sums) {
    __local int s[TN_SCAN_B];
    const int lid = get_local_id(0), g = get_group_id(0);
    const int base = g * TN_SCAN_B;
    s[2 * lid] = base + 2 * lid < n ? in[base + 2 * lid] : 0;
    s[2 * lid + 1] = base + 2 * lid + 1 < n ? in[base + 2 * lid + 1] : 0;
    int off = 1;

    for (int dd = TN_SCAN_B >> 1; dd > 0; dd >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < dd) {
            const int ai = off * (2 * lid + 1) - 1, bi = off * (2 * lid + 2) - 1;
            s[bi] += s[ai];
        }

        off <<= 1;
    }

    if (lid == 0) {
        sums[g] = s[TN_SCAN_B - 1];
        s[TN_SCAN_B - 1] = 0;
    }

    for (int dd = 1; dd < TN_SCAN_B; dd <<= 1) {
        off >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < dd) {
            const int ai = off * (2 * lid + 1) - 1, bi = off * (2 * lid + 2) - 1;
            const int t = s[ai];
            s[ai] = s[bi];
            s[bi] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (base + 2 * lid < n) {
        out[base + 2 * lid] = s[2 * lid];
    }

    if (base + 2 * lid + 1 < n) {
        out[base + 2 * lid + 1] = s[2 * lid + 1];
    }
}

__kernel void k_scan_add(__global int* out, int n, __global const int* sums) {
    const int i = get_global_id(0);

    if (i < n) {
        out[i] += sums[i / TN_SCAN_B];
    }
}

// ---- hash (counting sort of node ids by bin key) ----------------------------------
__kernel void k_keys(TnHash H, TN_DIMS_ARGS, __global const float* hvox, __global const float* P, int n, float t,
                     float skin, __global int* key, __global int* cnt, __global float* hn) {
    const int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    TN_MKDIMS(d);
    const float h = tn_h_at(d, hvox, P[3 * i], P[3 * i + 1], P[3 * i + 2]);
    hn[i] = h;
    const int k = tn_bin_key(&H, tn_level_of(&H, (t + skin) * h), P[3 * i], P[3 * i + 1], P[3 * i + 2]);
    key[i] = k;
    atomic_inc(&cnt[k]);
}

__kernel void k_scatter(__global const int* key, int n, __global int* cursor, __global int* sorted) {
    const int i = get_global_id(0);

    if (i < n) {
        sorted[atomic_inc(&cursor[key[i]])] = i;
    }
}

__kernel void k_neighbors(TnHash H, __global const float* hn, __global const float* P, __global const ushort* lab,
                          __global const uchar* typ, __global const int* cstart, __global const int* sorted, float t,
                          float skin, int n, __global int* nbr, __global int* nnb) {
    const int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    tn_neighbors(&H, hn, P, lab, typ, cstart, sorted, t, skin, i, nbr, nnb);
}

// ---- force / move -----------------------------------------------------------------
__kernel void k_force(__global const float* hn, __global const float* P, __global const uchar* typ,
                      __global const int* nbr, __global const int* nnb, float fscale, float fsurf, int n,
                      __global float* F) {
    const int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    tn_force(hn, P, typ, nbr, nnb, fscale, fsurf, i, F + 4 * i);
}

__kernel void k_move(TN_KFIELD_ARGS, TN_DIMS_ARGS, __global const float* hvox, __global const float* F, float dt,
                     float maxstep, float snap, int voxmode, int n, __global float* P, __global const ushort* lab,
                     __global uchar* typ, __global ushort* part, __global float* mv, __global float* hn) {
    const int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    TN_MKDIMS(d);
    mv[i] = tn_move(d, L, bl_cnt, bl_lab, bl_slot, phi, gI, gTW, gm, hvox, F, dt, maxstep, snap, voxmode, i, P, lab, typ, part, hn);
}

// ---- per-iteration reductions + Verlet bookkeeping --------------------------------
// stats[0..31]: log2 histogram of |dp|/h; stats[32]: max |dp|/h (float bits);
// stats[33]: nodes past skin/2 since the build. A node past a full skin (a
// surface snap) refreshes its own list here, as the host path does.
__kernel void k_stats(TnHash H, __global const float* hn, __global const float* P,
                      __global float* P0, __global const float* mv, __global const ushort* lab,
                      __global const uchar* typ, __global const int* cstart, __global const int* sorted, float t,
                      float skin, int n, __global int* nbr, __global int* nnb, __global int* stats) {
    __local int lh[32];
    __local int lmax, lhalf;
    const int i = get_global_id(0), lid = get_local_id(0);

    if (lid < 32) {
        lh[lid] = 0;
    }

    if (lid == 0) {
        lmax = 0;
        lhalf = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < n) {
        const float m = mv[i];
        const int b = m > 0.0f ? clamp(20 + (int)floor(log2(m)), 0, 31) : 0;
        atomic_inc(&lh[b]);
        atomic_max(&lmax, as_int(m));
        const float ex = P[3 * i] - P0[3 * i], ey = P[3 * i + 1] - P0[3 * i + 1], ez = P[3 * i + 2] - P0[3 * i + 2];
        const float h = hn[i];
        const float m2 = ex * ex + ey * ey + ez * ez, s2 = skin * skin * h * h;

        if (m2 > s2) {
            tn_neighbors(&H, hn, P, lab, typ, cstart, sorted, t, skin, i, nbr, nnb);
            P0[3 * i] = P[3 * i];
            P0[3 * i + 1] = P[3 * i + 1];
            P0[3 * i + 2] = P[3 * i + 2];
        } else if (m2 > 0.25f * s2) {
            atomic_inc(&lhalf);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 32 && lh[lid]) {
        atomic_add(&stats[lid], lh[lid]);
    }

    if (lid == 0) {
        atomic_max(&stats[32], lmax);
        atomic_add(&stats[33], lhalf);
    }
}
