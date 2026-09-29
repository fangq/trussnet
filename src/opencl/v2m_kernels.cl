// SPDX-License-Identifier: GPL-3.0-or-later
//
// v2mesh -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// v2m_kernels.cl -- device kernels of the relaxation (stages 4-9) and the grid
// extras; the program is built from
//   #define V2M_G __global + v2m_grid_body.cl + v2m_seed_body.cl + v2m_particle_body.cl
//   + this file.
// Every kernel is one work-item per voxel or per node over the shared bodies,
// so the device path computes exactly what the OpenMP reference does.

#define V2M_DIMS_ARGS int nx, int ny, int nz, int nbx, int nby, int nbz, float vx, float vy, float vz
#define V2M_MKDIMS(d)  \
    V2mDims d;         \
    d.nx = nx;        \
    d.ny = ny;        \
    d.nz = nz;        \
    d.nbx = nbx;      \
    d.nby = nby;      \
    d.nbz = nbz;      \
    d.vx = vx;        \
    d.vy = vy;        \
    d.vz = vz
#define V2M_KFIELD_ARGS __global const ushort* L, __global const int* bl_cnt, __global const ushort* bl_lab, \
    __global const int* bl_slot, __global const float* phi, __global const float* gI, __global const float* gTW, int gm

// ---- grid extras -----------------------------------------------------------------
__kernel void g_thick(__global const ushort* L, V2M_DIMS_ARGS, __global const int* bl_cnt,
                      __global const ushort* bl_lab, __global const int* bl_slot, __global const float* phi,
                      float sigma, float thick, float floor_mm, float vmin, __global float* h) {
    const size_t v = get_global_id(0);

    if (v >= (size_t)nx * ny * nz) {
        return;
    }

    V2M_MKDIMS(d);
    const int i = (int)(v % nx), j = (int)((v / nx) % ny), k = (int)(v / ((size_t)nx * ny));
    const float t = v2m_thick_voxel(d, L, bl_cnt, bl_lab, bl_slot, phi, sigma, 2, i, j, k);

    if (t < 1e29f) {
        h[v] = fmin(h[v], fmax(floor_mm, t * vmin / thick));
    }
}

// ---- exclusive scan (Blelloch, 2 elements per work-item, local size V2M_SCAN_LS) ----
#define V2M_SCAN_LS 256
#define V2M_SCAN_B (2 * V2M_SCAN_LS)

__kernel void k_scan_block(__global const int* in, __global int* out, int n, __global int* sums) {
    __local int s[V2M_SCAN_B];
    const int lid = get_local_id(0), g = get_group_id(0);
    const int base = g * V2M_SCAN_B;
    s[2 * lid] = base + 2 * lid < n ? in[base + 2 * lid] : 0;
    s[2 * lid + 1] = base + 2 * lid + 1 < n ? in[base + 2 * lid + 1] : 0;
    int off = 1;

    for (int dd = V2M_SCAN_B >> 1; dd > 0; dd >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < dd) {
            const int ai = off * (2 * lid + 1) - 1, bi = off * (2 * lid + 2) - 1;
            s[bi] += s[ai];
        }

        off <<= 1;
    }

    if (lid == 0) {
        sums[g] = s[V2M_SCAN_B - 1];
        s[V2M_SCAN_B - 1] = 0;
    }

    for (int dd = 1; dd < V2M_SCAN_B; dd <<= 1) {
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
        out[i] += sums[i / V2M_SCAN_B];
    }
}

// ---- hash (counting sort of node ids by bin key) ----------------------------------
__kernel void k_keys(V2mHash H, V2M_DIMS_ARGS, __global const float* hvox, __global const float* P, int n, float t,
                     float skin, __global int* key, __global int* cnt, __global float* hn) {
    const int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    V2M_MKDIMS(d);
    const float h = v2m_h_at(d, hvox, P[3 * i], P[3 * i + 1], P[3 * i + 2]);
    hn[i] = h;
    const int k = v2m_bin_key(&H, v2m_level_of(&H, (t + skin) * h), P[3 * i], P[3 * i + 1], P[3 * i + 2]);
    key[i] = k;
    atomic_inc(&cnt[k]);
}

__kernel void k_scatter(__global const int* key, int n, __global int* cursor, __global int* sorted) {
    const int i = get_global_id(0);

    if (i < n) {
        sorted[atomic_inc(&cursor[key[i]])] = i;
    }
}

// positions + sizes gathered into bin order (see v2m_neighbors)
__kernel void k_gather(__global const int* sorted, __global const float* P, __global const float* hn, int n,
                       __global float* Ps) {
    const int s = get_global_id(0);

    if (s < n) {
        const int j = sorted[s];
        Ps[4 * s] = P[3 * j];
        Ps[4 * s + 1] = P[3 * j + 1];
        Ps[4 * s + 2] = P[3 * j + 2];
        Ps[4 * s + 3] = hn[j];
    }
}

__kernel void k_neighbors(V2mHash H, __global const float* hn, __global const float* P, __global const ushort* lab,
                          __global const uchar* typ, __global const int* cstart, __global const int* sorted,
                          __global const float* Ps, float t, float skin, int n, __global int* nbr, __global int* nnb) {
    __local float lds[V2M_NBR_WG * V2M_K];   // launched with V2M_NBR_WG work-items
    __local int lids[V2M_NBR_WG * V2M_K];
    const int i = get_global_id(0), lid = get_local_id(0);

    if (i >= n) {
        return;
    }

    v2m_neighbors(&H, hn, P, lab, typ, cstart, sorted, Ps, t, skin, i, nbr, nnb, lds + lid, lids + lid, V2M_NBR_WG);
}

// ---- force / move -----------------------------------------------------------------
__kernel void k_force(__global const float* hn, __global const float* P, __global const uchar* typ,
                      __global const int* nbr, __global const int* nnb, float fscale, float fsurf, int n,
                      __global float* F) {
    const int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    v2m_force(hn, P, typ, nbr, nnb, fscale, fsurf, i, F + 4 * i);
}

__kernel void k_move(V2M_KFIELD_ARGS, V2M_DIMS_ARGS, __global const float* hvox, __global const float* F, float dt,
                     float maxstep, float snap, int voxmode, int n, __global float* P, __global const ushort* lab,
                     __global uchar* typ, __global ushort* part, __global float* mv, __global float* hn,
                     __global const int* order, int fire, __global float* V, float fdt, float falpha,
                     __global float* pw) {
    // nodes grouped by type (order): an interface node's projection costs ~10x an
    // interior node's step, and a warp waits for its slowest lane. v2m_move touches
    // only node i, so the order does not change the result.
    const int gid = get_global_id(0);

    if (gid >= n) {
        return;
    }

    const int i = order[gid];
    V2M_MKDIMS(d);
    mv[i] = v2m_move(d, L, bl_cnt, bl_lab, bl_slot, phi, gI, gTW, gm, hvox, F, dt, maxstep, snap, voxmode, i, P, lab, typ, part, hn,
                    fire, V, fdt, falpha, pw);
}

// ---- per-iteration reductions + Verlet bookkeeping --------------------------------
// stats[0..31]: log2 histogram of |dp|/h; stats[32]: max |dp|/h (float bits);
// stats[33]: nodes past skin/2 since the build. A node past a full skin (a
// surface snap) refreshes its own list here, as the host path does.
// fire: psum[group] = the group's sum of pw (FIRE's power, summed on the host).
__kernel void k_stats(V2mHash H, __global const float* hn, __global const float* P,
                      __global float* P0, __global const float* mv, __global const ushort* lab,
                      __global const uchar* typ, __global const int* cstart, __global const int* sorted,
                      __global const float* Ps, float t,
                      float skin, int n, __global int* nbr, __global int* nnb, __global int* stats,
                      int fire, __global const float* pw, __global float* psum) {
    __local int lh[32];
    __local float lp[V2M_STATS_WG];
    __local int lmax, lhalf;
    __local float lds[V2M_STATS_WG * V2M_K];   // launched with V2M_STATS_WG work-items
    __local int lids[V2M_STATS_WG * V2M_K];
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
            v2m_neighbors(&H, hn, P, lab, typ, cstart, sorted, Ps, t, skin, i, nbr, nnb, lds + lid, lids + lid,
                         V2M_STATS_WG);
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

    if (fire) {   // uniform per launch: every work-item takes the barriers
        lp[lid] = i < n ? pw[i] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int w = V2M_STATS_WG / 2; w > 0; w >>= 1) {
            if (lid < w) {
                lp[lid] += lp[lid + w];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            psum[get_group_id(0)] = lp[0];
        }
    }
}
