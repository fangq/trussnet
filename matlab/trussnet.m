function [node, elem, face, info] = trussnet(vol, varargin)
    %
    % [node, elem, face, info] = trussnet(vol)
    % [node, elem, face, info] = trussnet(vol, opt)
    % [node, elem, face, info] = trussnet(vol, 'name1', value1, 'name2', value2, ...)
    %
    % GPU particle (truss) multi-label / gray-scale / tissue-probability tetrahedral mesher;
    % a 2-D image gives a triangle mesh (the same method, see "2-D images" below).
    %
    % author: Qianqian Fang (q.fang at neu.edu)
    %
    % input:
    %     vol: a 3-D array, either
    %          - integer labels: 0 = exterior (never meshed), 1..N = tissues; every
    %            label is meshed with conforming shared interfaces, or
    %          - a gray-scale intensity, when opt.thresholds is given: the label of
    %            a voxel is the number of thresholds <= its intensity, and the
    %            interfaces are the (sub-voxel) iso-surfaces vol = thresholds(k);
    %          or a 4-D (x, y, z, class) tissue-probability map (TPM): the labels
    %          are the argmax of the classes (enclosed exterior pockets filled);
    %          or a file name (.nii, .nii.gz, .jnii, .bnii: labels, gray-scale or a
    %          4-D TPM), meshed in the file's world coordinates
    %     opt: a struct, or name/value pairs (names are case-insensitive, '_' is ignored):
    %          sizing (mm):
    %            size       default element size (default 3 x voxel)
    %            hmin/hmax  smallest / largest element size (default size/3, size)
    %            lsize      per-label size: a vector (lsize(l) = size of label l, 0 = default)
    %                       or an N x 2 [label size] matrix
    %            sizing     a user sizing (mm): an array of vol's (spatial) size, a sizing
    %                       field, 0 = the automatic size at that voxel; or a vector, one
    %                       size per label (N, or N+1 from label 0), per threshold level
    %                       or per TPM channel, 0 = default
    %            K          elements per radian of curvature (default 3)
    %            grad       sizing gradient limit (default 0.3)
    %            sigma      indicator smoothing, voxels (default 1)
    %            thick      thin layers: h <= local thickness / thick (0 = off)
    %          gray-scale:
    %            thresholds  iso-values, e.g. 2.5 or [2 2.5 3 3.5]
    %            graysigma   Gaussian pre-smoothing of the intensity, voxels (default 0)
    %          tissue probabilities (4-D vol):
    %            tpmexterior  exterior channels, 1-based (default: for a file, the ones
    %                         named background / air; else none: exterior = 1 - sum)
    %            tpmmap       the label of each channel (0 = exterior; shared = summed)
    %            tpmspm6      1: merge the 18 siamize classes to SPM6 (GM WM CSF Bone Soft)
    %            tpmsigma     Gaussian smoothing of the probabilities, voxels (default 0)
    %            tpmholes     1: keep the enclosed exterior pockets (default 0: filled)
    %            tpmfields    1: interfaces from the probabilities (p_a = p_b; with
    %                         sigma = 0 unsmoothed, the most accurate for a smooth TPM)
    %          device:
    %            gpu        1: run on the first OpenCL GPU (falls back to the CPU); 0: CPU (default)
    %            gpuid      1-based OpenCL device index (implies gpu = 1)
    %          quality / tessellation:
    %            reratio    max radius-edge ratio (alias q; default 2, 0 = off)
    %            opt        sliver repair: flips, collapses, Steiner points (default 1)
    %            smooth     quality-guarded smoothing passes (default 5)
    %            repair     max conformity repair rounds (default 6)
    %          relaxation:
    %            iters      max relaxation iterations (default 500)
    %            fscale, fsurf, dt, snap, nseed, jseed, corners, trap ('smooth'|'voxel')
    %          coordinates:
    %            voxelsize  [dx dy dz] or a scalar, mm (default 1)
    %            affine     4x4 voxel (0-based i,j,k) -> world matrix (e.g. a NIfTI
    %                       header's; the voxel size, unless given, from its columns);
    %                       default: node coordinates in MATLAB index space
    %                       scaled by voxelsize (voxel (i,j,k) centre at [i j k].*voxelsize)
    %          verbose    1: print the per-stage progress and statistics
    %
    % 2-D images: a 2-D vol (labels, or gray-scale with thresholds) is meshed into
    %     triangles: node N x 2, elem M x 4 [v1 v2 v3 label] (1-based, counter-clockwise),
    %     face P x 4 [v1 v2 inner outer] (the boundary and interface edges, the inner
    %     region on the left), info (nodes, tris, junctions, badedges, spanning, minangle,
    %     qmin/qp5/qmedian with q = 4 sqrt(3) A / sum(l^2), labelarea / labelpixels, ms_*).
    %     Options: size hmin hmax lsize sizing K grad sigma (default 0.5) thresholds
    %     graysigma nseed iters fscale fsurf dt snap repair smooth pixelsize affine (2x3/3x3).
    %
    % output:
    %     node: N x 3 node coordinates
    %     elem: M x 5 [v1 v2 v3 v4 label], 1-based
    %     face: P x 5 [v1 v2 v3 inner outer], 1-based: the exterior surface (outer = 0)
    %           and every interface between two labels (once), normals pointing from
    %           the inner label to the outer one (computed only when requested)
    %     info: a struct: counts, conformity (badfaces/badedges/spanning, 0 = conforming),
    %           quality (mindihedral, joeliumin/p5/median), timings (ms_*), usedgpu,
    %           tpmfilled (TPM: enclosed exterior voxels filled)
    %
    % example:
    %     [xi, yi, zi] = ndgrid(1:60);
    %     vol = uint8(sqrt((xi-30).^2 + (yi-30).^2 + (zi-30).^2) < 25);
    %     vol(sqrt((xi-30).^2 + (yi-30).^2 + (zi-30).^2) < 12) = 2;
    %     [node, elem, face] = trussnet(vol, 'size', 3, 'gpu', 1);
    %     plotmesh(node, face(:, 1:4));        % iso2mesh
    %
    % -- this function is part of trussnet (https://github.com/fangq/trussnet)
    % License: GPL-3.0-or-later, Copyright (C) 2026 Qianqian Fang <q.fang at neu.edu>
    %

    if nargin < 1
        error('trussnet: usage: [node, elem, face, info] = trussnet(vol, opt)');
    end
    if ~ischar(vol) && ((~isnumeric(vol) && ~islogical(vol)) || ndims(vol) > 4)
        error('trussnet: vol must be a 2-D image, a 3-D (labels / gray-scale) or 4-D (TPM) array, or a file name');
    end

    opt = struct();
    if numel(varargin) == 1 && isstruct(varargin{1})
        opt = varargin{1};
    elseif numel(varargin) >= 1
        if mod(numel(varargin), 2) ~= 0
            error('trussnet: options must be a struct or name/value pairs');
        end
        for i = 1:2:numel(varargin)
            if ~ischar(varargin{i})
                error('trussnet: option names must be strings');
            end
            opt.(varargin{i}) = varargin{i + 1};
        end
    end

    if islogical(vol)
        vol = uint8(vol);
    end

    if nargout > 3
        [node, elem, face, info] = trussnet_mex(vol, opt);
    elseif nargout > 2
        [node, elem, face] = trussnet_mex(vol, opt);
    else
        [node, elem] = trussnet_mex(vol, opt);
    end
