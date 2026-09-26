function nfail = run_trussnet_tests()
    %
    % nfail = run_trussnet_tests()
    %
    % Unit tests of the trussnet MEX binding; runs in MATLAB and GNU Octave (no
    % test framework needed). Returns the number of failed tests.
    %
    %   matlab -batch "addpath('matlab/tests'); exit(run_trussnet_tests() > 0)"
    %   octave --eval "addpath('matlab/tests'); exit(run_trussnet_tests() > 0)"
    %
    % -- this function is part of trussnet (https://github.com/fangq/trussnet)
    % License: GPL-3.0-or-later, Copyright (C) 2026 Qianqian Fang <q.fang at neu.edu>
    %

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..'));

    tests = {@test_outputs, @test_one_based, @test_conformity, @test_face_orientation, ...
             @test_node_space, @test_options_struct_and_pairs, @test_voxelsize, @test_affine, ...
             @test_lsize, @test_logical_input, @test_gray_single, @test_gray_multi, ...
             @test_deterministic, @test_errors, @test_gpu, @test_tpm, @test_tpm_no_exterior, ...
             @test_tpm_map_holes, @test_tpm_raw_fields, @test_tpm_file, @test_sizing_field, ...
             @test_sizing_vectors, @test_2d, @test_2d_gray_sizing};
    nfail = 0;
    for i = 1:numel(tests)
        name = func2str(tests{i});
        try
            tests{i}();
            fprintf('  PASS  %s\n', name);
        catch err
            nfail = nfail + 1;
            fprintf('  FAIL  %s: %s\n', name, err.message);
        end
    end
    fprintf('%d of %d tests passed\n', numel(tests) - nfail, numel(tests));

    % ---- fixtures ------------------------------------------------------------------

function [vol, r] = spheres(n, rout, rin)
    if nargin < 1
        n = 48;
    end
    if nargin < 2
        rout = 19;
    end
    if nargin < 3
        rin = 9;
    end
    [xi, yi, zi] = ndgrid(1:n);
    c = (n + 1) / 2;
    r = sqrt((xi - c).^2 + (yi - c).^2 + (zi - c).^2);
    vol = uint8(r < rout);
    vol(r < rin) = 2;

function v = tetvol(node, elem)
    a = node(elem(:, 1), :);
    b = node(elem(:, 2), :) - a;
    c = node(elem(:, 3), :) - a;
    d = node(elem(:, 4), :) - a;
    v = abs(sum(cross(b, c, 2) .* d, 2)) / 6;

function v = enclosed(node, tri)   % divergence theorem, outward triangles
    a = node(tri(:, 1), :);
    b = node(tri(:, 2), :);
    c = node(tri(:, 3), :);
    v = sum(sum(a .* cross(b, c, 2), 2)) / 6;

function check(cond, msg)
    if ~cond
        error(msg);
    end

    % ---- tests ---------------------------------------------------------------------

function test_outputs
    [node, elem, face, info] = trussnet(spheres(), 'size', 3);
    check(size(node, 2) == 3 && size(elem, 2) == 5 && size(face, 2) == 5, 'output widths');
    check(size(elem, 1) > 1000, 'too few elements');
    check(isstruct(info) && isfield(info, 'badfaces') && isfield(info, 'ms_total'), 'info fields');
    check(ischar(info.version) && ~isempty(regexp(info.version, '^\d+\.\d+\.\d+$', 'once')), 'info.version');
    [n2, e2] = trussnet(spheres(), 'size', 3);   % fewer outputs: no face
    check(isequal(size(n2), size(node)) && isequal(size(e2), size(elem)), 'nargout = 2');

function test_one_based
    [node, elem, face] = trussnet(spheres(), 'size', 3);
    check(min(min(elem(:, 1:4))) == 1 && max(max(elem(:, 1:4))) == size(node, 1), 'elem indices');
    check(min(min(face(:, 1:3))) >= 1 && max(max(face(:, 1:3))) <= size(node, 1), 'face indices');
    check(numel(unique(elem(:, 1:4))) == size(node, 1), 'unused nodes');

function test_conformity
    [node, elem, face, info] = trussnet(spheres(), 'size', 3);
    check(isequal(unique(elem(:, 5))', [1 2]), 'labels');
    check(info.badfaces == 0 && info.badedges == 0 && info.spanning == 0, 'not conforming');
    check(min(tetvol(node, elem)) > 0, 'degenerate tet');
    check(info.mindihedral > 5 && info.joeliumedian > 0.8, 'quality');

function test_face_orientation
    [node, elem, face] = trussnet(spheres(), 'size', 3);
    kinds = unique(face(:, 4:5), 'rows');
    check(isequal(kinds, [1 0; 2 1]), 'face kinds');
    tv = tetvol(node, elem);
    ext = face(face(:, 5) == 0, :);
    check(abs(enclosed(node, ext) / sum(tv) - 1) < 1e-6, 'exterior orientation');
    in2 = face(face(:, 4) == 2 & face(:, 5) == 1, :);
    check(abs(enclosed(node, in2) / sum(tv(elem(:, 5) == 2)) - 1) < 1e-6, 'interface orientation');

function test_node_space   % MATLAB index space: the ball of radius 19 around (24.5, 24.5, 24.5)
    node = trussnet(spheres(), 'size', 3);
    rn = sqrt(sum((node - 24.5).^2, 2));
    check(abs(max(rn) - 19) < 1, 'node coordinates not in index space');

function test_options_struct_and_pairs
    vol = spheres(32, 12, 5);
    [n1, e1] = trussnet(vol, struct('size', 3, 'reratio', 2));
    [n2, e2] = trussnet(vol, 'size', 3, 'reratio', 2);
    [n3, e3] = trussnet(vol, 'SIZE', 3, 're_ratio', 2);   % case / '_' insensitive
    check(isequal(e1, e2) && isequal(e1, e3) && isequal(n1, n2), 'option forms differ');

function test_voxelsize
    vol = spheres(32, 12, 5);
    [n1, e1] = trussnet(vol, 'size', 3);
    [n2, e2] = trussnet(vol, 'size', 6, 'voxelsize', 2);
    check(isequal(e1, e2), 'voxelsize changed the mesh');
    check(max(abs(n2(:) - 2 * n1(:))) < 1e-4, 'voxelsize scaling');

function test_affine   % a 0-based voxel -> world matrix
    vol = spheres(32, 12, 5);
    A = [2 0 0 10; 0 2 0 -5; 0 0 2 3; 0 0 0 1];
    [n1, e1] = trussnet(vol, 'size', 3);
    [n2, e2] = trussnet(vol, 'size', 6, 'affine', A);   % (voxel size 2 from A's columns)
    check(isequal(e1, e2), 'affine changed the mesh');
    ref = 2 * (n1 - 1) + repmat([10 -5 3], size(n1, 1), 1);   % index space is 1-based
    check(max(abs(n2(:) - ref(:))) < 1e-4, 'affine mapping');

function test_lsize
    vol = spheres(40, 16, 7);
    [n0, e0] = trussnet(vol, 'size', 3);
    [n1, e1] = trussnet(vol, 'size', 3, 'lsize', [0 1.5]);     % vector: lsize(l)
    [n2, e2] = trussnet(vol, 'size', 3, 'lsize', [2 1.5]);     % 1 x 2 is a vector, too
    [n3, e3] = trussnet(vol, 'size', 3, 'lsize', [2 1.5; 1 3]);   % N x 2 [label size]
    check(sum(e1(:, 5) == 2) > 1.4 * sum(e0(:, 5) == 2), 'lsize vector');
    check(isequal(e1, e3), 'lsize N x 2');
    check(sum(e2(:, 5) == 1) > 1.4 * sum(e0(:, 5) == 1), 'lsize [2 1.5]: label 1 at 2, label 2 at 1.5');

function test_logical_input
    vol = spheres(32, 12, 5) > 0;
    [node, elem] = trussnet(vol, 'size', 3);
    check(isequal(unique(elem(:, 5))', 1), 'logical volume');

function test_gray_single   % radial ramp: level 5 = the sphere of radius 15
    [vol, r] = spheres(40);
    [node, elem, face, info] = trussnet(20 - r, 'thresholds', 5, 'size', 3);
    check(isequal(unique(elem(:, 5))', 1), 'gray single labels');
    check(abs(max(sqrt(sum((node - 20.5).^2, 2))) - 15) < 0.3, 'iso-surface position');
    check(info.badfaces == 0, 'gray single conformity');

function test_gray_multi
    [vol, r] = spheres(40);
    [node, elem, face, info] = trussnet(single(20 - r), 'thresholds', [5 10 14], 'size', 2);
    check(isequal(unique(elem(:, 5))', [1 2 3]), 'gray multi labels');
    check(isequal(unique(face(:, 4:5), 'rows'), [1 0; 2 1; 3 2]), 'gray multi faces');
    check(info.badfaces == 0 && info.spanning == 0, 'gray multi conformity');
    [n2, e2] = trussnet(single(20 - r), 'thresholds', [14 5 10], 'size', 2);
    check(isequal(elem, e2), 'threshold order');

function test_deterministic
    vol = spheres(32, 12, 5);
    [n1, e1] = trussnet(vol, 'size', 3);
    [n2, e2] = trussnet(vol, 'size', 3);
    check(isequal(n1, n2) && isequal(e1, e2), 'not deterministic');

function test_errors
    vol = spheres(24, 9, 4);
    check(throws(@() trussnet(zeros(16, 16, 16, 'uint8'))), 'empty volume');
    check(throws(@() trussnet(ones(1, 10))), '1-D input');
    check(throws(@() trussnet(double(vol) - 1)), 'negative labels');
    check(throws(@() trussnet(vol, 'trap', 'nope')), 'bad trap');
    check(throws(@() trussnet(vol, 'size')), 'odd name/value list');
    check(throws(@() trussnet(vol, 'affine', eye(3))), 'bad affine');

function t = throws(f)
    t = false;
    try
        f();
    catch
        t = true;
    end

function test_gpu
    vol = spheres();
    [ng, eg, fg, ig] = trussnet(vol, 'size', 3, 'gpu', 1);
    if ~ig.usedgpu
        fprintf('        (no OpenCL device: skipped)\n');
        return
    end
    [nc, ec, fc, ic] = trussnet(vol, 'size', 3, 'gpu', 0);
    check(~ic.usedgpu, 'gpu = 0 used the GPU');
    check(ig.badfaces == 0 && ig.spanning == 0 && ic.badfaces == 0, 'gpu conformity');
    check(abs(size(eg, 1) - size(ec, 1)) / size(ec, 1) < 0.05, 'gpu / cpu sizes differ');

    % ---- tissue-probability (4-D) input ---------------------------------------------

function [tpm, r] = tpm_spheres(background)   % [background,] shell, inner ball: logistic edges
    if nargin < 1
        background = true;
    end
    [xi, yi, zi] = ndgrid(1:40);
    r = sqrt((xi - 20.5).^2 + (yi - 20.5).^2 + (zi - 20.5).^2);
    pout = 1 ./ (1 + exp((r - 16) / 1.5 * 4));
    pin = 1 ./ (1 + exp((r - 7) / 1.5 * 4));
    tpm = single(cat(4, 1 - pout, pout - pin, pin));
    if ~background
        tpm = tpm(:, :, :, 2:3);
    end

function test_tpm
    tpm = tpm_spheres();
    [node, elem, face, info] = trussnet(tpm, 'size', 3, 'tpmexterior', 1);   % 1-based channel
    check(isequal(unique(elem(:, 5))', [1 2]), 'tpm labels');
    check(info.badfaces == 0 && info.badedges == 0 && info.spanning == 0, 'tpm conformity');
    check(isequal(unique(face(:, 4:5), 'rows'), [1 0; 2 1]), 'tpm faces');
    tv = tetvol(node, elem);
    soft = squeeze(sum(sum(sum(tpm, 1), 2), 3));
    check(abs(sum(tv(elem(:, 5) == 1)) / soft(2) - 1) < 0.05, 'tpm shell volume');

function test_tpm_no_exterior   % tissues only: exterior = 1 - sum
    [n1, e1] = trussnet(tpm_spheres(), 'size', 3, 'tpmexterior', 1);
    [n2, e2] = trussnet(tpm_spheres(false), 'size', 3);
    check(isequal(unique(e2(:, 5))', [1 2]), 'no-exterior labels');
    check(abs(size(e2, 1) - size(e1, 1)) / size(e1, 1) < 0.05, 'no-exterior size');

function test_tpm_map_holes
    tpm = tpm_spheres();
    four = cat(4, tpm(:, :, :, 1), tpm(:, :, :, 2) / 2, tpm(:, :, :, 2) / 2, tpm(:, :, :, 3));
    [n1, e1] = trussnet(tpm, 'size', 3, 'tpmexterior', 1);
    [n2, e2] = trussnet(four, 'size', 3, 'tpmmap', [0 1 1 2]);   % the shell split in two, merged back
    check(isequal(e1, e2), 'tpmmap merge');
    pocket = tpm;
    pocket(8:13, 18:23, 18:23, :) = repmat(reshape(single([1 0 0]), 1, 1, 1, 3), [6 6 6 1]);
    [n3, e3, f3, i3] = trussnet(pocket, 'size', 2, 'tpmexterior', 1);
    [n4, e4, f4, i4] = trussnet(pocket, 'size', 2, 'tpmexterior', 1, 'tpmholes', 1);
    check(i3.tpmfilled == 216 && i4.tpmfilled == 0, 'tpm hole counts');
    check(sum(tetvol(n3, e3)) - sum(tetvol(n4, e4)) > 100, 'tpm kept pocket meshed');

function test_tpm_raw_fields   % p = 0.5 interfaces: the ball within 5% of the sphere
    tpm = tpm_spheres();
    [node, elem, face, info] = trussnet(tpm, 'size', 3, 'tpmexterior', 1, 'tpmfields', 1, 'sigma', 0);
    tv = tetvol(node, elem);
    check(abs(sum(tv(elem(:, 5) == 2)) / (4 / 3 * pi * 7^3) - 1) < 0.05, 'raw-field ball volume');
    check(info.badfaces == 0 && info.spanning == 0, 'raw-field conformity');

function test_tpm_file   % a 4-D NIfTI-1 written by hand, meshed in its world coordinates
    tpm = tpm_spheres(false);
    fn = [tempname() '.nii'];
    write_nifti4d(fn, tpm, [2 2 2], [-40 -40 -40]);
    cleaner = onCleanup(@() delete(fn));
    [n1, e1] = trussnet(fn, 'size', 6);
    [n2, e2] = trussnet(tpm, 'size', 6, 'affine', [2 0 0 -40; 0 2 0 -40; 0 0 2 -40; 0 0 0 1]);
    check(isequal(e1, e2) && max(abs(n1(:) - n2(:))) < 1e-4, 'tpm file vs array');

function write_nifti4d(fn, img, vs, origin)
    hdr = zeros(1, 348, 'uint8');
    sz = size(img);
    hdr = put_bytes(hdr, 0, 348, 'int32');
    hdr = put_bytes(hdr, 40, [4 sz(1:4) 1 1 1], 'int16');
    hdr = put_bytes(hdr, 70, [16 32], 'int16');                          % float32
    hdr = put_bytes(hdr, 76, [1 vs 1 1 1 1], 'single');
    hdr = put_bytes(hdr, 108, 352, 'single');                            % vox_offset
    hdr = put_bytes(hdr, 112, [1 0], 'single');                          % scl_slope, scl_inter
    hdr = put_bytes(hdr, 254, 1, 'int16');                               % sform_code
    hdr = put_bytes(hdr, 280, [vs(1) 0 0 origin(1) 0 vs(2) 0 origin(2) 0 0 vs(3) origin(3)], 'single');
    hdr = put_bytes(hdr, 344, uint8('n+1'), 'uint8');
    fid = fopen(fn, 'wb');
    fwrite(fid, hdr, 'uint8');
    fwrite(fid, zeros(1, 4, 'uint8'), 'uint8');
    fwrite(fid, single(img), 'single');
    fclose(fid);

function h = put_bytes(h, off, val, cls)   % little-endian bytes of val (as class cls) at offset off
    b = typecast(cast(val, cls), 'uint8');
    h(off + (1:numel(b))) = b;

    % ---- user sizing ------------------------------------------------------------------

function test_sizing_field
    [vol, r] = spheres(40, 16, 7);
    [n0, e0] = trussnet(vol, 'size', 3);
    [n1, e1] = trussnet(vol, 'size', 3, 'sizing', zeros(size(vol)));   % 0 = automatic everywhere
    check(isequal(e0, e1), 'zero field changed the mesh');
    [n2, e2, f2, i2] = trussnet(vol, 'size', 3, 'sizing', 1.5 * (r < 10));
    check(sum(e2(:, 5) == 2) > 1.5 * sum(e0(:, 5) == 2), 'local field did not refine');
    check(i2.badfaces == 0 && i2.spanning == 0, 'field conformity');
    [n3, e3] = trussnet(vol, 'size', 3, 'sizing', 4 * ones(size(vol)));
    [n4, e4] = trussnet(vol, 'size', 3, 'sizing', 2 * ones(size(vol)));
    check(size(e4, 1) > 4 * size(e3, 1), 'uniform field does not set the size');

function test_sizing_vectors
    vol = spheres(40, 16, 7);
    [n0, e0] = trussnet(vol, 'size', 3, 'lsize', [0 1.5]);
    [n1, e1] = trussnet(vol, 'size', 3, 'sizing', [0 1.5]);       % labels 1..N
    [n2, e2] = trussnet(vol, 'size', 3, 'sizing', [0 0 1.5]);     % labels 0..N
    check(isequal(e0, e1) && isequal(e0, e2), 'label sizing vector');
    tpm = tpm_spheres();
    [n3, e3] = trussnet(tpm, 'size', 3, 'tpmexterior', 1, 'lsize', [0 1.5]);
    [n4, e4] = trussnet(tpm, 'size', 3, 'tpmexterior', 1, 'sizing', [0 0 1.5]);   % per channel
    check(isequal(e3, e4), 'TPM channel sizing vector');
    check(throws(@() trussnet(vol, 'sizing', [1 2 3 4])), 'bad sizing length');

    % ---- 2-D images -------------------------------------------------------------------

function [lab, r] = disk_labels()
    [xi, yi] = ndgrid(1:120, 1:100);
    r = sqrt((xi - 60.5).^2 + (yi - 50.5).^2);
    lab = uint8(r < 40);
    lab(r < 40 & xi > 60.5) = 2;
    lab(r < 40 & yi > 60.5) = 3;

function test_2d
    lab = disk_labels();
    [node, elem, face, info] = trussnet(lab, 'size', 4);
    check(size(node, 2) == 2 && size(elem, 2) == 4 && size(face, 2) == 4, '2-D output widths');
    check(isequal(unique(elem(:, 4))', [1 2 3]), '2-D labels');
    check(info.badedges == 0 && info.spanning == 0 && info.junctions == 4, '2-D conformity / junctions');
    a = node(elem(:, 1), :);
    b = node(elem(:, 2), :);
    c = node(elem(:, 3), :);
    ar = 0.5 * ((b(:, 1) - a(:, 1)) .* (c(:, 2) - a(:, 2)) - (b(:, 2) - a(:, 2)) .* (c(:, 1) - a(:, 1)));
    check(min(ar) > 0, '2-D orientation');
    check(all(abs(info.labelarea(2:4) ./ info.labelpixels(2:4) - 1) < 0.03), '2-D areas');
    check(abs(max(sqrt(sum((node - [60.5 50.5]).^2, 2))) - 40) < 1, '2-D index space');

function test_2d_gray_sizing
    [lab, r] = disk_labels();
    [n1, e1, f1, i1] = trussnet(50 - r, 'thresholds', [10 25], 'size', 3);
    check(isequal(unique(e1(:, 4))', [1 2]) && i1.badedges == 0 && i1.spanning == 0, '2-D gray-scale');
    [n2, e2] = trussnet(lab, 'size', 4, 'lsize', [0 2]);
    [n3, e3] = trussnet(lab, 'size', 4, 'sizing', [0 2 0]);
    [n4, e4] = trussnet(lab, 'size', 4, 'sizing', zeros(size(lab)));
    [n5, e5] = trussnet(lab, 'size', 4);
    check(isequal(e2, e3) && isequal(e4, e5), '2-D sizing forms');
