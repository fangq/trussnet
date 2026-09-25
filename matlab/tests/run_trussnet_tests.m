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
    %

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..'));

    tests = {@test_outputs, @test_one_based, @test_conformity, @test_face_orientation, ...
             @test_node_space, @test_options_struct_and_pairs, @test_voxelsize, @test_affine, ...
             @test_lsize, @test_logical_input, @test_gray_single, @test_gray_multi, ...
             @test_deterministic, @test_errors, @test_gpu};
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
    [n2, e2] = trussnet(vol, 'size', 6, 'voxelsize', 2, 'affine', A);
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
    check(throws(@() trussnet(ones(10, 10))), '2-D volume');
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
