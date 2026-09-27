function [node, elem, face, info] = tnmesh(cmd, varargin)
    %
    % [node, elem, face, info] = tnmesh(cmd, ...)
    %
    % The stages of the trussnet mesher on their own, and meshes / surfaces as input
    % (the command-line --mode). All indices are 1-based; elem is M x 5 [v1 v2 v3 v4
    % label], face P x 5 [v1 v2 v3 inner outer] (outer = 0: the exterior).
    %
    % author: Qianqian Fang (q.fang at neu.edu)
    %
    % commands:
    %     [node, ~, face, info] = tnmesh('surface', vol, opt)
    %          the region and exterior surfaces of a volume (trussnet, faces only;
    %          node holds only the surfaces' nodes); only the surface nodes are
    %          tessellated -- about twice as fast -- unless opt.exacttess = 1
    %     [node, ~, ~, info] = tnmesh('points', vol, opt)
    %          the relaxed nodes before tessellation; info.nodelabel, info.nodetype
    %          (0 interior, 1 interface, 2 junction, 3 corner), info.nodepartner (the
    %          other labels, 0 = none)
    %     [node, elem, ~, info] = tnmesh('tessellate', node, label, opt)
    %          the Delaunay tets of a point cloud (label: per node, optional; each tet
    %          takes the most frequent label of its nodes); opt.gpu: the OpenCL Delaunay
    %     [node, elem, ~, info] = tnmesh('optimize', node, elem, opt)
    %          the mesher's optimiser alone: flips, collapses, Steiner points, smoothing;
    %          the region interfaces and the boundary are kept (opt.q, opt.optrounds)
    %     [node, elem, face, info] = tnmesh('cdt', node, face, opt)
    %          labelled tets of closed, non-self-intersecting surfaces, the surfaces kept
    %          exactly (constrained Delaunay); opt.fill: the spacing of the interior
    %          points (default opt.size, else 1.5 x the mean edge; 0 = none); then the
    %          optimiser unless opt.opt = 0
    %     [node, elem, face, info] = tnmesh('remesh', node, face, opt)
    %          labelled tets of the regions of closed surfaces that may self-intersect,
    %          overlap or be oriented either way: rasterized into per-region fields
    %          (opt.rastervoxel), then the whole mesher (every trussnet option applies)
    %     [node, ~, face, info] = tnmesh('repair', node, face, opt)
    %          as remesh, returning the region surfaces: closed, no self-intersections
    %     info = tnmesh('check', node, elem, face)
    %          a report: quality, inverted tets, open / junction edges, self-
    %          intersections, and info.ok (elem or face may be [])
    %
    % Surface regions come from face's inner / outer labels (P x 5), else each closed
    % component is a shell: shells nest, the innermost containing one's label wins
    % (P x 4: a label per face; P x 3: the nesting depth + 1). opt may be a struct or
    % name / value pairs.
    %
    % example:
    %     [no, ~, fc] = tnmesh('surface', vol, struct('size', 3));
    %     [no2, el2] = tnmesh('cdt', no, fc);
    %     info = tnmesh('check', no2, el2, []);
    %
    % -- this function is part of trussnet (https://github.com/fangq/trussnet)
    % License: GPL-3.0-or-later, Copyright (C) 2026 Qianqian Fang <q.fang at neu.edu>
    %

    if nargin < 2 || ~ischar(cmd)
        error('tnmesh: usage: [node, elem, face, info] = tnmesh(cmd, ...)');
    end

    node = [];
    elem = [];
    face = [];
    info = struct();

    switch lower(cmd)
        case 'surface'
            opt = getopt(varargin(2:end));
            if ~isfield(opt, 'surfaceonly')   % only the surface nodes tessellated (opt.exacttess: all)
                opt.surfaceonly = ~(isfield(opt, 'exacttess') && opt.exacttess);
            end
            if isfield(opt, 'exacttess')
                opt = rmfield(opt, 'exacttess');
            end
            [no, ~, fc, info] = trussnet(varargin{1}, opt);
            [used, ~, idx] = unique(fc(:, 1:3));
            node = no(used, :);
            face = [reshape(idx, [], 3), fc(:, 4:5)];
        case 'points'
            opt = getopt(varargin(2:end));
            opt.points = 1;
            [node, ~, ~, info] = trussnet(varargin{1}, opt);
        case 'check'
            args = [varargin, cell(1, 3)];
            node = trussnet_mex('check', args{1}, args{2}, args{3});   % (the report is the first output)
        case {'tessellate', 'optimize', 'cdt', 'remesh', 'repair'}
            if numel(varargin) < 1
                error('tnmesh: %s needs node', cmd);
            end
            second = [];
            if numel(varargin) >= 2
                second = varargin{2};
            end
            opt = getopt(varargin(3:end));
            [node, elem, face, info] = trussnet_mex(lower(cmd), varargin{1}, second, opt);
        otherwise
            error('tnmesh: unknown command ''%s''', cmd);
    end
end

function opt = getopt(args)
    opt = struct();
    if numel(args) == 1 && isstruct(args{1})
        opt = args{1};
    elseif numel(args) >= 1
        if mod(numel(args), 2) ~= 0
            error('tnmesh: options must be a struct or name/value pairs');
        end
        for i = 1:2:numel(args)
            opt.(args{i}) = args{i + 1};
        end
    end
end
