classdef baobzi < handle
% BAOBZI  MATLAB handle class wrapping a piecewise-Chebyshev approximator.
%
% This is a thin classdef over the mwrap-generated bz_*.m stubs (see
% baobzi.mw). The opaque baobzi_t handle lives in the `mwptr` property, which
% mwrap reads/writes via its R2008OO object-pointer convention.
%
% CONSTRUCTION
%   obj = baobzi(f, a, b, tol)
%   obj = baobzi(f, a, b, tol, Name, Value, ...)
%
%   f   — function_handle  @(x) y   (x: 1×dim row, y: 1×out_dim row)
%   a,b — 1×dim double domain corners (lower/upper)
%   tol — scalar double tolerance
%
%   Name/Value options:
%     'dim'                   input dimension (default: numel(a))
%     'out_dim'               output dimension (default: 1)
%     'degree'                Chebyshev degree: 7, 11, or 15 (default: 11)
%     'policy'                0=LATENCY,1=THROUGHPUT,2=BALANCED (default: 2)
%     'tol_kind'              0=REL_TAIL..5=ABS_L2 (default: 2=REL_MAX)
%     'max_depth'             max tree depth (default: 50)
%     'max_memory_mib'        memory budget in MiB (default: 4)
%     'allow_max_depth_leaves' int bool (default: 0)
%     'min_uniform_depth'     (default: 0)
%
% USAGE
%   y = obj.eval(X)       X is N×dim, y is N×out_dim
%   y = obj(X)            same (via subsref)
%   n = obj.memory_usage()
%   obj.print_stats()
%   delete(obj)           frees C-side memory

    properties (SetAccess = private)
        mwptr        % opaque baobzi_t handle (mwrap object pointer)
    end
    properties (Access = private)
        input_dim_
        output_dim_
    end

    methods

        function obj = baobzi(f, a, b, tol, varargin)
            p = inputParser;
            addRequired(p, 'f');
            addRequired(p, 'a');
            addRequired(p, 'b');
            addRequired(p, 'tol');
            addParameter(p, 'dim',                    numel(a));
            addParameter(p, 'out_dim',                1);
            addParameter(p, 'degree',                 11);
            addParameter(p, 'policy',                 2);
            addParameter(p, 'tol_kind',               2);
            addParameter(p, 'max_depth',              50);
            addParameter(p, 'max_memory_mib',         4);
            addParameter(p, 'allow_max_depth_leaves', 0);
            addParameter(p, 'min_uniform_depth',      0);
            parse(p, f, a, b, tol, varargin{:});
            r = p.Results;

            a = double(r.a(:)');
            b = double(r.b(:)');

            % The generated gateway reads scalars as mxDOUBLE_CLASS, so pass
            % every numeric option as a plain double (no int32 casts).
            obj.mwptr = bz_fit(r.f, a, b, double(r.tol), ...
                double(r.dim), double(r.out_dim), double(r.degree), ...
                double(r.policy), double(r.tol_kind), ...
                double(r.max_depth), double(r.max_memory_mib), ...
                double(r.allow_max_depth_leaves), double(r.min_uniform_depth));

            obj.input_dim_  = r.dim;
            obj.output_dim_ = r.out_dim;
        end

        function y = eval(obj, X)
        % EVAL  Evaluate at N points. X: N×dim. Returns N×out_dim.
            X = double(X);
            if isvector(X) && obj.input_dim_ == 1
                X = X(:);
            end
            N = size(X, 1);
            if N == 1
                % Single-point fast path. y comes back out_dim×1.
                y = bz_eval1(obj, X(:)', obj.output_dim_);
                y = reshape(y, 1, obj.output_dim_);
            else
                % Batch: X is N×dim (row-per-point). Transpose to dim×N so
                % column-major storage = AoS point-major.
                Xflat = X';                       % dim×N, point-major
                % bz_eval_multi returns Yflat as out_dim×N.
                Yflat = bz_eval_multi(obj, Xflat(:), obj.output_dim_, N);
                y = Yflat';                       % N×out_dim
            end
        end

        function varargout = subsref(obj, S)
            if numel(S) == 1 && strcmp(S.type, '()')
                varargout{1} = obj.eval(S.subs{:});
            else
                [varargout{1:nargout}] = builtin('subsref', obj, S);
            end
        end

        function bytes = memory_usage(obj)
            bytes = bz_memory(obj);
        end

        function print_stats(obj)
            bz_stats(obj);
        end

        function d = input_dim(obj)
            d = obj.input_dim_;
        end

        function d = output_dim(obj)
            d = obj.output_dim_;
        end

        function delete(obj)
            if ~isempty(obj.mwptr)
                bz_free(obj);
                obj.mwptr = [];
            end
        end

    end
end
