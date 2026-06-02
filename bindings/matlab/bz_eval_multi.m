function [Yflat] = bz_eval_multi(self, Xflat, output_dim, n)
mex_id_ = 'bz_eval_multi_w(c i baobzi_function*, c i double[], c o double[xx], c i int, c i int64_t)';
[Yflat] = baobzi_mex(mex_id_, self, Xflat, output_dim, n, output_dim, n);

% -----------------------------------------------------------------------
% Sorted 1-D batch eval (dim == 1; caller guarantees ascending x).
