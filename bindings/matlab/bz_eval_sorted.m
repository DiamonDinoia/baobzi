function [Yflat] = bz_eval_sorted(self, x, output_dim, n)
mex_id_ = 'bz_eval_sorted_w(c i baobzi_function*, c i double[], c o double[xx], c i int, c i int64_t)';
[Yflat] = baobzi_mex(mex_id_, self, x, output_dim, n, output_dim, n);

% -----------------------------------------------------------------------
% Batch SoA eval. Xflat = dim contiguous coordinate planes (n each).
