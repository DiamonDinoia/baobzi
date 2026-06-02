function [y] = bz_eval1(self, x, output_dim)
mex_id_ = 'bz_eval1_w(c i baobzi_function*, c i double[], c o double[x], c i int)';
[y] = baobzi_mex(mex_id_, self, x, output_dim, output_dim);

% -----------------------------------------------------------------------
% Batch AoS eval. Xflat = (input_dim*n) doubles, point-major.
