% example_vector.m — 2-D input, 3-D output vector fit
addpath(fileparts(fileparts(mfilename('fullpath'))));

f   = @(x) [sin(x(1)+x(2)); cos(x(1)-x(2)); x(1)*x(2)];
obj = baobzi(f, [-1,-1], [1,1], 1e-6, ...
             'dim', 2, 'out_dim', 3, 'degree', 11, 'max_memory_mib', 64);

[gx, gy] = meshgrid(linspace(-1,1,30));
Xgrid    = [gx(:), gy(:)];
Yhat     = obj.eval(Xgrid);   % 900×3
Yref     = [sin(Xgrid(:,1)+Xgrid(:,2)), ...
             cos(Xgrid(:,1)-Xgrid(:,2)), ...
             Xgrid(:,1).*Xgrid(:,2)];
fprintf('2D->3D max abs error: %.3e\n', max(max(abs(Yhat - Yref))));
fprintf('Memory: %.1f KiB\n', obj.memory_usage()/1024);
delete(obj);
