function test_baobzi()
% test_baobzi -- smoke/parity tests for the baobzi MATLAB wrapper.
% Exits via error() on any failure.
% Run headless:  matlab -batch "addpath('.'); test_baobzi"

passed = 0;
failed = 0;

    function check(cond, name)
        if cond
            fprintf('PASS  %s\n', name);
            passed = passed + 1;
        else
            fprintf('FAIL  %s\n', name);
            failed = failed + 1;
        end
    end

% ================================================================== %
% 1. 1-D scalar fit: exp(0.5*x)+sin(3*x) on [0,1]
% ================================================================== %
f1   = @(x) exp(0.5*x(1)) + sin(3*x(1));
obj1 = baobzi(f1, [0], [1], 1e-8, 'dim', 1, 'out_dim', 1, 'degree', 11);

% baobzi's upper domain corner is exclusive (eval at exactly b yields NaN,
% the out-of-domain sentinel), so sample strictly inside [a, b).
Xtest = linspace(0, 1, 201)';
Xtest(end) = [];
Yhat  = obj1.eval(Xtest);
Yref  = exp(0.5*Xtest) + sin(3*Xtest);
err1  = max(abs(Yhat(:) - Yref(:)));
check(err1 < 1e-5, sprintf('1D accuracy (max err = %.2e)', err1));

Yhat2 = obj1(Xtest);
check(isnumeric(Yhat2) && max(abs(Yhat2(:) - Yhat(:))) < 1e-12, ...
      '1D subsref syntax obj(X)');

y1    = obj1.eval([0.5]);
yref1 = exp(0.25) + sin(1.5);
check(abs(y1 - yref1) < 1e-5, '1D single-point eval');

delete(obj1);

% ================================================================== %
% 2. 2-D -> 3-D vector fit
% ================================================================== %
f2   = @(x) [sin(x(1)+x(2)); cos(x(1)-x(2)); x(1)*x(2)];
obj2 = baobzi(f2, [-1,-1], [1,1], 1e-6, ...
              'dim', 2, 'out_dim', 3, 'degree', 11, 'max_memory_mib', 64);

[gx, gy] = meshgrid(linspace(-1,1,50), linspace(-1,1,50));
Xgrid    = [gx(:), gy(:)];
Ygrid    = obj2.eval(Xgrid);
Yref2    = [sin(Xgrid(:,1)+Xgrid(:,2)), ...
             cos(Xgrid(:,1)-Xgrid(:,2)), ...
             Xgrid(:,1).*Xgrid(:,2)];

err2 = max(max(abs(Ygrid - Yref2)));
check(err2 < 1e-4, sprintf('2D->3D accuracy (max err = %.2e)', err2));
check(size(Ygrid,1)==2500 && size(Ygrid,2)==3, '2D->3D output shape 2500x3');
check(obj2.memory_usage() > 0, '2D->3D memory_usage > 0');

delete(obj2);

% ================================================================== %
% 3. NaN out-of-domain
% ================================================================== %
f3   = @(x) sin(x(1));
obj3 = baobzi(f3, [0], [1], 1e-8);
y_out = obj3.eval([-5]);
check(isnan(y_out), 'Out-of-domain returns NaN');
delete(obj3);

% ================================================================== %
% 4. Erroring callback is caught and surfaced (no crash/segfault)
% ================================================================== %
f_bad  = @(x) error('boom: intentional test error');
caught = false;
msg_ok = false;
try
    obj_bad = baobzi(f_bad, [0], [1], 1e-8);
    delete(obj_bad);
catch ME
    caught = true;
    msg_ok = contains(ME.message, 'boom') || ...
             strcmp(ME.identifier, 'baobzi:callback');
end
check(caught,  'Erroring callback: constructor threw');
check(msg_ok,  'Erroring callback: message surfaced');

% ================================================================== %
% 5. Too-tight tol + low max_depth -> fit error
% ================================================================== %
f5      = @(x) sin(100*x(1));
caught5 = false;
msg5_ok = false;
try
    obj5 = baobzi(f5, [0], [1], 1e-15, ...
                  'max_depth', 2, 'max_memory_mib', 1);
    delete(obj5);
catch ME5
    caught5 = true;
    msg5_ok = contains(ME5.message, 'MaxDepth')     || ...
              contains(ME5.message, 'MemoryBudget')  || ...
              contains(ME5.message, 'depth')          || ...
              contains(ME5.message, 'memory')         || ...
              strcmp(ME5.identifier, 'baobzi:fit');
end
check(caught5,  'Tight-tol/low-depth: error thrown');
check(msg5_ok,  'Tight-tol/low-depth: message mentions MaxDepth/MemoryBudget or baobzi:fit');

% ================================================================== %
% Summary
% ================================================================== %
fprintf('\n--- Results: %d passed, %d failed ---\n', passed, failed);
if failed > 0
    error('test_baobzi:failures', '%d test(s) failed.', failed);
end

end  % function test_baobzi
