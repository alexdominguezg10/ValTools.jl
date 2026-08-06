% Independent MATLAB verification of ValTools.jl's variance-ellipse formula
% (current_ellipse_metrics / mooring_variance_ellipse -- src/Metrics/stats.jl
% and src/Observations/mooring.jl, byte-identical math in both places).
%
% This is NOT a jLab crosscheck. jLab's jEllipse/ellparams.m computes the
% MODULATED ellipse of an analytic signal (a different quantity, already
% ported+verified against real jLab as ValTools' `ellipsefit`). jLab has no
% princax-equivalent for the static, whole-record covariance ellipse, so
% this script implements the standard "principal axis" / variance-ellipse
% method directly via MATLAB's own eig() -- independent of ValTools'
% closed-form 2x2 trace/determinant eigenvalue algebra. Reference method:
% e.g. Emery & Thomson, "Data Analysis Methods in Physical Oceanography",
% principal axis / variance ellipse section.
%
% Usage (from ValTools.jl root):
%   matlab -batch "run('scripts/crosscheck_variance_ellipse.m')"

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'crosscheck_variance_ellipse');

sites = {'LR75DW_mid', 'WH600DW_bottom'};

for i = 1:numel(sites)
    name = sites{i};
    uv = csvread(fullfile(OUTDIR, [name '_uv.csv']));
    u = uv(:,1);
    v = uv(:,2);

    u_anom = u - mean(u);
    v_anom = v - mean(v);
    C = [mean(u_anom.^2),      mean(u_anom.*v_anom);
         mean(u_anom.*v_anom), mean(v_anom.^2)];

    % Independent eigen-decomposition via MATLAB's own eig(), NOT the
    % trace/determinant closed form ValTools uses. eig() returns eigenvalues
    % in ascending order for a symmetric 2x2 -- take max/min explicitly
    % rather than relying on that order.
    evals = eig(C);
    l1 = max(evals);   % major-axis variance
    l2 = min(evals);   % minor-axis variance

    semi_major = sqrt(max(l1, 0));
    semi_minor = sqrt(max(l2, 0));

    % Same inclination-from-north convention as ValTools (Julia atan(y,x) ==
    % MATLAB atan2(y,x)): theta from the covariance matrix, then rotated to
    % a 0-180 deg bearing.
    theta = 0.5 * atan2(2*C(1,2), C(1,1) - C(2,2));
    inclination = mod(90.0 - rad2deg(theta), 180.0);

    eke = 0.5 * (var(u) + var(v));   % MATLAB var() defaults to N-1, same as Julia's

    fprintf('%s: semi_major=%.6f semi_minor=%.6f inclination=%.4f EKE=%.6f (n=%d)\n', ...
        name, semi_major, semi_minor, inclination, eke, numel(u));

    dlmwrite(fullfile(OUTDIR, [name '_matlab.csv']), ...
        [semi_major; semi_minor; inclination; eke], 'precision', '%.17g');
end

fprintf('Independent MATLAB variance-ellipse computation done for %d site(s).\n', numel(sites));
