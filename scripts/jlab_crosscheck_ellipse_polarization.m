% Independent MATLAB verification of ValTools.jl's `ellipse_polarization`,
% run through jLab's OWN real pipeline: sleptap -> mspec -> specdiag (which
% itself just calls polparams.m -- confirmed by reading jSpectral/specdiag.m
% directly) -- rather than a from-scratch reimplementation, since
% `ellipse_polarization` IS meant to be a direct port of these two exact
% jLab functions and has never been checked against them.
%
% Real ARE/LR75DW mooring data (see the .jl companion) -- dt is read from
% dt_hours.txt (computed there from the file's own real timestamps, 0.5 h
% for this record), not hardcoded here, so the two sides can never silently
% drift out of sync on sample rate.
%
% N<=512 deliberately: sleptap.m spline-interpolates its tapers from an
% M=512 base case for any M>512, so keeping N<=512 here means BOTH sides
% use jLab's exact tridiagonal-method Slepian tapers, not that
% approximation -- see the .jl companion's header comment for why this
% matters (a spline-interpolation difference would masquerade as a
% port bug otherwise).
%
% Usage (from ValTools.jl root):
%   matlab -batch "run('scripts/jlab_crosscheck_ellipse_polarization.m')"

JLAB_PATH = '/Users/alexdominguez/ADominguez/TOOLS/MATLAB/jlab';
addpath(JLAB_PATH);
jlab_addpath;

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'jlab_crosscheck_ellipse_polarization');

uv = csvread(fullfile(OUTDIR, 'uv.csv'));
u = uv(:,1);
v = uv(:,2);
N = numel(u);

dt = dlmread(fullfile(OUTDIR, 'dt_hours.txt'));
nw = 4.0;
K = 7;

% 'cyclic' only rescales the returned f (f=f/2/pi internally, confirmed by
% reading mspec.m directly) -- Sxx/Syy/Sxy (and hence d1/d2/theta/nu/P/
% alpha/beta below) are IDENTICAL with or without it. It's required here
% only so `bin = round(f*N*dt)` below lines up with the .jl companion's
% cyclic-frequency bin index -- without it, f is in rad/hour, and every
% bin label comes out wrong by a factor of ~2*pi, silently pairing rows at
% UNRELATED frequencies when the .jl side matches by bin number (this was
% hit for real: first version of this script omitted 'cyclic' and produced
% large, non-constant-looking mismatches that looked like a formula bug in
% ellipse_polarization but were actually just misaligned rows).
[psi, lambda] = sleptap(N, nw, K);
[f, Sxx, Syy, Sxy] = mspec(dt, u, v, psi, 'cyclic');

[d1, d2, th, nu] = specdiag(Sxx, Syy, Sxy);
[~, P, alpha, beta] = polparams(Sxx, Syy, Sxy);

% FFT bin index for each row of mspec's output (f(1)=0/DC through Nyquist),
% used by the .jl companion to align rows without assuming a fixed offset.
bin = round(f * N * dt);

out = [bin(:), f(:), d1(:), d2(:), th(:), nu(:), P(:), alpha(:), real(beta(:)), imag(beta(:))];
dlmwrite(fullfile(OUTDIR, 'jlab_polparams.csv'), out, 'precision', '%.17g');

fprintf('jLab sleptap/mspec/specdiag/polparams done: N=%d, %d frequency bins, K=%d tapers\n', N, numel(f), K);
