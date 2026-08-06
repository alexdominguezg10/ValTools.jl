% Independent MATLAB verification of ValTools.jl's `spectral_multitaper`
% (Multitaper.jl-backed), run through jLab's OWN sleptap.m -> mspec.m
% pipeline on the same real ARE/LR75DW mooring data (see the .jl companion).
%
% mspec.m defaults to RADIAN frequency; the 'cyclic' flag below switches it
% to cycles/time, matching Multitaper.jl's convention (confirmed via
% ValTools' own test_spectral.jl, which matches a cos(2*pi*f0*t) signal's
% f0 directly against the returned freq axis) -- without this flag the two
% sides would differ by a spurious factor of 2*pi that has nothing to do
% with a real bug.
%
% dt is read from dt_hours.txt (written by the .jl companion from the
% file's own real timestamps: 0.5 h for this record), not hardcoded.
%
% N<=512 deliberately: sleptap.m spline-interpolates its tapers from an
% M=512 base case for any M>512 -- see the .jl companion's header comment.
%
% Usage (from ValTools.jl root):
%   matlab -batch "run('scripts/jlab_crosscheck_spectral_multitaper.m')"

JLAB_PATH = '/Users/alexdominguez/ADominguez/TOOLS/MATLAB/jlab';
addpath(JLAB_PATH);
jlab_addpath;

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'jlab_crosscheck_spectral_multitaper');

x = csvread(fullfile(OUTDIR, 'x.csv'));
N = numel(x);

dt = dlmread(fullfile(OUTDIR, 'dt_hours.txt'));
nw = 4.0;
K = 7;

[psi, lambda] = sleptap(N, nw, K);
[f, S] = mspec(dt, x, psi, 'cyclic');

out = [f(:), S(:)];
dlmwrite(fullfile(OUTDIR, 'jlab_mspec.csv'), out, 'precision', '%.17g');

fprintf('jLab sleptap/mspec (cyclic) done: N=%d, %d frequency bins, K=%d tapers\n', N, numel(f), K);
