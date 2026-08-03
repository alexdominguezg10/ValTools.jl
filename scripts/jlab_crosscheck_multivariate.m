% Cross-check ValTools.jl's multivariate `instmom` port against REAL jLab
% MATLAB output (working-agreement rule 11: verify against real jLab
% output, not just source reading), following the established
% jlab_crosscheck_* export->matlab->diff pattern.
%
% Calling convention confirmed by reading jRidges/instmom.m directly and
% matching the exact working forms used in jLab's own
% jFigures/jlab_makefigs.m:1451 and jEllipse/ellband.m:371 (both trivariate
% ellband_test4 cases): instmom(dt, X, dim, jdim) where X is the (N x n_ch)
% stacked signal matrix, dim=1 (time), jdim=2 (channels).
%
% Usage (from ValTools.jl root):
%   matlab -batch "run('scripts/jlab_crosscheck_multivariate.m')"

JLAB_PATH = '/Users/alexdominguez/ADominguez/TOOLS/MATLAB/jlab';
addpath(JLAB_PATH);
jlab_addpath;

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'jlab_crosscheck_multivariate');
if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end

% Julia exports interleaved [re1 im1 re2 im2 re3 im3] columns (csvread can't
% parse complex numbers) -- reconstruct the (N x 3) complex matrix here.
Xri = csvread(fullfile(OUTDIR, 'signals.csv'));
X = [Xri(:,1)+1i*Xri(:,2), Xri(:,3)+1i*Xri(:,4), Xri(:,5)+1i*Xri(:,6)];

[ja, jomega, jupsilon, jxi] = instmom(1, X, 1, 2);

% ja, jomega, jupsilon, jxi are all REAL (jointmom's jupsilon/jxi are
% magnitudes -- sqrt of a weighted mean of |.|^2 -- not complex, even
% though the per-channel univariate xi is complex).
dlmwrite(fullfile(OUTDIR, 'jl_ja.csv'), ja, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_jomega.csv'), jomega, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_jupsilon.csv'), jupsilon, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_jxi.csv'), jxi, 'precision', '%.17g');

fprintf('jLab instmom joint moments computed: size=%s\n', mat2str(size(ja)));
