% Part 2 of the N-D wavetrans cross-check (jlab_crosscheck_wavetrans_nd.jl
% is the driver — it exports the inputs, invokes this via `matlab -batch`,
% and diffs the outputs).
%
% Runs jLab's REAL wavetrans.m (not our Julia port) on:
%   (a) the full (N x n_sig) signal MATRIX in one call — jLab's own
%       column-signal batching, the semantics ValTools' N-D wavetrans ports;
%   (b) each column separately — to confirm jLab's matrix path is itself
%       identical to its per-column path (sanity check on (a)).
% Both use the exported shared frequency grid and 'mirror' boundary (the
% boundary variant our port matches jLab's timeseries_boundary.m exactly on).
%
% Writes, per column k: jl_wt_<k>_re.csv / jl_wt_<k>_im.csv (matrix-call
% output), and a summary.csv with jLab's own matrix-vs-percolumn max diff.

JLAB_PATH = '/Users/alexdominguez/ADominguez/TOOLS/MATLAB/jlab';
addpath(JLAB_PATH);
jlab_addpath;

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'jlab_crosscheck_wavetrans_nd');

X  = csvread(fullfile(OUTDIR, 'signals.csv'));      % (N x n_sig)
fs = csvread(fullfile(OUTDIR, 'fs.csv'));           % (n_fs x 1), rad/sample
ga = 3; be = 8;
[N, n_sig] = size(X);

% (a) one matrix call — jLab returns (N x n_fs x n_sig)
wt_mat = wavetrans(X, {ga, be, fs(:)', 'bandpass'}, 'mirror');

% (b) per-column calls, and jLab-internal consistency
max_internal = 0;
for k = 1:n_sig
    wt_k = wavetrans(X(:,k), {ga, be, fs(:)', 'bandpass'}, 'mirror');
    max_internal = max(max_internal, max(abs(wt_mat(:,:,k) - wt_k), [], 'all'));
    dlmwrite(fullfile(OUTDIR, sprintf('jl_wt_%d_re.csv', k)), real(wt_mat(:,:,k)), 'precision', '%.17g');
    dlmwrite(fullfile(OUTDIR, sprintf('jl_wt_%d_im.csv', k)), imag(wt_mat(:,:,k)), 'precision', '%.17g');
end

fid = fopen(fullfile(OUTDIR, 'summary.csv'), 'w');
fprintf(fid, 'n_sig,n_fs,max_matrix_vs_percolumn\n');
fprintf(fid, '%d,%d,%.17g\n', n_sig, length(fs), max_internal);
fclose(fid);

fprintf('jLab wavetrans done: %d cols, %d freqs, matrix-vs-percolumn max diff = %.3g\n', ...
        n_sig, length(fs), max_internal);
