% Companion to jlab_crosscheck_fig3_ridges.jl -- runs REAL jLab
% (wavetrans.m, ridgewalk.m) on the exported GDP drifter 44000 (2005) full
% record and eddy-band frequency grid, both rotary senses, matching
% flare_fig3_rotary_wavelet_ridge.jl's actual setup exactly (unlike
% jlab_crosscheck_flare_fig23.m's Part B, which used a 2000-hr subset, a
% generic frequency grid, and CCW only).
%
% Usage: matlab -batch "run('scripts/jlab_crosscheck_fig3_ridges.m')"

JLAB_PATH = '/Users/alexdominguez/ADominguez/TOOLS/MATLAB/jlab';
addpath(JLAB_PATH);
jlab_addpath;

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'jlab_crosscheck_fig3_ridges');

uv = csvread(fullfile(OUTDIR, 'uv_full.csv'));
u = uv(:,1); v = uv(:,2);
w = u + 1i*v;                          % cm/s, w = u + iv
fs = csvread(fullfile(OUTDIR, 'fs_eddy.csv'));
ga = 3; be = 3;

fprintf('wavetrans.m: N=%d, n_freqs=%d, beta=%g, gamma=%g ...\n', length(w), length(fs), be, ga);

% Same complex-input convention as jlab_crosscheck_flare_fig23.m:
% WP=(1/sqrt(2))*(WX+iWY) for CCW, WN=(1/sqrt(2))*(WX-iWY) for CW.
wp = wavetrans(w, {ga, be, fs(:)', 'bandpass'}, 'mirror');
wn = wavetrans(conj(w), {ga, be, fs(:)', 'bandpass'}, 'mirror');

dlmwrite(fullfile(OUTDIR, 'jl_wp_re.csv'), real(wp), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_wp_im.csv'), imag(wp), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_wn_re.csv'), real(wn), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_wn_im.csv'), imag(wn), 'precision', '%.17g');
fprintf('  wavetrans done: size(wp)=%s\n', mat2str(size(wp)));

% Ridgewalk, BOTH senses this time (jlab_crosscheck_flare_fig23.m only
% ever ran the CCW/wp branch). P=sqrt(beta*gamma), M=1, matching
% ValTools' own min_cycles=2*sqrt(beta*gamma)/pi and alpha=0.25 defaults
% for this same beta=3,gamma=3 pair -- see jlab_crosscheck_flare_fig23.m's
% own comment for the derivation.
P = sqrt(ga * be);
M = 1;

fprintf('ridgewalk.m: CCW branch ...\n');
[wr_ccw, ir_ccw, jr_ccw, omega_ccw] = ridgewalk(1, wp, fs(:)', P, M);
dlmwrite(fullfile(OUTDIR, 'jl_ridge_ccw_ir.csv'), ir_ccw, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_ccw_omega.csv'), omega_ccw, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_ccw_wr_re.csv'), real(wr_ccw), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_ccw_wr_im.csv'), imag(wr_ccw), 'precision', '%.17g');
fprintf('  CCW: %d ridge point(s) incl. NaN separators\n', length(ir_ccw));

fprintf('ridgewalk.m: CW branch ...\n');
[wr_cw, ir_cw, jr_cw, omega_cw] = ridgewalk(1, wn, fs(:)', P, M);
dlmwrite(fullfile(OUTDIR, 'jl_ridge_cw_ir.csv'), ir_cw, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_cw_omega.csv'), omega_cw, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_cw_wr_re.csv'), real(wr_cw), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_cw_wr_im.csv'), imag(wr_cw), 'precision', '%.17g');
fprintf('  CW: %d ridge point(s) incl. NaN separators\n', length(ir_cw));

fprintf('\nAll jLab outputs written to %s\n', OUTDIR);
