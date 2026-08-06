% Companion to jlab_crosscheck_flare_fig23.jl -- runs REAL jLab (mspec.m,
% sleptap.m, wavetrans.m, ridgewalk.m) on the exported GDP drifter 44000
% (2005) data, so the Julia driver can diff ValTools' Metrics.rotary_spectrum
% and JLab.rotary_wavetrans/ridgechains_jlab against jLab's own output.
%
% Usage: matlab -batch "run('scripts/jlab_crosscheck_flare_fig23.m')"

JLAB_PATH = '/Users/alexdominguez/ADominguez/TOOLS/MATLAB/jlab';
addpath(JLAB_PATH);
jlab_addpath;

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'jlab_crosscheck_flare_fig23');

% ═══════════════════════════════════════════════════════════════════════
% PART A: rotary multitaper spectrum, K=15 tapers, nw=8, full 2005 record
% ═══════════════════════════════════════════════════════════════════════
fprintf('Part A: mspec.m rotary spectrum (K=15, nw=8) ...\n');
uv = csvread(fullfile(OUTDIR, 'uv_full.csv'));
u = uv(:,1); v = uv(:,2);
z = u + 1i*v;                       % cm/s, w = u + iv

N = length(z);
[psi, lambda] = sleptap(N, 8, 15);  % P=8 (time-bandwidth), K=15 tapers
[f, spp, snn] = mspec(1, z, psi, 'demean');   % dt=1 (hour), demean only (matches our detrend="constant")

dlmwrite(fullfile(OUTDIR, 'jl_mspec_f.csv'), f, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_mspec_spp.csv'), spp, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_mspec_snn.csv'), snn, 'precision', '%.17g');
fprintf('  mspec done: %d frequencies\n', length(f));

% ═══════════════════════════════════════════════════════════════════════
% PART B: rotary wavelet transform (beta=3, gamma=3, mirror boundary) +
% single-branch ridgewalk on the CCW (positive-rotary) transform, on a
% 2000-hr subset.
% ═══════════════════════════════════════════════════════════════════════
fprintf('Part B: wavetrans.m + ridgewalk.m (beta=3, gamma=3, mirror) ...\n');
uv_sub = csvread(fullfile(OUTDIR, 'uv_sub.csv'));
u_sub = uv_sub(:,1); v_sub = uv_sub(:,2);
w_sub = u_sub + 1i*v_sub;
fs = csvread(fullfile(OUTDIR, 'fs_shared.csv'));
ga = 3; be = 3;

% Direct complex-input rotary transform: WP=(1/sqrt(2))*(WX+iWY),
% WN=(1/sqrt(2))*(WX-iWY) -- jLab's own documented convention for
% wavetrans(x+iy,psi)/wavetrans(x-iy,psi). This is the exact path
% ValTools' rotary_wavetrans is meant to reproduce.
wp = wavetrans(w_sub, {ga, be, fs(:)', 'bandpass'}, 'mirror');
wn = wavetrans(conj(w_sub), {ga, be, fs(:)', 'bandpass'}, 'mirror');

dlmwrite(fullfile(OUTDIR, 'jl_wp_re.csv'), real(wp), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_wp_im.csv'), imag(wp), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_wn_re.csv'), real(wn), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_wn_im.csv'), imag(wn), 'precision', '%.17g');
fprintf('  wavetrans done: size(wp)=%s\n', mat2str(size(wp)));

% Ridge on the CCW (positive-rotary) branch alone, single-transform form:
% [WR,IR,JR,OMEGA]=RIDGEWALK(DT,W,FS,P,M). P=sqrt(beta*gamma) characterizes
% the wavelet; M=1 means the minimum ridge length is exactly M*(2P/pi)
% cycles = 2*3/pi ~= 1.9099 cycles, matching ValTools' own
% min_cycles=2*sqrt(beta*gamma)/pi default used on the Julia side for this
% same beta=3,gamma=3 pair. alpha (chaining aggressiveness) defaults to
% 1/4 in ridgewalk.m, matching ValTools' own alpha=0.25 default.
P = sqrt(ga * be);
M = 1;
[wr, ir, jr, omega] = ridgewalk(1, wp, fs(:)', P, M);

dlmwrite(fullfile(OUTDIR, 'jl_ridge_ir.csv'), ir, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_omega.csv'), omega, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_wr_re.csv'), real(wr), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_wr_im.csv'), imag(wr), 'precision', '%.17g');
fprintf('  ridgewalk done: %d ridge point(s) (including NaN separators)\n', length(ir));

fprintf('\nAll jLab outputs written to %s\n', OUTDIR);
