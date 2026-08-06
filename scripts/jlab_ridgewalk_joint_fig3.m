% Cheap follow-up to jlab_crosscheck_fig3_ridges.m: re-runs ONLY ridgewalk.m,
% in its JOINT (bivariate) form -- [W1R,W2R,IR,JR,OMEGA]=RIDGEWALK(W1,W2,FS) --
% reusing the wp/wn wavelet transforms already saved to disk by the earlier
% run, so wavetrans.m (the expensive step on this N=8761 record) is NOT
% recomputed. This is the correct jLab entry point for what
% ValTools._rotary_ridge_properties_core actually replicates (joint local
% maxima on sqrt(|w+|^2+|w-|^2)); the earlier jlab_crosscheck_fig3_ridges.m
% called the UNIVARIATE single-transform form on wp and wn separately,
% which is a different algorithm and not a fair comparison.
%
% Usage: matlab -batch "run('scripts/jlab_ridgewalk_joint_fig3.m')"

JLAB_PATH = '/Users/alexdominguez/ADominguez/TOOLS/MATLAB/jlab';
addpath(JLAB_PATH);
jlab_addpath;

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'jlab_crosscheck_fig3_ridges');

wp = complex(csvread(fullfile(OUTDIR, 'jl_wp_re.csv')), csvread(fullfile(OUTDIR, 'jl_wp_im.csv')));
wn = complex(csvread(fullfile(OUTDIR, 'jl_wn_re.csv')), csvread(fullfile(OUTDIR, 'jl_wn_im.csv')));
fs = csvread(fullfile(OUTDIR, 'fs_eddy.csv'));
fprintf('Loaded wp/wn: size=%s, n_freqs=%d\n', mat2str(size(wp)), length(fs));

ga = 3; be = 3;
P = sqrt(ga * be);
M = 1;   % same chaining params as the (wrong-mode) univariate run, for apples-to-apples

fprintf('ridgewalk.m: JOINT form on (wp, wn) ...\n');
[wr_p, wr_n, ir, jr, omega] = ridgewalk(1, wp, wn, fs(:)', P, M);

dlmwrite(fullfile(OUTDIR, 'jl_ridge_joint_ir.csv'), ir, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_joint_omega.csv'), omega, 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_joint_wrp_re.csv'), real(wr_p), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_joint_wrp_im.csv'), imag(wr_p), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_joint_wrn_re.csv'), real(wr_n), 'precision', '%.17g');
dlmwrite(fullfile(OUTDIR, 'jl_ridge_joint_wrn_im.csv'), imag(wr_n), 'precision', '%.17g');
fprintf('  joint ridgewalk done: %d point(s) incl. NaN separators\n', length(ir));
fprintf('\nAll outputs written to %s\n', OUTDIR);
