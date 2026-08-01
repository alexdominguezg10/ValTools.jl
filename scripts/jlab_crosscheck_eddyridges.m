% Part 2 of 3 of the jLab cross-check harness (jlab_crosscheck_*).
%
% Runs jLab's REAL eddyridges.m (not our Julia port) on every segment
% exported by jlab_crosscheck_export.jl, using the same parameters as
% jLab's own Gulf-of-Mexico-census call
% (jFigures/makefigs_gulfcensus.m:460, onegulfeddy=eddyridges(...,2,1/64,
% sqrt(6),1,0)): FMAX=2, FMIN=1/64, P=sqrt(6), M=1, RHO=0.
%
% Writes one row per jLab-found ridge to jlab_ridges.csv for
% jlab_crosscheck_compare.jl to diff against our own Julia output.
%
% Requires the jLab toolbox; set JLAB_PATH below if it's not at the
% default location.
%
% Usage (from ValTools.jl root, or adjust OUTDIR):
%   matlab -batch "run('scripts/jlab_crosscheck_eddyridges.m')"

JLAB_PATH = '/Users/alexdominguez/ADominguez/TOOLS/MATLAB/jlab';
addpath(JLAB_PATH);
jlab_addpath;

OUTDIR = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'jlab_crosscheck');

manifest = readtable(fullfile(OUTDIR, 'manifest.csv'));
n_cases = height(manifest);

out_fid = fopen(fullfile(OUTDIR, 'jlab_ridges.csv'), 'w');
fprintf(out_fid, 'case_ridge_id,sub_idx,t0,t1,npts,L,omega_day,xi\n');

for k = 1:n_cases
    rid = manifest.ridge_id(k);
    fname = fullfile(OUTDIR, sprintf('seg_%d.csv', rid));
    if ~isfile(fname)
        fprintf('MISSING %s, skipping\n', fname);
        continue
    end
    M = csvread(fname);
    num = M(:,1); lat = M(:,2); lon = M(:,3);

    fprintf('--- case %d/%d: ridge %d, N=%d ---\n', k, n_cases, rid, length(num));

    try
        s = eddyridges(num, lat, lon, 2, 1/64, sqrt(6), 1, 0);
    catch err
        fprintf('  ERROR: %s\n', err.message);
        continue
    end

    for r = 1:length(s.num)
        nn = s.num{r};
        if isempty(nn)
            continue
        end
        om = s.omega{r};
        xi = s.xi{r};
        len = s.len{r};
        fprintf(out_fid, '%d,%d,%.6f,%.6f,%d,%.4f,%.5f,%.4f\n', ...
            rid, r, nn(1), nn(end), length(nn), len(end), mean(om), mean(xi));
    end
end

fclose(out_fid);
fprintf('\nDone. Wrote %s\n', fullfile(OUTDIR, 'jlab_ridges.csv'));
fprintf('Next: julia --project=envs/cpu scripts/jlab_crosscheck_compare.jl\n');
