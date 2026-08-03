# GPU verification for the N-D wavetrans engine (run on Ixachi, oc_gpu).
# Checks, on real hardware:
#   1. GPU == CPU for the N-D path, boundary=:zeros
#   2. GPU == CPU for boundary=:mirror  <- the defect fixed 2026-08-03:
#      the old GPU path hardcoded [1:N] extraction and silently returned
#      the wrong window under :mirror
#   3. CuArray auto-detect path forwards fs/boundary kwargs (second fixed
#      defect) and matches the explicit gpu=true call
#
# Usage (from /LUSTRE/adomingu/ValTools.jl, in a CUDA+Multitaper-equipped
# env, e.g. the valtools_gpu_scratch_env pattern):
#   julia --project=<env> scripts/verify_gpu_wavetrans_nd.jl
#
# NOTE: Multitaper must be loaded even though nothing below calls it.
# ValToolsCUDAExt is declared in Project.toml as triggered by BOTH CUDA
# and Multitaper (`ValToolsCUDAExt = ["CUDA", "Multitaper"]` -- shared
# with the spectral_multitaper_gpu path in the same extension file), so
# `using CUDA` alone never activates it: _wavetrans_nd_gpu silently falls
# through to the error-stub in src/JLab/wavelets.jl instead of dispatching
# to the real GPU method. Same category of gotcha as the GOMED-era `ext/`
# sync miss (project_gomed_validation_results memory) -- a weakdep
# extension trigger not firing, not a logic bug in the code it gates.

using CUDA
using Multitaper  # required to activate ValToolsCUDAExt -- see note above
using ValTools.JLab

@assert CUDA.functional() "CUDA not functional on this node"

N, n_sig = 4096, 32
X = randn(N, n_sig)

for boundary in (:zeros, :mirror)
    wt_cpu, fs = wavetrans(X; boundary=boundary)
    wt_gpu, _  = wavetrans(X; boundary=boundary, gpu=true)
    md = maximum(abs.(wt_cpu .- wt_gpu)) / maximum(abs.(wt_cpu))
    println("boundary=$boundary: GPU vs CPU max rel diff = $md")
    @assert md < 1e-10 "GPU/CPU mismatch for boundary=$boundary"
end

# CuArray auto-detect with explicit fs + mirror (previously dropped kwargs)
x = randn(N)
_, fs0 = wavetrans(x)
fs_sub = fs0[1:2:end]
wt_ref, _ = wavetrans(x; fs=fs_sub, boundary=:mirror, gpu=true)
wt_cu, fs_cu = wavetrans(CuArray(x); fs=fs_sub, boundary=:mirror)
@assert fs_cu == fs_sub "CuArray path ignored fs kwarg"
md = maximum(abs.(wt_ref .- wt_cu)) / maximum(abs.(wt_ref))
println("CuArray auto-detect (fs+mirror): max rel diff vs explicit gpu=true = $md")
@assert md < 1e-12

println("\nALL GPU CHECKS PASSED")
