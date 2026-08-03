"""
ValToolsCUDAExt — GPU acceleration for ValTools.JLab

Loaded automatically when both ValTools and CUDA are in the environment.
Provides GPU-accelerated implementations of:
- Wavelet transforms (wavetrans, wavetrans_batch) via CUFFT
- Spectral analysis (mspec) via batched GPU FFTs

The strategy:
1. Move signal + filter bank to GPU as CuArrays
2. Use CUDA's FFT (CUFFT) — same underlying library as MATLAB's GPU fft()
3. Batched broadcast-multiply replaces per-scale loops
4. Move results back to CPU

For a 100k-sample signal with 64 scales, this gives ~20–50× speedup on H200.
"""
module ValToolsCUDAExt

using ValTools
using CUDA
using CUDA.CUFFT
using Random
using Multitaper: dpss_tapers

const JL = ValTools.JLab

import ValTools.JLab: _wavetrans_nd_gpu,
                      spectral_multitaper_gpu, spectral_multitaper_batch_gpu

# ============================================================================
# GPU WAVELET TRANSFORM — N-D engine backend
# One implementation for every wavetrans entry point (vector, matrix,
# higher-rank trailing dims, rotary complex input): receives the
# already-padded (N_fft × n_sig) signal matrix plus the boundary `offset`,
# so :zeros and :mirror both extract the correct N samples (the old
# single-signal GPU path hardcoded [1:N] and silently returned the wrong
# window under boundary=:mirror). Chunks across signals to keep CUFFT
# plans within GPU memory.
# ============================================================================

function JL._wavetrans_nd_gpu(X_pad::AbstractMatrix{<:Union{Float64, ComplexF64}},
                              bank::Matrix{Float64}, N::Int, offset::Int)
    N_fft, n_sig = size(X_pad)
    n_scales = size(bank, 2)

    d_bank = CuArray(ComplexF64.(bank))   # complex for conj
    wt = Array{ComplexF64, 3}(undef, N, n_scales, n_sig)

    # Chunk: target ~16 GB per 3D array (H200 has 141 GB)
    bytes_per_element = 16  # ComplexF64
    max_bytes = 16 * 1024^3
    chunk_sig = max(1, min(n_sig, floor(Int, max_bytes / (N_fft * n_scales * bytes_per_element))))

    for s_start in 1:chunk_sig:n_sig
        s_end = min(s_start + chunk_sig - 1, n_sig)
        nc = s_end - s_start + 1

        d_X_chunk = CuArray(ComplexF64.(X_pad[:, s_start:s_end]))

        d_Xf = CUFFT.fft(d_X_chunk, 1)

        d_Xf3   = reshape(d_Xf, N_fft, 1, nc)
        d_bank3 = reshape(conj.(d_bank), N_fft, n_scales, 1)
        d_prod  = d_Xf3 .* d_bank3

        d_result = CUFFT.ifft(d_prod, 1)

        wt[:, :, s_start:s_end] .= Array(d_result[offset+1:offset+N, :, :])
    end

    return wt
end

# ============================================================================
# AUTO-DETECT: CuArray input → GPU path
# Forwards ALL kwargs (the old CuVector method dropped fs and boundary,
# making them silently unusable on this path); the trailing explicit
# gpu=true wins over any gpu in kwargs.
# ============================================================================

function JL.wavetrans(X::CuArray{<:Number}; kwargs...)
    JL.wavetrans(Array(X); kwargs..., gpu=true)
end

# ============================================================================
# GPU MULTITAPER SPECTRAL ANALYSIS
# ============================================================================

include("spectral_multitaper_gpu.jl")

# ============================================================================
# GPU SIGMA COORDINATES — sigma_to_z, interp_z, lic_texture
# ============================================================================

include("sigma_gpu.jl")
include("interp_z_gpu.jl")
include("lic_gpu.jl")

# ── Init ────────────────────────────────────────────────────────────────────

function __init__()
    if CUDA.functional()
        @info "ValTools CUDA: GPU acceleration activated ($(CUDA.device()))" *
              " — sigma_to_z_gpu, interp_z_gpu, lic_texture_gpu, wavetrans(gpu=true)"
    else
        @warn "ValTools CUDA: CUDA loaded but not functional, GPU path unavailable"
    end
end

end # module
