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

import ValTools.JLab: _wavetrans_gpu, _wavetrans_batch_gpu

# ============================================================================
# GPU WAVELET TRANSFORM — single signal
# ============================================================================

function JL._wavetrans_gpu(x_pad::Vector{Float64}, bank::Matrix{Float64},
                           N::Int, N_fft::Int, n_scales::Int)

    d_x    = CuArray(ComplexF64.(x_pad))
    d_bank = CuArray(ComplexF64.(bank))   # complex for conj

    d_X = CUFFT.fft(d_x)

    # Chunk scales to fit in GPU memory
    chunk_size = min(n_scales, 128)
    wt = zeros(ComplexF64, N, n_scales)

    for j_start in 1:chunk_size:n_scales
        j_end = min(j_start + chunk_size - 1, n_scales)
        n_chunk = j_end - j_start + 1

        d_X_rep = repeat(d_X, 1, n_chunk)
        d_prod  = d_X_rep .* conj.(@view(d_bank[:, j_start:j_end]))
        d_result = CUFFT.ifft(d_prod, 1)

        wt[:, j_start:j_end] .= Array(d_result[1:N, :])
    end

    return wt
end

# ============================================================================
# GPU WAVELET TRANSFORM — batch of signals
# Chunks across signals to keep CUFFT plans within GPU memory.
# ============================================================================

function JL._wavetrans_batch_gpu(X::Matrix{Float64}, bank::Matrix{Float64},
                                 N::Int, N_fft::Int, n_scales::Int, n_sig::Int)

    d_bank = CuArray(ComplexF64.(bank))   # complex for conj
    wt = zeros(ComplexF64, N, n_scales, n_sig)

    # Chunk: target ~16 GB per 3D array (H200 has 141 GB)
    bytes_per_element = 16  # ComplexF64
    max_bytes = 16 * 1024^3
    chunk_sig = max(1, min(n_sig, floor(Int, max_bytes / (N_fft * n_scales * bytes_per_element))))

    for s_start in 1:chunk_sig:n_sig
        s_end = min(s_start + chunk_sig - 1, n_sig)
        nc = s_end - s_start + 1

        X_chunk = vcat(X[:, s_start:s_end], zeros(N_fft - N, nc))
        d_X_chunk = CuArray(ComplexF64.(X_chunk))

        d_Xf = CUFFT.fft(d_X_chunk, 1)

        d_Xf3   = reshape(d_Xf, N_fft, 1, nc)
        d_bank3 = reshape(conj.(d_bank), N_fft, n_scales, 1)
        d_prod  = d_Xf3 .* d_bank3

        d_result = CUFFT.ifft(d_prod, 1)

        wt[:, :, s_start:s_end] .= Array(d_result[1:N, :, :])
    end

    return wt
end

# ============================================================================
# AUTO-DETECT: CuArray input → GPU path
# ============================================================================

function JL.wavetrans(x::CuVector;
                      dt::Real=1.0,
                      scales::Union{AbstractVector, Nothing}=nothing,
                      nv::Int=8,
                      mother::String="morse",
                      gamma::Real=3.0,
                      beta::Real=8.0,
                      gpu::Bool=true)
    JL.wavetrans(Array(x); dt, scales, nv, mother, gamma, beta, gpu=true)
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
