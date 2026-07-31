"""
Batch multitaper spectral estimation on GPU.

Computes multitaper spectra for one or more signals efficiently on GPU
by batching all FFTs (across tapers × signals) in a single CUFFT call.
"""

import FFTW: fftfreq

function JL.spectral_multitaper_batch_gpu(X::Matrix{Float64},
                                          tapers::Matrix{Float64},
                                          lambdas::Vector{Float64},
                                          dt::Real, N_fft::Int)

    N, M = size(X)
    K = length(lambdas)

    # Step 1: Create tapered copies — broadcast multiply
    # X_tapered[i, k, m] = X[i, m] * tapers[i, k]
    X_tapered = reshape(X, N, 1, M) .* reshape(tapers, N, K, 1)  # (N, K, M)

    # Step 2: Zero-pad to N_fft
    X_padded = vcat(X_tapered, zeros(N_fft - N, K, M))  # (N_fft, K, M)

    # Step 3: Move to GPU as complex
    d_X_padded = CuArray(ComplexF64.(X_padded))

    # Step 4: Batch FFT across dimension 1 (frequency axis)
    # All N_fft * K * M FFTs computed in one CUFFT call
    d_X_hat = CUFFT.fft(d_X_padded, 1)  # (N_fft, K, M)

    # Step 5: Compute power, weight by lambda, sum across tapers
    # |X_hat[f, k, m]|^2 * lambda[k] summed over k
    d_power_raw = abs2.(d_X_hat)  # (N_fft, K, M)

    # Weight: reshape lambdas to (1, K, 1) for broadcasting
    d_lambdas = CuArray(reshape(lambdas, 1, K, 1))
    d_weighted = d_power_raw .* d_lambdas  # (N_fft, K, M) × (1, K, 1)
    d_psd = sum(d_weighted, dims=2) ./ sum(lambdas)  # (N_fft, 1, M) normalized
    d_psd = dropdims(d_psd, dims=2)  # (N_fft, M)

    # Step 6: Scale by dt (convert to power spectral density)
    d_psd .*= dt

    # Step 7: Extract positive frequencies and return to CPU
    freqs_all = fftfreq(N_fft, 1.0/dt)
    pos_mask = freqs_all .>= 0
    freqs_pos = freqs_all[pos_mask]

    psd_pos = Array(d_psd[pos_mask, :])  # (n_pos, M)

    return freqs_pos, psd_pos
end


"""
    spectral_multitaper_gpu(x::Union{Vector, Matrix}, dt::Real=1.0;
                            nw::Float64=4.0, ntapers::Int=0,
                            detrend::String="linear", gpu::Bool=true)

GPU-accelerated multitaper spectral estimation (single or batch).

# Arguments
- `x`: (N,) vector (single) or (N, M) matrix (batch of M signals)
- `dt`: sampling interval
- `nw`: time-bandwidth product (default 4.0)
- `ntapers`: number of tapers (default: 2*nw-1)
- `detrend`: "none", "constant", or "linear"
- `gpu`: use GPU if available (default true)

# Returns
MTSpectrum struct with fields:
- `f`: frequency vector
- `S`: power spectral density
- `params`: MTParameters (nw, K, dt)
"""
function JL.spectral_multitaper_gpu(x::Union{Vector{Float64}, Matrix{Float64}},
                                    dt::Real=1.0;
                                    nw::Float64=4.0, ntapers::Int=0,
                                    detrend::String="linear", gpu::Bool=true)

    # Handle both single signal (vector) and batch (matrix)
    if x isa Vector
        x = reshape(x, :, 1)  # (N,) → (N, 1)
    end

    N, M = size(x)

    # Default ntapers
    if ntapers <= 0
        ntapers = max(1, 2*floor(Int, nw) - 1)
    end

    # Detrend
    x_detrended = copy(x)
    if detrend == "linear"
        for m in 1:M
            x_detrended[:, m] = _detrend_linear_vec(x_detrended[:, m])
        end
    elseif detrend == "constant"
        for m in 1:M
            x_detrended[:, m] .-= mean(x_detrended[:, m])
        end
    end

    # Generate DPSS tapers
    tapers, lambdas = dpss_tapers(N, nw, ntapers, :both)

    # FFT length
    N_fft = 2^ceil(Int, log2(2*N - 1))

    # Call GPU function
    if gpu
        freqs, psd = JL.spectral_multitaper_batch_gpu(x_detrended, tapers,
                                                       lambdas, dt, N_fft)
    else
        error("CPU path not yet implemented for batch")
    end

    # For now, return only the first signal's spectrum
    # (Full MTSpectrum wrapper will be added in next step)
    if M == 1
        return freqs, psd[:, 1]
    else
        return freqs, psd
    end
end


function _detrend_linear_vec(x::Vector)
    n = length(x)
    t = collect(1.0:n)
    A = [ones(n) t]
    coeffs = A \ x
    return x .- A * coeffs
end
