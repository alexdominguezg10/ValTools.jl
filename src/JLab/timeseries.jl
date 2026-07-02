"""
Time Series Processing & Filtering

Ported from Jonathan M. Lilly's jCommon (https://github.com/jonathanlilly/jlab)

**Topics:** Detrending, gap filling, bandpass filtering, Hilbert transform
"""

# Note: DSP.jl is optional; basic filtering using Julia's built-in capabilities

"""
    detrend(x::AbstractVector; order::Int=1, kind::String="polynomial")

Remove polynomial trend from time series.

# Arguments
- `x::AbstractVector`: Input signal
- `order::Int`: Polynomial order (1=linear, 2=quadratic, etc.)
- `kind::String`: "polynomial" (default) or "constant" (mean only)

# Returns
- `x_detrended::Vector`: Detrended signal
- `trend::Vector`: Estimated trend
- `coeffs::Vector`: Polynomial coefficients [highest order, ..., constant]

# References
jCommon/detrend.m
"""
function detrend(x::AbstractVector; order::Int=1, kind::String="polynomial")
    x = vec(x)
    n = length(x)

    if kind == "polynomial"
        # Fit polynomial
        t = collect(1:n)
        A = ones(n, order+1)
        for p in 1:order
            A[:, p+1] = t .^ p
        end

        # Least squares fit
        coeffs = A \ x

        # Reconstruct trend
        trend = A * coeffs

        return x .- trend, trend, reverse(coeffs)
    elseif kind == "constant"
        trend = fill(mean(x), n)
        return x .- trend, trend, [mean(x)]
    else
        error("Unknown detrend kind: $kind")
    end
end

"""
    bandpass(x::AbstractVector{<:Real}, dt::Real, f_low::Real, f_high::Real;
             order::Int=4, kind::String="butterworth")

Apply bandpass filter to time series.

# Arguments
- `x::AbstractVector`: Input signal
- `dt::Real`: Sampling interval
- `f_low::Real`: Lower cutoff frequency
- `f_high::Real`: Upper cutoff frequency
- `order::Int`: Filter order (not used in current FFT implementation)
- `kind::String`: Filter type ("butterworth" or "fft")

# Returns
- `x_filt::Vector`: Filtered signal

# References
jCommon/bandpass.m

# Notes
- Current implementation uses FFT-based filtering (zero-phase by design)
- For Butterworth, use DSP.jl when available
"""
function bandpass(x::AbstractVector{<:Real}, dt::Real, f_low::Real, f_high::Real;
                 order::Int=4, kind::String="fft")
    x = vec(x)
    N = length(x)

    if kind == "fft"
        # FFT-based bandpass filter
        X = fft(x)
        freqs = fftfreq(N, 1.0/dt)

        # Create mask for frequencies in [f_low, f_high]
        mask = (abs.(freqs) .>= f_low) .& (abs.(freqs) .<= f_high)

        # Zero out frequencies outside band
        X_filt = X .* mask

        # Inverse transform
        return real(ifft(X_filt))
    else
        error("Filter kind '$kind' not available (use DSP.jl for 'butterworth')")
    end
end

"""
    highpass(x::AbstractVector{<:Real}, dt::Real, f_cutoff::Real;
             order::Int=4)

Apply highpass filter (FFT-based).

# Arguments
- `x::AbstractVector`: Input signal
- `dt::Real`: Sampling interval
- `f_cutoff::Real`: Cutoff frequency

# Returns
- `x_filt::Vector`: Filtered signal
"""
function highpass(x::AbstractVector{<:Real}, dt::Real, f_cutoff::Real; order::Int=4)
    x = vec(x)
    N = length(x)

    X = fft(x)
    freqs = fftfreq(N, 1.0/dt)

    # High-pass: keep |freq| > f_cutoff
    mask = abs.(freqs) .>= f_cutoff

    X_filt = X .* mask

    return real(ifft(X_filt))
end

"""
    lowpass(x::AbstractVector{<:Real}, dt::Real, f_cutoff::Real;
            order::Int=4)

Apply lowpass filter (FFT-based).

# Arguments
- `x::AbstractVector`: Input signal
- `dt::Real`: Sampling interval
- `f_cutoff::Real`: Cutoff frequency

# Returns
- `x_filt::Vector`: Filtered signal
"""
function lowpass(x::AbstractVector{<:Real}, dt::Real, f_cutoff::Real; order::Int=4)
    x = vec(x)
    N = length(x)

    X = fft(x)
    freqs = fftfreq(N, 1.0/dt)

    # Low-pass: keep |freq| <= f_cutoff
    mask = abs.(freqs) .<= f_cutoff

    X_filt = X .* mask

    return real(ifft(X_filt))
end

"""
    fillgaps(x::AbstractVector; method::String="linear", max_gap::Int=10)

Interpolate missing values (NaN) in time series.

# Arguments
- `x::AbstractVector`: Input with NaN values
- `method::String`: "linear" (default), "cubic", or "nearest"
- `max_gap::Int`: Maximum gap size to fill

# Returns
- `x_filled::Vector`: Series with gaps filled
"""
function fillgaps(x::AbstractVector; method::String="linear", max_gap::Int=10)
    x = vec(copy(x))
    n = length(x)

    # Find gaps
    valid_idx = findall(!isnan, x)

    if length(valid_idx) < 2
        return x  # Cannot interpolate
    end

    # Interpolate missing values
    for i in 1:n
        if isnan(x[i])
            # Find neighbors
            before = findall(<(i), valid_idx)
            after = findall(>(i), valid_idx)

            if !isempty(before) && !isempty(after)
                i1 = valid_idx[last(before)]
                i2 = valid_idx[first(after)]

                if i2 - i1 <= max_gap
                    # Linear interpolation
                    w = (i - i1) / (i2 - i1)
                    x[i] = (1-w) * x[i1] + w * x[i2]
                end
            end
        end
    end

    return x
end

"""
    hilbert(x::AbstractVector{<:Real})

Analytic signal via Hilbert transform.

Computes complex analytic signal z = x + i*H(x)
where H is the Hilbert transform.

# Arguments
- `x::AbstractVector`: Real input signal

# Returns
- `z::Vector{ComplexF64}`: Analytic signal

# Notes
- |z| = instantaneous amplitude envelope
- angle(z) = instantaneous phase
- Uses FFT-based Hilbert transform

# References
jCommon Hilbert; classic DSP technique
"""
function hilbert(x::AbstractVector{<:Real})
    x = vec(x)
    n = length(x)

    # FFT
    X = fft(x)

    # Hilbert transform in frequency domain:
    # H(x) has imaginary part multiplied by -i for positive freqs, +i for negative
    H = similar(X)

    # DC and Nyquist components unchanged
    H[1] = X[1]
    if iseven(n)
        H[n÷2 + 1] = X[n÷2 + 1]
    end

    # Positive frequencies: multiply by -2i (equivalent to dividing imaginary part)
    for k in 2:ceil(Int, n/2)
        H[k] = 2 * X[k]
    end

    # Negative frequencies: zero
    if iseven(n)
        for k in n÷2 + 2:n
            H[k] = 0
        end
    else
        for k in (n+3)÷2:n
            H[k] = 0
        end
    end

    # Inverse FFT
    h = ifft(H)

    # Analytic signal: x + i*Hilbert(x)
    return x .+ im .* imag(h)
end

