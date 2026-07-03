# JL.spectral_multitaper / mspec / sleptap — Multitaper.jl-backed implementations.
# Moved verbatim from src/JLab/spectral.jl when Multitaper.jl became a weakdep;
# see the stub docstrings there for the public API documentation.

function JL.spectral_multitaper(x::Union{Vector{Float64}, Matrix{Float64}},
                                dt::Real=1.0;
                                nw::Float64=4.0, ntapers::Int=0,
                                detrend::String="linear",
                                unit=Unitful.NoUnits)

    # Ensure Float64
    x = Float64.(x)
    if x isa Vector
        x = reshape(x, :, 1)
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
            x_detrended[:, m] = _jlab_detrend_linear(x_detrended[:, m])
        end
    elseif detrend == "constant"
        for m in 1:M
            x_detrended[:, m] .-= mean(x_detrended[:, m])
        end
    elseif detrend != "none"
        error("detrend must be 'none', 'constant', or 'linear'")
    end

    # Call Multitaper.multispec for each signal
    # Features enabled: adaptive=true, Ftest=true, jk=true
    if M == 1
        spec = multispec(x_detrended[:, 1]; NW=nw, K=ntapers, dt=dt,
                        ctr=false, a_weight=true, Ftest=true, jk=true)
        return _to_spectral_estimate(spec, nw, ntapers, dt, detrend, unit)
    else
        # Batch: return vector of SpectralEstimate
        specs = [multispec(x_detrended[:, m]; NW=nw, K=ntapers, dt=dt,
                          ctr=false, a_weight=true, Ftest=true, jk=true)
                 for m in 1:M]
        return [_to_spectral_estimate(s, nw, ntapers, dt, detrend, unit) for s in specs]
    end
end

# Convert a Multitaper.jl MTSpectrum into a ValTools Types.SpectralEstimate
function _to_spectral_estimate(spec::MTSpectrum, nw, ntapers, dt, detrend, unit)
    freq = collect(Float64, spec.f)
    power = spec.S .* unit^2
    ftest_pval = spec.Fpval === nothing ? nothing : collect(Float64, spec.Fpval)
    jkvar = spec.jkvar === nothing ? nothing : collect(Float64, spec.jkvar)
    params = (nw=nw, ntapers=ntapers, dt=dt, detrend=detrend, N=spec.params.N)
    return ValTools.Types.SpectralEstimate(freq, power, ftest_pval, jkvar, params)
end

function JL.sleptap(n::Integer, k::Integer; bandwidth::Real=4.0)
    n > 0 || error("sleptap: n must be positive, got $n")
    k > 0 || error("sleptap: k must be positive, got $k")
    tapers, lambdas = dpss_tapers(n, Float64(bandwidth), k, :both)
    return tapers, lambdas
end

function JL.mspec(x::Union{Vector{Float64}, Matrix{Float64}}, dt::Real=1.0;
                  ntapers::Int=5, detrend::String="linear", gpu::Bool=false)
    if gpu
        JL.spectral_multitaper_gpu(x, dt; ntapers=ntapers, detrend=detrend, gpu=gpu)
    end
    spec = JL.spectral_multitaper(x, dt; ntapers=ntapers, detrend=detrend)
    return spec.freq, spec.power
end

function _jlab_detrend_linear(x::AbstractVector)
    n = length(x)
    t = collect(1.0:n)
    A = [ones(n) t]
    coeffs = A \ x
    return x .- A * coeffs
end
