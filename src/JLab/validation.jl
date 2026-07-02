"""
Model-Observation Spectral Validation

Connects JLab spectral analysis (wavelet, multitaper, rotary) with ValTools
validation metrics to provide spectral-domain model skill assessment.

Typical workflow:
1. Colocate model and observation time series (use ValTools.Colocation)
2. Call `validate_spectra()` for frequency-domain skill scores
3. Call `kinetic_energy_budget()` for band-partitioned energy comparison
4. Call `eddy_census()` for wavelet-ridge-based eddy detection comparison

**References:**
- Lilly, J. M. & S. C. Olhede (2009). Bivariate instantaneous frequency…
- Lilly, J. M. & P. Perez-Brunius (2021). Extracting statistically significant
  eddy signals from large Lagrangian datasets using wavelet ridge analysis,
  with application to the Gulf of Mexico. Nonlinear Processes in Geophysics.
  https://doi.org/10.5194/npg-28-181-2021
- Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
"""

import FFTW: fft, fftfreq

# ============================================================================
# STANDARD FREQUENCY BANDS (oceanographic)
# ============================================================================

const FREQ_BANDS_HOURS = Dict{String, Tuple{Float64, Float64}}(
    "subtidal"    => (0.0,      1.0/36.0),      # > 36 h
    "diurnal"     => (1.0/28.0, 1.0/20.0),      # 20–28 h
    "semidiurnal" => (1.0/14.0, 1.0/11.0),      # 11–14 h
    "inertial"    => (0.0,      0.0),            # placeholder — set by latitude
    "supertidal"  => (1.0/6.0,  0.5),            # < 6 h
    "mesoscale"   => (1.0/720.0, 1.0/48.0),     # 2–30 days
    "submesoscale" => (1.0/48.0, 1.0/6.0),      # 6–48 h
)

"""
    inertial_frequency(lat::Real)

Local inertial frequency in cycles per hour.

f = 2Ω sin(φ), returned in cycles/hour.
"""
function inertial_frequency(lat::Real)
    omega_earth = 7.2921e-5   # rad/s
    f_rad_s = 2 * omega_earth * abs(sin(deg2rad(lat)))
    return f_rad_s / (2π) * 3600.0   # cycles per hour
end

"""
    get_freq_band(name::String; lat::Real=25.0, width::Real=0.2)

Get frequency band limits in cycles/hour.

For `"inertial"`, uses latitude to compute local inertial frequency
with ± `width` fractional bandwidth.

Standard bands: "subtidal", "diurnal", "semidiurnal", "inertial",
"supertidal", "mesoscale", "submesoscale".
"""
function get_freq_band(name::String; lat::Real=25.0, width::Real=0.2)
    if name == "inertial"
        f_i = inertial_frequency(lat)
        return (f_i * (1 - width), f_i * (1 + width))
    end
    haskey(FREQ_BANDS_HOURS, name) || error("Unknown band: $name. " *
        "Available: $(join(keys(FREQ_BANDS_HOURS), ", "))")
    return FREQ_BANDS_HOURS[name]
end

# ============================================================================
# SPECTRAL VALIDATION
# ============================================================================

"""
    validate_spectra(model::AbstractVector, obs::AbstractVector, dt::Real;
                     ntapers=5, freq_bands=String[], lat=25.0,
                     gpu=false)

Compare model and observation power spectral densities.

# Arguments
- `model`, `obs`: Time series (same length, same dt)
- `dt`: Sampling interval (hours)
- `ntapers`: Number of Slepian tapers for multitaper estimate
- `freq_bands`: Named frequency bands to evaluate (e.g., `["inertial", "mesoscale"]`)
- `lat`: Latitude for inertial frequency calculation
- `gpu`: Use GPU FFTs

# Returns
`Dict` with keys:
- `"freqs"`, `"psd_model"`, `"psd_obs"`: Full spectra
- `"log_spectral_ratio"`: log10(PSD_model / PSD_obs) — 0 = perfect
- `"spectral_correlation"`: Pearson r of log(PSD) — how well shapes match
- `"spectral_slope_model"`, `"spectral_slope_obs"`: Fitted k^α slopes
- Per-band entries (if `freq_bands` given):
  `"<band>_energy_model"`, `"<band>_energy_obs"`, `"<band>_ratio"`

# References
Thomson (1982); jSpectral/mspec.m

# Example
```julia
dt_hours = 1.0
skill = validate_spectra(u_model, u_obs, dt_hours;
                         freq_bands=["inertial", "semidiurnal", "mesoscale"],
                         lat=25.0)
println("Spectral correlation: ", skill["spectral_correlation"])
println("Inertial energy ratio: ", skill["inertial_ratio"])
```
"""
function validate_spectra(model::AbstractVector, obs::AbstractVector, dt::Real;
                          ntapers::Int=5, freq_bands::Vector{String}=String[],
                          lat::Real=25.0, gpu::Bool=false)

    length(model) == length(obs) || error("model and obs must have same length")

    f_m, psd_m = mspec(collect(Float64, model), Float64(dt); ntapers=ntapers, gpu=gpu)
    f_o, psd_o = mspec(collect(Float64, obs),   Float64(dt); ntapers=ntapers, gpu=gpu)

    # Skip DC
    valid = f_m .> 0
    f_m   = f_m[valid];   psd_m = psd_m[valid]
    f_o   = f_o[valid];   psd_o = psd_o[valid]

    result = Dict{String, Any}(
        "freqs"     => f_m,
        "psd_model" => psd_m,
        "psd_obs"   => psd_o,
    )

    # Log spectral ratio (0 = perfect match)
    pos = (psd_m .> 0) .& (psd_o .> 0)
    log_ratio = fill(NaN, length(f_m))
    log_ratio[pos] .= log10.(psd_m[pos] ./ psd_o[pos])
    result["log_spectral_ratio"] = log_ratio
    result["mean_log_spectral_ratio"] = mean(log_ratio[pos])

    # Spectral correlation (shape similarity)
    if sum(pos) > 3
        lm = log10.(psd_m[pos])
        lo = log10.(psd_o[pos])
        r = cor(lm, lo)
        result["spectral_correlation"] = isfinite(r) ? r : NaN
    else
        result["spectral_correlation"] = NaN
    end

    # Spectral slope fit: PSD ~ f^α in log-log
    result["spectral_slope_model"] = _fit_spectral_slope(f_m[pos], psd_m[pos])
    result["spectral_slope_obs"]   = _fit_spectral_slope(f_o[pos], psd_o[pos])

    # Band-integrated energy
    for band_name in freq_bands
        f_lo, f_hi = get_freq_band(band_name; lat=lat)
        in_band = (f_m .>= f_lo) .& (f_m .<= f_hi)

        if any(in_band)
            df = length(f_m) > 1 ? f_m[2] - f_m[1] : 1.0
            e_m = sum(psd_m[in_band]) * df
            e_o = sum(psd_o[in_band]) * df
            result["$(band_name)_energy_model"] = e_m
            result["$(band_name)_energy_obs"]   = e_o
            result["$(band_name)_ratio"] = e_o > 0 ? e_m / e_o : NaN
        else
            result["$(band_name)_energy_model"] = 0.0
            result["$(band_name)_energy_obs"]   = 0.0
            result["$(band_name)_ratio"]        = NaN
        end
    end

    return result
end

# ============================================================================
# ROTARY SPECTRAL VALIDATION
# ============================================================================

"""
    validate_rotary(u_model, v_model, u_obs, v_obs, dt;
                    ntapers=5, freq_bands=String[], lat=25.0)

Compare rotary spectra (CW/CCW) between model and observations.

Rotary decomposition separates clockwise (inertial, eddies in NH)
from counter-clockwise (tidal, wind-driven) energy.

# Returns
`Dict` with:
- `"freqs"`, `"cw_model"`, `"ccw_model"`, `"cw_obs"`, `"ccw_obs"`
- `"cw_correlation"`, `"ccw_correlation"`: Shape match per component
- `"rotary_coefficient_model"`, `"rotary_coefficient_obs"`:
  (CW − CCW)/(CW + CCW) — polarization. +1 = pure CW, −1 = pure CCW
- Per-band CW/CCW ratios

# References
Gonella (1972); Lilly (2010); jSpectral/polparams.m
"""
function validate_rotary(u_model::AbstractVector, v_model::AbstractVector,
                         u_obs::AbstractVector, v_obs::AbstractVector,
                         dt::Real;
                         ntapers::Int=5,
                         freq_bands::Vector{String}=String[],
                         lat::Real=25.0)

    length(u_model) == length(v_model) == length(u_obs) == length(v_obs) ||
        error("All inputs must have same length")

    f_m, cw_m, ccw_m = rotary(collect(Float64, u_model), collect(Float64, v_model), Float64(dt); ntapers=ntapers)
    f_o, cw_o, ccw_o = rotary(collect(Float64, u_obs),   collect(Float64, v_obs),   Float64(dt); ntapers=ntapers)

    n = min(length(f_m), length(f_o))
    f_m = f_m[1:n]; cw_m = cw_m[1:n]; ccw_m = ccw_m[1:n]
    f_o = f_o[1:n]; cw_o = cw_o[1:n]; ccw_o = ccw_o[1:n]

    result = Dict{String, Any}(
        "freqs"     => f_m,
        "cw_model"  => cw_m,  "ccw_model"  => ccw_m,
        "cw_obs"    => cw_o,  "ccw_obs"    => ccw_o,
    )

    # Correlations in log space
    pos_cw  = (cw_m .> 0)  .& (cw_o .> 0)
    pos_ccw = (ccw_m .> 0) .& (ccw_o .> 0)

    if sum(pos_cw) > 3
        result["cw_correlation"] = cor(log10.(cw_m[pos_cw]), log10.(cw_o[pos_cw]))
    else
        result["cw_correlation"] = NaN
    end
    if sum(pos_ccw) > 3
        result["ccw_correlation"] = cor(log10.(ccw_m[pos_ccw]), log10.(ccw_o[pos_ccw]))
    else
        result["ccw_correlation"] = NaN
    end

    # Rotary coefficient: (CW-CCW)/(CW+CCW)
    total_m = cw_m .+ ccw_m
    total_o = cw_o .+ ccw_o
    rc_m = fill(NaN, n);  rc_o = fill(NaN, n)
    for i in 1:n
        total_m[i] > 0 && (rc_m[i] = (cw_m[i] - ccw_m[i]) / total_m[i])
        total_o[i] > 0 && (rc_o[i] = (cw_o[i] - ccw_o[i]) / total_o[i])
    end
    result["rotary_coefficient_model"] = rc_m
    result["rotary_coefficient_obs"]   = rc_o

    # Band energies
    for band_name in freq_bands
        f_lo, f_hi = get_freq_band(band_name; lat=lat)
        in_band = (f_m .>= f_lo) .& (f_m .<= f_hi)
        if any(in_band)
            df = length(f_m) > 1 ? f_m[2] - f_m[1] : 1.0
            result["$(band_name)_cw_model"]  = sum(cw_m[in_band]) * df
            result["$(band_name)_ccw_model"] = sum(ccw_m[in_band]) * df
            result["$(band_name)_cw_obs"]    = sum(cw_o[in_band]) * df
            result["$(band_name)_ccw_obs"]   = sum(ccw_o[in_band]) * df
        end
    end

    return result
end

# ============================================================================
# KINETIC ENERGY BUDGET
# ============================================================================

"""
    kinetic_energy_budget(u::AbstractVector, v::AbstractVector, dt::Real;
                          freq_bands=Dict{String,Tuple{Float64,Float64}}(),
                          lat=25.0, ntapers=5)

Partition kinetic energy by frequency bands using multitaper spectra.

# Arguments
- `u`, `v`: Velocity time series (m/s)
- `dt`: Sampling interval (hours)
- `freq_bands`: Named frequency bands. If empty, uses standard oceanographic bands.
- `lat`: Latitude for inertial band
- `ntapers`: Slepian tapers

# Returns
- `ke_total::Float64`: Total kinetic energy (m²/s²)
- `ke_by_band::Dict{String, Float64}`: KE in each band
- `ke_spectra::Dict`: Full spectral details (freqs, psd_u, psd_v)

# Example
```julia
ke, ke_bands, spec = kinetic_energy_budget(u, v, 1.0;
    lat=25.0)
println("Total KE: ", ke, " m²/s²")
println("Inertial KE: ", ke_bands["inertial"], " m²/s²")
println("Mesoscale KE: ", ke_bands["mesoscale"], " m²/s²")
```
"""
function kinetic_energy_budget(u::AbstractVector, v::AbstractVector, dt::Real;
                               freq_bands::Dict{String, Tuple{Float64, Float64}}=
                                   Dict{String, Tuple{Float64, Float64}}(),
                               lat::Real=25.0, ntapers::Int=5)

    length(u) == length(v) || error("u and v must have same length")

    # Default bands if not specified
    if isempty(freq_bands)
        f_i = inertial_frequency(lat)
        freq_bands = Dict(
            "subtidal"     => (0.0,        1.0/36.0),
            "diurnal"      => (1.0/28.0,   1.0/20.0),
            "semidiurnal"  => (1.0/14.0,   1.0/11.0),
            "inertial"     => (f_i * 0.8,  f_i * 1.2),
            "supertidal"   => (1.0/6.0,    0.5),
            "mesoscale"    => (1.0/720.0,  1.0/48.0),
        )
    end

    f_u, psd_u = mspec(collect(Float64, u), Float64(dt); ntapers=ntapers)
    f_v, psd_v = mspec(collect(Float64, v), Float64(dt); ntapers=ntapers)

    # Total KE = 0.5 * ∫(PSD_u + PSD_v) df
    valid = f_u .> 0
    df = length(f_u) > 1 ? f_u[2] - f_u[1] : 1.0
    ke_total = 0.5 * sum((psd_u[valid] .+ psd_v[valid])) * df

    ke_by_band = Dict{String, Float64}()
    for (name, (f_lo, f_hi)) in freq_bands
        in_band = (f_u .>= f_lo) .& (f_u .<= f_hi)
        if any(in_band)
            ke_by_band[name] = 0.5 * sum((psd_u[in_band] .+ psd_v[in_band])) * df
        else
            ke_by_band[name] = 0.0
        end
    end

    ke_spectra = Dict{String, Any}(
        "freqs" => f_u,
        "psd_u" => psd_u,
        "psd_v" => psd_v,
    )

    return ke_total, ke_by_band, ke_spectra
end

# ============================================================================
# EDDY CENSUS via wavelet ridges
# ============================================================================

"""
    eddy_census(u::AbstractVector, v::AbstractVector, dt::Real;
                lat=25.0, nv=8, gamma=3.0, beta=8.0,
                amp_thresh=0.05, min_duration=3.0, gpu=false)

Detect coherent eddy-like events via wavelet ridge analysis.

Uses the Lilly & Perez-Brunius (2021) approach: wavelet transform of
complex velocity, extract ridges, filter by amplitude & duration.

# Arguments
- `u`, `v`: Velocity time series
- `dt`: Sampling interval (hours)
- `lat`: Latitude (for inertial frequency reference)
- `nv`: Voices per octave
- `amp_thresh`: Minimum ridge amplitude (fraction of max)
- `min_duration`: Minimum eddy duration (in units of dt)
- `gpu`: Use GPU

# Returns
`Vector{Dict}` — one entry per detected eddy event:
- `"start"`, `"stop"`: Time indices
- `"duration"`: In time steps
- `"mean_frequency"`: Mean ridge frequency (cycles/hour)
- `"mean_amplitude"`: Mean wavelet amplitude
- `"sense"`: `:cw` or `:ccw` (rotation sense)

# References
Lilly & Olhede (2009); Lilly & Perez-Brunius (2021, NPG)
"""
function eddy_census(u::AbstractVector, v::AbstractVector, dt::Real;
                     lat::Real=25.0, nv::Int=8,
                     gamma::Real=3.0, beta::Real=8.0,
                     amp_thresh::Real=0.05, min_duration::Real=3.0,
                     gpu::Bool=false)

    length(u) == length(v) || error("u and v must have same length")
    N = length(u)

    # Complex velocity
    z = collect(Float64, u) .+ im .* collect(Float64, v)

    # Wavelet transform of complex velocity
    wt, scales = wavetrans(real.(z); dt=Float64(dt), nv=nv,
                           gamma=Float64(gamma), beta=Float64(beta), gpu=gpu)

    # Also transform imaginary part for CW/CCW separation
    wt_v, _ = wavetrans(imag.(z); dt=Float64(dt), scales=scales,
                        gamma=Float64(gamma), beta=Float64(beta), gpu=gpu)

    # Ridge detection
    amp = abs.(wt)
    max_amp = maximum(amp)
    thresh_abs = amp_thresh * max_amp

    ridge_freq, ridge_amp, ridge_qual = ridgemap(wt, scales; thresh=thresh_abs, quality=true)

    # Chain ridge points into events
    events = Dict[]
    in_event = false
    event_start = 0

    for i in 1:N
        if !isnan(ridge_freq[i]) && ridge_amp[i] > thresh_abs
            if !in_event
                in_event = true
                event_start = i
            end
        else
            if in_event
                duration = i - event_start
                if duration >= min_duration
                    seg = event_start:i-1
                    valid_seg = seg[.!isnan.(ridge_freq[seg])]

                    if !isempty(valid_seg)
                        # Determine rotation sense from phase progression
                        phase_u = angle.(wt[valid_seg, argmax(abs.(wt[valid_seg[1], :]))])
                        dp = diff(phase_u)
                        sense = mean(dp) > 0 ? :ccw : :cw

                        push!(events, Dict(
                            "start"          => event_start,
                            "stop"           => i - 1,
                            "duration"       => duration,
                            "mean_frequency" => mean(ridge_freq[valid_seg]),
                            "mean_amplitude" => mean(ridge_amp[valid_seg]),
                            "max_amplitude"  => maximum(ridge_amp[valid_seg]),
                            "mean_quality"   => mean(ridge_qual[valid_seg]),
                            "sense"          => sense,
                        ))
                    end
                end
                in_event = false
            end
        end
    end

    # Handle event still open at end
    if in_event
        duration = N - event_start + 1
        if duration >= min_duration
            seg = event_start:N
            valid_seg = seg[.!isnan.(ridge_freq[seg])]
            if !isempty(valid_seg)
                phase_u = angle.(wt[valid_seg, argmax(abs.(wt[valid_seg[1], :]))])
                dp = diff(phase_u)
                sense = length(dp) > 0 && mean(dp) > 0 ? :ccw : :cw
                push!(events, Dict(
                    "start"          => event_start,
                    "stop"           => N,
                    "duration"       => duration,
                    "mean_frequency" => mean(ridge_freq[valid_seg]),
                    "mean_amplitude" => mean(ridge_amp[valid_seg]),
                    "max_amplitude"  => maximum(ridge_amp[valid_seg]),
                    "mean_quality"   => mean(ridge_qual[valid_seg]),
                    "sense"          => sense,
                ))
            end
        end
    end

    return events
end

# ============================================================================
# FULL VALIDATION REPORT
# ============================================================================

"""
    validate_model_spectra(u_model, v_model, u_obs, v_obs, dt;
                           lat=25.0, ntapers=5,
                           freq_bands=["inertial","semidiurnal","mesoscale"],
                           gpu=false)

Comprehensive spectral validation of model velocities vs. observations.

Combines: multitaper PSD, rotary spectra, KE budget, and eddy census
into a single validation report.

# Returns
`Dict` with sections:
- `"scalar"`: Scalar spectral validation (u-component PSD comparison)
- `"rotary"`: CW/CCW decomposition comparison
- `"ke_model"`, `"ke_obs"`: Kinetic energy budgets
- `"eddies_model"`, `"eddies_obs"`: Detected eddy events (count, stats)
- `"summary"`: One-line skill scores

# References
Lilly & Olhede (2009); Thomson (1982); Gonella (1972)
"""
function validate_model_spectra(u_model::AbstractVector, v_model::AbstractVector,
                                u_obs::AbstractVector, v_obs::AbstractVector,
                                dt::Real;
                                lat::Real=25.0, ntapers::Int=5,
                                freq_bands::Vector{String}=
                                    ["inertial", "semidiurnal", "mesoscale"],
                                gpu::Bool=false)

    n = min(length(u_model), length(u_obs))
    um = collect(Float64, u_model[1:n])
    vm = collect(Float64, v_model[1:n])
    uo = collect(Float64, u_obs[1:n])
    vo = collect(Float64, v_obs[1:n])
    dt_f = Float64(dt)

    report = Dict{String, Any}()

    # 1. Scalar spectral validation (u-component)
    report["scalar_u"] = validate_spectra(um, uo, dt_f;
        ntapers=ntapers, freq_bands=freq_bands, lat=lat, gpu=gpu)
    report["scalar_v"] = validate_spectra(vm, vo, dt_f;
        ntapers=ntapers, freq_bands=freq_bands, lat=lat, gpu=gpu)

    # 2. Rotary spectral validation
    report["rotary"] = validate_rotary(um, vm, uo, vo, dt_f;
        ntapers=ntapers, freq_bands=freq_bands, lat=lat)

    # 3. Kinetic energy budgets
    ke_m, kb_m, ks_m = kinetic_energy_budget(um, vm, dt_f; lat=lat, ntapers=ntapers)
    ke_o, kb_o, ks_o = kinetic_energy_budget(uo, vo, dt_f; lat=lat, ntapers=ntapers)
    report["ke_model"] = Dict("total" => ke_m, "by_band" => kb_m)
    report["ke_obs"]   = Dict("total" => ke_o, "by_band" => kb_o)

    # 4. Summary scores
    su = report["scalar_u"]
    sv = report["scalar_v"]
    rot = report["rotary"]

    summary = Dict{String, Any}(
        "spectral_corr_u"    => su["spectral_correlation"],
        "spectral_corr_v"    => sv["spectral_correlation"],
        "spectral_slope_model_u" => su["spectral_slope_model"],
        "spectral_slope_obs_u"   => su["spectral_slope_obs"],
        "cw_correlation"     => rot["cw_correlation"],
        "ccw_correlation"    => rot["ccw_correlation"],
        "ke_total_model"     => ke_m,
        "ke_total_obs"       => ke_o,
        "ke_ratio"           => ke_o > 0 ? ke_m / ke_o : NaN,
    )

    # Band energy ratios
    for band in freq_bands
        key_m = "$(band)_energy_model"
        key_o = "$(band)_energy_obs"
        if haskey(su, key_m) && haskey(su, key_o)
            e_m = su[key_m]
            e_o = su[key_o]
            summary["$(band)_energy_ratio_u"] = e_o > 0 ? e_m / e_o : NaN
        end
    end

    report["summary"] = summary

    return report
end

# ============================================================================
# HELPERS
# ============================================================================

function _fit_spectral_slope(freqs::AbstractVector, psd::AbstractVector)
    # Fit PSD ~ f^α in log-log space
    lf = log10.(freqs)
    lp = log10.(psd)
    valid = isfinite.(lf) .& isfinite.(lp)
    if sum(valid) < 3
        return NaN
    end
    A = [ones(sum(valid)) lf[valid]]
    coeffs = A \ lp[valid]
    return coeffs[2]   # spectral slope α
end
