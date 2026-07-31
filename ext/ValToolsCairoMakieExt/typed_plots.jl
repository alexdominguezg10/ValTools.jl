# Makie plotting functions dispatching on ValTools' typed structs
# (Types/types_def.jl). Axis labels are generated from Unitful units where
# the struct carries them; dimensionless (bare Float64) fields get an
# unlabeled axis. One function name per "kind" of plot, with multiple
# dispatch across the type(s) it applies to (e.g. `plot_timeseries` covers
# TimeSeriesVector, TimeSeriesMatrix, and TimeSeriesCollection) -- the
# "Makie recipes per type" from the Stage 4a roadmap entry, adapted to the
# types that actually exist (SpatialField/ValidationResult/RidgeChain from
# the original plan were never built; see Types/types_def.jl instead).

# z=1.96 is the two-sided 95% normal critical value, matching the default
# `confidence=0.95` convention used everywhere else CIs are computed in
# this package (rotary_spectrum.jl, ellipse_polarization.jl). Not derived
# from the caller's actual `confidence` field since these fields don't
# carry it in the returned struct; consumers needing an exact match should
# use `jkvar`/`ci_*` from `params.confidence` directly instead of the ribbon.
const _Z95 = 1.9600

_unit_str(::Type{Q}) where {Q<:Unitful.Quantity} = string(Unitful.unit(Q))
_unit_str(::Type{<:Number}) = ""

_axis_label(name::AbstractString, ::Type{Q}) where {Q<:Number} =
    isempty(_unit_str(Q)) ? name : "$name [$(_unit_str(Q))]"

# ── TimeSeriesVector / TimeSeriesMatrix / TimeSeriesCollection ────────

"""
    plot_timeseries(ts::Types.TimeSeriesVector; kwargs...)
    plot_timeseries(tm::Types.TimeSeriesMatrix; kwargs...)
    plot_timeseries(tc::Types.TimeSeriesCollection; kwargs...)

Plot a typed time series with a Unitful-derived y-axis label. `TimeSeriesMatrix`
draws one line per channel (legend from `channels`); `TimeSeriesCollection`
overlays each member series on its own time axis (legend from each series'
`name`), since collection members don't share a clock.

# Keyword Arguments
- `title=""`, `figsize=(900, 400)`

# Returns
`Figure`
"""
function ValTools.plot_timeseries(ts::ValTools.Types.TimeSeriesVector{Q};
                                   title::String="",
                                   figsize::Tuple{Int,Int}=(900, 400)) where {Q}
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; xlabel="Time", ylabel=_axis_label(ts.name, Q),
              title=isempty(title) ? ts.name : title)
    ax.xtickformat = _ = i -> [Dates.format(ts.time[clamp(round(Int, x), 1, length(ts.time))], "yyyy-mm-dd") for x in i]
    lines!(ax, Float64.(1:length(ts.time)), Unitful.ustrip.(ts.value); color=:steelblue)
    return fig
end

function ValTools.plot_timeseries(tm::ValTools.Types.TimeSeriesMatrix{Q};
                                   title::String="",
                                   figsize::Tuple{Int,Int}=(900, 400)) where {Q}
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; xlabel="Time", ylabel=_axis_label(tm.name, Q),
              title=isempty(title) ? tm.name : title)
    colors = Makie.wong_colors()
    x = Float64.(1:length(tm.time))
    for (j, ch) in enumerate(tm.channels)
        lines!(ax, x, Unitful.ustrip.(view(tm.value, :, j));
               color=colors[mod1(j, length(colors))], label=ch)
    end
    length(tm.channels) > 1 && axislegend(ax; position=:rt)
    return fig
end

function ValTools.plot_timeseries(tc::ValTools.Types.TimeSeriesCollection{Q};
                                   title::String="",
                                   figsize::Tuple{Int,Int}=(900, 400)) where {Q}
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; xlabel="Time", ylabel=_axis_label(tc.name, Q),
              title=isempty(title) ? tc.name : title)
    colors = Makie.wong_colors()
    for (i, s) in enumerate(tc.series)
        isempty(s.time) && continue
        t0 = s.time[1]
        x = [Dates.value(Dates.Millisecond(t - t0)) / 1000 for t in s.time]
        lines!(ax, x, Unitful.ustrip.(s.value);
               color=colors[mod1(i, length(colors))], label=s.name)
    end
    ax.xlabel = "Time since each series' start [s]"
    length(tc.series) > 1 && axislegend(ax; position=:rt)
    return fig
end

# ── SpectralEstimate ───────────────────────────────────────────────────

"""
    plot_spectrum(se::Types.SpectralEstimate; kwargs...)

Log-log power spectral density plot. Draws a jackknife confidence ribbon
from `se.jkvar` (log-power-space variance, per `Multitaper.jl` convention)
when present, and marks F-test-significant line components (`ftest_pval`
below `alpha`) when present.

# Keyword Arguments
- `alpha=0.05`: F-test significance threshold for marking peaks
- `title=""`, `figsize=(700, 450)`

# Returns
`Figure`
"""
function ValTools.plot_spectrum(se::ValTools.Types.SpectralEstimate{Q};
                                 alpha::Real=0.05,
                                 title::String="",
                                 figsize::Tuple{Int,Int}=(700, 450)) where {Q}
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; xlabel="Frequency", ylabel=_axis_label("Power", Q),
              xscale=log10, yscale=log10, title=title)
    power = Unitful.ustrip.(se.power)

    if se.jkvar !== nothing
        half = _Z95 .* sqrt.(se.jkvar)
        lo = power .* exp.(-half)
        hi = power .* exp.(half)
        band!(ax, se.freq, max.(lo, eps()), hi; color=(:steelblue, 0.25))
    end
    lines!(ax, se.freq, power; color=:steelblue, linewidth=1.5)

    if se.ftest_pval !== nothing
        sig = se.ftest_pval .< alpha
        any(sig) && scatter!(ax, se.freq[sig], power[sig]; color=:firebrick,
                              markersize=8, marker=:utriangle, label="F-test p<$alpha")
        any(sig) && axislegend(ax; position=:rt)
    end
    return fig
end

# ── RotarySpectralEstimate ─────────────────────────────────────────────

"""
    plot_rotary_spectrum(rse::Types.RotarySpectralEstimate; kwargs...)

CCW/CW rotary power spectra with jackknife CI ribbons (`ci_ccw`/`ci_cw`,
when present) and F-test significance markers (`ftest_ccw`/`ftest_cw`,
when present).

# Keyword Arguments
- `alpha=0.05`, `title=""`, `figsize=(700, 450)`

# Returns
`Figure`
"""
function ValTools.plot_rotary_spectrum(rse::ValTools.Types.RotarySpectralEstimate;
                                        alpha::Real=0.05,
                                        title::String="",
                                        figsize::Tuple{Int,Int}=(700, 450))
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; xlabel="Frequency", ylabel="Power",
              xscale=log10, yscale=log10, title=title)

    rse.ci_ccw !== nothing && band!(ax, rse.freq, max.(rse.ci_ccw[1], eps()), rse.ci_ccw[2];
                                     color=(:steelblue, 0.2))
    rse.ci_cw !== nothing && band!(ax, rse.freq, max.(rse.ci_cw[1], eps()), rse.ci_cw[2];
                                    color=(:firebrick, 0.2))
    lines!(ax, rse.freq, rse.S_ccw; color=:steelblue, linewidth=1.5, label="CCW")
    lines!(ax, rse.freq, rse.S_cw; color=:firebrick, linewidth=1.5, label="CW")

    if rse.ftest_ccw !== nothing
        sig = rse.ftest_ccw .< alpha
        any(sig) && scatter!(ax, rse.freq[sig], rse.S_ccw[sig]; color=:navy,
                              markersize=8, marker=:utriangle)
    end
    if rse.ftest_cw !== nothing
        sig = rse.ftest_cw .< alpha
        any(sig) && scatter!(ax, rse.freq[sig], rse.S_cw[sig]; color=:darkred,
                              markersize=8, marker=:utriangle)
    end
    axislegend(ax; position=:rt)
    return fig
end

# ── RotaryCoherenceEstimate / CrossSpectralEstimate ────────────────────
# Shared two-panel (coherence, phase) layout for both -- the rotary
# version just has two branches (CCW/CW) per panel instead of one.

"""
    plot_rotary_coherence(rce::Types.RotaryCoherenceEstimate; kwargs...)

Two-panel plot: magnitude-squared coherence (CCW/CW) with a
`significance_level` threshold line, and cross-spectral phase (CCW/CW).

# Keyword Arguments
- `title=""`, `figsize=(700, 600)`

# Returns
`Figure`
"""
function ValTools.plot_rotary_coherence(rce::ValTools.Types.RotaryCoherenceEstimate;
                                         title::String="",
                                         figsize::Tuple{Int,Int}=(700, 600))
    fig = Figure(size=figsize)
    !isempty(title) && Label(fig[0, 1], title; fontsize=14)

    ax1 = Axis(fig[1, 1]; xlabel="Frequency", ylabel="Coherence²", xscale=log10)
    lines!(ax1, rce.freq, rce.coh_ccw; color=:steelblue, label="CCW")
    lines!(ax1, rce.freq, rce.coh_cw; color=:firebrick, label="CW")
    isfinite(rce.significance_level) && hlines!(ax1, [rce.significance_level];
        color=:gray40, linestyle=:dash, label="95% sig.")
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[2, 1]; xlabel="Frequency", ylabel="Phase [rad]", xscale=log10)
    lines!(ax2, rce.freq, rce.phase_ccw; color=:steelblue)
    lines!(ax2, rce.freq, rce.phase_cw; color=:firebrick)
    linkxaxes!(ax1, ax2)
    return fig
end

"""
    plot_cross_spectrum(cse::Types.CrossSpectralEstimate; kwargs...)

Two-panel plot: magnitude-squared coherence with a `significance_level`
threshold line, and cross-spectral phase.

# Keyword Arguments
- `title=""`, `figsize=(700, 600)`

# Returns
`Figure`
"""
function ValTools.plot_cross_spectrum(cse::ValTools.Types.CrossSpectralEstimate;
                                       title::String="",
                                       figsize::Tuple{Int,Int}=(700, 600))
    fig = Figure(size=figsize)
    !isempty(title) && Label(fig[0, 1], title; fontsize=14)

    ax1 = Axis(fig[1, 1]; xlabel="Frequency", ylabel="Coherence²", xscale=log10)
    lines!(ax1, cse.freq, cse.coherence; color=:steelblue)
    isfinite(cse.significance_level) && hlines!(ax1, [cse.significance_level];
        color=:gray40, linestyle=:dash, label="95% sig.")
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[2, 1]; xlabel="Frequency", ylabel="Phase [rad]", xscale=log10)
    lines!(ax2, cse.freq, cse.phase; color=:steelblue)
    linkxaxes!(ax1, ax2)
    return fig
end

# ── EllipsePolarizationEstimate ────────────────────────────────────────

"""
    plot_ellipse_polarization(epe::Types.EllipsePolarizationEstimate; kwargs...)

Two-panel plot: major/minor-axis eigen-power spectra (`d1`/`d2`, with
jackknife CI ribbons when present), and ellipse orientation `theta` with
its circular-jackknife CI ribbon (`ci_theta`) when present. `theta`'s
wraparound at the branch cut is not corrected for display -- a ribbon that
appears to span the full `(-pi/2, pi/2]` range at a given frequency means
the orientation genuinely isn't resolved there (see
`EllipsePolarizationEstimate` docstring), not a plotting bug.

# Keyword Arguments
- `title=""`, `figsize=(700, 600)`

# Returns
`Figure`
"""
function ValTools.plot_ellipse_polarization(epe::ValTools.Types.EllipsePolarizationEstimate;
                                             title::String="",
                                             figsize::Tuple{Int,Int}=(700, 600))
    fig = Figure(size=figsize)
    !isempty(title) && Label(fig[0, 1], title; fontsize=14)

    ax1 = Axis(fig[1, 1]; xlabel="Frequency", ylabel="Eigen-power",
               xscale=log10, yscale=log10)
    epe.ci_d1 !== nothing && band!(ax1, epe.freq, max.(epe.ci_d1[1], eps()), epe.ci_d1[2];
                                    color=(:steelblue, 0.2))
    epe.ci_d2 !== nothing && band!(ax1, epe.freq, max.(epe.ci_d2[1], eps()), epe.ci_d2[2];
                                    color=(:firebrick, 0.2))
    lines!(ax1, epe.freq, epe.d1; color=:steelblue, label="d1 (major)")
    lines!(ax1, epe.freq, epe.d2; color=:firebrick, label="d2 (minor)")
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[2, 1]; xlabel="Frequency", ylabel="theta [rad]", xscale=log10)
    if epe.ci_theta !== nothing
        band!(ax2, epe.freq, epe.ci_theta[1], epe.ci_theta[2]; color=(:seagreen, 0.25))
    end
    lines!(ax2, epe.freq, epe.theta; color=:seagreen)
    linkxaxes!(ax1, ax2)
    return fig
end

# ── ColocatedObservation ────────────────────────────────────────────────

"""
    plot_colocation(co::Types.ColocatedObservation; kwargs...)

Overlay a colocated model/obs time series pair with a metrics annotation
(rmse/correlation/skill/bias, whichever keys are present in `co.metrics`).

# Keyword Arguments
- `title=""`, `figsize=(900, 400)`

# Returns
`Figure`
"""
function ValTools.plot_colocation(co::ValTools.Types.ColocatedObservation;
                                   title::String="",
                                   figsize::Tuple{Int,Int}=(900, 400))
    Qm = eltype(co.model.value)
    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; xlabel="Time", ylabel=_axis_label(co.model.name, Qm),
              title=title)

    lines!(ax, Float64.(1:length(co.obs.time)), Unitful.ustrip.(co.obs.value);
           color=:black, linewidth=1.5, label="obs")
    lines!(ax, Float64.(1:length(co.model.time)), Unitful.ustrip.(co.model.value);
           color=:steelblue, linewidth=1.5, linestyle=:dash, label="model")
    axislegend(ax; position=:rt)

    if !isempty(co.metrics)
        txt = join(["$k=$(round(v; digits=3))" for (k, v) in pairs(co.metrics)
                     if v isa Real], "\n")
        !isempty(txt) && text!(fig.scene, 0.02, 0.95; text=txt, fontsize=10,
                                space=:relative, align=(:left, :top))
    end
    return fig
end
