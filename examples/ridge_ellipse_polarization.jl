# # Synthesizing and characterizing a modulated ellipse (ellsig/ellpol)
#
# `Metrics.ellipse_polarization` describes polarization in the *frequency*
# domain — one averaged spectral matrix per frequency band. `JLab.ellpol`
# takes the complementary *time*-domain route: given a wavelet ridge's
# time-varying ellipse parameters — RMS amplitude `kappa`, linearity
# `lambda` (0 = circular, ±1 = rectilinear), orientation `theta`, and
# orbital phase `phi` — `ellsig` reconstructs the actual `(x, y)` signal
# those parameters describe, and `ellpol` summarizes the whole record with
# one polarization state `(P, alpha, beta)`.
#
# We build a slowly-precessing, slowly-more-eccentric ellipse by hand — the
# kind of signature a wavelet ridge would hand back for a real eddy whose
# orientation drifts and whose shape isn't perfectly circular — synthesize
# it with `ellsig`, and check the identity `P² = alpha² + beta²` that has to
# hold for any consistent polarization decomposition.
#
# One thing worth noticing before running this: `ellpol` reports ONE
# time-averaged polarization state for the whole record. A single
# instantaneous ellipse is always fully polarized (P=1) in this sense — but
# here the orientation sweeps a full half-turn over the record. Averaging
# the second-moment matrix across many different orientations partially
# cancels itself out, the same way averaging many unit vectors pointing in
# different directions gives something shorter than a unit vector. So `P`
# well below 1 here isn't a discrepancy — it's `ellpol` correctly reporting
# that "one fixed polarization state" is a poor summary of a *strongly
# modulated* signal, exactly the situation Lilly & Olhede's own modulation
# framework (used elsewhere in this package for `instmom`/
# `multivariate_ridges`) is built to quantify.
#
# **Reference:** Lilly & Olhede (2009); jLab: `jEllipse/ellpol.m`, `jEllipse/ellsig.m`

# ## Setup: synthesize a precessing, breathing ellipse

using ValTools.JLab, CairoMakie

N = 400
t = collect(0.0:N-1)

kappa  = fill(1.0, N)                       # constant RMS amplitude
lambda = 0.15 .+ 0.35 .* sin.(2π .* t ./ N) # linearity drifts: more circular <-> more elliptical
theta  = 2π .* t ./ (2 * N)                 # orientation precesses half a turn over the record
phi    = 2π .* 0.05 .* t                    # orbital phase -- sets the rotation rate

# ## Synthesize and characterize

x, y = ellsig(kappa, lambda, theta, phi)
r = ellpol(kappa, lambda, theta, phi)

P_val = r.P
alpha_val = r.alpha
beta_val = r.beta
identity_lhs = r.P^2
identity_rhs = r.alpha^2 + r.beta^2
kbar_val = r.kbar

println("Time-averaged polarization state:")
println("  P (total polarization):        ", round(P_val, digits=3),
        "  (well below 1: orientation sweeps a half-turn, see header note)")
println("  alpha (rotary excess, CCW-CW):  ", round(alpha_val, digits=3))
println("  beta (linear-motion component): ", round(beta_val, digits=3))
println("  Identity check  P^2 vs alpha^2+beta^2:  ",
        round(identity_lhs, digits=6), "  vs  ", round(identity_rhs, digits=6))
println("  kbar (mean RMS axis length):     ", round(kbar_val, digits=3))

# ## Visualization: ellipse trajectory and parameter time series

fig = Figure(size=(900, 400))

ax1 = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Precessing, breathing ellipse",
           aspect=DataAspect())
lines!(ax1, real.(x), real.(y), color=t, colormap=:viridis, linewidth=1.5)
scatter!(ax1, [real(x[1])], [real(y[1])], color=:red, markersize=12, label="start")
axislegend(ax1, position=:rb)

ax2 = Axis(fig[1, 2], xlabel="Time", ylabel="Value",
           title="True ellipse parameters used to synthesize it")
lines!(ax2, t, lambda, color=:dodgerblue, linewidth=2, label="λ (linearity)")
lines!(ax2, t, theta ./ π, color=:seagreen, linewidth=2, label="θ/π (orientation)")
axislegend(ax2, position=:lt)

save(joinpath(@__DIR__, "ridge_ellipse_polarization.png"), fig)
println("Saved ridge_ellipse_polarization.png")

# ## Verification

# The fundamental identity P² = alpha² + beta² must hold exactly for any consistent decomposition
identity_error = abs(identity_lhs - identity_rhs)
@assert identity_error < 1e-10 "Identity P² = alpha² + beta² violated"
@assert P_val < 0.8 "P should be <0.8 for strongly modulated signal (orientation sweep)"
@assert kbar_val ≈ 1.0 "kbar should be ~1.0 (constant amplitude)"
println("✓ Verification passed: identity P² = alpha² + beta² holds to numerical precision")
