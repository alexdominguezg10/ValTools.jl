```@meta
EditURL = "../../../examples/svd_polarization_detection.jl"
```

# Detecting a coherent signal with SVD-based polarization (msvd)

`Metrics.ellipse_polarization` characterizes polarization by pooling
per-taper spectra into one averaged spectral matrix and eigendecomposing
it. `msvd` (Park, Vernon & Lindberg 1987) instead singular-value-decomposes
each frequency band's raw `channels × looks` eigentransform matrix
directly, *before* any pooling — and that gives something the pooled route
can't: the fraction of a band's total power explained by a single rank-1
(fully coherent, fully polarized) structure, `d₁²/trace(S)`. For a
genuinely coherent bivariate oscillation this ratio sits near 1; for two
independent noise channels its *asymptotic* (many-look) value is 0.5 — a
clean, single-run signal-detection statistic. With only `K` looks per
band, picking the larger of two random sample eigenvalues is itself
biased upward from that limit (a standard finite-sample effect, not a
flaw in the statistic) — `K=40` below keeps that bias small enough that
the true signal still stands out sharply above the noise floor.

`msvd` itself doesn't care where the `K` "looks" of each band come from —
real Slepian tapers, or (as here, to keep this example dependency-free) `K`
non-overlapping Hann-windowed segments of the record, each giving one
independent estimate of every frequency band. We bury a circularly
polarized oscillation in independent per-channel noise and ask `msvd` to
find the frequency band where the signal actually lives, purely from how
"rank-1" that band's structure is.

**Reference:** Park et al. (1987); jLab: `jSpectral/msvd.m`

## Setup: circularly polarized signal + independent noise

````julia
using ValTools.Metrics, FFTW, Statistics, CairoMakie, Random
Random.seed!(5)

K  = 40                        # more independent looks -> less finite-K bias in the noise baseline
ns = 300                       # samples per segment (sets frequency resolution)
n  = K * ns
dt = 1.0
f0 = 0.08                      # cycles/sample -- the true signal frequency
u = 0.4 .* cos.(2π * f0 .* (0:n-1)) .+ 0.5 .* randn(n)
v = 0.4 .* sin.(2π * f0 .* (0:n-1)) .+ 0.5 .* randn(n)   # circularly polarized (CCW) + noise
````

## Segmented spectral estimation with Hann windowing

````julia
hann = 0.5 .- 0.5 .* cos.(2π .* (0:ns-1) ./ (ns - 1))
````

True positive frequencies only (indices 2:ns÷2 of a real-input FFT) —
NOT `(0:ns-1)./(ns*dt) .> 0`, which for a real signal's Hermitian-symmetric
spectrum would keep the upper half's *aliased mirror* of the negative
frequencies (bins above Nyquist wrap around to negative freq, not to freq
values above 1/dt) and corrupt every band above 0.5 cycles/sample.

````julia
pos = 2:(ns ÷ 2)
freqs_pos = (pos .- 1) ./ (ns * dt)
nfreq = length(freqs_pos)
````

Build the (J, N=2, K) eigentransform array msvd expects: one complex FFT
coefficient per (band, channel, segment).

````julia
W = Array{ComplexF64}(undef, nfreq, 2, K)
for k in 1:K
    seg = (k-1)*ns+1:k*ns
    W[:, 1, k] = fft(u[seg] .* hann)[pos]
    W[:, 2, k] = fft(v[seg] .* hann)[pos]
end
````

## MSVD: rank-1 power fraction (coherence detection statistic)

````julia
r = msvd(W)
explained = r.d[:, 1] .^ 2 ./ r.trS   # fraction of power in the dominant (rank-1) mode

peak = argmax(explained)
noise_bands = abs.(freqs_pos .- f0) .> 0.05
peak_explained = explained[peak]
noise_mean_explained = mean(explained[noise_bands])

println("Peak explained-power frequency: ", round(freqs_pos[peak], digits=4),
        "  (true: ", f0, ")")
println("Explained fraction at the signal band: ", round(peak_explained, digits=3),
        "  (1.0 = fully coherent/rank-1)")
println("Mean explained fraction elsewhere:     ", round(noise_mean_explained, digits=3),
        "  (asymptotic limit for independent noise is 0.5; finite K=$K biases this up)")
````

## Visualization: rank-1 power fraction vs. frequency

````julia
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="Frequency (cycles/sample)",
          ylabel="Explained fraction, d₁²/trace(S)",
          title="MSVD rank-1 power fraction, K=$K segments")
lines!(ax, freqs_pos, explained, color=:dodgerblue, linewidth=1.5)
hlines!(ax, [0.5], color=:gray, linestyle=:dash, label="asymptotic noise floor (K→∞)")
vlines!(ax, [f0], color=:orangered, linestyle=:dash, label="true signal freq")
xlims!(ax, 0, 0.25)
axislegend(ax)

save(joinpath(@__DIR__, "svd_polarization_detection.png"), fig)
println("Saved svd_polarization_detection.png")
````

## Verification

````julia
@assert abs(freqs_pos[peak] - f0) < 0.005 "Peak should be within 0.005 of true frequency"
@assert peak_explained > 0.8 "Signal coherence should be high (>0.8)"
@assert noise_mean_explained < 0.7 "Noise floor should stay below 0.7 (finite-K bias)"
println("✓ Verification passed: signal detected at correct frequency with high coherence")
````

