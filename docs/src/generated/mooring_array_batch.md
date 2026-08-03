```@meta
EditURL = "../../../examples/mooring_array_batch.jl"
```

# Batching a mooring array through one wavelet transform

A mooring line typically carries a handful of current meters at fixed
depths, all sampled on the same clock. Looping `wavetrans` once per depth
works, but it also refits the same filter bank and re-runs the same-size
FFT N times for no reason — the whole point of a shared clock is that one
batched transform can do all depths at once. `wavetrans` accepts any array
with time along dimension 1 and independent signals along the trailing
dimensions (jLab's own column-signal convention), so an `(N, depths)`
matrix goes through in a single call.

We simulate an 8-depth array, each depth carrying the same semidiurnal
tidal frequency with an amplitude that decays away from two depths where
it's locally reinforced (a stratification/mode-shape effect), batch them
through one `wavetrans` call, and read off the amplitude-vs-depth profile
directly from the 3-D output.

**Reference:** jLab N-D wavelet support: `jRidges/wavetrans.m` (trailing-dims batching)

## Setup: simulate 8-depth mooring array

````julia
using ValTools.JLab, Statistics, CairoMakie, Random
Random.seed!(11)

dt = 1.0                              # hours
N  = 24 * 8                           # 8 days, hourly
t  = (0:N-1) .* dt
f_tide = 1 / 12.42                    # M2 semidiurnal period, hours

depths = collect(10.0:40.0:290.0)     # 8 depths, meters
````

Amplitude profile with a mode-shape-like double hump (not a monotonic
decay) -- a case where a hand-picked "typical" depth wouldn't represent
the others well, motivating looking at the whole profile at once.

````julia
true_amp = 0.10 .+ 0.15 .* abs.(sin.(2π .* depths ./ 260.0))

X = hcat([true_amp[k] .* cos.(2π * f_tide .* t) .+ 0.015 .* randn(N)
          for k in eachindex(depths)]...)
````

## Batched wavelet transform: one call for all 8 depths

````julia
wt, fs = wavetrans(X; dt=dt, nv=8)    # (N, n_freq, 8) in ONE batched FFT call
amp = tiredecode(wt, fs; kind="amp")  # (N, n_freq, 8), still fully batched
````

For each depth, the tidal amplitude is the mean ridge amplitude at the
scale closest to the true M2 frequency.

````julia
target_scale = argmin(abs.(fs .- 2π * f_tide))
amp_by_depth = vec(mean(amp[:, target_scale, :]; dims=1))

println("Batched transform shape: ", size(wt), "  (time x freq x depth)")
println("Depth (m)   true amp   recovered amp")
for k in eachindex(depths)
    println("  ", round(depths[k], digits=0), "        ", round(true_amp[k], digits=3),
            "      ", round(amp_by_depth[k], digits=3))
end
````

## Visualization: time series and recovered amplitude profile

````julia
fig = Figure(size=(900, 400))

ax1 = Axis(fig[1, 1], xlabel="Time (days)", ylabel="Depth (m)",
           title="Mooring array (8 depths)", yreversed=true)
for k in eachindex(depths)
    lines!(ax1, t ./ 24, depths[k] .+ 20 .* X[:, k], color=:dodgerblue, linewidth=0.8)
end

ax2 = Axis(fig[1, 2], xlabel="M2 tidal amplitude (m/s)", ylabel="Depth (m)",
           title="Amplitude-vs-depth, one batched wavetrans call", yreversed=true)
depth_fine = range(0, 300; length=200)
lines!(ax2, 0.10 .+ 0.15 .* abs.(sin.(2π .* depth_fine ./ 260.0)), depth_fine,
       color=:gray, linestyle=:dash, label="true profile")
scatter!(ax2, amp_by_depth, depths, color=:orangered, markersize=14, label="recovered")
axislegend(ax2, position=:rb)

save(joinpath(@__DIR__, "mooring_array_batch.png"), fig)
println("Saved mooring_array_batch.png")
````

## Verification

Check that batched recovery matches the true amplitude profile to within ~15%

````julia
rms_error = sqrt(mean((amp_by_depth .- true_amp) .^ 2))
@assert rms_error < 0.04 "RMS error in amplitude profile should be <0.04 m/s"
println("✓ Verification passed: amplitude profile recovered within RMS error = $rms_error m/s")
````

