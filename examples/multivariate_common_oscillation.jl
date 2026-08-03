# # A common oscillation across a mooring: multivariate ridge analysis
#
# Near-inertial waves are generated at the surface by wind events and
# propagate downward, losing energy as they go — three current meters at
# different depths on the same mooring line all feel the *same* oscillation,
# just with different (decaying) amplitude and a small phase lag. Treating
# each depth separately throws away the fact that they share one underlying
# signal. `multivariate_ridges` instead finds ONE joint ridge across all N
# channels at once — jLab's genuinely N-general generalization of wavelet
# ridge analysis (Lilly & Olhede 2012), not a per-channel loop.
#
# We simulate three depths (50, 150, 300 m) sharing one near-inertial
# oscillation with amplitude decaying with depth, ask `multivariate_ridges`
# to recover it, and check that the ridge's reconstructed transform
# (`wt_ridge`) picks the true depth-decay profile back out.

using ValTools.JLab, Statistics, CairoMakie, Random
Random.seed!(3)

dt = 1.0                          # hours
N  = 24 * 10                      # 10 days, hourly
t  = (0:N-1) .* dt
f0 = 1 / 18.5                     # cycles/hour, near-inertial period ~18.5 h

depths = [50.0, 150.0, 300.0]     # meters
true_amp = 0.30 .* exp.(-depths ./ 200.0)   # e-folding decay scale 200 m

phase_lag = [0.0, 0.25, 0.55]     # radians, small lag increasing with depth
X = hcat([true_amp[k] .* cos.(2π * f0 .* t .+ phase_lag[k]) .+ 0.02 .* randn(N)
          for k in eachindex(depths)]...)

ridges = multivariate_ridges(X; dt=dt, nv=8)
r = only(ridges)   # one clean ridge spans the whole record

amp_recovered = vec(mean(abs.(r.wt_ridge); dims=1))
amp_recovered ./= amp_recovered[1] / true_amp[1]   # scale to the surface (50 m) amplitude

println("Joint ridge period:    ", round(2π / r.omega_bar, digits=2), " h  (true: 18.5 h)")
println("Depth   true amp   recovered amp")
for k in eachindex(depths)
    println("  ", Int(depths[k]), " m   ", round(true_amp[k], digits=3),
            "      ", round(amp_recovered[k], digits=3))
end

# One ridge, three channels: the left panel shows all three raw records
# together (same oscillation, shrinking with depth); the right panel
# compares the ridge's recovered amplitude-vs-depth profile against the
# true decay curve it was simulated with.

fig = Figure(size=(900, 400))

ax1 = Axis(fig[1, 1], xlabel="Time (days)", ylabel="Velocity (m/s)",
           title="Three depths, one oscillation")
colors = [:dodgerblue, :seagreen, :orangered]
for k in eachindex(depths)
    lines!(ax1, t ./ 24, X[:, k], color=colors[k], linewidth=1.5,
           label="$(Int(depths[k])) m")
end
axislegend(ax1)

ax2 = Axis(fig[1, 2], xlabel="Amplitude (m/s)", ylabel="Depth (m)",
           title="Joint-ridge amplitude vs. depth", yreversed=true)
depth_fine = range(0, 350; length=100)
lines!(ax2, 0.30 .* exp.(-depth_fine ./ 200.0), depth_fine, color=:gray,
       linestyle=:dash, label="true e-folding decay")
scatter!(ax2, amp_recovered, depths, color=colors, markersize=16, label="ridge-recovered")
axislegend(ax2)

save(joinpath(@__DIR__, "multivariate_common_oscillation.png"), fig)
println("Saved multivariate_common_oscillation.png")
