# # Tracking an eddy's spin-down with wavelet ridges
#
# Real eddies don't sit at one fixed frequency forever — they spin down as
# they lose energy to friction and radiate away. A single rotary spectrum
# smears that decay across many frequencies. `rotary_wavetrans`/`rotary_ridge`
# instead track the *instantaneous* CCW/CW amplitude and frequency through
# time, the same way you'd watch a decaying pendulum on a scope rather than
# just reading off its average frequency.
#
# We simulate an anticyclonic eddy's rotary velocity, amplitude decaying
# exponentially over ~8 days, and ask `rotary_ridge` to recover the decay.

using ValTools.JLab, Statistics, CairoMakie

dt = 1.0                       # hours
N  = 24 * 12                   # 12 days, hourly
t  = (0:N-1) .* dt
f0 = 1 / 20.0                  # cycles/hour, ~20 h rotation period
decay_hours = 96.0             # e-folding time, ~4 days

envelope = 0.3 .* exp.(-t ./ decay_hours)
u = envelope .* cos.(2π * f0 .* t)
v = envelope .* sin.(2π * f0 .* t)   # CCW

result = rotary_ridge(u, v; dt=dt, nv=8)

println("CCW ridge amplitude, day 1 vs day 10: ",
        round(mean(filter(!isnan, result.amp_ccw[1:24])), digits=3), " -> ",
        round(mean(filter(!isnan, result.amp_ccw[end-23:end])), digits=3))
println("Mean rotary coefficient: ",
        round(mean(filter(!isnan, result.rotary_coefficient)), digits=2), "  (+1 = purely CCW)")

fig = Figure(size=(900, 400))

ax1 = Axis(fig[1, 1], xlabel="Time (days)", ylabel="Ridge amplitude (m/s)",
           title="Eddy spin-down (CCW ridge)")
lines!(ax1, t ./ 24, result.amp_ccw, color=:dodgerblue, linewidth=2, label="tracked amplitude")
lines!(ax1, t ./ 24, envelope, color=:gray, linestyle=:dash, label="true envelope")
axislegend(ax1)

ax2 = Axis(fig[1, 2], xlabel="Time (days)", ylabel="Ridge period (hours)",
           title="Tracked rotation period")
lines!(ax2, t ./ 24, 2π ./ result.freq_ccw, color=:seagreen, linewidth=2)
hlines!(ax2, [20.0], color=:gray, linestyle=:dash)

save(joinpath(@__DIR__, "eddy_spindown.png"), fig)
println("Saved eddy_spindown.png")