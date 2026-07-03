# # Detecting an inertial oscillation
#
# Inertial oscillations are the ocean's signature of the Coriolis force: a
# parcel of water given a push rotates freely at the local inertial frequency
# `f = 2*Ω*sin(latitude)`, tracing a circle that is **counter-clockwise in the
# Northern Hemisphere** and clockwise in the Southern. Gonella (1972) showed
# that this handedness falls straight out of a rotary spectral decomposition:
# split a velocity time series `w = u + iv` into its positive- and
# negative-frequency parts, and the CCW/CW power ratio tells you which way
# the water is spinning.
#
# Here we simulate forty days of hourly current-meter data at 25°N (inertial
# period ≈ 26.7 h), buried in noise, and ask `rotary_spectrum` to find it.

using ValTools, Multitaper, CairoMakie, Random
Random.seed!(1)

dt = 1.0                    # hours between samples
t  = 0:dt:(24*40)           # 40 days, hourly — enough frequency resolution to nail 26.7 h
f_inertial = 1 / 26.7       # cycles/hour at 25°N

u = 0.15 .* cos.(2π * f_inertial .* t) .+ 0.03 .* randn(length(t))
v = 0.15 .* sin.(2π * f_inertial .* t) .+ 0.03 .* randn(length(t))  # CCW = inertial (NH)

spec = rotary_spectrum(u, v; dt_hours=dt)

peak = argmax(spec.S_ccw)
println("Peak CCW period:     ", round(1 / spec.freq[peak], digits=1), " h  (true: 26.7 h)")
println("Rotary coefficient:  ", round(spec.rotary_coefficient[peak], digits=2), "  (+1 = purely CCW/inertial, -1 = purely CW)")

# Two views of the same signal: the hodograph shows the velocity vector
# literally spiraling counter-clockwise over time, and the rotary spectrum
# shows *why* — almost all the energy sits in the CCW branch at 26.7 h.

fig = Figure(size=(900, 400))

ax1 = Axis(fig[1, 1], xlabel="u (m/s)", ylabel="v (m/s)", title="Hodograph — first 48 h",
           aspect=DataAspect())
lines!(ax1, u[1:48], v[1:48], color=1:48, colormap=:viridis)
scatter!(ax1, [u[1]], [v[1]], color=:red, markersize=14, label="start")
axislegend(ax1)

ax2 = Axis(fig[1, 2], xlabel="Period (hours)", ylabel="Power", title="Rotary spectrum",
           xscale=log10)
periods = 1 ./ spec.freq
lines!(ax2, periods, spec.S_ccw, label="CCW (inertial)", color=:dodgerblue, linewidth=2)
lines!(ax2, periods, spec.S_cw, label="CW", color=:orangered, linewidth=2)
vlines!(ax2, [26.7], color=:gray, linestyle=:dash, label="26.7 h")
axislegend(ax2)

save(joinpath(@__DIR__, "inertial_oscillation.png"), fig)
println("Saved inertial_oscillation.png")
