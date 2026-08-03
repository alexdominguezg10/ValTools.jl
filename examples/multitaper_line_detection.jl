# # Multitaper spectral estimation — telling a real line from noise
#
# A single periodogram is a noisy, inconsistent estimator of a spectrum — its
# variance doesn't shrink as you add more data, it just gets more finely
# sampled in frequency. Thomson's (1982) multitaper method fixes this by
# averaging several nearly-independent spectral estimates (one per Slepian
# taper) of the *same* record, trading a little frequency resolution for a
# lot less variance — and it comes with an F-test for line components for
# free, so you can say *which* peak is a real periodic signal and which is
# just broadband noise piled up in one bin.
#
# We bury a clean 0.02 cycles/sample signal in noise and ask
# `spectral_multitaper` to find it. The fun part: the raw power spectrum's
# highest bin isn't quite at 0.02 — sampling noise nudges it one bin over.
# The F-test isn't fooled.
#
# **Reference:** Thomson (1982); jLab: `jSpectral/spectral_multitaper.m`

# ## Setup: signal + noise with known frequency

using ValTools.JLab, Multitaper, Unitful, CairoMakie, Random
Random.seed!(7)

N = 2000
x = 0.5 .* sin.(2π * 0.02 .* (1:N)) .+ randn(N)  # m/s: signal + noise

# ## Multitaper estimation with F-test for line components

spec = spectral_multitaper(x, 1.0; nw=4.0, unit=u"m/s")

power_peak = argmax(spec.power)
line_peak  = argmin(spec.ftest_pval)   # smallest p-value = most significant line

power_freq = spec.freq[power_peak]
line_freq = spec.freq[line_peak]
line_pval = spec.ftest_pval[line_peak]

println("Raw power peak:  freq=", round(power_freq, digits=4),
        "   F-test p-value there: ", round(spec.ftest_pval[power_peak], digits=4))
println("F-test line peak: freq=", round(line_freq, digits=4),
        "   p-value: ", round(line_pval, digits=6),
        "  (true frequency: 0.02)")
println()
println(power_peak == line_peak ?
    "This run: power peak and F-test peak agree." :
    "This run: the power spectrum's highest bin is off by one from the true line — " *
    "the F-test correctly lands on 0.02 anyway.")

# ## Visualization: multitaper power spectrum with F-test line

fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="Frequency (cycles/sample)", ylabel="Power (m²/s²)",
          title="Multitaper spectrum, nw=4, K=$(spec.params.ntapers) tapers", yscale=log10)
lines!(ax, spec.freq, Unitful.ustrip.(spec.power), color=:dodgerblue, linewidth=1.5)
vlines!(ax, [line_freq], color=:seagreen, linestyle=:dash, linewidth=2,
        label="F-test line (p=$(round(line_pval, digits=4)))")
xlims!(ax, 0, 0.1)
axislegend(ax)

save(joinpath(@__DIR__, "multitaper_spectrum.png"), fig)
println("Saved multitaper_spectrum.png")

# ## Verification

@assert abs(line_freq - 0.02) < 0.001 "Detected line should be within 0.001 of true 0.02"
@assert line_pval < 0.001 "F-test p-value should be highly significant (<0.001)"
println("✓ Verification passed: line detected at correct frequency with p<0.001")