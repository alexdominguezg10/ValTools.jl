# # Unit-safe model-vs-observation validation
#
# A classic silent bug in ocean model validation: comparing a model in m/s
# against a mooring reported in cm/s and getting an RMSE that's off by 100x,
# with no error, no warning — just a wrong number that looks plausible.
# `TimeSeriesVector` bakes the unit into the data itself (via Unitful.jl), so
# `Dispatch.validate` simply cannot make that mistake: mismatched dimensions
# raise `Unitful.DimensionError` immediately, and compatible-but-different
# units (m/s vs cm/s) are converted automatically before comparison.
#
# (This example is exactly how we caught a real bug in `skill_score`/`bias`
# while writing it — an earlier version stripped each series' units
# *independently* instead of converting to a common scale first, which
# silently produced a skill score of -10069 for a genuinely good model.
# Correlation happened to survive unscathed since Pearson's r is invariant
# to per-variable rescaling, but `bias`/`skill_score` are not.)
#
# **Reference:** Unitful.jl philosophy: types carry units, not strings.

# ## Setup: observations and model with different units

using ValTools, ValTools.Dispatch, Unitful, Dates, Random
Random.seed!(42)

t = Dates.now() .+ Dates.Hour.(0:99)

# "Observed" current-meter record in m/s
obs_val = 0.4 .* sin.(2π .* (0:99) ./ 24) .* u"m/s" .+ randn(100) .* 0.05u"m/s"
obs = TimeSeriesVector(t, obs_val, "mooring", (;))

# A model that's slightly biased and slightly noisier — but reported in cm/s,
# as models sometimes are (a real-world scenario!)
model_val = uconvert.(u"cm/s", obs_val .+ 0.03u"m/s" .+ randn(100) .* 0.02u"m/s")
model = TimeSeriesVector(t, model_val, "model", (;))

println("obs unit:   ", Dispatch.unit_of(obs))
println("model unit: ", Dispatch.unit_of(model))

# ## Validate: automatic unit conversion, no silent errors

result = Dispatch.validate(model, obs)
println()
rmse_val = result.rmse
corr_val = result.correlation
skill_val = result.skill
bias_val = result.bias

println("RMSE:        ", rmse_val)               # unit-converted automatically, carries units
println("Correlation: ", round(corr_val, digits=3))
println("Skill score: ", round(skill_val, digits=3))
println("Bias:        ", round(bias_val, digits=3), " ", Dispatch.unit_of(model), "  (bias is in model's unit — here cm/s)")

# ## Catch dimensional errors immediately

# Try comparing against something dimensionally incompatible — this is a
# programming error we *want* to catch immediately, not silently misreport
bad = TimeSeriesVector(t, randn(100) .* u"kg", "not_velocity", (;))
error_caught = false
try
    model + bad
catch e
    println("\nCaught expected error: ", sprint(showerror, e))
    error_caught = true
end

# ## Verification

@assert ustrip(rmse_val) > 0 "RMSE should be positive"
@assert corr_val > 0.9 "Correlation should be high (nearly-true model)"
@assert abs(ustrip(bias_val)) < 5.0 "Bias should be small (biased model) in cm/s"
@assert error_caught "Dimensional error should be caught immediately"
println("✓ Verification passed: units tracked correctly, dimensional errors caught")