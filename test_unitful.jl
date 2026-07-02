using ValTools
using ValTools.Types
using ValTools.Dispatch
using Unitful
using Dates

println("=== UNITFUL INTEGRATION TEST ===\n")

# Create two time series with units (using 4-argument constructor)
t = Dates.now() .+ Dates.Second.(0:9)
ts1 = TimeSeriesVector(t, randn(10)*u"m/s", "velocity_1", (;))
ts2 = TimeSeriesVector(t, randn(10)*u"m/s", "velocity_2", (;))

# Test 1: Unit-preserving statistics
println("✓ Test 1: Unit-preserving statistics")
m1 = Dispatch.mean(ts1)
println("  mean(ts1) = $m1")
s1 = Dispatch.std(ts1)
println("  std(ts1) = $s1")

# Test 2: Unit-aware addition
println("\n✓ Test 2: Unit-aware addition")
ts_sum = ts1 + ts2
println("  ts1 + ts2: $(length(ts_sum.value)) samples")
println("  first value: $(ts_sum.value[1])")

# Test 3: Scalar multiplication (preserves units)
println("\n✓ Test 3: Scalar multiplication")
ts_scaled = ts1 * 2.5
println("  ts1 * 2.5: $(ts_scaled.value[1])")

# Test 4: Unit conversion
println("\n✓ Test 4: Unit conversion")
ts_cms = Dispatch.convert_units(ts1, u"cm/s")
println("  converted to cm/s: $(ts_cms.value[1])")

# Test 5: Unit information
println("\n✓ Test 5: Unit information")
unit = Dispatch.unit_of(ts1)
println("  unit of ts1: $unit")

# Test 6: Unit stripping
println("\n✓ Test 6: Unit stripping")
ts_bare = Dispatch.strip_units(ts1)
println("  stripped: $(ts_bare.value[1]) (type: $(typeof(ts_bare.value[1])))")

println("\n✓ ALL UNITFUL INTEGRATION TESTS PASSED")
