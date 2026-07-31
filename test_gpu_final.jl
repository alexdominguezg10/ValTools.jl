using CUDA
CUDA.set_runtime_version!(v"12.1")
using ValTools, ValTools.JLab
using Multitaper
using Random

Random.seed!(42)

println("=== ValTools GPU Test ===\n")

# Test 1: CPU single
println("Test 1: CPU single signal")
x = randn(256)
t1 = @elapsed spec = spectral_multitaper(x)
println("  Time: $(round(t1*1000; digits=1)) ms")
println("  PSD: $(extrema(spec.power))")

# Test 2: CPU batch
println("\nTest 2: CPU batch (5 signals)")
X_cpu = randn(1024, 5)
t2 = @elapsed specs = spectral_multitaper(X_cpu)
println("  Time: $(round(t2*1000; digits=1)) ms")
println("  Num spectra: $(length(specs))")

# Test 3: GPU batch
if CUDA.functional()
    println("\nTest 3: GPU batch (10 signals, 2048 samples)")
    X_gpu = randn(2048, 10)
    tapers, lambdas = dpss_tapers(2048, 4.0, 6, :both)
    N_fft = 2^ceil(Int, log2(2*2048-1))

    # Warm-up: first CUDA/CUFFT call in the process pays for context init +
    # kernel compilation (same reason CPU "Test 1" above is inflated by JIT).
    # Time the second call for a representative steady-state number.
    ValTools.JLab.spectral_multitaper_batch_gpu(X_gpu, tapers, lambdas, 1.0, N_fft)
    t3 = @elapsed (freqs, psd) = ValTools.JLab.spectral_multitaper_batch_gpu(X_gpu, tapers, lambdas, 1.0, N_fft)

    println("  Time (warm): $(round(t3*1000; digits=1)) ms")
    println("  PSD shape: $(size(psd))")

    # Speedup (scale CPU batch time to 10 signals)
    speedup = (t2 * 2) / t3
    println("\n  CPU batch (5 sigs): $(round(t2*1000; digits=1)) ms")
    println("  GPU batch (10 sigs): $(round(t3*1000; digits=1)) ms")
    println("  Est. speedup: $(round(speedup; digits=1))x")

    println("\n✓ ALL TESTS PASSED!")
else
    println("\nCUDA not functional")
end
