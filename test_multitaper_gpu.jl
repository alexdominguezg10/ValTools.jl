"""
Test multitaper GPU implementation on H200.

Usage: julia --project test_multitaper_gpu.jl
"""

using CUDA
using Random
using Statistics
using ValTools.JLab

println("============================================================")
println("ValTools Multitaper GPU Test")
println("============================================================")
println("Julia version: ", VERSION)
println("CUDA functional: ", CUDA.functional())
if CUDA.functional()
    dev = CUDA.device()
    println("GPU device: ", dev)
end
println()

Random.seed!(42)

# ============================================================================
# Test 1: CPU single signal
# ============================================================================
println("TEST 1: CPU single signal")
println("------------------------------------------------------------")
x = randn(256)
t1 = @elapsed spec = spectral_multitaper(x, 1.0; nw=4.0)
println("✓ CPU single signal works")
println("  Time: $(round(t1*1000, digits=2)) ms")
println("  PSD length: ", length(spec.S))
println("  PSD range: [", round(minimum(spec.S), digits=3), ", ",
        round(maximum(spec.S), digits=3), "]")
println()

# ============================================================================
# Test 2: CPU batch (multiple signals)
# ============================================================================
println("TEST 2: CPU batch (5 signals)")
println("------------------------------------------------------------")
X_cpu = randn(1024, 5)
t2 = @elapsed specs_cpu = spectral_multitaper(X_cpu, 1.0; nw=4.0)
println("✓ CPU batch works")
println("  Time: $(round(t2*1000, digits=2)) ms")
println("  Batch result type: ", typeof(specs_cpu))
println("  Number of spectra: ", length(specs_cpu))
if !isempty(specs_cpu)
    println("  First spectrum PSD length: ", length(specs_cpu[1].S))
end
println()

# ============================================================================
# Test 3: GPU single signal (if CUDA available)
# ============================================================================
if CUDA.functional()
    println("TEST 3: GPU single signal")
    println("------------------------------------------------------------")
    try
        x_single = randn(256)
        t3 = @elapsed spec_gpu = spectral_multitaper_gpu(x_single, 1.0; nw=4.0, gpu=true)
        println("✓ GPU single signal works")
        println("  Time: $(round(t3*1000, digits=2)) ms")
        println("  Speedup vs CPU: $(round(t1/t3, digits=1))x")
        println()
    catch e
        println("✗ GPU single signal failed")
        println("  Error: ", e)
        println()
    end

    # ========================================================================
    # Test 4: GPU batch (the real speedup)
    # ========================================================================
    println("TEST 4: GPU batch (10 signals, 2048 samples each)")
    println("------------------------------------------------------------")
    try
        X_gpu = randn(2048, 10)
        t4 = @elapsed spec_batch_gpu = spectral_multitaper_gpu(X_gpu, 1.0; nw=4.0, gpu=true)
        println("✓ GPU batch works")
        println("  Time: $(round(t4*1000, digits=2)) ms")
        println("  Result shape: ", size(spec_batch_gpu[2]))
        println("  Number of spectra: ", length(spec_batch_gpu[2][1, :])")

        println()
        println("  Speedup analysis:")
        println("    CPU batch (5 sigs): $(round(t2*1000, digits=1)) ms")
        println("    GPU batch (10 sigs): $(round(t4*1000, digits=1)) ms")
        println("    Est. speedup: ~$(round(t2*2/t4, digits=1))x (accounting for 2x more signals)")
        println()
    catch e
        println("✗ GPU batch failed")
        println("  Error: ", e)
        println()
    end

    # ========================================================================
    # Test 5: Numerical equivalence (CPU vs GPU on small batch)
    # ========================================================================
    println("TEST 5: Numerical equivalence (CPU vs GPU)")
    println("------------------------------------------------------------")
    try
        X_test = randn(512, 2)

        # CPU
        specs_cpu_test = spectral_multitaper(X_test, 1.0; nw=4.0)

        # GPU
        freqs_gpu, psd_gpu = spectral_multitaper_gpu(X_test, 1.0; nw=4.0, gpu=true)

        # Compare first signal
        psd_cpu_1 = specs_cpu_test[1].S
        psd_gpu_1 = psd_gpu[:, 1]

        # Relative error
        rel_error = maximum(abs.(psd_cpu_1 .- psd_gpu_1) ./ (abs.(psd_cpu_1) .+ 1e-10))

        println("✓ Numerical comparison done")
        println("  CPU PSD[1] range: [", round(minimum(psd_cpu_1), digits=3), ", ",
                round(maximum(psd_cpu_1), digits=3), "]")
        println("  GPU PSD[1] range: [", round(minimum(psd_gpu_1), digits=3), ", ",
                round(maximum(psd_gpu_1), digits=3), "]")
        println("  Max relative error: ", round(rel_error, digits=2e-10))

        if rel_error < 1e-10
            println("  ✓ Numerically equivalent (machine epsilon)")
        elseif rel_error < 1e-6
            println("  ⚠ Close but not identical (expected for GPU)")
        else
            println("  ✗ Significant difference!")
        end
        println()
    catch e
        println("✗ Equivalence test failed")
        println("  Error: ", e)
        println()
    end

else
    println("CUDA not functional — GPU tests skipped")
    println()
end

# ============================================================================
# Summary
# ============================================================================
println("============================================================")
println("Test Complete")
println("============================================================")
