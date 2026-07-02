using CUDA
# Fix CUDA runtime for HPC environment
try
    CUDA.set_runtime_version!(v"12.1")
catch
end

using ValTools
using ValTools.JLab
using Random

Random.seed!(42)

println("Starting tests...")
if CUDA.functional()
    println("GPU device: ", CUDA.device())
else
    println("CUDA not functional")
end

# Test 1: Single signal
println("\nTest 1: CPU single signal")
x = randn(256)
t1 = @elapsed spec = spectral_multitaper(x, 1.0; nw=4.0)
println("Time: $(round(t1*1000; digits=2)) ms")
println("PSD OK: ", all(spec.S .>= 0))

# Test 2: Batch
println("\nTest 2: CPU batch (5 signals)")
X = randn(1024, 5)
t2 = @elapsed specs = spectral_multitaper(X, 1.0; nw=4.0)
println("Time: $(round(t2*1000; digits=2)) ms")
println("Num spectra: ", length(specs))

# Test 3: GPU batch
if CUDA.functional()
    println("\nTest 3: GPU batch (10 signals)")
    X_gpu = randn(2048, 10)
    t3 = @elapsed specs_gpu = spectral_multitaper_gpu(X_gpu, 1.0; nw=4.0, gpu=true)
    println("Time: $(round(t3*1000; digits=2)) ms")
    println("Speedup: $(round(t2*2/t3; digits=1))x")
else
    println("CUDA not functional")
end

println("\nDone!")
