using ValTools, ValTools.JLab
using Random

Random.seed!(42)

println("CPU Performance Test\n")

# Test 1: Single signal
x = randn(256)
t1 = @elapsed spec = spectral_multitaper(x)
println("Single (256 samples): $(round(t1*1000; digits=1)) ms")

# Test 2: Small batch
X = randn(1024, 5)
t2 = @elapsed specs = spectral_multitaper(X)
println("Batch (1024x5): $(round(t2*1000; digits=1)) ms")

# Test 3: Large batch
X_large = randn(2048, 10)
t3 = @elapsed specs = spectral_multitaper(X_large)
println("Batch (2048x10): $(round(t3*1000; digits=1)) ms")

# Throughput
throughput = (10 * 2048) / t3 / 1e6
println("\nCPU Throughput: $(round(throughput; digits=2)) Msamples/sec")
println("✓ CPU Path Works!")
