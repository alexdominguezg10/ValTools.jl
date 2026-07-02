#!/bin/bash
#SBATCH --job-name=valtools-gpu-build
#SBATCH --partition=oc_gpu
#SBATCH --gres=gpu:H200:1
#SBATCH --time=00:30:00
#SBATCH --output=build_gpu.log
#SBATCH --error=build_gpu.err

set -e

VALTOOLS_DIR="/LUSTRE/adomingu/ValTools.jl"
cd "$VALTOOLS_DIR"

echo "=========================================="
echo "ValTools.jl GPU Precompilation"
echo "=========================================="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

julia --project << 'EOF'
using Pkg
println("Precompiling with CUDA...")
Pkg.instantiate()

using CUDA
println("CUDA version: ", CUDA.version())
println("GPU device: ", CUDA.device())

using ValTools.JLab
println("✓ GPU precompilation complete")

# Quick test
using Random
Random.seed!(42)
x = randn(256)
spec = spectral_multitaper(x, 1.0; nw=4.0)
println("✓ CPU path works on GPU node")

# Test GPU function exists
if CUDA.functional()
    println("✓ CUDA functional")
    # Load GPU extension
    using ValTools
    println("✓ ValTools fully loaded with GPU support")
end
EOF

echo ""
echo "=========================================="
echo "✓ GPU precompilation done!"
echo "Now run tests with: julia --project test_multitaper_gpu.jl"
echo "=========================================="
