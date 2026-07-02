#!/bin/bash
# Build script for Ixachi (CPU + GPU architectures)
# Usage: bash build_ixachi.sh

set -e

VALTOOLS_DIR="/LUSTRE/adomingu/ValTools.jl"
cd "$VALTOOLS_DIR"

echo "=========================================="
echo "ValTools.jl Build for Ixachi"
echo "=========================================="

# Step 1: CPU compilation
echo ""
echo "STEP 1: CPU Precompilation"
echo "Run this on a CPU node (or login node)"
echo ""
julia --project << 'EOF'
using Pkg
println("Installing dependencies...")
Pkg.instantiate()
println("Precompiling ValTools on CPU...")
using ValTools
println("✓ CPU precompilation complete")
EOF

echo ""
echo "=========================================="
echo "Now submit to GPU node with:"
echo "  sbatch build_ixachi_gpu.sh"
echo "=========================================="
