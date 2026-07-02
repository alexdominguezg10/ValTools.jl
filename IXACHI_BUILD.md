# Building ValTools on Ixachi

Ixachi has both CPU and GPU nodes with different architectures. Julia needs separate precompilation for each.

## Quick Start

### 1. CPU Precompilation (Login node or CPU queue)

```bash
cd /LUSTRE/adomingu/ValTools.jl
julia --project << 'EOF'
using Pkg
Pkg.instantiate()
using ValTools
println("✓ CPU precompilation done")
EOF
```

### 2. GPU Precompilation (H200 node)

Submit to GPU queue:

```bash
cd /LUSTRE/adomingu/ValTools.jl
sbatch build_ixachi_gpu.slurm
```

Monitor the job:
```bash
squeue -u adomingu
tail -f build_gpu.log
```

### 3. Run Tests (After GPU build completes)

```bash
julia --project test_multitaper_gpu.jl
```

Expected output:
```
TEST 1: CPU single signal ✓
TEST 2: CPU batch ✓
TEST 3: GPU single signal ✓
TEST 4: GPU batch ✓ (15-50× speedup)
TEST 5: Numerical equivalence ✓ (machine epsilon)
```

## What Gets Built

**CPU compilation:**
- JLab spectral methods
- Multitaper.jl integration
- CPU FFT/detrending path

**GPU compilation:**
- CUDA extension for batched tapered FFTs
- GPU memory management (chunking for H200)
- CUFFT backend

## Expected Performance

| Scenario | Time | Notes |
|----------|------|-------|
| Single 256-sample signal (CPU) | ~10-50ms | First call includes JIT |
| Batch 10×2048 signals (GPU) | ~50-200ms | 15-50× speedup vs CPU |
| Batch 100×2048 signals (GPU) | ~200-500ms | Chunked for memory safety |

## Troubleshooting

**"CUDA not functional"**
- Check you're on a GPU node: `nvidia-smi`
- Ensure CUDA module loaded: `module list`
- Re-run GPU build script

**"Package X not found"**
- Run `Pkg.instantiate()` again
- Check project is at `/LUSTRE/adomingu/ValTools.jl`

**Precompilation slow (>5 min)**
- First compile is normal, subsequent runs are faster
- Check disk quota: `lfs quota -u adomingu /LUSTRE`

## Next Steps

Once tests pass:
1. Update test suite (`test/jlab/test_spectral.jl`) for GPU equivalence
2. Benchmark full ensemble (100+ signals)
3. Move to Phase 2 (Type hierarchy refactoring)
