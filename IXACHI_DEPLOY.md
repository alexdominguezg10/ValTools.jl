# ValTools.jl on Ixachi — Deployment Guide

## 1. Copy project to Ixachi

```bash
# From your local machine
rsync -avz --exclude='.git' --exclude='Manifest.toml' \
    ValTools.jl/ adomingu@ixachi:/LUSTRE/adomingu/ValTools.jl/
```

## 2. Setup CPU environment (login node)

```bash
ssh ixachi

# Julia should be available; if not:
# module load julia   # or use your juliaup install

cd /LUSTRE/adomingu/ValTools.jl
julia envs/setup_envs.jl cpu
```

## 3. Setup GPU environment (GPU node)

```bash
# Get an interactive GPU session
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash

cd /LUSTRE/adomingu/ValTools.jl

# Set CUDA runtime to match H200 driver
julia -e 'using Pkg; Pkg.activate("envs/gpu"); Pkg.add("CUDA"); using CUDA; CUDA.set_runtime_version!(v"13.1"; local_toolkit=true)'

# Then setup
julia envs/setup_envs.jl gpu
```

**IMPORTANT:** Always build the GPU env on a GPU node (gotcha from CICOILPhysics.jl —
precompiling on login node means CUDA.functional() = false, and it will recompile
everything again on the GPU node).

## 4. Usage — Interactive GPU session

```bash
srun --partition=gpu --gres=gpu:1 --time=02:00:00 --pty bash

cd /LUSTRE/adomingu/ValTools.jl
julia --project=envs/gpu
```

```julia
using CUDA                      # triggers ValToolsCUDAExt
using ValTools
using ValTools.JLab

# Verify GPU
println(CUDA.device())          # should show H200

# ── Example: wavelet transform on GPU ──
using Random; Random.seed!(42)

# Large signal: 100k samples, typical of 1-year hourly mooring data
N = 100_000
x = randn(N)

# CPU baseline
@time wt_cpu, scales = wavetrans(x; dt=1.0, nv=8)

# GPU
@time wt_gpu, scales = wavetrans(x; dt=1.0, nv=8, gpu=true)

# Verify match
println("Max difference: ", maximum(abs.(wt_cpu .- wt_gpu)))

# ── Batch: 64 mooring depths simultaneously ──
X = randn(N, 64)
@time wt3d, scales = wavetrans_batch(X; dt=1.0, nv=8, gpu=true)
println("Batch result: ", size(wt3d))  # (100000, n_scales, 64)
```

## 5. Usage — SLURM batch script

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=valtools_gpu_%j.out
#SBATCH --error=valtools_gpu_%j.err

cd /LUSTRE/adomingu/ValTools.jl

julia --project=envs/gpu scripts/my_analysis.jl
```

## 6. Usage — CPU only (no GPU needed)

```bash
cd /LUSTRE/adomingu/ValTools.jl
julia --project=envs/cpu
```

```julia
using ValTools
using ValTools.JLab

# Everything works, just no gpu=true
wt, scales = wavetrans(x; dt=1.0, nv=8)
freqs, psd = mspec(x, dt; ntapers=5)
```

## 7. Updating after local changes

```bash
# From local machine — push changes
rsync -avz --exclude='.git' --exclude='Manifest.toml' \
    ValTools.jl/ adomingu@ixachi:/LUSTRE/adomingu/ValTools.jl/

# On Ixachi — recompile (the Pkg.develop link means changes are instant,
# but if you added new deps you need to re-instantiate)
julia --project=envs/gpu -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

## Gotchas (learned from CICOILPhysics.jl)

1. **CUDA runtime version:** H200 on Ixachi has CUDA 13.1 — run
   `CUDA.set_runtime_version!(v"13.1"; local_toolkit=true)` once
2. **Precompile on GPU node:** Don't precompile `envs/gpu` on login node
3. **Manifest.toml:** Don't rsync it — let each machine resolve its own
4. **FFTW threads:** Set `FFTW.set_num_threads(N)` for CPU path on multi-core nodes
5. **Memory:** H200 has 141 GB HBM3e — you can fit very large wavelet transforms
