#=
Setup script for ValTools.jl dual environments (CPU / GPU)

Run this ONCE on Ixachi to create both environments:

    # On login node (CPU env — no GPU needed):
    julia envs/setup_envs.jl cpu

    # On GPU node (GPU env — needs CUDA driver):
    srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash
    julia envs/setup_envs.jl gpu

IMPORTANT: Build the GPU env on a GPU node so CUDA.jl detects the
H200 driver and precompiles correctly (gotcha #8 from CICOILPhysics.jl).
=#

using Pkg

target = length(ARGS) >= 1 ? ARGS[1] : "both"
project_root = dirname(dirname(@__FILE__))

function setup_env(env_name)
    env_path = joinpath(project_root, "envs", env_name)
    println("=" ^ 60)
    println("  Setting up '$env_name' environment")
    println("  Path: $env_path")
    println("=" ^ 60)

    Pkg.activate(env_path)

    # Develop the parent ValTools package (local, editable)
    Pkg.develop(path=project_root)

    # Resolve and precompile
    Pkg.instantiate()
    Pkg.precompile()

    println("\n✓ Environment '$env_name' ready\n")
end

if target ∈ ("cpu", "both")
    setup_env("cpu")
end

if target ∈ ("gpu", "both")
    setup_env("gpu")
end

if target ∉ ("cpu", "gpu", "both")
    println("Usage: julia envs/setup_envs.jl [cpu|gpu|both]")
    exit(1)
end
