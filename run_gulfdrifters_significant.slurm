#!/bin/bash
#SBATCH --job-name=gomed-full-census
#SBATCH --partition=oc_cpus
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=gomed_census.log
#SBATCH --error=gomed_census.err

set -e

VALTOOLS_DIR="/LUSTRE/adomingu/ValTools.jl"
SCRATCH_ENV="/LUSTRE/adomingu/valtools_cpu_scratch_env"
cd "$VALTOOLS_DIR"

# Ixachi home directory has limited space -- keep cached datasets (currently
# GulfDriftersOpen.nc, ~96MB) on Lustre instead of the default ~/.valtools/jdata.
# _jdata_cache_dir() (src/JLab/jdata.jl:84-89) already reads this env var, no
# code change needed.
export VALTOOLS_JDATA_DIR="/LUSTRE/adomingu/jdata"

echo "=========================================="
echo "GulfDrifters full-census significance test"
echo "=========================================="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# envs/cpu/Project.toml doesn't list Multitaper as a direct dependency
# (deliberate weakdep-only decision for the published package, see
# project_multitaper_licensing_decision memory) -- reuse the established
# scratch-env workaround instead of editing the tracked envs/cpu/Project.toml.
# Idempotent: safe to rerun, Pkg skips work that's already done.
julia -e "
import Pkg
Pkg.activate(\"$SCRATCH_ENV\")
Pkg.develop(path=\"$VALTOOLS_DIR\")
Pkg.add([\"Multitaper\"])
Pkg.instantiate()
"

echo ""
echo "Running full census with $SLURM_CPUS_PER_TASK threads..."
echo ""

julia --project="$SCRATCH_ENV" -t "$SLURM_CPUS_PER_TASK" scripts/validate_gulfdrifters_significant.jl

echo ""
echo "=========================================="
echo "Done. Output CSV: results/gulfdrifters_significant_full.csv"
echo "=========================================="
