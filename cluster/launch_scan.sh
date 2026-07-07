#!/bin/bash

# Parameter-scan array job for MotilePacemaker, adapted from an existing
# SoftDisks3 SLURM template. One array task = one line of parameters_array.txt
# = one call to julia/scripts/run_scan.jl.
#
# ONE-TIME SETUP (not part of this script -- do this once per cluster
# checkout, not per scan, and never inside the array job itself, since
# N array tasks racing on `Pkg.instantiate()` against the same shared
# depot would corrupt it):
#   cd <project>/julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
# PER-SCAN SETUP (run locally, then rsync/scp just the scan's config dir up,
# or git push/pull it along with the rest of the repo):
#   python python/scripts/generate_param_scan.py configs/scans/<scan_name>.toml
#   -> prints "Set #SBATCH --array=1-N"; put that N below.

##SBATCH --time=48:00:00
##SBATCH --partition=medium

#medium 2-00:00:00
#short 02:00:00
#long 14-00:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

#SBATCH --mail-type END

# EDIT PER SCAN: array range from generate_param_scan.py's printed count,
# and a descriptive job name (by convention: "<scan_name>", matching the
# sweep spec's file stem and the output naming below).
##SBATCH --array=1-9
##SBATCH --job-name="260708ScanChemistry"

#SBATCH --output=/home/lkoehler/Documents/Slurm/MotilePacemaker/R-%A_%a.out
#SBATCH --error=/home/lkoehler/Documents/Slurm/MotilePacemaker/R-%A_%a.err

set -e

#module load python
#module load julia

# EDIT PER CHECKOUT: path to the cloned repo on the cluster
project="/home/lkoehler/Documents/MotilePacemaker"
# EDIT PER SCAN: must match #SBATCH --job-name above
scan_name="$SLURM_JOB_NAME"

results="/data/biophys/lkoehler/MotilePacemaker/$scan_name"
scratch="/scratch/$USER/$SLURM_JOB_ID"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Make sure julia and its packages are found in the right place
export JULIA_DEPOT_PATH=/home/lkoehler/.julia/
export PATH="$PATH:/home/lkoehler/Documents/julia-1.11.3/bin"

manifest="$project/configs/scans/$scan_name/manifest.toml"
params_file="$project/configs/scans/$scan_name/parameters_array.txt"

# <scan_name>_<task_id>, e.g. example_epsilon_width_7 -- written straight to
# scratch, matching the existing workflow's scratch-then-copy pattern
run_name="${scan_name}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$scratch"
cd "$project"

# The trailing sed expansion is intentionally unquoted: it expands to
# however many values that line of parameters_array.txt has, so this script
# never needs to know in advance how many parameters are being scanned.
# Passing $SLURM_ARRAY_TASK_ID as task_index is what makes two identical
# parameter lines (e.g. to repeat a run with a different initial condition)
# get two different default random seeds automatically.
srun julia --project="$project/julia" "$project/julia/scripts/run_scan.jl" \
    "$manifest" \
    "$SLURM_ARRAY_TASK_ID" \
    "$scratch/${run_name}.h5" \
    $(sed -n -e "${SLURM_ARRAY_TASK_ID}p" "$params_file")

# copy just the result to an accessible location, named the same as in scratch
mkdir -p "$results"
cp "$scratch/${run_name}.h5" "$results/${run_name}.h5"

# clean up after yourself
rm -rf "$scratch"

exit 0
