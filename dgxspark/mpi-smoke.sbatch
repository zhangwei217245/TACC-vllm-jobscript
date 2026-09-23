#!/usr/bin/env bash
#SBATCH --job-name=mpi-smoke
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --output=mpi-smoke-%j.out

set -euo pipefail
# Load your site's Open MPI module before sbatch, or add module load here.
# Submit from a shared directory containing mpi-smoke.c and this batch script.
# The same MPI installation and the shared executable must exist on all nodes.
cd -- "$SLURM_SUBMIT_DIR"
command -v mpicc
command -v mpirun
mpirun --version

# Per-job paths prevent concurrent tests from overwriting each other's files.
run_dir="$SLURM_SUBMIT_DIR/mpi-smoke-$SLURM_JOB_ID"
mkdir -p -- "$run_dir"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$run_dir/hosts.txt"
printf 'Allocated nodes:\n'
cat -- "$run_dir/hosts.txt"

mpicc -std=c99 -O2 -Wall -Wextra mpi-smoke.c -o "$run_dir/mpi-smoke"
# mpirun obtains the allocation from Slurm; hosts.txt is informational here.
mpirun -np 4 --map-by ppr:1:node "$run_dir/mpi-smoke"

