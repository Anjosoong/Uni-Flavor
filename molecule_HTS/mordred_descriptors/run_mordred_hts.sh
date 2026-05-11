#!/bin/bash
#SBATCH -J coconut_mordred
#SBATCH -p kshcnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2000  # 2GB per CPU × 32 CPUs = 64GB total
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --time=24:00:00

# SLURM job script for Mordred descriptor calculation (HTS, 32 cores / 64 logical CPUs)
# Optimized for COCONUT database (730,000 molecules)

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Cores: $SLURM_CPUS_PER_TASK"

# Load required modules (adjust based on your cluster)
# module load python/3.8
# module load rdkit/2022.03

# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Change to working directory
cd /path/to/working/directory || exit 1

# Activate conda environment
source activate mordred2

# Redirect stdout/stderr to stable files in the working directory.
# This avoids multiprocessing flush failures when Slurm spool streams become unavailable.
RUNTIME_STDOUT="coconut_mordred_${SLURM_JOB_ID}.out"
RUNTIME_STDERR="coconut_mordred_${SLURM_JOB_ID}.err"
exec >>"${RUNTIME_STDOUT}" 2>>"${RUNTIME_STDERR}"

echo "Runtime stdout: ${RUNTIME_STDOUT}"
echo "Runtime stderr: ${RUNTIME_STDERR}"

# Run the descriptor calculation
echo "Starting Mordred descriptor calculation..."
echo "Input: coconut_calculation.csv"
echo "Output dir: mordred_output"
echo "Output: coconut_mordred_descriptors.csv"
echo "Batch size: 512"
echo "CPU cores: 32"
echo ""

python3 mordred_descriptors_hts.py \
    --input coconut_calculation.csv \
    --output-dir mordred_output \
    --output coconut_mordred_descriptors.csv \
    --batch-size 512 \
    --nproc 32 \
    --smiles-col SMILES \
    --id-col id

echo "Job finished at: $(date)"

# Check if output file was created
if [ -f "mordred_output/coconut_mordred_descriptors.csv" ]; then
    echo "SUCCESS: Output file created"
    wc -l mordred_output/coconut_mordred_descriptors.csv
    ls -lh mordred_output/coconut_mordred_descriptors.csv
else
    echo "ERROR: Output file not found"
    exit 1
fi
