#!/bin/bash -l
#SBATCH --job-name=Grevys_Experiment_5
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1               # one Slurm task
#SBATCH --cpus-per-task=4        # …with 4 CPU cores
#SBATCH --mem=50G
#SBATCH --account=PAS2136
#SBATCH --gres=gpu:1             # or --gpus=1 on some clusters
#SBATCH --output=/users/PAS2136/upadha2/GGR/GGR.out
#SBATCH --error=/users/PAS2136/upadha2/GGR/GGR.err

conda activate smk_pipeline

cd '/users/PAS2136/upadha2/GGR'

# ── Run Snakemake on the 4 cores we requested ─────────────────────────────────
snakemake -s snakefile.smk --cores "$SLURM_CPUS_PER_TASK" 

