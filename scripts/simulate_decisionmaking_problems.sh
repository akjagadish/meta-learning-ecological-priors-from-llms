#!/bin/bash -l
#SBATCH -o ./logs/%A_%a.out
#SBATCH -e ./logs/%A_%a.err
#SBATCH --job-name=SyntheticTaskGeneration
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com


cd ~/ermi/

module purge
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


# python mi/simulate_data.py --num-tasks 10000 --num-dims 2 --max-steps 20 --paradigm decisionmaking
python mi/simulate_data.py --num-tasks 10000 --num-dims 4 --max-steps 20 --paradigm decisionmaking
python mi/simulate_data.py --num-tasks 400 --num-dims 4 --max-steps 20 --paradigm decisionmaking --ranked
python mi/simulate_data.py --num-tasks 400 --num-dims 4 --max-steps 20 --paradigm decisionmaking --direction
python mi/simulate_data.py --num-tasks 400 --num-dims 4 --max-steps 20 --paradigm decisionmaking