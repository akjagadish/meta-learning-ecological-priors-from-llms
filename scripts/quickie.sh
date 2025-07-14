#!/bin/bash -l
#SBATCH -o ./logs/%A_%a.out
#SBATCH -e ./logs/%A_%a.err
#SBATCH --job-name=sim
#SBATCH --time=1:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com


cd ~/ermi/
module purge
module load anaconda/3/2023.03
module load gcc/13 impi/2021.9
module load cuda/12.1
module load pytorch/gpu-cuda-12.1/2.2.0
pip3 install --user ipdb torch transformers tensorboard ipdb tqdm schedulefree


# python mi/simulate_bermi_categorylearning.py
python mi/fitted_simulations.py --model-name bermi --method bounded --task-name devraj2022 --optimizer differential_evolution