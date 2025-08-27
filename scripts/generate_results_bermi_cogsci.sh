#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=generate_plots
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=72
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
# python mi/fitted_simulations.py --model-name bermi --method bounded --task-name devraj2022 --optimizer differential_evolution
python categorisation/baselines/run_gcm.py --num-iter 10 --task-name devraj2022 --loss mse_transfer --num-blocks 11  --model-name bermi --method bounded
python categorisation/baselines/run_pm.py --num-iter 10 --prototypes from_data --task-name devraj2022 --loss mse_transfer --num-blocks 11  --model-name bermi --method bounded