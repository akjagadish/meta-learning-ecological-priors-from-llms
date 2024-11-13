#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=generate_plots
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=72
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

# cd ~/ermi/categorisation/

# module purge
# module load anaconda/3/2023.03
# pip install groupBMC==1.0
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# python make_plots.py

# cd ~/ermi/functionlearning/

# module purge
# module load anaconda/3/2023.03
# pip install groupBMC==1.0
# pip3 install --user openai ipdb transformers tensorboard anthropic openml wordcloud mycolorpy Pillow
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# python make_plots.py


cd ~/ermi/

module purge
module load anaconda/3/2023.03
pip install groupBMC==1.0
pip3 install --user openai ipdb transformers tensorboard anthropic openml wordcloud mycolorpy Pillow
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python decisionmaking/make_plots.py
# python categorisation/make_plots.py
# python functionlearning/make_plots.py
