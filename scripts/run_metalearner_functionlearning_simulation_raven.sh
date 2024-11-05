#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=sim_functionlearning
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

## little2022
# python mi/simulate_model.py --ess ${SLURM_ARRAY_TASK_ID} --offset 0 --scale 0.25 --job-array --paradigm functionlearning --task-name little2022 --policy greedy --use-base-model-name --model-name env=claude_dim1_maxsteps25_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_ess0.0_run=0_regall_essinit0.0_annealed_schedulefree
# python mi/simulate_model.py --ess ${SLURM_ARRAY_TASK_ID} --offset 0 --scale 0.25 --job-array --paradigm functionlearning --task-name little2022 --policy greedy --use-base-model-name --model-name env=synthetic_dim1_maxsteps25_dim1_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.01_shuffleTrue_ess0.0_run=0_synthetic_regall_essinit0.0_annealed_schedulefree

#  syntheticfunctionlearning
# python mi/simulate_model.py --ess ${SLURM_ARRAY_TASK_ID} --offset 0 --scale 0.25 --job-array --paradigm functionlearning --task-name syntheticfunctionlearning --policy greedy --use-base-model-name --model-name env=claude_dim1_maxsteps25_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.0_shuffleTrue_ess0.0_run=0_regall_essinit0.0_annealed_schedulefree
python mi/simulate_model.py --ess ${SLURM_ARRAY_TASK_ID} --offset 0 --scale 0.25 --job-array --paradigm functionlearning --task-name syntheticfunctionlearning --policy greedy --use-base-model-name --model-name env=synthetic_dim1_maxsteps25_dim1_model=transformer_num_episodes1000000_num_hidden=8_lr0.0003_num_layers=2_d_model=64_num_head=8_noise0.01_shuffleTrue_ess0.0_run=0_synthetic_regall_essinit0.0_annealed_schedulefree
python mi/simulate_model.py --paradigm functionlearning --task-name little2022 --policy greedy --use-filename --model-name env=claude_dim1_maxsteps25_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
python mi/simulate_model.py --paradigm functionlearning --task-name syntheticfunctionlearning --policy greedy --use-filename --model-name env=claude_dim1_maxsteps25_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_ess0.0_run=0_regall_schedulefree
python mi/simulate_model.py --paradigm functionlearning --task-name syntheticfunctionlearning --policy greedy --use-filename --model-name env=claude_dim1_maxsteps25_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_ess0.0_run=0_regall

#python mi/simulate_model.py --paradigm functionlearning --task-name little2022 --policy greedy --use-filename --model-name env=synthetic_dim1_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.01_shuffleTrue_run=0_synthetic
#python mi/simulate_model.py --paradigm functionlearning --task-name little2022 --policy greedy --use-filename --model-name env=claude_dim1_maxsteps25_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
#python mi/simulate_model.py --paradigm functionlearning --task-name little2022 --policy greedy --use-filename --model-name env=synthetic_dim1_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.01_shuffleTrue_run=0_synthetic
#python mi/simulate_model.py --paradigm functionlearning --task-name syntheticfunctionlearning --policy greedy --use-filename --model-name env=claude_dim1_maxsteps25_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
#python mi/simulate_model.py --paradigm functionlearning --task-name syntheticfunctionlearning --policy greedy --use-filename --model-name env=synthetic_dim1_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.01_shuffleTrue_run=0_synthetic

# python mi/simulate_model.py --paradigm functionlearning --task-name little2022 --model-name env=claude_generated_functionlearningtasks_paramsNA_dim1_data25_tasks9991_run0_procid0_pversion2_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_norm0.5
# python mi/simulate_model.py --paradigm functionlearning --task-name little2022 --model-name env=claude_dim1_maxsteps25_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# python mi/simulate_model.py --paradigm functionlearning --task-name little2022 --use-filename --model-name env=synthetic_dim1_model=transformer_num_episodes100000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.01_shuffleTrue_run=0_synthetic

