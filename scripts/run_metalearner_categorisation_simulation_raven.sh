#!/bin/bash -l
#SBATCH -o ./logs/%A_%a.out
#SBATCH -e ./logs/%A_%a.err
#SBATCH --job-name=sim_categorisation
#SBATCH --time=01:00:00
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


## johannsen task
# python mi/simulate.py --job-id ${SLURM_ARRAY_TASK_ID} --experiment johanssen_categorisation --num-runs 1 --model-name env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# python mi/simulate.py --job-id ${SLURM_ARRAY_TASK_ID} --experiment johanssen_categorisation --num-runs 1 --model-name env=dim4synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_synthetic 
# python mi/simulate.py --job-id ${SLURM_ARRAY_TASK_ID} --experiment johanssen_categorisation --num-runs 1 --model-name env=dim4synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_syntheticnonlinear

## smith task
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.  --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.1 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.2 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.3 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.4 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.5 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.6 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.7 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.8 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.9 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate.py --experiment smith_categorisation --num-runs 10 --beta 1.0 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1

## shepard task
# python mi/simulate.py --experiment shepard_categorisation --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python mi/simulate_model.py --use-base-model-name --ess 15 --offset 0 --scale 0.5 --job-array --paradigm categorisation --task-name shepard1961 --policy bernoulli --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_lossbce_ess3.5_run=3_regall_essinit0.0_annealed
# python mi/simulate_model.py --use-base-model-name --ess ${SLURM_ARRAY_TASK_ID} --offset 0 --scale 0.5 --job-array --paradigm categorisation --task-name shepard1961 --policy bernoulli --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_lossbce_ess3.5_run=3_regall_essinit0.0_annealed
