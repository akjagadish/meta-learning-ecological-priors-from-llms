import sys
sys.path.append('/u/ajagadish/ermi/categorisation/')
sys.path.append('/u/ajagadish/ermi/mi/')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from collections import Counter, defaultdict
from wordcloud import WordCloud
from mycolorpy import colorlist as mcp
import torch.nn as nn
import math
import ivon
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal, Gamma
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights, TransformerDecoderRegression, TransformerDecoderRegressionLinearWeights, TransformerDecoderLinearWeightsConstrained
import torch.nn.utils.rnn as rnn_utils
FONTSIZE=20
SYS_PATH = '/u/ajagadish/ermi' #'/raven/u/ajagadish/vanilla-llama/'
from envs import ShepardsTask

##  simulate bermi on Shepards tasks

esses = np.arange(0.0, 8.0, 0.5)
betas = np.arange(0.0, 1.1, 0.1)
num_runs = 5
num_blocks = 15
num_trials_per_block = 16
num_tasks = 6
max_steps = num_blocks * num_trials_per_block
policy = 'binomial'
shuffle_trials=True
device = 'cpu'
block_errors_esses = torch.zeros((len(esses), len(betas), num_tasks, num_blocks))

for e_idx, ess in enumerate(esses):
    for b_idx, beta in enumerate(betas):
        model_path = f'{SYS_PATH}/categorisation/trained_models/env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_lossbce_ess{ess}_run=3_regall_essinit0.0_annealed.pt'

        # load model
        env = ShepardsTask(task=1, max_steps=max_steps, shuffle_trials=shuffle_trials)
        model = TransformerDecoderClassification(num_input=env.num_dims, num_output=env.num_choices, num_hidden=256, num_layers=6, d_model=64, num_head=8, max_steps=300, loss='bce', device=device).to(device)
        state_dict = torch.load(model_path, map_location=device, weights_only=False)[1]
        model.load_state_dict(state_dict)
        model.to(device)

        corrects_over_tasks = []
        for task_idx in range(1, num_tasks+1):
            env = ShepardsTask(task=task_idx, max_steps=max_steps, shuffle_trials=shuffle_trials)
            corrects = []
            for run_idx in range(num_runs):
                with torch.no_grad():
                    model.eval()
                    packed_inputs, sequence_lengths, targets = env.sample_batch()
                    model.beta = beta  # model beta is adjustable at test time
                    model.device = device
                    model_choices = model(
                        packed_inputs.float().to(device), sequence_lengths)

                    # sample from model choices probs using binomial distribution
                    if policy == 'binomial':
                        model_choices = torch.distributions.Binomial(
                            probs=model_choices).sample()
                    elif policy == 'greedy':
                        model_choices = model_choices.round()

                    model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
                        sequence_lengths)], axis=0).squeeze().float()
                    true_choices = torch.concat(targets, axis=0).float().to(device)
                    accuracy = (model_choices == true_choices).sum() / (model_choices.shape[0])

                cum_sum = np.array(sequence_lengths).cumsum()
                for idx, _ in enumerate(cum_sum):
                    task_correct = (model_choices == true_choices.squeeze())[
                                    cum_sum[idx]-cum_sum[0]:cum_sum[idx]]
                    corrects.append(task_correct)

            corrs = torch.stack(corrects).sum(0)/(num_runs*env.batch_size)
            corrects_over_tasks.append(corrs)

        corrs_over_tasks = 1-torch.stack(corrects_over_tasks)
        block_errors = []
        for t_idx in range(num_tasks):
            b_errors =torch.stack(np.split(corrs_over_tasks[t_idx], num_blocks)).mean(1)
            block_errors.append(b_errors)
        block_errors = torch.stack(block_errors)
        block_errors_esses[e_idx, b_idx] = block_errors

# save block errors
block_errors_esses = block_errors_esses.numpy() 
np.save(f'{SYS_PATH}/categorisation/data/model_simulation/bermi_block_errors_esses_betas.npy', block_errors_esses)

## compute distance to humans

# load json file containing the data
with open(f'{SYS_PATH}/categorisation/data/human/nosofsky1994.json') as json_file:
    data = json.load(json_file)

mse_distance = np.zeros((len(esses), len(betas), num_tasks))
for e_idx, ess in enumerate(esses):
    for b_idx, beta in enumerate(betas):
        for t_idx, rule in enumerate(data.keys()):
            block_errors = block_errors_esses[e_idx, b_idx, t_idx]
            human_block_error = data[rule]['y'][:num_blocks]
            mse_distance[e_idx, b_idx, t_idx] = np.mean((block_errors-human_block_error)**2)

# mean across tasks
mse_distance = mse_distance.mean(2)
ind = np.unravel_index(np.argmin(mse_distance, axis=None), mse_distance.shape)
print(f'ess={esses[ind[0]]}, beta={betas[ind[1]]}, mse={mse_distance[ind]}')