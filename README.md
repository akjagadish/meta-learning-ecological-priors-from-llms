[![arXiv Badge](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=fff&style=for-the-badge)]() 


# Meta-learning ecological priors from large language models explains human learning and decision making
This repository contains the code for the project "Meta-learning ecological priors from large language models explains human learning and decision making"


<p align="center">
  <img src="ermi.png" />
</p>

## Abstract
Human cognition is profoundly shaped by the environments in which it unfolds. Yet, it remains an open question whether learning and decision making can be explained as a principled adaptation to the statistical structure of real-world tasks. We introduce ecologically rational analysis, a computational framework that unifies the normative foundations of rational analysis with ecological grounding. Leveraging large language models to generate ecologically valid cognitive tasks at scale, and using meta-learning to derive rational models optimized for these environments, we develop a new class of learning algorithms: Ecologically Rational Meta-learned Inference (ERMI). ERMI internalizes the statistical regularities of naturalistic problem spaces and adapts flexibly to novel situations, without requiring hand-crafted heuristics or explicit parameter updates. We show that ERMI captures human behavior across 15 experiments spanning function learning, category learning, and decision making, outperforming several established cognitive models in trial-by-trial prediction. Our results suggest that much of human cognition may reflect adaptive alignment to the ecological structure of the problems we encounter in everyday life.

## Project Structure

```bash
.
├── categorisation
│   ├── baselines
│   │   ├── gcm.py
│   │   ├── llm.py
│   │   ├── pm.py
│   │   ├── rulex.py
│   │   ├── run_gcm.py
│   │   ├── run_llm.py
│   │   ├── run_pm.py
│   │   ├── run_rulex.py
│   │   └── simulate_llm.py
│   ├── benchmark
│   │   ├── eval.py
│   │   └── save_eval_data.py
│   ├── data
│   │   ├── benchmark
│   │   ├── fitted_simulation
│   │   ├── generated_tasks
│   │   ├── human
│   │   ├── llm
│   │   ├── meta_learner
│   │   ├── model_comparison
│   │   ├── stats
│   │   └── task_labels
│   ├── mi
│   │   ├── baseline_classifiers.py
│   │   ├── envs.py
│   │   ├── evaluate.py
│   │   ├── fit_humans.py
│   │   ├── fitted_simulations.py
│   │   ├── human_envs.py
│   │   ├── model.py
│   │   ├── model_utils.py
│   │   ├── simulate_johanssen2002.py
│   │   ├── simulate_mi.py
│   │   ├── simulate_shepard1961.py
│   │   └── train_transformer.py
├── decisionmaking
├── functionlearning
├── taskgeneration
│   ├── generate_linear_data.py
│   ├── generate_real_data.py
│   ├── generate_synthetic_data.py
│   ├── generate_tasklabels.py
│   ├── generate_tasks.py
│   ├── parse_generated_tasks.py
│   ├── prompts.py
│   └── utils.py
├── trained_models
├── figures
├── scripts
├── notebooks
├── logs
└── README.md

```

The project also contains an .env file for storing environment variables and a requirements.txt file for installing the required Python libraries.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Installation
Clone the repository to your local machine. Then, install the required Python libraries from ` requirements.txt` and install the ermi package using pip:
    
```bash
git clone https://github.com/akjagadish/ermi.git
cd ermi
pip install -r requirements.txt
pip install .
```

### Configuration
The project uses a configuration file called .env to store environment variables. The .env file should be located in the root directory of the project and should contain the following variables:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
HF_API_KEY=your_huggingface_api_key
BERMI_DIR=/PATH/to/dir/
BERMI_WEIGHTS==/PATH/to/dir/
```
Replace your_anthropic_api_key with your actual Anthropic API key. You can obtain an API key by signing up for Anthropic's API service.
For using opensource models from Huggingface, you need to sign up for Huggingface and get an API key. Replace your_huggingface_api_key with your actual Huggingface API key.
Add the path to the directory where you have cloned the repository to the BERMI_DIR variable. Add the path to the directory where you want to save the trained models to the BERMI_WEIGHTS variable. Alternatively, you can export these variables in your shell.
```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key
export HF_API_KEY=your_huggingface_api_key
export BERMI_DIR=/PATH/to/dir/
export BERMI_WEIGHTS==/PATH/to/dir/
```

## Usage

I will run through how to generate category learning tasks from Claude-v2 for three dimensional stimuli, train ERMI model on this task, simulate data  from the trained model, and fit the model to human data from Badham et al. 2017 task. But the same steps can be used for other tasks and models (MI and PFN) as well. I will also show how to fit a baseline model on Badham et al. 2017 task and how to run benchmarking on OpenML-CC18 benchmark. All scripts used for running experiments are located in the scripts directory. Note that all the *.sh files in there are written for the HPC cluster we use, therefore it likely won't run out of the box on other systems. The python scripts should be portable as is.


### Generate category learning tasks from Claude-v2
To generate category learning tasks from Claude-v2, there are two steps. 

Step 1: Generate task labels using the following command:
```bash
# Generate task labels in 100 separte runs for category learning tasks from Claude-v2
python task_generation/generate_tasklabels.py --model NA --proc-id 0 --num-runs 100 --num-tasks 250 --num-dim 3 --max-length 10000 --run-gpt claude --prompt-version 5 

# Pool the generated task labels into a single pandas dataframe
python task_generation/generate_tasklabels.py --model NA --proc-id 0 --num-runs 100 --num-tasks 250 --num-dim 3 --max-length 10000 --run-gpt claude --prompt-version 5 --pool --path /PATH/to/dir/categorisation/data/tasklabels

``` 

Step 2: Generate category learning tasks using the following command:
```bash
python task_generation/generate_tasks.py --model NA --proc-id 0  --num-tasks 10000 --start-task-id 0 --num-dim 3 --num-data 100 --max-length 4000 --run-gpt claude --prompt-version 4 --use-generated-tasklabels --file-name-tasklabels claude_generated_tasklabels_paramsNA_dim3_tasks23426_pversion5 --path-tasklabels /PATH/to/dir/categorisation/data/tasklabels
```

### Train ERMI model

To train ERMI model on the generated tasks, use the following command:
```bash
python mi/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --env-dir /PATH/to/dir/categorisation/data/generated_tasks --shuffle-features --first-run-id 0
```

### Fit ERMI model to human data

To fit ERMI model to human data from Badham et al. 2017 task, use the following command:
```bash
python mi/fit_humans.py --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0 --task-name badham2017 --optimizer
```

### Simulate data from ERMI model using fitted parameters

To simulate data from ERMI model using fitted parameters, use the following command:
```bash
python mi/fitted_simulations.py --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0 --task-name badham2017 --optimizer differential_evolution

```

### Fit baseline models to human data for model comparison

To run a baseline model on the Badham et al. 2017 task, use the following command:
```bash
python baselines/run_gcm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --task-name badham2017 
python baselines/run_pm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --prototypes from_data --task-name badham2017
python baselines/run_rulex.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --task-name badham2017
python baselines/run_llm.py --num-iter 1 --loss 'nll' --num-blocks 1 --fit-human-data --dataset badham2017
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This project is for research purposes only and should not be used for any other purposes.

## Citation

If you use our work, please cite our
[Coming soon]() as such:

``` bibtex
Coming soon
```
