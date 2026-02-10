import numpy as np
import torch
from envs import Binz2022, Badham2017, Devraj2022, Little2022, SyntheticFunctionlearningTask, DeLosh1997, EvaluateFunctionLearning, HandCraftedFunctions, ExperimentFunctions
import argparse
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights, TransformerDecoderRegression, TransformerDecoderRegressionLinearWeights, TransformerDecoderLinearWeightsConstrained
import sys
import re
# import ivon
from model_utils import parse_model_path
from torch.distributions import Bernoulli
import os
from dotenv import load_dotenv
load_dotenv()
# sys.path.insert(0, '/scratch/gpfs/GRIFFITHS/aj9225/meta-learning-ecological-priors-from-llms/mi')
SYS_PATH = os.getenv('SYS_PATH')

def compute_mse(x, y, axis=1, per_trial=False):
    return ((x - y) ** 2).mean(axis=axis) if not per_trial else ((x - y) ** 2)

def compute_loglikelihood_human_choices_under_model(env=None, model_path=None, participant=0, beta=1., epsilon=0., policy='greedy', constraint=True, device='cpu', paired=False, **kwargs):

    # parse model parameters
    num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, kwargs)

    # initialise model
    if paired:
        if constraint:
            model = TransformerDecoderLinearWeightsConstrained(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                    num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
        else:
            model = TransformerDecoderLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                    num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

    else:
        model = TransformerDecoderClassification(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                 num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
    
    # load model weights
    state_dict = torch.load(model_path, map_location=device, weights_only=False)[1]
    model.load_state_dict(state_dict)
    model.to(device)

    with torch.no_grad():

        # model setup: eval mode and set beta
        model.eval()
        model.beta = beta
        model.device = device  
        # env setup: sample batch from environment and unpack
        outputs = env.sample_batch(participant, paired=paired)

        if not hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, correct_choices, human_choices, _ = outputs
        elif hasattr(env, 'return_prototype') and (env.return_prototype is True):
            packed_inputs, sequence_lengths, correct_choices, human_choices, _, _ = outputs

        # get model choices
        model_choice_probs = model(packed_inputs.float().to(device), sequence_lengths)
        model_choices = model_choice_probs.round() if policy == 'greedy' else Bernoulli(
                    probs=model_choice_probs).sample()
        l2_norm = torch.norm(torch.cat([p.flatten() for p in model.parameters() if p is not None]), 2)

        # compute metrics
        per_trial_model_accuracy =(model_choices == correct_choices)
        correct_choice_probs = torch.concat([1-model_choice_probs, model_choice_probs], 2).gather(2, correct_choices.to(torch.int64))
        expected_log_likelihood = torch.log(correct_choice_probs)
        per_trial_human_accuracy = (human_choices == correct_choices)
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = torch.concat([correct_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = correct_choices.reshape(-1).float().to(device)
        model_accuracy = (model_choices == correct_choices).sum() / \
            correct_choices.numel()
        human_accuracy = (human_choices.reshape(-1) ==
                          correct_choices).sum() / correct_choices.numel()
        model_coefficients = model.w.detach().numpy() if paired else None

    return model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy, model_coefficients, expected_log_likelihood, l2_norm

def compute_mses_human_predictions_under_model(env=None, model_path=None, participant=0, device='cpu', paired=False, policy='greedy', constraint=True, **kwargs):
    
    # parse model parameters
    num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, kwargs)

    # initialise model
    if paired:
        if constraint:
            pass
        #     model = TransformerDecoderLinearWeightsConstrained(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
        #                                             num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
        else:
            model = TransformerDecoderRegressionLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                        num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

    else:
        model = TransformerDecoderRegression(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                 num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
    
    #load model weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)[1]
    model.load_state_dict(state_dict)
    model.to(device)

    with torch.no_grad():

        # model setup: eval mode and set beta
        model.eval()
        model.device = device

        # env setup: sample batch from environment and unpack
        if kwargs.get('synthetic'):
            outputs = env.sample_batch()
            packed_inputs, sequence_lengths, targets, inputs, kernel_choices = outputs[:5]
            raw_inputs = outputs[5] if len(outputs) > 5 else None
            raw_targets = outputs[6] if len(outputs) > 6 else None

            # get model preds
            model_preds = model(packed_inputs.float().to(device), sequence_lengths)
            model_preds = model_preds.mean if policy == 'greedy' else model_preds.sample()

            # compute metrics
            model_error = compute_mse(model_preds, targets.unsqueeze(2))
            per_trial_model_error = compute_mse(model_preds.squeeze(), targets, per_trial=True)
            
            return model_preds, model_error, per_trial_model_error, targets, inputs, kernel_choices, raw_inputs, raw_targets
        
        else:

            packed_inputs, sequence_lengths, targets, human_preds, ground_truth_functions = env.sample_batch(participant, paired=paired)
         
            # get model preds
            model_preds = model(packed_inputs.float().to(device), sequence_lengths)
            model_preds = model_preds.mean if policy == 'greedy' else model_preds.sample()

            # compute metrics
            model_error = compute_mse(model_preds[:, -1], targets[:, -1])
            model_preds = torch.concat([model_preds[i, [-1]] for i, _ in enumerate(
                sequence_lengths)], axis=0).squeeze().float()
            targets = torch.concat([targets[i, [-1]] for i, _ in enumerate(
                sequence_lengths)], axis=0).squeeze().float()
            targets = targets.reshape(-1).float().to(device)

            return model_preds, model_error, None, targets, human_preds, ground_truth_functions


def sample_model(args):

    model_path = f"{SYS_PATH}/{args.paradigm}/trained_models/{args.model_name}.pt"
    if args.task_name == 'badham2017':
        env = Badham2017()
        task_features = {'model_max_steps': 96, 'human_data': True}
    elif args.task_name == 'devraj2022':
        env = Devraj2022()
        task_features = {'model_max_steps': 616, 'human_data': True}
    elif args.task_name == 'binz2022':
        env = Binz2022(experiment_id=args.exp_id)
        task_features = {'model_max_steps': 10, 'human_data': True}
    elif args.task_name == 'little2022':
        env = Little2022(condition=args.exp_id, evaluate_human_preds=True)
        task_features = {'model_max_steps': 25, 'human_data': True}
    elif args.task_name == 'syntheticfunctionlearning':
        env = SyntheticFunctionlearningTask(num_dims=1, mode='test', max_steps=25)
        env.num_samples = 10
        env.batch_size = 100
        task_features = {'model_max_steps': 25, 'synthetic': True}
    elif args.task_name == 'evaluatefunctionlearning':
        env = EvaluateFunctionLearning(num_dims=1, max_steps=25, noise=0.1)
        env.num_samples = 100
        env.batch_size = 1000
        task_features = {'model_max_steps': 25, 'synthetic': True}
    elif args.task_name == 'delosh1997':
        env = DeLosh1997(max_steps=args.model_max_steps)
        task_features = {'model_max_steps': args.model_max_steps, 'synthetic': True}
        env.num_samples = 2
    elif args.task_name == 'kwantes2006':
        env = DeLosh1997(max_steps=args.model_max_steps, offset=True)
        task_features = {'model_max_steps': args.model_max_steps, 'synthetic': True}
        env.num_samples = 2
    elif args.task_name == 'handcrafted_functions':
        env = HandCraftedFunctions(max_steps=20, scale=0.25)
        env.num_samples = 1
        task_features = {'model_max_steps': 25, 'synthetic': True}
    elif args.task_name == 'experiment_functions_interpolation':
        env = ExperimentFunctions(
            n_train_per_seg=10, n_test_per_seg=5,
            noise_std=1.0, seed=42, scale=0.50,
            save_data=True,
            output_path=f"{SYS_PATH}/functionlearning/data/experiment_data/interpolation/experiment_functions_stimuli_interpolation.json",
            figures_output_path=f"{SYS_PATH}/functionlearning/data/experiment_data/interpolation/plots/",
        )
        env.num_samples = 1
        env._eval_mode = 'interpolation'
        task_features = {'model_max_steps': 25, 'synthetic': True}
    elif args.task_name == 'experiment_functions_extrapolation':
        env = ExperimentFunctions(
            n_train_per_seg=10, n_test_per_seg=5,
            noise_std=1.0, seed=42, scale=0.50,
            save_data=True,
            output_path=f"{SYS_PATH}/functionlearning/data/experiment_data/extrapolation/experiment_functions_stimuli_extrapolation.json",
            figures_output_path=f"{SYS_PATH}/functionlearning/data/experiment_data/extrapolation/plots/",
        )
        env.num_samples = 1
        env._eval_mode = 'extrapolation'
        task_features = {'model_max_steps': 25, 'synthetic': True}
    else:
        raise NotImplementedError
   
    participants = env.data.participant.unique() if task_features.get('human_data') else range(env.num_samples)
    
    if args.task_name in ['little2022', 'syntheticfunctionlearning', 'delosh1997', 'evaluatefunctionlearning', 'kwantes2006', 'handcrafted_functions',
                          'experiment_functions_interpolation', 'experiment_functions_extrapolation']:

        model_errors, per_trial_model_errors, model_preds, targets, human_preds, ground_truth_functions, raw_inputs_list, raw_targets_list = [], [], [], [], [], [], [], []
        for participant in participants:
            results = compute_mses_human_predictions_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                                                                             paired=args.paired, constraint=args.constraint, **task_features)
            model_pred, model_error, per_trial_model_error, target, human_pred, ground_truth_function = results[:6]
            raw_inp = results[6] if len(results) > 6 else None
            raw_tgt = results[7] if len(results) > 7 else None
            model_preds.append(model_pred)
            model_errors.append(model_error)
            per_trial_model_errors.append(per_trial_model_error)
            targets.append(target)
            human_preds.append(human_pred)
            ground_truth_functions.append(ground_truth_function)
            raw_inputs_list.append(raw_inp)
            raw_targets_list.append(raw_tgt)
        raw_inputs_out = np.stack(raw_inputs_list) if raw_inputs_list[0] is not None else None
        raw_targets_out = np.stack(raw_targets_list) if raw_targets_list[0] is not None else None
        return np.stack(model_preds), np.stack(model_errors), np.stack(per_trial_model_errors), np.stack(targets), np.stack(human_preds), np.stack(ground_truth_functions), raw_inputs_out, raw_targets_out
        
    elif args.task_name in ['badham2017', 'devraj2022', 'binz2022']:

        per_trial_accs, per_trial_human_accs, human_accs, accs, coeffs, exp_logs, l2_norms = [], [], [], [], [], [], []
        for participant in participants:  
            beta, epsilon = 1., 0.
            model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy, model_coeffs, expected_log_likelihood, l2_norm = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, shuffle_trials=True,
                                                                                                                    beta=beta, epsilon=epsilon, policy=args.policy, paired=args.paired, constraint=args.constraint, **task_features)
            human_accs.append(human_accuracy)
            per_trial_accs.append(per_trial_model_accuracy)
            per_trial_human_accs.append(per_trial_human_accuracy)
            accs.append(model_accuracy)
            coeffs.append(model_coeffs)
            exp_logs.append(expected_log_likelihood)
            l2_norms.append(l2_norm)
        
        return np.array(accs), torch.stack(per_trial_accs).squeeze().sum(1), np.array(human_accs), np.stack(per_trial_human_accs).squeeze().sum(1), np.stack(coeffs), torch.stack(exp_logs).squeeze(), np.array(l2_norms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='save meta-learners choices for a given task within a paradigm')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--paradigm', type=str, default='categorisation')
    parser.add_argument('--task-name', type=str,
                        required=True, help='task name')
    parser.add_argument('--exp-id', type=int, default=1, help='experiment id')
    parser.add_argument('--model-name', type=str,
                        required=True, help='model name')
    parser.add_argument('--paired', action='store_true',
                        default=False, help='paired')
    parser.add_argument('--policy', type=str, default='greedy',
                        help='method to use for computing model choices')
    parser.add_argument('--ess', type=float, default=None,
                         help='weight for the nll loss term in the ELBO')
    parser.add_argument('--constraint', action='store_true',
                        default=False, help='use constrained model')
    parser.add_argument('--job-array', action='store_true',
                        default=False, help='passed as a job array')
    parser.add_argument('--scale', type=float, default=0.,
                        help='scale for the job array')
    parser.add_argument('--offset', type=int, default=0,
                        help='offset for the job array')
    parser.add_argument('--use-filename', action='store_true',
                        default=False, help='use filename for saving')
    parser.add_argument('--use-base-model-name', action='store_true',
                        default=False, help='use base model name for saving')
    parser.add_argument('--model_max_steps', type=int, default=None,
                        help='maximum number of steps for the model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    
    args.ess = args.ess * args.scale + args.offset if args.job_array else args.ess
    args.model_name = re.sub(r'ess\d+\.\d+', f'ess{args.ess}', args.model_name) if args.use_base_model_name else args.model_name
    # (for ivon based models) args.model_name = args.model_name.replace('essNone', f'ess{str(int(args.ess))}') if (args.ess is not None) and (args.job_array) else args.model_name
    if args.use_filename:
        save_path = f"{args.paradigm}/data/model_simulation/{args.model_name}_{args.task_name}.npz"

    else:
        num_hidden, num_layers, d_model, num_head, loss_fn, _, source, condition, _ = parse_model_path(args.model_name, {}, return_data_info=True)
        save_path = f"{args.paradigm}/data/model_simulation/task={args.task_name}_experiment={args.exp_id}_source={source}_condition={condition}_loss={loss_fn}_paired={args.paired}_policy={args.policy}.npz"
        save_path = save_path.replace('.npz', f"_ess={str(round(float(args.ess), 4))}.npz") if args.ess is not None else save_path
        save_path = save_path.replace('.npz', f"_max_steps={str(args.model_max_steps)}.npz") if args.model_max_steps is not None else save_path
        
    
    if args.paradigm == 'functionlearning':
        results = sample_model(args)
        model_preds, model_errors, per_trial_model_errors, targets, human_preds, ground_truth_functions = results[:6]
        raw_inputs = results[6] if len(results) > 6 else None
        raw_targets = results[7] if len(results) > 7 else None
        print('saving')
        # save list of results
        save_dict = dict(model_preds=model_preds, model_errors=model_errors, per_trial_model_errors=per_trial_model_errors, targets=targets, 
                         human_preds=human_preds, ground_truth_functions=ground_truth_functions)
        if raw_inputs is not None:
            save_dict['raw_inputs'] = raw_inputs
        if raw_targets is not None:
            save_dict['raw_targets'] = raw_targets
        np.savez(save_path, **save_dict)
    else:
        model_accuracy, per_trial_model_accuracy, human_accuracy, per_trial_human_accuracy, model_coefficients, expected_log_likelihood, l2_norms  = sample_model(args)
        print('saving')
        # save list of results
        np.savez(save_path, model_accuracy=model_accuracy, per_trial_model_accuracy=per_trial_model_accuracy, 
                 human_accuracy=human_accuracy, per_trial_human_accuracy=per_trial_human_accuracy, model_coefficients=model_coefficients,
                 expected_log_likelihood=expected_log_likelihood, l2_norms=l2_norms)
