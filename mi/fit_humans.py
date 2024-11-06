import numpy as np
import torch
from envs import Binz2022, Badham2017, Devraj2022
import argparse
from tqdm import tqdm
from scipy.optimize import differential_evolution, minimize
import sys
import re
from model import TransformerDecoderClassification, TransformerDecoderLinearWeights
from model_utils import parse_model_path
from torch.distributions import Bernoulli
import pandas as pd
# sys.path.insert(0, '/u/ajagadish/ermi/mi')
# SYS_PATH = '/u/ajagadish/ermi'
from os import getenv
from dotenv import load_dotenv
load_dotenv()
SYS_PATH = getenv('BERMI_DIR')

def compute_loglikelihood_human_choices_under_model(env, model, participant=0, beta=1., epsilon=0., method='soft_sigmoid', policy='greedy', paired=False, model_path=None, **kwargs):

    with torch.no_grad():

        if method in ['bounded_soft_sigmoid', 'bounded_resources', 'grid_search']:
            model = model if kwargs['state_dict'] else torch.load(model_path)[1].to(device) if device=='cuda' else torch.load(model_path, map_location=torch.device('cpu'))[1].to(device)
            state_dict = torch.load(
                model_path, map_location=device)[1] if kwargs['state_dict'] else model.state_dict()  
            for key in state_dict.keys():
                state_dict[key][..., [np.random.choice(state_dict[key].shape[-1], int(state_dict[key].shape[-1] * epsilon), replace=False)]] = 0
            model.load_state_dict(state_dict)
        
        # model setup: eval mode, set device, and fix beta
        model.eval()
        model.to(device)
        model.beta = beta

        # env setup: sample batch from environment and unpack
        outputs = env.sample_batch(participant, paired=paired)

        if not hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, correct_choices, human_choices, _ = outputs
        elif hasattr(env, 'return_prototype') and (env.return_prototype is True):
            packed_inputs, sequence_lengths, correct_choices, human_choices, _, _ = outputs
        else:
            packed_inputs, sequence_lengths, correct_choices, human_choices, _ = outputs

        # get model choices
        model_choice_probs = model(
            packed_inputs.float().to(device), sequence_lengths)
        model_choices = model_choice_probs.round() if policy == 'greedy' else Bernoulli(
                    probs=model_choice_probs).sample()
    
        # compute log likelihoods of human choices under model choice probs (binomial distribution)
        loglikehoods = Bernoulli(
                probs=model_choice_probs.to('cpu')).log_prob(human_choices.float())
        summed_loglikelihoods = torch.vstack(
            [loglikehoods[idx, :sequence_lengths[idx]].sum() for idx in range(len(loglikehoods))]).sum()
       
        # sum log likelihoods only for unpadded trials per condition and compute chance log likelihood
        chance_loglikelihood = sum(sequence_lengths) * np.log(0.5)

        # task performance
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = torch.concat([correct_choices[i, :seq_len] for i, seq_len in enumerate(
            sequence_lengths)], axis=0).squeeze().float()
        correct_choices = correct_choices.reshape(-1).float().to(device)
        model_accuracy = (model_choices == correct_choices).sum() / \
            correct_choices.numel()
    
    return summed_loglikelihoods, chance_loglikelihood, model_accuracy.cpu().numpy()

def optimize(args):

    model_path = f"{SYS_PATH}/{args.paradigm}/trained_models/{args.model_name}.pt"
    if args.task_name == 'badham2017':
        env = Badham2017()
        task_features = {'model_max_steps': 96, 'state_dict': False}
    elif args.task_name == 'devraj2022':
        env = Devraj2022()
        task_features = {'model_max_steps': 616, 'state_dict': False}
    elif args.task_name == 'binz2022':
        env = Binz2022(experiment_id=args.exp_id)
        task_features = {'model_max_steps': 10, 'state_dict': True}
    else:
        raise NotImplementedError
    
    # parse model parameters
    num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, task_features)
    
    # initialise model
    if args.paired:
        model = TransformerDecoderLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

    else:
        model = TransformerDecoderClassification(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                 num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
    
    # load model weights
    if args.method == 'soft_sigmoid':
        if task_features['state_dict']:
            state_dict = torch.load(
                model_path, map_location=device)[1]    
            model.load_state_dict(state_dict)
        else:
            model = torch.load(model_path)[1].to(device) if device=='cuda' else torch.load(model_path, map_location=torch.device('cpu'))[1].to(device)    

    def objective(x, participant):
        epsilon = x[0] if args.method == 'bounded_resources' else 0.
        beta = x[0] if args.method == 'soft_sigmoid' else 1.
        if args.method == 'bounded_soft_sigmoid':
            epsilon = x[0]
            beta = x[1]
        elif args.method == 'grid_search':
            epsilon = args.epsilon
            beta = res.x[0]
        ll, _, _ = compute_loglikelihood_human_choices_under_model(env=env, model=model, participant=participant, shuffle_trials=True,
                                                                   beta=beta, epsilon=epsilon, method=args.method, paired=args.paired, model_path=model_path, ** task_features)
        return -ll.numpy()

    if args.method == 'soft_sigmoid':
        bounds = [(0., 1.)]
    elif args.method == 'bounded_resources':
        bounds = [(0., 0.5)]
    elif args.method == 'bounded_soft_sigmoid':
        bounds = [(0., .5), (0., 1.)]
    elif args.method == 'grid_search':
        bounds = [(0., 1.)]
    else:
        raise NotImplementedError

    pr2s, nlls, accs, parameters = [], [], [], []
    participants = env.data.participant.unique()
    for participant in tqdm(participants):
        res_fun = np.inf
        for _ in range(args.num_iters):

            # x0 = [np.random.uniform(x, y) for x, y in bounds]
            # result = minimize(objective, x0, args=(participant), bounds=bounds, method='SLSQP')
            result = differential_evolution(
                func=objective, args=(participant,), bounds=bounds)

            if result.fun < res_fun:
                res_fun = result.fun
                res = result
                print(f"min nll and parameter: {res_fun, res.x}")

        epsilon = res.x[0] if args.method == 'bounded_resources' else 0.
        beta = res.x[0] if args.method == 'soft_sigmoid' else 1.
        if args.method == 'bounded_soft_sigmoid':
            epsilon = res.x[0]
            beta = res.x[1]
        elif args.method == 'grid_search':
            epsilon = args.epsilon
            beta = res.x[0]

        ll, chance_ll, acc = compute_loglikelihood_human_choices_under_model(env=env, model=model, participant=participant, shuffle_trials=True,
                                                                             beta=beta, epsilon=epsilon, method=args.method, paired=args.paired, model_path=model_path, **task_features)
        nlls.append(res_fun)
        pr2s.append(1 - (res_fun/chance_ll))
        accs.append(acc)
        parameters.append(res.x)

    return np.array(pr2s), np.array(nlls), accs, parameters

def grid_search(args):

    model_path = f"{SYS_PATH}/{args.paradigm}/trained_models/{args.model_name}.pt"
    if args.task_name == 'badham2017':
        env = Badham2017()
        task_features = {'model_max_steps': 96, 'state_dict': False}
    elif args.task_name == 'devraj2022':
        env = Devraj2022()
        task_features = {'model_max_steps': 616, 'state_dict': False}
    elif args.task_name == 'binz2022':
        env = Binz2022(experiment_id=args.exp_id)
        task_features = {'model_max_steps': 10, 'state_dict': True}
    else:
        raise NotImplementedError
    
    # parse model parameters
    num_hidden, num_layers, d_model, num_head, loss_fn, model_max_steps = parse_model_path(model_path, task_features)
    
    # initialise model
    if args.paradigm == 'decisionmaking':
        model = TransformerDecoderLinearWeights(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)

    elif args.paradigm == 'functionlearning':
        pass
    elif args.paradigm == 'categorisation':
        model = TransformerDecoderClassification(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                                 num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=model_max_steps, loss=loss_fn, device=device).to(device)
    
    # load model weights
    state_dict = torch.load(
        model_path, map_location=device)[1]    
    model.load_state_dict(state_dict)

    pr2s, nlls, accs, parameters = [], [], [], []
    participants = env.data.participant.unique()
    for participant in tqdm(participants):
        res_fun = np.inf
        for _ in range(args.num_iters):
            for beta in np.linspace(0., 1., 100):
                ll, chance_ll, acc = compute_loglikelihood_human_choices_under_model(env=env, model=model, participant=participant, shuffle_trials=True,
                                                                                    beta=beta, epsilon=None, method=args.method, paired=args.paired, model_path=model_path, **task_features)
                if -ll.numpy() < res_fun:
                    res_fun = -ll.numpy()
                    res_acc = acc
                    res_beta = beta
                    print(f"current min nll and parameter: {res_fun, res_beta}")

        nlls.append(res_fun)
        pr2s.append(1 - (-res_fun/chance_ll))
        accs.append(res_acc)
        parameters.append([res_beta, args.ess])

    return np.array(pr2s), np.array(nlls), accs, parameters

def find_best_model_gs(args):
    
    bermi_esses = np.array([0.0, 0.5, 1., 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
    ermi_esses = np.array([0.0])
    data = pd.read_csv(f'{PARADIGM_PATH}/data/human/binz2022heuristics_exp{args.exp_id}.csv')
    sources = ['claude', 'synthetic']
    conditions = ['unknown', 'rank', 'pseudoranked'] if args.exp_id == 1 else ['unknown', 'direction', 'pseudodirection'] if args.exp_id == 2 else ['unknown']
    
    for condition in conditions:
        for source in sources:
            for esses in [bermi_esses, ermi_esses]:
                fitted_beta = np.zeros((len(esses), data.participant.max()+1))
                fitted_nlls = np.zeros((len(esses), data.participant.max()+1))
                for idx, ess in enumerate(esses):
                    results = np.load(f'{PARADIGM_PATH}/data/model_comparison/task={args.task_name}_experiment={args.exp_id}_source={source}_condition={condition}_ess={str(float(ess))}_loss=nll_paired=True_method=bounded_optimizer=grid_search_numiters=5.npz')
                    fitted_beta[idx] = results['betas'][:, 0]
                    fitted_nlls[idx] = results['nlls']

                best_idx = np.argmin(fitted_nlls, axis=0)
                best_ess = esses[best_idx]
                best_beta = fitted_beta[best_idx, np.arange(data.participant.max()+1)]
                best_nlls = fitted_nlls.min(0)

                method = 'unbounded' if ess == 0 and len(esses)==1  else 'bounded'
                np.savez(f"{PARADIGM_PATH}/data/model_comparison/task={args.task_name}_experiment={args.exp_id}_source={source}_condition={condition}_loss=nll_paired=True_method={method}_optimizer=grid_search_numiters=5.npz", ess=best_ess, beta=best_beta, nlls=best_nlls)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='save meta-learners choices for a given task within a paradigm')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--paradigm', required=True, type=str, default='categorisation')
    parser.add_argument('--task-name', type=str,
                        required=True, help='task name')
    parser.add_argument('--exp-id', type=int, default=1, help='experiment id')
    parser.add_argument('--model-name', type=str,
                        required=False, help='model name')
    parser.add_argument('--method', type=str, default='soft_sigmoid',
                        help='method for computing model choice probabilities')
    parser.add_argument('--epsilon', type=float, default=0.,
                        help='epsilon for grid_search')
    parser.add_argument('--ess', type=float, default=0.0,
                        help='regularisation strength for grid_search')
    parser.add_argument('--num-iters', type=int, default=5)
    parser.add_argument('--paired', action='store_true',
                        default=False, help='paired')
    parser.add_argument('--optimizer', type=str, default='de', help='optimizer')
    parser.add_argument('--use-base-model-name', action='store_true',
                        default=False, help='use filename')
    parser.add_argument('--scale', type=float, default=1,
                        help='scale for the job array')
    parser.add_argument('--offset', type=float, default=0,
                        help='offset for the job array')
    parser.add_argument('--find-best-model', action='store_true',
                        default=False, help='find best model')

    args = parser.parse_args()
    PARADIGM_PATH = f"{SYS_PATH}/{args.paradigm}"
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.find_best_model:
        find_best_model_gs(args)
        sys.exit()
    else:
        args.ess = args.ess*args.scale + args.offset
        args.model_name = re.sub(r'ess\d+\.\d+', f'ess{args.ess}', args.model_name) if args.use_base_model_name else args.model_name
        assert args.method in ['soft_sigmoid', 'bounded_soft_sigmoid', 'bounded'], 'method not implemented'
        assert args.optimizer in ['de', 'grid_search'], 'optimizer not implemented'
        if args.optimizer == 'de':
            pr2s, nlls, accs, parameters = optimize(args)
        elif args.optimizer == 'grid_search':
            pr2s, nlls, accs, parameters = grid_search(args)

        # save list of results
        num_hidden, num_layers, d_model, num_head, loss_fn, _, source, condition, _ = parse_model_path(args.model_name, {}, return_data_info=True)
        save_path = f"{args.paradigm}/data/model_comparison/task={args.task_name}_experiment={args.exp_id}_source={source}_condition={condition}_ess={str(round(float(args.ess), 4))}_loss={loss_fn}_paired={args.paired}_method={args.method}_optimizer={args.optimizer}_numiters={args.num_iters}.npz"
        np.savez(save_path, betas=parameters, nlls=nlls, pr2s=pr2s, accs=accs)
