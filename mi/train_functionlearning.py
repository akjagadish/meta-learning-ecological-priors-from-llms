import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
import wandb
from envs import FunctionlearningTask, SyntheticFunctionlearningTask
from model import TransformerDecoderRegression
import argparse
from tqdm import tqdm
from evaluate import evaluate_regression
from model_utils import annealed_lambda
import schedulefree

def run(env_name, restart_training, restart_episode_id, num_episodes, ess, ess_init, annealing_fraction, synthetic, num_dims, max_steps, sample_to_match_max_steps, dynamic_scaling, noise, shuffle, shuffle_features, print_every, save_every, num_hidden, num_layers, d_model, num_head, loss, save_dir, device, lr, regularize, optim, batch_size=64):

    writer = SummaryWriter('runs/' + save_dir)
    if synthetic:
       env = SyntheticFunctionlearningTask(num_dims=num_dims, max_steps=max_steps,dynamic_scaling=dynamic_scaling, batch_size=batch_size, noise=noise, device=device).to(device)
    else:
        env = FunctionlearningTask(data=env_name, num_dims=num_dims, max_steps=max_steps, dynamic_scaling=dynamic_scaling, sample_to_match_max_steps=sample_to_match_max_steps,
                                   batch_size=batch_size, noise=noise, shuffle_trials=shuffle, shuffle_features=shuffle_features, device=device).to(device)

    # setup model
    model = TransformerDecoderRegression(num_input=env.num_dims, num_output=env.num_choices, num_hidden=num_hidden,
                                             num_layers=num_layers, d_model=d_model, num_head=num_head, max_steps=max_steps, loss=loss, device=device).to(device)
    start_id = 0
    
    if restart_training and os.path.exists(save_dir):
        #t, model = torch.load(save_dir)
        t, state_dict, _, _, _ = torch.load(save_dir)
        model.load_state_dict(state_dict)
        model = model.to(device)
        print(f'Loaded model from {save_dir}')
        start_id = t if restart_episode_id == 0 else restart_episode_id
    # setup optimizer
    if optim == 'schedulefree':
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=ess_init)
    elif optim == 'adamw':
            ess_init = ess_init or ess
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=ess_init)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=start_id, T_max=num_episodes)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []  # keep track of losses
    accuracy = []  # keep track of accuracies

    # train for num_episodes
    for t in tqdm(range(start_id, int(num_episodes))):
        if optim == 'schedulefree':
            optimizer.train()
        elif optim != 'schedulefree' and t > 0:
            scheduler.step()
        model.train()
        # sample batch
        packed_inputs, sequence_lengths, targets = env.sample_batch()
        with torch.no_grad():
            norm = torch.norm(torch.cat([p.flatten() for p in model.parameters() if p is not None]), 2).item()

        loss = model.compute_loss(packed_inputs, targets, sequence_lengths)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        # Calculate and log gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        wandb.log({"gradient_norm": total_norm})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        if annealing_fraction > 0:
          for param_group in optimizer.param_groups:
                ess_t = annealed_lambda(t+1, num_episodes, ess_init, ess, annealing_fraction)
                param_group['weight_decay'] = ess_t
                wandb.log({"annealing lambda": ess_t})
        # logging
        losses.append(loss.item())
        if (not t % print_every):
            wandb.log({"loss": loss, "l2 norm": norm, "episode": t})

        if (not t % save_every):
            torch.save([t, model.state_dict(), optimizer.state_dict(), ess], save_dir)
            experiment = 'synthetic' if synthetic else 'llm_generated'
            acc = evaluate_regression(env_name=env_name, model_path=save_dir, experiment=experiment,
                                      env=None, model=model, mode='val', loss='mse', shuffle_trials=shuffle, max_steps=max_steps, 
                                      num_dims=num_dims, optimizer=optimizer, optim=optim, device=device)
            accuracy.append(acc)
            wandb.log({"Val. Acc.": acc, "episode": t})

    return losses, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='meta-learning for functionlearning')
    parser.add_argument('--num-episodes', type=int, default=1e6,
                        help='number of trajectories for training')
    parser.add_argument('--job-array', action='store_true',
                        default=False, help='job array')
    parser.add_argument('--ess', type=float, default=None,
                         help='weight for the nll loss term in the ELBO')
    parser.add_argument('--ess-init', type=float, default=None,
                            help='initial weight for the nll loss term in the ELBO')
    parser.add_argument('--annealing-fraction', type=float, default=0,
                        help='fraction of the training time for annealing')
    parser.add_argument('--num-dims', type=int, default=3,
                        help='number of dimensions')
    parser.add_argument('--max-steps', type=int, default=8,
                        help='number of data points per task')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--print-every', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save-every', type=int,
                        default=100, help='how often to save')
    parser.add_argument('--runs', type=int, default=1,
                        help='total number of runs')
    parser.add_argument('--first-run-id', type=int,
                        default=0, help='id of the first run')
    parser.add_argument('--num_hidden', type=int,
                        default=128, help='number of hidden units')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--num_layers', type=int,
                        default=1, help='number of layers')
    parser.add_argument('--d_model', type=int, default=256,
                        help='dimension of the model')
    parser.add_argument('--num_head', type=int,
                        default=4, help='number of heads')
    parser.add_argument('--loss', default='nll', help='loss function')
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--env-name', required=True, help='name of the environment')
    parser.add_argument('--env-type', default=None, 
                        help='name of the environment when name of the dataset does not explain the model fully')
    parser.add_argument('--env-dir', help='name of the environment', required=False)
    parser.add_argument('--save-dir', help='directory to save models', required=True)
    parser.add_argument('--test', action='store_true',
                        default=False, help='test runs')
    parser.add_argument('--synthetic', action='store_true',
                        default=False, help='train models on synthetic data')
    parser.add_argument('--noise', type=float, default=0., help='noise level')
    parser.add_argument('--shuffle', action='store_true',
                        default=False, help='shuffle trials')
    parser.add_argument('--shuffle-features', action='store_true',
                        default=False, help='shuffle features')
    parser.add_argument('--model-name', default='transformer',
                        help='name of the model')
    parser.add_argument('--sample-to-match-max-steps', action='store_true',
                        default=False, help='sample to match max steps')
    parser.add_argument('--dynamic-scaling', action='store_true',
                        default=False, help='dynamic scaling')
    parser.add_argument('--restart-training', action='store_true',
                        default=False, help='restart training')
    parser.add_argument('--restart-episode-id', type=int,
                        default=0, help='restart episode id')
    parser.add_argument('--scale', type=float, default=1,
                        help='scale for the job array')
    parser.add_argument('--offset', type=float, default=0,
                        help='offset for the job array')
    parser.add_argument('--regularize', default='all',
                        help='regularize the specific model parameters or all')
    parser.add_argument('--optimizer', default='adamw',
                        help='optimizer')
    # parser.add_argument('--eval', default='categorisation', help='what to eval your meta-learner on')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    env = f'{args.env_name}_dim{args.num_dims}' if args.synthetic else args.env_name if args.env_type is None else args.env_type
    args.ess = args.ess * args.scale + args.offset if args.job_array else args.ess
    # wandb configuration
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="BERMI - functionlearning",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": args.model_name,
        "dataset": env,
        "epochs": args.num_episodes,
        "num_hidden": args.num_hidden,
        "num_layers": args.num_layers,
        "d_model": args.d_model,
        "num_head": args.num_head,
        "noise": args.noise,
        "shuffle": args.shuffle,
        "loss": args.loss,
        "ess": args.ess,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "annealing_fraction": args.annealing_fraction,
        "regularize": args.regularize,
        "dynamic_scaling": args.dynamic_scaling,
        }
    )

    for i in range(args.runs):

        save_dir = f'{args.save_dir}env={env}_model={args.model_name}_num_episodes{str(args.num_episodes)}_num_hidden={str(args.num_hidden)}_lr{str(args.lr)}_num_layers={str(args.num_layers)}_d_model={str(args.d_model)}_num_head={str(args.num_head)}_noise{str(args.noise)}_shuffle{str(args.shuffle)}_ess{str(round(float(args.ess), 4))}_run={str(args.first_run_id + i)}.pt'
        save_dir = save_dir.replace(
                '.pt', f'_synthetic.pt') if args.synthetic else save_dir
        save_dir = save_dir.replace(
            '.pt', '_test.pt') if args.test else save_dir
        env_name = f'/{args.env_dir}/{args.env_name}.csv' if not args.synthetic else None
        save_dir = save_dir.replace('.pt', f'_reg{args.regularize}.pt') if args.ess >= 0 else save_dir
        save_dir = save_dir.replace('.pt', f'_essinit{str(args.ess_init)}_annealed.pt') if args.annealing_fraction > 0 else save_dir
        save_dir = save_dir.replace('.pt', f'_schedulefree.pt') if args.optimizer == 'schedulefree' else save_dir
        save_dir = save_dir.replace('.pt', f'_dynamic_scaling.pt') if args.dynamic_scaling else save_dir
        wandb.run.name = save_dir[len(args.save_dir):]
        wandb.run.save()        
        
        run(env_name, args.restart_training, args.restart_episode_id, args.num_episodes, args.ess, args.ess_init, args.annealing_fraction, args.synthetic, args.num_dims, args.max_steps, args.sample_to_match_max_steps, args.dynamic_scaling,
            args.noise, args.shuffle, args.shuffle_features, args.print_every, args.save_every, args.num_hidden, args.num_layers, args.d_model, args.num_head, args.loss, save_dir, device, args.lr, args.regularize, args.optimizer, args.batch_size)
