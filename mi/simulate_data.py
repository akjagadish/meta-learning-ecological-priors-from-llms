from envs import SyntheticDecisionmakingTask, SyntheticFunctionlearningTask
import argparse
import torch


def simulate_synthetic_tasks(paradigm, num_tasks, ranked, direction, num_dims, max_steps, noise, device, batch_size):

    if paradigm == 'decisionmaking':
        env = SyntheticDecisionmakingTask(num_dims=num_dims, max_steps=max_steps, synthesize_tasks=True, ranking=ranked, direction=direction,
                                        batch_size=batch_size, noise=noise, device=device).to(device)
    elif paradigm == 'functionlearning':
        env = SyntheticFunctionlearningTask(num_dims=num_dims, max_steps=max_steps, batch_size=batch_size, 
                                            noise=noise, device=device).to(device)

    env.save_synthetic_data(num_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='simulate and save synthetic data')
    parser.add_argument('--paradigm', type=str, default='decisionmaking',
                        help='paradigm to simulate data for decisionmaking or functionlearning')
    parser.add_argument('--num-tasks', type=int,
                        default=10000, help='number of tasks')
    parser.add_argument('--num-dims', type=int, default=3,
                        help='number of dimensions')
    parser.add_argument('--max-steps', type=int, default=8,
                        help='number of data points per task')
    parser.add_argument('--batch-size', type=int,
                        default=100, help='batch size')
    parser.add_argument('--noise', type=float, default=0.0, help='noise level')
    parser.add_argument('--ranked', action='store_true',
                        default=False, help='simulate ranked features based synthetic data')
    parser.add_argument('--direction', action='store_true',
                        default=False, help='simulate positive direction based synthetic data')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    simulate_synthetic_tasks(args.paradigm,args.num_tasks, args.ranked, args.direction, args.num_dims,
                                            args.max_steps, args.noise, device, args.batch_size)
