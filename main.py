import argparse
import torch
from bpql import BPQLAgent
from trainer import Trainer
from utils import set_seed, make_delayed_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', default='HalfCheetah-v3', type=str)

    parser.add_argument('--obs-delayed-steps', default=4, type=int)  # Delayed timesteps (Observation, Reward)
    parser.add_argument('--act-delayed-steps', default=5, type=int)  # Delayed timesteps (Action)

    parser.add_argument('--random-seed', default=-1, type=int)
    parser.add_argument('--eval_flag', default=True, type=bool)
    parser.add_argument('--eval-freq', default=5000, type=int)
    parser.add_argument('--eval-episode', default=5, type=int)
    parser.add_argument('--automating-temperature', default=True, type=bool)
    parser.add_argument('--temperature', default=0.2, type=float)
    parser.add_argument('--start-step', default=10000, type=int)
    parser.add_argument('--max-step', default=1000000, type=int)
    parser.add_argument('--update_after', default=1000, type=int)
    parser.add_argument('--hidden-dims', default=(256, 256))
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--buffer-size', default=1000000, type=int)
    parser.add_argument('--update-every', default=50, type=int)
    parser.add_argument('--log_std_bound', default=[-20, 2])
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--actor-lr', default=3e-4, type=float)
    parser.add_argument('--critic-lr', default=3e-4, type=float)
    parser.add_argument('--temperature-lr', default=3e-4, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--show-loss', default=False, type=bool)
    args = parser.parse_args()

    # Set Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set Seed
    random_seed = set_seed(args.random_seed)

    # Create Delayed Environment
    env, eval_env = make_delayed_env(args, random_seed, obs_delayed_steps=args.obs_delayed_steps, act_delayed_steps=args.act_delayed_steps)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low[0], env.action_space.high[0]]

    print(f"Environment: {args.env_name}, Obs. Delayed Steps: {args.obs_delayed_steps}, Act. Delayed Steps: {args.act_delayed_steps}, Random Seed: {args.random_seed}", "\n")

    # Create Agent
    agent = BPQLAgent(args, state_dim, action_dim, action_bound, env.action_space, device)

    # Create Trainer & Train
    trainer = Trainer(env, eval_env, agent, args)
    trainer.train()
