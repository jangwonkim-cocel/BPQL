import numpy as np
import random
import torch
import torch.nn as nn
from wrapper import DelayedEnv


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def hard_update(network, target_network):
    with torch.no_grad():
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(param.data)


def soft_update(network, target_network, tau):
    with torch.no_grad():
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    return random_seed


def make_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    eval_env = gym.make(env_name)
    eval_env.seed(random_seed)
    eval_env.action_space.seed(random_seed)

    return env, eval_env


def make_delayed_env(args, random_seed, obs_delayed_steps, act_delayed_steps):
    import gym
    # openai gym
    env_name = args.env_name

    env = gym.make(env_name)
    delayed_env = DelayedEnv(env, seed=random_seed, obs_delayed_steps=obs_delayed_steps, act_delayed_steps=act_delayed_steps)

    eval_env = gym.make(env_name)
    eval_delayed_env = DelayedEnv(eval_env, seed=random_seed, obs_delayed_steps=obs_delayed_steps, act_delayed_steps=act_delayed_steps)

    return delayed_env, eval_delayed_env

def log_to_txt(env_name, random_seed, total_step, result):
    seed = '(' + str(random_seed) + ')'
    f = open('./log/' + env_name + '_seed' + seed + '.txt', 'a')
    log = str(total_step) + ' ' + str(result) + '\n'
    f.write(log)
    f.close()