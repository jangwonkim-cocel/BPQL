from collections import deque
import gym
import numpy as np


class DelayedEnv(gym.Wrapper):
    def __init__(self, env, seed, obs_delayed_steps, act_delayed_steps):
        super(DelayedEnv, self).__init__(env)
        assert obs_delayed_steps + act_delayed_steps > 0
        self.env.action_space.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._max_episode_steps = self.env._max_episode_steps

        self.obs_buffer = deque(maxlen=obs_delayed_steps)
        self.reward_buffer = deque(maxlen=obs_delayed_steps)
        self.done_buffer = deque(maxlen=obs_delayed_steps)

        self.action_buffer = deque(maxlen=act_delayed_steps)

        self.obs_delayed_steps = obs_delayed_steps
        self.act_delayed_steps = act_delayed_steps

    def reset(self):
        for _ in range(self.act_delayed_steps):
            self.action_buffer.append(np.zeros_like(self.env.action_space.sample()))

        init_state, _ = self.env.reset()
        for _ in range(self.obs_delayed_steps):
            self.obs_buffer.append(init_state)
            self.reward_buffer.append(0)
            self.done_buffer.append(False)
        return init_state

    def step(self, action):
        if self.act_delayed_steps > 0:
            delayed_action = self.action_buffer.popleft()
            self.action_buffer.append(action)
        else:
            delayed_action = action

        current_obs, current_reward, current_terminated, current_truncated, _ = self.env.step(delayed_action)
        current_done = current_terminated or current_truncated

        if self.obs_delayed_steps > 0:
            delayed_obs = self.obs_buffer.popleft()
            delayed_reward = self.reward_buffer.popleft()
            delayed_done = self.done_buffer.popleft()

            self.obs_buffer.append(current_obs)
            self.reward_buffer.append(current_reward)
            self.done_buffer.append(current_done)
        else:
            delayed_obs = current_obs
            delayed_reward = current_reward
            delayed_done = current_done

        return delayed_obs, delayed_reward, delayed_done, {'current_obs': current_obs, 'current_reward': current_reward,
                                                           'current_done': current_done}




