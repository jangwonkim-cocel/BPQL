import numpy as np

from utils import log_to_txt


class Trainer:
    def __init__(self, env, eval_env, agent, args):
        self.args = args
        self.agent = agent

        self.delayed_env = env
        self.eval_delayed_env = eval_env

        self.start_step = args.start_step
        self.update_after = args.update_after
        self.max_step = args.max_step
        self.batch_size = args.batch_size
        self.update_every = args.update_every

        self.eval_flag = args.eval_flag
        self.eval_episode = args.eval_episode
        self.eval_freq = args.eval_freq

        self.episode = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_local_step = 0
        self.eval_num = 0
        self.finish_flag = False

        self.total_delayed_steps = args.obs_delayed_steps + self.args.act_delayed_steps

    def train(self):
        # The train process starts here.
        while not self.finish_flag:
            self.episode += 1
            self.local_step = 0

            # Initialize the delayed environment & the temporal buffer
            self.delayed_env.reset()
            self.agent.temporary_buffer.clear()
            done = False

            # Episode starts here.
            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.local_step < self.total_delayed_steps:  # if t < d
                    action = np.zeros_like(self.delayed_env.action_space.sample())  # Select the 'no-op' action
                    _, _, _, _ = self.delayed_env.step(action)

                    self.agent.temporary_buffer.actions.append(action)
                elif self.local_step == self.total_delayed_steps:  # if t == d
                    if self.total_step < self.start_step:
                        action = self.delayed_env.action_space.sample()
                    else:
                        action = np.zeros_like(self.delayed_env.action_space.sample())  # Select the 'no-op' action

                    next_observed_state, _, _, _ = self.delayed_env.step(action)
                    #                s(1)       <-     Env: a(d)
                    self.agent.temporary_buffer.actions.append(action)  # Put a(d) to the temporary buffer
                    self.agent.temporary_buffer.states.append(next_observed_state)  # Put s(1) to the temporary buffer
                else:  # if t > d
                    last_observed_state = self.agent.temporary_buffer.states[-1]
                    first_action_idx = len(self.agent.temporary_buffer.actions) - self.total_delayed_steps

                    # Get the augmented state(t)
                    augmented_state = self.agent.temporary_buffer.get_augmented_state(last_observed_state, first_action_idx)

                    if self.total_step < self.start_step:
                        action = self.delayed_env.action_space.sample()
                    else:
                        action = self.agent.get_action(augmented_state, evaluation=False)
                        # a(t) <- policy: augmented_state(t)
                    next_observed_state, reward, done, info = self.delayed_env.step(action)
                    #          s(t+1-d),  r(t-d)      <-      Env: a(t)
                    true_done = 0.0 if self.local_step == self.delayed_env._max_episode_steps + self.args.obs_delayed_steps else float(done)

                    self.agent.temporary_buffer.actions.append(action)  # Put a(t) to the temporary buffer
                    self.agent.temporary_buffer.states.append(next_observed_state)  # Put s(t+1-d) to the temporary buffer

                    if self.local_step > 2 * self.total_delayed_steps:  # if t > 2d
                        augmented_s, s, a, next_augmented_s, next_s = self.agent.temporary_buffer.get_tuple()
                        #  aug_s(t-d),  s(t-d),  a(t-d),  aug_s(t+1-d),  s(t+1-d)  <- Temporal Buffer
                        self.agent.replay_memory.push(augmented_s, s, a, reward, next_augmented_s, next_s, true_done)
                        #  Store (aug_s(t-d), s(t-d), a(t-d), r(t-d), aug_s(t+1-d), s(t+1-d)) in the replay memory.

                # Update parameters
                if self.agent.replay_memory.size >= self.batch_size and self.total_step >= self.update_after and \
                        self.total_step % self.update_every == 0:
                    total_actor_loss = 0
                    total_critic_loss = 0
                    total_log_alpha_loss = 0
                    for i in range(self.update_every):
                        # Train the policy and the beta Q-network (critic).
                        critic_loss, actor_loss, log_alpha_loss = self.agent.train()
                        total_critic_loss += critic_loss
                        total_actor_loss += actor_loss
                        total_log_alpha_loss += log_alpha_loss

                    # Print the loss.
                    if self.args.show_loss:
                        print("Loss  |  Actor loss {:.3f}  |  Critic loss {:.3f}  |  Log-alpha loss {:.3f}"
                              .format(total_actor_loss / self.update_every, total_critic_loss / self.update_every,
                                      total_log_alpha_loss / self.update_every))

                # Evaluate.
                if self.eval_flag and self.total_step % self.eval_freq == 0:
                    self.evaluate()

                # Raise finish flag.
                if self.total_step == self.max_step:
                    self.finish_flag = True

    def evaluate(self):
        # Evaluate process
        self.eval_num += 1
        reward_list = []

        for epi in range(self.eval_episode):
            episode_reward = 0
            self.eval_delayed_env.reset()
            self.agent.eval_temporary_buffer.clear()
            done = False
            self.eval_local_step = 0

            while not done:
                self.eval_local_step += 1
                if self.eval_local_step < self.total_delayed_steps:
                    action = np.zeros_like(self.delayed_env.action_space.sample())
                    _, _, _, _ = self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)
                elif self.eval_local_step == self.total_delayed_steps:
                    action = np.zeros_like(self.eval_delayed_env.action_space.sample())
                    next_observed_state, _, _, _ = self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)
                    self.agent.eval_temporary_buffer.states.append(next_observed_state)
                else:
                    last_observed_state = self.agent.eval_temporary_buffer.states[-1]
                    first_action_idx = len(self.agent.eval_temporary_buffer.actions) - self.total_delayed_steps
                    augmented_state = self.agent.eval_temporary_buffer.get_augmented_state(last_observed_state,
                                                                                          first_action_idx)
                    action = self.agent.get_action(augmented_state, evaluation=True)
                    next_observed_state, reward, done, _ = self.eval_delayed_env.step(action)
                    self.agent.eval_temporary_buffer.actions.append(action)
                    self.agent.eval_temporary_buffer.states.append(next_observed_state)
                    episode_reward += reward

            reward_list.append(episode_reward)

        log_to_txt(self.args.env_name, self.args.random_seed, self.total_step, sum(reward_list) / len(reward_list))
        print("Eval  |  Total Steps {}  |  Episodes {}  |  Average Reward {:.2f}  |  Max reward {:.2f}  |  "
              "Min reward {:.2f}".format(self.total_step, self.episode, sum(reward_list) / len(reward_list),
                                          max(reward_list), min(reward_list)))










