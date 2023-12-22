import torch
import numpy as np


class ReplayMemory:
    def __init__(self, delayed_steps, state_dim, action_dim, device, capacity=1e6):
        self.device = device
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0

        self.augmented_state_buffer = np.empty(shape=(self.capacity, state_dim + action_dim * delayed_steps), dtype=np.float32)
        self.action_buffer = np.empty(shape=(self.capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.next_augmented_state_buffer = np.empty(shape=(self.capacity, state_dim + action_dim * delayed_steps), dtype=np.float32)
        self.done_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)
        self.next_state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)

    def push(self, augmented_state, state, action, reward, next_augmented_state, next_state, done):
        self.size = min(self.size + 1, self.capacity)

        self.augmented_state_buffer[self.position] = augmented_state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_augmented_state_buffer[self.position] = next_augmented_state
        self.done_buffer[self.position] = done
        self.state_buffer[self.position] = state
        self.next_state_buffer[self.position] = next_state

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        augmented_states = torch.FloatTensor(self.augmented_state_buffer[idxs]).to(self.device)
        actions = torch.FloatTensor(self.action_buffer[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.reward_buffer[idxs]).to(self.device)
        next_augmented_states = torch.FloatTensor(self.next_augmented_state_buffer[idxs]).to(self.device)
        dones = torch.FloatTensor(self.done_buffer[idxs]).to(self.device)
        states = torch.FloatTensor(self.state_buffer[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_state_buffer[idxs]).to(self.device)

        return augmented_states, actions, rewards, next_augmented_states, dones, states, next_states
