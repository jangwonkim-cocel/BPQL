import numpy as np
from collections import deque


class TemporaryBuffer:
    def __init__(self, delayed_steps):
        self.d = delayed_steps
        self.states = deque(maxlen=delayed_steps + 2)
        self.actions = deque(maxlen=2 * delayed_steps + 1)

    def clear(self):
        self.states.clear()
        self.actions.clear()

    def get_augmented_state(self, last_observed_state, first_action_idx):
        aug_state = np.concatenate([last_observed_state, self.actions[first_action_idx]])
        for i in range(first_action_idx + 1, first_action_idx + self.d):
            aug_state = np.concatenate([aug_state, self.actions[i]])
        return aug_state

    def get_tuple(self):
        assert len(self.states) == self.d + 2 and len(self.actions) == 2 * self.d + 1

        aug_s = self.get_augmented_state(self.states[0], 0)
        s = self.states[-2]
        a = self.actions[self.d]

        next_aug_s = self.get_augmented_state(self.states[1], 1)
        next_s = self.states[-1]

        self.states.popleft()
        self.actions.popleft()
        return aug_s, s, a, next_aug_s, next_s

