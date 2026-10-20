
import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent implemented from scratch."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=42,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros(
            (state_size, action_size), dtype=np.float64
        )
        self.rng = np.random.default_rng(seed)

    def select_action(self, state):
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))

        # Exploitation: select the best known action
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, terminated):
        # No future reward is available after a terminal state
        future_value = (
            0.0
            if terminated
            else np.max(self.q_table[next_state])
        )

        old_value = self.q_table[state, action]

        target = reward + self.gamma * future_value

        self.q_table[state, action] = old_value + self.alpha * (
            target - old_value
        )

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save(self, path="q_table.npy"):
        np.save(path, self.q_table)

    def load(self, path="q_table.npy"):
        self.q_table = np.load(path)
