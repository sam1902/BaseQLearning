import numpy as np
from QLearner import QLearner


class AdvancedQLearner(QLearner):
    def __init__(self, reward_matrix: np.ndarray, discount_factor: float = 0.8, learning_rate: float = 0.7):
        super().__init__(reward_matrix, discount_factor)
        self.learning_rate = learning_rate

    def update_q_matrix(self, state: int, action: int) -> None:
        """
        Updates the Q matrix for a given state after exploring an action
        :param state: starting state
        :param action: starting action to perform
        """
        states_row = self.q_matrix[action, :]
        best_state = np.where(states_row == np.max(states_row))[0]
        best_state = int(best_state) if best_state.shape[0] == 1 else self.choose_action_random(best_state)

        best_value = self.q_matrix[action, best_state]
        # Old value
        self.q_matrix[state, action] = (1 - self.learning_rate) \
                                       * self.q_matrix[state, action]
        # Exploration value
        self.q_matrix[state, action] = self.learning_rate \
                                       * (self.reward_matrix[state, action] + self.discount_factor * best_value)

