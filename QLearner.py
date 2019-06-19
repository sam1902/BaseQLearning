import numpy as np
from tqdm import tqdm
from typing import List


class QLearner:
    def __init__(self, reward_matrix: np.ndarray, discount_factor: float = 0.8):
        """
        Initialise the Q-learning trainer
        :param reward_matrix: Reward matrix corresponding to the problem, 1 for possible, 100 for goal, 0 for impossible
        :param discount_factor: Discount factor, whether to value early rewards more than future rewards, 0.8 means it'll value
        80% more immediate reward than subsequent rewards.
        """
        self.discount_factor = discount_factor
        # Row : current state, Col : next state,   Values : transition's reward
        self.reward_matrix = reward_matrix
        # Row : current state, Col : next actions, Values : action's fitness
        self.q_matrix = np.zeros_like(reward_matrix)

    def possible_actions(self, state: int) -> List[int]:
        """
        Looks up in the reward matrix the list of possible actions
        :param state: current state
        :return: list of possible actions (int)
        """
        current_row = self.reward_matrix[state, :]
        possible_actions = np.where(current_row > 0)[0]
        return possible_actions

    @staticmethod
    def choose_action_random(actions: List[int]) -> int:
        """
        Pick an actions at random
        :param actions: list of possible actions from which to pick from
        :return: an action
        """
        return int(np.random.choice(actions))

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
        self.q_matrix[state, action] = self.reward_matrix[state, action] + self.discount_factor * best_value

    def train(self, n: int = 70) -> None:
        """
        Trains the Q learner with the given parameters
        :param n: number of possible starting states to explore
        """
        for _ in tqdm(range(n)):
            state = np.random.randint(0, self.reward_matrix.shape[0])
            actions = self.possible_actions(state)
            action = self.choose_action_random(actions)
            self.update_q_matrix(state, action)

    def normalise(self) -> None:
        """
        Normalises linearly the q learning matrix for values to be between 0 and 1.
        """
        if np.max(self.q_matrix) > 0:
            self.q_matrix = self.q_matrix / np.max(self.q_matrix)

    def predict(self, state: int) -> int:
        """
        Predicts the next best state to go to from the current state using the (trained) Q matrix.
        :param state: current state from which to predict from
        :return: the best state to go to next.
        """
        return int(np.argmax(self.q_matrix[state, :]))
