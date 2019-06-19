import numpy as np
from AdvancedQLearner import AdvancedQLearner


def main():
    reward_matrix = np.array([[0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 1],
                              [0, 0, 100, 1, 0, 0],
                              [0, 1, 1, 0, 1, 0],
                              [1, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 0]])
    ql = AdvancedQLearner(reward_matrix, discount_factor=0.8, learning_rate=0.5)

    ql.train(n=70)
    ql.normalise()

    state = 5  # a.k.a. state F
    prev_state = -1
    count = 0
    while prev_state != state and count < 100:
        prev_state = state
        print("Current state {}".format(state), end=' ')
        state = ql.predict(state)
        print("==> Next state {}".format(state))
        count += 1

    if reward_matrix[state, state] == reward_matrix.max():
        print('Success')
    else:
        print('Failure')


if __name__ == '__main__':
    main()
