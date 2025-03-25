import gymnasium as gym
import numpy as np
from q_learning import random_q_table


def get_discrete_state(state, low_value, window_size):
    new_state = (state - low_value) / window_size
    return tuple(new_state.astype(np.int))


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='human')

    num_actions = env.action_space.n

    observation_space_size = [3, 3]
    observation_space_low_value = env.observation_space.low
    observation_space_high_value = env.observation_space.high
    observation_window_size = (observation_space_high_value - observation_space_low_value) / observation_space_size

    q_table = random_q_table(-1, 0, (observation_space_size + [num_actions]))

    state, _ = env.reset()

    discrete_state = get_discrete_state(state, observation_space_low_value, observation_window_size)

    env.render()
