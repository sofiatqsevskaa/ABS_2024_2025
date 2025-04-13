import gymnasium as gym
import numpy as np
from q_learning import get_best_action, random_q_table, calculate_new_q_value


def get_discrete_state(state, low_value, window_size):
    new_state = (state - low_value) / window_size
    return tuple(new_state.astype(int))


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode=None)

    observation_space_size = [3, 3]
    observation_space_low_value = env.observation_space.low
    observation_space_high_value = env.observation_space.high
    observation_window_size = (
        observation_space_high_value - observation_space_low_value) / observation_space_size

    num_actions = env.action_space.n
    q_table = random_q_table(-1, 0, observation_space_size + [num_actions])

    learning_rates = [0.1, 0.01]
    discount_factors = [0.5, 0.9]
    num_episodes = [50, 100]

    sum_rewards = 0
    sum_steps = 0

    disc = 1
    lear = 1
    num = 1

    for episode in range(num_episodes[num]):
        state, _ = env.reset()
        discrete_state = get_discrete_state(
            state, observation_space_low_value, observation_window_size)
        while True:
            action = get_best_action(q_table, discrete_state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_discrete_state = get_discrete_state(
                next_state, observation_space_low_value, observation_window_size)

            new_q = calculate_new_q_value(
                q_table,
                old_state=discrete_state,
                new_state=next_discrete_state,
                action=action,
                reward=reward,
                lr=learning_rates[lear],
                discount_factor=discount_factors[disc]
            )

            q_table[discrete_state + (action,)] = new_q

            sum_rewards += reward
            sum_steps += 1

            discrete_state = next_discrete_state

            if done:
                break

    average_reward = sum_rewards / num_episodes[num]
    average_steps = sum_steps / num_episodes[num]

    print(f"Average reward during training: {average_reward}")
    print(f"Average steps during training: {average_steps}")

    sum_rewards = 0
    sum_steps = 0

    for episode in range(num_episodes[num]):
        state, _ = env.reset()
        discrete_state = get_discrete_state(
            state, observation_space_low_value, observation_window_size)
        while True:
            action = get_best_action(q_table, discrete_state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            sum_rewards += reward
            sum_steps += 1

            discrete_state = get_discrete_state(
                next_state, observation_space_low_value, observation_window_size)

            if done:
                break

    average_reward = sum_rewards / num_episodes[num]
    average_steps = sum_steps / num_episodes[num]

    print(f"Average reward during testing: {average_reward}")
    print(f"Average steps during testing: {average_steps}")

    # did not perform well because of the limited number of episodes and the limited number of steps
    env.close()
