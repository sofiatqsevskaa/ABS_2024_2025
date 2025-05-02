import gymnasium as gym
import torch.nn as nn
import torch
from deep_q_learning_torch import DQN
import numpy as np


def build_model(state_space_shape, num_actions):
    return nn.Sequential(
        nn.Linear(state_space_shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions)
    )


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='human')
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = build_model(state_shape, num_actions)
    target_model = build_model(state_shape, num_actions)

    agent = DQN(state_shape, num_actions, model, target_model,
                learning_rate=0.001, discount_factor=0.99)

    num_episodes = 100
    num_steps_per_episode = 100
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    epsilon = epsilon_start
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(num_steps_per_episode):
            action = agent.get_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -1
            agent.update_memory(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(
            f'Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}')

        if episode % 5 == 0:
            agent.update_target_model()

    agent.save('cartpole', num_episodes)
    print(
        f'Average reward over 100 episodes: {np.mean(rewards[-100:]):.2f}')

    test_episodes = 50
    total_rewards = []
    for _ in range(test_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state, epsilon=0)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    avg_test_reward = np.mean(total_rewards)
    print(
        f'Average reward over {test_episodes} test episodes: {avg_test_reward:.2f}')

    # The results are better than the Q-learning version, because the DQN is able to learn more complex policies.
    # The Q-learning version is limited by the size of the Q-table, while the DQN can learn from a larger state space and can generalize better.
    # The results were better the more episodes were run, as the DQN was able to learn more complex policies.

    # Plotting the rewards

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
