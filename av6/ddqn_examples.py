import gymnasium as gym
import torch.nn as nn
import torch
from deep_q_learning_torch import DDQN
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

    agent = DDQN(state_shape, num_actions, model, target_model,
                 learning_rate=0.001, discount_factor=0.99)

    num_episodes = 1000
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

    # increasing the number of episodes was crucial in this approach
    # this is because the mountaincar environment has sparse rewards
    # the agent receives a constant negative reward until it reaches the goal

    # ddqn helps reduce overestimation bias but still needs many episodes to learn in sparse reward settings
    # more episodes mean more chances for the agent to reach the goal by chance
    # this allows the replay buffer to store useful experiences
    # over time the agent learns to associate state action pairs with better returns
    # with fewer episodes the agent cannot explore enough and keeps failing to climb the hill by doing the same action

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
