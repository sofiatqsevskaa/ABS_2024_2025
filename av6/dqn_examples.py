import gymnasium as gym
import torch.nn as nn
from deep_q_learning_torch import DQN


def build_model(state_space_shape, num_actions):
    return ...


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    env.reset()
    env.render()

    num_episodes = 5000
    num_steps_per_episode = 100

    ...

    for episode in range(num_episodes):
        ...
        for step in range(num_steps_per_episode):
            ...
            action = agent.get_action(...)
            next_state, reward, done, _, _ = env.step(action)

            agent.update_memory(state, action, reward, next_state, done)

        agent.train()

        if episode % 5:
            agent.update_target_model()

    print()













