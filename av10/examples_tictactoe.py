import random
import numpy as np
from pettingzoo.classic import tictactoe_v3


# https://pettingzoo.farama.org/environments/classic/tictactoe/


def policy(obs):
    return random.choice(np.flatnonzero(obs['action_mask']))


if __name__ == '__main__':
    env = tictactoe_v3.env(render_mode='human')

    env.reset()

    env.render()

    q_table_agent_1 = ...
    q_table_agent_2 = ...

    num_episodes = 5

    for _ in range(num_episodes):
        env.reset()
        for agent in env.agent_iter():
            state, reward, terminated, truncated, info = env.last()
            observation = state['observation']
            action_mask = state['action_mask']
            if terminated or truncated:
                env.reset()
                break
            else:
                action = policy(state)
                env.step(action)
