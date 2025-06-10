import random
import numpy as np
from pettingzoo.classic import tictactoe_v3
from collections import defaultdict


def get_action(obs, q_table, epsilon, random_agent=False):
    if random_agent:
        return random.choice(np.flatnonzero(obs['action_mask']))
    state_key = tuple(obs['observation'].flatten())
    if random.random() < epsilon or state_key not in q_table:
        return random.choice(np.flatnonzero(obs['action_mask']))
    q_values = q_table[state_key]
    mask = obs['action_mask']
    masked_q = [q if mask[i] else -np.inf for i, q in enumerate(q_values)]
    return int(np.argmax(masked_q))


def update_q_table(q_table, state, action, reward, next_state, alpha, gamma, next_action_mask):
    state_key = tuple(state.flatten())
    next_key = tuple(next_state.flatten())
    if next_key not in q_table:
        q_table[next_key] = np.zeros(9)
    max_q_next = max([q_table[next_key][a] for a in np.flatnonzero(
        next_action_mask)]) if np.any(next_action_mask) else 0
    q_table[state_key][action] += alpha * (
        reward + gamma * max_q_next - q_table[state_key][action]
    )


if __name__ == '__main__':
    env = tictactoe_v3.env()
    env.reset()

    q_table_agent = defaultdict(lambda: np.zeros(9))

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    num_episodes = 50000

    for ep in range(num_episodes):
        env.reset()
        done = False
        prev_obs = None
        prev_action = None

        while not done:
            agent = env.agent_selection
            state, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            if not done:
                obs = state['observation']
                action_mask = state['action_mask']

                if agent == 'player_0':
                    action = get_action(
                        state, q_table_agent, epsilon, random_agent=False)
                else:
                    action = get_action(
                        state, q_table_agent, epsilon, random_agent=True)

                if prev_obs is not None and agent == 'player_0':
                    update_q_table(
                        q_table_agent,
                        prev_obs, prev_action, reward,
                        obs, alpha, gamma, action_mask
                    )

                if agent == 'player_0':
                    prev_obs = obs
                    prev_action = action

                env.step(action)

        if prev_obs is not None:
            reward = env.rewards['player_0']
            update_q_table(
                q_table_agent, prev_obs, prev_action, reward,
                np.zeros((3, 3, 3)), alpha, gamma, np.ones(9)
            )

        if ep % 500 == 0:
            print(f'train ep {ep}')

    total_reward = 0
    wins = {'player_0': 0, 'player_1': 0, 'draw': 0}

    for game_i in range(50):
        env.reset()
        done = False
        while not done:
            agent = env.agent_selection
            state, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            if not done:
                if agent == 'player_0':
                    action = get_action(state, q_table_agent,
                                        0, random_agent=False)
                else:
                    action = get_action(state, None, 0, random_agent=True)
                env.step(action)

        r = env.rewards.get('player_0', 0)
        total_reward += r

        if r > 0:
            wins['player_0'] += 1
        elif r < 0:
            wins['player_1'] += 1
        else:
            wins['draw'] += 1

        print(f'game {game_i}')

    print('avg reward player_0:', total_reward, '/50')
    print('avg reward player_1:', -total_reward, '/50')
    print('wins:', wins)


# with 5000 training episodes:

# avg reward player_0: -12 /50
# avg reward player_1: 12 /50
# wins: {'player_0': 16, 'player_1': 28, 'draw': 6}

# despite player_0 learning, results show player_1 still wins more often
# draws are low, indicating a relatively decisive game outcome distribution
# may need more training episodes or better exploration strategy to improve player_0 performance


# with 50000 training episodes:

# avg reward player_0: 0 /50
# avg reward player_1: 0 /50
# wins: {'player_0': 0, 'player_1': 0, 'draw': 50}

# after increasing training episodes to 50000, player_0 is drawing player_1 every time
