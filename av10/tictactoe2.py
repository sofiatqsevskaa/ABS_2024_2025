import random
import numpy as np
from pettingzoo.classic import tictactoe_v3
from collections import defaultdict


def get_action(obs, action_mask, q_table, epsilon):
    state_key = tuple(obs.flatten())
    if random.random() < epsilon or state_key not in q_table:
        return random.choice(np.flatnonzero(action_mask))
    q_values = q_table[state_key]
    masked_q = [q if action_mask[i] else -
                np.inf for i, q in enumerate(q_values)]
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

    q_table_agent_1 = defaultdict(lambda: np.zeros(9))
    q_table_agent_2 = defaultdict(lambda: np.zeros(9))

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2
    num_episodes = 5000

    for episode in range(num_episodes):
        env.reset()
        done = False
        prev_obs = {}
        prev_action = {}
        if episode % 500 == 0:
            print(f'train ep {episode}')
        while not done:
            agent = env.agent_selection
            state, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            if not done:
                obs = state['observation']
                action_mask = state['action_mask']
                q_table = q_table_agent_1 if agent == 'player_0' else q_table_agent_2
                action = get_action(obs, action_mask, q_table, epsilon)
                if agent in prev_obs:
                    q_table = q_table_agent_1 if agent == 'player_0' else q_table_agent_2
                    update_q_table(
                        q_table,
                        prev_obs[agent], prev_action[agent], reward,
                        obs, alpha, gamma, action_mask
                    )

                prev_obs[agent] = obs
                prev_action[agent] = action
                env.step(action)
        for agent in prev_obs:
            reward = env.rewards[agent]
            q_table = q_table_agent_1 if agent == 'player_0' else q_table_agent_2
            update_q_table(
                q_table, prev_obs[agent], prev_action[agent], reward,
                np.zeros((3, 3, 3)), alpha, gamma, np.ones(9)
            )

    agents = ['player_0', 'player_1']
    total_rewards = {agent: 0 for agent in agents}
    wins = {'player_0': 0, 'player_1': 0, 'draw': 0}

    for match in range(50):
        env.reset()
        done = False
        if match % 10 == 0:
            print(f'game {match}')
        while not done:
            agent = env.agent_selection
            state, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            if not done:
                obs = state['observation']
                action_mask = state['action_mask']
                q_table = q_table_agent_1 if agent == 'player_0' else q_table_agent_2
                action = get_action(obs, action_mask, q_table, 0)
                env.step(action)

        r0 = env.rewards.get('player_0', 0)
        r1 = env.rewards.get('player_1', 0)
        print(f"Game ended. Rewards: player_0={r0}, player_1={r1}")
        total_rewards['player_0'] += r0
        total_rewards['player_1'] += r1
        if r0 > r1:
            wins['player_0'] += 1
        elif r1 > r0:
            wins['player_1'] += 1
        else:
            wins['draw'] += 1

    print('avg reward player_0:', total_rewards['player_0'], '/ 50')
    print('avg reward player_1:', total_rewards['player_1'], '/ 50')
    print('wins:', wins)


# experiment 1:

# avg reward player_0: 0 / 50
# avg reward player_1: 0 / 50
# wins: {'player_0': 0, 'player_1': 0, 'draw': 50}

# the large number of draws suggests both agents have learned to avoid losing

# since the q functions were trained independently for both agents, player_0 and player_1 likely learned the same policy

# experiment 2:

# avg reward player_0: 0 / 50
# avg reward player_1: 0 / 50
# wins: {'player_0': 0, 'player_1': 0, 'draw': 50}
