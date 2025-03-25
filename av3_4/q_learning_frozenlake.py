import gymnasium as gym
from q_learning import get_random_action, get_best_action, get_action, \
    random_q_table, calculate_new_q_value

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='human')

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = random_q_table(-1, 0, (num_states, num_actions))

    learning_rate = 0.001
    discount_factor = 0.1
    epsilon = 0.5
    num_episodes = 10
    num_steps_per_episode = 5

    for episode in range(num_episodes):
        state, _ = env.reset()
        for step in range(num_steps_per_episode):
            action = get_random_action(env)  # 1
            action = get_best_action(q_table, state)  # 2
            action = get_action(env, q_table, state, epsilon)

            new_state, reward, terminated, _, _ = env.step(action)

            new_q = calculate_new_q_value(q_table,
                                          state, new_state,
                                          action, reward,
                                          learning_rate, discount_factor)

            q_table[state, action] = new_q

            state = new_state

            print()
