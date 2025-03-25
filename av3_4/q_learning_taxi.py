import gymnasium as gym
from q_learning import get_random_action, get_best_action, get_action, \
    random_q_table, calculate_new_q_value

if __name__ == '__main__':
    env = gym.make('Taxi-v3', 10)

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    q_table = random_q_table(-1, 0, (num_states, num_actions))

    learning_rates = [0.1, 0.01]
    discount_factors = [0.5, 0.9]
    epsilon = 0.5
    num_episodes = [50, 100]
    num_steps_per_episode = 5

    sum_rewards = 0
    average_reward = 0

    sum_steps = 0
    average_steps = 0

    disc = 1
    lear = 1
    num = 1
    for episode in range(num_episodes[num]):
        state, _ = env.reset()
        while(True):
            #action = get_random_action(env)
            action = get_best_action(q_table, state)
            #action = get_action(env, q_table, state, epsilon)

            new_state, reward, terminated, _, _ = env.step(action)

            new_q = calculate_new_q_value(q_table,
                                          state, new_state,
                                          action, reward,
                                          learning_rates[lear], discount_factors[disc])

            q_table[state, action] = new_q

            sum_rewards += reward

            sum_steps += 1

            state = new_state

            if terminated:
                break

    average_reward = sum_rewards/num_episodes[num]
    print(f"Average reward is {average_reward}")
    average_steps = sum_steps/num_episodes[num]
    print(f"Average number of steps is {average_steps}")

    env = gym.make('Taxi-v3')

    sum_rewards = 0
    average_reward = 0

    sum_steps = 0
    average_steps = 0

    for episode in range(num_episodes[num]):
        state, _ = env.reset()
        while(True):
            #action = get_random_action(env)
            action = get_best_action(q_table, state)
            #action = get_action(env, q_table, state, epsilon)

            new_state, reward, terminated, _, _ = env.step(action)

            sum_rewards += reward

            sum_steps += 1

            state = new_state

            if terminated:
                break

    average_reward = sum_rewards/num_episodes[num]
    print(f"Average reward is {average_reward}")
    average_steps = sum_steps/num_episodes[num]
    print(f"Average number of steps is {average_steps}")

    #By choosing a random action the agent doesn't perform well - the average reward is 0 no matter the learning rate or discount factor
    #Best action showed best results

