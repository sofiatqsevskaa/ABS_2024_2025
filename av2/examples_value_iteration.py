import gymnasium as gym
from mdp import value_iteration, policy_iteration
from mdp import *

if __name__ == '__main__':
    env = gym.make('Taxi-v3')

    init_state, _ = env.reset()
    env.render()

    # policy, V = value_iteration(env=env,
    #                 num_actions=env.action_space.n,
    #                 num_states=env.observation_space.n,
    #                 theta = 0.00001,
    #                 discount_factor=0.5)

    # result: 2.2 average iterations, 20 average reward - 50 iter
    # result: 2.18 average iterations, 20 average reward - 100 iter

    # policy, V = value_iteration(env=env,
    #                 num_actions=env.action_space.n,
    #                 num_states=env.observation_space.n,
    #                 theta = 0.00001,
    #                 discount_factor=0.7)

    # result: 2.28 average iterations, 20 average reward - 50 iter
    # result: 2.14 average iterations, 20 average reward - 100 iter

    # policy, V = value_iteration(env=env,
    #                 num_actions=env.action_space.n,
    #                 num_states=env.observation_space.n,
    #                 theta = 0.00001,
    #                 discount_factor=0.9)

    # result: 2.2 average iterations, 20 average reward - 50 iter
    # result: 2.14 average iterations, 20 average reward - 100 iter

    # policy, V = policy_iteration(env=env,
    #                             num_actions=env.action_space.n,
    #                             num_states=env.observation_space.n,
    #                             discount_factor=0.5)

    # result: 2.26 average iterations, 20 average reward - 50 iter
    # result: 2.10 average iterations, 20 average reward - 100 iter

    policy, V = policy_iteration(env=env,
                                num_actions=env.action_space.n,
                                num_states=env.observation_space.n,
                                discount_factor=0.7)

    # result: 2.18 average iterations, 20 average reward - 50 iter
    # result: 2.08 average iterations, 20 average reward - 100 iter

    # policy, V = policy_iteration(env=env,
    #                             num_actions=env.action_space.n,
    #                             num_states=env.observation_space.n,
    #                             discount_factor=0.9)

    # result: 2.22 average iterations, 20 average reward - 50 iter
    # result: 2.14 average iterations, 20 average reward - 100 iter

    actions = list(policy[init_state])
    max_act = max(actions)
    max_ind = actions.index(max_act)

    avg_iter = 0
    avg_reward = 0


    # for i in range (50):
    #     num_iter = 0
    #     num_reward = 0
    #     while(True):
    #         state, reward, done, truncated, info = env.step(max_ind)
    #         env.render()
    #         actions = list(policy[state])
    #         max_act = max(actions)
    #         max_ind = actions.index(max_act)
    #         num_iter += 1
    #         if done:
    #             break
    #
    #     avg_reward += reward
    #     avg_iter += num_iter
    #
    # print(avg_iter/50)
    # print(avg_reward/50)


    for i in range (100):
        num_iter = 0
        num_reward = 0
        while(True):
            state, reward, done, truncated, info = env.step(max_ind)
            env.render()
            actions = list(policy[state])
            max_act = max(actions)
            max_ind = actions.index(max_act)
            num_iter += 1
            if done:
                break

        avg_reward += reward
        avg_iter += num_iter

    print(avg_iter/100)
    print(avg_reward/100)
