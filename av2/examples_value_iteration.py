import gymnasium as gym
from mdp import value_iteration, policy_iteration

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='human',
                   map_name='8x8', is_slippery=True)
    env.reset()
    env.render()

    policy, V = value_iteration(env=env,
                    num_actions=env.action_space.n,
                    num_states=env.observation_space.n)


    print()
