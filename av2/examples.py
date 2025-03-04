import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='human',
                   map_name='8x8', is_slippery=True)
    env.reset()
    env.render()

    env.step(2)
    env.render()
    env.step(2)
    env.render()
    env.step(2)
    env.render()
    env.step(2)
    env.render()

    print()
