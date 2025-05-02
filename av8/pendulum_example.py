import gymnasium as gym
from deep_q_learning_torch import DDPG, OrnsteinUhlenbeckActionNoise

if __name__ == '__main__':
    env = gym.make('Pendulum-v1', render_mode='human')
    env.reset()

    #env.render()

    agent = DDPG(state_space_shape=(3,), action_space_shape=(1,),
                 learning_rate_actor=0.001, learning_rate_critic=0.001,
                 discount_factor=0.9, batch_size=64, memory_size=1000)

    num_episodes = 100
    num_steps_per_episode = 50

    noise = OrnsteinUhlenbeckActionNoise(action_space_shape=(1,))

    for episode in range(num_episodes):
        state, _ = env.reset()
        for step in range(num_steps_per_episode):
            if episode % 5:
                action = agent.get_action(state, discrete=False) + noise()
            else:
                action = agent.get_action(state, discrete=False)
            new_state, reward, terminated, _, _ = env.step(action)

            agent.update_memory(state, action, reward, new_state, terminated)

            state = new_state

        agent.train()

        if (episode + 1) % 20:
            agent.update_target_model()

    print()



