import gymnasium as gym
import numpy as np
from PIL import Image
from deep_q_learning_torch import DuelingDQN


def preprocess_state(state):
    img = Image.fromarray(state)
    img = img.convert('L')
    grayscale_img = np.array(img, dtype=np.float32)
    grayscale_img = grayscale_img / 255.0
    return grayscale_img[np.newaxis, :, :]


def preprocess_reward(reward):
    return np.clip(reward, -1000.0, 1000.0)


if __name__ == '__main__':
    env = gym.make('ALE/MsPacman-v5')
    state, _ = env.reset()

    state_space_shape = (1, env.observation_space.shape[0], env.observation_space.shape[1])
    num_actions = env.action_space.n

    agent = DuelingDQN(state_space_shape=state_space_shape, num_actions=num_actions)

    num_episodes = 100
    epsilon = 0.5

    for episode in range(num_episodes):
        preprocessed_state = preprocess_state(state)
        action = agent.get_action(preprocessed_state, epsilon)

        new_state, reward, terminated, _, _ = env.step(action)
        new_preprocessed_state = preprocess_state(new_state)

        agent.update_memory(preprocessed_state, action, reward, new_preprocessed_state, terminated)

        state = new_state

        if (episode + 1) % 5:
            agent.train()

        if (episode + 1) & 20:
            agent.update_target_model()
