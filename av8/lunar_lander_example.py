import gymnasium as gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import random


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, state_dim, action_dim, max_action,
                 actor_lr=1e-3, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, buffer_size=1000000, batch_size=64, device='cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)
        state, action, reward, next_state, done = [
            x.to(self.device) for x in (state, action, reward, next_state, done)]

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, x0=None, device='cpu', seed=None):
        self.size = size if isinstance(size, (tuple, list)) else (size,)
        self.mu = torch.full(self.size, mu, device=device, dtype=torch.float32)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.device = device
        self.x0 = torch.zeros(self.size, device=device) if x0 is None else torch.tensor(
            x0, device=device, dtype=torch.float32)
        self.reset()
        if seed is not None:
            torch.manual_seed(seed)

    def reset(self):
        self.x_prev = self.x0.clone()

    def __call__(self):
        noise = self.theta * (self.mu - self.x_prev) * self.dt
        noise += self.sigma * torch.sqrt(torch.tensor(
            self.dt, device=self.device)) * torch.randn(self.size, device=self.device)
        self.x_prev = self.x_prev + noise
        return self.x_prev.clone()


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = DDPG(state_dim, action_dim, max_action, device=device)

    noise = OrnsteinUhlenbeckNoise(action_dim, sigma=0.3, device=device)

    num_episodes = 1000
    recent_rewards = deque(maxlen=10)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            action += noise().cpu().numpy()
            action = np.clip(action, -max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.add_experience(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward

        recent_rewards.append(episode_reward)
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        print(
            f'Episode {episode + 1} Reward: {episode_reward:.2f}')

    total_test_reward = 0
    for test_num in range(100):
        state, _ = env.reset()
        done = False
        test_reward = 0

        while not done:
            action = agent.select_action(state)
            action = np.clip(action, -max_action, max_action)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            test_reward += reward

        print(f'Test Episode {test_num + 1} Total Reward: {test_reward:.2f}')
        total_test_reward += test_reward

    print(f"Average Reward is {total_test_reward / 100:.2f}")
    env.close()

# this ddpg implementation works by combining an actor network that learns a policy to select continuous actions
# with a critic network that evaluates the q value of state action pairs enabling stable policy updates
# the replay buffer stores past experiences to break correlations between samples and improve learning
# target networks are used with soft updates to stabilize training and prevent divergence
# ornstein uhlenbeck noise is added to actions during training to encourage exploration in continuous action space
#
# after training for 1000 episodes and testing over 100 episodes the average reward is around 20 63
# this relatively low reward indicates that while the agent learns some useful behaviors it has not fully
# mastered the environment possible reasons include insufficient training time suboptimal hyperparameters
# or the complexity of lunarlandercontinuous requiring more exploration and finetuning
# further training hyperparameter tuning or using enhancements like prioritized replay or better exploration
# strategies could improve performance and increase average rewards
