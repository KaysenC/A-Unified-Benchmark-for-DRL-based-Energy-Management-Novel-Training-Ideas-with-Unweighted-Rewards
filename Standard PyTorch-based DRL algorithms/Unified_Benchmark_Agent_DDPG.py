import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = nn.functional.leaky_relu(self.fc1(state))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = nn.functional.leaky_relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, state, action):
        x = nn.functional.leaky_relu(self.fc1(torch.cat([state, action], dim=1)))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = nn.functional.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DDPG_ENG:
    def __init__(self, state_dim, action_dim, learning_rate, memory_size, batch_size, gamma, tau, target_update):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=memory_size)
        self.pointer = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update = target_update

    def choose_action(self, state, var_conunt):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).data.cpu().numpy().flatten()

        if np.random.random() <= var_conunt:
            action = np.random.uniform(low=0, high=1.0, size=action.shape)
        return action

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        self.pointer += 1

    def learn(self, step):
        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor(np.array([transition[0] for transition in batch])).to(self.device)
        action_batch = torch.FloatTensor(np.array([transition[1] for transition in batch])).to(self.device)
        action_batch = torch.unsqueeze(action_batch, dim=1)
        reward_batch = torch.FloatTensor(np.array([transition[2] for transition in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([transition[3] for transition in batch])).to(self.device)

        q_values = self.critic(state_batch, action_batch)
        next_actions = self.target_actor(next_state_batch)
        target_q_values = self.target_critic(next_state_batch, next_actions.detach())
        target_q_values = reward_batch.unsqueeze(1) + self.gamma * target_q_values
        critic_loss = nn.MSELoss()(q_values, target_q_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if step % self.target_update == 0:
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, episode):
        torch.save(self.actor.state_dict(), './Training_Saved_Files/actor' + '_{}episode'.format(episode) + '.pth')
        torch.save(self.critic.state_dict(), './Training_Saved_Files/critic' + '_{}episode'.format(episode) + '.pth')

    def load(self):
        actor_ems_path = './Policy_EMS_Net.pth'
        if os.path.exists(actor_ems_path):
            self.actor.load_state_dict(torch.load(actor_ems_path, map_location='cpu'))
            print("Models loaded successfully.")
        else:
            print("No saved models found.")