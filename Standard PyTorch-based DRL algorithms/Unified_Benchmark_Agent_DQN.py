import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = nn.functional.leaky_relu(self.fc1(state))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = nn.functional.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQN_ENG:
    def __init__(self, state_dim, action_dim, learning_rate, memory_size, batch_size, gamma, tau, target_update):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(state_dim, action_dim).to(self.device)
        self.target_dqn = DQN(state_dim, action_dim).to(self.device)
        self.dqn_optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.pointer = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update = target_update

    def choose_action(self, state, epsilon):
        state = torch.FloatTensor(state).to(self.device)
        if np.random.random() <= epsilon:
            action = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.dqn(state)
                action = torch.argmax(q_values).item()
        return action

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        self.pointer += 1

    def learn(self, step):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([transition[0] for transition in batch])).to(self.device)
        actions = torch.LongTensor(np.array([transition[1] for transition in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([transition[2] for transition in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([transition[3] for transition in batch])).to(self.device)

        current_q_values = self.dqn(states).gather(1, actions)
        next_q_values = self.target_dqn(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        if step % self.target_update == 0:
            for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, episode):
        torch.save(self.dqn.state_dict(), './Training_Saved_Files/DQN_EMS' + '_{}'.format(episode) + '.pth')

    def load(self):
        actor_ems_path = './Policy_EMS_Net.pth'
        if os.path.exists(actor_ems_path):
            self.dqn.load_state_dict(torch.load(actor_ems_path, map_location='cpu'))
            print("Models loaded successfully.")
        else:
            print("No saved models found.")