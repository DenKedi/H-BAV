# H-BAV/agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# Wähle das Gerät (GPU, falls verfügbar, sonst CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Die von dir definierte Netzwerkarchitektur."""

    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    """Der von dir definierte Replay Buffer."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))

        return (
            torch.from_numpy(np.vstack(states)).float().to(device),
            torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.from_numpy(np.vstack(next_states)).float().to(device),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Diese Klasse kapselt den Agenten und seine Lernlogik.
    """

    def __init__(self, state_dim, action_dim, buffer_size=200000, batch_size=64, gamma=0.99, lr=0.00025):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target-Netzwerk ist nur zur Evaluation

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.criterion = nn.MSELoss()

    def select_action(self, state, epsilon):
        """Wählt eine Aktion mit der Epsilon-Greedy-Strategie."""
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                # wähle die Aktion mit dem höchsten Q-Wert
                return q_values.max(1)[1].item()
        else:
            # wähle eine zufällige Aktion
            return random.randrange(self.action_dim)

    def learn(self):
        """Führt einen Lernschritt aus."""
        if len(self.replay_buffer) < self.batch_size:
            return None  # Nicht genug Samples im Buffer

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Q-Werte der Aktionen, die tatsächlich ausgeführt wurden
        q_expected = self.policy_net(states).gather(1, actions)

        # Q-Werte der nächsten Zustände vom Target-Netzwerk
        q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)

        # Berechnung des Ziel-Q-Wertes (Bellman-Gleichung)
        # Für terminale Zustände (done=1) ist der zukünftige Wert 0
        q_target = rewards + (self.gamma * q_next * (1 - dones))

        # Verlust berechnen
        loss = self.criterion(q_expected, q_target)

        # Netzwerk optimieren
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        """Kopiert die Gewichte vom Policy-Netzwerk zum Target-Netzwerk."""
        self.target_net.load_state_dict(self.policy_net.state_dict())