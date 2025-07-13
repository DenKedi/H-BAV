# H-BAV/main.py

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time  # NEU: time wird für eine kurze Pause benötigt

from environment import HBAVEnv
from agent import DQNAgent, QNetwork, device  # NEU: QNetwork wird für den Watch-Mode importiert
from utils import get_epsilon_exp

# NEU: Wähle den Modus: "train" oder "watch"
MODE = "watch"

# --- Konfiguration & Hyperparameter ---
ENV_NAME = "H-BAV"
STATE_VERSION = 1
STATE_DIM = 8 if STATE_VERSION == 1 else 7
ACTION_DIM = 6

# Trainingsparameter
MAX_EPISODES = 1000
MAX_STEPS = 1000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.00025

# Epsilon-Greedy
EPS_START = 0.7
EPS_END = 0.05
EPS_DECAY_FACTOR = (EPS_END / EPS_START) ** (1 / 12000)

# Buffer & Update-Frequenzen
BUFFER_SIZE = 200000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 500


# NEU: Der bisherige Code wird in eine Trainingsfunktion verpackt
def train_agent():
    """Startet den Trainingsprozess für den Agenten."""
    env = HBAVEnv(version=STATE_VERSION)
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                     gamma=GAMMA, lr=LEARNING_RATE)

    writer = SummaryWriter(f"runs/{ENV_NAME}")
    episode_rewards = deque(maxlen=100)
    global_step = 0
    epsilon = EPS_START

    print("Training beginnt...")
    print(f"Verwende Gerät: {device}")

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        episode_reward = 0

        for step in range(1, MAX_STEPS + 1):
            global_step += 1
            epsilon = get_epsilon_exp(epsilon, EPS_END, EPS_DECAY_FACTOR)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(agent.replay_buffer) > MIN_REPLAY_SIZE:
                loss = agent.learn()
                if global_step % 100 == 0 and loss is not None:
                    writer.add_scalar("train/loss", loss, global_step)

            if global_step % TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)

        writer.add_scalar("rollout/episode_reward", episode_reward, episode)
        writer.add_scalar("rollout/avg_reward_100", avg_reward, episode)
        writer.add_scalar("rollout/epsilon", epsilon, episode)
        writer.add_scalar("rollout/score", env.steps_survived, episode)

        print(
            f"Episode {episode}: Reward = {episode_reward:.2f}, Score = {env.steps_survived}, Avg Reward (100) = {avg_reward:.2f}, Epsilon = {epsilon:.3f}")

    env.close()
    writer.close()
    print("Training abgeschlossen.")
    torch.save(agent.policy_net.state_dict(), f"{ENV_NAME}_model.pth")
    print(f"Modell gespeichert als {ENV_NAME}_model.pth")


# NEU: Funktion, um dem trainierten Agenten zuzuschauen
def watch_agent_play():
    """Lädt ein trainiertes Modell und lässt es spielen."""
    MODEL_PATH = f"{ENV_NAME}_model.pth"
    EPISODES_TO_WATCH = 10

    env = HBAVEnv(version=STATE_VERSION)
    policy_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)

    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print(f"Fehler: Modelldatei nicht gefunden unter '{MODEL_PATH}'.")
        print("Bitte trainiere zuerst den Agenten, indem du MODE = 'train' setzt.")
        return

    policy_net.eval()  # Wichtig: Setzt das Netzwerk in den Evaluationsmodus

    print(f"Lade Modell von {MODEL_PATH} und schaue {EPISODES_TO_WATCH} Episoden zu...")

    for episode in range(1, EPISODES_TO_WATCH + 1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            env.render()  # Zeige das Spiel an

            # Aktion ohne Zufall (Epsilon = 0) auswählen
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()

            next_state, _, done, _ = env.step(action)
            state = next_state
            score = env.steps_survived

            # Spielgeschwindigkeit für den Zuschauer anpassen
            env.clock.tick(30)

        print(f"Episode {episode}: Endpunktzahl = {score}")
        time.sleep(1)  # Kurze Pause zwischen den Episoden

    env.close()


# NEU: Wählt basierend auf dem MODE die auszuführende Funktion
if __name__ == '__main__':
    if MODE == "train":
        train_agent()
    elif MODE == "watch":
        watch_agent_play()
    else:
        print(f"Unbekannter Modus '{MODE}'. Bitte 'train' oder 'watch' wählen.")