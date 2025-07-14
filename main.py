import time
from collections import deque
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from agent import DQNAgent, QNetwork
from environment import HBAVEnv
from utils import get_epsilon_exp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modus: "train" oder "watch"
MODE = "watch"

# --- Konfiguration & Hyperparameter ---
ENV_NAME = "H-BAV"
STATE_DIM = 16
ACTION_DIM = 6
MAX_EPISODES = 700
MAX_STEPS = 1000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.00025
EPS_START = 0.7
EPS_END = 0.05
EPS_DECAY_FACTOR = (EPS_END / EPS_START) ** (1 / 12000)
BUFFER_SIZE = 200000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 500


def plot_stats(episode_history, reward_history, avg_reward_history, std_dev_history, loss_history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Trainingsfortschritt', fontsize=16)
    ax1.plot(episode_history, reward_history, label='Belohnung pro Episode', color='lightblue', alpha=0.6)
    ax1.plot(episode_history, avg_reward_history, label='Gleitender Durchschnitt (100 Ep.)', color='blue')
    avg_rewards = np.array(avg_reward_history)
    std_devs = np.array(std_dev_history)
    ax1.fill_between(episode_history, avg_rewards - std_devs, avg_rewards + std_devs, color='blue', alpha=0.2,
                     label='Standardabweichung')
    ax1.set_ylabel('Belohnung')
    ax1.set_title('Belohnungs-Verlauf')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(episode_history, loss_history, label='Durchschnittlicher Verlust pro Episode', color='red')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Verlust (Loss)')
    ax2.set_title('Verlust-Verlauf')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = f"{ENV_NAME}_training_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot gespeichert als '{plot_filename}'")


def train_agent():
    env = HBAVEnv()
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                     gamma=GAMMA, lr=LEARNING_RATE)

    episode_rewards = deque(maxlen=100)

    epsilon = EPS_START

    history_episodes, history_rewards, history_avg_rewards, history_std_devs, history_avg_losses = [], [], [], [], []

    stats_filename = f"{ENV_NAME}_training_stats.csv"

    with open(stats_filename, 'w', newline='') as csvfile:
        stats_writer = csv.writer(csvfile)
        header = ["Episode", "Total_Reward", "Avg_Reward_100", "Std_Dev_Reward_100", "Avg_Loss_Episode", "Epsilon",
                  "Steps_Survived"]
        stats_writer.writerow(header)

        print("Training beginnt...")
        print(f"Verwende Gerät: {device}")
        print(f"Statistiken werden in '{stats_filename}' gespeichert.")

        global_step = 0
        for episode in range(1, MAX_EPISODES + 1):
            state = env.reset()
            episode_reward, episode_losses = 0, []

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
                    if loss is not None:
                        episode_losses.append(loss)

                if global_step % TARGET_UPDATE_FREQ == 0:
                    agent.update_target_net()

                if done:
                    break

            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards)
            std_dev_reward = np.std(episode_rewards) if len(episode_rewards) > 1 else 0.0
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0

            history_episodes.append(episode)
            history_rewards.append(episode_reward)
            history_avg_rewards.append(avg_reward)
            history_std_devs.append(std_dev_reward)
            history_avg_losses.append(avg_loss)

            stats_row = [episode, f"{episode_reward:.2f}", f"{avg_reward:.2f}", f"{std_dev_reward:.2f}",
                         f"{avg_loss:.4f}", f"{epsilon:.3f}", env.steps_survived]
            stats_writer.writerow(stats_row)

            print(
                f"Episode {episode}: Reward = {episode_reward:.2f}, Score = {env.steps_survived}, Avg Reward (100) = {avg_reward:.2f}")

    env.close()
    print("Training abgeschlossen.")
    plot_stats(history_episodes, history_rewards, history_avg_rewards, history_std_devs, history_avg_losses)
    torch.save(agent.policy_net.state_dict(), f"{ENV_NAME}_model.pth")
    print(f"Modell gespeichert als {ENV_NAME}_model.pth")


def watch_agent_play():
    MODEL_PATH = f"{ENV_NAME}_model.pth"
    EPISODES_TO_WATCH = 10
    env = HBAVEnv()
    policy_net = QNetwork(STATE_DIM, ACTION_DIM).to(device)
    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Fehler: Modelldatei nicht gefunden unter '{MODEL_PATH}'.")
        return
    policy_net.eval()
    print(f"Lade Modell von {MODEL_PATH} und schaue {EPISODES_TO_WATCH} Episoden zu...")
    for episode in range(1, EPISODES_TO_WATCH + 1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()
            next_state, _, done, _ = env.step(action)
            state = next_state
            score = env.steps_survived
            env.clock.tick(30)
        print(f"Episode {episode}: Endpunktzahl = {score}")
        time.sleep(1)
    env.close()


if __name__ == '__main__':
    if MODE == "train":
        train_agent()
    elif MODE == "watch":
        watch_agent_play()
    else:
        print(f"Unbekannter Modus '{MODE}'. 'train' oder 'watch' wählen.")