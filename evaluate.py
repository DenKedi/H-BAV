import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from environment import HBAVEnv
from agent import QNetwork

# --- Konfiguration ---
MODEL_PATH = "H-BAV_model.pth"
NUM_EPISODES = 500
SHOW_GAME = False
OUTPUT_EXCEL_FILE = "H-BAV_Evaluation.xlsx"
OUTPUT_PLOT_FILE = "H-BAV_Evaluation_Plot.png"


def evaluate_agent():

    env = HBAVEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = QNetwork(input_dim=16, output_dim=6).to(device)
    try:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Fehler: Modelldatei nicht gefunden unter '{MODEL_PATH}'.")
        return

    policy_net.eval()

    print(f"Modell von '{MODEL_PATH}' geladen. Starte Evaluation über {NUM_EPISODES} Episoden...")

    all_rewards = []
    all_final_ball_counts = []

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0

        for _ in range(5000):  # Maximale Schrittzahl pro Episode
            if SHOW_GAME:
                env.render()
                env.clock.tick(60)

            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                break

        all_rewards.append(episode_reward)
        all_final_ball_counts.append(len(env.balls))
        print(f"Episode {episode}/{NUM_EPISODES} abgeschlossen | Reward: {episode_reward:.2f}")

    env.close()
    print("\nEvaluation abgeschlossen.")

    # --- Datenverarbeitung und Speicherung ---
    df = pd.DataFrame({
        'Episode': range(1, NUM_EPISODES + 1),
        'Reward': all_rewards,
        'Finale_Ballanzahl': all_final_ball_counts
    })

    summary_stats = {
        'Metrik': ['Reward', 'Finale Ballanzahl'],
        'Durchschnitt': [np.mean(all_rewards), np.mean(all_final_ball_counts)],
        'Standardabweichung': [np.std(all_rewards), np.std(all_final_ball_counts)],
        'Median': [np.median(all_rewards), np.median(all_final_ball_counts)],
        'Maximalwert': [np.max(all_rewards), np.max(all_final_ball_counts)],
        'Minimalwert': [np.min(all_rewards), np.min(all_final_ball_counts)],
    }
    summary_df = pd.DataFrame(summary_stats)

    with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Episoden_Daten', index=False)
        summary_df.to_excel(writer, sheet_name='Zusammenfassung', index=False)
    print(f"Excel-Datei gespeichert als: '{OUTPUT_EXCEL_FILE}'")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df['Episode'], df['Reward'], linestyle='-', color='lightblue', alpha=0.5,
            label='Reward pro Episode (Rohdaten)')

    window_size = 25  # Die Anzahl der Episoden, über die gemittelt wird
    rolling_avg = df['Reward'].rolling(window=window_size).mean()
    ax.plot(df['Episode'], rolling_avg, color='navy', linestyle='-', linewidth=2,
            label=f'Gleitender Durchschnitt ({window_size} Ep.)')

    avg_reward = np.mean(all_rewards)
    ax.axhline(avg_reward, color='red', linestyle='--', linewidth=2, label=f'Gesamtdurchschnitt: {avg_reward:.2f}')

    ax.set_title(f'Reward-Verlauf über {NUM_EPISODES} Evaluationsepisoden', fontsize=16)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Gesamte Belohnung (Reward)', fontsize=12)
    ax.legend()

    tick_step = max(1, NUM_EPISODES // 10)
    ax.set_xticks(np.arange(0, NUM_EPISODES + 1, tick_step))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"Plot gespeichert als: '{OUTPUT_PLOT_FILE}'")


if __name__ == '__main__':
    evaluate_agent()