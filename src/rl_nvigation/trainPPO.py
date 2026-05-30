import rclpy
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from robotEnv import HardWorldEnv


# ======================
# CALLBACK (UNCHANGED ✅)
# ======================
class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.collisions = []
        self.efficiency = []
        self.smoothness = []
        self.curr_reward = 0

    def _on_step(self):
        self.curr_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            env = self.training_env.envs[0].env

            self.rewards.append(self.curr_reward)
            self.collisions.append(env.collision_count)

            shortest = np.linalg.norm(env.goal)
            eff = shortest / (env.path_length + 1e-6)
            self.efficiency.append(eff)

            self.smoothness.append(env.smoothness)

            self.curr_reward = 0

        return True


# ======================
# TRAINING PLOTS (UNCHANGED ✅)
# ======================
def plot(data, title, name):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.grid()
    plt.savefig(name)
    plt.close()


# ======================
# 🔥 EVALUATION FUNCTION (NEW)
# ======================
def evaluate_model(model, env, num_episodes=500):
    stats = {
        "rewards": [],
        "smoothness": [],
        "efficiency": [],
        "collisions": [],
        "success_rate": []
    }

    obs = env.reset()
    current_episode_reward = 0
    episodes_completed = 0

    print(f"\n📊 Evaluating for {num_episodes} episodes...")

    while episodes_completed < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)

        current_episode_reward += reward[0]

        if done[0]:
            info = infos[0]

            stats["rewards"].append(current_episode_reward)
            stats["smoothness"].append(info.get("smoothness", 0))
            stats["efficiency"].append(info.get("path_efficiency", 0))
            stats["collisions"].append(info.get("collision_occurred", 0))
            stats["success_rate"].append(info.get("is_success", 0))

            print(
                f"Ep {episodes_completed+1}: "
                f"Reward={current_episode_reward:.2f}, "
                f"Eff={info.get('path_efficiency', 0):.2f}, "
                f"Success={info.get('is_success', 0)}"
            )

            current_episode_reward = 0
            episodes_completed += 1

    return stats


# ======================
# 🔥 FINAL EVAL PLOTS (NEW)
# ======================
def save_evaluation_summary(stats):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3)

    axs[0, 0].plot(stats["efficiency"], marker='o')
    axs[0, 0].set_title("Path Efficiency (1.0 = Optimal)")
    axs[0, 0].set_ylim(0, 1.1)
    axs[0, 0].grid(True)

    axs[0, 1].plot(stats["smoothness"], marker='s')
    axs[0, 1].set_title("Trajectory Smoothness (Lower = Better)")
    axs[0, 1].grid(True)

    axs[1, 0].plot(stats["rewards"])
    axs[1, 0].set_title("Total Reward per Episode")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].grid(True)

    total = len(stats["collisions"])
    coll_percent = (sum(stats["collisions"]) / total) * 100
    succ_percent = (sum(stats["success_rate"]) / total) * 100

    axs[1, 1].bar(["Success Rate", "Collision Rate"], [succ_percent, coll_percent])
    axs[1, 1].set_title(f"Overall Performance (%) over {total} Episodes")
    axs[1, 1].set_ylim(0, 100)

    for i, v in enumerate([succ_percent, coll_percent]):
        axs[1, 1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.suptitle("RL Navigation Agent Evaluation - HardWorld", fontsize=16)

    filename = "FINAL_EVALUATION.png"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ Evaluation plots saved at: {os.getcwd()}/{filename}")
    plt.close()

    # Save CSV
    pd.DataFrame(stats).to_csv("evaluation_metrics.csv", index=False)


# ======================
# MAIN
# ======================
def main():
    rclpy.init()

    env = DummyVecEnv([lambda: Monitor(HardWorldEnv())])
    callback = MetricsCallback()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./Final_PPO_LOGS"
    )

    # ======================
    # TRAINING
    # ======================
    print("🚀 Training...")
    model.learn(130000, callback=callback)

    model.save("edited_ppo_model2")
    print("✅ Saved")

    # ======================
    # TRAINING PLOTS (KEEPED ✅)
    # ======================
    plot(callback.rewards, "Reward", "FINAL_reward_world.png")
    plot(callback.collisions, "Collisions", "FINAL_collision_world.png")
    plot(callback.efficiency, "Efficiency", "FINAL_efficiency_world.png")
    plot(callback.smoothness, "Smoothness", "FINAL_smoothness_world.png")

    # ======================
    # 🔥 EVALUATION (NEW)
    # ======================
    stats = evaluate_model(model, env, num_episodes=500)
    save_evaluation_summary(stats)

    # ======================
    # INFERENCE TIME
    # ======================
    obs = env.reset()
    done = False
    inf_times = []

    while not done:
        start = time.time()
        action, _ = model.predict(obs)
        inf_times.append(time.time() - start)

        obs, _, dones, _ = env.step(action)
        done = dones[0]

    print("⚡ Avg inference time:", np.mean(inf_times))

    rclpy.shutdown()


if __name__ == "__main__":
    main()
