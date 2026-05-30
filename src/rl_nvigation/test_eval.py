import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN   # ✅ CHANGE HERE
from stable_baselines3.common.vec_env import DummyVecEnv
import rclpy
from robotDQNEnv import HardWorldEnvDQN   # ✅ CHANGE ENV
import os
import time

# --- 1. DEFINE PLOTTING FUNCTION ---
def save_evaluation_summary(stats):
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    plt.subplots_adjust(hspace=0.4)

    # Path Efficiency
    axs[0, 0].plot(stats["efficiency"], color='blue')
    axs[0, 0].set_title("Path Efficiency")

    # Smoothness
    axs[0, 1].plot(stats["smoothness"], color='purple')
    axs[0, 1].set_title("Smoothness")

    # Reward
    axs[1, 0].plot(stats["rewards"], color='green')
    axs[1, 0].set_title("Total Reward")

    # Performance %
    total = len(stats["collisions"])
    succ = (sum(stats["success_rate"]) / total) * 100 if total > 0 else 0
    coll = (sum(stats["collisions"]) / total) * 100 if total > 0 else 0

    axs[1, 1].bar(["Success", "Collision"], [succ, coll], color=['green', 'red'])
    axs[1, 1].set_title("Performance %")

    # Episode Time
    axs[2, 0].plot(stats["time"], color='orange', marker='o')
    axs[2, 0].set_title("Episode Duration (Seconds)")
    axs[2, 0].set_ylabel("Seconds")

    # Avg Time
    axs[2, 1].text(0.5, 0.5, f"Avg Time: {np.mean(stats['time']):.2f}s",
                   ha='center', fontsize=14, fontweight='bold')
    axs[2, 1].axis('off')

    plt.savefig("DQN_Final_Metrics_With_Time.png")
    print("✅ Saved: DQN_Final_Metrics_With_Time.png")
    plt.close()


# --- 2. MAIN ---
def main():
    rclpy.init()

    env = DummyVecEnv([lambda: HardWorldEnvDQN()])

    if not os.path.exists("edited_dqn_model.zip"):
        print("❌ Error: DQN model not found.")
        return

    # ✅ LOAD DQN MODEL
    model = DQN.load("edited_dqn_model.zip", env=env)

    obs = env.reset()
    env0 = env.envs[0]

    stats = {
        "rewards": [],
        "smoothness": [],
        "efficiency": [],
        "collisions": [],
        "success_rate": [],
        "time": []
    }

    episode_reward = 0
    episode_traj = []

    print("🚀 Starting DQN Evaluation...")

    episode_start = time.time()

    for i in range(2000):  # same structure
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, infos = env.step(action)

        pos = env0.get_robot_position()
        episode_traj.append(pos)
        episode_reward += reward[0]

        if done[0]:
            info = infos[0]
            duration = time.time() - episode_start
            stats["time"].append(duration)

            # --- Efficiency ---
            if len(episode_traj) > 1:
                start_p = np.array(episode_traj[0])
                end_p = np.array(episode_traj[-1])
                goal_p = np.array(env0.goal)

                actual_dist = np.sum(
                    np.linalg.norm(np.diff(episode_traj, axis=0), axis=1)
                )

                initial_gap = np.linalg.norm(start_p - goal_p)
                final_gap = np.linalg.norm(end_p - goal_p)
                progress = initial_gap - final_gap

                eff_score = max(0, progress) / (actual_dist + 1e-6)
                stats["efficiency"].append(float(eff_score))
            else:
                stats["efficiency"].append(0.0)

            # --- Other metrics ---
            stats["rewards"].append(episode_reward)
            stats["smoothness"].append(info.get("smoothness", 0))
            stats["collisions"].append(info.get("collision_occurred", 0))
            stats["success_rate"].append(info.get("is_success", 0))

            print(f"Episode {len(stats['rewards'])} | Reward: {episode_reward:.1f} | Collided: {info.get('collision_occurred', 0)}")
            print(f"Ep {len(stats['rewards'])} | Time: {duration:.2f}s | Success: {info.get('is_success')}")

            # Reset
            episode_start = time.time()
            episode_reward = 0
            episode_traj = []
            obs = env.reset()

    save_evaluation_summary(stats)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
