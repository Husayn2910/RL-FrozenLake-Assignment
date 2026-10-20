
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent


def train_agent(episodes=20_000, seed=42):
    env = gym.make("FrozenLake-v1", is_slippery=False)

    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        seed=seed,
    )

    rewards = []
    successes = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0

        while True:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            agent.update(
                state,
                action,
                reward,
                next_state,
                terminated,
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)
        successes.append(int(total_reward > 0))

        if (episode + 1) % 2000 == 0:
            recent = np.mean(successes[-2000:])
            print(
                f"Episode {episode + 1:5d} | "
                f"Success rate: {recent:.1%} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()

    agent.save("q_table.npy")
    np.save("episode_rewards.npy", np.array(rewards))
    np.save("episode_successes.npy", np.array(successes))

    # Plot the success rate over training
    window = 500
    success_rate = np.convolve(
        np.array(successes, dtype=float),
        np.ones(window) / window,
        mode="valid",
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        np.arange(window, episodes + 1),
        success_rate,
    )
    plt.xlabel("Training episode")
    plt.ylabel("Success rate (500-episode window)")
    plt.title("FrozenLake Q-learning Training Progress")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()

    print("\nTraining finished.")
    print("Saved Q-table to q_table.npy")
    print("Saved training graph to training_curve.png")

    return agent


if __name__ == "__main__":
    train_agent()
