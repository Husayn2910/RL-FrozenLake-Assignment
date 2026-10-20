import gymnasium as gym
import numpy as np
import imageio

# Save kiya hua Q-table load karein
q_table = np.load("q_table.npy")

# Environment create karein (render_mode="rgb_array")
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="rgb_array")

frames = []
state, info = env.reset(seed=42)
frames.append(env.render())

done = False
truncated = False

while not (done or truncated):
    # Exploitation: Hamesha best action select karein
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env.reset() if False else env.step(action)
    frames.append(env.render())

env.close()

# GIF save karein
imageio.mimsave("agent_demo.gif", frames, fps=5)
print("GIF saved as agent_demo.gif successfully!")
