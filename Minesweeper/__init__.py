import gymnasium
from gymnasium.envs.registration import register

register(
     id="Minesweeper-features-v0.1",
     entry_point="Minesweeper.env.minesweeper_env:MinesweeperEnv",
     max_episode_steps=3000,
     kwargs = { "obs_type":"features"}
)

register(
     id="Minesweeper-pixels-v0.1",
     entry_point="Minesweeper.env.minesweeper_env:MinesweeperEnv",
     max_episode_steps=3000,
     kwargs = { "obs_type":"pixels"}
)


print("Environment registered!")