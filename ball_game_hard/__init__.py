from gymnasium.envs.registration import register

register(
     id="BallGameHard-pixels-v0.1",
     entry_point="ball_game_hard.envs.BallGameHard:BallGameHardEnv",
     max_episode_steps=3000,
     kwargs = { "obs_type":"features"}
)



print("Environment registered!")