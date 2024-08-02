from gymnasium.envs.registration import register

register(
     id="BallGame-pixels-v0.1",
     entry_point="ball_game.envs.BallGame:BallGameEnv",
     max_episode_steps=3000,
     kwargs = { "obs_type":"features"}
)



print("Environment registered!")