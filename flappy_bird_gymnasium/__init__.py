from gymnasium.envs.registration import register

# Exporting envs:
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv



register(
    id="FlappyBird-v0",
    entry_point="flappy_bird_gymnasium:FlappyBirdEnv",
    kwargs={
        "render_mode" : "rgb_array",
        "obs_type" : "features",
        "human" : "True"
    }
)

register(
    id = "FlappyBird-features-v0",
    entry_point="flappy_bird_gymnasium:FlappyBirdEnv",
    kwargs={
        "render_mode" : "rgb_array",
        "obs_type" : "features",
        "human" : "False"
    }
)

register(
    id = "FlappyBird-pixels-v0",
    entry_point="flappy_bird_gymnasium:FlappyBirdEnv",
    kwargs={
        "render_mode" : "rgb_array",
        "obs_type" : "pixels",
        "human" : "False"
    }
)
#