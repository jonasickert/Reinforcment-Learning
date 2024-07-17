import gymnasium as gym
from gymnasium.utils.play import play
import Minesweeper


# w: up, a: left, s: down, d: right, spacer: uncover cell
_keys_to_action = {"w": 3, "a": 2, "s": 1, "d": 0, " ": 4, "-":5}
env = gym.make(id="Minesweeper-features-v0.1", render_mode="rgb_array", kwargs="feature")
play(env, keys_to_action=_keys_to_action, noop=5)

