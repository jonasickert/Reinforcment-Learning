import cv2
import gymnasium as gym
import numpy as np
from PIL import Image


class Wrapper():
    """class wrapper contains the conversion of pixel observation to new parameters"""

    class RewardClipper(gym.RewardWrapper):
        """class for reard clipping 1.a"""

        def __init__(self, env):
            """:param env: environment where the reward should be clipped"""
            super().__init__(env)
            print("init RewardClipping done!")

        def reward(self, reward: float):
            """
            :param reward: the reward that should be clipped between -1 and 1
            :return: is reward < -1 it's clipped to -1 (max), if reward > 1 it's clipped to 1 (min)
            """
            return min(max(-0.01, reward), 0.01)

    class Preprocessing(gym.ObservationWrapper):
        """
        Wrapper has a method atariPreprocessing, but it only converts atari frames witch 210x180 pixel.
        Own preprocessing necessary because our games are not in 210x180
        """

        def __init__(self, env):
            """:param env: environment where the reward should be clipped"""
            super().__init__(env)
            print("init Preprocessing done!")

        def observation(self, obs):
            """
            geschrieben von Artur! erstetz die zuvor geschriebenen step und reset Methoden.
            modifies the obsertvation. Caution, not reward terminated and info.
            :param obs:observation to modify
            :return: grayscale "image" as an array
            """
            if len(obs.shape) == 3 and obs.shape[2] == 3:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (84, 110))
            obs = obs[13:97, :]
            return obs

    def __init__(self, envi: gym.Env):
        """:param envi: environmnet that has to be changed"""
        self.env = self.RewardClipper(envi)
        self.env = self.Preprocessing(self.env)
        self.env = gym.wrappers.FrameStack(self.env, 4)
        print("init Wrapper done!")

    def get_env(self):
        """
        funciton to return changed env, not possible in __init__()
        :return: env with changed parameters
        """
        return self.env


# ----------------Test--------------------
env = gym.make("Hopper-v4", render_mode='rgb_array')
twopac = Wrapper(env)
pic = Image.fromarray(env.reset()[0])
pic.show()
print("before step()")
for i in range(20):
    action = twopac.env.action_space.sample()
    x = twopac.env.step(action)
    print(x)
    if x[2]:
        twopac.env.reset()
print("finish")
