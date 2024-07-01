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

    class Preprocessing(gym.Wrapper):
        """
        Wrapper has a method atariPreprocessing, but it only converts atari frames witch 210x180 pixel.
        Own preprocessing necessary because our games are not in 210x180
        """
        def __init__(self, env):
            """:param env: environment where the reward should be clipped"""
            super().__init__(env)
            print("init Preprocessing done!")

        def reset(self):
            """
            Takes the normal observation from env.
            Return of reset() in env is a np.array and not an image => call preprocess()
            :return: loock return preprocess()
            """
            norm_obs, info = self.env.reset()
            new_obs = self.preprocess(norm_obs)
            return new_obs, info

        def step(self, action):
            """
            Takes the normal observation from env.
            Return of step() in env is also a np.array and not an image => call preprocess()
            :return: loock return preprocess()
            """
            norm_obs, rew, terminated, info, done = self.env.step(action)
            new_obs = self.preprocess(norm_obs)
            return new_obs, rew, terminated, info, done

        def preprocess(self, obs_array):
            """
            :param obs_array: np.array to convert into a grayscale image
            first, array has to be converted into image and then into grayscale
            :return: new grayscale image
            """
            gray_image = Image.fromarray(obs_array)
            gray_image = gray_image.convert("L")
            return np.asarray(gray_image)

    def __init__(self, envi: gym.Env):
        """:param envi: environmnet that has to be changed"""
        self.env = self.RewardClipper(envi)
        self.env = self.Preprocessing(self.env)
        self.env = gym.wrappers.FrameStack(self.env, 1)
        print("init Wrapper done!")

    def get_env(self):
        """
        funciton to return changed env, not possible in __init__()
        :return: env with changed parameters
        """
        return self.env


#----------------Test--------------------
env = gym.make("Hopper-v4", render_mode="rgb_array")
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

