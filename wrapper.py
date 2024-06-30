import gymnasium as gym
from PIL import Image

class wrapper():
    """class wrapper contains the conversion of pixel observation to new parameters"""

    class RewardClipper(gym.RewardWrapper):
        """class for reard clipping 1.a"""
        def __init__(self, env):
            """:param env: environment where the reward should be clipped"""
            super().__init__(env)

        def reward(self, reward: float):
            """
            :param reward: the reward that should be clipped between -1 and 1
            :return: is reward < -1 it's clipped to -1 (max), if reward > 1 it's clipped to 1 (min)
            """
            return min(max(-1, reward), 1)

    class Preprocessing(gym.Wrapper):
        """
        Wrapper has a method atariPreprocessing, but it only converts atari frames witch 210x180 pixel.
        Own preprocessing necessary because our games are not in 210x180
        """
        def __init__(self, env):
            """:param env: environment where the reward should be clipped"""
            super().__init__(env)

        def reset(self):
            """
            Takes the normal observation from env.
            Return of reset() in env is a np.array and not an image => call preprocess()
            :return: loock return preprocess()
            """
            norm_obs = self.env.reset()
            new_obs = self.preprocess(norm_obs[0])
            return new_obs

        def step(self, action):
            """
            Takes the normal observation from env.
            Return of step() in env is also a np.array and not an image => call preprocess()
            :return: loock return preprocess()
            """
            norm_obs, rew, done, info = self.env.step(action)
            new_obs = self.preprocess(norm_obs[0])
            return new_obs


        def preprocess(self, obs_array):
            """
            :param obs_array: np.array to convert into an grayscale image
            first, array has to be converted into image and than into grayscale
            :return: new grayscale image
            """
            img = Image.fromarray(obs_array)
            gray_image = img.convert("L")
            return gray_image

    def __init__(self, envi: gym.Env):
        """:param envi: environmnet that has to be changed"""
        self.env = self.RewardClipper(envi)
        self.env = self.Preprocessing(self.env)
        self.env = gym.wrappers.FrameStack(self.env, 4)

    def get_env(self):
        """
        funciton to return changed env, not possible in __init__()
        :return: env with changed parameters
        """
        return self.env
