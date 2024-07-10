import cv2
import gymnasium as gym
import numpy as np
from PIL import Image
from gymnasium import spaces


class Wrapper():
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
            return min(max(-0.01, reward), 0.01)

    class Preprocessing(gym.ObservationWrapper):
        """
        Wrapper has a method atariPreprocessing, but it only converts atari frames witch 210x180 pixel.
        Own preprocessing necessary because our games are not in 210x180
        """

        def __init__(self, env):
            """:param env: environment where the reward should be clipped"""
            super().__init__(env)

        def observation(self, obs):
            """
            geschrieben von Artur! erstetz die zuvor geschriebenen step und reset Methoden.
            modifies the obsertvation. Caution, not reward terminated and info.
            :param obs:observation to modify
            :return: grayscale "image" as an array
            """
            if len(obs.shape) == 3 and obs.shape[2] == 3:
                # Konvertieren des Bildes in Graustufen
                obs = cv2.cvtColor(obs.astype('uint8'), cv2.COLOR_RGB2GRAY)
                # Ändern der Größe des Bildes
                obs = cv2.resize(obs, (84, 110))
                # Zuschneiden des Bildes
                obs = obs[13:97, :]
                # Konvertieren in uint8, falls notwendig
                if obs.dtype != np.uint8:
                    obs = (obs * 255).astype('uint8')
                return obs
            return obs


    def __init__(self, envi: gym.Env):
        """:param envi: environmnet that has to be changed"""
        self.env = self.RewardClipper(envi)
        self.env = self.Preprocessing(self.env)
        self.env = gym.wrappers.FrameStack(self.env, 1)

    def get_env(self):
        """
        funciton to return changed env, not possible in __init__()
        :return: env with changed parameters
        """
        return self.env


envi = gym.make("Hopper-v4")

def testPreProcessing():
    """
    testPreProcessing tests the @Wrapper.PreProcessing
    """
    process = Wrapper.Preprocessing(envi)
    pic = (np.random.randn(400,400,3)* 255).astype('uint8')
    img_array = Image.fromarray(pic.astype('uint8')).convert('RGBA')
    img_array.show()
    img_new = process.observation(pic)
    img_new = Image.fromarray(img_new.astype('uint8'))
    img_new.show()

def testRewardClipping():
    """
    tests the @Wrapper.RewardClipper
    """
    env = Wrapper(envi)
    env = env.get_env()
    env.reset()
    x = env.step(envi.action_space.sample())
    print("reward with clipping: "+str(x[1]))

testPreProcessing()
envi.reset()
x = envi.step(envi.action_space.sample())
print("reward without clipping: " + str(x[1]))
testRewardClipping()


