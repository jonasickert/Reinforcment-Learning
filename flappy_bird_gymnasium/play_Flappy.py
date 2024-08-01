import gymnasium
import numpy as np
import pygame
from gymnasium.utils import play

import flappy_bird_gymnasium


controls = {
    (pygame.K_SPACE,) : 1
}
env = gymnasium.make ("FlappyBird-v0")
play.play(env,keys_to_action=controls,noop=0)
