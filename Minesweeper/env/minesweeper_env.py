import cProfile
import re

import time
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

from gymnasium.wrappers import FrameStack


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    # declaration of the game options. contains info about game size, grids, play-stile, number of cells and mines,
    # lists of cells and mines, list of sprites, the cell length, factors for scaling, number of uncovered sprites
    # and the image for the sprites.
    class GameOptions:
        # ----------------- Game Options -------------------- #
        play_random = True
        amount_of_cells_width = 8
        size = width, height = 840, 840
        screen_size = 840
        ai = False
        # --------------------------------------------------- #
        pos_ai = (0, 7)
        amount_of_mines = np.rint((amount_of_cells_width*amount_of_cells_width)/8)
        minesweeper_image = pygame.image.load("Minesweeper\\env\\resources\\treasurehunt.png")
        cells_with_random_mines = []
        grid = []
        mines = []
        sprites_list = pygame.sprite.Group()
        cell_length: float = float(screen_size / amount_of_cells_width)
        factor = float(cell_length / 32)
        uncovered_sprites = 0
        grid_size = amount_of_cells_width * amount_of_cells_width
        counter = 0

        def __init__(self):
            if self.ai: self.amount_of_mines = 10

    # init of the Gym Environment => windows, clocks, observation spaces, render modes
    def __init__(self, render_mode, obs_type, **kwargs):
        self._agent_location = None
        self.game_option = self.GameOptions()
        self.window_size = self.game_option.screen_size
        self.size = self.game_option.amount_of_cells_width
        #if render_mode == "rgb_array":
        #    self.obs_type = "pixels"
        #else:
        #    self.obs_type = "features"
        self.obs_type = obs_type
        self.window = None
        self.clock = None
        # observation space for obs_type == features
        self.observation_space = spaces.Dict(
            {
                # due to task 1, the position is fixed between 0 and 1, first is x, second is y
                "agent": spaces.Box(0, 1, shape=(2,), dtype=np.float64),
                "cells": spaces.Box(-1, 1, shape=(self.game_option.grid_size, 3), dtype=np.float64),
            })
        # observation space for ops_type == pixels
        if obs_type == "pixels":
            self.observation_space = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        # render mode has to be human or rgb-array
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        # the first value corresponds to the right direction, negativ left, positive right
        # the second value corresponds to the up direction, negativ down, positive up
        # if both are zero => uncover cell
        self._action_space = {
            0: np.array([1 / (self.game_option.amount_of_cells_width - 1), 0]),
            1: np.array([0, 1 / (self.game_option.amount_of_cells_width - 1)]),
            2: np.array([-1 / (self.game_option.amount_of_cells_width - 1), 0]),
            3: np.array([0, -1 / (self.game_option.amount_of_cells_width - 1)]),
            4: np.array([0, 0]),
            5: np.array([-1, -1])}

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        #self.canvas.fill((255, 255, 255))
        self.action = True
        self.first_uncover_at_random = True
        # 0 -> no changes at all
        # 1 -> just change circle
        # 2 -> change sprites list
        self.change_image = 0
        #self.count_frames = 0
        #self.start_time = time.time()

    # return the observation spaces
    def _get_obs(self):
        if self.obs_type == "features":
            #print("in features")
            return {"agent": self._agent_location, "cells": self.observation_cells, }#"mines": [self.game_option.amount_of_mines]}
        if self.obs_type == "pixels":
            #print("in pixels")
            eightyfour = pygame.transform.scale(self.canvas, (84, 84))
            eightyfour = pygame.transform.flip(eightyfour, True, False)
            eightyfour = pygame.transform.rotate(eightyfour, 90)
            #pic = np.array(pygame.surfarray.pixels3d(eightyfour))
            #img_array = Image.fromarray(pic.astype('uint8')).convert('RGBA')
            #img_array.show()
            #pygame.image.save(eightyfour)
            a = np.array(pygame.surfarray.pixels3d(eightyfour))
            #print(a)
            return a

    # @_conf_observation inits the observation for obs_type features.
    # observation_cells is an array a that contains arrays b with length 3.
    # b contains x and y position of within the grid and the information if a cell is covered (-1) or the number
    #   of neighboring mines within 0,1.
    # @_conf_observation fills the observation_cells with the positions which are during the game constant.
    def _conf_observation(self):
        step: float = np.round(1 / (self.game_option.amount_of_cells_width - 1), 140)
        self._agent_location = (0, 1)
        self.observation_cells = np.full((self.game_option.grid_size, 3), float(-1))
        x: float = 0
        y: float = 0
        for cell in self.observation_cells:
            cell[0] = x
            cell[1] = y
            x = x + step
            if x > 1:
                x = 0
                y = y + step

    # @reset resets the game and environment.
    # we do not only need the observation space for the return, but also for the game logic.
    # when obs_type is pixels, we still calculate the observation space for features but do not return them.
    # when obs_type is pixels, we return the return of @_render_frames
    def reset(self, **kwargs):
        self.game_option.counter = 0
        super().reset()
        self._conf_observation()
        self.window = None
        if self.obs_type is "pixels":
            self.create_canvas()
        if self.render_mode == "human":
            self._render_frame()
        #self._render_frame()
        observation = self._get_obs()
        #if self.obs_type == "pixels":
        #    self._render_frame()
        #    observation = self._get_obs()
        return observation, {}

    # this Cell only exists for creating the minesweeper grid
    # when a cell is created, the cell is automatically covered
    # within the game the Cell is no longer used and uselessi
    class Cell:
        def __init__(self, x, y, mines):
            self.x = x
            self.y = y
            self.mines = mines
            self.covered = True

    # Sprites, which later contains all information about a Cell.
    # Contains the coordinate (gridPosX, gridPosY) and position (rect.x, rect.y) of the cell
    # contains the image of the cell
    class Sprites(pygame.sprite.Sprite):
        img = pygame.image.load(
            "Minesweeper\\env\\resources\\treasurehunt.png")

        def __init__(self, grid_x, grid_y, pos_x, pos_y, spr_x, spr_y, mines, factor, cell_length):
            super().__init__()
            self.mines = mines
            self.covered = True
            self.gridPosX: int = grid_x
            self.gridPosY: int = grid_y
            #self.image = pygame.Surface((200, 200))
            self.image = pygame.Surface((32, 32))
            self.image.blit(self.img, (0, 0), (spr_x, spr_y, 32, 32))
            self.bigger_img = pygame.transform.scale_by(self.image, factor)
            self.image = self.bigger_img
            self.rect = self.image.get_rect()
            self.rect.x = pos_x
            self.rect.y = pos_y

    # @get_mines() creates a list with a fixed number of mines, depended on the grid size
    # when the game is random => if
    # when game is not random => else, fixed array, best for training, changeable at GameOption
    def get_mines(self, x, y):
        cells_with_mines = []
        if self.game_option.play_random:
            # when x and y is -1, we do not use the feature that at the first chosen cell and all neighbors are no
            #   mines, when x y are not -1, we use this feature
            neighbouring_cells = [(x, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y),
                                  (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
            while cells_with_mines.__len__() < self.game_option.amount_of_mines:
                cell = (np.round(np.random.uniform(0, self.game_option.amount_of_cells_width - 1)),
                        np.round(np.random.uniform(0, self.game_option.amount_of_cells_width - 1)))
                if x == -1 and y == -1:
                    if cell not in cells_with_mines:
                        cells_with_mines.append(cell)
                else:
                    if cell not in cells_with_mines and cell not in neighbouring_cells:
                        cells_with_mines.append(cell)

            return cells_with_mines
        else:
            return [(2, 0), (7, 0), (0, 1), (0, 3), (6, 3), (2, 5), (3, 6), (4, 5), (3, 7), (5, 7)]

    # this function creates a list of all cells with all information.
    # the position in the grid is identically with the position of cell in the game.
    # basically for every cell we count the neighbor cells which have mines, the sum is the number of
    # mines in the neighborhood, info is not stored in (0,1), but in (0,8), -1 when cell is a mine
    def create_grid(self, cells_with_mines):
        grid_temp = []
        for y in range(self.game_option.amount_of_cells_width):
            for x in range(self.game_option.amount_of_cells_width):
                has_mine = (x, y) in cells_with_mines
                if has_mine:
                    has_mine = -1
                else:
                    has_mine = 0
                neighboring_mines = 0
                if (x - 1, y - 1) in cells_with_mines:
                    neighboring_mines += 1
                if (x - 1, y) in cells_with_mines:
                    neighboring_mines += 1
                if (x - 1, y + 1) in cells_with_mines:
                    neighboring_mines += 1
                if (x, y - 1) in cells_with_mines:
                    neighboring_mines += 1
                if (x, y + 1) in cells_with_mines:
                    neighboring_mines += 1
                if (x + 1, y - 1) in cells_with_mines:
                    neighboring_mines += 1
                if (x + 1, y) in cells_with_mines:
                    neighboring_mines += 1
                if (x + 1, y + 1) in cells_with_mines:
                    neighboring_mines += 1
                if has_mine == 0 and neighboring_mines != 0:
                    has_mine = neighboring_mines
                cell = self.Cell(x, y, has_mine)
                grid_temp.append(cell)
        return grid_temp

    # this function creates for every cell a sprite and adds it into the sprites_list.
    # don't worry why the position on image stays the same, create_sprites is called only at the beginning of the
    # game, so the sprite image is all the time the same.
    def create_sprites(self, grid_temp):
        index = 0
        while index < self.game_option.amount_of_cells_width * self.game_option.amount_of_cells_width:
            cell = grid_temp[index]
            pos_x: float = self.game_option.cell_length * cell.x
            pos_y: float = self.game_option.cell_length * cell.y
            sprite = self.Sprites(int(cell.x), int(cell.y), pos_x, pos_y, 0, 64, cell.mines, self.game_option.factor,
                                  self.game_option.cell_length)
            # noinspection PyTypeChecker
            self.game_option.sprites_list.add(sprite)
            index += 1

    # changes the sprite image based on mine information.
    def change_sprite_image(self, sprite: Sprites):
        if sprite.mines < 8 and sprite.mines != 0 and sprite.covered:
            sprite.image.blit(self.game_option.minesweeper_image, (0, 0), (32 * sprite.mines, 0, 32, 32))
            self.game_option.uncovered_sprites += 1
            if self.game_option.factor < 0:
                sprite.image = pygame.transform.scale_by(sprite.image, 1/self.game_option.factor)
            else: sprite.image = pygame.transform.scale_by(sprite.image, self.game_option.factor)
        else:
            if sprite.mines == 8 and sprite.covered:
                sprite.image.blit(self.game_option.minesweeper_image, (0, 0), (0, 32, 32, 32))
                self.game_option.uncovered_sprites += 1
                if self.game_option.factor < 0:
                    sprite.image = pygame.transform.scale_by(sprite.image, 1 / self.game_option.factor)
                else:
                    sprite.image = pygame.transform.scale_by(sprite.image, self.game_option.factor)
            else:
                if sprite.mines == 0 and sprite.covered:
                    sprite.image.blit(self.game_option.minesweeper_image, (0, 0), (0, 0, 32, 32))
                    self.game_option.uncovered_sprites += 1
                    if self.game_option.factor < 0:
                        sprite.image = pygame.transform.scale_by(sprite.image, 1 / self.game_option.factor)
                    else:
                        sprite.image = pygame.transform.scale_by(sprite.image, self.game_option.factor)
        #sprite.image = pygame.transform.scale_by(sprite.image, self.  game_option.factor)
        sprite.covered = False
        #print("uncovered sprites in change sprite: " + str(self.game_option.uncovered_sprites))

    # algorithm to uncover all neighboring cells without mine and no neighboring mines.
    # called when the clicked cell has no mine and no neighboring mines.
    # when a cell has no mines or mine = 0 => uncover cell and add all neighbors who are still covered to the queue
    # when a cell has mine > 0 uncover the cell but do not add neighbors to queue
    def uncover_cells_with_no_neighbour_mines(self, grid_x, grid_y, cell_with_mines):
        queue_to_uncover = [(grid_x, grid_y)]
        uncovered_queue = []
        counter = 1
        while queue_to_uncover:
            queue_to_uncover = [item for item in queue_to_uncover if item not in uncovered_queue]
            if queue_to_uncover.__len__() == 0:
                break
            else:
                cell = queue_to_uncover.pop()
                x, y = cell
                counter += 1
                neighbouring_cells = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x - 1, y), (x + 1, y),
                                      (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
                for sprite in self.game_option.sprites_list:
                    if sprite.mines == 0 and sprite.covered and sprite.gridPosX == x and sprite.gridPosY == y:
                        #sprite.image.blit(self.game_option.minesweeper_image, (0, 0), (0, 0, 32, 32))
                        #sprite.image = pygame.transform.scale_by(sprite.image, self.game_option.factor)
                        #sprite.covered = False
                        self.change_sprite_image(sprite)
                        uncovered_queue.append((x, y))
                        cell = self.observation_cells[y*self.game_option.amount_of_cells_width+x]
                        if (np.round(cell[0], 14) == np.round(x / (self.game_option.amount_of_cells_width - 1),
                                                              14) and
                                np.round(cell[1], 14) == np.round(y / (self.game_option.amount_of_cells_width - 1),
                                                                  14)):
                            cell[2] = np.round(sprite.mines/8,14)
                        """for cell in self.observation_cells:
                            if (np.round(cell[0], 14) == np.round(x / (self.game_option.amount_of_cells_width - 1),
                                                                  14) and
                                    np.round(cell[1], 14) == np.round(y / (self.game_option.amount_of_cells_width - 1),
                                                                      14)):
                                cell[2] = sprite.mines / 8
                                break"""
                        for neighbour in neighbouring_cells:
                            queue_to_uncover.append(neighbour)
                        #self.game_option.uncovered_sprites += 1
                        break
                    else:
                        if cell_with_mines != 0 and sprite.covered and sprite.gridPosX == x and sprite.gridPosY == y:
                            self.change_sprite_image(sprite)
                            for cell in self.observation_cells:
                                if (np.round(cell[0], 14) == np.round(x / (self.game_option.amount_of_cells_width - 1),
                                                                      14) and
                                        np.round(cell[1], 14) == np.round(
                                            y / (self.game_option.amount_of_cells_width - 1), 14)):
                                    cell[2] = sprite.mines / 8
                                    break
                            uncovered_queue.append((x, y))

    def ai_start(self):
        new_location = [0,1]
        x = np.rint(new_location[0] * (self.game_option.amount_of_cells_width - 1))
        y = np.rint(new_location[1] * (self.game_option.amount_of_cells_width - 1))
        for sprite in self.game_option.sprites_list:
            if sprite.gridPosX == x and sprite.gridPosY == y:
                if sprite.mines == -1:
                    terminated = True
                    reward = float(-1)
                if sprite.mines > 0:
                    reward = float(0.1)
                    self.change_sprite_image(sprite)
                if sprite.mines == 0:
                    reward = float(0.1)
                    self.uncover_cells_with_no_neighbour_mines(sprite.gridPosX, sprite.gridPosY,
                                                               self.game_option.mines)
                if (self.game_option.grid_size - self.game_option.amount_of_mines ==
                        self.game_option.uncovered_sprites):
                    terminated = True
                    reward = float(1)
                for cell in self.observation_cells:
                    if cell[0] == self._agent_location[0] and cell[1] == self._agent_location[1]:
                        cell[2] = sprite.mines / 8



    # start_game starts the game, and restarts it
    # before starting the game all lists are set to default, mostly just empty lists
    def start_game(self, position):
        self.game_option.uncovered_sprites = 0
        self.game_option.sprites_list = pygame.sprite.Group()
        self.action = True
        if self.game_option.ai:
            #self.game_option.ai = False
            self.game_option.mines = self.get_mines(-1, -1)
            self.game_option.grid = self.create_grid(self.game_option.mines)
            self.create_sprites(self.game_option.grid)
            self.ai_start()
        else:
            if position == (-1, -1):
                self.game_option.mines = self.get_mines(-1, -1)
                self.game_option.grid = self.create_grid(self.game_option.mines)
                self.create_sprites(self.game_option.grid)
            else:
                self.game_option.mines = self.get_mines(position[0], position[1])
                #print(position)
                self.game_option.grid = self.create_grid(self.game_option.mines)
                self.create_sprites(self.game_option.grid)

    # @step method for the environment, here the action happens, hehe
    # if the agent is moving the new_location is not equal to _agent_location => just moves the position, no
    #   termination, no rewards, no win.
    # if the agent is not moving, the agent wants to uncover the cell => lookup if the cell has a mine, if yes
    #   terminated = True, reward -1, if not, lookup if won, if yes terminated = True, reward = 1, if not,
    #   terminated = False, reward = 0.1.
    def step(self, action):
        self.game_option.counter += 1
        move = self._action_space[action]
        #print(move)
        # noop return
        if move[0]==-1 and move[1]==-1: #not in self._action_space:
            #print("in move 5")
            self.change_image = 0
            self.action = False
            return self._get_obs(), float(0), False, False, {},
        else:
            # where to declare the click????
            # move = self._action_space[action]
            self.action = True
            new_location = np.clip(self._agent_location + move, 0, 1)

            if (self.first_uncover_at_random and
                    new_location[0] == self._agent_location[0] and new_location[1] == self._agent_location[1]):
                self.start_game((np.rint(new_location[0]*(self.game_option.amount_of_cells_width-1)),
                                 np.rint(new_location[1]*(self.game_option.amount_of_cells_width-1))))
                # here, change all the cells and
                # grid will be changed => sprites have to be changed
                # this will be called before uncover cells and update observation_space
                # call
                self.first_uncover_at_random = False
            reward: float = -0.02
            terminated = False
            if (new_location[0] == self._agent_location[0] and new_location[1] == self._agent_location[1] and
                move is  self._action_space[4] and self.first_uncover_at_random == False):
                x = np.rint(self._agent_location[0] * (self.game_option.amount_of_cells_width - 1))
                y = np.rint(self._agent_location[1] * (self.game_option.amount_of_cells_width - 1))
                for sprite in self.game_option.sprites_list:
                    if sprite.gridPosX == x and sprite.gridPosY == y:
                        if sprite.mines == -1 and sprite.covered:
                            terminated = True
                            reward = float(-1)
                            self.first_uncover_at_random = True
                        if sprite.mines > 0 and sprite.covered:
                            reward = float(1)
                            self.change_sprite_image(sprite)
                        if sprite.mines == 0 and sprite.covered:
                            reward = float(1)
                            self.uncover_cells_with_no_neighbour_mines(sprite.gridPosX, sprite.gridPosY,
                                                                       self.game_option.mines)
                        if (self.game_option.grid_size - self.game_option.amount_of_mines ==
                                self.game_option.uncovered_sprites):
                            self.first_uncover_at_random = True
                            terminated = True
                            reward = float(1)
                        for cell in self.observation_cells:
                            if cell[0] == self._agent_location[0] and cell[1] == self._agent_location[1]:
                                cell[2] = sprite.mines / 8
            else:
                reward = -0.02
            self._agent_location = new_location

            if self.obs_type is "pixels":
                self.create_canvas()

            observation = self._get_obs()

            if self.obs_type == "pixels":
                #self._render_frame()
                observation = self._get_obs()
                # self.create_canvas()

            if self.render_mode == "human":
                #print("in step in human calling _render_frame")
                self._render_frame()
            #print("uncovered sprites" + str(self.game_option.uncovered_sprites))
            #print(self.game_option.grid_size)
            #print(self.game_option.amount_of_mines)
            return observation, float(reward), terminated, False, {}

    # don't know why this exists; maybe for the AI who plays the game later on, fixed methods call?
    def render(self):
        return self._render_frame()


    # @_render_frame updates the pygame window.
    # at first call; game is created, till here a window does not exist.
    # creates a canvas, puts all sprites from a list into that canvas.
    # why canvas? it's resizable for the pixel space.
    # if obs_type pixels the rendert image from _render_frame is the observation for the AI
    # image has to be scaled to 84x84

    def create_canvas(self):
        #print("in create canvas")
        self.canvas = None
        self.canvas = pygame.Surface((self.window_size, self.window_size))
        # self.canvas.fill((0, 0, 0))
        self.game_option.sprites_list.update()
        self.game_option.sprites_list.draw(self.canvas)
        pos_on_canvas = (
            (self._agent_location[0] * (
                        self.game_option.amount_of_cells_width - 1)) * self.game_option.cell_length + self.game_option.cell_length / 2,
            self._agent_location[1] * (
                        self.game_option.amount_of_cells_width - 1) * self.game_option.cell_length + self.game_option.cell_length / 2)
        pygame.draw.circle(self.canvas, (205, 38, 38), center=pos_on_canvas,
                           radius=self.game_option.cell_length / 2, width=8)
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))


    def _render_frame(self):
        if self.window is None:
            self.start_game((-1, -1))
            pygame.init()
            self.window = pygame.display.set_mode(self.game_option.size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.action:
            self.action = False
            """
            self.canvas = None
            self.canvas = pygame.Surface((self.window_size, self.window_size))
            #self.canvas.fill((0, 0, 0))
            self.game_option.sprites_list.update()
            self.game_option.sprites_list.draw(self.canvas)
            pos_on_canvas = (
            (self._agent_location[0] * (self.game_option.amount_of_cells_width-1)) * self.game_option.cell_length + self.game_option.cell_length / 2,
            self._agent_location[1] * (self.game_option.amount_of_cells_width-1) * self.game_option.cell_length + self.game_option.cell_length / 2)
            pygame.draw.circle(self.canvas, (205, 38, 38), center=pos_on_canvas,
                               radius=self.game_option.cell_length / 2, width=8)
            """
            self.create_canvas()
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.display.flip()
        self.clock.tick(24)

        """
        if self.render_mode == "rgb_array":
            # factor = 84 / self.game_option.screen_size
            # eightyfour = pygame.transform.scale_by(canvas, factor)
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))
        """
    # @close closes the game, not the environment
    def close(self):
        pygame.quit()

# From here just for testing if everything work sand returns are correct
"""
env = MinesweeperEnv(render_mode="human", obs_type="features")
env.reset()
env.render()
count_frames = 0
start_time = time.time()
while count_frames < 20000:
    count_frames += 1
    rand = np.random.randint(0,5)
    x = env.step(rand)
    if x[2]:
        env.reset()
end_time = time.time()

needed_time_1 = 0

env = MinesweeperEnv(render_mode="rgb_array", obs_type="pixels")
#env = FrameStack(env, 4)

env.reset()
#env.render()
count_frames = 0
start_time = time.time()
while count_frames < 20:
    count_frames += 1
    rand = np.random.randint(0,5)
    x = env.step(rand)
    print(x[1])
    if x[2]:
        env.reset()
end_time = time.time()
needed_time = end_time-start_time
print("FPS features: " + str(20000/needed_time_1)+", Steps: 20000, Needed Time: "+ str(needed_time_1)+", grid: 8x8")
print("FPS pixels: " + str(20000/needed_time)+", Steps: 20000, Needed Time: "+ str(needed_time)+", grid: 8x8")"""
