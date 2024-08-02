import gymnasium as gym
from gymnasium import spaces
import pygame
from PIL import Image
import numpy as np


class BallGameHardEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode, obs_type):
        super(BallGameHardEnv, self).__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        # Bildschirmgröße
        self.screen_width = 800
        self.screen_height = 800

        # Ball Eigenschaften
        self.ball_radius = 40
        self.ball_x = self.screen_width // 2
        self.ball_y = self.screen_height // 2
        self.ball_speed_y = 20
        self.colission_speed = 3
        self.colission_line = None
        self.colission_line_thick = 30

        # Action space (0: nichts tun, 1: Leertaste drücken)
        self.action_space = spaces.Discrete(2)

        # Observation space (Ballhöhe und Geschwindigkeit)
        self.observation_space = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)

        self.window = None

        # Pygame initialisieren
        #pygame.init()
        pygame.init()
        pygame.display.set_caption("Dont hit the f****** Border Bro! Get me trough this Project Bud")

        self.clock = pygame.time.Clock()
        self.reset()

    def get_obs(self):
        self.create_canvas()
        scaled = pygame.transform.scale(self.canvas, (84, 84))
        scaled = pygame.transform.flip(scaled, True, False)
        scaled = pygame.transform.rotate(scaled, 90)
        return np.array(pygame.surfarray.pixels3d(scaled))

    def reset(self, **kwargs):
        super().reset()
        self.ball_y = self.screen_height // 2
        self.ball_speed_y = 0
        observation = self.get_obs()
        self.colission_line = None
        return observation, {}

    def step(self, action):
        if action == 1:
            self.ball_speed_y = -20

        self.ball_speed_y += 1.5  # Gravitationseffekt
        self.ball_y += self.ball_speed_y

        if self.colission_line is None:
            #rint("in if self.colission_line is None")
            x = np.random.random()
            if x<0.1:
                self.height = np.random.randint(400, 500)
                self.colission_line = (0, 0)
                b = np.random.random()
                if b<0.5:

                    self.colission_line = (0, 0)
                else:
                    self.colission_line = (1, self.screen_height)

        if self.colission_line is not None:
            #print("in self.colission line")
            # von oben
            direction, y = self.colission_line

            if direction==0:
                if y < self.height:
                    y += self.colission_speed
                    self.colission_line = (direction, y)
                else:
                    self.height = 0
                    y -= self.colission_speed
                    self.colission_line = (direction, y)
            else:

                if y > self.screen_height-self.height:
                    y -= self.colission_speed
                    self.colission_line = (direction, y)
                else:
                    self.height = 0
                    y += self.colission_speed
                    self.colission_line = (direction, y)
            if y==0 or y>=self.screen_height or y<=0:
                self.colission_line = None


        # Ball auf dem Boden aufprallen lassen
        if self.ball_y >= self.screen_height - self.ball_radius:
            self.ball_y = self.screen_height - self.ball_radius
            self.ball_speed_y = 0

        state = np.array([self.ball_y, self.ball_speed_y], dtype=np.float32)

        done = False
        if self.ball_y >= self.screen_height - self.ball_radius:
            self.ball_y = self.screen_height - self.ball_radius
            self.ball_speed_y = 0
            done = True
        elif self.ball_y <= self.ball_radius:
            self.ball_y = self.ball_radius
            self.ball_speed_y = 0
            done = True
        elif self.colission_line is not  None:

            direction, y = self.colission_line
            if direction == 0:
                ball_height_y = self.ball_y - self.ball_radius
                if ball_height_y <= y+self.colission_line_thick/2:
                    done = True
            else:
                ball_height_y = self.ball_radius + self.ball_y
                if ball_height_y >= y-self.colission_line_thick/2:
                    done = True


        reward = float(0.1) if not done else float(-1)

        if self.render_mode == "human":
            self.render_frame()

        obs = self.get_obs()

        return obs, reward, done, False, {}

    def create_canvas(self):
        self.canvas = pygame.Surface((800,800))
        self.canvas.fill((255, 255, 255))
        #pygame.draw.line(self.canvas, (126, 200, 80), (0, self.screen_height-15),
        #                 (self.screen_width, self.screen_height-15 ), 30)
        #pygame.draw.line(self.canvas, (135, 206, 235), (0, self.screen_height - self.screen_height + 13),
        #                 (self.screen_width, self.screen_height - self.screen_height + 13), 30)
        if self.colission_line:
            pygame.draw.line(self.canvas, (0,0,0), (0, self.colission_line[1]),
                             (self.screen_width, self.colission_line[1]), self.colission_line_thick)

        pygame.draw.circle(self.canvas, (0, 0, 0), (self.ball_x, int(self.ball_y)), self.ball_radius)

        #canvas_array =
        #print(canvas_array.shape)
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def render(self, mode='human'):
        return self.create_canvas()


    def render_frame(self):
        if self.window is None:

            self.window = pygame.display.set_mode((800, 800))
        self.create_canvas()
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.display.flip()
        self.clock.tick(24)

    def close(self):
        pygame.quit()


# Beispiel, um die Umgebung zu verwenden

"""if __name__ == "__main__":
    env = BallGameHardEnv(render_mode="human", obs_type="pixels")
    state = env.reset()
    done = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                env.close()
                pygame.quit()

        action = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action = 1

        #print(action)
        state, reward, done, _ , l = env.step(action)
        print(reward)
        #img_array = Image.fromarray(state)
        #img_array.show()
        if done:
            env.reset()

    env.close()"""
