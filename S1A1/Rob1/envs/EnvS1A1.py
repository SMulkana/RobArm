"""
Scenario 1:
Robot Arm has picked an object from position A and has to place it at position B. An obstacle 
Obs1 is placed in it's path. It has to maneuver around the obstacle and reach positon B.

1. object is placed at fixed position
2. target location is fixed
3. obstacle is static and at fixed  position

Assumption: 
-Object has been gripped and picked up by the gripper at positon A
-Robot Arm has to plan trajectory to get to position B
-Robot Arm stops at position B
-2D Observation and action space taken 
-robot arm taken as point mass
- robot arm can only move up, down, left, right
- simple grid world
-obstacle is placed in the middle of object and goal
"""

import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces

class SimpleEnv(gym.Env):
    metadata = { "render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size #Size of the observation space
        self.window_size = 512 #Size of the action space 

        self.observation_space = spaces.Dict(
            {
                "Obj": spaces.Box(0, size -1, shape=(2,), dtype=int),
                "Goal": spaces.Box(0, size -1, shape=(2,), dtype=int),
                "Obs1": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
    

        ) 

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
             "Obj": self._Obj_location, 
             "Goal": self._Goal_location, 
             "Obs1": self._Obs1_location
        }
   

    def _get_info(self):
    # Calculate distance from agent to obstacle
        distance_Obj_to_obstacle = np.linalg.norm(self._Obj_location - self._Obs1_location, ord=1)

    # Calculate distance from obstacle to target
        distance_Obs1_to_target = np.linalg.norm(self._Obs1_location - self._Goal_location, ord=1)

    # Calculate total distance, accounting for the obstacle
        total_distance = distance_Obj_to_obstacle + distance_Obs1_to_target

        return {
               "distance": total_distance
    }
    
    def reset(self, seed=None, options=None):
                # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._Obj_location = self.np_random.integers(0, self.size, size=2, dtype=int)
       

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._Goal_location = self._Obj_location
        while np.array_equal(self._Goal_location, self._Obj_location):
            self._Goal_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        self._Obs1_location = (self._Obj_location + self._Goal_location) // 2
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        next_location = np.clip(self._Obj_location + direction, 0, self.size - 1)

    # Check if the potential next location is not blocked by an obstacle
        if next_location != self._Obs1_location:
            self._Obj_location = next_location
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._Obj_location, self._Goal_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._Goal_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._Obj_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        pygame.draw.rect(
            canvas,
            (255, 255, 0),  # You can use a different color (e.g., yellow) for the obstacle
            pygame.Rect(
                pix_square_size * self._Obs1_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            pygame.time.delay(1000)  # Add a 1-second delay (1000 milliseconds)
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
