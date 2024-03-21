import math
from typing import List

import gym
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.game import SnakeGame


class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, enlarge_size=7, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = SnakeGame(
            seed=seed,
            board_size=board_size,
            silent_mode=silent_mode
        )
        self.game.reset()

        self.silent_mode = silent_mode

        self.action_space = 4  # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN

        self.observation_width = board_size * enlarge_size
        self.observation_height = board_size * enlarge_size

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.observation_width, self.observation_height, 3),
            dtype=np.uint8
        )

        self.board_size = board_size
        self.enlarge_size = enlarge_size
        self.grid_size = board_size ** 2  # Max length of snake is board_size^2
        self.init_snake_size = len(self.game.snake)
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        if limit_step:
            self.step_limit = self.grid_size * 4  # More than enough steps to get the food.
        else:
            self.step_limit = 1e9  # Basically no limit.
        self.reward_step_counter = 0

    def reset(self):
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs

    def step(self, action):
        self.done, info = self.game.step(action)
        # info = {"snake_size": int, "snake_head_pos": np.array,
        # "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        self.reward_step_counter += 1

        if info["snake_size"] == self.grid_size:  # Snake fills up the entire board. Game over.
            reward = self.max_growth * 0.1  # Victory reward
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, self.done, info

        if self.reward_step_counter > self.step_limit:  # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True

        if self.done:  # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            reward = - math.pow(self.max_growth, (
                    self.grid_size - info["snake_size"]) / self.max_growth)  # (-max_growth, -1)
            reward = reward * 0.1
            return obs, reward, self.done, info

        elif info["food_obtained"]:  # Food eaten. Reward boost on snake size.
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0  # Reset reward step counter

        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(
                    info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1

        return obs, reward, self.done, info

    def render(self, mode='human'):
        self.game.render()

    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space)]])

    # Check if the action is against the current direction of the snake or is ending the game.
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0:  # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1:  # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2:  # RIGHT
            if current_direction == "LEFT":
                return False
            else:
                col += 1

        elif action == 3:  # DOWN
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall.
        # Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                    (row, col) in snake_list  # The snake won't pop the last cell if it ate food.
                    or row < 0
                    or row >= self.board_size
                    or col < 0
                    or col >= self.board_size
            )
        else:
            game_over = (
                    (row, col) in snake_list[:-1]  # The snake will pop the last cell if it did not eat food.
                    or row < 0
                    or row >= self.board_size
                    or col < 0
                    or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)

        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)

        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Set the food to red
        obs[self.game.food] = [0, 0, 255]

        # Enlarge the observation to 84x84
        obs = np.repeat(
            np.repeat(obs, self.enlarge_size, axis=0),
            self.enlarge_size, axis=1
        )

        obs = np.transpose(obs, (2, 0, 1))
        return obs


def make_env(seed=0, board_size=12, silent_mode=True):
    def _init():
        env = SnakeEnv(seed=seed, board_size=board_size, silent_mode=silent_mode)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        env.seed(seed)
        return env

    return _init


def create_multiprocess_env(seed_set: List[int]):
    return SubprocVecEnv([make_env(seed=s) for s in seed_set])
