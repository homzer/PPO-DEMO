import random
import time

import fire
import torch

from src.args import ModelArgs
from src.env import SnakeEnv
from src.policy import ActorCritic


def run(ckpt_file: str = None):
    args = ModelArgs()
    model = ActorCritic(args.observation_space, args.num_actions)
    if ckpt_file is not None:
        model.load(ckpt_file)
    model.cuda() if torch.cuda.is_available() else model.cpu()
    env = SnakeEnv(seed=random.randint(1, 1e5), silent_mode=False)

    for episode in range(10):
        done = False
        obs = env.reset()
        steps = 0
        sum_step_reward = 0
        print(f"=================== Episode {episode + 1} ==================")
        while not done:
            obs = torch.tensor(obs, dtype=torch.float32)[None]
            action = model.predict(obs, action_masks=env.get_action_mask())
            obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                if info["snake_size"] == env.game.grid_size:
                    print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
                else:
                    print(f"Game over Penalty: {reward:.4f}. Score: {info['score']}")
                steps = 0
            elif info["food_obtained"]:
                print(
                    f"Food obtained at step {steps:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
                sum_step_reward = 0
            else:
                sum_step_reward += reward

            env.render()
            time.sleep(0.02)

        time.sleep(5)

    env.close()


if __name__ == '__main__':
    fire.Fire(run)
