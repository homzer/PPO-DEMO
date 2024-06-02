import torch
import numpy as np

from src.env import SnakeEnv
from src.policy import ActorCritic
from src.utils import Timer


class SnakeGameEvaluator:
    def __init__(self, model: ActorCritic):
        self.model = model
        self.env = SnakeEnv(silent_mode=True)

    def forward(self, num_episodes: int = 1):
        scores = []
        timer = Timer(num_episodes, episode=5)
        for episode in range(num_episodes):
            done = False
            obs = self.env.reset()
            score = 0
            while not done:
                obs = torch.tensor(obs, dtype=torch.float32)[None]
                action = self.model.predict(obs, action_masks=self.env.get_action_mask())
                obs, reward, done, info = self.env.step(action)
                score = info['score']
            timer.step()
            # print(f"Episode {episode} Score: ", score)
            scores.append(score)
        print("Final Average Score: ", np.mean(scores), "Std: ", np.std(scores))
        return np.mean(scores)

    def __del__(self):
        self.env.close()
