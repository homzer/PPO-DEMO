from typing import Union

import numpy as np
import gym
import torch
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common import vec_env
from tqdm import trange

from src.args import ModelArgs
from src.buffer import RolloutBuffer
from src.policy import ActorCritic


class BufferCollector:
    def __init__(
            self, args: ModelArgs,
            env: Union[gym.Env, vec_env.VecEnv],
            policy: ActorCritic
    ):
        self.args = args
        self.env = env
        self.policy = policy

    def renew(self, env: Union[gym.Env, vec_env.VecEnv]):
        self.env = env

    def collect(self) -> RolloutBuffer:
        self.policy.eval()

        rollout_buffer = RolloutBuffer(
            buffer_size=self.args.n_collect_steps,
            observation_space=self.args.observation_space,
            action_space=self.env.action_space,
            device=self.args.device,
            gae_lambda=self.args.gae_lambda,
            gamma=self.args.gamma,
            n_envs=self.args.num_envs
        )
        rollout_buffer.reset()
        last_obs = self.env.reset()
        # whether the last episode is the starting state (or game over in the last episode)
        last_episode_starts = np.ones((self.args.num_envs,), dtype=bool)

        print("Collecting data ...")
        for _ in trange(self.args.n_collect_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    last_obs, device=self.args.device
                )
                # obs_tensor = _reshape_observation(obs_tensor)
                action_masks = get_action_masks(self.env)
                actions, values, log_probs = self.policy.forward(
                    obs_tensor, action_masks=action_masks
                )
            actions = actions.cpu().numpy()
            values = values.flatten().cpu().numpy()
            log_probs = log_probs.cpu().numpy()
            # despite game over, it still continue
            new_obs, rewards, dones, infos = self.env.step(actions)
            actions = actions.reshape(-1, 1)

            rollout_buffer.add(
                obs=last_obs,
                action=actions,
                reward=rewards,
                episode_start=last_episode_starts,
                values=values,
                log_probs=log_probs,
                action_masks=action_masks
            )

            last_obs = new_obs
            last_episode_starts = dones

        with torch.no_grad():
            last_obs = torch.tensor(last_obs, device=self.args.device)
            # last_obs = _reshape_observation(last_obs)
            last_values = self.policy.predict_values(last_obs)
        rollout_buffer.compute_returns_and_advantage(
            last_values, last_episode_starts
        )

        return rollout_buffer
