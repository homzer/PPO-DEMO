import os

import numpy as np
import torch
import torch.nn.functional as F

from src.args import ModelArgs
from src.buffer import RolloutBuffer
from src.policy import ActorCritic


class TrainerForPpo:
    def __init__(self, args: ModelArgs, policy: ActorCritic):
        self.args = args
        self.policy = policy
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=args.lr
        )

    def train(self, rollout_buffer: RolloutBuffer):
        self.policy.train()

        clip_range = 0.07  # TODO: schedule function

        # For logging
        entropy_losses = []
        policy_losses = []
        value_losses = []
        losses = []

        for _ in range(self.args.n_update_epochs):
            for rollout_data in rollout_buffer.get(self.args.batch_size):
                actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    obs=rollout_data.observations,
                    actions=actions,
                    action_masks=rollout_data.action_masks
                )

                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss using the TD(Temporal Difference)(gae_lambda) target
                # Regression training for value function (or critic)
                values = values.flatten()
                value_loss = F.mse_loss(rollout_data.returns, values)

                # Entropy loss favor exploration (smoothing action distribution)
                entropy_loss = - torch.mean(entropy)

                loss = policy_loss + self.args.ent_coef * entropy_loss + self.args.vf_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # logging
                policy_losses.append(policy_loss.detach().cpu().numpy())
                value_losses.append(value_loss.detach().cpu().numpy())
                entropy_losses.append(entropy_loss.detach().cpu().numpy())
                losses.append(loss.detach().cpu().numpy())

        print("\n===============================")
        print("train/entropy_loss", np.mean(entropy_losses))
        print("train/policy_gradient_loss", np.mean(policy_losses))
        print("train/value_loss", np.mean(value_losses))
        print("train/loss", np.mean(losses))
        print("===============================")

    def predict(self, observation, action_masks=None):
        observation = torch.tensor(observation, dtype=torch.float32, device=self.args.device)
        if len(observation.shape) == 3:
            observation = observation[None]
        # observation = _reshape_observation(observation)
        return self.policy.predict(observation, action_masks=action_masks)

    def load(self, saved_dir: str):
        state_dict = torch.load(saved_dir)
        self.policy.load_state_dict(state_dict)

    def save(self, saved_dir: str, name: str):
        os.makedirs(saved_dir, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(saved_dir, f"{name}.bin"))
