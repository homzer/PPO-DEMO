import collections
import os

import torch
import torch.nn.functional as F

from src.args import ModelArgs
from src.buffer import MaskableRolloutBufferSamples
from src.criterion import KLDivLoss
from src.policy import ActorCritic, Model, Actor


class Trainer:
    def __init__(self, policy: Model, optimizer: torch.optim.Optimizer):
        self.policy = policy
        self.optimizer = optimizer

    def _back_propagation(self, loss: torch.Tensor):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_optimizer(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        print(f'Saving optimizer to {save_dir} ......')
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, f'optimizer.bin'))
        print(f'Saving done !')

    def load_optimizer(self, ckpt_file: str):
        if not os.path.exists(ckpt_file):
            return
        print(f'Loading optimizer from {ckpt_file} .....')
        state_dict = torch.load(ckpt_file)
        self.optimizer.load_state_dict(state_dict)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda() if torch.cuda.is_available() else v
        print(f'Loading done !')

    def load(self, ckpt_dir: str):
        self.policy.load(os.path.join(ckpt_dir, "model.bin"))
        self.load_optimizer(os.path.join(ckpt_dir, "optimizer.bin"))

    def save(self, save_dir: str):
        self.policy.save(save_dir)
        self.save_optimizer(save_dir)


class TrainerForActorCritic(Trainer):
    def __init__(self, args: ModelArgs, policy: ActorCritic, optimizer: torch.optim.Optimizer):
        super().__init__(policy, optimizer)
        self.args = args
        self.clip_range = 0.07
        self.policy = policy

    def forward(self, rollout_data: MaskableRolloutBufferSamples):
        self.policy.train()

        actions = rollout_data.actions.long().flatten()
        values, log_prob, entropy, logits = self.policy.evaluate_actions(
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
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Value loss using the TD(Temporal Difference)(gae_lambda) target
        # Regression training for value function (or critic)
        values = values.flatten()
        value_loss = F.mse_loss(rollout_data.returns, values)

        # Entropy loss favor exploration (smoothing action distribution)
        entropy_loss = - torch.mean(entropy)

        loss = policy_loss + self.args.ent_coef * entropy_loss + self.args.vf_coef * value_loss
        self._back_propagation(loss)

        Outputs = collections.namedtuple("TrainerOutputs", ["entropy_loss", "policy_loss", "value_loss", "loss"])
        return Outputs(
            entropy_loss=entropy_loss.detach().cpu().item(),
            policy_loss=policy_loss.detach().cpu().item(),
            value_loss=value_loss.detach().cpu().item(),
            loss=loss.detach().cpu().item()
        )

    def predict(self, observation, action_masks=None):
        observation = torch.tensor(observation, dtype=torch.float32, device=self.args.device)
        if len(observation.shape) == 3:
            observation = observation[None]
        # observation = _reshape_observation(observation)
        return self.policy.predict(observation, action_masks=action_masks)


class KLDivTrainerForActorCritic(Trainer):
    def __init__(self, args: ModelArgs, policy: ActorCritic, optimizer: torch.optim.Optimizer):
        super().__init__(policy, optimizer)
        self.args = args
        self.policy = policy
        self.criterion = KLDivLoss(return_scalar=False)

    def forward(self, rollout_data: MaskableRolloutBufferSamples):
        self.policy.train()

        actions = rollout_data.actions.long().flatten()
        values, log_prob, entropy, logits = self.policy.evaluate_actions(
            obs=rollout_data.observations,
            actions=actions,
            action_masks=rollout_data.action_masks
        )

        # Normalize advantage
        advantages = rollout_data.advantages  # [b]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        signs = torch.sign(advantages) * 1e5
        labels = logits.detach().clone()
        labels[torch.arange(labels.size(0)), actions] = signs
        kl_loss = self.criterion.forward(logits, labels)
        kl_loss = self.args.kl_coef * torch.mean(advantages * kl_loss.sequeeze(-1))

        values = values.flatten()
        value_loss = self.args.vf_coef * F.mse_loss(rollout_data.returns, values)

        loss = value_loss + kl_loss
        self._back_propagation(loss)

        Outputs = collections.namedtuple("TrainerOutputs", ["loss", "kl_loss", "value_loss"])
        return Outputs(loss=loss.item(), kl_loss=kl_loss.item(), value_loss=value_loss.item())

    def predict(self, observation, action_masks=None):
        observation = torch.tensor(observation, dtype=torch.float32, device=self.args.device)
        if len(observation.shape) == 3:
            observation = observation[None]
        # observation = _reshape_observation(observation)
        return self.policy.predict(observation, action_masks=action_masks)
