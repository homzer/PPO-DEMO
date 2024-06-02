import os

import numpy as np
import torch
import torch.nn as nn


def logits_normalize(logits):
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def load(self, ckpt_file: str):
        print(f"Loading model from {ckpt_file} ......")
        self.load_state_dict(torch.load(ckpt_file))
        print("Loading done")

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "model.bin"))


class NatureCNN(nn.Module):
    def __init__(self, observation_space: tuple, features_dim: int = 512):
        super().__init__()
        n_input_channels = observation_space[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=(4, 4), stride=(2, 2), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=0),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            inputs = torch.ones(size=observation_space)[None].float()
            # inputs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn.forward(inputs).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim * 4),
            nn.ReLU(),
            nn.Linear(features_dim * 4, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        features = self.cnn.forward(observations)
        return self.linear.forward(features)


class ActorCritic(Model):
    def __init__(self, observation_space: tuple, num_actions: int, features_dim: int = 512):
        super().__init__()
        self.feature_extractor = NatureCNN(observation_space, features_dim)
        self.action_net = nn.Linear(features_dim, num_actions)
        self.value_net = nn.Linear(features_dim, 1)

    def forward(self, obs, action_masks: np.ndarray = None):
        features = self.feature_extractor.forward(obs.float())
        values = self.value_net.forward(features)
        logits = self.action_net.forward(features)
        logits = logits_normalize(logits)
        if action_masks is not None:
            action_masks = torch.as_tensor(
                action_masks, dtype=torch.float, device=logits.device
            ).reshape(logits.shape)
            logits += (1 - action_masks) * -1e8
            logits = logits_normalize(logits)
        probs = torch.softmax(logits, dim=-1)  # [..., 4]
        actions = torch.multinomial(probs, num_samples=1).squeeze()
        action_logits = torch.gather(logits, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

        return actions, values, action_logits

    def predict_values(self, obs):
        features = self.feature_extractor.forward(obs.float())
        return self.value_net.forward(features)

    def evaluate_actions(self, obs, actions, action_masks):
        features = self.feature_extractor.forward(obs.float())
        values = self.value_net.forward(features)
        logits = self.action_net.forward(features)
        logits = logits_normalize(logits)
        if action_masks is not None:
            action_masks = torch.as_tensor(
                action_masks, dtype=torch.float, device=logits.device
            ).reshape(logits.shape)
            logits += (1 - action_masks) * -1e8
            logits = logits_normalize(logits)
        probs = torch.softmax(logits, dim=-1)  # [..., 4]
        action_logits = torch.gather(logits, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        logits_times_probs = logits * probs
        if action_masks is not None:
            logits_times_probs = logits_times_probs * action_masks

        return values, action_logits, - logits_times_probs.sum(-1)

    def predict(self, observation, action_masks=None):
        self.eval()

        with torch.no_grad():
            observation = observation.to(next(self.parameters()).device)
            features = self.feature_extractor.forward(observation)
            logits = self.action_net.forward(features)
            logits = logits_normalize(logits)
            if action_masks is not None:
                action_masks = torch.as_tensor(
                    action_masks, dtype=torch.float, device=logits.device
                ).reshape(logits.shape)
                logits += (1 - action_masks) * -1e8
                logits = logits_normalize(logits)
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze()
        actions = actions.cpu().numpy()
        return actions


class Actor(Model):
    def __init__(self, observation_space: tuple, num_actions: int, features_dim: int = 512):
        super().__init__()
        self.feature_extractor = NatureCNN(observation_space, features_dim)
        self.action_net = nn.Linear(features_dim, num_actions)

    def forward(self, obs, action_masks: np.ndarray = None):
        features = self.feature_extractor.forward(obs.float())
        logits = self.action_net.forward(features)
        if action_masks is not None:
            action_masks = torch.as_tensor(
                action_masks, dtype=torch.float, device=logits.device
            ).reshape(logits.shape)
            logits += (1 - action_masks) * -1e5
        logits = logits_normalize(logits)
        return logits
