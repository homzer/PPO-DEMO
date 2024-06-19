from dataclasses import dataclass


@dataclass
class ModelArgs:
    epochs: int = 4000
    observation_space: tuple = (3, 84, 84)
    num_actions: int = 4
    lr: float = 1e-5
    num_envs: int = 8
    n_collect_steps: int = 4096
    n_update_epochs: int = 6
    batch_size: int = 1024
    device: str = "cuda"
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    kl_coef: float = 50.0
    gamma: float = 0.94
    gae_lambda: float = 0.95
