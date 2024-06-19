import multiprocessing as mp
import os
import random

import fire
import torch

from src.args import ModelArgs
from src.collector import ParallelBufferCollector
from src.env import SnakeEnv
from src.policy import ActorCritic
from src.trainer import KLDivTrainerForActorCritic
from src.utils import Timer, set_seed


def run(
        ckpt_file: str = None,  # "results/model-280.bin"
        save_dir: str = 'results',
        epochs: int = 1000,
        n_update_epochs: int = 4,
        n_collect_steps: int = 4096,
        batch_size: int = 512
):
    set_seed()
    num_envs = mp.cpu_count()
    args = ModelArgs(
        epochs=epochs,
        num_envs=num_envs,
        n_update_epochs=n_update_epochs,
        n_collect_steps=n_collect_steps,
        batch_size=batch_size
    )
    print(f"Found {args.num_envs} CPUs.")
    policy = ActorCritic(
        observation_space=args.observation_space,
        num_actions=args.num_actions
    )
    policy.cuda() if torch.cuda.is_available() else policy.cpu()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    envs = [SnakeEnv(seed=random.randint(0, 1000)) for _ in range(args.num_envs)]
    collector = ParallelBufferCollector(args, envs=envs, policy=policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    trainer = KLDivTrainerForActorCritic(args, policy=policy, optimizer=optimizer)
    if ckpt_file is not None:
        trainer.load(ckpt_file)

    timer = Timer(args.epochs)
    for epoch in range(args.epochs):
        rollout_buffer = collector.forward()
        trainer_outputs = None
        for _ in range(args.n_update_epochs):
            for rollout_data in rollout_buffer.get(args.batch_size):
                trainer_outputs = trainer.forward(rollout_data)
        timer.step()
        print("\n===============================")
        print("train/loss", trainer_outputs.loss)
        print("train/kl_loss", trainer_outputs.kl_loss)
        print("train/value_loss", trainer_outputs.value_loss)
        print("===============================")

        if epoch % 100 == 0:
            seed_set = [random.randint(0, 1000) for _ in range(args.num_envs)]
            print(f"Epoch {epoch} of {args.epochs}. Resetting random seeds for environments: ", seed_set)
            collector.renew([SnakeEnv(seed=random.randint(0, 1000)) for _ in range(args.num_envs)])
            trainer.save(os.path.join(save_dir, str(epoch)))

    trainer.save(os.path.join(save_dir, str(args.epochs)))


if __name__ == '__main__':
    fire.Fire(run)
