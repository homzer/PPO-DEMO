import multiprocessing as mp
import os
import random

import fire
import torch

from src.args import ModelArgs
from src.policy import ActorCritic
from src.trainer import TrainerForActorCritic
from src.collector import BufferCollector
from src.env import create_multiprocess_env
from src.utils import Timer


def run(
        ckpt_file=None,  # "results/model-280.bin"
        save_dir: str = 'results'
):
    num_envs = mp.cpu_count()
    args = ModelArgs(
        epochs=1000,
        num_envs=num_envs,
        n_update_epochs=4,
        n_collect_steps=1024,
        batch_size=512
    )
    print(f"Found {args.num_envs} CPUs.")
    policy = ActorCritic(
        observation_space=args.observation_space,
        num_actions=args.num_actions
    )
    policy.cuda()
    env = create_multiprocess_env(
        [random.randint(0, 1e5) for _ in range(args.num_envs)]
    )
    collector = BufferCollector(args, env=env, policy=policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    trainer = TrainerForActorCritic(args, policy=policy, optimizer=optimizer)
    if ckpt_file is not None:
        trainer.load(ckpt_file)

    timer = Timer(args.epochs)
    for epoch in range(args.epochs):
        rollout_buffer = collector.collect()
        trainer_outputs = None
        for _ in range(args.n_update_epochs):
            for rollout_data in rollout_buffer.get(args.batch_size):
                trainer_outputs = trainer.forward(rollout_data)
        timer.step()
        print("\n===============================")
        print("train/entropy_loss", trainer_outputs.entropy_loss)
        print("train/policy_gradient_loss", trainer_outputs.policy_loss)
        print("train/value_loss", trainer_outputs.value_loss)
        print("train/loss", trainer_outputs.loss)
        print("===============================")

        if epoch % 100 == 0:
            seed_set = [random.randint(0, 1e5) for _ in range(args.num_envs)]
            print(f"Epoch {epoch} of {args.epochs}. Resetting random seeds for environments: ", seed_set)
            collector.renew(create_multiprocess_env(seed_set))
            trainer.save(os.path.join(save_dir, str(epoch)))

    trainer.save(os.path.join(save_dir, str(args.epochs)))


if __name__ == '__main__':
    fire.Fire(run)
