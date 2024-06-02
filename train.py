import multiprocessing as mp
import random

import fire

from src.args import ModelArgs
from src.policy import ActorCritic
from src.trainer import TrainerForPpo
from src.collector import BufferCollector
from src.env import create_multiprocess_env


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
    trainer = TrainerForPpo(args, policy=policy)
    if ckpt_file is not None:
        trainer.load(ckpt_file)

    for epoch in range(args.epochs):
        rollout_buffer = collector.collect()
        trainer.train(rollout_buffer)

        if epoch % 100 == 0:
            seed_set = [random.randint(0, 1e5) for _ in range(args.num_envs)]
            print(f"Epoch {epoch} of {args.epochs}. Resetting random seeds for environments: ", seed_set)
            collector.renew(create_multiprocess_env(seed_set))
            trainer.save(save_dir, f"model-{epoch}")

    trainer.save(save_dir, "model-final")


if __name__ == '__main__':
    fire.Fire(run)
