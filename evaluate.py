import fire
import torch

from src.args import ModelArgs
from src.evaluator import SnakeGameEvaluator
from src.policy import ActorCritic
from src.utils import set_seed


def run():
    set_seed()
    args = ModelArgs()
    model = ActorCritic(args.observation_space, args.num_actions)
    for ckpt_file in ['results/model-40.bin', 'results/model-480.bin', 'results/model-730.bin']:
        model.load(ckpt_file)
        model.cuda()
        evaluator = SnakeGameEvaluator(model)
        evaluator.forward(20)


if __name__ == '__main__':
    fire.Fire(run)
