import random
import time

import numpy as np
import torch


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Timer:
    def __init__(self, total: int, episode: int = 1):
        self.total = total
        self.ticktock = 0
        self.last = None
        self.avg_time = 0
        self.episode = episode

    @staticmethod
    def format_clock(period):
        hour, minute, second = period // 3600, (period % 3600) // 60, period % 60
        return int(hour), int(minute), int(second)

    def step(self):
        if self.last is not None:
            period = time.time() - self.last
            self.avg_time = (self.avg_time * (self.ticktock - 1) + period) / self.ticktock
            h1, m1, s1 = self.format_clock(self.avg_time * (self.ticktock + 1))
            h2, m2, s2 = self.format_clock(self.avg_time * (self.total - self.ticktock))
            if self.ticktock % self.episode == 0:
                print(
                    f"STEP {self.ticktock}/{self.total} | USED: %02d:%02d:%02d | AVG %.2f s/it | "
                    f"ETA: %02d:%02d:%02d" % (h1, m1, s1, self.avg_time, h2, m2, s2)
                )
        self.last = time.time()
        self.ticktock += 1
        if self.ticktock == self.total:
            self.reset()

    def reset(self):
        self.ticktock = 0
        self.last = None
        self.avg_time = 0


def powmax(tensor, exponent=1, dim=-1, eps=7e-5):
    """ Similar to softmax, perform power max on vectors along one specific dimension. """
    numerator = torch.pow(tensor, exponent=exponent)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / (denominator + eps)


def masked_mean(x, mask=None, dim: int = -1, keepdim: bool = False, eps: float = 1e-12):
    if type(x) is torch.Tensor:
        if mask is None:
            mask = torch.full_like(x, fill_value=True)
        assert x.shape == mask.shape
        mask = mask.to(x.dtype)
        return torch.sum(
            x * mask, dim=dim, keepdim=keepdim
        ) / (torch.sum(mask, dim=dim, keepdim=keepdim) + eps)
    elif type(x) is np.ndarray:
        if mask is None:
            mask = np.full_like(x, fill_value=True)
        assert x.shape == mask.shape
        mask = mask.astype(x.dtype)
        return np.sum(
            x * mask, axis=dim, keepdims=keepdim
        ) / (np.sum(mask, axis=dim, keepdims=keepdim) + eps)
    else:
        raise TypeError
