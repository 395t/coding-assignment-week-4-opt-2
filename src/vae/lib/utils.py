import glob
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Logger:
    def __init__(self, log_dir):
        if os.path.exists(log_dir):
            assert len(glob.glob('events.out.tfevents.*')) == 0, (
                f'Tensorboard log already exists in {log_dir}')
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

    def flush(self):
        self.writer.flush()

    def log_scalar(self, tag, val, step):
        if hasattr(val, 'item'):
            val = val.item()
        self.writer.add_scalar(tag, val, global_step=step)

    def log_scalars(self, tag_value_dict, step: int):
        for tag, val in tag_value_dict.items():
            self.log_scalar(tag, val, step)

    def log_image(self, tag: str, img: torch.Tensor, step: int):
        assert img.ndim == 3
        self.writer.add_image(tag, img, global_step=step, dataformats='CHW')

    def log_image_grid(self, tag, imgs, step: int, **kwargs):
        assert imgs.ndim == 4
        img_grid = make_grid(imgs, **kwargs)
        self.log_image(tag, img_grid, step)

    def add_graph(self, *args, **kwargs):
        self.writer.add_graph(*args, **kwargs)


def normal_logpdf(x, mu=None, sig=None):
    D = x.shape[1]
    mu = torch.zeros_like(x) if mu is None else mu
    sig = torch.ones_like(x) if sig is None else sig
    return -0.5 * D * np.log(2*np.pi) - (
        torch.log(sig) + 0.5 / (sig**2) * ((x - mu) ** 2)).sum(dim=1)


def sumflat(x):
    return x.flatten(1).sum(dim=1)


def print_model_info(model: torch.nn.Module):
    cnt_train = sum(np.prod(p.shape) for p in model.parameters() if p.requires_grad)
    cnt_total = sum(np.prod(p.shape) for p in model.parameters())
    print(f'Model has {cnt_train} trainable parameters ({cnt_total} total). Training = {model.training}')


def tanh_to_uint8(x):
    x = ((x + 1) / 2 * 255.).byte().cpu()
    assert x.dtype == torch.uint8
    return x


# def test():
#     z = torch.randn(20)
#     mu = torch.randn(20) 
#     sig = torch.randn(20).clamp(0.0) + 0.5
#     logp = normal_logpdf(z[None], mu=mu[None], sig=sig[None])
#     print(logp)
#     from scipy.stats import multivariate_normal
#     mvn = multivariate_normal(mean=mu.numpy(), cov=np.diag(sig.numpy() ** 2))
#     print(mvn.logpdf(z.numpy()))
#     import ipdb; ipdb.set_trace()

# test()