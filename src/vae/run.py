import fire
from lib import trainer


def train_single(*, out_dir: str, optimizer: str, z_dim: int, use_cuda: bool = False):
    trainer.train(out_dir=out_dir, z_dim=10, optimizer=optimizer, use_cuda=use_cuda)


if __name__ == '__main__':
    fire.Fire({
        'train_single': train_single,
    })