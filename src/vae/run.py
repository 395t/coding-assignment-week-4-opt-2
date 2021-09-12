import os
import shutil
import traceback
import fire
from lib import trainer
from lib.utils import HParams


def compare_opts(lrs=None):
    if lrs is None:
        lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    elif not isinstance(lrs, tuple):
        lrs = (lrs,)
    print(f'Running for LRs: {lrs}')
    opt_names = ['adagrad', 'adadelta', 'adam', 'amsgrad', 'adamw']
    for lr in lrs:
        for opt_name in opt_names: 
            out_dir = f'results/compare_opts/lr={lr}/{opt_name}'
            os.makedirs(out_dir, exist_ok=True)
            if len(os.listdir(out_dir)) > 0:
                shutil.rmtree(out_dir)
                os.makedirs(out_dir, exist_ok=False)
            hparams = HParams(out_dir=out_dir,
                              z_dim=20,
                              lr=lr,
                              epochs=150,
                              ckpt_freq=30,
                              sample_freq=5,
                              opt_name=opt_name)
            print(f'\n[compare_opts] Running for lr={hparams.lr} optimizer={hparams.opt_name}')
            try:
                trainer.train(hparams)
            except KeyboardInterrupt:
                print(f'[compare_opts] Run skipped manually.')
                raise KeyboardInterrupt
            except Exception:
                print(traceback.format_exc())
                print(f'[compare_opts] Unknown error encountered for lr={hparams.lr} optimizer={hparams.opt_name}')


if __name__ == '__main__':
    fire.Fire({
        'compare_opts': compare_opts,
    })