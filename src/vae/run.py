import functools
import glob
import os
import shutil
import traceback

# For plots
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
import seaborn as sns; sns.set(context='paper', style='whitegrid', font_scale=1.5, font='Times New Roman')

import fire
import numpy as np
import torch
from lib import trainer
from lib.utils import HParams, OPT_NAMES


_MARKERS = ['o', '^', 's', 'x', '+']


_OPT_LABELS = {
    'adagrad': 'Adagrad',
    'adadelta': 'Adadelta',
    'adam': 'Adam',
    'amsgrad': 'Amsgrad',
    'adamw': 'AdamW'}


def compare_opts(lrs=None):
    if lrs is None:
        lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    elif not isinstance(lrs, tuple):
        lrs = (lrs,)
    print(f'Running for LRs: {lrs}')
    for lr in lrs:
        for opt_name in OPT_NAMES: 
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


def plot(fn_prefix, results, smooth=None, skip_first=None, set_kwargs=None):
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    for i, name in enumerate(results):
        result = results[name]
        x, y = result['x'], result['y']
        if smooth is not None:
            y[smooth-1:] = np.convolve(y, np.ones(smooth), 'valid') / smooth
        if skip_first is not None:
            x = x[skip_first:]
            y = y[skip_first:]
        ax.plot(x, y, label=_OPT_LABELS[name])
    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_title(f'Training Loss')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Negative ELBO')
    if set_kwargs is not None:
        ax.set(**set_kwargs)
    ax.legend(loc='upper right')
    fig.savefig(fn_prefix+'.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(fn_prefix+'.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curves(split, lrs):
    assert split in ('train', 'test')
    for lr in lrs:
        results = {}
        for opt_name in OPT_NAMES:
            out_dir = f'results/compare_opts/lr={lr}/{opt_name}'
            if not os.path.exists(os.path.join(out_dir, 'FINISHED')):
                print(f'Skipping lr={lr} due to missing runs.')
                continue

            fn = sorted(glob.glob(os.path.join(out_dir, 'ckpt_*')))[-1]
            print(f'Processing {fn}')
            stats = torch.load(fn)['stats']
            if split == 'train':
                results[opt_name] = {
                    'x': 10 * np.arange(1, len(stats['loss']) + 1),
                    'y': stats['loss'],
                }
            elif split == 'test':
                results[opt_name] = {
                    'x': 10 * np.arange(1, len(stats['eval_loss']) + 1),
                    'y': np.array(stats['eval_loss']),
                }

        plot(f'plots/{split}_loss..lr={lr}',
            results,
            smooth=100 if split == 'train' else 10,
            skip_first=100 if split == 'train' else 10,
            set_kwargs={
                'ylim': (40, 120) if split == 'train' else (50, 160),
                'title': f'Training Loss (lr={lr})' if split == 'train' else f'Test Loss (lr={lr})',
                'xlabel': 'Step' if split == 'train' else 'Epoch',
                'ylabel': 'Negative ELBO',
            })

def print_final_losses(lrs):
    results = {}
    for opt_name in OPT_NAMES:
        results[opt_name] = {}
        for lr in lrs:
            out_dir = f'results/compare_opts/lr={lr}/{opt_name}'
            if not os.path.exists(os.path.join(out_dir, 'FINISHED')):
                print(f'Skipping lr={lr} due to missing runs.')
                continue

            fn = sorted(glob.glob(os.path.join(out_dir, 'ckpt_*')))[-1]
            print(f'Processing {fn}')
            stats = torch.load(fn)['stats']
            results[opt_name][lr] = stats['eval_loss'][-1]

    for opt_name in OPT_NAMES:
        print(f'Optimizer {opt_name}:')
        for lr in lrs:
            print(f'  -> lr={lr}: {results[opt_name][lr]:.2f}')


if __name__ == '__main__':
    fire.Fire({
        'compare_opts': compare_opts,
        'plot_train_loss': functools.partial(plot_loss_curves, 'train'),
        'plot_test_loss': functools.partial(plot_loss_curves, 'test'),
        'print_final_losses': print_final_losses,
    })