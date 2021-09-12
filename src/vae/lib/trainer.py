import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor 
from lib.utils import Logger, normal_logpdf, sumflat, print_model_info, tanh_to_uint8, get_optimizer
from lib.vae import VAE


def train(hp):
    os.makedirs(hp.out_dir, exist_ok=True)
    device = torch.device('cuda' if hp.use_cuda else 'cpu')
    dataset = SVHN(root='svhn', split='train', download=True, transform=ToTensor())
    eval_dataset = SVHN(root='svhn', split='test', download=True, transform=ToTensor())
    model = VAE(hp.z_dim).to(device)
    print_model_info(model)
    opt = get_optimizer(hp.opt_name, model.parameters(), lr=hp.lr, **hp.opt_kwargs)
    logger = Logger(hp.out_dir)
    total_step = 0
    error_occured = False

    start_time = time.time()
    stats = {
        'loss': [],
        'loss_kl': [],
        'loss_rec': [],
        'eval_loss': [],
        'start_time': start_time,
        'epoch_times': [],
    }
    for epoch in range(1, hp.epochs+1):
        loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True)
        for x, _ in loader:
            total_step += 1
            x = x.to(device) * 2 - 1.0
            z, mu, sigma, x_hat = model(x)

            loss_rec = 0.5 * sumflat((x - x_hat) ** 2)
            loss_kl = normal_logpdf(z, mu, sigma) - normal_logpdf(z)
            loss = (loss_rec + loss_kl).mean()
            if torch.isnan(loss).item():
                error_occured = True
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

            if total_step % 10 == 0:
                stats['loss'].append(loss.cpu().item())
                stats['loss_rec'].append(loss_rec.cpu().mean().item())
                stats['loss_kl'].append(loss_kl.cpu().mean().item())
                logger.log_scalars({
                    'train/loss': stats['loss'][-1],
                    'train/loss_rec': stats['loss_rec'][-1],
                    'train/loss_kl': stats['loss_kl'][-1],

                }, total_step)

                print(f'\rep {epoch:02d} step {total_step:03d} '
                    f'loss {stats["loss"][-1]:.2f} '
                    f'loss_rec {stats["loss_rec"][-1]:.2f} '
                    f'loss_kl {stats["loss_kl"][-1]:.2f} '
                    f'({time.time() - start_time:.2f} sec) '
                    '                   ',
                    end='', flush=True)

        print()
        if error_occured:
            print('NaN detected -- Ending training!')
            break
        stats['epoch_times'].append(time.time())
        eval_loss = evaluate(model=model, dataset=eval_dataset, logger=logger,
                             step=total_step, epoch=epoch, device=device, hparams=hp)
        stats['eval_loss'].append(eval_loss.cpu().mean().item())

        if epoch % hp.ckpt_freq == 0 or epoch == hp.epochs:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'total_step': total_step,
                    'stats': stats,
                    'hparams': vars(hp),
                },
                os.path.join(hp.out_dir, f'ckpt_ep={epoch:03d}.pt'))

    end_time = time.time()
    with open(os.path.join(hp.out_dir, 'FINISHED'), 'w') as f:
        f.write(f'Started: {start_time}\n')
        f.write(f'Finished: {end_time}\n')
        f.write(f'Total time: {end_time - start_time:.2f}\n')


@torch.no_grad()
def evaluate(*, model: torch.nn.Module, dataset, logger: Logger, step: int, epoch: int, device, hparams):
    loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False, drop_last=False)

    model.eval()
    losses = []
    for i, (x, _) in enumerate(loader):
        x = x.to(device) * 2 - 1.0
        z, mu, sigma, x_hat = model(x)

        loss_rec = 0.5 * sumflat((x - x_hat) ** 2)
        loss_kl = normal_logpdf(z, mu, sigma) - normal_logpdf(z)
        loss = loss_rec + loss_kl
        losses.append(loss.cpu())

        if i == 0 and (epoch % hparams.sample_freq == 0 or epoch == hparams.epochs):
            n = 6
            samples = model.decoder(torch.randn(n**2, hparams.z_dim, device=device))
            logger.log_image_grid('reconstructions', tanh_to_uint8(x_hat[:n**2]), step, nrow=n)
            logger.log_image_grid('samples', tanh_to_uint8(samples), step, nrow=n)

    losses = torch.cat(losses)
    logger.log_scalar('eval/loss', losses.mean().item(), step)
    model.train()
    return losses