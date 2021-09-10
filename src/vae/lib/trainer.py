import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor 
from lib.utils import Logger, normal_logpdf, sumflat, print_model_info, tanh_to_uint8
from lib.vae import VAE


def get_optimizer(name, params, **kwargs):
    optim_cls = {
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta,
        'adam': optim.Adam,
        'amsgrad': optim.Adam,
        'adamw': optim.AdamW,
    }[name]

    if name in ('adam', 'amsgrad'):
        return optim_cls(params, amsgrad=(name == 'amsgrad'), **kwargs)
    else:
        return optim_cls(params, **kwargs)

    
def train(*,
          out_dir: str, z_dim: int, optimizer: str,
          epochs: int = 50, use_cuda: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if use_cuda else 'cpu')
    dataset = SVHN(root='svhn', split='train', download=True, transform=ToTensor())
    eval_dataset = SVHN(root='svhn', split='test', download=True, transform=ToTensor())
    model = VAE(z_dim).to(device)
    print_model_info(model)
    opt = get_optimizer(optimizer, model.parameters(), lr=0.005)
    logger = Logger(out_dir)
    total_step = 0

    for epoch in range(1, epochs+1):
        loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
        for x, _ in loader:
            total_step += 1
            x = x.to(device) * 2 - 1.0
            z, mu, sigma, x_hat = model(x)

            loss_rec = 0.5 * sumflat((x - x_hat) ** 2)
            loss_kl = normal_logpdf(z, mu, sigma) - normal_logpdf(z)
            loss = (loss_rec + loss_kl).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if total_step % 10 == 0:
                logger.log_scalars({
                    'train/loss': loss.cpu().item(),
                    'train/loss_rec': loss_rec.cpu().mean().item(),
                    'train/loss_kl': loss_kl.cpu().mean().item(),

                }, total_step)

            print(f'\rep {epoch:02d} step {total_step:03d} loss {loss:.2f} '
                  f'loss_rec {loss_rec.mean():.2f} '
                  f'loss_kl {loss_kl.mean():.2f} ',
                  end='', flush=True)

        print()
        evaluate(model=model, dataset=eval_dataset, logger=logger, step=total_step, use_cuda=use_cuda)

@torch.no_grad()
def evaluate(*, model: torch.nn.Module, dataset, logger: Logger, step: int, use_cuda: bool):
    device = torch.device('cuda' if use_cuda else 'cpu')
    loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, drop_last=False)

    model.eval()
    losses = []
    for i, (x, _) in enumerate(loader):
        x = x.to(device) * 2 - 1.0
        z, mu, sigma, x_hat = model(x)

        loss_rec = 0.5 * sumflat((x - x_hat) ** 2)
        loss_kl = normal_logpdf(z, mu, sigma) - normal_logpdf(z)
        loss = (loss_rec + loss_kl)
        losses.append(loss.cpu())

        if i == 0:
            n = 6
            samples = model.decoder(torch.randn(n**2, 10, device=device))
            logger.log_image_grid('reconstructions', tanh_to_uint8(x_hat[:n**2]), step, nrow=n)
            logger.log_image_grid('samples', tanh_to_uint8(samples), step, nrow=n)

    losses = torch.cat(losses)
    logger.log_scalar('eval/loss', losses.mean().item(), step)
    model.train()