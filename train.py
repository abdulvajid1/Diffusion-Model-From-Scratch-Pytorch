from utils import setup_logging, get_data, save_images, save_model, load_model
from ddpm import UNET, Diffusion
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
import torch.nn as nn
import logging
from tqdm import tqdm
import torch

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, 
                    datefmt="%I: %M: %S")

import copy
import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.model.named_parameters())
        model_params = dict(model.named_parameters())

        for name in ema_params:
            ema_params[name].data.mul_(self.decay)
            ema_params[name].data.add_(
                model_params[name].data * (1 - self.decay)
            )

    def to(self, device):
        self.model.to(device)


def train(args):
    setup_logging(args.run_name)
    dataloader = get_data(args)
    model = UNET().to(args.device)
    ema = EMA(model, decay=0.998)
    ema.to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    global_step = 0
    if args.load_model:
        global_step = load_model(model, optimizer, global_step=args.global_steps, args=args)
    logging.info(f"Starting from step {global_step}")
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=args.device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    data_len = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}")
        pbar = tqdm(dataloader)

        for step, (images, _) in enumerate(pbar):
            images = images.to(args.device)
            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            x_t, noice = diffusion.forward_diffusion_sample(images, t)
            predicted_noice = model(x_t, t)
            loss = mse_loss(noice, predicted_noice)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=(epoch*data_len)+step+global_step)
        
        sampled_images = diffusion.sample(ema.model, n=4)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch+global_step//509}.jpg"))
        save_model(ema.model, optimizer, global_step=(epoch*data_len)+step+global_step, run_name=args.run_name)


import argparse

def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = 'ddpm_unconditional'
    args.epochs = 1000
    args.batch_size = 16
    args.img_size = 64
    args.dataset_path = "dataset"
    args.device = 'cuda'
    args.lr = 2e-4
    args.load_model = False
    args.global_steps = 103718
    logging.info("starting")
    train(args)

if __name__ == "__main__":
    launch()
