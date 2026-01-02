from utils import setup_logging, get_data, save_images
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


def train(args):
    setup_logging(args.run_name)
    dataloader = get_data(args)
    model = UNET().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
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
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=(epoch*data_len)+step)
        
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_names, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}.jpg"))


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
    args.lr = 3e-4
    logging.info("starting")
    train(args)

if __name__ == "__main__":
    launch()
