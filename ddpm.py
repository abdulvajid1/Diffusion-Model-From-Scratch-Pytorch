import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import logging

import torch.nn.functional as F

# output_size = ⌊ (input_size − kernel_size + 2 × padding) / stride ⌋ + 1
# max pool out  = H_out = ⌊ (H − 2) / 2 ⌋ + 1

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, 
                    datefmt="%I: %M: %S")

class Diffusion(nn.Module):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.linear_beta_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def forward_diffusion_sample(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # unsqueezing to match same shape as image so it will broadcast
        one_minus_sqrt_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + one_minus_sqrt_alpha_hat * noise, noise
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    additional_noise = torch.randn_like(x)
                else:
                    additional_noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (
        x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise
    ) + torch.sqrt(beta) * additional_noise

        
        model.train()
        x = (x.clamp(-1, 1) + 1)/2
        x = (x*255).type(torch.uint8)
        return x


class UNET(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)

        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)

        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)

        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128) # 512 cuz we have 256 skip connection
        self.sa4 = SelfAttention(128, 16)

        self.up2 = Up(256, 64) # 126 skip
        self.sa5 = SelfAttention(64, 32)

        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1) # 1x1


    def pos_encoding(self, t, channel):
        inv_freq = 1 / (10000 ** (torch.arange(0, channel, 2, device=self.device).float() / channel))
        pos_enc_a = torch.sin(t.repeat(1, channel // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channel // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.to(self.device)
    
    def forward(self, x, t):
        t = t.float().unsqueeze(-1)
        t_emb = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x) # (b, 3, 64, 64) -> (b, 3, 64, 64)

        x2 = self.down1(x1, t_emb) # 64 -> 32
        x2 = self.sa1(x2) # 32 -> 32

        x3 = self.down2(x2, t_emb) # 32 -> 16
        x3 = self.sa2(x3) # 16 -> 16

        x4 = self.down3(x3, t_emb) # 16 -> 8
        x4 = self.sa3(x4) # 8 -> 8

        x4 = self.bot1(x4) # 8
        x4 = self.bot2(x4) # 8
        x4 = self.bot3(x4) # 8

        x = self.up1(x4, x3, t_emb) # 8 -> 16
        x = self.sa4(x) # 16

        x = self.up2(x, x2, t_emb) # 16 -> 32
        x = self.sa5(x) # 32

        x = self.up3(x, x1, t_emb) # 32 -> 64
        x = self.sa6(x) # 64

        output = self.outc(x) # 1x1
        return output
    

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        self.residual = residual
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return F.gelu(self.double_conv(x))
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch=in_channels, out_ch=in_channels, residual=True),
            DoubleConv(in_ch=in_channels, out_ch=out_channels, residual=False)
            )
        
        self.embed_layer = nn.Sequential( # a projection Layer to feature maps dim , [b, embed_dim] -> [b, channels_dim] so each channel will get scalar pos info
            nn.Linear(emb_dim, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, t):
        x = self.max_pool(x)
        emb = self.embed_layer(t)[:, :, None, None] #.repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb # each channel will get seperate pos number that get added
    
class Up(nn.Module):
    def __init__(self, in_channel, out_channel, emb_dim=256) -> None:
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Sequential(
            DoubleConv(in_ch=in_channel, out_ch=out_channel),
            DoubleConv(in_ch=out_channel, out_ch=out_channel)
        )

        self.emb_layer = nn.Linear(in_features=emb_dim, out_features=out_channel)

    def forward(self, x, x_skip, t_emb):
        x = self.up(x)
        x = torch.concat([x_skip, x], dim=1)
        x = self.conv(x)
        t_emb = self.emb_layer(t_emb)[:, :, None, None]
        return x + t_emb
    

class SelfAttention(nn.Module):
    def __init__(self, channel_dim, size) -> None:
        super().__init__()
        self.size = size
        self.channel_dim = channel_dim 
        self.mha = nn.MultiheadAttention(channel_dim, num_heads=4)
        self.norm = nn.LayerNorm([channel_dim])
        self.final_layer = nn.Sequential(
            nn.LayerNorm([channel_dim]),
            nn.Linear(channel_dim, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, channel_dim)
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.channel_dim, self.size * self.size).transpose(1, 2) # num, channel, siz*siz -> num, siz*siz, channel
        norm_x = self.norm(x)
        attn_x, _= self.mha(norm_x, norm_x, norm_x)
        attn_x = attn_x + x
        x = self.final_layer(attn_x) + attn_x
        return x.transpose(1, 2).contiguous().view(-1, self.channel_dim, self.size, self.size)










