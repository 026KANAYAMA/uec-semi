import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import time
import random
import glob
import itertools
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.multiprocessing.spawn import spawn
import torchvision.models as models
import torchvision.datasets as dst
from torchvision.io import read_image
from torchvision.transforms import v2


# settings
DATA_PATH_A = "/export/space0/yanai/media/foodimg128/rice/"
DATA_PATH_B = "/export/space0/yanai/media/foodimg128/unadon/"
img_size = 128
epochs = 100
batch_size = 32
lr=2e-4
lambda_L1 = 100


def transform_image(img):
    transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    return transforms(img)


# 初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CustomDataset(Dataset):
    def __init__(self, data_path_a, data_path_b):
        self.img_a_paths = glob.glob(os.path.join(data_path_a, "img*.jpg"))
        self.img_b_pths = glob.glob(os.path.join(data_path_b, "img*.jpg"))

    def __len__(self):
        return len(self.img_a_paths)
    
    def __getitem__(self, idx):
        img_a = read_image(self.img_a_paths[idx])
        img_a = transform_image(img_a)
        idx = random.randint(0, len(self.img_b_pths) - 1)
        img_b = read_image(self.img_b_pths[idx]) # bはランダム
        img_b = transform_image(img_b)
        return img_a, img_b


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            # down‑sample ×2
            *self._down(ngf, ngf*2),
            *self._down(ngf*2, ngf*4),
            # residual blocks
            *[ResnetBlock(ngf*4) for _ in range(n_blocks)],
            # up‑sample ×2
            *self._up(ngf*4, ngf*2),
            *self._up(ngf*2, ngf),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7), nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)
    def _down(self, in_c, out_c):
        return [nn.Conv2d(in_c, out_c, 3, 2, 1),
                nn.InstanceNorm2d(out_c), nn.ReLU(True)]
    def _up(self, in_c, out_c):
        return [nn.ConvTranspose2d(in_c, out_c, 3, 2, 1, 1),
                nn.InstanceNorm2d(out_c), nn.ReLU(True)]
    def forward(self, x): return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4; pad = 1
        seq = [nn.Conv2d(in_ch, ndf, kw, 2, pad), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            seq += [nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, 2, pad),
                    nn.InstanceNorm2d(ndf*nf_mult), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(ndf*nf_mult, 1, kw, 1, pad)]  # 70×70 map
        self.model = nn.Sequential(*seq)
    def forward(self, x): return self.model(x)


def main(rank, world_size):
    torch.autograd.set_detect_anomaly(True)
    # 環境変数，プロセスグループ初期化
    os.environ['MASTER_ADDR'] = '127.0.0.1' # ランデブーサーバー（プロセス間を仲介）
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )

    # seed
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # data
    dataset = CustomDataset(
        DATA_PATH_A,
        DATA_PATH_B
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # 代わりにepoch内で設定
        sampler=sampler,
        num_workers=1,
        pin_memory=True
    )

    # model
    G_X2Y_raw = Generator()
    F_Y2X_raw = Generator()
    D_x_raw = PatchDiscriminator()
    D_y_raw = PatchDiscriminator()

    G_X2Y_raw.apply(weights_init)
    F_Y2X_raw.apply(weights_init)
    D_x_raw.apply(weights_init)
    D_y_raw.apply(weights_init)

    G_X2Y = nn.parallel.DistributedDataParallel(
        G_X2Y_raw.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )
    F_Y2X = nn.parallel.DistributedDataParallel(
        F_Y2X_raw.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )
    D_X = nn.parallel.DistributedDataParallel(
        D_x_raw.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )
    D_Y = nn.parallel.DistributedDataParallel(
        D_y_raw.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    # others
    opt_G = torch.optim.Adam(itertools.chain(G_X2Y.parameters(), F_Y2X.parameters()), 2e-4, (0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_Y.parameters(), D_X.parameters()),   2e-4, (0.5, 0.999))

    criterion_GAN = nn.MSELoss()   # LSGAN
    criterion_L1  = nn.L1Loss()

    # learning
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total_loss_D = 0
        total_loss_G = 0
        for real_X, real_Y in data_loader:
            real_X, real_Y = real_X.float().to(device), real_Y.float().to(device)

            # 1.Train Generator
            fake_Y = G_X2Y(real_X)
            rec_X  = F_Y2X(fake_Y)
            fake_X = F_Y2X(real_Y)
            rec_Y  = G_X2Y(fake_X)

            loss_G = (
                criterion_GAN(D_Y(fake_Y), torch.ones_like(D_Y(fake_Y))) +
                criterion_GAN(D_X(fake_X), torch.ones_like(D_X(fake_X))) +
                10 * (criterion_L1(rec_X, real_X) + criterion_L1(rec_Y, real_Y))
            ) / 2
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            total_loss_G += loss_G.item()

            # 2.Train Disc
            loss_D_Y = 0.5 * (
                criterion_GAN(D_Y(real_Y), torch.ones_like(D_Y(real_Y))) +
                criterion_GAN(D_Y(fake_Y.detach()), torch.zeros_like(D_Y(fake_Y)))
            )
            loss_D_X = 0.5 * (
                criterion_GAN(D_X(real_X), torch.ones_like(D_X(real_X))) +
                criterion_GAN(D_X(fake_X.detach()), torch.zeros_like(D_X(fake_X)))
            )
            opt_D.zero_grad()
            loss_D = loss_D_Y + loss_D_X
            (loss_D_Y + loss_D_X).backward()
            opt_D.step()
            total_loss_D += loss_D.item()

        avg_loss_G = total_loss_G / len(data_loader)
        avg_loss_D = total_loss_D / len(data_loader)

        if rank == 0:
            print(f"epoch: {epoch+1}, loss_G: {avg_loss_G:.3f}, loss_D: {avg_loss_D:.3f}")


    # 保存
    torch.save(G_X2Y.module.state_dict(), "10_1_cycle_gan.pth")
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(world_size)
    spawn(main, args=(world_size,), nprocs=world_size, join=True)