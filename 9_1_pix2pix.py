import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import time
import random
import glob
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
DATA_PATH = "/export/space0/yanai/media/foodimg128/omurice/"
img_size = 128
epochs = 100
batch_size = 128
lr=2e-4
lambda_L1   = 100


def transform_for_sketch(img):
    transforms = v2.Compose([
        v2.Grayscale(num_output_channels=1), #念の為
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,))   # [0,1]→[-1,1]
    ])
    return transforms(img)


def transform_for_real(img):
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
    def __init__(self, data_path):
        self.sketch_img_pths = glob.glob(os.path.join(data_path, "hed*.png"))
        self.real_img_pths = glob.glob(os.path.join(data_path, "img*.jpg"))
        assert len(self.sketch_img_pths)==len(self.real_img_pths)

    def __len__(self):
        return len(self.sketch_img_pths)

    def __getitem__(self, idx):
        sketch_img = read_image(self.sketch_img_pths[idx])
        sketch_img = transform_for_sketch(sketch_img)
        real_img = read_image(self.real_img_pths[idx])
        real_img = transform_for_real(real_img)
        return sketch_img, real_img
    

class Generator(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, ngf=64):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch,    ngf,   4, 2, 1, bias=False),  # 256→128
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf,      ngf*2, 4, 2, 1, bias=False),  # 128→64
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*2,    ngf*4, 4, 2, 1, bias=False),  # 64→32
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf*4,    ngf*8, 4, 2, 1, bias=False),  # 32→16
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True),
        )
        # decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),  # 16→32
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),  # 32→64
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),  # 64→128
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   out_ch,4, 2, 1),               # 128→256
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
    

# Discriminator(PatchGAN)
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch_A=1, in_ch_B=3, features=64):
        super().__init__()
        in_channels = in_ch_A + in_ch_B  # 1 + 3 = 4
        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels,      features, normalize=False),
            block(features,         features*2),
            block(features*2,       features*4),
            block(features*4,       features*8),
            nn.Conv2d(features*8, 1, 4, 1, 1)  # 出力はパッチごとのマップ
        )
    def forward(self, A, B):
        x = torch.cat([A, B], dim=1)  # (B,4,H,W)
        return self.model(x)


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
        DATA_PATH,
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

    # Losses & Labels
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1  = nn.L1Loss()
    real_label = 1.0
    fake_label = 0.0

    # model
    G_raw = Generator()
    D_raw = PatchDiscriminator()

    G_raw.apply(weights_init)
    D_raw.apply(weights_init)

    # これを追加したことで， Error detected in CudnnBatchNormBackward0を回避
    G_raw = nn.SyncBatchNorm.convert_sync_batchnorm(G_raw)
    D_raw = nn.SyncBatchNorm.convert_sync_batchnorm(D_raw)

    G = nn.parallel.DistributedDataParallel(
        G_raw.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )
    D = nn.parallel.DistributedDataParallel(
        D_raw.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # learning
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total_loss_D = 0
        total_loss_G = 0
        for i, (sketch, real) in enumerate(data_loader):
            sketch, real = sketch.to(device), real.to(device)

            # 1) Train Generator
            fake_img = G(sketch)
            pred_fake = D(sketch, fake_img)

            valid = torch.full_like(pred_fake, real_label, dtype=torch.float32, device=device)
            fake  = torch.full_like(pred_fake, fake_label,  dtype=torch.float32, device=device)

            loss_G_GAN = criterion_GAN(pred_fake, valid)
            loss_G_L1  = criterion_L1(fake_img, real) * lambda_L1
            loss_G     = loss_G_GAN + loss_G_L1

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            total_loss_G += loss_G.item()

            # 2) Train Discriminator
            pred_real  = D(sketch, real)
            loss_D_real = criterion_GAN(pred_real, valid)
            pred_fake  = D(sketch, fake_img.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            total_loss_D += loss_D.item()

        avg_loss_G = total_loss_G / len(data_loader)
        avg_loss_D = total_loss_D / len(data_loader)

        if rank == 0:
            print(f"epoch: {epoch+1}, loss_G: {avg_loss_G:.3f}, loss_D: {avg_loss_D:.3f}")

    # 保存
    torch.save(G.module.state_dict(), "9_1_pix2pix.pth")

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(world_size)
    spawn(main, args=(world_size,), nprocs=world_size, join=True)