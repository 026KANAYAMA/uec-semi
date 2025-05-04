import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import time
import random
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
import torchvision.datasets as dst
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader, DistributedSampler


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

    # settings
    batch_size = 128 # 512

    # data
    dataroot = "/export/data/dataset/mnist_10k"

    dataset = dst.ImageFolder( #ImageFolderはGrayでもRGBで読み込む
        root=dataroot,
        transform=v2.Compose([
            v2.Grayscale(num_output_channels=1),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize([0.5], [0.5])
        ]),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # 代わりにepoch内で設定
        sampler=sampler,
        num_workers=1,
        pin_memory=True
    )

    # settings
    epochs = 100
    z_dim = 100

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d( # 100*1*1 -> 256*7*7
                    in_channels=z_dim,
                    out_channels=256,
                    kernel_size=7,
                    stride=1,
                    padding=0,
                    bias=False
                ),
            nn.BatchNorm2d(256), nn.ReLU(False), # 256*7*7 -> 128*12*14
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(False),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            )
        def forward(self, z):
            return self.net(z.view(-1, z_dim, 1, 1))
        
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                # 1×28×28 → 128×14×14
                nn.Conv2d(1, 128, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=False),
                # 128×14×14 → 256×7×7
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=False),
                # 平坦化して出力
                nn.Flatten(),
                nn.Linear(256*7*7, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)
        
    # exp
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    G_raw = Generator()
    D_raw = Discriminator()

    # 初期化
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

    # DCGAN論文にてb1=0.5，lr=2e-4が指定
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    # learning
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total_loss_D = 0
        total_loss_G = 0
        for real_imgs, _ in dataloader:
            real = real_imgs.to(device) 

            # 1.Generator更新
            for p in D.parameters():             # D の勾配を止める
                p.requires_grad_(False)

            G.zero_grad()
            z2 = torch.randn(real.size(0), z_dim, device=device)
            fake2 = G(z2)
            pred2 = D(fake2)
            loss_G = criterion(pred2, torch.ones_like(pred2))
            loss_G.backward()
            opt_G.step()
            total_loss_G += loss_G.item()

            # 2.Discriminator更新
            for p in D.parameters():
                p.requires_grad_(True)
            D.zero_grad()
            real_labels = torch.ones(real.size(0),1, device=device) #本物ラベル
            pred_real = D(real)
            loss_real = criterion(pred_real, real_labels)

            fake_labels = torch.zeros(real.size(0),1, device=device)
            pred_fake = D(fake2.detach())
            loss_fake = criterion(pred_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()
            total_loss_D += loss_D.item()

        avg_loss_D = total_loss_D / len(dataloader)
        avg_loss_G = total_loss_G / len(dataloader)

        if rank == 0:
            print(f"epoch: {epoch+1}, loss_D: {avg_loss_D:.3f}, loss_G: {avg_loss_G:.3f}")

    # 保存
    torch.save(G.module.state_dict(), "6_2_dcgan.pth")

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(world_size)
    spawn(main, args=(world_size,), nprocs=world_size, join=True)
    
    