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
    epochs = 100
    batch_size = 128 # 512
    lr = 2e-4
    z_dim = 100
    num_classes = 10

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

    # one-hot → チャネル埋め込み用ヘルパー
    def expand_label(y, H, W):
        # y: (B,) int ラベル → (B, C, H, W) one-hot レプリケート
        onehot = torch.zeros(y.size(0), num_classes, H, W, device=y.device)
        onehot.scatter_(1, y.view(-1,1,1,1).repeat(1,1,H,W), 1)
        return onehot

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                # input: (z_dim + num_classes)×1×1
                nn.ConvTranspose2d(z_dim + num_classes, 256, 7, 1, 0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # → 256×7×7
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # → 128×14×14
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # → 64×28×28
                nn.Conv2d(64, 1, 3, 1, 1),
                nn.Tanh()
                # 出力: 1×28×28
            )
        def forward(self, z, y):
            # z: (B, z_dim), y: (B,) int
            B = z.size(0)
            z = z.view(B, z_dim, 1, 1)
            y_map = expand_label(y, 1, 1)         # (B, C, 1, 1)
            x = torch.cat([z, y_map], dim=1)      # (B, z_dim+C,1,1)
            return self.net(x)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
            # input: (1 + num_classes)×28×28
            nn.Conv2d(1 + num_classes, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # → 64×14×14
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # → 128×7×7
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # → 256×4×4
            nn.Flatten(),
            nn.Linear(256*4*4, 1),
            nn.Sigmoid()
        )
        def forward(self, img, y):
            # img: (B,1,28,28), y: (B,)
            B = img.size(0)
            y_map = expand_label(y, 28, 28)       # (B,C,28,28)
            x = torch.cat([img, y_map], dim=1)    # (B,1+C,28,28)
            return self.net(x)

    # 初期化
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # model
    G_raw = Generator()
    D_raw = Discriminator()

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

    # 最適化手法 & 損失関数
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    # learning
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        total_loss_D = 0
        total_loss_G = 0
        for imgs, labels in dataloader:
            B = imgs.size(0)
            real_imgs = imgs.to(device)
            labels    = labels.to(device)

            # 1.Discriminator更新
            D.zero_grad()
            # 本物
            real_preds = D(real_imgs, labels)
            loss_real = criterion(real_preds, torch.ones(B,1,device=device))
            # 偽物
            z1 = torch.randn(B, z_dim, device=device)
            fake_imgs = G(z1, labels)
            fake_preds = D(fake_imgs.detach(), labels)
            loss_fake = criterion(fake_preds, torch.zeros(B,1,device=device))
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()
            total_loss_D += loss_D.item()

            # 2.Generator更新
            G.zero_grad()
            z2 = torch.randn(B, z_dim, device=device)
            fake_preds = D(G(z2, labels), labels)
            loss_G = criterion(fake_preds, torch.ones(B,1,device=device))
            loss_G.backward()
            opt_G.step()
            total_loss_G += loss_G.item()

        avg_loss_D = total_loss_D / len(dataloader)
        avg_loss_G = total_loss_G / len(dataloader)

        if rank == 0:
            print(f"epoch: {epoch+1}, loss_D: {avg_loss_D:.3f}, loss_G: {avg_loss_G:.3f}")


    # 保存
    torch.save(G.module.state_dict(), "6_3_cdcgan.pth")

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(world_size)
    spawn(main, args=(world_size,), nprocs=world_size, join=True)

