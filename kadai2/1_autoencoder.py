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
img_size = 32 #CIFAR10デフォルト
epochs = 20
batch_size = 128


def transform_image(img):
    transforms = v2.Compose([
        v2.ToTensor(), 
        v2.ToDtype(torch.float32, scale=True),
    ])
    return transforms(img)


class CustomDataset(Dataset):
    def __init__(self):
        self.dataset = dst.CIFAR10(
            root="../data",
            train=True,
            download=False,
            transform=transform_image
        )

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        img = transform_image(img)
        label = self.dataset[idx][1]
        return img, label


class Encoder(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        # conv 層で次第にチャネル数↑・空間サイズ↓
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 32→16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # 16→8
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# 8→4
            nn.ReLU(True),
        )
        self.fc = nn.Linear(256*4*4, bottleneck_dim)

    def forward(self, x):
        h = self.conv(x)
        h_flat = h.view(x.size(0), -1)
        z = self.fc(h_flat)
        return z, h  # ボトルネックと中間マップを返す


class Decoder(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        self.fc = nn.Linear(bottleneck_dim, 256*4*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 4→8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8→16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),    # 16→32
            nn.Sigmoid(),  # 出力を [0,1] に
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 256, 4, 4)
        x_recon = self.deconv(h)
        return x_recon
    

class Autoencoder(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        self.encoder = Encoder(bottleneck_dim)
        self.decoder = Decoder(bottleneck_dim)

    def forward(self, x):
        z, h = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, h


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
    dataset = CustomDataset()

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
    model = Autoencoder(bottleneck_dim=256)
    model = nn.parallel.DistributedDataParallel(
        model.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    # others
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # leraning
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for img, label in data_loader:
            img = img.to(device)
            optimizer.zero_grad()
            recon, _, _ = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        if rank == 0:
            print(f"epoch: {epoch+1}, train loss: {avg_loss:.3f}")

    # 保存
    torch.save(model.module.state_dict(), "./1_autoencoder.pth")
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(world_size)
    spawn(main, args=(world_size,), nprocs=world_size, join=True)