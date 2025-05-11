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
from torchvision.datasets import VOCSegmentation
from torchvision.io import read_image
from torchvision.transforms import v2


epochs = 20
batch_size = 32


def transform_image(img):
    transforms = v2.Compose([
        v2.Resize((256,256)),
        v2.ToTensor(), 
        v2.ToDtype(torch.float32, scale=True),
    ])
    return transforms(img)

def transform_mask(mask):
    # マスクは最近傍補間でリサイズ
    transform_mask = v2.Compose([
        v2.ToTensor(), 
        v2.Resize((256,256), interpolation=v2.InterpolationMode.NEAREST),
    ])
    return transform_mask(mask)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels,       middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels,   middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels,
                            kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc1 = _EncoderBlock(3,   64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128,256)
        self.enc4 = _EncoderBlock(256,512, dropout=True)

        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4   = _DecoderBlock(1024, 512, 256)
        self.dec3   = _DecoderBlock(512,  256, 128)
        self.dec2   = _DecoderBlock(256,  128,  64)

        # 最下段のデコーダ
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)   # 32→32→32→pool→16
        enc2 = self.enc2(enc1)  # 16→16→16→pool→8
        enc3 = self.enc3(enc2)  # 8 →8 →8 →pool→4
        enc4 = self.enc4(enc3)  # 4 →4 →4 →pool→2

        center = self.center(enc4)  # 2→2→2→up→4

        # skip connection + upsample (bilinear)
        dec4 = self.dec4(torch.cat([
            center,
            F.interpolate(enc4, size=center.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1))  # →4

        dec3 = self.dec3(torch.cat([
            dec4,
            F.interpolate(enc3, size=dec4.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1))  # →8

        dec2 = self.dec2(torch.cat([
            dec3,
            F.interpolate(enc2, size=dec3.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1))  # →16

        dec1 = self.dec1(torch.cat([
            dec2,
            F.interpolate(enc1, size=dec2.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1))  # →32

        out = self.final(dec1)  # [B, num_classes, 32, 32]
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
    

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
    dataset = VOCSegmentation(
            root="../data",
            year="2012",
            image_set="train",
            download=False,
            transform=transform_image,
            target_transform=transform_mask
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
    num_classes = 21   # 例：クラス数
    model = UNet(num_classes=num_classes)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(
        model.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    # others
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # learning
    print("学習開始")
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        for imgs, masks in data_loader:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        if rank == 0:
            print(f"epoch: {epoch+1}, train loss: {avg_loss:.3f}")

    # 保存
    torch.save(model.module.state_dict(), "./3_u_net.pth")
    dist.destroy_process_group()
    print("学習終了")


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(world_size)
    spawn(main, args=(world_size,), nprocs=world_size, join=True)
