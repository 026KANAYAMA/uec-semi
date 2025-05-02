# Distributed Data Parallelを試す

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torchvision.models import resnet152
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader, DistributedSampler


def main(rank, world_size):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # datasest
    transform = v2.Compose([
        v2.ToTensor(),
        v2.ToDtype(torch.float, scale=True)
    ])
    dataset = CIFAR10(
        root="./data",
        transform=transform,
        download=True
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size= 2048 // world_size,
        sampler=sampler,
        num_workers=2,
        drop_last=False
    )

    # model
    model = resnet152(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    # Others
    lr = 0.0001
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0001
    )

    # learning
    start_time = time.time()
    epochs = 5
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)

        train_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = train_loss / len(dataloader)
        if rank == 0:
            print(f"epoch: {epoch+1}, train loss: {avg_loss}")

    if rank == 0:
        end_time = time.time()
        print(f"学習時間: {end_time - start_time:.1f}seconds")

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    spawn(main, args=(world_size,), nprocs=world_size, join=True)

    # Total Batch=2048, GPU4
    # epoch: 1, train loss: 1.419136197566986
    # epoch: 2, train loss: 0.6253176712989807
    # epoch: 3, train loss: 0.30425582706928256
    # epoch: 4, train loss: 0.11427476674318314
    # epoch: 5, train loss: 0.043002016693353656
    # 学習時間: 113.8seconds

    # Total Batch=1012, GPU4
    # epoch: 1, train loss: 1.227011546796682
    # epoch: 2, train loss: 0.518278950939373
    # epoch: 3, train loss: 0.2220302725933036
    # epoch: 4, train loss: 0.09900690653190321
    # epoch: 5, train loss: 0.05891844653049294
    # 学習時間: 126.5seconds

    # Total Batch=2048, GPU4, num_works=4 -> 1 (マルチスレッドより4*4=16使ってた．物理は12)
    # epoch: 1, train loss: 1.419136197566986
    # epoch: 2, train loss: 0.6253176712989807
    # epoch: 3, train loss: 0.30425582706928256
    # epoch: 4, train loss: 0.11427476674318314
    # epoch: 5, train loss: 0.043002016693353656
    # 学習時間: 46.5seconds

    # Total Batch=2048, GPU4, num_works=2
    # epoch: 1, train loss: 1.419136197566986
    # epoch: 2, train loss: 0.6253176712989807
    # epoch: 3, train loss: 0.30425582706928256
    # epoch: 4, train loss: 0.11427476674318314
    # epoch: 5, train loss: 0.043002016693353656
    # 学習時間: 66.1seconds
    

