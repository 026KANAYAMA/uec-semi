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
from torch.multiprocessing.spawn import spawn
import torchvision.models as models
import torchvision.datasets as dst
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# settings
DATA_LOOT = "/export/data/dataset/COCO/train2014"
STYLE_IMG_PATH = "./data/NST/gohho.png"
DATA_SIZE = 128

BATCH_SIZE = 128 # 128*4=512
epochs = 50
lr = 1e-3
content_weight = 1.0
style_weight   = 1e5


def preprocessing(img, img_size):
    transforms = v2.Compose([
        v2.Resize(img_size),
        v2.CenterCrop(img_size),
        v2.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])
    return transforms(img)


def gram_matrix(feat):
    b, c, h, w = feat.size()
    F = feat.view(b, c, h*w)
    return torch.bmm(F, F.transpose(1,2)) / (c*h*w)


class CustomDataset(Dataset):
    def __init__(self, data_root, data_size):
        self.content_paths = glob.glob(os.path.join(data_root, "*.jpg"))
        self.data_size = data_size

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        content_img = read_image(self.content_paths[idx])
        content_tensor = preprocessing(content_img, self.data_size)
        return content_tensor
    

# --- 損失ネットワーク（VGG16 Features） ---
class VGG16LossNet(nn.Module):
    def __init__(self, content_layers, style_layers, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.to(device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg_layers = list(vgg)
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x):
        content_feats, style_feats = {}, {}
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.content_layers:
                content_feats[i] = x
            if i in self.style_layers:
                style_feats[i] = x
        return content_feats, style_feats
    

# --- Transformer Network ---
class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Down-sampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 9, stride=1, padding=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
        )
        # Residual Blocks
        res_blocks = []
        for _ in range(5):
            res_blocks += [nn.Sequential(
                nn.Conv2d(128,128,3,padding=1),
                nn.InstanceNorm2d(128, affine=True),
                nn.ReLU(True),
                nn.Conv2d(128,128,3,padding=1),
                nn.InstanceNorm2d(128, affine=True),
            )]
        self.residuals = nn.Sequential(*res_blocks)
        # Up-sampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,1,output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,3,2,1,output_padding=1),
            nn.ReLU(True),
            nn.Conv2d(32,3,9,stride=1,padding=4),
            nn.Tanh(),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.residuals(y) + y
        return (self.upsample(y) + 1) / 2  # [0,1] にスケーリング


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
        DATA_LOOT,
        DATA_SIZE
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 代わりにepoch内で設定
        sampler=sampler,
        num_workers=1,
        pin_memory=True
    )

    # スタイル画像，Gram行列事前計算
    style_img = read_image(STYLE_IMG_PATH)
    style_tensor = preprocessing(style_img, DATA_SIZE).unsqueeze(0).to(device)
    loss_net = VGG16LossNet(
        content_layers=[21], 
        style_layers=[0,5,10,19,28], 
        device=device).to(device)
    

    _, style_feats = loss_net(style_tensor)
    style_grams = {i: gram_matrix(feat) for i, feat in style_feats.items()}

    # model & optimizer & loss func
    transformer_raw = TransformerNet()
    transformer = nn.parallel.DistributedDataParallel(
        transformer_raw.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # learning
    for epoch in range(epochs):
        transformer.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        for content_tensor in data_loader:
            content_tensor = content_tensor.to(device)
            # 出力
            gen_img = transformer(content_tensor)
            c_gen_img, s_gen_img = loss_net(gen_img)
            c_content_tensor, _ = loss_net(content_tensor)
            # 損失
            loss_c = content_weight * mse_loss(c_gen_img[21], c_content_tensor[21])
            loss_s = 0
            for i in style_grams:
                loss_s += mse_loss(gram_matrix(s_gen_img[i]), style_grams[i])
            loss_s *= style_weight

            loss = loss_c + loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)

        if rank == 0:
            print(f"epoch: {epoch+1}, loss: {avg_loss:.3f}")

    # 保存
    torch.save(transformer.module.state_dict(), "8_1_rst.pth")

    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(world_size)
    spawn(main, args=(world_size,), nprocs=world_size, join=True)





