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
from torch.utils.data import DataLoader, DistributedSampler


def load_image(path):
    img = Image.open(path).convert('RGB')
    transform = v2.Compose([
        v2.Resize([128, 128]),
        v2.ToTensor(),
        v2.Lambda(lambda x: x.mul(255)), #VGGは256スケールで学習
        v2.Normalize(mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375]),
    ])
    return transform(img).unsqueeze(0)  # (1,3,H,W)


def deprocess(tensor):
    # 正規化前に戻して PIL で可視化できるように
    mean = torch.tensor([123.675, 116.28, 103.53], device=tensor.device).view(1,3,1,1)
    std  = torch.tensor([58.395, 57.12, 57.375], device=tensor.device).view(1,3,1,1)
    img = tensor * std + mean
    img = img.clamp(0,255).detach().cpu().squeeze(0).permute(1,2,0).numpy().astype('uint8')
    return Image.fromarray(img)


# --- Gram 行列計算 ---
def gram_matrix(feat):
    b, c, h, w = feat.size()
    F = feat.view(b, c, h*w)
    return torch.bmm(F, F.transpose(1,2)) / (c * h * w) #バッチ行列積，除算は正規化


# --- VGG16 から必要層を取り出すモジュール ---
class VGG16Features(nn.Module):
    def __init__(self, content_layers, style_layers):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x):
        content_feats = {}
        style_feats   = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.content_layers:
                content_feats[i] = x
            if i in self.style_layers:
                style_feats[i] = x
        return content_feats, style_feats



# seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# settings
content_layers = [21]                 # conv4_2
style_layers   = [0, 5, 10, 19, 28]   # conv1_1～conv5_1
feats_extractor = VGG16Features(content_layers, style_layers).to(device)

# data
content_img = load_image('data/NST/skytree.png').to(device)
style_img   = load_image('data/NST/gohho.png').to(device) 
# 初期ターゲットはコンテンツ画像のコピー
target = content_img.clone().requires_grad_(True)

# --- 損失関数とオプティマイザ ---
mse = nn.MSELoss()
optimizer = torch.optim.LBFGS([target], lr=1.0)

# 事前にスタイル特徴の Gram 行列を取得
_, style_feats = feats_extractor(style_img)
style_grams = {i: gram_matrix(f) for i, f in style_feats.items()}
content_feats, _   = feats_extractor(content_img)
content_feat = content_feats[21]    # conv4_2 の出力を保存

# --- 最適化ループ ---
style_weight   = 1e6
content_weight = 1.0
max_iter = 300
run = [0]
while run[0] < max_iter:
    def closure():
        optimizer.zero_grad()
        c_feats, s_feats = feats_extractor(target)
        # コンテンツ損失
        c_loss = content_weight * mse(c_feats[21], content_feat)
        # スタイル損失
        s_loss = 0
        for i in style_layers:
            gm_t = gram_matrix(s_feats[i])
            gm_s = style_grams[i]
            s_loss += mse(gm_t, gm_s)
        s_loss *= style_weight

        loss = c_loss + s_loss
        loss.backward()
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Iter {run[0]}: Content Loss {c_loss.item():.2f}, Style Loss {s_loss.item():.2f}")
        return loss

    optimizer.step(closure)

# --- 結果の保存／表示 ---
result_img = deprocess(target)
result_img.save('data/NST/result.png')
