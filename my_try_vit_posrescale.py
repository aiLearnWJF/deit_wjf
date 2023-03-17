#%%
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

import models
import models_v2

from torchinfo import summary

import utils
import math
import torch.nn.functional as F
model = create_model(
    "deit_tiny_patch16_224",
    pretrained=False,
    num_classes=1000,
    drop_rate=0.,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=(224,448)
)
# modelfull.pre_logits = torch.nn.Identity()
# model.head = torch.nn.Identity()
# checkpoint = torch.load("/vehicle/yckj3860/code/fast-reid-1.3.0/pretrained/deit_tiny_patch16_224-a1311bcf.pth")
# modelfull.load_state_dict(checkpoint["model"])

def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

# ***** start
pretrain_path = "/vehicle/yckj3860/code/fast-reid-1.3.0/pretrained/deit_tiny_patch16_224-a1311bcf.pth"
state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))

if 'model' in state_dict:
    state_dict = state_dict.pop('model')
if 'state_dict' in state_dict:
    state_dict = state_dict.pop('state_dict')
for k, v in state_dict.items():
    if 'head' in k or 'dist' in k:
        continue
    if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
        # For old models that I trained prior to conv based patchification
        O, I, H, W = model.patch_embed.proj.weight.shape
        v = v.reshape(O, -1, H, W)
    elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
        # To resize pos embedding when using model at different size from pretrained weights
        if 'distilled' in pretrain_path:
            v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
        # import pdb;pdb.set_trace()
        v = resize_pos_embed(v, model.pos_embed.data, 14, 28)
    state_dict[k] = v
# ***** end

model.load_state_dict(state_dict)
x     = torch.randn(10, 3, 224, 448)
y = model(x)
import pdb;pdb.set_trace()
