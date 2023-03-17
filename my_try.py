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
modelfull = create_model(
    "deit_tiny_distilled_patch16_224",
    pretrained=False,
    num_classes=1000,
    drop_rate=0.,
    drop_path_rate=0.1,
    drop_block_rate=None,
    img_size=224
)
modelfull.pre_logits = torch.nn.Identity()
modelfull.head = torch.nn.Identity()
modelfull.head_dist = torch.nn.Identity()

x     = torch.randn(10, 3, 224, 224)
y = modelfull(x)
import pdb;pdb.set_trace()
