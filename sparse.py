import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.transforms import transforms

import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Top-k selection')
    parser.add_argument('--dataset', type=str, default='awa', help='dataset')

    args=parser.parse_args()

    if args.dataset=='awa':
        num_classes=50
        