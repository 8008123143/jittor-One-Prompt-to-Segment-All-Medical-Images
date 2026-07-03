"""
Validation/evaluation script for One-Prompt (Jittor version).
Usage: python val.py -net oneprompt -mod one_adpt -exp_name One-ISIC -weights <weight_path> -b 1 -dataset isic -data_path ../dataset/isic -vis 10 -baseline 'unet'
"""

import os
import numpy as np
import jittor as jt
from jittor import nn
from PIL import Image

import cfg
from utils import get_network, set_log_dir, create_logger
from dataset import ISIC2016, REFUGE
import function

args = cfg.parse_args()

# Build network
net = get_network(args, args.net, use_gpu=args.gpu,
                  gpu_device=args.gpu_device, distribution=args.distributed)

# Load pretrained weights
assert args.weights != 0
print(f'=> resuming from {args.weights}')
assert os.path.exists(args.weights)
checkpoint = jt.load(args.weights)
start_epoch = checkpoint['epoch']
best_tol = checkpoint['best_tol']

state_dict = checkpoint['state_dict']
if args.distributed != 'none':
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
else:
    new_state_dict = state_dict

net.load_state_dict(new_state_dict)

args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)

# Simple transforms (no torchvision)
def resize_to_tensor(size):
    def fn(img):
        img = img.resize((size, size), Image.BILINEAR)
        return jt.array(np.array(img)).permute(2, 0, 1).float() / 255.0
    return fn
t = resize_to_tensor(args.image_size)

# Dataset loading
if args.dataset == 'isic':
    isic_train_dataset = ISIC2016(args, args.data_path, transform=t, mode='Training')
    isic_test_dataset = ISIC2016(args, args.data_path, transform=t, mode='Test')
    nice_train_loader = isic_train_dataset
    nice_test_loader = isic_test_dataset

elif args.dataset == 'REFUGE':
    refuge_train_dataset = REFUGE(args, args.data_path, transform=t, mode='Training')
    refuge_test_dataset = REFUGE(args, args.data_path, transform=t, mode='Test')
    nice_train_loader = refuge_train_dataset
    nice_test_loader = refuge_test_dataset

elif args.dataset == 'oneprompt':
    from utils import get_decath_loader
    nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

# Run evaluation
net.eval()
tol, metrics = function.validation_one(args, nice_test_loader, 0, net)
logger.info(f'Total score: {tol}, IOU: {metrics[0]}, DICE: {metrics[1]} || @ epoch {start_epoch}.')
print(f'Evaluation complete. Total score: {tol}, IOU: {metrics[0]}, DICE: {metrics[1]}')
