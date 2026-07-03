"""
Core utility functions for One-Prompt (Jittor version).
Includes model loading, logging, visualization, evaluation metrics.
"""

import os
import sys
import math
import logging
import time
import collections
from datetime import datetime
from typing import Union, Optional, List, Tuple
import pathlib

import numpy as np
import jittor as jt
from jittor import nn
from PIL import Image

import cfg

args = cfg.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════

def get_network(args, net_name, use_gpu=True, gpu_device=0, distribution=True):
    """Return the requested network."""
    if net_name == 'oneprompt':
        from models.oneprompt import one_model_registry
        net = one_model_registry[args.baseline](args)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if distribution != 'none':
        net = nn.DataParallel(net)
    return net


# ═══════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    jt.save(states, os.path.join(output_dir, filename))
    if is_best:
        jt.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def make_grid(tensor, nrow=8, padding=2, normalize=False,
              value_range=None, scale_each=False, pad_value=0):
    """Create a grid of images (compatible with torchvision.utils.make_grid)."""
    if isinstance(tensor, list):
        tensor = jt.stack(tensor, dim=0)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = jt.concat([tensor, tensor, tensor], 0)
        tensor = tensor.unsqueeze(0)

    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = jt.concat([tensor, tensor, tensor], 1)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = jt.ones((num_channels, height * ymaps + padding, width * xmaps + padding)) * pad_value

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid_narrow_h = grid[:, y * height + padding: (y + 1) * height]
            grid_narrow_hw = grid_narrow_h[:, :, x * width + padding: (x + 1) * width]
            grid_narrow_hw.update(tensor[k])
            k += 1
    return grid


def save_image(tensor, fp, format=None, **kwargs):
    """Save a tensor as an image file."""
    grid = make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse=False, points=None):
    """Visualize segmentation results and save to disk."""
    b, c, h, w = pred_masks.shape
    row_num = min(b, 4)

    if pred_masks.max() > 1 or pred_masks.min() < 0:
        pred_masks = jt.sigmoid(pred_masks)

    if reverse:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks

    if c == 2:
        # For cup/disc dual output
        pred_disc = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        pred_cup = pred_masks[:, 1, :, :].unsqueeze(1).expand(b, 3, h, w)
        gt_disc = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        gt_cup = gt_masks[:, 1, :, :].unsqueeze(1).expand(b, 3, h, w)
        compose = jt.concat([pred_disc[:row_num], pred_cup[:row_num],
                              gt_disc[:row_num], gt_cup[:row_num]], 0)
        save_image(compose, fp=save_path, nrow=row_num, padding=10)
    else:
        imgs = jt.nn.interpolate(imgs, size=(h, w))
        if imgs.shape[1] == 1:
            imgs = imgs[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        pred_masks = pred_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        gt_masks = gt_masks[:, 0, :, :].unsqueeze(1).expand(b, 3, h, w)
        compose = jt.concat([imgs[:row_num], pred_masks[:row_num],
                              gt_masks[:row_num]], 0)
        save_image(compose, fp=save_path, nrow=row_num, padding=10)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════

def iou_numpy(outputs: np.ndarray, labels: np.ndarray):
    """IoU for numpy arrays."""
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean()


def dice_coeff(input, target):
    """Dice coefficient for a batch of predictions."""
    eps = 0.0001
    inter = (input.reshape(-1) * target.reshape(-1)).sum()
    union = input.sum() + target.sum() + eps
    return (2 * inter + eps) / union


def eval_seg(pred, true_mask_p, thresholds):
    """Evaluate segmentation with multiple thresholds."""
    b, c, h, w = pred.shape
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0, 0, 0, 0
        for th in thresholds:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.numpy()
            disc_pred = vpred_cpu[:, 0, :, :].astype('int32')
            cup_pred = vpred_cpu[:, 1, :, :].astype('int32')
            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).numpy().astype('int32')
            cup_mask = gt_vmask_p[:, 1, :, :].squeeze(1).numpy().astype('int32')

            iou_d += iou_numpy(disc_pred, disc_mask)
            iou_c += iou_numpy(cup_pred, cup_mask)
            disc_dice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
            cup_dice += dice_coeff(vpred[:, 1, :, :], gt_vmask_p[:, 1, :, :]).item()
        return iou_d / len(thresholds), iou_c / len(thresholds), \
               disc_dice / len(thresholds), cup_dice / len(thresholds)
    else:
        eiou, edice = 0, 0
        for th in thresholds:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.numpy()
            disc_pred = vpred_cpu[:, 0, :, :].astype('int32')
            disc_mask = gt_vmask_p[:, 0, :, :].squeeze(1).numpy().astype('int32')

            eiou += iou_numpy(disc_pred, disc_mask)
            edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
        return eiou / len(thresholds), edice / len(thresholds)


# ═══════════════════════════════════════════════════════════════════════
# Prompt Generation
# ═══════════════════════════════════════════════════════════════════════

def np_random_click(mask, point_labels=1, inout=1):
    """Generate random click from numpy mask."""
    indices = np.argwhere(mask == inout)
    if len(indices) == 0:
        return np.array([0, 0])
    return indices[np.random.randint(len(indices))]


def generate_click_prompt(img, msk, pt_label=1):
    """Generate click prompts for 3D data."""
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.shape
    msk = msk[:, 0, :, :, :]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j, :, :, i]
            indices = jt.nonzero(msk_s)
            if indices.shape[0] == 0:
                random_index = jt.randint(0, h, (2,))
                new_s = msk_s
            else:
                random_index = indices[jt.randint(0, indices.shape[0], (1,))[0]]
                label = msk_s[random_index[0], random_index[1]]
                new_s = (msk_s == label).float()
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = jt.stack(pt_list_s, dim=0)
        msks = jt.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = jt.stack(pt_list, dim=-1)
    msk = jt.stack(msk_list, dim=-1)
    msk = msk.unsqueeze(1)
    return img, pt, msk  # [b, 2, d], [b, c, h, w, d]


# ═══════════════════════════════════════════════════════════════════════
# MONAI Decathlon Loader (Placeholder)
# ═══════════════════════════════════════════════════════════════════════

def get_decath_loader(args):
    """Load multi-dataset decathlon data (requires MONAI JSON file).

    Note: This function requires MONAI and a properly formatted
    dataset_0.json file. For Jittor-only usage, use ISIC2016/REFUGE.
    """
    raise NotImplementedError(
        "MONAI-based decathlon loader is not available in Jittor. "
        "Please use --dataset isic or --dataset REFUGE instead."
    )
