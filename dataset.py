"""
Dataset classes for One-Prompt (Jittor version).
Auto-scans folders — no CSV needed.
"""

import os
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
from PIL import Image


def random_click(mask, point_labels=1, inout=1):
    """Generate a random point click within the mask region."""
    indices = np.argwhere(mask == inout)
    if len(indices) == 0:
        return np.array([0, 0])
    return indices[np.random.randint(len(indices))]


class ISIC2016(Dataset):
    """ISIC 2016 — auto-scans image/mask pairs from folder structure."""

    def __init__(self, args, data_path, transform=None, transform_msk=None,
                 mode='Training', prompt='click', plane=False):
        super().__init__()
        img_dir = os.path.join(data_path, f'ISBI2016_ISIC_Part1_{mode}_Data')
        gt_dir = os.path.join(data_path, f'ISBI2016_ISIC_Part1_{mode}_GroundTruth')
        self.pairs = []
        gt_files = set(os.listdir(gt_dir))
        for f in sorted(os.listdir(img_dir)):
            if not f.endswith('.jpg'):
                continue
            name = f.replace('.jpg', '')
            m = f'{name}_Segmentation.png'
            if m in gt_files:
                self.pairs.append(
                    (os.path.join(img_dir, f), os.path.join(gt_dir, m)))
        self.img_size = args.image_size
        self.prompt = prompt
        self.transform = transform
        self.transform_msk = transform_msk
        self.total_len = len(self.pairs)
        print(f'ISIC2016 ({mode}): {self.total_len} image-mask pairs loaded')

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        point_label, inout = 1, 1
        img_path, msk_path = self.pairs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size))
        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)
            pt = jt.array(pt)
        if self.transform:
            img = self.transform(img)
        if self.transform_msk:
            mask = self.transform_msk(mask)
        else:
            mask = jt.array(np.array(mask)).unsqueeze(0).float() / 255.0
        name = img_path.split('/')[-1].split('.')[0]
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'image_meta_dict': {'filename_or_obj': name},
        }


class REFUGE(Dataset):
    """REFUGE — auto-scans subfolders for image/mask pairs."""

    def __init__(self, args, data_path, transform=None, transform_msk=None,
                 mode='Training', prompt='click', plane=False):
        super().__init__()
        subdir = os.path.join(data_path, mode + '-400')
        self.subfolders = [f.path for f in os.scandir(subdir) if f.is_dir()]
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.prompt = prompt
        self.transform = transform
        self.transform_msk = transform_msk
        self.total_len = len(self.subfolders)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        point_label, inout = 1, 1
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]
        img_path = os.path.join(subfolder, name + '.jpg')
        img = Image.open(img_path).convert('RGB')

        # Load multi-rater masks
        cup_paths = [os.path.join(subfolder, f'{name}_seg_cup_{i}.png')
                     for i in range(1, 8)]
        disc_paths = [os.path.join(subfolder, f'{name}_seg_disc_{i}.png')
                      for i in range(1, 8)]

        if self.prompt == 'click':
            # Use mean of raters for click generation
            cup_np = [np.array(Image.open(p).convert('L').resize(
                (self.img_size, self.img_size))) for p in cup_paths]
            disc_np = [np.array(Image.open(p).convert('L').resize(
                (self.img_size, self.img_size))) for p in disc_paths]
            pt_disc = random_click(
                np.mean(np.stack(disc_np), axis=0) / 255, point_label, inout)
            pt_disc = jt.array(pt_disc)

        if self.transform:
            img = self.transform(img)
            multi_rater_disc = []
            for p in disc_paths:
                m = Image.open(p).convert('L')
                m_arr = jt.array(np.array(m)).unsqueeze(0).float() / 255.0
                multi_rater_disc.append(m_arr)
            multi_rater_disc = jt.stack(multi_rater_disc, dim=0)
            mask_disc = jt.nn.interpolate(
                multi_rater_disc.unsqueeze(0),
                size=(self.mask_size, self.mask_size),
                mode='bilinear', align_corners=False
            ).squeeze(0).mean(dim=0)

        return {
            'image': img,
            'label': mask_disc,
            'p_label': point_label,
            'pt': pt_disc,
            'image_meta_dict': {'filename_or_obj': name},
        }
