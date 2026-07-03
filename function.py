"""
Training and validation functions for One-Prompt (Jittor version).
Includes TensorBoard logging for loss curves and metrics.
"""

import os
import json
import numpy as np
import jittor as jt
from jittor import nn
from tqdm import tqdm
from PIL import Image

import cfg
from conf import settings
from utils import (
    save_image, vis_image, eval_seg, dice_coeff,
    generate_click_prompt,
)

args = cfg.parse_args()

# Loss functions
criterion_G = nn.BCEWithLogitsLoss()

# Thresholds for evaluation
threshold = (0.1, 0.3, 0.5, 0.7, 0.9)


def train_one(args, net, optimizer, train_loader, epoch, writer, vis=50):
    """Train one epoch with TensorBoard logging."""
    epoch_loss = 0
    ind = 0
    step_losses = []  # Track per-step loss for curve plotting
    net.train()
    optimizer.zero_grad()

    with tqdm(total=train_loader.total_len, desc=f'Train Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            if ind == 0:
                # --- Set up template image from first sample ---
                tmp_img = pack['image'].float()
                if tmp_img.ndim == 4:
                    tmp_img = tmp_img[0:1].repeat(args.b, 1, 1, 1)
                tmp_mask = pack['label'].float()
                if tmp_mask.ndim == 4:
                    tmp_mask = tmp_mask[0:1].repeat(args.b, 1, 1, 1)

                if 'pt' not in pack:
                    tmp_img, pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']

                if point_labels[0] != -1:
                    point_coords = pt
                    if not isinstance(point_coords, jt.Var):
                        point_coords = jt.array(point_coords)
                    if not isinstance(point_labels, jt.Var):
                        point_labels_jt = jt.array(point_labels) if isinstance(point_labels, (list, np.ndarray)) else point_labels
                    else:
                        point_labels_jt = point_labels
                    point_coords, point_labels_jt = point_coords.float(), point_labels_jt.int()
                    point_coords, point_labels_jt = point_coords[None, :, :], point_labels_jt[None, :]
                    pt = (point_coords, point_labels_jt)

            imgs = pack['image'].float()
            masks = pack['label'].float()
            name = pack['image_meta_dict']['filename_or_obj']

            # --- 3D handling ---
            if args.thd:
                pt_tensor, _ = pt
                pt_tensor = pt_tensor.reshape(
                    pt_tensor.shape[0] * pt_tensor.shape[1], -1, pt_tensor.shape[-1])
                imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1],
                                    imgs.shape[2], imgs.shape[3], imgs.shape[4])
                masks = masks.reshape(masks.shape[0] * masks.shape[1],
                                      masks.shape[2], masks.shape[3], masks.shape[4])
                imgs = imgs.repeat(1, 3, 1, 1)
                imgs = jt.nn.interpolate(imgs, size=(args.image_size, args.image_size))
                masks = jt.nn.interpolate(masks, size=(args.out_size, args.out_size))

            ind += 1

            # --- Align batch sizes ---
            if tmp_img.shape[0] != imgs.shape[0]:
                tmp_img = tmp_img[0:1].repeat(imgs.shape[0], 1, 1, 1)
                p0, p1 = pt
                p0 = p0[0:1].repeat(imgs.shape[0], 1, 1)
                p1 = p1[0:1].repeat(imgs.shape[0], 1)
                pt = (p0, p1)

            # --- Forward pass ---
            imge, skips = net.image_encoder(imgs)
            timge, tskips = net.image_encoder(tmp_img)

            p1, p2, se, de = net.prompt_encoder(
                points=pt, boxes=None, doodles=None, masks=None,
            )

            pred, _ = net.mask_decoder(
                skips_raw=skips, skips_tmp=tskips,
                raw_emb=imge, tmp_emb=timge,
                pt1=p1, pt2=p2,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )

            # Align mask size to pred size
            if masks.shape[-2:] != pred.shape[-2:]:
                masks = jt.nn.interpolate(masks, size=pred.shape[-2:],
                                           mode='bilinear', align_corners=False)

            loss = criterion_G(pred, masks)
            loss_val = loss.item()

            pbar.set_postfix(**{'loss': f'{loss_val:.4f}'})
            epoch_loss += loss_val
            step_losses.append(loss_val)

            # --- TensorBoard: log per-step loss ---
            if writer is not None:
                global_step = epoch * train_loader.total_len + ind
                writer.add_scalar('Train/StepLoss', loss_val, global_step)

            # --- Backward ---
            optimizer.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # --- Visualization (save images to disk + TensorBoard) ---
            if vis and ind % vis == 0:
                namecat = 'Train_'
                for na in name[:4]:  # max 4 names
                    namecat += na.split('/')[-1].split('.')[0] + '_'
                save_path = os.path.join(
                    args.path_helper['sample_path'],
                    namecat + f'epoch_{epoch}_step_{ind}.jpg')
                vis_image(imgs, pred, masks, save_path, reverse=False)

                # TensorBoard: log sample images
                if writer is not None:
                    try:
                        from utils import make_grid as mg_fn
                        grid = mg_fn(imgs[:4], nrow=4, padding=2, normalize=True)
                        writer.add_image('Train/Input', grid, global_step)
                        grid_pred = mg_fn(pred[:4], nrow=4, padding=2, normalize=True)
                        writer.add_image('Train/Prediction', grid_pred, global_step)
                        grid_gt = mg_fn(masks[:4], nrow=4, padding=2, normalize=True)
                        writer.add_image('Train/GroundTruth', grid_gt, global_step)
                    except Exception:
                        pass  # image logging is best-effort

            pbar.update()

    avg_loss = epoch_loss / max(ind, 1)

    # --- TensorBoard: log epoch-level metrics ---
    if writer is not None:
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)

    return avg_loss, step_losses


def validation_one(args, val_loader, epoch, net, clean_dir=True):
    """Run validation with TensorBoard logging."""
    net.eval()

    n_val = val_loader.total_len
    mix_res = (0, 0, 0, 0)
    tot = 0

    with tqdm(total=n_val, desc=f'Val Epoch {epoch}', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            if ind == 0:
                tmp_img = pack['image'].float()
                if tmp_img.ndim == 4:
                    tmp_img = tmp_img[0:1].repeat(args.b, 1, 1, 1)
                tmp_mask = pack['label'].float()
                if tmp_mask.ndim == 4:
                    tmp_mask = tmp_mask[0:1].repeat(args.b, 1, 1, 1)

                if 'pt' not in pack:
                    tmp_img, pt, tmp_mask = generate_click_prompt(tmp_img, tmp_mask)
                else:
                    pt = pack['pt']
                    point_labels = pack['p_label']

                if point_labels[0] != -1:
                    point_coords = pt
                    if not isinstance(point_coords, jt.Var):
                        point_coords = jt.array(point_coords)
                    if not isinstance(point_labels, jt.Var):
                        point_labels_jt = jt.array(point_labels) if isinstance(point_labels, (list, np.ndarray)) else point_labels
                    else:
                        point_labels_jt = point_labels
                    point_coords, point_labels_jt = point_coords.float(), point_labels_jt.int()
                    point_coords, point_labels_jt = point_coords[None, :, :], point_labels_jt[None, :]
                    pt = (point_coords, point_labels_jt)

            imgs = pack['image'].float()
            masks = pack['label'].float()
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                pt_tensor, _ = pt
                pt_tensor = pt_tensor.reshape(
                    pt_tensor.shape[0] * pt_tensor.shape[1], -1, pt_tensor.shape[-1])
                imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1],
                                    imgs.shape[2], imgs.shape[3], imgs.shape[4])
                masks = masks.reshape(masks.shape[0] * masks.shape[1],
                                      masks.shape[2], masks.shape[3], masks.shape[4])
                imgs = imgs.repeat(1, 3, 1, 1)
                imgs = jt.nn.interpolate(imgs, size=(args.image_size, args.image_size))
                masks = jt.nn.interpolate(masks, size=(args.out_size, args.out_size))

            # --- Forward (no grad) ---
            prev_grad = jt.flag.grad
            jt.flag.grad = False

            imge, skips = net.image_encoder(imgs)
            timge, tskips = net.image_encoder(tmp_img)

            p1, p2, se, de = net.prompt_encoder(
                points=pt, boxes=None, doodles=None, masks=None,
            )

            pred, _ = net.mask_decoder(
                skips_raw=skips, skips_tmp=tskips,
                raw_emb=imge, tmp_emb=timge,
                pt1=p1, pt2=p2,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
            )

            if masks.shape[-2:] != pred.shape[-2:]:
                masks = jt.nn.interpolate(masks, size=pred.shape[-2:],
                                           mode='bilinear', align_corners=False)
            tot += criterion_G(pred, masks).item()

            # --- Visualization ---
            if args.vis and ind % args.vis == 0:
                namecat = 'Val_'
                for na in name[:4]:
                    img_name = na.split('/')[-1].split('.')[0]
                    namecat += img_name + '_'
                save_path = os.path.join(
                    args.path_helper['sample_path'],
                    namecat + f'epoch_{epoch}_step_{ind}.jpg')
                vis_image(imgs, pred, masks, save_path, reverse=False)

            temp = eval_seg(pred, masks, threshold)
            mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            jt.flag.grad = prev_grad
            pbar.update()

    avg_loss = tot / n_val
    n = len(threshold)
    eiou = mix_res[0] / n if len(mix_res) == 2 else mix_res[0] / n_val
    edice = mix_res[1] / n if len(mix_res) == 2 else mix_res[1] / n_val

    return avg_loss, (eiou, edice)


# ═══════════════════════════════════════════════════════════════════════
# Metrics History (for Loss/Curve Plotting)
# ═══════════════════════════════════════════════════════════════════════

class MetricsHistory:
    """Stores training history and can save/load to JSON + plot curves."""

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_iou = []
        self.val_dice = []
        self.step_losses = []  # flattened per-step losses

    def update(self, train_loss, val_loss=None, val_iou=None, val_dice=None,
               step_losses=None):
        self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_iou is not None:
            self.val_iou.append(val_iou)
        if val_dice is not None:
            self.val_dice.append(val_dice)
        if step_losses is not None:
            self.step_losses.extend(step_losses)

    def save(self, filepath):
        """Save metrics to JSON."""
        data = {
            'train_loss': [float(x) for x in self.train_loss],
            'val_loss': [float(x) for x in self.val_loss],
            'val_iou': [float(x) for x in self.val_iou],
            'val_dice': [float(x) for x in self.val_dice],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Metrics saved to {filepath}')

    def load(self, filepath):
        """Load metrics from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.train_loss = data.get('train_loss', [])
        self.val_loss = data.get('val_loss', [])
        self.val_iou = data.get('val_iou', [])
        self.val_dice = data.get('val_dice', [])
        return self

    def plot(self, save_dir, prefix='training'):
        """Generate and save loss/metric curves as PNG images."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # --- Figure 1: Loss Curves ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Train Loss
        ax = axes[0]
        ax.plot(range(1, len(self.train_loss) + 1), self.train_loss,
                'b-', linewidth=1.5, label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Train + Val Loss
        ax = axes[1]
        ax.plot(range(1, len(self.train_loss) + 1), self.train_loss,
                'b-', linewidth=1.5, label='Train Loss')
        if self.val_loss:
            val_epochs = range(1, len(self.val_loss) + 1)
            ax.plot(val_epochs, self.val_loss, 'r-o', linewidth=1.5,
                    markersize=4, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        loss_path = os.path.join(save_dir, f'{prefix}_loss_curves.png')
        fig.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Loss curves saved to {loss_path}')

        # --- Figure 2: IoU & Dice Curves ---
        if self.val_iou or self.val_dice:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            if self.val_iou:
                epochs = range(1, len(self.val_iou) + 1)
                ax.plot(epochs, self.val_iou, 'g-o', linewidth=1.5, markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU')
            ax.set_title('Validation IoU')
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            if self.val_dice:
                epochs = range(1, len(self.val_dice) + 1)
                ax.plot(epochs, self.val_dice, 'm-o', linewidth=1.5, markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Dice')
            ax.set_title('Validation Dice')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            metric_path = os.path.join(save_dir, f'{prefix}_metric_curves.png')
            fig.savefig(metric_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Metric curves saved to {metric_path}')

        # --- Figure 3: Smoothed per-step loss ---
        if len(self.step_losses) > 100:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            steps = range(len(self.step_losses))
            ax.plot(steps, self.step_losses, 'b-', alpha=0.15, linewidth=0.5,
                    label='Raw')
            # Moving average
            window = max(len(self.step_losses) // 200, 10)
            if window > 1:
                smoothed = np.convolve(self.step_losses,
                                       np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.step_losses)),
                        smoothed, 'r-', linewidth=1.5,
                        label=f'Smoothed (w={window})')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Per-Step Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            step_path = os.path.join(save_dir, f'{prefix}_step_loss.png')
            fig.savefig(step_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Step loss curve saved to {step_path}')

    def log_to_tensorboard(self, writer, epoch):
        """Write all epoch-level metrics to TensorBoard."""
        if writer is None:
            return
        if self.train_loss:
            writer.add_scalar('Epoch/TrainLoss', self.train_loss[-1], epoch)
        if self.val_loss:
            writer.add_scalar('Epoch/ValLoss', self.val_loss[-1], epoch)
        if self.val_iou:
            writer.add_scalar('Epoch/IoU', self.val_iou[-1], epoch)
        if self.val_dice:
            writer.add_scalar('Epoch/Dice', self.val_dice[-1], epoch)
        writer.flush()
