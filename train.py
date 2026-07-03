"""
Training script for One-Prompt (Jittor version).
Features:
  - TensorBoard scalar logging (loss, IoU, Dice)
  - Automatic loss/metric curve plotting via matplotlib
  - Segmentation visualization saved to disk + TensorBoard
  - Metrics history saved to JSON for post-hoc analysis

Usage:
  python train.py -net oneprompt -mod one_adpt -exp_name basic_exp -b 8 \
      -dataset isic -data_path ../data -baseline 'unet' -vis 50
"""

import os
import time
import json
import numpy as np
import jittor as jt
from PIL import Image
from jittor import nn
from jittor import optim

import cfg
from conf import settings
from utils import (
    get_network, set_log_dir, create_logger, save_checkpoint,
)
from function import (
    train_one, validation_one, MetricsHistory,
)

args = cfg.parse_args()

# ═══════════════════════════════════════════════════════════════════════
# Build network
# ═══════════════════════════════════════════════════════════════════════
net = get_network(args, args.net, use_gpu=args.gpu,
                  gpu_device=args.gpu_device, distribution=args.distributed)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                       eps=1e-08, weight_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ═══════════════════════════════════════════════════════════════════════
# Resume or start fresh
# ═══════════════════════════════════════════════════════════════════════
start_epoch = 0
metrics_history = MetricsHistory()

if args.weights != 0:
    print(f'=> Resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint = jt.load(args.weights)
    start_epoch = checkpoint['epoch']
    best_tol = checkpoint['best_tol']
    if args.distributed != 'none':
        net.module.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint['state_dict'])
    args.path_helper = checkpoint['path_helper']
    logger = create_logger(args.path_helper['log_path'])

    # Resume metrics history if available
    metrics_json = os.path.join(args.path_helper.get('prefix', 'logs'),
                                'metrics_history.json')
    if os.path.exists(metrics_json):
        metrics_history.load(metrics_json)
        print(f'=> Loaded metrics history ({len(metrics_history.train_loss)} epochs)')
else:
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])

logger.info(f"Args: {args}")
print(f"Log directory: {args.path_helper['prefix']}")

# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════
if args.dataset == 'oneprompt':
    from utils import get_decath_loader
    nice_train_loader, nice_test_loader, transform_train, transform_val, \
        train_list, val_list = get_decath_loader(args)
elif args.dataset == 'isic':
    from dataset import ISIC2016
    # Simple transforms (no torchvision needed)
    def resize_to_tensor(size):
        def fn(img):
            img = img.resize((size, size), Image.BILINEAR)
            return jt.array(np.array(img)).permute(2, 0, 1).float() / 255.0
        return fn
    t = resize_to_tensor(args.image_size)
    isic_train = ISIC2016(args, args.data_path, transform=t, mode='Training')
    isic_test = ISIC2016(args, args.data_path, transform=t, mode='Test')
    nice_train_loader = isic_train
    nice_test_loader = isic_test
elif args.dataset == 'REFUGE':
    from dataset import REFUGE
    def resize_to_tensor(size):
        def fn(img):
            img = img.resize((size, size), Image.BILINEAR)
            return jt.array(np.array(img)).permute(2, 0, 1).float() / 255.0
        return fn
    t = resize_to_tensor(args.image_size)
    refuge_train = REFUGE(args, args.data_path, transform=t, mode='Training')
    refuge_test = REFUGE(args, args.data_path, transform=t, mode='Test')
    nice_train_loader = refuge_train
    nice_test_loader = refuge_test
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

# ═══════════════════════════════════════════════════════════════════════
# TensorBoard
# ═══════════════════════════════════════════════════════════════════════
try:
    from tensorboardX import SummaryWriter
    tb_log_dir = os.path.join(args.path_helper['prefix'], 'tensorboard')
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logging to: {tb_log_dir}")
except ImportError:
    print("Warning: tensorboardX not installed. Loss curves will only be "
          "available via matplotlib plots, not TensorBoard.")
    writer = None

# ═══════════════════════════════════════════════════════════════════════
# Checkpoint directory
# ═══════════════════════════════════════════════════════════════════════
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net,
                               settings.TIME_NOW)
os.makedirs(checkpoint_path, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════
best_tol = getattr(checkpoint, 'best_tol', 1e4) if args.weights != 0 else 1e4
print(f"\n{'='*60}")
print(f"Training started — {settings.EPOCH} epochs")
print(f"Sample outputs: {args.path_helper['sample_path']}")
print(f"{'='*60}\n")

for epoch in range(start_epoch, settings.EPOCH):
    net.train()
    t_start = time.time()

    # --- Train ---
    train_loss, step_losses = train_one(
        args, net, optimizer, nice_train_loader, epoch, writer, vis=args.vis)
    scheduler.step()

    t_train = time.time() - t_start
    logger.info(f'[Epoch {epoch}] Train Loss: {train_loss:.4f} | Time: {t_train:.1f}s')

    # --- Validation ---
    net.eval()
    if (epoch and epoch % args.val_freq == 0) or epoch == settings.EPOCH - 1:
        val_loss, (eiou, edice) = validation_one(
            args, nice_test_loader, epoch, net)

        logger.info(f'[Epoch {epoch}] Val Loss: {val_loss:.4f} | '
                     f'IoU: {eiou:.4f} | Dice: {edice:.4f}')

        # --- TensorBoard epoch metrics ---
        if writer is not None:
            writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            writer.add_scalar('Epoch/IoU', eiou, epoch)
            writer.add_scalar('Epoch/Dice', edice, epoch)
            writer.flush()

        # --- Update metrics history ---
        metrics_history.update(
            train_loss=train_loss,
            val_loss=val_loss,
            val_iou=eiou,
            val_dice=edice,
            step_losses=step_losses,
        )

        # --- Save checkpoint ---
        sd = net.module.state_dict() if args.distributed != 'none' \
            else net.state_dict()

        is_best = val_loss < best_tol
        if is_best:
            best_tol = val_loss

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'],
            filename=f"checkpoint_epoch_{epoch}.pth")

        # --- Save metrics JSON ---
        metrics_history.save(
            os.path.join(args.path_helper['prefix'], 'metrics_history.json'))

        # --- Plot curves every N validations ---
        plot_interval = max(args.val_freq * 2, 100)
        if epoch % plot_interval == 0 or epoch == settings.EPOCH - 1:
            metrics_history.plot(args.path_helper['prefix'], prefix='training')

        if is_best:
            print(f'  >>> Best model saved! Val Loss: {val_loss:.4f}')
    else:
        # Non-validation epoch: still record train loss
        metrics_history.update(train_loss=train_loss, step_losses=step_losses)

    t_total = time.time() - t_start
    print(f'[Epoch {epoch}] Train: {train_loss:.4f} | Total: {t_total:.1f}s')

# ═══════════════════════════════════════════════════════════════════════
# Post-training: final plots & summary
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Training complete! Generating final plots...")

# Final metrics save
metrics_history.save(
    os.path.join(args.path_helper['prefix'], 'metrics_history_final.json'))

# Final curve plots
try:
    metrics_history.plot(args.path_helper['prefix'], prefix='final')
except Exception as e:
    print(f"Warning: Could not generate plots: {e}")

# Print summary
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Log directory:  {args.path_helper['prefix']}")
print(f"Checkpoints:    {args.path_helper['ckpt_path']}")
print(f"Sample images:  {args.path_helper['sample_path']}")
print(f"Train Loss:     {args.path_helper['log_path']}")
if metrics_history.val_iou:
    print(f"Best IoU:       {max(metrics_history.val_iou):.4f}")
if metrics_history.val_dice:
    print(f"Best Dice:      {max(metrics_history.val_dice):.4f}")
if metrics_history.val_loss:
    print(f"Best Val Loss:  {min(metrics_history.val_loss):.4f}")
print(f"\nTo view TensorBoard: tensorboard --logdir={tb_log_dir}")

if writer is not None:
    writer.close()
