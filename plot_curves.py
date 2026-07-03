#!/usr/bin/env python
"""
Standalone Loss & Metrics Curve Plotter.
Reads metrics_history.json from a training run and generates plots.

Usage:
  python plot_curves.py --input ./logs/your_exp_xxx/metrics_history.json
  python plot_curves.py --input ./logs/your_exp_xxx/metrics_history.json --output ./my_plots/
"""

import os
import json
import argparse
import numpy as np


def plot_curves(metrics_json_path, output_dir=None):
    """Load metrics JSON and generate loss/metric plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # --- Load data ---
    with open(metrics_json_path, 'r') as f:
        data = json.load(f)

    train_loss = data.get('train_loss', [])
    val_loss = data.get('val_loss', [])
    val_iou = data.get('val_iou', [])
    val_dice = data.get('val_dice', [])

    if not train_loss:
        print("Error: No training data found in JSON file.")
        return

    # --- Output directory ---
    if output_dir is None:
        output_dir = os.path.dirname(metrics_json_path)
    os.makedirs(output_dir, exist_ok=True)

    prefix = os.path.splitext(os.path.basename(metrics_json_path))[0]

    # --- Figure 1: Loss Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Train Loss only
    ax = axes[0]
    ax.plot(range(1, len(train_loss) + 1), train_loss,
            'b-', linewidth=1.5, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Train + Val Loss
    ax = axes[1]
    ax.plot(range(1, len(train_loss) + 1), train_loss,
            'b-', linewidth=1.5, alpha=0.7, label='Train Loss')
    if val_loss:
        val_epochs = np.linspace(1, len(train_loss), len(val_loss))
        ax.plot(val_epochs, val_loss, 'r-o', linewidth=1.5,
                markersize=5, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_path = os.path.join(output_dir, f'{prefix}_loss.png')
    fig.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[1/3] Loss curves → {loss_path}')

    # --- Figure 2: IoU & Dice ---
    if val_iou or val_dice:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        if val_iou:
            epochs = np.linspace(1, len(train_loss), len(val_iou))
            ax.plot(epochs, val_iou, 'g-o', linewidth=1.5, markersize=5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU')
            ax.set_title('Validation IoU')
            ax.grid(True, alpha=0.3)
            # Annotate max
            max_idx = np.argmax(val_iou)
            ax.annotate(f'Best: {val_iou[max_idx]:.4f}',
                        xy=(epochs[max_idx], val_iou[max_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color='darkgreen',
                        arrowprops=dict(arrowstyle='->', color='darkgreen'))

        ax = axes[1]
        if val_dice:
            epochs = np.linspace(1, len(train_loss), len(val_dice))
            ax.plot(epochs, val_dice, 'm-o', linewidth=1.5, markersize=5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Dice')
            ax.set_title('Validation Dice')
            ax.grid(True, alpha=0.3)
            # Annotate max
            max_idx = np.argmax(val_dice)
            ax.annotate(f'Best: {val_dice[max_idx]:.4f}',
                        xy=(epochs[max_idx], val_dice[max_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color='darkmagenta',
                        arrowprops=dict(arrowstyle='->', color='darkmagenta'))

        plt.tight_layout()
        metric_path = os.path.join(output_dir, f'{prefix}_metrics.png')
        fig.savefig(metric_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[2/3] Metric curves → {metric_path}')
    else:
        print('[2/3] No IoU/Dice data to plot.')

    # --- Figure 3: Combined Summary ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('One-Prompt Training Summary', fontsize=14, fontweight='bold')

    # (0,0): Train Loss
    ax = axes[0, 0]
    ax.plot(range(1, len(train_loss) + 1), train_loss, 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # (0,1): Train + Val Loss
    ax = axes[0, 1]
    ax.plot(range(1, len(train_loss) + 1), train_loss, 'b-', alpha=0.6, linewidth=1, label='Train')
    if val_loss:
        val_epochs = np.linspace(1, len(train_loss), len(val_loss))
        ax.plot(val_epochs, val_loss, 'r-o', markersize=4, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0): IoU
    ax = axes[1, 0]
    if val_iou:
        epochs = np.linspace(1, len(train_loss), len(val_iou))
        ax.plot(epochs, val_iou, 'g-o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Validation IoU')
    ax.grid(True, alpha=0.3)

    # (1,1): Dice
    ax = axes[1, 1]
    if val_dice:
        epochs = np.linspace(1, len(train_loss), len(val_dice))
        ax.plot(epochs, val_dice, 'm-o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.set_title('Validation Dice')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_path = os.path.join(output_dir, f'{prefix}_summary.png')
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[3/3] Summary dashboard → {summary_path}')

    # --- Print statistics ---
    print(f"\n{'='*50}")
    print("Training Statistics")
    print(f"{'='*50}")
    print(f"  Epochs trained: {len(train_loss)}")
    print(f"  Final Train Loss: {train_loss[-1]:.4f}")
    print(f"  Min Train Loss:   {min(train_loss):.4f} (epoch {np.argmin(train_loss)+1})")
    if val_loss:
        print(f"  Final Val Loss:   {val_loss[-1]:.4f}")
        print(f"  Min Val Loss:     {min(val_loss):.4f}")
    if val_iou:
        print(f"  Max IoU:          {max(val_iou):.4f}")
    if val_dice:
        print(f"  Max Dice:         {max(val_dice):.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot loss and metric curves from metrics_history.json')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to metrics_history.json')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for plots (default: same as input)')
    args = parser.parse_args()

    plot_curves(args.input, args.output)
