Model Checkpoints
==================

This directory contains Jittor model checkpoint files (.pth).

Files:
  checkpoint_epoch_*.pth    — Per-epoch checkpoints (saved every 50 epochs)
  checkpoint_best.pth        — Best model (epoch 278, Val Dice: 0.8702)

To load a checkpoint:
  import jittor as jt
  ckpt = jt.load('logs/oneprompt_isic_300ep_2026_06_25_09_15_00/Model/checkpoint_best.pth')
  model.load_state_dict(ckpt['state_dict'])

Note: These are Jittor binary files, NOT PyTorch compatible.
      Use the conversion script if cross-framework loading is needed.

Best checkpoint saved at: 2026-06-26 01:15:00