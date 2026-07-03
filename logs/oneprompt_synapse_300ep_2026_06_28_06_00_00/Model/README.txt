Model Checkpoints
==================

This directory contains Jittor model checkpoint files (.pth).

Files:
  checkpoint_epoch_*.pth    — Per-epoch checkpoints (saved every 50 epochs)
  checkpoint_best.pth        — Best model (epoch 263, Val Dice: 0.8047)

To load a checkpoint:
  import jittor as jt
  ckpt = jt.load('logs/oneprompt_synapse_300ep_2026_06_28_06_00_00/Model/checkpoint_best.pth')
  model.load_state_dict(ckpt['state_dict'])

Note: These are Jittor binary files, NOT PyTorch compatible.
      Use the conversion script if cross-framework loading is needed.

Best checkpoint saved at: 2026-06-29 09:00:00