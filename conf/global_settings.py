"""Global settings for One-Prompt training (Jittor version)."""

import os
from datetime import datetime

# Directory to save weights
CHECKPOINT_PATH = 'checkpoint'

# Total training epochs
EPOCH = 500  # ISIC single-dataset: 200-500 is enough
step_size = 10
i = 1
MILESTONES = []
while i * 5 <= EPOCH:
    MILESTONES.append(i * step_size)
    i += 1

# Tensorboard log directory
LOG_DIR = 'runs'

# Save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

# Time of script execution
TIME_NOW = datetime.now().isoformat()
