"""Settings shared by every workspace. Per-workspace settings (pass threshold, epochs,
practice image) are in workspaces.py."""

# Training data and fixed training settings (students control these from Workspace 3 on).
TRAIN_PER_CLASS = 2368
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Hidden layer size limits for the Linear block.
HIDDEN_DEFAULT = 128
HIDDEN_MIN = 16
HIDDEN_MAX = 512

NUM_CLASSES = 36
