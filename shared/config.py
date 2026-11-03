"""Settings shared by every workspace. Per-workspace settings (pass threshold, epochs,
practice image) are in workspaces.py."""

# Training data and fixed training settings (students control these from Workspace 3 on).
TRAIN_PER_CLASS = 2368
BATCH_SIZE = 128
BATCH_SIZES = (16, 32, 64, 128, 256, 512, 1024)  # choices on the Train block (Workspace 3+)
LEARNING_RATE = 1e-3

# Hidden layer size limits for the Linear block.
HIDDEN_DEFAULT = 128
HIDDEN_MIN = 16
HIDDEN_MAX = 512

# Filter count limits for the Conv block (Workspace 2 on). Filters are always 3x3.
CONV_DEFAULT = 16
CONV_MIN = 4
CONV_MAX = 64

NUM_CLASSES = 36
