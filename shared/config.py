"""Settings shared by the workspace server, harness and grader."""

# Minimum "visible" practice-grid accuracy to pass Workspace 1: look-alike confusions
# (0/O, 1/I, 2/Z, 8/B, 6/G, 5/S) count as correct, since they barely show in the image.
# Baselines (scripts/benchmark_ws1.py, 5 seeds): linear 85.3-87.4%, ReLU-128 90.5-94.9%.
PRACTICE_THRESHOLD = 0.89

# Fixed training settings for Workspace 1 (students control only epochs).
TRAIN_PER_CLASS = 2368
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 5
MAX_EPOCHS = 10

# Hidden layer size limits for the Linear block.
HIDDEN_DEFAULT = 128
HIDDEN_MIN = 16
HIDDEN_MAX = 512

NUM_CLASSES = 36
