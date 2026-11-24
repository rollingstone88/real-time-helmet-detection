"""Real-time helmet detection for motorcycle riders (PyTorch + OpenCV)."""
import os

# Some torchvision detection ops have no Apple-Silicon (MPS) kernel; fall back to CPU
# for those ops instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "1.0.0"
