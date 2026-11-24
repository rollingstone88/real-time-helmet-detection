import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import helmet  # noqa: E402,F401  (sets the MPS fallback before torch is imported)
