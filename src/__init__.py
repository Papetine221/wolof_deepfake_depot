"""Package src pour la détection de deepfakes audio en wolof (Chapitre IV)."""
import os
from pathlib import Path

_cache_root = Path(__file__).resolve().parent.parent / ".cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_cache_root / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mpl"))
