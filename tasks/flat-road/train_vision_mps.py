"""Run Jack Opus's train_vision.py unchanged on Apple MPS (installs the sparse-core drop-in first)."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ.get('FLYHARD_ROOT', Path(__file__).resolve().parents[3] / 'flyhard')) / 'src'))  # sibling flyhard checkout, or set FLYHARD_ROOT
import flyhard.connectome_mps  # noqa: F401
import train_vision
if __name__ == '__main__':
    train_vision.main()
