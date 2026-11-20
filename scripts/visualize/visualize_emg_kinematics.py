"""Inspect synchronized elbow/wrist estimates and preprocessed EMG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.emg_kinematics.viewer import main


if __name__ == '__main__':
    main()
