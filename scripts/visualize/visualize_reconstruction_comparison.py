"""Open the legacy G99 and TUM native v002 comparison viewer."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.motion_reconstruction.comparisons.viewer import main

if __name__ == '__main__':
    main()
