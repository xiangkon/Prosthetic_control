"""Screen every existing TUM trial for forearm-rotation discontinuities."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == '__main__':
    from src.motion_reconstruction.legacy_g99.pronation_screening import main
    main()
