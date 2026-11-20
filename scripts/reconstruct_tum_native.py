"""Run the independent TUM-native method from any working directory."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
if __name__ == '__main__':
    from src.motion_reconstruction.tum_native.anchored_pipeline import main
    main()
