"""Recover upper-limb joint angles from TUM three-dimensional markers."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    arguments = sys.argv[1:]
    from src.motion_reconstruction.dispatch import select_algorithm
    algorithm, arguments = select_algorithm(arguments)
    if algorithm == 'tum_native':
        from src.motion_reconstruction.tum_native.anchored_pipeline import main
        main(arguments)
    elif '--angle-statistics' in arguments:
        from src.motion_reconstruction.legacy_g99.angle_statistics import main
        main([argument for argument in arguments if argument != '--angle-statistics'])
    else:
        from src.motion_reconstruction.legacy_g99.runner import main
        main(arguments)
