"""Recover upper-limb joint angles from TUM three-dimensional markers."""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from src.motion_reconstruction.runner import main


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
