"""Compare existing legacy and v002 reconstruction trends without reconstruction."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from src.motion_reconstruction.comparisons.trend_analysis import main

if __name__=='__main__':
    main()
