"""Rank subjects and tasks using saved reconstruction trend metrics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.motion_reconstruction.comparisons.trend_ranking import main

if __name__=='__main__': main()
