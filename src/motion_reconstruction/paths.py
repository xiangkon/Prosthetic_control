"""Project-relative defaults shared by the reconstruction entry points."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / 'datasets/TUM/TUM'
OUTPUT_ROOT = PROJECT / 'outputs/TUM/reconstruction'
ANALYSIS_ROOT = PROJECT / 'outputs/TUM/marker_analysis'
REFERENCE_TABLES = PROJECT / 'configs/motion_reconstruction/garner1999_tables.json'
