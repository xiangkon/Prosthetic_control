"""Project-relative defaults shared by the reconstruction entry points."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / 'datasets/TUM/TUM'
RECONSTRUCTION_ROOT = PROJECT / 'outputs/TUM/reconstruction'
LEGACY_ROOT = RECONSTRUCTION_ROOT / 'legacy_g99'
NATIVE_ROOT = RECONSTRUCTION_ROOT / 'tum_native'
COMPARISON_ROOT = RECONSTRUCTION_ROOT / 'comparisons'
OUTPUT_ROOT = LEGACY_ROOT  # compatibility alias for legacy-only clients
ANALYSIS_ROOT = LEGACY_ROOT / 'marker_analysis'
REFERENCE_TABLES = PROJECT / 'configs/motion_reconstruction/legacy_g99/garner1999_tables.json'


def require_writable_legacy_output(output):
    """Protect the complete frozen baseline, including ancestor/descendant targets."""
    output, baseline = Path(output).resolve(), LEGACY_ROOT.resolve()
    if output == baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError('legacy_g99 is a frozen baseline. Select an independent --output directory for recomputation.')


def resolve_historical_path(path):
    """Resolve preserved historical path fields through the immutable migration map."""
    import json
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT / candidate
    candidate = candidate.resolve()
    if candidate.exists():
        return candidate
    for inventory in sorted((PROJECT / 'outputs/TUM/_checks/layout_migration').glob('*/inventory.json'), reverse=True):
        data = json.loads(inventory.read_text(encoding='utf-8'))
        for old, new in sorted(data['moves'], key=lambda pair: len(pair[0]), reverse=True):
            source = (PROJECT / old).resolve()
            if candidate == source or candidate.is_relative_to(source):
                target = PROJECT / new / candidate.relative_to(source)
                if target.exists():
                    return target
    return candidate
