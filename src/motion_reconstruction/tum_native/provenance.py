"""Source fingerprints for the maintained pose-anchored method.

The frozen tokens preserve existing v002 cache IDs after removing unused
historical entry code. They are hashes, not backup source copies. Compatibility
is allowed only while every retained numerical source matches the reviewed
consolidation baseline; any later numerical edit generates a new fingerprint.
"""
from .schema import PROJECT, canonical_hash, sha256

NUMERICAL_SOURCES = ('src/motion_reconstruction/tum_native/schema.py', 'src/motion_reconstruction/tum_native/observations.py', 'src/motion_reconstruction/tum_native/calibration.py', 'src/motion_reconstruction/tum_native/model.py', 'src/motion_reconstruction/tum_native/quality.py', 'src/motion_reconstruction/tum_native/anchored_pipeline.py', 'src/motion_reconstruction/tum_native/anchored.py', 'src/motion_reconstruction/tum_native/anchored_quality.py', 'src/motion_reconstruction/tum_native/functional_axes.py', 'src/motion_reconstruction/common/motion_io.py', 'src/motion_reconstruction/common/_matlab.py', 'src/motion_reconstruction/common/emg_sync.py')
_CONSOLIDATION_BASELINE = '933b861c21bffbe7174c51ea62afb5f25d8caa7d3710ffd2a5e398874034493f'
_HISTORICAL_SHARED_TOKEN = '09cee3767dcdcc201d84828a3c3b179433527274e4e0594e30c6297478119f07'
_HISTORICAL_RUNNER_TOKEN = 'caab23cbbee96eafe59c0c97ad024517074302daaa399a5ea1bc3d133d33afe2'


def runtime_fingerprint(config):
    sources = {name: sha256(PROJECT/name) for name in NUMERICAL_SOURCES}
    if canonical_hash(sources) == _CONSOLIDATION_BASELINE:
        base = 'src/motion_reconstruction/tum_native/'
        return canonical_hash(dict(config=config, shared=_HISTORICAL_SHARED_TOKEN,
            v2={name: _HISTORICAL_RUNNER_TOKEN if name == 'anchored_pipeline.py' else sources[base+name]
                for name in ['anchored_pipeline.py','anchored.py','anchored_quality.py','functional_axes.py']}))
    return canonical_hash(dict(config=config, numerical_sources=sources, fingerprint_schema='pose-anchored-v2'))
